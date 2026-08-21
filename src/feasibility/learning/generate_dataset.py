"""Generates a trial-style ostrich-vs-helhest_stack comparison HDF5 for
feasibility.learning.custom_dataset.PoseErrorDataset -- N randomized (spawn pose, constant body
twist) initial conditions on ONE fixed centered-box heightmap
(feasibility.heightmap.create_box_obstacles.centered_box_paths()), each replayed once in ostrich
(dynamics) and once in helhest_stack (kinematic twin), producing exactly the mapping
custom_dataset.py expects:

    x = (v_drive, wz_drive, spawn_x, spawn_y, spawn_yaw)   [n, 5]
    y = (e_pos, e_rot)                                      [n, 2]  -- computed by custom_dataset
                                                                        from the saved poses, not
                                                                        stored here

Unlike comparator/compare_box_obstacles.py (one fixed spawn+twist, swept across a heightmap
SERIES) or compare_on_surface.py (a handful of hand-picked Trials on one terrain), this samples
MANY random (spawn, twist) trials on one terrain, and runs them as N robots in N replicated
ostrich worlds sharing that one terrain (comparator.common.HelhestBatchSimulator/run_hstack_batch
now both accept a per-world/per-row [N, 3] spawn_pose) rather than rebuilding the model once per
sample -- see the module's own docstrings for how. This is what makes N in the hundreds-to-
thousands practical; run_trial_comparison's one-build-per-trial loop would not be.

Spawn sampling: (x, y) on a SPAWN_STEP-meter lattice within a fixed +-SPAWN_LIMIT m square
centered on the heightmap (10m x 10m by default, well inside its 16m x 16m default extent), yaw
one of N_YAW evenly-spaced headings. A pose is rejected if the robot's oriented footprint would
overlap the box's (inflated-by-ramp-and-safety-margin) footprint -- an exact separating-axis
test, not a distance heuristic -- but nothing biases sampling TOWARD the box: trials that spawn
facing away and simply drive off are valid, useful samples too. (v_drive, wz_drive) are drawn
uniformly from V_RANGE/WZ_RANGE independently of spawn pose.

Usage:
    python src/feasibility/learning/generate_dataset.py                       # 512 defaults
    python src/feasibility/learning/generate_dataset.py +n_samples=2000 +seed=1
    python src/feasibility/learning/generate_dataset.py +box_height=0.5 +chunk=64
    python src/feasibility/learning/generate_dataset.py +duration_s=1.2       # exact multiple of
                                                                                # both sims' dt
"""
from __future__ import annotations

import time
import warnings

import hydra
import numpy as np
import warp as wp
from examples.helhest_junior.common import HelhestJuniorConfig
from omegaconf import DictConfig
from ostrich import EngineConfig
from ostrich import LoggingConfig
from ostrich import RenderingConfig
from ostrich import SimulationConfig

from helhest import dynamics

from feasibility.comparator.common import build_setpoints
from feasibility.comparator.common import CONFIG_PATH
from feasibility.comparator.common import euler_zyx_to_quat_xyzw
from feasibility.comparator.common import K_P
from feasibility.comparator.common import OUT_DIR
from feasibility.comparator.common import run_hstack_batch
from feasibility.comparator.common import run_ostrich_batch
from feasibility.comparator.provenance import write_comparison
from feasibility.heightmap import HeightMapReader
from feasibility.heightmap.create_box_obstacles import BOX_SIZE
from feasibility.heightmap.create_box_obstacles import centered_box_path
from feasibility.heightmap.create_box_obstacles import DEFAULT_INCLINE_DEG

# --- hyperparameters (module constants -- override any of them with a Hydra `+key=value`) -------

DEFAULT_N = 128  # trajectory pairs to generate
DEFAULT_DURATION_S = 2.4  # s, command hold time -- see module docstring / generate()'s duration
# check: ostrich dt=3e-2, hstack dt=0.1 (see comparator.common), so 0.6s is the smallest exact
# multiple of both that is >= the originally requested 0.5s. An override that isn't a multiple
# of both still runs, just with a printed warning (see generate()).
DEFAULT_BOX_HEIGHT = 0.70  # m, which centered_box_paths() height to load
DEFAULT_SEED = 0
DEFAULT_CHUNK = 128  # robots per ostrich model build -- a tuning knob, not a hard limit; see
# generate_dataset.py's plan / ostrich/experiments/4_scalability for headroom (65536 worlds at
# 6.3 GB WITH an adjoint tape -- this is forward-only and graph-captured, far cheaper per world).

SPAWN_STEP = 0.5  # m, (x, y) lattice pitch
SPAWN_LIMIT = 2.0  # m, |x|, |y| <= this -- spawns stay within a FIXED 10m x 10m square centered
# on the heightmap, independent of the heightmap's own extent (default centered_box_paths() grid
# is 16m x 16m, leaving 3m of clearance on each side beyond the spawn square before the grid
# edge). This bounds where the robot can SPAWN, not how far it travels during the rollout: a
# trial that spawns near the edge of this square and then drives at up to
# V_RANGE[1]*duration_s can still cross the grid edge if that distance exceeds the remaining
# clearance -- HeightMapReader.sample() clamps silently rather than erroring outside the grid,
# so this is no longer a hold-by-construction invariant the way it was when SPAWN_LIMIT was
# derived from containment (see git history); widen the heightmap's --extent or shrink
# V_RANGE/duration_s together if that matters for your run.
N_YAW = 8  # yaw = k * 360/N_YAW deg, k in 0..N_YAW-1
V_RANGE = (0.0, 1.5)  # m/s, forward body velocity command
WZ_RANGE = (-1.0, 1.0)  # rad/s, yaw-rate command
SAFETY_MARGIN = 0.05  # m, extra clearance between the robot's footprint and the box's (already
# ramp-inflated) footprint at spawn time

# Robot whole-body AABB in the body frame (X forward, Y left, origin = front-wheel axle
# midpoint) -- rear-wheel rim to front-wheel rim, half-track + wheel half-width laterally. Same
# geometry comparator/compare_box_obstacles.py's module docstring derives, read from
# HelhestJuniorConfig instead of re-hardcoded.
ROBOT_X_MIN = float(HelhestJuniorConfig.REAR_WHEEL_POS[0]) - HelhestJuniorConfig.WHEEL_RADIUS
ROBOT_X_MAX = HelhestJuniorConfig.WHEEL_RADIUS
ROBOT_Y_HALF = float(HelhestJuniorConfig.LEFT_WHEEL_POS[1]) + HelhestJuniorConfig.WHEEL_WIDTH / 2.0


def _robot_box_disjoint(x: float, y: float, yaw: float, box_half: float) -> bool:
    """Exact separating-axis test: does the robot's oriented rectangular footprint at (x, y,
    yaw) overlap the box's axis-aligned footprint (a box_half x box_half square centered at the
    origin -- the centered series' box is always at (0, 0), see create_box_obstacles.py)?
    Returns True iff they do NOT overlap (i.e. this spawn pose is legal)."""
    corners_local = np.array(
        [
            [ROBOT_X_MIN, -ROBOT_Y_HALF],
            [ROBOT_X_MAX, -ROBOT_Y_HALF],
            [ROBOT_X_MAX, ROBOT_Y_HALF],
            [ROBOT_X_MIN, ROBOT_Y_HALF],
        ]
    )
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    corners = corners_local @ rot.T + np.array([x, y])
    box_corners = np.array(
        [[-box_half, -box_half], [box_half, -box_half], [box_half, box_half], [-box_half, box_half]]
    )
    for axis in (np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([c, s]), np.array([-s, c])):
        p_robot = corners @ axis
        p_box = box_corners @ axis
        if p_robot.max() < p_box.min() or p_box.max() < p_robot.min():
            return True
    return False


def box_half_extent(box_height: float, incline_deg: float = DEFAULT_INCLINE_DEG) -> float:
    """The box's flat-top half-extent (BOX_SIZE/2) plus its ramp bevel (see
    create_box_obstacles.py's module docstring: ramp_width = height / tan(incline)) plus
    SAFETY_MARGIN -- the half-extent a spawn's footprint must clear."""
    ramp_width = box_height / np.tan(np.radians(incline_deg))
    return BOX_SIZE / 2.0 + ramp_width + SAFETY_MARGIN


def legal_spawn_poses(box_half: float) -> np.ndarray:
    """All (x, y, yaw) on the SPAWN_STEP/N_YAW lattice within +-SPAWN_LIMIT whose robot
    footprint doesn't overlap the box -- [M, 3], M <= (2*SPAWN_LIMIT/SPAWN_STEP + 1)**2 * N_YAW.
    Precomputed once per box height and drawn from by sample_dataset(), not filtered per-draw."""
    lattice = np.arange(-SPAWN_LIMIT, SPAWN_LIMIT + 1e-9, SPAWN_STEP)
    yaws = np.arange(N_YAW) * (2.0 * np.pi / N_YAW)
    poses = [
        (x, y, yaw)
        for x in lattice
        for y in lattice
        for yaw in yaws
        if _robot_box_disjoint(x, y, yaw, box_half)
    ]
    return np.array(poses, dtype=np.float64)


def sample_dataset(
    n: int, seed: int, box_height: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draws n (spawn_pose, v_drive, wz_drive) samples. spawn_pose is drawn WITH replacement
    from the legal lattice (a repeated pose under a different twist is still a useful sample);
    v_drive/wz_drive are independent continuous draws, uniform over V_RANGE/WZ_RANGE -- spawn
    pose and command are not correlated, so a trial that starts facing away from the box and
    simply drives off is exactly as likely (and exactly as valid a sample) as one that drives
    into it. Returns spawn_pose [n, 3] float64, v_drive [n] float32, wz_drive [n] float32."""
    legal = legal_spawn_poses(box_half_extent(box_height))
    rng = np.random.default_rng(seed)
    spawn_pose = legal[rng.integers(0, len(legal), size=n)]
    v_drive = rng.uniform(*V_RANGE, size=n).astype(np.float32)
    wz_drive = rng.uniform(*WZ_RANGE, size=n).astype(np.float32)
    return spawn_pose, v_drive, wz_drive


def generate(cfg: DictConfig) -> None:
    wp.init()

    n = int(cfg.get("n_samples", DEFAULT_N))
    seed = int(cfg.get("seed", DEFAULT_SEED))
    box_height = float(cfg.get("box_height", DEFAULT_BOX_HEIGHT))
    duration_s = float(cfg.get("duration_s", DEFAULT_DURATION_S))
    chunk = int(cfg.get("chunk", DEFAULT_CHUNK))
    mu = float(cfg.get("mu", 0.8))
    k_turn = float(cfg.get("k_turn", dynamics.K_TURN))
    device = str(cfg.get("device", "cuda:0"))

    terrain_path = centered_box_path(box_height)
    terrain = HeightMapReader.load(terrain_path)

    spawn_pose, v_drive, wz_drive = sample_dataset(n, seed, box_height)
    labels = np.array([f"s{i:05d}" for i in range(n)])

    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)
    render_config.vis_type = "null"  # headless -- see HelhestBatchSimulator docstring for why a
    # GL viewer's set_world_offsets would scatter N robots off the one shared terrain anyway

    ostrich_dt = sim_config.target_timestep_seconds
    hstack_dt = dynamics.DT
    T_o = int(round(duration_s / ostrich_dt))
    T_h = int(round(duration_s / hstack_dt))
    end_o, end_h = T_o * ostrich_dt, T_h * hstack_dt
    # Comparing reconstructed end times (rather than duration_s % dt directly) sidesteps binary
    # float noise: e.g. 0.5 % 0.1 == 0.09999... in Python floats even though 0.5 IS an exact
    # multiple of 0.1 -- round-tripping through the actual step count T is what both sims
    # themselves do, so it's the correct definition of "divisible" here, not just a proxy for it.
    if abs(end_o - end_h) > 1e-6:
        warnings.warn(
            f"duration_s={duration_s} is not an exact multiple of BOTH ostrich's dt={ostrich_dt} "
            f"and hstack's dt={hstack_dt}, so the two sims don't land on the same end time "
            f"(ostrich {T_o} steps -> {end_o:.4f}s, hstack {T_h} steps -> {end_h:.4f}s) -- "
            f"final-pose error will carry a systematic bias of up to one ostrich step of travel "
            f"(v_drive*{ostrich_dt}={V_RANGE[1] * ostrich_dt:.4f} m worst case). Use a duration_s "
            f"that's a multiple of both dt (e.g. any multiple of 0.3s here) to avoid this -- "
            f"duration_s={DEFAULT_DURATION_S} (the default) already is.",
            stacklevel=2,
        )
    print(f"[ostrich]  {n} samples x {T_o} steps @ dt={ostrich_dt}, chunk={chunk}")
    print(f"[hstack]   {n} samples x {T_h} steps @ dt={hstack_dt}, chunk={chunk}")

    ostrich_poses, ostrich_wheel_qds, ostrich_cmds = [], [], []
    h_controlleds, h_deriveds, h_clearances, h_residuals, h_turnings, h_wheel_qds, hstack_cmds = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        b = end - start
        t0 = time.time()
        sim_config.num_worlds = b
        spawn_chunk = spawn_pose[start:end]
        v_chunk, wz_chunk = v_drive[start:end], wz_drive[start:end]

        ostrich_setpoints = np.stack(
            [build_setpoints(ostrich_dt, v_chunk[i], wz_chunk[i], duration_s) for i in range(b)], axis=1
        )  # [T_o, b, 3]
        hstack_setpoints = np.stack(
            [build_setpoints(hstack_dt, v_chunk[i], wz_chunk[i], duration_s) for i in range(b)], axis=1
        )  # [T_h, b, 3]

        pose, wheel_qd = run_ostrich_batch(
            sim_config, render_config, engine_config, logging_config, terrain, ostrich_setpoints, mu, spawn_chunk
        )
        controlled, derived, clearance, residual, turning, wheel_qd_h = run_hstack_batch(
            hstack_setpoints, terrain, hstack_dt, k_turn, mu, device, spawn_chunk
        )

        ostrich_poses.append(pose)
        ostrich_wheel_qds.append(wheel_qd)
        ostrich_cmds.append(ostrich_setpoints)
        h_controlleds.append(controlled)
        h_deriveds.append(derived)
        h_clearances.append(clearance)
        h_residuals.append(residual)
        h_turnings.append(turning)
        h_wheel_qds.append(wheel_qd_h)
        hstack_cmds.append(hstack_setpoints)

        print(f"  samples {start}..{end - 1} ({b}) done in {time.time() - t0:.1f}s")

    ostrich_pose = np.concatenate(ostrich_poses, axis=1)  # [T_o, n, 7]
    ostrich_wheel_qd = np.concatenate(ostrich_wheel_qds, axis=1)
    ostrich_cmd_wheel_omega = np.concatenate(ostrich_cmds, axis=1)
    h_controlled = np.concatenate(h_controlleds, axis=1)  # [T_h, n, 3]
    h_derived = np.concatenate(h_deriveds, axis=1)
    h_clearance = np.concatenate(h_clearances, axis=1)
    h_residual = np.concatenate(h_residuals, axis=1)
    h_turning = np.concatenate(h_turnings, axis=1)
    h_wheel_qd = np.concatenate(h_wheel_qds, axis=1)
    hstack_cmd_wheel_omega = np.concatenate(hstack_cmds, axis=1)

    h_quat = euler_zyx_to_quat_xyzw(h_controlled[..., 2], h_derived[..., 1], h_derived[..., 2])
    h_pose = np.concatenate([h_controlled[..., :2], h_derived[..., :1], h_quat], axis=-1).astype(np.float32)

    out_path = OUT_DIR / f"dataset_box_h{round(box_height * 100):03d}cm_n{n}.h5"
    write_comparison(
        out_path,
        root=dict(
            n=n, variant_name="sample", obstacle_x=0.0, duration_s=duration_s, mu=mu, k_turn=k_turn, k_p=K_P
        ),
        per_variant=dict(
            # no single scalar parameter here -- sample index, purely so the field stays
            # populated for any generic consumer that expects it (see run_trial_comparison)
            variant_value=np.arange(n, dtype=np.float32),
            variant_label=labels,
            spawn_pose=spawn_pose.astype(np.float32),
            v_drive=v_drive,
            wz_drive=wz_drive,
        ),
        terrain_entries=[(terrain_path, terrain)] * n,
        ostrich=dict(
            dt=ostrich_dt,
            t=np.arange(T_o, dtype=np.float32) * ostrich_dt,
            cmd_wheel_omega=ostrich_cmd_wheel_omega,
            pose=ostrich_pose,
            wheel_qd=ostrich_wheel_qd,
        ),
        hstack=dict(
            dt=hstack_dt,
            t=np.arange(T_h, dtype=np.float32) * hstack_dt,
            cmd_wheel_omega=hstack_cmd_wheel_omega,
            pose=h_pose,
            controlled=h_controlled,
            derived=h_derived,
            turning=h_turning,
            clearance=h_clearance,
            residual=h_residual,
            wheel_qd=h_wheel_qd,
        ),
    )


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest", version_base=None)
def main(cfg: DictConfig) -> None:
    generate(cfg)


if __name__ == "__main__":
    main()
