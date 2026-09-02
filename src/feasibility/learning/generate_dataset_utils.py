"""Shared machinery for the two dataset generators, learning/generate_init_pose_dataset.py and
learning/generate_dataset_body_centered_patch.py: everything that has nothing to do with WHICH
feature set (spawn pose vs. terrain patch) ends up in the output file.

Two pieces:

* Spawn-pose sampling on the loaded heightmap -- the "lattice"/"continuous" spawn modes, and
  sample_dataset(), which draws n (spawn_pose, v_drive, wz_drive) trials. See sample_dataset()'s
  docstring for the full spawn sampling story (spawn modes, the SPAWN_LIMIT square, why
  V_RANGE/WZ_RANGE are independent of spawn pose). Spawn poses are no longer filtered against a
  box obstacle's footprint (that SAT test assumed the fixed centered-box series, which a generic
  `+map=` heightmap can't guarantee) -- every pose in the SPAWN_LIMIT square is legal now, so
  callers that need obstacle clearance should pick a map/SPAWN_LIMIT combination that keeps the
  spawn square off any obstacle themselves.
* resolve_map_path(): turns a `+map=` CLI value into a loadable HeightMapReader path -- absolute
  paths pass through, everything else resolves against REPO_ROOT (so `+map=assets/foo/bar` works
  regardless of the invoking cwd, matching every heightmap/create_*.py generator's own asset
  layout).
* simulate_dataset_rollout(): the chunked ostrich-vs-hstack batch rollout both generators run
  identically (they only differ in what non-simulator feature array -- spawn_pose columns vs. a
  terrain patch -- gets written alongside its output).

CLI parameters shared by both generators (Hydra overrides read via generate(); `+` prefix
required since none of these exist in the base "helhest" config):
    +n_samples=INT          trajectory pairs to generate (default: 128)
    +seed=INT               RNG seed for spawn/command sampling (default: 0)
    +map=STR                path (absolute, or relative to the repo root) to the heightmap to
                            load, PNG+YAML pair, extension optional, e.g.
                            assets/speed_bumps/speed_bump_h010cm (default: DEFAULT_MAP, the
                            centered box terrain this used to default to via box_height=0.70)
    +duration_s=FLOAT       command hold time in seconds; ideally an exact multiple of both
                            sims' dt (ostrich 3e-2, hstack 0.1) or a warning is printed
                            (default: 2.4)
    +chunk=INT              robots per ostrich model build/batch (tuning knob only, not a hard
                            limit) (default: 128)
    +settle_steps=INT       ostrich steps spent settling the robot onto the terrain before
                            recording starts, paid once per chunk; raise it if a spawn is still
                            moving at t=0 (default: 12)
    +spawn_mode=STR         "lattice" (grid poses, drawn with replacement) or "continuous"
                            (uniform + rejection sampling) (default: "lattice")
    +mu=FLOAT               ground friction coefficient (default: 0.8)
    +k_turn=FLOAT           helhest_stack ICR turning-rate gain (default: dynamics.K_TURN)
    +device=STR             helhest_stack torch/warp device (default: "cuda:0")
    Also accepts any standard Hydra config-group override against the "helhest" base config
    (e.g. engine=mujoco, logging=..., simulation=...) -- see ostrich/examples/conf/helhest.yaml
    for the groups. rendering is forced to headless by simulate_dataset_rollout() and cannot be
    overridden.
"""
from __future__ import annotations

import pathlib
import time
import warnings

import hydra
import numpy as np
from omegaconf import DictConfig
from ostrich import EngineConfig
from ostrich import LoggingConfig
from ostrich import RenderingConfig
from ostrich import SimulationConfig

from helhest import dynamics

from feasibility.comparator.common import build_setpoints
from feasibility.comparator.common import euler_zyx_to_quat_xyzw
from feasibility.comparator.common import run_hstack_batch
from feasibility.comparator.common import run_ostrich_batch
from feasibility.heightmap import HeightMapReader

# --- hyperparameters (module constants -- override any of them with a Hydra `+key=value`) -------

DEFAULT_N = 128  # trajectory pairs to generate
DEFAULT_DURATION_S = 2.4  # s, command hold time -- see module docstring / simulate_dataset_
# rollout()'s duration check: ostrich dt=3e-2, hstack dt=0.1 (see comparator.common), so 0.6s is
# the smallest exact multiple of both that is >= the originally requested 0.5s. An override that
# isn't a multiple of both still runs, just with a printed warning.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]  # matches every heightmap/create_*.py
# generator's own REPO_ROOT (learning -> feasibility -> src -> repo root)
DEFAULT_MAP = "assets/box_centered/box_centered_h070cm"  # repo-root-relative -- the terrain this
# used to default to via box_height=0.70 before +map= replaced it
DEFAULT_SEED = 0
DEFAULT_CHUNK = 128  # robots per ostrich model build -- a tuning knob, not a hard limit; see
# ostrich/experiments/4_scalability for headroom (65536 worlds at 6.3 GB WITH an adjoint tape --
# this is forward-only and graph-captured, far cheaper per world).
DEFAULT_SETTLE_STEPS = 12  # ostrich steps spent dropping the robot onto the terrain at zero
# command before recording starts. Overrides run_ostrich_batch's None default, which would let
# _resolve_settle_steps pick max(60, 0.5s/dt) = 60 steps (1.8 s) at ostrich's dt=3e-2 -- a floor
# sized for the dt=5e-4 replay case. Measured on a 32-world continuous batch on DEFAULT_MAP:
# the chassis free-falls for 6 steps from its +0.5 m spawn (velocity tracking 9.81*t exactly),
# lands on step 6, and is static from step 7 on -- height pinned to 4 decimals, residual speed
# ~2 mm/s. 12 is that with ~1.7x margin. Unlike the compare_*.py sweeps, which settle once for a
# whole run, this is paid once PER CHUNK (n/chunk times), so it was ~35% of total ostrich time.

SPAWN_STEP = 0.5  # m, (x, y) lattice pitch
SPAWN_LIMIT = 2.0  # m, |x|, |y| <= this -- spawns stay within a FIXED 10m x 10m square centered
# on the world origin, independent of the loaded heightmap's own extent or contents (DEFAULT_MAP's
# 16m x 16m grid leaves 3m of clearance on each side beyond the spawn square before the grid
# edge; a different `+map=` should be picked with the same margin in mind). This bounds where the
# robot can SPAWN, not how far it travels during the rollout: a
# trial that spawns near the edge of this square and then drives at up to
# V_RANGE[1]*duration_s can still cross the grid edge if that distance exceeds the remaining
# clearance -- HeightMapReader.sample() clamps silently rather than erroring outside the grid,
# so this is no longer a hold-by-construction invariant the way it was when SPAWN_LIMIT was
# derived from containment (see git history); widen the heightmap's --extent or shrink
# V_RANGE/duration_s together if that matters for your run.
N_YAW = 8  # yaw = k * 360/N_YAW deg, k in 0..N_YAW-1 -- `lattice` mode only
DEFAULT_SPAWN_MODE = "lattice"  # see module docstring; "continuous" is what large-n runs want
SPAWN_MODES = ("lattice", "continuous")
SPAWN_MODE_TAGS = {"lattice": "", "continuous": "_cont"}  # output-filename suffix per mode, so a
# continuous run can't silently overwrite a lattice run of the same n/height (the default mode
# keeps an empty tag, leaving every already-generated filename untouched)
V_RANGE = (0.0, 1.5)  # m/s, forward body velocity command
WZ_RANGE = (-1.0, 1.0)  # rad/s, yaw-rate command


def resolve_map_path(map_arg: str) -> pathlib.Path:
    """Turns a `+map=` CLI value into a loadable HeightMapReader path: absolute paths pass
    through unchanged, everything else resolves against REPO_ROOT -- so `+map=assets/foo/bar`
    works the same regardless of the invoking cwd, matching every heightmap/create_*.py
    generator's own asset layout (assets/box_centered/, assets/speed_bumps/, ...)."""
    p = pathlib.Path(map_arg)
    return p if p.is_absolute() else REPO_ROOT / p


def legal_spawn_poses() -> np.ndarray:
    """All (x, y, yaw) on the SPAWN_STEP/N_YAW lattice within +-SPAWN_LIMIT -- [M, 3],
    M = (2*SPAWN_LIMIT/SPAWN_STEP + 1)**2 * N_YAW. Drawn from by sample_dataset()'s "lattice"
    mode. Row order matches the nested x -> y -> yaw enumeration it always had, so a given seed
    keeps drawing the same poses. No longer filtered against a box footprint (see module
    docstring) -- every pose in the square is legal."""
    lattice = np.arange(-SPAWN_LIMIT, SPAWN_LIMIT + 1e-9, SPAWN_STEP)
    yaws = np.arange(N_YAW) * (2.0 * np.pi / N_YAW)
    grid_x, grid_y, grid_yaw = np.meshgrid(lattice, lattice, yaws, indexing="ij")
    return np.stack([grid_x.ravel(), grid_y.ravel(), grid_yaw.ravel()], axis=1)


def continuous_spawn_poses(n: int, rng: np.random.Generator) -> np.ndarray:
    """n (x, y, yaw) poses drawn uniformly -- (x, y) ~ U(-SPAWN_LIMIT, SPAWN_LIMIT), yaw ~
    U(0, 2pi). [n, 3]."""
    xy = rng.uniform(-SPAWN_LIMIT, SPAWN_LIMIT, size=(n, 2))
    yaw = rng.uniform(0.0, 2.0 * np.pi, size=n)
    return np.column_stack([xy, yaw]).astype(np.float64)


def sample_dataset(
    n: int, seed: int, spawn_mode: str = DEFAULT_SPAWN_MODE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draws n (spawn_pose, v_drive, wz_drive) samples. spawn_mode selects how spawn_pose is
    drawn (see module docstring): "lattice" draws WITH replacement from the SPAWN_STEP/N_YAW
    grid (a repeated pose under a different twist is still a useful sample), "continuous" draws
    uniformly over the spawn square and the full yaw circle.

    v_drive/wz_drive are independent continuous draws, uniform over V_RANGE/WZ_RANGE in either
    mode -- spawn pose and command are not correlated.

    Returns spawn_pose [n, 3] float64, v_drive [n] float32, wz_drive [n] float32. The rng is
    consumed pose-first then v then wz, so a "lattice" run at a given seed still produces
    exactly the poses it did before this mode existed."""
    if spawn_mode not in SPAWN_MODES:
        raise ValueError(f"spawn_mode must be one of {SPAWN_MODES}, got {spawn_mode!r}")

    rng = np.random.default_rng(seed)
    if spawn_mode == "lattice":
        legal = legal_spawn_poses()
        spawn_pose = legal[rng.integers(0, len(legal), size=n)]
    else:
        spawn_pose = continuous_spawn_poses(n, rng)
    v_drive = rng.uniform(*V_RANGE, size=n).astype(np.float32)
    wz_drive = rng.uniform(*WZ_RANGE, size=n).astype(np.float32)
    return spawn_pose, v_drive, wz_drive


def simulate_dataset_rollout(
    cfg: DictConfig,
    terrain: HeightMapReader,
    spawn_pose: np.ndarray,
    v_drive: np.ndarray,
    wz_drive: np.ndarray,
) -> tuple[dict, dict, float, float]:
    """Runs the chunked ostrich-vs-hstack batch rollout shared by both generate_*.py dataset
    scripts: replays every (spawn_pose[i], v_drive[i], wz_drive[i]) trial in ostrich and
    helhest_stack, `chunk` worlds at a time (comparator.common.HelhestBatchSimulator/
    run_hstack_batch both take a per-world [N, 3] spawn_pose, so a chunk of trials becomes N
    parallel worlds sharing `terrain` rather than one build per trial -- see
    comparator/common.py's module docstring for why that can't span DIFFERENT terrains, which is
    fine here since both generators use one fixed heightmap for the whole run).

    Reads n_samples-independent knobs from `cfg` (see module docstring): duration_s, chunk,
    settle_steps, mu, k_turn, device, plus any Hydra config-group override against the "helhest"
    base config (engine=, simulation=, logging=...). rendering is forced to headless.

    Returns (ostrich_fields, hstack_fields, mu, k_turn) -- the first two are ready to splice
    straight into comparator.provenance.write_comparison's `ostrich=`/`hstack=` kwargs; mu/k_turn
    are returned too since callers also record them in write_comparison's `root=` dict."""
    n = spawn_pose.shape[0]
    duration_s = float(cfg.get("duration_s", DEFAULT_DURATION_S))
    chunk = int(cfg.get("chunk", DEFAULT_CHUNK))
    settle_steps = int(cfg.get("settle_steps", DEFAULT_SETTLE_STEPS))
    mu = float(cfg.get("mu", 0.8))
    k_turn = float(cfg.get("k_turn", dynamics.K_TURN))
    device = str(cfg.get("device", "cuda:0"))

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
    print(
        f"[ostrich]  {n} samples x {T_o} steps @ dt={ostrich_dt}, chunk={chunk}, "
        f"settle={settle_steps} steps/chunk"
    )
    print(f"[hstack]   {n} samples x {T_h} steps @ dt={hstack_dt}")

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
            sim_config, render_config, engine_config, logging_config, terrain,
            ostrich_setpoints, mu, spawn_chunk, settle_steps,
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

    ostrich_fields = dict(
        dt=ostrich_dt,
        t=np.arange(T_o, dtype=np.float32) * ostrich_dt,
        cmd_wheel_omega=ostrich_cmd_wheel_omega,
        pose=ostrich_pose,
        wheel_qd=ostrich_wheel_qd,
    )
    hstack_fields = dict(
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
    )
    return ostrich_fields, hstack_fields, mu, k_turn
