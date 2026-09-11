"""Generates a lattice_learning divergence dataset: one sample is a single body-frame terrain
patch plus ONE scalar curvature command, replayed as the router's OWN forward-arc primitive
(design.md sections 1, 2, 4, 7) --

    x = (patch [24, 28] @ 0.125 m, kappa)      -- v = V_NOM, L = ARC_LEN are PINNED, not sampled
    y = (e_pos, e_rot)  -- ostrich vs. the ARC-PLUS-SETTLE reference, computed by custom_dataset.py

Three things this generator does that no sibling generator does, all from design.md:

* **Warm start (section 2).** Every trial is entered already moving at V_NOM: `warmup_s` seconds
  of CAPTURED, commanded ostrich rollout are prepended to the setpoint array and sliced off the
  returned log -- `t0_pose` (ostrich's actual pose row `w_o - 1`) is the arc's true origin, not
  the nominal spawn.
* **`OSTRICH_DT` is re-pinned to 2.5e-2 s** (section 2a), the finest `dt = 0.1/k` that divides
  both `ARC_LEN/V_NOM = 0.5 s` and the twin's `DT = 0.1 s` exactly -- overridden in code on
  `sim_config`, never in `examples/conf/simulation/helhest.yaml` (shared with `submodule_test/`'s
  characterization baseline). Kept a multiple of the twin's own dt even though this generator does
  not run the twin, so a future arc-vs-twin diagnostic pass can replay the same trials.
* **The reference is the arc integrated from `t0_pose`, plus a static settle at its endpoint**
  (section 1b) -- not the twin. `settle_batch()` below re-derives `costtogo.py`'s own
  `ForwardSimulator` settle convention (`n_steps=1`, zero command, row 0 of `derived` IS the
  static settle) rather than importing anything from `helhest.planning`, since that machinery is
  embedded in `CostToGo`'s captured-graph setup and isn't meant to be called standalone per trial.
  This is the ONLY use of `helhest_stack`'s `ForwardSimulator` here -- a static, zero-command,
  single-step settle at a given pose, not a dynamic rollout. The label itself only ever compares
  ostrich against this arc-plus-settle reference (`custom_dataset.py`); the twin's own dynamic
  trajectory over the arc is not simulated at all in this prototype. Section 1a's arc-vs-twin /
  twin-vs-ostrich diagnostic split is a possible FUTURE addition, not implemented here -- adding
  it means running `comparator.common.run_hstack_batch`'s dynamic rollout alongside ostrich's,
  which this generator deliberately does not pay for yet.

`swept_clear` (section 7b/8, a REPORTING split, not a training filter) approximates
`_relax_lattice_pose_kernel`'s sweep test by settling the robot at several points along the SAME
arc, at the arc's OWN (fixed) heading -- matching the kernel's own convention of indexing
`blocked[..., t]` at the sweep's outer heading, not the locally-varying arc heading.

Deliberately independent of `feasibility.learning`, `feasibility.grid_learning` and
`feasibility.grid_learning_2` (design.md section 11a): the obstacle/footprint filter is
re-derived here (as every sibling generator's own copy is) rather than imported.
`feasibility.comparator` (the batch-rollout core, and `provenance`'s terrain/git embedding) and
`feasibility.heightmap` are shared infrastructure and ARE imported; so are this package's own
`arc.py`/`patch.py`, since the dataset schema is a contract between this generator and this
package's `model.py`/`custom_dataset.py`. `helhest`/`ostrich` are objects of study, imported
directly.

Writes through a LOCAL writer (`write_arc_dataset`), not `comparator.provenance.write_comparison`:
that shared writer hard-requires both an `ostrich/` and an `hstack/` group, and (see above) this
generator never runs the twin. `write_arc_dataset` reuses `provenance.terrain_fields`/
`git_provenance` verbatim -- the same reuse `grid_learning_2/generate_dataset.py`'s own local
writer makes, for the identical reason (its schema doesn't fit `write_comparison`'s two-sim
assumption either) -- so terrain embedding and git provenance still can't drift from
`write_comparison`'s own files.

CLI parameters (Hydra overrides; `+` prefix required since none exist in the base "helhest"
config):
    +n_maps=INT          maps drawn (without replacement) from maps_dir     (default: 4)
    +trials_per_map=INT  trials per map -- n = n_maps * this                (default: 16)
    +maps_dir=STR        repo-root-relative or absolute map directory
                         (default: assets/large_box_random/0)
    +seed=INT            RNG seed for map selection, poses, kappa, jitter   (default: 0)
    +kappa_max=FLOAT     sample kappa ~ U(-this, this), 1/m                 (default: arc.KAPPA_MAX)
    +warmup_s=FLOAT      captured warm-up duration before the recorded arc (default: 0.375) --
                         design.md section 2's "measured quantity"; NOT independently re-measured
                         in this checkout, see DEFAULT_WARMUP_S's comment
    +settle_steps=INT    ostrich steps dropping onto the terrain at zero command, paid once per
                         chunk, BEFORE the (uncaptured) warm-up window begins  (default: 15)
    +chunk=INT           trials per ostrich model build                    (default: 64)
    +mu=FLOAT            ground friction                                   (default: 0.8)
    +device=STR          torch/warp device for ostrich AND the settle      (default: "cuda:0")
    +router_cell=FLOAT   the router's own lattice cell -- sets xy_jitter = this/2 (section 1c)
                         (default: 0.24, demos/navigate_partial_view.py's lat_coarsen=4 example)
    +n_theta=INT         the router's own heading bin count -- sets yaw_jitter = pi/this
                         (default: 24)
    +near_obstacle_frac=FLOAT  fraction of each map's spawns biased towards the obstacle
                         boundary rather than drawn uniformly over the map (default: 0.0, i.e.
                         unbiased) -- see sample_spawn_poses' docstring; uniform sampling alone
                         lands an arc's endpoint on the obstacle only 1-3% of the time
    +near_obstacle_band=FLOAT  max distance in meters from the obstacle boundary a biased
                         candidate is drawn from (default: 1.0)
    +dry_run=BOOL        geometry/filter/count only -- no simulation, no output (default: false)
    Also accepts any standard Hydra config-group override against the "helhest" base config
    (engine=mujoco, simulation=..., logging=...); rendering is forced headless.

Usage:
    python src/feasibility/lattice_learning/generate_dataset.py +dry_run=true   # cheap CPU check
    python src/feasibility/lattice_learning/generate_dataset.py +n_maps=1 +trials_per_map=2 +chunk=2
    python src/feasibility/lattice_learning/generate_dataset.py                 # M=4, 16/map
    python src/feasibility/lattice_learning/generate_dataset.py +chunk=32       # if the GPU OOMs
"""
from __future__ import annotations

import pathlib
import time

import h5py
import hydra
import numpy as np
from omegaconf import DictConfig
from ostrich import EngineConfig
from ostrich import LoggingConfig
from ostrich import RenderingConfig
from ostrich import SimulationConfig

from helhest import dynamics
from helhest import friction as friction_mod
from helhest.engine import ForwardSimulator
from helhest.engine import RobotParams

from feasibility.comparator.common import cmd_to_wheels
from feasibility.comparator.common import CONFIG_PATH
from feasibility.comparator.common import euler_zyx_to_quat_xyzw
from feasibility.comparator.common import init_warp_device
from feasibility.comparator.common import K_P
from feasibility.comparator.common import OUT_DIR
from feasibility.comparator.common import run_ostrich_batch
from feasibility.comparator.provenance import git_provenance
from feasibility.comparator.provenance import terrain_fields
from feasibility.heightmap import HeightMapReader
from feasibility.lattice_learning.arc import ARC_LEN
from feasibility.lattice_learning.arc import integrate_arc
from feasibility.lattice_learning.arc import KAPPA_MAX
from feasibility.lattice_learning.arc import V_NOM
from feasibility.lattice_learning.patch import patch_overhangs
from feasibility.lattice_learning.patch import PatchSpec
from feasibility.lattice_learning.patch import patch_spec_to_attrs
from feasibility.lattice_learning.patch import sample_patches
from feasibility.lattice_learning.patch import WHEEL_CONTACTS_LOCAL

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

# --- pinned physics constants (design.md section 2a) ---------------------------------------------

OSTRICH_DT = 2.5e-2  # s. The finest dt = 0.1/k dividing both ARC_LEN/V_NOM = 0.5 s and the twin's
# DT = 0.1 s exactly; overridden on `sim_config` in code, never in the shared helhest.yaml (see
# module docstring and design.md section 2a).
ARC_DURATION_S = ARC_LEN / V_NOM  # 0.5 s -- the time the pinned arc itself takes to travel


def _exact_steps(duration_s: float, dt: float) -> int:
    """round(duration_s / dt), asserting the division is exact -- design.md section 2a's whole
    argument is that this must hold, so a near-miss is a bug to raise on, not round away."""
    n = duration_s / dt
    steps = round(n)
    if abs(steps - n) > 1e-6:
        raise ValueError(f"{duration_s}s is not an exact multiple of dt={dt}s ({n} steps)")
    return steps


T_RECORD_OSTRICH = _exact_steps(ARC_DURATION_S, OSTRICH_DT)  # 20

# --- hyperparameters (module constants -- override any of them with a Hydra `+key=value`) --------

DEFAULT_N_MAPS = 4
DEFAULT_TRIALS_PER_MAP = 16
DEFAULT_SEED = 0
DEFAULT_MAPS_DIR = "assets/large_box_random/0"  # heightmap/create_large_box_obstacles.py's series
# -- the same starvation fix grid_learning_2 uses (design.md section 7c)
MAP_GLOB = "*.png"

DEFAULT_WARMUP_S = 0.375  # s -- design.md section 2's "measured quantity": run a pilot batch,
# find the step by which body speed is within 2% of v_nom and yaw rate within 2% of v_nom*kappa,
# take that with ~1.7x margin; design.md expects order 12-18 ostrich steps (~0.3-0.45s) at
# OSTRICH_DT=2.5e-2. NOT independently re-measured against ostrich in this checkout -- this is a
# placeholder within the expected range, exposed as +warmup_s= so the real pilot measurement can
# be plugged in with no code change.
DEFAULT_SETTLE_STEPS = 15  # ostrich steps dropping onto the terrain at zero command, paid once
# per chunk, BEFORE the warm-up window -- learning/generate_dataset_utils.py measured 12 steps at
# ostrich dt=3e-2 (~0.36s of physical settle time); scaled to this dataset's finer OSTRICH_DT to
# cover the same physical time, not independently re-measured here.
DEFAULT_CHUNK = 64  # trials per ostrich model build -- a tuning knob, not a hard limit (design.md
# section 7c: "raise chunk hard" once this is validated on real hardware; kept modest here as a
# default that fits a small GPU).
DEFAULT_ROUTER_CELL = 0.24  # m -- demos/navigate_partial_view.py's lat_coarsen=4 example (design.md
# section 6a); only used to derive xy_jitter = this/2 (section 1c)
DEFAULT_N_THETA = 24  # the router's own heading bin count; only used to derive yaw_jitter = pi/this
DEFAULT_NEAR_OBSTACLE_FRAC = 0.0  # fraction of each map's spawns biased towards the obstacle
# boundary instead of drawn uniformly -- 0.0 reproduces the original unbiased sampling exactly;
# see sample_spawn_poses' docstring for why this knob exists (measured 1-3% "arc lands on the
# obstacle" rate under uniform sampling, regardless of the obstacle's own size).
DEFAULT_NEAR_OBSTACLE_BAND = 1.0  # m -- max distance from the obstacle boundary a biased
# candidate is drawn from (see sample_spawn_poses)

OBSTACLE_MARGIN_FRACTION = 0.15  # a footprint point counts as "on an obstacle" once its height
# clears this fraction of the way from the terrain's median height to its max -- same convention
# and same value as learning/generate_dataset_utils.py and grid_learning_2/generate_dataset.py's
# own re-derivations (design.md section 11a): re-derived, not imported, but not re-tuned either.

MAX_SPAWN_DISPLACEMENT = 3.0 * ARC_LEN  # m -- design.md section 7b: a final pose farther than
# this from the arc's own origin t0_pose is an implausible/diverged solve, not a large-but-real
# collision displacement (the arc only travels ARC_LEN=0.3m nominally).

SWEPT_SAMPLES = 6  # points sampled along the arc's own curve for the swept_clear reporting split
# (design.md section 7b/8) -- each is settled and checked against the SAME feasibility criterion
# costtogo.py's _feasibility_kernel uses, at the arc's OWN fixed heading (matching
# _relax_lattice_pose_kernel's blocked[..., t] convention, t held at the sweep's outer heading
# rather than the locally-varying arc heading). Re-derived rather than importing that kernel,
# which is embedded in CostToGo's captured-graph machinery and not meant to be called per-trial.


def resolve_path(arg: str) -> pathlib.Path:
    """`+maps_dir=` -> a loadable path: absolute passes through, else resolves against the repo
    root (matches every heightmap/create_*.py generator's own asset layout)."""
    p = pathlib.Path(arg)
    return p if p.is_absolute() else REPO_ROOT / p


def select_maps(maps_dir: pathlib.Path, n_maps: int, rng: np.random.Generator) -> list[pathlib.Path]:
    """`n_maps` heightmap stems drawn without replacement from every PNG in `maps_dir`, sorted
    first so a given seed always picks the same maps regardless of filesystem ordering."""
    candidates = sorted(p.with_suffix("") for p in maps_dir.glob(MAP_GLOB))
    if len(candidates) < n_maps:
        raise ValueError(
            f"{maps_dir} holds {len(candidates)} map(s) matching {MAP_GLOB}, need {n_maps}. "
            f"Generate more with: python src/feasibility/heightmap/create_large_box_obstacles.py "
            f"--n {n_maps}"
        )
    return [candidates[i] for i in rng.choice(len(candidates), size=n_maps, replace=False)]


def obstacle_height_threshold(terrain: HeightMapReader) -> float:
    """Height above which a footprint point counts as "on an obstacle" -- see
    OBSTACLE_MARGIN_FRACTION."""
    baseline = float(np.median(terrain.H))
    return baseline + OBSTACLE_MARGIN_FRACTION * (terrain.max_z - baseline)


def footprint_clear(terrain: HeightMapReader, poses: np.ndarray, threshold: float) -> np.ndarray:
    """[n] bool -- True where `poses` [n, 3] = (x, y, yaw)'s wheel-contact + body-center footprint
    (WHEEL_CONTACTS_LOCAL, patch.py's geometry) samples entirely below `threshold`."""
    x, y, yaw = poses[:, 0], poses[:, 1], poses[:, 2]
    c, s = np.cos(yaw), np.sin(yaw)
    local_x = np.append(WHEEL_CONTACTS_LOCAL[:, 0], 0.0)
    local_y = np.append(WHEEL_CONTACTS_LOCAL[:, 1], 0.0)
    wx = x[:, None] + c[:, None] * local_x - s[:, None] * local_y
    wy = y[:, None] + s[:, None] * local_x + c[:, None] * local_y
    heights = np.asarray(terrain.sample(wx, wy), dtype=np.float64)
    return heights.max(axis=1) <= threshold


def _obstacle_boundary_points(terrain: HeightMapReader, threshold: float) -> np.ndarray:
    """[k, 2] world (x, y) cell centers on the obstacle/flat-ground boundary (4-connected: an
    obstacle cell -- H > threshold -- with at least one non-obstacle neighbor). Local
    re-derivation of heightmap/create_large_box_obstacles.py's `accessible_frontier_score`
    boundary mask (without its access-limit restriction, which is specific to grid_learning's
    fixed lattice and irrelevant here) -- same re-derive-don't-import convention this module
    already uses for `footprint_clear`/`obstacle_height_threshold`."""
    obstacle = terrain.H > threshold
    boundary = np.zeros_like(obstacle)
    boundary[:-1, :] |= obstacle[:-1, :] & ~obstacle[1:, :]
    boundary[1:, :] |= obstacle[1:, :] & ~obstacle[:-1, :]
    boundary[:, :-1] |= obstacle[:, :-1] & ~obstacle[:, 1:]
    boundary[:, 1:] |= obstacle[:, 1:] & ~obstacle[:, :-1]
    ys_idx, xs_idx = np.nonzero(boundary)
    x = terrain.x0 + (xs_idx + 0.5) * terrain.cell
    y = terrain.y0 + (ys_idx + 0.5) * terrain.cell
    return np.column_stack([x, y])


def sample_spawn_poses(
    terrain: HeightMapReader,
    spec: PatchSpec,
    n: int,
    rng: np.random.Generator,
    *,
    near_obstacle_frac: float = 0.0,
    near_obstacle_band: float = 1.0,
) -> np.ndarray:
    """`n` continuous (x, y, yaw) poses on `terrain`, rejection-sampled against the obstacle
    footprint filter AND `spec`'s overhang radius (design.md section 7b/7c) so a valid row's patch
    never needs `HeightMapReader.sample`'s clamp-at-the-edge fallback. The sampling square is
    sized against THIS map's own extent, not a fixed SPAWN_LIMIT (design.md section 7c: a
    0.3-0.6 m arc needs far less clearance than learning/'s 10x10 m spawn square).

    `near_obstacle_frac` (0-1, default 0.0 -- exactly reproduces the old unbiased behaviour)
    steers that fraction of `n` poses towards the obstacle instead of drawing them uniformly over
    the whole map: measured empirically, uniform sampling only lands an arc's own endpoint on the
    (single, small) obstacle 1-3% of the time, since that regime is gated by spawn-to-obstacle
    PROXIMITY, not by obstacle size (see the create_rough_terrain_plus_boxes.py discussion this
    knob comes from). Biased candidates are drawn within `near_obstacle_band` meters of a random
    point on the obstacle's own boundary (`_obstacle_boundary_points`) and pass through the EXACT
    SAME footprint_clear/patch_overhangs filters as a uniform candidate, so biasing can only
    change which valid poses get sampled, never weaken validity. A map with no obstacle at all
    (an all-flat threshold) falls back to uniform sampling for that share too."""
    if not 0.0 <= near_obstacle_frac <= 1.0:
        raise ValueError(f"near_obstacle_frac must be in [0, 1], got {near_obstacle_frac}")
    margin = spec.reach + 0.1  # m, a little slack beyond the exact overhang radius
    x_lo, x_hi = terrain.x0 + margin, terrain.x0 + terrain.nx * terrain.cell - margin
    y_lo, y_hi = terrain.y0 + margin, terrain.y0 + terrain.ny * terrain.cell - margin
    if x_hi <= x_lo or y_hi <= y_lo:
        raise ValueError(
            f"terrain is too small for a patch reach of {spec.reach:.2f} m with margin "
            f"(extent {terrain.nx * terrain.cell:.2f} x {terrain.ny * terrain.cell:.2f} m)"
        )
    threshold = obstacle_height_threshold(terrain)
    boundary_xy = (
        _obstacle_boundary_points(terrain, threshold) if near_obstacle_frac > 0.0 else None
    )

    def propose_uniform(count: int) -> np.ndarray:
        x = rng.uniform(x_lo, x_hi, size=count)
        y = rng.uniform(y_lo, y_hi, size=count)
        yaw = rng.uniform(0.0, 2.0 * np.pi, size=count)
        return np.column_stack([x, y, yaw])

    def propose_near_obstacle(count: int) -> np.ndarray:
        if boundary_xy is None or len(boundary_xy) == 0:
            return propose_uniform(count)  # no obstacle on this map -- nothing to bias towards
        pick = rng.integers(0, len(boundary_xy), size=count)
        r = rng.uniform(0.0, near_obstacle_band, size=count)
        theta = rng.uniform(0.0, 2.0 * np.pi, size=count)
        x = np.clip(boundary_xy[pick, 0] + r * np.cos(theta), x_lo, x_hi)
        y = np.clip(boundary_xy[pick, 1] + r * np.sin(theta), y_lo, y_hi)
        yaw = rng.uniform(0.0, 2.0 * np.pi, size=count)
        return np.column_stack([x, y, yaw])

    def accept(propose, count: int) -> np.ndarray:
        accepted = np.empty((0, 3), dtype=np.float64)
        for _ in range(20):  # bounded retries, same convention as generate_dataset_utils.py
            missing = count - len(accepted)
            if missing <= 0:
                break
            candidates = propose(2 * missing)
            ok = footprint_clear(terrain, candidates, threshold) & ~patch_overhangs(
                terrain, candidates, spec
            )
            accepted = np.concatenate([accepted, candidates[ok]])
        return accepted[:count]

    n_near = round(n * near_obstacle_frac)
    near = accept(propose_near_obstacle, n_near) if n_near else np.empty((0, 3), dtype=np.float64)
    uniform = accept(propose_uniform, n - n_near)
    accepted = np.concatenate([near, uniform])
    if len(accepted) < n:
        raise ValueError(
            f"sample_spawn_poses: only found {len(accepted)}/{n} clear poses "
            f"({len(near)}/{n_near} near-obstacle, {len(uniform)}/{n - n_near} uniform) after 20 "
            "rejection-sampling rounds each -- check the map's obstacle coverage / extent."
        )
    rng.shuffle(accepted)  # so chunk order never correlates with which regime a pose came from
    return accepted


def _quat_to_yaw(q: np.ndarray) -> np.ndarray:
    """[..., 4] (qx,qy,qz,qw) -> yaw [...] (rad) -- same formula as comparator.common's private
    `_quat_yaw`, re-derived here to avoid importing a leading-underscore symbol."""
    qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def settle_batch(
    terrain: HeightMapReader, poses: np.ndarray, mu: float, device: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Static (zero-command) settle of the twin at `poses` [n, 3] = (x, y, yaw) -- design.md
    section 1b's producer of (z, pitch, roll), and the ONLY use of `ForwardSimulator` in this
    module (see the module docstring: no dynamic twin rollout is run here). `n_steps=1` + a
    zeroed `target_wheel_omega` reproduces exactly `costtogo.py`'s own `settle_sim` convention:
    row 0 of `derived` (the pre-step state) IS the static settle, independent of n_steps or
    command (see costtogo.py's `_feasibility_kernel` comment "row 0 = the static settle").
    Returns (derived [n, 3] = (z, pitch, roll), residual [n], clearance [n])."""
    elevation, grid = terrain.to_hstack(device)
    mu_xlim = (terrain.x0, terrain.x0 + (terrain.nx - 1) * terrain.cell)
    mu_ylim = (terrain.y0, terrain.y0 + (terrain.ny - 1) * terrain.cell)
    mu_field = friction_mod.uniform(mu, xlim=mu_xlim, ylim=mu_ylim, cell=terrain.cell)
    n = poses.shape[0]
    sim = ForwardSimulator(
        dynamics.robot_params(), dynamics.execution_solver(), grid, batch_size=n, n_steps=1,
        device=device,
    )
    sim.set_terrain(elevation)
    sim.set_friction(mu_field)
    sim.start_pose.assign(np.ascontiguousarray(poses, np.float32))
    sim.target_wheel_omega.zero_()
    sim.rollout_launch()
    derived = sim.derived.numpy()[0]  # [n, 3]
    residual = sim.residual.numpy()[0]  # [n]
    clearance = sim.clearance.numpy()[0]  # [n]
    return derived, residual, clearance


def settle_feasible(
    derived: np.ndarray, residual: np.ndarray, clearance: np.ndarray, robot: RobotParams
) -> np.ndarray:
    """[n] bool -- reproduces costtogo.py's `_feasibility_kernel` OR, in numpy: direction-aware
    envelope (climb = nose-up = NEGATIVE pitch) plus residual/clearance thresholds."""
    pitch, roll = derived[:, 1], derived[:, 2]
    over_envelope = (
        (np.abs(roll) > robot.max_roll)
        | (pitch < -robot.max_pitch_up)
        | (pitch > robot.max_pitch_down)
    )
    return ~(over_envelope | (residual > robot.resid_tol) | (clearance < robot.clear_margin))


def swept_clear_batch(
    terrain: HeightMapReader, t0_pose: np.ndarray, kappa: np.ndarray, mu: float, device: str,
    robot: RobotParams,
) -> np.ndarray:
    """[n] bool -- see SWEPT_SAMPLES' comment. Samples SWEPT_SAMPLES positions along the exact arc
    from `t0_pose` [n, 3] at curvature `kappa` [n], holds heading at `t0_pose`'s own yaw for every
    sample (the router's own convention), settles all of them at once, and ANDs feasibility."""
    n = t0_pose.shape[0]
    ss = np.linspace(0.0, ARC_LEN, SWEPT_SAMPLES)
    poses = np.stack([integrate_arc(t0_pose, kappa, s) for s in ss], axis=0)  # [S, n, 3]
    poses[..., 2] = t0_pose[None, :, 2]
    derived, residual, clearance = settle_batch(terrain, poses.reshape(-1, 3), mu, device)
    feasible = settle_feasible(derived, residual, clearance, robot).reshape(SWEPT_SAMPLES, n)
    return feasible.all(axis=0)


def simulate_map(
    terrain: HeightMapReader,
    n: int,
    rng: np.random.Generator,
    *,
    spec: PatchSpec,
    xy_jitter: float,
    yaw_jitter: float,
    sim_config: SimulationConfig,
    render_config: RenderingConfig,
    engine_config: EngineConfig,
    logging_config: LoggingConfig,
    chunk: int,
    settle_steps: int,
    warmup_s: float,
    kappa_max: float,
    mu: float,
    device: str,
    near_obstacle_frac: float = 0.0,
    near_obstacle_band: float = 1.0,
) -> dict[str, np.ndarray]:
    """Replays `n` continuous (spawn pose, kappa) trials on ONE terrain, `chunk` at a time, and
    returns every per_variant/ostrich array design.md section 7b's schema needs (minus the
    deferred hstack diagnostic -- see the module docstring). See the module docstring for the
    warm-start / reference / swept_clear pieces this stitches together."""
    robot = RobotParams()
    spawn_pose = sample_spawn_poses(
        terrain, spec, n, rng,
        near_obstacle_frac=near_obstacle_frac, near_obstacle_band=near_obstacle_band,
    )
    kappa = rng.uniform(-kappa_max, kappa_max, size=n).astype(np.float32)

    w_o = round(warmup_s / OSTRICH_DT)
    T_o = w_o + T_RECORD_OSTRICH

    t0_pose = np.zeros((n, 3), dtype=np.float64)
    belief_pose = np.zeros((n, 3), dtype=np.float64)
    arc_end_pose = np.zeros((n, 3), dtype=np.float64)
    ref_pose = np.zeros((n, 7), dtype=np.float32)
    patch = np.zeros((n, spec.ny * spec.nx), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    swept_clear = np.zeros(n, dtype=bool)

    ostrich_pose = np.zeros((T_RECORD_OSTRICH, n, 7), dtype=np.float32)
    ostrich_wheel_qd = np.zeros((T_RECORD_OSTRICH, n, 3), dtype=np.float32)
    ostrich_cmd = np.zeros((T_RECORD_OSTRICH, n, 3), dtype=np.float32)

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        b = end - start
        t0_chunk_t = time.time()
        sim_config.num_worlds = b  # mutated in place per chunk, like every sibling generator's
        # HelhestBatchSimulator.build_model cross-checks it against spawn_pose's row count
        spawn_chunk = spawn_pose[start:end]
        kappa_chunk = kappa[start:end]
        omega_chunk = V_NOM * kappa_chunk
        wheels_chunk = np.stack(
            [cmd_to_wheels(V_NOM, om) for om in omega_chunk]
        ).astype(np.float32)  # [b, 3]

        ostrich_setpoints = np.tile(wheels_chunk[None], (T_o, 1, 1))  # [T_o, b, 3]

        pose_log, wheel_qd_o = run_ostrich_batch(
            sim_config, render_config, engine_config, logging_config, terrain,
            ostrich_setpoints, mu, spawn_chunk, settle_steps,
        )
        t0_xyyaw = np.column_stack(
            [pose_log[w_o - 1, :, 0], pose_log[w_o - 1, :, 1], _quat_to_yaw(pose_log[w_o - 1, :, 3:7])]
        )  # [b, 3] -- the arc's true origin (design.md section 1c)

        arc_end_chunk = integrate_arc(t0_xyyaw, kappa_chunk, ARC_LEN)  # [b, 3]

        dx = rng.uniform(-xy_jitter, xy_jitter, size=b)
        dy = rng.uniform(-xy_jitter, xy_jitter, size=b)
        dyaw = rng.uniform(-yaw_jitter, yaw_jitter, size=b)
        belief_chunk = t0_xyyaw + np.stack([dx, dy, dyaw], axis=-1)  # design.md section 1c

        patch_chunk = sample_patches(terrain, belief_chunk, spec).reshape(b, -1)

        endpoint_derived, endpoint_residual, endpoint_clearance = settle_batch(
            terrain, arc_end_chunk, mu, device
        )
        ref_quat = euler_zyx_to_quat_xyzw(
            arc_end_chunk[:, 2], endpoint_derived[:, 1], endpoint_derived[:, 2]
        )
        ref_pose_chunk = np.concatenate(
            [arc_end_chunk[:, :2], endpoint_derived[:, :1], ref_quat], axis=-1
        ).astype(np.float32)

        swept_clear_chunk = swept_clear_batch(terrain, t0_xyyaw, kappa_chunk, mu, device, robot)

        ostrich_final_xy = pose_log[-1, :, :2]
        displacement = np.linalg.norm(ostrich_final_xy - t0_xyyaw[:, :2], axis=1)
        finite = (
            np.isfinite(pose_log[-1]).all(axis=1)
            & np.isfinite(t0_xyyaw).all(axis=1)
            & np.isfinite(ref_pose_chunk).all(axis=1)
        )
        settle_ok = settle_feasible(endpoint_derived, endpoint_residual, endpoint_clearance, robot)
        displacement_ok = displacement <= MAX_SPAWN_DISPLACEMENT
        overhang_ok = ~patch_overhangs(terrain, belief_chunk, spec)
        valid_chunk = finite & settle_ok & displacement_ok & overhang_ok

        sl = slice(start, end)
        t0_pose[sl] = t0_xyyaw
        belief_pose[sl] = belief_chunk
        arc_end_pose[sl] = arc_end_chunk
        ref_pose[sl] = ref_pose_chunk
        patch[sl] = patch_chunk
        valid[sl] = valid_chunk
        swept_clear[sl] = swept_clear_chunk

        ostrich_pose[:, sl] = pose_log[w_o:]
        ostrich_wheel_qd[:, sl] = wheel_qd_o[w_o:]
        ostrich_cmd[:, sl] = ostrich_setpoints[w_o:]

        n_bad = int((~valid_chunk).sum())
        bad = f", {n_bad} invalid" if n_bad else ""
        print(f"    trials {start}..{end - 1} ({b}) done in {time.time() - t0_chunk_t:.1f}s{bad}")

    return dict(
        spawn_pose=spawn_pose.astype(np.float32),
        t0_pose=t0_pose.astype(np.float32),
        belief_pose=belief_pose.astype(np.float32),
        patch=patch,
        kappa=kappa,
        arc_end_pose=arc_end_pose.astype(np.float32),
        ref_pose=ref_pose,
        valid=valid,
        swept_clear=swept_clear,
        ostrich_pose=ostrich_pose,
        ostrich_wheel_qd=ostrich_wheel_qd,
        ostrich_cmd=ostrich_cmd,
    )


def _write_fields(group: h5py.Group, fields: dict[str, np.ndarray]) -> None:
    """Write each array in `fields` as a compressed dataset of `group` -- gzip/4, matching
    comparator.provenance's own convention. Its `_write_group` is private, so this is a local
    restatement rather than an import, the same choice grid_learning_2/generate_dataset.py's own
    writer makes for the identical reason."""
    for name, arr in fields.items():
        arr = np.asarray(arr)
        if arr.dtype.kind == "U":
            group.create_dataset(name, data=arr.astype(object), dtype=h5py.string_dtype())
        elif arr.shape == ():
            group.create_dataset(name, data=arr)
        else:
            group.create_dataset(name, data=arr, compression="gzip", compression_opts=4)


def write_arc_dataset(
    path: pathlib.Path,
    *,
    root: dict[str, object],
    per_variant: dict[str, np.ndarray],
    terrain_entries: list[tuple[pathlib.Path, HeightMapReader]],
    ostrich: dict[str, np.ndarray],
) -> None:
    """Writes dataset_arc_*.h5. A local writer rather than
    `comparator.provenance.write_comparison`: that schema hard-requires BOTH an `ostrich/` and an
    `hstack/` group, and this generator never runs the twin's dynamic rollout (see the module
    docstring). `terrain_fields()`/`git_provenance()` are reused verbatim, so terrain embedding
    and git provenance still can't drift from `write_comparison`'s own files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for name, value in root.items():
            f.attrs[name] = value

        _write_fields(f, per_variant)

        grp_terrain = f.create_group("terrain")
        _write_fields(grp_terrain, terrain_fields(terrain_entries))

        grp_git = f.create_group("git")
        for name, value in git_provenance().items():
            grp_git.attrs[name] = value

        grp_ostrich = f.create_group("ostrich")
        fields = dict(ostrich)
        grp_ostrich.attrs["dt"] = fields.pop("dt")
        _write_fields(grp_ostrich, fields)

    print(f"saved {path}")


def generate(cfg: DictConfig) -> None:
    n_maps = int(cfg.get("n_maps", DEFAULT_N_MAPS))
    trials_per_map = int(cfg.get("trials_per_map", DEFAULT_TRIALS_PER_MAP))
    seed = int(cfg.get("seed", DEFAULT_SEED))
    maps_dir = resolve_path(str(cfg.get("maps_dir", DEFAULT_MAPS_DIR)))
    kappa_max = float(cfg.get("kappa_max", KAPPA_MAX))
    warmup_s = float(cfg.get("warmup_s", DEFAULT_WARMUP_S))
    settle_steps = int(cfg.get("settle_steps", DEFAULT_SETTLE_STEPS))
    chunk = int(cfg.get("chunk", DEFAULT_CHUNK))
    mu = float(cfg.get("mu", 0.8))
    device = str(cfg.get("device", "cuda:0"))
    router_cell = float(cfg.get("router_cell", DEFAULT_ROUTER_CELL))
    n_theta = int(cfg.get("n_theta", DEFAULT_N_THETA))
    near_obstacle_frac = float(cfg.get("near_obstacle_frac", DEFAULT_NEAR_OBSTACLE_FRAC))
    near_obstacle_band = float(cfg.get("near_obstacle_band", DEFAULT_NEAR_OBSTACLE_BAND))
    dry_run = bool(cfg.get("dry_run", False))

    spec = PatchSpec()
    xy_jitter = router_cell / 2.0  # design.md section 1c
    yaw_jitter = np.pi / n_theta

    print(f"[arc]      v_nom={V_NOM} m/s  arc_len={ARC_LEN} m  kappa in [-{kappa_max}, {kappa_max}] "
          f"1/m  duration={ARC_DURATION_S}s -> {T_RECORD_OSTRICH} ostrich steps")
    print(f"[warmup]   {warmup_s}s -> {round(warmup_s / OSTRICH_DT)} ostrich steps, entered "
          f"already moving at v_nom")
    print(f"[patch]    {spec.ny}x{spec.nx} cells @ {spec.cell} m, reference={spec.reference}")
    print(f"[jitter]   xy=+-{xy_jitter:.4f} m (router_cell={router_cell}), "
          f"yaw=+-{yaw_jitter:.4f} rad (n_theta={n_theta})")

    rng = np.random.default_rng(seed)
    map_paths = select_maps(maps_dir, n_maps, rng)
    n = n_maps * trials_per_map
    print(f"[maps]     {n_maps} from {maps_dir}, {trials_per_map} trial(s) each -> {n} rows")

    if dry_run:
        # Cheap CPU-only self-check: spawn sampling + filters on a synthetic flat map, and on the
        # first REAL map if one is available -- no ostrich rollout, no GPU kernel launched.
        flat = HeightMapReader.flat(xlim=(-8.0, 8.0), ylim=(-8.0, 8.0))
        flat_spawns = sample_spawn_poses(flat, spec, 16, rng)
        assert flat_spawns.shape == (16, 3)
        assert not patch_overhangs(flat, flat_spawns, spec).any(), "flat ground must never overhang"
        real = HeightMapReader.load(map_paths[0])
        real_spawns = sample_spawn_poses(real, spec, 16, rng)
        threshold = obstacle_height_threshold(real)
        assert footprint_clear(real, real_spawns, threshold).all()
        assert not patch_overhangs(real, real_spawns, spec).any()
        assert T_RECORD_OSTRICH * OSTRICH_DT == ARC_DURATION_S
        print(f"[dry-run]  {map_paths[0].name}: 16/16 sampled spawns clear + non-overhanging, "
              "step-count/duration self-checks ok, nothing simulated")
        return

    init_warp_device(device)

    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)
    render_config.vis_type = "null"  # headless

    # design.md section 2a: override in code, never in the shared helhest.yaml (submodule_test's
    # characterization baseline depends on that file staying untouched).
    sim_config.target_timestep_seconds = OSTRICH_DT

    per_variant_fields: dict[str, list[np.ndarray]] = {}
    ostrich_fields: dict[str, list[np.ndarray]] = {}
    map_index_all, map_path_all = [], []
    terrain_entries: list[tuple[pathlib.Path, HeightMapReader]] = []

    for m, p in enumerate(map_paths):
        print(f"[map {m + 1}/{n_maps}] {p.name}")
        terrain = HeightMapReader.load(p)
        result = simulate_map(
            terrain, trials_per_map, rng, spec=spec, xy_jitter=xy_jitter, yaw_jitter=yaw_jitter,
            sim_config=sim_config, render_config=render_config, engine_config=engine_config,
            logging_config=logging_config, chunk=chunk, settle_steps=settle_steps,
            warmup_s=warmup_s, kappa_max=kappa_max, mu=mu, device=device,
            near_obstacle_frac=near_obstacle_frac, near_obstacle_band=near_obstacle_band,
        )
        for key in ("spawn_pose", "t0_pose", "belief_pose", "patch", "kappa", "arc_end_pose",
                    "ref_pose", "valid", "swept_clear"):
            per_variant_fields.setdefault(key, []).append(result[key])
        for key in ("ostrich_pose", "ostrich_wheel_qd", "ostrich_cmd"):
            ostrich_fields.setdefault(key, []).append(result[key])
        map_index_all.append(np.full(trials_per_map, m, dtype=np.int64))
        map_path_all.extend([str(p)] * trials_per_map)
        terrain_entries.extend([(p, terrain)] * trials_per_map)

        n_valid = int(result["valid"].sum())
        print(f"    -> {n_valid}/{trials_per_map} valid, "
              f"{int(result['swept_clear'].sum())}/{trials_per_map} swept-clear")

    per_variant = {k: np.concatenate(v, axis=0) for k, v in per_variant_fields.items()}
    per_variant["map_index"] = np.concatenate(map_index_all, axis=0)
    per_variant["map_path"] = np.array(map_path_all)
    ostrich = {k.removeprefix("ostrich_"): np.concatenate(v, axis=1) for k, v in ostrich_fields.items()}
    ostrich["dt"] = OSTRICH_DT
    ostrich["t"] = np.arange(T_RECORD_OSTRICH, dtype=np.float32) * OSTRICH_DT

    tag = f"{maps_dir.parent.name}{maps_dir.name}" if maps_dir.name.isdigit() else maps_dir.name
    out_path = OUT_DIR / f"dataset_arc_{tag}_M{n_maps}_R{trials_per_map}_seed{seed}.h5"
    write_arc_dataset(
        out_path,
        root=dict(
            v_nom=V_NOM, arc_len=ARC_LEN, min_turn_radius=0.5,  # design.md section 4c's pinned
            # guard trio -- see arc.py; kept literal here so this file has no import-time
            # dependency beyond arc.py's already-imported constants
            kappa_min=-kappa_max, kappa_max=kappa_max,
            warmup_s=warmup_s, settle_steps=settle_steps,
            xy_jitter=xy_jitter, yaw_jitter=yaw_jitter, router_cell=router_cell, n_theta=n_theta,
            near_obstacle_frac=near_obstacle_frac, near_obstacle_band=near_obstacle_band,
            maps_dir=str(maps_dir), map_glob=MAP_GLOB,
            mu=mu, k_p=K_P,
            ostrich_dt=OSTRICH_DT,
            n_maps=n_maps, trials_per_map=trials_per_map, n=n, seed=seed,
            **patch_spec_to_attrs(spec),
        ),
        per_variant=per_variant,
        terrain_entries=terrain_entries,
        ostrich=ostrich,
    )

    n_valid = int(per_variant["valid"].sum())
    n_swept = int(per_variant["swept_clear"].sum())
    print("=" * 60)
    print(f"[summary]  {n} trials across {n_maps} maps")
    print(f"  valid       {n_valid:>7d}  ({100 * n_valid / n:5.1f}%)")
    print(f"  swept_clear {n_swept:>7d}  ({100 * n_swept / n:5.1f}%)  -- reporting split only")
    print("=" * 60)


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest", version_base=None)
def main(cfg: DictConfig) -> None:
    generate(cfg)


if __name__ == "__main__":
    main()
