"""Generates a lattice_learning divergence dataset: one sample is a single body-frame terrain
patch plus ONE scalar curvature command, replayed as the router's OWN forward-arc primitive
(design.md sections 1, 2, 4, 7) --

    x = (patch [24, 28] @ 0.125 m, kappa)      -- v = V_NOM, L = ARC_LEN are PINNED, not sampled
    y = (e_pos, e_rot)  -- ostrich vs. the ARC-PLUS-SETTLE reference, computed by custom_dataset.py

Three things this generator does that no sibling generator does, all from design.md:

* **Warm start (section 2).** Every trial is entered already moving at V_NOM: `warmup_s` seconds
  of CAPTURED, commanded ostrich rollout are prepended to the setpoint array and sliced off the
  returned log -- `t0_pose` (ostrich's actual pose row `w_o - 1`) is the arc's true origin, not
  the nominal spawn. Ostrich spawns at helhest_stack's static-settle pose at the spawn, stored as
  `spawn_zpr` (absolute z incl. `SPAWN_CLEARANCE`, pitch, roll) -- NOT the level `+0.5 m` spawn
  the other generators use, which ejects the robot when a wheel starts inside a step. The sliced-off pre-roll (the `settle_steps` zero-command drop, then the
  warm-up) is still stored, as `ostrich/preroll_pose`/`preroll_wheel_qd` with negative
  `ostrich/preroll_t`, for `gl_replay_arc.py` to show; nothing in training reads it.
* **`OSTRICH_DT` is re-pinned to 2.5e-2 s** (section 2a), the finest `dt = 0.1/k` that divides
  both `ARC_LEN/V_NOM = 0.5 s` and the twin's `DT = 0.1 s` exactly -- overridden in code on
  `sim_config`, never in `examples/conf/simulation/helhest.yaml` (shared with `submodule_test/`'s
  characterization baseline). Kept a multiple of the twin's own dt even though this generator does
  not run the twin, so a future arc-vs-twin diagnostic pass can replay the same trials.
* **The reference is the arc integrated from `t0_pose`, plus a static settle at its endpoint**
  (section 1b) -- not the twin. `settle.settle_batch()` re-derives `costtogo.py`'s own
  `ForwardSimulator` settle convention (`n_steps=1`, zero command, row 0 of `derived` IS the
  static settle) rather than importing anything from `helhest.planning`, since that machinery is
  embedded in `CostToGo`'s captured-graph setup and isn't meant to be called standalone per trial.
  That static, zero-command, single-step settle (here and in spawn validity) is the ONLY use of
  `helhest_stack`'s `ForwardSimulator` -- never a dynamic rollout. The label itself only ever compares
  ostrich against this arc-plus-settle reference (`custom_dataset.py`); the twin's own dynamic
  trajectory over the arc is not simulated at all in this prototype. Section 1a's arc-vs-twin /
  twin-vs-ostrich diagnostic split is a possible FUTURE addition, not implemented here -- adding
  it means running `comparator.common.run_hstack_batch`'s dynamic rollout alongside ostrich's,
  which this generator deliberately does not pay for yet.

`swept_clear` (section 7b/8, a REPORTING split, not a training filter) approximates
`_relax_lattice_pose_kernel`'s sweep test by settling the robot at several points along the SAME
arc, at the arc's OWN (fixed) heading -- matching the kernel's own convention of indexing
`blocked[..., t]` at the sweep's outer heading, not the locally-varying arc heading.

Trial selection -- spawn pose AND kappa -- lives in this package's `spawn_sampling.py`
(`sample_trials`): a spawn is valid iff the static settle there is feasible, and on targeted maps
`+interact_frac` of the trials are drawn from those whose arc meets terrain (`arc_relief`, stored
per row, together with a per-row `targeted` flag). Maps whose sidecar `category` is in
`+untargeted_categories` (default: rough -- flat and low-amplitude rough ground) are sampled
uniformly.

Deliberately independent of `feasibility.learning`, `feasibility.grid_learning` and
`feasibility.grid_learning_2` (design.md section 11a). `feasibility.comparator` (the batch-rollout
core, and `provenance`'s terrain/git embedding) and `feasibility.heightmap` are shared
infrastructure and ARE imported; so are this package's own `arc.py`/`patch.py`/`settle.py`/
`spawn_sampling.py`, since the dataset schema is a contract between this generator and this
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
    +interact_frac=FLOAT exact share of a targeted map's trials whose arc meets terrain, i.e.
                         arc_relief > interact_relief (default: 0.5); null = uniform on every map
    +interact_relief=FLOAT  m, plane-relative terrain relief along the arc that counts as meeting
                         terrain (default: 0.05)
    +untargeted_categories=[..]  sidecar categories sampled uniformly (default: [rough]; [] for
                         none). Maps without a `category` key are targeted.
    +dry_run=BOOL        trial sampling (settle on CPU) on every selected map, no ostrich, no
                         output (default: false)
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
from feasibility.lattice_learning.settle import settle_batch
from feasibility.lattice_learning.settle import settle_feasible
from feasibility.lattice_learning.spawn_sampling import DEFAULT_INTERACT_FRAC
from feasibility.lattice_learning.spawn_sampling import DEFAULT_INTERACT_RELIEF
from feasibility.lattice_learning.spawn_sampling import DEFAULT_UNTARGETED_CATEGORIES
from feasibility.lattice_learning.spawn_sampling import map_category
from feasibility.lattice_learning.spawn_sampling import sample_trials
from feasibility.lattice_learning.spawn_sampling import SpawnBatch

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
MAX_SPAWN_DISPLACEMENT = 3.0 * ARC_LEN  # m -- design.md section 7b: a final pose farther than
# this from the arc's own origin t0_pose is an implausible/diverged solve, not a large-but-real
# collision displacement (the arc only travels ARC_LEN=0.3m nominally).

SPAWN_CLEARANCE = 0.03  # m -- ostrich spawns at helhest_stack's static-settle pose (z, pitch, roll)
# at the spawn (x, y, yaw), lifted this far along world z. Replaces comparator.common's level
# `terrain(axle center) + 0.5` spawn, which put a wheel INSIDE any step higher than 0.15 m under it
# and got the robot ejected in the first settle step (M50_R16_seed0 trials 42/44: 0.15-0.18 m of
# overlap, 4.5-5.7 m/s). The margin covers the two models' small geometry differences (yaw-binned
# cylinder envelope vs. ostrich's collision cylinder, bilinear grid vs. the triangulated mesh) and
# keeps the drop short (~0.77 m/s landing vs ~1.7 m/s from the old 0.15 m fall).

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


def _quat_to_yaw(q: np.ndarray) -> np.ndarray:
    """[..., 4] (qx,qy,qz,qw) -> yaw [...] (rad) -- same formula as comparator.common's private
    `_quat_yaw`, re-derived here to avoid importing a leading-underscore symbol."""
    qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


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
    interact_frac: float | None = None,
    interact_relief: float = DEFAULT_INTERACT_RELIEF,
) -> tuple[dict[str, np.ndarray], SpawnBatch]:
    """Replays `n` continuous (spawn pose, kappa) trials on ONE terrain, `chunk` at a time, and
    returns every per_variant/ostrich array design.md section 7b's schema needs (minus the
    deferred hstack diagnostic -- see the module docstring), plus the sampler's own summary.
    Trials come from `spawn_sampling.sample_trials` (`interact_frac=None` = untargeted). See the
    module docstring for the warm-start / reference / swept_clear pieces this stitches together."""
    robot = RobotParams()
    w_o = round(warmup_s / OSTRICH_DT)
    T_o = w_o + T_RECORD_OSTRICH

    trials = sample_trials(
        terrain, spec, n, rng, kappa_max=kappa_max, lead=w_o * OSTRICH_DT * V_NOM, mu=mu,
        device=device, interact_frac=interact_frac, interact_relief=interact_relief, robot=robot,
    )
    spawn_pose, kappa = trials.pose, trials.kappa
    # The same static settle sample_trials accepted each spawn on (helhest_stack is bit-exact), kept
    # this time as ostrich's starting pose -- see SPAWN_CLEARANCE.
    spawn_derived, _, _ = settle_batch(terrain, spawn_pose, mu, device)
    spawn_zpr = spawn_derived.astype(np.float64)
    spawn_zpr[:, 0] += SPAWN_CLEARANCE

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
    # Viewer-only: the settle drop followed by the warm-up, i.e. everything before the arc.
    ostrich_preroll_pose = np.zeros((settle_steps + w_o, n, 7), dtype=np.float32)
    ostrich_preroll_wheel_qd = np.zeros((settle_steps + w_o, n, 3), dtype=np.float32)

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

        pose_log, wheel_qd_o, settle_pose, settle_wheel_qd = run_ostrich_batch(
            sim_config, render_config, engine_config, logging_config, terrain,
            ostrich_setpoints, mu, spawn_chunk, settle_steps, record_settle=True,
            spawn_zpr=spawn_zpr[start:end],
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
        ostrich_preroll_pose[:, sl] = np.concatenate([settle_pose, pose_log[:w_o]], axis=0)
        ostrich_preroll_wheel_qd[:, sl] = np.concatenate([settle_wheel_qd, wheel_qd_o[:w_o]], axis=0)

        n_bad = int((~valid_chunk).sum())
        bad = f", {n_bad} invalid" if n_bad else ""
        print(f"    trials {start}..{end - 1} ({b}) done in {time.time() - t0_chunk_t:.1f}s{bad}")

    return dict(
        spawn_pose=spawn_pose.astype(np.float32),
        spawn_zpr=spawn_zpr.astype(np.float32),
        t0_pose=t0_pose.astype(np.float32),
        belief_pose=belief_pose.astype(np.float32),
        patch=patch,
        kappa=kappa,
        arc_relief=trials.arc_relief,
        arc_end_pose=arc_end_pose.astype(np.float32),
        ref_pose=ref_pose,
        valid=valid,
        swept_clear=swept_clear,
        ostrich_pose=ostrich_pose,
        ostrich_wheel_qd=ostrich_wheel_qd,
        ostrich_cmd=ostrich_cmd,
        ostrich_preroll_pose=ostrich_preroll_pose,
        ostrich_preroll_wheel_qd=ostrich_preroll_wheel_qd,
    ), trials


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
    interact_frac_cfg = cfg.get("interact_frac", DEFAULT_INTERACT_FRAC)
    interact_frac = None if interact_frac_cfg is None else float(interact_frac_cfg)
    interact_relief = float(cfg.get("interact_relief", DEFAULT_INTERACT_RELIEF))
    untargeted_categories = tuple(
        str(c) for c in cfg.get("untargeted_categories", DEFAULT_UNTARGETED_CATEGORIES)
    )
    dry_run = bool(cfg.get("dry_run", False))

    def map_interact_frac(stem: pathlib.Path) -> float | None:
        """None (uniform draws) for maps whose sidecar category is untargeted, else the knob."""
        return None if map_category(stem) in untargeted_categories else interact_frac

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
    print(f"[trials]   interact_frac={interact_frac} (arc_relief > {interact_relief} m), "
          f"untargeted categories: {', '.join(untargeted_categories) or 'none'}")

    rng = np.random.default_rng(seed)
    map_paths = select_maps(maps_dir, n_maps, rng)
    n = n_maps * trials_per_map
    print(f"[maps]     {n_maps} from {maps_dir}, {trials_per_map} trial(s) each -> {n} rows")

    if dry_run:
        # CPU-only check: trial sampling (including its static settle, run on CPU) on every
        # selected map -- no ostrich rollout, nothing written.
        assert T_RECORD_OSTRICH * OSTRICH_DT == ARC_DURATION_S
        lead = round(warmup_s / OSTRICH_DT) * OSTRICH_DT * V_NOM
        for p in map_paths:
            frac = map_interact_frac(p)
            trials = sample_trials(
                HeightMapReader.load(p), spec, trials_per_map, rng, kappa_max=kappa_max, lead=lead,
                mu=mu, device="cpu", interact_frac=frac, interact_relief=interact_relief,
            )
            n_int = int((trials.arc_relief > interact_relief).sum())
            short = f", {trials.shortfall} short" if trials.shortfall else ""
            print(f"[dry-run]  {p.name}: {n_int}/{trials_per_map} interacting "
                  f"({'targeted' if trials.targeted else 'untargeted'}{short})")
        print("[dry-run]  sampling + step-count/duration self-checks ok, nothing simulated")
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
    map_index_all, map_path_all, targeted_all = [], [], []
    terrain_entries: list[tuple[pathlib.Path, HeightMapReader]] = []

    for m, p in enumerate(map_paths):
        print(f"[map {m + 1}/{n_maps}] {p.name}")
        terrain = HeightMapReader.load(p)
        result, trials = simulate_map(
            terrain, trials_per_map, rng, spec=spec, xy_jitter=xy_jitter, yaw_jitter=yaw_jitter,
            sim_config=sim_config, render_config=render_config, engine_config=engine_config,
            logging_config=logging_config, chunk=chunk, settle_steps=settle_steps,
            warmup_s=warmup_s, kappa_max=kappa_max, mu=mu, device=device,
            interact_frac=map_interact_frac(p), interact_relief=interact_relief,
        )
        for key in ("spawn_pose", "spawn_zpr", "t0_pose", "belief_pose", "patch", "kappa", "arc_relief",
                    "arc_end_pose", "ref_pose", "valid", "swept_clear"):
            per_variant_fields.setdefault(key, []).append(result[key])
        for key in ("ostrich_pose", "ostrich_wheel_qd", "ostrich_cmd",
                    "ostrich_preroll_pose", "ostrich_preroll_wheel_qd"):
            ostrich_fields.setdefault(key, []).append(result[key])
        map_index_all.append(np.full(trials_per_map, m, dtype=np.int64))
        map_path_all.extend([str(p)] * trials_per_map)
        targeted_all.append(np.full(trials_per_map, trials.targeted, dtype=bool))
        terrain_entries.extend([(p, terrain)] * trials_per_map)

        n_valid = int(result["valid"].sum())
        n_int = int((trials.arc_relief > interact_relief).sum())
        short = f", {trials.shortfall} short of target" if trials.shortfall else ""
        print(f"    -> {n_valid}/{trials_per_map} valid, "
              f"{int(result['swept_clear'].sum())}/{trials_per_map} swept-clear, "
              f"{n_int}/{trials_per_map} interacting "
              f"({'targeted' if trials.targeted else 'untargeted'}{short})")

    per_variant = {k: np.concatenate(v, axis=0) for k, v in per_variant_fields.items()}
    per_variant["map_index"] = np.concatenate(map_index_all, axis=0)
    per_variant["map_path"] = np.array(map_path_all)
    per_variant["targeted"] = np.concatenate(targeted_all, axis=0)
    ostrich = {k.removeprefix("ostrich_"): np.concatenate(v, axis=1) for k, v in ostrich_fields.items()}
    ostrich["dt"] = OSTRICH_DT
    ostrich["t"] = np.arange(T_RECORD_OSTRICH, dtype=np.float32) * OSTRICH_DT
    n_preroll = settle_steps + round(warmup_s / OSTRICH_DT)
    # Negative times, so preroll_t continues straight into t (the arc's first step is t=0).
    ostrich["preroll_t"] = (np.arange(n_preroll, dtype=np.float32) - n_preroll) * OSTRICH_DT

    tag = f"{maps_dir.parent.name}{maps_dir.name}" if maps_dir.name.isdigit() else maps_dir.name
    out_path = OUT_DIR / f"dataset_arc_{tag}_M{n_maps}_R{trials_per_map}_seed{seed}.h5"
    write_arc_dataset(
        out_path,
        root=dict(
            v_nom=V_NOM, arc_len=ARC_LEN, min_turn_radius=0.5,  # design.md section 4c's pinned
            # guard trio -- see arc.py; kept literal here so this file has no import-time
            # dependency beyond arc.py's already-imported constants
            kappa_min=-kappa_max, kappa_max=kappa_max,
            warmup_s=warmup_s, settle_steps=settle_steps, spawn_clearance=SPAWN_CLEARANCE,
            xy_jitter=xy_jitter, yaw_jitter=yaw_jitter, router_cell=router_cell, n_theta=n_theta,
            # NaN = untargeted everywhere (an HDF5 attr cannot hold None)
            interact_frac=np.nan if interact_frac is None else interact_frac,
            interact_relief=interact_relief,
            untargeted_categories=",".join(untargeted_categories),
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
