"""Generates a lattice_learning divergence dataset: one sample is a single body-frame terrain
patch plus ONE commanded body twist, replayed as one of the router's OWN primitives for
ARC_DURATION_S = 0.5 s (design.md sections 1, 2, 4, 7) --

    x = (patch [24, 28] @ 0.125 m, (v_drive, wz_drive))
    y = (e_pos, e_rot)  -- ostrich vs. the PRIMITIVE-PLUS-SETTLE reference, computed by custom_dataset.py

Two primitive kinds, both pinned (arc.py), never sampled beyond their sign/curvature:
  * forward arc: v_drive = V_NOM, wz_drive = V_NOM * kappa, travelling ARC_LEN = the lattice step;
    `kappa` is stored too, for the kappa-input model.
  * pivot (`rotate_in_place` rows): v_drive = 0, wz_drive = +-OMEGA_NOM, turning PIVOT_ANGLE = one
    heading bin in place -- helhest_stack's point-turn primitive (`pivot_cost > 0`). `kappa` is
    NaN: a pivot has no curvature, so only the (v, wz)-input model can train on these rows.
Everything below reads the per-row twist, so the two kinds share one code path; where the text says
"arc", a pivot's arc is the in-place turn and its "arc end" the same xy one bin over.

Three things this generator does that no sibling generator does, all from design.md:

* **Warm start (section 2).** Every trial is entered already moving at its own twist -- driving
  at V_NOM, or for a pivot already spinning at +-OMEGA_NOM: `warmup_s` seconds
  of CAPTURED, commanded ostrich rollout are prepended to the setpoint array and sliced off the
  returned log -- `t0_pose` (ostrich's actual pose row `w_o - 1`) is the arc's true origin, not
  the nominal spawn. Ostrich spawns at helhest_stack's static-settle pose at the spawn, stored as
  `spawn_zpr` (absolute z incl. `SPAWN_CLEARANCE`, pitch, roll) -- NOT the level `+0.5 m` spawn
  the other generators use, which ejects the robot when a wheel starts inside a step. The sliced-off pre-roll (the `settle_steps` zero-command drop, then the
  warm-up) is still stored, as `ostrich/preroll_pose`/`preroll_wheel_qd` with negative
  `ostrich/preroll_t`, for `gl_replay_arc.py` to show; nothing in training reads it.
* **Feedforward slip compensation (`trial.yaw_gain`).** Ideal no-slip wheel speeds do not
  produce the yaw rate they are computed for -- a skid-steer slips, and a point turn is pure slip,
  so ostrich realizes only ~0.49 of any commanded `wz`. Every yaw command is therefore divided by
  `yaw_gain` before it reaches the wheels (`_wheel_setpoints`, which documents the measurement),
  so the robot actually turns what the primitive says. The stored `v_drive`/`wz_drive`, the
  reference, and the model's input all stay NOMINAL -- only the command moves. `yaw_gain: 1.0`
  reproduces the uncompensated behaviour of files written before the key existed; the value used
  is in the root attrs.
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

Trial selection -- spawn pose AND kappa -- lives in this package's `spawn_sampling.py`, and WHICH
strategy runs on which maps, with which params and on what share of the rows, in the config's `mix`
(`dataset_config.py`: validation, the strategy registry, and the allocation of maps and rows).
Entries on the same map category share its maps, so one ramp map can carry `ramp_up` and
`ramp_down` rows. Every strategy requires the static settle to be feasible at the spawn; `uniform`
and `targeted` also require it at the nominal arc end, `ramp_up`, `ramp_down` and `edge` do not.
  - `ramp_up` / `ramp_down` (ramps maps): head-on up a face from before its foot / down a face from
    its plateau, so divergence can be read against the face's slope. Stored per row as `ramp_deg`,
    plus `ramp_s` = the arc origin's position onto the face from its entry point (foot / crest);
    both are NaN for every other row.
  - `edge` (curbs_and_walls maps): arc origins near a height edge, `interact_frac` of the trials
    running into it, `down_frac` of those driving down, the rest climbing up.
  - `uniform` (any map, typically rough): uniform (pose, kappa).
  - `targeted` (any map): uniform proposals, `interact_frac` of the trials meeting terrain.
  - `rotate_in_place` (poles_and_walls / curbs_and_walls maps): PIVOTS next to a pole or wall, no
    wheel touching it during the warm-up spin, `interact_frac` of them sweeping a wheel into it
    during the recorded one.
Every row stores `sampling` (its strategy), `map_category`, `arc_relief`, `interact_dir` (+1 up /
-1 down / 0 none) and a `targeted` flag.

`valid` is the data-quality gate only: finite poses, a plausible displacement, a patch on the map.
It does NOT include the static settle at the arc end -- that is stored separately per row as
`endpoint_blocked` (True where helhest_stack's settle at the true arc end is infeasible, i.e. the
planner would call the edge `blocked`), so custom_dataset.py's `drop_blocked_endpoints` can remove
those rows. The root attr `valid_excludes_endpoint_settle` marks files written this way; older
files folded the endpoint settle into `valid` and have no `endpoint_blocked` column.

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

CLI: ONE argument, the dataset config -- a name in `configs/` or a path to a YAML file. Every
parameter (seed, map dir and counts, warm-up, friction, jitter, chunking, device, dry run, ostrich
Hydra overrides, and the sampling mix with each strategy's params) lives in that file, so the file
alone reproduces a dataset; the h5 embeds its text as the root attr `config_yaml`.
`configs/default.yaml` documents every key; `dataset_config.py` validates it.

Output: outputs/dataset_arc_<config name>_M<n_maps>_R<trials_per_map>_seed<seed>.h5

Usage:
    python src/feasibility/lattice_learning/generate_dataset.py default    # configs/default.yaml
    python src/feasibility/lattice_learning/generate_dataset.py my_config
    python src/feasibility/lattice_learning/generate_dataset.py path/to/my_dataset.yaml
Set `run.dry_run: true` in a copy of a config for a cheap CPU check of the allocation and sampling.
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
import time

import h5py
import hydra
import numpy as np
from hydra import compose
from hydra import initialize_config_dir
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
from feasibility.comparator.common import MU_LAT_RATIO
from feasibility.comparator.common import OUT_DIR
from feasibility.comparator.common import run_ostrich_batch
from feasibility.comparator.provenance import git_provenance
from feasibility.comparator.provenance import terrain_fields
from feasibility.heightmap import HeightMapReader
from feasibility.lattice_learning.arc import ARC_DURATION_S
from feasibility.lattice_learning.arc import ARC_LEN
from feasibility.lattice_learning.arc import integrate_twist
from feasibility.lattice_learning.arc import N_THETA
from feasibility.lattice_learning.arc import OMEGA_NOM
from feasibility.lattice_learning.arc import PIVOT_ANGLE
from feasibility.lattice_learning.arc import V_NOM
from feasibility.lattice_learning.dataset_config import allocate
from feasibility.lattice_learning.dataset_config import AllocatedMap
from feasibility.lattice_learning.dataset_config import check_maps
from feasibility.lattice_learning.dataset_config import ConfigError
from feasibility.lattice_learning.dataset_config import DatasetConfig
from feasibility.lattice_learning.dataset_config import load_config
from feasibility.lattice_learning.dataset_config import MAP_GLOB
from feasibility.lattice_learning.dataset_config import sample_map_mix
from feasibility.lattice_learning.patch import patch_overhangs
from feasibility.lattice_learning.patch import PatchSpec
from feasibility.lattice_learning.patch import patch_spec_to_attrs
from feasibility.lattice_learning.patch import sample_patches
from feasibility.lattice_learning.settle import settle_batch
from feasibility.lattice_learning.settle import settle_feasible
from feasibility.lattice_learning.spawn_sampling import map_metadata
from feasibility.lattice_learning.spawn_sampling import SpawnBatch
from feasibility.lattice_learning.tiled_terrain import tile_offsets
from feasibility.lattice_learning.tiled_terrain import TiledTerrain


# --- pinned physics constants (design.md section 2a) ---------------------------------------------

OSTRICH_DT = 2.5e-2  # s. The finest dt = 0.1/k dividing both ARC_LEN/V_NOM = 0.5 s and the twin's
# DT = 0.1 s exactly; overridden on `sim_config` in code, never in the shared helhest.yaml (see
# module docstring and design.md section 2a). ARC_DURATION_S (arc.py, 0.5 s) is the time one
# primitive takes -- a forward arc or a pivot alike.


def _exact_steps(duration_s: float, dt: float) -> int:
    """round(duration_s / dt), asserting the division is exact -- design.md section 2a's whole
    argument is that this must hold, so a near-miss is a bug to raise on, not round away."""
    n = duration_s / dt
    steps = round(n)
    if abs(steps - n) > 1e-6:
        raise ValueError(f"{duration_s}s is not an exact multiple of dt={dt}s ({n} steps)")
    return steps


T_RECORD_OSTRICH = _exact_steps(ARC_DURATION_S, OSTRICH_DT)  # 20

# Every former hyperparameter (map counts, warm-up, settle steps, chunking, jitter, sampling) now
# lives in the dataset config -- see configs/default.yaml for each value and its rationale.
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


def _quat_to_yaw(q: np.ndarray) -> np.ndarray:
    """[..., 4] (qx,qy,qz,qw) -> yaw [...] (rad) -- same formula as comparator.common's private
    `_quat_yaw`, re-derived here to avoid importing a leading-underscore symbol."""
    qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def swept_clear_batch(
    terrain: HeightMapReader, t0_pose: np.ndarray, v: np.ndarray, wz: np.ndarray, mu: float,
    device: str, robot: RobotParams,
) -> np.ndarray:
    """[n] bool -- see SWEPT_SAMPLES' comment. Samples SWEPT_SAMPLES positions along the exact
    primitive from `t0_pose` [n, 3] under the twist (`v`, `wz`) [n], holds heading at `t0_pose`'s own
    yaw for every sample (the router's own convention), settles all of them at once, and ANDs
    feasibility. A pivot's samples all collapse onto `t0_pose` itself -- exactly the lattice pivot's
    sweep, the cell it stands on."""
    n = t0_pose.shape[0]
    ts = np.linspace(0.0, ARC_DURATION_S, SWEPT_SAMPLES)
    poses = np.stack([integrate_twist(t0_pose, v, wz, t) for t in ts], axis=0)  # [S, n, 3]
    poses[..., 2] = t0_pose[None, :, 2]
    derived, residual, clearance = settle_batch(terrain, poses.reshape(-1, 3), mu, device)
    feasible = settle_feasible(derived, residual, clearance, robot).reshape(SWEPT_SAMPLES, n)
    return feasible.all(axis=0)


@dataclasses.dataclass
class PreparedMap:
    """One map's trials, everything drawn from the rng before any ostrich rollout (see
    `prepare_map`)."""

    terrain: HeightMapReader
    category: str  # the map's sidecar category
    trials: SpawnBatch
    spawn_zpr: np.ndarray  # [n, 3] absolute z (incl. SPAWN_CLEARANCE), pitch, roll
    jitter: np.ndarray  # [n, 3] belief (dx, dy, dyaw), design.md section 1c

    @property
    def n(self) -> int:
        return self.trials.pose.shape[0]


def _wheel_setpoints(v: np.ndarray, wz: np.ndarray, yaw_gain: float) -> np.ndarray:
    """[n, 3] wheel velocity setpoints for the body twists (`v`, `wz`) [n] -- (V_NOM, V_NOM * kappa)
    for an arc, (0, +-OMEGA_NOM) for a pivot -- with the yaw command divided by `yaw_gain`.

    THE FEEDFORWARD SLIP COMPENSATION, and the only place it is applied. `cmd_to_wheels` is ideal
    no-slip differential drive: it computes the wheel speeds that would produce `wz` if nothing
    slipped. A skid-steer always slips, and a point turn is nothing but slip, so ostrich realizes
    only `yaw_gain` of whatever yaw rate is asked for. Measured on flat ground at the friction this
    generator runs (`MU_LAT_RATIO` 0.5, mu 0.8), warm-started, over the recorded window: 0.49 at
    OMEGA_NOM, and flat to +-0.03 across the whole commanded range 0.13 .. 1.4 rad/s -- that
    flatness is what makes ONE scalar the right shape for this correction. Commanding `wz /
    yaw_gain` then lands the realized rotation on the primitive's nominal value: measured
    1.00-1.04x nominal for pivots, 0.985 for the +-1/(2R) arcs, 0.90 for the +-1/R arcs (the
    tightest arcs command ~2.45 rad/s, past where the gain was measured flat, and still fall a
    little short -- a single scalar cannot fix all five curvatures at once).

    Uncompensated, the shortfall is a pure command-side bias with no terrain in it, and it lands
    squarely in the label: flat ground read e_rot = 0.13 rad for a pivot and 0.34 rad for a
    kappa_max arc, the latter 0.9x the TAU_ROT the planner gates on. Compensated it reads 0.015
    and 0.08. What does NOT go away is e_pos: the robot's real centre of rotation sits ~0.19 m from
    where the primitive assumes it, so a correctly-turning pivot still drifts ~0.05 m. That is a
    rigid per-primitive offset and belongs in the router's `prim_dr`/`prim_dc`, not here.

    `yaw_gain = 1.0` restores the uncompensated command every dataset before this key used.

    Only the COMMAND is scaled. The stored `v_drive`/`wz_drive`, the reference the label is
    measured against, and the model's own input all stay NOMINAL -- they are the primitive the
    planner believes in, and compensating them would move the reference along with the robot and
    measure nothing."""
    return np.stack(
        [cmd_to_wheels(float(a), float(b) / yaw_gain) for a, b in zip(v, wz)]
    ).astype(np.float32)


def prepare_map(
    allocated: AllocatedMap, cfg: DatasetConfig, rng: np.random.Generator, *, spec: PatchSpec
) -> PreparedMap:
    """Samples a map's trials -- `counts` rows per mix entry, by `dataset_config.sample_map_mix` --
    the static-settle spawn pose ostrich starts from, and the belief jitter.

    The jitter is drawn HERE, per `run.chunk` slice in dx/dy/dyaw order, after the trials --
    nothing else touches the rng, so the stream (and therefore every trial) is identical whether
    one map or `run.maps_per_build` maps share an ostrich build."""
    robot = RobotParams()
    n = sum(allocated.counts.values())
    chunk, device = cfg.run.chunk, cfg.run.device
    xy_jitter, yaw_jitter = cfg.trial.xy_jitter, cfg.trial.yaw_jitter
    terrain = HeightMapReader.load(allocated.path)
    trials = sample_map_mix(
        terrain, map_metadata(allocated.path), spec, allocated.counts, cfg, rng, lead=lead_m(cfg),
        device=device, robot=robot,
    )
    # The same static settle sample_trials accepted each spawn on (helhest_stack is bit-exact), kept
    # this time as ostrich's starting pose -- see SPAWN_CLEARANCE.
    spawn_derived, _, _ = settle_batch(terrain, trials.pose, cfg.trial.mu, device)
    spawn_zpr = spawn_derived.astype(np.float64)
    spawn_zpr[:, 0] += SPAWN_CLEARANCE

    jitter = np.zeros((n, 3), dtype=np.float64)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        b = end - start
        dx = rng.uniform(-xy_jitter, xy_jitter, size=b)
        dy = rng.uniform(-xy_jitter, xy_jitter, size=b)
        dyaw = rng.uniform(-yaw_jitter, yaw_jitter, size=b)
        jitter[start:end] = np.stack([dx, dy, dyaw], axis=-1)
    return PreparedMap(terrain=terrain, category=allocated.category, trials=trials,
                       spawn_zpr=spawn_zpr, jitter=jitter)


def warmup_steps(cfg: DatasetConfig) -> int:
    """Ostrich steps of warm-up; `trial.warmup_s` must be an exact multiple of OSTRICH_DT."""
    return _exact_steps(cfg.trial.warmup_s, OSTRICH_DT)


def lead_m(cfg: DatasetConfig) -> float:
    """m travelled along the arc's curvature during the warm-up, before the recorded arc."""
    return warmup_steps(cfg) * OSTRICH_DT * V_NOM


def rollout_group(
    maps: list[PreparedMap],
    *,
    sim_config: SimulationConfig,
    render_config: RenderingConfig,
    engine_config: EngineConfig,
    logging_config: LoggingConfig,
    chunk: int,
    settle_steps: int,
    warmup_s: float,
    mu: float,
    yaw_gain: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Runs every trial of `maps` through ostrich, `chunk` worlds per model build, and returns per
    map (pose [settle_steps + T_o, n, 7], wheel_qd [settle_steps + T_o, n, 3]) in that map's own
    coordinates.

    One map: its own `HeightMapReader` is the terrain, exactly the per-map build. Several maps
    (`+maps_per_build`): a `TiledTerrain` places each on its own tile, each trial spawns at its
    map's offset and the offset is subtracted back from the logged x/y, so a chunk can freely mix
    maps."""
    w_o = round(warmup_s / OSTRICH_DT)
    T_o = w_o + T_RECORD_OSTRICH
    if len(maps) == 1:
        terrain = maps[0].terrain
        offsets = np.zeros((1, 2), dtype=np.float64)
    else:
        offsets = tile_offsets([m.terrain for m in maps])
        terrain = TiledTerrain([m.terrain for m in maps], offsets)

    row_offset = np.concatenate([np.repeat(offsets[i : i + 1], m.n, axis=0) for i, m in enumerate(maps)])
    spawn_pose = np.concatenate([m.trials.pose for m in maps]).astype(np.float64)
    spawn_pose[:, :2] += row_offset
    spawn_zpr = np.concatenate([m.spawn_zpr for m in maps])
    wheels = np.concatenate(
        [_wheel_setpoints(m.trials.v, m.trials.wz, yaw_gain) for m in maps]
    )  # [N, 3] -- the COMPENSATED command, see _wheel_setpoints
    n = spawn_pose.shape[0]

    pose = np.zeros((settle_steps + T_o, n, 7), dtype=np.float32)
    wheel_qd = np.zeros((settle_steps + T_o, n, 3), dtype=np.float32)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        b = end - start
        t0_chunk_t = time.time()
        sim_config.num_worlds = b  # mutated in place per chunk, like every sibling generator's
        # HelhestBatchSimulator.build_model cross-checks it against spawn_pose's row count

        # The settle rides in the CAPTURED rollout as `settle_steps` leading rows of zero wheel
        # speed, instead of run_ostrich_batch's own uncaptured settle loop (settle_steps=0 skips
        # it). Same physics per step -- collide, zero wheel target, solve -- but replayed as a
        # CUDA graph: ~6.5 ms/step instead of ~0.6 s/step of Python kernel launches, which made
        # the settle ~2/3 of every map's wall time. Sliced back apart in finish_map.
        setpoints = np.concatenate(
            [np.zeros((settle_steps, b, 3), dtype=np.float32), np.tile(wheels[None, start:end], (T_o, 1, 1))],
            axis=0,
        )
        chunk_pose, chunk_wheel_qd = run_ostrich_batch(
            sim_config, render_config, engine_config, logging_config, terrain, setpoints, mu,
            spawn_pose[start:end], settle_steps=0, spawn_zpr=spawn_zpr[start:end],
        )
        chunk_pose[..., :2] -= row_offset[None, start:end].astype(np.float32)
        pose[:, start:end] = chunk_pose
        wheel_qd[:, start:end] = chunk_wheel_qd
        print(f"    ostrich trials {start}..{end - 1} ({b} worlds, {len(maps)} map(s)) "
              f"done in {time.time() - t0_chunk_t:.1f}s")

    results, row = [], 0
    for m in maps:
        results.append((pose[:, row : row + m.n], wheel_qd[:, row : row + m.n]))
        row += m.n
    return results


def finish_map(
    prepared: PreparedMap,
    full_pose: np.ndarray,
    full_wheel_qd: np.ndarray,
    *,
    spec: PatchSpec,
    chunk: int,
    settle_steps: int,
    warmup_s: float,
    mu: float,
    yaw_gain: float,
    device: str,
) -> dict[str, np.ndarray]:
    """Turns one map's ostrich rollout (`rollout_group`) into every per_variant/ostrich array
    design.md section 7b's schema needs (minus the deferred hstack diagnostic -- see the module
    docstring). See the module docstring for the warm-start / reference / swept_clear pieces this
    stitches together. Uses no rng."""
    robot = RobotParams()
    terrain, trials = prepared.terrain, prepared.trials
    n = prepared.n
    w_o = round(warmup_s / OSTRICH_DT)
    T_o = w_o + T_RECORD_OSTRICH
    spawn_pose, kappa = trials.pose, trials.kappa
    # (v_drive, wz_drive) -- comparator.common's own naming for the commanded body twist -- is the
    # command every row actually got: (V_NOM, V_NOM * kappa) for an arc, (0, +-OMEGA_NOM) for a
    # pivot, whose kappa is NaN. The (v, omega)-input model reads these, not kappa.
    v_drive, wz_drive = trials.v, trials.wz

    t0_pose = np.zeros((n, 3), dtype=np.float64)
    belief_pose = np.zeros((n, 3), dtype=np.float64)
    arc_end_pose = np.zeros((n, 3), dtype=np.float64)
    ref_pose = np.zeros((n, 7), dtype=np.float32)
    patch = np.zeros((n, spec.ny * spec.nx), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    endpoint_blocked = np.zeros(n, dtype=bool)
    swept_clear = np.zeros(n, dtype=bool)

    settle_pose, pose_log = full_pose[:settle_steps], full_pose[settle_steps:]
    settle_wheel_qd, wheel_qd_o = full_wheel_qd[:settle_steps], full_wheel_qd[settle_steps:]
    assert pose_log.shape[0] == T_o

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sl = slice(start, end)
        v_chunk, wz_chunk = v_drive[sl], wz_drive[sl]
        last = pose_log[w_o - 1, sl]
        t0_xyyaw = np.column_stack(
            [last[:, 0], last[:, 1], _quat_to_yaw(last[:, 3:7])]
        )  # [b, 3] -- the primitive's true origin (design.md section 1c)

        # the arc's end, or for a pivot the same xy one heading bin over
        arc_end_chunk = integrate_twist(t0_xyyaw, v_chunk, wz_chunk, ARC_DURATION_S)  # [b, 3]
        belief_chunk = t0_xyyaw + prepared.jitter[sl]  # design.md section 1c

        patch_chunk = sample_patches(terrain, belief_chunk, spec).reshape(end - start, -1)

        endpoint_derived, endpoint_residual, endpoint_clearance = settle_batch(
            terrain, arc_end_chunk, mu, device
        )
        ref_quat = euler_zyx_to_quat_xyzw(
            arc_end_chunk[:, 2], endpoint_derived[:, 1], endpoint_derived[:, 2]
        )
        ref_pose_chunk = np.concatenate(
            [arc_end_chunk[:, :2], endpoint_derived[:, :1], ref_quat], axis=-1
        ).astype(np.float32)

        swept_clear_chunk = swept_clear_batch(terrain, t0_xyyaw, v_chunk, wz_chunk, mu, device, robot)

        ostrich_final_xy = pose_log[-1, sl, :2]
        displacement = np.linalg.norm(ostrich_final_xy - t0_xyyaw[:, :2], axis=1)
        finite = (
            np.isfinite(pose_log[-1, sl]).all(axis=1)
            & np.isfinite(t0_xyyaw).all(axis=1)
            & np.isfinite(ref_pose_chunk).all(axis=1)
        )
        settle_ok = settle_feasible(endpoint_derived, endpoint_residual, endpoint_clearance, robot)
        displacement_ok = displacement <= MAX_SPAWN_DISPLACEMENT
        overhang_ok = ~patch_overhangs(terrain, belief_chunk, spec)
        # the endpoint settle is NOT part of valid -- stored as endpoint_blocked instead, see the
        # module docstring
        valid_chunk = finite & displacement_ok & overhang_ok

        t0_pose[sl] = t0_xyyaw
        belief_pose[sl] = belief_chunk
        arc_end_pose[sl] = arc_end_chunk
        ref_pose[sl] = ref_pose_chunk
        patch[sl] = patch_chunk
        valid[sl] = valid_chunk
        endpoint_blocked[sl] = ~settle_ok
        swept_clear[sl] = swept_clear_chunk

    # what the wheels were actually told to do -- compensated, unlike the nominal v_drive/wz_drive
    ostrich_cmd = np.tile(
        _wheel_setpoints(v_drive, wz_drive, yaw_gain)[None], (T_RECORD_OSTRICH, 1, 1)
    )
    return dict(
        spawn_pose=spawn_pose.astype(np.float32),
        spawn_zpr=prepared.spawn_zpr.astype(np.float32),
        t0_pose=t0_pose.astype(np.float32),
        belief_pose=belief_pose.astype(np.float32),
        patch=patch,
        kappa=kappa,
        v_drive=v_drive,
        wz_drive=wz_drive,
        arc_relief=trials.arc_relief,
        interact_dir=trials.interact_dir,
        ramp_deg=trials.ramp_deg,
        ramp_s=trials.ramp_s,
        sampling=trials.strategy,
        map_category=np.full(n, prepared.category),
        targeted=trials.targeted,
        arc_end_pose=arc_end_pose.astype(np.float32),
        ref_pose=ref_pose,
        valid=valid,
        endpoint_blocked=endpoint_blocked,
        swept_clear=swept_clear,
        ostrich_pose=np.ascontiguousarray(pose_log[w_o:]),
        ostrich_wheel_qd=np.ascontiguousarray(wheel_qd_o[w_o:]),
        ostrich_cmd=ostrich_cmd,
        # Viewer-only: the settle drop followed by the warm-up, i.e. everything before the arc.
        ostrich_preroll_pose=np.concatenate([settle_pose, pose_log[:w_o]], axis=0),
        ostrich_preroll_wheel_qd=np.concatenate([settle_wheel_qd, wheel_qd_o[:w_o]], axis=0),
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


def describe_trials(trials: SpawnBatch) -> str:
    """One-line summary of a map's sampled trials, per strategy, for the per-map progress print."""
    parts = []
    for strategy in dict.fromkeys(trials.strategy.tolist()):
        k = trials.strategy == strategy
        n_up, n_down = int((trials.interact_dir[k] > 0).sum()), int((trials.interact_dir[k] < 0).sum())
        text = f"{strategy} {int(k.sum())}: {n_up} up/{n_down} down"
        n_blocked = int((~trials.endpoint_feasible[k]).sum())
        if n_blocked:
            text += f", {n_blocked} nominal ends blocked"
        on_ramp = k & np.isfinite(trials.ramp_deg)
        if on_ramp.any():
            text += (f", faces {np.nanmin(trials.ramp_deg[on_ramp]):.0f}-"
                     f"{np.nanmax(trials.ramp_deg[on_ramp]):.0f} deg, "
                     f"{int((trials.kappa[on_ramp] == 0).sum())} straight")
            if on_ramp.sum() < k.sum():
                text += f", {int(k.sum() - on_ramp.sum())} fallback"
        pivot = k & (trials.v == 0)
        if pivot.any():
            text += f", pivots {int((trials.wz[pivot] > 0).sum())} CCW/{int((trials.wz[pivot] < 0).sum())} CW"
        parts.append(text)
    short = f" ({trials.shortfall} short)" if trials.shortfall else ""
    return "; ".join(parts) + short


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


def compose_ostrich_config(overrides: tuple[str, ...]) -> tuple[
    SimulationConfig, RenderingConfig, EngineConfig, LoggingConfig
]:
    """Ostrich's "helhest" base config with the dataset config's `ostrich_overrides`, composed
    through Hydra's compose API (no @hydra.main: the only CLI argument is the dataset config)."""
    with initialize_config_dir(config_dir=str(CONFIG_PATH), version_base=None):
        hcfg = compose(config_name="helhest", overrides=list(overrides))
    return (
        hydra.utils.instantiate(hcfg.simulation),
        hydra.utils.instantiate(hcfg.rendering),
        hydra.utils.instantiate(hcfg.engine),
        hydra.utils.instantiate(hcfg.logging),
    )


def generate(cfg: DatasetConfig) -> None:
    seed = cfg.seed
    n_maps, trials_per_map = cfg.maps.n_maps, cfg.maps.trials_per_map
    maps_dir = cfg.maps.path
    warmup_s, settle_steps, mu = cfg.trial.warmup_s, cfg.trial.settle_steps, cfg.trial.mu
    yaw_gain = cfg.trial.yaw_gain
    chunk, maps_per_build, device = cfg.run.chunk, cfg.run.maps_per_build, cfg.run.device
    w_o = warmup_steps(cfg)  # raises unless warmup_s is a multiple of OSTRICH_DT
    lead = lead_m(cfg)

    spec = PatchSpec()
    xy_jitter, yaw_jitter = cfg.trial.xy_jitter, cfg.trial.yaw_jitter

    print(f"[config]   {cfg.path}")
    print(f"[arc]      v_nom={V_NOM} m/s  arc_len={ARC_LEN} m  kappa in [-{cfg.trial.kappa_max}, "
          f"{cfg.trial.kappa_max}] 1/m  duration={ARC_DURATION_S}s -> {T_RECORD_OSTRICH} ostrich steps")
    print(f"[pivot]    omega_nom=+-{OMEGA_NOM:.4f} rad/s -> {np.degrees(PIVOT_ANGLE):.1f} deg = one of "
          f"{N_THETA} heading bins in place, same duration")
    print(f"[warmup]   {warmup_s}s -> {w_o} ostrich steps ({lead:.3f} m, or "
          f"{np.degrees(OMEGA_NOM * warmup_s):.1f} deg in place), entered already moving at v_nom / omega_nom")
    print(f"[friction] mu={mu} longitudinal (rolling), {mu * MU_LAT_RATIO:.3f} lateral (skid) "
          f"-- MU_LAT_RATIO={MU_LAT_RATIO}, see comparator.common.friction_kwargs")
    print(f"[yaw_gain] {yaw_gain} -- yaw commands scaled by 1/{yaw_gain} = {1.0 / yaw_gain:.3f} "
          f"(pivot commands +-{OMEGA_NOM / yaw_gain:.4f} rad/s to realize +-{OMEGA_NOM:.4f}); "
          f"stored twist, reference and model input stay nominal"
          + ("  [NO COMPENSATION]" if yaw_gain == 1.0 else ""))
    print(f"[patch]    {spec.ny}x{spec.nx} cells @ {spec.cell} m, reference={spec.reference}")
    print(f"[jitter]   xy=+-{xy_jitter:.4f} m (router_cell={cfg.trial.router_cell}), "
          f"yaw=+-{yaw_jitter:.4f} rad (n_theta={cfg.trial.n_theta})")
    for e in cfg.mix:
        params = ", ".join(f"{k}={v}" for k, v in dataclasses.asdict(e.params).items())
        print(f"[mix]      {e.percent:6.2f}% {e.strategy} on {e.map}"
              f"{'' if e.spec.spawn_only else ' (arc end must be settle-feasible)'}"
              f"{f' -- {params}' if params else ''}")

    rng = np.random.default_rng(seed)
    pool = check_maps(cfg, lead)
    allocation = allocate(cfg, pool, rng)
    n = cfg.n
    print(f"[maps]     {n_maps} from {maps_dir} ("
          + ", ".join(f"{c}: {len(pool[c])} available" for c in cfg.categories)
          + f"), {trials_per_map} trial(s) each -> {n} rows")
    print(allocation.table())

    if cfg.run.dry_run:
        # CPU-only check: trial sampling (including its static settle, run on CPU) on every
        # allocated map -- no ostrich rollout, nothing written.
        assert T_RECORD_OSTRICH * OSTRICH_DT == ARC_DURATION_S
        for i, m in enumerate(allocation.maps):
            trials = sample_map_mix(
                HeightMapReader.load(m.path), map_metadata(m.path), spec, m.counts, cfg, rng,
                lead=lead, device="cpu",
            )
            print(f"[dry-run]  {i + 1}/{n_maps} {m.path.name}: {describe_trials(trials)}")
        print("[dry-run]  allocation + sampling + step-count/duration self-checks ok, nothing simulated")
        return

    init_warp_device(device)
    sim_config, render_config, engine_config, logging_config = compose_ostrich_config(
        cfg.ostrich_overrides
    )
    render_config.vis_type = "null"  # headless

    # design.md section 2a: override in code, never in the shared helhest.yaml (submodule_test's
    # characterization baseline depends on that file staying untouched).
    sim_config.target_timestep_seconds = OSTRICH_DT

    per_variant_fields: dict[str, list[np.ndarray]] = {}
    ostrich_fields: dict[str, list[np.ndarray]] = {}
    map_index_all, map_path_all = [], []
    terrain_entries: list[tuple[pathlib.Path, HeightMapReader]] = []

    finished: list[tuple[int, pathlib.Path, HeightMapReader, dict, SpawnBatch]] = []
    for g in range(0, n_maps, maps_per_build):
        group = allocation.maps[g : g + maps_per_build]
        prepared = []
        for m, am in enumerate(group, start=g):
            print(f"[map {m + 1}/{n_maps}] {am.path.name}: sampling trials")
            prepared.append(prepare_map(am, cfg, rng, spec=spec))
        t_build = time.time()
        rollouts = rollout_group(
            prepared, sim_config=sim_config, render_config=render_config,
            engine_config=engine_config, logging_config=logging_config, chunk=chunk,
            settle_steps=settle_steps, warmup_s=warmup_s, mu=mu, yaw_gain=yaw_gain,
        )
        print(f"[ostrich]  maps {g + 1}..{g + len(group)} simulated in {time.time() - t_build:.1f}s")
        for m, (am, prep, (full_pose, full_wheel_qd)) in enumerate(
            zip(group, prepared, rollouts), start=g
        ):
            result = finish_map(
                prep, full_pose, full_wheel_qd, spec=spec, chunk=chunk, settle_steps=settle_steps,
                warmup_s=warmup_s, mu=mu, yaw_gain=yaw_gain, device=device,
            )
            finished.append((m, am.path, prep.terrain, result, prep.trials))

    for m, p, terrain, result, trials in finished:
        for key in ("spawn_pose", "spawn_zpr", "t0_pose", "belief_pose", "patch", "kappa", "v_drive",
                    "wz_drive", "arc_relief", "interact_dir", "ramp_deg", "ramp_s", "sampling",
                    "map_category", "targeted", "arc_end_pose", "ref_pose", "valid",
                    "endpoint_blocked", "swept_clear"):
            per_variant_fields.setdefault(key, []).append(result[key])
        for key in ("ostrich_pose", "ostrich_wheel_qd", "ostrich_cmd",
                    "ostrich_preroll_pose", "ostrich_preroll_wheel_qd"):
            ostrich_fields.setdefault(key, []).append(result[key])
        map_index_all.append(np.full(trials_per_map, m, dtype=np.int64))
        map_path_all.extend([str(p)] * trials_per_map)
        terrain_entries.extend([(p, terrain)] * trials_per_map)

        n_valid = int(result["valid"].sum())
        print(f"[map {m + 1}/{n_maps}] {p.name} -> {n_valid}/{trials_per_map} valid, "
              f"{int(result['endpoint_blocked'].sum())}/{trials_per_map} endpoint-blocked, "
              f"{int(result['swept_clear'].sum())}/{trials_per_map} swept-clear; "
              f"{describe_trials(trials)}")

    per_variant = {k: np.concatenate(v, axis=0) for k, v in per_variant_fields.items()}
    per_variant["map_index"] = np.concatenate(map_index_all, axis=0)
    per_variant["map_path"] = np.array(map_path_all)
    ostrich = {k.removeprefix("ostrich_"): np.concatenate(v, axis=1) for k, v in ostrich_fields.items()}
    ostrich["dt"] = OSTRICH_DT
    ostrich["t"] = np.arange(T_RECORD_OSTRICH, dtype=np.float32) * OSTRICH_DT
    n_preroll = settle_steps + w_o
    # Negative times, so preroll_t continues straight into t (the arc's first step is t=0).
    ostrich["preroll_t"] = (np.arange(n_preroll, dtype=np.float32) - n_preroll) * OSTRICH_DT

    out_path = OUT_DIR / f"dataset_arc_{cfg.name}_M{n_maps}_R{trials_per_map}_seed{seed}.h5"
    write_arc_dataset(
        out_path,
        root=dict(
            config_name=cfg.name, config_path=str(cfg.path), config_yaml=cfg.raw_yaml,
            v_nom=V_NOM, arc_len=ARC_LEN, min_turn_radius=0.5,  # design.md section 4c's pinned
            # guard trio -- see arc.py; kept literal here so this file has no import-time
            # dependency beyond arc.py's already-imported constants
            kappa_min=-cfg.trial.kappa_max, kappa_max=cfg.trial.kappa_max,
            omega_nom=OMEGA_NOM, pivot_angle=PIVOT_ANGLE, pivot_n_theta=N_THETA,  # the pivot
            # primitive's pinned trio (arc.py), for rows with v_drive == 0
            warmup_s=warmup_s, settle_steps=settle_steps, spawn_clearance=SPAWN_CLEARANCE,
            xy_jitter=xy_jitter, yaw_jitter=yaw_jitter, router_cell=cfg.trial.router_cell,
            n_theta=cfg.trial.n_theta, interact_relief=cfg.trial.interact_relief,
            valid_excludes_endpoint_settle=True,
            maps_dir=str(maps_dir), map_glob=MAP_GLOB,
            mu=mu, mu_lat_ratio=MU_LAT_RATIO, yaw_gain=yaw_gain, k_p=K_P,  # mu is LONGITUDINAL; the lateral
            # (skid) coefficient is mu * mu_lat_ratio -- comparator.common.friction_kwargs.
            # Recorded because a file generated before the anisotropy existed is isotropic
            # (ratio 1.0) and its yaw labels are not comparable with a newer file's.
            ostrich_dt=OSTRICH_DT,
            n_maps=n_maps, trials_per_map=trials_per_map, n=n, seed=seed,
            maps_per_build=maps_per_build, chunk=chunk,
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
    n_blocked = int((per_variant["valid"] & per_variant["endpoint_blocked"]).sum())
    print(f"  valid       {n_valid:>7d}  ({100 * n_valid / n:5.1f}%)")
    print(f"  of which endpoint_blocked {n_blocked:>5d}  -- custom_dataset.py drop_blocked_endpoints")
    print(f"  swept_clear {n_swept:>7d}  ({100 * n_swept / n:5.1f}%)  -- reporting split only")
    for strategy in dict.fromkeys(per_variant["sampling"].tolist()):
        k = per_variant["sampling"] == strategy
        print(f"  {strategy:<15} {int(k.sum()):>7d} rows, {int((per_variant['valid'] & k).sum())} valid")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", help="dataset config: a name in configs/ or a path to a .yaml file")
    args = parser.parse_args()
    try:
        generate(load_config(args.config))
    except ConfigError as err:  # a bad file, or a file that does not fit its map directory
        parser.exit(2, f"config error: {err}\n")


if __name__ == "__main__":
    main()
