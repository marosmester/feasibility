"""Generates an mppi_learning divergence dataset (design.md sections 5, 8 step 5): one sample is a
body-frame terrain patch plus ONE 1 s MPPI window's command, and the label is where helhest_stack's
twin and ostrich end up after that window --

    x = (patch [24, 36] @ 0.125 m, (v_mean, v_slope, wz_mean, wz_slope))
    y = twin end pose vs ostrich end pose  -- the errors are computed by the training side

One trial, per row:

1. **Placement** (`spawn_sampling`, chosen by the config's `mix`): a nominal window origin, an entry
   twist `(v, wz)` and one window of wheel speeds from MPPI's own priors (`command.py`).
2. **Ostrich** runs everything in one captured rollout: `settle_steps` at zero command, `warmup_s`
   at the entry twist, then the window's wheel speeds held per 0.1 s step (`command.upsample`).
   Ostrich stands in for the real robot, which turns faster than it does (outdoors 0.55 of an
   ideal yaw rate against ostrich's 0.48), so warm-up and window are both commanded
   `command.compensate(.., trial.ostrich_yaw_gain)`: the yaw rate multiplied by the gain (1.15). The
   sampler keeps those compensated wheels inside the box. Placement integrates the warm-up at
   `entry.yaw_ratio`, what ostrich realizes of MPPI's yaw rate under that compensation.
3. **The twin** (`twin.run_twin`, MPPI's `planning_solver` at `trial.k_turn`) gets MPPI's own
   command, uncompensated, and starts from ostrich's REALIZED state at
   the window start -- pose `t0_pose`, wheel speeds, and the body twist as ostrich's pose
   difference over the warm-up's last MPPI step (0.1 s: a turning ostrich stick-slips, and a
   single 0.025 s step catches a random point of it) -- as MPPI starts every replan from the
   measured state. The
   patch is taken at `t0_pose` too; `origin_drift` records how far it is from the nominal origin.

`valid` is the data-quality gate only: finite poses, ostrich's displacement over the window at most
`MAX_WINDOW_DISPLACEMENT`, the patch on the map. `endpoint_feasible` (the static settle at the
NOMINAL window end, from placement) and the twin's own `twin_min_clearance` / `twin_max_residual`
are stored beside it for the training side to filter on.

Reuses `lattice_learning/generate_dataset.py`'s pinned `OSTRICH_DT`, its settle-pose spawn
(`SPAWN_CLEARANCE`), its Hydra composition and its h5 writer; the rollout loop is this module's own
because every world here has a time-varying setpoint profile.

`run.dry_run: true` samples every allocated map on CPU and, on a CUDA `run.device`, runs the
flat-ground twin-vs-ostrich check (design.md section 8 step 2): the same rollout on a flat map,
printing twin-vs-ostrich end-pose error per command family. Nothing is written.

Output: outputs/dataset_mppi_<config>_maps<maps dir name>_M<n_maps>_R<trials_per_map>_seed<seed>.h5

Usage:
    python src/feasibility/mppi_learning/generate_dataset.py smoke      # configs/smoke.yaml (dry run)
    python src/feasibility/mppi_learning/generate_dataset.py default
    python src/feasibility/mppi_learning/generate_dataset.py path/to/config.yaml
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
import time

import numpy as np
from helhest import dynamics
from ostrich import EngineConfig
from ostrich import LoggingConfig
from ostrich import RenderingConfig
from ostrich import SimulationConfig

from feasibility.comparator.common import init_warp_device
from feasibility.comparator.common import K_P
from feasibility.comparator.common import MU_LAT_RATIO
from feasibility.comparator.common import OUT_DIR
from feasibility.comparator.common import run_ostrich_batch
from feasibility.heightmap import HeightMapReader
from feasibility.lattice_learning.dataset_config import allocate
from feasibility.lattice_learning.dataset_config import AllocatedMap
from feasibility.lattice_learning.dataset_config import ConfigError
from feasibility.lattice_learning.dataset_config import MAP_GLOB
from feasibility.lattice_learning.generate_dataset import compose_ostrich_config
from feasibility.lattice_learning.generate_dataset import OSTRICH_DT
from feasibility.lattice_learning.generate_dataset import SPAWN_CLEARANCE
from feasibility.lattice_learning.generate_dataset import write_arc_dataset
from feasibility.lattice_learning.patch import patch_overhangs
from feasibility.lattice_learning.patch import patch_spec_to_attrs
from feasibility.lattice_learning.patch import sample_patches
from feasibility.lattice_learning.settle import settle_batch
from feasibility.lattice_learning.spawn_sampling import map_metadata
from feasibility.lattice_learning.tiled_terrain import tile_offsets
from feasibility.lattice_learning.tiled_terrain import TiledTerrain
from feasibility.mppi_learning.command import compensate
from feasibility.mppi_learning.command import encode
from feasibility.mppi_learning.command import FAMILIES
from feasibility.mppi_learning.command import HALF_TRACK
from feasibility.mppi_learning.command import MPPI_DT
from feasibility.mppi_learning.command import upsample
from feasibility.mppi_learning.command import WHEEL_RADIUS
from feasibility.mppi_learning.command import WINDOW_S
from feasibility.mppi_learning.command import WINDOW_STEPS
from feasibility.mppi_learning.dataset_config import check_edge_reach
from feasibility.mppi_learning.dataset_config import check_mppi_maps
from feasibility.mppi_learning.dataset_config import DatasetConfig
from feasibility.mppi_learning.dataset_config import load_config
from feasibility.mppi_learning.dataset_config import platform_length
from feasibility.mppi_learning.dataset_config import sample_map_mix
from feasibility.mppi_learning.dataset_config import trial_context
from feasibility.mppi_learning.spawn_sampling import PATCH_SPEC
from feasibility.mppi_learning.spawn_sampling import sample_trials
from feasibility.mppi_learning.spawn_sampling import WindowBatch
from feasibility.mppi_learning.twin import run_twin


def _exact_steps(duration_s: float, dt: float) -> int:
    n = duration_s / dt
    if abs(round(n) - n) > 1e-6:
        raise ValueError(f"{duration_s}s is not an exact multiple of dt={dt}s ({n} steps)")
    return round(n)


T_WINDOW_OSTRICH = _exact_steps(WINDOW_S, OSTRICH_DT)  # 40
MAX_WINDOW_DISPLACEMENT = 2.0 * 4.0 * WHEEL_RADIUS * WINDOW_S  # m, twice the box's farthest window
# (wmax 4 rad/s for 1 s = 1.4 m): past that, ostrich's solve diverged, not the robot


def quat_to_yaw(q: np.ndarray) -> np.ndarray:
    """[..., 4] (qx, qy, qz, qw) -> yaw [...]."""
    qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def twist_to_wheels(twist: np.ndarray) -> np.ndarray:
    """[n, 2] (v, wz) -> [n, 3] ideal no-slip wheel speeds (left, right, rear), rad/s -- the inverse
    of `command.wheels_to_twist`, with the robot geometry the command box is defined in."""
    v, wz = twist[:, 0], twist[:, 1]
    left, right = (v - wz * HALF_TRACK) / WHEEL_RADIUS, (v + wz * HALF_TRACK) / WHEEL_RADIUS
    return np.stack([left, right, (left + right) / 2.0], axis=-1).astype(np.float32)


@dataclasses.dataclass
class PreparedMap:
    """One map's trials, everything drawn from the rng before any rollout."""

    terrain: HeightMapReader
    category: str
    trials: WindowBatch
    spawn_zpr: np.ndarray  # [n, 3] absolute z (incl. SPAWN_CLEARANCE), pitch, roll

    @property
    def n(self) -> int:
        return len(self.trials.pose)


def prepare_trials(terrain: HeightMapReader, category: str, trials: WindowBatch, mu: float, device: str) -> PreparedMap:
    """Ostrich spawns at helhest_stack's static settle at each spawn, lifted SPAWN_CLEARANCE."""
    spawn_zpr, _, _ = settle_batch(terrain, trials.pose, mu, device)
    spawn_zpr = spawn_zpr.astype(np.float64)
    spawn_zpr[:, 0] += SPAWN_CLEARANCE
    return PreparedMap(terrain, category, trials, spawn_zpr)


def prepare_map(allocated: AllocatedMap, cfg: DatasetConfig, rng: np.random.Generator) -> PreparedMap:
    """Samples one map's trials by the config's mix. The only rng consumer, so the trials do not
    depend on `run.maps_per_build`."""
    terrain = HeightMapReader.load(allocated.path)
    trials = sample_map_mix(terrain, map_metadata(allocated.path), allocated.counts, cfg, rng, device=cfg.run.device)
    return prepare_trials(terrain, allocated.category, trials, cfg.trial.mu, cfg.run.device)


def window_setpoints(trials: WindowBatch, yaw_gain: float) -> np.ndarray:
    """[T_WINDOW_OSTRICH, n, 3] what ostrich is commanded over the window: MPPI's wheel speeds,
    yaw-compensated, held per MPPI step."""
    return upsample(compensate(np.ascontiguousarray(trials.omega.transpose(1, 0, 2)), yaw_gain), dt_to=OSTRICH_DT)


def setpoints_of(trials: WindowBatch, settle_steps: int, warmup_steps: int, yaw_gain: float) -> np.ndarray:
    """[settle_steps + warmup_steps + T_WINDOW_OSTRICH, n, 3] ostrich wheel setpoints: rest, the entry
    twist, then the window held per MPPI step -- both yaw-compensated by `yaw_gain`."""
    n = len(trials.pose)
    entry = compensate(twist_to_wheels(trials.entry.astype(np.float64)), yaw_gain)
    return np.concatenate([
        np.zeros((settle_steps, n, 3), np.float32),
        np.broadcast_to(entry[None], (warmup_steps, n, 3)),
        window_setpoints(trials, yaw_gain),
    ]).astype(np.float32)


def rollout_group(
    maps: list[PreparedMap],
    configs: tuple[SimulationConfig, RenderingConfig, EngineConfig, LoggingConfig],
    *,
    chunk: int,
    settle_steps: int,
    warmup_steps: int,
    mu: float,
    yaw_gain: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Every trial of `maps` through ostrich, `chunk` worlds per build; per map (pose [T, n, 7],
    wheel_qd [T, n, 3]) in the map's own coordinates, T = settle + warm-up + window. Several maps
    share one build on `TiledTerrain` tiles, as in lattice_learning."""
    sim_config, render_config, engine_config, logging_config = configs
    if len(maps) == 1:
        terrain, offsets = maps[0].terrain, np.zeros((1, 2))
    else:
        offsets = tile_offsets([m.terrain for m in maps])
        terrain = TiledTerrain([m.terrain for m in maps], offsets)
    row_offset = np.concatenate([np.repeat(offsets[i : i + 1], m.n, axis=0) for i, m in enumerate(maps)])
    spawn_pose = np.concatenate([m.trials.pose for m in maps]).astype(np.float64)
    spawn_pose[:, :2] += row_offset
    spawn_zpr = np.concatenate([m.spawn_zpr for m in maps])
    setpoints = np.concatenate([setpoints_of(m.trials, settle_steps, warmup_steps, yaw_gain) for m in maps], axis=1)
    n = len(spawn_pose)

    pose = np.zeros((setpoints.shape[0], n, 7), np.float32)
    wheel_qd = np.zeros((setpoints.shape[0], n, 3), np.float32)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        t0 = time.time()
        sim_config.num_worlds = end - start
        chunk_pose, chunk_wheel_qd = run_ostrich_batch(
            sim_config, render_config, engine_config, logging_config, terrain,
            np.ascontiguousarray(setpoints[:, start:end]), mu, spawn_pose[start:end], settle_steps=0,
            spawn_zpr=spawn_zpr[start:end],
        )
        chunk_pose[..., :2] -= row_offset[None, start:end].astype(np.float32)
        pose[:, start:end], wheel_qd[:, start:end] = chunk_pose, chunk_wheel_qd
        print(f"    ostrich trials {start}..{end - 1} ({end - start} worlds, {len(maps)} map(s)) "
              f"done in {time.time() - t0:.1f}s")

    results, row = [], 0
    for m in maps:
        results.append((pose[:, row : row + m.n], wheel_qd[:, row : row + m.n]))
        row += m.n
    return results


def realized_start(pose_log: np.ndarray, wheel_log: np.ndarray, w: int, steps_per_mppi_step: int | None = None
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ostrich's state at the window start from the warm-up's logs [w, n, ...] (row k is the state
    after step k + 1, so row w - 1 is the window start): (x, y, yaw) [n, 3], wheel speeds [n, 3],
    body twist (vx, vy, yaw_rate) [n, 3] as the mean over the last MPPI step (the pose difference
    across it, rotated into the body frame at the mid-step heading). `steps_per_mppi_step` is the
    ostrich steps in one MPPI step for logs run at another dt (None: OSTRICH_DT's, the dataset's)."""
    k = _exact_steps(MPPI_DT, OSTRICH_DT) if steps_per_mppi_step is None else int(steps_per_mppi_step)
    last, prev = pose_log[w - 1].astype(np.float64), pose_log[w - 1 - k].astype(np.float64)
    yaw, yaw_prev = quat_to_yaw(last[:, 3:7]), quat_to_yaw(prev[:, 3:7])
    turn = np.angle(np.exp(1j * (yaw - yaw_prev)))
    dx, dy = (last[:, 0] - prev[:, 0]) / MPPI_DT, (last[:, 1] - prev[:, 1]) / MPPI_DT
    c, s = np.cos(yaw_prev + turn / 2), np.sin(yaw_prev + turn / 2)
    yaw_rate = turn / MPPI_DT
    twist = np.stack([c * dx + s * dy, -s * dx + c * dy, yaw_rate], axis=-1)
    return np.stack([last[:, 0], last[:, 1], yaw], axis=-1), wheel_log[w - 1].astype(np.float32), twist.astype(np.float32)


def finish_map(
    prepared: PreparedMap, full_pose: np.ndarray, full_wheel_qd: np.ndarray, *, settle_steps: int,
    warmup_steps: int, mu: float, k_turn: float, yaw_gain: float, device: str,
) -> dict[str, np.ndarray]:
    """One map's ostrich logs -> the twin from ostrich's realized window start, the patch there, and
    every per-row array. Uses no rng."""
    terrain, trials, n = prepared.terrain, prepared.trials, prepared.n
    log, wheel_log = full_pose[settle_steps:], full_wheel_qd[settle_steps:]
    assert log.shape[0] == warmup_steps + T_WINDOW_OSTRICH
    t0_pose, t0_wheels, t0_twist = realized_start(log, wheel_log, warmup_steps)
    ostrich_end = log[-1]

    finite_start = np.isfinite(t0_pose).all(axis=1) & np.isfinite(t0_wheels).all(axis=1) & np.isfinite(t0_twist).all(axis=1)
    # a diverged warm-up must not poison the twin's batch: run those rows from the nominal origin at
    # rest, and let `valid` drop them
    start = np.where(finite_start[:, None], t0_pose, trials.origin)
    twin_pose, clearance, residual = run_twin(
        terrain, start, np.ascontiguousarray(trials.omega.transpose(1, 0, 2)),
        init_wheel_omega=np.where(finite_start[:, None], t0_wheels, 0.0),
        init_twist=np.where(finite_start[:, None], t0_twist, 0.0), mu=mu, k_turn=k_turn, device=device,
        diagnostics=True,
    )
    twin_pose, clearance, residual = twin_pose[0], clearance[0], residual[0]

    displacement = np.linalg.norm(ostrich_end[:, :2] - start[:, :2], axis=1)
    valid = (
        finite_start
        & np.isfinite(ostrich_end).all(axis=1)
        & np.isfinite(twin_pose).all(axis=1)
        & (displacement <= MAX_WINDOW_DISPLACEMENT)
        & ~patch_overhangs(terrain, start, PATCH_SPEC)
    )
    window_cmd = window_setpoints(trials, yaw_gain)  # what ostrich got; `omega` is MPPI's
    return dict(
        spawn_pose=trials.pose.astype(np.float32),
        spawn_zpr=prepared.spawn_zpr.astype(np.float32),
        origin=trials.origin.astype(np.float32),
        t0_pose=t0_pose.astype(np.float32),
        t0_wheel_omega=t0_wheels,
        t0_twist=t0_twist,
        origin_drift=np.linalg.norm(t0_pose[:, :2] - trials.origin[:, :2], axis=1).astype(np.float32),
        entry=trials.entry,
        omega=trials.omega,
        command=encode(np.ascontiguousarray(trials.omega.transpose(1, 0, 2))),
        family=trials.family,
        patch=sample_patches(terrain, start, PATCH_SPEC).reshape(n, -1).astype(np.float32),
        twin_pose=twin_pose.astype(np.float32),
        twin_min_clearance=clearance.astype(np.float32),
        twin_max_residual=residual.astype(np.float32),
        relief=trials.relief,
        interact_dir=trials.interact_dir,
        ramp_deg=trials.ramp_deg,
        ramp_s=trials.ramp_s,
        sampling=trials.strategy,
        map_category=np.full(n, prepared.category),
        targeted=trials.targeted,
        endpoint_feasible=trials.endpoint_feasible,
        valid=valid,
        ostrich_pose=np.ascontiguousarray(log[warmup_steps:]),
        ostrich_wheel_qd=np.ascontiguousarray(wheel_log[warmup_steps:]),
        ostrich_cmd=window_cmd,
        # Viewer-only: the settle drop followed by the warm-up.
        ostrich_preroll_pose=np.ascontiguousarray(full_pose[: settle_steps + warmup_steps]),
        ostrich_preroll_wheel_qd=np.ascontiguousarray(full_wheel_qd[: settle_steps + warmup_steps]),
    )


PER_ROW = ("spawn_pose", "spawn_zpr", "origin", "t0_pose", "t0_wheel_omega", "t0_twist", "origin_drift",
           "entry", "omega", "command", "family", "patch", "twin_pose", "twin_min_clearance",
           "twin_max_residual", "relief", "interact_dir", "ramp_deg", "ramp_s", "sampling", "map_category",
           "targeted", "endpoint_feasible", "valid")
OSTRICH = ("ostrich_pose", "ostrich_wheel_qd", "ostrich_cmd", "ostrich_preroll_pose", "ostrich_preroll_wheel_qd")


def describe_trials(trials: WindowBatch) -> str:
    """One line per map: rows per strategy, interaction split and command families."""
    parts = []
    for strategy in dict.fromkeys(trials.strategy.tolist()):
        k = trials.strategy == strategy
        fam = np.bincount(trials.family[k], minlength=len(FAMILIES))
        parts.append(f"{strategy} {int(k.sum())}: {int((trials.interact_dir[k] > 0).sum())} up/"
                     f"{int((trials.interact_dir[k] < 0).sum())} down, "
                     + "/".join(f"{f[0]}{c}" for f, c in zip(FAMILIES, fam) if c))
    return "; ".join(parts) + (f" ({trials.shortfall} short)" if trials.shortfall else "")


def ostrich_configs(cfg: DatasetConfig) -> tuple[SimulationConfig, RenderingConfig, EngineConfig, LoggingConfig]:
    configs = compose_ostrich_config(cfg.ostrich_overrides)
    configs[1].vis_type = "null"  # headless
    configs[0].target_timestep_seconds = OSTRICH_DT  # in code, never in the shared helhest.yaml
    return configs


def flat_check(cfg: DatasetConfig, n_per_family: int = 16) -> None:
    """design.md section 8 step 2: on flat ground the twin and ostrich should agree up to the twin's
    slip model. Reports, per command family, the twin-vs-ostrich end-pose error and how far ostrich's
    realized origin landed from the nominal one."""
    from feasibility.lattice_learning.custom_dataset import pose_to_se3
    from feasibility.lattice_learning.custom_dataset import se3_errors

    device = cfg.run.device
    init_warp_device(device)
    flat = HeightMapReader.flat(xlim=(-8.0, 8.0), ylim=(-8.0, 8.0), cell=0.05)
    ctx = trial_context(cfg, flat, device)
    trials = sample_trials(ctx, n_per_family * len(FAMILIES), np.random.default_rng(cfg.seed), interact_frac=None)
    prepared = prepare_trials(flat, "flat", trials, cfg.trial.mu, device)
    settle_steps, warmup_steps = cfg.trial.settle_steps, _exact_steps(cfg.trial.warmup_s, OSTRICH_DT)
    (pose, wheel_qd), = rollout_group(
        [prepared], ostrich_configs(cfg), chunk=cfg.run.chunk, settle_steps=settle_steps,
        warmup_steps=warmup_steps, mu=cfg.trial.mu, yaw_gain=cfg.trial.ostrich_yaw_gain,
    )
    out = finish_map(prepared, pose, wheel_qd, settle_steps=settle_steps, warmup_steps=warmup_steps,
                     mu=cfg.trial.mu, k_turn=cfg.trial.k_turn, yaw_gain=cfg.trial.ostrich_yaw_gain,
                     device=device)
    e_pos, e_rot = se3_errors(pose_to_se3(out["twin_pose"]), pose_to_se3(out["ostrich_pose"][-1]))
    print(f"[flat]     twin vs ostrich after one {WINDOW_S:.0f} s window on flat ground, "
          f"{int(out['valid'].sum())}/{len(e_pos)} valid (median / max):")
    for code, name in enumerate(FAMILIES):
        k = (out["family"] == code) & out["valid"]
        if k.any():
            print(f"[flat]     {name:<8} n={int(k.sum()):>3}  e_pos {np.median(e_pos[k]):.3f} / {e_pos[k].max():.3f} m  "
                  f"e_rot {np.median(e_rot[k]):.3f} / {e_rot[k].max():.3f} rad  "
                  f"origin drift {np.median(out['origin_drift'][k]):.3f} m")


def generate(cfg: DatasetConfig) -> None:
    n_maps, trials_per_map, seed = cfg.maps.n_maps, cfg.maps.trials_per_map, cfg.seed
    settle_steps, mu = cfg.trial.settle_steps, cfg.trial.mu
    warmup_steps = _exact_steps(cfg.trial.warmup_s, OSTRICH_DT)
    chunk, maps_per_build, device = cfg.run.chunk, cfg.run.maps_per_build, cfg.run.device
    maps_dir = cfg.maps.path

    print(f"[config]   {cfg.path}")
    print(f"[window]   {WINDOW_S} s = {WINDOW_STEPS} MPPI steps = {T_WINDOW_OSTRICH} ostrich steps; "
          f"wheels in [{cfg.command.wmin}, {cfg.command.wmax}] rad/s, spins from {cfg.command.spin_min}, "
          f"families {dict(zip(FAMILIES, cfg.command.mix))}")
    print(f"[warmup]   {cfg.trial.warmup_s} s -> {warmup_steps} ostrich steps after {settle_steps} settle steps; "
          f"entry = the window's first command, {cfg.entry.jump_frac:.0%} jumps (v <= {cfg.entry.v_max} m/s, "
          f"|wz| <= {cfg.entry.wz_max} rad/s, box-limited); placed at yaw_ratio {cfg.entry.yaw_ratio}")
    print(f"[twin]     planning_solver: motor_tau {dynamics.MOTOR_TAU} s, k_turn {cfg.trial.k_turn} "
          f"(alpha {1 + cfg.trial.k_turn * mu:.2f}), mu {mu}")
    print(f"[ostrich]  commanded yaw rate x {cfg.trial.ostrich_yaw_gain} (warm-up and window), "
          f"mu_lat_ratio {MU_LAT_RATIO}")
    print(f"[patch]    {PATCH_SPEC.ny}x{PATCH_SPEC.nx} cells @ {PATCH_SPEC.cell} m, x [{PATCH_SPEC.x_min}, {PATCH_SPEC.x_max}]")
    print(f"[ramps]    ramp_down needs plateaus >= {platform_length(cfg):.2f} m; its start is capped per face "
          f"by the plateau")

    rng = np.random.default_rng(seed)
    pool = check_mppi_maps(cfg)
    allocation = allocate(cfg, pool, rng)
    check_edge_reach(allocation)
    print(f"[maps]     {n_maps} from {maps_dir}, {trials_per_map} trial(s) each -> {cfg.n} rows")
    print(allocation.table())

    if cfg.run.dry_run:
        for i, m in enumerate(allocation.maps):
            trials = sample_map_mix(HeightMapReader.load(m.path), map_metadata(m.path), m.counts, cfg, rng, device="cpu")
            print(f"[dry-run]  {i + 1}/{n_maps} {m.path.name}: {describe_trials(trials)}")
        if device.startswith("cuda"):
            flat_check(cfg)
        print("[dry-run]  allocation + sampling ok, nothing written")
        return

    init_warp_device(device)
    configs = ostrich_configs(cfg)
    finished: list[tuple[int, AllocatedMap, PreparedMap, dict]] = []
    for g in range(0, n_maps, maps_per_build):
        group = allocation.maps[g : g + maps_per_build]
        prepared = []
        for m, am in enumerate(group, start=g):
            print(f"[map {m + 1}/{n_maps}] {am.path.name}: sampling trials")
            prepared.append(prepare_map(am, cfg, rng))
        t_build = time.time()
        rollouts = rollout_group(prepared, configs, chunk=chunk, settle_steps=settle_steps,
                                 warmup_steps=warmup_steps, mu=mu, yaw_gain=cfg.trial.ostrich_yaw_gain)
        print(f"[ostrich]  maps {g + 1}..{g + len(group)} simulated in {time.time() - t_build:.1f}s")
        for m, (am, prep, (pose, wheel_qd)) in enumerate(zip(group, prepared, rollouts), start=g):
            result = finish_map(prep, pose, wheel_qd, settle_steps=settle_steps, warmup_steps=warmup_steps,
                                mu=mu, k_turn=cfg.trial.k_turn, yaw_gain=cfg.trial.ostrich_yaw_gain,
                                device=device)
            finished.append((m, am, prep, result))
            print(f"[map {m + 1}/{n_maps}] {am.path.name} -> {int(result['valid'].sum())}/{prep.n} valid, "
                  f"origin drift median {np.median(result['origin_drift']):.2f} m; {describe_trials(prep.trials)}")

    per_variant = {k: np.concatenate([r[k] for *_, r in finished]) for k in PER_ROW}
    per_variant["map_index"] = np.concatenate([np.full(p.n, m, np.int64) for m, _, p, _ in finished])
    per_variant["map_path"] = np.array([str(am.path) for _, am, p, _ in finished for _ in range(p.n)])
    ostrich = {k.removeprefix("ostrich_"): np.concatenate([r[k] for *_, r in finished], axis=1) for k in OSTRICH}
    ostrich["dt"] = OSTRICH_DT
    ostrich["t"] = np.arange(T_WINDOW_OSTRICH, dtype=np.float32) * OSTRICH_DT
    n_preroll = settle_steps + warmup_steps
    ostrich["preroll_t"] = (np.arange(n_preroll, dtype=np.float32) - n_preroll) * OSTRICH_DT

    write_arc_dataset(
        OUT_DIR / f"dataset_mppi_{cfg.name}_maps{maps_dir.name}_M{n_maps}_R{trials_per_map}_seed{seed}.h5",
        root=dict(
            config_name=cfg.name, config_path=str(cfg.path), config_yaml=cfg.raw_yaml,
            window_s=WINDOW_S, window_steps=WINDOW_STEPS, mppi_dt=MPPI_DT, ostrich_dt=OSTRICH_DT,
            command_columns="v_mean,v_slope,wz_mean,wz_slope", families=",".join(FAMILIES),
            wmin=cfg.command.wmin, wmax=cfg.command.wmax, sigma=cfg.command.sigma,
            sigma_knot=cfg.command.sigma_knot, spin_min=cfg.command.spin_min, family_mix=np.asarray(cfg.command.mix),
            entry_v_max=cfg.entry.v_max, entry_wz_max=cfg.entry.wz_max,
            warmup_s=cfg.trial.warmup_s, settle_steps=settle_steps, spawn_clearance=SPAWN_CLEARANCE,
            entry_jump_frac=cfg.entry.jump_frac, entry_yaw_ratio=cfg.entry.yaw_ratio,
            interact_relief=cfg.trial.interact_relief,
            twin_solver="planning_solver", motor_tau=dynamics.MOTOR_TAU, k_turn=cfg.trial.k_turn,
            ostrich_yaw_gain=cfg.trial.ostrich_yaw_gain,
            mu=mu, mu_lat_ratio=MU_LAT_RATIO, k_p=K_P,
            max_window_displacement=MAX_WINDOW_DISPLACEMENT,
            maps_dir=str(maps_dir), map_glob=MAP_GLOB, n_maps=n_maps, trials_per_map=trials_per_map,
            n=cfg.n, seed=seed, maps_per_build=maps_per_build, chunk=chunk,
            **patch_spec_to_attrs(PATCH_SPEC),
        ),
        per_variant=per_variant,
        terrain_entries=[(am.path, p.terrain) for _, am, p, _ in finished for _ in range(p.n)],
        ostrich=ostrich,
    )
    n_valid = int(per_variant["valid"].sum())
    print("=" * 60)
    print(f"[summary]  {cfg.n} trials across {n_maps} maps, {n_valid} valid ({100 * n_valid / cfg.n:.1f}%)")
    for strategy in dict.fromkeys(per_variant["sampling"].tolist()):
        k = per_variant["sampling"] == strategy
        print(f"  {strategy:<15} {int(k.sum()):>7d} rows, {int((per_variant['valid'] & k).sum())} valid")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="dataset config: a name in configs/ or a path to a .yaml file")
    args = parser.parse_args()
    try:
        generate(load_config(args.config))
    except ConfigError as err:
        parser.exit(2, f"config error: {err}\n")


if __name__ == "__main__":
    main()
