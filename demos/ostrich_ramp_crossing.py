"""Ostrich ground truth for helhest_stack's ramp-series benchmark: can Helhest Junior actually drive
over each ridge, end to end?

`helhest_stack/scripts/bench_ramp_series.py` asks the lattice planner whether a ridge is crossable
and gets two opinions: the settle's `blocked` field (crosses only up to 15 deg) and no `blocked`
at all (crosses everything up to 75 deg). Neither is checked against physics. This script is the
referee: it loads the same 15 maps (`feasibility.heightmap.create_ramp_series`, which builds them
with helhest_stack's own `rampmaps`), spawns the robot at each map's benchmark start, and drives it
straight at the goal in ostrich. The ridge is symmetric and full-width, so one straight drive tests
the climb AND the descent, and there is no way around it.

`+series=uphill` drives the uphill-only series instead (`feasibility.heightmap.create_uphill_series`,
the maps `benchmarks/bench_uphill_nn.py` plans on): one full-width face and a plateau holding the
goal, so the drive tests the climb alone. It is the ostrich check on that benchmark's assumed
60/65 deg climb limit, which was read off the ridge runs. The robot stops on the plateau at the
goal x; "where" reports approach / up face / plateau.

How it runs
    * All maps in ONE ostrich build: `lattice_learning.tiled_terrain.TiledTerrain` puts each map on
      its own tile of one global mesh, and each world spawns on its tile. `+repeats=R` gives every
      map R worlds, because ostrich is intrinsically nondeterministic on contact-rich driving
      (submodule_test measured it), so a single run per angle is a coin flip near the limit.
    * Open loop: a constant forward twist `(+v, 0)`, the same wheel servo as `comparator.common`.
      There is no heading controller. Yaw and lateral drift are reported so a run that wandered
      rather than failed is visible.
    * A device-side stop latch: once a world's chassis passes its goal x, its wheel targets are
      zeroed for the rest of the rollout (a Warp kernel inside the captured step, no host round
      trip). Without it a robot that crosses early would keep driving off the end of the map and
      fall into the empty space between tiles.
    * The command runs for `+slack` x the nominal time to the goal, so a slow but successful climb
      is not mistaken for a stall.

Verdict per run (chassis pose = the front-axle base frame, as everywhere in comparator/)
    crossed    base reached the goal x without flipping
    flipped    chassis up-vector pointed below the horizon at some point before the goal
    stalled    time ran out; the furthest x reached is reported as a ridge segment
               (approach / up face / plateau / down face / run-out)
    off_map    drifted within 0.5 m of the map's Y edge
    nonfinite  the solve produced NaN/inf
Also reported, up to the crossing: peak climb pitch (nose up), peak descend pitch (nose down), peak
|roll|, time to cross vs nominal, final yaw and |y| drift.

Output: `outputs/ostrich_ramp_crossing.h5` (`+series=uphill`: `outputs/ostrich_uphill_crossing.h5`)
in comparator/provenance's schema (settle rows sliced off, poses shifted back to each map's own
coordinates, empty `hstack/` group), so any run replays with:
    python src/feasibility/replay/gl_replay.py --file outputs/ostrich_ramp_crossing.h5 --id K --which ostrich
where K = map_index * repeats + repeat (printed in the table).

Hydra overrides (append as key=value; `+` for the ones below):
    +series=ridge|uphill  map family (default: ridge)
    +maps_dir=PATH        series dir (default: assets/ramp_series, or assets/uphill_series)
    +angles=[20,40]       only these face angles, deg (default: every map in maps_dir)
    +repeats=INT          worlds per map (default: 3)
    +v=FLOAT              forward speed, m/s (default: 0.6, lattice_learning's V_NOM)
    +slack=FLOAT          command duration as a multiple of nominal time to goal (default: 2.0)
    +mu=FLOAT             ground friction (default: 0.8, as comparator/ and the datasets)
    +settle_steps=INT     zero-command drop onto the terrain before driving (default: 20)
    +out=PATH             output HDF5 (default: outputs/ostrich_<ramp|uphill>_crossing.h5)

Usage:
    python src/feasibility/heightmap/create_ramp_series.py     # maps first
    python demos/ostrich_ramp_crossing.py
    python demos/ostrich_ramp_crossing.py +repeats=5 +angles=[30,35,40,45]
    python src/feasibility/heightmap/create_uphill_series.py
    python demos/ostrich_ramp_crossing.py +series=uphill
    python demos/ostrich_ramp_crossing.py simulation.target_timestep_seconds=0.025
"""
from __future__ import annotations

import gc
import math
import pathlib
import time

import examples
import hydra
import numpy as np
import warp as wp
import yaml
from examples.helhest_junior.replay_real import WHEEL_DOF_OFFSET
from omegaconf import DictConfig
from ostrich import EngineConfig
from ostrich import LoggingConfig
from ostrich import RenderingConfig
from ostrich import SimulationConfig

from feasibility.comparator.common import _batch_control_kernel
from feasibility.comparator.common import cmd_to_wheels
from feasibility.comparator.common import HelhestBatchSimulator
from feasibility.comparator.common import friction_kwargs
from feasibility.comparator.common import K_P
from feasibility.comparator.provenance import write_comparison
from feasibility.heightmap import HeightMapReader
from feasibility.heightmap.create_ramp_series import ASSETS_DIR
from feasibility.heightmap.create_ramp_series import ramp_series_paths
from feasibility.heightmap.create_uphill_series import ASSETS_DIR as UPHILL_ASSETS_DIR
from feasibility.heightmap.create_uphill_series import uphill_series_paths
from feasibility.lattice_learning.tiled_terrain import tile_offsets
from feasibility.lattice_learning.tiled_terrain import TiledTerrain
from feasibility.planning.evaluation import pitch_roll

CONFIG_PATH = pathlib.Path(examples.__file__).parent.joinpath("conf")
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# The two map families this script drives. Both sidecars carry `up_deg`, `height`, `start` and
# `goal`, so everything but the file layout, the defaults and the reported reference is shared.
# blocked_deg is where helhest_stack's settle `blocked` stops the planner on that family: the
# ridge's 15 deg descend limit, or the uphill face's 25 deg climb limit.
SERIES = {
    "ridge": dict(
        paths=ramp_series_paths, assets_dir=ASSETS_DIR, glob="ramp_a*.png",
        generator="heightmap/create_ramp_series.py", blocked_deg=15,
        out=REPO_ROOT / "outputs" / "ostrich_ramp_crossing.h5",
    ),
    "uphill": dict(
        paths=uphill_series_paths, assets_dir=UPHILL_ASSETS_DIR, glob="uphill_a*.png",
        generator="heightmap/create_uphill_series.py", blocked_deg=25,
        out=REPO_ROOT / "outputs" / "ostrich_uphill_crossing.h5",
    ),
}

DEFAULT_V = 0.6  # m/s -- lattice_learning.arc.V_NOM, the speed the divergence net was trained at
DEFAULT_REPEATS = 3
DEFAULT_SLACK = 2.0
DEFAULT_MU = 0.8
DEFAULT_SETTLE_STEPS = 20  # the +0.5 m drop is at rest by ~step 7 at dt 3e-2 (comparator.common)
SPAWN_Z = 0.5  # m above the flat start, comparator.common's level-spawn convention
CHASSIS_LOCAL_IDX = 0  # body 0 of each world is the chassis -- same index the pose logger reads
OFF_MAP_MARGIN = 0.5  # m from the Y edge


@wp.kernel
def _stop_latch_kernel(
    body_q: wp.array(dtype=wp.transform),
    stop_x: wp.array(dtype=wp.float32),  # [W] world-frame x past which a world stops driving
    stopped: wp.array(dtype=wp.int32),  # [W] latched 1 once stop_x was reached
    joint_target_vel: wp.array(dtype=wp.float32),
    bodies_per_world: int,
    dofs_per_world: int,
    wheel_dof_offset: int,
):
    w = wp.tid()
    p = wp.transform_get_translation(body_q[w * bodies_per_world + CHASSIS_LOCAL_IDX])
    if p[0] >= stop_x[w]:
        stopped[w] = 1
    if stopped[w] == 1:
        base = w * dofs_per_world + wheel_dof_offset
        joint_target_vel[base + 0] = 0.0
        joint_target_vel[base + 1] = 0.0
        joint_target_vel[base + 2] = 0.0


class RampCrossingSimulator(HelhestBatchSimulator):
    """HelhestBatchSimulator plus the per-world stop latch, applied after the setpoint control
    kernel inside the captured step so it overrides the drive command once a world arrives."""

    def __init__(self, *args, stop_x: np.ndarray, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        device = self.model.device
        self._stop_x = wp.array(np.asarray(stop_x, np.float32), dtype=wp.float32, device=device)
        self._stopped = wp.zeros(self.simulation_config.num_worlds, dtype=wp.int32, device=device)

    def _batch_physics_step(self) -> None:
        # HelhestBatchSimulator._batch_physics_step with one extra launch after the control kernel
        num_worlds = self.simulation_config.num_worlds
        self.current_state.clear_forces()
        self.contacts = self.model.collide(self.current_state)
        wp.launch(
            kernel=_batch_control_kernel,
            dim=num_worlds,
            inputs=[
                self._setpoints_wp, self._step_buf, self._T, self.control.joint_target_vel,
                self.dofs_per_world, WHEEL_DOF_OFFSET,
            ],
            device=self.model.device,
        )
        wp.launch(
            kernel=_stop_latch_kernel,
            dim=num_worlds,
            inputs=[
                self.current_state.body_q, self._stop_x, self._stopped,
                self.control.joint_target_vel, self.bodies_per_world, self.dofs_per_world,
                WHEEL_DOF_OFFSET,
            ],
            device=self.model.device,
        )
        self.solver.step(
            state_in=self.current_state,
            state_out=self.next_state,
            control=self.control,
            contacts=self.contacts,
            dt=self.clock.dt,
        )
        self._copy_state(self.current_state, self.next_state)
        self._log_step(self._step_buf, self._T, self._pose_log, self._wheel_log)


def yaw_of(q: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def ridge_segment(x: float, meta: dict) -> str:
    """Which part of the (origin-centred) ridge a base x sits on."""
    run = meta["height"] / math.tan(math.radians(meta["up_deg"]))
    if "crest_x" in meta:  # uphill series: no plateau end and no down face, the plateau runs on
        crest = meta["crest_x"]
        return "approach" if x < crest - run else "up face" if x < crest else "plateau"
    half = 0.5 * meta["plateau"]
    if x < -half - run:
        return "approach"
    if x < -half:
        return "up face"
    if x <= half:
        return "plateau"
    if x <= half + run:
        return "down face"
    return "run-out"


def judge(pose: np.ndarray, dt: float, meta: dict, terrain: HeightMapReader, v: float) -> dict:
    """pose [T, 7] of one world in its map's own coordinates -> verdict + metrics."""
    goal_x = meta["goal"][0]
    start_x = meta["start"][0]
    y_edge = terrain.y0 + terrain.ny * terrain.cell - OFF_MAP_MARGIN
    finite = np.isfinite(pose).all(axis=1)
    reached = np.nonzero(finite & (pose[:, 0] >= goal_x))[0]
    end = int(reached[0]) + 1 if len(reached) else int(finite.sum()) if finite.all() else int(np.argmin(finite))
    p = pose[: max(end, 1)]
    pitch, roll, up_z = pitch_roll(p[:, 3:7])
    flip_idx = np.nonzero(up_z < 0.0)[0]
    off_idx = np.nonzero(np.abs(p[:, 1]) > y_edge)[0]
    if not finite.all() and not len(reached):
        status = "nonfinite"
    elif len(flip_idx):
        status = "flipped"
    elif len(off_idx):
        status = "off_map"
    elif len(reached):
        status = "crossed"
    else:
        status = "stalled"
    furthest = float(np.nanmax(p[:, 0]))
    event_x = {
        "flipped": float(p[flip_idx[0], 0]) if len(flip_idx) else furthest,
        "off_map": float(p[off_idx[0], 0]) if len(off_idx) else furthest,
    }.get(status, furthest)
    last10 = max(0, len(p) - int(round(10.0 / dt)))
    return dict(
        status=status,
        where="" if status == "crossed" else ridge_segment(event_x, meta),
        furthest_x=furthest,
        # only for a real crossing: an inverted Helhest can still roll on its oversized wheels, so a
        # flipped run sometimes reaches the goal too
        t_cross=float(reached[0] * dt) if status == "crossed" else float("nan"),
        t_nominal=(goal_x - start_x) / v,
        climb_deg=float(np.degrees(np.maximum(-pitch, 0.0).max())),
        descend_deg=float(np.degrees(np.maximum(pitch, 0.0).max())),
        roll_deg=float(np.degrees(np.abs(roll).max())),
        yaw_drift_deg=float(np.degrees(yaw_of(p[-1, 3:7]) - yaw_of(p[0, 3:7]))),
        y_drift=float(np.abs(p[:, 1] - p[0, 1]).max()),
        progress_last10s=float(p[-1, 0] - p[last10, 0]),
    )


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest", version_base=None)
def ostrich_ramp_crossing(cfg: DictConfig) -> None:
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)
    render_config.vis_type = "null"  # batched captured rollout; watch runs with gl_replay.py

    series_name = str(cfg.get("series", "ridge"))
    if series_name not in SERIES:
        raise SystemExit(f"+series must be one of {sorted(SERIES)}, got {series_name!r}")
    series = SERIES[series_name]
    maps_dir = pathlib.Path(cfg.get("maps_dir", series["assets_dir"]))
    repeats = int(cfg.get("repeats", DEFAULT_REPEATS))
    v = float(cfg.get("v", DEFAULT_V))
    slack = float(cfg.get("slack", DEFAULT_SLACK))
    mu = float(cfg.get("mu", DEFAULT_MU))
    settle_steps = int(cfg.get("settle_steps", DEFAULT_SETTLE_STEPS))
    out = pathlib.Path(cfg.get("out", series["out"]))
    dt = float(sim_config.target_timestep_seconds)

    paths = series["paths"](maps_dir)
    if not paths:
        raise SystemExit(f"no {series['glob']} in {maps_dir} -- run {series['generator']} first")
    metas = [yaml.safe_load(p.with_suffix(".yaml").read_text()) for p in paths]
    if cfg.get("angles", None) is not None:
        wanted = {round(float(a), 3) for a in cfg.angles}
        keep = [i for i, m in enumerate(metas) if round(m["up_deg"], 3) in wanted]
        paths, metas = [paths[i] for i in keep], [metas[i] for i in keep]
    terrains = [HeightMapReader.load(p) for p in paths]
    k = len(paths)
    n = k * repeats

    # --- one tiled build: world w = map (w // repeats), repeat (w % repeats) ---------------------
    offsets = tile_offsets(terrains)
    map_of = np.repeat(np.arange(k), repeats)
    row_offset = offsets[map_of]  # [n, 2]
    spawn_pose = np.array([m["start"] for m in metas], np.float64)[map_of]
    spawn_pose[:, :2] += row_offset
    spawn_zpr = np.tile([SPAWN_Z, 0.0, 0.0], (n, 1))
    stop_x = np.array([m["goal"][0] for m in metas])[map_of] + row_offset[:, 0]

    t_nominal = max((m["goal"][0] - m["start"][0]) / v for m in metas)
    T = int(math.ceil(slack * t_nominal / dt))
    setpoints = np.concatenate(
        [np.zeros((settle_steps, n, 3), np.float32), np.tile(np.float32(cmd_to_wheels(v, 0.0)), (T, n, 1))]
    )
    sim_config.num_worlds = n
    print(f"{k} map(s) x {repeats} repeat(s) = {n} worlds, v={v} m/s, mu={mu}, dt={dt}, "
          f"{settle_steps} settle + {T} drive steps ({T * dt:.0f} s = {slack}x nominal)")

    t_start = time.time()
    sim = None
    try:
        sim = RampCrossingSimulator(
            sim_config, render_config, engine_config, logging_config,
            k_p=K_P, **friction_kwargs(mu), terrain=TiledTerrain(terrains, offsets),
            spawn_pose=spawn_pose, spawn_zpr=spawn_zpr, stop_x=stop_x,
        )
        # settle rides in the captured rollout as leading zero rows (generate_dataset.py's pattern)
        pose, wheel_qd = sim.replay_graph_batch(setpoints, settle_steps=0)
    finally:
        sim = None
        gc.collect()
    print(f"ostrich rollout done in {time.time() - t_start:.1f} s")
    pose, wheel_qd = pose[settle_steps:], wheel_qd[settle_steps:]
    pose[..., :2] -= row_offset[None].astype(np.float32)

    # --- verdicts ---------------------------------------------------------------------------------
    results = [judge(pose[:, w], dt, metas[map_of[w]], terrains[map_of[w]], v) for w in range(n)]
    print(f"\n{'deg':>4} {'id':>3} {'status':>9} {'where':>9} {'t_cross':>7} {'t_nom':>5} {'climb':>5} "
          f"{'desc':>5} {'roll':>5} {'yaw_dr':>6} {'y_dr':>5} {'furthest':>8} {'last10s':>7}")
    for w, r in enumerate(results):
        print(f"{metas[map_of[w]]['up_deg']:4.0f} {w:3d} {r['status']:>9} {r['where']:>9} "
              f"{r['t_cross']:7.1f} {r['t_nominal']:5.1f} {r['climb_deg']:5.1f} {r['descend_deg']:5.1f} "
              f"{r['roll_deg']:5.1f} {r['yaw_drift_deg']:6.1f} {r['y_drift']:5.2f} "
              f"{r['furthest_x']:8.2f} {r['progress_last10s']:7.2f}")

    print(f"\n{'deg':>4}  crossed  outcomes")
    crossed_up_to = None
    for i, m in enumerate(metas):
        rs = results[i * repeats : (i + 1) * repeats]
        n_ok = sum(r["status"] == "crossed" for r in rs)
        outcomes = ", ".join(r["status"] + (f"@{r['where']}" if r["where"] else "") for r in rs)
        print(f"{m['up_deg']:4.0f}  {n_ok:>3}/{repeats:<3}  {outcomes}")
        if n_ok == repeats and (crossed_up_to is None or crossed_up_to == metas[i - 1]["up_deg"]):
            crossed_up_to = m["up_deg"]
    print(f"\nevery repeat crosses, contiguously from the shallowest map, up to: "
          f"{'none' if crossed_up_to is None else f'{crossed_up_to:.0f} deg'}   "
          f"(helhest_stack settle 'blocked': {series['blocked_deg']} deg)")

    write_comparison(
        out,
        root=dict(
            name=out.stem, series=series_name, n=n, repeats=repeats, v=v, mu=mu, slack=slack,
            settle_steps=settle_steps, obstacle_x=0.0, maps_dir=str(maps_dir),
        ),
        per_variant=dict(
            spawn_pose=np.array([m["start"] for m in metas], np.float32)[map_of],
            v_drive=np.full(n, v, np.float32),
            wz_drive=np.zeros(n, np.float32),
            variant_value=np.array([m["up_deg"] for m in metas], np.float32)[map_of],
            variant_label=np.array(
                [f"ramp_{metas[map_of[w]]['up_deg']:.0f}deg_r{w % repeats}" for w in range(n)]
            ),
            map_index=map_of,
            status=np.array([r["status"] for r in results]),
            where=np.array([r["where"] for r in results]),
            **{key: np.array([r[key] for r in results], np.float32)
               for key in ("furthest_x", "t_cross", "climb_deg", "descend_deg", "roll_deg",
                           "yaw_drift_deg", "y_drift")},
        ),
        terrain_entries=[(paths[i], terrains[i]) for i in map_of],
        ostrich=dict(dt=dt, t=np.arange(T, dtype=np.float32) * dt, pose=pose, wheel_qd=wheel_qd),
        hstack=dict(dt=dt),
    )


if __name__ == "__main__":
    ostrich_ramp_crossing()
