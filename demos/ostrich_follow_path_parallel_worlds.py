"""`ostrich_follow_path.py` for many (map, planner) pairs at once: plan them all, then drive EVERY
(map, planner, repeat) as its own world in one ostrich build -- the series that took 20+ minutes
one process per pair becomes one rollout.

How the worlds are laid out (what ostrich/newton allow)
    * Replicated worlds (`finalize_replicated`) only ever collide with the GLOBAL shapes, never
      with each other, so every planner and repeat on one map shares that map's footprint.
    * The terrain is one global mesh per build, so different maps can't be separate builds inside
      one run. `lattice_learning.tiled_terrain` (what `generate_dataset.py +maps_per_build` uses)
      merges the maps into one mesh, each on its own tile 5 m apart. A world spawns at its map's
      start + that tile's offset, its planned path is shifted the same way, and the offset is
      subtracted from the logged poses again, so verdicts, PNGs and h5 files are in map coordinates.
    * The engine runs a fixed iteration count, so a flipped or finished world costs the same as a
      driving one and doesn't slow the batch. Worlds are sorted by path length and split into
      builds of at most --worlds-per-build (GPU memory); each build steps as long as its longest path.
    * Controller: feasibility.planning.pure_pursuit's kernel, each world reading its own path out
      of one flat waypoint buffer, inside the captured step.

Start/goal pairs per map (--pairs, default 3)
    Pair 0 is always the sidecar's own start/goal (on the uphill series: the centre straight line,
    facing +X). Pairs 1.. are drawn from the heightmap alone, seeded per map name (--seed), so a
    map keeps its pairs whatever else is on the command line:
    * start (x, y, yaw), yaw ~ U(-yaw_range, +yaw_range) around +X: the LOWER part of the map --
      the three wheel contacts and the body origin sit at the map's lowest level, and the terrain
      on rings around each contact stays under that wheel's rim (plus START_GAP), so the robot
      spawns on the flat ground below the face without a wheel touching the foot;
    * goal (x, y): the UPPER part -- every point within GOAL_CLEARANCE (wheelbase + margin) of the
      goal is at the map's highest level, so the robot ends fully on the plateau whatever heading
      it arrives with;
    * both keep every checked point EDGE_MARGIN inside the grid.
    Rejection sampling; a map with no admissible start or goal raises.

Planning is feasibility.planning.planners.plan_path, as in the single-map script;
a (map, pair, planner) with no feasible path gets no worlds and shows as "no path" in the summary.

Outputs (<out-dir>/)
    <map>_p<K>_<planner>.png / .h5   as ostrich_follow_path.py writes them, one per start/goal
                                     pair K (replay with gl_replay.py --file ... --id R --which ostrich)
    summary.yaml                     (map, pair) x planners arrived counts, the worst arc's predicted
                                     errors on each planned path (max_err: e_pos_rot / e_rot / fused
                                     from the pos_rot net, also for the ungated vanilla planners, so
                                     their outcomes can calibrate the fused taus), the pairs,
                                     settings, wall time

Run as a module (imports benchmarks/ and demos/, needs the repo root on sys.path):
    python -m demos.ostrich_follow_path_parallel_worlds
    python -m demos.ostrich_follow_path_parallel_worlds --planners nn-gated-rot --tau-rot 0.49
    python -m demos.ostrich_follow_path_parallel_worlds --planners nn-gated-rot nn-gated-fused --pairs 5
    python -m demos.ostrich_follow_path_parallel_worlds \
        --maps assets/uphill_series/uphill_a0050 assets/uphill_series/uphill_a0600 --repeats 5

CLI parameters:
    --maps PATH ...        map stems (PNG + YAML with start/goal), or directories whose *.png are
                           all taken (default: assets/uphill_series)
    --planners NAME ...    any of feasibility.planning.planners.PLANNERS
                           (default: nn-gated-pos nn-gated-pitch nn-gated-rot nn-gated-fused)
    --pairs INT            start/goal pairs per map, the sidecar's first (default: 3)
    --seed INT             pair sampling seed, combined with each map's name (default: 0)
    --yaw-range DEG        sampled start yaw is within +-DEG of +X (default: 90)
    --repeats INT          worlds per (map, pair, planner) (default: 3)
    --worlds-per-build INT cap on worlds in one ostrich build; more worlds -> several builds (default: 64)
    --out-dir PATH         output directory (default: outputs/follow_path_parallel)
    --tau-pos/--tau-pitch/--tau-rot/--tau-fused-pos/--tau-fused-rot, --checkpoint/--checkpoint-rot, --torch-device, --chunk,
    --v, --lookahead, --mu, --slack, --settle-steps, --override
                           as in ostrich_follow_path.py
"""
from __future__ import annotations

import argparse
import gc
import math
import pathlib
import time
import zlib
from dataclasses import dataclass

import numpy as np
import yaml

from demos.ostrich_follow_path import add_shared_args
from demos.ostrich_follow_path import plot_run
from demos.ostrich_follow_path import REPO_ROOT
from feasibility.comparator.common import K_P
from feasibility.comparator.common import WHEEL_RADIUS
from feasibility.comparator.provenance import write_comparison
from feasibility.heightmap import HeightMapReader
from feasibility.heightmap.create_uphill_series import ASSETS_DIR
from feasibility.lattice_learning.patch import WHEEL_CONTACTS_LOCAL
from feasibility.lattice_learning.generate_dataset import compose_ostrich_config
from feasibility.lattice_learning.tiled_terrain import tile_offsets
from feasibility.lattice_learning.tiled_terrain import TiledTerrain
from feasibility.planning.evaluation import judge
from feasibility.planning.planners import plan_path
from feasibility.planning.planners import PlanContext
from feasibility.planning.planners import PlannerConfig
from feasibility.planning.planners import PLANNERS
from feasibility.planning.pure_pursuit import controller_kappa_max
from feasibility.planning.pure_pursuit import PurePursuitSimulator
from feasibility.planning.pure_pursuit import resample_polyline

DEFAULT_PLANNERS = ("nn-gated-pos", "nn-gated-pitch", "nn-gated-rot", "nn-gated-fused")
SPAWN_HEIGHT = 0.5  # [m] above the terrain, HelhestBatchSimulator's own default spawn
# Start clearance: at horizontal distance d from a wheel contact the rim is r - sqrt(r^2 - d^2) above
# the ground, so terrain there may rise that high without touching the wheel -- lets a start sit
# right at a gentle foot (the 5 deg map has only 2.2 m of flat ground) yet keeps a wheel radius from
# a steep face. Checked on rings at START_RING_RADII, with START_GAP [m] of extra horizontal clearance.
START_RING_RADII = (0.1, 0.2, 0.3, float(WHEEL_RADIUS) - 0.01)
START_GAP = 0.05
GOAL_CLEARANCE = float(-WHEEL_CONTACTS_LOCAL[:, 0].min()) + 0.2  # [m] wheelbase + margin
# [m] every checked point this far inside the grid: judge's off_map is 0.5 m on the chassis, and pure
# pursuit overshoots the lattice's 0.5 m turn by ~0.5 m when a sideways start swings toward the goal
EDGE_MARGIN = 1.0
LEVEL_TOL = 1e-3  # [m] "at the lowest / highest level" (8-bit maps quantize to ~3 mm, flats exactly)
MAX_PAIR_TRIES = 100_000


@dataclass
class Job:
    """One world: a repeat of one (map, pair, planner) plan."""

    map_idx: int
    pair: int
    start: tuple[float, float, float]
    planner: str
    repeat: int
    path: np.ndarray  # [P, 2] resampled waypoints, map coordinates
    path_len: float
    drive_steps: int


# --- stages ---------------------------------------------------------------------------------------


def resolve_maps(entries: list[pathlib.Path]) -> list[pathlib.Path]:
    stems = []
    for e in entries:
        if e.is_dir():
            stems += [p.with_suffix("") for p in sorted(e.glob("*.png"))]
        else:
            stems.append(e.with_suffix(""))
    if not stems:
        raise SystemExit(f"no maps found in {[str(e) for e in entries]}")
    return stems


def _inside(terrain: HeightMapReader, x: np.ndarray, y: np.ndarray) -> bool:
    """Every point (x, y) is EDGE_MARGIN inside the grid."""
    x_lo, x_hi = terrain.x0 + EDGE_MARGIN, terrain.x0 + terrain.nx * terrain.cell - EDGE_MARGIN
    y_lo, y_hi = terrain.y0 + EDGE_MARGIN, terrain.y0 + terrain.ny * terrain.cell - EDGE_MARGIN
    return bool(x.min() >= x_lo and x.max() <= x_hi and y.min() >= y_lo and y.max() <= y_hi)


def sample_pairs(
    terrain: HeightMapReader,
    sidecar: tuple[tuple[float, float, float], tuple[float, float]],
    n: int,
    yaw_range: float,
    rng: np.random.Generator,
) -> list[tuple[tuple[float, float, float], tuple[float, float]]]:
    """`n` (start, goal) pairs: the sidecar's first, then starts on the map's lowest level and goals
    on its highest -- see the module docstring."""
    lo, hi = float(terrain.H.min()), float(terrain.H.max())
    if hi - lo < 10 * LEVEL_TOL:
        raise ValueError("map has no distinct lower and upper part")
    x_min, x_max = terrain.x0, terrain.x0 + terrain.nx * terrain.cell
    y_min, y_max = terrain.y0, terrain.y0 + terrain.ny * terrain.cell
    r = float(WHEEL_RADIUS)
    ang = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    dirs = np.stack([np.cos(ang), np.sin(ang)], axis=-1)  # [A, 2]
    ring = np.vstack([d * dirs for d in START_RING_RADII])  # [R, 2]
    rim = np.repeat([r - math.sqrt(r * r - max(d - START_GAP, 0.0) ** 2) for d in START_RING_RADII],
                    len(ang))  # [R] allowed rise above the lowest level
    gx, gy = np.meshgrid(np.arange(-GOAL_CLEARANCE, GOAL_CLEARANCE + 1e-9, 0.5 * terrain.cell),
                         np.arange(-GOAL_CLEARANCE, GOAL_CLEARANCE + 1e-9, 0.5 * terrain.cell))
    disc = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    disc = disc[np.hypot(disc[:, 0], disc[:, 1]) <= GOAL_CLEARANCE]

    def draw(admissible, what: str) -> np.ndarray:
        for _ in range(MAX_PAIR_TRIES):
            q = np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max),
                          rng.uniform(-yaw_range, yaw_range)])
            if admissible(q):
                return q
        raise SystemExit(f"no admissible {what} in {MAX_PAIR_TRIES} draws")

    def start_ok(q: np.ndarray) -> bool:
        c, s = math.cos(q[2]), math.sin(q[2])
        wx = q[0] + c * WHEEL_CONTACTS_LOCAL[:, 0] - s * WHEEL_CONTACTS_LOCAL[:, 1]  # [3]
        wy = q[1] + s * WHEEL_CONTACTS_LOCAL[:, 0] + c * WHEEL_CONTACTS_LOCAL[:, 1]
        rx, ry = wx[:, None] + ring[None, :, 0], wy[:, None] + ring[None, :, 1]  # [3, R]
        if not _inside(terrain, rx, ry):
            return False
        on_floor = np.append(terrain.sample(wx, wy), terrain.sample(q[0], q[1]))
        return bool(np.all(np.abs(on_floor - lo) < LEVEL_TOL)
                    and np.all(terrain.sample(rx, ry) - lo < rim[None] + LEVEL_TOL))

    def goal_ok(q: np.ndarray) -> bool:
        x, y = q[0] + disc[:, 0], q[1] + disc[:, 1]
        return _inside(terrain, x, y) and bool(np.all(np.abs(terrain.sample(x, y) - hi) < LEVEL_TOL))

    pairs = [sidecar]
    for _ in range(n - 1):
        s_ = draw(start_ok, "start")
        g_ = draw(goal_ok, "goal")
        pairs.append(((float(s_[0]), float(s_[1]), float(s_[2])), (float(g_[0]), float(g_[1]))))
    return pairs


def plan_all(
    stems: list[pathlib.Path],
    terrains: list[HeightMapReader],
    pairs: list[list[tuple[tuple[float, float, float], tuple[float, float]]]],
    args: argparse.Namespace,
    dt: float,
) -> tuple[dict[tuple[int, int, str], dict], list[Job]]:
    ctx = PlanContext(PlannerConfig.from_args(args))
    plans, jobs = {}, []
    for m, (stem, terrain) in enumerate(zip(stems, terrains)):
        for k, (start, goal) in enumerate(pairs[m]):
            for planner in args.planners:
                plan = plan_path(terrain, start, goal, planner, ctx, audit=True)
                name = f"{stem.name} p{k} {planner}" + (f" ({plan['gate']})" if plan["gate"] else "")
                plans[(m, k, planner)] = plan
                if not plan["reached"]:
                    print(f"  {name}: no path (V* = {plan['v_start']:.2f})")
                    continue
                path = resample_polyline(np.vstack([plan["xy"], goal]))
                path_len = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
                plan.update(path=path, path_len=path_len)
                drive_steps = int(math.ceil(args.slack * path_len / args.v / dt))
                jobs += [Job(m, k, start, planner, r, path, path_len, drive_steps)
                         for r in range(args.repeats)]
                print(f"  {name}: {plan['path_m']:.2f} m planned, {args.repeats} world(s)")
    del ctx
    gc.collect()  # planner buffers go before the ostrich builds (3 GiB GPU)
    return plans, jobs


def simulate(
    jobs: list[Job],
    terrains: list[HeightMapReader],
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], list[np.ndarray], float]:
    """Every job as a world, chunked into builds. Returns per job (pose [drive_steps, 7],
    wheel_qd [drive_steps, 3]) in map coordinates with the settle sliced off, and dt."""
    if len(terrains) == 1:
        offsets = np.zeros((1, 2))
        terrain = terrains[0]
    else:
        offsets = tile_offsets(terrains)
        terrain = TiledTerrain(terrains, offsets)
    kappa_max = controller_kappa_max()
    order = sorted(range(len(jobs)), key=lambda j: jobs[j].drive_steps)
    poses: list[np.ndarray] = [None] * len(jobs)
    wheels: list[np.ndarray] = [None] * len(jobs)
    dt = math.nan
    for c0 in range(0, len(order), args.worlds_per_build):
        idx = order[c0 : c0 + args.worlds_per_build]
        chunk = [jobs[j] for j in idx]
        b = len(chunk)
        sim_config, render_config, engine_config, logging_config = compose_ostrich_config(
            tuple(args.override))
        render_config.vis_type = "null"
        sim_config.num_worlds = b
        dt = float(sim_config.target_timestep_seconds)
        off = np.array([offsets[j.map_idx] for j in chunk])  # [b, 2]
        spawn_pose = np.array([j.start for j in chunk], np.float64)
        spawn_zpr = np.zeros((b, 3), np.float64)
        spawn_zpr[:, 0] = [float(terrains[j.map_idx].sample(*spawn_pose[i, :2])) + SPAWN_HEIGHT
                           for i, j in enumerate(chunk)]
        spawn_pose[:, :2] += off
        T = args.settle_steps + max(j.drive_steps for j in chunk)
        t0 = time.time()
        sim = None
        try:
            sim = PurePursuitSimulator(
                sim_config, render_config, engine_config, logging_config,
                k_p=K_P, mu_front=args.mu, mu_rear=args.mu, terrain=terrain,
                spawn_pose=spawn_pose, spawn_zpr=spawn_zpr,
                paths=[j.path + off[i] for i, j in enumerate(chunk)],
                settle_steps=args.settle_steps, lookahead=args.lookahead, v=args.v,
                kappa_max=kappa_max,
            )
            t_build = time.time() - t0
            pose, wheel_qd = sim.rollout(T)
        finally:
            sim = None
            gc.collect()
        t_roll = time.time() - t0 - t_build
        print(f"  build {c0 // args.worlds_per_build}: {b} worlds x {T} steps, build {t_build:.1f} s, "
              f"rollout {t_roll:.1f} s ({1e3 * t_roll / T:.1f} ms/step)")
        pose[..., :2] -= off[None].astype(np.float32)
        for i, (j, job) in enumerate(zip(idx, chunk)):
            s = slice(args.settle_steps, args.settle_steps + job.drive_steps)
            poses[j], wheels[j] = pose[s, i], wheel_qd[s, i]
    return poses, wheels, dt


# --- main -----------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--maps", type=pathlib.Path, nargs="+", default=[ASSETS_DIR])
    parser.add_argument("--planners", nargs="+", choices=list(PLANNERS), default=list(DEFAULT_PLANNERS))
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--yaw-range", type=float, default=90.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worlds-per-build", type=int, default=64)
    parser.add_argument("--out-dir", type=pathlib.Path,
                        default=REPO_ROOT / "outputs" / "follow_path_parallel")
    add_shared_args(parser)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("--pairs must be >= 1 (pair 0 is the sidecar's start/goal)")
    t_start = time.time()

    stems = resolve_maps(args.maps)
    metas = [yaml.safe_load(s.with_suffix(".yaml").read_text()) for s in stems]
    for s, meta in zip(stems, metas):
        if meta.get("start") is None or meta.get("goal") is None:
            raise SystemExit(f"{s.name}.yaml has no start/goal")
    terrains = [HeightMapReader.load(s) for s in stems]
    pairs = []
    for stem, meta, terrain in zip(stems, metas, terrains):
        sidecar = (tuple(float(v) for v in meta["start"]),
                   (float(meta["goal"][0]), float(meta["goal"][1])))
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, zlib.crc32(stem.name.encode())]))
        pairs.append(sample_pairs(terrain, sidecar, args.pairs, math.radians(args.yaw_range), rng))
    dt_cfg = float(compose_ostrich_config(tuple(args.override))[0].target_timestep_seconds)

    # --- A. plan --------------------------------------------------------------------------------
    print(f"=== planning {len(stems)} map(s) x {args.pairs} pair(s) x {len(args.planners)} planner(s) ===")
    for stem, map_pairs in zip(stems, pairs):
        print(f"  {stem.name}: " + "  ".join(
            f"p{k} ({s[0]:.2f}, {s[1]:.2f}, {math.degrees(s[2]):.0f} deg) -> ({g[0]:.2f}, {g[1]:.2f})"
            for k, (s, g) in enumerate(map_pairs)))
    t0 = time.time()
    plans, jobs = plan_all(stems, terrains, pairs, args, dt_cfg)
    print(f"planning done in {time.time() - t0:.1f} s; {len(jobs)} world(s) to simulate")

    # --- B. simulate ----------------------------------------------------------------------------
    results: dict[int, dict] = {}
    poses: list[np.ndarray] = []
    wheels: list[np.ndarray] = []
    dt = dt_cfg
    if jobs:
        print(f"=== ostrich: {len(jobs)} worlds, <= {args.worlds_per_build} per build, v={args.v} m/s, "
              f"lookahead {args.lookahead} m, mu={args.mu}, dt={dt_cfg} ===")
        t0 = time.time()
        poses, wheels, dt = simulate(jobs, terrains, args)
        print(f"ostrich done in {time.time() - t0:.1f} s")
        for j, job in enumerate(jobs):
            results[j] = judge(poses[j], dt, job.path, terrains[job.map_idx])

    # --- C. verdicts + outputs ------------------------------------------------------------------
    summary: dict[str, dict[str, str]] = {}
    path_err: dict[str, dict[str, dict[str, float]]] = {}  # worst arc on each planned path
    for m, stem in enumerate(stems):
        for k, (start, goal) in enumerate(pairs[m]):
            row_name = f"{stem.name}_p{k}"
            summary[row_name] = {}
            path_err[row_name] = {}
            for planner in args.planners:
                plan = plans[(m, k, planner)]
                path_err[row_name][planner] = {h: round(float(v), 4) for h, v in plan["max_err"].items()}
                gate = f", gate {plan['gate']}" if plan["gate"] else ""
                if not plan["reached"]:
                    summary[row_name][planner] = "no path"
                    continue
                ids = [j for j, job in enumerate(jobs)
                       if job.map_idx == m and job.pair == k and job.planner == planner]
                res = [results[j] for j in ids]
                n_ok = sum(r["status"] == "arrived" for r in res)
                summary[row_name][planner] = f"{n_ok}/{len(res)}"
                path, path_len = plan["path"], plan["path_len"]
                run_name = f"{row_name}_{planner}"
                print(f"\n--- {run_name}{gate}: ({start[0]:.2f}, {start[1]:.2f}, "
                      f"{math.degrees(start[2]):.0f} deg) -> ({goal[0]:.2f}, {goal[1]:.2f}), "
                      f"planned {plan['path_m']:.2f} m ---")
                print(f"{'id':>3} {'status':>9} {'t_arr':>6} {'climb':>6} {'roll':>5} {'cte_max':>7} "
                      f"{'cte_mean':>8} {'cte@climb':>9} {'progress':>12}")
                for w, r in enumerate(res):
                    print(f"{w:3d} {r['status']:>9} {r['t_arrive']:6.1f} {r['climb_deg']:6.1f} "
                          f"{r['roll_deg']:5.1f} {r['cte_max']:7.3f} {r['cte_mean']:8.3f} "
                          f"{r['cte_at_climb']:9.3f} {r['progress_m']:5.2f}/{path_len:<5.2f}")
                title = (f"{stem.name} pair {k}: {planner}{gate}\nplanned {plan['path_m']:.1f} m, "
                         f"{n_ok}/{len(res)} arrived in ostrich (v {args.v} m/s, lookahead {args.lookahead} m)")
                plot_run(terrains[m], plan, path, res, start, title, args.out_dir / f"{run_name}.png")
                n = len(ids)
                pose = np.stack([poses[j] for j in ids], axis=1)
                wheel_qd = np.stack([wheels[j] for j in ids], axis=1)
                write_comparison(
                    args.out_dir / f"{run_name}.h5",
                    root=dict(
                        name=run_name, map=str(stem), pair=k, planner=planner, head=str(plan["head"]),
                        tau=plan["tau"], gate=plan["gate"], n=n, v=args.v, lookahead=args.lookahead,
                        mu=args.mu, slack=args.slack, settle_steps=args.settle_steps, path_m=path_len,
                        goal_x=goal[0], goal_y=goal[1], obstacle_x=0.5 * (start[0] + goal[0]),
                    ),
                    per_variant=dict(
                        spawn_pose=np.tile(np.array(start, np.float32), (n, 1)),
                        v_drive=np.full(n, args.v, np.float32),
                        wz_drive=np.zeros(n, np.float32),
                        variant_value=np.arange(n, dtype=np.float32),
                        variant_label=np.array([f"{run_name}_r{w}" for w in range(n)]),
                        status=np.array([r["status"] for r in res]),
                        planned_path=path.astype(np.float32),
                        **{key: np.array([r[key] for r in res], np.float32)
                           for key in ("t_arrive", "climb_deg", "roll_deg", "cte_max", "cte_mean",
                                       "cte_at_climb", "progress_m")},
                    ),
                    terrain_entries=[(stem, terrains[m])] * n,
                    ostrich=dict(dt=dt, t=np.arange(len(pose), dtype=np.float32) * dt, pose=pose,
                                 wheel_qd=wheel_qd),
                    hstack=dict(dt=dt),
                )

    # tile isolation check: every world's first logged xy is its own start
    drift = [float(np.linalg.norm(poses[j][0, :2] - np.array(job.start[:2]))) for j, job in enumerate(jobs)]
    wall = time.time() - t_start
    width = max(len(name) for name in summary)
    print(f"\n=== summary (arrived / worlds), {wall:.0f} s total ===")
    print(" " * width + "".join(f"{p:>16}" for p in args.planners))
    for name, row in summary.items():
        print(f"{name:<{width}}" + "".join(f"{row[p]:>16}" for p in args.planners))
    if drift:
        print(f"max start drift after un-tiling: {max(drift):.3f} m")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.yaml").write_text(yaml.safe_dump(dict(
        arrived=summary,
        max_err=path_err,
        pairs={f"{stem.name}_p{k}": dict(start=list(s), goal=list(g))
               for stem, map_pairs in zip(stems, pairs) for k, (s, g) in enumerate(map_pairs)},
        taus=dict(e_pos=args.tau_pos, e_pitch=args.tau_pitch, e_rot=args.tau_rot,
                  fused_e_pos=args.tau_fused_pos, fused_e_rot=args.tau_fused_rot),
        checkpoint=str(args.checkpoint), checkpoint_rot=str(args.checkpoint_rot),
        seed=args.seed, yaw_range_deg=args.yaw_range, repeats=args.repeats, v=args.v,
        lookahead=args.lookahead, mu=args.mu, slack=args.slack,
        worlds=len(jobs), worlds_per_build=args.worlds_per_build, wall_s=round(wall, 1),
        max_start_drift_m=max(drift) if drift else None,
    ), sort_keys=False))
    print(f"saved {args.out_dir / 'summary.yaml'}")


if __name__ == "__main__":
    main()
