"""Plan on a heightmap with one of the uphill benchmark's planners, then have Helhest Junior FOLLOW that
path in ostrich -- the physics check on a planned path, not just on a straight drive.

`benchmarks/bench_uphill_nn.py` shows the NN-gated planners reaching the 60 deg plateau by crossing
the face obliquely, while `demos/ostrich_ramp_crossing.py` only ever drove straight at it. This
demo drives the planned line itself:

    1. load a heightmap (any PNG+YAML; start/goal from its sidecar or --start/--goal), optionally
       shifted sideways with --start-side / --goal-side
    2. plan start -> goal with --planner (feasibility.planning.planners.plan_path)
       (vanilla-off: no `blocked`; vanilla-on: settle `blocked`; nn-gated-pos / nn-gated-pitch /
       nn-gated-rot: arcs pruned where the network's predicted e_pos / e_pitch / e_rot exceeds tau;
       nn-gated-fused: pruned where e_pos > --tau-fused-pos OR e_rot > --tau-fused-rot; e_rot and
       the fused gate's e_pos come from the pos_rot checkpoint --checkpoint-rot, the others from
       --checkpoint)
    3. drive it in ostrich with a pure-pursuit controller running INSIDE the captured physics step
       (a Warp kernel reading the chassis pose, no host round trip), --repeats worlds in one build
       because ostrich is nondeterministic on contact-rich driving
    4. report per repeat: arrived / flipped / off_map / stalled / nonfinite, peak climb pitch and
       |roll|, cross-track error to the planned path (max, mean, and at the peak-pitch moment)

Path and controller: see feasibility.planning.pure_pursuit (waypoints, pursuit law, KAPPA_GAIN
clamp, STOP_RADIUS / GOAL_TOLERANCE stop). Planners and thresholds: feasibility.planning.planners;
verdict: feasibility.planning.evaluation.judge.

Outputs (<out-dir>/<map>_<planner>.*)
    .png  bird's-eye elevation, planned path dashed, each repeat's executed track coloured by outcome
    .h5   comparator/provenance's schema (poses after the settle, empty hstack/ group), plus the
          planned path as `planned_path`; replay one repeat K with
          python src/feasibility/replay/gl_replay.py --file <out-dir>/<map>_<planner>.h5 --id K --which ostrich

Run:
    python demos/ostrich_follow_path.py --map assets/uphill_series/uphill_a0600 --planner nn-gated-pos
    python demos/ostrich_follow_path.py --map assets/uphill_series/uphill_a0600 --planner vanilla-off --view
    python demos/ostrich_follow_path.py --map assets/uphill_series/uphill_a0650 --planner nn-gated-pitch \
        --tau-pitch 0.2 --repeats 5
    python demos/ostrich_follow_path.py --map assets/uphill_series/uphill_a0500 --planner nn-gated-fused \
        --start-side left --goal-side right --view

CLI parameters:
    --map PATH            heightmap stem (PNG + YAML), required
    --planner NAME        vanilla-off | vanilla-on | nn-gated-pos | nn-gated-pitch | nn-gated-rot |
                          nn-gated-fused, required
    --start X Y YAW       start pose (default: the sidecar's `start`)
    --goal X Y            goal (default: the sidecar's `goal`)
    --start-side SIDE     left | center | right: shift the start sideways (default: center = unshifted)
    --goal-side SIDE      same for the goal. The shift is perpendicular to start -> goal (left = the
                          robot's left looking at the goal), to SIDE_MARGIN of the map's edge; not
                          combinable with an explicit --start / --goal
    --tau-pos FLOAT       nn-gated-pos threshold [m] (default: planners.TAU_POS = 0.1741)
    --tau-pitch FLOAT     nn-gated-pitch threshold [rad] (default: planners.TAU_PITCH = 0.0933)
    --tau-rot FLOAT       nn-gated-rot threshold [rad] (default: TAU_ROT = 0.38)
    --tau-fused-pos FLOAT nn-gated-fused e_pos threshold [m] (default: TAU_FUSED_POS = 0.25)
    --tau-fused-rot FLOAT nn-gated-fused e_rot threshold [rad] (default: TAU_FUSED_ROT = 0.46)
    --checkpoint PATH     pos_rpy network checkpoint (default: ..._M200_R8_seed0_rpy.pt)
    --checkpoint-rot PATH pos_rot network checkpoint (default: ..._M200_R8_seed0.pt)
    --torch-device STR    network inference device (default: cpu)
    --chunk INT           patches per network batch (default: 4096)
    --v FLOAT             forward speed [m/s] (default: 0.6)
    --lookahead FLOAT     pure-pursuit lookahead [m] (default: 0.4)
    --repeats INT         worlds following the same path (default: 3; 1 with --view)
    --mu FLOAT            ground friction (default: 0.8)
    --slack FLOAT         drive time as a multiple of path length / v (default: 2.0)
    --settle-steps INT    zero-command drop onto the terrain before driving (default: 20)
    --view                live GL viewer instead of the headless rollout; the camera starts at the
                          corner behind-right of the start, looking at the whole path from above
                          (with no feasible path: the robot settled at the start, parked, until closed)
                          starts PAUSED: SPACE runs/pauses, "." steps once while paused
    --out-dir PATH        output directory (default: outputs/follow_path)
    --override K=V ...    ostrich Hydra overrides, e.g. simulation.target_timestep_seconds=0.03
"""
from __future__ import annotations

import argparse
import gc
import math
import pathlib
import time

import numpy as np
import warp as wp
import yaml

from feasibility.comparator.common import friction_kwargs
from feasibility.comparator.common import K_P
from feasibility.comparator.provenance import write_comparison
from feasibility.heightmap import HeightMapReader
from feasibility.lattice_learning.generate_dataset import compose_ostrich_config
from feasibility.planning.evaluation import judge
from feasibility.planning.planners import DEFAULT_CHECKPOINT
from feasibility.planning.planners import DEFAULT_CHECKPOINT_ROT
from feasibility.planning.planners import plan_path
from feasibility.planning.planners import PlanContext
from feasibility.planning.planners import PlannerConfig
from feasibility.planning.planners import PLANNERS
from feasibility.planning.planners import TAU_FUSED_POS
from feasibility.planning.planners import TAU_FUSED_ROT
from feasibility.planning.planners import TAU_PITCH
from feasibility.planning.planners import TAU_POS
from feasibility.planning.planners import TAU_ROT
from feasibility.planning.pure_pursuit import controller_kappa_max
from feasibility.planning.pure_pursuit import PurePursuitSimulator
from feasibility.planning.pure_pursuit import resample_polyline

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SIDES = ("left", "center", "right")
SIDE_MARGIN = 1.5  # [m] from the map edge to a side-shifted start/goal (robot ~0.4 m + judge's 0.5 m)
CAMERA_AZIMUTH = 60.0  # [deg] of the camera from straight behind the start, towards its right
CAMERA_ELEVATION = 30.0  # [deg] above the horizontal, looking down at the path's midpoint
STATUS_COLORS = {"arrived": "tab:green", "flipped": "tab:red", "off_map": "tab:orange",
                 "stalled": "white", "nonfinite": "magenta"}


# --- start/goal, camera --------------------------------------------------------------------------


def shift_sideways(
    start: tuple[float, float, float],
    goal: tuple[float, float],
    start_side: str,
    goal_side: str,
    terrain: HeightMapReader,
) -> tuple[tuple[float, float, float], tuple[float, float]]:
    """Move start and/or goal perpendicular to the start -> goal line, left or right by the largest
    offset that keeps both a side-shifted start and goal SIDE_MARGIN inside the map. Yaw unchanged."""
    s, g = np.array(start[:2]), np.array(goal)
    along = (g - s) / np.linalg.norm(g - s)
    left = np.array([-along[1], along[0]])
    lo = np.array([terrain.x0, terrain.y0]) + SIDE_MARGIN
    hi = np.array([terrain.x0 + terrain.nx * terrain.cell, terrain.y0 + terrain.ny * terrain.cell]) - SIDE_MARGIN

    def reach(p: np.ndarray, d: np.ndarray) -> float:  # how far p can move along d staying in [lo, hi]
        steps = [((hi if d_k > 0 else lo)[k] - p[k]) / d_k for k, d_k in enumerate(d) if abs(d_k) > 1e-9]
        return max(min(steps), 0.0)

    offset = min(reach(p, sign * left) for p in (s, g) for sign in (1.0, -1.0))
    sign = {"left": 1.0, "center": 0.0, "right": -1.0}
    s = s + sign[start_side] * offset * left
    g = g + sign[goal_side] * offset * left
    return (float(s[0]), float(s[1]), float(start[2])), (float(g[0]), float(g[1]))


def corner_camera(
    terrain: HeightMapReader, start: tuple[float, float, float], path: np.ndarray, fov_deg: float
) -> tuple[np.ndarray, float, float, float]:
    """Camera behind-right of the start, CAMERA_ELEVATION above the path's midpoint, far enough to
    frame the whole path and the terrain it crosses. Returns (pos, pitch_deg, yaw_deg, distance),
    Newton's Z-up convention (front = (cos yaw cos pitch, sin yaw cos pitch, sin pitch))."""
    xy = np.vstack([np.array(start[:2])[None], path])
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    z = terrain.sample(xy[:, 0], xy[:, 1])
    target = np.array([*(0.5 * (lo + hi)), 0.5 * (z.min() + z.max())])
    radius = 0.5 * float(np.linalg.norm(hi - lo)) + 1.5
    dist = 0.75 * radius / math.tan(math.radians(0.5 * fov_deg))
    travel = path[-1] - np.array(start[:2])
    travel /= np.linalg.norm(travel)
    right = np.array([travel[1], -travel[0]])
    az = math.radians(CAMERA_AZIMUTH)
    horiz = -math.cos(az) * travel + math.sin(az) * right  # target -> camera, on the ground plane
    el = math.radians(CAMERA_ELEVATION)
    pos = target + dist * np.array([*(math.cos(el) * horiz), math.sin(el)])
    front = target - pos
    yaw = math.degrees(math.atan2(front[1], front[0]))
    pitch = math.degrees(math.asin(front[2] / np.linalg.norm(front)))
    return pos, pitch, yaw, dist


def show_parked(
    args: argparse.Namespace,
    terrain: HeightMapReader,
    start: tuple[float, float, float],
    goal: tuple[float, float],
) -> None:
    """No path to follow: open the GL viewer on the robot settled at `start`, commanded to stand
    still (v = 0 and a one-point path, so the pursuit kernel stops at once), until the window closes."""
    sim_config, render_config, engine_config, logging_config = compose_ostrich_config(tuple(args.override))
    render_config.vis_type = "gl"
    sim_config.num_worlds = 1
    here = np.array([start[:2], start[:2]], np.float32)
    sim = None
    try:
        sim = PurePursuitSimulator(
            sim_config, render_config, engine_config, logging_config,
            k_p=K_P, **friction_kwargs(args.mu), terrain=terrain,
            spawn_pose=np.array([start], np.float64), paths=[here],
            settle_steps=args.settle_steps, lookahead=args.lookahead, v=0.0, kappa_max=controller_kappa_max(),
        )
        # frame the start -> goal line the planner could not connect
        pos, pitch, yaw, dist = corner_camera(terrain, start, np.array([goal]), sim.viewer.camera.fov)
        sim.viewer.set_camera(pos=wp.vec3(*pos), pitch=pitch, yaw=yaw)
        sim.viewer.camera.sync_pivot_to_view(dist)
        sim.rollout(args.settle_steps, view=True)  # the settle drop, then held until closed
    finally:
        sim = None
        gc.collect()


# --- outputs --------------------------------------------------------------------------------------


def plot_run(
    terrain: HeightMapReader,
    plan: dict,
    path: np.ndarray,
    results: list[dict],
    start: tuple[float, float, float],
    title: str,
    out: pathlib.Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = terrain
    extent = (t.x0, t.x0 + t.nx * t.cell, t.y0, t.y0 + t.ny * t.cell)
    fig, ax = plt.subplots(figsize=(12, 5.5), layout="constrained")
    im = ax.imshow(t.H, origin="lower", extent=extent, cmap="viridis", interpolation="nearest")
    ax.plot(plan["xy"][:, 0], plan["xy"][:, 1], "--", color="black", lw=2, label="planned path")
    for i, r in enumerate(results):
        ax.plot(r["xy"][:, 0], r["xy"][:, 1], "-", lw=1.2, color=STATUS_COLORS[r["status"]],
                label=f"repeat {i}: {r['status']}")
    ax.plot(start[0], start[1], "o", ms=9, color="white", mec="black", label="start")
    ax.plot(path[-1, 0], path[-1, 1], "*", ms=14, color="gold", mec="black", label="goal")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(im, ax=ax, label="elevation z [m]", shrink=0.8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"saved {out}")


# --- main -----------------------------------------------------------------------------------------


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    """Planner thresholds/networks and controller/physics flags, shared with
    ostrich_follow_path_parallel_worlds.py."""
    parser.add_argument("--tau-pos", type=float, default=TAU_POS)
    parser.add_argument("--tau-pitch", type=float, default=TAU_PITCH)
    parser.add_argument("--tau-rot", type=float, default=TAU_ROT)
    parser.add_argument("--tau-fused-pos", type=float, default=TAU_FUSED_POS)
    parser.add_argument("--tau-fused-rot", type=float, default=TAU_FUSED_ROT)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-rot", type=pathlib.Path, default=DEFAULT_CHECKPOINT_ROT)
    parser.add_argument("--torch-device", default="cpu")
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--v", type=float, default=0.6)
    parser.add_argument("--lookahead", type=float, default=0.4)
    parser.add_argument("--mu", type=float, default=0.8)
    parser.add_argument("--slack", type=float, default=2.0)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--override", nargs="*", default=[])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--map", type=pathlib.Path, required=True)
    parser.add_argument("--planner", choices=list(PLANNERS), required=True)
    parser.add_argument("--start", type=float, nargs=3, metavar=("X", "Y", "YAW"))
    parser.add_argument("--goal", type=float, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--start-side", choices=SIDES, default="center")
    parser.add_argument("--goal-side", choices=SIDES, default="center")
    add_shared_args(parser)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--out-dir", type=pathlib.Path, default=REPO_ROOT / "outputs" / "follow_path")
    args = parser.parse_args()

    map_path = args.map.with_suffix("")
    meta = yaml.safe_load(map_path.with_suffix(".yaml").read_text())
    start = tuple(args.start) if args.start is not None else meta.get("start")
    goal = tuple(args.goal) if args.goal is not None else meta.get("goal")
    if start is None or goal is None:
        raise SystemExit(f"{map_path.name}.yaml has no start/goal -- pass --start X Y YAW --goal X Y")
    if (args.start is not None and args.start_side != "center") or (
        args.goal is not None and args.goal_side != "center"
    ):
        raise SystemExit("--start-side/--goal-side shift the sidecar's start/goal; drop --start/--goal")
    start = tuple(float(v) for v in start)
    goal = (float(goal[0]), float(goal[1]))
    terrain = HeightMapReader.load(map_path)
    start, goal = shift_sideways(start, goal, args.start_side, args.goal_side, terrain)
    repeats = 1 if args.view else args.repeats
    sides = "" if args.start_side == args.goal_side == "center" else f"_s{args.start_side[0]}_g{args.goal_side[0]}"
    run_name = f"{map_path.name}{sides}_{args.planner}"

    # --- 1. plan --------------------------------------------------------------------------------
    print(f"=== {map_path.name}: planning with {args.planner}, start {start} -> goal {goal} ===")
    ctx = PlanContext(PlannerConfig.from_args(args))
    plan = plan_path(terrain, start, goal, args.planner, ctx)
    del ctx
    gc.collect()  # CostToGo / gated solver buffers go before the ostrich build (3 GiB GPU)
    gate = f", gate {plan['gate']}" if plan["gate"] else ""
    if not plan["reached"]:
        why = "no feasible path" if not plan["reachable"] else "V* finite but the policy did not arrive"
        msg = f"{args.planner}{gate}: {why} (V* = {plan['v_start']:.2f}) -- nothing to simulate"
        if args.view:
            print(f"{msg}; opening the viewer at the start pose, robot parked")
            show_parked(args, terrain, start, goal)
        raise SystemExit(msg)
    max_err = "  ".join(f"max predicted {k} {v:.4f}" for k, v in plan["max_err"].items())
    print(f"plan{gate}: {plan['path_m']:.2f} m, V* = {plan['v_start']:.2f}, {len(plan['xy'])} lattice "
          f"poses, {plan['n_settle_bad']} settle-infeasible  {max_err}")
    path = resample_polyline(np.vstack([plan["xy"], goal]))
    path_len = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())

    # --- 2. simulate ----------------------------------------------------------------------------
    sim_config, render_config, engine_config, logging_config = compose_ostrich_config(tuple(args.override))
    render_config.vis_type = "gl" if args.view else "null"
    sim_config.num_worlds = repeats
    dt = float(sim_config.target_timestep_seconds)
    drive_steps = int(math.ceil(args.slack * path_len / args.v / dt))
    T = args.settle_steps + drive_steps
    kappa_max = controller_kappa_max()
    print(f"ostrich: {repeats} world(s), v={args.v} m/s, lookahead {args.lookahead} m, mu={args.mu}, "
          f"dt={dt}, {args.settle_steps} settle + {drive_steps} drive steps ({drive_steps * dt:.0f} s)")
    t0 = time.time()
    sim = None
    try:
        sim = PurePursuitSimulator(
            sim_config, render_config, engine_config, logging_config,
            k_p=K_P, **friction_kwargs(args.mu), terrain=terrain,
            spawn_pose=np.tile(np.array(start, np.float64), (repeats, 1)),
            paths=[path] * repeats, settle_steps=args.settle_steps, lookahead=args.lookahead, v=args.v,
            kappa_max=kappa_max,
        )
        if args.view:
            pos, pitch, yaw, dist = corner_camera(terrain, start, path, sim.viewer.camera.fov)
            sim.viewer.set_camera(pos=wp.vec3(*pos), pitch=pitch, yaw=yaw)
            sim.viewer.camera.sync_pivot_to_view(dist)  # mouse orbit turns around the path's midpoint
            sim.viewer._paused = True  # start paused: SPACE runs, "." steps (no public setter)
        pose, wheel_qd = sim.rollout(T, args.view)
    finally:
        sim = None
        gc.collect()
    print(f"ostrich rollout done in {time.time() - t0:.1f} s")
    pose, wheel_qd = pose[args.settle_steps :], wheel_qd[args.settle_steps :]
    if len(pose) == 0:
        raise SystemExit("viewer closed during the settle -- nothing to judge")

    # --- 3. verdict -----------------------------------------------------------------------------
    results = [judge(pose[:, w], dt, path, terrain) for w in range(repeats)]
    print(f"\n{'id':>3} {'status':>9} {'t_arr':>6} {'climb':>6} {'roll':>5} {'cte_max':>7} "
          f"{'cte_mean':>8} {'cte@climb':>9} {'progress':>12}")
    for w, r in enumerate(results):
        print(f"{w:3d} {r['status']:>9} {r['t_arrive']:6.1f} {r['climb_deg']:6.1f} {r['roll_deg']:5.1f} "
              f"{r['cte_max']:7.3f} {r['cte_mean']:8.3f} {r['cte_at_climb']:9.3f} "
              f"{r['progress_m']:5.2f}/{path_len:<5.2f}")
    n_ok = sum(r["status"] == "arrived" for r in results)
    print(f"\n{n_ok}/{repeats} arrived  ({args.planner}{gate}, {map_path.name})")

    title = (f"{map_path.name}: {args.planner}{gate}\nplanned {plan['path_m']:.1f} m, "
             f"{n_ok}/{repeats} arrived in ostrich (v {args.v} m/s, lookahead {args.lookahead} m)")
    plot_run(terrain, plan, path, results, start, title, args.out_dir / f"{run_name}.png")
    write_comparison(
        args.out_dir / f"{run_name}.h5",
        root=dict(
            name=run_name, map=str(map_path), planner=args.planner, head=str(plan["head"]),
            tau=plan["tau"], gate=plan["gate"], n=repeats, v=args.v, lookahead=args.lookahead, mu=args.mu,
            slack=args.slack, settle_steps=args.settle_steps, path_m=path_len,
            obstacle_x=0.5 * (start[0] + goal[0]),
        ),
        per_variant=dict(
            spawn_pose=np.tile(np.array(start, np.float32), (repeats, 1)),
            v_drive=np.full(repeats, args.v, np.float32),
            wz_drive=np.zeros(repeats, np.float32),
            variant_value=np.arange(repeats, dtype=np.float32),
            variant_label=np.array([f"{run_name}_r{w}" for w in range(repeats)]),
            status=np.array([r["status"] for r in results]),
            planned_path=path.astype(np.float32),
            **{key: np.array([r[key] for r in results], np.float32)
               for key in ("t_arrive", "climb_deg", "roll_deg", "cte_max", "cte_mean",
                           "cte_at_climb", "progress_m")},
        ),
        terrain_entries=[(map_path, terrain)] * repeats,
        ostrich=dict(dt=dt, t=np.arange(len(pose), dtype=np.float32) * dt, pose=pose,
                     wheel_qd=wheel_qd),
        hstack=dict(dt=dt),
    )


if __name__ == "__main__":
    main()
