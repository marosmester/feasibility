"""Plan out of the garage, then actually DRIVE that plan in ostrich -- the physics check on a route
whose decisive manoeuvre is a point turn.

`demos/garage_gate.py` is the planner-side experiment on `heightmap/create_garage.py`'s A/B pair: a
U-shaped garage the robot is parked in facing a side wall, empty in map A and with one 0.20 m curb
across the floor in map B. The only way out of either is a 90 deg POINT TURN, and in map B that turn
drags the rear wheel sideways into the curb -- which the settle cannot see (the body tilts inside its
envelope, so `blocked` stays 0) and `nn-gated-pivot` refuses. Nothing is ever simulated there: the
verdict rests on a predicted `e_pitch` and on geometry.

`demos/ostrich_follow_path.py` is the physics-side counterpart for the uphill series, but it cannot
be pointed at this map. Pure pursuit holds a constant forward speed and derives `wz = v * kappa`,
so `v = 0` means `wz = 0`: a point turn is not a manoeuvre it can express, and `resample_polyline`
deletes pivots (repeated vertices) before the controller would see them anyway. Handed the garage it
would drive forward into the side wall 0.95 m ahead -- which is not a tuning failure but the map
working as designed, since the garage is dimensioned so that no forward arc gets the nose round.

So this demo drives the plan with `feasibility.planning.pivot_pursuit` instead: the plan's own
primitives become PIVOT phases (closed-loop on the measured yaw -- an in-place skid-steer turn runs
~2x slower than its no-slip command, so an open-loop one simply under-turns) and DRIVE phases
(pure pursuit, unchanged). Otherwise the shape is `ostrich_follow_path.py`'s:

    1. load the garage map (A or B, or any heightmap stem), start/goal from its sidecar
    2. plan start -> goal with --planner, `pivot_cost` > 0 so the lattice may turn in place
    3. report the PLANNED turn: how many bins, and what its wheels are standing on
    4. drive it in ostrich, --repeats worlds in one build (ostrich is nondeterministic on
       contact-rich driving, and a point turn on near-tangential contacts is the worst case)
    5. report per repeat: arrived / flipped / off_map / stalled, plus a turn table -- what the
       chassis did WHILE it was turning, which is the only window the curb can show up in

The turn table exists because `evaluation.judge`'s `climb_deg` is `max(-pitch, 0)`, nose-UP only,
and the garage's curb is sized against the REAR wheel -- it pitches the body nose-DOWN inside
`max_pitch_down` while a front wheel on it would break `max_roll`. The existing metric would miss
precisely the feature the map was built around, so the turn table reports both signs.

The interesting runs are a controlled pair. `vanilla-on` plans the same route on both maps (the
settle cannot see the curb, so there is nothing to plan around), so running it on A and on B drives
ONE plan over two terrains that differ by one feature -- `comparator`'s own pattern, with the
difference attributable to the curb alone. `nn-gated-pivot` then refuses map B outright: there is no
path to drive, which is the result rather than an error, and --view parks the robot where the plan
gave up.

CLI parameters:
    --map A|B|PATH        garage_a / garage_b under --map-dir, or any heightmap stem (so
                          create_pivot_pocket.py's map works too). Default A
    --map-dir PATH        where create_garage.py wrote the pair (default assets/garage)
    --planner NAME        any planners.PLANNERS key (default vanilla-on)
    --start X Y YAW       override the sidecar's start;  --goal X Y  likewise
    --pivot-cost M        m-equivalent per 15 deg bin for a point turn (default 0.15). 0 builds a
                          forward-only lattice, which finds no way out of the garage at all
    --omega RAD_S         commanded pivot yaw rate (default arc.OMEGA_NOM, the rate the gating
                          network's pivot rows were generated at)
    --yaw-tol DEG         a pivot phase is done this close to its target heading (default 2)
    --anchor-yaw          turn the PLANNED number of degrees instead of driving the lattice's own
                          bin centres (see pivot_pursuit.plan_phases; worth up to 7.5 deg)
    --pivot-slack X       timeout budget per bin, x its no-slip time (default 4, ~2x measured)
    --relief M            wheel relief counted as standing on something (default 0.05)
    --repeats N           worlds following the same plan (default 3; 1 with --view)
    --view                live GL viewer instead of the headless rollout; starts PAUSED, SPACE runs
    --torch-device D      network inference device (default cuda -- nn-gated-pivot needs a
                          predicted error at EVERY lattice pose, minutes on CPU)
    ... plus ostrich_follow_path.py's shared flags (--tau-*, --checkpoint*, --v, --lookahead, --mu,
        --slack, --settle-steps, --chunk, --override)

Outputs (<out-dir>/<map>_<planner>.*)
    .png       bird's-eye elevation, planned path dashed, each repeat's executed track by outcome
    _turn.png  pitch / roll / yaw against time with the point turns shaded -- this demo's own plot
    .h5        comparator/provenance's schema, plus the per-step phase index; replay repeat K with
               python src/feasibility/replay/gl_replay.py --file <...>.h5 --id K --which ostrich

Run:
    python demos/ostrich_follow_plan_turning.py --map A --planner vanilla-on
    python demos/ostrich_follow_plan_turning.py --map B --planner vanilla-on     # the same plan
    python demos/ostrich_follow_plan_turning.py --map B --planner nn-gated-pivot # refused
    python demos/ostrich_follow_plan_turning.py --map A --planner nn-gated-pivot --view
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
from feasibility.planning.evaluation import pitch_roll
from feasibility.planning.gated_lattice import N_PRIM_ARC
from feasibility.planning.pivot_pursuit import bin_centre
from feasibility.planning.pivot_pursuit import KIND_PIVOT
from feasibility.planning.pivot_pursuit import plan_phases
from feasibility.planning.pivot_pursuit import PIVOT_OMEGA
from feasibility.planning.pivot_pursuit import PIVOT_TIMEOUT_SLACK
from feasibility.planning.pivot_pursuit import pivot_timeout_steps
from feasibility.planning.pivot_pursuit import PivotPursuitSimulator
from feasibility.planning.pivot_pursuit import YAW_TOL
from feasibility.planning.planners import plan_path
from feasibility.planning.planners import PlanContext
from feasibility.planning.planners import PlannerConfig
from feasibility.planning.planners import PLANNERS
from feasibility.planning.pure_pursuit import controller_kappa_max

try:  # `python demos/x.py` puts demos/ on sys.path, `python -m demos.x` puts the repo root
    from demos import ostrich_follow_path as ofp
    from demos import view_planned_path as vpp
except ModuleNotFoundError:  # pragma: no cover - the bare-script path
    import ostrich_follow_path as ofp
    import view_planned_path as vpp

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MAP_DIR = REPO_ROOT / "assets" / "garage"
RELIEF = 0.05  # [m] a wheel standing on something (lattice_learning's trial.interact_relief)


def resolve_map(arg: str, map_dir: pathlib.Path) -> pathlib.Path:
    """"A"/"B" name create_garage.py's pair; anything else is taken as a heightmap stem."""
    if arg.upper() in ("A", "B"):
        return map_dir / f"garage_{arg.lower()}"
    return pathlib.Path(arg).with_suffix("")


def report_plan(plan: dict, terrain: HeightMapReader, threshold: float, tau: float) -> dict:
    """What the plan intends to do, before any of it is simulated: how much of the route is turned
    in place, and what the wheels are standing on while it turns.

    Relief is measured with `view_planned_path.wheel_relief` -- the HIGHEST ground within a
    `wheel_radius` of each wheel centre, which is the settle's own dilated envelope and the tyre's.
    Sampling the height under the centre instead reads a wheel dragging its rim along a curb face as
    standing on flat ground, which is exactly the geometry this demo is pointed at.
    """
    poses = vpp.path_poses(plan)
    prims = np.asarray(plan["prims"], int) if plan["prims"] else np.zeros(0, int)
    pivot = prims >= N_PRIM_ARC
    # a pivot's contact is under the poses it turns BETWEEN, so take both ends of each pivot arc
    idx = np.nonzero(pivot)[0]
    touched = np.unique(np.concatenate([idx, idx + 1])) if len(idx) else np.zeros(0, int)
    relief = vpp.wheel_relief(poses, terrain)
    turn_deg = float(len(idx) * 360.0 / 24.0)
    worst = float(relief[touched].max()) if len(touched) else 0.0
    n_on = int((relief[touched] > threshold).any(axis=1).sum()) if len(touched) else 0
    print(f"plan: {plan['path_m']:.2f} m, {plan['n_poses']} lattice poses, {len(idx)} point turns "
          f"({turn_deg:.0f} deg in place), {plan['n_settle_bad']} settle-infeasible")
    print(f"      at the {len(touched)} turning poses: {n_on} have a wheel on relief > "
          f"{threshold:.2f} m, worst {worst:.2f} m")
    errs = plan.get("arc_errors", {})
    if "e_pitch" in errs and len(idx):
        p = np.asarray(errs["e_pitch"])[pivot]
        print(f"      the net's predicted e_pitch on those turns: max {p.max():.4f}, median "
              f"{np.median(p):.4f} rad, {int((p > tau).sum())}/{len(p)} over tau {tau:.4f}")
    return dict(n_pivot=len(idx), turn_deg=turn_deg, worst_relief=worst, n_on_relief=n_on)


def yaw_of(q: np.ndarray) -> np.ndarray:
    """(qx, qy, qz, qw) [..., 4] -> heading about z. `evaluation.pitch_roll` deliberately stops at
    pitch and roll, the two axes a verdict on a CLIMB needs; a turn is judged on the third."""
    return np.arctan2(2.0 * (q[..., 3] * q[..., 2] + q[..., 0] * q[..., 1]),
                      1.0 - 2.0 * (q[..., 1] ** 2 + q[..., 2] ** 2))


def turn_metrics(
    pose: np.ndarray,
    phase: np.ndarray,
    phases: dict[str, np.ndarray],
    dt: float,
    terrain: HeightMapReader,
) -> dict:
    """What the chassis did while it was TURNING IN PLACE -- the window the curb can act in.

    Reports pitch both ways round. `judge`'s `climb_deg` is `max(-pitch, 0)`, nose-up only, and the
    garage curb is sized against the REAR wheel: it pitches the body nose-DOWN within
    `max_pitch_down` while a front wheel on it would break `max_roll`. Nose-up alone cannot see it.

    `relief_m` is the same audit `report_plan` runs on the PLANNED poses, run instead on the ones
    ostrich actually reached: the highest ground within a `wheel_radius` of any wheel centre while
    turning. It is what makes a null result legible -- a turn that comes out flat because the robot
    never reached the obstacle reads very differently from one that drove over it and shrugged, and
    a point turn drifts (the reference point is the front axle, the rear wheel is dragged), so where
    the plan put a wheel is not where the wheel went.

    Several disjoint turns are pooled for the extremes and the elapsed time; the yaw and the drift
    are measured from the first turning step to the last, which is the whole manoeuvre.
    """
    is_pivot = phases["kind"] == KIND_PIVOT
    mask = is_pivot[np.clip(phase, 0, len(is_pivot) - 1)]
    if not mask.any():
        return dict(n_steps=0, t_turn=math.nan, pitch_up_deg=math.nan, pitch_down_deg=math.nan,
                    roll_deg=math.nan, yaw_deg=math.nan, drift_m=math.nan, relief_m=math.nan)
    lo, hi = int(np.nonzero(mask)[0][0]), int(np.nonzero(mask)[0][-1])
    pitch, roll, _ = pitch_roll(pose[mask, 3:7])
    yaw = yaw_of(pose[mask, 3:7])
    d = float(yaw_of(pose[hi, 3:7]) - yaw_of(pose[lo, 3:7]))
    driven = np.column_stack([pose[mask, 0], pose[mask, 1], yaw])
    return dict(
        n_steps=int(mask.sum()),
        t_turn=float(mask.sum() * dt),
        pitch_up_deg=float(np.degrees(max(-pitch.min(), 0.0))),
        pitch_down_deg=float(np.degrees(max(pitch.max(), 0.0))),
        roll_deg=float(np.degrees(np.abs(roll).max())),
        yaw_deg=float(np.degrees(math.atan2(math.sin(d), math.cos(d)))),
        drift_m=float(np.linalg.norm(pose[hi, :2] - pose[lo, :2])),
        relief_m=float(vpp.wheel_relief(driven, terrain).max()),
    )


def plot_turn(
    pose: np.ndarray,
    phase: np.ndarray,
    phases: dict[str, np.ndarray],
    dt: float,
    results: list[dict],
    title: str,
    out: pathlib.Path,
) -> None:
    """Pitch / |roll| and yaw against time, with the point turns shaded -- this demo's own plot.
    The turn is where the curb acts, so shading it is what makes an A/B pair readable side by side."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = pose.shape[1]
    t = np.arange(len(pose)) * dt
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, layout="constrained")
    is_pivot = phases["kind"] == KIND_PIVOT
    mask = is_pivot[np.clip(phase[:, 0], 0, len(is_pivot) - 1)]  # world 0 sets the shading
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    for lo, hi in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]):
        ax0.axvspan(t[lo], t[min(hi, len(t) - 1)], color="0.85", zorder=0)
        ax1.axvspan(t[lo], t[min(hi, len(t) - 1)], color="0.85", zorder=0,
                    label="_" if lo else "turning in place")
    for w in range(n):
        pitch, roll, _ = pitch_roll(pose[:, w, 3:7])
        c = ofp.STATUS_COLORS[results[w]["status"]]
        ax0.plot(t, np.degrees(pitch), "-", lw=1.2, color=c, label=f"pitch {w} ({results[w]['status']})")
        ax0.plot(t, np.degrees(np.abs(roll)), "--", lw=1.0, color=c, alpha=0.7, label=f"|roll| {w}")
        ax1.plot(t, np.degrees(np.unwrap(yaw_of(pose[:, w, 3:7]))), "-", lw=1.2, color=c,
                 label=f"yaw {w}")
    for target in phases["target"][is_pivot]:
        ax1.axhline(math.degrees(target), color="black", lw=0.6, ls=":", alpha=0.6)
    ax0.set_ylabel("pitch (nose-up < 0) and |roll| [deg]")
    ax0.axhline(0.0, color="black", lw=0.6)
    ax0.legend(fontsize=7, ncol=2, loc="upper left")
    ax1.set_ylabel("yaw [deg], dotted = the planned bin centres")
    ax1.set_xlabel("t [s]")
    ax1.legend(fontsize=7, loc="upper left")
    ax0.set_title(title, fontsize=10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--map", default="A", help="A | B | a heightmap stem")
    parser.add_argument("--map-dir", type=pathlib.Path, default=DEFAULT_MAP_DIR)
    parser.add_argument("--planner", choices=list(PLANNERS), default="vanilla-on")
    parser.add_argument("--start", type=float, nargs=3, metavar=("X", "Y", "YAW"))
    parser.add_argument("--goal", type=float, nargs=2, metavar=("X", "Y"))
    ofp.add_shared_args(parser)
    # A forward-only lattice cannot leave the garage at all, and nn-gated-pivot has nothing to gate
    # without the point turns; the network gate also needs a predicted error at EVERY lattice pose,
    # which is ~17 s on a GPU and minutes on a CPU. Both differ from the uphill demo's defaults.
    parser.set_defaults(pivot_cost=0.15, torch_device="cuda")
    parser.add_argument("--omega", type=float, default=PIVOT_OMEGA, help="[rad/s] pivot yaw rate")
    parser.add_argument("--yaw-tol", type=float, default=math.degrees(YAW_TOL), help="[deg]")
    parser.add_argument("--anchor-yaw", action="store_true",
                        help="turn the planned angle rather than onto the lattice's bin centres")
    parser.add_argument("--pivot-slack", type=float, default=PIVOT_TIMEOUT_SLACK)
    parser.add_argument("--relief", type=float, default=RELIEF)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--out-dir", type=pathlib.Path,
                        default=REPO_ROOT / "outputs" / "follow_plan_turning")
    args = parser.parse_args()

    map_path = resolve_map(args.map, args.map_dir)
    if not map_path.with_suffix(".png").exists():
        raise SystemExit(f"{map_path}.png does not exist -- run "
                         f"`python src/feasibility/heightmap/create_garage.py`")
    meta = yaml.safe_load(map_path.with_suffix(".yaml").read_text())
    start = tuple(args.start) if args.start is not None else meta.get("start")
    goal = tuple(args.goal) if args.goal is not None else meta.get("goal")
    if start is None or goal is None:
        raise SystemExit(f"{map_path.name}.yaml has no start/goal -- pass --start X Y YAW --goal X Y")
    start = tuple(float(v) for v in start)
    goal = (float(goal[0]), float(goal[1]))
    terrain = HeightMapReader.load(map_path)
    repeats = 1 if args.view else args.repeats
    run_name = f"{map_path.name}_{args.planner}"
    if args.pivot_cost <= 0.0:
        print("warning: --pivot-cost 0 builds a forward-only lattice, which cannot turn in place")

    # --- 1. plan --------------------------------------------------------------------------------
    curb = float(meta.get("curb_height", 0.0) or 0.0)
    what = f"curb {curb:.2f} m at x = {meta['curb_x']:.2f}" if curb else "no curb"
    print(f"=== {map_path.name} ({what}): planning with {args.planner}, pivot_cost "
          f"{args.pivot_cost}, start {start} -> goal {goal} ===")
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
            ofp.show_parked(args, terrain, start, goal)
        raise SystemExit(msg)

    # --- 2. what the plan intends ---------------------------------------------------------------
    print(f"gate: {plan['gate'] or 'none (the settle alone)'}")
    intent = report_plan(plan, terrain, args.relief, float(plan["tau"]))
    anchor = (start[2] - bin_centre(start[2])) if args.anchor_yaw else 0.0
    path, phases = plan_phases(plan, goal, anchor=anchor)
    if intent["n_pivot"] == 0:
        print("note: this plan never turns in place, so the follower reduces to plain pure pursuit")
    path_len = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())

    # --- 3. simulate ----------------------------------------------------------------------------
    sim_config, render_config, engine_config, logging_config = compose_ostrich_config(tuple(args.override))
    render_config.vis_type = "gl" if args.view else "null"
    sim_config.num_worlds = repeats
    dt = float(sim_config.target_timestep_seconds)
    budget = pivot_timeout_steps(dt, args.pivot_slack, args.omega)
    drive_steps = int(math.ceil(args.slack * path_len / args.v / dt))
    T = args.settle_steps + drive_steps + intent["n_pivot"] * budget
    print(f"ostrich: {repeats} world(s), v={args.v} m/s, lookahead {args.lookahead} m, mu={args.mu}, "
          f"dt={dt}, {args.settle_steps} settle + {drive_steps} drive + "
          f"{intent['n_pivot']} x {budget} turn steps ({T * dt:.0f} s)")
    t0 = time.time()
    sim, timeouts = None, np.zeros(repeats, int)
    try:
        sim = PivotPursuitSimulator(
            sim_config, render_config, engine_config, logging_config,
            k_p=K_P, **friction_kwargs(args.mu), terrain=terrain,
            spawn_pose=np.tile(np.array(start, np.float64), (repeats, 1)),
            paths=[path] * repeats, phases=[phases] * repeats, settle_steps=args.settle_steps,
            lookahead=args.lookahead, v=args.v, kappa_max=controller_kappa_max(),
            omega=args.omega, yaw_tol=math.radians(args.yaw_tol), pivot_timeout_steps=budget,
        )
        if args.view:
            pos, pitch, yaw, dist = ofp.corner_camera(terrain, start, path, sim.viewer.camera.fov)
            sim.viewer.set_camera(pos=wp.vec3(*pos), pitch=pitch, yaw=yaw)
            sim.viewer.camera.sync_pivot_to_view(dist)  # mouse orbit turns around the path's midpoint
            sim.viewer._paused = True  # start paused: SPACE runs, "." steps (no public setter)
        pose, wheel_qd, phase_log = sim.rollout(T, args.view)
        timeouts = sim.timeouts()
    finally:
        sim = None
        gc.collect()
    print(f"ostrich rollout done in {time.time() - t0:.1f} s")
    pose, wheel_qd = pose[args.settle_steps :], wheel_qd[args.settle_steps :]
    phase_log = phase_log[args.settle_steps :]
    if len(pose) == 0:
        raise SystemExit("viewer closed during the settle -- nothing to judge")

    # --- 4. verdict -----------------------------------------------------------------------------
    results = [judge(pose[:, w], dt, path, terrain) for w in range(repeats)]
    turns = [turn_metrics(pose[:, w], phase_log[:, w], phases, dt, terrain)
             for w in range(repeats)]
    print(f"\n{'id':>3} {'status':>9} {'t_arr':>6} {'roll':>5} {'cte_max':>7} {'cte_mean':>8} "
          f"{'progress':>12}")
    for w, r in enumerate(results):
        print(f"{w:3d} {r['status']:>9} {r['t_arrive']:6.1f} {r['roll_deg']:5.1f} "
              f"{r['cte_max']:7.3f} {r['cte_mean']:8.3f} {r['progress_m']:5.2f}/{path_len:<5.2f}")
    print(f"\nwhile turning in place ({intent['n_pivot']} planned bins = {intent['turn_deg']:.0f} deg):")
    print(f"{'id':>3} {'t_turn':>7} {'yaw':>7} {'nose_up':>8} {'nose_down':>10} {'|roll|':>7} "
          f"{'drift':>6} {'relief':>7} {'timeouts':>9}")
    for w, m in enumerate(turns):
        print(f"{w:3d} {m['t_turn']:7.1f} {m['yaw_deg']:7.1f} {m['pitch_up_deg']:8.1f} "
              f"{m['pitch_down_deg']:10.1f} {m['roll_deg']:7.1f} {m['drift_m']:6.2f} "
              f"{m['relief_m']:7.2f} {int(timeouts[w]):9d}")
    reached = max(m["relief_m"] for m in turns)
    if intent["worst_relief"] > args.relief and reached <= args.relief:
        print(f"the PLAN put a wheel on {intent['worst_relief']:.2f} m of relief while turning, but "
              f"no repeat actually reached it ({reached:.2f} m):\n  the turn drifts, so a flat turn "
              f"table here says the robot missed the feature, not that the feature is harmless.")
    n_ok = sum(r["status"] == "arrived" for r in results)
    print(f"\n{n_ok}/{repeats} arrived  ({args.planner}{gate}, {map_path.name})")
    if timeouts.any():
        print("a point turn hit its timeout without reaching the heading -- a wheel was jammed, or "
              "--pivot-slack is too tight")

    # --- 5. outputs -----------------------------------------------------------------------------
    title = (f"{map_path.name}: {args.planner}{gate}\nplanned {plan['path_m']:.1f} m with "
             f"{intent['n_pivot']} point turns, {n_ok}/{repeats} arrived in ostrich")
    ofp.plot_run(terrain, plan, path, results, start, title, args.out_dir / f"{run_name}.png")
    plot_turn(pose, phase_log, phases, dt, results, title, args.out_dir / f"{run_name}_turn.png")
    write_comparison(
        args.out_dir / f"{run_name}.h5",
        root=dict(
            name=run_name, map=str(map_path), planner=args.planner, head=str(plan["head"]),
            tau=plan["tau"], gate=plan["gate"], n=repeats, v=args.v, lookahead=args.lookahead,
            mu=args.mu, slack=args.slack, settle_steps=args.settle_steps, path_m=path_len,
            pivot_cost=args.pivot_cost, omega=args.omega, yaw_tol=args.yaw_tol,
            anchor_yaw=anchor, curb_height=curb, n_pivot=intent["n_pivot"],
            planned_turn_deg=intent["turn_deg"], worst_pivot_relief=intent["worst_relief"],
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
            phase_kind=phases["kind"].astype(np.int32),
            phase_target=phases["target"].astype(np.float32),
            pivot_timeouts=timeouts.astype(np.int32),
            **{key: np.array([r[key] for r in results], np.float32)
               for key in ("t_arrive", "climb_deg", "roll_deg", "cte_max", "cte_mean",
                           "cte_at_climb", "progress_m")},
            **{f"turn_{key}": np.array([m[key] for m in turns], np.float32)
               for key in ("t_turn", "yaw_deg", "pitch_up_deg", "pitch_down_deg", "roll_deg",
                           "drift_m", "relief_m")},
        ),
        terrain_entries=[(map_path, terrain)] * repeats,
        ostrich=dict(dt=dt, t=np.arange(len(pose), dtype=np.float32) * dt, pose=pose,
                     wheel_qd=wheel_qd, phase=phase_log.astype(np.int32)),
        hstack=dict(dt=dt),
    )


if __name__ == "__main__":
    main()
