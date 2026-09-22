"""Does the network see the curb? Plan out of a U-shaped garage with and without one.

`heightmap/create_garage.py` writes two maps that differ by a single feature: a three-walled
garage the robot is parked in facing a side wall, empty in map A and with one 0.20 m curb across
the floor in map B. The only way out of either is a POINT TURN -- the garage is narrow enough that
no forward arc can swing the nose round -- and in map B that turn drags the rear wheel sideways
into the curb. The settle cannot see it (the body tilts 14.9 deg, inside its 15 deg envelope), so
`blocked` stays 0 and the plan goes straight over it.

This runs three arms on each map and prints them side by side:

    vanilla-on, pivot_cost 0   the settle, forward arcs only      -- must find NO route
    vanilla-on                 the settle, point turns allowed    -- the shipped pipeline
    nn-gated-pivot             the same, plus the v_wz net's e_pitch > TAU_PIVOT_PITCH on the
                               POINT TURNS only, forward arcs left open

and then the evidence underneath the verdict: the predicted e_pitch of every point turn the
lattice can place inside the garage, split by whether that turn's swept tyre actually reaches the
curb. The ground truth is geometric (`create_garage.feature_distances`), measured to the wheel's
own radius rather than to its centre -- a wheel whose centre is 0.2 m short of a curb face is
already riding up it, and a centre-only audit reads that as flat ground.

Gating needs a predicted error at EVERY lattice pose, not just along a path, so each map costs one
`arc_network.vwz_error_fields` pass: ~17 s on this machine's GPU, minutes on CPU. `--torch-device`
defaults to cuda for that reason; the root .venv's cu128 torch covers this GPU, and CLAUDE.md's
note about needing `.venv-cu126` applies to the ThinkPad's GTX 1050, not here.

`--view` then draws what the arms decided, in the Newton GL viewer: the garage mesh, the Helhest
Junior parked at the start pose, and one RED ARROW per planned pose of the chosen arm's path (a
point turn spirals up, as in `view_planned_path.py`, whose drawing this reuses). An arm that found
NO ROUTE -- map B under the gate, which is the result the demo exists for -- still gets its scene,
so the robot is visible boxed in where the plan gave up, but no path is drawn over it and the
console says so. With both maps, RIGHT/N and LEFT/P switch between them in the one window.

CLI parameters:
    --map A|B|both    which garage to run (default both)
    --map-dir PATH    where create_garage.py wrote them (default assets/garage)
    --tau RAD         the point-turn gate's e_pitch tolerance (default planners.TAU_PIVOT_PITCH)
    --tau-sweep       also report which taus keep a route open, once the field is in hand
    --torch-device D  cuda (default) or cpu
    --pivot-cost M    m-equivalent per 15 deg bin for a point turn (default 0.15)
    --view            after the numbers, open the GL viewer on the plan
    --view-arm NAME   which arm --view draws: nn-gated-pivot (default), vanilla-on,
                      vanilla-on-pivot0
    --arrow-len M     arrow length (default view_planned_path.ARROW_LEN)
    --no-robot        skip the scale model; --no-trail drops the line linking the poses

Usage:
    python demos/garage_gate.py
    python demos/garage_gate.py --map B --tau-sweep
    python demos/garage_gate.py --view                 # both maps, gated plan, side by side
    python demos/garage_gate.py --map A --view --view-arm vanilla-on
    python demos/view_planned_path.py --map assets/garage/garage_b --planner nn-gated-pivot \
        --torch-device cuda            # the same plan, drawn, with its per-pose table
"""
from __future__ import annotations

import argparse
import math
import pathlib

import numpy as np
import warp as wp
import yaml

from feasibility.heightmap import create_garage as garage
from feasibility.heightmap.heightmap_reader import HeightMapReader
from feasibility.planning.gated_lattice import gated_solve
from feasibility.planning.gated_lattice import lattice_state
from feasibility.planning.gated_lattice import N_PRIM_ARC
from feasibility.planning.gated_lattice import N_THETA
from feasibility.planning.planners import plan_path
from feasibility.planning.planners import PlanContext
from feasibility.planning.planners import PlannerConfig
from feasibility.planning.planners import TAU_PIVOT_PITCH

try:  # `python demos/garage_gate.py` puts demos/ on sys.path, `python -m demos.…` puts the root
    from demos import view_planned_path as vpp
except ModuleNotFoundError:  # pragma: no cover - the bare-script path
    import view_planned_path as vpp

path_poses, wheel_relief = vpp.path_poses, vpp.wheel_relief

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MAP_DIR = REPO_ROOT / "assets" / "garage"
RELIEF = 0.05  # m, what counts as a wheel standing on something (trial.interact_relief)
# --view-arm name -> the key run_map files that arm's result under
VIEW_ARMS = {"nn-gated-pivot": "gated", "vanilla-on": "on", "vanilla-on-pivot0": "no_pivot"}
# Spread across the band `planning.tune_pivot_tau.py` puts the gate's threshold in (0.06-0.11 on
# the current checkpoint) and below it, because the width of the window the two maps part company
# in -- and whether the calibrated tau is inside it -- is the thing worth knowing. TAU_PIVOT_PITCH
# itself is always evaluated too, wherever it currently sits.
TAU_SWEEP = tuple(sorted(  # sorted/deduped, so the rows read in order wherever the constant sits
    {0.03, 0.05, 0.07, 0.0873, 0.095, 0.105, 0.12, 0.15, 0.20, 0.30, TAU_PIVOT_PITCH}
))


def pivot_truth(terrain: HeightMapReader, ctg, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """(touch, legal) [ny, nx, n_theta, 2] over the lattice's own poses: does each single-bin point
    turn put a wheel's TYRE within reach of the curb, and would the settle allow it at all?

    Geometry, not physics -- this is what the network's predicted e_pitch is scored against. The
    turn is sampled at sub-headings across the bin because the wheel sweeps continuously through
    it, and `feature_distances` rasterises the walls and the curb separately from the rectangles
    the map was built from, so the wall's own ramp is never mistaken for curb.
    """
    curb_x = params["curb_x"] or garage.CURB_X
    # the spur is read off the sidecar, not defaulted: a map written with --spur-x records None
    # and has to be scored as the rib-only map it is
    walls, curbs = garage.garage_rects(params["wall_gap"], params["back_gap"],
                                       params["garage_x"][1], -params["garage_x"][0], curb_x,
                                       params.get("spur_x") or curb_x,
                                       params.get("spur_y") or garage.SPUR_Y)
    dist_wall, dist_curb = garage.feature_distances(terrain, walls, curbs, params["wall_height"],
                                                    params["curb_height"])
    grid = ctg.grid
    ny, nx = grid.cells_y, grid.cells_x
    jj, ii = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
    xy = np.stack([grid.origin_x + jj * grid.cell_size, grid.origin_y + ii * grid.cell_size], -1)
    r, sub = float(garage._ROBOT.wheel_radius), np.linspace(0.0, 1.0, 5)
    touch = np.zeros((ny, nx, N_THETA, 2), bool)
    legal = np.zeros((ny, nx, N_THETA, 2), bool)
    for p, d in enumerate((-1, 1)):  # the solver's own pivot order: cw then ccw
        for b in range(N_THETA):
            yaw = (b + 0.5) * garage.BIN + d * garage.BIN * sub
            local = garage.wheels_of(np.stack([np.zeros_like(yaw), np.zeros_like(yaw), yaw], -1))
            w = local[None, None] + xy[:, :, None, None, :]
            near = lambda f: garage.sample_dist(f, terrain, w).reshape(ny, nx, -1).min(-1)
            touch[:, :, b, p], legal[:, :, b, p] = near(dist_curb) < r, near(dist_wall) >= r
    return touch, legal


def report_field(fields: dict, touch: np.ndarray, keep: np.ndarray, tau: float) -> dict:
    """The predicted e_pitch of every point turn the lattice can place in the garage, split by the
    geometric ground truth. This is the calibration check on a map the network never saw: a
    tolerance in radians is only meaningful if the turns that really touch something score above
    it and the ones on open floor score below."""
    pivots = fields["e_pitch"][..., N_PRIM_ARC:]
    out = {}
    print(f"    {'':<26} {'n':>6} {'median':>8} {'q90':>8} {'max':>8} {'over tau':>9}")
    for tag, key, mask in (("tyre reaches the curb", "touch", keep & touch),
                           ("clear of it", "clear", keep & ~touch)):
        if not mask.any():
            print(f"    {tag:<26} {'none':>6}")
            continue
        v = pivots[mask]
        out[key] = dict(n=int(mask.sum()), median=float(np.median(v)), over=float((v > tau).mean()))
        print(f"    {tag:<26} {int(mask.sum()):6d} {np.median(v):8.4f} "
              f"{np.quantile(v, 0.9):8.4f} {v.max():8.4f} {(v > tau).mean():8.0%}")
    return out


def run_arm(terrain, start, goal, planner, ctx) -> dict:
    """One planner on one map -> the row printed by `print_arm`, with the plan's own audit: how
    many of its point turns really stand on something, measured with the settle's wheel envelope
    rather than the height under the centre."""
    result = plan_path(terrain, start, goal, planner, ctx)
    poses = path_poses(result)
    relief = wheel_relief(poses, terrain)
    is_pivot = np.zeros(len(poses), bool)
    for i in range(1, len(poses)):
        is_pivot[i] = np.allclose(poses[i, :2], poses[i - 1, :2])
    on = (relief > RELIEF).any(axis=1)
    errs = result.get("arc_errors", {})
    prim_pivot = np.asarray(result["prims"]) >= N_PRIM_ARC if result["prims"] else np.zeros(0, bool)
    return dict(
        result=result, n_pivot=int(is_pivot.sum()), pivot_on=int((is_pivot & on).sum()),
        worst_pivot_relief=float(relief[is_pivot].max()) if is_pivot.any() else 0.0,
        pivot_pred=errs["e_pitch"][prim_pivot] if "e_pitch" in errs and len(prim_pivot) else None,
    )


def print_arm(label: str, arm: dict, tau: float) -> None:
    r = arm["result"]
    if not r["reachable"]:
        print(f"    {label:<28} NO ROUTE")
        return
    pred = arm["pivot_pred"]
    col = "" if pred is None or not len(pred) else (
        f"  predicted e_pitch max {pred.max():.3f}, {int((pred > tau).sum())} over tau"
    )
    print(f"    {label:<28} reached {str(r['reached']):<5} {r['n_poses']:3d} poses "
          f"{r['path_m']:6.2f} m   {arm['n_pivot']:2d} point turns, {arm['pivot_on']} of them on "
          f"relief (worst {arm['worst_pivot_relief']:.2f} m){col}")


def run_map(path: pathlib.Path, args: argparse.Namespace) -> dict:
    terrain = HeightMapReader.load(path)
    params = yaml.safe_load(path.with_suffix(".yaml").read_text())
    start = tuple(float(v) for v in params["start"])
    goal = (float(params["goal"][0]), float(params["goal"][1]))
    curb = float(params["curb_height"])
    what = (f"curb {curb:.2f} m at x = {params['curb_x']:.2f}" if curb
            else "no curb -- the control")
    print(f"\n=== {path.name}: {what} ===\n    {params['clear_width']:.2f} m clear, walls "
          f"{params['wall_height']:.2f} m, start {start[0]:.2f} {start[1]:.2f} facing "
          f"{math.degrees(start[2]):.0f} deg -> goal {goal}")

    # forward arcs only: no inference, and it is the premise everything else rests on
    zero = PlanContext(PlannerConfig(pivot_cost=0.0, torch_device=args.torch_device))
    no_pivot = run_arm(terrain, start, goal, "vanilla-on", zero)
    print_arm("vanilla-on, pivot_cost 0", no_pivot, args.tau)
    del zero

    ctx = PlanContext(PlannerConfig(pivot_cost=args.pivot_cost, torch_device=args.torch_device,
                                    tau_pivot_pitch=args.tau))
    on = run_arm(terrain, start, goal, "vanilla-on", ctx)
    print_arm("vanilla-on", on, args.tau)
    gated = run_arm(terrain, start, goal, "nn-gated-pivot", ctx)
    print_arm(f"nn-gated-pivot (tau {args.tau:.4f})", gated, args.tau)

    ctg, solver = ctx.planner_for(terrain)
    fields = ctx.vwz_fields(terrain)
    touch, legal = pivot_truth(terrain, ctg, params)
    gx = ctg.grid.origin_x + np.arange(ctg.grid.cells_x) * ctg.grid.cell_size
    gy = ctg.grid.origin_y + np.arange(ctg.grid.cells_y) * ctg.grid.cell_size
    inside = ((gx > params["garage_x"][0]) & (gx < params["garage_x"][1]))[None, :] & (
        (gy > -params["back_gap"]) & (gy < params["wall_gap"]))[:, None]
    keep = legal & inside[:, :, None, None] & (ctg.blocked.numpy()[..., None] < 0.5)
    print(f"\n    every point turn the lattice can place inside the garage, by whether its swept "
          f"tyre reaches the curb:")
    split = report_field(fields, touch, keep, args.tau)

    sweep = {}
    if args.tau_sweep:
        print(f"\n    which tolerances still leave a route (the field is already in hand, so each "
              f"row is one solve):")
        tilt = ctg.graded_tilt
        err = wp.array(fields["e_pitch"], dtype=wp.float32, device=ctg.device)
        rct = lattice_state(*start, ctg)
        for tau in TAU_SWEEP:
            taus = np.full(ctg.solver.n_prim, math.inf, np.float32)
            taus[N_PRIM_ARC:] = tau
            v = gated_solve(solver, ctg, ctg.blocked, tilt, err, taus)
            sweep[tau] = bool(v[rct] < float(ctg._vcap) * 0.9)
            pruned = (fields["e_pitch"][..., N_PRIM_ARC:][keep] > tau).mean()
            print(f"      tau {tau:6.4f} ({math.degrees(tau):4.1f} deg)  "
                  f"{'route' if sweep[tau] else 'NO ROUTE':>8}   "
                  f"prunes {pruned:4.0%} of the garage's point turns")
    out = dict(name=path.name, curb=curb, no_pivot=no_pivot, on=on, gated=gated, split=split,
               sweep=sweep, terrain=terrain, start=start, goal=goal)
    del ctx  # the CostToGo's settle buffers go before the viewer builds its own model (3 GiB GPU)
    return out


def view_scenes(rows: list[dict], arm: str, args: argparse.Namespace) -> None:
    """The A/B drawn: terrain mesh, the Helhest Junior parked at the start pose for scale, and one
    RED ARROW per planned pose of `arm`'s plan -- `view_planned_path.py`'s drawing reused rather
    than re-derived, point-turn spiral (ARROW_DZ per pose sharing a cell) and amber trail included.

    A map whose chosen arm found NO ROUTE -- which is the whole point of map B under the gate --
    still gets its scene, so the robot can be seen boxed in where the plan gave up, but nothing is
    drawn over it: no arrows, no trail, no goal marker, and the console says so. Passing None to
    log_arrows/log_lines/log_points clears those batches, so a path does not linger from the map
    shown before it.

    With both maps in hand this is one viewer cycling between them (terrain_browser.py's pattern:
    the rebuild happens in the render loop, never inside pyglet's key callback), since two ViewerGL
    windows in one process is not a thing worth finding out about.

    newton/pyglet are imported here rather than at module top so the headless run -- the normal
    one, the numbers being the deliverable -- never needs a GL stack.
    """
    import newton
    import pyglet

    viewer = newton.viewer.ViewerGL()
    pending = {"step": 0}
    if len(rows) > 1:
        def on_key_press(symbol: int, modifiers: int) -> None:
            if symbol in (pyglet.window.key.RIGHT, pyglet.window.key.N):
                pending["step"] += 1
            elif symbol in (pyglet.window.key.LEFT, pyglet.window.key.P):
                pending["step"] -= 1

        viewer.renderer.register_key_press(on_key_press)

    def load(i: int) -> dict:
        row = rows[i]
        terrain, start, result = row["terrain"], row["start"], row[arm]["result"]
        model = vpp.build_model(terrain, start, not args.no_robot)
        viewer.set_model(model)  # may rebind viewer.device, so take it after
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        label = f"[{i + 1}/{len(rows)}] {row['name']} -- {arm}"
        viewer.renderer.set_title(label)
        draw = dict(state=state, arrow_a=None, arrow_b=None, trail_a=None, trail_b=None,
                    goal_pt=None, goal_r=None, goal_c=None)
        if not result["reachable"]:
            print(f"  {label}: NO PATH FOUND -- terrain and robot only, nothing to draw")
            return draw
        dev = viewer.device
        poses = path_poses(result)
        starts, ends = vpp.arrow_segments(poses, terrain, args.arrow_len)
        goal = row["goal"]
        draw["arrow_a"] = wp.array(starts, dtype=wp.vec3, device=dev)
        draw["arrow_b"] = wp.array(ends, dtype=wp.vec3, device=dev)
        if len(poses) > 1 and not args.no_trail:  # a one-pose plan has no segment to link
            draw["trail_a"] = wp.array(starts[:-1], dtype=wp.vec3, device=dev)
            draw["trail_b"] = wp.array(starts[1:], dtype=wp.vec3, device=dev)
        draw["goal_pt"] = wp.array(
            [[goal[0], goal[1], float(terrain.sample(*goal)) + vpp.ARROW_Z]],
            dtype=wp.vec3, device=dev,
        )
        # log_points hands radii/colors straight to a kernel, so both have to be arrays even for
        # the single goal point -- see view_planned_path.py's note.
        draw["goal_r"] = wp.array([vpp.GOAL_RADIUS], dtype=wp.float32, device=dev)
        draw["goal_c"] = wp.array([vpp.COLOR_GOAL], dtype=wp.vec3, device=dev)
        print(f"  {label}: {len(poses)} red heading arrows, {result['path_m']:.2f} m planned")
        return draw

    index = 0
    draw = load(index)
    first = rows[0]["terrain"]
    cx = first.x0 + first.nx * first.cell / 2.0
    cy = first.y0 + first.ny * first.cell / 2.0
    extent = max(first.nx, first.ny) * first.cell
    viewer.set_camera(
        pos=wp.vec3(cx, cy - vpp.CAMERA_BACK * extent, first.max_z + vpp.CAMERA_UP * extent),
        pitch=vpp.CAMERA_PITCH,
        yaw=vpp.CAMERA_YAW,
    )
    print("  orbit with the mouse, F re-frames, ESC quits"
          + (", RIGHT/N and LEFT/P switch maps" if len(rows) > 1 else ""))
    while viewer.is_running():
        if pending["step"]:
            index = (index + pending["step"]) % len(rows)
            pending["step"] = 0
            draw = load(index)
        viewer.begin_frame(0.0)
        viewer.log_state(draw["state"])
        viewer.log_arrows("planned_path", draw["arrow_a"], draw["arrow_b"], vpp.COLOR_PATH)
        viewer.log_lines("planned_trail", draw["trail_a"], draw["trail_b"], vpp.COLOR_TRAIL)
        viewer.log_points("goal", draw["goal_pt"], draw["goal_r"], draw["goal_c"])
        viewer.end_frame()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--map", choices=("A", "B", "both"), default="both")
    ap.add_argument("--map-dir", type=pathlib.Path, default=DEFAULT_MAP_DIR)
    ap.add_argument("--tau", type=float, default=TAU_PIVOT_PITCH,
                    help="[rad] the point-turn gate's e_pitch tolerance")
    ap.add_argument("--tau-sweep", action="store_true")
    ap.add_argument("--torch-device", default="cuda")
    ap.add_argument("--pivot-cost", type=float, default=0.15)
    ap.add_argument("--view", action="store_true",
                    help="after the numbers, draw the plan in the Newton GL viewer")
    ap.add_argument("--view-arm", choices=tuple(VIEW_ARMS), default="nn-gated-pivot",
                    help="which arm's plan --view draws (default the gated one, which is the arm "
                         "the two maps part company on)")
    ap.add_argument("--arrow-len", type=float, default=vpp.ARROW_LEN)
    ap.add_argument("--no-robot", action="store_true", help="skip the scale model")
    ap.add_argument("--no-trail", action="store_true", help="drop the line linking the poses")
    args = ap.parse_args()

    names = {"A": ["garage_a"], "B": ["garage_b"], "both": ["garage_a", "garage_b"]}[args.map]
    paths = [args.map_dir / n for n in names]
    for p in paths:
        if not p.with_suffix(".png").exists():
            raise SystemExit(
                f"{p}.png does not exist -- run `python src/feasibility/heightmap/create_garage.py`"
            )

    wp.init()
    rows = [run_map(p, args) for p in paths]
    if len(rows) == 2:
        print_summary(*rows)
    if args.view:
        print(f"\n=== drawing the {args.view_arm} plan ===")
        view_scenes(rows, VIEW_ARMS[args.view_arm], args)


def print_summary(a: dict, b: dict) -> None:
    """The A/B verdict, printed once both maps have been run."""
    print(f"\n=== {a['name']} vs {b['name']}, the only difference being the curb ===")
    for tag, row in (("without the curb", a), ("with it", b)):
        g = row["gated"]["result"]
        print(f"    {tag:<18} ungated {row['on']['result']['path_m']:5.2f} m with "
              f"{row['on']['n_pivot']} point turns ({row['on']['pivot_on']} on relief)   ->   "
              f"gated {'NO ROUTE' if not g['reachable'] else format(g['path_m'], '.2f') + ' m'}")
    print(
        "\n    The gate reads one head of one network and prunes nothing but the point turns; the\n"
        "    forward arcs and the settle are untouched between the two runs. So whatever it does\n"
        "    differently on map B, it did because of the curb."
    )
    # how wide is the window that separates the two maps? the sweep already answered it
    window = sorted(t for t in a["sweep"] if a["sweep"][t] and not b["sweep"].get(t, True))
    if window:
        print(f"\n    Tolerances that separate the two maps -- a route on A, none on B: "
              f"{window[0]:.4f} to {window[-1]:.4f} rad\n"
              f"    ({math.degrees(window[0]):.1f} to {math.degrees(window[-1]):.1f} deg) of the "
              f"{len(TAU_SWEEP)} tried. TAU_PIVOT_PITCH is {TAU_PIVOT_PITCH:.4f}, calibrated on the\n"
              f"    checkpoint's own held-out maps and never on either of these.")
    # the honest caveat, from this run's own numbers rather than a claim
    if "clear" in a["split"] and "touch" in b["split"]:
        pct = lambda v: f"{v:.0%}"
        # whether that already costs map A its route is this run's to say, not a fixed claim:
        # it has been both, and it moves with every retrain
        a_gated, a_on = a["gated"]["result"], a["on"]["result"]
        toll = (
            f"It has not cost map A its route here -- its gated plan is the ungated one, "
            f"{a_on['path_m']:.2f} m --\n    but it is why that can come out longer, and it has."
            if a_gated["reachable"] and abs(a_gated["path_m"] - a_on["path_m"]) < 1e-6 else
            f"That, not a curb, is why map A's gated route is "
            + ("NO ROUTE" if not a_gated["reachable"]
               else f"{a_gated['path_m']:.2f} m against the ungated {a_on['path_m']:.2f}")
            + "\n    rather than identical."
        )
        print(
            f"\n    Read map A's row with the split above in mind. {pct(a['split']['clear']['over'])} of the point turns on its\n"
            f"    bare floor are already over tau, because the head reads the 1 m WALLS too: a turn\n"
            f"    with its rim close to one really does diverge, it is just not the feature this map\n"
            f"    is about. {toll} The curb still moves the population it is meant to -- on map B\n"
            f"    the turns whose tyre reaches it sit at {b['split']['touch']['median']:.3f} rad against "
            f"{b['split']['clear']['median']:.3f} for the ones that do not,\n"
            f"    {pct(b['split']['touch']['over'])} of them over tau against {pct(b['split']['clear']['over'])}."
        )


if __name__ == "__main__":
    main()
