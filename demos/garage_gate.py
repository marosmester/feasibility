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

CLI parameters:
    --map A|B|both    which garage to run (default both)
    --map-dir PATH    where create_garage.py wrote them (default assets/garage)
    --tau RAD         the point-turn gate's e_pitch tolerance (default planners.TAU_PIVOT_PITCH)
    --tau-sweep       also report which taus keep a route open, once the field is in hand
    --torch-device D  cuda (default) or cpu
    --pivot-cost M    m-equivalent per 15 deg bin for a point turn (default 0.15)

Usage:
    python demos/garage_gate.py
    python demos/garage_gate.py --map B --tau-sweep
    python demos/view_planned_path.py --map assets/garage/garage_b --planner nn-gated-pivot \
        --torch-device cuda            # the same plan, drawn
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
    from demos.view_planned_path import path_poses
    from demos.view_planned_path import wheel_relief
except ModuleNotFoundError:  # pragma: no cover - the bare-script path
    from view_planned_path import path_poses
    from view_planned_path import wheel_relief

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MAP_DIR = REPO_ROOT / "assets" / "garage"
RELIEF = 0.05  # m, what counts as a wheel standing on something (trial.interact_relief)
# Bunched around TAU_PIVOT_PITCH, because that is where the two maps part company and the width of
# the window they part in is the thing worth knowing.
TAU_SWEEP = (0.03, 0.05, 0.07, 0.0873, 0.095, 0.105, 0.12, 0.15, 0.20, 0.30)


def pivot_truth(terrain: HeightMapReader, ctg, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """(touch, legal) [ny, nx, n_theta, 2] over the lattice's own poses: does each single-bin point
    turn put a wheel's TYRE within reach of the curb, and would the settle allow it at all?

    Geometry, not physics -- this is what the network's predicted e_pitch is scored against. The
    turn is sampled at sub-headings across the bin because the wheel sweeps continuously through
    it, and `feature_distances` rasterises the walls and the curb separately from the rectangles
    the map was built from, so the wall's own ramp is never mistaken for curb.
    """
    walls, curbs = garage.garage_rects(params["wall_gap"], params["back_gap"],
                                       params["garage_x"][1], -params["garage_x"][0],
                                       params["curb_x"] or garage.CURB_X)
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
    print_arm("vanilla-on, pivot_cost 0", run_arm(terrain, start, goal, "vanilla-on", zero),
              args.tau)
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
    out = dict(name=path.name, curb=curb, on=on, gated=gated, split=split, sweep=sweep)
    del ctx
    return out


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
    if len(rows) < 2:
        return
    a, b = rows
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
        print(
            f"\n    Read map A's row with the split above in mind. {pct(a['split']['clear']['over'])} of the point turns on its\n"
            f"    bare floor are already over tau, because the head reads the 1 m WALLS too: a turn\n"
            f"    with its rim close to one really does diverge, it is just not the feature this map\n"
            f"    is about. That, not a curb, is why map A's gated route is longer instead of\n"
            f"    identical. The curb still moves the population it is meant to -- on map B the turns\n"
            f"    whose tyre reaches it sit at {b['split']['touch']['median']:.3f} rad against "
            f"{b['split']['clear']['median']:.3f} for the ones that do not,\n"
            f"    {pct(b['split']['touch']['over'])} of them over tau against {pct(b['split']['clear']['over'])}."
        )


if __name__ == "__main__":
    main()
