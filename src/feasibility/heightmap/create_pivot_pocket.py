"""A map whose only route out needs an IN-PLACE TURN, with a curb the planner is blind to.

`CostToGo`'s forward arcs are capped at `RobotParams.min_turn_radius` (0.5 m), so a forward-only
lattice can only reverse its heading by looping -- and the loop needs room for the whole robot,
not just the front-axle reference point it plans for: the body reaches `rear_offset + wheel_radius`
= 1.10 m behind that point, so a U-turn sweeps an annulus ~3.2 m across. An in-place point turn
(`make_cost_to_go(pivot_cost > 0)`) needs only the 1.10 m rim reach, ~2.2 m. A corridor BETWEEN
those two widths is therefore a route the pivot lattice can solve and the forward-only one cannot,
which is what this map is: a dead-end pocket the robot starts inside, facing the closed end, with
the goal outside behind it.

    y
    ^   +-----------------------------------+        tall walls (WALL_H, blocked by the settle)
    |   |             |  |  |  |  |         |        low ribs (CURB_H, INVISIBLE to the settle)
    |   |             |  |  |  |  |         |
  --+---|        >>> start     |  |     goal|x       start faces the closed end (+x)
    |   |             |  |  |  |  |         |
    |   |             |  |  |  |  |         |
    |   +-----------------------------------+
        ^ mouth (open to -x)  ^ turn zone   ^ back wall

The two heights do different jobs, and the split is the point of the map:

  * The POCKET walls are WALL_H = 0.70 m. At that height the settle blocks poses well before the
    robot reaches them (a wheel lifted that far breaks `max_roll` / `max_pitch_down`), so the
    planner genuinely respects the pocket and has to turn around inside it.
  * The RIBS are CURB_H = 0.15 m, and the settle is blind to them at EVERY heading: a wheel lifted
    0.15 m tilts the body ~11.5 deg, inside the 15 deg envelope, so `blocked` stays 0 and the only
    trace left is a small `graded_tilt` penalty. They run ACROSS the pocket at CURB_PITCH, inside
    the rear wheel's pivot sweep (it orbits the front-axle midpoint at `rear_offset` = 0.75 m), so
    a point turn anywhere in the zone drags a wheel through one -- sideways, into a vertical face,
    which is not something the kinematic twin can represent at all.

"Anywhere" is a claim about geometry, and `pivot_relief_coverage` asserts it on the rasterised
heightmap rather than arguing it from the constants. Arguing it is what went wrong the first time:
the ribs were two rails ALONG the pocket at +-0.85 m, which the write-up justified from the start
pose alone. A pivot centre 0.20 m off the centreline pulls the rear wheel's orbit to |y| = 0.55,
inside the rails' 0.65 m inner edge, and the planner found exactly that -- turning with all three
wheels at z = 0.000 while the report showed a correctly-low predicted error. Rails cannot be fixed
by narrowing: to be unavoidable one has to sit inside `rear_offset - max_pivot_offset(width)` of
the centreline, to be drivable outside `half_track`, and across the (2.2, 3.2) m band of widths
this map lives in the first bound is always the tighter one. Ribs across the pocket have no
sideways escape at all, which is why they replaced them.

So the planner is forced to do the one thing it models worst. `demos/view_planned_path.py` draws
the plan, re-checks the premise on the path that was actually planned -- the check no generator
can make, since the planner turns where it likes -- and reports what a v_wz network predicts for
each primitive; part B is gating those pivots with it.

Usage:
    python src/feasibility/heightmap/create_pivot_pocket.py
    python src/feasibility/heightmap/create_pivot_pocket.py --width 3.0 --curb-height 0.2
    python src/feasibility/heightmap/create_pivot_pocket.py --out-dir assets/pivot_pocket
    python src/feasibility/heightmap/create_pivot_pocket.py --series   # curbs 0.1 .. 1.0 m

`--series` writes `pivot_pocket_h010cm` ... `pivot_pocket_h100cm`, everything else fixed. Walls stay
at WALL_H (0.70 m), so from 0.7 m up the ribs are as tall as / taller than the walls and the
settle blocks them too -- the "invisible curb" premise above only holds at the low end.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import yaml
from helhest.engine import RobotParams

from feasibility.heightmap.create_curbs_and_walls import rects_layer
from feasibility.heightmap.heightmap_reader import HeightMapReader

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = REPO_ROOT / "assets" / "pivot_pocket"
DEFAULT_NAME = "pivot_pocket"

DEFAULT_EXTENT = 12.0  # m, square map
DEFAULT_CELL = 0.1  # m
INCLINE_DEG = 80.0  # side slope of every feature, as in create_curbs_and_walls

WALL_H = 0.70  # m -- tall enough that the settle blocks poses beside it (measured: see module doc)
WALL_T = 0.30  # m wall thickness
CURB_H = 0.15  # m -- low enough that the settle blocks NOTHING at any heading
CURB_T = 0.20  # m rib thickness, along x
SERIES_HEIGHTS = [round(0.1 * i, 1) for i in range(1, 11)]  # m, 0.1 .. 1.0 for --series

DEFAULT_WIDTH = 2.8  # m clear, inner wall face to inner wall face

# The curbs are RIBS running across the pocket, spanning its full width, so a pivot cannot be
# placed to the side of them (see `max_pivot_offset`). `CURB_PITCH` is what makes them
# unavoidable along x too: over any 90 deg of a turn the rear wheel's x sweeps at least
# `rear_offset` (0.75 m) of its orbit, so a pitch below that guarantees a crossing wherever the
# turn is placed.
#
# They cover the TURN ZONE (`CURB_X0` to the closed end), not the whole pocket. The robot starts
# facing the closed end and the lattice has no reverse, so it cannot translate towards the mouth
# before it has turned -- every plan's point turns land within a metre of START, which is why the
# zone only has to hold them. Ribbing the whole pocket also works and is more obviously airtight,
# but it puts the exit run over seven more ribs, and driving head-on into a step turns out to be
# the larger divergence of the two (e_rot up to 0.98 against a pivot's 0.19): the point turns the
# map exists to expose end up the quietest thing on the path.
CURB_PITCH = 0.65  # m between rib centrelines; gap = CURB_PITCH - CURB_T = 0.45 m, wide enough
# that a 0.70 m wheel drops ~0.08 m into it rather than bridging rib to rib. Also the coarsest
# pitch that clears the START footprint: `_rib_phase` can win (0.75 mod pitch, pitch) / 2 of
# clearance, and 0.55 m tops out at 0.175 m -- under the rasterised rib's own 0.20 m half-width.

CURB_X0 = 1.3  # m, mouth-side end of the ribbed turn zone (START's rear wheel sits at 1.75)

POCKET_X0 = -1.0  # m, the open mouth
POCKET_X1 = 4.0  # m, inner face of the back wall
START = (2.5, 0.0, 0.0)  # inside the pocket, facing the closed end
GOAL = (-4.0, 0.0)  # outside it, behind the robot

_ROBOT = RobotParams()
RIM_REACH = _ROBOT.rear_offset + _ROBOT.wheel_radius  # 1.10 m, the pivot's own swept radius


def max_pivot_offset(width: float) -> float:
    """[m] how far off the pocket centreline a pivot centre can sit -- the lateral freedom the
    planner has when choosing WHERE to turn.

    NOT `width/2 - RIM_REACH`. That would be the bound if a pivot had to fit its whole swept
    annulus between the walls, but the lattice reverses a heading in 15 deg bins and needs only
    ~180 deg of them, so the rear wheel visits ONE side of its orbit: a turn placed off-centre
    simply sweeps toward the far wall, and gains clearance rather than losing it. What actually
    binds is the widest part of the robot that is present at EVERY heading -- a front wheel at
    `half_track`, dilated by the settle's own `wheel_radius` sphere (engine/envelope.py).

    This is why the curbs run ACROSS the pocket and not along it. A longitudinal rail has to be
    inside `rear_offset - max_pivot_offset(width)` of the centreline to be unavoidable, and
    outside `half_track` to leave the robot a lane to drive in on -- and for every width in the
    (2.2, 3.2) m band this map lives in, the first bound is the tighter one, so no rail satisfies
    both. The original map picked the drivable side of that trade and was therefore avoidable:
    at 2.8 m the planner could place a pivot 0.69 m off-centre, and it used 0.20 m, threading the
    rear wheel through the gap between the rails to turn on ground as flat as open floor.
    """
    return width / 2.0 - (_ROBOT.half_track + _ROBOT.wheel_radius)


def _rib_phase() -> float:
    """[m] the ribs' offset (mod CURB_PITCH), chosen to put every START wheel as far from a rib
    as the pitch allows.

    The robot has to SPAWN somewhere, and ostrich drops it onto the terrain before the run, so
    `write_map` asserts all three wheel centres start on flat ground. That constrains the phase:
    the rear wheel is `rear_offset` = 0.75 m behind the fronts, which is not a whole number of
    pitches, so the two are at different phases and only some offsets clear both. Searching for
    the offset that maximises the smallest clearance keeps this true if the pitch is retuned,
    instead of pinning a magic number that silently stops working.

    It does NOT weaken the premise: this places the wheels between ribs at ONE heading, and
    `pivot_relief_coverage` requires every centre to meet a rib at SOME heading.
    """
    wheel_x = np.array([START[0], START[0], START[0] - _ROBOT.rear_offset])
    candidates = np.arange(0.0, CURB_PITCH, 0.005)
    gap = np.abs(
        (wheel_x[None, :] - candidates[:, None] + CURB_PITCH / 2.0) % CURB_PITCH - CURB_PITCH / 2.0
    )
    return float(candidates[int(np.argmax(gap.min(axis=1)))])


def rib_centres() -> np.ndarray:
    """[k] world x of each rib centreline, across the turn zone at CURB_PITCH."""
    k0 = int(np.ceil((CURB_X0 - _rib_phase()) / CURB_PITCH))
    k1 = int(np.floor((POCKET_X1 - _rib_phase()) / CURB_PITCH))
    return _rib_phase() + np.arange(k0, k1 + 1) * CURB_PITCH


def pivot_relief_coverage(
    terrain: HeightMapReader, width: float, threshold: float = 0.05, n_theta: int = 24
) -> tuple[float, tuple[float, float]]:
    """Is a point turn anywhere in the TURN ZONE forced onto relief? Returns the fraction of
    candidate pivot centres for which SOME heading bin puts a wheel on relief above `threshold`,
    and the worst centre found.

    This is the map's own premise, checked on the rasterised heightmap rather than argued from
    the constants -- the argument is what was wrong before. Candidate centres are every 0.1 m of
    the ribbed zone the robot's own footprint fits in; headings are the lattice's own `n_theta`
    bins, because those are the only poses `CostToGo` can route through.

    Scoped to the zone, not the pocket, for the reason CURB_X0 exists: the flat mouth is not
    reachable facing +x, so a pivot centre there is not a plan the planner can actually build.
    That makes this a weaker check than the rib-everywhere version -- it assumes the turn starts
    near START rather than proving it -- so `demos/view_planned_path.py` re-checks the premise on
    the path that was really planned, which is the claim that matters.
    """
    lo = _ROBOT.half_track + _ROBOT.wheel_radius
    xs = np.arange(CURB_X0 + lo, POCKET_X1 - lo, 0.1)
    ys = np.arange(-(width / 2.0 - lo), width / 2.0 - lo + 1e-9, 0.1)
    yaws = (np.arange(n_theta) + 0.5) * 2.0 * np.pi / n_theta
    covered, worst, worst_xy = 0, np.inf, (float("nan"), float("nan"))
    for x in xs:
        for y in ys:
            best = max(
                float(terrain.sample(wx, wy).max())
                for yaw in yaws
                for wx, wy in [wheel_xy((x, y, yaw)).T]
            )
            covered += best > threshold
            if best < worst:
                worst, worst_xy = best, (float(x), float(y))
    return covered / (len(xs) * len(ys)), worst_xy


def build_pivot_pocket(
    width: float = DEFAULT_WIDTH,
    curb_height: float = CURB_H,
    wall_height: float = WALL_H,
    extent: float = DEFAULT_EXTENT,
    cell: float = DEFAULT_CELL,
) -> tuple[HeightMapReader, dict]:
    """The pocket as a (reader, params) pair -- a pure builder, all IO in main().

    Walls and curbs are separate `rects_layer` calls because they sit at DIFFERENT heights, and are
    combined by an elementwise maximum (the union-of-solids convention the other generators use;
    `build_multi_box`'s sum is only correct for one shared height).
    """
    half_w = width / 2.0  # inner faces at +-half_w
    wall_cy = half_w + WALL_T / 2.0
    walls = [
        # the two sides, run past POCKET_X1 so they meet the back wall with no seam
        (POCKET_X1 + WALL_T - POCKET_X0, WALL_T, (POCKET_X0 + POCKET_X1 + WALL_T) / 2.0, wall_cy, 0.0),
        (POCKET_X1 + WALL_T - POCKET_X0, WALL_T, (POCKET_X0 + POCKET_X1 + WALL_T) / 2.0, -wall_cy, 0.0),
        # the closed end, spanning the full outer width
        (WALL_T, 2.0 * wall_cy + WALL_T, POCKET_X1 + WALL_T / 2.0, 0.0, 0.0),
    ]
    # ribs span the full clear width, so a pivot has no curb-free cell to slide sideways to
    curbs = [(CURB_T, width, cx, 0.0, 0.0) for cx in rib_centres()]
    H = np.maximum(
        rects_layer(walls, wall_height, INCLINE_DEG, extent, cell),
        rects_layer(curbs, curb_height, INCLINE_DEG, extent, cell),
    )
    terrain = HeightMapReader(H, origin=(-extent / 2.0, -extent / 2.0), cell=cell)
    params = {
        "kind": "pivot_pocket",
        "extent": float(extent),
        "cell": float(cell),
        "width": float(width),
        "wall_height": float(wall_height),
        "wall_thickness": float(WALL_T),
        "curb_height": float(curb_height),
        "curb_thickness": float(CURB_T),
        "curb_pitch": float(CURB_PITCH),
        "curb_centres_x": [float(v) for v in rib_centres()],
        "incline_deg": float(INCLINE_DEG),
        "pocket_x": [float(POCKET_X0), float(POCKET_X1)],
        "start": [float(v) for v in START],
        "goal": [float(v) for v in GOAL],
    }
    return terrain, params


def wheel_xy(pose: tuple[float, float, float]) -> np.ndarray:
    """[3, 2] world (x, y) of the three wheel centres at body pose (x, y, yaw): the two front
    wheels at +-half_track and the rear one `rear_offset` behind, per `RobotParams.build`."""
    x, y, yaw = pose
    b, l = _ROBOT.half_track, _ROBOT.rear_offset
    local = np.array([[0.0, b], [0.0, -b], [-l, 0.0]])
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([x + c * local[:, 0] - s * local[:, 1], y + s * local[:, 0] + c * local[:, 1]], -1)


def series_name(curb_height: float) -> str:
    """Stem for one map of the series, e.g. `pivot_pocket_h015cm` (cm, as `large_box_i*_h070cm`)."""
    return f"{DEFAULT_NAME}_h{round(curb_height * 100):03d}cm"


def write_map(args: argparse.Namespace, curb_height: float, stem: str) -> None:
    """Build, assert the design and write one pocket with the given curb height."""
    terrain, params = build_pivot_pocket(
        args.width, curb_height, args.wall_height, args.extent, args.cell
    )

    # --- the design, asserted (the map is only useful if all of these hold) ---------------------
    half_w = args.width / 2.0
    assert args.width < 2.0 * (RIM_REACH + 0.5), (
        f"pocket {args.width:.2f} m wide leaves room for a min_turn_radius U-turn "
        f"({2.0 * (RIM_REACH + 0.5):.2f} m) -- the forward-only lattice would not need a pivot"
    )
    assert args.width > 2.0 * RIM_REACH, (
        f"pocket {args.width:.2f} m wide does not fit the pivot's own {2.0 * RIM_REACH:.2f} m sweep"
    )
    assert CURB_PITCH < _ROBOT.rear_offset, (
        f"rib pitch {CURB_PITCH} m is wider than the {_ROBOT.rear_offset} m of its orbit the rear "
        "wheel's x sweeps in one turn -- a pivot could be placed between two ribs"
    )
    coverage, worst = pivot_relief_coverage(terrain, args.width)
    assert coverage == 1.0, (
        f"only {coverage:.0%} of the pocket's pivot centres are forced onto relief; a turn at "
        f"({worst[0]:.2f}, {worst[1]:.2f}) keeps every wheel on flat ground at all 24 headings -- "
        "the planner would find it, and the map's premise would be false on the planned path"
    )
    for name, pose in (("start", START), ("goal", (*GOAL, 0.0))):
        assert abs(terrain.sample(pose[0], pose[1])) < 1e-9, f"{name} is not on flat ground"
    for i, (wx, wy) in enumerate(wheel_xy(START)):
        assert abs(terrain.sample(wx, wy)) < 1e-9, f"start wheel {i} is not on flat ground"
    # the design intent, at the start specifically: turning there rides the rear wheel onto a rib
    start_rear_z = [
        float(terrain.sample(*wheel_xy((START[0], START[1], yaw))[2]))
        for yaw in np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    ]
    assert max(start_rear_z) > 0.9 * curb_height, (
        f"turning at START never puts the rear wheel above z={max(start_rear_z):.3f} -- the ribs "
        f"do not reach its {_ROBOT.rear_offset} m orbit"
    )
    assert float(terrain.sample(*GOAL)) == 0.0 and abs(GOAL[0]) < args.extent / 2.0
    assert terrain.max_z == max(args.wall_height, curb_height), terrain.max_z

    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / stem
    terrain.save(path)
    meta = yaml.safe_load(path.with_suffix(".yaml").read_text())
    meta.update(source="feasibility.heightmap.create_pivot_pocket", **params)
    path.with_suffix(".yaml").write_text(yaml.safe_dump(meta, sort_keys=False))

    print(
        f"wrote {path}.png + .yaml  ({terrain.nx}x{terrain.ny} cells at {args.cell} m, "
        f"z in [{terrain.min_z:.2f}, {terrain.max_z:.2f}] m)\n"
        f"  pocket  x in [{POCKET_X0}, {POCKET_X1}], {args.width:.2f} m clear "
        f"(pivot needs {2 * RIM_REACH:.2f}, a U-turn {2 * (RIM_REACH + 0.5):.2f})\n"
        f"  walls   {args.wall_height:.2f} m -- the settle blocks these\n"
        f"  ribs    {curb_height:.2f} m, {len(rib_centres())} across the full width at x = "
        f"{', '.join(f'{v:.2f}' for v in rib_centres())}"
        f"{'' if curb_height > 0.15 else ' -- the settle does NOT block these'}\n"
        f"  premise every pivot centre in the pocket meets relief at some heading\n"
        f"  start   {START}  ->  goal {GOAL}\n"
        f"  view it: python demos/view_planned_path.py --map {path}.png --pivot-cost 0.15"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--name", default=DEFAULT_NAME, help="map stem written into --out-dir")
    ap.add_argument("--width", type=float, default=DEFAULT_WIDTH, help="clear pocket width [m]")
    ap.add_argument("--curb-height", type=float, default=CURB_H)
    ap.add_argument("--wall-height", type=float, default=WALL_H)
    ap.add_argument("--extent", type=float, default=DEFAULT_EXTENT)
    ap.add_argument("--cell", type=float, default=DEFAULT_CELL)
    ap.add_argument(
        "--series",
        action="store_true",
        help=f"write one map per --series-heights curb height as {series_name(0.1)} ...; "
        "ignores --curb-height and --name",
    )
    ap.add_argument(
        "--series-heights",
        type=float,
        nargs="+",
        default=SERIES_HEIGHTS,
        help="curb heights [m] for --series (default 0.1 to 1.0 in 0.1 steps)",
    )
    args = ap.parse_args()

    if args.series:
        for h in args.series_heights:
            write_map(args, h, series_name(h))
    else:
        write_map(args, args.curb_height, args.name)


if __name__ == "__main__":
    main()
