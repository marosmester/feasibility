"""A U-shaped garage whose only way out is a 90 deg POINT TURN -- with and without a curb inside.

Two maps, identical but for one feature, built for `demos/garage_gate.py`:

    garage_a   three walls, empty floor
    garage_b   the same, plus ONE curb on the floor: a rib across it parallel to the garage
               door, with a SPUR off the rib reaching back towards the robot

The robot is parked deep inside, facing a SIDE wall, 90 deg from the opening -- no reverse
or loop needed, just a turn on the spot and out. In map B that turn drags the REAR wheel
sideways into a 0.20 m curb: the body tilts 15 deg (which the settle tolerates) while
ostrich's wheel climbs a vertical face the kinematic twin has no term for. That gap is what
`planning.planners.TAU_PIVOT_PITCH` is calibrated to catch.

    y
    ^   +-------------------------+      tall walls (WALL_H, the settle blocks these)
    |   |                     | | |      low curb (CURB_H, INVISIBLE to the settle)
  --+---           ^ start    | | |      start faces the +y side wall
    |   |          R      +---+ | |
    |   |                 | spur| |      the spur reaches back on the side the rear wheel
    |   +-----------------+---+---+      swings through -- see WHERE THE CURB GOES
        ^ door (open to -x)     ^ curb   ^ back wall

WHY 90 DEG NEEDS AN ASYMMETRIC GARAGE (measured off `RobotParams` by `clearances()`, printed
by `main()`). Over a 90 deg point turn the wheel RIMS stay inside x in [-0.71, +1.09], y in
[-1.09, +0.71] of the reference point; the tightest FORWARD 90 deg turn instead throws the
outer front wheel 1.13 m sideways. So the faced wall must sit in the window
0.71 m < WALL_GAP < 1.13 m -- close enough that no forward arc swings the nose round, far
enough a point turn still fits. That window sits ABOVE the 1.09 m the turn needs BEHIND the
robot, so a centred robot has no valid width at all; it must be parked close to the wall it
faces, which is what a garage looks like anyway. `WALL_GAP` 0.95 m / `BACK_GAP` 1.35 m give
2.30 m clear width with ~0.2 m margin on both bounds, both asserted.

WHERE THE CURB GOES is the delicate part -- `create_pivot_pocket.py` is the cautionary tale
of a feature argued from the start pose alone turning out to be avoidable. Three
constraints:

  * CLEAR OF THE START. The settle dilates the heightmap by `wheel_radius`, so a curb ramp
    foot within 0.35 m of a start wheel CENTRE blocks the spawn. At `CURB_X` 0.90 m the
    nearest start wheel sits 0.50 m off it.
  * REACHED IN PRACTICE, not just in the plan -- what `SPUR_X` is for; see THE SPUR below.
  * ONLY THE REAR WHEEL. A front wheel lifted `CURB_H` rolls past `max_roll` (the settle
    blocks it, no network needed); the rear wheel pitches only 14.9 deg, INSIDE
    `max_pitch_down`, so `blocked` stays 0 and the net is the only instrument left.

THE PREMISE IS A REACHABILITY CLAIM, and `reach_closure` makes it one: a breadth-first walk
over the lattice states `CostToGo` plans on, run three ways --

    forward arcs only                 must NOT get out -- the garage needs a point turn
    point turns free                  must get out     -- there is a route to find
    only CURB-FREE point turns        must NOT get out -- pruning them is what closes it

The third caught the first draft: point turns near the door were curb-free, and with the
mouth 1.5 m away the robot could shuffle there on short arcs. `WALL_GAP` is what shuts that
(the shuffle works at 1.00 m of curb offset, not at 0.90 m) -- a finding no amount of
constant-arguing would have produced. `demos/garage_gate.py` measures the same three arms on
the real planner and network.

THE SPUR: a rib alone is reached by the PLANNED turn (rear rim 0.05 m over it) but
`demos/ostrich_follow_plan_turning.py` shows the wheel missing it in practice, because an
in-place skid-steer turn DRIFTS -- the reference point slides ~0.22 m back toward the door,
so the rear wheel ends its 45 deg pivot 0.24 m short of where the plan put it. A curb sized
against the plan measures nothing in physics. The rib can't simply move closer (the start's
front wheel pins any full-width rib at x >= 0.71), but the wheel that pins it and the wheel
that must reach it sit at DIFFERENT y -- so `SPUR_X`/`SPUR_Y` extend the rib 0.40 m further
back over just the far half of the floor, clear of every spawn wheel (asserted at both the
start yaw and the lattice's bin centre).

How far the spur may reach is a WINDOW: the settle is blind to a TYRE over the curb but NOT
to a wheel CENTRE on it (a centre on it lifts the wheel the full height and blocks the pose,
which would let map B plan around it and break the controlled A/B pair). The spur is placed
to leave the centre just outside and the tyre well in (0.10 m / 0.25 m here); `write_map`
asserts both edges plus what tyre survives the drift -- a two-cell-wide target on a 0.1 m
grid, which is why every number here is measured rather than assumed.

Usage:
    python src/feasibility/heightmap/create_garage.py
    python src/feasibility/heightmap/create_garage.py --curb-height 0.15
    python src/feasibility/heightmap/create_garage.py --spur-x 0.50 --spur-y -0.55
    # (--spur-x at or past the rib drops the spur -- the old map B, which the asserts below
    #  now refuse: it leaves 0.05 m of tyre over the curb and the drift spends all of it)
"""
from __future__ import annotations

import argparse
import math
import pathlib

import numpy as np
import yaml
from helhest.engine import RobotParams
from scipy.ndimage import distance_transform_edt

from feasibility.heightmap.create_curbs_and_walls import rects_layer
from feasibility.heightmap.create_pivot_pocket import wheel_xy
from feasibility.heightmap.heightmap_reader import HeightMapReader

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = REPO_ROOT / "assets" / "garage"
NAME_A, NAME_B = "garage_a", "garage_b"

DEFAULT_EXTENT = 7.0  # m, square map -- 70 x 70 cells, and the v_wz gate's field is ny*nx*n_theta
DEFAULT_CELL = 0.1
INCLINE_DEG = 80.0  # side slope of every feature, as in create_curbs_and_walls

# The lattice the maps are designed for: helhest_stack's CostToGo at planning.gated_lattice's
# constants. Restated rather than imported, so a heightmap generator does not pull in the planner
# (create_pivot_pocket.pivot_relief_coverage takes the same 24 as a default argument).
N_THETA = 24
ARC_STEP = 0.3  # m, one forward primitive

WALL_H = 1.00  # m -- "at least 1 m"; the settle blocks poses beside it at every heading
WALL_T = 0.30  # m wall thickness
CURB_H = 0.20  # m -- the rear wheel climbs it, the settle stays blind (see module doc)
CURB_T = 0.20  # m curb thickness, along x

START = (0.0, 0.0, math.pi / 2.0)  # reference point at the origin, facing the +y side wall
GOAL = (-3.0, 0.0)  # outside the door
EXIT_BINS = 6  # heading bins of the point turn that gets the robot out: 6 * 15 deg = 90 deg
# of those bins, how many the planner really turns ON THE SPOT: after 45 deg the forward arcs
# clear the faced wall and driving is cheaper, so the plan pivots three times and arcs out (what
# `demos/view_planned_path.py` draws and `demos/ostrich_follow_plan_turning.py` reports). The curb
# has to be reached inside THOSE bins -- the last three of the 90 deg never happen on the spot.
DRIVEN_BINS = 3
# [m] where the turn ostrich really drives ends up relative to the plan: an in-place skid-steer
# turn slides, measured at (-0.22, -0.10) over this exit by
# `demos/ostrich_follow_plan_turning.py` -- back towards the door, away from the curb. Applied
# as one rigid offset, which over-states the early bins but matches the late (curb-reaching) ones
# -- the conservative reading.
PIVOT_DRIFT = (-0.22, -0.10)
CONTACT_MARGIN = 0.10  # m of rear tyre that must still be over the curb once drifted
PLANNED_OVERLAP = 0.20  # m of rear tyre over it in the PLAN -- what the drift is spent out of

WALL_GAP = 0.95  # m, START to the inner face of the wall it FACES (the 0.71 .. 1.13 window)
BACK_GAP = 1.35  # m, START to the inner face of the far side wall -> 2.30 m clear width
DEPTH_BACK = 1.35  # m, START to the inner face of the closed end
DEPTH_DOOR = 1.50  # m, START to the open mouth -> a 2.85 m deep garage
CURB_X = 0.90  # m, the curb's centreline, parallel to the door (i.e. a rib spanning y)
# [m] the SPUR: the same curb, extended back over the far half of the floor so the turn
# ostrich really drives reaches it, not just the plan (see module doc). SPUR_X is its near
# face, SPUR_Y the y it starts at -- both the middle of a window bounded BELOW by the spawn
# (wheels need `wheel_radius` clearance; kept 0.40 m) and ABOVE by the settle, which is blind
# to a rim over the curb but not to a wheel CENTRE on it (one cell further and the settle
# blocks the pivot pose, breaking the A/B pair). Both edges are a cell away, so `write_map`
# asserts each rather than arguing them.
SPUR_X = 0.45
SPUR_Y = -0.60

RELIEF = 0.05  # m, what counts as terrain relief (lattice_learning's trial.interact_relief)

_ROBOT = RobotParams()
BIN = 2.0 * math.pi / N_THETA
# [m] the lift at ONE wheel that first breaks the settle's tilt envelope -- roll for a front wheel
# (the two are 2*half_track apart), pitch for the rear one (rear_offset behind). The two come out
# within a centimetre of each other, and CURB_H sits between them: the rear wheel climbs the curb
# and the body stays inside the envelope, a front wheel would not. Used as the height above which a
# feature counts as WALL, i.e. as something the settle stops the planner on.
BLOCK_H = 2.0 * float(_ROBOT.half_track) * math.tan(float(_ROBOT.max_roll))


def bin_centre(yaw: float) -> float:
    """The heading `CostToGo` actually plans at, given a world yaw: its bin's centre. START's
    90 deg is not a bin centre (they sit at 7.5, 22.5, ...), so every clearance below is computed
    at 97.5 deg, the pose the lattice would really turn from."""
    return (math.floor((yaw % (2.0 * math.pi)) / BIN) + 0.5) * BIN


def pivot_wheels(pose: tuple[float, float, float], n_bins: int = EXIT_BINS, ccw: bool = True):
    """[n_bins + 1, 3, 2] world wheel centres over a point turn of `n_bins` lattice bins from
    `pose`, one entry per heading the turn passes through -- the poses `CostToGo` chains together
    when it takes the pivot primitive `n_bins` times."""
    x, y, yaw = pose
    step = BIN if ccw else -BIN
    yaws = bin_centre(yaw) + step * np.arange(n_bins + 1)
    return np.array([wheel_xy((x, y, float(t))) for t in yaws])


def arc_poses(pose: tuple[float, float, float], kappa: float, n_sub: int = 13) -> np.ndarray:
    """[n_sub, 3] poses along ONE forward primitive of curvature `kappa` [1/m] and ARC_STEP metres,
    starting at `pose` -- whose heading is used as given, so a caller chaining arcs keeps the exact
    continuous turn rather than re-snapping to a bin each step. Sampled rather than end-point only,
    because `_relax_gated_kernel` checks the cells an arc SWEEPS, not just where it lands."""
    x, y, yaw = pose
    s = np.linspace(0.0, ARC_STEP, n_sub)
    t = yaw + kappa * s
    if abs(kappa) < 1e-9:
        px, py = x + s * math.cos(yaw), y + s * math.sin(yaw)
    else:  # centre of curvature a radius to the left (kappa > 0) or right of the start pose
        r = 1.0 / kappa
        cx, cy = x - r * math.sin(yaw), y + r * math.cos(yaw)
        px, py = cx + r * np.sin(t), cy - r * np.cos(t)
    return np.stack([px, py, t], axis=-1)


def wheels_of(poses: np.ndarray) -> np.ndarray:
    """[..., 3, 2] wheel centres of a stack of (x, y, yaw) poses -- `wheel_xy` vectorised, asserted
    against it below, so the bulk checks here and the per-pose ones agree by construction."""
    b, l = float(_ROBOT.half_track), float(_ROBOT.rear_offset)
    local = np.array([[0.0, b], [0.0, -b], [-l, 0.0]])
    c, s = np.cos(poses[..., 2])[..., None], np.sin(poses[..., 2])[..., None]
    return np.stack(
        [poses[..., 0, None] + c * local[:, 0] - s * local[:, 1],
         poses[..., 1, None] + s * local[:, 0] + c * local[:, 1]],
        axis=-1,
    )


def arc_wheels(pose: tuple[float, float, float], n_arcs: int, radius: float, ccw: bool = True):
    """[n_arcs + 1, 3, 2] world wheel centres at the end of each of `n_arcs` forward primitives at
    `radius`. Entry 0 is the starting pose, so entry k is where the robot stands after k arcs."""
    kappa = (1.0 if ccw else -1.0) / radius
    p = np.array([pose[0], pose[1], bin_centre(pose[2])], dtype=float)
    out = [p]
    for _ in range(n_arcs):
        p = arc_poses(tuple(p), kappa)[-1]
        out.append(p)
    return wheels_of(np.array(out))


def reach_closure(
    terrain: HeightMapReader,
    dist_wall: np.ndarray,
    dist_curb: np.ndarray,
    bounds: tuple[float, float, float, float],
    stop_x: float,
    allow_pivots: bool,
    gate_curb: bool = False,
    start: tuple[float, float, float] = START,
    n_kappa: int = 21,
    max_states: int = 60000,
) -> dict:
    """Can the robot get OUT -- past `stop_x`? A breadth-first walk over the same (row, col,
    heading bin) states `CostToGo` plans on, from `start`.

    Three arms of the map's design, predicted here from geometry alone before any network is
    loaded (`demos/garage_gate.py` then measures the same three on the real planner):

        allow_pivots=False                  must NOT escape -- the garage needs a point turn
        allow_pivots=True                   must escape     -- there is a route when pivots are free
        allow_pivots=True, gate_curb=True   must NOT escape -- pruning curb-touching point turns
                                                               takes that route away

    A forward arc is drivable when no wheel centre comes within `wheel_radius` of a wall
    anywhere ALONG it (the settle's spherical envelope). Curvature is swept continuously
    rather than at the lattice's five `primitive_kappas`, so this closure is a SUPERSET of
    what the planner can really drive -- whatever it can't reach here, the planner certainly
    can't. A point turn is one heading bin, checked at sub-headings since the wheel sweeps
    continuously through it; with `gate_curb` it must also stay clear of the curb.
    """
    r, cell = float(_ROBOT.wheel_radius), terrain.cell
    x0, x1, y0, y1 = bounds
    kappas = np.linspace(-1.0, 1.0, n_kappa) / float(_ROBOT.min_turn_radius)
    state = lambda p: (int(round((p[1] - terrain.y0) / cell)), int(round((p[0] - terrain.x0) / cell)),
                       int(math.floor((p[2] % (2.0 * math.pi)) / BIN)) % N_THETA)
    pose = lambda s: (terrain.x0 + s[1] * cell, terrain.y0 + s[0] * cell, (s[2] + 0.5) * BIN)
    sub = np.linspace(0.0, 1.0, 5)[1:]  # sub-headings across one pivot bin, endpoint included

    s0 = state((start[0], start[1], bin_centre(start[2])))
    seen, frontier, depth, n_pivots = {s0}, [s0], 0, 0
    while frontier and len(seen) < max_states:
        nxt = []
        for s in frontier:
            p = pose(s)
            for k in kappas:
                swept = arc_poses(p, float(k))
                if sample_dist(dist_wall, terrain, wheels_of(swept)).min() < r:
                    continue
                e = state(swept[-1])
                if not (x0 <= swept[-1][0] <= x1 and y0 <= swept[-1][1] <= y1) or e in seen:
                    continue
                if swept[-1][0] < stop_x:
                    return dict(escaped=True, n_states=len(seen), depth=depth, n_pivots=n_pivots,
                                states=seen)
                seen.add(e)
                nxt.append(e)
            if not allow_pivots:
                continue
            for d in (-1, 1):
                turn = np.stack([np.full_like(sub, p[0]), np.full_like(sub, p[1]),
                                 p[2] + d * BIN * sub], axis=-1)
                w = wheels_of(turn)
                if sample_dist(dist_wall, terrain, w).min() < r:
                    continue
                if gate_curb and sample_dist(dist_curb, terrain, w).min() < r:
                    continue  # the gate prunes this one, so the closure may not use it
                e = (s[0], s[1], (s[2] + d) % N_THETA)
                if e not in seen:
                    seen.add(e)
                    nxt.append(e)
                    n_pivots += 1
        frontier, depth = nxt, depth + 1
    xs = np.array([pose(s)[0] for s in seen])
    return dict(escaped=False, n_states=len(seen), depth=depth, n_pivots=n_pivots, states=seen,
                x_min=float(xs.min()), x_max=float(xs.max()))


def clearances() -> dict:
    """Every bound the garage's dimensions are chosen from, measured off `RobotParams` rather than
    written down: how much room a 90 deg point turn needs in each direction, and how much room the
    same 90 deg done on FORWARD arcs would need to the side. `main()` prints the lot."""
    r = _ROBOT.wheel_radius
    out = {}
    for tag, ccw in (("ccw", True), ("cw", False)):
        w = pivot_wheels((0.0, 0.0, START[2]), EXIT_BINS, ccw)
        out[f"pivot_{tag}"] = dict(
            ahead=float(w[..., 1].max()) + r, behind=-float(w[..., 1].min()) + r,
            fore=float(w[..., 0].max()) + r, aft=-float(w[..., 0].min()) + r,
        )
    # the forward alternative, per primitive: how far sideways the outer front wheel's rim reaches
    # after k arcs of the tightest turn the lattice has. The FIRST k over WALL_GAP is where the
    # planner runs out of forward road, and 90 deg needs ceil(90 / (ARC_STEP / min_turn_radius)).
    n_need = math.ceil(math.radians(90.0) / (ARC_STEP / float(_ROBOT.min_turn_radius)))
    for tag, ccw in (("ccw", True), ("cw", False)):
        w = arc_wheels((0.0, 0.0, START[2]), n_need, float(_ROBOT.min_turn_radius), ccw)
        out[f"arc_{tag}"] = dict(
            n_for_90deg=n_need,
            ahead=[float(v) + r for v in w[..., 1].max(axis=1)],  # per arc index, 0 = at START
            travel=[ARC_STEP * k for k in range(n_need + 1)],
        )
    return out


def garage_rects(
    wall_gap: float = WALL_GAP,
    back_gap: float = BACK_GAP,
    depth_back: float = DEPTH_BACK,
    depth_door: float = DEPTH_DOOR,
    curb_x: float = CURB_X,
    spur_x: float = SPUR_X,
    spur_y: float = SPUR_Y,
) -> tuple[list, list]:
    """(wall rectangles, curb rectangles) as `rects_layer` takes them -- (w, d, cx, cy, yaw), w
    along x. Split out so `feature_distances` can rasterise each feature ALONE: telling the two
    apart by a height threshold would file the walls' own 80 deg ramp, which passes through the
    curb's height on its way up, as curb.

    The curb is the rib plus its spur, two rectangles at the SAME height: `rects_layer` merges
    them by an elementwise maximum, so they fuse into one solid with no seam and the spur's near
    face is the only new edge. `spur_x` at or beyond the rib's own near face means no spur."""
    back_x, door_x = depth_back, -depth_door
    width = wall_gap + back_gap  # clear, inner face to inner face
    mid_y = (wall_gap - back_gap) / 2.0  # the garage's own centreline, which START is NOT on
    walls = [
        # the two sides, run past the closed end so they meet the back wall with no seam
        (back_x + WALL_T - door_x, WALL_T, (door_x + back_x + WALL_T) / 2.0, wall_gap + WALL_T / 2.0, 0.0),
        (back_x + WALL_T - door_x, WALL_T, (door_x + back_x + WALL_T) / 2.0, -(back_gap + WALL_T / 2.0), 0.0),
        # the closed end, spanning the full outer width
        (WALL_T, width + 2.0 * WALL_T, back_x + WALL_T / 2.0, mid_y, 0.0),
    ]
    # one rib spanning the full clear width, parallel to the door: no sideways way past it
    curbs = [(CURB_T, width, curb_x, mid_y, 0.0)]
    if spur_x < curb_x - CURB_T / 2.0:  # the spur, back to the far side wall so nothing gets past
        far = curb_x + CURB_T / 2.0  # runs into the rib's far face, so the two fuse
        curbs.append((far - spur_x, spur_y + back_gap, (spur_x + far) / 2.0,
                      (spur_y - back_gap) / 2.0, 0.0))
    return walls, curbs


def build_garage(
    curb_height: float = CURB_H,
    wall_height: float = WALL_H,
    wall_gap: float = WALL_GAP,
    back_gap: float = BACK_GAP,
    depth_back: float = DEPTH_BACK,
    depth_door: float = DEPTH_DOOR,
    curb_x: float = CURB_X,
    spur_x: float = SPUR_X,
    spur_y: float = SPUR_Y,
    extent: float = DEFAULT_EXTENT,
    cell: float = DEFAULT_CELL,
) -> tuple[HeightMapReader, dict]:
    """The garage as a (reader, params) pair -- a pure builder, all IO in main(). `curb_height` 0
    omits the curb layer entirely, which is map A.

    Walls and curb are separate `rects_layer` calls because they sit at DIFFERENT heights, combined
    by an elementwise maximum -- the union-of-solids convention the other generators use.
    """
    back_x, door_x = depth_back, -depth_door
    width = wall_gap + back_gap  # clear, inner face to inner face
    walls, curbs = garage_rects(wall_gap, back_gap, depth_back, depth_door, curb_x, spur_x, spur_y)
    H = rects_layer(walls, wall_height, INCLINE_DEG, extent, cell)
    if curb_height > 0.0:
        H = np.maximum(H, rects_layer(curbs, curb_height, INCLINE_DEG, extent, cell))
    terrain = HeightMapReader(H, origin=(-extent / 2.0, -extent / 2.0), cell=cell)
    params = {
        "kind": "garage",
        "extent": float(extent),
        "cell": float(cell),
        "clear_width": float(width),
        "wall_gap": float(wall_gap),
        "back_gap": float(back_gap),
        "wall_height": float(wall_height),
        "wall_thickness": float(WALL_T),
        "curb_height": float(curb_height),
        "curb_thickness": float(CURB_T),
        "curb_x": float(curb_x) if curb_height > 0.0 else None,
        "spur_x": float(spur_x) if curb_height > 0.0 and len(curbs) > 1 else None,
        "spur_y": float(spur_y) if curb_height > 0.0 and len(curbs) > 1 else None,
        "incline_deg": float(INCLINE_DEG),
        "garage_x": [float(door_x), float(back_x)],
        "exit_bins": int(EXIT_BINS),
        "start": [float(v) for v in START],
        "goal": [float(v) for v in GOAL],
    }
    return terrain, params


def feature_distances(
    terrain: HeightMapReader,
    walls: list,
    curbs: list,
    wall_height: float,
    curb_height: float,
    relief: float = RELIEF,
) -> tuple[np.ndarray, np.ndarray]:
    """(dist_wall, dist_curb) [ny, nx] in metres: distance from each cell to the WALL structure
    and to the CURB, each rasterised from its own rectangles (ramps included). Two fields
    because they play opposite roles -- a wheel within `wheel_radius` of a wall is a pose the
    settle refuses, one within `wheel_radius` of the curb is a pose the settle accepts and only
    the network can catch. Distance to the footprint, not height under the centre, since the
    settle dilates the heightmap by the wheel radius and the tyre reaches a feature first."""
    extent, cell = (terrain.nx - 1) * terrain.cell, terrain.cell
    dist = lambda m: distance_transform_edt(~m) * cell
    # The wall's blocking footprint is only the part tall enough to break the tilt envelope: a
    # wheel on the very bottom of an 80 deg ramp is lifted a millimetre, not stopped. Taking the
    # PERMISSIVE reading is deliberate -- it lets `forward_reach` over-estimate where the robot can
    # drive, and lets `pivot_curb_coverage` count point turns the settle might actually refuse, so
    # both asserts are harder to pass than the truth.
    d_wall = dist(rects_layer(walls, wall_height, INCLINE_DEG, extent, cell) > BLOCK_H)
    if curb_height <= 0.0:
        return d_wall, np.full(terrain.H.shape, np.inf)
    return d_wall, dist(rects_layer(curbs, curb_height, INCLINE_DEG, extent, cell) > relief)


def sample_dist(field: np.ndarray, terrain: HeightMapReader, xy: np.ndarray) -> np.ndarray:
    """`field` (a per-cell quantity from `feature_distances`) at world points `xy[..., 2]`, nearest
    cell -- the same cell-centre convention as `HeightMapReader.sample`, clamped at the edges."""
    j = np.clip(np.round((xy[..., 0] - terrain.x0) / terrain.cell - 0.5).astype(int), 0, terrain.nx - 1)
    i = np.clip(np.round((xy[..., 1] - terrain.y0) / terrain.cell - 0.5).astype(int), 0, terrain.ny - 1)
    return field[i, j]


def pivot_primitive_audit(
    terrain: HeightMapReader,
    dist_wall: np.ndarray,
    dist_curb: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> dict:
    """Every single-bin POINT TURN placeable in the garage, split by whether it touches the curb --
    the ground truth the v_wz network's `e_pitch` field is scored against in `demos/garage_gate.py`,
    computed here from geometry so the two can be compared.

    A turn counts when the settle would allow it (no wheel centre within `wheel_radius` of a wall
    at any sub-heading) and touches the curb when some wheel comes within `wheel_radius` of it --
    the tyre reaching the feature, not the centre crossing it, which is the contact ostrich sees
    and the kinematic twin does not.
    """
    r, cell = float(_ROBOT.wheel_radius), terrain.cell
    xs = np.arange(x_range[0], x_range[1] + 1e-9, cell)
    ys = np.arange(y_range[0], y_range[1] + 1e-9, cell)
    gx, gy = (v.ravel() for v in np.meshgrid(xs, ys, indexing="ij"))
    sub = np.linspace(0.0, 1.0, 5)
    n_legal, n_touch, free_x = 0, 0, -math.inf
    for d in (-1, 1):
        for b in range(N_THETA):
            yaw = (b + 0.5) * BIN + d * BIN * sub
            local = wheels_of(np.stack([np.zeros_like(yaw), np.zeros_like(yaw), yaw], -1))
            w = local[None] + np.stack([gx, gy], -1)[:, None, None, :]
            flat = lambda f: sample_dist(f, terrain, w).reshape(len(gx), -1).min(axis=1)
            legal, touch = flat(dist_wall) >= r, flat(dist_curb) < r
            n_legal += int(legal.sum())
            n_touch += int((legal & touch).sum())
            if (legal & ~touch).any():
                free_x = max(free_x, float(gx[legal & ~touch].max()))
    return dict(n_legal=n_legal, n_touch=n_touch, free_x=free_x, n_centres=len(gx))


def write_map(args: argparse.Namespace, curb_height: float, stem: str) -> None:
    """Build, assert the design and write one garage."""
    terrain, params = build_garage(
        curb_height, args.wall_height, args.wall_gap, args.back_gap, args.depth_back,
        args.depth_door, args.curb_x, args.spur_x, args.spur_y, args.extent, args.cell,
    )
    r, c = float(_ROBOT.wheel_radius), clearances()
    pivot_ahead = max(c["pivot_ccw"]["ahead"], c["pivot_cw"]["ahead"])
    pivot_behind = max(c["pivot_ccw"]["behind"], c["pivot_cw"]["behind"])
    pivot_fore = max(c["pivot_ccw"]["fore"], c["pivot_cw"]["fore"])
    # the PEAK of the forward turn, not where it ends: the robot has to pass through it
    arc_ahead = min(max(c["arc_ccw"]["ahead"]), max(c["arc_cw"]["ahead"]))

    # --- the design, asserted: the map is only useful if all of these hold --------------------
    assert args.wall_gap > pivot_ahead, (
        f"the wall the robot faces is {args.wall_gap:.2f} m away but a 90 deg point turn throws a "
        f"wheel rim {pivot_ahead:.2f} m towards it -- the settle would block the only way out"
    )
    assert args.wall_gap < arc_ahead, (
        f"the wall the robot faces is {args.wall_gap:.2f} m away and a 90 deg FORWARD turn needs "
        f"only {arc_ahead:.2f} m -- the forward-only lattice could drive out and the map proves "
        "nothing about point turns"
    )
    assert args.back_gap > pivot_behind, (
        f"{args.back_gap:.2f} m behind the robot does not fit the point turn's {pivot_behind:.2f} m"
    )
    assert args.depth_back > pivot_fore, (
        f"{args.depth_back:.2f} m to the closed end does not fit the point turn's {pivot_fore:.2f} m"
    )
    pitch = math.atan2(curb_height, float(_ROBOT.rear_offset))
    roll = math.atan2(curb_height, 2.0 * float(_ROBOT.half_track))
    if curb_height > 0.0:
        assert pitch < float(_ROBOT.max_pitch_down), (
            f"a {curb_height:.2f} m curb pitches the body {math.degrees(pitch):.1f} deg with the "
            f"rear wheel on it, past max_pitch_down {math.degrees(_ROBOT.max_pitch_down):.1f} -- "
            "the settle would block the pivot on its own and the network would have nothing to add"
        )
    walls, curbs = garage_rects(args.wall_gap, args.back_gap, args.depth_back, args.depth_door,
                                args.curb_x, args.spur_x, args.spur_y)
    dist_wall, dist_curb = feature_distances(terrain, walls, curbs, args.wall_height, curb_height)
    # both spawn poses: the yaw the robot is really placed at, and the bin centre the lattice
    # rounds it to -- the settle has to accept the second or the planner has nowhere to start
    start_w = wheel_xy(START)
    spawn_w = np.stack([start_w, wheel_xy((START[0], START[1], bin_centre(START[2])))])
    for i, (wx, wy) in enumerate(start_w):
        assert abs(terrain.sample(wx, wy)) < 1e-9, f"start wheel {i} is not on flat ground"
    start_clear = float(sample_dist(dist_curb, terrain, spawn_w).min())
    assert start_clear >= r, (
        f"a start wheel centre is {start_clear:.2f} m from the curb, inside the settle's own "
        f"{r:.2f} m wheel envelope -- the spawn pose would be blocked before the robot moves"
    )
    assert float(sample_dist(dist_wall, terrain, start_w).min()) >= r, "a start wheel is on a wall"
    for name, pose in (("start", START), ("goal", (*GOAL, 0.0))):
        assert abs(terrain.sample(pose[0], pose[1])) < 1e-9, f"{name} is not on flat ground"
    assert abs(GOAL[0]) < args.extent / 2.0 and GOAL[0] < -args.depth_door, "goal is not outside"
    assert terrain.max_z == max(args.wall_height, curb_height), terrain.max_z

    # --- the premise, as a reachability closure over the lattice the planner will really use ----
    door_x = -args.depth_door
    bounds = (door_x - 1.2, args.depth_back, -args.back_gap, args.wall_gap)
    stop_x = door_x - 0.6  # clear of the mouth: the robot is out
    kw = dict(bounds=bounds, stop_x=stop_x)
    no_pivot = reach_closure(terrain, dist_wall, dist_curb, allow_pivots=False, **kw)
    assert not no_pivot["escaped"], (
        f"the robot drives out of the garage on forward arcs alone ({no_pivot['n_states']} states "
        "explored) -- the map does not force a point turn and proves nothing about gating them"
    )
    free = reach_closure(terrain, dist_wall, dist_curb, allow_pivots=True, **kw)
    assert free["escaped"], (
        f"even with point turns free the robot cannot get out ({free['n_states']} states, "
        f"{free['n_pivots']} turns) -- the garage is too tight for the route it is built around"
    )
    gated, audit = None, None
    if curb_height > 0.0:
        # the map's whole point: pruning the point turns that touch the curb takes the route away
        gated = reach_closure(terrain, dist_wall, dist_curb, allow_pivots=True, gate_curb=True, **kw)
        assert not gated["escaped"], (
            f"the robot still gets out using only point turns that stay clear of the curb "
            f"({gated['n_pivots']} of them) -- the curb is avoidable, so a correct network would "
            "predict a low error for the turn the planner takes and the map would prove nothing"
        )
        # And the START turn specifically -- the one the demo plans and reports. CCW is the exit
        # (swings the rear wheel over the curb toward the closed end); CW turns the nose into the
        # back wall instead, so only CCW is asserted here (`reach_closure` above covers the rest).
        # How deep it takes the rear wheel into the curb must land in a WINDOW: too shallow and
        # only the rim grazes (the drift below wipes it out); too deep and the wheel CENTRE is on
        # the curb, which the settle DOES see and blocks -- breaking the A/B pair.
        driven = pivot_wheels(START, DRIVEN_BINS, True)
        d_ccw = float(sample_dist(dist_curb, terrain, driven)[:, 2].min())
        assert 0.0 < d_ccw < r - PLANNED_OVERLAP, (
            f"the exit turn's rear wheel centre passes {d_ccw:.2f} m from the curb over the "
            f"{DRIVEN_BINS} bins the planner turns on the spot, outside the "
            f"(0, {r - PLANNED_OVERLAP:.2f}) m window: " + (
                "the centre is ON the curb, which the settle blocks -- map B will plan a "
                "different route from map A and the two are no longer one plan over two terrains"
                if d_ccw <= 0.0 else
                f"that leaves only {max(r - d_ccw, 0.0):.2f} m of tyre over it, and the drift "
                "below spends more than that"
            )
        )
        # ... and the same turn where ostrich really puts it: still a tyre on the curb, or the
        # physics side of this map measures nothing (see THE SPUR in the module doc)
        d_drift = float(sample_dist(dist_curb, terrain, driven + np.asarray(PIVOT_DRIFT))[:, 2].min())
        assert d_drift < r - CONTACT_MARGIN, (
            f"displaced by the {np.hypot(*PIVOT_DRIFT):.2f} m an in-place turn really slides, the "
            f"rear wheel centre passes {d_drift:.2f} m from the curb, leaving "
            f"{max(r - d_drift, 0.0):.2f} m of tyre over it against the {CONTACT_MARGIN:.2f} m "
            "this map is built to guarantee -- the curb has to reach further back"
        )
        audit = pivot_primitive_audit(terrain, dist_wall, dist_curb, (door_x, args.depth_back),
                                      (-args.back_gap, args.wall_gap))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / stem
    terrain.save(path)
    meta = yaml.safe_load(path.with_suffix(".yaml").read_text())
    meta.update(source="feasibility.heightmap.create_garage", **params)
    path.with_suffix(".yaml").write_text(yaml.safe_dump(meta, sort_keys=False))

    arcs = c["arc_ccw"]["ahead"]
    blocked_at = next((k for k, v in enumerate(arcs) if v > args.wall_gap), None)
    print(
        f"wrote {path}.png + .yaml  ({terrain.nx}x{terrain.ny} cells at {args.cell} m, "
        f"z in [{terrain.min_z:.2f}, {terrain.max_z:.2f}] m)\n"
        f"  garage  x in [{-args.depth_door:.2f}, {args.depth_back:.2f}] deep, "
        f"{args.wall_gap + args.back_gap:.2f} m clear wide, walls {args.wall_height:.2f} m\n"
        f"  start   {START[0]:.2f}, {START[1]:.2f} facing {math.degrees(bin_centre(START[2])):.1f} "
        f"deg (bin centre)  ->  goal {GOAL}\n"
        f"  pivot   90 deg on the spot needs {pivot_ahead:.2f} m ahead, {pivot_behind:.2f} m "
        f"behind, {pivot_fore:.2f} m to the closed end -- has {args.wall_gap:.2f} / "
        f"{args.back_gap:.2f} / {args.depth_back:.2f}\n"
        f"  arcs    the same 90 deg on forward arcs needs {arc_ahead:.2f} m ahead; the "
        f"{'' if blocked_at is None else f'{blocked_at}. '}arc is blocked at {args.wall_gap:.2f} m, "
        f"so the lattice turns at most "
        f"{math.degrees((blocked_at - 1) * ARC_STEP / float(_ROBOT.min_turn_radius)) if blocked_at else 90.0:.0f}"
        f" deg of the {90} it needs\n"
        + (
            f"  curb    {curb_height:.2f} m at x = {args.curb_x:.2f}, spanning the full width"
            + (f", with a spur back to x = {args.spur_x:.2f} over y < {args.spur_y:.2f} "
               f"(start wheels {start_clear:.2f} m clear, needs {r:.2f}); "
               if len(curbs) > 1 else "; ")
            + f"rear wheel on it pitches {math.degrees(pitch):.1f} deg "
            f"(max_pitch_down {math.degrees(_ROBOT.max_pitch_down):.0f}, so the settle is BLIND), "
            f"a front wheel would roll {math.degrees(roll):.1f} deg "
            f"(max_roll {math.degrees(_ROBOT.max_roll):.0f}, so the settle "
            f"{'blocks that' if roll > float(_ROBOT.max_roll) else 'is blind to that too'})\n"
            f"  turns   {audit['n_touch']} of the {audit['n_legal']} single-bin point turns the "
            f"settle allows in the garage touch it ({audit['n_touch'] / audit['n_legal']:.0%}); "
            f"the rest sit near the door, the nearest at x = {audit['free_x']:.2f}\n"
            f"  reach   over the {DRIVEN_BINS} bins the planner turns on the spot the rear wheel "
            f"centre passes {d_ccw:.2f} m from the curb, {r - d_ccw:.2f} m of tyre over it "
            f"(centre ON it and the settle would block the pose); displaced by the "
            f"{np.hypot(*PIVOT_DRIFT):.2f} m ostrich really slides, {r - d_drift:.2f} m of tyre "
            f"is still over it (needs {CONTACT_MARGIN:.2f})\n"
            f"  premise forward arcs alone cannot get out ({no_pivot['n_states']} states); with "
            f"point turns it can; with only CURB-FREE point turns it cannot again "
            f"({gated['n_states']} states, {gated['n_pivots']} turns) -- so pruning them is what "
            "closes the route\n"
            if curb_height > 0.0
            else f"  curb    none -- this is map A, the control; forward arcs alone still cannot "
            f"get out ({no_pivot['n_states']} states), point turns can\n"
        )
        + f"  view it: python demos/view_planned_path.py --map {path}.png --pivot-cost 0.15"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--curb-height", type=float, default=CURB_H, help="[m] map B's curb")
    ap.add_argument("--wall-height", type=float, default=WALL_H)
    ap.add_argument("--wall-gap", type=float, default=WALL_GAP,
                    help="[m] START to the wall it faces; must be inside the window clearances() prints")
    ap.add_argument("--back-gap", type=float, default=BACK_GAP)
    ap.add_argument("--depth-back", type=float, default=DEPTH_BACK)
    ap.add_argument("--depth-door", type=float, default=DEPTH_DOOR)
    ap.add_argument("--curb-x", type=float, default=CURB_X)
    ap.add_argument("--spur-x", type=float, default=SPUR_X,
                    help="[m] near face of the curb's spur; >= curb-x drops it (the rib-only map)")
    ap.add_argument("--spur-y", type=float, default=SPUR_Y,
                    help="[m] the y the spur starts at, running from there to the far side wall")
    ap.add_argument("--extent", type=float, default=DEFAULT_EXTENT)
    ap.add_argument("--cell", type=float, default=DEFAULT_CELL)
    ap.add_argument("--only", choices=("a", "b"), help="write just one of the two maps")
    args = ap.parse_args()

    # the vectorised wheel geometry the bulk checks run on IS `wheel_xy`, not a second copy of it
    probe = np.array([[1.0, 2.0, 0.3], [-0.4, 0.7, 2.9], [0.0, 0.0, bin_centre(START[2])]])
    assert np.allclose(wheels_of(probe), [wheel_xy(tuple(p)) for p in probe]), "wheels_of drifted"

    c = clearances()
    print("point turn (90 deg, both directions) needs, from the reference point [m]:")
    for tag in ("pivot_ccw", "pivot_cw"):
        v = c[tag]
        print(f"  {tag:<10} ahead {v['ahead']:.3f}  behind {v['behind']:.3f}  "
              f"fore {v['fore']:.3f}  aft {v['aft']:.3f}")
    print(f"the same 90 deg on forward arcs, outer wheel rim sideways after each "
          f"{ARC_STEP} m arc [m]:")
    for tag in ("arc_ccw", "arc_cw"):
        print(f"  {tag:<10} " + "  ".join(f"{v:.3f}" for v in c[tag]["ahead"]))
    print(f"-> WALL_GAP must lie in ({max(c['pivot_ccw']['ahead'], c['pivot_cw']['ahead']):.3f}, "
          f"{min(c['arc_ccw']['ahead'][-1], c['arc_cw']['ahead'][-1]):.3f}) m; it is {args.wall_gap}\n")

    if args.only != "b":
        write_map(args, 0.0, NAME_A)
    if args.only != "a":
        write_map(args, args.curb_height, NAME_B)


if __name__ == "__main__":
    main()
