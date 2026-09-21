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
    |   |    ~~~~~~~~~~~~~~~~~~~~~~~~~~~    |        low curbs (CURB_H, INVISIBLE to the settle)
    |   |                                   |
  --+---|            >>> start          goal|x       start faces the closed end (+x)
    |   |                                   |
    |   |    ~~~~~~~~~~~~~~~~~~~~~~~~~~~    |
    |   +-----------------------------------+
        ^ mouth (open to -x)                ^ back wall

The two heights do different jobs, and the split is the point of the map:

  * The POCKET walls are WALL_H = 0.70 m. At that height the settle blocks poses well before the
    robot reaches them (a wheel lifted that far breaks `max_roll` / `max_pitch_down`), so the
    planner genuinely respects the pocket and has to turn around inside it.
  * The CURBS are CURB_H = 0.15 m, and the settle is blind to them at EVERY heading: a wheel lifted
    0.15 m tilts the body ~11.5 deg, inside the 15 deg envelope, so `blocked` stays 0 and the only
    trace left is a small `graded_tilt` penalty. They run the full length of the pocket at
    +-CURB_OFFSET, which is inside the rear wheel's pivot sweep (it orbits the front-axle midpoint
    at `rear_offset` = 0.75 m), so a point turn ANYWHERE in the pocket drags a wheel through one --
    sideways, into a vertical face, which is not something the kinematic twin can represent at all.

So the planner is forced to do the one thing it models worst. `demos/view_planned_path.py` draws
the plan; part B is gating those pivots with a v_wz network.

Usage:
    python src/feasibility/heightmap/create_pivot_pocket.py
    python src/feasibility/heightmap/create_pivot_pocket.py --width 3.0 --curb-height 0.2
    python src/feasibility/heightmap/create_pivot_pocket.py --out-dir assets/pivot_pocket
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
CURB_T = 0.40  # m curb thickness

DEFAULT_WIDTH = 2.8  # m clear, inner wall face to inner wall face
POCKET_X0 = -1.0  # m, the open mouth
POCKET_X1 = 4.0  # m, inner face of the back wall
CURB_OFFSET = 0.85  # m, |y| of the curb centreline -- inside the rear wheel's pivot orbit
START = (2.5, 0.0, 0.0)  # inside the pocket, facing the closed end
GOAL = (-4.0, 0.0)  # outside it, behind the robot

_ROBOT = RobotParams()
RIM_REACH = _ROBOT.rear_offset + _ROBOT.wheel_radius  # 1.10 m, the pivot's own swept radius


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
    curbs = [  # full pocket length, so there is no curb-free cell to pivot on
        (POCKET_X1 - POCKET_X0, CURB_T, (POCKET_X0 + POCKET_X1) / 2.0, +CURB_OFFSET, 0.0),
        (POCKET_X1 - POCKET_X0, CURB_T, (POCKET_X0 + POCKET_X1) / 2.0, -CURB_OFFSET, 0.0),
    ]
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
        "curb_offset": float(CURB_OFFSET),
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
    args = ap.parse_args()

    terrain, params = build_pivot_pocket(
        args.width, args.curb_height, args.wall_height, args.extent, args.cell
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
    for name, pose in (("start", START), ("goal", (*GOAL, 0.0))):
        assert abs(terrain.sample(pose[0], pose[1])) < 1e-9, f"{name} is not on flat ground"
    for i, (wx, wy) in enumerate(wheel_xy(START)):
        assert abs(terrain.sample(wx, wy)) < 1e-9, f"start wheel {i} is not on flat ground"
    # the design intent: a point turn at the start drags the rear wheel through a curb, both ways
    for yaw in (np.pi / 2.0, -np.pi / 2.0):
        rear = wheel_xy((START[0], START[1], yaw))[2]
        z = float(terrain.sample(*rear))
        assert z > 0.9 * args.curb_height, (
            f"rear wheel at yaw {np.degrees(yaw):.0f} deg sits at z={z:.3f}, not on the curb -- "
            f"CURB_OFFSET {CURB_OFFSET} is outside the {_ROBOT.rear_offset} m pivot orbit"
        )
    assert float(terrain.sample(*GOAL)) == 0.0 and abs(GOAL[0]) < args.extent / 2.0
    assert terrain.max_z == args.wall_height, terrain.max_z

    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / args.name
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
        f"  curbs   {args.curb_height:.2f} m at y = +-{CURB_OFFSET} -- the settle does NOT\n"
        f"  start   {START}  ->  goal {GOAL}\n"
        f"  view it: python demos/view_planned_path.py --map {path}.png --pivot-cost 0.15"
    )


if __name__ == "__main__":
    main()
