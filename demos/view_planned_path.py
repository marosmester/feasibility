"""Plan on a saved heightmap and look at the resulting path in the Newton GL viewer.

Runs one `feasibility.planning` planner on a map, then draws the lattice's own traced policy --
one RED ARROW per planned pose (x, y, yaw), pointing the way the robot faces there -- over the
terrain mesh, with the Helhest Junior model parked at the start for scale. No physics is stepped:
this shows what was PLANNED, not what ostrich would do with it.

Built for `heightmap/create_pivot_pocket.py`'s map, whose only route out is an in-place turn, so
the drawing has to survive a run of poses that share one cell. A pivot turns 15 deg without moving,
which as flat arrows would stack into an unreadable star at one point -- so consecutive poses in
the SAME cell are lifted by ARROW_DZ each, and a point turn reads as a little spiral staircase
while ordinary driving stays flat. An amber trail links the poses in order (`--no-trail` drops it).

The per-pose table also audits the plan against the terrain the planner half-ignores: for every
pose it samples the ground under each of the three wheel centres and flags the ones standing on
relief above `--relief` (0.05 m, the dataset's `trial.interact_relief`). The settle only blocks a
pose once a wheel is lifted far enough to break the tilt envelope, so a low curb shows up here
while `blocked` stays 0 -- those are the arcs a v_wz network has to gate.

CLI parameters:
    --map PATH        heightmap stem or .png (default: assets/pivot_pocket/pivot_pocket)
    --planner NAME    a PLANNERS key (default vanilla-on); the nn-gated ones need --pivot-cost 0
    --pivot-cost M    m-equivalent per 15 deg bin for an in-place turn (default 0.15; 0 = no pivots)
    --start X Y YAW / --goal X Y   override the map sidecar's own
    --relief M        wheel-relief flagged in the table (default 0.05)
    --arrow-len M     arrow length (default 0.45)
    --no-robot        skip the scale model; --no-trail drops the connecting line
    --dry-run         plan and print, no viewer (works headless)

Usage:
    python demos/view_planned_path.py
    python demos/view_planned_path.py --pivot-cost 0 --dry-run     # no route without pivots
    python demos/view_planned_path.py --map assets/lattice_maps/0/ramps_i0000 --pivot-cost 0
"""
from __future__ import annotations

import argparse
import math
import pathlib

import newton
import numpy as np
import warp as wp
import yaml
from examples.helhest_junior.common import create_helhest_junior_model
from examples.helhest_junior.common import HelhestJuniorConfig
from ostrich.core.model_builder import OstrichModelBuilder

from feasibility.heightmap.create_pivot_pocket import wheel_xy
from feasibility.heightmap.heightmap_reader import HeightMapReader
from feasibility.planning.gated_lattice import N_THETA
from feasibility.planning.planners import plan_path
from feasibility.planning.planners import PlanContext
from feasibility.planning.planners import PlannerConfig
from feasibility.planning.planners import PLANNERS

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MAP = REPO_ROOT / "assets" / "pivot_pocket" / "pivot_pocket"

ARROW_LEN = 0.45  # [m] drawn heading arrow -- longer than the 0.3 m arc, so a turn is legible
ARROW_Z = 0.12  # [m] above the ground under the pose, so arrows clear the terrain mesh
ARROW_DZ = 0.10  # [m] extra lift per consecutive pose in the SAME cell (see module docstring)
COLOR_PATH = (0.95, 0.15, 0.15)  # red -- the planned heading at each pose
COLOR_TRAIL = (1.0, 0.75, 0.2)  # amber -- the order the poses are visited in
COLOR_GOAL = (0.2, 0.9, 0.3)
GOAL_RADIUS = 0.15

# Camera framing, as in plotting/terrain_visualizer.py (yaw=90 faces +Y in Newton's convention).
CAMERA_YAW, CAMERA_PITCH = 90.0, -40.0
CAMERA_BACK, CAMERA_UP = 0.55, 0.65  # fractions of the larger terrain extent


def path_poses(result: dict) -> np.ndarray:
    """[n, 3] world (x, y, yaw) of the traced lattice states. `plan_path` already puts the cell
    corners in `xy`; the heading is its bin's centre, the same pose `CostToGo`'s settle judged."""
    t = result["states"][:, 2]
    return np.column_stack([result["xy"], (t + 0.5) * 2.0 * np.pi / N_THETA])


def arrow_segments(
    poses: np.ndarray, terrain: HeightMapReader, length: float
) -> tuple[np.ndarray, np.ndarray]:
    """(starts, ends) [n, 3] for one heading arrow per pose, lifted clear of the terrain. Poses in
    a run of identical cells (a point turn) each get one more ARROW_DZ, so they spiral instead of
    overdrawing each other."""
    stack = np.zeros(len(poses), dtype=np.int64)
    for i in range(1, len(poses)):
        stack[i] = stack[i - 1] + 1 if np.allclose(poses[i, :2], poses[i - 1, :2]) else 0
    z = np.asarray(terrain.sample(poses[:, 0], poses[:, 1])) + ARROW_Z + stack * ARROW_DZ
    starts = np.column_stack([poses[:, 0], poses[:, 1], z])
    tip = np.column_stack([length * np.cos(poses[:, 2]), length * np.sin(poses[:, 2]), np.zeros(len(poses))])
    return starts, starts + tip


def wheel_relief(poses: np.ndarray, terrain: HeightMapReader) -> np.ndarray:
    """[n, 3] ground height under each wheel centre, per pose -- what the plan is really standing
    on. Relief the settle tolerates (a low curb) is invisible in `blocked` but shows up here."""
    return np.array([[float(terrain.sample(x, y)) for x, y in wheel_xy(tuple(p))] for p in poses])


def report(result: dict, poses: np.ndarray, relief: np.ndarray, threshold: float) -> None:
    """The per-pose table: position, heading, whether the pose is a point turn, and the wheels
    standing on relief above `threshold`."""
    print(f"\n  {'#':>3}  {'x':>6} {'y':>6}  {'yaw':>6}  move   wheels on relief > "
          f"{threshold:.2f} m  (L / R / rear, m)")
    on_curb = 0
    for i, (x, y, yaw) in enumerate(poses):
        pivot = i > 0 and np.allclose(poses[i, :2], poses[i - 1, :2])
        hits = relief[i] > threshold
        on_curb += int(hits.any())
        cells = "  ".join(f"{'*' if h else ' '}{z:5.2f}" for z, h in zip(relief[i], hits))
        print(f"  {i:3d}  {x:6.2f} {y:6.2f}  {math.degrees(yaw):6.1f}  "
              f"{'PIVOT' if pivot else '     '}  {cells}")
    n_pivot = sum(
        1 for i in range(1, len(poses)) if np.allclose(poses[i, :2], poses[i - 1, :2])
    )
    print(
        f"\n  reachable {result['reachable']}   reached {result['reached']}   "
        f"poses {result['n_poses']}   billed {result['path_m']:.2f} m\n"
        f"  in-place turns {n_pivot} of {len(poses) - 1} primitives "
        f"({n_pivot * 360.0 / N_THETA:.0f} deg turned on the spot)\n"
        f"  poses the settle calls BLOCKED: {result['n_settle_bad']}\n"
        f"  poses with a wheel on relief > {threshold:.2f} m: {on_curb}"
    )


def build_model(terrain: HeightMapReader, start: tuple[float, float, float], with_robot: bool):
    """Terrain mesh + (optionally) a static Helhest Junior at the start pose, for scale. Same
    OstrichModelBuilder / eval_fk pattern as plotting/terrain_visualizer.py -- the wheel joints use
    a custom attribute only that builder registers, and finalize() leaves the robot at the origin
    until joint_q is propagated."""
    builder = OstrichModelBuilder()
    builder.add_shape_mesh(
        body=-1, mesh=terrain.to_ostrich_mesh(), cfg=newton.ModelBuilder.ShapeConfig(mu=0.8)
    )
    if with_robot:
        z = float(terrain.sample(start[0], start[1])) + HelhestJuniorConfig.WHEEL_RADIUS
        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(start[2]))
        create_helhest_junior_model(
            builder, xform=wp.transform(wp.vec3(start[0], start[1], z), rot)
        )
    return builder.finalize()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--map", type=pathlib.Path, default=DEFAULT_MAP)
    ap.add_argument("--planner", choices=list(PLANNERS), default="vanilla-on")
    ap.add_argument("--pivot-cost", type=float, default=0.15)
    ap.add_argument("--start", type=float, nargs=3, metavar=("X", "Y", "YAW"))
    ap.add_argument("--goal", type=float, nargs=2, metavar=("X", "Y"))
    ap.add_argument("--relief", type=float, default=0.05, help="wheel relief flagged in the table")
    ap.add_argument("--arrow-len", type=float, default=ARROW_LEN)
    ap.add_argument("--no-robot", action="store_true")
    ap.add_argument("--no-trail", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="plan and print, no viewer")
    args = ap.parse_args()

    map_path = args.map.with_suffix("")
    terrain = HeightMapReader.load(map_path)
    meta = yaml.safe_load(map_path.with_suffix(".yaml").read_text())
    start = tuple(args.start) if args.start else meta.get("start")
    goal = tuple(args.goal) if args.goal else meta.get("goal")
    if start is None or goal is None:
        raise SystemExit(f"{map_path.name}.yaml has no start/goal -- pass --start X Y YAW --goal X Y")
    start = tuple(float(v) for v in start)
    goal = (float(goal[0]), float(goal[1]))

    wp.init()
    print(f"=== {map_path.name}: {args.planner}, pivot_cost {args.pivot_cost}, "
          f"start {start} -> goal {goal} ===")
    ctx = PlanContext(PlannerConfig(pivot_cost=args.pivot_cost))
    try:
        result = plan_path(terrain, start, goal, args.planner, ctx)
    except NotImplementedError as exc:  # a gated planner on a pivot lattice -- a CLI combination
        raise SystemExit(f"{exc}\n\nRe-run with --pivot-cost 0, or --planner vanilla-on.")
    del ctx  # the CostToGo's settle buffers go before the viewer's model build (3 GiB GPU)

    poses = path_poses(result)
    report(result, poses, wheel_relief(poses, terrain), args.relief)
    if not result["reachable"]:
        print("\n  no route from this start -- nothing to draw"
              + ("" if args.pivot_cost > 0 else "; try --pivot-cost 0.15"))
        return
    starts, ends = arrow_segments(poses, terrain, args.arrow_len)
    ground = np.asarray(terrain.sample(poses[:, 0], poses[:, 1]))
    print(f"  {len(poses)} arrows, {float((starts[:, 2] - ground).min()):.2f}-"
          f"{float((starts[:, 2] - ground).max()):.2f} m above the ground under each pose "
          f"(the point turn spirals up by ARROW_DZ = {ARROW_DZ} m per bin)")
    if args.dry_run:
        return

    model = build_model(terrain, start, not args.no_robot)
    cx = terrain.x0 + terrain.nx * terrain.cell / 2.0
    cy = terrain.y0 + terrain.ny * terrain.cell / 2.0
    extent = max(terrain.nx, terrain.ny) * terrain.cell

    viewer = newton.viewer.ViewerGL()
    viewer.set_model(model)
    viewer.set_camera(
        pos=wp.vec3(cx, cy - CAMERA_BACK * extent, terrain.max_z + CAMERA_UP * extent),
        pitch=CAMERA_PITCH,
        yaw=CAMERA_YAW,
    )
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    dev = viewer.device
    arrow_a = wp.array(starts, dtype=wp.vec3, device=dev)
    arrow_b = wp.array(ends, dtype=wp.vec3, device=dev)
    trail_a = wp.array(starts[:-1], dtype=wp.vec3, device=dev)
    trail_b = wp.array(starts[1:], dtype=wp.vec3, device=dev)
    goal_pt = wp.array(
        [[goal[0], goal[1], float(terrain.sample(*goal)) + ARROW_Z]], dtype=wp.vec3, device=dev
    )
    # log_points' `radii`/`colors` are typed to accept a scalar and an RGB tuple, but ViewerGL
    # hands both straight to a kernel / .numpy() and a non-array raises. log_lines/log_arrows do
    # expand a tuple colour, so only the points need this. One entry, one goal point.
    goal_r = wp.array([GOAL_RADIUS], dtype=wp.float32, device=dev)
    goal_c = wp.array([COLOR_GOAL], dtype=wp.vec3, device=dev)

    print(f"\n  drawing {len(poses)} red heading arrows"
          f"{'' if args.no_trail else ' + the amber trail linking them'}; orbit with the mouse")
    while viewer.is_running():
        viewer.begin_frame(0.0)
        viewer.log_state(state)
        viewer.log_arrows("planned_path", arrow_a, arrow_b, COLOR_PATH)
        if not args.no_trail:
            viewer.log_lines("planned_trail", trail_a, trail_b, COLOR_TRAIL)
        viewer.log_points("goal", goal_pt, goal_r, goal_c)
        viewer.end_frame()


if __name__ == "__main__":
    main()
