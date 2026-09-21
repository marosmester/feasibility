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

`nn-report` plans exactly like vanilla-on, then asks the v_wz network for the predicted error of
every primitive the path takes -- the in-place turns included -- and adds one column per head the
checkpoint carries to the table (the arc that ARRIVED at each pose). Nothing is pruned by it yet.
A pos_rpy checkpoint is the default and the better instrument here: on pivot rows its e_pitch
separates an interacting turn from a near-miss 70.7x where a pos_rot sibling's e_rot manages 9.2x
(see `planning.arc_network.load_network_vwz`), and its flat-ground floor is an order of magnitude
lower, which is what decides how many good arcs a gate throws away.

The per-pose table also audits the plan against the terrain the planner half-ignores: for every
pose it takes the highest ground within one wheel radius of each of the three wheel centres --
the settle's own spherical envelope, and the tyre's -- and flags the ones standing on relief above
`--relief` (0.05 m, the dataset's `trial.interact_relief`). The settle only blocks a pose once a
wheel is lifted far enough to break the tilt envelope, so a low curb shows up here while `blocked`
stays 0 -- those are the arcs a v_wz network has to gate. Sampling the centre alone, which is what
this did first, reads a wheel dragging its rim along a curb face as standing on flat ground.

That column is also how the MAP is checked. A heightmap generator can assert its design at the
start pose, but the planner turns where it likes, so the premise "every point turn drags a wheel
through a curb" is only true of a path once it has been planned. It was not: the first version of
`create_pivot_pocket.py` used rails along the pocket, and the planner placed its pivots 0.20 m
off the centreline where the rear wheel's orbit threads between them, turning on ground as flat
as open floor -- with the network correctly predicting a low error, which read as the network
failing. A plan whose point turns are all on flat ground now prints PREMISE FAILED instead.

The last block scores the point turns against `planners.TAU_PIVOT_PITCH`, the threshold calibrated
for them alone on the held-out maps of the v_wz checkpoint's own dataset. It is not a ranking of
pivots against forward arcs -- on this map the arcs score higher, since driving head-on into a rib
diverges more than pivoting on one -- because the e_pitch head is near-unbiased on a point turn, so
the gate compares each one to a tolerance in radians instead of to the other class.

CLI parameters:
    --map PATH        heightmap stem or .png (default: assets/pivot_pocket/pivot_pocket_h020cm,
                      the 0.2 m-curb map from `create_pivot_pocket.py --series`)
    --planner NAME    a PLANNERS key (default nn-report); the nn-gated ones need --pivot-cost 0
    --checkpoint-vwz PATH   the v_wz net nn-report queries, pos_rpy or pos_rot (see DEFAULT_CHECKPOINT_VWZ)
    --pivot-cost M    m-equivalent per 15 deg bin for an in-place turn (default 0.15; 0 = no pivots)
    --start X Y YAW / --goal X Y   override the map sidecar's own
    --relief M        wheel-relief flagged in the table (default 0.05)
    --tau-pivot-pitch RAD   the point-turn gate's tolerance the report scores against
                      (default planners.TAU_PIVOT_PITCH)
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
from helhest.engine import RobotParams
from ostrich.core.model_builder import OstrichModelBuilder

from feasibility.heightmap.create_pivot_pocket import wheel_xy
from feasibility.heightmap.heightmap_reader import HeightMapReader
from feasibility.planning.gated_lattice import N_PRIM_ARC
from feasibility.planning.gated_lattice import N_THETA
from feasibility.planning.planners import DEFAULT_CHECKPOINT_VWZ
from feasibility.planning.planners import plan_path
from feasibility.planning.planners import PlanContext
from feasibility.planning.planners import PlannerConfig
from feasibility.planning.planners import PLANNERS
from feasibility.planning.planners import TAU_PIVOT_PITCH

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MAP = REPO_ROOT / "assets" / "pivot_pocket" / "pivot_pocket_h020cm"

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
    """[n, 3] the ground each wheel is really standing on, per pose: the HIGHEST terrain within one
    `wheel_radius` of the centre, not the height under the centre itself.

    The settle dilates the heightmap by that radius (an isotropic spherical envelope) and so does
    the real tyre -- a wheel whose centre is 0.2 m short of a curb face is already riding up it.
    Sampling the centre alone is what made the pivot pocket's point turns read as standing on flat
    ground while ostrich was dragging a rim through a rib, so the audit here uses the same envelope
    the settle does. Relief the settle TOLERATES (a low curb tilts the body inside its limits) is
    still invisible in `blocked` and shows up only here -- which is the arc a network has to gate.
    """
    r = float(RobotParams().wheel_radius)
    k = int(math.ceil(r / terrain.cell))
    d = np.arange(-k, k + 1) * terrain.cell
    ox, oy = (v.ravel() for v in np.meshgrid(d, d, indexing="ij"))
    disc = (ox**2 + oy**2) <= r**2 + 1e-12
    ox, oy = ox[disc], oy[disc]
    out = np.empty((len(poses), 3))
    for i, p in enumerate(poses):
        for j, (x, y) in enumerate(wheel_xy(tuple(p))):
            out[i, j] = float(np.max(terrain.sample(x + ox, y + oy)))
    return out


def report(
    result: dict, poses: np.ndarray, relief: np.ndarray, threshold: float, tau_pivot: float
) -> None:
    """The per-pose table: position, heading, whether the pose is a point turn, the wheels standing
    on relief above `threshold` and, for `nn-report`, the network's predicted error of the
    primitive that ARRIVED at the pose (blank for the start), one column per checkpoint head.

    Ends with the map's own premise re-checked ON THE PLANNED PATH. The heightmap generator can
    only assert its design over the poses it knows about (`create_pivot_pocket`'s
    `pivot_relief_coverage`); the planner turns where it likes. A plan whose point turns all sit
    on flat ground has nothing for a network to predict, so that
    is reported as a FAILED premise rather than left to be read off a column of zeros.
    """
    errs = result.get("arc_errors")
    heads = list(errs) if errs else []
    print(f"\n  {'#':>3}  {'x':>6} {'y':>6}  {'yaw':>6}  move   wheels on relief > "
          f"{threshold:.2f} m  (L / R / rear, m)"
          + ("   " + " ".join(f"{h:>7}" for h in heads) if heads else ""))
    is_pivot_pose = np.zeros(len(poses), bool)
    on_curb = 0
    for i, (x, y, yaw) in enumerate(poses):
        is_pivot_pose[i] = i > 0 and np.allclose(poses[i, :2], poses[i - 1, :2])
        hits = relief[i] > threshold
        on_curb += int(hits.any())
        cells = "  ".join(f"{'*' if h else ' '}{z:5.2f}" for z, h in zip(relief[i], hits))
        nn = "   " + " ".join(f"{errs[h][i - 1]:7.3f}" for h in heads) if heads and i > 0 else ""
        print(f"  {i:3d}  {x:6.2f} {y:6.2f}  {math.degrees(yaw):6.1f}  "
              f"{'PIVOT' if is_pivot_pose[i] else '     '}  {cells}{nn}")
    n_pivot = int(is_pivot_pose.sum())
    # a pivot's own contact is the relief under the pose it turns INTO, so count the pivot poses
    pivot_on_curb = int((is_pivot_pose & (relief > threshold).any(axis=1)).sum())
    print(
        f"\n  reachable {result['reachable']}   reached {result['reached']}   "
        f"poses {result['n_poses']}   billed {result['path_m']:.2f} m\n"
        f"  in-place turns {n_pivot} of {len(poses) - 1} primitives "
        f"({n_pivot * 360.0 / N_THETA:.0f} deg turned on the spot)\n"
        f"  poses the settle calls BLOCKED: {result['n_settle_bad']}\n"
        f"  poses with a wheel on relief > {threshold:.2f} m: {on_curb}"
        f"  (of which in-place turns: {pivot_on_curb}/{n_pivot})"
    )
    if n_pivot and not pivot_on_curb:
        print(
            f"\n  !! PREMISE FAILED: all {n_pivot} planned point turns stand on flat ground.\n"
            "     The planner turned somewhere the obstacle does not reach, so there is no\n"
            "     divergence here for a network to predict -- a low predicted error is CORRECT.\n"
            "     Fix the map (create_pivot_pocket.pivot_relief_coverage) before reading the\n"
            "     numbers below as evidence about the network."
        )
    if not heads or not len(result["prims"]):  # an unreachable goal has no primitives to report
        return
    prim_pivot = np.asarray(result["prims"]) >= N_PRIM_ARC
    print(f"\n  v_wz network, per primitive taken (gate: {result['gate']})")
    print(f"    {'':<15} {'n':>3}   " + "  ".join(f"{h:^17}" for h in heads))
    for label, mask in (("in-place turns", prim_pivot), ("forward arcs", ~prim_pivot)):
        if not mask.any():
            continue
        cells = "  ".join(
            f"max {errs[h][mask].max():5.3f} mu {errs[h][mask].mean():5.3f}" for h in heads
        )
        print(f"    {label:<15} {int(mask.sum()):3d}   {cells}")
    # The point turns against their own calibrated threshold. Not a ranking against the forward
    # arcs: on this map the arcs score HIGHER (driving head-on into a rib diverges more than
    # pivoting on one), yet that costs the gate nothing, because `planners.TAU_PIVOT_PITCH` is a
    # tolerance in radians the head is calibrated against, not a cut between the two classes.
    if "e_pitch" in heads and prim_pivot.any():
        p = errs["e_pitch"][prim_pivot]
        print(f"\n  pivot gate: e_pitch > {tau_pivot:.4f} rad ({math.degrees(tau_pivot):.1f} deg, "
              f"planners.TAU_PIVOT_PITCH)")
        print(f"    would prune {int((p > tau_pivot).sum())} of the {len(p)} point turns taken   "
              f"predicted e_pitch " + " ".join(f"{v:.3f}" for v in p))
        a = errs["e_pitch"][~prim_pivot]
        if len(a):
            print(f"    the same number on the {len(a)} forward arcs, for scale: "
                  f"{int((a > tau_pivot).sum())} over it, worst {a.max():.3f}")


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
    ap.add_argument("--planner", choices=list(PLANNERS), default="nn-report")
    ap.add_argument("--checkpoint-vwz", type=pathlib.Path, default=DEFAULT_CHECKPOINT_VWZ)
    ap.add_argument("--pivot-cost", type=float, default=0.15)
    ap.add_argument("--start", type=float, nargs=3, metavar=("X", "Y", "YAW"))
    ap.add_argument("--goal", type=float, nargs=2, metavar=("X", "Y"))
    ap.add_argument("--relief", type=float, default=0.05, help="wheel relief flagged in the table")
    ap.add_argument("--torch-device", default="cpu",
                    help="where the network runs; nn-gated-pivot needs a field over every lattice "
                         "pose, which is minutes on cpu (see planning.arc_network.vwz_error_fields)")
    ap.add_argument("--tau-pivot-pitch", type=float, default=TAU_PIVOT_PITCH,
                    help="[rad] the point-turn gate's e_pitch tolerance the report scores against")
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
    cfg = PlannerConfig(pivot_cost=args.pivot_cost, checkpoint_vwz=args.checkpoint_vwz,
                        tau_pivot_pitch=args.tau_pivot_pitch, torch_device=args.torch_device)
    ctx = PlanContext(cfg)
    try:
        result = plan_path(terrain, start, goal, args.planner, ctx)
    except NotImplementedError as exc:  # a gated planner on a pivot lattice -- a CLI combination
        raise SystemExit(f"{exc}\n\nRe-run with --pivot-cost 0, or --planner vanilla-on.")
    del ctx  # the CostToGo's settle buffers go before the viewer's model build (3 GiB GPU)

    poses = path_poses(result)
    report(result, poses, wheel_relief(poses, terrain), args.relief, cfg.tau_pivot_pitch)
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
