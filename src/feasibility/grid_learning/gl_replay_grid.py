"""GL viewer for a grid_learning dataset_grid_*.h5 (generate_dataset.py's output): steps through
the sampled trials of one (map, commanded wz) row, showing each lattice cell's ostrich/hstack
FINAL poses on the real terrain, for manual sanity-checking before the dataset is fed to a
network -- does the spawn lattice line up with the obstacle(s), are masked cells the ones
actually on/near an obstacle rather than an unexplained diverged solve, do a cell's two final
poses look sane (not NaN, not flung across the map).

Unlike replay/gl_replay.py, there is no trajectory to interpolate: dataset_grid_*.h5 stores only
`y[R, G, G, 14]` (ostrich pose(7) + hstack pose(7) per lattice cell of the R-th row) and
`mask[R, G, G]` -- one keyframe per cell, not a time series. So this viewer is a stepper, not a
continuous replay: it steps through a row's G x G lattice cells (or one, via --cell), freezing
the two robot meshes at that cell's final poses each time, with no wheel spin (the dataset logs
no wheel velocity to integrate, only the final chassis pose -- see replay/gl_replay.py's own
wheel-angle-integration docstring for the corresponding caveat there).

The whole lattice is also drawn as small static spheres (Newton ModelBuilder.add_shape_sphere,
body=-1, same static-world pattern the ground mesh itself uses), three-way colored: green =
mask=True (valid solve), gray = never simulated (spawn footprint on an obstacle, recomputed
locally via footprint_clear() -- same math as generate_dataset.py's own obstacle filter, copied
rather than imported so this stays a standalone script, see that module's docstring on why
grid_learning/ re-derives rather than shares), red = was footprint-clear but mask is False anyway
-- a diverged solve, the case actually worth flagging. The currently-shown cell's sphere is
recolored white as a highlight. pos_error/rot_error (T_err = T1^-1 @ T2 convention) are the same
formula learning/pose_error.py uses, reimplemented locally for the same independence reason, so
printed numbers stay comparable to what that script/test_nn.py report elsewhere.

CLI parameters:
    --file PATH             dataset_grid_*.h5 path (required -- filenames encode M/L/g, no
                            single sensible default)
    --map-index INT         which of the file's n_maps maps to show (default: 0)
    --command-index INT     which of the file's n_commands wz draws for that map (default: 0)
    --cell I J               freeze on one specific lattice cell instead of stepping through all
                            of them (skips --dwell/--loop)
    --which {ostrich,hstack,both}   which robot mesh/meshes to render (default: both)
    --dwell FLOAT            seconds spent on each cell before advancing (default: 1.5)
    --loop                  wrap back to the first cell instead of freezing on the last
    --show-masked            also step through mask=False cells (default: skip them)
    --dry-run                load + validate the row and print summary stats, no GL viewer --
                            the only piece of this script testable without a display, mirroring
                            generate_dataset.py's own +dry_run convention

Usage:
    python src/feasibility/grid_learning/gl_replay_grid.py --file outputs/dataset_grid_box_random_M2_L5_g15.h5 --dry-run
    python src/feasibility/grid_learning/gl_replay_grid.py --file outputs/dataset_grid_box_random_M2_L5_g15.h5
    python src/feasibility/grid_learning/gl_replay_grid.py --file outputs/dataset_grid_box_random_M2_L5_g15.h5 --map-index 1 --command-index 2
    python src/feasibility/grid_learning/gl_replay_grid.py --file outputs/dataset_grid_box_random_M2_L5_g15.h5 --cell 7 7 --which ostrich
    python src/feasibility/grid_learning/gl_replay_grid.py --file outputs/dataset_grid_box_random_M2_L5_g15.h5 --show-masked --loop --dwell 0.5
"""
from __future__ import annotations

import argparse
import pathlib
import time

import h5py
import newton
import numpy as np
import warp as wp
from examples.helhest_junior.common import create_helhest_junior_model
from examples.helhest_junior.common import HelhestJuniorConfig
from ostrich.core.model_builder import OstrichModelBuilder

from feasibility.comparator.provenance import terrain_from_h5
from feasibility.heightmap import HeightMapReader

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

# create_helhest_junior_model always adds exactly 4 joints per robot -- see replay/gl_replay.py's
# build_model, same layout reused verbatim here.
JOINTS_PER_ROBOT = 4
WHEEL_NAMES = ("left", "right", "rear")
ROBOT_COLOR = {"ostrich": (0.9, 0.55, 0.1), "hstack": (0.55, 0.2, 0.85)}

# Lattice-marker appearance -- see module docstring for what each color means.
SPHERE_RADIUS = 0.08  # m
SPHERE_Z_OFFSET = 0.04  # m, lifts markers just above the terrain surface to avoid z-fighting
COLOR_VALID = (0.15, 0.85, 0.25)  # green -- mask=True
COLOR_BLOCKED = (0.55, 0.55, 0.55)  # gray -- spawn footprint on an obstacle, never simulated
COLOR_DIVERGED = (0.9, 0.15, 0.15)  # red -- footprint-clear but no valid solve
COLOR_HIGHLIGHT = (1.0, 1.0, 1.0)  # white -- the cell currently on screen

# Camera: parked above and back from the terrain's own center, framed off its own grid extent --
# same convention plotting/terrain_visualizer.py uses for a static "look at the whole map"
# camera, unlike replay/gl_replay.py's obstacle-relative one (this dataset's box position is
# random per map, there is no single obstacle_x to frame against).
CAMERA_YAW = 90.0
CAMERA_PITCH = -25.0
CAMERA_BACK = 0.7  # camera distance behind terrain center, as a fraction of the larger extent
CAMERA_UP = 0.4  # camera height above max_z, as a fraction of the larger extent

# Reference/spawn robot: every lattice cell's rollout starts from the SAME (yaw, resting-on-
# ground) configuration -- only its (x, y) differs per cell, and generate_dataset.py never logs
# that starting pose itself (only the two sims' FINAL poses). Parking one static, distinctly
# colored robot at a map corner -- outside the spawn lattice's own footprint, so it never
# overlaps a real trial -- gives a fixed "this is what upright, facing +X looked like before any
# of this diverged" reference alongside the moving ostrich/hstack pair.
REFERENCE_MARGIN = 0.8  # m inset from the terrain's min corner
REFERENCE_COLOR = (0.8, 0.8, 0.85)

# Same obstacle-vs-background footprint filter generate_dataset.py uses to decide which lattice
# cells get simulated at all -- copied rather than imported (see module docstring), so the
# constant below must stay in step with that module's own OBSTACLE_MARGIN_FRACTION by hand.
OBSTACLE_MARGIN_FRACTION = 0.15

WHEEL_CONTACTS_LOCAL = np.array(
    [
        [float(p[0]), float(p[1])]
        for p in (
            HelhestJuniorConfig.LEFT_WHEEL_POS,
            HelhestJuniorConfig.RIGHT_WHEEL_POS,
            HelhestJuniorConfig.REAR_WHEEL_POS,
        )
    ]
)  # [3, 2] body-frame (x, y) of the three wheel contacts


def obstacle_height_threshold(terrain: HeightMapReader) -> float:
    """Height above which a footprint point counts as "on an obstacle" -- see
    generate_dataset.py's identical helper for the full rationale."""
    baseline = float(np.median(terrain.H))
    return baseline + OBSTACLE_MARGIN_FRACTION * (terrain.max_z - baseline)


def footprint_clear(terrain: HeightMapReader, xy: np.ndarray, spawn_yaw: float) -> np.ndarray:
    """[G, G] bool -- True where a robot spawned at xy[i,j] facing spawn_yaw has its whole
    footprint (three wheel contacts plus body center) below obstacle_height_threshold. Identical
    math to generate_dataset.py's footprint_clear(), which decided which cells got simulated at
    all -- used here purely to explain a False mask (blocked vs. diverged), not to filter
    anything."""
    threshold = obstacle_height_threshold(terrain)
    c, s = np.cos(spawn_yaw), np.sin(spawn_yaw)
    local_x = np.append(WHEEL_CONTACTS_LOCAL[:, 0], 0.0)
    local_y = np.append(WHEEL_CONTACTS_LOCAL[:, 1], 0.0)
    wx = xy[..., 0, None] + c * local_x - s * local_y
    wy = xy[..., 1, None] + s * local_x + c * local_y
    heights = np.asarray(terrain.sample(wx, wy), dtype=np.float64)
    return heights.max(axis=-1) <= threshold


def pose_to_se3(pose: np.ndarray) -> np.ndarray:
    """(x, y, z, qx, qy, qz, qw) -> 4x4 SE(3) matrix -- same formula as learning/pose_error.py,
    reimplemented locally (see module docstring)."""
    x, y, z, qx, qy, qz, qw = pose
    R = np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ]
    )
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = (x, y, z)
    return T


def se3_error(T1: np.ndarray, T2: np.ndarray) -> tuple[float, float]:
    """T_err = T1^-1 @ T2 -> (pos_error [m], rot_error [rad]) -- same formula as
    learning/pose_error.py's se3_error, reimplemented locally (see module docstring)."""
    T_err = np.linalg.inv(T1) @ T2
    pos_error = float(np.linalg.norm(T_err[:3, 3]))
    trace = np.clip((np.trace(T_err[:3, :3]) - 1) / 2, -1.0, 1.0)
    rot_error = float(np.arccos(trace))
    return pos_error, rot_error


def cell_color(i: int, j: int, mask_row: np.ndarray, clear: np.ndarray) -> tuple[float, float, float]:
    if mask_row[i, j]:
        return COLOR_VALID
    return COLOR_BLOCKED if not clear[i, j] else COLOR_DIVERGED


def build_model(
    terrain: HeightMapReader,
    which: tuple[str, ...],
    lattice: np.ndarray,
    mask_row: np.ndarray,
    clear: np.ndarray,
    spawn_yaw: float,
) -> tuple[newton.Model, dict[str, dict[str, int]], np.ndarray]:
    """Ground mesh + one static reference/spawn robot (see its own constants' docstring) + one
    robot per `which` (frozen, poses written per-cell in main()) + one static sphere per lattice
    cell, colored per cell_color(). Returns (model, robots, sphere_shape_idx [G, G] int -- shape
    index of each cell's marker, for later recoloring).

    The reference robot occupies joint block 0 (create_helhest_junior_model always adds exactly
    JOINTS_PER_ROBOT joints); `which`'s movable robots are offset by one block to make room for
    it, hence `(i + 1) * JOINTS_PER_ROBOT` below instead of `i * JOINTS_PER_ROBOT`."""
    G = lattice.shape[0]
    builder = OstrichModelBuilder()
    ground_cfg = newton.ModelBuilder.ShapeConfig(mu=0.8)
    builder.add_shape_mesh(body=-1, mesh=terrain.to_ostrich_mesh(), cfg=ground_cfg)

    extent = max(terrain.nx, terrain.ny) * terrain.cell
    ref_x = terrain.x0 + terrain.nx * terrain.cell / 2.0 - extent / 2.0 + REFERENCE_MARGIN
    ref_y = terrain.y0 + terrain.ny * terrain.cell / 2.0 - extent / 2.0 + REFERENCE_MARGIN
    ref_z = float(terrain.sample(ref_x, ref_y)) + HelhestJuniorConfig.WHEEL_RADIUS
    ref_quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), spawn_yaw)
    ref_shape_start = builder.shape_count
    create_helhest_junior_model(builder, xform=wp.transform(wp.vec3(ref_x, ref_y, ref_z), ref_quat))
    ref_shapes = (ref_shape_start, builder.shape_count)

    robots: dict[str, dict[str, int]] = {}
    for i, name in enumerate(which):
        shape_start = builder.shape_count
        create_helhest_junior_model(builder, xform=wp.transform_identity())
        j_base = (i + 1) * JOINTS_PER_ROBOT
        robots[name] = {
            "base": j_base, "left": j_base + 1, "right": j_base + 2, "rear": j_base + 3,
            "shapes": (shape_start, builder.shape_count),
        }

    sphere_idx = np.zeros((G, G), dtype=np.int64)
    for i in range(G):
        for j in range(G):
            x, y = float(lattice[i, j, 0]), float(lattice[i, j, 1])
            z = float(terrain.sample(x, y)) + SPHERE_Z_OFFSET
            sphere_idx[i, j] = builder.add_shape_sphere(
                body=-1,
                xform=wp.transform(wp.vec3(x, y, z), wp.quat_identity()),
                radius=SPHERE_RADIUS,
                as_site=True,  # visual marker only, no collision response / contact cost
                color=cell_color(i, j, mask_row, clear),
            )

    model = builder.finalize()
    colors = model.shape_color.numpy()
    colors[ref_shapes[0]:ref_shapes[1]] = REFERENCE_COLOR
    if len(which) > 1:
        for name in which:
            start, end = robots[name]["shapes"]
            colors[start:end] = ROBOT_COLOR[name]
    model.shape_color.assign(colors)
    return model, robots, sphere_idx


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=pathlib.Path, required=True, help="dataset_grid_*.h5 path")
    ap.add_argument("--map-index", type=int, default=0, help="which map to show (default: 0)")
    ap.add_argument("--command-index", type=int, default=0, help="which wz draw for that map (default: 0)")
    ap.add_argument("--cell", type=int, nargs=2, metavar=("I", "J"), default=None,
                     help="freeze on one specific lattice cell (skips --dwell/--loop)")
    ap.add_argument("--which", choices=("ostrich", "hstack", "both"), default="both")
    ap.add_argument("--dwell", type=float, default=1.5, help="seconds per cell (default: 1.5)")
    ap.add_argument("--loop", action="store_true", help="wrap to the first cell instead of freezing on the last")
    ap.add_argument("--show-masked", action="store_true", help="also step through mask=False cells")
    ap.add_argument("--dry-run", action="store_true", help="validate + print stats only, no GL viewer")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    which = ("ostrich", "hstack") if args.which == "both" else (args.which,)

    with h5py.File(args.file, "r") as f:
        n_maps, n_commands = int(f.attrs["n_maps"]), int(f.attrs["n_commands"])
        n_rows, G = int(f.attrs["n_rows"]), int(f.attrs["grid_n"])
        if not (0 <= args.map_index < n_maps):
            raise SystemExit(f"--map-index must be in [0, {n_maps}), got {args.map_index}")
        if not (0 <= args.command_index < n_commands):
            raise SystemExit(f"--command-index must be in [0, {n_commands}), got {args.command_index}")
        row = args.map_index * n_commands + args.command_index

        spawn_yaw = float(f.attrs["spawn_yaw"])
        wz = float(f["wz"][row])
        map_path = f["map_path"].asstr()[row]
        terrain = terrain_from_h5(f, row)
        lattice = f["spawn_xy"][:]  # [G, G, 2], shared by every row
        mask_row = f["mask"][row]  # [G, G] bool
        y_row = f["y"][row]  # [G, G, 14]

    clear = footprint_clear(terrain, lattice, spawn_yaw)
    inconsistent = mask_row & ~clear
    if inconsistent.any():
        print(
            f"WARNING: {int(inconsistent.sum())} cell(s) are mask=True in the file but fail this "
            f"script's own footprint_clear() -- check OBSTACLE_MARGIN_FRACTION hasn't drifted "
            f"from generate_dataset.py's"
        )

    n_valid = int(mask_row.sum())
    n_blocked = int((~clear).sum())
    n_diverged = int((clear & ~mask_row).sum())
    print(
        f"[row {row}]  map={map_path}  commanded omega_z={wz:+.3f} rad/s   {G}x{G} lattice: "
        f"{n_valid} valid, {n_blocked} blocked (obstacle), {n_diverged} diverged"
    )

    if args.dry_run:
        print("[dry-run]  loaded + validated, nothing rendered")
        return

    if args.cell is not None:
        ci, cj = args.cell
        if not (0 <= ci < G and 0 <= cj < G):
            raise SystemExit(f"--cell must be within [0, {G}) x [0, {G}), got {args.cell}")
        cells = [(ci, cj)]
    else:
        cells = [(i, j) for i in range(G) for j in range(G) if mask_row[i, j] or args.show_masked]
        if not cells:
            raise SystemExit("no cells to show -- every cell is masked; pass --show-masked to include them")

    wp.init()
    model, robots, sphere_idx = build_model(terrain, which, lattice, mask_row, clear, spawn_yaw)
    viewer = newton.viewer.ViewerGL()
    viewer.set_model(model)
    cx = terrain.x0 + terrain.nx * terrain.cell / 2.0
    cy = terrain.y0 + terrain.ny * terrain.cell / 2.0
    extent = max(terrain.nx, terrain.ny) * terrain.cell
    viewer.set_camera(
        pos=wp.vec3(cx, cy - CAMERA_BACK * extent, terrain.max_z + CAMERA_UP * extent),
        pitch=CAMERA_PITCH, yaw=CAMERA_YAW,
    )
    state = model.state()

    q_start = model.joint_q_start.numpy()
    joint_q = model.joint_q.numpy().copy()
    joint_q_wp = wp.array(joint_q, dtype=wp.float32, device=model.device)
    joint_qd_wp = wp.zeros_like(model.joint_qd)  # eval_fk only needs joint_q for body_q

    colors = model.shape_color.numpy()
    highlighted: list[tuple[int, int] | None] = [None]

    def show_cell(i: int, j: int) -> None:
        if highlighted[0] is not None:
            pi, pj = highlighted[0]
            colors[sphere_idx[pi, pj]] = cell_color(pi, pj, mask_row, clear)
        colors[sphere_idx[i, j]] = COLOR_HIGHLIGHT
        model.shape_color.assign(colors)
        highlighted[0] = (i, j)

        x, y = lattice[i, j]
        if mask_row[i, j]:
            for name in which:
                base_q = q_start[robots[name]["base"]]
                joint_q[base_q:base_q + 7] = y_row[i, j, :7] if name == "ostrich" else y_row[i, j, 7:14]
            pos_e, rot_e = se3_error(pose_to_se3(y_row[i, j, :7]), pose_to_se3(y_row[i, j, 7:14]))
            status = f"valid  pos_err={pos_e:.3f} m  rot_err={rot_e:.3f} rad ({np.degrees(rot_e):.1f} deg)"
        elif not clear[i, j]:
            status = "blocked (on obstacle, never simulated) -- robots left at previous cell"
        else:
            status = "DIVERGED (footprint-clear but no valid solve) -- robots left at previous cell"
        print(f"  [cell {i:2d},{j:2d}] xy=({x:.2f},{y:.2f})  {status}")

    idx = 0
    show_cell(*cells[idx])
    freeze = len(cells) == 1
    last_switch = time.perf_counter()
    t0 = time.perf_counter()
    print(
        f"stepping {len(cells)} cell(s)  [{', '.join(which)}]  dwell={args.dwell}s  loop={args.loop}  "
        f"commanded omega_z={wz:+.3f} rad/s  (grey robot at the corner = shared spawn config)"
    )

    while viewer.is_running():
        now = time.perf_counter()
        if not freeze and now - last_switch >= args.dwell:
            if idx + 1 < len(cells):
                idx += 1
                show_cell(*cells[idx])
            elif args.loop:
                idx = 0
                show_cell(*cells[idx])
            else:
                freeze = True
            last_switch = now

        joint_q_wp.assign(joint_q)
        newton.eval_fk(model, joint_q_wp, joint_qd_wp, state)
        contacts = model.collide(state)
        viewer.begin_frame(now - t0)
        viewer.log_state(state)
        viewer.log_contacts(contacts, state)
        viewer.end_frame()
        wp.synchronize()


if __name__ == "__main__":
    main()
