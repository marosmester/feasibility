"""GL browser for an mppi_learning dataset_mppi_*.h5 (generate_dataset.py's output), after
`lattice_learning/gl_replay_arc.py`: it replays ONE row at a time over the row's own terrain in
Newton's ViewerGL. N / P (or RIGHT / LEFT) step through the rows.

What is drawn is the label, the two poses whose difference the net learns:

  * **ostrich** (orange robot): the whole stored rollout -- the settle drop and the warm-up at
    negative time, then the 1 s window from t = 0. Ostrich's own window path is an orange line,
    the warm-up before it a grey one.
  * **the twin** (purple robot): helhest_stack's `planning_solver` rollout of MPPI's command from
    ostrich's realized window start (`t0_pose`, `t0_wheel_omega`, `t0_twist`). The file stores only
    its END pose, so the browser re-runs it with `twin.run_twin(..., every_step=True)` and animates
    the path. The purple robot waits at the window start until t = 0 and then drives, and its path
    is a purple line. A WARNING is printed if the rerun's end pose differs from the stored
    `twin_pose` (the twin is deterministic, so a difference means the inputs have drifted).
  * **the patch** (cyan rectangle): the net's terrain input, taken at `t0_pose`.

Time: the file's `ostrich/t` and `preroll_t` stamp log row k at k * dt, but row k is the state
AFTER step k + 1. The browser adds one dt so that t = 0 is the window start and t = 1 s its end,
which are the times the twin's path is on. The twin's wheels spin at its COMMANDED speeds (it logs
no realized wheel speed through `run_twin`); that is cosmetic only.

Each row prints two lines: map, strategy, family, interaction and command, then e_pos / e_rot
(twin end vs ostrich end, `custom_dataset.se3_errors`), VALID / INVALID, the nominal end's
`endpoint_feasible`, and the twin's `twin_min_clearance` / `twin_max_residual`. `valid` is also
recomputed from the stored fields (finite, window displacement, patch overhang, as in
`generate_dataset.finish_map`), and a WARNING is printed if that disagrees with the file.

CLI parameters:
    --file PATH      dataset_mppi_*.h5 path (required)
    --id INT         row to open first (default: the first selected row)
    --sampling STR   only rows drawn by this strategy (ramp_up, ramp_down, edge, rotate_in_place,
                     uniform, targeted)
    --family STR     only rows of this command family (wide, straight, narrow, spin)
    --interact INT   only rows with this interact_dir (-1, 0, 1)
    --invalid        only rows with valid == False
    --sort KEY       order the selected rows: index (default), e_pos or e_rot (largest first)
    --speed FLOAT    playback speed multiplier (default 1.0 = real time)
    --loop           loop playback instead of freezing on the last frame
    --window-only    skip the settle and warm-up and play only the window
    --rot-display    rot (default) prints e_rot; rpy prints the (roll, pitch, yaw) error in degrees
    --device STR     warp device for the twin rerun (default cuda:0)
    --dry-run        print the selection and the first row's lines, no viewer (needs no DISPLAY)

Usage:
    python src/feasibility/mppi_learning/gl_dataset_browser.py --file outputs/dataset_mppi_<...>.h5 --dry-run
    python src/feasibility/mppi_learning/gl_dataset_browser.py --file outputs/dataset_mppi_<...>.h5
    python src/feasibility/mppi_learning/gl_dataset_browser.py --file outputs/dataset_mppi_<...>.h5 --sampling edge --sort e_pos
    python src/feasibility/mppi_learning/gl_dataset_browser.py --file outputs/dataset_mppi_<...>.h5 --family spin --speed 0.25 --loop
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
import time

import h5py
import newton
import numpy as np
import pyglet
import warp as wp

from feasibility.comparator.common import init_warp_device
from feasibility.comparator.provenance import terrain_from_h5
from feasibility.heightmap import HeightMapReader
from feasibility.lattice_learning.custom_dataset import pose_to_se3
from feasibility.lattice_learning.custom_dataset import rpy_errors
from feasibility.lattice_learning.custom_dataset import se3_errors
from feasibility.lattice_learning.patch import patch_overhangs
from feasibility.lattice_learning.patch import patch_spec_from_attrs
from feasibility.lattice_learning.patch import PatchSpec
from feasibility.mppi_learning.command import MPPI_DT
from feasibility.mppi_learning.twin import run_twin
from feasibility.replay.gl_replay import build_model
from feasibility.replay.gl_replay import integrate_wheel_angle
from feasibility.replay.gl_replay import interp_pose
from feasibility.replay.gl_replay import interp_series
from feasibility.replay.gl_replay import ROBOT_COLOR
from feasibility.replay.gl_replay import WHEEL_NAMES

TWIN = "hstack"  # build_model's name for the second robot, colored purple by gl_replay.ROBOT_COLOR
OSTRICH_PATH_COLOR = ROBOT_COLOR["ostrich"]
TWIN_PATH_COLOR = ROBOT_COLOR[TWIN]
WARMUP_PATH_COLOR = (0.6, 0.6, 0.6)
PATCH_COLOR = (0.2, 0.9, 1.0)
LINE_Z_OFFSET = 0.03  # m above the terrain, against z-fighting
TWIN_DISAGREE_TOL = 1e-3  # m / quaternion units: rerun end vs stored twin_pose

NEXT_KEYS = (pyglet.window.key.N, pyglet.window.key.RIGHT)
PREV_KEYS = (pyglet.window.key.P, pyglet.window.key.LEFT)

# terrain-center-relative framing, as in gl_replay_arc.py
CAMERA_YAW = 90.0
CAMERA_PITCH = -25.0
CAMERA_BACK = 0.7
CAMERA_UP = 0.4

SORT_KEYS = ("index", "e_pos", "e_rot")


def ground_polyline(terrain: HeightMapReader, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """[n, 3] points at (x, y), LINE_Z_OFFSET above the terrain under each."""
    z = np.asarray(terrain.sample(x, y), dtype=np.float64) + LINE_Z_OFFSET
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def patch_rectangle(terrain: HeightMapReader, spec: PatchSpec, pose: np.ndarray) -> np.ndarray:
    """[5, 3] closed outline of the patch's body-frame rectangle at `pose` (x, y, yaw)."""
    bx = np.array([spec.x_min, spec.x_max, spec.x_max, spec.x_min, spec.x_min])
    by = np.array([spec.y_min, spec.y_min, spec.y_max, spec.y_max, spec.y_min])
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return ground_polyline(terrain, pose[0] + c * bx - s * by, pose[1] + s * bx + c * by)


@dataclasses.dataclass
class Row:
    """One dataset row materialized to numpy, plus the twin's re-run path."""

    index: int
    attrs: dict
    map_index: int
    map_path: str
    category: str
    sampling: str
    family: str
    command: np.ndarray  # [4] v_mean, v_slope, wz_mean, wz_slope
    entry: np.ndarray  # [2] v, wz
    interact_dir: int
    relief: float
    ramp_deg: float
    ramp_s: float
    valid: bool
    endpoint_feasible: bool
    twin_min_clearance: float
    twin_max_residual: float
    origin_drift: float
    start: np.ndarray  # [3] where the twin started: t0_pose, or the nominal origin if that is not finite
    finite_start: bool
    twin_pose: np.ndarray  # [7] stored label pose
    terrain: HeightMapReader
    ostrich_pose: np.ndarray  # [T, 7] the window only
    play_t: np.ndarray  # [P] s, 0 = window start
    play_pose: np.ndarray  # [P, 7]
    wheel_theta: np.ndarray  # [P, 3]
    warmup_xy: np.ndarray  # [W, 2] ostrich's warm-up path (empty with --window-only)
    twin_t: np.ndarray  # [WINDOW_STEPS + 1] s
    twin_path: np.ndarray  # [WINDOW_STEPS + 1, 7] re-run
    twin_theta: np.ndarray  # [WINDOW_STEPS + 1, 3]


def load_row(path: pathlib.Path, i: int, device: str, window_only: bool = False) -> Row:
    with h5py.File(path, "r") as f:
        attrs = dict(f.attrs)
        dt = float(f["ostrich"].attrs["dt"])
        t = f["ostrich/t"][:].astype(np.float64) + dt
        pose = f["ostrich/pose"][:, i, :].astype(np.float64)
        wheel_qd = f["ostrich/wheel_qd"][:, i, :].astype(np.float64)
        pre_t = f["ostrich/preroll_t"][:].astype(np.float64) + dt
        pre_pose = f["ostrich/preroll_pose"][:, i, :].astype(np.float64)
        pre_qd = f["ostrich/preroll_wheel_qd"][:, i, :].astype(np.float64)
        settle = int(attrs["settle_steps"])
        warmup_xy = pre_pose[settle:, :2]
        if window_only:  # keep the window-start row so t = 0 is still on the playback
            pre_t, pre_pose, pre_qd, warmup_xy = pre_t[-1:], pre_pose[-1:], pre_qd[-1:], warmup_xy[:0]
        play_t, play_pose = np.concatenate([pre_t, t]), np.concatenate([pre_pose, pose])

        t0_pose, origin = f["t0_pose"][i].astype(np.float64), f["origin"][i].astype(np.float64)
        t0_wheels, t0_twist = f["t0_wheel_omega"][i], f["t0_twist"][i]
        finite_start = bool(np.isfinite(t0_pose).all() and np.isfinite(t0_wheels).all() and np.isfinite(t0_twist).all())
        omega = f["omega"][i].astype(np.float32)  # [WINDOW_STEPS, 3], MPPI's command, uncompensated
        families = str(attrs["families"]).split(",")
        terrain = terrain_from_h5(f, i)
        row = dict(
            index=i, attrs=attrs, map_index=int(f["map_index"][i]), map_path=f["map_path"].asstr()[i],
            category=f["map_category"].asstr()[i], sampling=f["sampling"].asstr()[i],
            family=families[int(f["family"][i])], command=f["command"][i].astype(np.float64),
            entry=f["entry"][i].astype(np.float64), interact_dir=int(f["interact_dir"][i]),
            relief=float(f["relief"][i]), ramp_deg=float(f["ramp_deg"][i]), ramp_s=float(f["ramp_s"][i]),
            valid=bool(f["valid"][i]), endpoint_feasible=bool(f["endpoint_feasible"][i]),
            twin_min_clearance=float(f["twin_min_clearance"][i]),
            twin_max_residual=float(f["twin_max_residual"][i]), origin_drift=float(f["origin_drift"][i]),
            twin_pose=f["twin_pose"][i].astype(np.float64),
        )

    # the twin exactly as finish_map ran it, a diverged warm-up included
    start = t0_pose if finite_start else origin
    twin_path = run_twin(
        terrain, start[None], omega[:, None],
        init_wheel_omega=(t0_wheels if finite_start else np.zeros(3, np.float32))[None],
        init_twist=(t0_twist if finite_start else np.zeros(3, np.float32))[None],
        mu=float(attrs["mu"]), k_turn=float(attrs["k_turn"]), device=device, every_step=True,
    )[:, 0].astype(np.float64)
    twin_t = np.arange(len(twin_path)) * MPPI_DT
    twin_qd = np.concatenate([np.where(finite_start, t0_wheels, 0.0)[None], omega]).astype(np.float64)
    return Row(
        **row, start=start, finite_start=finite_start, terrain=terrain, ostrich_pose=pose,
        play_t=play_t, play_pose=play_pose,
        wheel_theta=integrate_wheel_angle(play_t, np.concatenate([pre_qd, wheel_qd])),
        warmup_xy=warmup_xy, twin_t=twin_t, twin_path=twin_path,
        twin_theta=integrate_wheel_angle(twin_t, twin_qd),
    )


def label_errors(twin_pose: np.ndarray, ostrich_end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[n] e_pos, e_rot of the twin's end pose against ostrich's, [n, 7] each."""
    e_pos, e_rot = se3_errors(pose_to_se3(twin_pose), pose_to_se3(ostrich_end))
    return np.asarray(e_pos, np.float64), np.asarray(e_rot, np.float64)


def select_rows(path: pathlib.Path, args: argparse.Namespace) -> np.ndarray:
    """Row indices passing the filters, in `--sort` order."""
    with h5py.File(path, "r") as f:
        n = int(f.attrs["n"])
        keep = np.ones(n, bool)
        if args.sampling is not None:
            keep &= f["sampling"].asstr()[:] == args.sampling
        if args.family is not None:
            families = str(f.attrs["families"]).split(",")
            if args.family not in families:
                raise SystemExit(f"--family must be one of {families}, got {args.family!r}")
            keep &= f["family"][:] == families.index(args.family)
        if args.interact is not None:
            keep &= f["interact_dir"][:] == args.interact
        if args.invalid:
            keep &= ~f["valid"][:]
        e_pos, e_rot = label_errors(f["twin_pose"][:].astype(np.float64), f["ostrich/pose"][-1].astype(np.float64))
    rows = np.flatnonzero(keep)
    if args.sort != "index":
        key = np.nan_to_num({"e_pos": e_pos, "e_rot": e_rot}[args.sort][rows], nan=np.inf)
        rows = rows[np.argsort(-key, kind="stable")]
    return rows


def report_row(row: Row, position: int, n_selected: int, rot_display: str) -> None:
    ostrich_end = row.ostrich_pose[-1]
    e_pos, e_rot = (float(e[0]) for e in label_errors(row.twin_pose[None], ostrich_end[None]))
    if rot_display == "rpy":
        roll, pitch, yaw = (float(np.degrees(e)) for e in rpy_errors(pose_to_se3(row.twin_pose), pose_to_se3(ostrich_end)))
        rot = f"roll={roll:.1f} pitch={pitch:.1f} yaw={yaw:.1f} deg"
    else:
        rot = f"e_rot={e_rot:.3f} rad ({np.degrees(e_rot):.1f} deg)"
    v_mean, v_slope, wz_mean, wz_slope = row.command
    ramp = f" ramp {row.ramp_deg:.0f} deg s {row.ramp_s:+.2f} m" if np.isfinite(row.ramp_deg) else ""
    print(
        f"[{position + 1}/{n_selected}  row {row.index}]  {pathlib.Path(row.map_path).name} ({row.category})  "
        f"{row.sampling} / {row.family}  interact {row.interact_dir:+d} relief {row.relief:.2f} m{ramp}\n"
        f"    cmd v {v_mean:.2f}{v_slope:+.2f} wz {wz_mean:+.2f}{wz_slope:+.2f}  entry v {row.entry[0]:.2f} "
        f"wz {row.entry[1]:+.2f}  |  e_pos={e_pos:.3f} m  {rot}  "
        f"{'VALID' if row.valid else 'INVALID'}  endpoint={'OK' if row.endpoint_feasible else 'BLOCKED'}  "
        f"twin clearance {row.twin_min_clearance:.3f} residual {row.twin_max_residual:.1e}  "
        f"origin drift {row.origin_drift:.2f} m"
    )

    # generate_dataset.finish_map's `valid`, from the stored fields
    displacement = float(np.linalg.norm(ostrich_end[:2] - row.start[:2]))
    overhang = bool(patch_overhangs(row.terrain, row.start[None], patch_spec_from_attrs(row.attrs))[0])
    recomputed = (
        row.finite_start and bool(np.isfinite(ostrich_end).all()) and bool(np.isfinite(row.twin_pose).all())
        and displacement <= float(row.attrs["max_window_displacement"]) and not overhang
    )
    if recomputed != row.valid:
        print(f"  WARNING: recomputed valid={recomputed} (finite start {row.finite_start}, displacement "
              f"{displacement:.2f} m, patch overhang {overhang}) contradicts the file's valid={row.valid}")
    drift = np.abs(row.twin_path[-1] - row.twin_pose)
    if not np.all(drift <= TWIN_DISAGREE_TOL):
        print(f"  WARNING: the twin rerun ends {np.nanmax(drift):.2e} away from the stored twin_pose -- "
              f"mu/k_turn/solver/terrain differ from generation")


@dataclasses.dataclass
class Scene:
    """One map's Newton model plus the per-row buffers the render loop writes each frame."""

    model: newton.Model
    robots: dict[str, dict[str, int]]
    state: newton.State
    q_start: np.ndarray
    joint_q: np.ndarray
    joint_q_wp: wp.array
    joint_qd_wp: wp.array
    lines: dict[str, tuple[wp.array, wp.array, tuple[float, float, float]]] = dataclasses.field(default_factory=dict)


def build_scene(viewer: newton.viewer.ViewerGL, row: Row) -> Scene:
    model, robots = build_model(row.terrain, ("ostrich", TWIN))
    viewer.set_model(model)
    joint_q = model.joint_q.numpy().copy()
    scene = Scene(
        model=model, robots=robots, state=model.state(), q_start=model.joint_q_start.numpy(), joint_q=joint_q,
        joint_q_wp=wp.array(joint_q, dtype=wp.float32, device=model.device),
        joint_qd_wp=wp.zeros_like(model.joint_qd),
    )
    place_row(scene, row)
    return scene


def place_row(scene: Scene, row: Row) -> None:
    """Rebuild the row's guide lines; the robots are written every frame by the render loop."""
    polylines = {
        "/ostrich_path": (ground_polyline(row.terrain, row.ostrich_pose[:, 0], row.ostrich_pose[:, 1]), OSTRICH_PATH_COLOR),
        "/twin_path": (ground_polyline(row.terrain, row.twin_path[:, 0], row.twin_path[:, 1]), TWIN_PATH_COLOR),
        "/warmup_path": (ground_polyline(row.terrain, row.warmup_xy[:, 0], row.warmup_xy[:, 1]), WARMUP_PATH_COLOR),
        "/patch": (patch_rectangle(row.terrain, patch_spec_from_attrs(row.attrs), row.start), PATCH_COLOR),
    }
    device = scene.model.device
    scene.lines = {
        name: (wp.array(p[:-1], dtype=wp.vec3, device=device), wp.array(p[1:], dtype=wp.vec3, device=device), color)
        for name, (p, color) in polylines.items() if len(p) >= 2
    }


def write_robot(scene: Scene, name: str, pose: np.ndarray, theta: np.ndarray) -> None:
    joints = scene.robots[name]
    base = int(scene.q_start[joints["base"]])
    scene.joint_q[base : base + 7] = pose
    for wheel_name, angle in zip(WHEEL_NAMES, theta):
        scene.joint_q[scene.q_start[joints[wheel_name]]] = angle


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=pathlib.Path, required=True, help="dataset_mppi_*.h5 path")
    ap.add_argument("--id", type=int, default=None, help="row to open first (default: the first selected)")
    ap.add_argument("--sampling", type=str, default=None, help="only rows drawn by this strategy")
    ap.add_argument("--family", type=str, default=None, help="only rows of this command family")
    ap.add_argument("--interact", type=int, default=None, choices=(-1, 0, 1), help="only rows with this interact_dir")
    ap.add_argument("--invalid", action="store_true", help="only rows with valid == False")
    ap.add_argument("--sort", type=str, default="index", choices=SORT_KEYS, help="row order (errors: largest first)")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier (default 1.0 = real time)")
    ap.add_argument("--loop", action="store_true", help="loop playback instead of freezing on the last frame")
    ap.add_argument("--window-only", action="store_true", help="skip the settle and warm-up, play only the window")
    ap.add_argument("--rot-display", type=str, default="rot", choices=("rot", "rpy"), help="e_rot, or the (roll, pitch, yaw) error in degrees")
    ap.add_argument("--device", type=str, default="cuda:0", help="warp device for the twin rerun (default cuda:0)")
    ap.add_argument("--dry-run", action="store_true", help="print the selection and the first row, no viewer")
    args = ap.parse_args()

    rows = select_rows(args.file, args)
    if len(rows) == 0:
        raise SystemExit("no row passes the filters")
    if args.id is None:
        position = 0
    elif args.id in rows:
        position = int(np.flatnonzero(rows == args.id)[0])
    else:
        raise SystemExit(f"--id {args.id} is not among the {len(rows)} selected rows")
    print(f"{len(rows)} rows selected, sorted by {args.sort}")

    init_warp_device(args.device)
    row = load_row(args.file, int(rows[position]), args.device, args.window_only)
    report_row(row, position, len(rows), args.rot_display)
    if args.dry_run:
        print("[dry-run]  loaded + checked, nothing rendered")
        return

    viewer = newton.viewer.ViewerGL()
    pending = {"step": 0}  # key presses only record the step; the render loop switches rows

    def on_key_press(symbol: int, modifiers: int) -> None:
        if symbol in NEXT_KEYS:
            pending["step"] += 1
        elif symbol in PREV_KEYS:
            pending["step"] -= 1

    viewer.renderer.register_key_press(on_key_press)

    scene = build_scene(viewer, row)
    terrain = row.terrain
    cx = terrain.x0 + terrain.nx * terrain.cell / 2.0
    cy = terrain.y0 + terrain.ny * terrain.cell / 2.0
    extent = max(terrain.nx, terrain.ny) * terrain.cell
    viewer.set_camera(
        pos=wp.vec3(cx, cy - CAMERA_BACK * extent, terrain.max_z + CAMERA_UP * extent),
        pitch=CAMERA_PITCH, yaw=CAMERA_YAW,
    )
    print(
        "N/RIGHT next, P/LEFT previous, F re-frame, ESC quit  (orange = ostrich, purple = the twin "
        f"from ostrich's window start, grey = warm-up path, cyan = patch)  speed={args.speed}"
    )

    t0 = time.perf_counter()
    while viewer.is_running():
        if pending["step"]:
            position = (position + pending["step"]) % len(rows)
            pending["step"] = 0
            row_next = load_row(args.file, int(rows[position]), args.device, args.window_only)
            report_row(row_next, position, len(rows), args.rot_display)
            if row_next.map_index == row.map_index:
                place_row(scene, row_next)
            else:
                scene = build_scene(viewer, row_next)
            row = row_next
            t0 = time.perf_counter()

        t = row.play_t
        t_start, span = float(t[0]), float(t[-1] - t[0])
        real_t = (time.perf_counter() - t0) * args.speed
        sim_t = t_start + ((real_t % span) if (args.loop and span > 0) else min(real_t, span))

        write_robot(scene, "ostrich", interp_pose(t, row.play_pose, sim_t), interp_series(t, row.wheel_theta, sim_t))
        write_robot(scene, TWIN, interp_pose(row.twin_t, row.twin_path, sim_t), interp_series(row.twin_t, row.twin_theta, sim_t))
        scene.joint_q_wp.assign(scene.joint_q)

        newton.eval_fk(scene.model, scene.joint_q_wp, scene.joint_qd_wp, scene.state)
        contacts = scene.model.collide(scene.state)
        viewer.begin_frame(sim_t)
        viewer.log_state(scene.state)
        for name, (starts, ends, color) in scene.lines.items():
            viewer.log_lines(name, starts, ends, colors=color)
        viewer.log_contacts(contacts, scene.state)
        viewer.end_frame()
        wp.synchronize()


if __name__ == "__main__":
    main()
