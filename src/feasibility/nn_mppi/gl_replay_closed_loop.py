"""GL replay of a `closed_loop.py` run (`outputs/nn_mppi/<map>_<tag>.h5`): ostrich's logged poses of
the chosen worlds over the run's own terrain, in Newton's ViewerGL, on the real Helhest Junior mesh.

Pose-only playback, as `replay/gl_replay.py`: every frame writes each robot's logged chassis pose and
a wheel angle integrated from the logged wheel speeds into `joint_q`, then `newton.eval_fk`. No
physics is stepped, and ostrich's settle drop before the first replan is skipped (t = 0 is the first
replan). A world keeps its log after it stops (arrived, timeout, ...), where ostrich got zero wheel
speeds, so it rolls to a halt where the run ended it.

Each robot and its whole path (a line just above the terrain) take its arm's colour, the same
colours as `closed_loop.py`'s PNG; the goal's arrival circle is red. The first view shows every chosen
world at once. N / P (or RIGHT / LEFT) step to each world ALONE and back, R restarts the playback.
Every view prints its worlds' status, time and path length, and names them in the window title.

CLI parameters:
    --file PATH         a closed_loop.py h5 (required)
    --arms NAME [...]   arms to show, exactly as the run named them (default: every arm)
    --repeat INT        which repeat of each arm (default 0; -1 = all repeats)
    --speed FLOAT       playback speed multiplier (default 1.0 = real time)
    --loop              loop playback instead of freezing on the last frame
    --dry-run           load the file and print the worlds, no viewer (needs no DISPLAY)

Usage:
    python src/feasibility/nn_mppi/gl_replay_closed_loop.py --file outputs/nn_mppi/<map>_<tag>.h5
    python src/feasibility/nn_mppi/gl_replay_closed_loop.py --file outputs/nn_mppi/<map>_<tag>.h5 \\
        --arms "nnflat[300,300]" "nnflat[300,300]w3" --speed 2
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
from examples.helhest_junior.common import create_helhest_junior_model
from ostrich.core.model_builder import OstrichModelBuilder

from feasibility.heightmap import HeightMapReader
from feasibility.mppi_learning.gl_dataset_browser import ground_polyline
from feasibility.mppi_learning.gl_dataset_browser import NEXT_KEYS
from feasibility.mppi_learning.gl_dataset_browser import PREV_KEYS
from feasibility.nn_mppi.closed_loop import arm_colors
from feasibility.replay.gl_replay import integrate_wheel_angle
from feasibility.replay.gl_replay import interp_pose
from feasibility.replay.gl_replay import interp_series
from feasibility.replay.gl_replay import JOINTS_PER_ROBOT

RESTART_KEY = pyglet.window.key.R
GOAL_COLOR = (1.0, 0.1, 0.1)
# camera behind the start, looking along start -> goal from above
CAMERA_BACK = 4.0  # [m]
CAMERA_UP = 6.0  # [m] above the highest terrain
CAMERA_PITCH = -40.0


@dataclasses.dataclass
class Run:
    """A closed_loop.py h5, materialized to numpy."""

    terrain: HeightMapReader
    start: np.ndarray  # [3]
    goal: np.ndarray  # [2]
    reach_radius: float
    arm: np.ndarray  # [W] str
    repeat: np.ndarray  # [W]
    status: np.ndarray  # [W] str
    end_time: np.ndarray  # [W] s, when the run stopped the world
    t: np.ndarray  # [T] s per log row, 0 = the first replan
    pose: np.ndarray  # [T, W, 7] ostrich, non-finite rows replaced by the last finite one
    wheel_theta: np.ndarray  # [W, T, 3]
    end_row: np.ndarray  # [W] the log row where the run stopped the world
    first_row: int  # the first log row driven by MPPI (closed_loop.report's path starts here)


def load_run(path: pathlib.Path) -> Run:
    with h5py.File(path, "r") as f:
        attrs = dict(f.attrs)
        grid = f["terrain"]
        terrain = HeightMapReader(grid[:], tuple(grid.attrs["origin"]), float(grid.attrs["cell"]))
        pose = f["ostrich_pose"][:].astype(np.float64)
        wheel_qd = f["ostrich_wheel_qd"][:].astype(np.float64)
        arm, status = f["arm"].asstr()[:], f["status"].asstr()[:]
        repeat, end_frame = f["repeat"][:], f["end_frame"][:]
    settle, dt, mppi_dt = int(attrs["settle_steps"]), float(attrs["ostrich_dt"]), float(attrs["mppi_dt"])
    # log row k is the state after ostrich step k + 1; the first replan reads row settle - 1
    t = (np.arange(pose.shape[0]) + 1 - settle) * dt
    for k in range(1, pose.shape[0]):  # a diverged world freezes at its last finite pose
        bad = ~np.isfinite(pose[k]).all(axis=-1)
        pose[k, bad] = pose[k - 1, bad]
    steps_per_frame = int(round(mppi_dt / dt))
    end_row = np.minimum(settle + end_frame * steps_per_frame, pose.shape[0]) - 1
    wheel_theta = np.stack([integrate_wheel_angle(t, np.nan_to_num(wheel_qd[:, w])) for w in range(pose.shape[1])])
    return Run(
        terrain=terrain, start=np.asarray(attrs["start"], np.float64), goal=np.asarray(attrs["goal"], np.float64),
        reach_radius=float(attrs["reach_radius"]), arm=arm, repeat=repeat, status=status,
        end_time=end_frame * mppi_dt, t=t, pose=pose, wheel_theta=wheel_theta, end_row=end_row, first_row=settle,
    )


def world_name(run: Run, w: int) -> str:
    return f"{run.arm[w]} #{run.repeat[w]}"


def report(run: Run, worlds: list[int], label: str) -> None:
    print(f"\n{label}")
    for w in worlds:
        xy = run.pose[run.first_row : run.end_row[w] + 1, w, :2]
        path = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
        print(f"  {world_name(run, w):>22}  {run.status[w]:>9}  {run.end_time[w]:5.1f} s  {path:6.2f} m")


@dataclasses.dataclass
class Scene:
    """One view's Newton model (the terrain plus one robot per shown world) and its buffers."""

    worlds: list[int]
    model: newton.Model
    state: newton.State
    q_start: np.ndarray
    joint_q: np.ndarray
    joint_q_wp: wp.array
    joint_qd_wp: wp.array


def polyline_segments(points: np.ndarray, device: wp.context.Device) -> tuple[wp.array, wp.array]:
    return wp.array(points[:-1], dtype=wp.vec3, device=device), wp.array(points[1:], dtype=wp.vec3, device=device)


def build_scene(viewer: newton.viewer.ViewerGL, run: Run, mesh: newton.Mesh, worlds: list[int],
                colors: dict[str, tuple[float, float, float]]) -> Scene:
    # create_helhest_junior_model's joints use a custom attribute only OstrichModelBuilder declares
    builder = OstrichModelBuilder()
    builder.add_shape_mesh(body=-1, mesh=mesh, cfg=newton.ModelBuilder.ShapeConfig(mu=0.8))
    shapes = []
    for _ in worlds:
        first = builder.shape_count
        create_helhest_junior_model(builder, xform=wp.transform_identity())
        shapes.append((first, builder.shape_count))
    model = builder.finalize()
    shape_color = model.shape_color.numpy()
    for (first, last), w in zip(shapes, worlds):
        shape_color[first:last] = colors[run.arm[w]]
    model.shape_color.assign(shape_color)
    viewer.set_model(model)  # also drops the previous view's lines

    # the paths and the goal are static: ViewerGL keeps logged lines and redraws them every frame
    for w in worlds:
        xy = run.pose[run.first_row : run.end_row[w] + 1, w, :2]
        if len(xy) >= 2:
            viewer.log_lines(f"/path_{w}", *polyline_segments(ground_polyline(run.terrain, xy[:, 0], xy[:, 1]), model.device),
                             colors=colors[run.arm[w]])
    angle = np.linspace(0.0, 2.0 * np.pi, 33)
    circle = ground_polyline(run.terrain, run.goal[0] + run.reach_radius * np.cos(angle),
                             run.goal[1] + run.reach_radius * np.sin(angle))
    viewer.log_lines("/goal", *polyline_segments(circle, model.device), colors=GOAL_COLOR)
    joint_q = model.joint_q.numpy().copy()
    return Scene(worlds=worlds, model=model, state=model.state(), q_start=model.joint_q_start.numpy(), joint_q=joint_q,
                 joint_q_wp=wp.array(joint_q, dtype=wp.float32, device=model.device),
                 joint_qd_wp=wp.zeros_like(model.joint_qd))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=pathlib.Path, required=True, help="a closed_loop.py h5")
    ap.add_argument("--arms", type=str, nargs="+", default=None, help="arms to show (default: every arm)")
    ap.add_argument("--repeat", type=int, default=0, help="which repeat of each arm (-1 = all)")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier (default 1.0 = real time)")
    ap.add_argument("--loop", action="store_true", help="loop playback instead of freezing on the last frame")
    ap.add_argument("--dry-run", action="store_true", help="print the worlds, no viewer")
    args = ap.parse_args()

    run = load_run(args.file)
    arms = list(dict.fromkeys(run.arm))
    colors = dict(zip(arms, arm_colors(arms)))
    if args.arms is not None:
        unknown = sorted(set(args.arms) - set(arms))
        if unknown:
            raise SystemExit(f"--arms {unknown} not in the run; its arms: {arms}")
        arms = [a for a in arms if a in args.arms]
    shown = [w for w in range(len(run.arm)) if run.arm[w] in arms and args.repeat in (-1, run.repeat[w])]
    if not shown:
        raise SystemExit(f"no world is repeat {args.repeat} of {arms}")
    # view 0: every shown world; view k: shown world k - 1 alone
    views = [shown] + [[w] for w in shown]
    labels = [f"{args.file.stem}: {len(shown)} worlds"] + [f"{args.file.stem}: {world_name(run, w)}" for w in shown]
    report(run, shown, labels[0])
    if args.dry_run:
        print("[dry-run]  loaded, nothing rendered")
        return

    wp.init()
    viewer = newton.viewer.ViewerGL()
    pending = {"step": 0, "restart": False}  # key presses only record; the render loop rebuilds

    def on_key_press(symbol: int, modifiers: int) -> None:
        if symbol in NEXT_KEYS:
            pending["step"] += 1
        elif symbol in PREV_KEYS:
            pending["step"] -= 1
        elif symbol == RESTART_KEY:
            pending["restart"] = True

    viewer.renderer.register_key_press(on_key_press)
    mesh = run.terrain.to_ostrich_mesh()
    view = 0
    scene = build_scene(viewer, run, mesh, views[view], colors)
    viewer.renderer.set_title(labels[view])
    heading = np.arctan2(run.goal[1] - run.start[1], run.goal[0] - run.start[0])
    viewer.set_camera(
        pos=wp.vec3(run.start[0] - CAMERA_BACK * np.cos(heading), run.start[1] - CAMERA_BACK * np.sin(heading),
                    run.terrain.max_z + CAMERA_UP),
        pitch=CAMERA_PITCH, yaw=float(np.degrees(heading)),
    )
    print(f"\nN/RIGHT next view, P/LEFT previous, R restart, F re-frame, ESC quit  "
          f"({len(views)} views: all, then each world alone)  speed={args.speed}")

    t_end = float(run.t[-1])
    t0 = time.perf_counter()
    while viewer.is_running():
        if pending["step"]:
            view = (view + pending["step"]) % len(views)
            pending["step"] = 0
            scene = build_scene(viewer, run, mesh, views[view], colors)
            viewer.renderer.set_title(labels[view])
            report(run, views[view], labels[view])
            t0 = time.perf_counter()
        if pending["restart"]:
            pending["restart"] = False
            t0 = time.perf_counter()

        real_t = (time.perf_counter() - t0) * args.speed
        sim_t = (real_t % t_end) if (args.loop and t_end > 0) else min(real_t, t_end)
        for i, w in enumerate(scene.worlds):
            joints = i * JOINTS_PER_ROBOT  # base, then left/right/rear wheel
            base = int(scene.q_start[joints])
            scene.joint_q[base : base + 7] = interp_pose(run.t, run.pose[:, w], sim_t)
            for k, angle in enumerate(interp_series(run.t, run.wheel_theta[w], sim_t)):
                scene.joint_q[scene.q_start[joints + 1 + k]] = angle
        scene.joint_q_wp.assign(scene.joint_q)

        newton.eval_fk(scene.model, scene.joint_q_wp, scene.joint_qd_wp, scene.state)
        viewer.begin_frame(sim_t)
        viewer.log_state(scene.state)
        viewer.end_frame()
        wp.synchronize()


if __name__ == "__main__":
    main()
