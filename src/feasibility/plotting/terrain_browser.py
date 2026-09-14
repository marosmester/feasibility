"""Newton GL viewer that steps through every heightmap in a directory -- terrain_visualizer.py
for a whole batch (e.g. heightmap/create_maps_for_lattice_learning.py's assets/lattice_maps/<seed>/)
instead of one file. No physics: each map is shown with a static Helhest Junior parked at its
center for scale, placed exactly as terrain_visualizer.py places it.

Keys (besides Newton's own mouse/WASD camera controls):
    RIGHT / N     next map (wraps around)
    LEFT  / P     previous map (wraps around)

LEFT/RIGHT are also bound to the viewer's camera strafe while held, so a tap nudges the camera a
little; N/P switch maps without that. The camera is kept across switches so maps of the same
extent can be compared from one viewpoint (press F to re-frame). The current map's name is shown
in the window title and printed with its height range.

Switching rebuilds the model and hands it to ViewerGL.set_model, which discards the previous
model's render state itself. The rebuild happens in the render loop, not inside the key callback,
so it never runs in the middle of pyglet's event dispatch.

CLI parameters:
    maps_dir      directory holding <stem>.png + <stem>.yaml heightmaps (all *.png, sorted by name)
    --start NAME  stem or index to open first (default: the first map)
    --no-robot    skip placing the Helhest Junior model, terrain only

Usage:
    python src/feasibility/plotting/terrain_browser.py assets/lattice_maps/0
    python src/feasibility/plotting/terrain_browser.py assets/lattice_maps/0 --start ramps_i0003
    python src/feasibility/plotting/terrain_browser.py assets/large_box_random/0 --no-robot
"""
from __future__ import annotations

import argparse
import pathlib

import newton
import pyglet
import warp as wp
from examples.helhest_junior.common import create_helhest_junior_model
from examples.helhest_junior.common import HelhestJuniorConfig
from ostrich.core.model_builder import OstrichModelBuilder

from feasibility.heightmap import HeightMapReader
from feasibility.plotting.terrain_visualizer import CAMERA_BACK
from feasibility.plotting.terrain_visualizer import CAMERA_PITCH
from feasibility.plotting.terrain_visualizer import CAMERA_UP
from feasibility.plotting.terrain_visualizer import CAMERA_YAW

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

NEXT_KEYS = (pyglet.window.key.RIGHT, pyglet.window.key.N)
PREV_KEYS = (pyglet.window.key.LEFT, pyglet.window.key.P)


def build_scene(terrain: HeightMapReader, robot: bool) -> tuple[newton.Model, newton.State]:
    """Terrain mesh plus an optional parked robot, with body_q already propagated."""
    cx = terrain.x0 + terrain.nx * terrain.cell / 2.0
    cy = terrain.y0 + terrain.ny * terrain.cell / 2.0
    # OstrichModelBuilder, not newton's: the wheel joints need its "joint_dof_mode" attribute.
    builder = OstrichModelBuilder()
    builder.add_shape_mesh(body=-1, mesh=terrain.to_ostrich_mesh(), cfg=newton.ModelBuilder.ShapeConfig(mu=0.8))
    if robot:
        z = float(terrain.sample(cx, cy)) + HelhestJuniorConfig.WHEEL_RADIUS
        create_helhest_junior_model(builder, xform=wp.transform(wp.vec3(cx, cy, z), wp.quat_identity()))
    model = builder.finalize()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    return model, state


def resolve_start(stems: list[pathlib.Path], start: str | None) -> int:
    if start is None:
        return 0
    if start.isdigit():
        return int(start) % len(stems)
    names = [s.name for s in stems]
    if start not in names:
        raise SystemExit(f"--start {start!r} not found in the directory")
    return names.index(start)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("maps_dir", type=pathlib.Path, help="directory of <stem>.png + <stem>.yaml heightmaps")
    ap.add_argument("--start", default=None, help="map stem or index to open first")
    ap.add_argument("--no-robot", action="store_true", help="skip placing the Helhest Junior model, terrain only")
    args = ap.parse_args()

    maps_dir = args.maps_dir if args.maps_dir.is_absolute() else REPO_ROOT / args.maps_dir
    stems = [p.with_suffix("") for p in sorted(maps_dir.glob("*.png"))]
    if not stems:
        raise SystemExit(f"no *.png heightmaps in {maps_dir}")
    index = resolve_start(stems, args.start)

    wp.init()
    viewer = newton.viewer.ViewerGL()

    # Key presses only record the requested step; the loop below does the rebuild.
    pending = {"step": 0}

    def on_key_press(symbol: int, modifiers: int) -> None:
        if symbol in NEXT_KEYS:
            pending["step"] += 1
        elif symbol in PREV_KEYS:
            pending["step"] -= 1

    viewer.renderer.register_key_press(on_key_press)

    def load(i: int) -> tuple[HeightMapReader, newton.State]:
        terrain = HeightMapReader.load(stems[i])
        model, state = build_scene(terrain, robot=not args.no_robot)
        viewer.set_model(model)
        label = f"[{i + 1}/{len(stems)}] {stems[i].name}"
        viewer.renderer.set_title(label)
        print(
            f"{label}  ({terrain.nx}x{terrain.ny} cells, cell={terrain.cell} m, "
            f"z in [{terrain.min_z:.3f}, {terrain.max_z:.3f}] m)"
        )
        return terrain, state

    first, state = load(index)
    extent = max(first.nx, first.ny) * first.cell
    cx = first.x0 + first.nx * first.cell / 2.0
    cy = first.y0 + first.ny * first.cell / 2.0
    viewer.set_camera(
        pos=wp.vec3(cx, cy - CAMERA_BACK * extent, first.max_z + CAMERA_UP * extent),
        pitch=CAMERA_PITCH,
        yaw=CAMERA_YAW,
    )
    print(f"{len(stems)} maps in {maps_dir} -- RIGHT/N next, LEFT/P previous, F re-frame, ESC quit")

    while viewer.is_running():
        if pending["step"]:
            index = (index + pending["step"]) % len(stems)
            pending["step"] = 0
            _, state = load(index)
        viewer.begin_frame(0.0)
        viewer.log_state(state)
        viewer.end_frame()


if __name__ == "__main__":
    main()
