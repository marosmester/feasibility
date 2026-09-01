"""Minimal Newton GL viewer for a saved HeightMapReader terrain (create_rough_terrain.py's
output, or any other assets/*/*.png+.yaml heightmap) -- no physics, no control, just the
terrain mesh plus a static Helhest Junior model parked at its center (for size reference)
displayed in Newton's interactive viewer so you can look at it and orbit/pan/zoom with the
mouse.

The robot is placed standing, not simulated: its base pose is set once so the wheel axles sit
at terrain_height(center) + WHEEL_RADIUS (chassis-local wheel z is 0, see
HelhestJuniorConfig.LEFT_WHEEL_POS etc.) -- exactly resting on the ground under a flat patch,
approximately resting elsewhere, close enough for a scale reference with no settle physics run.

CLI parameters:
    --file PATH   heightmap path stem, no extension (default: assets/rough/rough_seed0000)
    --no-robot    skip placing the Helhest Junior model, terrain only

Usage:
    python src/feasibility/heightmap/terrain_visualizer.py
    python src/feasibility/heightmap/terrain_visualizer.py --file assets/rough/rough_seed0007
    python src/feasibility/heightmap/terrain_visualizer.py --file assets/surface/surface
    python src/feasibility/heightmap/terrain_visualizer.py --no-robot
"""
from __future__ import annotations

import argparse
import pathlib

import newton
import warp as wp
from examples.helhest_junior.common import create_helhest_junior_model
from examples.helhest_junior.common import HelhestJuniorConfig
from ostrich.core.model_builder import OstrichModelBuilder

from feasibility.heightmap import HeightMapReader

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_FILE = REPO_ROOT / "assets" / "rough" / "rough_seed0000"

# Camera framing derived from the terrain's own footprint at load time -- yaw=90 faces +Y, same
# convention as replay/gl_replay.py's camera (see its comment re Newton's get_front()).
CAMERA_YAW = 90.0
CAMERA_PITCH = -25.0
CAMERA_BACK = 0.7  # camera distance behind terrain center, as a fraction of the larger extent
CAMERA_UP = 0.4  # camera height above max_z, as a fraction of the larger extent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=pathlib.Path, default=DEFAULT_FILE, help=f"heightmap path stem (default {DEFAULT_FILE})")
    ap.add_argument("--no-robot", action="store_true", help="skip placing the Helhest Junior model, terrain only")
    args = ap.parse_args()

    terrain = HeightMapReader.load(args.file)
    cx = terrain.x0 + terrain.nx * terrain.cell / 2.0
    cy = terrain.y0 + terrain.ny * terrain.cell / 2.0

    # create_helhest_junior_model's wheel joints use the "joint_dof_mode" custom attribute,
    # which only OstrichModelBuilder registers -- see replay/gl_replay.py's build_model.
    builder = OstrichModelBuilder()
    builder.add_shape_mesh(body=-1, mesh=terrain.to_ostrich_mesh(), cfg=newton.ModelBuilder.ShapeConfig(mu=0.8))
    if not args.no_robot:
        z = float(terrain.sample(cx, cy)) + HelhestJuniorConfig.WHEEL_RADIUS
        create_helhest_junior_model(builder, xform=wp.transform(wp.vec3(cx, cy, z), wp.quat_identity()))
    model = builder.finalize()

    extent = max(terrain.nx, terrain.ny) * terrain.cell
    camera_pos = wp.vec3(cx, cy - CAMERA_BACK * extent, terrain.max_z + CAMERA_UP * extent)

    wp.init()
    viewer = newton.viewer.ViewerGL()
    viewer.set_model(model)
    viewer.set_camera(pos=camera_pos, pitch=CAMERA_PITCH, yaw=CAMERA_YAW)
    state = model.state()
    # finalize()'s default joint_q only fixes body_q for shapes attached to body=-1 (the
    # terrain); the robot's free/revolute joints need one eval_fk to propagate joint_q -> body_q
    # so it renders at the xform we placed it at rather than at the world origin -- same call
    # replay/gl_replay.py makes every frame, just once here since nothing moves.
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    print(
        f"viewing {args.file}  "
        f"({terrain.nx}x{terrain.ny} cells, cell={terrain.cell} m, z in [{terrain.min_z:.3f}, {terrain.max_z:.3f}] m)"
    )

    while viewer.is_running():
        viewer.begin_frame(0.0)
        viewer.log_state(state)
        viewer.end_frame()


if __name__ == "__main__":
    main()
