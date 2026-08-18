"""Batch ostrich-vs-helhest_stack comparison: turn the robot IN PLACE beside a box obstacle so
its rear wheel/body sweeps into the box's side face, replayed once in ostrich (dynamics) and
once in helhest_stack (kinematic twin) on each heightmap in the box-obstacle series from
feasibility.heightmap.create_box_obstacles.

Unlike compare_speed_bumps.py (drive straight over an obstacle -- vertical clearance), this
probes LATERAL body collision, which helhest_stack cannot represent at all: its quasi-static
settle only resolves vertical support against the heightmap (chassis high-centering is
detection-only by design, see helhest_stack's README "Known limitations"), so it has no
horizontal collision response and will happily keep rotating through the box. Ostrich, with real
rigid-body contact, is expected to physically jam against taller boxes. The gap between the two
net-yaw columns in run_comparison's printed summary IS the measurement.

Geometry (Helhest Junior body frame, origin at the front-wheel axle -- see
ostrich/examples/helhest_junior/common.py's HelhestJuniorConfig): front wheels at X=0, Y=+-0.365;
rear wheel at X=-0.75. A turn-in-place command (v=0, opposite signs on the two front wheels) is a
pure rotation about the front-axle midpoint, and the REAR wheel is the outermost point on that
arc (radius 1.101 m, vs 0.858 m for the rear chassis corner and 0.543 m for a front wheel) -- it
is what strikes first and hardest.

The robot spawns beside the box (SPAWN_Y = BOX_CY - SIDE_OFFSET, i.e. offset from the box's -Y
flank, X aligned with the box's center), facing +X, and is commanded a constant negative yaw
rate (clockwise) for DURATION_S. At rest the front wheel -- the closest point at spawn -- clears
even the tallest box's ramp foot by ~0.22 m; rotating clockwise swings the rear wheel 0.25 m INTO
the box at the tallest height, with first contact at ~48 deg of commanded yaw. The front wheels
swing away from the box as yaw increases, so the rear is the only contact point throughout.
DURATION_S=10s at WZ_DRIVE=-0.8 rad/s is ~458 deg unobstructed -- more than a full turn, so an
unblocked run (the shortest box, h=0.10, acts as that control) completes its rotation with room
to spare, while a blocked run visibly falls short.

`cmd_to_wheels(v=0, wz=WZ_DRIVE)` drives the two front wheels at opposite signs and the rear
wheel at exactly 0 -- "opposite commands to front wheels" with the rear simply dragged along,
per the scenario request.

Usage:
    python -m feasibility.comparator.compare_box_obstacles                # all BOX_HEIGHTS
    python -m feasibility.comparator.compare_box_obstacles +mu=0.5
    python -m feasibility.comparator.compare_box_obstacles +heights=[0.2,0.5,0.8]  # subset, faster
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from feasibility.comparator.common import CONFIG_PATH
from feasibility.comparator.common import run_comparison
from feasibility.comparator.common import ScenarioSpec
from feasibility.heightmap.create_box_obstacles import BOX_CX
from feasibility.heightmap.create_box_obstacles import BOX_CY
from feasibility.heightmap.create_box_obstacles import BOX_X0
from feasibility.heightmap.create_box_obstacles import box_obstacle_paths

# m, gap from the spawn pivot to the box's Y centerline -- see module docstring for the resulting
# clearances/contact angle. Not folded into ScenarioSpec: it's specific to this scenario's
# "beside the box" placement, unlike compare_speed_bumps.py's simple "upstream of it".
SIDE_OFFSET = 1.6


def spec() -> ScenarioSpec:
    return ScenarioSpec(
        name="box_obstacles",
        variants=box_obstacle_paths(),
        obstacle_x=BOX_X0,
        value_name="box_height",
        value_header="box h [m]",
        label_fmt="box_h={:.2f}m",
        spawn_x=BOX_CX,  # aligned with the box's X center
        spawn_y=BOX_CY - SIDE_OFFSET,  # beside its -Y flank
        spawn_yaw=0.0,  # facing +X, i.e. along the box's side
        v_drive=0.0,
        wz_drive=-0.8,  # turn in place, clockwise -> rear wheel swings at the box
        duration_s=10.0,  # ~458 deg unobstructed at 0.8 rad/s -- more than a full turn
    )


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest", version_base=None)
def main(cfg: DictConfig) -> None:
    run_comparison(cfg, spec())


if __name__ == "__main__":
    main()
