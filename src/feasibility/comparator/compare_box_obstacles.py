"""Batch ostrich-vs-helhest_stack comparison: turn the robot IN PLACE right next to a box
obstacle's corner so its rear wheel sweeps into the box's side face, replayed once in ostrich
(dynamics) and once in helhest_stack (kinematic twin) on each heightmap in the box-obstacle
series from feasibility.heightmap.create_box_obstacles.

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
pure rotation about the front-axle midpoint. The rear wheel's axle sits 1.101 m from that pivot
(vs 0.858 m for the rear chassis corner, 0.543 m for a front wheel), and the wheel itself is a
further ~0.35 m radius disc riding at ground level on top of that -- its rim's effective reach is
well beyond any bare chassis point, so it is expected to be what actually contacts a ground-level
obstacle first, ahead of the elevated chassis body.

The robot spawns diagonally near the box's SE corner (CORNER_MARGIN beyond
(BOX_X0+BOX_SIZE, BOX_CY-BOX_SIZE/2) in both X and Y), facing +X, and is commanded a constant
negative yaw rate (clockwise) for DURATION_S. This replaces an earlier version of this scenario
that centered the spawn on the box's -Y side with a large offset (1.6 m) -- that placement left
~35-48 deg of free rotation before any contact; parking right at the corner instead means even
the shortest box is reached within ~10-25 deg, and taller boxes (whose ramp reaches further
outward) within single digits, matching a request to make contact with a high obstacle near-
certain rather than dependent on exactly how wide that obstacle's ramp bevel is. The front wheels
stay clear of the box/ramp footprint at spawn for every height in the series (by construction of
CORNER_MARGIN, see spec() below), and swing further away as CW rotation proceeds, so they never
become the contact point.

`cmd_to_wheels(v=0, wz=WZ_DRIVE)` drives the two front wheels at opposite signs and the rear
wheel at exactly 0 -- "opposite commands to front wheels" with the rear simply dragged along,
per the scenario request.

CLI parameters (Hydra overrides read by comparator.common.run_comparison; `+` prefix required
since none of these exist in the base "helhest" config):
    +mu=FLOAT              ground friction coefficient (default: 0.8)
    +k_turn=FLOAT          helhest_stack ICR turning-rate gain (default: dynamics.K_TURN)
    +device=STR            helhest_stack torch/warp device (default: "cuda:0")
    +heights=[F,F,...]     subset of create_box_obstacles.BOX_HEIGHTS to run, values must match
                           an already-generated asset to 2 decimals (default: full series)
    Also accepts any standard Hydra config-group override against the "helhest" base config
    (e.g. engine=mujoco, logging=..., simulation=...) -- see ostrich/examples/conf/helhest.yaml
    for the groups. rendering/num_worlds are forced by run_comparison and cannot be overridden.

Usage:
    python src/feasibility/comparator/compare_box_obstacles.py                # all BOX_HEIGHTS
    python src/feasibility/comparator/compare_box_obstacles.py +mu=0.5
    python src/feasibility/comparator/compare_box_obstacles.py +heights=[0.2,0.5,0.8]  # subset, faster
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from feasibility.comparator.common import CONFIG_PATH
from feasibility.comparator.common import run_comparison
from feasibility.comparator.common import ScenarioSpec
from feasibility.heightmap.create_box_obstacles import BOX_CY
from feasibility.heightmap.create_box_obstacles import BOX_SIZE
from feasibility.heightmap.create_box_obstacles import BOX_X0
from feasibility.heightmap.create_box_obstacles import box_obstacle_paths

# m, diagonal clearance from the spawn pivot to the box's SE corner (BOX_X0+BOX_SIZE,
# BOX_CY-BOX_SIZE/2), added to both X and Y. Chosen so the front wheels still clear the
# widest ramp bevel in the series (the tallest box, h=0.80, whose 75deg bevel is ~0.22m wide)
# by a small margin at spawn -- see module docstring for the resulting contact angles.
CORNER_MARGIN = 0.35


def spec() -> ScenarioSpec:
    return ScenarioSpec(
        name="box_obstacles",
        variants=box_obstacle_paths(),
        obstacle_x=BOX_X0,
        value_name="box_height",
        value_header="box h [m]",
        label_fmt="box_h={:.2f}m",
        spawn_x=BOX_X0 + BOX_SIZE + CORNER_MARGIN,  # just past the box's far (+X) edge
        spawn_y=BOX_CY - BOX_SIZE / 2 - CORNER_MARGIN,  # just past its near (-Y) edge
        spawn_yaw=0.0,  # facing +X
        v_drive=0.0,
        wz_drive=-0.8,  # turn in place, clockwise -> rear wheel swings toward the corner
        duration_s=10.0,  # plenty of runway to observe the jam once contact happens
    )


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest", version_base=None)
def main(cfg: DictConfig) -> None:
    run_comparison(cfg, spec())


if __name__ == "__main__":
    main()
