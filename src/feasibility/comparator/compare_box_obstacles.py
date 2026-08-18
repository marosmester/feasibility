"""Batch ostrich-vs-helhest_stack comparison: the SAME straight-line approach, replayed once in
ostrich (dynamics) and once in helhest_stack (kinematic twin) on each heightmap in the
box-obstacle series from feasibility.heightmap.create_box_obstacles.

Variants differ only in which heightmap they run on -- the box heights from
create_box_obstacles.BOX_HEIGHTS -- not in the command. The robot spawns upstream of the box on
its centerline (BOX_X0 - spawn_back), facing +X, and drives straight at it. Unlike the speed
bump (which spans the full Y width and has negligible X extent), the box has a real 1.5 m X
footprint, so `drive_s` is longer here to clear it -- see ScenarioSpec below. The actual
sweep/rollout/npz-writing logic lives in comparator.common.run_comparison, shared with
compare_speed_bumps.py and any other scenario driver -- this file only describes the scenario.

Usage:
    python -m feasibility.comparator.compare_box_obstacles                # all BOX_HEIGHTS
    python -m feasibility.comparator.compare_box_obstacles +mu=0.5 +k_turn=1.0
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from feasibility.comparator.common import CONFIG_PATH
from feasibility.comparator.common import run_comparison
from feasibility.comparator.common import ScenarioSpec
from feasibility.heightmap.create_box_obstacles import BOX_X0
from feasibility.heightmap.create_box_obstacles import box_obstacle_paths


def spec() -> ScenarioSpec:
    return ScenarioSpec(
        name="box_obstacles",
        variants=box_obstacle_paths(),
        obstacle_x=BOX_X0,
        value_name="box_height",
        value_header="box h [m]",
        label_fmt="box_h={:.2f}m",
        spawn_back=3.0,  # spawn_x = BOX_X0 - 3.0 = -1.0, on flat ground even for the tallest box
        drive_s=7.0,  # -> ends at spawn_x + 7.0 = 6.0, past the box's far ramp end at every height
    )


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest", version_base=None)
def main(cfg: DictConfig) -> None:
    run_comparison(cfg, spec())


if __name__ == "__main__":
    main()
