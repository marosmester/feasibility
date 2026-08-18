"""Batch ostrich-vs-helhest_stack comparison: the SAME straight-line approach, replayed once in
ostrich (dynamics) and once in helhest_stack (kinematic twin) on each heightmap in the
speed-bump series from feasibility.heightmap.create_speed_bumps, on the same initial condition
as demos/ostrich_vel_cmd.py / demos/hstack_vel_cmd.py.

Variants differ only in which heightmap they run on -- the bump heights from
create_speed_bumps.BUMP_HEIGHTS -- not in the command. The robot always spawns upstream of the
bump on its centerline (BUMP_X0 - spawn_back), facing +X, i.e. perpendicular to the bump (which
spans the full Y width), and drives straight at it. The actual sweep/rollout/npz-writing logic
lives in comparator.common.run_comparison, shared with every other compare_*.py scenario driver
(e.g. compare_box_obstacles.py) -- this file only describes the scenario via a ScenarioSpec.

Usage:
    python -m feasibility.comparator.compare_speed_bumps                # all BUMP_HEIGHTS
    python -m feasibility.comparator.compare_speed_bumps +mu=0.5 +k_turn=1.0
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from feasibility.comparator.common import CONFIG_PATH
from feasibility.comparator.common import run_comparison
from feasibility.comparator.common import ScenarioSpec
from feasibility.heightmap.create_speed_bumps import BUMP_X0
from feasibility.heightmap.create_speed_bumps import speed_bump_paths


def spec() -> ScenarioSpec:
    return ScenarioSpec(
        name="speed_bumps",
        variants=speed_bump_paths(),
        obstacle_x=BUMP_X0,
        value_name="bump_height",
        value_header="bump h [m]",
        label_fmt="bump_h={:.2f}m",
    )


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest", version_base=None)
def main(cfg: DictConfig) -> None:
    run_comparison(cfg, spec())


if __name__ == "__main__":
    main()
