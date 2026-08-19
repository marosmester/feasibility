"""Batch ostrich-vs-helhest_stack comparison on ostrich/examples/assets/surface.obj -- a single,
fixed hilly terrain (rasterized once by feasibility.heightmap.create_surface into
assets/surface/surface) -- replayed once in ostrich (dynamics) and once in helhest_stack
(kinematic twin) for each Trial in TRIALS below.

Unlike compare_speed_bumps.py/compare_box_obstacles.py, which hold spawn pose + command fixed
and batch across a heightmap SERIES, this scenario holds the terrain fixed and batches across
DIFFERENT initial conditions and control sequences (see comparator.common.Trial /
TrialScenarioSpec / run_trial_comparison) -- still serving the two simulators exactly the same
thing (spawn pose + commanded body twist) per trial, just varying that shared input across
trials instead of across terrain. Each trial probes a different feature of the surface:

  flat_baseline          -- flat ground, off the hill entirely; sanity check the two sims agree
                             with no terrain-driven divergence at all.
  climb_south_face       -- drive straight up the hill's southern face toward the summit
                             (z rises ~0.49 -> 1.36 m over the run): vertical-clearance /
                             pitch divergence, ostrich's real inertia vs the kinematic twin's
                             quasi-static settle.
  descend_from_summit    -- spawn near the summit (z~1.52 m) and drive down its steep east
                             face: descent, where ostrich's momentum can carry it further than
                             the kinematic twin's per-step quasi-static settle predicts.
  flank_traverse          -- drive across the hill's northern shoulder roughly along a contour
                             (undulating, not monotonic climb/descent): sustained roll/pitch
                             coupling from uneven ground.
  turn_in_place_on_slope -- spawn partway up the southern face and rotate in place (v=0):
                             yaw-rate tracking on an incline, no forward travel to help settle.

Surface extent is x,y in [-15.12, 15.12] m (see create_surface.py); every trial's spawn + travel
stays well inside that.

Usage:
    python src/feasibility/comparator/compare_on_surface.py                     # all TRIALS
    python src/feasibility/comparator/compare_on_surface.py +mu=0.5
    python src/feasibility/comparator/compare_on_surface.py +trials=[climb_south_face,descend_from_summit]
"""

from __future__ import annotations

import math

import hydra
from omegaconf import DictConfig

from feasibility.comparator.common import CONFIG_PATH
from feasibility.comparator.common import run_trial_comparison
from feasibility.comparator.common import Trial
from feasibility.comparator.common import TrialScenarioSpec
from feasibility.heightmap.create_surface import surface_path

# rad/s, shared by every turn-in-place trial -- same magnitude as compare_box_obstacles.py's
# WZ_DRIVE, just positive (no obstacle side to swing toward here).
WZ_TURN = 0.6

TRIALS = [
    Trial(label="flat_baseline", spawn_x=-13.0, spawn_y=-13.0, spawn_yaw=0.0, v_drive=1.0),
    Trial(label="climb_south_face", spawn_x=2.5, spawn_y=-11.0, spawn_yaw=math.pi / 2, v_drive=1.0),
    Trial(label="descend_from_summit", spawn_x=2.5, spawn_y=-2.5, spawn_yaw=0.0, v_drive=1.0),
    Trial(label="flank_traverse", spawn_x=-8.0, spawn_y=3.0, spawn_yaw=0.0, v_drive=1.0),
    Trial(label="turn_in_place_on_slope", spawn_x=2.5, spawn_y=-5.5, spawn_yaw=0.0, v_drive=0.0, wz_drive=WZ_TURN),
]


def spec() -> TrialScenarioSpec:
    return TrialScenarioSpec(
        name="on_surface",
        terrain_path=surface_path(),
        trials=TRIALS,
        obstacle_x=0.0,  # no single obstacle -- terrain center, roughly the hill's footprint
        duration_s=9.0,  # s, shared by every trial (see TrialScenarioSpec) -- long enough for
        # climb_south_face to reach the summit and turn_in_place_on_slope to complete a full
        # rotation (WZ_TURN * 9s ~= 309 deg)
    )


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest", version_base=None)
def main(cfg: DictConfig) -> None:
    run_trial_comparison(cfg, spec())


if __name__ == "__main__":
    main()
