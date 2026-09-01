"""Batch ostrich-vs-helhest_stack comparison on one shared terrain: TERRAIN_PATH (default the
fixed hilly mesh rasterized by feasibility.heightmap.create_surface into assets/surface/surface)
replayed once in ostrich (dynamics) and once in helhest_stack (kinematic twin) for each Trial
build_trials() derives below.

Unlike compare_speed_bumps.py/compare_box_obstacles.py, which hold spawn pose + command fixed
and batch across a heightmap SERIES, this scenario holds the terrain fixed and batches across
DIFFERENT initial conditions and control sequences (see comparator.common.Trial /
TrialScenarioSpec / run_trial_comparison) -- still serving the two simulators exactly the same
thing (spawn pose + commanded body twist) per trial, just varying that shared input across
trials instead of across terrain.

TERRAIN-MESH-AGNOSTIC BY DESIGN: build_trials() computes every spawn/heading purely from the
loaded terrain's own grid bounds and its single highest cell (see _TerrainFeatures) -- there is
no coordinate anywhere tuned to the specific hill mesh. "Highest cell" stands in for "the
interesting feature" on ANY terrain: a real summit on the hill mesh, or merely the tallest bump
on a heightmap.create_rough_terrain.py random field -- either way, "approach/stand on/pass by
the highest point" is a well-defined probe. Every reach (how far back a spawn is placed, how far
sideways a traverse offsets) is clipped by _reach() to whatever room the grid actually has inside
a MARGIN-m buffer from the boundary, so a small or oddly-shaped terrain degrades to a shorter
trial instead of spawning out of bounds -- fixes the previous version, whose five Trials were
absolute (x, y) coordinates tuned to this one 30x30 m hill and silently sampled
HeightMapReader.sample()'s clamped edge value (i.e. lost their meaning) on anything smaller or
differently shaped, e.g. create_rough_terrain.py's default 16x16 m grid.

  flat_baseline           -- long straight run from one corner toward the terrain center; the
                              sanity check that the two sims agree with minimal terrain-driven
                              divergence (exactly flat on the hill mesh's empty corners; on a
                              stationary rough field it's just ordinary background roughness,
                              still a useful baseline).
  climb_to_peak           -- approach the terrain's single highest cell head-on: vertical-
                              clearance / pitch divergence, ostrich's real inertia vs the
                              kinematic twin's quasi-static settle.
  descend_from_peak       -- spawn AT the highest cell, continue past it: descent, where
                              ostrich's momentum can carry it further than the kinematic twin's
                              per-step quasi-static settle predicts.
  flank_traverse          -- pass by the highest cell laterally, offset to one side, heading
                              perpendicular to the approach direction: sustained roll/pitch
                              coupling from uneven ground, no monotonic climb/descent.
  turn_in_place_on_slope  -- spawn partway up the approach to the peak and rotate in place
                              (v=0): yaw-rate tracking on an incline, no forward travel to help
                              settle.

CLI parameters:
    TERRAIN_PATH            positional, optional -- heightmap asset path stem (no extension),
                             e.g. assets/rough/rough_seed0000 from
                             heightmap/create_rough_terrain.py (default: the fixed hill,
                             heightmap.create_surface.surface_path()). Consumed before Hydra
                             parses the rest of argv (see _pop_terrain_path_arg), since Hydra's
                             own CLI grammar has no positional-argument concept.
    Remaining args are Hydra overrides read by comparator.common.run_trial_comparison (`+`
    prefix required since none of these exist in the base "helhest" config):
      +mu=FLOAT              ground friction coefficient (default: 0.8)
      +k_turn=FLOAT          helhest_stack ICR turning-rate gain (default: dynamics.K_TURN)
      +device=STR            helhest_stack torch/warp device (default: "cuda:0")
      +trials=[LABEL,...]    subset of build_trials()' labels to run, e.g. flat_baseline,
                             climb_to_peak, descend_from_peak, flank_traverse,
                             turn_in_place_on_slope (default: all trials)
    Also accepts any standard Hydra config-group override against the "helhest" base config
    (e.g. engine=mujoco, logging=..., simulation=...) -- see ostrich/examples/conf/helhest.yaml
    for the groups. rendering/num_worlds are forced by run_trial_comparison and cannot be
    overridden.

Usage:
    python src/feasibility/comparator/compare_on_surface.py                     # all trials, hill surface
    python src/feasibility/comparator/compare_on_surface.py +mu=0.5
    python src/feasibility/comparator/compare_on_surface.py +trials=[climb_to_peak,descend_from_peak]
    python src/feasibility/comparator/compare_on_surface.py assets/rough/rough_seed0000
"""

from __future__ import annotations

import math
import pathlib
import sys
from dataclasses import dataclass

import hydra
import numpy as np
from omegaconf import DictConfig

from feasibility.comparator.common import CONFIG_PATH
from feasibility.comparator.common import run_trial_comparison
from feasibility.comparator.common import Trial
from feasibility.comparator.common import TrialScenarioSpec
from feasibility.heightmap import HeightMapReader
from feasibility.heightmap.create_surface import surface_path

# rad/s, shared by every turn-in-place trial -- same magnitude as compare_box_obstacles.py's
# WZ_DRIVE, just positive (no obstacle side to swing toward here).
WZ_TURN = 0.6

V_DRIVE = 1.0  # m/s, shared forward speed for every driving trial
DURATION_S = 9.0  # s, shared by every trial (see TrialScenarioSpec) -- long enough for a v=1.0
# trial to cross most of a default-extent terrain and for turn_in_place to complete a full
# rotation (WZ_TURN * 9s ~= 309 deg)

# m, kept clear of the terrain grid's boundary by every spawn/reach computation below -- covers
# the robot's own footprint (0.75 m wheelbase, 0.73 m track) plus its ~1.1 m turning reach (see
# compare_box_obstacles.py's CORNER_MARGIN comment), so an in-place turn's wheels never swing
# past the grid edge into HeightMapReader.sample()'s clamped region.
MARGIN = 1.5


@dataclass(frozen=True)
class _TerrainFeatures:
    """Everything build_trials needs from a loaded terrain, geometry only -- no assumption
    about what the terrain looks like (hill, box, band-limited noise, ...), just its grid
    bounds and its single highest cell."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    center: tuple[float, float]
    peak: tuple[float, float]


def _terrain_features(terrain: HeightMapReader) -> _TerrainFeatures:
    iy, ix = np.unravel_index(np.argmax(terrain.H), terrain.H.shape)
    peak = (terrain.x0 + (ix + 0.5) * terrain.cell, terrain.y0 + (iy + 0.5) * terrain.cell)
    center = (terrain.x0 + terrain.nx * terrain.cell / 2.0, terrain.y0 + terrain.ny * terrain.cell / 2.0)
    return _TerrainFeatures(
        xmin=terrain.x0,
        xmax=terrain.x0 + terrain.nx * terrain.cell,
        ymin=terrain.y0,
        ymax=terrain.y0 + terrain.ny * terrain.cell,
        center=center,
        peak=peak,
    )


def _unit(dx: float, dy: float, fallback: tuple[float, float] = (1.0, 0.0)) -> tuple[float, float]:
    """Unit vector along (dx, dy), or `fallback` if it's ~zero (e.g. a perfectly flat terrain,
    where argmax ties at the terrain center and there is no meaningful "peak direction")."""
    norm = math.hypot(dx, dy)
    return (dx / norm, dy / norm) if norm > 1e-9 else fallback


def _reach(ox: float, oy: float, ux: float, uy: float, feats: _TerrainFeatures) -> float:
    """Distance from (ox, oy) to the nearest grid boundary (inset by MARGIN) along unit
    direction (ux, uy) -- a ray/box clip. Bounds how far a spawn can be pushed back along an
    approach direction. Floors at 0 rather than going negative, so a tiny or oddly-shaped
    terrain (or an origin already inside the margin) degrades to a zero-length approach --
    spawn right at the anchor -- instead of landing out of bounds."""
    xmin, xmax = feats.xmin + MARGIN, feats.xmax - MARGIN
    ymin, ymax = feats.ymin + MARGIN, feats.ymax - MARGIN
    candidates = []
    if ux > 1e-9:
        candidates.append((xmax - ox) / ux)
    elif ux < -1e-9:
        candidates.append((xmin - ox) / ux)
    if uy > 1e-9:
        candidates.append((ymax - oy) / uy)
    elif uy < -1e-9:
        candidates.append((ymin - oy) / uy)
    return max(0.0, min(candidates)) if candidates else 0.0


def build_trials(terrain: HeightMapReader) -> list[Trial]:
    """Five generic probes, positioned/aimed purely from `terrain`'s own bounds and highest
    cell -- see module docstring for what each one probes and _TerrainFeatures/_reach for how
    they stay in bounds on any terrain."""
    feats = _terrain_features(terrain)
    cx, cy = feats.center
    px, py = feats.peak
    ux, uy = _unit(px - cx, py - cy)
    yaw = math.atan2(uy, ux)
    perp_ux, perp_uy = -uy, ux
    perp_yaw = math.atan2(perp_uy, perp_ux)

    climb_reach = min(V_DRIVE * DURATION_S * 0.9, _reach(px, py, -ux, -uy, feats))
    climb_x, climb_y = px - climb_reach * ux, py - climb_reach * uy
    turn_x, turn_y = px - 0.5 * climb_reach * ux, py - 0.5 * climb_reach * uy

    flank_anchor_x, flank_anchor_y = cx + 0.5 * (px - cx), cy + 0.5 * (py - cy)  # midway center<->peak
    flank_reach = min(V_DRIVE * DURATION_S * 0.5, _reach(flank_anchor_x, flank_anchor_y, -perp_ux, -perp_uy, feats))
    flank_x, flank_y = flank_anchor_x - flank_reach * perp_ux, flank_anchor_y - flank_reach * perp_uy

    corner_x, corner_y = feats.xmin + MARGIN, feats.ymin + MARGIN
    baseline_yaw = math.atan2(cy - corner_y, cx - corner_x)

    return [
        Trial(label="flat_baseline", spawn_x=corner_x, spawn_y=corner_y, spawn_yaw=baseline_yaw, v_drive=V_DRIVE),
        Trial(label="climb_to_peak", spawn_x=climb_x, spawn_y=climb_y, spawn_yaw=yaw, v_drive=V_DRIVE),
        Trial(label="descend_from_peak", spawn_x=px, spawn_y=py, spawn_yaw=yaw, v_drive=V_DRIVE),
        Trial(label="flank_traverse", spawn_x=flank_x, spawn_y=flank_y, spawn_yaw=perp_yaw, v_drive=V_DRIVE),
        Trial(
            label="turn_in_place_on_slope",
            spawn_x=turn_x,
            spawn_y=turn_y,
            spawn_yaw=0.0,
            v_drive=0.0,
            wz_drive=WZ_TURN,
        ),
    ]


def spec(terrain_path: pathlib.Path | None = None) -> TrialScenarioSpec:
    path = terrain_path or surface_path()
    terrain = HeightMapReader.load(path)
    return TrialScenarioSpec(
        name="on_surface",
        terrain_path=path,
        trials=build_trials(terrain),
        obstacle_x=_terrain_features(terrain).center[0],  # camera anchor -- terrain center
        duration_s=DURATION_S,
    )


def _pop_terrain_path_arg() -> pathlib.Path | None:
    """Extracts the optional positional TERRAIN_PATH from sys.argv before Hydra gets to parse
    the rest -- Hydra's own CLI grammar is strictly `key=value` overrides / `--flag`s, so the
    first bare token (no '=', not a leading '-') is unambiguously ours to claim, never a valid
    Hydra override. Mutates sys.argv (pops the token out) so Hydra's own parser, which reads
    sys.argv again when `main()` is called below, never sees it."""
    for i, arg in enumerate(sys.argv[1:], start=1):
        if "=" not in arg and not arg.startswith("-"):
            return pathlib.Path(sys.argv.pop(i))
    return None


# Set from sys.argv by the __main__ guard below, before the @hydra.main-wrapped main() call
# re-parses sys.argv as Hydra overrides -- see _pop_terrain_path_arg's docstring for why this
# can't just be an extra parameter on main(cfg).
_terrain_path: pathlib.Path | None = None


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest", version_base=None)
def main(cfg: DictConfig) -> None:
    run_trial_comparison(cfg, spec(_terrain_path))


if __name__ == "__main__":
    _terrain_path = _pop_terrain_path_arg()
    main()
