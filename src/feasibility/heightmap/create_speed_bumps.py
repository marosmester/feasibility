"""Generate a series of speed-bump heightmaps: flat ground (z=0) everywhere except one
box-shaped bump spanning the full Y width of the grid, oriented perpendicular to a robot
driving straight along +X -- i.e. a real speed bump, not an angled ramp. The heightmaps
in the series differ ONLY in the bump's z height; grid extent, cell size, and the bump's
X position/width are shared. Used by feasibility.comparator.batch_compare to probe how
ostrich (dynamics) vs helhest_stack (kinematic twin) diverge as the bump grows from
negligible to significant relative to the wheel radius (0.35 m).

Usage:
    python -m feasibility.heightmap.create_speed_bumps
"""
from __future__ import annotations

import pathlib

import numpy as np

from feasibility.heightmap import HeightMapReader

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
ASSETS_DIR = REPO_ROOT / "assets"

# Grid extent/resolution -- matches HeightMapReader.flat()'s own default, so a spawn
# point tuned against that flat default lands in the same place here.
XLIM = (-2.0, 6.0)
YLIM = (-3.0, 3.0)
CELL = 0.05

BUMP_X0 = 2.0  # m, leading (near) edge of the bump
BUMP_WIDTH = 0.4  # m, extent along X (direction of travel)

BUMP_HEIGHTS =  (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70)  # m, one heightmap per height


def bump_path(height: float) -> pathlib.Path:
    """assets/speed_bump_h<height, cm, no dot> -- no extension, HeightMapReader appends
    .png/.yaml via pathlib's with_suffix(), which treats the LAST '.' in the name as an
    extension separator. A dotted decimal like "h0.20" would collide "h0.20" and "h0.30"
    onto the same "h0.{yaml,png}" (both get stem "h0", suffix replaced wholesale) -- cm
    integers side-step that."""
    return ASSETS_DIR / f"speed_bump_h{round(height * 100):03d}cm"


def speed_bump_paths(heights: tuple[float, ...] = BUMP_HEIGHTS) -> list[tuple[float, pathlib.Path]]:
    """(height, path) pairs for the series -- the single source of truth batch_compare
    reads back, so the two stay in sync without duplicating the height list."""
    return [(h, bump_path(h)) for h in heights]


def build_speed_bump(height: float) -> HeightMapReader:
    """Flat ground except one box at X in [BUMP_X0, BUMP_X0+BUMP_WIDTH), spanning the
    full Y range -- perpendicular to a robot driving straight along +X, so any Y offset
    within the grid hits the bump square-on rather than at an angle."""
    nx = int(round((XLIM[1] - XLIM[0]) / CELL)) + 1
    ny = int(round((YLIM[1] - YLIM[0]) / CELL)) + 1
    H = np.zeros((ny, nx), dtype=np.float64)
    xs = XLIM[0] + (np.arange(nx) + 0.5) * CELL
    H[:, (xs >= BUMP_X0) & (xs < BUMP_X0 + BUMP_WIDTH)] = height
    return HeightMapReader(H, origin=(XLIM[0], YLIM[0]), cell=CELL)


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for height, path in speed_bump_paths():
        build_speed_bump(height).save(path)
        print(f"saved {path}.png / {path}.yaml  (bump height {height:.2f} m)")


if __name__ == "__main__":
    main()
