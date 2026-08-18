"""Generate a series of speed-bump heightmaps: flat ground (z=0) everywhere except one
bump spanning the full Y width of the grid, oriented perpendicular to a robot driving
straight along +X -- i.e. a real speed bump, not an angled ramp. The heightmaps in the
series differ ONLY in the bump's z height; grid extent, cell size, incline angle, and the
bump's X position/width are shared. Used by feasibility.comparator.batch_compare to probe how
ostrich (dynamics) vs helhest_stack (kinematic twin) diverge as the bump grows from
negligible to significant relative to the wheel radius (0.35 m).

The bump's leading/trailing edges are a linear ramp, not an instantaneous step -- a step
forces the whole height jump into whatever one grid cell straddles it (Newton's heightfield
collision triangulates every cell exactly, see newton/_src/utils/heightfield.py's
_heightfield_surface_query, so a one-cell step still gets a real contact plane, just a very
steep one). Ramp width is derived PER bump height from --incline-deg
(ramp_width = height / tan(incline)) rather than fixed, so every height in the series shares
the same incline angle instead of the same ramp width -- a fixed ramp width would make tall
bumps steeper than short ones.

Usage:
    python -m feasibility.heightmap.create_speed_bumps
    python -m feasibility.heightmap.create_speed_bumps --cell 0.01 --incline-deg 10
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np

from feasibility.heightmap import HeightMapReader

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
ASSETS_DIR = REPO_ROOT / "assets"

# Grid extent -- matches HeightMapReader.flat()'s own default, so a spawn point tuned against
# that flat default lands in the same place here.
XLIM = (-2.0, 6.0)
YLIM = (-3.0, 3.0)

BUMP_X0 = 2.0  # m, leading (near) edge of the bump's flat top
BUMP_WIDTH = 0.4  # m, extent of the flat top along X (direction of travel)

DEFAULT_CELL = 0.05  # m, grid resolution
DEFAULT_INCLINE_DEG = 15.0  # deg, ramp slope shared by every height in the series

BUMP_HEIGHTS =  (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70)  # m, one heightmap per height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", type=float, default=DEFAULT_CELL, help="grid resolution in meters")
    parser.add_argument(
        "--incline-deg",
        type=float,
        default=DEFAULT_INCLINE_DEG,
        help="ramp incline angle in degrees, shared by every bump height in the series",
    )
    return parser.parse_args()


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


def build_speed_bump(height: float, cell: float, incline_deg: float) -> HeightMapReader:
    """Flat ground except a trapezoidal bump straddling [BUMP_X0, BUMP_X0+BUMP_WIDTH)
    (flat top at `height`, linear ramps on each side sloped at `incline_deg`), spanning the
    full Y range -- perpendicular to a robot driving straight along +X, so any Y offset
    within the grid hits the bump square-on rather than at an angle."""
    ramp_width = height / np.tan(np.radians(incline_deg))
    nx = int(round((XLIM[1] - XLIM[0]) / cell)) + 1
    ny = int(round((YLIM[1] - YLIM[0]) / cell)) + 1
    xs = XLIM[0] + (np.arange(nx) + 0.5) * cell
    # np.interp clamps to fp's end values outside [xp[0], xp[-1]], both 0 here, so this
    # also covers "flat ground everywhere else" with no separate masking.
    breakpoints = [BUMP_X0 - ramp_width, BUMP_X0, BUMP_X0 + BUMP_WIDTH, BUMP_X0 + BUMP_WIDTH + ramp_width]
    profile = np.interp(xs, breakpoints, [0.0, height, height, 0.0])
    H = np.tile(profile, (ny, 1))
    return HeightMapReader(H, origin=(XLIM[0], YLIM[0]), cell=cell)


def main() -> None:
    args = parse_args()
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for height, path in speed_bump_paths():
        build_speed_bump(height, args.cell, args.incline_deg).save(path)
        print(
            f"saved {path}.png / {path}.yaml  "
            f"(bump height {height:.2f} m, cell {args.cell} m, incline {args.incline_deg:.1f} deg)"
        )


if __name__ == "__main__":
    main()
