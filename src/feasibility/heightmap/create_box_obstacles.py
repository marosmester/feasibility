"""Generate a series of box-obstacle heightmaps: flat ground (z=0) everywhere except one
square obstacle with a 1.5m x 1.5m flat top, centered on a robot driving straight along +X.
The heightmaps in the series differ ONLY in the obstacle's z height; grid extent, cell size,
incline angle, and the obstacle's X/Y position/footprint are shared.

Unlike create_speed_bumps.py's 1D ramp profile (a bump spans the full Y width, so only its X
edges need ramping), a box obstacle has a finite footprint in BOTH axes, so all four sides need
ramping to avoid an instantaneous step -- same rationale as create_speed_bumps.py's module
docstring: Newton's heightfield/mesh collision gives a one-cell step a real (if very steep)
contact plane, so a smooth-but-steep ramp is preferable to a cliff. The height field is a
standard 2D box signed-distance-field: zero (i.e. flat top at `height`) inside the footprint,
falling off LINEARLY with Euclidean distance to the footprint's boundary, reaching 0 at
`ramp_width` away -- producing a mitered truncated-pyramid (frustum), not a cross/plus artifact.
`ramp_width` is derived PER obstacle height from --incline-deg (ramp_width = height / tan(incline))
rather than fixed, so every height in the series shares the same incline angle instead of the
same ramp width -- a fixed ramp width would make tall obstacles steeper than short ones.

DEFAULT_INCLINE_DEG is 75 deg, much steeper than create_speed_bumps.py's 15 deg drive-up ramp --
a box obstacle is meant to look and behave like a box (something to get stuck against), not a
second ramp, and the steep angle keeps ramp_width small (0.27*height, <=0.22m over the whole
series) so it stays a thin collision-friendly bevel well inside the grid at every height,
instead of ballooning into the terrain the way a 15 deg ramp would (2.99m at h=0.8, wider than
the box itself and large enough to run off the grid's edges).

Also generates a SECOND series, box_centered_h* (on a wider square grid -- 12m x 12m by default,
--extent -- with the box footprint centered at the world origin instead of near one edge) -- for
a random-spawn dataset generator that needs room to place the robot on every side of the
obstacle. Each run generates ONE of the two series, never both: the centered series by default,
or the original off-center 8m x 6m BOX_X0-anchored series (box_obstacle_h*, still what
compare_box_obstacles.py reads) via --off-center (whose extent isn't a knob -- it's tied to
compare_box_obstacles.py's tuned corner spawn, see CORNER_MARGIN there). They're kept as
separate series rather than widening BOX_CX/BOX_CY in place because the robot's turning reach
(1.101m, see comparator/compare_box_obstacles.py) plus the box's own half-footprint (up to
0.964m at h=0.8) leaves almost no legal non-overlapping spawn positions on the original grid
once you also need clearance from its edges.

CLI parameters:
    --cell FLOAT          grid resolution in meters (default: 0.05)
    --incline-deg FLOAT   ramp incline angle in degrees, shared by every obstacle height in the
                           series (default: 75.0)
    --off-center          generate the original off-center 8m x 6m box_obstacle_h* series
                           instead of the default centered box_centered_h* series
    --extent FLOAT        full width/height in meters of the centered series' square grid
                           (default: 16.0); ignored with --off-center, whose extent is fixed

Usage:
    python src/feasibility/heightmap/create_box_obstacles.py                  # centered series (default)
    python src/feasibility/heightmap/create_box_obstacles.py --extent 20      # wider centered grid
    python src/feasibility/heightmap/create_box_obstacles.py --off-center     # original off-center series
    python src/feasibility/heightmap/create_box_obstacles.py --cell 0.01 --incline-deg 60
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np

from feasibility.heightmap import HeightMapReader

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
ASSETS_DIR = REPO_ROOT / "assets" / "box"

# Grid extent -- matches HeightMapReader.flat()'s own default and create_speed_bumps.py's, so a
# spawn point tuned against that flat default lands in the same place here.
XLIM = (-2.0, 6.0)
YLIM = (-3.0, 3.0)

BOX_X0 = 2.0  # m, leading (near) edge of the obstacle's flat-top footprint
BOX_SIZE = 1.5  # m, footprint extent along BOTH X and Y (square)
BOX_CX = BOX_X0 + BOX_SIZE / 2.0  # m, footprint center X
BOX_CY = (YLIM[0] + YLIM[1]) / 2.0  # m, footprint center Y -- centered in the grid

DEFAULT_CELL = 0.05  # m, grid resolution
DEFAULT_INCLINE_DEG = 75.0  # deg, ramp slope shared by every height in the series -- see module docstring

BOX_HEIGHTS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)  # m, one heightmap per height

# --- centered series: wider square grid, box footprint at the origin -- see module docstring ---
DEFAULT_CENTERED_EXTENT = 16.0  # m, full width/height of the centered series' square grid --
# knob: --extent. Box footprint is always centered in it (CENTERED_CX/CY), so any extent keeps
# the box at the grid's center.
CENTERED_ASSETS_DIR = REPO_ROOT / "assets" / "box_centered"
CENTERED_CX = 0.0
CENTERED_CY = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", type=float, default=DEFAULT_CELL, help="grid resolution in meters")
    parser.add_argument(
        "--incline-deg",
        type=float,
        default=DEFAULT_INCLINE_DEG,
        help="ramp incline angle in degrees, shared by every obstacle height in the series",
    )
    parser.add_argument(
        "--off-center",
        action="store_true",
        help="generate the original off-center 8m x 6m box_obstacle_h* series instead of the "
        "default centered box_centered_h* series",
    )
    parser.add_argument(
        "--extent",
        type=float,
        default=DEFAULT_CENTERED_EXTENT,
        help="full width/height in meters of the centered series' square grid (default "
        f"{DEFAULT_CENTERED_EXTENT}); ignored with --off-center, whose extent is fixed",
    )
    args = parser.parse_args()
    if args.off_center and args.extent != DEFAULT_CENTERED_EXTENT:
        parser.error("--extent has no effect with --off-center (its grid extent is fixed)")
    return args


def box_path(height: float) -> pathlib.Path:
    """assets/box/box_obstacle_h<height, cm, no dot> -- no extension, HeightMapReader appends
    .png/.yaml via pathlib's with_suffix(), which treats the LAST '.' in the name as an
    extension separator. A dotted decimal like "h0.20" would collide "h0.20" and "h0.30"
    onto the same "h0.{yaml,png}" (both get stem "h0", suffix replaced wholesale) -- cm
    integers side-step that."""
    return ASSETS_DIR / f"box_obstacle_h{round(height * 100):03d}cm"


def box_obstacle_paths(heights: tuple[float, ...] = BOX_HEIGHTS) -> list[tuple[float, pathlib.Path]]:
    """(height, path) pairs for the series -- the single source of truth other callers can
    read back, so the height list isn't duplicated elsewhere."""
    return [(h, box_path(h)) for h in heights]


def centered_box_path(height: float) -> pathlib.Path:
    """assets/box_centered/box_centered_h<height, cm, no dot> -- see box_path's docstring for
    why cm integers, not a dotted decimal."""
    return CENTERED_ASSETS_DIR / f"box_centered_h{round(height * 100):03d}cm"


def centered_box_paths(heights: tuple[float, ...] = BOX_HEIGHTS) -> list[tuple[float, pathlib.Path]]:
    """(height, path) pairs for the centered series -- mirrors box_obstacle_paths()."""
    return [(h, centered_box_path(h)) for h in heights]


def build_box_obstacle(
    height: float,
    cell: float,
    incline_deg: float,
    *,
    xlim: tuple[float, float] = XLIM,
    ylim: tuple[float, float] = YLIM,
    cx: float = BOX_CX,
    cy: float = BOX_CY,
) -> HeightMapReader:
    """Flat ground except a square frustum centered at (cx, cy): flat top at `height` over the
    BOX_SIZE x BOX_SIZE footprint, linear ramp down to ground on all four sides (mitered
    corners, via Euclidean distance to the footprint rectangle) sloped at `incline_deg`."""
    ramp_width = height / np.tan(np.radians(incline_deg))
    nx = int(round((xlim[1] - xlim[0]) / cell)) + 1
    ny = int(round((ylim[1] - ylim[0]) / cell)) + 1
    xs = xlim[0] + (np.arange(nx) + 0.5) * cell
    ys = ylim[0] + (np.arange(ny) + 0.5) * cell
    X, Y = np.meshgrid(xs, ys)  # [ny, nx], row=y, col=x

    half = BOX_SIZE / 2.0
    dx = np.maximum(0.0, np.abs(X - cx) - half)
    dy = np.maximum(0.0, np.abs(Y - cy) - half)
    d = np.hypot(dx, dy)  # Euclidean distance outside the footprint rectangle, 0 inside it
    H = height * np.clip(1.0 - d / ramp_width, 0.0, 1.0)
    return HeightMapReader(H, origin=(xlim[0], ylim[0]), cell=cell)


def build_centered_box(
    height: float, cell: float, incline_deg: float, extent: float = DEFAULT_CENTERED_EXTENT
) -> HeightMapReader:
    """build_box_obstacle on a square grid `extent` meters wide/tall, footprint at the origin."""
    half = extent / 2.0
    return build_box_obstacle(
        height, cell, incline_deg, xlim=(-half, half), ylim=(-half, half), cx=CENTERED_CX, cy=CENTERED_CY
    )


def main() -> None:
    args = parse_args()
    if args.off_center:
        ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        for height, path in box_obstacle_paths():
            build_box_obstacle(height, args.cell, args.incline_deg).save(path)
            print(
                f"saved {path}.png / {path}.yaml  "
                f"(box height {height:.2f} m, cell {args.cell} m, incline {args.incline_deg:.1f} deg)"
            )
    else:
        CENTERED_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        for height, path in centered_box_paths():
            build_centered_box(height, args.cell, args.incline_deg, args.extent).save(path)
            print(
                f"saved {path}.png / {path}.yaml  "
                f"(centered box height {height:.2f} m, extent {args.extent:.1f} m, "
                f"cell {args.cell} m, incline {args.incline_deg:.1f} deg)"
            )


if __name__ == "__main__":
    main()
