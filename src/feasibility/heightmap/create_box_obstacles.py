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

Also generates a wider square grid (10m x 10m by default, --extent) for a random-spawn dataset
generator that needs room to place the robot on every side of the obstacle -- the robot's
turning reach (1.101m, see comparator/compare_box_obstacles.py) plus the box's own
half-footprint (up to 0.964m at h=0.8) leaves almost no legal non-overlapping spawn positions
on the original 8m x 6m grid once you also need clearance from its edges. On this wider grid,
each run places the (shared, series-wide) box footprint at ONE of two positions:

  * random (default) -- uniformly sampled anywhere such that the box AND its ramp (sized for
    the series' tallest height, so the same sampled position stays valid across the whole
    height series) stay fully within the grid -- written to box_random_h* under
    assets/box_random/, a position genuinely different on every run (--seed to reproduce one).
  * --center -- footprint centered at the grid origin -- written to box_centered_h* under
    assets/box_centered/, unchanged from run to run. Kept in its own directory (rather than
    reusing box_random_h* with cx=cy=0) because other code hard-assumes an origin-centered box:
    terrain_patch.py's self-test relies on 4-fold symmetry about (0,0), and
    generate_dataset_utils.py's DEFAULT_MAP / smoke test assert a point at the origin is "on
    the box". Re-running the default (random) case must never silently invalidate those.

This replaces the old --off-center flag, which generated a THIRD, separate series (box_obstacle_h*
under assets/box/, anchored at fixed BOX_X0 on the original 8m x 6m grid -- still what
compare_box_obstacles.py reads, via a fixed spawn pose computed algebraically from BOX_X0/BOX_CY,
so that series' generating code -- box_path, box_obstacle_paths, build_box_obstacle's XLIM/YLIM
defaults -- stays in this module for those imports, it's just no longer reachable from this
script's CLI. Regenerate it (e.g. after changing --cell/--incline-deg) via a one-off Python call:
`build_box_obstacle(height, cell, incline_deg).save(box_path(height))` per height in BOX_HEIGHTS.

--batch generates a THIRD mode: N single-height maps (one obstacle height, --height, shared by
all of them -- NOT the BOX_HEIGHTS series), each its OWN independently-sampled random position
-- for a dataset generator that wants position variety across many terrains rather than a height
sweep at one position. Written to box_random_i<index>_h<height>cm under assets/box_random/
(same directory as the single-position random default, but index-prefixed so the two naming
patterns never collide). Since each map carries only one height, its position's valid sampling
region is sized for THAT height's own ramp width, not the whole series' worst case -- larger and
less conservative than the default mode's shared-position bound.

CLI parameters:
    --cell FLOAT          grid resolution in meters (default: 0.05)
    --incline-deg FLOAT   ramp incline angle in degrees, shared by every obstacle height in the
                           series (default: 75.0)
    --extent FLOAT        full width/height in meters of the square grid (default: 10.0)
    --center               center the box footprint at the grid origin (box_centered_h* series)
                           instead of the default random position (box_random_h* series)
    --batch                generate N single-height maps with independently random positions
                           (box_random_i*_h*cm) instead of the default height-series-at-one-
                           position behavior; mutually exclusive with --center
    --n INT                number of maps to generate in --batch mode (default: 100)
    --height FLOAT         obstacle height in meters shared by every map in --batch mode
                           (default: 0.50)
    --seed INT             RNG seed for the random position(s); ignored with --center

Usage:
    python src/feasibility/heightmap/create_box_obstacles.py                  # random position (default)
    python src/feasibility/heightmap/create_box_obstacles.py --seed 0         # reproducible random position
    python src/feasibility/heightmap/create_box_obstacles.py --center         # centered at the origin
    python src/feasibility/heightmap/create_box_obstacles.py --extent 20      # wider grid
    python src/feasibility/heightmap/create_box_obstacles.py --cell 0.01 --incline-deg 60
    python src/feasibility/heightmap/create_box_obstacles.py --batch          # 100 random-position maps, h=0.50m
    python src/feasibility/heightmap/create_box_obstacles.py --batch --n 500 --height 0.30 --seed 0
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
DEFAULT_CENTERED_EXTENT = 10.0  # m, full width/height of the centered series' square grid --
# knob: --extent. Box footprint is always centered in it (CENTERED_CX/CY), so any extent keeps
# the box at the grid's center.
CENTERED_ASSETS_DIR = REPO_ROOT / "assets" / "box_centered"
CENTERED_CX = 0.0
CENTERED_CY = 0.0

# --- random series: same wider square grid, box footprint placed anywhere within bounds ---
RANDOM_ASSETS_DIR = REPO_ROOT / "assets" / "box_random"

# --- batch mode: N single-height maps, each its own random position -- see module docstring ---
DEFAULT_BATCH_N = 100
DEFAULT_BATCH_HEIGHT = 0.50  # m


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
        "--extent",
        type=float,
        default=DEFAULT_CENTERED_EXTENT,
        help=f"full width/height in meters of the square grid (default {DEFAULT_CENTERED_EXTENT})",
    )
    parser.add_argument(
        "--center",
        action="store_true",
        help="center the box footprint at the grid origin (box_centered_h* series) instead of "
        "the default random position (box_random_h* series)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="generate --n single-height maps with independently random positions "
        "(box_random_i*_h*cm) instead of the default height-series-at-one-position behavior; "
        "mutually exclusive with --center",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_BATCH_N,
        help=f"number of maps to generate in --batch mode (default: {DEFAULT_BATCH_N})",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=DEFAULT_BATCH_HEIGHT,
        help=f"obstacle height in meters shared by every map in --batch mode (default: {DEFAULT_BATCH_HEIGHT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for the random position(s); ignored with --center",
    )
    args = parser.parse_args()
    if args.batch and args.center:
        parser.error("--batch and --center are mutually exclusive")
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


def random_box_path(height: float) -> pathlib.Path:
    """assets/box_random/box_random_h<height, cm, no dot> -- see box_path's docstring for why
    cm integers, not a dotted decimal."""
    return RANDOM_ASSETS_DIR / f"box_random_h{round(height * 100):03d}cm"


def random_box_paths(heights: tuple[float, ...] = BOX_HEIGHTS) -> list[tuple[float, pathlib.Path]]:
    """(height, path) pairs for the random series -- mirrors box_obstacle_paths()."""
    return [(h, random_box_path(h)) for h in heights]


BATCH_INDEX_WIDTH = 4  # zero-padding width for --batch indices -- fixed (not n-derived) so
# filenames sort consistently regardless of which --n a given batch used, up to 9999 maps.


def random_batch_path(index: int, height: float) -> pathlib.Path:
    """assets/box_random/box_random_i<index, zero-padded to BATCH_INDEX_WIDTH>_h<height, cm, no
    dot> -- index-prefixed so --batch's per-map files never collide with random_box_path()'s
    single shared-position series in the same directory."""
    return RANDOM_ASSETS_DIR / f"box_random_i{index:0{BATCH_INDEX_WIDTH}d}_h{round(height * 100):03d}cm"


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
    height: float,
    cell: float,
    incline_deg: float,
    extent: float = DEFAULT_CENTERED_EXTENT,
    *,
    cx: float = CENTERED_CX,
    cy: float = CENTERED_CY,
) -> HeightMapReader:
    """build_box_obstacle on a square grid `extent` meters wide/tall, footprint centered at
    (cx, cy) -- the grid origin by default, or anywhere else within bounds (see
    sample_random_center)."""
    half = extent / 2.0
    return build_box_obstacle(height, cell, incline_deg, xlim=(-half, half), ylim=(-half, half), cx=cx, cy=cy)


def max_ramp_width(height: float, incline_deg: float) -> float:
    """Ramp width at `height` -- the footprint inflation an obstacle of this height needs on
    every side before reaching ground level (ramp_width = height / tan(incline), see module
    docstring). Callers sharing one position across a height series pass max(BOX_HEIGHTS) to
    get the worst case any height in the series needs."""
    return height / np.tan(np.radians(incline_deg))


def sample_random_center(
    extent: float, incline_deg: float, rng: np.random.Generator, max_height: float = max(BOX_HEIGHTS)
) -> tuple[float, float]:
    """Uniformly sample a footprint center (cx, cy) such that the box AND its ramp (sized for
    `max_height`, the tallest height any map generated at this position will use) stay fully
    within the `extent` x `extent` square grid."""
    ramp = max_ramp_width(max_height, incline_deg)
    bound = extent / 2.0 - BOX_SIZE / 2.0 - ramp
    if bound < 0.0:
        raise ValueError(
            f"--extent {extent} m is too small to fit the box (size {BOX_SIZE} m) plus its ramp "
            f"(up to {ramp:.3f} m at incline {incline_deg} deg) anywhere inside the grid"
        )
    return rng.uniform(-bound, bound), rng.uniform(-bound, bound)


def main() -> None:
    args = parse_args()

    # (label, height, cx, cy, path) per mode -- one shared build+save+print loop below.
    tasks: list[tuple[str, float, float, float, pathlib.Path]]
    if args.batch:
        RANDOM_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(args.seed)
        tasks = []
        for i in range(args.n):
            cx, cy = sample_random_center(args.extent, args.incline_deg, rng, max_height=args.height)
            tasks.append((f"random (x={cx:.3f}, y={cy:.3f})", args.height, cx, cy, random_batch_path(i, args.height)))
    elif args.center:
        CENTERED_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        tasks = [("centered", h, CENTERED_CX, CENTERED_CY, path) for h, path in centered_box_paths()]
    else:
        RANDOM_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(args.seed)
        cx, cy = sample_random_center(args.extent, args.incline_deg, rng)
        label = f"random (x={cx:.3f}, y={cy:.3f})"
        tasks = [(label, h, cx, cy, path) for h, path in random_box_paths()]

    for label, height, cx, cy, path in tasks:
        build_centered_box(height, args.cell, args.incline_deg, args.extent, cx=cx, cy=cy).save(path)
        print(
            f"saved {path}.png / {path}.yaml  "
            f"({label} box height {height:.2f} m, extent {args.extent:.1f} m, "
            f"cell {args.cell} m, incline {args.incline_deg:.1f} deg)"
        )


if __name__ == "__main__":
    main()
