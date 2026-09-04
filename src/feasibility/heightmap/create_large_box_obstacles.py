"""Generate heightmaps with larger, non-square rectangular obstacle(s) than
create_box_obstacles.py's fixed 1.5m x 1.5m box, aimed at grid_learning/generate_dataset.py's
whole-map divergence-field labeling: a single small box only touches a thin ring of the 15x15
spawn lattice (~6-12% of cells), so a batch of those maps starves the CNN of "interesting"
(ostrich/hstack-diverging) training rows. Bigger, more elongated rectangles expose far more
obstacle/flat-ground frontier per map -- more lattice cells sit near a boundary -- while a hard
cap on total obstacle+ramp area keeps a healthy share of the map genuinely flat, for the
"robot entirely on flat ground" baseline rows the CNN also needs.

Like create_box_obstacles.py, obstacles are flat-topped frustums: zero outside their footprint,
rising LINEARLY with Euclidean distance to the (rectangular, not necessarily square) footprint
boundary, reaching full `height` inside it, ramp incline shared across every obstacle
(--incline-deg, default 75 deg -- same steep, stuck-against-a-box rationale as
create_box_obstacles.py, not a second drive-up ramp).

Height is FIXED (--height, default 0.70 m) for every obstacle on every map -- unlike
create_box_obstacles.py's BOX_HEIGHTS series, there is no per-height sweep here. That fixed
height is what makes composing multiple obstacles simple: layers are combined with an
elementwise MAXIMUM, not a sum. create_box_obstacles.py's build_multi_box sums layers and
requires (and asserts) non-overlapping footprints, because summing two overlapping ramps of
DIFFERENT heights would fabricate a spurious taller-than-either tower. Here every layer tops
out at the same `height`, so touching or even overlapping rectangles just merge into one
contiguous flat-topped region under a max -- no separation bookkeeping needed, and "obstacles
can touch" (explicitly wanted) falls out for free.

--n-boxes is an UPPER BOUND, not an exact count: obstacle dimensions (width and depth, each
independently a multiple of 0.5 m from 0.5 to 6.0 m) are drawn to fill a fixed-size AREA BUDGET
(--max-area-fraction, default 0.5 of the extent x extent map), using each candidate box's
INFLATED footprint -- (w + 2*ramp) x (d + 2*ramp), i.e. including its sloped skirt, which is
not flat ground either -- against the remaining budget. That inflated-sum test is deliberately
conservative: build_multi_rect's actual unioned area can only be <= this estimate, since any
overlap between boxes only shrinks the true footprint below the sum of its parts, so the 50%
cap holds without ever having to rasterize a candidate to check it. Once a box no longer fits
the remaining budget after MAX_SIZE_ATTEMPTS draws, sizing stops early -- a map may end up with
fewer than --n-boxes obstacles, never more.

Given a fixed list of obstacle sizes, WHERE to place them is chosen by a best-of-N random
search (--position-trials): each candidate places every box's center uniformly within its own
fits-inside-the-grid bound, builds the actual height field, and scores it via
accessible_frontier_score -- the count of obstacle/flat-ground BOUNDARY cells (4-connected,
computed directly on the raster) that also lie within DEFAULT_ACCESS_LIMIT of the map center.
That single definition buys two things at once: a boundary cell between two touching/
overlapping obstacles is never counted (both its neighbors are "obstacle", not "flat"), and a
boundary segment further out than DEFAULT_ACCESS_LIMIT is excluded because no spawn-lattice
cell can ever reach it. DEFAULT_ACCESS_LIMIT (3.5 m) is deliberately equal to
grid_learning/generate_dataset.py's SPAWN_LIMIT -- re-derived here rather than imported, the
same "independent trees" stance grid_learning's own module docstring already takes toward
heightmap/ and comparator/ (see its "Deliberately does not import from learning/" note): this
module has no import-time dependency on grid_learning, so the two can keep changing
independently, at the cost of a duplicated constant a reader has to notice matches on purpose.
The highest-scoring candidate out of --position-trials tries wins; this is deliberately a
cheap random search, not a combinatorial placement optimizer.

Written to assets/large_box_random/<seed>/large_box_i<index>_h<height,cm>cm[.png/.yaml] --
--seed is REQUIRED (as in create_box_obstacles.py's --batch mode) both to seed the RNG and to
namespace the output directory, so re-running one seed always regenerates the same maps in the
same place and different seeds never collide. The filename pattern deliberately does NOT start
with "box_random_" (create_box_obstacles.py's prefix), so grid_learning/generate_dataset.py's
default --map-glob (box_random_*_h*.png) will never accidentally pick these up; point it here
explicitly via +maps_dir=assets/large_box_random/<seed> +map_glob=large_box_i*_h*.png.

CLI parameters:
    --seed INT                 RNG seed; REQUIRED -- also names the
                                assets/large_box_random/<seed>/ output subdirectory
    --n INT                     number of maps to generate (default: 100)
    --n-boxes INT               upper bound on obstacles per map, each independently sized+
                                placed (default: 3); a map may end up with fewer if the area
                                budget is exhausted first
    --height FLOAT              obstacle height in meters, shared by every obstacle on every
                                map (default: 0.70)
    --max-area-fraction FLOAT   max total obstacle+ramp footprint as a fraction of the map's
                                area (default: 0.5)
    --extent FLOAT              full width/height in meters of the square grid (default: 10.0)
    --cell FLOAT                grid resolution in meters (default: 0.05)
    --incline-deg FLOAT          ramp incline angle in degrees, shared by every obstacle
                                (default: 75.0)
    --position-trials INT        number of random placement candidates scored per map, the
                                best-of-N frontier search (default: 30)

Usage:
    python src/feasibility/heightmap/create_large_box_obstacles.py --seed 0
    python src/feasibility/heightmap/create_large_box_obstacles.py --seed 0 --n 500
    python src/feasibility/heightmap/create_large_box_obstacles.py --seed 0 --n-boxes 5 --max-area-fraction 0.4
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np

from feasibility.heightmap import HeightMapReader

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
ASSETS_DIR = REPO_ROOT / "assets" / "large_box_random"

DEFAULT_HEIGHT = 0.70  # m, shared by every obstacle on every map -- see module docstring
DEFAULT_EXTENT = 10.0  # m, full width/height of the square grid
DEFAULT_CELL = 0.05  # m, grid resolution
DEFAULT_INCLINE_DEG = 75.0  # deg, ramp slope shared by every obstacle -- see create_box_obstacles.py's rationale

SIZE_MIN, SIZE_MAX, SIZE_STEP = 0.5, 6.0, 0.5  # m, width/depth drawn independently from this range
SIZE_CHOICES = np.round(np.arange(SIZE_MIN, SIZE_MAX + 1e-9, SIZE_STEP), 2)

DEFAULT_MAX_AREA_FRACTION = 0.5  # cap on total obstacle+ramp footprint as a fraction of extent**2
DEFAULT_N_BOXES_UPPER = 3  # knob: --n-boxes, an UPPER bound -- see module docstring
DEFAULT_POSITION_TRIALS = 30  # knob: --position-trials, best-of-N placement search

MAX_SIZE_ATTEMPTS = 50  # draws per box before giving up on adding it (budget likely exhausted)

DEFAULT_ACCESS_LIMIT = 3.5  # m, deliberately == grid_learning/generate_dataset.py's SPAWN_LIMIT
# -- see module docstring for why it's re-derived here rather than imported.

BATCH_INDEX_WIDTH = 4  # zero-padding width for map indices, same rationale as create_box_obstacles.py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="RNG seed; also names the assets/large_box_random/<seed>/ output subdirectory",
    )
    parser.add_argument("--n", type=int, default=100, help="number of maps to generate (default: 100)")
    parser.add_argument(
        "--n-boxes",
        type=int,
        default=DEFAULT_N_BOXES_UPPER,
        help=f"upper bound on obstacles per map (default: {DEFAULT_N_BOXES_UPPER}); a map may "
        f"end up with fewer once the area budget is exhausted",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=DEFAULT_HEIGHT,
        help=f"obstacle height in meters, shared by every obstacle (default: {DEFAULT_HEIGHT})",
    )
    parser.add_argument(
        "--max-area-fraction",
        type=float,
        default=DEFAULT_MAX_AREA_FRACTION,
        help=f"max total obstacle+ramp footprint as a fraction of the map area "
        f"(default: {DEFAULT_MAX_AREA_FRACTION})",
    )
    parser.add_argument(
        "--extent",
        type=float,
        default=DEFAULT_EXTENT,
        help=f"full width/height in meters of the square grid (default: {DEFAULT_EXTENT})",
    )
    parser.add_argument("--cell", type=float, default=DEFAULT_CELL, help="grid resolution in meters")
    parser.add_argument(
        "--incline-deg",
        type=float,
        default=DEFAULT_INCLINE_DEG,
        help="ramp incline angle in degrees, shared by every obstacle",
    )
    parser.add_argument(
        "--position-trials",
        type=int,
        default=DEFAULT_POSITION_TRIALS,
        help=f"random placement candidates scored per map (default: {DEFAULT_POSITION_TRIALS})",
    )
    args = parser.parse_args()
    if args.n_boxes < 1:
        parser.error("--n-boxes must be >= 1")
    if not (0.0 < args.max_area_fraction <= 1.0):
        parser.error("--max-area-fraction must be in (0, 1]")
    if args.position_trials < 1:
        parser.error("--position-trials must be >= 1")
    return args


def large_box_batch_path(index: int, seed: int, height: float) -> pathlib.Path:
    """assets/large_box_random/<seed>/large_box_i<index, zero-padded>_h<height, cm, no dot> --
    cm-integer height avoids the dotted-decimal filename collision create_box_obstacles.py's
    box_path docstring explains; the "large_box_" prefix (not "box_random_") is deliberate so
    grid_learning/generate_dataset.py's default --map-glob never picks these up by accident."""
    return ASSETS_DIR / str(seed) / f"large_box_i{index:0{BATCH_INDEX_WIDTH}d}_h{round(height * 100):03d}cm"


def build_rect_obstacle(
    height: float,
    cell: float,
    incline_deg: float,
    extent: float,
    w: float,
    d: float,
    cx: float,
    cy: float,
) -> HeightMapReader:
    """Flat ground except one rectangular frustum centered at (cx, cy): flat top at `height`
    over a w x d footprint (independent half-extents, so w != d / non-square is fine), linear
    ramp down to ground on all four sides via Euclidean distance to the footprint rectangle,
    sloped at `incline_deg` -- generalizes create_box_obstacles.py's build_box_obstacle (which
    hardcodes one shared half-extent for a square footprint)."""
    half = extent / 2.0
    ramp_width = height / np.tan(np.radians(incline_deg))
    nx = int(round(extent / cell)) + 1
    ny = nx
    xs = -half + (np.arange(nx) + 0.5) * cell
    ys = -half + (np.arange(ny) + 0.5) * cell
    X, Y = np.meshgrid(xs, ys)  # [ny, nx], row=y, col=x

    hx, hy = w / 2.0, d / 2.0
    dx = np.maximum(0.0, np.abs(X - cx) - hx)
    dy = np.maximum(0.0, np.abs(Y - cy) - hy)
    dist = np.hypot(dx, dy)
    H = height * np.clip(1.0 - dist / ramp_width, 0.0, 1.0)
    return HeightMapReader(H, origin=(-half, -half), cell=cell)


def build_multi_rect(
    height: float,
    cell: float,
    incline_deg: float,
    extent: float,
    boxes: list[tuple[float, float, float, float]],
) -> HeightMapReader:
    """build_rect_obstacle combined over one layer per (w, d, cx, cy) in `boxes`, via an
    elementwise MAXIMUM rather than a sum -- see module docstring for why max is correct here
    (fixed shared `height`, no non-overlap requirement, so touching/overlapping footprints just
    merge). A single box reproduces build_rect_obstacle exactly."""
    if not boxes:
        raise ValueError("boxes must hold at least one (w, d, cx, cy)")
    layers = [build_rect_obstacle(height, cell, incline_deg, extent, w, d, cx, cy) for w, d, cx, cy in boxes]
    H = np.maximum.reduce([layer.H for layer in layers])
    return HeightMapReader(H, origin=(layers[0].x0, layers[0].y0), cell=cell, min_z=0.0, max_z=height)


def sample_rect_sizes(
    rng: np.random.Generator,
    n_boxes_upper: int,
    incline_deg: float,
    height: float,
    max_area_fraction: float,
    extent: float,
) -> list[tuple[float, float]]:
    """Up to `n_boxes_upper` (width, depth) pairs, each drawn from SIZE_CHOICES, filling a
    running area budget of `max_area_fraction * extent**2` -- tested against each candidate
    box's INFLATED footprint (w + 2*ramp) x (d + 2*ramp), so the sloped skirt counts as
    non-flat ground too. This sum is a conservative upper bound on build_multi_rect's actual
    (possibly-overlapping) unioned area, so the budget test alone enforces the cap without ever
    rasterizing a candidate -- see module docstring. Stops early, before `n_boxes_upper` is
    reached, once a box no longer fits after MAX_SIZE_ATTEMPTS draws: --n-boxes is an upper
    bound, not an exact count."""
    ramp = height / np.tan(np.radians(incline_deg))
    budget = max_area_fraction * extent ** 2
    sizes: list[tuple[float, float]] = []
    for _ in range(n_boxes_upper):
        for _ in range(MAX_SIZE_ATTEMPTS):
            w = float(rng.choice(SIZE_CHOICES))
            d = float(rng.choice(SIZE_CHOICES))
            inflated_area = (w + 2.0 * ramp) * (d + 2.0 * ramp)
            if inflated_area <= budget:
                sizes.append((w, d))
                budget -= inflated_area
                break
        else:
            break
    if not sizes:
        raise ValueError(
            f"--max-area-fraction {max_area_fraction} on a {extent} m grid leaves no budget "
            f"for even the smallest ({SIZE_MIN} m) obstacle plus its ramp"
        )
    return sizes


def accessible_frontier_score(
    H: np.ndarray, cell: float, extent: float, access_limit: float, eps: float = 1e-6
) -> int:
    """Count of obstacle/flat-ground boundary cells (4-connected: an obstacle cell -- H > eps --
    with at least one non-obstacle neighbor) whose world position lies within `access_limit` of
    the map center. A boundary cell between two touching/overlapping obstacles is never counted
    (both neighbors are "obstacle"), and a boundary cell further out than `access_limit` is
    excluded because no spawn-lattice cell can ever reach it -- see module docstring."""
    obstacle = H > eps
    boundary = np.zeros_like(obstacle)
    boundary[:-1, :] |= obstacle[:-1, :] & ~obstacle[1:, :]
    boundary[1:, :] |= obstacle[1:, :] & ~obstacle[:-1, :]
    boundary[:, :-1] |= obstacle[:, :-1] & ~obstacle[:, 1:]
    boundary[:, 1:] |= obstacle[:, 1:] & ~obstacle[:, :-1]

    ny, nx = H.shape
    half = extent / 2.0
    xs = -half + (np.arange(nx) + 0.5) * cell
    ys = -half + (np.arange(ny) + 0.5) * cell
    X, Y = np.meshgrid(xs, ys)
    reachable = (np.abs(X) <= access_limit) & (np.abs(Y) <= access_limit)
    return int(np.count_nonzero(boundary & reachable))


def choose_best_layout(
    sizes: list[tuple[float, float]],
    extent: float,
    incline_deg: float,
    height: float,
    cell: float,
    access_limit: float,
    rng: np.random.Generator,
    n_trials: int,
) -> tuple[HeightMapReader, list[tuple[float, float, float, float]]]:
    """Best-of-`n_trials` random placement search: each trial places every box in `sizes`
    uniformly within its own fits-inside-the-grid bound, scores the resulting map via
    accessible_frontier_score, and the highest-scoring trial wins -- see module docstring."""
    ramp = height / np.tan(np.radians(incline_deg))
    half = extent / 2.0
    best_score = -1
    best_H: HeightMapReader | None = None
    best_boxes: list[tuple[float, float, float, float]] = []
    for _ in range(n_trials):
        boxes = []
        for w, d in sizes:
            bx = half - w / 2.0 - ramp
            by = half - d / 2.0 - ramp
            cx = rng.uniform(-bx, bx)
            cy = rng.uniform(-by, by)
            boxes.append((w, d, cx, cy))
        H = build_multi_rect(height, cell, incline_deg, extent, boxes)
        score = accessible_frontier_score(H.H, cell, extent, access_limit)
        if score > best_score:
            best_score, best_H, best_boxes = score, H, boxes
    assert best_H is not None
    return best_H, best_boxes


def main() -> None:
    args = parse_args()
    (ASSETS_DIR / str(args.seed)).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for i in range(args.n):
        sizes = sample_rect_sizes(
            rng, args.n_boxes, args.incline_deg, args.height, args.max_area_fraction, args.extent
        )
        H, boxes = choose_best_layout(
            sizes, args.extent, args.incline_deg, args.height, args.cell,
            DEFAULT_ACCESS_LIMIT, rng, args.position_trials,
        )
        score = accessible_frontier_score(H.H, args.cell, args.extent, DEFAULT_ACCESS_LIMIT)
        area_fraction = float(np.count_nonzero(H.H > 1e-6)) / H.H.size
        path = large_box_batch_path(i, args.seed, args.height)
        H.save(path)
        box_desc = " ".join(f"({w:.1f}x{d:.1f} @ x={cx:.2f},y={cy:.2f})" for w, d, cx, cy in boxes)
        print(
            f"saved {path}.png / {path}.yaml  "
            f"({len(boxes)}/{args.n_boxes} box(es) {box_desc}, area {area_fraction:.1%}, "
            f"frontier {score} cells, height {args.height:.2f} m, extent {args.extent:.1f} m)"
        )


if __name__ == "__main__":
    main()
