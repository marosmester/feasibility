"""Combine two HeightMapReader assets into one by ADDING their elevation fields -- e.g. layering
create_box_obstacles.py's box obstacle on top of create_rough_terrain.py's random rough field,
so the rough texture is kept everywhere and a real obstacle appears where the box's footprint
is, instead of picking one terrain or the other.

Addition is the right combination, not a max/replace: create_box_obstacles.py's height field is
already a RELATIVE bump layer -- 0 outside its footprint+ramp, rising to `height` only inside it
(see that module's docstring) -- not an absolute elevation map. Adding it to any base terrain
reproduces the exact same bump shape riding on top of whatever the base terrain has locally,
with the ramp blending back into the untouched base exactly where the overlay's own height
decays to 0. This generalizes past the rough+box case: OVERLAY_PATH can be any HeightMapReader
asset defined the same way (create_speed_bumps.py's bumps included).

Grid handling: BASE_PATH's grid is always the output grid. When OVERLAY_PATH shares the exact
same origin/cell/shape (the common case -- create_rough_terrain.py and create_box_obstacles.py's
centered series both default to a 16x16 m grid at cell=0.05, origin (-8, -8)), the two arrays
are added directly, no interpolation. Otherwise the overlay is bilinearly resampled onto the
base's own cell centers via HeightMapReader.sample(), which CLAMPS outside its own grid -- since
the centered box series' footprint is small relative to its 16 m grid and decays to exactly 0
well before the border, a base grid larger than the overlay's still gets a clean 0 added outside
the overlay's own extent, not a clamped nonzero smear.

CLI parameters:
    BASE_PATH      positional, optional -- terrain kept as-is outside the overlay's footprint
                   (default: heightmap.create_rough_terrain.rough_path(0), i.e.
                   assets/rough/rough_seed0000)
    OVERLAY_PATH   positional, optional -- relative bump layer added on top (default:
                   heightmap.create_box_obstacles.centered_box_path(0.80), the tallest/steepest
                   box in that series)
    --out PATH     output path stem (no extension), overriding the default
                   assets/mended/<base stem>_plus_<overlay stem> naming

Usage:
    python src/feasibility/heightmap/mend_two_meshes.py                              # rough_seed0000 + box h080cm
    python src/feasibility/heightmap/mend_two_meshes.py assets/rough/rough_seed0007 assets/box_centered/box_centered_h040cm
    python src/feasibility/heightmap/mend_two_meshes.py --out assets/mended/my_terrain
"""
from __future__ import annotations

import argparse
import math
import pathlib

import numpy as np

from feasibility.heightmap import HeightMapReader
from feasibility.heightmap.create_box_obstacles import centered_box_path
from feasibility.heightmap.create_rough_terrain import rough_path

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
ASSETS_DIR = REPO_ROOT / "assets" / "mended"

DEFAULT_BASE = rough_path(0)
DEFAULT_OVERLAY = centered_box_path(0.80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "base", type=pathlib.Path, nargs="?", default=DEFAULT_BASE,
        help=f"terrain kept as-is outside the overlay's footprint (default {DEFAULT_BASE})",
    )
    parser.add_argument(
        "overlay", type=pathlib.Path, nargs="?", default=DEFAULT_OVERLAY,
        help=f"relative bump layer added on top (default {DEFAULT_OVERLAY})",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="output path stem (no extension), overriding the default assets/mended/<base>_plus_<overlay> naming",
    )
    return parser.parse_args()


def mended_path(base_path: pathlib.Path, overlay_path: pathlib.Path) -> pathlib.Path:
    """assets/mended/<base stem>_plus_<overlay stem> -- no extension, HeightMapReader appends
    .png/.yaml."""
    return ASSETS_DIR / f"{base_path.stem}_plus_{overlay_path.stem}"


def build_mended_terrain(base_path: pathlib.Path, overlay_path: pathlib.Path) -> tuple[HeightMapReader, bool]:
    """base.H + overlay.H, on base's grid -- see module docstring for why addition (not
    max/replace) is correct and how the grid mismatch case is handled."""
    base = HeightMapReader.load(base_path)
    overlay = HeightMapReader.load(overlay_path)

    same_grid = (
        base.nx == overlay.nx
        and base.ny == overlay.ny
        and math.isclose(base.cell, overlay.cell, abs_tol=1e-9)
        and math.isclose(base.x0, overlay.x0, abs_tol=1e-9)
        and math.isclose(base.y0, overlay.y0, abs_tol=1e-9)
    )
    if same_grid:
        overlay_H = overlay.H
    else:
        xs = base.x0 + (np.arange(base.nx) + 0.5) * base.cell
        ys = base.y0 + (np.arange(base.ny) + 0.5) * base.cell
        X, Y = np.meshgrid(xs, ys)  # [ny, nx], row=y, col=x -- same layout as base.H
        overlay_H = overlay.sample(X, Y)

    H = base.H + overlay_H
    return HeightMapReader(H, origin=(base.x0, base.y0), cell=base.cell), same_grid


def main() -> None:
    args = parse_args()
    hmap, same_grid = build_mended_terrain(args.base, args.overlay)
    path = pathlib.Path(args.out) if args.out else mended_path(args.base, args.overlay)
    path.parent.mkdir(parents=True, exist_ok=True)
    hmap.save(path)

    print(
        f"saved {path}.png / {path}.yaml  "
        f"({hmap.nx}x{hmap.ny} cells, cell={hmap.cell} m, "
        f"base={args.base}, overlay={args.overlay} ({'exact grid' if same_grid else 'resampled'}), "
        f"z in [{hmap.min_z:.3f}, {hmap.max_z:.3f}] m)"
    )


if __name__ == "__main__":
    main()
