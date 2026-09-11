"""Generate heightmaps combining `create_rough_terrain.py`'s band-limited rough field with ONE
`create_large_box_obstacles.py`-style rectangular obstacle, purpose-built for
`lattice_learning/generate_dataset.py` rather than `grid_learning/`'s whole-map divergence field:

* `assets/large_box_random/` is tuned for `grid_learning`'s fixed 15x15 spawn lattice -- up to
  `--max-area-fraction 0.5` obstacle coverage and a best-of-N placement search that maximizes
  frontier cells reachable from the map CENTER. `lattice_learning/generate_dataset.py` instead
  rejection-samples spawn poses over a map's WHOLE extent for a tiny 0.3-0.6 m arc
  (`sample_spawn_poses`), so a map from that series can leave too little clear ground for its
  20-round rejection budget to find enough poses -- exactly the failure this script exists to
  avoid, by capping obstacle coverage far lower and never concentrating it.
* Every `large_box_random` map is flat ground plus box(es), nothing else -- binary "on the
  obstacle or not" signal. Adding continuous rough terrain underneath (same ADD-not-max
  combination `mend_two_meshes.py` uses, since the box layer is a relative bump -- 0 outside its
  footprint -- riding on top of any base) gives `lattice_learning`'s patch-conditioned regression
  a graded divergence signal almost everywhere, not just at one obstacle's edge.

Reuses `create_rough_terrain.build_rough_terrain` and `create_large_box_obstacles.sample_rect_sizes`/
`build_multi_rect` directly rather than re-deriving them: both live in `heightmap/`, which composes
its own generators freely (the "independent trees" stance in CLAUDE.md/design.md is about
`lattice_learning`/`grid_learning`/`learning` not importing each other, not about `heightmap/`
itself). The frontier-score placement search (`choose_best_layout`/`accessible_frontier_score`)
is deliberately NOT reused: it exists to bias a box towards `grid_learning`'s fixed-lattice
reachable region, and `lattice_learning` has no such fixed lattice -- a single uniformly random
position is exactly as good, so `place_boxes_randomly` below is that same per-trial placement
with the best-of-N scoring loop removed.

Each map's rough realization is independently seeded (drawn from the batch's own `--seed`-derived
RNG, one integer per map index) so a batch of `--n` maps gets `--n` different rough textures, not
one texture reused everywhere; the SAME per-map RNG state then draws that map's box size(s) and
position(s), so the whole batch is reproducible from `--seed` alone, same convention
`create_large_box_obstacles.py --seed` uses.

CLI parameters:
    --seed INT                 RNG seed; REQUIRED -- also names the
                                assets/rough_box/<seed>/ output subdirectory
    --n INT                     number of maps to generate (default: 100)
    --n-boxes INT               upper bound on obstacles per map (default: 1 -- "one random
                                block obstacle"); a map may end up with fewer if the area budget
                                is exhausted first (see sample_rect_sizes)
    --height FLOAT              obstacle height in meters (default: 0.70, the same platform-scale
                                obstacle height every other box generator uses)
    --max-area-fraction FLOAT   max total obstacle+ramp footprint as a fraction of the map's area
                                (default: 0.12 -- far below large_box_random's 0.5: this series
                                doesn't need a whole-map divergence field, just enough near-
                                obstacle trials mixed with plenty of clear ground for
                                sample_spawn_poses to always succeed)
    --extent FLOAT              full width/height in meters of the square grid (default: 10.0,
                                matching create_large_box_obstacles.py's default -- leaves a
                                healthy clear margin past PatchSpec's ~2.5 m reach even with a
                                box present)
    --cell FLOAT                grid resolution in meters (default: 0.05)
    --incline-deg FLOAT         obstacle ramp incline angle in degrees (default: 75.0)
    --cutoff-wavelength FLOAT   longest rough-terrain wavelength let through, in meters -- see
                                create_rough_terrain.py's module docstring (default: 3.0)
    --min-wavelength FLOAT      shortest rough-terrain wavelength let through, in meters (default:
                                0.6)
    --beta FLOAT                rough-terrain power-spectrum slope S(k) ~ k^-beta (default: 2.5)
    --rms FLOAT                 target RMS elevation of the rough layer alone, in meters (default:
                                0.035 -- same as create_rough_terrain.py's own default)

Usage:
    python src/feasibility/heightmap/create_rough_terrain_plus_boxes.py --seed 0
    python src/feasibility/heightmap/create_rough_terrain_plus_boxes.py --seed 0 --n 200
    python src/feasibility/lattice_learning/generate_dataset.py +maps_dir=assets/rough_box/0
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import yaml

from feasibility.heightmap import HeightMapReader
from feasibility.heightmap.create_large_box_obstacles import build_multi_rect
from feasibility.heightmap.create_large_box_obstacles import sample_rect_sizes
from feasibility.heightmap.create_rough_terrain import build_rough_terrain
from feasibility.heightmap.create_rough_terrain import GROUND_CLEARANCE

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
ASSETS_DIR = REPO_ROOT / "assets" / "rough_box"

DEFAULT_HEIGHT = 0.70  # m -- same platform-scale obstacle height every box generator uses
DEFAULT_EXTENT = 10.0  # m, full width/height of the square grid
DEFAULT_CELL = 0.05  # m, grid resolution
DEFAULT_INCLINE_DEG = 75.0  # deg, obstacle ramp slope
DEFAULT_N_BOXES = 1  # "one random block obstacle" -- see module docstring
DEFAULT_MAX_AREA_FRACTION = 0.12  # far below large_box_random's 0.5 -- see module docstring

DEFAULT_CUTOFF_WAVELENGTH = 3.0  # m -- create_rough_terrain.py's own default
DEFAULT_MIN_WAVELENGTH = 0.6  # m -- create_rough_terrain.py's own default
DEFAULT_BETA = 2.5
DEFAULT_RMS = 0.035  # m -- create_rough_terrain.py's own default

BATCH_INDEX_WIDTH = 4  # zero-padding width for map indices, same convention as
# create_large_box_obstacles.py's large_box_batch_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--seed", type=int, required=True,
        help="RNG seed; also names the assets/rough_box/<seed>/ output subdirectory",
    )
    parser.add_argument("--n", type=int, default=100, help="number of maps to generate (default: 100)")
    parser.add_argument(
        "--n-boxes", type=int, default=DEFAULT_N_BOXES,
        help=f"upper bound on obstacles per map (default: {DEFAULT_N_BOXES})",
    )
    parser.add_argument(
        "--height", type=float, default=DEFAULT_HEIGHT,
        help=f"obstacle height in meters (default: {DEFAULT_HEIGHT})",
    )
    parser.add_argument(
        "--max-area-fraction", type=float, default=DEFAULT_MAX_AREA_FRACTION,
        help=f"max total obstacle+ramp footprint as a fraction of the map area "
        f"(default: {DEFAULT_MAX_AREA_FRACTION})",
    )
    parser.add_argument(
        "--extent", type=float, default=DEFAULT_EXTENT,
        help=f"full width/height in meters of the square grid (default: {DEFAULT_EXTENT})",
    )
    parser.add_argument("--cell", type=float, default=DEFAULT_CELL, help="grid resolution in meters")
    parser.add_argument(
        "--incline-deg", type=float, default=DEFAULT_INCLINE_DEG,
        help="obstacle ramp incline angle in degrees",
    )
    parser.add_argument(
        "--cutoff-wavelength", type=float, default=DEFAULT_CUTOFF_WAVELENGTH,
        help="longest rough-terrain wavelength let through, in meters",
    )
    parser.add_argument(
        "--min-wavelength", type=float, default=DEFAULT_MIN_WAVELENGTH,
        help="shortest rough-terrain wavelength let through, in meters",
    )
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA, help="rough-terrain power-spectrum slope S(k) ~ k^-beta")
    parser.add_argument(
        "--rms", type=float, default=DEFAULT_RMS,
        help="target RMS elevation of the rough layer alone, in meters",
    )
    args = parser.parse_args()
    if args.n_boxes < 1:
        parser.error("--n-boxes must be >= 1")
    if not (0.0 < args.max_area_fraction <= 1.0):
        parser.error("--max-area-fraction must be in (0, 1]")
    if args.min_wavelength < 2.0 * args.cell:
        parser.error(
            f"--min-wavelength {args.min_wavelength} m is below the grid's Nyquist wavelength "
            f"{2.0 * args.cell} m (=2*cell) -- lower --cell or raise --min-wavelength"
        )
    if args.cutoff_wavelength <= args.min_wavelength:
        parser.error(
            f"--cutoff-wavelength {args.cutoff_wavelength} m must be greater than "
            f"--min-wavelength {args.min_wavelength} m -- the passband would be empty"
        )
    return args


def rough_box_batch_path(index: int, seed: int, height: float) -> pathlib.Path:
    """assets/rough_box/<seed>/rough_box_i<index, zero-padded>_h<height, cm, no dot> -- same
    naming convention as create_large_box_obstacles.py's large_box_batch_path, distinct prefix so
    a directory listing is never confused with that series or with plain assets/rough/."""
    return ASSETS_DIR / str(seed) / f"rough_box_i{index:0{BATCH_INDEX_WIDTH}d}_h{round(height * 100):03d}cm"


def place_boxes_randomly(
    sizes: list[tuple[float, float]], extent: float, incline_deg: float, height: float,
    rng: np.random.Generator,
) -> list[tuple[float, float, float, float]]:
    """One random layout of `sizes` (w, d) pairs -- each box's center drawn uniformly within its
    own fits-inside-the-grid bound, independent of every other box. This is
    create_large_box_obstacles.choose_best_layout's per-trial placement with the frontier-score
    search and its best-of-N loop removed -- see module docstring for why a single random draw is
    exactly as good here."""
    ramp = height / np.tan(np.radians(incline_deg))
    half = extent / 2.0
    boxes = []
    for w, d in sizes:
        bx = half - w / 2.0 - ramp
        by = half - d / 2.0 - ramp
        cx = rng.uniform(-bx, bx)
        cy = rng.uniform(-by, by)
        boxes.append((w, d, cx, cy))
    return boxes


def _write_params(path: pathlib.Path, **params: float) -> None:
    """Merges generation params into the .yaml sidecar HeightMapReader.save() just wrote -- same
    pattern create_rough_terrain.py's own (private) helper uses; re-derived rather than imported
    since it's a leading-underscore symbol."""
    yaml_path = path.with_suffix(".yaml")
    meta = yaml.safe_load(yaml_path.read_text())
    meta.update(params)
    yaml_path.write_text(yaml.safe_dump(meta))


def main() -> None:
    args = parse_args()
    (ASSETS_DIR / str(args.seed)).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for i in range(args.n):
        rough_seed = int(rng.integers(0, 2 ** 31 - 1))
        rough = build_rough_terrain(
            args.extent, args.cell, args.cutoff_wavelength, args.min_wavelength, args.beta,
            args.rms, rough_seed,
        )

        sizes = sample_rect_sizes(
            rng, args.n_boxes, args.incline_deg, args.height, args.max_area_fraction, args.extent
        )
        boxes = place_boxes_randomly(sizes, args.extent, args.incline_deg, args.height, rng)
        box_layer = build_multi_rect(args.height, args.cell, args.incline_deg, args.extent, boxes)

        assert rough.H.shape == box_layer.H.shape  # both built on the same extent/cell/origin
        H = rough.H + box_layer.H
        hmap = HeightMapReader(H, origin=(rough.x0, rough.y0), cell=args.cell)

        path = rough_box_batch_path(i, args.seed, args.height)
        hmap.save(path)

        rough_rms_achieved = float(rough.H.std())
        rough_peak_to_peak = float(rough.H.max() - rough.H.min())
        box_area_fraction = float(np.count_nonzero(box_layer.H > 1e-6)) / box_layer.H.size
        combined_peak_to_peak = float(hmap.H.max() - hmap.H.min())
        _write_params(
            path,
            extent=args.extent,
            cell=args.cell,
            seed=args.seed,
            index=i,
            rough_seed=rough_seed,
            cutoff_wavelength=args.cutoff_wavelength,
            min_wavelength=args.min_wavelength,
            beta=args.beta,
            rms_target=args.rms,
            rms_achieved=rough_rms_achieved,
            box_height=args.height,
            box_incline_deg=args.incline_deg,
            box_max_area_fraction=args.max_area_fraction,
            box_area_fraction=box_area_fraction,
            boxes=[[float(v) for v in box] for box in boxes],
        )

        box_desc = " ".join(f"({w:.1f}x{d:.1f} @ x={cx:.2f},y={cy:.2f})" for w, d, cx, cy in boxes)
        print(
            f"saved {path}.png / {path}.yaml  "
            f"(rough seed={rough_seed} rms target/achieved={args.rms:.3f}/{rough_rms_achieved:.3f} m, "
            f"{len(boxes)}/{args.n_boxes} box(es) {box_desc}, box area {box_area_fraction:.1%}, "
            f"combined peak-to-peak {combined_peak_to_peak:.3f} m)"
        )
        if rough_peak_to_peak > GROUND_CLEARANCE:
            print(
                f"WARNING: rough layer alone has peak-to-peak {rough_peak_to_peak:.3f} m, "
                f"exceeding the {GROUND_CLEARANCE:.2f} m chassis ground clearance -- expect "
                "high-centering away from the obstacle too; lower --rms or raise "
                "--cutoff-wavelength"
            )


if __name__ == "__main__":
    main()
