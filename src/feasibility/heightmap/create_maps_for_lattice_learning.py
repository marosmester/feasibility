"""Generate a mixed batch of heightmaps for lattice_learning/generate_dataset.py, with set ratios.

Four map categories, each built by a helper in this package (this script only samples, mixes and
writes):

    boxes   create_large_box_obstacles.build_box_map, with height drawn per map from a
            continuous range instead of that script's fixed 0.70 m
    walls   create_curbs_and_walls.build_walls_map: curbs, thin walls, L-corners, wall-gaps
    ramps   create_ramps.build_ramp_map: finite-width ramps, sharp kinks, side drop-offs
    rough   create_rough_terrain.build_rough_terrain: exactly flat, or low-amplitude rough

Why these, and not create_large_box_obstacles.py's default 0.70 m boxes: helhest_stack's static
settle already blocks poses on or against obstacles that tall, so the planner never looks up an
edge cost there. The value is in terrain the settle calls feasible but a moving robot handles
badly -- see lattice_learning/design.md. Flat/rough maps are the negatives the network must predict
~0 on.

Ratios and every sampling range live in the CONFIG block below; edit them there, or override just
the ratios with --ratios. Per-category counts use largest-remainder rounding, so they always sum
to exactly --n.

All maps go into ONE flat directory, <out-dir>/<category>_i<NNNN>.png/.yaml, plus manifest.yaml
(seed, ratios, counts, extent, cell, and the full per-category config), so
`generate_dataset.py +maps_dir=<out-dir>` works unchanged. That script draws n_maps at random from
the directory, so the ratios hold on average, and exactly when n_maps equals --n. Each map's
.yaml sidecar also carries the helper's params (feature list, heights, ...); HeightMapReader.load
ignores the extra keys.

Seeding is per map: SeedSequence([seed, category_id, index]). Changing ratios or --n only adds or
removes maps at the END of a category; every existing <category>_i<NNNN> regenerates identically.
The script refuses to write into a directory holding PNGs it would not produce, since
generate_dataset.py would glob those stale maps too.

`base_rms` (off by default) adds a create_rough_terrain layer under boxes/walls/ramps maps, so
steps and ramps are also seen on non-flat ground.

The `category` key in each sidecar is read back by lattice_learning/spawn_sampling.py: `rough`
maps are sampled uniformly, the others are targeted at trials whose arc meets terrain.

CLI parameters:
    --seed INT         RNG seed; REQUIRED -- also names the default output subdirectory
    --n INT            total number of maps (default: 200)
    --ratios STR       comma-separated category=weight, e.g. boxes=0.2,walls=0.3,ramps=0.3,rough=0.2
                       (default: DEFAULT_RATIOS below); normalized, weights must be >= 0
    --extent FLOAT     full width/height of every square map in meters (default: 12.0)
    --cell FLOAT       grid resolution in meters (default: 0.1)
    --out-dir PATH     output directory (default: assets/lattice_maps/<seed>)
    --dry-run          print counts and build one example per category, write nothing

Usage:
    python src/feasibility/heightmap/create_maps_for_lattice_learning.py --seed 0 --dry-run
    python src/feasibility/heightmap/create_maps_for_lattice_learning.py --seed 0 --n 400
    python src/feasibility/heightmap/create_maps_for_lattice_learning.py --seed 0 --ratios walls=1,rough=1
    python src/feasibility/lattice_learning/generate_dataset.py +maps_dir=assets/lattice_maps/0
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
from typing import Callable

import numpy as np
import yaml

from feasibility.heightmap import HeightMapReader
from feasibility.heightmap.create_curbs_and_walls import GROUND_EPS
from feasibility.heightmap.create_curbs_and_walls import WallsConfig
from feasibility.heightmap.create_curbs_and_walls import build_walls_map
from feasibility.heightmap.create_large_box_obstacles import build_box_map
from feasibility.heightmap.create_ramps import RampsConfig
from feasibility.heightmap.create_ramps import build_ramp_map
from feasibility.heightmap.create_rough_terrain import build_rough_terrain

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
ASSETS_DIR = REPO_ROOT / "assets" / "lattice_maps"

# ================================ CONFIG ==========================================================

DEFAULT_RATIOS: dict[str, float] = {"boxes": 0.2, "walls": 0.3, "ramps": 0.3, "rough": 0.2}

DEFAULT_N = 200
DEFAULT_EXTENT = 12.0  # m; generate_dataset.py spawns >= 2.1 m from the edge, leaving ~7.8 m square
DEFAULT_CELL = 0.1  # m

BOXES = {
    "height": (0.05, 0.5),  # m, one height per map (build_box_map shares it across its boxes)
    "incline_deg": 80.0,
    "n_boxes": 4,  # upper bound
    "max_area_fraction": 0.25,  # below build_box_map's 0.5 so spawn sampling keeps clear ground
    "position_trials": 30,
    "base_rms": (0.0, 0.0),  # m; upper bound > 0 adds a rough base layer
}
WALLS = {
    "config": WallsConfig(),  # every range lives in create_curbs_and_walls.WallsConfig
    "base_rms": (0.0, 0.0),
}
RAMPS = {
    "config": RampsConfig(),  # every range lives in create_ramps.RampsConfig
    "base_rms": (0.0, 0.0),
}
ROUGH = {
    "flat_prob": 0.3,  # share of rough-category maps that are exactly flat
    "rms": (0.005, 0.03),  # m
    "cutoff_wavelength": (1.0, 3.0),  # m
    "min_wavelength": 0.6,  # m
    "beta": 2.5,
}
# Rough-base spectrum under boxes/walls/ramps, when their base_rms is enabled.
BASE_ROUGH = {"cutoff_wavelength": 2.0, "min_wavelength": 0.6, "beta": 2.5}

# =================================================================================================

# Stable ids feed the per-map SeedSequence -- append new categories, never reorder.
CATEGORY_IDS: dict[str, int] = {"boxes": 0, "walls": 1, "ramps": 2, "rough": 3}
INDEX_WIDTH = 4


def parse_ratios(text: str) -> dict[str, float]:
    """"boxes=0.2,walls=0.3" -> {"boxes": 0.2, "walls": 0.3, "ramps": 0.0, "rough": 0.0}."""
    ratios = {name: 0.0 for name in CATEGORY_IDS}
    for item in text.split(","):
        name, sep, value = item.partition("=")
        name = name.strip()
        if not sep or name not in CATEGORY_IDS:
            raise ValueError(f"bad ratio {item!r}; expected <category>=<weight>, categories "
                             f"{list(CATEGORY_IDS)}")
        ratios[name] = float(value)
    return ratios


def allocate_counts(ratios: dict[str, float], n: int) -> dict[str, int]:
    """Largest-remainder split of n by the (unnormalized, >= 0) ratios; sums to exactly n."""
    if any(w < 0.0 for w in ratios.values()):
        raise ValueError(f"ratios must be >= 0, got {ratios}")
    total = sum(ratios.values())
    if total <= 0.0:
        raise ValueError("at least one ratio must be positive")
    raw = {name: n * w / total for name, w in ratios.items()}
    counts = {name: int(np.floor(v)) for name, v in raw.items()}
    leftover = n - sum(counts.values())
    by_remainder = sorted(raw, key=lambda name: (-(raw[name] - counts[name]), CATEGORY_IDS[name]))
    for name in by_remainder[:leftover]:
        counts[name] += 1
    return counts


def map_rng(seed: int, category: str, index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, CATEGORY_IDS[category], index]))


def add_rough_base(
    hmap: HeightMapReader, base_rms: tuple[float, float], rng: np.random.Generator, extent: float,
    cell: float,
) -> tuple[HeightMapReader, dict]:
    """Feature map + a create_rough_terrain layer (added, like mend_two_meshes.py), if enabled."""
    if base_rms[1] <= 0.0:
        return hmap, {}
    rms = float(rng.uniform(*base_rms))
    rough_seed = int(rng.integers(0, 2**31 - 1))
    base = build_rough_terrain(extent, cell, rms=rms, seed=rough_seed, **BASE_ROUGH)
    assert base.H.shape == hmap.H.shape
    combined = HeightMapReader(hmap.H + base.H, origin=(hmap.x0, hmap.y0), cell=cell)
    return combined, {"base_rms": rms, "base_rough_seed": rough_seed, **BASE_ROUGH}


def build_boxes(rng: np.random.Generator, extent: float, cell: float) -> tuple[HeightMapReader, dict]:
    height = float(rng.uniform(*BOXES["height"]))
    hmap, params = build_box_map(
        rng, height=height, extent=extent, cell=cell, incline_deg=BOXES["incline_deg"],
        n_boxes=BOXES["n_boxes"], max_area_fraction=BOXES["max_area_fraction"],
        position_trials=BOXES["position_trials"],
    )
    hmap, base = add_rough_base(hmap, BOXES["base_rms"], rng, extent, cell)
    return hmap, {**params, **base, "n_features": len(params["boxes"])}


def build_walls(rng: np.random.Generator, extent: float, cell: float) -> tuple[HeightMapReader, dict]:
    hmap, params = build_walls_map(rng, WALLS["config"], extent, cell)
    hmap, base = add_rough_base(hmap, WALLS["base_rms"], rng, extent, cell)
    return hmap, {**params, **base, "n_features": len(params["features"])}


def build_ramps(rng: np.random.Generator, extent: float, cell: float) -> tuple[HeightMapReader, dict]:
    hmap, params = build_ramp_map(rng, RAMPS["config"], extent, cell)
    hmap, base = add_rough_base(hmap, RAMPS["base_rms"], rng, extent, cell)
    return hmap, {**params, **base, "n_features": len(params["ramps"])}


def build_rough(rng: np.random.Generator, extent: float, cell: float) -> tuple[HeightMapReader, dict]:
    if rng.uniform() < ROUGH["flat_prob"]:
        rms, cutoff, rough_seed = 0.0, 0.0, -1
    else:
        rms = float(rng.uniform(*ROUGH["rms"]))
        cutoff = float(rng.uniform(*ROUGH["cutoff_wavelength"]))
        rough_seed = int(rng.integers(0, 2**31 - 1))
    if rms == 0.0:
        half = extent / 2.0
        n_axis = int(round(extent / cell)) + 1
        hmap = HeightMapReader(np.zeros((n_axis, n_axis)), origin=(-half, -half), cell=cell)
    else:
        hmap = build_rough_terrain(
            extent, cell, cutoff_wavelength=cutoff, min_wavelength=ROUGH["min_wavelength"],
            beta=ROUGH["beta"], rms=rms, seed=rough_seed,
        )
    params = {
        "flat": rms == 0.0,
        "rms": rms,
        "cutoff_wavelength": cutoff,
        "min_wavelength": ROUGH["min_wavelength"],
        "beta": ROUGH["beta"],
        "rough_seed": rough_seed,
        "n_features": 0,
    }
    return hmap, params


BUILDERS: dict[str, Callable[[np.random.Generator, float, float], tuple[HeightMapReader, dict]]] = {
    "boxes": build_boxes,
    "walls": build_walls,
    "ramps": build_ramps,
    "rough": build_rough,
}


def to_plain(value: object) -> object:
    """Dataclasses / tuples / numpy scalars -> plain Python, which yaml.safe_dump accepts."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_plain(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def describe(category: str, name: str, hmap: HeightMapReader, params: dict) -> str:
    covered = float(np.count_nonzero(np.abs(hmap.H - np.median(hmap.H)) > GROUND_EPS)) / hmap.H.size
    return (
        f"{name:<14} {category:<5} features {params['n_features']:>2}  "
        f"relief {hmap.H.max() - hmap.H.min():.3f} m  non-flat {covered:6.1%}"
    )


def write_sidecar_params(path: pathlib.Path, params: dict) -> None:
    """Merge generation params into the .yaml HeightMapReader.save() just wrote -- same pattern as
    create_rough_terrain.py's private _write_params."""
    yaml_path = path.with_suffix(".yaml")
    meta = yaml.safe_load(yaml_path.read_text())
    meta.update(to_plain(params))
    yaml_path.write_text(yaml.safe_dump(meta, sort_keys=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--ratios", type=str, default=None)
    parser.add_argument("--extent", type=float, default=DEFAULT_EXTENT)
    parser.add_argument("--cell", type=float, default=DEFAULT_CELL)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    ratios = parse_ratios(args.ratios) if args.ratios else dict(DEFAULT_RATIOS)
    counts = allocate_counts(ratios, args.n)
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else ASSETS_DIR / str(args.seed)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    print(f"{args.n} maps -> {out_dir}  " + "  ".join(f"{k}={v}" for k, v in counts.items()))

    names = [
        (category, i, f"{category}_i{i:0{INDEX_WIDTH}d}")
        for category in CATEGORY_IDS
        for i in range(counts[category])
    ]

    if args.dry_run:
        for category in CATEGORY_IDS:
            if counts[category]:
                hmap, params = BUILDERS[category](map_rng(args.seed, category, 0), args.extent, args.cell)
                print("example " + describe(category, f"{category}_i0000", hmap, params))
        print("dry run -- nothing written")
        return

    planned = {name for _, _, name in names}
    stale = sorted(p.stem for p in out_dir.glob("*.png") if p.stem not in planned)
    if stale:
        raise SystemExit(
            f"{out_dir} holds {len(stale)} PNG(s) this run would not produce (e.g. {stale[0]}); "
            "generate_dataset.py would glob them too. Remove them or pick another --out-dir."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    for category, i, name in names:
        hmap, params = BUILDERS[category](map_rng(args.seed, category, i), args.extent, args.cell)
        path = out_dir / name
        hmap.save(path)
        write_sidecar_params(
            path, {"category": category, "index": i, "seed": args.seed, "extent": args.extent,
                   "cell": args.cell, **params},
        )
        print(describe(category, name, hmap, params))

    manifest = {
        "seed": args.seed,
        "n": args.n,
        "ratios": ratios,
        "counts": counts,
        "extent": args.extent,
        "cell": args.cell,
        "category_ids": CATEGORY_IDS,
        "config": {"boxes": BOXES, "walls": WALLS, "ramps": RAMPS, "rough": ROUGH,
                   "base_rough": BASE_ROUGH},
    }
    (out_dir / "manifest.yaml").write_text(yaml.safe_dump(to_plain(manifest), sort_keys=False))
    print(f"wrote {len(names)} maps + manifest.yaml to {out_dir}")


if __name__ == "__main__":
    main()
