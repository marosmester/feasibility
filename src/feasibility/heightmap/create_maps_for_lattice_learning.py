"""Generate a mixed batch of heightmaps for lattice_learning/generate_dataset.py, with set ratios.

Four map categories, each built by a helper in this package (this script only samples, mixes and
writes). lattice_learning/generate_dataset.py reads the sidecar `category` back and applies the
spawn_sampling strategies its dataset YAML's `mix` assigns to that category (see
lattice_learning/dataset_config.py for which strategy is eligible where):

    category         builder                                   strategies (lattice_learning)
    ramps            create_ramps.build_ramp_map: 0.7 m tall,  `ramp_up`: head-on up a face;
                     finite-width ramps, rising face 5-80 deg, `ramp_down`: from the plateau, head-
                     plateau >= 2.1 m (a standing platform),   on down a face; ramp_deg per trial
                     14 m maps
    curbs_and_walls  create_curbs_and_walls.build_walls_map:   `edge`: near an edge, some of them
                     2 of curb/wall/L-corner/box, 1 m apart,   running into it (climb up or drive
                     0.2-1.0 m tall, 80 deg sides              down)
    poles_and_walls  create_poles_and_walls.build_poles_and_    `edge`: same as curbs_and_walls --
                     walls_map: up to 2 of pole/wall, 1 m       edge_field is found from the
                     apart, 0.1-1.0 m tall, 80 deg sides        heightmap alone, no sidecar needed
    rough            create_rough_terrain.build_rough_terrain: `uniform`
                     exactly flat, or low-amplitude rough

The ramps span a continuous slope so the angle where ostrich and helhest_stack start to diverge
can be read off; the curbs/walls and poles/walls go from edges a wheel can mount to ones none can
(the latter narrower and sparser -- a pole or a lone thin wall rather than curbs, corners and
boxes); flat/rough maps are the negatives the network must predict ~0 on. On ramps,
curbs_and_walls and poles_and_walls maps a trial's arc end need not be settle-feasible -- the
dataset flags those rows `endpoint_blocked`.

Ratios and every sampling range live in the CONFIG block below; edit them there, or override just
the ratios with --ratios. Per-category counts use largest-remainder rounding, so they always sum
to exactly --n.

All maps go into ONE flat directory, <out-dir>/<category>_i<NNNN>.png/.yaml, plus manifest.yaml
(seed, ratios, counts, per-category extent, cell, and the full per-category config), which a
dataset YAML's `maps.dir` points at. generate_dataset.py draws each category's maps from here in
the proportions of its own `mix`, so these ratios only need to leave every category enough maps;
it also checks each ramp's sidecar plateau before allowing `ramp_down`. Each map's
.yaml sidecar also carries the helper's params (feature list, heights, ...); HeightMapReader.load
ignores the extra keys.

Seeding is per map: SeedSequence([seed, category_id, index]). Changing ratios or --n only adds or
removes maps at the END of a category; every existing <category>_i<NNNN> regenerates identically.
The script refuses to write into a directory holding PNGs it would not produce, since
generate_dataset.py would glob those stale maps too.

`base_rms` (off by default) adds a create_rough_terrain layer under ramps/curbs_and_walls/
poles_and_walls maps, so steps and ramps are also seen on non-flat ground. It breaks the ramp
sampler's "wheels on this face's surface" check (every ramp trial would fall back), so leave it
off for ramps.

CLI parameters:
    --seed INT         RNG seed; REQUIRED -- also names the default output subdirectory
    --n INT            total number of maps (default: 200)
    --ratios STR       comma-separated category=weight, e.g.
                       ramps=1,curbs_and_walls=1,poles_and_walls=1,rough=1
                       (default: DEFAULT_RATIOS below); normalized, weights must be >= 0
    --extent FLOAT     full width/height of every square map in meters, overriding every category's
                       own CONFIG "extent" (default: per category -- ramps 14.0, others 12.0)
    --cell FLOAT       grid resolution in meters (default: 0.1)
    --out-dir PATH     output directory (default: assets/lattice_maps/<seed>)
    --dry-run          print counts and build one example per category, write nothing

Usage:
    python src/feasibility/heightmap/create_maps_for_lattice_learning.py --seed 0 --dry-run
    python src/feasibility/heightmap/create_maps_for_lattice_learning.py --seed 0 --n 400
    python src/feasibility/heightmap/create_maps_for_lattice_learning.py --seed 0 --ratios ramps=1
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
from typing import Callable

import numpy as np
import yaml

from feasibility.heightmap import HeightMapReader
from feasibility.heightmap.create_curbs_and_walls import WallsConfig
from feasibility.heightmap.create_curbs_and_walls import build_walls_map
from feasibility.heightmap.create_poles_and_walls import PolesAndWallsConfig
from feasibility.heightmap.create_poles_and_walls import build_poles_and_walls_map
from feasibility.heightmap.create_ramps import RampsConfig
from feasibility.heightmap.create_ramps import build_ramp_map
from feasibility.heightmap.create_ramps import DEFAULT_EXTENT as RAMPS_DEFAULT_EXTENT
from feasibility.heightmap.create_rough_terrain import build_rough_terrain
from feasibility.heightmap.lattice_maps_utils import GROUND_EPS

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
ASSETS_DIR = REPO_ROOT / "assets" / "lattice_maps"

# ================================ CONFIG ==========================================================

DEFAULT_RATIOS: dict[str, float] = {
    "ramps": 0.25, "curbs_and_walls": 0.25, "poles_and_walls": 0.25, "rough": 0.25,
}

DEFAULT_N = 200
DEFAULT_EXTENT = 12.0  # m, curbs_and_walls and rough; generate_dataset.py spawns >= 2.1 m from the
# edge, leaving ~7.8 m square
DEFAULT_CELL = 0.1  # m

RAMPS = {
    "config": RampsConfig(),  # every range lives in create_ramps.RampsConfig: 0.7 m, 5-80 deg
    "base_rms": (0.0, 0.0),  # m; upper bound > 0 adds a rough base layer
    "extent": RAMPS_DEFAULT_EXTENT,  # m, 14: a 5 deg ramp plus its standing platform is 10.5 m long
}
CURBS_AND_WALLS = {
    "config": WallsConfig(),  # every range lives in create_curbs_and_walls.WallsConfig: 0.2-1.0 m
    "base_rms": (0.0, 0.0),
    "extent": DEFAULT_EXTENT,
}
POLES_AND_WALLS = {
    "config": PolesAndWallsConfig(),  # every range lives in create_poles_and_walls.
    # PolesAndWallsConfig: 0.1-1.0 m, up to 2 of pole/wall
    "base_rms": (0.0, 0.0),
    "extent": DEFAULT_EXTENT,
}
ROUGH = {
    "flat_prob": 0.3,  # share of rough-category maps that are exactly flat
    "rms": (0.005, 0.03),  # m
    "cutoff_wavelength": (1.0, 3.0),  # m
    "min_wavelength": 0.6,  # m
    "beta": 2.5,
    "extent": DEFAULT_EXTENT,
}
# Rough-base spectrum under ramps/curbs_and_walls, when their base_rms is enabled.
BASE_ROUGH = {"cutoff_wavelength": 2.0, "min_wavelength": 0.6, "beta": 2.5}

# =================================================================================================

# Stable ids feed the per-map SeedSequence -- append new categories, never reorder or reuse an id.
# Retired: 0 = boxes (create_large_box_obstacles.build_box_map), 1 = walls (0.05-0.5 m curbs and
# walls, superseded by curbs_and_walls).
CATEGORY_IDS: dict[str, int] = {"ramps": 2, "rough": 3, "curbs_and_walls": 4, "poles_and_walls": 5}
INDEX_WIDTH = 4


def parse_ratios(text: str) -> dict[str, float]:
    """"ramps=0.5,rough=0.5" -> {"ramps": 0.5, "rough": 0.5, "curbs_and_walls": 0.0}."""
    ratios = {name: 0.0 for name in CATEGORY_IDS}
    for item in text.split(","):
        name, sep, value = item.partition("=")
        name = name.strip()
        if not sep or name not in CATEGORY_IDS:
            raise ValueError(f"bad ratio {item!r}; expected <category>=<weight>, categories "
                             f"{list(CATEGORY_IDS)}")
        ratios[name] = float(value)
    return ratios


def largest_remainder(weights: list[float], total: int) -> list[int]:
    """Non-negative ints proportional to the (unnormalized, >= 0) `weights`, summing to exactly
    `total`; ties go to the earlier weight. Also lattice_learning/dataset_config.py's allocation."""
    w = np.asarray(weights, dtype=np.float64)
    if (w < 0.0).any():
        raise ValueError(f"weights must be >= 0, got {weights}")
    if w.sum() <= 0.0:
        raise ValueError("at least one weight must be positive")
    raw = total * w / w.sum()
    counts = np.floor(raw).astype(int)
    order = sorted(range(len(w)), key=lambda i: (-(raw[i] - counts[i]), i))
    for i in order[: total - int(counts.sum())]:
        counts[i] += 1
    return counts.tolist()


def allocate_counts(ratios: dict[str, float], n: int) -> dict[str, int]:
    """Largest-remainder split of n by the (unnormalized, >= 0) ratios; sums to exactly n. Ties go
    to the lower category id."""
    names = sorted(ratios, key=CATEGORY_IDS.__getitem__)
    counts = dict(zip(names, largest_remainder([ratios[k] for k in names], n)))
    return {name: counts[name] for name in ratios}


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


def build_curbs_and_walls(
    rng: np.random.Generator, extent: float, cell: float
) -> tuple[HeightMapReader, dict]:
    hmap, params = build_walls_map(rng, CURBS_AND_WALLS["config"], extent, cell)
    hmap, base = add_rough_base(hmap, CURBS_AND_WALLS["base_rms"], rng, extent, cell)
    return hmap, {**params, **base, "n_features": len(params["features"])}


def build_ramps(rng: np.random.Generator, extent: float, cell: float) -> tuple[HeightMapReader, dict]:
    hmap, params = build_ramp_map(rng, RAMPS["config"], extent, cell)
    hmap, base = add_rough_base(hmap, RAMPS["base_rms"], rng, extent, cell)
    return hmap, {**params, **base, "n_features": len(params["ramps"])}


def build_poles_and_walls(
    rng: np.random.Generator, extent: float, cell: float
) -> tuple[HeightMapReader, dict]:
    hmap, params = build_poles_and_walls_map(rng, POLES_AND_WALLS["config"], extent, cell)
    hmap, base = add_rough_base(hmap, POLES_AND_WALLS["base_rms"], rng, extent, cell)
    return hmap, {**params, **base, "n_features": len(params["features"])}


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
    "ramps": build_ramps,
    "curbs_and_walls": build_curbs_and_walls,
    "poles_and_walls": build_poles_and_walls,
    "rough": build_rough,
}
CATEGORY_CONFIGS: dict[str, dict] = {
    "ramps": RAMPS, "curbs_and_walls": CURBS_AND_WALLS, "poles_and_walls": POLES_AND_WALLS, "rough": ROUGH,
}


def category_extent(category: str, override: float | None) -> float:
    """m -- `--extent` when given (every category), else the category's own CONFIG "extent"."""
    return float(override) if override is not None else float(CATEGORY_CONFIGS[category]["extent"])


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
        f"{name:<24} {category:<15} features {params['n_features']:>2}  "
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
    parser.add_argument("--extent", type=float, default=None)
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
                hmap, params = BUILDERS[category](
                    map_rng(args.seed, category, 0), category_extent(category, args.extent), args.cell
                )
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
        extent = category_extent(category, args.extent)
        hmap, params = BUILDERS[category](map_rng(args.seed, category, i), extent, args.cell)
        path = out_dir / name
        hmap.save(path)
        write_sidecar_params(
            path, {"category": category, "index": i, "seed": args.seed, "extent": extent,
                   "cell": args.cell, **params},
        )
        print(describe(category, name, hmap, params))

    manifest = {
        "seed": args.seed,
        "n": args.n,
        "ratios": ratios,
        "counts": counts,
        "extent": {c: category_extent(c, args.extent) for c in CATEGORY_IDS},
        "cell": args.cell,
        "category_ids": CATEGORY_IDS,
        "config": {**CATEGORY_CONFIGS, "base_rough": BASE_ROUGH},
    }
    (out_dir / "manifest.yaml").write_text(yaml.safe_dump(to_plain(manifest), sort_keys=False))
    print(f"wrote {len(names)} maps + manifest.yaml to {out_dir}")


if __name__ == "__main__":
    main()
