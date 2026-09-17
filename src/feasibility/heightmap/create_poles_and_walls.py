"""Thin poles and thin walls, for lattice_learning's `poles_and_walls` maps -- the sparse,
narrow-obstacle counterpart to create_curbs_and_walls.py's wider curbs/corners/boxes.

Every feature is a thin, near-vertical obstacle 0.1-1.0 m tall (~0.3-2.9 wheel radii): a `pole`
(a thin vertical cylinder -- a post, bollard or sign) or a `wall` (one long, thin rectangle, same
shape as create_curbs_and_walls.py's `wall` kind). A map is up to 2 features
(PolesAndWallsConfig.n_features), each drawn independently -- its own kind, height and, for a
wall, yaw.

Poles are the new shape here: create_large_box_obstacles.build_rect_obstacle has no round
counterpart, so `build_cylinder_obstacle` below is its circular-footprint analogue -- flat top at
`height` over a disc of `radius`, ramped down to ground at `incline_deg` by Euclidean distance to
the disc (build_rect_obstacle's distance is to a RECTANGLE instead; a wall reuses it directly).

Both kinds are steep-sided (80 deg default) and thin on purpose: lattice_learning's patch samples
terrain on a `PATCH_PITCH` (0.125 m) grid, and a feature thinner than that grid's own resolving
power could fall entirely between sample points and be invisible to the network while ostrich
still hits it (the same concern create_curbs_and_walls.py's wall_thickness floor addresses). A
wall's thinnest dimension is floored at 0.15 m, same floor and same rationale as
create_curbs_and_walls.WallsConfig.wall_thickness. A pole is isotropic, so the equivalent floor is
a RADIUS bound rather than a thickness: on a regular sampling grid of pitch `p`, the farthest any
point can be from the nearest grid intersection is the cell's half-diagonal, `p * sqrt(2) / 2`
(worst case: the point sits at the cell center) -- so `MIN_POLE_RADIUS` guarantees at least one
patch sample lands inside the pole's flat top regardless of where the grid happens to fall.

Feature centers, overlap margin, geometry helpers (`rect_inside_grid`, `GROUND_EPS`) and the
placement/overlap/area-budget loop (`place_features`) all come from lattice_maps_utils, shared
with create_curbs_and_walls.build_walls_map and create_ramps.build_ramp_map so none of the three
map builders imports from another.

`build_poles_and_walls_map` does no file IO -- create_maps_for_lattice_learning.py composes it and
owns the output layout. Running this module directly is its smoke test.

CLI parameters:
    --seed INT       RNG seed for the example map (default: 0)
    --extent FLOAT   full width/height of the square grid in meters (default: 12.0)
    --cell FLOAT     grid resolution in meters (default: 0.1)
    --out PATH       if given, save the example map to this stem (.png/.yaml)

Usage:
    python src/feasibility/heightmap/create_poles_and_walls.py
    python src/feasibility/heightmap/create_poles_and_walls.py --seed 3 --out /tmp/poles
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib

import numpy as np
from scipy.ndimage import distance_transform_edt

from feasibility.heightmap import HeightMapReader
from feasibility.heightmap.create_large_box_obstacles import build_rect_obstacle
from feasibility.heightmap.lattice_maps_utils import GROUND_EPS
from feasibility.heightmap.lattice_maps_utils import PLACEMENT_MARGIN
from feasibility.heightmap.lattice_maps_utils import grid_axes
from feasibility.heightmap.lattice_maps_utils import place_features
from feasibility.heightmap.lattice_maps_utils import rect_inside_grid

DEFAULT_EXTENT = 12.0  # m
DEFAULT_CELL = 0.1  # m

FEATURE_KINDS = ("pole", "wall")

PATCH_PITCH = 0.125  # m, lattice_learning's patch sample pitch -- module docstring
MIN_POLE_RADIUS = PATCH_PITCH * np.sqrt(2.0) / 2.0  # m, ~0.0884 -- module docstring


@dataclasses.dataclass(frozen=True)
class PolesAndWallsConfig:
    """Sampling ranges for build_poles_and_walls_map; every (lo, hi) pair is a uniform draw."""

    height: tuple[float, float] = (0.1, 1.0)  # m, per feature
    n_features: tuple[int, int] = (2, 2)  # inclusive; an UPPER bound if placement runs out
    kind_weights: tuple[float, float] = (0.5, 0.5)  # FEATURE_KINDS order: pole, wall
    pole_radius: tuple[float, float] = (0.09, 0.2)  # m; >= MIN_POLE_RADIUS so the 0.125 m patch
    # always sees it
    wall_length: tuple[float, float] = (1.5, 5.0)  # m
    wall_thickness: tuple[float, float] = (0.15, 0.25)  # m, >= 0.15, same floor as
    # create_curbs_and_walls.WallsConfig.wall_thickness
    incline_deg: float = 80.0  # side slope; sharp, but no single-cell mesh sliver
    overlap_margin: float = 1.0  # m, clear ground between two features' footprints
    max_area_fraction: float = 0.3  # of the whole map, keeps clear ground for spawn sampling

    def __post_init__(self) -> None:
        if self.pole_radius[0] < MIN_POLE_RADIUS:
            raise ValueError(
                f"pole_radius[0]={self.pole_radius[0]} m is below MIN_POLE_RADIUS="
                f"{MIN_POLE_RADIUS:.4f} m -- a thinner pole can fall entirely between "
                f"{PATCH_PITCH} m patch samples"
            )


def build_cylinder_obstacle(
    height: float,
    cell: float,
    incline_deg: float,
    extent: float,
    radius: float,
    cx: float,
    cy: float,
) -> HeightMapReader:
    """Flat ground except one circular frustum (a thin pole) centered at (cx, cy): flat top at
    `height` over a disc of `radius`, linear ramp down to ground via Euclidean distance to the
    disc, sloped at `incline_deg` -- the round-footprint analogue of
    create_large_box_obstacles.build_rect_obstacle (whose distance is to a rectangle instead)."""
    half = extent / 2.0
    ramp_width = height / np.tan(np.radians(incline_deg))
    xs = ys = grid_axes(extent, cell)
    X, Y = np.meshgrid(xs, ys)  # [n, n], row=y, col=x
    dist = np.maximum(0.0, np.hypot(X - cx, Y - cy) - radius)
    H = height * np.clip(1.0 - dist / ramp_width, 0.0, 1.0)
    return HeightMapReader(H, origin=(-half, -half), cell=cell)


def circle_inside_grid(
    radius: float, cx: float, cy: float, height: float, incline_deg: float, extent: float
) -> bool:
    """True when the pole's footprint, including its sloped skirt, lies inside the grid --
    create_curbs_and_walls.rect_inside_grid's circular counterpart."""
    skirt = height / np.tan(np.radians(incline_deg))
    reach = radius + skirt
    limit = extent / 2.0
    return abs(cx) + reach <= limit and abs(cy) + reach <= limit


def build_poles_and_walls_map(
    rng: np.random.Generator,
    cfg: PolesAndWallsConfig = PolesAndWallsConfig(),
    extent: float = DEFAULT_EXTENT,
    cell: float = DEFAULT_CELL,
) -> tuple[HeightMapReader, dict]:
    """One map of poles and thin walls on flat ground, plus a params dict for a .yaml sidecar --
    lattice_maps_utils.place_features runs the placement/overlap/area-budget loop; the first
    feature that cannot land stops the map, so n_features is an upper bound."""
    center_limit = extent / 2.0 - PLACEMENT_MARGIN
    if center_limit <= 0.0:
        raise ValueError(f"extent {extent} m leaves no room inside the {PLACEMENT_MARGIN} m margin")
    weights = np.asarray(cfg.kind_weights, dtype=np.float64)
    weights = weights / weights.sum()

    def make_attempt():
        def attempt():
            kind = FEATURE_KINDS[int(rng.choice(len(FEATURE_KINDS), p=weights))]
            height = rng.uniform(*cfg.height)
            cx, cy = rng.uniform(-center_limit, center_limit, size=2)
            if kind == "pole":
                radius = rng.uniform(*cfg.pole_radius)
                if not circle_inside_grid(radius, cx, cy, height, cfg.incline_deg, extent):
                    return None
                layer = build_cylinder_obstacle(height, cell, cfg.incline_deg, extent, radius, cx, cy).H
                yaw, shape = 0.0, {"radius": float(radius)}
            else:
                yaw = rng.uniform(0.0, np.pi)
                length = rng.uniform(*cfg.wall_length)
                thickness = rng.uniform(*cfg.wall_thickness)
                rect = (length, thickness, cx, cy, yaw)
                if not rect_inside_grid(rect, height, cfg.incline_deg, extent):
                    return None
                layer = build_rect_obstacle(
                    height, cell, cfg.incline_deg, extent, length, thickness, cx, cy, yaw
                ).H
                shape = {"length": float(length), "thickness": float(thickness)}
            record = {
                "kind": kind,
                "height": float(height),
                "yaw": float(yaw),
                "center": [float(cx), float(cy)],
                **shape,
            }
            return layer, record

        return attempt

    n_target = int(rng.integers(cfg.n_features[0], cfg.n_features[1] + 1))
    H, features = place_features(
        extent, cell, n_target, cfg.overlap_margin, cfg.max_area_fraction, make_attempt
    )

    half = extent / 2.0
    hmap = HeightMapReader(H, origin=(-half, -half), cell=cell, min_z=0.0, max_z=float(H.max()))
    params = {
        "incline_deg": float(cfg.incline_deg),
        "n_features_target": n_target,
        "features": features,
        "area_fraction": float(np.count_nonzero(H > GROUND_EPS)) / H.size,
    }
    return hmap, params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--extent", type=float, default=DEFAULT_EXTENT)
    parser.add_argument("--cell", type=float, default=DEFAULT_CELL)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    cfg = PolesAndWallsConfig()
    rng0 = np.random.default_rng(1)

    # --- a pole's cross-section is radially symmetric -- checked on a raster fine enough that -----
    # --- bilinear interpolation error from the grid's own (non-rotationally-symmetric) cell -------
    # --- layout is negligible next to the 2e-3 m tolerance -----------------------------------------
    fine_cell = 0.005
    probe = build_cylinder_obstacle(0.5, fine_cell, cfg.incline_deg, 2.0, 0.15, 0.3, -0.2)
    thetas = np.linspace(0.0, 2.0 * np.pi, 9)[:-1]
    for r in (0.05, 0.15, 0.4):
        vals = [
            float(probe.sample(np.array([0.3 + r * np.cos(t)]), np.array([-0.2 + r * np.sin(t)]))[0])
            for t in thetas
        ]
        assert np.ptp(vals) < 2e-3, (r, vals)
    print("[radial] pole cross-section identical at every angle, for several fixed radii")

    # --- pole visibility: the smallest configured radius always catches >= 1 patch sample, -------
    # --- regardless of the patch grid's phase relative to the pole center ------------------------
    pole_r, pole_h = cfg.pole_radius[0], cfg.height[0]
    pole_probe = build_cylinder_obstacle(pole_h, args.cell, cfg.incline_deg, args.extent, pole_r, 0.0, 0.0)
    worst = np.inf
    for _ in range(200):
        ox, oy = rng0.uniform(0.0, PATCH_PITCH, size=2)
        xs = np.arange(-1.0, 1.0, PATCH_PITCH) + ox
        ys = np.arange(-1.0, 1.0, PATCH_PITCH) + oy
        X, Y = np.meshgrid(xs, ys)
        worst = min(worst, float(pole_probe.sample(X.ravel(), Y.ravel()).max()))
    assert worst >= 0.5 * pole_h, (
        f"a {pole_r:.3f} m pole can hide between {PATCH_PITCH} m patch samples (seen {worst:.3f})"
    )
    print(f"[visibility] {pole_r:.3f} m pole radius: worst-case peak seen at {PATCH_PITCH} m patch "
          f"pitch {worst:.3f} / {pole_h:.3f}")

    # --- wall visibility: same check as create_curbs_and_walls.py's thinnest wall -----------------
    thin, wall_h = cfg.wall_thickness[0], cfg.height[0]
    wall_probe = HeightMapReader(
        build_rect_obstacle(wall_h, args.cell, cfg.incline_deg, args.extent, 4.0, thin, 0.0, 0.0, 0.7).H,
        origin=(-args.extent / 2.0, -args.extent / 2.0), cell=args.cell,
    )
    normal = np.array([-np.sin(0.7), np.cos(0.7)])  # across the wall
    worst = np.inf
    for offset in rng0.uniform(0.0, PATCH_PITCH, size=200):
        t = np.arange(-1.0, 1.0, PATCH_PITCH) + offset
        worst = min(worst, float(wall_probe.sample(t * normal[0], t * normal[1]).max()))
    assert worst >= 0.5 * wall_h, f"a {thin} m wall can hide between {PATCH_PITCH} m samples (seen {worst:.3f})"
    print(f"[visibility] {thin} m wall: worst-case peak seen at {PATCH_PITCH} m pitch {worst:.3f} / {wall_h:.3f}")

    # --- a full random map ------------------------------------------------------------------------
    hmap, params = build_poles_and_walls_map(np.random.default_rng(args.seed), cfg, args.extent, args.cell)
    heights = [f["height"] for f in params["features"]]
    assert params["features"], "no feature fit on the map"
    assert hmap.H.min() == 0.0
    assert hmap.H.max() <= max(heights) + 1e-12, "max-merge produced a taller-than-any feature"
    assert all(cfg.height[0] <= h <= cfg.height[1] for h in heights)
    assert params["area_fraction"] <= cfg.max_area_fraction

    # --- kinds over many maps: every kind appears, every map fills to n_features, and no two -----
    # --- features come within overlap_margin of each other ----------------------------------------
    seen: dict[str, int] = {k: 0 for k in FEATURE_KINDS}
    for i in range(40):
        _, p = build_poles_and_walls_map(np.random.default_rng([args.seed, i]), cfg, args.extent, args.cell)
        assert len(p["features"]) == cfg.n_features[1], f"map {i}: {len(p['features'])} features"
        covered = []
        for f in p["features"]:
            if f["kind"] == "pole":
                layer = build_cylinder_obstacle(
                    f["height"], args.cell, cfg.incline_deg, args.extent, f["radius"], *f["center"]
                ).H
            else:
                layer = build_rect_obstacle(
                    f["height"], args.cell, cfg.incline_deg, args.extent, f["length"], f["thickness"],
                    *f["center"], f["yaw"],
                ).H
            covered.append(layer > GROUND_EPS)
        for a in range(len(covered)):
            gap = distance_transform_edt(~covered[a]) * args.cell
            for b in range(a + 1, len(covered)):
                assert gap[covered[b]].min() > cfg.overlap_margin, f"map {i}: features {a},{b} touch"
        for f in p["features"]:
            seen[f["kind"]] += 1
            assert cfg.height[0] <= f["height"] <= cfg.height[1]
            if f["kind"] == "pole":
                assert cfg.pole_radius[0] <= f["radius"] <= cfg.pole_radius[1]
    assert all(seen.values()), seen
    print(f"[kinds] 40 maps: {seen}")
    again, _ = build_poles_and_walls_map(np.random.default_rng(args.seed), cfg, args.extent, args.cell)
    assert np.array_equal(hmap.H, again.H), "same seed must reproduce the same map"
    kinds = ", ".join(f"{f['kind']}@{f['height']:.2f}m" for f in params["features"])
    print(f"[map] seed {args.seed}: {len(heights)}/{params['n_features_target']} features ({kinds}), "
          f"area {params['area_fraction']:.1%}, max {hmap.H.max():.3f} m")

    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        hmap.save(out)
        print(f"saved {out}.png / {out}.yaml")
    print("all self-checks ok")
