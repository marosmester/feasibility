"""Curbs, thin walls, L-corners, wall-gaps and boxes of continuous height, for lattice_learning's
`curbs_and_walls` maps.

Every feature is steep-sided (80 deg) and 0.2-1.0 m tall (~0.6-2.9 wheel radii), from an edge a
wheel may still mount to one no wheel can. lattice_learning's spawn_sampling.py samples these maps
with its `edge` strategy: every trial starts near an edge and half of them run into one, climbing
up or driving down, with the arc end NOT required to be settle-feasible -- so the tall end of the
range shows what ostrich does when helhest_stack says `blocked`.

A map is K features, each drawn independently -- its own height, yaw and kind:

    curb    one long rectangle, 0.3-0.8 m wide (a step up onto it, then straight off again)
    wall    one long, thin rectangle
    corner  two walls meeting at a right angle (an L), the sampled center is the corner point
    gap     two parallel walls with a clear passage between them, about the robot's width
    box     a rectangle with both sides >= 1.5 m, so the whole robot fits on top -- the only kind
            a trial can drive DOWN from

Every rectangle is create_large_box_obstacles.build_rect_obstacle (with its `yaw`), and layers
combine with an elementwise MAXIMUM: the union of solid extrusions standing on flat ground, which
stays correct when two features of DIFFERENT heights overlap (a sum would stack them into a tower).

Wall thickness is floored at 0.15 m on purpose. lattice_learning's patch samples terrain every
0.125 m, so anything thinner can fall entirely between two sample points and be invisible to the
network while ostrich still hits it; the __main__ check below asserts a minimum-thickness wall is
always seen.

Feature centers are drawn inside |x|, |y| <= extent/2 - PLACEMENT_MARGIN. lattice_learning keeps
spawn poses ~2.5 m from the map edge (patch reach + warm-up lead) and its edge strategy starts
trials up to 1.5 m from an edge, so a feature placed 1.1 m from the edge is still reachable.
Re-derived, not imported: heightmap/ is shared infrastructure and must not depend on an
experiment tree.

`build_walls_map` does no file IO -- create_maps_for_lattice_learning.py composes it and owns the
output layout. Running this module directly is its smoke test.

CLI parameters:
    --seed INT       RNG seed for the example map (default: 0)
    --extent FLOAT   full width/height of the square grid in meters (default: 12.0)
    --cell FLOAT     grid resolution in meters (default: 0.1)
    --out PATH       if given, save the example map to this stem (.png/.yaml)

Usage:
    python src/feasibility/heightmap/create_curbs_and_walls.py
    python src/feasibility/heightmap/create_curbs_and_walls.py --seed 3 --out /tmp/walls
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib

import numpy as np

from feasibility.heightmap import HeightMapReader
from feasibility.heightmap.create_large_box_obstacles import build_rect_obstacle

DEFAULT_EXTENT = 12.0  # m
DEFAULT_CELL = 0.1  # m
PLACEMENT_MARGIN = 1.1  # m, feature centers stay this far inside the grid edge -- module docstring
MAX_FEATURE_ATTEMPTS = 50  # draws per feature before the map stops growing
GROUND_EPS = 1e-6  # m, a cell above this counts as covered by a feature

FEATURE_KINDS = ("curb", "wall", "corner", "gap", "box")

# (w, d, cx, cy, yaw) of one rectangle in world coordinates, w along its own rotated x axis
Rect = tuple[float, float, float, float, float]


@dataclasses.dataclass(frozen=True)
class WallsConfig:
    """Sampling ranges for build_walls_map; every (lo, hi) pair is a uniform draw."""

    height: tuple[float, float] = (0.2, 1.0)  # m, per feature
    n_features: tuple[int, int] = (3, 8)  # inclusive; an UPPER bound once the area budget binds
    kind_weights: tuple[float, ...] = (0.2, 0.2, 0.2, 0.15, 0.25)  # FEATURE_KINDS order
    curb_length: tuple[float, float] = (2.0, 6.0)  # m
    curb_width: tuple[float, float] = (0.3, 0.8)  # m
    wall_length: tuple[float, float] = (1.5, 5.0)  # m, also each corner arm / gap wall
    wall_thickness: tuple[float, float] = (0.15, 0.25)  # m, >= 0.15 so the 0.125 m patch sees it
    gap_width: tuple[float, float] = (1.1, 1.8)  # m, clear passage between the two gap walls
    box_side: tuple[float, float] = (1.5, 4.0)  # m, each side; >= 1.5 so the robot fits on top
    incline_deg: float = 80.0  # side slope; sharp, but no single-cell mesh sliver
    max_area_fraction: float = 0.3  # of the whole map, keeps clear ground for spawn sampling


def grid_axes(extent: float, cell: float) -> np.ndarray:
    """Cell-center coordinates along one axis, identical to build_rect_obstacle's and
    create_rough_terrain.build_rough_terrain's grid, so layers from all three line up."""
    n = int(round(extent / cell)) + 1
    return -extent / 2.0 + (np.arange(n) + 0.5) * cell


def feature_rects(
    kind: str,
    rng: np.random.Generator,
    cfg: WallsConfig,
    cx: float,
    cy: float,
    yaw: float,
) -> tuple[list[Rect], dict]:
    """The rectangle(s) making up one feature placed at (cx, cy, yaw), plus its shape params.
    Offsets are laid out in the feature's own frame and rotated into the world."""
    c, s = np.cos(yaw), np.sin(yaw)

    def place(w: float, d: float, ox: float, oy: float) -> Rect:
        return (w, d, cx + c * ox - s * oy, cy + s * ox + c * oy, yaw)

    if kind == "curb":
        length, width = rng.uniform(*cfg.curb_length), rng.uniform(*cfg.curb_width)
        return [place(length, width, 0.0, 0.0)], {"length": length, "width": width}
    if kind == "box":
        side_x, side_y = rng.uniform(*cfg.box_side), rng.uniform(*cfg.box_side)
        return [place(side_x, side_y, 0.0, 0.0)], {"side_x": side_x, "side_y": side_y}
    thickness = rng.uniform(*cfg.wall_thickness)
    if kind == "wall":
        length = rng.uniform(*cfg.wall_length)
        return [place(length, thickness, 0.0, 0.0)], {"length": length, "thickness": thickness}
    if kind == "corner":
        arm_x, arm_y = rng.uniform(*cfg.wall_length), rng.uniform(*cfg.wall_length)
        # Both arms start at the corner cell's outer edge, so they share exactly one t x t square.
        rects = [
            place(arm_x, thickness, arm_x / 2.0 - thickness / 2.0, 0.0),
            place(thickness, arm_y, 0.0, arm_y / 2.0 - thickness / 2.0),
        ]
        return rects, {"arm_x": arm_x, "arm_y": arm_y, "thickness": thickness}
    if kind == "gap":
        length, gap = rng.uniform(*cfg.wall_length), rng.uniform(*cfg.gap_width)
        offset = gap / 2.0 + thickness / 2.0
        rects = [place(length, thickness, 0.0, offset), place(length, thickness, 0.0, -offset)]
        return rects, {"length": length, "gap": gap, "thickness": thickness}
    raise ValueError(f"unknown feature kind {kind!r}, expected one of {FEATURE_KINDS}")


def rect_inside_grid(rect: Rect, height: float, incline_deg: float, extent: float) -> bool:
    """True when the rectangle's footprint, including its sloped skirt, lies inside the grid."""
    w, d, cx, cy, yaw = rect
    skirt = height / np.tan(np.radians(incline_deg))
    hx, hy = w / 2.0 + skirt, d / 2.0 + skirt
    c, s = np.cos(yaw), np.sin(yaw)
    reach_x = abs(c) * hx + abs(s) * hy
    reach_y = abs(s) * hx + abs(c) * hy
    limit = extent / 2.0
    return abs(cx) + reach_x <= limit and abs(cy) + reach_y <= limit


def rects_layer(
    rects: list[Rect], height: float, incline_deg: float, extent: float, cell: float
) -> np.ndarray:
    """[ny, nx] heights of one feature: its rectangles max-merged at one shared height."""
    layers = [
        build_rect_obstacle(height, cell, incline_deg, extent, w, d, cx, cy, yaw).H
        for w, d, cx, cy, yaw in rects
    ]
    return np.maximum.reduce(layers)


def build_walls_map(
    rng: np.random.Generator,
    cfg: WallsConfig = WallsConfig(),
    extent: float = DEFAULT_EXTENT,
    cell: float = DEFAULT_CELL,
) -> tuple[HeightMapReader, dict]:
    """One map of curbs/walls/corners/gaps on flat ground, plus a params dict for a .yaml
    sidecar. Each feature gets up to MAX_FEATURE_ATTEMPTS draws to land fully inside the grid
    without pushing the covered area past cfg.max_area_fraction; the first feature that cannot
    stops the map, so n_features is an upper bound."""
    center_limit = extent / 2.0 - PLACEMENT_MARGIN
    if center_limit <= 0.0:
        raise ValueError(f"extent {extent} m leaves no room inside the {PLACEMENT_MARGIN} m margin")
    weights = np.asarray(cfg.kind_weights, dtype=np.float64)
    weights = weights / weights.sum()

    n_axis = grid_axes(extent, cell).size
    H = np.zeros((n_axis, n_axis), dtype=np.float64)
    features: list[dict] = []
    n_target = int(rng.integers(cfg.n_features[0], cfg.n_features[1] + 1))
    for _ in range(n_target):
        for _ in range(MAX_FEATURE_ATTEMPTS):
            kind = FEATURE_KINDS[int(rng.choice(len(FEATURE_KINDS), p=weights))]
            height = rng.uniform(*cfg.height)
            yaw = rng.uniform(0.0, np.pi)
            cx, cy = rng.uniform(-center_limit, center_limit, size=2)
            rects, shape = feature_rects(kind, rng, cfg, cx, cy, yaw)
            if not all(rect_inside_grid(r, height, cfg.incline_deg, extent) for r in rects):
                continue
            candidate = np.maximum(H, rects_layer(rects, height, cfg.incline_deg, extent, cell))
            if np.count_nonzero(candidate > GROUND_EPS) / candidate.size > cfg.max_area_fraction:
                continue
            H = candidate
            features.append(
                {
                    "kind": kind,
                    "height": float(height),
                    "yaw": float(yaw),
                    "center": [float(cx), float(cy)],
                    **{k: float(v) for k, v in shape.items()},
                    "rects": [[float(v) for v in r] for r in rects],  # (w, d, cx, cy, yaw) each
                }
            )
            break
        else:
            break

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
    cfg = WallsConfig()

    # --- a 90 deg yaw is the same wall with length and thickness swapped -----------------------
    rotated = build_rect_obstacle(0.3, args.cell, cfg.incline_deg, args.extent, 3.0, 0.2, 0.4, -0.7,
                                  yaw=np.pi / 2.0).H
    swapped = build_rect_obstacle(0.3, args.cell, cfg.incline_deg, args.extent, 0.2, 3.0, 0.4, -0.7).H
    assert np.allclose(rotated, swapped, atol=1e-9), np.abs(rotated - swapped).max()
    print("[yaw] 90 deg rotated wall == length/thickness-swapped wall")

    # --- the thinnest wall is always seen at the patch's 0.125 m sample pitch --------------------
    thin, cell_patch = cfg.wall_thickness[0], 0.125
    probe = HeightMapReader(
        rects_layer([(4.0, thin, 0.0, 0.0, 0.7)], 0.3, cfg.incline_deg, args.extent, args.cell),
        origin=(-args.extent / 2.0, -args.extent / 2.0), cell=args.cell,
    )
    probe_rng = np.random.default_rng(1)
    normal = np.array([-np.sin(0.7), np.cos(0.7)])  # across the wall
    worst = np.inf
    for offset in probe_rng.uniform(0.0, cell_patch, size=200):
        t = np.arange(-1.0, 1.0, cell_patch) + offset
        worst = min(worst, float(probe.sample(t * normal[0], t * normal[1]).max()))
    assert worst >= 0.5 * 0.3, f"a {thin} m wall can hide between 0.125 m samples (seen {worst:.3f})"
    print(f"[visibility] {thin} m wall: worst-case peak seen at 0.125 m pitch {worst:.3f} / 0.300 m")

    # --- a full random map --------------------------------------------------------------------
    hmap, params = build_walls_map(np.random.default_rng(args.seed), cfg, args.extent, args.cell)
    heights = [f["height"] for f in params["features"]]
    assert params["features"], "no feature fit on the map"
    assert hmap.H.min() == 0.0
    assert hmap.H.max() <= max(heights) + 1e-12, "max-merge produced a taller-than-any feature"
    assert all(cfg.height[0] <= h <= cfg.height[1] for h in heights)
    assert params["area_fraction"] <= cfg.max_area_fraction

    # --- kinds over many maps: every kind appears, every box top fits the robot -----------------
    seen: dict[str, int] = {k: 0 for k in FEATURE_KINDS}
    for i in range(40):
        _, p = build_walls_map(np.random.default_rng([args.seed, i]), cfg, args.extent, args.cell)
        for f in p["features"]:
            seen[f["kind"]] += 1
            assert cfg.height[0] <= f["height"] <= cfg.height[1]
            if f["kind"] == "box":
                assert min(f["side_x"], f["side_y"]) >= cfg.box_side[0]
    assert all(seen.values()), seen
    print(f"[kinds] 40 maps: {seen}")
    again, _ = build_walls_map(np.random.default_rng(args.seed), cfg, args.extent, args.cell)
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
