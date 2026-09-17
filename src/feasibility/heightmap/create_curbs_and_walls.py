"""Curbs, thin walls, L-corners and boxes of continuous height, for lattice_learning's
`curbs_and_walls` maps.

Every feature is steep-sided (80 deg) and 0.2-1.0 m tall (~0.6-2.9 wheel radii), from an edge a
wheel may still mount to one no wheel can. lattice_learning's spawn_sampling.py samples these maps
with its `edge` strategy: every trial starts near an edge and half of them run into one, climbing
up or driving down, with the arc end NOT required to be settle-feasible -- so the tall end of the
range shows what ostrich does when helhest_stack says `blocked`.

A map is 2 features (WallsConfig.n_features), each drawn independently -- its own height, yaw and
kind:

    curb    one long rectangle, 0.3-0.8 m wide (a step up onto it, then straight off again)
    wall    one long, thin rectangle
    corner  two walls meeting at a right angle (an L), the sampled center is the corner point
    box     a rectangle with both sides >= 1.5 m, so the whole robot fits on top -- the only kind
            a trial can drive DOWN from

Few features, kept apart: the footprints of two features (sloped skirts included) stay more than
WallsConfig.overlap_margin (1.0 m) clear of each other, so each one stays a recognizable shape
instead of merging into a clutter of overlapping walls and boxes.

Every rectangle is create_large_box_obstacles.build_rect_obstacle (with its `yaw`), and the
rectangles of one feature (a corner's two arms) combine with an elementwise MAXIMUM.

Wall thickness is floored at 0.15 m on purpose. lattice_learning's patch samples terrain every
0.125 m, so anything thinner can fall entirely between two sample points and be invisible to the
network while ostrich still hits it; the __main__ check below asserts a minimum-thickness wall is
always seen.

Feature centers are drawn inside |x|, |y| <= extent/2 - PLACEMENT_MARGIN (lattice_maps_utils; see
its docstring for the margin's rationale). The placement/overlap/area-budget loop itself is
lattice_maps_utils.place_features, shared with create_poles_and_walls.py and create_ramps.py so
none of the three imports from another.

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
from scipy.ndimage import distance_transform_edt

from feasibility.heightmap import HeightMapReader
from feasibility.heightmap.create_large_box_obstacles import build_rect_obstacle
from feasibility.heightmap.lattice_maps_utils import GROUND_EPS
from feasibility.heightmap.lattice_maps_utils import PLACEMENT_MARGIN
from feasibility.heightmap.lattice_maps_utils import Rect
from feasibility.heightmap.lattice_maps_utils import place_features
from feasibility.heightmap.lattice_maps_utils import rect_inside_grid

DEFAULT_EXTENT = 12.0  # m
DEFAULT_CELL = 0.1  # m

FEATURE_KINDS = ("curb", "wall", "corner", "box")


@dataclasses.dataclass(frozen=True)
class WallsConfig:
    """Sampling ranges for build_walls_map; every (lo, hi) pair is a uniform draw."""

    height: tuple[float, float] = (0.2, 1.0)  # m, per feature
    n_features: tuple[int, int] = (2, 2)  # inclusive; an UPPER bound if placement runs out
    kind_weights: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)  # FEATURE_KINDS order
    curb_length: tuple[float, float] = (2.0, 6.0)  # m
    curb_width: tuple[float, float] = (0.3, 0.8)  # m
    wall_length: tuple[float, float] = (1.5, 5.0)  # m, also each corner arm
    wall_thickness: tuple[float, float] = (0.15, 0.25)  # m, >= 0.15 so the 0.125 m patch sees it
    box_side: tuple[float, float] = (1.5, 4.0)  # m, each side; >= 1.5 so the robot fits on top
    incline_deg: float = 80.0  # side slope; sharp, but no single-cell mesh sliver
    overlap_margin: float = 1.0  # m, clear ground between two features' footprints
    max_area_fraction: float = 0.3  # of the whole map, keeps clear ground for spawn sampling


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
    raise ValueError(f"unknown feature kind {kind!r}, expected one of {FEATURE_KINDS}")


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
    """One map of curbs/walls/corners/boxes on flat ground, plus a params dict for a .yaml
    sidecar -- lattice_maps_utils.place_features runs the placement/overlap/area-budget loop; the
    first feature that cannot land stops the map, so n_features is an upper bound."""
    center_limit = extent / 2.0 - PLACEMENT_MARGIN
    if center_limit <= 0.0:
        raise ValueError(f"extent {extent} m leaves no room inside the {PLACEMENT_MARGIN} m margin")
    weights = np.asarray(cfg.kind_weights, dtype=np.float64)
    weights = weights / weights.sum()

    def make_attempt():
        def attempt():
            kind = FEATURE_KINDS[int(rng.choice(len(FEATURE_KINDS), p=weights))]
            height = rng.uniform(*cfg.height)
            yaw = rng.uniform(0.0, np.pi)
            cx, cy = rng.uniform(-center_limit, center_limit, size=2)
            rects, shape = feature_rects(kind, rng, cfg, cx, cy, yaw)
            if not all(rect_inside_grid(r, height, cfg.incline_deg, extent) for r in rects):
                return None
            layer = rects_layer(rects, height, cfg.incline_deg, extent, cell)
            record = {
                "kind": kind,
                "height": float(height),
                "yaw": float(yaw),
                "center": [float(cx), float(cy)],
                **{k: float(v) for k, v in shape.items()},
                "rects": [[float(v) for v in r] for r in rects],  # (w, d, cx, cy, yaw) each
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

    # --- kinds over many maps: every kind appears, every box top fits the robot, every map is -----
    # --- filled to n_features, and no two features come within overlap_margin of each other -----
    seen: dict[str, int] = {k: 0 for k in FEATURE_KINDS}
    for i in range(40):
        _, p = build_walls_map(np.random.default_rng([args.seed, i]), cfg, args.extent, args.cell)
        assert len(p["features"]) == cfg.n_features[1], f"map {i}: {len(p['features'])} features"
        covered = [
            rects_layer(f["rects"], f["height"], cfg.incline_deg, args.extent, args.cell) > GROUND_EPS
            for f in p["features"]
        ]
        for a in range(len(covered)):
            gap = distance_transform_edt(~covered[a]) * args.cell
            for b in range(a + 1, len(covered)):
                assert gap[covered[b]].min() > cfg.overlap_margin, f"map {i}: features {a},{b} touch"
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
