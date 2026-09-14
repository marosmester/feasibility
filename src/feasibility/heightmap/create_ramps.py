"""Finite-width ramps with sharp foot/top kinks and side drop-offs, for lattice_learning's maps.

helhest_stack's static settle handles a ramp's FACE well: the settled body's pitch is the slope,
and the tilt cost charges it. What the settle smooths over are the transitions -- the kink at the
foot where the nose meets the slope, the crest where the front wheel goes over the top, the side
edge of a narrow ramp where one wheel drops off and the body rolls. This series is built out of
exactly those, so they carry the divergence signal lattice_learning's network is meant to find.

One ramp is a closed-form piecewise-linear solid. In its own frame, with s along the ramp axis
measured from the foot and t across it:

    z = max(0, min( tan(up) * s,                      rising face
                    H,                                plateau
                    tan(down) * (s_end - s),          far side: a second ramp, or a steep drop
                    H + tan(side) * (w/2 - |t|) ))    side edges

    s_end = H/tan(up) + plateau + H/tan(down)

The minimum of linear pieces makes every kink sharp by construction (no fillet). `w` is the TOP
width: the side skirt flares outward below the plateau exactly the way
create_large_box_obstacles.build_rect_obstacle's frustum skirt does, so a steep `side` angle
reads as a drop-off. Several ramps on one map combine with an elementwise MAXIMUM (the union of
solids), same as create_curbs_and_walls.py.

Ranges are chosen against RobotParams: `up` spans 5-30 deg across the 25 deg climb limit, and `w`
starts at 0.9 m, just above the ~0.83 m wheel-to-wheel width, so narrow ramps put the side edges
under the wheels. Ramp centers use create_curbs_and_walls.PLACEMENT_MARGIN (see that module) and
every footprint must lie fully inside the grid.

`build_ramp_map` does no file IO -- create_maps_for_lattice_learning.py composes it and owns the
output layout. Running this module directly is its smoke test.

CLI parameters:
    --seed INT       RNG seed for the example map (default: 0)
    --extent FLOAT   full width/height of the square grid in meters (default: 12.0)
    --cell FLOAT     grid resolution in meters (default: 0.1)
    --out PATH       if given, save the example map to this stem (.png/.yaml)

Usage:
    python src/feasibility/heightmap/create_ramps.py
    python src/feasibility/heightmap/create_ramps.py --seed 3 --out /tmp/ramp
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib

import numpy as np

from feasibility.heightmap import HeightMapReader
from feasibility.heightmap.create_curbs_and_walls import GROUND_EPS
from feasibility.heightmap.create_curbs_and_walls import PLACEMENT_MARGIN
from feasibility.heightmap.create_curbs_and_walls import grid_axes

DEFAULT_EXTENT = 12.0  # m
DEFAULT_CELL = 0.1  # m
MAX_RAMP_ATTEMPTS = 50  # draws per ramp before the map stops growing


@dataclasses.dataclass(frozen=True)
class RampsConfig:
    """Sampling ranges for build_ramp_map; every (lo, hi) pair is a uniform draw."""

    n_ramps: tuple[int, int] = (1, 3)  # inclusive; an UPPER bound once placement/area binds
    up_deg: tuple[float, float] = (5.0, 30.0)  # rising face, spans the 25 deg climb limit
    height: tuple[float, float] = (0.2, 1.0)  # m, plateau height, capped by max_run below
    max_run: float = 3.0  # m, longest horizontal run of a sloped face
    plateau: tuple[float, float] = (0.5, 2.5)  # m, flat top length along the axis
    width: tuple[float, float] = (0.9, 3.0)  # m, TOP width
    drop_prob: float = 0.5  # chance the far side is a steep drop instead of a down-ramp
    drop_deg: float = 80.0  # far-side angle when it is a drop
    side_deg: float = 80.0  # side edges: drop-offs, not side slopes
    max_area_fraction: float = 0.35  # of the whole map, keeps clear ground for spawn sampling


@dataclasses.dataclass(frozen=True)
class Ramp:
    """One ramp's geometry, placed: (cx, cy) is the midpoint of its footprint along the axis."""

    cx: float
    cy: float
    yaw: float  # rad, direction of travel UP the rising face
    up_deg: float
    height: float
    plateau: float
    down_deg: float
    width: float
    side_deg: float

    @property
    def length(self) -> float:
        """s_end: foot of the rising face to the bottom of the far side, in meters."""
        up, down = np.radians(self.up_deg), np.radians(self.down_deg)
        return self.height / np.tan(up) + self.plateau + self.height / np.tan(down)

    @property
    def half_width(self) -> float:
        """Half the footprint width at ground level, skirt included."""
        return self.width / 2.0 + self.height / np.tan(np.radians(self.side_deg))


def ramp_layer(ramp: Ramp, extent: float, cell: float) -> np.ndarray:
    """[ny, nx] heights of one ramp on flat ground -- the closed form in the module docstring."""
    axis = grid_axes(extent, cell)
    X, Y = np.meshgrid(axis, axis)  # row = y, col = x
    c, s_yaw = np.cos(ramp.yaw), np.sin(ramp.yaw)
    s = c * (X - ramp.cx) + s_yaw * (Y - ramp.cy) + ramp.length / 2.0
    t = -s_yaw * (X - ramp.cx) + c * (Y - ramp.cy)
    z = np.minimum.reduce(
        [
            np.tan(np.radians(ramp.up_deg)) * s,
            np.full_like(s, ramp.height),
            np.tan(np.radians(ramp.down_deg)) * (ramp.length - s),
            ramp.height + np.tan(np.radians(ramp.side_deg)) * (ramp.width / 2.0 - np.abs(t)),
        ]
    )
    return np.maximum(z, 0.0)


def ramp_inside_grid(ramp: Ramp, extent: float) -> bool:
    """True when the ramp's whole ground footprint lies inside the grid."""
    hx, hy = ramp.length / 2.0, ramp.half_width
    c, s = abs(np.cos(ramp.yaw)), abs(np.sin(ramp.yaw))
    limit = extent / 2.0
    return abs(ramp.cx) + c * hx + s * hy <= limit and abs(ramp.cy) + s * hx + c * hy <= limit


def sample_ramp(rng: np.random.Generator, cfg: RampsConfig, center_limit: float) -> Ramp:
    """One random ramp. Height is drawn directly inside what max_run allows at the drawn slope
    (not rejected), and a down-ramp's angle is floored so its run respects max_run too."""
    up_deg = rng.uniform(*cfg.up_deg)
    height_cap = min(cfg.height[1], cfg.max_run * np.tan(np.radians(up_deg)))
    if height_cap < cfg.height[0]:
        raise ValueError(
            f"max_run {cfg.max_run} m at {up_deg:.1f} deg cannot reach the minimum height "
            f"{cfg.height[0]} m -- raise max_run or the lower up_deg bound"
        )
    height = rng.uniform(cfg.height[0], height_cap)
    if rng.uniform() < cfg.drop_prob:
        down_deg = cfg.drop_deg
    else:
        down_floor = np.degrees(np.arctan(height / cfg.max_run))
        down_deg = rng.uniform(max(cfg.up_deg[0], down_floor), cfg.up_deg[1])
    cx, cy = rng.uniform(-center_limit, center_limit, size=2)
    return Ramp(
        cx=float(cx),
        cy=float(cy),
        yaw=float(rng.uniform(0.0, 2.0 * np.pi)),
        up_deg=float(up_deg),
        height=float(height),
        plateau=float(rng.uniform(*cfg.plateau)),
        down_deg=float(down_deg),
        width=float(rng.uniform(*cfg.width)),
        side_deg=float(cfg.side_deg),
    )


def build_ramp_map(
    rng: np.random.Generator,
    cfg: RampsConfig = RampsConfig(),
    extent: float = DEFAULT_EXTENT,
    cell: float = DEFAULT_CELL,
) -> tuple[HeightMapReader, dict]:
    """One map of ramps on flat ground, plus a params dict for a .yaml sidecar. Each ramp gets up
    to MAX_RAMP_ATTEMPTS draws to fit inside the grid without pushing the covered area past
    cfg.max_area_fraction; the first that cannot stops the map, so n_ramps is an upper bound."""
    center_limit = extent / 2.0 - PLACEMENT_MARGIN
    if center_limit <= 0.0:
        raise ValueError(f"extent {extent} m leaves no room inside the {PLACEMENT_MARGIN} m margin")
    n_axis = grid_axes(extent, cell).size
    H = np.zeros((n_axis, n_axis), dtype=np.float64)
    ramps: list[Ramp] = []
    n_target = int(rng.integers(cfg.n_ramps[0], cfg.n_ramps[1] + 1))
    for _ in range(n_target):
        for _ in range(MAX_RAMP_ATTEMPTS):
            ramp = sample_ramp(rng, cfg, center_limit)
            if not ramp_inside_grid(ramp, extent):
                continue
            candidate = np.maximum(H, ramp_layer(ramp, extent, cell))
            if np.count_nonzero(candidate > GROUND_EPS) / candidate.size > cfg.max_area_fraction:
                continue
            H = candidate
            ramps.append(ramp)
            break
        else:
            break

    half = extent / 2.0
    max_z = float(H.max())
    hmap = HeightMapReader(H, origin=(-half, -half), cell=cell, min_z=0.0, max_z=max_z)
    params = {
        "n_ramps_target": n_target,
        "ramps": [{**dataclasses.asdict(r), "length": float(r.length)} for r in ramps],
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
    cfg = RampsConfig()
    axis = grid_axes(args.extent, args.cell)

    # --- one axis-aligned ramp, checked against its own closed form -----------------------------
    probe = Ramp(cx=0.0, cy=0.0, yaw=0.0, up_deg=15.0, height=0.6, plateau=1.5, down_deg=80.0,
                 width=1.0, side_deg=80.0)
    Hp = ramp_layer(probe, args.extent, args.cell)
    row = int(np.argmin(np.abs(axis)))  # t ~ 0
    foot_x = -probe.length / 2.0
    up_run = probe.height / np.tan(np.radians(probe.up_deg))
    face = (axis > foot_x + args.cell) & (axis < foot_x + up_run - args.cell)
    slope = np.diff(Hp[row, face]) / args.cell
    assert np.allclose(slope, np.tan(np.radians(probe.up_deg)), atol=1e-9), slope
    print(f"[face] along-axis slope {slope.mean():.4f} == tan(15 deg) {np.tan(np.radians(15)):.4f}")

    plateau_col = int(np.argmin(np.abs(axis - (foot_x + up_run + probe.plateau / 2.0))))
    assert np.isclose(Hp[row, plateau_col], probe.height), Hp[row, plateau_col]
    print(f"[plateau] height {Hp[row, plateau_col]:.3f} m == H")

    X, Y = np.meshgrid(axis, axis)
    outside = (np.abs(X) > probe.length / 2.0 + args.cell) | (np.abs(Y) > probe.half_width + args.cell)
    assert np.all(Hp[outside] == 0.0), "ramp height leaked outside its footprint"
    across = Hp[:, plateau_col]
    assert np.isclose(across[row], probe.height)
    assert across[np.abs(axis) > probe.half_width].max() == 0.0
    edge = (np.abs(axis) > probe.width / 2.0) & (np.abs(axis) < probe.half_width)
    print(f"[sides] plateau cross-section: {probe.height:.2f} m on top, drops to 0 within "
          f"{probe.half_width - probe.width / 2.0:.3f} m of the edge ({int(edge.sum())} skirt cells)")

    # --- a full random map --------------------------------------------------------------------
    hmap, params = build_ramp_map(np.random.default_rng(args.seed), cfg, args.extent, args.cell)
    assert params["ramps"], "no ramp fit on the map"
    assert hmap.H.min() == 0.0
    assert hmap.H.max() <= max(r["height"] for r in params["ramps"]) + 1e-12
    assert params["area_fraction"] <= cfg.max_area_fraction
    for r in params["ramps"]:
        assert r["height"] / np.tan(np.radians(r["up_deg"])) <= cfg.max_run + 1e-9
    again, _ = build_ramp_map(np.random.default_rng(args.seed), cfg, args.extent, args.cell)
    assert np.array_equal(hmap.H, again.H), "same seed must reproduce the same map"
    desc = ", ".join(
        f"{r['up_deg']:.0f}deg/{r['height']:.2f}m/w{r['width']:.1f}"
        f"{'/drop' if r['down_deg'] == cfg.drop_deg else ''}"
        for r in params["ramps"]
    )
    print(f"[map] seed {args.seed}: {len(params['ramps'])}/{params['n_ramps_target']} ramps ({desc}), "
          f"area {params['area_fraction']:.1%}, max {hmap.H.max():.3f} m")

    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        hmap.save(out)
        print(f"saved {out}.png / {out}.yaml")
    print("all self-checks ok")
