"""Finite-width ramps with sharp foot/top kinks and side drop-offs, for lattice_learning's maps.

The series exists to read ostrich-vs-helhest_stack divergence against a CONTINUOUS slope angle:
every ramp is 0.7 m tall and its rising face is drawn uniformly from 5 deg (an 8 m run) to 80 deg
(a 0.12 m run, wall-like), across helhest_stack's 25 deg climb limit. lattice_learning's
spawn_sampling.py drives every trial on a `ramps` map head-on up a face, so the transitions the
static settle smooths over -- the kink at the foot where the nose meets the slope, the crest, the
side edge of a narrow ramp -- are exactly what those trials see.

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
solids), same as create_curbs_and_walls.py, but never touch: a new ramp's footprint must stay
`overlap_margin` clear of every existing one, since a second ramp on a face breaks the sampler's
"wheels on this face" check.

Placement must not bias the angle distribution. A gentle ramp is long, so re-drawing the whole
ramp whenever it does not fit would quietly favour steep ones. Instead each ramp slot draws its
SHAPE once (angle, plateau, far side, width) and then retries only the PLACEMENT: the rising
face's foot uniformly inside `extent/2 - foot_edge_margin` (spawn_sampling.py starts trials just
before the foot, and its 2.0 m patch must stay on the map there) and a uniform yaw. `max_length`
caps s_end so even a 5 deg ramp has placements that fit; the plateau and a far-side down-ramp are
shortened to respect it. The __main__ check asserts the placed angles are uniform (KS test).

`w` starts at 0.9 m, just above the ~0.83 m wheel-to-wheel width, so narrow ramps put the side
edges under the wheels. `ramp_height` evaluates the closed form at arbitrary points.

`build_ramp_map` does no file IO -- create_maps_for_lattice_learning.py composes it and owns the
output layout. Running this module directly is its smoke test.

CLI parameters:
    --seed INT       RNG seed for the example map (default: 0)
    --extent FLOAT   full width/height of the square grid in meters (default: 14.0)
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
from scipy.ndimage import distance_transform_edt

from feasibility.heightmap import HeightMapReader
from feasibility.heightmap.lattice_maps_utils import GROUND_EPS
from feasibility.heightmap.lattice_maps_utils import grid_axes
from feasibility.heightmap.lattice_maps_utils import place_features

DEFAULT_EXTENT = 14.0  # m -- 2 m more than the other lattice map categories: a 5 deg ramp
# with its 2.1 m platform is up to 10.5 m long and must still place with its foot 2.2 m inside the edge
DEFAULT_CELL = 0.1  # m
MAX_PLACEMENT_ATTEMPTS = 200  # placements per drawn ramp shape before the map stops growing


@dataclasses.dataclass(frozen=True)
class RampsConfig:
    """Sampling ranges for build_ramp_map; every (lo, hi) pair is a uniform draw."""

    n_ramps: tuple[int, int] = (1, 3)  # inclusive; an UPPER bound once placement/area binds
    up_deg: tuple[float, float] = (5.0, 80.0)  # rising face: gentle to wall-like
    height: tuple[float, float] = (0.7, 0.7)  # m, plateau height
    max_length: float = 10.5  # m, cap on s_end (foot to far-side bottom); >= a 5 deg run of 8.0 m
    # plus the plateau floor plus an 80 deg drop's 0.12 m
    plateau: tuple[float, float] = (2.1, 3.0)  # m, flat top length, shortened to fit max_length
    # but never below the floor: a STANDING PLATFORM for lattice_learning's ramp_down trials, which
    # start with the whole robot on the plateau -- spawn_sampling.required_platform_length (2.08 m
    # at the default warm-up); dataset_config.check_maps refuses shorter plateaus for ramp_down.
    width: tuple[float, float] = (0.9, 3.0)  # m, TOP width
    drop_prob: float = 0.5  # chance the far side is a steep drop instead of a down-ramp
    drop_deg: float = 80.0  # far-side angle when it is a drop
    side_deg: float = 80.0  # side edges: drop-offs, not side slopes
    max_area_fraction: float = 0.35  # of the whole map, keeps clear ground for spawn sampling
    overlap_margin: float = 1.0  # m, clear ground between two ramps' footprints
    foot_edge_margin: float = 2.2  # m, the rising face's foot stays this far inside the grid edge
    # (both axes) -- lattice_learning drives trials head-on up the face from just before its foot,
    # and its 2.0 m patch reach must stay on the map there; re-derived, not imported


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
    return ramp_height(ramp, X, Y)


def ramp_height(ramp: Ramp, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Heights of one ramp on flat ground at arbitrary world points (X, Y) -- the closed form in
    the module docstring, evaluated off-grid."""
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


def foot_inside_margin(ramp: Ramp, extent: float, margin: float) -> bool:
    """True when the rising face's foot (the ramp-axis end opposite `yaw`) is at least `margin`
    inside the grid edge along both axes."""
    half = ramp.length / 2.0
    fx = ramp.cx - half * np.cos(ramp.yaw)
    fy = ramp.cy - half * np.sin(ramp.yaw)
    limit = extent / 2.0 - margin
    return abs(fx) <= limit + 1e-9 and abs(fy) <= limit + 1e-9


def sample_ramp_shape(rng: np.random.Generator, cfg: RampsConfig) -> dict:
    """Everything about one ramp except where it stands: the Ramp fields minus cx, cy, yaw. The
    plateau and a far-side down-ramp are drawn inside what `max_length` leaves (not rejected), so
    the rising angle stays exactly uniform."""
    up_deg = rng.uniform(*cfg.up_deg)
    height = rng.uniform(*cfg.height)
    up_run = height / np.tan(np.radians(up_deg))
    drop_run = height / np.tan(np.radians(cfg.drop_deg))
    if up_run + cfg.plateau[0] + drop_run > cfg.max_length:
        raise ValueError(
            f"a {up_deg:.1f} deg, {height:.2f} m ramp needs {up_run + cfg.plateau[0] + drop_run:.2f} m "
            f"> max_length {cfg.max_length} m -- raise max_length or the lower up_deg bound"
        )
    plateau_hi = min(cfg.plateau[1], cfg.max_length - up_run - drop_run)
    plateau = rng.uniform(cfg.plateau[0], max(plateau_hi, cfg.plateau[0]))
    remaining = cfg.max_length - up_run - plateau  # horizontal room left for the far side
    down_floor = np.degrees(np.arctan(height / remaining))
    if rng.uniform() < cfg.drop_prob or down_floor >= cfg.up_deg[1]:
        down_deg = cfg.drop_deg
    else:
        down_deg = rng.uniform(max(cfg.up_deg[0], down_floor), cfg.up_deg[1])
    return {
        "up_deg": float(up_deg),
        "height": float(height),
        "plateau": float(plateau),
        "down_deg": float(down_deg),
        "width": float(rng.uniform(*cfg.width)),
        "side_deg": float(cfg.side_deg),
    }


def place_ramp(rng: np.random.Generator, shape: dict, extent: float, foot_limit: float) -> Ramp:
    """`shape` at a uniform yaw with its rising face's foot uniform in |x|, |y| <= foot_limit."""
    fx, fy = rng.uniform(-foot_limit, foot_limit, size=2)
    yaw = rng.uniform(0.0, 2.0 * np.pi)
    probe = Ramp(cx=0.0, cy=0.0, yaw=0.0, **shape)
    half = probe.length / 2.0
    return Ramp(
        cx=float(fx + half * np.cos(yaw)), cy=float(fy + half * np.sin(yaw)), yaw=float(yaw), **shape
    )


def build_ramp_map(
    rng: np.random.Generator,
    cfg: RampsConfig = RampsConfig(),
    extent: float = DEFAULT_EXTENT,
    cell: float = DEFAULT_CELL,
) -> tuple[HeightMapReader, dict]:
    """One map of ramps on flat ground, plus a params dict for a .yaml sidecar --
    lattice_maps_utils.place_features runs the placement/overlap/area-budget loop. Each ramp slot
    draws its shape once (retrying placement must never bias the angle distribution) and then gets
    up to MAX_PLACEMENT_ATTEMPTS placements; the first shape that cannot be placed stops the map,
    so n_ramps is an upper bound."""
    foot_limit = extent / 2.0 - cfg.foot_edge_margin
    if foot_limit <= 0.0:
        raise ValueError(f"extent {extent} m leaves no room inside the {cfg.foot_edge_margin} m margin")

    def make_attempt():
        shape = sample_ramp_shape(rng, cfg)

        def attempt():
            ramp = place_ramp(rng, shape, extent, foot_limit)
            if not ramp_inside_grid(ramp, extent):
                return None
            return ramp_layer(ramp, extent, cell), ramp

        return attempt

    n_target = int(rng.integers(cfg.n_ramps[0], cfg.n_ramps[1] + 1))
    H, ramps = place_features(
        extent, cell, n_target, cfg.overlap_margin, cfg.max_area_fraction, make_attempt,
        max_attempts=MAX_PLACEMENT_ATTEMPTS,
    )

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
    from scipy.stats import kstest

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

    # --- the extremes of the angle range build and place -----------------------------------------
    extremes = RampsConfig(n_ramps=(1, 1))
    for deg in cfg.up_deg:
        one = dataclasses.replace(extremes, up_deg=(deg, deg))
        _, p = build_ramp_map(np.random.default_rng(args.seed), one, args.extent, args.cell)
        assert len(p["ramps"]) == 1 and np.isclose(p["ramps"][0]["up_deg"], deg), p
        r = p["ramps"][0]
        print(f"[extremes] {deg:.0f} deg: length {r['length']:.2f} m placed at "
              f"({r['cx']:+.2f}, {r['cy']:+.2f})")

    # --- a full random map --------------------------------------------------------------------
    hmap, params = build_ramp_map(np.random.default_rng(args.seed), cfg, args.extent, args.cell)
    assert params["ramps"], "no ramp fit on the map"
    assert hmap.H.min() == 0.0
    assert hmap.H.max() <= max(r["height"] for r in params["ramps"]) + 1e-12
    assert params["area_fraction"] <= cfg.max_area_fraction
    again, _ = build_ramp_map(np.random.default_rng(args.seed), cfg, args.extent, args.cell)
    assert np.array_equal(hmap.H, again.H), "same seed must reproduce the same map"
    desc = ", ".join(
        f"{r['up_deg']:.0f}deg/{r['height']:.2f}m/w{r['width']:.1f}"
        f"{'/drop' if r['down_deg'] == cfg.drop_deg else ''}"
        for r in params["ramps"]
    )
    print(f"[map] seed {args.seed}: {len(params['ramps'])}/{params['n_ramps_target']} ramps ({desc}), "
          f"area {params['area_fraction']:.1%}, max {hmap.H.max():.3f} m")

    # --- many maps: placement keeps the angle uniform, footprints never touch ---------------------
    placed, per_map = [], []
    for i in range(300):
        _, p = build_ramp_map(np.random.default_rng([args.seed, i]), cfg, args.extent, args.cell)
        per_map.append(len(p["ramps"]))
        for r in p["ramps"]:
            ramp = Ramp(**{k: v for k, v in r.items() if k != "length"})
            assert foot_inside_margin(ramp, args.extent, cfg.foot_edge_margin)
            assert ramp.length <= cfg.max_length + 1e-9
            placed.append(r["up_deg"])
        layers = [ramp_layer(Ramp(**{k: v for k, v in r.items() if k != "length"}), args.extent,
                             args.cell) > GROUND_EPS for r in p["ramps"]]
        for a in range(len(layers)):
            for b in range(a + 1, len(layers)):
                gap = distance_transform_edt(~layers[a]) * args.cell
                assert gap[layers[b]].min() > cfg.overlap_margin, "two ramps' footprints touch"
    lo, hi = cfg.up_deg
    ks = kstest(placed, "uniform", args=(lo, hi - lo))
    print(f"[uniform] {len(placed)} ramps on 300 maps (mean {np.mean(per_map):.2f}/map): up_deg "
          f"KS vs U({lo:.0f}, {hi:.0f}) p = {ks.pvalue:.3f}, below 25 deg {np.mean(np.array(placed) < 25):.1%}")
    assert ks.pvalue > 0.01, "placement biases the rising angle"

    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        hmap.save(out)
        print(f"saved {out}.png / {out}.yaml")
    print("all self-checks ok")
