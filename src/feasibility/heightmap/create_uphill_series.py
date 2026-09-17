"""An UPHILL-only ramp series for `benchmarks/bench_uphill_nn.py`: flat ground, one rising face at
`up_deg`, then a plateau that runs to the end of the grid, with the planning goal on the plateau.

`create_ramp_series.py`'s full ridges test the climb AND the descent in one crossing, and in ostrich
the descent is what fails first (every 35-60 deg ridge flipped on the way down). This series drops
the descent so the benchmark isolates the one question the steepness sweep can answer cleanly: can
the robot climb a face this steep. Like the ridge, the face spans the full Y width, so there is no
way around it.

Layout (shared by every map, so start and goal are shared too):

    x0 ... start (x0 + margin) ... foot (-run) /face/ crest (x = 0) ... goal (+GOAL_PAST_CREST)
       ... BEYOND_GOAL of plateau ... grid end

* The CREST is pinned at x = 0 on every map and only the foot moves with the angle, so the goal sits
  at one fixed spot on the plateau. The grid is sized once, from the shallowest (longest) face.
* The goal is GOAL_PAST_CREST (2.0 m) past the crest, so the whole 0.75 m wheelbase is up on the
  plateau, not balancing over the edge.
* BEYOND_GOAL (2.0 m) of plateau past the goal: the divergence net's patch looks 2.0 m ahead
  (`lattice_learning.patch.DEFAULT_X_RANGE`) and `HeightMapReader.sample` clamps at the grid edge,
  so without this the patch and settle near the goal would read clamped-edge values, not real ones.

Geometry constants come from `helhest.planning.rampmaps` (height, cell, margin, Y extent, renderable
limit) so this series and helhest_stack's ridge series stay one family. The face is built here
because rampmaps only has the symmetric ridge.

8-bit storage: `HeightMapReader.save` quantizes elevation to height/255 = 2.9 mm. That is checked on
the saved files, the same as in create_ramp_series.py.

Output: `<out-dir>/uphill_a<tenths of deg, 4 digits>.png/.yaml` (a sorted glob returns sweep order),
each .yaml sidecar extended with `up_deg`, `measured_deg` (steepest rendered rise on the float
grid), `height`, `crest_x`, `start` [x, y, yaw] and `goal` [x, y]; plus `manifest.yaml`.

CLI parameters:
    --angles FLOAT...   face angles in deg (default: 5 10 ... 75)
    --out-dir PATH      output directory (default: assets/uphill_series)

Usage:
    python src/feasibility/heightmap/create_uphill_series.py
    python src/feasibility/heightmap/create_uphill_series.py --angles 55 60 65 --out-dir /tmp/uphill
"""
from __future__ import annotations

import argparse
import math
import pathlib

import numpy as np
import yaml
from helhest.planning.rampmaps import DEFAULT_CELL
from helhest.planning.rampmaps import DEFAULT_EXTENT_Y
from helhest.planning.rampmaps import DEFAULT_HEIGHT
from helhest.planning.rampmaps import DEFAULT_MARGIN
from helhest.planning.rampmaps import max_renderable_deg
from helhest.planning.rampmaps import ramp_run

from feasibility.heightmap import HeightMapReader

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
ASSETS_DIR = REPO_ROOT / "assets" / "uphill_series"

DEFAULT_ANGLES = tuple(float(d) for d in range(5, 76, 5))
GOAL_PAST_CREST = 2.0  # [m] goal x relative to the crest -- see module docstring
BEYOND_GOAL = 2.0  # [m] plateau kept past the goal -- the patch's forward reach
SLACK_CELLS = 2  # per grid end, so rounding never pushes start/goal stencils off flat ground


def series_grid(
    shallowest_deg: float,
    height: float = DEFAULT_HEIGHT,
    margin: float = DEFAULT_MARGIN,
    cell: float = DEFAULT_CELL,
) -> tuple[float, int]:
    """(x0, nx) of the one X grid the whole series shares, sized from its shallowest face. x0 is
    a whole number of cells from the crest at x = 0, so every map puts the crest on the same cell
    boundary."""
    behind = ramp_run(shallowest_deg, height) + margin  # crest back to the start
    x0 = -(math.ceil(behind / cell) + SLACK_CELLS) * cell
    x_end = (math.ceil((GOAL_PAST_CREST + BEYOND_GOAL) / cell) + SLACK_CELLS) * cell
    return x0, int(round((x_end - x0) / cell))


def uphill_ramp(
    up_deg: float,
    x0: float,
    nx: int,
    height: float = DEFAULT_HEIGHT,
    extent_y: float = DEFAULT_EXTENT_Y,
    cell: float = DEFAULT_CELL,
) -> HeightMapReader:
    """Flat z = 0, a full-width face rising `height` at `up_deg` to a crest at x = 0, then plateau
    at `height` to the grid end. Raises when the face is too steep for `cell` to render faithfully
    (same guard as rampmaps.bump_ridge) or the grid cannot hold the face."""
    if not 0.0 < up_deg < 90.0:
        raise ValueError(f"up_deg must be in (0, 90) deg, got {up_deg}")
    run = ramp_run(up_deg, height)
    if run < 2.0 * cell:
        raise ValueError(
            f"a {up_deg:.1f} deg face rising {height:.2f} m has a {run * 100:.1f} cm run, under two "
            f"{cell * 100:.0f} cm cells; steepest renderable is "
            f"{max_renderable_deg(height, cell):.1f} deg -- use a finer cell"
        )
    if x0 > -run:
        raise ValueError(f"grid starting at x0={x0:.2f} m cannot hold a {run:.2f} m face")
    ny = int(round(extent_y / cell))
    xs = x0 + (np.arange(nx) + 0.5) * cell  # cell centers
    # np.interp clamps outside the breakpoints: 0 before the foot, `height` past the crest
    profile = np.interp(xs, [-run, 0.0], [0.0, height])
    return HeightMapReader(
        np.tile(profile, (ny, 1)), (x0, -0.5 * ny * cell), cell, min_z=0.0, max_z=height
    )


def start_goal(
    terrain: HeightMapReader, margin: float = DEFAULT_MARGIN
) -> tuple[tuple[float, float, float], tuple[float, float]]:
    """Start pose (x, y, yaw) `margin` in from the low X edge facing +X; goal (x, y) on the
    plateau GOAL_PAST_CREST past the crest. Both centred in Y."""
    y_center = terrain.y0 + 0.5 * terrain.ny * terrain.cell
    return (terrain.x0 + margin, y_center, 0.0), (GOAL_PAST_CREST, y_center)


def uphill_path(out_dir: pathlib.Path, up_deg: float) -> pathlib.Path:
    """<out_dir>/uphill_a0600 -- tenths of a degree, zero-padded, no dot."""
    return out_dir / f"uphill_a{round(up_deg * 10):04d}"


def uphill_series_paths(out_dir: pathlib.Path = ASSETS_DIR) -> list[pathlib.Path]:
    """Every saved map stem in `out_dir`, in sweep order."""
    return [p.with_suffix("") for p in sorted(out_dir.glob("uphill_a*.png"))]


def steepest_rise_deg(terrain: HeightMapReader) -> float:
    """Steepest single-cell rise along +X as an angle [deg] -- what the grid actually holds."""
    return math.degrees(math.atan(float(np.max(np.diff(terrain.H[0]))) / terrain.cell))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--angles", type=float, nargs="+", default=list(DEFAULT_ANGLES))
    parser.add_argument("--out-dir", type=pathlib.Path, default=ASSETS_DIR)
    args = parser.parse_args()

    angles = sorted(args.angles)
    height, margin, cell = DEFAULT_HEIGHT, DEFAULT_MARGIN, DEFAULT_CELL
    x0, nx = series_grid(angles[0], height, margin, cell)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    quantum = height / 255.0

    print(f"{'deg':>5} {'measured':>8} {'max|dH| mm':>10} {'grid':>9}  file")
    for up_deg in angles:
        terrain = uphill_ramp(up_deg, x0, nx, height, cell=cell)
        (sx, sy, syaw), (gx, gy) = start_goal(terrain, margin)
        measured = steepest_rise_deg(terrain)
        # every admitted angle must render exactly (the 2-cell guard in uphill_ramp)
        assert abs(measured - up_deg) < 0.01, f"{up_deg} deg face rendered as {measured:.2f} deg"
        assert terrain.sample(sx, sy) == 0.0, "start is not on flat ground"
        assert sx + cell < -ramp_run(up_deg, height), "start stencil reaches the face foot"
        assert abs(terrain.sample(gx, gy) - height) < 1e-9, "goal is not on the plateau"

        path = uphill_path(args.out_dir, up_deg)
        terrain.save(path)
        loaded = HeightMapReader.load(path)
        err = float(np.abs(loaded.H - terrain.H).max())
        assert (loaded.nx, loaded.ny, loaded.cell) == (terrain.nx, terrain.ny, terrain.cell)
        assert (loaded.x0, loaded.y0) == (terrain.x0, terrain.y0), "origin changed on save"
        assert err <= 0.5 * quantum + 1e-9, f"{path.name}: {err} m exceeds half an 8-bit quantum"

        meta = yaml.safe_load(path.with_suffix(".yaml").read_text())
        meta.update(
            up_deg=float(up_deg), measured_deg=float(measured), height=float(height),
            crest_x=0.0, start=[float(sx), float(sy), float(syaw)], goal=[float(gx), float(gy)],
        )
        path.with_suffix(".yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
        print(f"{up_deg:5.1f} {measured:8.2f} {1e3 * err:10.2f} "
              f"{terrain.nx:4d}x{terrain.ny:<4d}  {path.name}")

    manifest = {
        "source": "feasibility.heightmap.create_uphill_series (uphill_ramp / start_goal)",
        "angles_deg": [float(a) for a in angles],
        "height": float(height), "margin": float(margin), "cell": float(cell),
        "x0": float(x0), "nx": nx, "extent_x": float(nx * cell), "extent_y": float(DEFAULT_EXTENT_Y),
        "crest_x": 0.0, "goal_past_crest": GOAL_PAST_CREST, "beyond_goal": BEYOND_GOAL,
        "start": [float(sx), float(sy), float(syaw)], "goal": [float(gx), float(gy)],
        "quantum_m": float(quantum),
    }
    (args.out_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    print(f"\nwrote {len(angles)} maps + manifest.yaml to {args.out_dir}  "
          f"({nx * cell:.1f} x {DEFAULT_EXTENT_Y:.1f} m at {cell} m; start ({sx:.2f}, {sy:.2f}) "
          f"-> goal ({gx:.2f}, {gy:.2f}) on the plateau)")


if __name__ == "__main__":
    main()
