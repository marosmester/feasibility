"""STAIR family: a lower floor where the robot starts, an upper platform with the goal on it, and
one rectangular step between them -- the only way up. For the premise question: does vanilla MPPI
take the short DIAGONAL line over the step (red), and does ostrich punish that where the twin does
not, against the longer way round that climbs it head-on (green)?

Bird's eye (x right, y up):

        upper platform, z = 2 h          goal X
    ---------------+----------+-------- y = 0  (platform face, 2 h tall beside the stair)
                   |  stair   |   tread at z = h, `--width` wide, `--depth` deep
                   +----------+-------- y = -depth
        floor z = 0      /
                        /  red: start -> goal, a straight line through the tread centre at
             robot  o--     `--alpha` off perpendicular; the start faces along it

  * Every face is `common.INCLINE_DEG` (80 deg) like the other families, built with
    `create_curbs_and_walls.rects_layer` and merged by elementwise max.
  * Start and goal sit `--reach` either side of the tread centre on that line, so the red line
    crosses both stair edges inside the stair's width (asserted, with a wheel's half-track to spare).
  * There is no ctrl map: removing the stair removes the only route. The sidecar instead records
    the cost-to-go V at the start, so it is checked that a route exists at all.

Maps: `stair_h<cm>_a<deg>`.

CLI parameters:
    --heights M ...   step height h (the platform is 2 h) (default 0.15 0.20 0.25)
    --alphas D ...    approach angle off perpendicular [deg] (default 30 45)
    --width M         stair width along x (default 2.4)
    --depth M         tread depth along y (default 1.2: the 1.10 m wheelbase-plus-rim fits on it)
    --reach M         start/goal distance from the tread centre along the line (default 2.6)
    --extent M        square map side (default 10.0)
    --cell M          grid resolution (default 0.05)
    --out-dir DIR     output directory (default assets/mppi_eval/stair)

Usage:
    python src/feasibility/heightmap/mppi_eval/create_stair.py
"""
from __future__ import annotations

import argparse
import math
import pathlib

import numpy as np
import warp as wp
import yaml
from helhest import dynamics

from feasibility.heightmap import HeightMapReader
from feasibility.heightmap.create_curbs_and_walls import rects_layer
from feasibility.heightmap.mppi_eval.common import DEVICE
from feasibility.heightmap.mppi_eval.common import INCLINE_DEG
from feasibility.heightmap.mppi_eval.common import start_value
from feasibility.heightmap.mppi_eval.common import write_manifest
from feasibility.lattice_learning.settle import settle_batch
from feasibility.lattice_learning.settle import settle_feasible

FAMILY = "stair"
ASSETS_DIR = pathlib.Path("assets/mppi_eval/stair")
MU = 0.8
PIVOT_COST = 0.15  # closed_loop's flags for this family: point turns allowed (the green way may need one) ...
SPIN_FRAC = 0.1  # ... and MPPI's SPIN prior
HEIGHTS = (0.15, 0.20, 0.25)
ALPHAS = (30.0, 45.0)  # 60 needs a stair wider than 2.8 m


def build_stair(height: float, width: float, depth: float, extent: float, cell: float) -> HeightMapReader:
    """Floor 0 for y < -depth, the tread `height` over |x| < width / 2, the platform 2 * height for y > 0."""
    half = extent / 2.0
    platform = (extent + 2.0, half + 1.0, 0.0, (half + 1.0) / 2.0, 0.0)  # (w, d, cx, cy, yaw), past the map edges
    tread = (width, depth, 0.0, -depth / 2.0, 0.0)
    H = np.maximum(rects_layer([platform], 2.0 * height, INCLINE_DEG, extent, cell),
                   rects_layer([tread], height, INCLINE_DEG, extent, cell))
    return HeightMapReader(H, (-half, -half), cell)


def red_line(alpha: float, depth: float, reach: float) -> tuple[tuple[float, float, float], tuple[float, float]]:
    """(start pose, goal) on the straight line through the tread centre, `alpha` off the +y climb."""
    ux, uy = math.sin(alpha), math.cos(alpha)
    cx, cy = 0.0, -depth / 2.0
    return (cx - reach * ux, cy - reach * uy, math.atan2(uy, ux)), (cx + reach * ux, cy + reach * uy)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--heights", type=float, nargs="+", default=list(HEIGHTS))
    parser.add_argument("--alphas", type=float, nargs="+", default=list(ALPHAS))
    parser.add_argument("--width", type=float, default=2.4)
    parser.add_argument("--depth", type=float, default=1.2)
    parser.add_argument("--reach", type=float, default=2.6)
    parser.add_argument("--extent", type=float, default=10.0)
    parser.add_argument("--cell", type=float, default=0.05)
    parser.add_argument("--out-dir", type=pathlib.Path, default=ASSETS_DIR)
    args = parser.parse_args()
    wp.init()
    robot = dynamics.robot_params()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for alpha_deg in args.alphas:
        alpha = math.radians(alpha_deg)
        start, goal = red_line(alpha, args.depth, args.reach)
        # the red line crosses both stair edges (y = -depth and y = 0) inside the width, a half-track to spare
        cross = args.depth / 2.0 * math.tan(alpha)  # both edges are depth / 2 from the tread centre
        assert cross + float(robot.half_track) < args.width / 2.0, f"alpha {alpha_deg}: the red line misses the stair"
        assert goal[1] > float(robot.rear_offset) + float(robot.wheel_radius), "the goal is not on the platform"
        for height in args.heights:
            terrain = build_stair(height, args.width, args.depth, args.extent, args.cell)
            derived, residual, clearance = settle_batch(terrain, np.array([start]), MU, DEVICE)
            assert settle_feasible(derived, residual, clearance, robot)[0], "the settle blocks the start pose"
            v, cap = start_value(terrain, start, goal, PIVOT_COST)
            v0, _ = start_value(terrain, start, goal, 0.0)
            assert v < cap, f"h {height}: the cost-to-go finds no way up (V {v:.1f}, cap {cap:.1f})"
            stem = f"{FAMILY}_h{round(height * 100):03d}_a{alpha_deg:03.0f}"
            terrain.save(args.out_dir / stem)
            sidecar_path = (args.out_dir / stem).with_suffix(".yaml")
            sidecar = yaml.safe_load(sidecar_path.read_text())
            sidecar.update(source="feasibility.heightmap.mppi_eval.create_stair", family=FAMILY, variant="with",
                           start=list(start), goal=list(goal), spin_frac=SPIN_FRAC, pivot_cost=PIVOT_COST,
                           step_height=height, alpha_deg=alpha_deg, width=args.width, depth=args.depth,
                           reach=args.reach, v_start=v, v_start_pc0=v0, v_cap=cap)
            sidecar_path.write_text(yaml.safe_dump(sidecar, sort_keys=False))
            rows.append(dict(stem=stem, step_height=height, alpha_deg=alpha_deg, v_start=v, v_start_pc0=v0))
            print(f"[{stem}] start {np.round(start, 2).tolist()} goal {np.round(goal, 2).tolist()} | "
                  f"V start {v:.2f} (no pivots {v0:.2f}, cap {cap:.1f})")
    write_manifest(args.out_dir, FAMILY, rows)
    print(f"wrote {len(rows)} maps to {args.out_dir}")


if __name__ == "__main__":
    main()
