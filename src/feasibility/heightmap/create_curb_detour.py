"""One map with a LOW CURB across the straight route and a clear way round its end, for
`nn_mppi/closed_loop.py` (mppi_learning/design.md section 9d.6): a map where avoiding the twin's
error means taking a different route, not only driving slower.

Layout (x forward from start to goal, the curb across it at x = 0):

    start (-5, 0, 0) ....... curb, from past the -y map edge up to y = gap_y ....... goal (5, 0)
                              above gap_y: flat ground, the detour

The curb is `create_curbs_and_walls`'s kind (one rectangle with 80 deg sides, `build_rect_obstacle`),
`--width` deep along x. Its height is what the map is about, and it has a window:

  * Low enough that helhest's SETTLE accepts every head-on crossing pose. Otherwise the cost-to-go
    routes round it and vanilla MPPI detours on its own. A crossing pitches the body by about
    atan(h / rear_offset), and with the rear wheel still on the curb that is nose-DOWN, where the
    limit is max_pitch_down, 15 deg. It was measured at 11.5 deg for 0.15 m, and 0.20 m is already
    blocked (15.5 deg).
  * So the default is 0.15 m. The cost-to-go then routes straight over it: V at the start equals a
    curb with no gap at all, against +1.6 for a forced detour at gap_y 1.0. That is 0.43 wheel
    radii, just under the 0.2 m floor of the training curbs (`create_curbs_and_walls`), so it is a
    mild extrapolation for the net.

The main block asserts both (settle and cost-to-go, on the GPU): if either fails, the map cannot
separate the arms.

Output: `<out-dir>/curb_detour_h<cm, 3 digits>_g<gap_y in dm, 2 digits>.png/.yaml`. The sidecar
adds `start`, `goal`, `height`, `gap_y`, `width`, `curb_x`, and the two cost-to-go values it checked.

CLI parameters:
    --height M     curb height (default 0.15)
    --gap-y M      the curb's +y end; the straight line y = 0 crosses it when > 0 (default 1.0)
    --width M      curb depth along x, footprint (default 0.5, inside the training curbs' 0.3-0.8)
    --extent M     square map side (default 14.0)
    --cell M       grid resolution (default 0.05)
    --out-dir DIR  output directory (default assets/curb_detour)

Usage:
    python src/feasibility/heightmap/create_curb_detour.py
    python src/feasibility/heightmap/create_curb_detour.py --height 0.12 --gap-y 0.5
"""
from __future__ import annotations

import argparse
import math
import pathlib

import numpy as np
import warp as wp
import yaml
from helhest import dynamics
from helhest.engine import GridParams
from helhest.planning.costtogo import CostToGo

from feasibility.heightmap import HeightMapReader
from feasibility.heightmap.create_large_box_obstacles import build_rect_obstacle
from feasibility.lattice_learning.settle import settle_batch
from feasibility.lattice_learning.settle import settle_feasible

ASSETS_DIR = pathlib.Path("assets/curb_detour")
START = (-5.0, 0.0, 0.0)
GOAL = (5.0, 0.0)
INCLINE_DEG = 80.0  # create_curbs_and_walls's side slope
BEYOND_EDGE = 1.0  # [m] the curb runs this far past the -y map edge, so there is no way round that side
# the premise checks use the planner closed_loop.py builds: the mppi_learning labels' twin (k_turn
# 1.0), its cost-to-go cell and robust margin, forward arcs only
K_TURN = 1.0
ROUTING_CELL = 0.32
ROBUST_MARGIN = 0.3
MU = 0.8
DEVICE = "cuda:0"


def build_curb_detour(height: float, gap_y: float, width: float, extent: float, cell: float) -> HeightMapReader:
    """Flat ground plus one curb across x = 0, from BEYOND_EDGE past the -y edge to y = gap_y."""
    length = extent / 2.0 + BEYOND_EDGE + gap_y
    return build_rect_obstacle(height, cell, INCLINE_DEG, extent, width, length, 0.0, gap_y - length / 2.0)


def start_value(terrain: HeightMapReader) -> float:
    """The cost-to-go at START (heading bin 0), on the map max-pooled to ~ROUTING_CELL as
    closed_loop.routing_field does."""
    robot = dynamics.robot_params()
    k = max(1, round(ROUTING_CELL / terrain.cell))
    ny, nx = terrain.ny // k, terrain.nx // k
    coarse = terrain.H[: ny * k, : nx * k].reshape(ny, k, nx, k).max(axis=(1, 3))
    c = terrain.cell * k
    ctg = CostToGo(GridParams(nx, ny, c, terrain.x0, terrain.y0), robot, dynamics.planning_solver(k_turn=K_TURN),
                   n_theta=24, robust_margin_m=ROBUST_MARGIN, pivot_cost=0.0, device=DEVICE)
    V = ctg.compute(wp.array(np.ascontiguousarray(coarse, np.float32), dtype=wp.float32, device=DEVICE), GOAL).numpy()
    i, j = int((START[1] - terrain.y0) / c), int((START[0] - terrain.x0) / c)
    return float(V[i, j, 0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--height", type=float, default=0.15)
    parser.add_argument("--gap-y", type=float, default=1.0)
    parser.add_argument("--width", type=float, default=0.5)
    parser.add_argument("--extent", type=float, default=14.0)
    parser.add_argument("--cell", type=float, default=0.05)
    parser.add_argument("--out-dir", type=pathlib.Path, default=ASSETS_DIR)
    args = parser.parse_args()
    wp.init()
    robot = dynamics.robot_params()
    terrain = build_curb_detour(args.height, args.gap_y, args.width, args.extent, args.cell)

    # --- geometry --------------------------------------------------------------------------------
    for name, (x, y) in (("start", START[:2]), ("goal", GOAL)):
        assert terrain.sample(x, y) == 0.0, f"{name} is not on flat ground"
    assert args.gap_y > 0.0 and abs(terrain.sample(0.0, 0.0) - args.height) < 1e-9, "the straight route misses the curb"
    # a robot centre that clears the curb end: half track + the settle's wheel envelope + margin
    pass_y = args.gap_y + float(robot.half_track) + float(robot.wheel_radius) + 0.2
    assert pass_y + 1.5 < args.extent / 2.0, f"no room round the curb end (pass at y {pass_y:.2f})"
    assert terrain.sample(0.0, pass_y) == 0.0, "the detour is not flat"

    # --- premise 1: the settle accepts every head-on crossing pose -------------------------------
    xs = np.arange(-1.5, 1.5, 0.025)
    for yaw_deg in (-15.0, 0.0, 15.0):
        poses = np.stack([xs, np.zeros_like(xs), np.full_like(xs, math.radians(yaw_deg))], -1)
        derived, residual, clearance = settle_batch(terrain, poses, MU, DEVICE)
        feasible = settle_feasible(derived, residual, clearance, robot)
        assert feasible.all(), (
            f"the settle blocks {int((~feasible).sum())} crossing poses at yaw {yaw_deg:+.0f} deg (pitch up to "
            f"{math.degrees(np.abs(derived[:, 1]).max()):.1f} deg) -- the cost-to-go routes round the curb and "
            "vanilla detours on its own; lower --height")
        print(f"[settle] yaw {yaw_deg:+.0f} deg: all {len(xs)} crossing poses feasible, "
              f"|pitch| <= {math.degrees(np.abs(derived[:, 1]).max()):.1f} deg")

    # --- premise 2: the cost-to-go routes over the curb, not round it -----------------------------
    v_gap = start_value(terrain)
    v_cross = start_value(build_curb_detour(args.height, args.extent, args.width, args.extent, args.cell))
    v_detour = start_value(build_curb_detour(1.0, args.gap_y, args.width, args.extent, args.cell))
    print(f"[route]  V(start): this map {v_gap:.2f}, curb with no gap {v_cross:.2f}, 1 m wall with the same gap {v_detour:.2f}")
    assert abs(v_gap - v_cross) < 1e-3, "the cost-to-go already prefers the detour -- vanilla would take it too"
    assert v_detour > v_gap, "the detour is not longer than the crossing"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"curb_detour_h{round(args.height * 100):03d}_g{round(args.gap_y * 10):02d}"
    terrain.save(stem)
    meta = yaml.safe_load(stem.with_suffix(".yaml").read_text())
    meta.update(
        source="feasibility.heightmap.create_curb_detour", height=args.height, gap_y=args.gap_y, width=args.width,
        curb_x=0.0, start=list(START), goal=list(GOAL), v_start=v_gap, v_start_forced_detour=v_detour,
    )
    stem.with_suffix(".yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    print(f"wrote {stem}.png/.yaml ({terrain.nx}x{terrain.ny} @ {args.cell} m)")


if __name__ == "__main__":
    main()
