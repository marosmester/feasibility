"""TURN-SPOT family: the robot must turn round in a dead-end corridor, and the spot it stands on is
the one place a point turn drags its rear tyre into a low curb -- for the premise check
(`nn_mppi/premise_check.py`) before any nn-MPPI data is made for it.

Layout (x along the corridor, the robot at the origin facing the closed end, the goal behind it):

    goal              open end                                     closed end
     X  .........  +---------------------------------------------------+
                   |          ====== curb ======                       |   tall walls (WALL_H)
                   |            R  -> start         |  turning bay  |  |
                   |          ====== curb ======                       |
                   +---------------------------------------------------+

  * The corridor (3.2-3.6 m) is too narrow for a forward U-turn but wide enough for a point turn
    on the cost-to-go closed_loop.py plans with (0.32 m cells, 0.3 m robust margin: measured, a
    point turn needs >= 3.2 m and a U-turn >= 4.0 m), so the only way out is a turn in place -- asserted: the
    cost-to-go finds no route with `pivot_cost` 0.
  * Low curbs line both walls in the START section only. Their inner edge (0.93-0.98 m) sits between the
    front rims' straight-driving reach (half_track + wheel_radius = 0.715 m), so driving along the
    centreline never touches them, and the rear rim's pivot reach (rear_offset + wheel_radius =
    1.10 m), so a point turn at the start sweeps the rear tyre into them. The rear wheel CENTRE
    (0.75 m) stays off them, and the curb is low enough that the settle accepts every pivot pose
    (the rear wheel lift pitches the body ~atan(h / rear_offset), inside max_pitch_down).
  * Ahead lies a curb-free TURNING BAY: a point turn there touches nothing. Vanilla MPPI (blind to
    the curb) should turn at the start, and an nn-MPPI that sees the drag should drive on first.

A draw that fails a premise assert is redrawn from the same seed's stream (and printed). Each
instance is written twice, `turn_spot_s<seed>` and `turn_spot_s<seed>_ctrl` (curbs removed),
and the cost-to-go is asserted identical on both (the twin is blind to the curb).

CLI parameters:
    --n INT        instances (seeds --seed .. --seed + n - 1, default 8)
    --seed INT     first seed (default 0)
    --extent M     square map side (default 12.0)
    --cell M       grid resolution (default 0.05)
    --out-dir DIR  output directory (default assets/mppi_eval/turn_spot)

Usage:
    python src/feasibility/heightmap/mppi_eval/create_turn_spot.py
    python src/feasibility/heightmap/mppi_eval/create_turn_spot.py --n 16 --seed 100
"""
from __future__ import annotations

import argparse
import math
import pathlib

import numpy as np
import warp as wp
from helhest import dynamics

from feasibility.heightmap.create_garage import wheels_of
from feasibility.heightmap.mppi_eval.common import build_terrain
from feasibility.heightmap.mppi_eval.common import DEVICE
from feasibility.heightmap.mppi_eval.common import distance_field
from feasibility.heightmap.mppi_eval.common import sample_field
from feasibility.heightmap.mppi_eval.common import start_value
from feasibility.heightmap.mppi_eval.common import write_manifest
from feasibility.heightmap.mppi_eval.common import write_pair
from feasibility.lattice_learning.settle import settle_batch
from feasibility.lattice_learning.settle import settle_feasible

FAMILY = "turn_spot"
FAMILY_ID = 1  # in the rng's SeedSequence, so the two families never share a stream
ASSETS_DIR = pathlib.Path("assets/mppi_eval/turn_spot")
START = (0.0, 0.0, 0.0)
WALL_H = 0.70  # [m] the settle blocks poses beside it
WALL_T = 0.30
MU = 0.8
PIVOT_COST = 0.15  # the closed_loop flags this family needs: point turns in the cost-to-go ...
SPIN_FRAC = 0.1  # ... and MPPI's SPIN prior
# drawn per seed
# [m] clear corridor width. Measured on closed_loop's cost-to-go (0.32 m cells, 0.3 m robust
# margin): a point turn needs >= 3.2 m, a forward U-turn >= 4.0 m.
WIDTH = (3.2, 3.6)
CURB_H = (0.10, 0.17)  # [m] 0.20 pitches the rear-wheel lift to the max_pitch_down edge (create_garage)
# [m] curb inner edge from the centreline. The cost-to-go max-pools the map to 0.30 m cells whose
# edges fall at y = 0.6, 0.9, ..., so an edge under 0.90 m (ramp included) lands the curb in the
# cell beside the centreline and RAISES V there (measured +0.26..0.45): the twin would no longer be
# blind. Above 0.93 m V is identical to the ctrl map; the rear tyre still reaches 0.12-0.17 m over.
CURB_EDGE = (0.93, 0.98)
CURB_BACK = (1.3, 1.6)  # [m] curb section behind the start ...
CURB_FWD = (1.3, 1.6)  # [m] ... and ahead of it: the rear rim's whole orbit plus drift
BAY = (2.7, 3.2)  # [m] curb-free corridor between the curb section and the closed end
GOAL_X = (-5.0, -4.0)  # [m] the goal, beyond the open end
OPEN_PAST_CURB = 0.4  # [m] corridor beyond the curb section on the open side
# premise margins
TYRE_OVER = 0.10  # [m] the start pivot's rear tyre must reach this far over the curb
BAY_CLEAR = 0.15  # [m] and the bay pivot must stay this far off it
MAX_DRAWS = 20


def draw(rng: np.random.Generator) -> dict:
    u = lambda lo_hi: float(rng.uniform(*lo_hi))
    return dict(width=u(WIDTH), curb_h=u(CURB_H), curb_edge=u(CURB_EDGE), curb_back=u(CURB_BACK),
                curb_fwd=u(CURB_FWD), bay=u(BAY), goal_x=u(GOAL_X))


def rects(p: dict) -> tuple[list, list]:
    """(walls, curbs) as `(w, d, cx, cy, yaw)`, w along x."""
    half = p["width"] / 2.0
    x_open = -(p["curb_back"] + OPEN_PAST_CURB)
    x_end = p["curb_fwd"] + p["bay"]  # inner face of the closed end
    length = x_end + WALL_T - x_open
    walls = [
        (length, WALL_T, (x_open + x_end + WALL_T) / 2.0, half + WALL_T / 2.0, 0.0),
        (length, WALL_T, (x_open + x_end + WALL_T) / 2.0, -(half + WALL_T / 2.0), 0.0),
        (WALL_T, p["width"] + 2.0 * WALL_T, x_end + WALL_T / 2.0, 0.0, 0.0),
    ]
    span = p["curb_back"] + p["curb_fwd"]
    cx, depth = (p["curb_fwd"] - p["curb_back"]) / 2.0, half - p["curb_edge"]
    curbs = [(span, depth, cx, p["curb_edge"] + depth / 2.0, 0.0),
             (span, depth, cx, -(p["curb_edge"] + depth / 2.0), 0.0)]
    return walls, curbs


def pivot_wheels_at(x: float, n: int = 97) -> np.ndarray:
    """[n, 3, 2] wheel centres over a full turn in place at (x, 0)."""
    yaw = np.linspace(0.0, 2.0 * math.pi, n)
    return wheels_of(np.stack([np.full_like(yaw, x), np.zeros_like(yaw), yaw], -1))


def check(p: dict, extent: float, cell: float) -> dict:
    """Build the pair and assert the premise. Returns the numbers worth recording."""
    robot = dynamics.robot_params()
    r = float(robot.wheel_radius)
    walls, curbs = rects(p)
    with_map = build_terrain(walls, WALL_H, curbs, p["curb_h"], extent, cell)
    ctrl_map = build_terrain(walls, WALL_H, curbs, 0.0, extent, cell)
    x0 = y0 = -extent / 2.0
    d_curb = distance_field(curbs, p["curb_h"], extent, cell)
    d_wall = distance_field(walls, WALL_H, extent, cell)
    at = lambda f, xy: sample_field(f, x0, y0, cell, xy)

    # 1. the settle accepts the start and every pose of a turn in place there
    yaws = (np.arange(48) + 0.5) * (2.0 * math.pi / 48)
    poses = np.stack([np.zeros_like(yaws), np.zeros_like(yaws), yaws], -1)
    derived, residual, clearance = settle_batch(with_map, poses, MU, DEVICE)
    feasible = settle_feasible(derived, residual, clearance, robot)
    assert feasible.all(), f"the settle blocks {int((~feasible).sum())}/48 start-pivot poses -- lower the curb"
    max_pitch = math.degrees(float(np.abs(derived[:, 1]).max()))

    # 2. the start pivot drags the rear tyre over the curb; no wheel CENTRE gets onto it
    start_w = pivot_wheels_at(START[0])
    rear = at(d_curb, start_w[:, 2])
    assert rear.min() < r - TYRE_OVER, f"the start pivot's rear tyre reaches only {r - rear.min():.2f} m over the curb"
    assert at(d_curb, start_w).min() > 0.0, "a wheel centre sits on the curb during the start pivot"
    # 3. driving along the centreline touches nothing: front rims stay off the curb
    xs = np.arange(-p["curb_back"] - 0.5, p["curb_fwd"] + 0.5, cell)
    line = wheels_of(np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], -1))
    assert at(d_curb, line).min() > r, "driving straight along the corridor touches the curb"
    # 4. the bay has a curb-free point turn, clear of the walls by the settle's envelope
    x_bay = p["curb_fwd"] + p["bay"] / 2.0
    bay_w = pivot_wheels_at(x_bay)
    assert at(d_curb, bay_w).min() > r + BAY_CLEAR, "a point turn in the bay touches the curb"
    assert at(d_wall, bay_w).min() > r, "a point turn in the bay hits a wall"

    # 5. the cost-to-go: blind to the curb, and no way out without a point turn
    v_with, cap = start_value(with_map, START, (p["goal_x"], 0.0), PIVOT_COST)
    v_ctrl, _ = start_value(ctrl_map, START, (p["goal_x"], 0.0), PIVOT_COST)
    v_nopivot, _ = start_value(ctrl_map, START, (p["goal_x"], 0.0), 0.0)
    assert v_with < cap - 1e-3, f"no route out at pivot_cost {PIVOT_COST} (V {v_with:.2f}, cap {cap:.2f})"
    assert abs(v_with - v_ctrl) < 1e-3, f"the cost-to-go sees the curb: V {v_with:.3f} vs ctrl {v_ctrl:.3f}"
    assert v_nopivot >= cap - 1e-3, f"a forward route exists without point turns (V {v_nopivot:.2f}) -- narrow the corridor"
    return dict(with_map=with_map, ctrl_map=ctrl_map, walls=walls, curbs=curbs, v_start=v_with, v_cap=cap,
                max_pitch_deg=max_pitch, tyre_over=float(r - rear.min()), x_bay=x_bay)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--extent", type=float, default=12.0)
    parser.add_argument("--cell", type=float, default=0.05)
    parser.add_argument("--out-dir", type=pathlib.Path, default=ASSETS_DIR)
    args = parser.parse_args()
    wp.init()
    rows = []
    for seed in range(args.seed, args.seed + args.n):
        rng = np.random.default_rng([seed, FAMILY_ID])
        for attempt in range(MAX_DRAWS):  # a draw that breaks the premise is redrawn, and reported
            p = draw(rng)
            try:
                c = check(p, args.extent, args.cell)
                break
            except AssertionError as error:
                print(f"[{FAMILY}_s{seed:03d}] draw {attempt} rejected: {error}")
        else:
            raise SystemExit(f"seed {seed}: no draw in {MAX_DRAWS} passes the premise")
        stem = f"{FAMILY}_s{seed:03d}"
        meta = dict(source="feasibility.heightmap.mppi_eval.create_turn_spot", family=FAMILY, seed=seed,
                    start=list(START), goal=[p["goal_x"], 0.0], spin_frac=SPIN_FRAC, pivot_cost=PIVOT_COST,
                    walls=[list(map(float, w)) for w in c["walls"]], wall_height=WALL_H,
                    curbs=[list(map(float, w)) for w in c["curbs"]], curb_height=p["curb_h"],
                    params=p, x_bay=c["x_bay"], v_start=c["v_start"])
        write_pair(args.out_dir, stem, c["with_map"], c["ctrl_map"], meta)
        rows.append(dict(stem=stem, **p))
        print(f"[{stem}] width {p['width']:.2f} curb {p['curb_h']:.2f} m at {p['curb_edge']:.2f} m, bay "
              f"{p['bay']:.2f} m: start-pivot pitch <= {c['max_pitch_deg']:.1f} deg, rear tyre {c['tyre_over']:.2f} m "
              f"over the curb, V {c['v_start']:.2f} (cap {c['v_cap']:.0f})")
    write_manifest(args.out_dir, FAMILY, rows)
    print(f"wrote {len(rows)} pairs to {args.out_dir}")


if __name__ == "__main__":
    main()
