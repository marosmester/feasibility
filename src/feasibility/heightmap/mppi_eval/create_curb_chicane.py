"""CURB-CHICANE family: an S-bend round tall wall ends, each end wrapped in a low curb APRON, so the
shortest way round crosses the apron obliquely right beside the wall -- for the premise check
(`nn_mppi/premise_check.py`) before any nn-MPPI data is made for it.

Layout (x from start to goal; one or two staggered walls from the map edge):

        +y edge  ---------------------------+------------------
                                            | wall 2 (from +y)
                     .apron.                |
    start  ......    : +---:    ......    .apron.   ......   goal
                       |                    :   :
                       | wall 1 (from -y)
        -y edge  ------+---------------------------------------

  * The walls are tall (WALL_H): the settle blocks them, and the cost-to-go keeps its 0.3 m robust
    margin off them. The route therefore hugs each wall END, which is where the apron is.
  * The apron is low (0.10-0.15 m, under `create_garage.BLOCK_H` 0.196 m, the lift at one front
    wheel that first breaks max_roll), so the settle accepts the poses on it. The coarse cost-to-go
    still charges it a little (V up 0.1-1.0 over the ctrl map in most draws; recorded as
    `v_start_ctrl`, not asserted equal), so the twin is not strictly blind here: what is asserted
    is that crossing stays cheaper than going round, and the premise check measures the contact. In ostrich a wheel
    meets a curb face at an angle, right next to a wall it can be deflected into.
  * Making the apron as tall as the wall must cost the route something (asserted): the curb really
    is on the cheapest route, and a route round it exists.

A draw that fails a premise assert is redrawn from the same seed's stream (and printed). Each
instance is written twice, `curb_chicane_s<seed>` and `curb_chicane_s<seed>_ctrl` (aprons removed).

CLI parameters:
    --n INT        instances (seeds --seed .. --seed + n - 1, default 8)
    --seed INT     first seed (default 0)
    --extent M     square map side (default 14.0)
    --cell M       grid resolution (default 0.05)
    --out-dir DIR  output directory (default assets/mppi_eval/curb_chicane)

Usage:
    python src/feasibility/heightmap/mppi_eval/create_curb_chicane.py
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import warp as wp

from feasibility.heightmap.mppi_eval.common import build_terrain
from feasibility.heightmap.mppi_eval.common import start_value
from feasibility.heightmap.mppi_eval.common import write_manifest
from feasibility.heightmap.mppi_eval.common import write_pair

FAMILY = "curb_chicane"
FAMILY_ID = 2  # in the rng's SeedSequence, so the two families never share a stream
ASSETS_DIR = pathlib.Path("assets/mppi_eval/curb_chicane")
START = (-5.0, 0.0, 0.0)
GOAL = (5.0, 0.0)
WALL_H = 0.70
WALL_T = 0.30
BEYOND_EDGE = 1.0  # [m] a wall runs this far past its map edge, so there is no way round that side
MIN_GAP = 3.0  # [m] wall end to the opposite map edge
PIVOT_COST = 0.0  # the node's: forward arcs only
SPIN_FRAC = 0.0
MAX_DRAWS = 20
ROUTE_COST = 0.05  # the tall-apron map must raise V(start) by at least this
# drawn per seed
END_Y = (0.5, 1.5)  # [m] how far past the centreline each wall reaches
WALL_X = (1.5, 2.5)  # [m] |x| of the two walls
CURB_H = (0.10, 0.15)
APRON_PAST = (0.6, 1.0)  # [m] apron beyond the wall end
APRON_SIDE = (0.4, 0.7)  # [m] apron to each side of the wall


def draw(rng: np.random.Generator) -> dict:
    u = lambda lo_hi: float(rng.uniform(*lo_hi))
    return dict(n_walls=int(rng.integers(1, 3)), end_y=u(END_Y), wall_x=u(WALL_X), curb_h=u(CURB_H),
                apron_past=u(APRON_PAST), apron_side=u(APRON_SIDE))


def rects(p: dict, extent: float) -> tuple[list, list]:
    """(walls, aprons) as `(w, d, cx, cy, yaw)`, w along x. Wall 1 rises from the -y edge at
    x = -wall_x (x = 0 when it is the only one), wall 2 hangs from the +y edge at x = +wall_x."""
    edge = extent / 2.0 + BEYOND_EDGE
    walls, aprons = [], []
    xs = (-p["wall_x"], p["wall_x"]) if p["n_walls"] == 2 else (0.0,)
    for k, x in enumerate(xs):
        s = 1.0 if k == 0 else -1.0  # wall 1 ends at +end_y, wall 2 at -end_y
        end = s * p["end_y"]
        length = edge + p["end_y"]
        walls.append((WALL_T, length, x, end - s * length / 2.0, 0.0))
        depth = p["apron_past"] + 0.3  # from 0.3 m inside the wall end to apron_past past it
        aprons.append((WALL_T + 2.0 * p["apron_side"], depth, x, end + s * (p["apron_past"] - 0.3) / 2.0, 0.0))
    return walls, aprons


def check(p: dict, extent: float, cell: float) -> dict:
    walls, aprons = rects(p, extent)
    assert extent / 2.0 - p["end_y"] >= MIN_GAP, "no room round a wall end"
    with_map = build_terrain(walls, WALL_H, aprons, p["curb_h"], extent, cell)
    ctrl_map = build_terrain(walls, WALL_H, aprons, 0.0, extent, cell)
    tall_map = build_terrain(walls, WALL_H, aprons, WALL_H, extent, cell)
    v_with, cap = start_value(with_map, START, GOAL, PIVOT_COST)
    v_ctrl, _ = start_value(ctrl_map, START, GOAL, PIVOT_COST)
    v_tall, _ = start_value(tall_map, START, GOAL, PIVOT_COST)
    assert v_with < cap - 1e-3, "no route"
    # NOT asserted equal to v_ctrl: the coarse cost-to-go does charge the apron a little (+0.1..1.0 in
    # most draws), so strict blindness only passed one-wall maps with small aprons. What the premise
    # needs is that crossing still beats going round, so vanilla keeps meeting the apron.
    assert v_tall > v_with + ROUTE_COST, f"the cheapest route misses the apron (V tall {v_tall:.3f} vs {v_with:.3f})"
    assert v_tall < cap - 1e-3, "no route round the aprons"
    return dict(with_map=with_map, ctrl_map=ctrl_map, walls=walls, curbs=aprons, v_start=v_with, v_ctrl=v_ctrl,
                v_tall=v_tall)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--extent", type=float, default=14.0)
    parser.add_argument("--cell", type=float, default=0.05)
    parser.add_argument("--out-dir", type=pathlib.Path, default=ASSETS_DIR)
    args = parser.parse_args()
    wp.init()
    rows = []
    for seed in range(args.seed, args.seed + args.n):
        rng = np.random.default_rng([seed, FAMILY_ID])
        for attempt in range(MAX_DRAWS):
            p = draw(rng)
            try:
                c = check(p, args.extent, args.cell)
                break
            except AssertionError as error:
                print(f"[{FAMILY}_s{seed:03d}] draw {attempt} rejected: {error}")
        else:
            raise SystemExit(f"seed {seed}: no draw in {MAX_DRAWS} passes the premise")
        stem = f"{FAMILY}_s{seed:03d}"
        meta = dict(source="feasibility.heightmap.mppi_eval.create_curb_chicane", family=FAMILY, seed=seed,
                    start=list(START), goal=list(GOAL), spin_frac=SPIN_FRAC, pivot_cost=PIVOT_COST,
                    walls=[list(map(float, w)) for w in c["walls"]], wall_height=WALL_H,
                    curbs=[list(map(float, w)) for w in c["curbs"]], curb_height=p["curb_h"],
                    params=p, v_start=c["v_start"], v_start_ctrl=c["v_ctrl"], v_start_tall_aprons=c["v_tall"])
        write_pair(args.out_dir, stem, c["with_map"], c["ctrl_map"], meta)
        rows.append(dict(stem=stem, **p))
        print(f"[{stem}] {p['n_walls']} wall(s) ending {p['end_y']:.2f} m past the centreline, apron "
              f"{p['curb_h']:.2f} m reaching {p['apron_past']:.2f} m: V {c['v_start']:.2f} (ctrl {c['v_ctrl']:.2f}, "
              f"aprons tall {c['v_tall']:.2f})")
    write_manifest(args.out_dir, FAMILY, rows)
    print(f"wrote {len(rows)} pairs to {args.out_dir}")


if __name__ == "__main__":
    main()
