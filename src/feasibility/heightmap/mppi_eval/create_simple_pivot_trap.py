"""SIMPLE-PIVOT-TRAP family: open ground, the goal off to the robot's left, and one block behind
its right shoulder that a turn in place towards the goal sweeps the rear tyre into -- for the
premise check (`nn_mppi/premise_check.py`) before any nn-MPPI data is made for it.

Bird's eye (the robot at the origin facing +x, the reference point on its front axle):

              goal (bearing B, distance D)
               X
                 \\  red: turn in place towards the goal, then drive straight (shorter)
                  \\         green: drive forward first, then arc left (longer, touches nothing)
                   \\
             rear o====[ front axle ]  ->  +x
                  \\
                   [block]   at PHI_HIT into the turn, behind-right

  * Helhest turns in place about its FRONT axle, so a CCW (left) turn swings the rear wheel to the
    RIGHT, along a circle of radius rear_offset (0.75 m), rim at 1.10 m. The block's inner face
    is centred on that circle's direction at PHI_HIT into the turn, OVERLAP inside the rim, so
    the red manoeuvre puts the rear tyre against it. Driving forward moves the rear wheel away.
  * This is a SWEEP, not a randomised family yet: block height x goal bearing, to find where
    vanilla MPPI in closed loop actually takes the red manoeuvre (and the block hurts it).
    A tall block is seen by the settle (it blocks the poses), a low one only partly; the
    sidecar records how many of the red turn's poses the settle blocks and the cost-to-go V at the
    start on both maps, at the family's `pivot_cost` and at 0 (no point turns: the green way).
  * The ctrl map has no block. It depends on the bearing only, so it is written once per bearing
    and every with-map's sidecar names it (`ctrl`), which premise_check.py runs once.

Maps: `simple_pivot_trap_h<cm>_b<deg>` and `simple_pivot_trap_b<deg>_ctrl`.

CLI parameters:
    --heights M ...   block heights (default 0.15 0.20 0.25 0.35)
    --bearings D ...  goal bearings [deg, CCW from the start heading] (default 60 90 120 150)
    --distance M      goal distance (default 3.5)
    --extent M        square map side (default 10.0)
    --cell M          grid resolution (default 0.05)
    --out-dir DIR     output directory (default assets/mppi_eval/simple_pivot_trap)

Usage:
    python src/feasibility/heightmap/mppi_eval/create_simple_pivot_trap.py
"""
from __future__ import annotations

import argparse
import math
import pathlib

import numpy as np
import warp as wp
import yaml
from helhest import dynamics

from feasibility.heightmap.create_garage import wheels_of
from feasibility.heightmap.mppi_eval.common import build_terrain
from feasibility.heightmap.mppi_eval.common import DEVICE
from feasibility.heightmap.mppi_eval.common import distance_field
from feasibility.heightmap.mppi_eval.common import sample_field
from feasibility.heightmap.mppi_eval.common import start_value
from feasibility.heightmap.mppi_eval.common import write_manifest
from feasibility.lattice_learning.settle import settle_batch
from feasibility.lattice_learning.settle import settle_feasible

FAMILY = "simple_pivot_trap"
ASSETS_DIR = pathlib.Path("assets/mppi_eval/simple_pivot_trap")
START = (0.0, 0.0, 0.0)
MU = 0.8
PIVOT_COST = 0.15  # the closed_loop flags this family needs: point turns in the cost-to-go ...
SPIN_FRAC = 0.1  # ... and MPPI's SPIN prior
BLOCK_SIDE = 0.8  # [m] square block
PHI_HIT = math.radians(45.0)  # how far into the CCW turn the rear wheel meets the block
OVERLAP = 0.20  # [m] of rear rim past the block's inner face (create_garage's PLANNED_OVERLAP)
HEIGHTS = (0.15, 0.20, 0.25, 0.35)  # 0.70 grows its ramp onto the rear wheel at the start
BEARINGS = (60.0, 90.0, 120.0, 150.0)


def block_rect() -> tuple[float, float, float, float, float]:
    """The block as `(w, d, cx, cy, yaw)`: its inner face at rear_offset + wheel_radius - OVERLAP from
    the reference point, square to the direction the rear wheel has at PHI_HIT into the turn."""
    robot = dynamics.robot_params()
    inner = float(robot.rear_offset) + float(robot.wheel_radius) - OVERLAP
    rho = inner + BLOCK_SIDE / 2.0
    ux, uy = -math.cos(PHI_HIT), -math.sin(PHI_HIT)  # the rear wheel's direction at PHI_HIT
    return (BLOCK_SIDE, BLOCK_SIDE, rho * ux, rho * uy, PHI_HIT)


def red_turn_poses(turn: float, n: int = 49) -> np.ndarray:
    """[n, 3] poses of a CCW turn in place at the start, `turn` radians."""
    yaw = START[2] + np.linspace(0.0, turn, n)
    return np.stack([np.full_like(yaw, START[0]), np.full_like(yaw, START[1]), yaw], -1)


def audit(height: float, bearing: float, distance: float, extent: float, cell: float) -> dict:
    """Build the (with, ctrl) maps for one sweep cell and measure, without asserting, what the
    twin sees of the block on the red manoeuvre."""
    robot = dynamics.robot_params()
    r = float(robot.wheel_radius)
    block = [block_rect()]
    goal = (distance * math.cos(bearing), distance * math.sin(bearing))
    with_map = build_terrain([], 0.0, block, height, extent, cell)
    ctrl_map = build_terrain([], 0.0, block, 0.0, extent, cell)
    x0 = y0 = -extent / 2.0
    d_block = distance_field(block, height, extent, cell)
    poses = red_turn_poses(bearing)
    wheels = wheels_of(poses)
    rear = sample_field(d_block, x0, y0, cell, wheels[:, 2])
    fronts = sample_field(d_block, x0, y0, cell, wheels[:, :2])
    assert fronts.min() > r, "a front tyre reaches the block during the red turn"
    straight = wheels_of(np.stack([np.linspace(0.0, 2.0, 41), np.zeros(41), np.zeros(41)], -1))
    assert sample_field(d_block, x0, y0, cell, straight).min() > r, "driving straight ahead touches the block"
    derived, residual, clearance = settle_batch(with_map, poses, MU, DEVICE)
    feasible = settle_feasible(derived, residual, clearance, robot)
    assert feasible[0], "the settle blocks the start pose"
    v = {}
    for name, terrain in (("with", with_map), ("ctrl", ctrl_map)):
        for pc in (PIVOT_COST, 0.0):
            v[f"v_{name}_pc{pc:g}"], cap = start_value(terrain, START, goal, pc)
    return dict(with_map=with_map, ctrl_map=ctrl_map, block=block, goal=goal, v_cap=cap,
                tyre_over=float(r - rear.min()), settle_blocked=int((~feasible).sum()), n_poses=len(poses),
                max_pitch_deg=math.degrees(float(np.abs(derived[feasible, 1]).max())), **v)


def save(terrain, path: pathlib.Path, meta: dict) -> None:
    terrain.save(path)
    sidecar = yaml.safe_load(path.with_suffix(".yaml").read_text())
    sidecar.update(meta)
    path.with_suffix(".yaml").write_text(yaml.safe_dump(sidecar, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--heights", type=float, nargs="+", default=list(HEIGHTS))
    parser.add_argument("--bearings", type=float, nargs="+", default=list(BEARINGS))
    parser.add_argument("--distance", type=float, default=3.5)
    parser.add_argument("--extent", type=float, default=10.0)
    parser.add_argument("--cell", type=float, default=0.05)
    parser.add_argument("--out-dir", type=pathlib.Path, default=ASSETS_DIR)
    args = parser.parse_args()
    wp.init()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for bearing_deg in args.bearings:
        bearing = math.radians(bearing_deg)
        ctrl = f"{FAMILY}_b{bearing_deg:03.0f}_ctrl"
        for height in args.heights:
            a = audit(height, bearing, args.distance, args.extent, args.cell)
            stem = f"{FAMILY}_h{round(height * 100):03d}_b{bearing_deg:03.0f}"
            common = dict(source="feasibility.heightmap.mppi_eval.create_simple_pivot_trap", family=FAMILY,
                          start=list(START), goal=list(a["goal"]), spin_frac=SPIN_FRAC, pivot_cost=PIVOT_COST,
                          walls=[], wall_height=0.0, curbs=[list(map(float, b)) for b in a["block"]],
                          curb_height=height, bearing_deg=bearing_deg, distance=args.distance)
            numbers = {k: float(v) for k, v in a.items() if k.startswith("v_") or k in ("tyre_over", "max_pitch_deg")}
            numbers["settle_blocked"] = a["settle_blocked"]
            save(a["with_map"], args.out_dir / stem, dict(common, variant="with", pair=stem, ctrl=ctrl, **numbers))
            if not (args.out_dir / f"{ctrl}.yaml").exists() or height == args.heights[0]:
                save(a["ctrl_map"], args.out_dir / ctrl, dict(common, variant="ctrl", pair=None))
            rows.append(dict(stem=stem, ctrl=ctrl, height=height, bearing_deg=bearing_deg, **numbers))
            pc = f"{PIVOT_COST:g}"
            print(f"[{stem}] rear tyre {a['tyre_over']:.2f} m over, settle blocks {a['settle_blocked']}/{a['n_poses']} "
                  f"red-turn poses (pitch <= {a['max_pitch_deg']:.1f} deg) | V start: with {a[f'v_with_pc{pc}']:.2f}"
                  f" (no pivots {a['v_with_pc0']:.2f}), ctrl {a[f'v_ctrl_pc{pc}']:.2f} (no pivots {a['v_ctrl_pc0']:.2f})")
    write_manifest(args.out_dir, FAMILY, rows)
    print(f"wrote {len(rows)} maps + {len(args.bearings)} ctrl maps to {args.out_dir}")


if __name__ == "__main__":
    main()
