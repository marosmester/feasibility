"""One map on which the vanilla and the NN-gated lattice planners take DIFFERENT routes, for
`demos/ostrich_follow_path.py`: a short straight line up a face ostrich cannot climb, and a longer
detour up a gentle ramp beside it.

The uphill series (`create_uphill_series.py`) spans the full width, so there is no way round and
the planners can only differ on WHETHER they plan. Here the same 0.75 m face carries a gentle ramp
at one side:

    start (x0 + margin, 0, 0) ...... steep face (--steep-deg, full width) | plateau ... goal (2, 0)
                                     gentle ramp at y = --ramp-y, crest at x = 0 too

* The STEEP face is `create_uphill_series.uphill_ramp`: crest pinned at x = 0, plateau to the grid
  end, goal GOAL_PAST_CREST onto it. 65 deg by default -- ostrich flips at the foot of the 65 deg
  face in 3/3 repeats, and TAU_POS / TAU_PITCH were calibrated to close that face head-on and keep
  60 deg open (`benchmarks/bench_uphill_nn.py`).
* The APPROACH (`--approach`, 4 m) is flat ground from the start to the gentle ramp's foot. It is
  what lets 65 deg work: with only ~0.25 m of it the detour is a sharp hook (pure pursuit snaps
  round it) and `nn-gated-pos` instead crosses the 65 deg face OBLIQUELY, at an angle the net
  under-predicts -- the blind spot planners.TAU_FUSED_POS records. With 4 m every gate goes round.
* The GENTLE ramp is one `create_ramps.Ramp` (yaw 0, 80 deg side drop-offs, the same closed form as
  lattice_learning's ramps), its crest also at x = 0 so its plateau runs into the main one; the
  two combine by elementwise maximum (union of solids). Its far side falls inside the main plateau,
  so it never shows.

The premise is two claims about the PLANNERS, and the main block asserts both with the exact
`plan_path` call the demo makes (GPU; the checkpoints are in gitignored outputs/, so `--no-check`
writes the map without them and records that in the sidecar):

  * `vanilla-off` (no `blocked`, only the graded tilt cost) goes straight up the steep face.
    Fails when the ramp is so close that the detour is cheaper even with the tilt cost -- move
    `--ramp-y` out.
  * `nn-gated-pos` and `nn-gated-pitch` go round, up the gentle ramp. Fails when a gate leaves some
    crossing of the face open (an oblique approach the net under-predicts) -- steepen `--steep-deg`.
`vanilla-on`, `nn-gated-rot` and `nn-gated-fused` are planned and reported, not asserted.

A route is classified where it crosses the middle of the steep face (x = -run/2): inside the ramp's
top width is `ramp`, outside its footprint `steep`, on its side skirt `edge`.

Output: `<out-dir>/ramp_detour_s<steep, tenths of deg>_g<gentle, tenths of deg>_a<approach, dm>.png/.yaml`.
The sidecar adds `start`, `goal`, `steep_deg`, `gentle_deg`, `ramp_y`, `ramp_width`, `approach`, `height`,
`crest_x`, `checked`, and per checked planner `routes.<planner>: {route, path_m}`.

CLI parameters:
    --steep-deg FLOAT   steep face angle [deg] (default 65)
    --gentle-deg FLOAT  gentle ramp angle [deg] (default 20)
    --ramp-y FLOAT      gentle ramp centre line [m] (default 2.5)
    --ramp-width FLOAT  gentle ramp top width [m] (default 2.5)
    --approach FLOAT    flat ground from the start to the gentle ramp's foot [m] (default 4.0)
    --torch-device STR  network inference device for the check (default cpu)
    --no-check          skip the planner premise check
    --out-dir PATH      output directory (default assets/ramp_detour)

Usage:
    python src/feasibility/heightmap/create_ramp_detour.py
    python src/feasibility/heightmap/create_ramp_detour.py --steep-deg 75 --ramp-y 3.0
"""
from __future__ import annotations

import argparse
import math
import pathlib

import numpy as np
import yaml
from helhest.planning.rampmaps import DEFAULT_CELL
from helhest.planning.rampmaps import DEFAULT_HEIGHT
from helhest.planning.rampmaps import DEFAULT_MARGIN
from helhest.planning.rampmaps import ramp_run

from feasibility.heightmap import HeightMapReader
from feasibility.heightmap.create_ramps import Ramp
from feasibility.heightmap.create_ramps import ramp_height
from feasibility.heightmap.create_uphill_series import series_grid
from feasibility.heightmap.create_uphill_series import start_goal
from feasibility.heightmap.create_uphill_series import uphill_ramp

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
ASSETS_DIR = REPO_ROOT / "assets" / "ramp_detour"
EXTENT_Y = 12.0  # [m] the series' 8 m plus room for the ramp clear of the edge
SIDE_DEG = 80.0  # gentle ramp's side drop-offs, create_ramps' default
RAMP_PLATEAU = 1.0  # [m] gentle ramp's own plateau past x = 0, ends inside the main plateau
DEFAULT_APPROACH = 4.0  # [m] flat run from the start to the gentle ramp's foot: room to steer onto
# the ramp in a gentle arc instead of a hook that pure pursuit has to snap round
EDGE_CLEARANCE = 1.5  # [m] ramp footprint to the +-y map edge (robot + a patch's lateral reach)
ASSERTED = {"vanilla-off": "steep", "nn-gated-pos": "ramp", "nn-gated-pitch": "ramp"}
REPORTED = ("vanilla-on", "nn-gated-rot", "nn-gated-fused")


def gentle_ramp(gentle_deg: float, ramp_y: float, width: float, height: float = DEFAULT_HEIGHT) -> Ramp:
    """The gentle ramp, yaw 0, its crest at x = 0 (foot at -run): Ramp.cx is its footprint midpoint."""
    proto = Ramp(0.0, ramp_y, 0.0, gentle_deg, height, RAMP_PLATEAU, SIDE_DEG, width, SIDE_DEG)
    return Ramp(proto.length / 2.0 - ramp_run(gentle_deg, height), ramp_y, 0.0, gentle_deg, height,
                RAMP_PLATEAU, SIDE_DEG, width, SIDE_DEG)


def build_ramp_detour(
    steep_deg: float, ramp: Ramp, approach: float = DEFAULT_APPROACH, cell: float = DEFAULT_CELL
) -> HeightMapReader:
    """The steep full-width face (uphill series geometry) united with `ramp` by elementwise max, on
    a grid holding `approach` of flat ground before the gentle ramp's foot plus DEFAULT_MARGIN
    behind the start."""
    x0, nx = series_grid(min(steep_deg, ramp.up_deg), ramp.height, approach + DEFAULT_MARGIN, cell)
    base = uphill_ramp(steep_deg, x0, nx, ramp.height, extent_y=EXTENT_Y, cell=cell)
    xs = base.x0 + (np.arange(base.nx) + 0.5) * cell
    ys = base.y0 + (np.arange(base.ny) + 0.5) * cell
    X, Y = np.meshgrid(xs, ys)
    H = np.maximum(base.H, ramp_height(ramp, X, Y))
    return HeightMapReader(H, (base.x0, base.y0), cell, min_z=0.0, max_z=ramp.height)


def row_rise_deg(terrain: HeightMapReader, y: float) -> float:
    """Steepest single-cell rise along +X on the grid row nearest `y` [deg]."""
    i = int((y - terrain.y0) / terrain.cell)
    return math.degrees(math.atan(float(np.max(np.diff(terrain.H[i]))) / terrain.cell))


def classify_route(xy: np.ndarray, x_cross: float, ramp: Ramp) -> str:
    """Where a planned polyline first crosses x = x_cross: `ramp` inside the top width, `steep`
    outside the footprint, `edge` on the side skirt."""
    k = int(np.argmax(xy[:, 0] >= x_cross))
    if xy[k, 0] < x_cross:
        return "none"
    a, b = xy[max(k - 1, 0)], xy[k]
    t = 0.0 if b[0] == a[0] else (x_cross - a[0]) / (b[0] - a[0])
    off = abs(a[1] + t * (b[1] - a[1]) - ramp.cy)
    if off <= ramp.width / 2.0:
        return "ramp"
    return "steep" if off > ramp.half_width else "edge"


def check_planners(
    terrain: HeightMapReader, start: tuple, goal: tuple, x_cross: float, ramp: Ramp, torch_device: str
) -> dict:
    """Plan with every planner of ASSERTED and REPORTED, as ostrich_follow_path.py does."""
    from feasibility.planning.planners import plan_path
    from feasibility.planning.planners import PlanContext
    from feasibility.planning.planners import PlannerConfig

    ctx = PlanContext(PlannerConfig(torch_device=torch_device))
    routes = {}
    for planner in (*ASSERTED, *REPORTED):
        plan = plan_path(terrain, start, goal, planner, ctx)
        route = classify_route(plan["xy"], x_cross, ramp) if plan["reached"] else "none"
        routes[planner] = {"route": route, "path_m": float(plan["path_m"]) if plan["reached"] else None}
        length = f"{plan['path_m']:6.2f} m" if plan["reached"] else "  no path"
        print(f"[route] {planner:>15}: {route:>5}  {length}  {plan['gate']}")
    return routes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steep-deg", type=float, default=65.0)
    parser.add_argument("--gentle-deg", type=float, default=20.0)
    parser.add_argument("--ramp-y", type=float, default=2.5)
    parser.add_argument("--ramp-width", type=float, default=2.5)
    parser.add_argument("--approach", type=float, default=DEFAULT_APPROACH)
    parser.add_argument("--torch-device", default="cpu")
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--out-dir", type=pathlib.Path, default=ASSETS_DIR)
    args = parser.parse_args()
    assert args.gentle_deg < args.steep_deg, "the gentle ramp must be gentler than the face"

    ramp = gentle_ramp(args.gentle_deg, args.ramp_y, args.ramp_width)
    terrain = build_ramp_detour(args.steep_deg, ramp, args.approach)
    start, goal = start_goal(terrain, DEFAULT_MARGIN)
    start = (start[0], 0.0, 0.0)
    goal = (goal[0], 0.0)
    height = ramp.height
    x_cross = -0.5 * ramp_run(args.steep_deg, height)

    # --- geometry --------------------------------------------------------------------------------
    assert terrain.sample(*start[:2]) == 0.0, "start is not on flat ground"
    assert abs(terrain.sample(*goal) - height) < 1e-9, "goal is not on the plateau"
    assert start[0] + terrain.cell < ramp.cx - ramp.length / 2.0, "start stencil reaches the ramp foot"
    assert classify_route(np.array([start[:2], goal]), x_cross, ramp) == "steep", (
        "the straight start -> goal line does not meet the steep face -- move --ramp-y out")
    steep, gentle = row_rise_deg(terrain, 0.0), row_rise_deg(terrain, ramp.cy)
    assert abs(steep - args.steep_deg) < 0.01, f"{args.steep_deg} deg face rendered as {steep:.2f} deg"
    assert abs(gentle - args.gentle_deg) < 0.01, f"{args.gentle_deg} deg ramp rendered as {gentle:.2f} deg"
    y_hi = terrain.y0 + terrain.ny * terrain.cell
    assert ramp.cy + ramp.half_width + EDGE_CLEARANCE <= y_hi, "ramp footprint too close to the +y edge"
    assert ramp.cy - ramp.half_width - EDGE_CLEARANCE >= terrain.y0, "ramp footprint too close to the -y edge"
    print(f"[geometry] face {steep:.1f} deg on y = 0, ramp {gentle:.1f} deg on y = {ramp.cy:.2f} "
          f"(top {ramp.width:.2f} m, footprint {2 * ramp.half_width:.2f} m), "
          f"{terrain.nx}x{terrain.ny} @ {terrain.cell} m")

    # --- premise: vanilla-off goes straight up, the gates go round --------------------------------
    routes = None
    if args.no_check:
        print("[route] --no-check: planner premise NOT verified")
    else:
        routes = check_planners(terrain, start, goal, x_cross, ramp, args.torch_device)
        for planner, want in ASSERTED.items():
            got = routes[planner]["route"]
            hint = ("move --ramp-y out: the detour is already cheap enough for the tilt cost" if want == "steep"
                    else "steepen --steep-deg: the gate leaves the face open")
            assert got == want, f"{planner} took route {got!r}, the premise needs {want!r} -- {hint}"
        for planner in ("nn-gated-pos", "nn-gated-pitch"):
            assert routes[planner]["path_m"] > routes["vanilla-off"]["path_m"], (
                f"{planner}'s detour is not longer than vanilla-off's straight line")

    # --- save ------------------------------------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = f"ramp_detour_s{round(args.steep_deg * 10):04d}_g{round(args.gentle_deg * 10):04d}_a{round(args.approach * 10):03d}"
    stem = args.out_dir / name
    terrain.save(stem)
    loaded = HeightMapReader.load(stem)
    err = float(np.abs(loaded.H - terrain.H).max())
    assert err <= 0.5 * height / 255.0 + 1e-9, f"{stem.name}: {err} m exceeds half an 8-bit quantum"
    meta = yaml.safe_load(stem.with_suffix(".yaml").read_text())
    meta.update(
        source="feasibility.heightmap.create_ramp_detour", steep_deg=float(args.steep_deg),
        gentle_deg=float(args.gentle_deg), ramp_y=float(ramp.cy), ramp_width=float(ramp.width),
        height=float(height), approach=float(args.approach), crest_x=0.0, start=[float(v) for v in start], goal=[float(v) for v in goal],
        checked=not args.no_check,
    )
    if routes is not None:
        meta["routes"] = routes
    stem.with_suffix(".yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    print(f"wrote {stem}.png/.yaml")


if __name__ == "__main__":
    main()
