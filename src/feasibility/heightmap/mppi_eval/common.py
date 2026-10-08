"""Shared by the `mppi_eval` map families: the cost-to-go check every premise rests on, the
feature-distance field the premise scorer reads, and writing a family's (with, ctrl) pair.

Each family writes every instance TWICE -- `<stem>` with its low features and `<stem>_ctrl` with
them removed -- so `nn_mppi/premise_check.py` can attribute a difference in vanilla MPPI's outcome to
the feature alone. That only holds if the twin is BLIND to the feature, i.e. the cost-to-go is the
same on both maps (`start_value`), which every generator asserts.

The sidecar of both maps carries the feature rectangles (`walls`, `curbs`: `(w, d, cx, cy, yaw)` as
`create_curbs_and_walls.rects_layer` takes them, w along x) and their heights, so a scorer can
rasterise each feature on its own; `closed_loop.py` reads `spin_frac`/`pivot_cost` from it.
"""
from __future__ import annotations

import math
import pathlib

import numpy as np
import warp as wp
import yaml
from helhest import dynamics
from helhest.engine import GridParams
from helhest.planning.costtogo import CostToGo
from scipy.ndimage import distance_transform_edt

from feasibility.heightmap import HeightMapReader
from feasibility.heightmap.create_curbs_and_walls import rects_layer

INCLINE_DEG = 80.0  # every feature's side slope, as in create_curbs_and_walls
RELIEF = 0.05  # [m] what counts as a feature's footprint (lattice_learning's trial.interact_relief)
# the cost-to-go closed_loop.py's planners sample: the mppi_learning labels' twin (k_turn 1.0), its
# routing cell and robust margin
K_TURN = 1.0
ROUTING_CELL = 0.32
ROBUST_MARGIN = 0.3
N_THETA = 24
DEVICE = "cuda:0"


def build_terrain(walls: list, wall_h: float, curbs: list, curb_h: float, extent: float, cell: float) -> HeightMapReader:
    """Flat ground plus the walls at `wall_h` and the curbs at `curb_h`, union of solids (max).
    An empty list or a zero height leaves that feature out -- the `_ctrl` map is `curb_h` 0."""
    n = int(round(extent / cell)) + 1  # rects_layer's grid
    H = np.zeros((n, n))
    if walls and wall_h > 0.0:
        H = np.maximum(H, rects_layer(walls, wall_h, INCLINE_DEG, extent, cell))
    if curbs and curb_h > 0.0:
        H = np.maximum(H, rects_layer(curbs, curb_h, INCLINE_DEG, extent, cell))
    return HeightMapReader(H, origin=(-extent / 2.0, -extent / 2.0), cell=cell)


def start_value(terrain: HeightMapReader, start: tuple[float, float, float], goal: tuple[float, float],
                pivot_cost: float) -> tuple[float, float]:
    """(V at `start`'s lattice state, the unreachable cap) on the map max-pooled to ~ROUTING_CELL,
    exactly as `closed_loop.routing_field` builds the field MPPI's goal cost samples."""
    k = max(1, round(ROUTING_CELL / terrain.cell))
    ny, nx = terrain.ny // k, terrain.nx // k
    coarse = terrain.H[: ny * k, : nx * k].reshape(ny, k, nx, k).max(axis=(1, 3))
    c = terrain.cell * k
    ctg = CostToGo(GridParams(nx, ny, c, terrain.x0, terrain.y0), dynamics.robot_params(),
                   dynamics.planning_solver(k_turn=K_TURN), n_theta=N_THETA, robust_margin_m=ROBUST_MARGIN,
                   pivot_cost=pivot_cost, device=DEVICE)
    V = ctg.compute(wp.array(np.ascontiguousarray(coarse, np.float32), dtype=wp.float32, device=DEVICE), goal).numpy()
    i, j = int((start[1] - terrain.y0) / c), int((start[0] - terrain.x0) / c)
    b = int(math.floor((start[2] % (2.0 * math.pi)) / (2.0 * math.pi / N_THETA))) % N_THETA
    return float(V[i, j, b]), float(ctg._vcap)


def distance_field(rects: list, height: float, extent: float, cell: float, above: float = RELIEF) -> np.ndarray:
    """[ny, nx] metres from each cell to the footprint of `rects` (where it stands above `above`).
    Infinite everywhere when there is no such feature."""
    if not rects or height <= above:
        n = int(round(extent / cell)) + 1
        return np.full((n, n), np.inf)
    return distance_transform_edt(~(rects_layer(rects, height, INCLINE_DEG, extent, cell) > above)) * cell


def sample_field(field: np.ndarray, x0: float, y0: float, cell: float, xy: np.ndarray) -> np.ndarray:
    """`field` at world points `xy[..., 2]`, nearest cell centre, clamped at the edges."""
    j = np.clip(np.round((xy[..., 0] - x0) / cell - 0.5).astype(int), 0, field.shape[1] - 1)
    i = np.clip(np.round((xy[..., 1] - y0) / cell - 0.5).astype(int), 0, field.shape[0] - 1)
    return field[i, j]


def write_pair(out_dir: pathlib.Path, stem: str, with_map: HeightMapReader, ctrl_map: HeightMapReader,
               meta: dict) -> list[str]:
    """Save `<stem>` (with) and `<stem>_ctrl` and add `meta` (+ `variant`) to both sidecars."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for terrain, variant, name in ((with_map, "with", stem), (ctrl_map, "ctrl", f"{stem}_ctrl")):
        path = out_dir / name
        terrain.save(path)
        sidecar = yaml.safe_load(path.with_suffix(".yaml").read_text())
        sidecar.update(meta, variant=variant, pair=stem)
        path.with_suffix(".yaml").write_text(yaml.safe_dump(sidecar, sort_keys=False))
        names.append(name)
    return names


def write_manifest(out_dir: pathlib.Path, family: str, rows: list[dict]) -> None:
    """`<out_dir>/manifest.yaml`: one row per instance (its stem and the drawn parameters)."""
    (out_dir / "manifest.yaml").write_text(yaml.safe_dump({"family": family, "instances": rows}, sort_keys=False))
