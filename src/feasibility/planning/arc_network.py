"""Predicted per-arc error fields from a `lattice_learning` ArcDivergenceNet, on `gated_lattice`'s
lattice: one value per (row, col, heading, primitive), the input `EdgeGatedLatticeSolver` gates on.

The network is queried at exactly the poses `CostToGo`'s settle judges -- (origin_x + c*cell,
origin_y + r*cell, heading-bin centre) -- with the curvature of each of the lattice's five forward
primitives (`arc.primitive_kappas`).

Inference usually runs on CPU (the cu128 torch of the root env does not support a GTX 1050; Warp
does, so the planning stays on CUDA). To keep that affordable: when every row of the map is
identical (true for the whole uphill series), a body-frame patch depends only on (column, heading)
-- `HeightMapReader.sample` clamps at the Y edges, which preserves the invariance exactly -- so one
row is evaluated and broadcast. This is checked, not assumed; any other map takes the full path
(~100 s/map on CPU).
"""
from __future__ import annotations

import math
import pathlib

import numpy as np
import torch
from helhest.planning.costtogo import CostToGo

from feasibility.heightmap import HeightMapReader
from feasibility.lattice_learning.arc import primitive_kappas
from feasibility.lattice_learning.model import ArcDivergenceNet
from feasibility.lattice_learning.patch import sample_patches
from feasibility.lattice_learning.train import load_checkpoint
from feasibility.planning.gated_lattice import N_PRIM_ARC
from feasibility.planning.gated_lattice import N_THETA
from feasibility.planning.gated_lattice import STEP


def load_network(
    path: pathlib.Path, device: torch.device, ctg: CostToGo, label_mode: str = "pos_rpy"
) -> ArcDivergenceNet:
    """Loads the checkpoint and asserts the pinned constants match the lattice it will gate
    (design.md section 4c): a net trained on other arcs is silently wrong, not broken."""
    model, ckpt = load_checkpoint(path, device)
    model.eval()
    assert ckpt["label_mode"] == label_mode, f"need a {label_mode} checkpoint, got {ckpt['label_mode']}"
    assert ckpt["command_mode"] == "kappa", f"expected command_mode kappa, got {ckpt['command_mode']}"
    assert math.isclose(ckpt["arc_len"], STEP), f"checkpoint arc_len {ckpt['arc_len']} != {STEP}"
    assert math.isclose(ckpt["min_turn_radius"], float(ctg.robot.min_turn_radius), rel_tol=1e-6), (
        f"checkpoint min_turn_radius {ckpt['min_turn_radius']} != robot "
        f"{ctg.robot.min_turn_radius}"
    )
    # arc.primitive_kappas is in _build_primitives' `turns` order (asserted by arc.py's own
    # self-test); pivots append primitives the net has no curvature for, so a pivot lattice cannot
    # be indexed by this field at all -- it needs v_wz-commanded fields, not a wider tau.
    assert ctg.solver.n_prim == N_PRIM_ARC, (
        f"expected {N_PRIM_ARC} forward primitives (pivot_cost 0), got {ctg.solver.n_prim} -- "
        "a kappa-indexed field cannot describe an in-place pivot"
    )
    return model


@torch.no_grad()
def predict_arcs(
    model: ArcDivergenceNet,
    terrain: HeightMapReader,
    poses: np.ndarray,
    kappas: np.ndarray,
    chunk: int,
    device: torch.device,
) -> np.ndarray:
    """poses [n, 3] -> [n, n_prim, K] physical errors (model.target_names order). The trunk runs
    once per pose and only the head once per curvature -- the caching design.md section 5a built
    the architecture for. Patches come from lattice_learning.patch.sample_patches itself, so the
    input is by construction what the dataset was built from."""
    out = np.empty((len(poses), len(kappas), len(model.target_names)), np.float32)
    assert model.target_transform is not None
    for i in range(0, len(poses), chunk):
        patch = torch.from_numpy(sample_patches(terrain, poses[i : i + chunk], model.patch_spec))
        code = model.terrain_code(patch[:, None].to(device))[..., 0, 0]  # [b, 256]
        for p, kappa in enumerate(kappas):
            command = torch.full((code.shape[0], 1), float(kappa), device=device)
            y = model.target_transform.inverse(model._head(code, command))
            out[i : i + chunk, p] = y.cpu().numpy()
    return out


def lattice_poses(ctg: CostToGo, rows: np.ndarray) -> np.ndarray:
    """[len(rows) * nx * n_theta, 3] (x, y, yaw) in C order over (row, col, heading) -- exactly the
    poses CostToGo.__init__ assigns to its settle (cell corner + bin-centre heading), so the settle
    and the network judge the same lattice states."""
    grid = ctg.grid
    rr, cc, tt = np.meshgrid(rows, np.arange(grid.cells_x), np.arange(N_THETA), indexing="ij")
    x = grid.origin_x + cc * grid.cell_size
    y = grid.origin_y + rr * grid.cell_size
    yaw = (tt + 0.5) * 2.0 * np.pi / N_THETA
    return np.stack([x, y, yaw], axis=-1).reshape(-1, 3)


def arc_error_fields(
    model: ArcDivergenceNet,
    terrain: HeightMapReader,
    ctg: CostToGo,
    chunk: int,
    device: torch.device,
    heads: tuple[str, ...] = ("e_pos", "e_pitch"),
) -> dict[str, np.ndarray]:
    """{head in `heads`} -> [ny, nx, n_theta, n_prim] predicted error per lattice arc. When all
    rows of the map are identical the patch cannot depend on the row (see module docstring), so row
    0 is evaluated and broadcast; otherwise every row is."""
    ny, nx = ctg.grid.cells_y, ctg.grid.cells_x
    kappas = np.array(primitive_kappas(float(ctg.robot.min_turn_radius)))
    row_invariant = bool(np.all(terrain.H == terrain.H[:1]))
    rows = np.arange(1) if row_invariant else np.arange(ny)
    pred = predict_arcs(model, terrain, lattice_poses(ctg, rows), kappas, chunk, device)
    pred = pred.reshape(len(rows), nx, N_THETA, len(kappas), -1)
    names = model.target_names
    fields = {name: pred[..., names.index(name)] for name in heads}
    if row_invariant:
        fields = {k: np.ascontiguousarray(np.broadcast_to(v, (ny, *v.shape[1:]))) for k, v in fields.items()}
    return fields
