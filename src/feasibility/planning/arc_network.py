"""Predicted per-arc error fields from a `lattice_learning` ArcDivergenceNet, on `gated_lattice`'s
lattice: one value per (row, col, heading, primitive), the input `EdgeGatedLatticeSolver` gates on.

The network is queried at exactly the poses `CostToGo`'s settle judges -- (origin_x + c*cell,
origin_y + r*cell, heading-bin centre) -- with the command of each lattice primitive: curvature for
a kappa net (`arc.primitive_kappas`, the five forward arcs only) via `arc_error_fields`, or
(v_drive, wz_drive) for a v_wz one (`primitive_commands`, point turns included) via
`vwz_error_fields`. Both are `_error_fields` with a different command table.

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
from feasibility.lattice_learning.arc import OMEGA_NOM
from feasibility.lattice_learning.arc import primitive_kappas
from feasibility.lattice_learning.arc import twist_from_kappa
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
    commands: np.ndarray,
    chunk: int,
    device: torch.device,
) -> np.ndarray:
    """poses [n, 3], commands [n_prim, C] (C = 1 curvature for a kappa net, 2 = (v, wz) for a v_wz
    one) -> [n, n_prim, K] physical errors (model.target_names order). The trunk runs once per pose
    and only the head once per command -- the caching design.md section 5a built the architecture
    for. Patches come from lattice_learning.patch.sample_patches itself, so the input is by
    construction what the dataset was built from."""
    out = np.empty((len(poses), len(commands), len(model.target_names)), np.float32)
    assert model.target_transform is not None
    for i in range(0, len(poses), chunk):
        patch = torch.from_numpy(sample_patches(terrain, poses[i : i + chunk], model.patch_spec))
        code = model.terrain_code(patch[:, None].to(device))[..., 0, 0]  # [b, 256]
        for p, cmd in enumerate(commands):
            command = torch.tensor(cmd, dtype=torch.float32, device=device).expand(code.shape[0], -1)
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


def _error_fields(
    model: ArcDivergenceNet,
    terrain: HeightMapReader,
    ctg: CostToGo,
    commands: np.ndarray,
    chunk: int,
    device: torch.device,
    heads: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """{head in `heads`} -> [ny, nx, n_theta, len(commands)] predicted error, one plane per command
    in the solver's own primitive order. When all rows of the map are identical the patch cannot
    depend on the row (see module docstring), so row 0 is evaluated and broadcast; otherwise every
    row is. `commands` is the only thing that differs between a kappa net and a v_wz one."""
    ny, nx = ctg.grid.cells_y, ctg.grid.cells_x
    row_invariant = bool(np.all(terrain.H == terrain.H[:1]))
    rows = np.arange(1) if row_invariant else np.arange(ny)
    pred = predict_arcs(model, terrain, lattice_poses(ctg, rows), commands, chunk, device)
    pred = pred.reshape(len(rows), nx, N_THETA, len(commands), -1)
    names = model.target_names
    fields = {name: pred[..., names.index(name)] for name in heads}
    if row_invariant:
        fields = {k: np.ascontiguousarray(np.broadcast_to(v, (ny, *v.shape[1:]))) for k, v in fields.items()}
    return fields


def arc_error_fields(
    model: ArcDivergenceNet,
    terrain: HeightMapReader,
    ctg: CostToGo,
    chunk: int,
    device: torch.device,
    heads: tuple[str, ...] = ("e_pos", "e_pitch"),
) -> dict[str, np.ndarray]:
    """{head in `heads`} -> [ny, nx, n_theta, n_prim] predicted error per lattice arc, from a
    CURVATURE-commanded net -- so only the five forward primitives, and `load_network` refuses a
    pivot lattice. `vwz_error_fields` is the counterpart that covers the point turns too."""
    kappas = np.array(primitive_kappas(float(ctg.robot.min_turn_radius)))
    return _error_fields(model, terrain, ctg, kappas[:, None], chunk, device, heads)


# --- v_wz networks: every primitive, pivots included, is a commanded body twist -------------------


def load_network_vwz(path: pathlib.Path, device: torch.device, ctg: CostToGo) -> ArcDivergenceNet:
    """`load_network`'s v_wz counterpart: a (v_drive, wz_drive)-commanded checkpoint, which unlike
    a kappa one can describe a pivot, so a pivot lattice (n_prim 7) is fine here. Same "silently
    wrong, not broken" asserts on the pinned lattice constants.

    Either label_mode is accepted and the caller reads `model.target_names`, because the two carry
    very different information about a PIVOT. Measured on the 1080 pivot rows of
    `dataset_arc_my_config_M300_R8_seed0.h5` (interacting vs near-miss medians):

        e_pos   0.0624 / 0.0595 =    1.0x   -- blind; ostrich's skid through a point turn is a
                                              near-constant the kinematic twin has no term for
        e_yaw   0.0256 / 0.0198 =    1.3x   -- blind for the same reason
        e_rot   0.2660 / 0.0209 =   12.7x   -- usable (pos_rot)
        e_pitch 0.2369 / 0.0002 = 1377x     -- the signal (pos_rpy): the rim climbs the feature
                                              while the twin's three contacts stay on flat ground

    So a pos_rpy checkpoint read on e_pitch is the sharper instrument for pivots; pos_rot's e_rot
    is the same signal diluted by the two blind axes it sums with.
    """
    if not pathlib.Path(path).exists():
        raise FileNotFoundError(f"v_wz checkpoint {path} does not exist (--checkpoint-vwz)")
    model, ckpt = load_checkpoint(path, device)
    model.eval()
    assert ckpt["command_mode"] == "v_wz", f"expected command_mode v_wz, got {ckpt['command_mode']}"
    assert ckpt["label_mode"] in ("pos_rot", "pos_rpy"), (
        f"need a pos_rot or pos_rpy checkpoint, got {ckpt['label_mode']}"
    )
    assert math.isclose(ckpt["arc_len"], STEP), f"checkpoint arc_len {ckpt['arc_len']} != {STEP}"
    assert math.isclose(ckpt["min_turn_radius"], float(ctg.robot.min_turn_radius), rel_tol=1e-6), (
        f"checkpoint min_turn_radius {ckpt['min_turn_radius']} != robot {ctg.robot.min_turn_radius}"
    )
    return model


def primitive_commands(ctg: CostToGo) -> np.ndarray:
    """[n_prim, 2] (v_drive, wz_drive) of each lattice primitive, in the solver's primitive order:
    the five forward arcs at (V_NOM, V_NOM * kappa), then -- with pivot_cost > 0 -- the two point
    turns at (0, -+OMEGA_NOM). Checked against the solver's own heading table, so an order or sign
    mismatch with `_build_primitives` fails here instead of mislabelling a primitive."""
    n_prim = ctg.solver.n_prim
    assert n_prim in (N_PRIM_ARC, N_PRIM_ARC + 2), n_prim
    v, wz = twist_from_kappa(np.array(primitive_kappas(float(ctg.robot.min_turn_radius))))
    commands = np.stack([v, wz], axis=-1).astype(np.float32)
    if n_prim > N_PRIM_ARC:
        pivots = np.array([[0.0, -OMEGA_NOM], [0.0, OMEGA_NOM]], dtype=np.float32)
        commands = np.concatenate([commands, pivots])
    heading = ctg.solver._prim_heading.numpy()  # [n_theta, n_prim] end bin
    dbin = (heading - np.arange(N_THETA)[:, None] + N_THETA // 2) % N_THETA - N_THETA // 2
    assert np.array_equal(np.sign(dbin), np.sign(commands[:, 1])[None].repeat(N_THETA, 0)), (
        "primitive order/turn direction differs from the solver's heading table"
    )
    if n_prim > N_PRIM_ARC:
        assert np.all(dbin[:, N_PRIM_ARC] == -1) and np.all(dbin[:, N_PRIM_ARC + 1] == 1)
    return commands


def vwz_error_fields(
    model: ArcDivergenceNet,
    terrain: HeightMapReader,
    ctg: CostToGo,
    chunk: int,
    device: torch.device,
    heads: tuple[str, ...] = ("e_pos", "e_pitch"),
) -> dict[str, np.ndarray]:
    """`arc_error_fields` for a v_wz checkpoint: {head} -> [ny, nx, n_theta, n_prim], covering EVERY
    primitive of the lattice -- the two in-place point turns included, which is the whole reason a
    v_wz net exists here. Commands come from `primitive_commands`, so the plane index is the
    solver's own primitive index and `EdgeGatedLatticeSolver`'s per-primitive tau lines up with it.

    This is the expensive one: unlike `path_arc_errors` it evaluates every lattice pose, ny * nx *
    n_theta of them, which is ~120k on a 7 m map at 0.1 m and minutes on CPU. A gate needs the
    field, though -- the value iteration has to know the error of arcs the final path never takes.
    """
    return _error_fields(model, terrain, ctg, primitive_commands(ctg), chunk, device, heads)


@torch.no_grad()
def path_arc_errors(
    model: ArcDivergenceNet,
    terrain: HeightMapReader,
    ctg: CostToGo,
    states: np.ndarray,
    prims: list[int],
    chunk: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """{head: [len(prims)]} for every head the checkpoint has (`model.target_names`, so pos_rot
    gives e_pos/e_rot and pos_rpy e_pos/e_roll/e_pitch/e_yaw): the v_wz net's predicted error of
    each primitive a traced path takes. Step i leaves lattice state states[i] (row, col, heading
    bin) by primitive prims[i], at the pose `lattice_poses` assigns that state. Only the path is
    evaluated -- a full field is ny*nx*n_theta poses, minutes on CPU for a map that is not
    row-invariant -- which is all a report needs; gating will need the field."""
    n = len(prims)
    assert model.target_transform is not None and len(states) == n + 1
    if n == 0:
        return {k: np.empty(0, np.float32) for k in model.target_names}
    grid = ctg.grid
    taken = np.asarray(states)[:-1]
    poses = np.stack(
        [
            grid.origin_x + taken[:, 1] * grid.cell_size,
            grid.origin_y + taken[:, 0] * grid.cell_size,
            (taken[:, 2] + 0.5) * 2.0 * np.pi / N_THETA,
        ],
        axis=-1,
    )
    commands = primitive_commands(ctg)[np.asarray(prims, dtype=np.int64)]  # [n, 2]
    out = np.empty((n, len(model.target_names)), np.float32)
    for i in range(0, n, chunk):
        patch = torch.from_numpy(sample_patches(terrain, poses[i : i + chunk], model.patch_spec))
        code = model.terrain_code(patch[:, None].to(device))[..., 0, 0]
        command = torch.from_numpy(commands[i : i + chunk]).to(device)
        out[i : i + chunk] = model.target_transform.inverse(model._head(code, command)).cpu().numpy()
    return {k: out[:, i] for i, k in enumerate(model.target_names)}


if __name__ == "__main__":
    import warp as wp

    from feasibility.heightmap.create_pivot_pocket import build_pivot_pocket
    from feasibility.lattice_learning.model import TargetTransform
    from feasibility.planning.gated_lattice import make_cost_to_go

    wp.init()
    if not wp.is_cuda_available():
        raise SystemExit("CUDA not available -- the lattice is built by Warp CUDA kernels.")
    terrain, _ = build_pivot_pocket()  # not row-invariant: the case the full field is slow on
    _, grid = terrain.to_hstack("cuda")
    cpu = torch.device("cpu")

    # --- the command table: order and turn direction agree with the solver's own primitives -----
    for pivot_cost, n_prim in ((0.0, N_PRIM_ARC), (0.15, N_PRIM_ARC + 2)):
        ctg = make_cost_to_go(grid, pivot_cost=pivot_cost)
        commands = primitive_commands(ctg)  # asserts the heading table inside
        assert commands.shape == (n_prim, 2) and np.all(commands[:N_PRIM_ARC, 0] == 0.6)
        if n_prim > N_PRIM_ARC:
            assert np.all(commands[N_PRIM_ARC:, 0] == 0.0)
            assert np.allclose(commands[N_PRIM_ARC:, 1], (-OMEGA_NOM, OMEGA_NOM))
        print(f"[commands] pivot_cost {pivot_cost}: {n_prim} primitives match the solver's table")

    # --- path_arc_errors == the model asked pose by pose (random weights, so any mix-up of poses,
    # primitives or command columns shows up as a number, not as a shape error) ------------------
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    def randomised(**kw) -> ArcDivergenceNet:
        net = ArcDivergenceNet(target_transform=TargetTransform.fit(torch.rand(64, 2)), **kw).eval()
        with torch.no_grad():
            for p in net.parameters():
                p.add_(0.05 * torch.randn_like(p))
        return net

    net = randomised(command_mode="v_wz", label_mode="pos_rot")
    n = 12
    states = np.column_stack([rng.integers(30, 90, n + 1), rng.integers(30, 90, n + 1),
                              rng.integers(0, N_THETA, n + 1)])
    prims = [int(p) for p in rng.integers(0, ctg.solver.n_prim, n)]
    prims[:2] = [N_PRIM_ARC, N_PRIM_ARC + 1]  # both pivots are in the sample
    got = path_arc_errors(net, terrain, ctg, states, prims, chunk=5, device=cpu)  # 3 chunks
    poses = lattice_poses(ctg, np.arange(grid.cells_y)).reshape(grid.cells_y, grid.cells_x, N_THETA, 3)
    for i, p in enumerate(prims):
        pose = poses[states[i, 0], states[i, 1], states[i, 2]][None]
        patch = torch.from_numpy(sample_patches(terrain, pose, net.patch_spec))[:, None]
        want = net.predict(patch, torch.from_numpy(primitive_commands(ctg)[p][None]))[0]
        assert np.allclose(got["e_pos"][i], want[0].item(), rtol=1e-4, atol=1e-6), i
        assert np.allclose(got["e_rot"][i], want[1].item(), rtol=1e-4, atol=1e-6), i
    assert got["e_pos"].std() > 1e-6, "random net gave a constant output: the check would be vacuous"
    assert len(path_arc_errors(net, terrain, ctg, states[:1], [], 5, cpu)["e_pos"]) == 0
    print(f"[path] {n} (pose, primitive) pairs == per-pose model.predict "
          f"(e_pos {got['e_pos'].min():.3f}..{got['e_pos'].max():.3f})")

    # --- a pos_rpy checkpoint returns ITS four heads, not a hardcoded pair ----------------------
    rpy = randomised(command_mode="v_wz", label_mode="pos_rpy")
    rpy.target_transform = TargetTransform.fit(torch.rand(64, 4))
    got_rpy = path_arc_errors(rpy, terrain, ctg, states, prims, chunk=5, device=cpu)
    assert tuple(got_rpy) == tuple(rpy.target_names) == ("e_pos", "e_roll", "e_pitch", "e_yaw"), got_rpy
    for i, p in enumerate(prims):
        pose = poses[states[i, 0], states[i, 1], states[i, 2]][None]
        patch = torch.from_numpy(sample_patches(terrain, pose, rpy.patch_spec))[:, None]
        want = rpy.predict(patch, torch.from_numpy(primitive_commands(ctg)[p][None]))[0]
        for k, name in enumerate(rpy.target_names):
            assert np.allclose(got_rpy[name][i], want[k].item(), rtol=1e-4, atol=1e-6), (i, name)
    assert len(path_arc_errors(rpy, terrain, ctg, states[:1], [], 5, cpu)) == 4
    print(f"[path] pos_rpy checkpoint returns all {len(got_rpy)} heads: {', '.join(got_rpy)}")

    # --- vwz_error_fields: the FIELD the pivot gate reads, on a small patch of the lattice -------
    # A field over the whole pocket is 345k poses, so check a cropped CostToGo instead: same code
    # path, same reshape, 4 rows of it. The net has random weights, so a swapped primitive plane or
    # a transposed reshape shows up as a wrong number rather than a shape error.
    small = HeightMapReader(terrain.H[:40, :40].copy(), origin=(terrain.x0, terrain.y0),
                            cell=terrain.cell)
    _, s_grid = small.to_hstack("cuda")
    s_ctg = make_cost_to_go(s_grid, pivot_cost=0.15)
    assert s_ctg.solver.n_prim == N_PRIM_ARC + 2
    fields = vwz_error_fields(rpy, small, s_ctg, chunk=512, device=cpu,
                              heads=("e_pos", "e_pitch", "e_yaw"))
    for name, f in fields.items():
        assert f.shape == (s_grid.cells_y, s_grid.cells_x, N_THETA, s_ctg.solver.n_prim), (name, f.shape)
    s_poses = lattice_poses(s_ctg, np.arange(s_grid.cells_y)).reshape(
        s_grid.cells_y, s_grid.cells_x, N_THETA, 3
    )
    s_cmd = primitive_commands(s_ctg)
    for r, c, t, p in ((0, 0, 0, 0), (7, 13, 5, N_PRIM_ARC), (22, 31, 19, N_PRIM_ARC + 1),
                       (39, 39, 23, 3)):
        patch = torch.from_numpy(sample_patches(small, s_poses[r, c, t][None], rpy.patch_spec))[:, None]
        want = rpy.predict(patch, torch.from_numpy(s_cmd[p][None]))[0]
        for k, name in enumerate(fields):
            got = fields[name][r, c, t, p]
            assert np.allclose(got, want[rpy.target_names.index(name)].item(), rtol=1e-4, atol=1e-6), (
                (r, c, t, p), name, got
            )
    pivots = fields["e_pitch"][..., N_PRIM_ARC:]
    assert not np.allclose(pivots[..., 0], pivots[..., 1]), "the two point turns predict the same"
    assert not np.allclose(pivots[..., 0], fields["e_pitch"][..., 0]), "pivot plane == an arc plane"
    print(f"[fields] vwz_error_fields {tuple(fields['e_pitch'].shape)} == per-pose model.predict, "
          f"all {s_ctg.solver.n_prim} primitives")
    del s_ctg

    # --- predict_arcs on kappa commands (the refactor) still equals the model asked directly -----
    knet = randomised(command_mode="kappa", label_mode="pos_rot")
    kappas = np.array(primitive_kappas(float(ctg.robot.min_turn_radius)))
    pose = poses[60, 60, :4].reshape(-1, 3)
    out = predict_arcs(knet, terrain, pose, kappas[:, None], chunk=3, device=cpu)
    for i in range(len(pose)):
        patch = torch.from_numpy(sample_patches(terrain, pose[i : i + 1], knet.patch_spec))[:, None]
        for p, k in enumerate(kappas):
            want = knet.predict(patch, torch.full((1, 1), float(k)))[0].numpy()
            assert np.allclose(out[i, p], want, rtol=1e-4, atol=1e-6), (i, p)
    print("[predict_arcs] kappa commands == per-pose model.predict")
