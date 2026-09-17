"""Lattice-planner benchmark on the uphill ramp series: no `blocked` vs `blocked` vs arcs gated by the
trained arc-divergence network.

    python src/feasibility/heightmap/create_uphill_series.py      # maps first
    python benchmarks/bench_uphill_nn.py
    python benchmarks/bench_uphill_nn.py --plot-dir outputs/bench_uphill_nn
    python benchmarks/bench_uphill_nn.py --tau-pos 0.15 --tau-pitch 0.08
    python benchmarks/bench_uphill_nn.py --self-test

Each map (`heightmap/create_uphill_series.py`) is flat ground, one full-width face rising 0.75 m at
`up_deg`, then a plateau holding the goal. Start and goal are shared by the whole series, so "does
the planner reach the goal" is a clean verdict on "can this robot climb a face this steep".

ARMS. All four share one `CostToGo` (helhest_stack's settle + lattice value iteration, n_theta=24,
0.3 m arcs) and its `graded_tilt` soft cost, so they differ ONLY in what makes a transition
infeasible:

    off       nothing -- `blocked` replaced by zeros (bench_ramp_series.py's OFF arm)
    on        the shipped pipeline: the settle's per-POSE `blocked` field
    nn_pos    `blocked` zeros; an ARC (row, col, heading, primitive) is pruned when the network
              predicts e_pos > tau_pos for it
    nn_pitch  same, with e_pitch > tau_pitch

The per-arc gate is a local copy of `lattice_solver._relax_lattice_pose_kernel` with one extra
check (`EdgeGatedLatticeSolver`); helhest_stack is not modified. The network is queried at exactly
the poses `CostToGo`'s settle judges -- (origin_x + c*cell, origin_y + r*cell, heading-bin centre)
-- with the curvature of each of the lattice's five forward primitives (`arc.primitive_kappas`).

THRESHOLDS. Two global constants, TAU_POS = 0.1741 m and TAU_PITCH = 0.0933 rad, applied to every
map (override with --tau-pos / --tau-pitch). They were calibrated ONCE, for the default checkpoint:
ostrich climbs the 60 deg face and flips at the foot of the 65 deg one
(`demos/ostrich_ramp_crossing.py +series=uphill`, 3/3 repeats each). For each of those two maps we
took tau*(map) = the smallest tau at which the gated planner still reaches the goal (the worst
predicted error on the best climbing path, found by a minimax value iteration), and set tau halfway
between the two:

    e_pos    tau*(60) = 0.1712 m    tau*(65) = 0.1769 m    -> 0.1741 m
    e_pitch  tau*(60) = 0.0894 rad  tau*(65) = 0.0972 rad  -> 0.0933 rad

A different checkpoint needs its own thresholds; these numbers do not carry over.

CAVEATS
    * Calibrated and evaluated on the same series, so the NN arms stopping at 60 deg is by
      construction. The gaps between the bracket maps are narrow (3% for e_pos, 8% for e_pitch).
    * The dataset only spawns trials at settle-feasible poses, so arc starts in the middle of a
      steep face are rare in training; predictions there are partly out of distribution.
    * A hard gate can disconnect the lattice (lattice_learning/design.md section 7a). An unreachable
      goal is reported as such, not hidden.

INFERENCE RUNS ON CPU. This env's torch (cu128) does not support the GTX 1050 (sm_61); Warp does, so
the planning stays on CUDA. The predicted error fields cross host->device once per map. To make CPU
inference affordable: when every row of the map is identical (true for this whole series), a
body-frame patch depends only on (column, heading) -- `HeightMapReader.sample` clamps at the Y
edges, which preserves the invariance exactly -- so one row is evaluated and broadcast. This is
checked, not assumed; any other map takes the full path (~100 s/map on CPU).

CLI parameters:
    --dir PATH            uphill series (default: assets/uphill_series)
    --checkpoint PATH     pos_rpy ArcDivergenceNet checkpoint
                          (default: outputs/checkpoints/dataset_arc_my_config_M200_R8_seed0_rpy.pt)
    --tau-pos FLOAT       e_pos arc threshold [m] (default: 0.1741, see THRESHOLDS)
    --tau-pitch FLOAT     e_pitch arc threshold [rad] (default: 0.0933, see THRESHOLDS)
    --chunk INT           patches per network batch (default: 4096)
    --torch-device STR    device for network inference (default: cpu, see above)
    --plot-dir PATH       also write one PNG per map, <dir>/<map>.png: a 2x2 bird's-eye view of the
                          elevation, one panel per arm, with that arm's planned path when it has one
    --self-test           synthetic checks: gated solver with the gate open == off arm bit-for-bit,
                          split-head inference == predict(), row-invariance shortcut == full gather

CUDA-only (Warp); skips cleanly without a GPU.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import time

import numpy as np
import torch
import warp as wp
import yaml
from helhest import dynamics
from helhest.planning.costtogo import CostToGo
from helhest.planning.lattice_solver import LatticeValueSolver

from feasibility.heightmap import HeightMapReader
from feasibility.heightmap.create_uphill_series import ASSETS_DIR
from feasibility.heightmap.create_uphill_series import uphill_ramp
from feasibility.heightmap.create_uphill_series import uphill_series_paths
from feasibility.lattice_learning.arc import primitive_kappas
from feasibility.lattice_learning.model import ArcDivergenceNet
from feasibility.lattice_learning.patch import sample_patches
from feasibility.lattice_learning.train import load_checkpoint

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "outputs" / "checkpoints" / "dataset_arc_my_config_M200_R8_seed0_rpy.pt"
)
N_THETA = 24
STEP = 0.3  # [m] CostToGo's default arc length -- asserted against the checkpoint's arc_len
# One-time calibration for DEFAULT_CHECKPOINT, see THRESHOLDS in the module docstring
TAU_POS = 0.1741  # [m]
TAU_PITCH = 0.0933  # [rad]
PASS_DEG, FAIL_DEG = 60.0, 65.0  # ostrich climbs / fails -- the bracket the taus were set from
CRITERIA = {"nn_pos": "e_pos", "nn_pitch": "e_pitch"}  # arm -> network head it gates on
ARMS = ("off", "on", *CRITERIA)


@wp.kernel
def _relax_gated_kernel(
    dist_in: wp.array(dtype=wp.float32, ndim=3),
    blocked: wp.array(dtype=wp.float32, ndim=3),
    tilt: wp.array(dtype=wp.float32, ndim=3),
    prim_dr: wp.array(dtype=wp.int32, ndim=2),
    prim_dc: wp.array(dtype=wp.int32, ndim=2),
    prim_heading: wp.array(dtype=wp.int32, ndim=2),
    prim_cost: wp.array(dtype=wp.float32, ndim=2),
    sweep_dr: wp.array(dtype=wp.int32, ndim=3),
    sweep_dc: wp.array(dtype=wp.int32, ndim=3),
    sweep_n: wp.array(dtype=wp.int32, ndim=2),
    n_prim: wp.int32,
    tilt_weight: wp.float32,
    inf: wp.float32,
    arc_error: wp.array(dtype=wp.float32, ndim=4),  # [h, w, n_theta, n_prim] predicted error
    tau: wp.float32,  # arc pruned when arc_error > tau
    dist_out: wp.array(dtype=wp.float32, ndim=3),
    changed: wp.array(dtype=wp.int32),
):
    """lattice_solver._relax_lattice_pose_kernel, verbatim, plus ONE gate: primitive p out of pose
    (r, c, t) is skipped when arc_error[r, c, t, p] > tau. Indexed at the source pose, like every
    other per-arc table the relaxation reads."""
    r, c, t = wp.tid()
    h = dist_in.shape[0]
    w = dist_in.shape[1]
    if blocked[r, c, t] > 0.5:
        dist_out[r, c, t] = inf
        return
    best = dist_in[r, c, t]
    for p in range(n_prim):
        ok = int(1)
        if arc_error[r, c, t, p] > tau:
            ok = 0
        ns = sweep_n[t, p]
        tsum = float(0.0)
        for s in range(ns):
            sr = r + sweep_dr[t, p, s]
            sc = c + sweep_dc[t, p, s]
            inb = int(0)
            if sr >= 0 and sr < h and sc >= 0 and sc < w:
                inb = 1
            scr = wp.clamp(sr, 0, h - 1)
            scc = wp.clamp(sc, 0, w - 1)
            if inb == 0 or blocked[scr, scc, t] > 0.5:
                ok = 0
            tsum += tilt[scr, scc, t]
        if ok == 1:
            nr = r + prim_dr[t, p]
            nc = c + prim_dc[t, p]
            if nr >= 0 and nr < h and nc >= 0 and nc < w:
                arc = prim_cost[t, p]
                if ns > 0:
                    arc = arc * (1.0 + tilt_weight * tsum / float(ns))
                best = wp.min(best, arc + dist_in[nr, nc, prim_heading[t, p]])
    dist_out[r, c, t] = best
    if best < dist_in[r, c, t]:
        changed[0] = 1


class EdgeGatedLatticeSolver(LatticeValueSolver):
    """LatticeValueSolver whose relaxation also prunes individual ARCS by a predicted error field.
    Built with the same arguments CostToGo uses, so its primitive tables are identical (asserted in
    build_gated_solver). `set_gate` must be called before `_record_solve`."""

    def set_gate(self, arc_error: wp.array, tau: float) -> None:
        assert arc_error.shape == (self.height, self.width, self.n_theta, self.n_prim)
        self._arc_error = arc_error
        self._tau = float(tau)

    def _relax(
        self,
        dist_in: wp.array,
        dist_out: wp.array,
        blocked: wp.array,
        tilt: wp.array,
        tilt_weight: float,
    ) -> None:
        wp.launch(
            _relax_gated_kernel,
            dim=(self.height, self.width, self.n_theta),
            inputs=[
                dist_in,
                blocked,
                tilt,
                self._prim_dr,
                self._prim_dc,
                self._prim_heading,
                self._prim_cost,
                self._sweep_dr,
                self._sweep_dc,
                self._sweep_n,
                self.n_prim,
                float(tilt_weight),
                self._inf,
                self._arc_error,
                self._tau,
            ],
            outputs=[dist_out, self._changed],
            device=self.device,
        )


def build_gated_solver(ctg: CostToGo) -> EdgeGatedLatticeSolver:
    """An EdgeGatedLatticeSolver with CostToGo's own lattice, primitive tables asserted identical --
    trace_states reads ctg.solver's tables for every arm."""
    grid = ctg.grid
    solver = EdgeGatedLatticeSolver(
        grid.cell_size,
        grid.cells_y,
        grid.cells_x,
        n_theta=N_THETA,
        turn_radius=ctg.robot.min_turn_radius,
        step=STEP,
        device=ctg.device,
    )
    for name in ("_prim_dr", "_prim_dc", "_prim_heading", "_prim_cost", "_sweep_dr", "_sweep_dc"):
        assert np.array_equal(getattr(solver, name).numpy(), getattr(ctg.solver, name).numpy()), name
    return solver


# --- network --------------------------------------------------------------------------------------


def load_network(path: pathlib.Path, device: torch.device, ctg: CostToGo) -> ArcDivergenceNet:
    """Loads the checkpoint and asserts the pinned constants match the lattice it will gate
    (design.md section 4c): a net trained on other arcs is silently wrong, not broken."""
    model, ckpt = load_checkpoint(path, device)
    model.eval()
    assert ckpt["label_mode"] == "pos_rpy", f"need a pos_rpy checkpoint, got {ckpt['label_mode']}"
    assert ckpt["command_mode"] == "kappa", f"expected command_mode kappa, got {ckpt['command_mode']}"
    assert math.isclose(ckpt["arc_len"], STEP), f"checkpoint arc_len {ckpt['arc_len']} != {STEP}"
    assert math.isclose(ckpt["min_turn_radius"], float(ctg.robot.min_turn_radius), rel_tol=1e-6), (
        f"checkpoint min_turn_radius {ckpt['min_turn_radius']} != robot "
        f"{ctg.robot.min_turn_radius}"
    )
    # arc.primitive_kappas is in _build_primitives' `turns` order (asserted by arc.py's own
    # self-test); pivots would append primitives the net has no curvature for
    assert ctg.solver.n_prim == 5, f"expected 5 forward primitives (pivots off), got {ctg.solver.n_prim}"
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
    poses CostToGo.__init__ assigns to its settle (cell corner + bin-centre heading), so the `on`
    arm and the network arms judge the same lattice states."""
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
) -> dict[str, np.ndarray]:
    """{"e_pos", "e_pitch"} -> [ny, nx, n_theta, n_prim] predicted error per lattice arc. When all
    rows of the map are identical the patch cannot depend on the row (see module docstring), so row
    0 is evaluated and broadcast; otherwise every row is."""
    ny, nx = ctg.grid.cells_y, ctg.grid.cells_x
    kappas = np.array(primitive_kappas(float(ctg.robot.min_turn_radius)))
    row_invariant = bool(np.all(terrain.H == terrain.H[:1]))
    rows = np.arange(1) if row_invariant else np.arange(ny)
    pred = predict_arcs(model, terrain, lattice_poses(ctg, rows), kappas, chunk, device)
    pred = pred.reshape(len(rows), nx, N_THETA, len(kappas), -1)
    names = model.target_names
    fields = {name: pred[..., names.index(name)] for name in CRITERIA.values()}
    if row_invariant:
        fields = {k: np.ascontiguousarray(np.broadcast_to(v, (ny, *v.shape[1:]))) for k, v in fields.items()}
    return fields


# --- planning -------------------------------------------------------------------------------------


def lattice_state(x: float, y: float, yaw: float, ctg: CostToGo) -> tuple[int, int, int]:
    """World pose -> (row, col, heading bin), floor mapping as _goal_cell_kernel. The small epsilon
    keeps a start sitting exactly on a cell boundary from flooring into the previous cell."""
    grid = ctg.grid
    return (
        int(math.floor((y - grid.origin_y) / grid.cell_size + 1e-6)),
        int(math.floor((x - grid.origin_x) / grid.cell_size + 1e-6)),
        int(math.floor((yaw % (2.0 * math.pi)) / (2.0 * math.pi / N_THETA))) % N_THETA,
    )


def trace_states(
    ctg: CostToGo,
    V: np.ndarray,
    blocked: np.ndarray,
    tilt: np.ndarray,
    start_rct: tuple[int, int, int],
    arc_error: np.ndarray | None = None,
    tau: float = math.inf,
    max_steps: int = 800,
) -> tuple[list[tuple[int, int, int]], list[int], float, bool]:
    """Follow the lattice's own policy from the start -> (states, primitive taken per step, billed
    m, reached). helhest_stack/scripts/bench_ramp_series.py's trace_states restated (scripts/ is not
    importable) with the arc gate added, so the NN arms trace the policy they were solved with."""
    s = ctg.solver
    pdr, pdc = s._prim_dr.numpy(), s._prim_dc.numpy()
    pheading, pcost = s._prim_heading.numpy(), s._prim_cost.numpy()
    sdr, sdc, sn = s._sweep_dr.numpy(), s._sweep_dc.numpy(), s._sweep_n.numpy()
    tilt_weight = float(ctg.flatness_weight)
    gr, gc = (int(v) for v in ctg._goal_rc.numpy())
    ny, nx, _ = V.shape
    vcap = float(ctg._vcap)

    r, c, t = start_rct
    states, prims, arc_len = [(r, c, t)], [], 0.0
    for _ in range(max_steps):
        if abs(r - gr) <= 1 and abs(c - gc) <= 1:
            return states, prims, arc_len, True
        best_p, best_val = -1, np.inf
        for p in range(s.n_prim):
            if arc_error is not None and arc_error[r, c, t, p] > tau:
                continue
            ns, ok, tsum = int(sn[t, p]), True, 0.0
            for si in range(ns):
                sr, sc = r + int(sdr[t, p, si]), c + int(sdc[t, p, si])
                if not (0 <= sr < ny and 0 <= sc < nx) or blocked[sr, sc, t] > 0.5:
                    ok = False
                    break
                tsum += tilt[sr, sc, t]
            if not ok:
                continue
            nr, nc, nt = r + int(pdr[t, p]), c + int(pdc[t, p]), int(pheading[t, p])
            if not (0 <= nr < ny and 0 <= nc < nx):
                continue
            arc = float(pcost[t, p]) * (1.0 + tilt_weight * tsum / ns if ns > 0 else 1.0)
            val = arc + V[nr, nc, nt]
            if val < best_val:
                best_val, best_p = val, p
        if best_p < 0 or best_val >= vcap * 0.9:
            return states, prims, arc_len, False
        arc_len += float(pcost[t, best_p])
        prims.append(best_p)
        r, c, t = r + int(pdr[t, best_p]), c + int(pdc[t, best_p]), int(pheading[t, best_p])
        states.append((r, c, t))
    return states, prims, arc_len, False


def arm_result(
    ctg: CostToGo,
    V: np.ndarray,
    arm_blocked: np.ndarray,
    tilt: np.ndarray,
    settle_blocked: np.ndarray,
    fields: dict[str, np.ndarray],
    start_rct: tuple[int, int, int],
    arc_error: np.ndarray | None = None,
    tau: float = math.inf,
) -> dict:
    """V at the start, the traced path, and an audit of that path against the settle's `blocked`
    and against both network heads (the arcs actually taken)."""
    V = np.minimum(V, ctg._vcap)
    states, prims, arc_len, reached = trace_states(
        ctg, V, arm_blocked, tilt, start_rct, arc_error, tau
    )
    rct = np.array(states)
    taken = rct[:-1]
    max_err = {
        name: float(field[taken[:, 0], taken[:, 1], taken[:, 2], prims].max()) if prims else math.nan
        for name, field in fields.items()
    }
    return dict(
        v_start=float(V[start_rct]),
        reachable=bool(V[start_rct] < ctg._vcap * 0.9),
        reached=reached,
        path_m=arc_len,
        n_poses=len(states),
        n_settle_bad=int((settle_blocked[rct[:, 0], rct[:, 1], rct[:, 2]] > 0.5).sum()),
        max_err=max_err,
        states=rct,
    )


def gated_solve(
    solver: EdgeGatedLatticeSolver,
    ctg: CostToGo,
    zeros: wp.array,
    tilt: wp.array,
    arc_error: wp.array,
    tau: float,
) -> np.ndarray:
    solver.set_gate(arc_error, tau)
    return solver._record_solve(zeros, tilt, ctg._goal_rc, ctg.flatness_weight, False).numpy()


def run_series(args: argparse.Namespace) -> None:
    paths = uphill_series_paths(args.dir)
    if not paths:
        raise SystemExit(f"no uphill_a*.png in {args.dir} -- run heightmap/create_uphill_series.py")
    metas = [yaml.safe_load(p.with_suffix(".yaml").read_text()) for p in paths]
    terrains = [HeightMapReader.load(p) for p in paths]
    _, grid = terrains[0].to_hstack("cuda")
    for p, t, m in zip(paths, terrains, metas):  # one grid, one start, one goal for the series
        assert (t.nx, t.ny, t.cell, t.x0, t.y0) == (grid.cells_x, grid.cells_y, grid.cell_size,
                                                    grid.origin_x, grid.origin_y), f"{p.name} off-grid"
        assert m["start"] == metas[0]["start"] and m["goal"] == metas[0]["goal"], p.name

    robot_params = dynamics.robot_params()
    ctg = CostToGo(grid, robot_params, dynamics.planning_solver(), n_theta=N_THETA, step=STEP,
                   device="cuda")
    gated = build_gated_solver(ctg)
    torch_device = torch.device(args.torch_device)
    model = load_network(args.checkpoint, torch_device, ctg)
    taus = {"nn_pos": float(args.tau_pos), "nn_pitch": float(args.tau_pitch)}
    start = tuple(float(v) for v in metas[0]["start"])
    goal = (float(metas[0]["goal"][0]), float(metas[0]["goal"][1]))
    start_rct = lattice_state(*start, ctg)
    ny, nx = grid.cells_y, grid.cells_x
    print(f"=== uphill NN benchmark: {len(paths)} maps, {ny}x{nx} @ {grid.cell_size} m x {N_THETA} "
          f"headings x {ctg.solver.n_prim} arcs = {ny * nx * N_THETA * ctg.solver.n_prim} arcs each; "
          f"network on {torch_device}, planning on {ctg.device} ===")
    print(f"thresholds: tau_pos = {taus['nn_pos']:.4f} m, tau_pitch = {taus['nn_pitch']:.4f} rad")

    zeros = wp.zeros_like(ctg.blocked)
    maps = []
    for path, terrain, meta in zip(paths, terrains, metas):
        t0 = time.time()
        elev, _ = terrain.to_hstack("cuda")
        v_on = ctg.compute(elev, goal).numpy().copy()
        blocked = ctg.blocked.numpy().copy()
        tilt = ctg.graded_tilt.numpy().copy()
        no_block = np.zeros_like(blocked)
        v_off = ctg.solver._record_solve(zeros, ctg.graded_tilt, ctg._goal_rc,
                                         ctg.flatness_weight, False).numpy().copy()
        fields = arc_error_fields(model, terrain, ctg, args.chunk, torch_device)
        m = dict(
            name=path.name, deg=float(meta["up_deg"]), terrain=terrain,
            on=arm_result(ctg, v_on, blocked, tilt, blocked, fields, start_rct),
            off=arm_result(ctg, v_off, no_block, tilt, blocked, fields, start_rct),
        )
        for arm, head in CRITERIA.items():
            err = wp.array(fields[head], dtype=wp.float32, device=ctg.device)
            v = gated_solve(gated, ctg, zeros, ctg.graded_tilt, err, taus[arm])
            m[arm] = arm_result(ctg, v, no_block, tilt, blocked, fields, start_rct, fields[head],
                                taus[arm])
        maps.append(m)
        print(f"  {path.name}: " + "  ".join(
            f"{arm} {'yes' if m[arm]['reached'] else 'no'}" for arm in ARMS
        ) + f"  ({time.time() - t0:.1f} s)")

    print_report(maps, taus, robot_params)
    if args.plot_dir is not None:
        plot_paths(maps, taus, ctg, start, goal, args.plot_dir)


# --- report ---------------------------------------------------------------------------------------


def _cell(result: dict) -> str:
    v = "UNREACH" if not result["reachable"] else f"{result['v_start']:7.2f}"
    return f"{v:>7} {'yes' if result['reached'] else 'no':>4} {result['n_settle_bad']:3d}"


def steepest_contiguous(maps: list[dict], arm: str) -> float | None:
    """Steepest angle such that the arm reaches the goal on it and on every shallower map."""
    best = None
    for m in maps:
        if not m[arm]["reached"]:
            break
        best = m["deg"]
    return best


def print_report(maps: list[dict], taus: dict, robot) -> None:
    print(f"\n{'deg':>4} | " + " | ".join(f"{arm:<16}" for arm in ARMS))
    print(f"{'':>4} | " + " | ".join(f"{'V*':>7} {'goal':>4} {'bad':>3}" for _ in ARMS))
    print("-" * (7 + 19 * len(ARMS)))
    for m in maps:
        print(f"{m['deg']:4.0f} | " + " | ".join(_cell(m[arm]) for arm in ARMS))
    print("V* = cost-to-go at the start pose; goal = the traced policy reached it; bad = poses on that\n"
          "arm's path the settle calls infeasible.")

    print("\nmax predicted error along each arm's path (e_pos m / e_pitch rad):")
    for m in maps:
        cells = [f"{arm} {m[arm]['max_err']['e_pos']:.3f}/{m[arm]['max_err']['e_pitch']:.3f}"
                 for arm in ARMS]
        print(f"  {m['deg']:4.0f}  " + "   ".join(cells))

    print(f"\nenvelope (settle `blocked`): climb {math.degrees(robot.max_pitch_up):.0f} deg, "
          f"descend {math.degrees(robot.max_pitch_down):.0f} deg, roll "
          f"{math.degrees(robot.max_roll):.0f} deg")
    print(f"physical reference (ostrich, uphill series): climbs {PASS_DEG:.0f} deg, fails "
          f"{FAIL_DEG:.0f} deg")
    for arm in ARMS:
        limit = steepest_contiguous(maps, arm)
        n = sum(m[arm]["reached"] for m in maps)
        tau = f", tau={taus[arm]:.4f}" if arm in CRITERIA else ""
        print(f"  {arm:<9} reaches the goal up to {'none' if limit is None else f'{limit:.0f} deg'} "
              f"({n}/{len(maps)} maps{tau})")


def plot_paths(
    maps: list[dict],
    taus: dict,
    ctg: CostToGo,
    start: tuple[float, float, float],
    goal: tuple[float, float],
    out_dir: pathlib.Path,
) -> None:
    """One PNG per map: 2x2 bird's-eye elevation panels, one per arm, each with the path that arm's
    policy traced from start to goal. A path is drawn only when it reached the goal; the panel title
    says why when it did not. Path vertices are the lattice states (cell corner poses), joined by
    straight segments -- each is one 0.3 m arc, so the chord is within a few cm of it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    grid = ctg.grid
    titles = {
        "off": "default planner, blocked OFF",
        "on": "default planner, blocked ON",
        "nn_pos": f"NN arc gate: e_pos > {taus['nn_pos']:.4f} m",
        "nn_pitch": f"NN arc gate: e_pitch > {taus['nn_pitch']:.4f} rad",
    }
    z_max = max(float(m["terrain"].H.max()) for m in maps)
    for m in maps:
        t = m["terrain"]
        extent = (t.x0, t.x0 + t.nx * t.cell, t.y0, t.y0 + t.ny * t.cell)
        fig, axes = plt.subplots(2, 2, figsize=(14, 7.5), sharex=True, sharey=True,
                                 layout="constrained")
        for ax, arm in zip(axes.flat, ARMS):
            im = ax.imshow(t.H, origin="lower", extent=extent, cmap="viridis", vmin=0.0, vmax=z_max,
                           interpolation="nearest")
            r = m[arm]
            if r["reached"]:
                rct = r["states"]
                ax.plot(grid.origin_x + rct[:, 1] * grid.cell_size,
                        grid.origin_y + rct[:, 0] * grid.cell_size, "-", color="tab:red", lw=2,
                        label="planned path")
                status = (f"path {r['path_m']:.1f} m, V* = {r['v_start']:.2f}, "
                          f"{r['n_settle_bad']} settle-infeasible poses on it")
            elif not r["reachable"]:
                status = "NO FEASIBLE PATH"
            else:
                status = "V* finite, but the traced policy did not arrive"
            ax.plot(start[0], start[1], "o", ms=9, color="white", mec="black", label="start")
            ax.plot(goal[0], goal[1], "*", ms=14, color="gold", mec="black", label="goal")
            ax.set_title(f"{titles[arm]}\n{status}", fontsize=10,
                         color="black" if r["reached"] else "tab:red")
            ax.set_aspect("equal")
        for ax in axes[1]:
            ax.set_xlabel("x [m]")
        for ax in axes[:, 0]:
            ax.set_ylabel("y [m]")
        axes[0, 0].legend(loc="upper left", fontsize=8)
        fig.colorbar(im, ax=axes, label="elevation z [m]", shrink=0.8)
        fig.suptitle(f"{m['name']}: {m['deg']:.0f} deg uphill face "
                     f"(ostrich climbs up to {PASS_DEG:.0f} deg)")
        out = out_dir / f"{m['name']}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
    print(f"saved {len(maps)} figures to {out_dir}/")


# --- self-test ------------------------------------------------------------------------------------


def self_test(args: argparse.Namespace) -> None:
    """Synthetic checks on a small uphill map. Each guards one way this benchmark could silently
    report something other than what it claims."""
    terrain = uphill_ramp(30.0, x0=-3.0, nx=60, extent_y=2.0)  # 60 x 20 cells
    elev, grid = terrain.to_hstack("cuda")
    ctg = CostToGo(grid, dynamics.robot_params(), dynamics.planning_solver(), n_theta=N_THETA,
                   step=STEP, device="cuda")
    gated = build_gated_solver(ctg)
    goal = (2.0, 0.0)
    ctg.compute(elev, goal)
    zeros = wp.zeros_like(ctg.blocked)
    shape = (grid.cells_y, grid.cells_x, N_THETA, ctg.solver.n_prim)

    # 1. gate wide open -> exactly the stock relaxation
    v_off = ctg.solver._record_solve(zeros, ctg.graded_tilt, ctg._goal_rc, ctg.flatness_weight,
                                     False).numpy().copy()
    open_gate = wp.zeros(shape, dtype=wp.float32, device=ctg.device)
    v_gated = gated_solve(gated, ctg, zeros, ctg.graded_tilt, open_gate, math.inf)
    assert np.array_equal(v_off, v_gated), "gated solver with an open gate differs from the stock one"
    print("[gate] open gate == stock LatticeValueSolver, bit for bit")

    # 2. gate shut -> nothing reaches the goal from the start
    start_rct = lattice_state(-2.0, 0.0, 0.0, ctg)
    shut = gated_solve(gated, ctg, zeros, ctg.graded_tilt, open_gate, -1.0)
    assert shut[start_rct] >= 0.5 * float(gated._inf), "a shut gate still reaches the goal"
    print("[gate] shut gate (every arc pruned) == goal unreachable")

    # 3. split-head inference == model.predict(); row shortcut == full gather
    device = torch.device(args.torch_device)
    model = load_network(args.checkpoint, device, ctg)
    kappas = np.array(primitive_kappas(float(ctg.robot.min_turn_radius)))
    poses = lattice_poses(ctg, np.array([3]))[::37]
    split = predict_arcs(model, terrain, poses, kappas, chunk=7, device=device)
    patch = torch.from_numpy(sample_patches(terrain, poses, model.patch_spec))[:, None]
    for p, kappa in enumerate(kappas):
        full = model.predict(patch, torch.full((len(poses), 1), float(kappa))).numpy()
        assert np.allclose(split[:, p], full, atol=1e-5), f"split-head inference differs at arc {p}"
    fields = arc_error_fields(model, terrain, ctg, args.chunk, device)
    rows = np.array([0, grid.cells_y // 2, grid.cells_y - 1])
    direct = predict_arcs(model, terrain, lattice_poses(ctg, rows), kappas, args.chunk, device)
    direct = direct.reshape(len(rows), grid.cells_x, N_THETA, len(kappas), -1)
    for name in CRITERIA.values():
        assert np.allclose(fields[name][rows], direct[..., model.target_names.index(name)], atol=1e-5), name
    print("[network] split-head == predict(); row-invariant broadcast == per-row inference")
    print("all self-checks ok")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=pathlib.Path, default=ASSETS_DIR)
    ap.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--tau-pos", type=float, default=TAU_POS)
    ap.add_argument("--tau-pitch", type=float, default=TAU_PITCH)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--torch-device", type=str, default="cpu")
    ap.add_argument("--plot-dir", type=pathlib.Path, default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    wp.init()
    if not wp.is_cuda_available():
        print("CUDA not available -- the lattice solves are Warp CUDA kernels. Skipping.")
        return
    if args.self_test:
        self_test(args)
    else:
        run_series(args)


if __name__ == "__main__":
    main()
