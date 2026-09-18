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
check (`feasibility.planning.gated_lattice.EdgeGatedLatticeSolver`); helhest_stack is not modified. The network is queried at exactly
the poses `CostToGo`'s settle judges -- (origin_x + c*cell, origin_y + r*cell, heading-bin centre)
-- with the curvature of each of the lattice's five forward primitives (`arc.primitive_kappas`).

THRESHOLDS. Two global constants, TAU_POS = 0.1741 m and TAU_PITCH = 0.0933 rad (kept in
`feasibility.planning.planners`, which every gated planner reads its defaults from), applied to every
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

INFERENCE RUNS ON CPU, with a checked row-invariance shortcut -- see `feasibility.planning.arc_network`.
The predicted error fields cross host->device once per map.

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

from feasibility.heightmap import HeightMapReader
from feasibility.heightmap.create_uphill_series import ASSETS_DIR
from feasibility.heightmap.create_uphill_series import uphill_ramp
from feasibility.heightmap.create_uphill_series import uphill_series_paths
from feasibility.lattice_learning.arc import primitive_kappas
from feasibility.lattice_learning.patch import sample_patches
from feasibility.planning.arc_network import arc_error_fields
from feasibility.planning.arc_network import lattice_poses
from feasibility.planning.arc_network import load_network
from feasibility.planning.arc_network import predict_arcs
from feasibility.planning.gated_lattice import arm_result
from feasibility.planning.gated_lattice import build_gated_solver
from feasibility.planning.gated_lattice import gated_solve
from feasibility.planning.gated_lattice import lattice_state
from feasibility.planning.gated_lattice import make_cost_to_go
from feasibility.planning.gated_lattice import N_THETA
from feasibility.planning.planners import DEFAULT_CHECKPOINT
from feasibility.planning.planners import TAU_PITCH
from feasibility.planning.planners import TAU_POS

PASS_DEG, FAIL_DEG = 60.0, 65.0  # ostrich climbs / fails -- the bracket the taus were set from
CRITERIA = {"nn_pos": "e_pos", "nn_pitch": "e_pitch"}  # arm -> network head it gates on
ARMS = ("off", "on", *CRITERIA)


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
    ctg = make_cost_to_go(grid)
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
    ctg = make_cost_to_go(grid)
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
