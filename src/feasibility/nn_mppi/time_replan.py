"""Replan timing of vanilla MPPI and the nn-cost MPPI on this machine: timing only, nothing is driven.

Builds the planner `closed_loop.py` drove (`closed_loop.node_planner`: the ROS node's MPPI at the
net's horizon 31 / 4 knots, the cost-to-go solved once) and times replans the way the node issues
them: `cost.update(start)`, `replan`, then the plan read back to the host, which waits for the GPU.
Two arms, built and timed one after the other so each has the GPU to itself:

  * `vanilla`: no hook;
  * `nn w<N>`: `WindowCost` charging windows 0 .. N-1 (knot-aligned, design.md section 3). Window 0
    has one patch, run through the torch trunk in `update`; windows 1-2 sample every rollout's patch
    inside the refine graph and run `WarpTrunk`, then the Warp head. The default baseline "none"
    is ONE inference per window; "flat" also runs the head on a level patch and subtracts it.

Each arm runs `--warmup` untimed replans (the first captures the CUDA graph; on a new machine Warp
also compiles its kernels once, into its kernel cache), then `--frames` timed ones from a start pose
jittered by a few cm per frame. Reported per arm: wall ms per replan (median, p90, p99, max), the GPU
ms of `update` (window 0's patch + torch trunk) and the mean GPU ms per refine stage from
`MppiGpu`'s CUDA events (the hook runs inside "cost"). Reading those events syncs after each refine,
which costs nothing at 1 refine (the readback syncs anyway) and a few percent with several.

Writes `outputs/nn_mppi/timing_<host>_b<batch>r<n_refine>[_<tag>].{h5,txt}`: every replan's wall
time per arm, the update times, the stage means, the config and the GPU/software versions.

CLI parameters:
    --checkpoint PATH     mppi_learning train.py checkpoint (required)
    --map PATH|flat       heightmap (.png or stem) with start/goal in its .yaml sidecar, or `flat`
                          (default; needs no assets)
    --batch INT           MPPI rollouts (default 512)
    --n-refine INT        refines per replan (default 1)
    --windows INT         windows the nn arm charges, 1 to 3 (default 3)
    --baseline none|flat  the nn cost's baseline (default none: one inference)
    --weight W            cost weight on every target (default 100; does not change the timing)
    --frames INT          timed replans per arm (default 200)
    --warmup INT          untimed replans per arm first (default 10)
    --tag STR             output file suffix

Usage:
    python src/feasibility/nn_mppi/time_replan.py --checkpoint outputs/checkpoints/<ckpt>.pt
    python src/feasibility/nn_mppi/time_replan.py --checkpoint outputs/checkpoints/<ckpt>.pt --batch 4096 --n-refine 3
"""
from __future__ import annotations

import argparse
import pathlib
import platform
import time

import h5py
import numpy as np
import torch
import warp as wp
from helhest.control.mppi import MppiGpu

from feasibility.comparator.common import init_warp_device
from feasibility.nn_mppi.closed_loop import load_map
from feasibility.nn_mppi.closed_loop import node_planner
from feasibility.nn_mppi.closed_loop import OUT
from feasibility.nn_mppi.closed_loop import routing_field
from feasibility.nn_mppi.closed_loop import STAGES
from feasibility.nn_mppi.mppi_cost import _load
from feasibility.nn_mppi.mppi_cost import WindowCost

DEVICE = "cuda:0"
ROUTING_CELL = 0.32  # closed_loop.py's --routing-cell default
JITTER = (0.05, 0.05, 0.05)  # start-pose jitter per frame (m, m, rad): replans differ, the warm start stays useful


def time_arm(planner: MppiGpu, cost: WindowCost | None, start: np.ndarray, goal: tuple[float, float],
             args: argparse.Namespace) -> dict[str, np.ndarray]:
    """`--warmup` + `--frames` replans as the node issues them; the timed ones' wall and update ms
    and the mean GPU ms per refine stage."""
    rng = np.random.default_rng(0)
    before, after = wp.Event(DEVICE, enable_timing=True), wp.Event(DEVICE, enable_timing=True)
    wall, update = [], []
    for frame in range(args.warmup + args.frames):
        if frame == args.warmup:
            planner.reset_timing()
        state = start + rng.normal(0.0, JITTER)
        t0 = time.perf_counter()
        if cost is not None:
            wp.record_event(before)
            cost.update(state)
            wp.record_event(after)
        planner.replan(state, goal, args.n_refine)
        planner.nominal()  # the plan read back, as the node does: waits for the GPU
        t1 = time.perf_counter()
        if frame >= args.warmup:
            wall.append((t1 - t0) * 1e3)
            if cost is not None:
                update.append(wp.get_event_elapsed_time(before, after))
    stats = planner.timing_stats()
    return {"wall_ms": np.asarray(wall, np.float32), "update_ms": np.asarray(update, np.float32),
            "stage_ms": np.array([stats[s]["mean_ms"] for s in STAGES], np.float32)}


def table(results: dict[str, dict[str, np.ndarray]]) -> str:
    lines = [f"{'arm':>10} | wall ms per replan: {'median':>7} {'p90':>7} {'p99':>7} {'max':>7} | {'update':>6} | "
             "per refine, GPU ms: " + " ".join(f"{s:>8}" for s in STAGES)]
    for name, r in results.items():
        w = r["wall_ms"]
        update = f"{np.median(r['update_ms']):6.2f}" if r["update_ms"].size else f"{'-':>6}"
        lines.append(f"{name:>10} |                     {np.median(w):7.2f} {np.percentile(w, 90):7.2f} "
                     f"{np.percentile(w, 99):7.2f} {w.max():7.2f} | {update} |                     "
                     + " ".join(f"{t:8.2f}" for t in r["stage_ms"]))
    vanilla, nn = (np.median(r["wall_ms"]) for r in results.values())
    lines.append(f"nn overhead: {nn - vanilla:.2f} ms per replan (median), {nn / vanilla:.1f}x vanilla")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True, help="mppi_learning checkpoint")
    parser.add_argument("--map", default="flat", help="heightmap path or 'flat'")
    parser.add_argument("--batch", type=int, default=512, help="MPPI rollouts")
    parser.add_argument("--n-refine", type=int, default=1, help="refines per replan")
    parser.add_argument("--windows", type=int, default=3, help="windows the nn arm charges, 1 to 3")
    parser.add_argument("--baseline", choices=("none", "flat"), default="none", help="the nn cost's baseline")
    parser.add_argument("--weight", type=float, default=100.0, help="cost weight on every target")
    parser.add_argument("--frames", type=int, default=200, help="timed replans per arm")
    parser.add_argument("--warmup", type=int, default=10, help="untimed replans per arm first")
    parser.add_argument("--tag", default="", help="output file suffix")
    args = parser.parse_args()
    if args.warmup < 1 or args.frames < 1:
        raise SystemExit("--warmup and --frames must be at least 1 (the first replan captures the graph)")

    init_warp_device(DEVICE)
    device = wp.get_device(DEVICE)
    system = {"host": platform.node(), "gpu": device.name, "arch": f"sm_{device.arch}", "torch": str(torch.__version__),
              "warp": wp.__version__, "cuda_driver": ".".join(map(str, wp.get_cuda_driver_version()))}
    terrain, map_name, start, goal = load_map(args.map, None, None)
    net, attrs, blur = _load(args.checkpoint)
    k_turn, mu = float(attrs["k_turn"]), float(attrs["mu"])
    lattice, lattice_grid, vcap = routing_field(terrain, goal, k_turn, ROUTING_CELL, 0.0, DEVICE)
    header = (f"[system]  " + ", ".join(f"{k} {v}" for k, v in system.items()) + "\n"
              f"[config]  {args.batch} rollouts, {args.n_refine} refine(s); nn: {args.windows} window(s), baseline "
              f"{args.baseline!r}; map {map_name}; {args.warmup} + {args.frames} replans per arm; {args.checkpoint.name}")
    print(header)

    results = {}
    for name, n_windows in (("vanilla", None), (f"nn w{args.windows}", args.windows)):
        planner = node_planner(terrain, k_turn, mu, args.batch, 0.0, seed=0, device=DEVICE)
        planner.set_lattice(lattice, lattice_grid)
        planner.cw.lattice_cap = vcap
        cost = None
        if n_windows is not None:
            cost = WindowCost(net, attrs, planner, {k: args.weight for k in net.target_names}, blur_terrain=blur,
                              baseline=args.baseline, n_windows=n_windows)
            planner.set_cost_hook(cost)
        results[name] = time_arm(planner, cost, start, goal, args)
        print(f"[arm]     {name}: median {np.median(results[name]['wall_ms']):.2f} ms")
        del planner, cost  # free this arm's buffers before the next is built
    report = table(results)
    print("\n" + report)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"timing_{system['host']}_b{args.batch}r{args.n_refine}" + (f"_{args.tag}" if args.tag else "")
    with h5py.File(OUT / f"{stem}.h5", "w") as f:
        f.attrs.update(system)
        f.attrs.update({k: str(v) for k, v in vars(args).items()})
        f.attrs["map"] = map_name
        f.attrs["stages"] = list(STAGES)
        for name, r in results.items():
            group = f.create_group(name.replace(" ", "_"))
            for key, value in r.items():
                group[key] = value
    (OUT / f"{stem}.txt").write_text(header + "\n\n" + report + "\n")
    print(f"\n[out]     {OUT / stem}.h5 / .txt")


if __name__ == "__main__":
    main()
