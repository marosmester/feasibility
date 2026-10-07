"""Times the terrain trunk as `WarpTrunk` against the TensorRT trunk engines from
`build_tensorRt_engine.py`, at the batch MPPI gives it when it charges several windows.

At `--rollouts` R and `--windows` W the trunk sees `mppi_cost.trunk_rows(R, W)` patches per refine
(windows 1 .. W - 1 per rollout, padded, plus the level patch); window 0's single patch is not
timed. The default 512 x 3 is 1025 patches. Forms, all on the same patches in one process:

  * torch eager, torch with `NCHWChannelLayerNorm`, and `HybridTrunk` (cuDNN convs + Warp norm),
    as references; all eager;
  * `WarpTrunk` eager and replayed as a captured CUDA graph (`WarpTrunk.captured`), the form
    `WindowCost` runs inside the refine graph;
  * each `<stem>.trunk.<precision>.engine` found next to the ONNX export, eager
    (`execute_async_v3` per call) and replayed from a torch CUDA graph that captured that call, both
    on one non-default stream and one execution context.

Every form is first checked against `net.terrain_code` (max relative difference, printed, not
asserted: fp16 sits near 1e-2 by design and `build_tensorRt_engine.py` already asserts its
tolerance). Timing interleaves the forms, one trial of each per round, and keeps each form's
fastest trial, because a laptop GPU drops its clock in bursts as it heats and whichever form ran
during a burst would otherwise lose (`channel_layer_norm._time_ms`).

The engines are specific to the GPU and TensorRT version that built them, so build them on this
machine first; an engine whose profile max is below the batch is skipped.

CLI parameters:
    --checkpoint PATH   mppi_learning train.py checkpoint the engines were exported from
    --onnx-dir PATH     where the engines live (default outputs/nn_mppi/onnx)
    --rollouts INT      MPPI rollouts (default 512)
    --windows INT       charged windows (default 3)
    --rounds INT        interleaved timing rounds (default 7)

Usage:
    python src/feasibility/nn_mppi/bench_trunk_tensorrt.py --checkpoint outputs/checkpoints/<ckpt>.pt
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
from collections.abc import Callable

import torch
import warp as wp

from feasibility.nn_mppi.build_tensorRt_engine import Engine
from feasibility.nn_mppi.channel_layer_norm import _load
from feasibility.nn_mppi.channel_layer_norm import _max_rel
from feasibility.nn_mppi.channel_layer_norm import _time_ms
from feasibility.nn_mppi.channel_layer_norm import nchw_layer_norm
from feasibility.nn_mppi.mppi_cost import trunk_rows
from feasibility.nn_mppi.warp_trunk import HybridTrunk
from feasibility.nn_mppi.warp_trunk import WarpTrunk


def graphed(engine: Engine, patch: torch.Tensor) -> tuple[Callable[[], object], torch.Tensor]:
    """`engine`'s call on `patch` captured into a torch CUDA graph on the engine's own stream:
    (replay, output). Shares the engine's execution context, whose activation memory is sized for
    the profile's max batch (~1.4 GB at 4097), so eager and graph forms cannot both have their own."""
    with torch.cuda.stream(engine.stream):
        engine(patch=patch)  # first enqueue does shape-dependent setup, which must not be captured
    engine.stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=engine.stream):
        out = engine(patch=patch)
    graph.replay()
    torch.cuda.synchronize()
    return graph.replay, out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True, help="mppi_learning checkpoint")
    parser.add_argument("--onnx-dir", type=pathlib.Path, default=pathlib.Path("outputs/nn_mppi/onnx"))
    parser.add_argument("--rollouts", type=int, default=512)
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=7, help="interleaved timing rounds")
    args = parser.parse_args()
    wp.init()
    device = torch.device("cuda:0")
    net = _load(args.checkpoint, device)
    fast = nchw_layer_norm(copy.deepcopy(net))
    n = trunk_rows(args.rollouts, args.windows)
    spec = net.patch_spec
    print(f"{torch.cuda.get_device_name(0)}: {n} patches ({args.rollouts} rollouts x {args.windows} windows)")

    torch.manual_seed(0)
    patch = torch.randn(n, 1, spec.ny, spec.nx, device=device) * 0.3
    flat = wp.from_torch(patch[:, 0].contiguous())
    warp_trunk = WarpTrunk(net, n)
    hybrid = HybridTrunk(net)
    forms: dict[str, Callable[[], object]] = {
        "torch eager": lambda: net.terrain_code(patch),
        "torch NCHW-norm": lambda: fast.terrain_code(patch),
        "hybrid (cuDNN+Warp)": lambda: hybrid(patch),
        "WarpTrunk eager": lambda: warp_trunk(flat),
        "WarpTrunk graph": lambda: warp_trunk.captured(flat),
    }
    with torch.no_grad():
        outputs = {"WarpTrunk graph": wp.to_torch(warp_trunk.captured(flat)).reshape(n, -1)}
        for path in sorted(args.onnx_dir.glob(f"{args.checkpoint.stem}.trunk.*.engine")):
            precision = path.suffixes[-2].lstrip(".")
            max_batch = json.loads(path.with_suffix(".json").read_text())["batch_min_opt_max"][2]
            if max_batch < n:
                print(f"[skip] {path.name}: profile max {max_batch} < {n}")
                continue
            # a non-default stream: on the default one TensorRT adds stream syncs to every enqueue
            engine = Engine(path.read_bytes(), torch.cuda.Stream())
            forms[f"TensorRT {precision} eager"] = lambda engine=engine: engine(patch=patch)
            outputs[f"TensorRT {precision} eager"] = engine(patch=patch)
            replay, out = graphed(engine, patch)
            forms[f"TensorRT {precision} graph"] = replay
            outputs[f"TensorRT {precision} graph"] = out
        reference = net.terrain_code(patch).flatten(1)
        torch.cuda.synchronize()
        for name, out in outputs.items():
            print(f"[check] {name}: max rel diff vs torch code {_max_rel(out, reference):.1e}")

        best = {name: float("inf") for name in forms}
        for _ in range(args.rounds):
            for name, fn in forms.items():
                best[name] = min(best[name], _time_ms(fn, trials=1, reps=5))
                torch.cuda.empty_cache()
    print(f"[trunk] ms per call at {n} patches, fastest of {args.rounds} interleaved rounds")
    for name, ms in best.items():
        print(f"  {name:22s} {ms:7.1f} ms   ({best['WarpTrunk graph'] / ms:.2f}x WarpTrunk graph's speed)")


if __name__ == "__main__":
    main()
