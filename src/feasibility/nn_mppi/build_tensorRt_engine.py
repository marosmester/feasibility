"""Builds a TensorRT engine from an `export_torch_model_to_onnx.py` graph, checks it against torch, and
times it. Run it on the machine that will use the engine: a plan is specific to the GPU and the
TensorRT version that built it.

One engine per ONNX file and precision, written next to the ONNX file as
`<onnx stem>.<precision>.engine`, with a `.json` sidecar: the ONNX file's `feasibility` metadata
(`target_names`, patch spec, command scales, `label_attrs`, source checkpoint) plus the build
(precision, batch profile, TensorRT version, GPU). The profile covers batches `--min-batch` to
`--max-batch`, tuned for `--opt-batch`; the defaults are the trunk's rows per refine at 512 rollouts
x 3 windows (opt) and at 4096 x 3 (max). The inputs and outputs stay fp32 at every precision.
Only the arithmetic inside changes:

  * `fp32`: TF32 off, so the engine computes what torch does to float rounding;
  * `tf32`: TensorRT's default on Ampere: fp32 storage, 10-bit-mantissa matmul/conv inputs;
  * `fp16`: half-precision kernels wherever TensorRT finds them faster.

Checks: the engine against the source checkpoint's net in torch on the same GPU, at every
`--check-batches` size, on patches in the trained range and `random_commands`. For a trunk engine
the check covers the code, and also that code pushed through the torch head to physical errors,
the number the MPPI cost uses. For a full engine it covers the physical errors directly. Each is
the max relative difference against the largest reference value, and it must stay under the
precision's tolerance (`TOLERANCE`). Then it times the engine against torch eager at
`--opt-batch`: median GPU ms over `--reps` runs, with CUDA events.

The checkpoint comes from the ONNX metadata (a path relative to the repo root, so run from there),
or `--checkpoint`. An ONNX file exported from random weights has none, so it is only built and timed.

Environment (the robot): TensorRT comes from JetPack, not pip. The repo `.venv` sees it through
`.venv/lib/python3.12/site-packages/zz_system_tensorrt.pth`, which adds `.venv/system-tensorrt/`
(a symlink to `/usr/lib/python3.12/dist-packages/tensorrt`) to `sys.path`. `onnx` reads the
metadata.

CLI parameters:
    onnx PATH ...           ONNX files from export_torch_model_to_onnx.py
    --precision P ...       fp32, tf32 and/or fp16, one engine each (default fp32 fp16)
    --min-batch INT         profile minimum (default 1)
    --opt-batch INT         profile optimum, also the timed batch (default 1025)
    --max-batch INT         profile maximum (default 8193)
    --checkpoint PATH       reference checkpoint (default: the one in the ONNX metadata)
    --check-batches INTS    batch sizes for the torch check (default 1 7 1025 8193)
    --reps INT              timed runs per arm (default 50)
    --workspace-mb INT      builder workspace limit (default 8192). TensorRT's own default on the
                            AGX Orin left ~1.8 GB. That is too little for the fp16 build at batch
                            8193, whose fused LayerNorm + SiLU asks for ~3.5 GB and has no fallback,
                            so the build failed.

Usage:
    python src/feasibility/nn_mppi/build_tensorRt_engine.py outputs/nn_mppi/onnx/<stem>.trunk.onnx
    python src/feasibility/nn_mppi/build_tensorRt_engine.py outputs/nn_mppi/onnx/<stem>.*.onnx --precision fp32 tf32 fp16
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
from collections.abc import Callable

import onnx
import tensorrt as trt
import torch

from feasibility.lattice_learning.patch import PatchSpec
from feasibility.lattice_learning.patch import patch_spec_from_attrs
from feasibility.mppi_learning.model import random_commands
from feasibility.mppi_learning.model import WindowDivergenceNet
from feasibility.mppi_learning.train import load_checkpoint
from feasibility.nn_mppi.mppi_cost import trunk_rows

PRECISIONS = ("fp32", "tf32", "fp16")
TOLERANCE = {"fp32": 1e-4, "tf32": 1e-2, "fp16": 3e-2}  # max rel diff vs torch fp32
LOGGER = trt.Logger(trt.Logger.WARNING)


def onnx_metadata(path: pathlib.Path) -> dict[str, object]:
    """The `feasibility` JSON that export_torch_model_to_onnx.py stores in the ONNX file."""
    props = {p.key: p.value for p in onnx.load(str(path), load_external_data=False).metadata_props}
    if "feasibility" not in props:
        raise ValueError(f"{path}: no `feasibility` metadata, not written by export_torch_model_to_onnx.py")
    return json.loads(props["feasibility"])


def build(path: pathlib.Path, precision: str, batches: tuple[int, int, int], workspace_mb: int | None) -> bytes:
    """ONNX file -> serialized engine, one optimization profile over the batch dim of every input."""
    builder = trt.Builder(LOGGER)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, LOGGER)
    if not parser.parse_from_file(str(path)):
        raise RuntimeError(f"{path}: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    if workspace_mb is not None:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    if precision == "fp32":
        config.clear_flag(trt.BuilderFlag.TF32)
    elif precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        rest = tuple(tensor.shape)[1:]
        profile.set_shape(tensor.name, *[(b, *rest) for b in batches])
    config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"{path}: TensorRT failed to build a {precision} engine")
    return bytes(plan)


class Engine:
    """A deserialized engine run on torch CUDA tensors, on `stream`."""

    def __init__(self, plan: bytes, stream: torch.cuda.Stream) -> None:
        self.engine = trt.Runtime(LOGGER).deserialize_cuda_engine(plan)
        self.context = self.engine.create_execution_context()
        self.stream = stream
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        assert len(self.outputs) == 1, self.outputs

    def __call__(self, **inputs: torch.Tensor) -> torch.Tensor:
        for name in self.inputs:
            tensor = inputs[name].contiguous()
            self.context.set_input_shape(name, tuple(tensor.shape))
            self.context.set_tensor_address(name, tensor.data_ptr())
        out = torch.empty(tuple(self.context.get_tensor_shape(self.outputs[0])), device="cuda")
        self.context.set_tensor_address(self.outputs[0], out.data_ptr())
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        return out


def _max_rel(got: torch.Tensor, expected: torch.Tensor) -> float:
    return ((got - expected).abs().max() / expected.abs().max().clamp_min(1e-6)).item()


def _inputs(spec: PatchSpec, n: int, generator: torch.Generator) -> dict[str, torch.Tensor]:
    return {"patch": (torch.randn(n, 1, spec.ny, spec.nx, generator=generator) * 0.3).cuda(),
            "command": random_commands(n, generator).cuda()}


@torch.no_grad()
def check(engine: Engine, net: WindowDivergenceNet, graph: str, precision: str, batches: list[int]) -> None:
    """Engine vs torch at every batch size; raises past TOLERANCE[precision]."""
    generator = torch.Generator().manual_seed(1)
    for n in batches:
        x = _inputs(net.patch_spec, n, generator)
        with torch.cuda.stream(engine.stream):
            got = engine(**{k: x[k] for k in engine.inputs})
            if graph == "trunk":
                code = net.terrain_code(x["patch"]).flatten(1)
                errors = {"code": _max_rel(got, code),
                          "error": _max_rel(net.target_transform.inverse(net._head(got, x["command"])),
                                            net.predict(x["patch"], x["command"]))}
            else:
                errors = {"error": _max_rel(got, net.predict(x["patch"], x["command"]))}
        engine.stream.synchronize()
        worst = max(errors.values())
        assert worst < TOLERANCE[precision], (graph, precision, n, errors)
        print(f"[check] {graph} {precision} batch {n}: max rel diff vs torch "
              + ", ".join(f"{k} {v:.1e}" for k, v in errors.items()))


def _time_ms(fn: Callable[[], object], stream: torch.cuda.Stream, reps: int) -> float:
    """Median GPU ms of `fn` on `stream`, after a few untimed runs."""
    times = []
    with torch.cuda.stream(stream):
        for i in range(reps + 5):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(stream)
            fn()
            end.record(stream)
            end.synchronize()
            if i >= 5:
                times.append(start.elapsed_time(end))
    return sorted(times)[len(times) // 2]


@torch.no_grad()
def bench(engine: Engine, net: WindowDivergenceNet | None, spec: PatchSpec, graph: str, precision: str, n: int,
          reps: int) -> None:
    """Times the engine, and torch eager fp32 beside it when there is a net."""
    x = _inputs(spec, n, torch.Generator().manual_seed(2))
    feed = {k: x[k] for k in engine.inputs}
    line = f"[bench] {graph} {precision} batch {n}: TensorRT {_time_ms(lambda: engine(**feed), engine.stream, reps):.2f} ms"
    if net is not None:
        torch_fn = (lambda: net.terrain_code(x["patch"])) if graph == "trunk" else (lambda: net.predict(x["patch"], x["command"]))
        line += f", torch eager fp32 {_time_ms(torch_fn, engine.stream, reps):.2f} ms"
    print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("onnx", type=pathlib.Path, nargs="+", help="ONNX files from export_torch_model_to_onnx.py")
    parser.add_argument("--precision", nargs="+", default=["fp32", "fp16"], choices=PRECISIONS)
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=trunk_rows(512, 3))
    parser.add_argument("--max-batch", type=int, default=trunk_rows(4096, 3))
    parser.add_argument("--checkpoint", type=pathlib.Path, default=None, help="reference checkpoint (default: from the ONNX metadata)")
    parser.add_argument("--check-batches", type=int, nargs="+", default=None, help="default: 1 7 opt max")
    parser.add_argument("--reps", type=int, default=50, help="timed runs per arm")
    parser.add_argument("--workspace-mb", type=int, default=8192, help="builder workspace limit")
    args = parser.parse_args()
    batches = (args.min_batch, args.opt_batch, args.max_batch)
    if not args.min_batch <= args.opt_batch <= args.max_batch:
        raise ValueError(f"need min <= opt <= max, got {batches}")
    check_batches = args.check_batches or sorted({1, 7, args.opt_batch, args.max_batch})
    torch.backends.cuda.matmul.allow_tf32 = False  # the reference is true fp32
    torch.backends.cudnn.allow_tf32 = False
    device = torch.cuda.get_device_properties(0)
    print(f"TensorRT {trt.__version__}, torch {torch.__version__}, {device.name} (sm_{device.major}{device.minor})")
    stream = torch.cuda.Stream()

    for path in args.onnx:
        meta = onnx_metadata(path)
        graph = meta["graph"]
        spec = patch_spec_from_attrs(meta["patch_spec"])
        checkpoint = args.checkpoint or (pathlib.Path(meta["checkpoint"]) if meta["checkpoint"] else None)
        net = load_checkpoint(checkpoint, torch.device("cuda"))[0].eval() if checkpoint is not None else None
        if net is not None and list(net.target_names) != meta["target_names"]:
            raise ValueError(f"{checkpoint} predicts {net.target_names}, {path} {meta['target_names']}")
        for precision in args.precision:
            start = time.perf_counter()
            plan = build(path, precision, batches, args.workspace_mb)
            seconds = time.perf_counter() - start
            engine_path = path.with_name(f"{path.name.removesuffix('.onnx')}.{precision}.engine")
            engine_path.write_bytes(plan)
            sidecar = dict(meta, onnx=str(path), precision=precision, batch_min_opt_max=list(batches),
                           tensorrt_version=trt.__version__, gpu=device.name,
                           compute_capability=f"{device.major}.{device.minor}", build_seconds=round(seconds, 1))
            engine_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
            print(f"[build] {engine_path} ({len(plan) / 1e6:.2f} MB, {seconds:.0f} s)")
            engine = Engine(plan, stream)
            if net is not None:
                check(engine, net, graph, precision, check_batches)
            else:
                print(f"[check] skipped: {path} has no checkpoint in its metadata, pass --checkpoint")
            bench(engine, net, spec, graph, precision, args.opt_batch, args.reps)
            del engine  # its context holds activations for the max batch; free them before the next build
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
