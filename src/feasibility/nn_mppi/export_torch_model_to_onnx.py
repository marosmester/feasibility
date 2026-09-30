"""Exports a `WindowDivergenceNet` checkpoint to ONNX, for a TensorRT engine built on the robot.

This is the ONNX -> TensorRT alternative to `warp_trunk.py`'s hand-written trunk. TensorRT 10 needs
sm_75 or newer, so it cannot run on the GTX 1050 (sm_61). This script therefore exports and checks
on the CPU with onnxruntime, and the engine is built on the robot with `trtexec` (it prints the
command). Two graphs per checkpoint, both fp32 with a dynamic batch dimension (TensorRT picks the
precision and the shape profile at build time):

  * `<stem>.trunk.onnx`: `patch` [B, 1, 24, 36] -> `code` [B, 256], i.e. `net.terrain_code` flattened.
    This is the part `WindowCost` runs in torch (window 0) or `WarpTrunk` (windows 1+), and the part
    worth replacing. The head stays in Warp.
  * `<stem>.full.onnx`: `patch` [B, 1, 24, 36] + `command` [B, 4] -> `error` [B, K] in physical units
    (m, rad), i.e. `net.predict` with `TargetTransform`'s inverse baked in. It is for standalone
    parity checks and benchmarks.

Each file carries a `feasibility` metadata entry (JSON) holding what a consumer needs from the
checkpoint: `target_names`, the patch spec, the command columns and scales, `label_attrs`, the source
checkpoint and git provenance. A checkpoint trained with `blur_terrain` is refused, because the blur
runs outside the net (`lattice_learning.train.prepare_patch`) and the graph would silently drop it.

Checks, per file: `onnx.checker` (full check), then onnxruntime against torch at every
`--check-batches` size, on patches in the trained range and `random_commands`. The pass criterion
is a max relative difference below 1e-4. The full graph's output must also be non-negative.
`--nchw-norm` exports after `channel_layer_norm.nchw_layer_norm` (the same numbers without the
permutes), which gives TensorRT a second form of the trunk to try. It is still checked against
the original net.

CLI parameters:
    --checkpoint PATH       mppi_learning train.py checkpoint (default: a random-weight net, which
                            checks the export but not a trained net's numbers)
    --out-dir PATH          output directory (default outputs/nn_mppi/onnx)
    --opset INT             ONNX opset (default 18)
    --check-batches INTS    batch sizes for the onnxruntime check (default 1 7 1025; 1025 is the
                            trunk's rows at 512 rollouts x 3 windows)
    --nchw-norm             export the trunk with `NCHWChannelLayerNorm` (stem suffix `_nchw`)

Usage:
    python src/feasibility/nn_mppi/export_torch_model_to_onnx.py
    python src/feasibility/nn_mppi/export_torch_model_to_onnx.py --checkpoint outputs/checkpoints/<ckpt>.pt
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from feasibility.comparator.provenance import git_provenance
from feasibility.lattice_learning.model import TargetTransform
from feasibility.lattice_learning.patch import N_CHANNELS
from feasibility.lattice_learning.patch import patch_spec_to_attrs
from feasibility.mppi_learning.model import COMMAND_COLUMNS
from feasibility.mppi_learning.model import random_commands
from feasibility.mppi_learning.model import V_SCALE
from feasibility.mppi_learning.model import WindowDivergenceNet
from feasibility.mppi_learning.model import WZ_SCALE
from feasibility.mppi_learning.train import load_checkpoint
from feasibility.nn_mppi.channel_layer_norm import _load
from feasibility.nn_mppi.channel_layer_norm import _max_rel
from feasibility.nn_mppi.channel_layer_norm import nchw_layer_norm
from feasibility.nn_mppi.mppi_cost import trunk_rows

OUT = pathlib.Path("outputs/nn_mppi/onnx")
MAX_REL = 1e-4
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"  # where JetPack installs it


class TrunkGraph(nn.Module):
    """patch [B, 1, ny, nx] -> terrain code [B, geometry_channels]."""

    def __init__(self, net: WindowDivergenceNet) -> None:
        super().__init__()
        self.net = net

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        return self.net.terrain_code(patch).flatten(1)


class FullGraph(nn.Module):
    """patch [B, 1, ny, nx] + command [B, 4] -> physical error [B, K]: `net.predict`, with the
    `TargetTransform` held as buffers so the export sees it as constants."""

    def __init__(self, net: WindowDivergenceNet) -> None:
        super().__init__()
        if net.target_transform is None:
            raise ValueError("the net needs its TargetTransform to give physical errors")
        self.net = net
        self.register_buffer("mean", net.target_transform.normalizer.mean.detach().clone())
        self.register_buffer("std", net.target_transform.normalizer.std.detach().clone())

    def forward(self, patch: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        return torch.expm1((self.net(patch, command) * self.std + self.mean).clamp_min(0.0))


def load(checkpoint: pathlib.Path | None) -> tuple[WindowDivergenceNet, dict[str, object] | None]:
    """(net on the CPU in eval mode, its checkpoint dict or None for the random-weight net)."""
    cpu = torch.device("cpu")
    if checkpoint is None:
        net = _load(None, cpu)
        net.target_transform = TargetTransform.fit(
            torch.rand(64, len(net.target_names), generator=torch.Generator().manual_seed(0)) * 2.0)
        return net, None
    net, ckpt = load_checkpoint(checkpoint, cpu)
    if ckpt["blur_terrain"]:
        raise ValueError(f"{checkpoint}: trained with blur_terrain, whose blur runs outside the net")
    return net.eval(), ckpt


def metadata(net: WindowDivergenceNet, ckpt: dict[str, object] | None, checkpoint: pathlib.Path | None,
             graph: str, opset: int, nchw_norm: bool) -> str:
    """The JSON stored under the `feasibility` metadata key."""
    fields = dict(
        graph=graph,
        target_names=list(net.target_names),
        label_mode=net.label_mode,
        patch_spec=patch_spec_to_attrs(net.patch_spec),
        command_columns=list(COMMAND_COLUMNS),
        v_scale=V_SCALE,
        wz_scale=WZ_SCALE,
        label_attrs=ckpt["label_attrs"] if ckpt is not None else None,
        checkpoint=str(checkpoint) if checkpoint is not None else None,
        nchw_norm=nchw_norm,
        opset=opset,
        torch_version=torch.__version__,
        git=git_provenance(),
    )
    return json.dumps(fields, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))


def export(module: nn.Module, args: tuple[torch.Tensor, ...], names: tuple[str, ...], output: str,
           path: pathlib.Path, opset: int, meta: str) -> None:
    """Exports `module` with a dynamic batch shared by every input, then stores `meta`."""
    batch = torch.export.Dim("batch", min=1)
    torch.onnx.export(
        module, args, str(path), dynamo=True, opset_version=opset, input_names=list(names),
        output_names=[output], dynamic_shapes={name: {0: batch} for name in names},
    )
    model = onnx.load(str(path))
    onnx.helper.set_model_props(model, {"feasibility": meta})
    onnx.save(model, str(path))
    onnx.checker.check_model(str(path), full_check=True)


def check(path: pathlib.Path, reference: nn.Module, names: tuple[str, ...], batches: list[int],
          ny: int, nx: int) -> None:
    """onnxruntime (CPU) against `reference` at every batch size; raises past MAX_REL."""
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [i.name for i in session.get_inputs()] == list(names), session.get_inputs()
    generator = torch.Generator().manual_seed(1)
    for n in batches:
        inputs = {"patch": torch.randn(n, N_CHANNELS, ny, nx, generator=generator) * 0.3,
                  "command": random_commands(n, generator)}
        inputs = {name: inputs[name] for name in names}
        with torch.no_grad():
            expected = reference(*inputs.values())
        got = torch.from_numpy(session.run(None, {k: v.numpy() for k, v in inputs.items()})[0])
        assert got.shape == expected.shape, (got.shape, expected.shape)
        err = _max_rel(got, expected)
        assert err < MAX_REL, (path.name, n, err)
        if "command" in names:
            assert (got >= 0).all(), f"{path.name}: negative physical error at batch {n}"
        print(f"[check] {path.name} batch {n}: max rel diff vs torch {err:.1e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=None, help="mppi_learning checkpoint (default: random weights)")
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT, help="output directory")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset")
    parser.add_argument("--check-batches", type=int, nargs="+", default=[1, 7, trunk_rows(512, 3)],
                        help="batch sizes for the onnxruntime check")
    parser.add_argument("--nchw-norm", action="store_true", help="export with NCHWChannelLayerNorm")
    args = parser.parse_args()

    net, ckpt = load(args.checkpoint)
    exported = nchw_layer_norm(copy.deepcopy(net)) if args.nchw_norm else net
    spec = net.patch_spec
    stem = (args.checkpoint.stem if args.checkpoint is not None else "random") + ("_nchw" if args.nchw_norm else "")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    patch = torch.randn(2, N_CHANNELS, spec.ny, spec.nx) * 0.3
    command = random_commands(2, torch.Generator().manual_seed(0))

    graphs = (
        ("trunk", TrunkGraph(exported), TrunkGraph(net), (patch,), ("patch",), "code"),
        ("full", FullGraph(exported), FullGraph(net), (patch, command), ("patch", "command"), "error"),
    )
    paths = {}
    for graph, module, reference, example, names, output in graphs:
        path = args.out_dir / f"{stem}.{graph}.onnx"
        meta = metadata(net, ckpt, args.checkpoint, graph, args.opset, args.nchw_norm)
        export(module.eval(), example, names, output, path, args.opset, meta)
        check(path, reference.eval(), names, args.check_batches, spec.ny, spec.nx)
        paths[graph] = path
        print(f"[export] {path} ({path.stat().st_size / 1e6:.2f} MB), inputs {names} -> {output}")

    shape = f"x{N_CHANNELS}x{spec.ny}x{spec.nx}"
    trunk = paths["trunk"]
    print("\nOn the robot (add --fp16 to time half precision, which the Orin has and the 1050 lacks):\n"
          f"  {TRTEXEC} --onnx={trunk.name} --saveEngine={trunk.with_suffix('.engine').name} "
          f"--minShapes=patch:1{shape} --optShapes=patch:{trunk_rows(512, 3)}{shape} "
          f"--maxShapes=patch:{trunk_rows(4096, 3)}{shape}")


if __name__ == "__main__":
    main()
