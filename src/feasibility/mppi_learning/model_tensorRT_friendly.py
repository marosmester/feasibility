"""`TrtFriendlyWindowDivergenceNet`: `model.WindowDivergenceNet` with a trunk TensorRT can fuse --
`nn_mppi/docs/trunk_speed.md`, "Potential speed-ups", idea 3.

TensorRT's profile of the original fp16 trunk spends only ~40% in convolution: the rest is the
replicate pad before every conv (~17%), the permute-based `ChannelLayerNorm` (~12-17%) and layout
copies between the two (~20-25%), none of which it fuses. Here every trunk block is instead

    Conv2d(k=3, zero padding, no bias) -> BatchNorm2d -> SiLU

and at inference BatchNorm is a per-channel affine that folds into the conv's weights and bias, so a
block is one conv + bias + SiLU kernel. Everything after the trunk (squeeze, geometry layer, FiLM
head, `TargetTransform`, command features) is `WindowDivergenceNet`'s, unchanged; channels and
strides follow `TRUNK_PLAN`, so the trunk still ends at 6 x 9.

What this gives up, against `lattice_learning/model.py`'s reasons for its blocks:

  * zero padding says "level ground at wheel-contact height" beyond the patch edge rather than
    "the edge extended" -- for a relief patch (heights relative to the wheels) that is a physical
    statement too, but a different one, and the net sees the border;
  * the patch <-> map crop equivalence no longer holds near the border, so `encode_map` is not the
    dense form of patch mode. MPPI (`nn_mppi`) only uses patch mode;
  * BatchNorm pools statistics over (batch, H, W) in TRAINING only; at eval it is a fixed affine,
    so the trunk's receptive field at inference is still local.

`train.py --trunk trt_friendly` trains it; the checkpoint's `trunk` key makes `load_checkpoint`
rebuild this class. `nn_mppi.warp_trunk` hard-codes the original blocks and refuses this net, so in
MPPI it runs through a TensorRT trunk (`bench_full_tensorrt.TrtTrunk`).

Usage:
    python src/feasibility/mppi_learning/model_tensorRT_friendly.py   # shape / causality / BN-fold checks
"""
from __future__ import annotations

import argparse
import copy
import io

import torch
import torch.nn as nn

from feasibility.lattice_learning.model import HEAD_FUSIONS
from feasibility.lattice_learning.model import LABEL_MODES
from feasibility.lattice_learning.model import LABEL_NAMES
from feasibility.lattice_learning.model import TargetTransform
from feasibility.lattice_learning.model import TRUNK_PLAN
from feasibility.lattice_learning.patch import N_CHANNELS
from feasibility.mppi_learning.model import mirror_command
from feasibility.mppi_learning.model import random_commands
from feasibility.mppi_learning.model import window_command_features
from feasibility.mppi_learning.model import WindowDivergenceNet
from feasibility.mppi_learning.spawn_sampling import PATCH_SPEC

__all__ = ["TRUNK_STYLE", "TRUNK_STYLES", "TrtConvBlock", "TrtFriendlyWindowDivergenceNet", "fold_batchnorm"]

TRUNK_STYLE = "trt_friendly"
TRUNK_STYLES = ("original", TRUNK_STYLE)  # train.py --trunk; a checkpoint without a `trunk` key is "original"


class TrtConvBlock(nn.Module):
    """conv3 (zero padding, no bias: BatchNorm's shift replaces it) -> BatchNorm2d -> SiLU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class TrtFriendlyWindowDivergenceNet(WindowDivergenceNet):
    """`WindowDivergenceNet` with `TrtConvBlock`s in the trunk; everything else inherited."""

    def __init__(self, *, base_width: int = 32, **kwargs: object) -> None:
        super().__init__(base_width=base_width, **kwargs)
        blocks: list[nn.Module] = []
        in_channels = N_CHANNELS
        for mult, block_stride in TRUNK_PLAN:
            blocks.append(TrtConvBlock(in_channels, base_width * mult, block_stride))
            in_channels = base_width * mult
        self.blocks = nn.ModuleList(blocks)


@torch.no_grad()
def fold_batchnorm(net: TrtFriendlyWindowDivergenceNet) -> TrtFriendlyWindowDivergenceNet:
    """An eval-mode copy whose trunk blocks are conv (with bias) -> SiLU, BatchNorm folded into the
    conv -- the fusion TensorRT does, written out so the self-test can show it is exact."""
    folded = copy.deepcopy(net).eval()
    for block in folded.blocks:
        conv, bn = block.conv, block.norm
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        fused = nn.Conv2d(conv.in_channels, conv.out_channels, 3, stride=conv.stride, padding=1, bias=True)
        fused.weight.copy_(conv.weight * scale[:, None, None, None])
        fused.bias.copy_(bn.bias - bn.running_mean * scale)
        block.conv, block.norm = fused, nn.Identity()
    return folded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-width", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--head-fusion", type=str, default="film", choices=HEAD_FUSIONS)
    args = parser.parse_args()
    torch.manual_seed(0)
    spec = PATCH_SPEC
    B = 8
    patch = torch.randn(B, N_CHANNELS, spec.ny, spec.nx)
    command = random_commands(B)
    command[0, 2:] = 0.0

    for label_mode in LABEL_MODES:
        net = TrtFriendlyWindowDivergenceNet(base_width=args.base_width, embed_dim=args.embed_dim,
                                             head_fusion=args.head_fusion, label_mode=label_mode)
        assert net(patch, command).shape == (B, len(LABEL_NAMES[label_mode]))
    original = WindowDivergenceNet(base_width=args.base_width, embed_dim=args.embed_dim, head_fusion=args.head_fusion)
    n_params = sum(p.numel() for p in net.parameters())
    n_original = sum(p.numel() for p in original.parameters())
    assert all(isinstance(b, TrtConvBlock) for b in net.blocks) and len(net.blocks) == len(TRUNK_PLAN)
    assert net.geometry.kernel_size == (6, 9) and net.terrain_code(patch).shape[-2:] == (1, 1)
    assert not any("ChannelLayerNorm" in type(m).__name__ for m in net.modules())
    print(f"[forward] TrtFriendlyWindowDivergenceNet ({n_params} params, original {n_original}): "
          f"6 x TrtConvBlock, trunk 6x9, y [B, K] in both label modes")

    # the trunk never sees the command
    probe = command.clone().requires_grad_(True)
    grad = torch.autograd.grad(net.terrain_code(patch).sum(), probe, allow_unused=True)[0]
    assert grad is None or torch.all(grad == 0)
    print("[causality] terrain_code(patch) has no gradient w.r.t. the command")

    f, fm = window_command_features(command), window_command_features(mirror_command(command))
    signed = torch.tensor([False, False, True, False, True, True, False])
    assert torch.allclose(fm[:, signed], -f[:, signed]) and torch.allclose(fm[:, ~signed], f[:, ~signed])
    print("[mirror] command features inherited unchanged")

    # give BatchNorm non-trivial running stats, then check the fold TensorRT will do is exact
    net.train()
    with torch.no_grad():
        for _ in range(5):
            net(torch.randn(64, N_CHANNELS, spec.ny, spec.nx) * 0.5 + 0.1, random_commands(64))
    net.eval()
    folded = fold_batchnorm(net)
    with torch.no_grad():
        code, code_folded = net.terrain_code(patch), folded.terrain_code(patch)
    err = ((code - code_folded).abs().max() / code.abs().max()).item()
    assert err < 1e-5, err
    print(f"[bn-fold] eval trunk == conv+bias+SiLU with BatchNorm folded in, max rel diff {err:.1e}")

    net.target_transform = TargetTransform.fit(torch.rand(64, len(net.target_names)) * 2.0)
    prediction = net.predict(patch, command)
    assert (prediction >= 0).all()
    buffer = io.BytesIO()
    torch.save(net.state_dict(), buffer)
    buffer.seek(0)
    again = TrtFriendlyWindowDivergenceNet(base_width=args.base_width, embed_dim=args.embed_dim,
                                           head_fusion=args.head_fusion, label_mode=net.label_mode,
                                           target_transform=net.target_transform).eval()
    again.load_state_dict(torch.load(buffer))
    assert torch.equal(again.predict(patch, command), prediction)
    print("[predict] non-negative; state_dict round-trip (BatchNorm running stats included) exact")

    try:
        WindowDivergenceNet(base_width=args.base_width).load_state_dict(net.state_dict())
        raise AssertionError("an original net accepted a trt_friendly state_dict")
    except RuntimeError:
        pass
    print("[state_dict] the original net refuses this state_dict (no silent cross-loading)")
    print("all self-checks ok")
