"""`NCHWChannelLayerNorm`: `lattice_learning.model.ChannelLayerNorm` computed in place over dim 1,
without the NCHW -> NHWC -> NCHW permutes, so the terrain trunk runs faster on the same weights.

`ChannelLayerNorm` normalises each spatial position over its channels by permuting to channels-last
and calling `nn.LayerNorm`. At 512 patches on the GTX 1050 that LayerNorm alone was half of
`terrain_code`'s GPU time (~60 of 120 ms: ~440k rows of 32-96 numbers each), plus ~18 ms of permute
copies. The maths is unchanged here: mean and BIASED variance over the channels, eps inside the
square root, then the LayerNorm's own affine `weight`/`bias` per channel. `nchw_layer_norm(net)`
swaps it into every trunk block of a loaded net. The replacement wraps the SAME `nn.LayerNorm`
module under the same attribute name, so the state_dict keys (`blocks.<i>.norm.norm.weight`) and
the parameters themselves are shared, and a checkpoint loads into either form. Outputs agree to
float rounding, not bit for bit (`__main__` checks it).

`__main__` checks:
  1. forward at every trunk width, and on activations offset far from zero (a one-pass variance
     would lose them to cancellation);
  2. gradients w.r.t. the input and the LayerNorm's weight/bias, so a net trained with the swap is
     the same net;
  3. `terrain_code` and `net.predict` end to end, and that the state_dict is unchanged;
then times `terrain_code` before and after the swap at 512, 1536 and 3584 patches (window 0, 3 and
7 windows of a 512-rollout MPPI), and profiles the swapped trunk's GPU kernels at 512.

CLI parameters:
    --checkpoint PATH   a mppi_learning train.py checkpoint (default: a random-weight net, which is
                        exactly as fast and checks the same arithmetic)
    --batches INTS      patch counts to time (default 512 1536 3584)
    --no-profile        skip the per-kernel profile

Usage:
    python src/feasibility/nn_mppi/channel_layer_norm.py
    python src/feasibility/nn_mppi/channel_layer_norm.py --checkpoint outputs/checkpoints/<ckpt>.pt
"""
from __future__ import annotations

import argparse
import copy
import pathlib
import time
from collections.abc import Callable

import torch
from torch import nn

from feasibility.lattice_learning.model import ChannelLayerNorm
from feasibility.mppi_learning.model import random_commands
from feasibility.mppi_learning.model import WindowDivergenceNet


class NCHWChannelLayerNorm(nn.Module):
    """`ChannelLayerNorm` without the permutes: [B, C, H, W] normalised over C at every (h, w)."""

    def __init__(self, norm: nn.LayerNorm) -> None:
        super().__init__()
        if len(norm.normalized_shape) != 1 or not norm.elementwise_affine:
            raise ValueError(f"expected an affine LayerNorm over one dim, got {norm}")
        self.norm = norm  # the original module: same parameters, same state_dict key

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # correction=0: nn.LayerNorm's variance is the biased one
        var, mean = torch.var_mean(x, dim=1, keepdim=True, correction=0)
        normed = (x - mean) * torch.rsqrt(var + self.norm.eps)
        return torch.addcmul(self.norm.bias[:, None, None], normed, self.norm.weight[:, None, None])


def nchw_layer_norm(net: nn.Module) -> nn.Module:
    """Replaces every `ChannelLayerNorm` in `net` with an `NCHWChannelLayerNorm` over the same
    `nn.LayerNorm`, IN PLACE, and returns `net`."""
    swapped = 0
    for module in list(net.modules()):
        for name, child in module.named_children():
            if type(child) is ChannelLayerNorm:
                setattr(module, name, NCHWChannelLayerNorm(child.norm))
                swapped += 1
    if swapped == 0:
        raise ValueError("no ChannelLayerNorm in the net")
    return net


# --- self-test and timing ------------------------------------------------------------------------


def _load(checkpoint: pathlib.Path | None, device: torch.device) -> WindowDivergenceNet:
    if checkpoint is not None:
        from feasibility.mppi_learning.train import load_checkpoint

        return load_checkpoint(checkpoint, device)[0].eval()
    torch.manual_seed(0)
    net = WindowDivergenceNet()
    with torch.no_grad():  # a fresh LayerNorm is the identity affine; make the affine matter
        for module in net.modules():
            if isinstance(module, ChannelLayerNorm):
                module.norm.weight.uniform_(0.5, 1.5)
                module.norm.bias.uniform_(-0.3, 0.3)
    return net.to(device).eval()


def _max_rel(got: torch.Tensor, expected: torch.Tensor) -> float:
    return ((got - expected).abs().max() / expected.abs().max().clamp_min(1e-6)).item()


def self_test(net: WindowDivergenceNet, device: torch.device) -> WindowDivergenceNet:
    """Asserts the swapped net computes what `net` does; returns the swapped copy."""
    fast = nchw_layer_norm(copy.deepcopy(net))
    spec = net.patch_spec
    generator = torch.Generator(device=device).manual_seed(1)

    # 1 + 2. per module: forward, and the gradients w.r.t. the input and the affine parameters
    for original in [m for m in net.modules() if isinstance(m, ChannelLayerNorm)]:
        width = original.norm.normalized_shape[0]
        swapped = NCHWChannelLayerNorm(copy.deepcopy(original.norm))
        for offset in (0.0, 100.0):
            x = torch.randn(8, width, 7, 9, device=device, generator=generator) * 3.0 + offset
            upstream = torch.randn(8, width, 7, 9, device=device, generator=generator)
            results = []
            for module in (original, swapped):
                xi = x.clone().requires_grad_(True)
                module.zero_grad()
                y = module(xi)
                (y * upstream).sum().backward()
                results.append((y, xi.grad, module.norm.weight.grad.clone(), module.norm.bias.grad.clone()))
            errors = [_max_rel(g, e) for g, e in zip(results[1], results[0])]
            assert max(errors) < 1e-4, (width, offset, errors)
        print(f"[module] C={width}: forward / d_input / d_weight / d_bias max rel diff "
              + " / ".join(f"{e:.1e}" for e in errors) + " (offset 100)")

    # 3. end to end, on relief in the trained range, and the state_dict is untouched
    patch = torch.randn(256, 1, spec.ny, spec.nx, device=device, generator=generator) * 0.3
    command = random_commands(256, torch.Generator().manual_seed(2)).to(device)
    with torch.no_grad():
        err_code = _max_rel(fast.terrain_code(patch), net.terrain_code(patch))
        err_head = _max_rel(fast(patch, command), net(patch, command))
    assert err_code < 1e-4 and err_head < 1e-4, (err_code, err_head)
    keys, fast_keys = net.state_dict().keys(), fast.state_dict().keys()
    assert list(keys) == list(fast_keys), set(keys) ^ set(fast_keys)
    print(f"[net] terrain_code max rel diff {err_code:.1e}, net output {err_head:.1e}; "
          f"state_dict keys identical ({len(keys)})")
    return fast


def _time_ms(fn: Callable[[], object], trials: int = 5, reps: int = 5) -> float:
    """The fastest of `trials` means over `reps` calls, after 3 warm-up calls: this laptop GPU drops
    its clock in bursts (software power cap), and one burst inflates every number it overlaps ~10x."""
    for _ in range(3):
        fn()
    best = float("inf")
    for _ in range(trials):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - start) / reps * 1e3)
    return best


@torch.no_grad()
def bench(net: WindowDivergenceNet, fast: WindowDivergenceNet, batches: list[int], profile: bool) -> None:
    spec = net.patch_spec
    for n in batches:
        patch = torch.randn(n, 1, spec.ny, spec.nx, device="cuda") * 0.3
        before = _time_ms(lambda: net.terrain_code(patch), trials=1, reps=10)
        after = _time_ms(lambda: fast.terrain_code(patch), trials=1, reps=10)
        print(f"[bench] terrain_code, {n} patches: ChannelLayerNorm {before:.1f} ms, "
              f"NCHWChannelLayerNorm {after:.1f} ms ({before / after:.2f}x)")
    if not profile:
        return
    from torch.profiler import profile as torch_profile
    from torch.profiler import ProfilerActivity

    patch = torch.randn(batches[0], 1, spec.ny, spec.nx, device="cuda") * 0.3
    reps = 5
    with torch_profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            fast.terrain_code(patch)
        torch.cuda.synchronize()
    events = sorted(prof.key_averages(), key=lambda e: e.self_device_time_total, reverse=True)
    total = sum(e.self_device_time_total for e in events)
    print(f"[profile] swapped trunk at {batches[0]} patches, GPU time per call by kernel:")
    for e in events[:10]:
        print(f"  {e.self_device_time_total / reps / 1e3:7.2f} ms  {100 * e.self_device_time_total / total:4.1f}%  "
              f"{e.key[:90]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=None, help="mppi_learning checkpoint (default: random weights)")
    parser.add_argument("--batches", type=int, nargs="+", default=[512, 1536, 3584], help="patch counts to time")
    parser.add_argument("--no-profile", action="store_true", help="skip the per-kernel profile")
    args = parser.parse_args()
    device = torch.device("cuda")
    net = _load(args.checkpoint, device)
    fast = self_test(net, device)
    bench(net, fast, args.batches, not args.no_profile)
