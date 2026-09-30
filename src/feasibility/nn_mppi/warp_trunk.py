"""The terrain trunk (`ArcDivergenceNet.terrain_code`: 6 conv blocks, squeeze, geometry) in Warp, in
two variants, timed against torch. This is the cost of charging MPPI windows beyond window 0, where
every rollout needs its own terrain code in every refine (mppi_learning/design.md section 9e).

`terrain_code` in torch spends most of its GPU time outside the convolutions. With
`channel_layer_norm.NCHWChannelLayerNorm` it still takes 73 ms at 512 patches on the GTX 1050, and
only about 20 ms of that is cuDNN's conv kernels. The rest goes to the replicate padding (a separate
copy before every conv), the norm's elementwise kernels and SiLU, each one a full pass over the
activations.

  * `HybridTrunk` keeps cuDNN's convolutions and replaces everything between them with one Warp
    kernel per block: LayerNorm over the channels, SiLU, and a write into the NEXT conv's input
    buffer with its replicate border already filled, so that conv runs with padding 0. torch and
    Warp share torch's stream, so it cannot be captured into a Warp graph.
  * `WarpTrunk` is all Warp, so it CAN be captured, which is what a hook inside `MppiGpu`'s refine
    graph needs. Activations are NHWC vec4 (`[B, H, W, C / 4]`). The convolution is direct and
    register-blocked like `mppi_cost._dense_kernel`: one thread computes 9 consecutive output
    pixels x 4 or 8 output channels, and replicate padding is a clamp on the input index. The
    LayerNorm + SiLU runs in place, one thread per pixel with its channels in registers. The
    squeeze and the geometry layer are dense layers through `mppi_cost._dense_kernel` itself; the
    geometry layer is one at patch size, since its kernel spans the squeeze output exactly.

Two changes made the Warp conv competitive, and both apply to any Warp kernel reading vec4:
  * one 128-bit `__ldg` per vec4 through `wp.func_native`. Warp's vec4 is 4-byte aligned, so
    `x[i]` compiles to four scalar loads. This took the trunk from 122 to 61 ms at 512 patches.
  * flat indices computed outside the inner loop, because a 4-D index costs four integer
    multiplies per load (emulated on Pascal, on the FMA cores). 61 -> 50 ms.
Measured at 512 patches: 47 ms for both Warp and the hybrid, against 73 for torch with
`NCHWChannelLayerNorm` and 119 for torch. A captured graph is no faster than eager launches, since
the trunk's ~15 launches cost nothing next to the kernels. At 3584 patches cuDNN gets relatively
faster and the hybrid wins, 298 ms against 338, while the Warp conv scales linearly.

Both take the PREPARED patch (`prepare_patch`'s output), as `terrain_code` does, and both use a
two-pass variance over values held in registers, so they agree with torch to float rounding.

`__main__` checks both against the original `net.terrain_code` on random relief, on patches sampled
from a box and a rough map, and on a batch that is not a multiple of 8. It then times each block's
conv and norm, Warp against torch, at the first batch size, and the whole trunk at every batch size
in five forms: torch, torch with `NCHWChannelLayerNorm`, the hybrid, Warp launched eagerly, and
Warp replayed as a captured CUDA graph.

CLI parameters:
    --checkpoint PATH   a mppi_learning train.py checkpoint (default: a random-weight net, which is
                        exactly as fast and checks the same arithmetic)
    --batches INTS      patch counts to time (default 512 1536 3584: window 0, 3 and 7 windows of a
                        512-rollout MPPI)

Usage:
    python src/feasibility/nn_mppi/warp_trunk.py
    python src/feasibility/nn_mppi/warp_trunk.py --checkpoint outputs/checkpoints/<ckpt>.pt
"""
from __future__ import annotations

import argparse
import copy
import pathlib
import time
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
import warp as wp

from feasibility.lattice_learning.model import ArcDivergenceNet
from feasibility.nn_mppi.channel_layer_norm import _load
from feasibility.nn_mppi.channel_layer_norm import nchw_layer_norm
from feasibility.nn_mppi.mppi_cost import _dense_kernel
from feasibility.nn_mppi.mppi_cost import _silu
from feasibility.nn_mppi.mppi_cost import _silu4
from feasibility.nn_mppi.mppi_cost import ROWS_PER_THREAD

# The norm kernels hold one pixel's channels in registers, which needs their per-channel loops
# unrolled (96 channels in the hybrid, 24 quads in WarpTrunk); Warp's default limit is 16.
wp.set_module_options({"max_unroll": 96})

PIXELS = wp.constant(9)  # output pixels per conv thread; divides every trunk width (36, 18, 9)
_pixel_index = wp.types.vector(length=9, dtype=wp.int32)

_KERNELS: dict[str, wp.Kernel] = {}


# Warp's vec4 is only 4-byte aligned, so `x[b, i, j, k]` compiles to four scalar loads. With four
# scalar loads per vec4 the conv kernel was bound by load issue (52 loads per 144 FMAs, 12% of FMA
# peak). These do ONE 128-bit load or store, `__ldg` going through the read-only (texture/L1)
# cache. Buffers here come from Warp/torch allocations, so every element is 16-byte aligned.
@wp.func_native("""
    const float4 v = __ldg(reinterpret_cast<const float4*>(wp::address(x, b, i, j, k)));
    return wp::vec_t<4, wp::float32>(v.x, v.y, v.z, v.w);
""")
def _load4(x: wp.array4d(dtype=wp.vec4), b: int, i: int, j: int, k: int) -> wp.vec4: ...


@wp.func_native("""
    *reinterpret_cast<float4*>(wp::address(y, b, i, j, k)) = make_float4(v[0], v[1], v[2], v[3]);
""")
def _store4(y: wp.array4d(dtype=wp.vec4), b: int, i: int, j: int, k: int, v: wp.vec4): ...


# The same on a flat array at a precomputed index: a 4-D index costs four integer multiplies per
# load, and Pascal runs integer multiplies (emulated) on the same cores as the FMAs.
@wp.func_native("""
    const float4 v = __ldg(reinterpret_cast<const float4*>(x.data) + i);
    return wp::vec_t<4, wp::float32>(v.x, v.y, v.z, v.w);
""")
def _ldg4(x: wp.array(dtype=wp.vec4), i: int) -> wp.vec4: ...


@wp.func_native("""
    reinterpret_cast<float4*>(y.data)[i] = make_float4(v[0], v[1], v[2], v[3]);
""")
def _stg4(y: wp.array(dtype=wp.vec4), i: int, v: wp.vec4): ...


def _conv_kernel(stride: int, quads_per_thread: int) -> wp.Kernel:
    """conv3x3 with replicate padding at `stride` on flat NHWC vec4 buffers, [B, H, W, Cin/4] ->
    [B, Ho, Wo, Cout/4]. One thread computes PIXELS consecutive output pixels x `quads_per_thread`
    x 4 output channels. An input load then serves 4 * quads_per_thread channels and a weight load
    serves PIXELS pixels. Launch (B, Ho, Wo / PIXELS, Cout / 4 / quads_per_thread). Output channels
    vary fastest across a warp, so its lanes read consecutive weights and share their input loads.

    Measured at 512 patches on the GTX 1050: 11 ms for block 1 (32 -> 32 at 24 x 36, ~40% of FMA
    peak), 10 ms for block 3, where cuDNN takes 14. Reusing each input column across the three kx
    taps (3x fewer input loads at stride 1) measured no faster, so loads are no longer the limit."""
    key = f"trunk_conv_s{stride}_q{quads_per_thread}"
    if key in _KERNELS:
        return _KERNELS[key]
    step = wp.constant(stride)
    qpt = wp.constant(quads_per_thread)
    accumulator_t = wp.types.matrix(shape=(9, 4 * quads_per_thread), dtype=float)

    def conv(
        x: wp.array(dtype=wp.vec4),
        weight: wp.array(dtype=wp.vec4),  # [9 * Cin, Cout / 4] flat, rows in (ky, kx, cin) order
        bias: wp.array(dtype=wp.vec4),  # [Cout / 4]
        height: int,
        width: int,
        in_quads: int,
        out_height: int,
        out_width: int,
        out_quads: int,
        y: wp.array(dtype=wp.vec4),
    ):
        b, oy, g, t = wp.tid()
        q0 = t * qpt
        ox0 = g * PIXELS
        acc = accumulator_t()
        for ky in range(3):
            row_base = (b * height + wp.clamp(oy * step + ky - 1, 0, height - 1)) * width
            for kx in range(3):
                base = _pixel_index()
                for i in range(PIXELS):
                    base[i] = (row_base + wp.clamp((ox0 + i) * step + kx - 1, 0, width - 1)) * in_quads
                w_index = (ky * 3 + kx) * in_quads * 4 * out_quads + q0
                for kq in range(in_quads):
                    xs = wp.matrix(shape=(PIXELS, 4), dtype=float)
                    for i in range(PIXELS):
                        xs[i] = _ldg4(x, base[i] + kq)
                    for s in range(4):
                        for qq in range(qpt):
                            w = _ldg4(weight, w_index + s * out_quads + qq)
                            for i in range(PIXELS):
                                xv = xs[i, s]
                                acc[i, 4 * qq + 0] += xv * w[0]
                                acc[i, 4 * qq + 1] += xv * w[1]
                                acc[i, 4 * qq + 2] += xv * w[2]
                                acc[i, 4 * qq + 3] += xv * w[3]
                    w_index += 4 * out_quads
        out_base = ((b * out_height + oy) * out_width + ox0) * out_quads + q0
        for qq in range(qpt):
            bq = bias[q0 + qq]
            for i in range(PIXELS):
                out = wp.vec4(acc[i, 4 * qq + 0], acc[i, 4 * qq + 1], acc[i, 4 * qq + 2], acc[i, 4 * qq + 3])
                _stg4(y, out_base + i * out_quads + qq, out + bq)

    _KERNELS[key] = wp.Kernel(conv, key=key)
    return _KERNELS[key]


def _nhwc_norm_kernel(quads: int) -> wp.Kernel:
    """ChannelLayerNorm + SiLU IN PLACE on NHWC vec4 [B, H, W, quads], one thread per pixel. Launch
    (B, H, W). A tile's and a register array's size are compile-time, so it is built per width."""
    key = f"trunk_nhwc_norm_{quads}"
    if key in _KERNELS:
        return _KERNELS[key]
    n = wp.constant(quads)
    values_t = wp.types.matrix(shape=(quads, 4), dtype=float)

    def norm(
        x: wp.array4d(dtype=wp.vec4),
        weight: wp.array(dtype=wp.vec4),  # LayerNorm affine, [quads]
        bias: wp.array(dtype=wp.vec4),
        eps: float,
    ):
        b, i, j = wp.tid()
        values = values_t()
        total = wp.vec4()
        for k in range(n):
            v = _load4(x, b, i, j, k)  # no other thread touches this pixel, so __ldg is safe in place
            values[k] = v
            total += v
        mean = (total[0] + total[1] + total[2] + total[3]) / float(4 * n)
        centre = wp.vec4(mean, mean, mean, mean)
        squares = float(0.0)
        for k in range(n):
            d = values[k] - centre
            squares += wp.dot(d, d)
        inv = 1.0 / wp.sqrt(squares / float(4 * n) + eps)  # biased variance, as nn.LayerNorm
        for k in range(n):
            _store4(x, b, i, j, k, _silu4(wp.cw_mul((values[k] - centre) * inv, weight[k]) + bias[k]))

    _KERNELS[key] = wp.Kernel(norm, key=key)
    return _KERNELS[key]


def _nchw_norm_pad_kernel(channels: int) -> wp.Kernel:
    """ChannelLayerNorm + SiLU on torch's NCHW conv output [B, C, H, W], written into
    [B, C, H + 2 pad, W + 2 pad] with a replicate border: one thread per OUTPUT pixel, reading the
    interior pixel it clamps to, so the border is recomputed rather than raced over. Consecutive
    threads are consecutive pixels, so every per-channel load is coalesced. Launch (B, Ho, Wo)."""
    key = f"trunk_nchw_norm_pad_{channels}"
    if key in _KERNELS:
        return _KERNELS[key]
    n = wp.constant(channels)
    values_t = wp.types.vector(length=channels, dtype=float)

    def norm_pad(
        x: wp.array4d(dtype=float),
        weight: wp.array(dtype=float),
        bias: wp.array(dtype=float),
        eps: float,
        pad: int,
        y: wp.array4d(dtype=float),
    ):
        b, py, px = wp.tid()
        sy = wp.clamp(py - pad, 0, x.shape[2] - 1)
        sx = wp.clamp(px - pad, 0, x.shape[3] - 1)
        values = values_t()
        total = float(0.0)
        for c in range(n):
            v = x[b, c, sy, sx]
            values[c] = v
            total += v
        mean = total / float(n)
        squares = float(0.0)
        for c in range(n):
            d = values[c] - mean
            squares += d * d
        inv = 1.0 / wp.sqrt(squares / float(n) + eps)
        for c in range(n):
            y[b, c, py, px] = _silu((values[c] - mean) * inv * weight[c] + bias[c])

    _KERNELS[key] = wp.Kernel(norm_pad, key=key)
    return _KERNELS[key]


@wp.kernel
def _pack_kernel(patch: wp.array3d(dtype=float), x: wp.array4d(dtype=wp.vec4)):
    # [B, ny, nx] relief -> NHWC vec4 with one real channel; the conv's weights are zero for the rest
    b, i, j = wp.tid()
    x[b, i, j, 0] = wp.vec4(patch[b, i, j], 0.0, 0.0, 0.0)


def _quads(t: torch.Tensor, device: wp.context.Device) -> wp.array:
    """[..., n] -> vec4 [..., n / 4] on `device`."""
    a = t.detach().float().cpu().numpy()
    return wp.array(np.ascontiguousarray(a.reshape(*a.shape[:-1], -1, 4)), dtype=wp.vec4, device=device)


def _check_trunk(net: ArcDivergenceNet) -> list[tuple[int, int, int, int, int]]:
    """Per block (Cin, Cout, stride, Ho, Wo) at patch size; raises on anything these kernels do not
    implement."""
    spec = net.patch_spec
    height, width, shapes = spec.ny, spec.nx, []
    for block in net.blocks:
        conv = block.conv
        if conv.kernel_size != (3, 3) or conv.padding != (1, 1) or conv.padding_mode != "replicate":
            raise ValueError(f"only replicate-padded 3x3 convs are implemented, got {conv}")
        stride = conv.stride[0]
        height, width = (height - 1) // stride + 1, (width - 1) // stride + 1
        shapes.append((conv.in_channels, conv.out_channels, stride, height, width))
    if net.geometry.kernel_size != (height, width):
        raise ValueError(f"geometry kernel {net.geometry.kernel_size} != trunk output {(height, width)}")
    if any(cout % 4 or wo % PIXELS for _, cout, _, _, wo in shapes) or net.squeeze.out_channels % 4:
        raise ValueError(f"widths must be multiples of 4 and output widths of {PIXELS}: {shapes}")
    return shapes


class WarpTrunk:
    """`net.terrain_code` in Warp for up to `max_batch` patches, all buffers preallocated, so a
    call is launches only and can be captured into a CUDA graph."""

    def __init__(self, net: ArcDivergenceNet, max_batch: int, device: str = "cuda:0") -> None:
        self.device = wp.get_device(device)
        self.shapes = _check_trunk(net)
        spec = net.patch_spec
        self.ny, self.nx = spec.ny, spec.nx
        # the dense layers take rows in groups of ROWS_PER_THREAD; batch rows past a call's B are
        # computed on stale buffers and never read
        self.max_batch = -(-max_batch // ROWS_PER_THREAD) * ROWS_PER_THREAD
        batch = self.max_batch
        largest = max([spec.ny * spec.nx * 4] + [ho * wo * cout for _, cout, _, ho, wo in self.shapes])
        with wp.ScopedDevice(self.device):
            # two ping-pong buffers big enough for any block; each layer gets a view of the right shape
            self._buffers = [wp.zeros(batch * largest // 4, dtype=wp.vec4) for _ in range(2)]
            self.steps: list[tuple[str, Callable[[int], None]]] = []
            source = self._view(1, (batch, spec.ny, spec.nx, 1))
            self._packed = source
            self.steps.append(("pack", self._launcher(_pack_kernel, None, lambda b: (b, self.ny, self.nx))))
            height, width, in_quads = spec.ny, spec.nx, 1
            for k, (block, (cin, cout, stride, ho, wo)) in enumerate(zip(net.blocks, self.shapes)):
                weight = block.conv.weight.detach().permute(2, 3, 1, 0)  # [3, 3, Cin, Cout]
                weight = F.pad(weight, (0, 0, 0, -cin % 4)).reshape(-1)  # Cin padded to quads
                target = self._view(k % 2, (batch, ho, wo, cout // 4))
                # two output quads per thread pay off from 64 channels (blocks 2, 4, 5: 6.4 / 4.6 /
                # 6.0 ms against 8.4 / 6.6 / 6.4); at 32 they halve the threads for less reuse and lose
                qpt = 2 if cout >= 64 and (cout // 4) % 2 == 0 else 1
                kernel = _conv_kernel(stride, qpt)
                conv_inputs = [self._buffers[(k + 1) % 2], _quads(weight, self.device), _quads(block.conv.bias, self.device),
                               height, width, in_quads, ho, wo, cout // 4, self._buffers[k % 2]]
                self.steps.append((f"block {k} conv", self._launcher(
                    kernel, conv_inputs,
                    lambda b, ho=ho, wo=wo, t=cout // 4 // qpt: (b, ho, wo // PIXELS, t))))
                height, width, in_quads = ho, wo, cout // 4
                layer_norm = block.norm.norm
                norm_inputs = [target, _quads(layer_norm.weight, self.device), _quads(layer_norm.bias, self.device),
                               float(layer_norm.eps)]
                self.steps.append((f"block {k} norm", self._launcher(
                    _nhwc_norm_kernel(cout // 4), norm_inputs, lambda b, ho=ho, wo=wo: (b, ho, wo))))
                source = target
            _, cout, _, ho, wo = self.shapes[-1]
            squeeze_out = net.squeeze.out_channels
            rows = ho * wo
            squeeze_in = self._view((len(self.shapes) - 1) % 2, (batch * rows, cout // 4))
            assert squeeze_in.ptr == source.ptr
            self._squeezed = wp.zeros((batch * rows, squeeze_out // 4), dtype=wp.vec4)
            squeeze_weight = net.squeeze.weight.detach().reshape(squeeze_out, cout).T
            self._dense_step("squeeze", squeeze_in, squeeze_weight, net.squeeze.bias, self._squeezed, rows)
            # the squeeze output, row-major over (b, h, w, c), IS the NHWC flattening per patch
            geometry_in = wp.array(ptr=self._squeezed.ptr, dtype=wp.vec4, shape=(batch, rows * squeeze_out // 4),
                                   device=self.device)
            geometry_weight = net.geometry.weight.detach().permute(2, 3, 1, 0).reshape(-1, net.geometry.out_channels)
            self.code = wp.zeros((batch, net.geometry.out_channels // 4), dtype=wp.vec4)
            self._dense_step("geometry", geometry_in, geometry_weight, net.geometry.bias, self.code, 1)
            self._keep = [squeeze_in, geometry_in]
        self._graphs: dict[int, object] = {}

    def _view(self, which: int, shape: tuple[int, ...]) -> wp.array:
        base = self._buffers[which]
        assert np.prod(shape) <= base.shape[0], (shape, base.shape)
        return wp.array(ptr=base.ptr, dtype=wp.vec4, shape=shape, device=self.device)

    def _launcher(self, kernel: wp.Kernel, inputs: list | None, dim: Callable[[int], tuple[int, ...]]) -> Callable[[int], None]:
        def launch(b: int) -> None:
            args = inputs if inputs is not None else [self._patch, self._packed]
            wp.launch(kernel, dim(b), inputs=args, device=self.device)
        return launch

    def _dense_step(self, name: str, x: wp.array, weight: torch.Tensor, bias: torch.Tensor, y: wp.array,
                    rows_per_patch: int) -> None:
        inputs = [x, _quads(weight, self.device), _quads(bias, self.device), 0, y]

        def launch(b: int) -> None:
            groups = -(-b // ROWS_PER_THREAD) * rows_per_patch  # b rounded up to whole row groups
            wp.launch(_dense_kernel, (groups, y.shape[1]), inputs=inputs, device=self.device)

        self.steps.append((name, launch))

    def __call__(self, patch: wp.array) -> wp.array:
        """[B, ny, nx] float relief -> [B, geometry_channels / 4] vec4 terrain codes (a view)."""
        b = patch.shape[0]
        if b > self.max_batch or patch.shape[1:] != (self.ny, self.nx):
            raise ValueError(f"patch {patch.shape}: at most {self.max_batch} x {self.ny} x {self.nx}")
        self._patch = patch
        for _, step in self.steps:
            step(b)
        return self.code[:b]

    def captured(self, patch: wp.array) -> wp.array:
        """The same as a call, replayed from a CUDA graph captured on the first call per `patch`
        buffer and batch size (the graph bakes in the buffer's address)."""
        key = (patch.ptr, patch.shape[0])
        if key not in self._graphs:
            self(patch)  # compile and load every kernel before capturing
            with wp.ScopedCapture(device=self.device) as capture:
                self(patch)
            self._graphs[key] = capture.graph
        wp.capture_launch(self._graphs[key])
        return self.code[: patch.shape[0]]


class HybridTrunk:
    """`net.terrain_code` with cuDNN's convolutions and a Warp norm + SiLU + replicate-pad kernel
    between them, on torch's current stream."""

    def __init__(self, net: ArcDivergenceNet) -> None:
        self.shapes = _check_trunk(net)
        self.net = net
        device = wp.device_from_torch(next(net.parameters()).device)
        self.blocks = []
        for block, (_, cout, stride, _, _) in zip(net.blocks, self.shapes):
            layer_norm = block.norm.norm
            affine = [wp.from_torch(t.detach().contiguous()) for t in (layer_norm.weight, layer_norm.bias)]
            self.blocks.append((block.conv.weight, block.conv.bias, stride, _nchw_norm_pad_kernel(cout),
                                affine, float(layer_norm.eps)))
        self.device = device

    def conv(self, k: int, padded: torch.Tensor) -> torch.Tensor:
        weight, bias, stride = self.blocks[k][:3]
        return F.conv2d(padded, weight, bias, stride=stride)

    def norm(self, k: int, h: torch.Tensor) -> torch.Tensor:
        """Block k's norm + SiLU, padded for the next conv unless k is the last block."""
        kernel, affine, eps = self.blocks[k][3:]
        pad = 0 if k == len(self.blocks) - 1 else 1
        b, c, ho, wo = h.shape
        out = torch.empty(b, c, ho + 2 * pad, wo + 2 * pad, device=h.device)
        wp.launch(kernel, (b, ho + 2 * pad, wo + 2 * pad), inputs=[wp.from_torch(h), *affine, eps, pad],
                  outputs=[wp.from_torch(out)], device=self.device, stream=wp.stream_from_torch(torch.cuda.current_stream()))
        return out

    def tail(self, h: torch.Tensor) -> torch.Tensor:
        """squeeze + geometry. The geometry layer's kernel spans its input exactly, so it is run as
        the dense layer it is (cuBLAS): as a conv, cuDNN took ~6 ms at 512 patches."""
        geometry = self.net.geometry
        h = self.net.squeeze(h).flatten(1)  # NCHW flattening, the order of the conv weight's (c, h, w)
        return F.linear(h, geometry.weight.flatten(1), geometry.bias)[:, :, None, None]

    @torch.no_grad()
    def __call__(self, patch: torch.Tensor) -> torch.Tensor:
        """[B, 1, ny, nx] prepared relief -> [B, geometry_channels, 1, 1], as `net.terrain_code`."""
        h = F.pad(patch, (1, 1, 1, 1), mode="replicate")
        for k in range(len(self.blocks)):
            h = self.norm(k, self.conv(k, h))
        return self.tail(h)


# --- self-test and timing ------------------------------------------------------------------------


def _max_rel(got: torch.Tensor, expected: torch.Tensor) -> float:
    return ((got - expected).abs().max() / expected.abs().max().clamp_min(1e-6)).item()


def _test_patches(net: ArcDivergenceNet, device: torch.device) -> dict[str, torch.Tensor]:
    from feasibility.heightmap.create_box_obstacles import build_centered_box
    from feasibility.heightmap.create_rough_terrain import build_rough_terrain
    from feasibility.lattice_learning.patch import sample_patches

    rng = np.random.default_rng(0)
    spec = net.patch_spec
    patches = {"random x0.3": torch.from_numpy(rng.normal(0.0, 0.3, (512, 1, spec.ny, spec.nx)).astype(np.float32))}
    for name, terrain in (("box", build_centered_box(0.3, 0.08, 70.0, 12.0)), ("rough", build_rough_terrain(extent=12.0))):
        pose = np.column_stack([rng.uniform(-4.0, 4.0, (509, 2)), rng.uniform(-np.pi, np.pi, 509)])
        patches[f"{name}, 509 poses"] = torch.from_numpy(sample_patches(terrain, pose, spec))[:, None]
    return {name: p.to(device) for name, p in patches.items()}


def self_test(net: ArcDivergenceNet, warp_trunk: WarpTrunk, hybrid: HybridTrunk, device: torch.device) -> None:
    for name, patch in _test_patches(net, device).items():
        with torch.no_grad():
            expected = net.terrain_code(patch).reshape(len(patch), -1)
            got_hybrid = hybrid(patch).reshape(len(patch), -1)
        torch.cuda.synchronize()
        flat = patch[:, 0].contiguous()
        code = warp_trunk(wp.from_torch(flat))
        wp.synchronize_device(warp_trunk.device)
        got_warp = wp.to_torch(code).reshape(len(patch), -1).clone()  # the graph below overwrites it
        got_graph = wp.to_torch(warp_trunk.captured(wp.from_torch(flat))).reshape(len(patch), -1).clone()
        wp.synchronize_device(warp_trunk.device)
        errors = {k: _max_rel(v, expected) for k, v in (("hybrid", got_hybrid), ("warp", got_warp), ("graph", got_graph))}
        assert max(errors.values()) < 1e-4, (name, errors)
        print(f"[check] {name}: max rel diff vs terrain_code " + ", ".join(f"{k} {v:.1e}" for k, v in errors.items()))


def _time_ms(fn: Callable[[], object], trials: int = 5, reps: int = 5) -> float:
    """The fastest of `trials` means over `reps` calls: this laptop GPU drops its clock in bursts
    (software power cap), and one burst inflates every number it overlaps ~10x, torch's included."""
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


def bench_layers(net: ArcDivergenceNet, fast: ArcDivergenceNet, warp_trunk: WarpTrunk, hybrid: HybridTrunk,
                 batch: int) -> None:
    """Per block at `batch` patches: conv (torch's replicate pad + cuDNN, cuDNN alone on the
    hybrid's pre-padded input, Warp) and norm + SiLU (torch NCHWChannelLayerNorm, the hybrid's Warp
    kernel, WarpTrunk's in-place one)."""
    spec = net.patch_spec
    patch = torch.randn(batch, 1, spec.ny, spec.nx, device="cuda") * 0.3
    warp_trunk(wp.from_torch(patch[:, 0].contiguous()))  # fills every buffer the steps read
    steps = dict(warp_trunk.steps)
    print(f"[layers] ms at {batch} patches   | conv: pad+cuDNN  cuDNN   Warp | norm+SiLU: torch  hybrid   Warp")
    h = patch
    with torch.no_grad():
        for k, block in enumerate(fast.blocks):
            padded = F.pad(h, (1, 1, 1, 1), mode="replicate")
            conv_out = hybrid.conv(k, padded)
            times = [_time_ms(lambda: block.conv(h)), _time_ms(lambda: hybrid.conv(k, padded)),
                     _time_ms(lambda: steps[f"block {k} conv"](batch)),
                     _time_ms(lambda: block.act(block.norm(conv_out))), _time_ms(lambda: hybrid.norm(k, conv_out)),
                     _time_ms(lambda: steps[f"block {k} norm"](batch))]
            cin, cout, stride, ho, wo = warp_trunk.shapes[k]
            print(f"  block {k} {cin:>2}->{cout:<2} s{stride} {ho:>2}x{wo:<2} |       "
                  + " ".join(f"{t:6.2f}" for t in times[:3]) + " |           " + " ".join(f"{t:6.2f}" for t in times[3:]))
            h = block(h)
        tail_torch, tail_hybrid = _time_ms(lambda: net.geometry(net.squeeze(h))), _time_ms(lambda: hybrid.tail(h))
    tail_warp = _time_ms(lambda: (steps["squeeze"](batch), steps["geometry"](batch)))
    print(f"  squeeze + geometry: torch (as convs) {tail_torch:.2f}, hybrid (conv + linear) {tail_hybrid:.2f}, "
          f"Warp {tail_warp:.2f}")


def bench_trunk(net: ArcDivergenceNet, fast: ArcDivergenceNet, warp_trunk: WarpTrunk, hybrid: HybridTrunk,
                batches: list[int]) -> None:
    """The whole trunk in five forms. Trials are interleaved across the forms, one trial of each per
    round, so a clock that sags as the GPU heats up slows every form alike instead of whichever
    is measured last (seconds per trial at 3584 patches)."""
    spec = net.patch_spec
    print("[trunk] ms per call      torch  torch+NCHW-norm  hybrid  Warp  Warp-graph")
    for n in batches:
        patch = torch.randn(n, 1, spec.ny, spec.nx, device="cuda") * 0.3
        flat = wp.from_torch(patch[:, 0].contiguous())
        forms = [lambda: net.terrain_code(patch), lambda: fast.terrain_code(patch), lambda: hybrid(patch),
                 lambda: warp_trunk(flat), lambda: warp_trunk.captured(flat)]
        times = [float("inf")] * len(forms)
        with torch.no_grad():
            for _ in range(5):
                for k, form in enumerate(forms):
                    times[k] = min(times[k], _time_ms(form, trials=1, reps=3))
                    torch.cuda.empty_cache()
        print(f"  {n:>5} patches  " + "  ".join(f"{t:8.1f}" for t in times))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=None, help="mppi_learning checkpoint (default: random weights)")
    parser.add_argument("--batches", type=int, nargs="+", default=[512, 1536, 3584], help="patch counts to time")
    args = parser.parse_args()
    wp.init()
    device = torch.device("cuda:0")
    net = _load(args.checkpoint, device)
    fast = nchw_layer_norm(copy.deepcopy(net))
    warp_trunk = WarpTrunk(net, max(args.batches + [512]))
    hybrid = HybridTrunk(net)
    self_test(net, warp_trunk, hybrid, device)
    bench_layers(net, fast, warp_trunk, hybrid, args.batches[0])
    bench_trunk(net, fast, warp_trunk, hybrid, args.batches)
