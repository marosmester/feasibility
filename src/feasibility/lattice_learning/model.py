"""`ArcDivergenceNet`: patch [B, N_CHANNELS, 24, 28] + one scalar `kappa` -> (e_pos, e_rot), the
literal implementation of design.md section 5. Read that file first; this module does not
re-derive the arguments, only the arithmetic.

    patch [B,1,24,28]                                   command kappa [B]
    ch0 = relief                                                 |
          |                                                      v
          v                                          command_encoder: Linear 3->64->64
    TERRAIN TRUNK -- 6 conv blocks, NO conditioning                |  e [B,64]
    replicate pad, ChannelLayerNorm, SiLU                          |
          |  [B,96,6,7]  (stride 4, RF 23 px = 2.875 m)            |
          v                                                       |
    Conv1x1 96->24 (channel squeeze)                               |
          |  [B,24,6,7]                                            |
          v                                                        |
    Conv2d 24->256, k=(6,7), valid pad  (THE GEOMETRY LAYER --      |
          |  c [B,256,1,1]   position-aware collapse = flatten+Linear)
          |  the terrain code; control-free, cached per pose        |
          +------------------------- FiLM --------------------------+
          |   gamma, beta = Linear(64 -> 512);  h = c*(1+gamma)+beta
          v
    LayerNorm+SiLU, Linear 256->256+SiLU, Linear 256->256+SiLU, {head_e_pos, head_e_rot}
          |
          v
    y_hat [B,2] in log1p / standardised space

Three things this module is NOT allowed to get wrong, each with its own `__main__` self-check
(design.md section 8):

* The trunk must never see `kappa` (design.md section 5a's caching argument -- one trunk pass
  serves every primitive at a pose). Enforced by the SIGNATURE of `terrain_code()`, which has no
  command argument at all; checked by autograd as a defence against a future refactor that
  accidentally threads one through.
* `ChannelLayerNorm`, not `GroupNorm`/`BatchNorm`/`InstanceNorm` -- those pool over space, which
  would make the trunk's receptive field the whole input and break the patch<->map equivalence
  design.md section 5d's 40x speedup depends on.
* The geometry layer is written as `nn.Conv2d(squeeze_channels, geometry_channels, kernel_size)`,
  not `nn.Linear` on a flattened feature map -- arithmetically identical on a single window
  (design.md section 5c), but this way `terrain_code()` is fully convolutional BY CONSTRUCTION:
  feed it a patch and it returns one code; feed it a whole relief map (`encode_map`) and it
  returns a dense field of codes at the SAME weights, no separate implementation.

`TargetTransform`/`Normalizer` live HERE, not in `custom_dataset.py` -- the opposite of
`learning/`'s arrangement, matching `grid_learning_2/model.py` -- so this module has no dataset
dependency and can run and self-check before any dataset file exists; `custom_dataset.py` imports
both from here (design.md section 11a).

CLI parameters:
    --base-width INT      trunk channel unit; layer widths are 1x/2x/3x this (default: 32)
    --embed-dim INT       command-embedding width, also the FiLM input width (default: 64)
    --head-fusion STR     film | concat -- design.md section 5b's ablation (default: film)

Usage:
    python src/feasibility/lattice_learning/model.py     # shape / RF / equivalence / causality checks
"""
from __future__ import annotations

import argparse
import dataclasses
import math

import torch
import torch.nn as nn

from feasibility.lattice_learning.arc import KAPPA_MAX
from feasibility.lattice_learning.patch import N_CHANNELS
from feasibility.lattice_learning.patch import PatchSpec

TARGET_NAMES = ("e_pos", "e_rot")
OUT_DIM = len(TARGET_NAMES)  # 2

DEFAULT_BASE_WIDTH = 32  # trunk channel unit -- the single scaling knob (design.md section 5b)
DEFAULT_EMBED_DIM = 64  # command encoder width, also FiLM's input width
DEFAULT_GEOMETRY_CHANNELS = 256  # width of the terrain code c
DEFAULT_SQUEEZE_CHANNELS = 24  # channel count feeding the geometry layer
DEFAULT_HEAD_WIDTH = 256  # head trunk width

# (kappa/K, |kappa|/K, sign(kappa)) -- design.md section 4b. |kappa| sets how far the rim sweeps,
# the sign sets which way; handing the net the split explicitly saves it building a V-shape out of
# a monotone input.
N_CMD_FEATURES = 3

# (channel multiplier of base_width, stride) per trunk block -- design.md section 5b's table,
# literally: 6 blocks, two stride-2 downsamples (total stride 4), channels 1x -> 1x -> 2x -> 2x ->
# 3x -> 3x. This list IS the receptive-field derivation (receptive_field() below) and it is chosen
# to land RF at 23 px = 2.875 m -- both load-bearing, not a free tuning knob.
TRUNK_PLAN: tuple[tuple[int, int], ...] = ((1, 1), (1, 1), (2, 2), (2, 1), (3, 2), (3, 1))

HEAD_FUSIONS = ("film", "concat")


def _conv3_out(n: int, stride: int) -> int:
    """Output size of one kernel=3, padding=1 ("same"-at-stride-1) conv -- the arithmetic behind
    every row of design.md section 5b's table."""
    return (n - 1) // stride + 1


def trunk_out_size(n: int, plan: tuple[tuple[int, int], ...] = TRUNK_PLAN) -> int:
    """Spatial size along one axis after the whole trunk -- 24 -> 6, 28 -> 7 at the defaults."""
    for _, stride in plan:
        n = _conv3_out(n, stride)
    return n


def total_stride(plan: tuple[tuple[int, int], ...] = TRUNK_PLAN) -> int:
    """Product of the trunk's strides -- 4 at the defaults: the input pixel pitch of one
    `terrain_code()` output cell (design.md section 6b)."""
    return math.prod(stride for _, stride in plan)


def receptive_field(plan: tuple[tuple[int, int], ...] = TRUNK_PLAN, kernel: int = 3) -> int:
    """Receptive field in input pixels of the trunk's last block, via the standard conv-stack
    recurrence RF += (kernel-1)*jump, jump *= stride -- design.md section 5b's RF column. Verified
    against an actual autograd probe in __main__, not trusted blindly (that probe is what would
    catch a substituted norm silently making the trunk's RF map-global)."""
    rf, jump = 1, 1
    for _, stride in plan:
        rf += (kernel - 1) * jump
        jump *= stride
    return rf


def command_features(kappa: torch.Tensor, kappa_max: float = KAPPA_MAX) -> torch.Tensor:
    """[...] curvature (1/m) -> [..., 3] = (kappa/K, |kappa|/K, sign(kappa)) -- design.md 4b."""
    k = kappa / kappa_max
    return torch.stack([k, k.abs(), torch.sign(kappa)], dim=-1)


@dataclasses.dataclass(frozen=True)
class Normalizer:
    """Per-column (x - mean) / std, std floored so a constant column never divides by zero --
    restated rather than imported, matching the rest of this tree's dependency stance (design.md
    section 11a): this module has no dataset to import one from yet."""

    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def from_tensor(cls, x: torch.Tensor, eps: float = 1e-8) -> "Normalizer":
        return cls(mean=x.mean(dim=0), std=x.std(dim=0).clamp_min(eps))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


@dataclasses.dataclass(frozen=True)
class TargetTransform:
    """Physical (e_pos m, e_rot rad) <-> log1p/standardized model space:

        forward:  y (m, rad) --log1p--> standardize --> what the loss is computed on
        inverse:  network output --unstandardize--> clamp at 0 --expm1--> y (m, rad)

    Both targets are non-negative and right-skewed (design.md section 7c's SNR note: most rows
    are near-zero, the informative ones are the collision tail), so log1p keeps a few large rows
    from dominating the gradient and keeps the network from predicting a physically impossible
    negative error. The clamp -- not expm1 -- is what enforces non-negativity: expm1 maps
    R -> (-1, inf), so an unclamped inverse of a confidently-wrong prediction could still hand back
    a negative distance. Round-tripping a genuine target is unaffected, since log1p of a real
    target is already >= 0.

    Fit on TRAIN rows only (custom_dataset.py's job); inverse() is for reporting/evaluation in
    physical units, and the training loss is computed in model space so the clamp's zero gradient
    can never mask an error."""

    normalizer: Normalizer

    @classmethod
    def fit(cls, y: torch.Tensor) -> "TargetTransform":
        return cls(normalizer=Normalizer.from_tensor(torch.log1p(y)))

    def _on(self, device: torch.device) -> Normalizer:
        if self.normalizer.mean.device == device:
            return self.normalizer
        return Normalizer(mean=self.normalizer.mean.to(device), std=self.normalizer.std.to(device))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Physical [*, 2] (m, rad) -> standardized log1p space."""
        return self._on(y.device)(torch.log1p(y))

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Standardized log1p space -> physical [*, 2] (m, rad), guaranteed non-negative."""
        return torch.expm1(self._on(y.device).inverse(y).clamp_min(0.0))


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the CHANNEL dimension only, applied independently at every spatial position
    -- the ConvNeXt-style "LayerNorm2d". NOT GroupNorm/InstanceNorm/BatchNorm: those normalise
    over (channels, H, W) JOINTLY, so their output at one position depends on every other
    position, silently making the block's receptive field the WHOLE input regardless of kernel and
    stride. That would break design.md section 5d's patch<->map equivalence (a norm with spatial
    extent pools over a 24x28 patch in one mode and over a whole map in the other -- different
    numbers) and would be caught by the receptive-field probe below."""

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvBlock(nn.Module):
    """One unconditioned trunk block: conv3 -> ChannelLayerNorm -> SiLU.

    `replicate` padding, never zeros: zero padding stamps a distinctive constant at the border --
    a route to encoding absolute position -- whereas replicate extends the edge height outward,
    the same boundary condition `HeightMapReader.sample` uses (it clamps to the nearest edge cell
    rather than raising), so it is the physically consistent choice, not a hack. It is also what
    makes design.md section 5d's equivalence test exact: embedding a patch inside a larger map by
    replicate-extending its own border reproduces exactly what this padding already assumes beyond
    the patch edge, so real map data and the trunk's padding assumption agree by construction."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1,
            padding_mode="replicate",
        )
        self.norm = ChannelLayerNorm(out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ArcDivergenceNet(nn.Module):
    """patch [B, N_CHANNELS, ny, nx] + kappa [B] -> [B, 2] in TargetTransform (log1p/standardized)
    space -- design.md section 5b.

    `target_transform` is a plain attribute, not a buffer: it is fitted data, not a learned
    parameter, so it stays out of `state_dict()` and must be saved alongside a checkpoint (and
    re-attached on load) for `predict()` to work -- same contract as `learning/model.py`'s
    `PoseErrorMLP` and `grid_learning_2/model.py`'s `GridDivergenceNet`."""

    def __init__(
        self,
        *,
        base_width: int = DEFAULT_BASE_WIDTH,
        embed_dim: int = DEFAULT_EMBED_DIM,
        geometry_channels: int = DEFAULT_GEOMETRY_CHANNELS,
        squeeze_channels: int = DEFAULT_SQUEEZE_CHANNELS,
        head_width: int = DEFAULT_HEAD_WIDTH,
        head_fusion: str = "film",
        patch_spec: PatchSpec | None = None,
        target_transform: TargetTransform | None = None,
    ) -> None:
        super().__init__()
        if head_fusion not in HEAD_FUSIONS:
            raise ValueError(f"head_fusion must be one of {HEAD_FUSIONS}, got {head_fusion!r}")
        spec = patch_spec or PatchSpec()
        stride = total_stride()
        geom_h, geom_w = spec.ny // stride, spec.nx // stride
        if spec.ny % stride or spec.nx % stride:
            raise ValueError(
                f"PatchSpec {spec.ny}x{spec.nx} is not an exact multiple of the trunk's total "
                f"stride {stride} -- the geometry layer's kernel size would not be well-defined"
            )
        self.patch_spec = spec
        self.head_fusion = head_fusion
        self.target_transform = target_transform

        blocks: list[nn.Module] = []
        in_channels = N_CHANNELS
        for mult, block_stride in TRUNK_PLAN:
            out_channels = base_width * mult
            blocks.append(ConvBlock(in_channels, out_channels, block_stride))
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)

        # Channel squeeze before the full-window kernel: cuts the geometry layer's parameter count
        # by (in_channels/squeeze_channels)x for free (design.md section 5b).
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        # THE GEOMETRY LAYER: a full-window Conv2d, valid padding. Arithmetically identical to
        # nn.Linear(squeeze_channels*geom_h*geom_w -> geometry_channels) on a single window
        # (design.md section 5c) -- written as a conv so it is ALSO the dense map-mode operator,
        # sliding over a bigger squeezed feature map with the SAME weights (section 6b).
        self.geometry = nn.Conv2d(squeeze_channels, geometry_channels, kernel_size=(geom_h, geom_w))

        self.command_encoder = nn.Sequential(
            nn.Linear(N_CMD_FEATURES, embed_dim), nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # design.md section 5d: FiLM supplies the MULTIPLICATIVE interaction (how much a terrain
        # feature matters is GATED by how far the rim sweeps, not merely offset by it), an MLP
        # after it supplies general nonlinear mixing, and neither subsumes the other in practice.
        # A concat head's first layer is additive in the command, and with one command scalar
        # against `geometry_channels` terrain channels the command's gradient is easily swamped
        # early. `--head-fusion concat` is the honest control for that claim, not an argument.
        if head_fusion == "film":
            self.film = nn.Linear(embed_dim, 2 * geometry_channels)
            head_in = geometry_channels
        else:
            self.film = None
            head_in = geometry_channels + embed_dim
        self.head_pre = nn.Sequential(nn.LayerNorm(head_in), nn.SiLU())
        self.head_trunk = nn.Sequential(
            nn.Linear(head_in, head_width), nn.SiLU(),
            nn.Linear(head_width, head_width), nn.SiLU(),
        )
        # Separate heads rather than one Linear(head_width, 2) purely for readability -- the two
        # are arithmetically identical, but named heads keep the (e_pos, e_rot) column order,
        # which TARGET_NAMES depends on, explicit at the definition site.
        self.head_e_pos = nn.Linear(head_width, 1)
        self.head_e_rot = nn.Linear(head_width, 1)

    def terrain_code(self, x: torch.Tensor) -> torch.Tensor:
        """[B, N_CHANNELS, H, W] relief -> [B, geometry_channels, H', W'] control-free terrain
        code -- NO `kappa` argument, by construction (design.md section 5a's caching guarantee).

        Valid for any (H, W) at least (patch_spec.ny, patch_spec.nx): patch mode (H'=W'=1) and the
        dense map mode of design.md section 6b are the SAME call, since the trunk, squeeze and
        geometry layer are all convolutions (section 5c) -- there is no separate map-mode
        implementation to drift out of sync with this one."""
        h = x
        for block in self.blocks:
            h = block(h)
        h = self.squeeze(h)
        return self.geometry(h)

    def encode_map(self, relief: torch.Tensor) -> torch.Tensor:
        """[B, H, W] or [B, N_CHANNELS, H, W] full-map relief -> dense terrain code
        [B, geometry_channels, H', W'] -- design.md section 6b's Phase 2 entry point. Same weights
        `forward()` uses; the only difference is the input's spatial extent. Applying FiLM/head to
        this dense field per output position is the planner's job (`infer.py`), not this module's."""
        if relief.ndim == 3:
            relief = relief.unsqueeze(1)
        return self.terrain_code(relief)

    def _head(self, c: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        """[B, geometry_channels] terrain code + [B] curvature -> [B, 2] in model space."""
        e = self.command_encoder(command_features(kappa))  # [B, embed_dim]
        if self.head_fusion == "film":
            gamma, beta = self.film(e).chunk(2, dim=-1)
            h = c * (1.0 + gamma) + beta  # `1 + gamma`: identity modulation at init
        else:
            h = torch.cat([c, e], dim=-1)
        h = self.head_trunk(self.head_pre(h))
        return torch.cat([self.head_e_pos(h), self.head_e_rot(h)], dim=-1)

    def forward(self, patch: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        """patch [B, N_CHANNELS, patch_spec.ny, patch_spec.nx], kappa [B] -> [B, 2] in
        TargetTransform (log1p/standardized) space. Patch mode: the geometry layer's window
        exactly spans the input, so `terrain_code` collapses to a single 1x1 code per row."""
        if patch.ndim != 4 or patch.shape[1] != N_CHANNELS:
            raise ValueError(f"patch must be [B, {N_CHANNELS}, H, W], got {tuple(patch.shape)}")
        code = self.terrain_code(patch)
        if code.shape[-2:] != (1, 1):
            raise ValueError(
                f"forward() is patch mode and expects a 1x1 terrain code, got spatial "
                f"{tuple(code.shape[-2:])} -- feed a {self.patch_spec.ny}x{self.patch_spec.nx} "
                f"patch, or use encode_map() for a dense field"
            )
        return self._head(code[..., 0, 0], kappa)

    @torch.no_grad()
    def predict(self, patch: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        """Same signature as forward(), returning physical (e_pos m, e_rot rad). Inference only --
        needs a target_transform, exactly like PoseErrorMLP.predict()."""
        if self.target_transform is None:
            raise RuntimeError(
                "predict() needs a target_transform to undo the log1p/standardize the net was "
                "trained in -- pass one to __init__ or assign .target_transform. Use forward() "
                "if you want the raw model-space output."
            )
        return self.target_transform.inverse(self(patch, kappa))


def _grad_support(net: ArcDivergenceNet, ny: int, nx: int, row: int, col: int) -> torch.Tensor:
    """[ny, nx] bool mask of which input pixels the TRUNK's feature cell (row, col) depends on,
    measured by autograd rather than derived -- shared by the receptive-field and mirror-
    registration checks below."""
    x = torch.zeros(1, N_CHANNELS, ny, nx, requires_grad=True)
    h = x
    for block in net.blocks:
        h = block(h)
    h[0, 0, row, col].backward()
    assert x.grad is not None
    return x.grad[0, 0].abs() > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-width", type=int, default=DEFAULT_BASE_WIDTH)
    parser.add_argument("--embed-dim", type=int, default=DEFAULT_EMBED_DIM)
    parser.add_argument("--head-fusion", type=str, default="film", choices=HEAD_FUSIONS)
    args = parser.parse_args()

    spec = PatchSpec()
    net = ArcDivergenceNet(
        base_width=args.base_width, embed_dim=args.embed_dim, head_fusion=args.head_fusion
    )
    n_params = sum(p.numel() for p in net.parameters())
    print(f"ArcDivergenceNet(base_width={args.base_width}, embed_dim={args.embed_dim}, "
          f"head_fusion={args.head_fusion}): {n_params} params")
    print(f"[trunk] {spec.ny}x{spec.nx} -> {trunk_out_size(spec.ny)}x{trunk_out_size(spec.nx)}, "
          f"total stride {total_stride()}")

    # --- shapes: patch mode collapses to one code / one prediction per row ----------------------
    B = 4
    patch = torch.randn(B, N_CHANNELS, spec.ny, spec.nx)
    kappa = torch.empty(B).uniform_(-KAPPA_MAX, KAPPA_MAX)
    code = net.terrain_code(patch)
    assert code.shape == (B, DEFAULT_GEOMETRY_CHANNELS, 1, 1), code.shape
    y_hat = net(patch, kappa)
    assert y_hat.shape == (B, OUT_DIM), y_hat.shape
    print(f"[forward] patch {tuple(patch.shape)}, kappa {tuple(kappa.shape)} "
          f"-> y_hat {tuple(y_hat.shape)}")

    # --- receptive field: measured at a trunk-output cell whose full 23 px RF sits INSIDE the
    # patch (row=3 of 6, col=3 of 7 -- see design.md 5b's table), so the probe isn't confounded by
    # replicate padding clipping the true value down to whatever the patch happens to contain. ---
    rf = receptive_field()
    support = _grad_support(net, spec.ny, spec.nx, row=3, col=3)
    rows = support.any(dim=1).nonzero()
    cols = support.any(dim=0).nonzero()
    measured_h = int(rows.max() - rows.min()) + 1
    measured_w = int(cols.max() - cols.min()) + 1
    assert measured_h == rf and measured_w == rf, (measured_h, measured_w, rf)
    print(f"[receptive field] formula {rf} px = {rf * 0.125:.3f} m, autograd probe "
          f"{measured_h}x{measured_w} px -- match")

    # --- command independence of the trunk (design.md section 5a/8): terrain_code() has no
    # `kappa` argument at all, so this is a defence against a future refactor accidentally
    # threading one through, not a probabilistic check. ------------------------------------------
    kappa_grad = torch.tensor([0.7], requires_grad=True)
    c_sum = net.terrain_code(patch[:1]).sum()
    grad = torch.autograd.grad(c_sum, kappa_grad, allow_unused=True)[0]
    assert grad is None or torch.all(grad == 0), "the trunk's output depends on kappa"
    print("[causality] terrain_code(patch) has zero gradient w.r.t. kappa")

    # --- patch <-> map equivalence (design.md section 5d): terrain_code() has no operation with
    # unbounded spatial extent (ChannelLayerNorm is per-position, every conv is local), so it must
    # be a genuinely POSITION-INDEPENDENT operator: querying it on a big map and on an honest,
    # generously-sized crop of that SAME map must agree EXACTLY at a position whose full receptive
    # field -- through the geometry layer's own aggregation, not just one trunk cell -- is
    # unclipped (no replicate padding invoked) in BOTH. This is the property section 6b's dense
    # map-mode pass actually depends on; a GroupNorm/BatchNorm swap would break it immediately,
    # since global statistics differ between a small crop and a big map even far from any edge.
    #
    # The literal PatchSpec-sized (24x28) training patch does NOT satisfy "unclipped" -- section
    # 3a deliberately sizes the receptive field (23 px) to approach the patch extent, so most of
    # its 6x7 trunk-output window IS replicate-padding-influenced. That is section 3a's causal-
    # support tradeoff, not a bug, and forward()'s accuracy against real terrain is train.py's job
    # to measure, not this structural test's -- crop_size below is chosen deliberately larger than
    # the training patch so the test isolates "is the machinery position-independent" from "is the
    # small patch's own boundary approximation small".
    crop_size = 64  # confirmed empirically to clear the geometry layer's full aggregation margin
    offset = 40
    big_map = torch.randn(1, N_CHANNELS, 2 * offset + crop_size + 20, 2 * offset + crop_size + 20)
    crop = big_map[:, :, offset:offset + crop_size, offset:offset + crop_size]
    with torch.no_grad():
        code_big = net.terrain_code(big_map)[0]
        code_crop = net.terrain_code(crop)[0]
    shift = offset // total_stride()
    cr, cc = code_crop.shape[-2] // 2, code_crop.shape[-1] // 2
    equiv_err = (code_big[:, shift + cr, shift + cc] - code_crop[:, cr, cc]).abs().max().item()
    assert equiv_err < 1e-4, f"terrain_code is not position-independent (err {equiv_err:.2e})"
    print(f"[patch<->map] interior crop vs. big-map code exact to {equiv_err:.2e}")

    # --- mirror REGISTRATION (design.md sections 7c/8): unlike grid_learning_2, this network's
    # output is a single collapsed code, not a spatial field, so there is no "output row p mirrors
    # output row G-1-p" to check on the TRUNK (and, empirically, there is none to find: with an
    # EVEN patch height and stride 4, the trunk's own output grid {0, 4, 8, 12, 16, 20} isn't
    # symmetric about the patch's center 11.5, so per-row receptive-field supports are not mirror
    # pairs -- that is a property of this even-sized, stride-4 grid, not a bug). What DOES need to
    # be exact, and is a precondition for section 7c's "flip the patch, negate kappa" augmentation
    # to be a real physical symmetry at all, is that the INPUT rows themselves are geometrically
    # mirrored: row i's body-Y and row (ny-1-i)'s body-Y must be exact negatives. Asserted on
    # PatchSpec.ys() directly, independent of the network. Model-space EQUIVARIANCE (the network
    # actually predicting the same output under the flip) is only reported, since a conv stack at
    # random init is not reflection-equivariant. ---------------------------------------------------
    ys = spec.ys()
    assert torch.allclose(torch.from_numpy(ys), -torch.from_numpy(ys[::-1].copy()), atol=1e-9), (
        "patch rows are not geometrically mirror-symmetric about y=0"
    )
    print(f"[mirror] patch row geometry is exactly mirror-symmetric about y=0 ({spec.ny} rows)")
    net.eval()
    with torch.no_grad():
        mirrored = net(torch.flip(patch, dims=[-2]), -kappa)
        diff = (y_hat - mirrored).abs().max().item()
    print(f"[mirror] model-space equivariance at init (untrained): {diff:.3e} "
          f"(expected O(1) until trained with the y-mirror + kappa-negate augmentation)")

    # --- TargetTransform round-trip and non-negativity -------------------------------------------
    y_phys = torch.rand(64, OUT_DIM) * 3.0  # stand-in physical (e_pos, e_rot), non-negative
    transform = TargetTransform.fit(y_phys)
    assert torch.allclose(transform.inverse(transform.forward(y_phys)), y_phys, atol=1e-5), "not invertible"
    assert (transform.inverse(torch.randn(64, OUT_DIM) * 5.0) >= 0).all(), "negative error escaped"

    net.target_transform = transform
    prediction = net.predict(patch, kappa)
    assert prediction.shape == (B, OUT_DIM) and (prediction >= 0).all()
    print(f"[predict] range: [{prediction.min():.4f}, {prediction.max():.4f}] "
          f"({', '.join(TARGET_NAMES)})")

    # --- the concat ablation must at least build and run to the same shape -----------------------
    concat_net = ArcDivergenceNet(
        base_width=args.base_width, embed_dim=args.embed_dim, head_fusion="concat"
    )
    assert concat_net(patch, kappa).shape == (B, OUT_DIM)
    n_concat = sum(p.numel() for p in concat_net.parameters())
    print(f"[ablation] head_fusion=concat builds and runs: {n_concat} params "
          f"({n_params - n_concat:+d} vs film)")
    print("all self-checks ok")
