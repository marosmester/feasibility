"""`GridPoseErrorNet`: the divergence-FIELD architecture described in design.md -- read
that file first, this module is the literal implementation of its section 4 (network), 3
(preprocessing) and 5a (target transform). Consumes exactly what
custom_dataset.GridPoseErrorDataset yields:

    heightmap [B, G_h, G_h]  (absolute world z, m)   +   wz [B]  (rad/s)
    -->  y_hat [B, G, G, 2]  in TargetTransform (log1p/standardized) space

plus `spawn_xy` [G, G, 2] (world x,y of every lattice cell, identical for every row -- read
straight from the dataset file, not hardcoded here).

Architecture, one line each (see design.md for the why):
    heightmap -> per-sample relief, in wheel radii     (section 3a)
    wz        -> (wz/WZ_MAX, |wz|/WZ_MAX)              (section 3b)
    8x FiLM'd conv blocks, valid-info-only via
      replicate padding, receptive field capped at
      35 px = 3.5 m                                    (section 4b/4c)
    F.grid_sample at spawn_xy, on the 25x25 @ 0.40 m
      feature map, corrected by readout_offset()       (section 4d)
    2x Conv1x1 head (shared across cells)               (section 4d)

Deliberately independent of feasibility.learning, same non-dependence stance as the rest of this
directory (see generate_dataset.py's module docstring) -- but custom_dataset.py and utils.py are
siblings IN this package, so their Normalizer/TARGET_NAMES/DEFAULT_EXTENT are reused via plain
`feasibility.grid_learning` imports, not reimplemented.

WHEEL_RADIUS is restated locally (matching HelhestJuniorConfig.WHEEL_RADIUS = 0.35 m) rather
than imported from `examples.helhest_junior.common`, which would drag ostrich/warp into every
training run for the sake of one float -- the same "restate rather than import" call
generate_dataset.py makes for WZ_RANGE, for a lighter reason here: that module already pays the
simulator import cost to run rollouts, this one (and the training script built on it) never
touches a simulator at all.

CLI parameters:
    --base-width INT   trunk channel unit; layer widths are 1x/2x/3x this (default: 32)
    --embed-dim INT    wz command-embedding width, also the FiLM input width (default: 32)
    --extent FLOAT     heightmap tensor extent, m -- must match the dataset file (default: 10.0)

Usage:
    python src/feasibility/grid_learning/model.py     # shape/RF/param-count smoke check
"""
from __future__ import annotations

import argparse
import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F

from feasibility.grid_learning.custom_dataset import Normalizer
from feasibility.grid_learning.custom_dataset import TARGET_NAMES
from feasibility.grid_learning.utils import DEFAULT_EXTENT

WHEEL_RADIUS = 0.35  # m -- HelhestJuniorConfig.WHEEL_RADIUS, restated (see module docstring)
WZ_MAX = 1.0  # rad/s -- generate_dataset.WZ_RANGE's magnitude, restated for the same reason

DEFAULT_BASE_WIDTH = 32
DEFAULT_EMBED_DIM = 32
OUT_DIM = len(TARGET_NAMES)  # 2 -- (e_pos, e_rot)

# (channel multiplier of base_width, stride) per trunk block -- design.md section 4c's
# table, literally: 8 blocks, two stride-2 downsamples, channels widen 1x -> 2x -> 3x. This list
# IS the receptive-field derivation -- changing it changes the 35 px / 3.5 m RF the whole trunk
# depth was chosen to hit, so treat it as load-bearing, not a free tuning knob.
TRUNK_PLAN: tuple[tuple[int, int], ...] = (
    (1, 1), (1, 1), (2, 2), (2, 1), (2, 1), (3, 2), (3, 1), (3, 1),
)


def receptive_field(plan: tuple[tuple[int, int], ...] = TRUNK_PLAN, kernel: int = 3) -> int:
    """Receptive field in input pixels of the last block of `plan`, via the standard conv-stack
    recurrence RF += (kernel-1)*jump, jump *= stride -- the arithmetic behind design.md's
    per-block RF column. Used both to report the number and, in the self-check below, to verify
    it against an actual autograd probe rather than trusting the formula blindly."""
    rf, jump = 1, 1
    for _, stride in plan:
        rf += (kernel - 1) * jump
        jump *= stride
    return rf


def readout_offset(extent: float, n_input: int, n_feat: int) -> float:
    """World-space offset (m) of the trunk's feature lattice from the map centre, which
    design.md section 4d's "spans world x,y in [-5.0, +5.0]" glosses over and `grid_sample`
    would otherwise get wrong.

    A stride-2 `padding=1` conv centres output cell k on INPUT cell 2k, so after two of them cell
    k sits on input cell `S*k` (S = 4). On an even 100-cell input those centres run
    -4.95 .. +4.65 m, not the -4.8 .. +4.8 m that `align_corners=False` assumes for a 25-cell grid
    spanning +-5.0 m -- a uniform -0.15 m shift on both axes. Left uncorrected it misregisters
    terrain against the spawn lattice by 1.5 heightmap pixels, and (worse) breaks section 6a's
    EXACT mirror symmetry: under a y-flip the shift becomes +0.15 m, so the augmentation would be
    showing the net two copies of the same map 0.30 m out of register and the mirror-equivariance
    check could never reach numerical noise.

    Derivation: feature centre k is at world `-extent/2 + res*(S*k + 0.5)`, so the lattice centre
    (k = (n_feat-1)/2) sits at `res * (1 - S) / 2` away from the origin. The pitch is `res*S`, so
    the lattice's half-width is still exactly `extent/2` -- only the centre moves, which is why
    the caller can keep dividing by `extent/2` after subtracting this."""
    resolution = extent / n_input
    total_stride = n_input // n_feat
    return resolution * (1 - total_stride) / 2.0


def relief(heightmap: torch.Tensor, wheel_radius: float = WHEEL_RADIUS) -> torch.Tensor:
    """[B, H, W] absolute world z (m) -> [B, 1, H, W] height relative to this SAMPLE's own median,
    scaled by wheel radius -- design.md section 3a. The median is computed per sample (each
    map has its own background elevation) and is a single scalar per sample: it recenters every
    pixel by the same amount, so it does not leak position-specific information about far-away
    terrain into a given pixel -- only "what counts as flat here", the same background estimate
    generate_dataset.obstacle_height_threshold uses. Flat ground -> exactly zero, checked below."""
    baseline = heightmap.flatten(1).median(dim=1).values.view(-1, 1, 1, 1)
    return (heightmap.unsqueeze(1) - baseline) / wheel_radius


def command_features(wz: torch.Tensor, wz_max: float = WZ_MAX) -> torch.Tensor:
    """[B] rad/s -> [B, 2] = (wz/wz_max, |wz|/wz_max) -- design.md section 3b. Two numbers
    because they mean different physical things: |wz| sets how far the fixed-duration turn
    sweeps (so how much of the disc is swept at all), the sign sets which way the rear wheel
    goes -- handing the net both spares it from carving |.| out of what is otherwise one linear
    input."""
    return torch.stack([wz / wz_max, wz.abs() / wz_max], dim=-1)


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the CHANNEL dimension only, applied independently at every spatial
    position -- normalizes each (h, w) location's C-vector by its own mean/std, never pooling
    over H or W. This is the ConvNeXt-style "LayerNorm2d", and the reason it is used here instead
    of GroupNorm/InstanceNorm/BatchNorm is not stylistic: GroupNorm computes its mean/std over
    (channels_in_group, H, W) JOINTLY per sample, so its output at any one position depends on
    statistics pooled from every other position -- silently making every block's receptive field
    the WHOLE map, regardless of kernel/stride. That is exactly the failure model.py's own
    receptive-field self-check caught (measured RF came back 100 px, not the intended 35, the
    first time this block used GroupNorm). A per-position channel norm has no such pooling, so it
    cannot leak position information and the conv geometry alone determines the RF."""

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class FiLMBlock(nn.Module):
    """One conv block, FiLM-conditioned on the command embedding `e` -- design.md section
    4b. `replicate` padding, never zeros: zero padding stamps a distinctive constant at the
    border (a route to encoding absolute position), whereas replicate extends the edge height
    outward, which is the SAME boundary condition HeightMapReader.sample uses (clamps to the
    nearest edge cell rather than raising) -- so this is the physically-consistent choice, not a
    hack. ChannelLayerNorm (not GroupNorm/BatchNorm -- see that class's docstring) because
    cross-position or cross-batch statistics would both break the receptive-field guarantee or
    behave differently train vs eval; SiLU to match learning/model.py. `(1 + gamma)` rather than
    `gamma` initializes the block to an identity modulation, and FiLM's own gamma/beta are a
    per-channel broadcast (same value at every position, computed only from `wz`) so they add no
    cross-position leakage either."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, embed_dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1,
            padding_mode="replicate",
        )
        self.norm = ChannelLayerNorm(out_channels)
        self.film = nn.Linear(embed_dim, 2 * out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        h = self.norm(self.conv(x))
        gamma, beta = self.film(e).chunk(2, dim=-1)
        h = h * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        return self.act(h)


@dataclasses.dataclass(frozen=True)
class TargetTransform:
    """Physical (e_pos m, e_rot rad) <-> log1p/standardized model space -- restated from
    learning/model.py's TargetTransform verbatim in spirit (see that module's docstring for the
    full non-negativity argument), wrapping this directory's own custom_dataset.Normalizer.

    Fit on TRAIN rows' VALID (mask == True) cells only: masked cells are exact zeros by
    construction (generate_dataset.py's `y[~mask] = 0.0`), and folding thousands of spurious
    zeros into the mean/std would badly skew the standardization toward "everything is zero" on
    a field that is mostly-flat-ground anyway (see design.md section 5a)."""

    normalizer: Normalizer

    @classmethod
    def fit(cls, y_valid: torch.Tensor) -> "TargetTransform":
        """Fit on [n, 2] physical targets, already filtered to mask==True, TRAIN rows only."""
        return cls(normalizer=Normalizer.from_tensor(torch.log1p(y_valid)))

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


class GridPoseErrorNet(nn.Module):
    """heightmap [B, G_h, G_h] + wz [B] -> divergence field [B, G, G, 2] at `spawn_xy`'s
    positions, in TargetTransform (log1p/standardized) space -- design.md section 4.

    `spawn_xy` and `extent` are forward() arguments rather than baked into the model: `spawn_xy`
    comes straight from the dataset file (so the lattice geometry is read from data, never
    hardcoded), and `extent` only has to match the file the heightmap tensor was resampled at
    (utils.heightmap_to_tensor's `extent`, DEFAULT_EXTENT unless overridden at generation time)
    -- it is passed at call time rather than fixed in `__init__` so one model can be evaluated
    against a differently-configured dataset file without reconstruction, exactly like
    `PoseErrorMLP.predict()` needs its `target_transform` passed in or assigned rather than
    frozen at construction.

    `target_transform` IS stored as a plain attribute (not a buffer): it is fitted data, not a
    learned parameter, so it stays out of state_dict() and must be saved alongside a checkpoint
    and re-attached on load for predict() to work -- same contract as PoseErrorMLP."""

    def __init__(
        self,
        *,
        base_width: int = DEFAULT_BASE_WIDTH,
        embed_dim: int = DEFAULT_EMBED_DIM,
        wheel_radius: float = WHEEL_RADIUS,
        target_transform: TargetTransform | None = None,
    ) -> None:
        super().__init__()
        self.wheel_radius = wheel_radius
        self.target_transform = target_transform

        self.command_mlp = nn.Sequential(
            nn.Linear(2, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim),
        )

        blocks = []
        in_channels = 1  # single-channel relief map
        for mult, stride in TRUNK_PLAN:
            out_channels = base_width * mult
            blocks.append(FiLMBlock(in_channels, out_channels, stride, embed_dim))
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)

        # design.md section 4a's per-cell head: Conv1x1 96 -> 64 -> 2. Both widths are
        # expressed in base_width (3x -> 2x) rather than as a fraction of the trunk's output, so
        # they scale with the one knob section 4e names and can't silently floor for a TRUNK_PLAN
        # whose last multiplier isn't divisible by 3.
        self.head = nn.Sequential(
            nn.Conv2d(TRUNK_PLAN[-1][0] * base_width, 2 * base_width, kernel_size=1), nn.SiLU(),
            nn.Conv2d(2 * base_width, OUT_DIM, kernel_size=1),
        )

    def conv_trunk(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """[B, 1, H, W] relief, [B, embed_dim] command embedding -> [B, C, H/4, W/4] feature map.
        Split out of `forward` so the receptive-field self-check below can probe the conv stack
        in isolation, without `relief()`'s median-centering (a single per-sample scalar, not a
        per-position effect -- see that function's docstring) in the loop."""
        for block in self.blocks:
            x = block(x, e)
        return x

    def forward(
        self, heightmap: torch.Tensor, wz: torch.Tensor, spawn_xy: torch.Tensor,
        extent: float = DEFAULT_EXTENT, blur_terrain: bool = False,
    ) -> torch.Tensor:
        """heightmap [B, H, W], wz [B], spawn_xy [G, G, 2] (world x, y; shared across the batch)
        -> [B, G, G, 2] in TargetTransform space. `extent` must match the heightmap tensor's own
        (utils.heightmap_to_tensor's `extent` at generation time), so the grid_sample readout
        queries world coordinates against the right normalization -- see design.md 4d.

        `blur_terrain` is design.md section 7's third baseline: flatten the terrain to a
        single number per sample, keeping the architecture and the command path intact, so a run
        with it on measures what the net can do WITHOUT terrain structure. It is applied to the
        RELIEF, not to the raw heightmap: relief() re-centres each map on its own median, so
        blurring first would leave exactly zero (verified -- a constant map has zero relief by
        construction) and would take the "is there a box somewhere at all" scalar down with it,
        collapsing this baseline into the per-wz mean field rather than sitting above it. Blurring
        after relief() keeps precisely that scalar -- the map's mean height above background, in
        wheel radii -- and destroys only WHERE the height is, which is the comparison section 7
        asks for."""
        e = self.command_mlp(command_features(wz))
        x = relief(heightmap, self.wheel_radius)
        if blur_terrain:
            x = x.mean(dim=(-2, -1), keepdim=True).expand_as(x)
        feat = self.conv_trunk(x, e)  # [B, C, H', W']

        offset = readout_offset(extent, heightmap.shape[-1], feat.shape[-1])
        grid = ((spawn_xy - offset) / (extent / 2.0)).unsqueeze(0).expand(feat.shape[0], -1, -1, -1)
        sampled = F.grid_sample(
            feat, grid, mode="bilinear", padding_mode="border", align_corners=False,
        )  # [B, C, G, G]
        return self.head(sampled).permute(0, 2, 3, 1)  # [B, G, G, 2]

    @torch.no_grad()
    def predict(
        self, heightmap: torch.Tensor, wz: torch.Tensor, spawn_xy: torch.Tensor,
        extent: float = DEFAULT_EXTENT, blur_terrain: bool = False,
    ) -> torch.Tensor:
        """Same signature as forward(), returning physical (e_pos m, e_rot rad) instead of model
        space. Inference only -- needs a target_transform, exactly like PoseErrorMLP.predict()."""
        if self.target_transform is None:
            raise RuntimeError(
                "predict() needs a target_transform to undo the log1p/standardize the net was "
                "trained in -- pass one to __init__ or assign .target_transform. Use forward() "
                "if you want the raw model-space output."
            )
        return self.target_transform.inverse(
            self(heightmap, wz, spawn_xy, extent, blur_terrain=blur_terrain)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-width", type=int, default=DEFAULT_BASE_WIDTH)
    parser.add_argument("--embed-dim", type=int, default=DEFAULT_EMBED_DIM)
    parser.add_argument("--extent", type=float, default=DEFAULT_EXTENT)
    args = parser.parse_args()

    net = GridPoseErrorNet(base_width=args.base_width, embed_dim=args.embed_dim)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"GridPoseErrorNet(base_width={args.base_width}, embed_dim={args.embed_dim}): {n_params} params")

    # --- receptive-field self-check: the formula's answer must match what the conv stack
    # ACTUALLY does under autograd, probed well away from the border so replicate-padding
    # boundary effects can't inflate the measured region. ---------------------------------------
    rf = receptive_field()
    print(f"[receptive field] formula: {rf} px = {rf * 0.10:.2f} m")
    H = 100
    x = torch.zeros(1, 1, H, H, requires_grad=True)
    e_probe = torch.zeros(1, args.embed_dim)
    feat = net.conv_trunk(x, e_probe)
    center = tuple(s // 2 for s in feat.shape[-2:])
    feat[0, 0, center[0], center[1]].backward()
    nonzero = (x.grad[0, 0].abs() > 0).nonzero()
    measured = int((nonzero[:, 0].max() - nonzero[:, 0].min()).item()) + 1
    assert measured == rf, f"measured RF {measured} px != formula RF {rf} px"
    print(f"[receptive field] autograd probe: {measured} px -- matches formula")

    # --- readout-geometry self-check: readout_offset() must equal where the trunk's feature
    # cells ACTUALLY sit, probed the same way as the RF above. Feature cell k draws on input
    # pixels centred at 4k, i.e. world -extent/2 + res*(4k + 0.5); the offset is how far the
    # lattice centre is from the map centre. --------------------------------------------------
    n_feat = feat.shape[-1]
    offset = readout_offset(args.extent, H, n_feat)
    resolution = args.extent / H
    probed = -args.extent / 2.0 + resolution * ((nonzero[:, 1].min() + nonzero[:, 1].max()) / 2.0 + 0.5)
    expected = offset + resolution * (H // n_feat) * (center[1] - (n_feat - 1) / 2.0)
    assert abs(float(probed) - expected) < 1e-6, (float(probed), expected)
    print(f"[readout] feature lattice offset: {offset:+.3f} m -- matches autograd probe "
          f"(cell {center[1]} at {float(probed):+.3f} m)")

    # --- flat-ground self-check: relief() of a constant heightmap must be exactly zero ---------
    flat = torch.full((2, H, H), 3.7)  # arbitrary constant elevation, any value should reduce to 0
    assert torch.allclose(relief(flat), torch.zeros(2, 1, H, H)), "flat ground must give zero relief"
    print("[relief] flat-ground self-check ok")

    # --- shape check + TargetTransform round-trip, mirroring learning/model.py's smoke test -----
    B, G = 4, 15
    heightmap = torch.randn(B, H, H)
    wz = torch.empty(B).uniform_(-1.0, 1.0)
    coords = torch.linspace(-3.5, 3.5, G)
    Y, X = torch.meshgrid(coords, coords, indexing="ij")
    spawn_xy = torch.stack([X, Y], dim=-1)  # [G, G, 2], same (x, y) order as generate_dataset

    y_hat = net(heightmap, wz, spawn_xy, extent=args.extent)
    assert y_hat.shape == (B, G, G, OUT_DIM), y_hat.shape
    print(f"[forward] heightmap {tuple(heightmap.shape)}, wz {tuple(wz.shape)} -> y_hat {tuple(y_hat.shape)}")

    # --- blur baseline self-check: blurring the RELIEF must keep the "is there a box" scalar
    # (design.md section 7), where blurring the raw heightmap would zero it out. ----------
    boxed = torch.zeros(1, H, H)
    boxed[0, 40:60, 40:60] = 0.7  # one 2 m box on flat ground
    blurred_relief = relief(boxed).mean(dim=(-2, -1))
    blurred_input = relief(boxed.mean(dim=(-2, -1), keepdim=True).expand_as(boxed).contiguous())
    assert blurred_relief.abs().item() > 1e-3, "relief-space blur lost the box-presence scalar"
    assert blurred_input.abs().max().item() == 0.0, "raw-input blur was expected to zero out"
    print(f"[blur] relief-space blur keeps box scalar {blurred_relief.item():.4f} wheel radii "
          f"(blurring the raw heightmap instead gives exactly 0)")
    assert not torch.allclose(
        net(heightmap, wz, spawn_xy, extent=args.extent, blur_terrain=True),
        net(heightmap, wz, spawn_xy, extent=args.extent),
    ), "blur_terrain=True changed nothing"

    y_phys = torch.rand(64, OUT_DIM) * 3.0  # stand-in physical (e_pos, e_rot), non-negative
    transform = TargetTransform.fit(y_phys)
    assert torch.allclose(transform.inverse(transform.forward(y_phys)), y_phys, atol=1e-5), "not invertible"
    assert (transform.inverse(torch.randn(64, OUT_DIM) * 5.0) >= 0).all(), "negative error escaped"

    net.target_transform = transform
    prediction = net.predict(heightmap, wz, spawn_xy, extent=args.extent)
    assert prediction.shape == (B, G, G, OUT_DIM) and (prediction >= 0).all()
    print(f"[predict] range: [{prediction.min():.4f}, {prediction.max():.4f}] ({', '.join(TARGET_NAMES)})")
    print("all self-checks ok")
