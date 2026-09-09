"""`GridDivergenceNet`: the per-cell-command architecture described in design.md -- read that
file first, this module is the literal implementation of its sections 3 (grid geometry), 4
(preprocessing) and 5 (network). Consumes

    heightmap [B, 81, 81]  (absolute world z, m)   +   wz [B, 15, 15]  (rad/s, ONE PER CELL)
    -->  y_hat [B, 15, 15, 2]  in TargetTransform (log1p/standardized) space

The two things that differ from grid_learning/model.py's `GridPoseErrorNet`:

  1. The command is a FIELD, not a scalar, so it may only be consumed by 1x1 operations
     (design.md section 2). The label at cell (i, j) came from one simulated trial that saw only
     `wz[i, j]`; any kernel > 1 applied after the command is injected would let cell A's
     prediction depend on cell B's command, which is a real deployment failure (a planner asking
     for a CONSTANT field would then be querying a distribution the net never saw). Hence: a
     completely unconditioned terrain trunk, and the command entering only in the head. The
     constraint is not asserted by argument -- the self-check below measures
     `d y_hat[p,q] / d wz[i,j]` under autograd and requires it to be exactly zero off-diagonal.

  2. There is no `F.grid_sample`. The input grid is odd and origin-centred and the trunk's total
     stride is exactly the spawn-lattice pitch in pixels (0.5 / 0.125 = 4), so the label lattice
     is an integer centre-crop of the feature map -- see utils.grid_coords_centered and design.md
     section 3. `spawn_xy` is therefore no longer a forward() argument at all; it becomes an
     assertion (`check_lattice_alignment`), run once at setup rather than interpolated with every
     batch. v1's `readout_offset()` has no counterpart here because there is nothing to correct.

Because the trunk is control-free it computes `N(H) -> c_hat(s)` from
docs/learned_divergence_penalty.md directly: one trunk pass gives the terrain code at every pose,
and the head can then be evaluated for as many commands per cell as the planner wants at 1x1
cost. That amortization is the reason for the reframing (design.md section 1a).

TARGET_NAMES / Normalizer / TargetTransform live HERE rather than in custom_dataset.py (the
opposite of grid_learning's arrangement) so this module has no sibling imports beyond utils.py's
pure grid arithmetic: the target space is a property of what the net regresses, and keeping it
next to the net means model.py can be run and checked before any dataset exists. custom_dataset.py
imports them from here.

WHEEL_RADIUS and WZ_MAX are restated locally rather than imported from
`examples.helhest_junior.common` / a sibling generator, which would drag ostrich/warp into every
training run for the sake of two floats -- the same "restate rather than import" call
grid_learning makes, and design.md section 11 keeps.

CLI parameters:
    --base-width INT      trunk channel unit; layer widths are 1x/2x/3x this (default: 32)
    --embed-dim INT       per-cell command-embedding width, also the FiLM input width (default: 32)
    --head-fusion STR     film | concat -- design.md section 5d's ablation (default: film)
    --cells INT           odd heightmap cells per side (default: 81)

Usage:
    python src/feasibility/grid_learning_2/model.py     # shape / RF / alignment / causality checks
"""
from __future__ import annotations

import argparse
import dataclasses
import math

import numpy as np
import torch
import torch.nn as nn

from feasibility.grid_learning_2.utils import DEFAULT_N_CELLS
from feasibility.grid_learning_2.utils import DEFAULT_RESOLUTION
from feasibility.grid_learning_2.utils import conv_out_size
from feasibility.grid_learning_2.utils import crop_offset
from feasibility.grid_learning_2.utils import downsampled_coords
from feasibility.grid_learning_2.utils import grid_coords_centered
from feasibility.grid_learning_2.utils import spawn_lattice

WHEEL_RADIUS = 0.35  # m -- HelhestJuniorConfig.WHEEL_RADIUS, restated (see module docstring)
WZ_MAX = 1.0  # rad/s -- the magnitude of generate_dataset's WZ_RANGE, restated likewise

TARGET_NAMES = ("e_pos", "e_rot")
OUT_DIM = len(TARGET_NAMES)  # 2

DEFAULT_BASE_WIDTH = 32
DEFAULT_EMBED_DIM = 32

# How many numbers the command field carries per cell -- (wz, |wz|) for now. Kept as a name
# rather than a literal 2 because design.md section 4b wants adding forward velocity `v` later to
# be a data-schema change, not an architecture change. V_DRIVE is 0 in every trial this dataset
# contains, so there is nothing else to carry yet.
N_CMD_FEATURES = 2

# (channel multiplier of base_width, stride) per trunk block -- design.md section 5b's table,
# literally: 7 blocks, two stride-2 downsamples, channels 1x -> 2x -> 3x. This list IS the
# receptive-field derivation: it is chosen to land the final RF at 27 px = 3.375 m, i.e. 1.69 m
# of radius against the rear wheel's 1.45 m rim sweep (1.101 m axle offset + 0.35 m wheel), and
# the total stride at exactly 4 = the 0.5 m lattice pitch in 0.125 m pixels. Both numbers break if
# it is edited, so treat it as load-bearing, not a free tuning knob.
TRUNK_PLAN: tuple[tuple[int, int], ...] = (
    (1, 1), (1, 1), (2, 2), (2, 1), (2, 1), (3, 2), (3, 1),
)

# Which blocks (0-based indices into TRUNK_PLAN) are tapped for the multi-scale readout --
# design.md section 5c. Block 5 (index 4) sees 1.875 m, block 7 (index 6) sees 3.375 m. A
# control-free trunk cannot know which sweep radius matters (that is what |wz| decides), so it
# hands the command-aware head both and lets the head select. The last block must be included.
READOUT_BLOCKS: tuple[int, ...] = (4, 6)

HEAD_FUSIONS = ("film", "concat")


def receptive_field(plan: tuple[tuple[int, int], ...] = TRUNK_PLAN, kernel: int = 3) -> int:
    """Receptive field in input pixels of the last block of `plan`, via the standard conv-stack
    recurrence RF += (kernel-1)*jump, jump *= stride -- the arithmetic behind design.md section
    5b's RF column. Used both to report the number and, in the self-check below, to verify it
    against an actual autograd probe rather than trusting the formula blindly (that probe is what
    caught GroupNorm silently making v1's RF map-global; see ChannelLayerNorm)."""
    rf, jump = 1, 1
    for _, stride in plan:
        rf += (kernel - 1) * jump
        jump *= stride
    return rf


def total_stride(plan: tuple[tuple[int, int], ...] = TRUNK_PLAN) -> int:
    """Product of the trunk's strides -- must equal the spawn-lattice pitch in input pixels
    (0.5 m / 0.125 m = 4) for the readout to be an integer crop."""
    return math.prod(stride for _, stride in plan)


def trunk_sizes(n_input: int, plan: tuple[tuple[int, int], ...] = TRUNK_PLAN) -> list[int]:
    """Spatial size after each block, so design.md section 3c's 81 -> 41 -> 21 chain is derived
    rather than assumed anywhere a shape is needed."""
    sizes, n = [], n_input
    for _, stride in plan:
        n = conv_out_size(n, stride)
        sizes.append(n)
    return sizes


def readout_slice_strides(
    plan: tuple[tuple[int, int], ...] = TRUNK_PLAN, taps: tuple[int, ...] = READOUT_BLOCKS
) -> tuple[int, ...]:
    """Per-tap stride needed to bring that tap's feature map down to the FINAL block's lattice:
    the product of the strides of every block AFTER the tap. For the defaults, block 5's 41x41
    map needs `[::2]` and block 7's 21x21 map needs `[::1]`.

    Sliced, never pooled (design.md section 5c): `[::2]` keeps indices 0, 2, ..., 40, which are
    exactly co-located with the 21-cell lattice, whereas `avg_pool2d(2)` would average adjacent
    cells and land half a cell (0.125 m) off -- reintroducing precisely the misregistration this
    whole design exists to remove."""
    if taps[-1] != len(plan) - 1:
        raise ValueError(f"the last block must be tapped, got taps={taps} for {len(plan)} blocks")
    return tuple(math.prod(stride for _, stride in plan[t + 1:]) for t in taps)


def relief(heightmap: torch.Tensor, wheel_radius: float = WHEEL_RADIUS) -> torch.Tensor:
    """[B, H, W] absolute world z (m) -> [B, 1, H, W] height relative to this SAMPLE's own median,
    scaled by wheel radius -- design.md section 4a, unchanged from v1. The median is one scalar
    per sample (each map has its own background elevation) so it recenters every pixel identically
    and leaks no position-specific information; a 0.70 m box reads as 2.0 wheel radii and flat
    ground reads as exactly zero, checked below."""
    baseline = heightmap.flatten(1).median(dim=1).values.view(-1, 1, 1, 1)
    return (heightmap.unsqueeze(1) - baseline) / wheel_radius


def command_features(wz: torch.Tensor, wz_max: float = WZ_MAX) -> torch.Tensor:
    """[B, G, G] rad/s -> [B, N_CMD_FEATURES, G, G] = (wz/wz_max, |wz|/wz_max) -- design.md
    section 4b. Same two numbers as v1 and for the same reason (|wz| sets how far the
    fixed-duration turn sweeps, the sign sets which way the rear wheel goes), now carried per cell
    as a channel-first field so every downstream op can be a Conv1x1."""
    return torch.stack([wz / wz_max, wz.abs() / wz_max], dim=1)


def lattice_coords(
    n_input: int = DEFAULT_N_CELLS,
    resolution: float = DEFAULT_RESOLUTION,
    n_lattice: int = 15,
    plan: tuple[tuple[int, int], ...] = TRUNK_PLAN,
) -> np.ndarray:
    """[n_lattice] world coordinate of each cropped readout cell along one axis -- what the
    network's output cells actually correspond to on the map. This is the quantity v1 had to get
    right with `readout_offset()` + `grid_sample`; here it is pure integer arithmetic and can be
    compared against a dataset's `spawn_xy` for exact equality."""
    coords = grid_coords_centered(n_input, resolution)
    n_feat = trunk_sizes(n_input, plan)[-1]
    feat_coords = downsampled_coords(coords, total_stride(plan), n_feat)
    off = crop_offset(n_feat, n_lattice)
    return feat_coords[off:off + n_lattice]


def check_lattice_alignment(
    spawn_xy: torch.Tensor | np.ndarray,
    n_input: int = DEFAULT_N_CELLS,
    resolution: float = DEFAULT_RESOLUTION,
    plan: tuple[tuple[int, int], ...] = TRUNK_PLAN,
    atol: float = 1e-6,
) -> float:
    """Assert that the cropped feature lattice sits exactly on `spawn_xy` (the dataset file's
    [G, G, 2] world coordinates), returning the max abs difference.

    This replaces v1's readout: there, `spawn_xy` was fed to `F.grid_sample` as interpolation
    coordinates, so a mismatch was silently absorbed as a resampling. Here it is an equality that
    either holds or does not, and calling this once at setup is enough -- the readout is fixed
    integer geometry, so it cannot drift batch to batch. Call it from the dataset/training setup
    (and it is called by the self-check below); forward() deliberately does not, since re-checking
    constant geometry on every batch is pure overhead."""
    spawn = np.asarray(spawn_xy.detach().cpu() if torch.is_tensor(spawn_xy) else spawn_xy)
    if spawn.ndim != 3 or spawn.shape[-1] != 2 or spawn.shape[0] != spawn.shape[1]:
        raise ValueError(f"spawn_xy must be [G, G, 2] square, got {spawn.shape}")
    coords = lattice_coords(n_input, resolution, spawn.shape[0], plan)
    X, Y = np.meshgrid(coords, coords)  # row = y, col = x, matching utils.spawn_lattice
    expected = np.stack([X, Y], axis=-1)
    err = float(np.abs(expected - spawn).max())
    if err > atol:
        raise AssertionError(
            f"readout lattice does not coincide with spawn_xy (max |diff| = {err:.3e} m > {atol}). "
            f"The heightmap grid ({n_input} cells @ {resolution} m) and the label lattice "
            f"({spawn.shape[0]} cells) are inconsistent -- see design.md section 3c."
        )
    return err


@dataclasses.dataclass(frozen=True)
class Normalizer:
    """Per-column (x - mean) / std, std floored so a constant column never divides by zero --
    restated from grid_learning/learning's Normalizer rather than imported (design.md section 11's
    dependency stance)."""

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
    """Physical (e_pos m, e_rot rad) <-> log1p/standardized model space -- design.md section 6,
    unchanged from v1 in substance.

    Fit on TRAIN rows' VALID (mask == True) cells only: masked cells are exact zeros by
    construction, and folding thousands of spurious zeros into the mean/std would skew the
    standardization toward "everything is zero" on a field that is mostly flat ground anyway.
    The inverse clamps at 0 before expm1 so a physically impossible negative error cannot
    escape."""

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


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the CHANNEL dimension only, applied independently at every spatial position
    -- the ConvNeXt-style "LayerNorm2d". Carried over from v1 verbatim, and for the same
    non-negotiable reason: GroupNorm/InstanceNorm/BatchNorm compute their statistics over
    (channels, H, W) JOINTLY, so their output at one position depends on every other position,
    silently making each block's receptive field the WHOLE map regardless of kernel and stride.
    v1's receptive-field probe measured 100 px instead of 35 the first time this block used
    GroupNorm. A per-position channel norm has no such pooling, so the conv geometry alone
    determines the RF -- which is exactly what design.md section 5b's table claims and the
    self-check below verifies."""

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvBlock(nn.Module):
    """One unconditioned trunk block: conv3 -> ChannelLayerNorm -> SiLU.

    This is v1's FiLMBlock with the conditioning removed, which design.md section 2 shows is
    forced rather than chosen -- a per-cell command cannot legally enter anything with a spatial
    footprint. (Note the asymmetry: v1's SCALAR command was a legal trunk input precisely because
    there was only one of it, so FiLM's per-channel broadcast added no cross-position dependence.)

    `replicate` padding, never zeros: zero padding stamps a distinctive constant at the border, a
    route to encoding absolute position, whereas replicate extends the edge height outward --
    which is the SAME boundary condition HeightMapReader.sample uses (it clamps to the nearest
    edge cell rather than raising), so it is the physically consistent choice, not a hack."""

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


class GridDivergenceNet(nn.Module):
    """heightmap [B, H, H] + wz field [B, G, G] -> divergence field [B, G, G, 2] in
    TargetTransform (log1p/standardized) space -- design.md section 5a.

    Note what is NOT in the signature, compared with v1's `GridPoseErrorNet.forward`: no
    `spawn_xy`, no `extent`. The readout is fixed integer geometry (section 3), so the lattice's
    world coordinates are a property of the architecture, not something to be looked up per batch;
    `check_lattice_alignment()` verifies once that the dataset agrees. G is read from the command
    field, which is the tensor that actually carries the per-cell structure.

    `target_transform` IS stored as a plain attribute (not a buffer): it is fitted data, not a
    learned parameter, so it stays out of state_dict() and must be saved alongside a checkpoint
    and re-attached on load for predict() to work -- same contract as v1 and learning's
    PoseErrorMLP."""

    def __init__(
        self,
        *,
        base_width: int = DEFAULT_BASE_WIDTH,
        embed_dim: int = DEFAULT_EMBED_DIM,
        head_fusion: str = "film",
        wheel_radius: float = WHEEL_RADIUS,
        target_transform: TargetTransform | None = None,
    ) -> None:
        super().__init__()
        if head_fusion not in HEAD_FUSIONS:
            raise ValueError(f"head_fusion must be one of {HEAD_FUSIONS}, got {head_fusion!r}")
        self.wheel_radius = wheel_radius
        self.head_fusion = head_fusion
        self.embed_dim = embed_dim
        self.target_transform = target_transform

        blocks = []
        in_channels = 1  # single-channel relief map
        for mult, stride in TRUNK_PLAN:
            out_channels = base_width * mult
            blocks.append(ConvBlock(in_channels, out_channels, stride))
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)

        self.slice_strides = readout_slice_strides()
        readout_channels = sum(TRUNK_PLAN[t][0] * base_width for t in READOUT_BLOCKS)

        # Per-cell command encoder: Conv1x1 only, so cell (i, j)'s embedding depends on wz[i, j]
        # and nothing else. Written as convs rather than Linears purely so the field stays
        # channel-first end to end.
        self.command_encoder = nn.Sequential(
            nn.Conv2d(N_CMD_FEATURES, embed_dim, kernel_size=1), nn.SiLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
        )

        # design.md section 5d: FiLM supplies the MULTIPLICATIVE interaction (how much terrain
        # feature k matters is gated by how far the turn sweeps), the MLP after it supplies
        # general nonlinear mixing, and neither subsumes the other in practice. A concat head's
        # first layer is additive in the command -- W_c @ c + W_e @ e -- so every multiplicative
        # effect has to be rebuilt out of downstream nonlinearities, and with 160 terrain channels
        # against one command the command's gradient is easily swamped early (predicting the
        # terrain-marginal mean is a real local optimum here). `--head-fusion concat` is the
        # honest control for that claim rather than an argument.
        head_width = 4 * base_width
        if head_fusion == "film":
            self.film = nn.Conv2d(embed_dim, 2 * readout_channels, kernel_size=1)
            head_in = readout_channels
            head_layers: list[nn.Module] = [nn.SiLU()]
        else:
            self.film = None
            head_in = readout_channels + embed_dim
            head_layers = []
        self.head = nn.Sequential(
            *head_layers,
            nn.Conv2d(head_in, head_width, kernel_size=1), nn.SiLU(),
            nn.Conv2d(head_width, head_width, kernel_size=1), nn.SiLU(),
            nn.Conv2d(head_width, OUT_DIM, kernel_size=1),
        )

    def terrain_code(self, heightmap: torch.Tensor, blur_terrain: bool = False) -> torch.Tensor:
        """heightmap [B, H, H] -> the control-free terrain code c(s) on the FULL feature lattice,
        [B, readout_channels, n_feat, n_feat] -- `N(H) -> c_hat` from
        docs/learned_divergence_penalty.md, before the crop.

        Split out of forward() so (a) the receptive-field self-check can probe the conv stack in
        isolation and (b) a planner can compute it ONCE per map and then evaluate the head for
        many commands per cell at 1x1 cost, which is the amortization design.md section 1a exists
        for.

        `blur_terrain` is design.md section 8's third baseline: flatten the terrain to one number
        per sample, keeping the architecture and the command path intact, so a run with it on
        measures what the net can do WITHOUT terrain structure. It is applied to the RELIEF, not
        the raw heightmap -- relief() re-centres each map on its own median, so blurring first
        would give exactly zero and take the "is there a box somewhere at all" scalar down with
        it, collapsing this baseline into the per-wz mean field instead of sitting above it."""
        x = relief(heightmap, self.wheel_radius)
        if blur_terrain:
            x = x.mean(dim=(-2, -1), keepdim=True).expand_as(x)

        taps: list[torch.Tensor] = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in READOUT_BLOCKS:
                taps.append(x)

        n_final = taps[-1].shape[-1]
        sliced = []
        for tap, stride in zip(taps, self.slice_strides):
            n_tap = tap.shape[-1]
            assert n_tap - 1 == stride * (n_final - 1), (
                f"readout tap of size {n_tap} is not co-located with the {n_final}-cell lattice "
                f"at stride {stride} -- see readout_slice_strides()"
            )
            sliced.append(tap[..., ::stride, ::stride])
        return torch.cat(sliced, dim=1)

    def head_from_code(self, code: torch.Tensor, wz: torch.Tensor) -> torch.Tensor:
        """Cropped terrain code [B, C, G, G] + command field [B, G, G] -> [B, G, G, 2].

        EVERY operation here is 1x1, which is design.md section 2's constraint discharged
        architecturally rather than by convention. Exposed separately from forward() so the
        amortized query (one `terrain_code`, many commands) is a supported call, not a trick."""
        e = self.command_encoder(command_features(wz))  # [B, embed, G, G]
        if self.head_fusion == "film":
            gamma, beta = self.film(e).chunk(2, dim=1)
            h = code * (1.0 + gamma) + beta  # `1 + gamma`: identity modulation at init
        else:
            h = torch.cat([code, e], dim=1)
        return self.head(h).permute(0, 2, 3, 1)  # [B, G, G, 2]

    def forward(
        self, heightmap: torch.Tensor, wz: torch.Tensor, blur_terrain: bool = False
    ) -> torch.Tensor:
        """heightmap [B, H, H] (absolute world z, m), wz [B, G, G] (rad/s per lattice cell)
        -> [B, G, G, 2] in TargetTransform space."""
        if wz.ndim != 3 or wz.shape[-1] != wz.shape[-2]:
            raise ValueError(f"wz must be a square [B, G, G] command field, got {tuple(wz.shape)}")
        code = self.terrain_code(heightmap, blur_terrain=blur_terrain)
        off = crop_offset(code.shape[-1], wz.shape[-1])
        g = wz.shape[-1]
        return self.head_from_code(code[..., off:off + g, off:off + g], wz)

    @torch.no_grad()
    def predict(
        self, heightmap: torch.Tensor, wz: torch.Tensor, blur_terrain: bool = False
    ) -> torch.Tensor:
        """Same signature as forward(), returning physical (e_pos m, e_rot rad) instead of model
        space. Inference only -- needs a target_transform, exactly like v1's predict()."""
        if self.target_transform is None:
            raise RuntimeError(
                "predict() needs a target_transform to undo the log1p/standardize the net was "
                "trained in -- pass one to __init__ or assign .target_transform. Use forward() "
                "if you want the raw model-space output."
            )
        return self.target_transform.inverse(self(heightmap, wz, blur_terrain=blur_terrain))


def _grad_support(net: GridDivergenceNet, n_input: int, row: int, col: int) -> torch.Tensor:
    """[H, H] bool mask of which input pixels the FULL-lattice feature cell (row, col) depends on,
    measured by autograd rather than derived. Used for both the receptive-field check and the
    mirror-geometry check below."""
    x = torch.zeros(1, 1, n_input, n_input, requires_grad=True)
    feat = x
    for block in net.blocks:
        feat = block(feat)
    feat[0, 0, row, col].backward()
    assert x.grad is not None
    return x.grad[0, 0].abs() > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-width", type=int, default=DEFAULT_BASE_WIDTH)
    parser.add_argument("--embed-dim", type=int, default=DEFAULT_EMBED_DIM)
    parser.add_argument("--head-fusion", type=str, default="film", choices=HEAD_FUSIONS)
    parser.add_argument("--cells", type=int, default=DEFAULT_N_CELLS)
    args = parser.parse_args()

    H = args.cells
    net = GridDivergenceNet(
        base_width=args.base_width, embed_dim=args.embed_dim, head_fusion=args.head_fusion
    )
    n_params = sum(p.numel() for p in net.parameters())
    sizes = trunk_sizes(H)
    print(f"GridDivergenceNet(base_width={args.base_width}, embed_dim={args.embed_dim}, "
          f"head_fusion={args.head_fusion}): {n_params} params")
    print(f"[trunk] {H} -> {' -> '.join(str(s) for s in sizes)}, total stride {total_stride()}, "
          f"taps {READOUT_BLOCKS} at sizes {[sizes[t] for t in READOUT_BLOCKS]} "
          f"sliced by {readout_slice_strides()}")

    # --- receptive field: the formula's answer must match what the conv stack ACTUALLY does under
    # autograd, probed at the lattice centre so replicate-padding borders cannot inflate it. This
    # is the check that catches a normalisation layer pooling over (H, W). --------------------
    rf = receptive_field()
    n_feat = sizes[-1]
    support = _grad_support(net, H, n_feat // 2, n_feat // 2)
    rows = support.any(dim=1).nonzero()
    measured = int(rows.max() - rows.min()) + 1
    assert measured == rf, f"measured RF {measured} px != formula RF {rf} px"
    print(f"[receptive field] formula {rf} px = {rf * DEFAULT_RESOLUTION:.3f} m, autograd probe "
          f"{measured} px -- match (rim sweep needs 2 x 1.45 m = 2.90 m)")

    # --- readout alignment: the cropped lattice must BE spawn_xy, to 1e-6. This replaces v1's
    # readout_offset probe; there it was a correction, here it is an equality. -----------------
    lattice = spawn_lattice()
    err = check_lattice_alignment(lattice, n_input=H, resolution=DEFAULT_RESOLUTION)
    coords = lattice_coords(H, DEFAULT_RESOLUTION, lattice.shape[0])
    print(f"[readout] crop [{crop_offset(n_feat, lattice.shape[0])}:...] -> {coords[0]:+.3f} .. "
          f"{coords[-1]:+.3f} m, max |diff| vs spawn_xy = {err:.2e} m (no interpolation)")

    B, G = 4, lattice.shape[0]
    heightmap = torch.randn(B, H, H)
    wz = torch.empty(B, G, G).uniform_(-1.0, 1.0)
    y_hat = net(heightmap, wz)
    assert y_hat.shape == (B, G, G, OUT_DIM), y_hat.shape
    print(f"[forward] heightmap {tuple(heightmap.shape)}, wz {tuple(wz.shape)} "
          f"-> y_hat {tuple(y_hat.shape)}")

    # --- per-cell command causality (design.md section 2), the constraint this whole architecture
    # exists to satisfy: perturbing wz[i, j] must not move ANY other cell's prediction. Exact,
    # not approximate -- every op after the command is injected is 1x1. -------------------------
    probe_hm = torch.randn(1, H, H)
    probe_wz = torch.empty(1, G, G).uniform_(-1.0, 1.0).requires_grad_(True)
    p, q = 4, 11  # an arbitrary off-centre output cell
    net(probe_hm, probe_wz)[0, p, q].sum().backward()
    grad = probe_wz.grad[0]
    assert grad[p, q].abs() > 0, "the cell's own command had no influence at all"
    off_diagonal = grad.clone()
    off_diagonal[p, q] = 0.0
    assert (off_diagonal == 0).all(), (
        f"command leaked across cells: max |d y_hat[{p},{q}] / d wz[i,j]| = "
        f"{off_diagonal.abs().max():.3e} over {int((off_diagonal != 0).sum())} other cells"
    )
    print(f"[causality] d y_hat[{p},{q}] / d wz[i,j] is exactly 0 for all {G * G - 1} other cells")

    # --- mirror GEOMETRY (design.md section 7c). What is exact at initialisation is the
    # REGISTRATION, not the network's response: a conv stack with random weights is not
    # reflection-equivariant, and only training on the augmentation makes f(flip(h), -flip(wz))
    # approach flip(f(h, wz)). So assert the two things that ARE exact -- the lattice is
    # antisymmetric about y = 0, and output row p and output row G-1-p draw on exactly mirrored
    # input rows -- and merely REPORT the model-space equivariance for reference. ---------------
    assert np.allclose(coords, -coords[::-1], atol=1e-12), "readout lattice is not symmetric about 0"
    off = crop_offset(n_feat, G)
    supp_a = _grad_support(net, H, off + 3, off + 6)
    supp_b = _grad_support(net, H, off + (G - 1 - 3), off + 6)
    assert torch.equal(supp_b, torch.flip(supp_a, dims=[0])), (
        "output rows p and G-1-p do not draw on mirrored input rows -- the crop is off-centre"
    )
    net.eval()
    with torch.no_grad():
        diff = (
            torch.flip(net(heightmap, wz), dims=[-3])
            - net(torch.flip(heightmap, dims=[-2]), -torch.flip(wz, dims=[-2]))
        ).abs().max().item()
    print(f"[mirror] lattice antisymmetric and receptive fields mirror exactly; model-space "
          f"equivariance at init: {diff:.3e} (expected O(1) until trained with the augmentation)")

    # --- flat ground -> exactly zero relief -----------------------------------------------------
    flat = torch.full((2, H, H), 3.7)  # arbitrary constant elevation, any value reduces to 0
    assert torch.allclose(relief(flat), torch.zeros(2, 1, H, H)), "flat ground must give zero relief"
    print("[relief] flat-ground self-check ok")

    # --- blur baseline: blurring the RELIEF keeps the "is there a box" scalar, where blurring the
    # raw heightmap would zero it out and collapse the baseline into the per-wz mean. -----------
    boxed = torch.zeros(1, H, H)
    boxed[0, 32:48, 32:48] = 0.7  # one 2 m box on flat ground
    blurred_relief = relief(boxed).mean(dim=(-2, -1))
    blurred_input = relief(boxed.mean(dim=(-2, -1), keepdim=True).expand_as(boxed).contiguous())
    assert blurred_relief.abs().item() > 1e-3, "relief-space blur lost the box-presence scalar"
    assert blurred_input.abs().max().item() == 0.0, "raw-input blur was expected to zero out"
    assert not torch.allclose(net(heightmap, wz, blur_terrain=True), net(heightmap, wz)), \
        "blur_terrain=True changed nothing"
    print(f"[blur] relief-space blur keeps box scalar {blurred_relief.item():.4f} wheel radii "
          f"(blurring the raw heightmap instead gives exactly 0)")

    # --- amortization: head_from_code() on a cached terrain code must equal a full forward, which
    # is what lets a planner price |A| commands per cell for one trunk pass (design.md 1a) ------
    with torch.no_grad():
        code = net.terrain_code(heightmap)
        cropped = code[..., off:off + G, off:off + G]
        assert torch.allclose(net.head_from_code(cropped, wz), net(heightmap, wz), atol=1e-6)
    print("[amortization] cached terrain_code + head_from_code == full forward")

    # --- TargetTransform round-trip and non-negativity ------------------------------------------
    y_phys = torch.rand(64, OUT_DIM) * 3.0  # stand-in physical (e_pos, e_rot), non-negative
    transform = TargetTransform.fit(y_phys)
    assert torch.allclose(transform.inverse(transform.forward(y_phys)), y_phys, atol=1e-5), "not invertible"
    assert (transform.inverse(torch.randn(64, OUT_DIM) * 5.0) >= 0).all(), "negative error escaped"

    net.target_transform = transform
    prediction = net.predict(heightmap, wz)
    assert prediction.shape == (B, G, G, OUT_DIM) and (prediction >= 0).all()
    print(f"[predict] range: [{prediction.min():.4f}, {prediction.max():.4f}] "
          f"({', '.join(TARGET_NAMES)})")

    # --- the concat ablation must at least build and run to the same shape ---------------------
    concat_net = GridDivergenceNet(
        base_width=args.base_width, embed_dim=args.embed_dim, head_fusion="concat"
    )
    assert concat_net(heightmap, wz).shape == (B, G, G, OUT_DIM)
    n_concat = sum(p.numel() for p in concat_net.parameters())
    print(f"[ablation] head_fusion=concat builds and runs: {n_concat} params "
          f"({n_params - n_concat:+d} vs film)")
    print("all self-checks ok")
