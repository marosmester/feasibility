"""`WindowDivergenceNet`: patch [B, 1, 24, 36] + one MPPI window's command (v_mean, v_slope,
wz_mean, wz_slope) -> the twin-vs-ostrich end-pose error (e_pos, e_rot) or (e_pos, e_roll,
e_pitch, e_yaw) -- design.md section 1.

`lattice_learning`'s `ArcDivergenceNet` as it is: the command-free terrain trunk, the command
encoder, the FiLM head and `TargetTransform`. Only two things change, both through the class hooks
that net exposes (`COMMAND_COLUMNS` / `N_CMD_FEATURES` / `encode_command`):

  * the command is `command.encode`'s four numbers, and the encoder is fed 7 features of them
    (`window_command_features`): lattice's value / |value| / sign split on the yaw rate, plus
    `wz_slope * sign(wz_mean)`, which says whether a turn is tightening or easing and, unlike the
    raw slope, survives the mirror;
  * the patch is `spawn_sampling.PATCH_SPEC` (x in [-1.5, 3.0] m, 24 x 36), so the trunk ends at
    6 x 9 and the geometry layer's kernel is (6, 9).

The feature scales are fixed by MPPI's wheel box, not fitted: `V_SCALE` is `v_max` (1.4 m/s) and
`WZ_SCALE` the ideal yaw rate of a full-spread spin (`v_max / half_track`). A checkpoint therefore
needs nothing but its weights and `TargetTransform`.

The y-mirror (a left/right flip of the patch) negates `wz_mean` and `wz_slope` (`mirror_command`),
which is train.py's augmentation.

Usage:
    python src/feasibility/mppi_learning/model.py     # shape / kernel / causality / mirror checks
"""
from __future__ import annotations

import argparse
import io

import torch

from feasibility.lattice_learning.model import ArcDivergenceNet
from feasibility.lattice_learning.model import HEAD_FUSIONS
from feasibility.lattice_learning.model import LABEL_MODES
from feasibility.lattice_learning.model import LABEL_NAMES
from feasibility.lattice_learning.model import Normalizer
from feasibility.lattice_learning.model import TargetTransform
from feasibility.lattice_learning.model import total_stride
from feasibility.lattice_learning.model import trunk_out_size
from feasibility.lattice_learning.patch import N_CHANNELS
from feasibility.lattice_learning.patch import PatchSpec
from feasibility.mppi_learning.command import CommandSpec
from feasibility.mppi_learning.command import HALF_TRACK
from feasibility.mppi_learning.spawn_sampling import PATCH_SPEC

__all__ = [
    "COMMAND_COLUMNS", "LABEL_MODES", "LABEL_NAMES", "MIRROR_SIGN", "Normalizer", "TargetTransform",
    "V_SCALE", "WZ_SCALE", "WindowDivergenceNet", "mirror_command", "window_command_features",
]

COMMAND_MODE = "mean_slope"
COMMAND_COLUMNS: tuple[str, ...] = ("v_mean", "v_slope", "wz_mean", "wz_slope")  # command.encode's order
MIRROR_SIGN: tuple[float, ...] = (1.0, 1.0, -1.0, -1.0)
N_CMD_FEATURES = 7

V_SCALE: float = CommandSpec().v_max  # m/s, 1.4
WZ_SCALE: float = V_SCALE / HALF_TRACK  # rad/s, wheels at -wmax / +wmax with no slip


def window_command_features(command: torch.Tensor) -> torch.Tensor:
    """[B, 4] (v_mean, v_slope, wz_mean, wz_slope) -> [B, 7] = (v_mean/V, v_slope/V, wz_mean/W,
    |wz_mean|/W, sign(wz_mean), wz_slope/W, wz_slope * sign(wz_mean)/W)."""
    v = command[:, :2] / V_SCALE
    wz, wz_slope = command[:, 2:3] / WZ_SCALE, command[:, 3:4] / WZ_SCALE
    sign = torch.sign(wz)
    return torch.cat([v, wz, wz.abs(), sign, wz_slope, wz_slope * sign], dim=-1)


def mirror_command(command: torch.Tensor) -> torch.Tensor:
    """[B, 4] command -> its y-mirror image: the yaw rate and its slope negated."""
    return command * command.new_tensor(MIRROR_SIGN)


class WindowDivergenceNet(ArcDivergenceNet):
    """`ArcDivergenceNet` with the window command and `PATCH_SPEC`; everything else inherited."""

    COMMAND_COLUMNS = {COMMAND_MODE: COMMAND_COLUMNS}
    N_CMD_FEATURES = {COMMAND_MODE: N_CMD_FEATURES}

    def __init__(self, *, command_mode: str = COMMAND_MODE, patch_spec: PatchSpec | None = None, **kwargs: object) -> None:
        super().__init__(command_mode=command_mode, patch_spec=patch_spec or PATCH_SPEC, **kwargs)

    def encode_command(self, command: torch.Tensor) -> torch.Tensor:
        return window_command_features(command)


def random_commands(n: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """[n, 4] commands spread over the box, for the self-test: v in [0, V_SCALE], wz in
    [-WZ_SCALE, WZ_SCALE], slopes of either sign."""
    u = torch.rand(n, 4, generator=generator)
    return torch.stack([u[:, 0] * V_SCALE, (u[:, 1] - 0.5) * V_SCALE, (2 * u[:, 2] - 1) * WZ_SCALE,
                        (u[:, 3] - 0.5) * WZ_SCALE], dim=-1)


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
    command[0, 2:] = 0.0  # a straight window: sign(0) = 0

    for label_mode in LABEL_MODES:
        net = WindowDivergenceNet(base_width=args.base_width, embed_dim=args.embed_dim,
                                  head_fusion=args.head_fusion, label_mode=label_mode)
        assert net(patch, command).shape == (B, len(LABEL_NAMES[label_mode]))
    n_params = sum(p.numel() for p in net.parameters())
    assert (spec.ny, spec.nx) == (24, 36) and net.geometry.kernel_size == (6, 9), net.geometry.kernel_size
    assert (trunk_out_size(spec.ny), trunk_out_size(spec.nx)) == (6, 9) and net.command_encoder[0].in_features == 7
    print(f"[forward] WindowDivergenceNet ({n_params} params): patch {tuple(patch.shape)} -> trunk 6x9, "
          f"geometry kernel {net.geometry.kernel_size}, y [B, K] in both label modes")

    # a command of the wrong width is refused, not broadcast
    try:
        net(patch, command[:, :2])
        raise AssertionError("a [B, 2] command was accepted")
    except ValueError:
        pass

    # the trunk never sees the command
    probe = command.clone().requires_grad_(True)
    grad = torch.autograd.grad(net.terrain_code(patch).sum(), probe, allow_unused=True)[0]
    assert grad is None or torch.all(grad == 0)
    print("[causality] terrain_code(patch) has no gradient w.r.t. the command; a [B, 2] command raises")

    # patch <-> map equivalence on this patch shape: an interior crop's code equals the map's
    crop, offset = 64, 40
    big = torch.randn(1, N_CHANNELS, 2 * offset + crop + 20, 2 * offset + crop + 20)
    with torch.no_grad():
        code_big, code_crop = net.terrain_code(big)[0], net.terrain_code(big[:, :, offset:offset + crop, offset:offset + crop])[0]
    shift = offset // total_stride()
    cr, cc = code_crop.shape[-2] // 2, code_crop.shape[-1] // 2
    err = (code_big[:, shift + cr, shift + cc] - code_crop[:, cr, cc]).abs().max().item()
    assert err < 1e-4, err
    print(f"[patch<->map] interior crop vs big-map code exact to {err:.1e}")

    # mirror: the features of a mirrored command are the originals with exactly the signed ones negated
    f, fm = window_command_features(command), window_command_features(mirror_command(command))
    signed = torch.tensor([False, False, True, False, True, True, False])
    assert torch.allclose(fm[:, signed], -f[:, signed]) and torch.allclose(fm[:, ~signed], f[:, ~signed])
    assert torch.equal(mirror_command(mirror_command(command)), command)
    assert torch.equal(f[0, 2:], torch.zeros(5)), f[0]
    print("[mirror] mirror_command negates wz_mean/wz_slope; |wz|, v and the tightening term are invariant")

    # physical output is non-negative, and a checkpoint round-trips exactly
    net.eval()
    net.target_transform = TargetTransform.fit(torch.rand(64, len(net.target_names)) * 2.0)
    prediction = net.predict(patch, command)
    assert (prediction >= 0).all()
    buffer = io.BytesIO()
    torch.save(net.state_dict(), buffer)
    buffer.seek(0)
    again = WindowDivergenceNet(base_width=args.base_width, embed_dim=args.embed_dim,
                                head_fusion=args.head_fusion, label_mode=net.label_mode,
                                target_transform=net.target_transform).eval()
    again.load_state_dict(torch.load(buffer))
    assert torch.equal(again.predict(patch, command), prediction)
    print(f"[predict] non-negative, range [{prediction.min():.3f}, {prediction.max():.3f}]; state_dict round-trip exact")

    concat = WindowDivergenceNet(base_width=args.base_width, embed_dim=args.embed_dim, head_fusion="concat")
    assert concat(patch, command).shape == (B, 2)
    print("[ablation] head_fusion=concat builds and runs")
    print("all self-checks ok")
