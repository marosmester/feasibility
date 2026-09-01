"""The pose-error regressor: a plain MLP over one comparator trial's inputs, predicting how far
the fast kinematic twin (helhest_stack) ends up from the accurate dynamics simulator (ostrich).

Consumes exactly what custom_dataset.PoseErrorDataset yields and nothing derived --
FEATURE_NAMES_RAW = (v, wz, x, y, yaw) in, TARGET_NAMES = (e_pos, e_rot) out:

    x [B, 5] --> Linear/LayerNorm/SiLU x3 @ 256 --> {head_e_pos, head_e_rot} --> y [B, 2]

`in_dim` is a constructor argument precisely because that 5 is not fixed: 6 under
yaw_encoding="sincos", and 578 in terrain-patch mode ((v, wz) plus a flattened 24x24 body-frame
patch -- see custom_dataset's module docstring and learning/terrain_patch.py). Pass
len(ds.FEATURE_NAMES) and the width is right in every case. At 578 inputs the first Linear
alone is 148k params against a few thousand training rows, so expect the patch configuration to
want --weight-decay / --patience where the 5-input one did not; a flattened patch through a
plain MLP is the deliberately simplest terrain encoding, not the most sample-efficient one.

Two design points worth stating, both measured on the outputs/dataset_box_*.h5 files:

* ONE trunk, TWO heads rather than two networks. e_pos and e_rot correlate 0.72-0.83 across
  every generated dataset, so they share almost all of their structure; splitting only at the
  last layer lets that structure be learned once.
* The model works in log1p space on STANDARDIZED targets, not in metres/radians. e_pos and
  e_rot are non-negative and right-skewed (skew 1.5-3.0 -- a handful of obstacle collisions sit
  far out in the tail), so a plain MSE on raw values lets those few rows dominate the gradient
  and lets the network predict physically impossible negative errors. TargetTransform owns that
  change of variables and its exact inverse; fit it on TRAIN rows only, for the same reason
  custom_dataset.make_dataloaders fits its Normalizers on train_subset.indices.

No optimizer, no scheduler, no training loop -- this module is architecture only. The trainer
constructs its own optimizer, matching how ostrich's PPOTrainer takes one as an argument.

CLI parameters:
    --in-dim INT    input columns (default: 5)
    --hidden INT    trunk width (default: 256)
    --depth INT     trunk layers (default: 3)

Usage:
    python src/feasibility/learning/model.py            # shape + round-trip smoke check
    python -c "
    import torch
    from feasibility.learning.custom_dataset import PoseErrorDataset
    from feasibility.learning.model import PoseErrorMLP, TargetTransform
    ds = PoseErrorDataset('outputs/dataset_box_h070cm_n128.h5')
    net = PoseErrorMLP(len(ds.FEATURE_NAMES), target_transform=TargetTransform.fit(ds.y))
    print(net.predict(ds.x[:4]))
    "
"""
from __future__ import annotations

import argparse
import dataclasses

import torch
import torch.nn as nn

from feasibility.learning.custom_dataset import FEATURE_NAMES_RAW
from feasibility.learning.custom_dataset import Normalizer
from feasibility.learning.custom_dataset import TARGET_NAMES

DEFAULT_IN_DIM = len(FEATURE_NAMES_RAW)  # 5 -- (v, wz, x, y, yaw), raw yaw, no sin/cos encoding
DEFAULT_HIDDEN = 256  # units per trunk layer
DEFAULT_DEPTH = 3  # trunk layers; ~135k params at the defaults, comfortable at n in the 1000s
OUT_DIM = len(TARGET_NAMES)  # 2 -- (e_pos, e_rot), one head each


@dataclasses.dataclass(frozen=True)
class TargetTransform:
    """The change of variables between physical targets and what the network regresses:

        forward:  y (m, rad) --log1p--> standardize --> what the loss is computed on
        inverse:  network output --unstandardize--> clamp at 0 --expm1--> y (m, rad)

    Wraps custom_dataset.Normalizer rather than re-deriving the mean/std arithmetic -- it
    already floors std, so a target column that happens to be constant can't divide by zero.

    The clamp is what actually enforces non-negativity, not expm1: expm1 maps R -> (-1, inf), so
    an unclamped inverse of a confidently-wrong prediction would hand back a negative distance.
    Since e_pos >= 0 and e_rot in [0, pi] by construction (see pose_error.se3_error), log1p of a
    real target is always >= 0, and clamping there projects onto the valid range. Round-tripping
    a genuine target is unaffected -- the clamp is a no-op on anything forward() produced.

    inverse() is for reporting and evaluation in physical units; compute the training loss in
    model space, where the clamp's zero gradient can't mask an error."""

    normalizer: Normalizer

    @classmethod
    def fit(cls, y: torch.Tensor) -> "TargetTransform":
        """Fit on [n, 2] physical targets -- TRAIN rows only, never the full set."""
        return cls(normalizer=Normalizer.from_tensor(torch.log1p(y)))

    def _on(self, device: torch.device) -> Normalizer:
        """The normalizer with its statistics on `device`. A fitted transform is built from CPU
        dataset tensors but applied to whatever device the net runs on, and a frozen dataclass
        can't migrate in place -- so re-wrap on demand (a no-op when already there)."""
        if self.normalizer.mean.device == device:
            return self.normalizer
        return Normalizer(mean=self.normalizer.mean.to(device), std=self.normalizer.std.to(device))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Physical [*, 2] (m, rad) -> standardized log1p space."""
        return self._on(y.device)(torch.log1p(y))

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Standardized log1p space -> physical [*, 2] (m, rad), guaranteed non-negative."""
        return torch.expm1(self._on(y.device).inverse(y).clamp_min(0.0))


class PoseErrorMLP(nn.Module):
    """(v, wz, x, y, yaw) -> (e_pos, e_rot), in standardized log1p space -- see TargetTransform.

    `in_dim` is a parameter rather than a hardcoded 5 because custom_dataset also yields 6
    columns under yaw_encoding="sincos" and 2 + ny*nx under patch_spec=; pass
    len(ds.FEATURE_NAMES) and it is right in every case.

    LayerNorm (not BatchNorm) between trunk layers: it behaves identically at train and eval
    time and doesn't depend on batch statistics, which matters when the last batch of a few
    thousand rows can be tiny. SiLU over ReLU because the target field is smooth away from the
    grazing-contact boundary and dead ReLU units waste an already-small network.

    `target_transform` is an ordinary attribute, not a buffer: it is fitted data, not a learned
    parameter, so it stays out of state_dict() and must be saved alongside a checkpoint (and
    re-attached on load) for predict() to work."""

    def __init__(
        self,
        in_dim: int = DEFAULT_IN_DIM,
        *,
        hidden: int = DEFAULT_HIDDEN,
        depth: int = DEFAULT_DEPTH,
        target_transform: TargetTransform | None = None,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth!r}")

        layers: list[nn.Module] = []
        width = in_dim
        for _ in range(depth):
            layers.extend([nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.SiLU()])
            width = hidden

        self.trunk = nn.Sequential(*layers)
        # Separate heads rather than one Linear(hidden, 2) purely for readability -- the two are
        # arithmetically identical, but named heads keep the (e_pos, e_rot) column order, which
        # error_visual.py and TARGET_NAMES both depend on, explicit at the definition site.
        self.head_e_pos = nn.Linear(hidden, 1)
        self.head_e_rot = nn.Linear(hidden, 1)
        self.target_transform = target_transform

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, in_dim] -> [B, 2] in standardized log1p space. Train against
        TargetTransform.forward(y), not against raw metres/radians."""
        h = self.trunk(x)
        return torch.cat([self.head_e_pos(h), self.head_e_rot(h)], dim=-1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """[B, in_dim] -> [B, 2] as (e_pos in m, e_rot in rad). Inference only."""
        if self.target_transform is None:
            raise RuntimeError(
                "predict() needs a target_transform to undo the log1p/standardize the net was "
                "trained in -- pass one to __init__ or assign .target_transform. Use forward() "
                "if you want the raw model-space output."
            )
        return self.target_transform.inverse(self(x))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--in-dim", type=int, default=DEFAULT_IN_DIM,
        help=f"input columns (default: {DEFAULT_IN_DIM})",
    )
    parser.add_argument(
        "--hidden", type=int, default=DEFAULT_HIDDEN,
        help=f"trunk width (default: {DEFAULT_HIDDEN})",
    )
    parser.add_argument(
        "--depth", type=int, default=DEFAULT_DEPTH,
        help=f"trunk layers (default: {DEFAULT_DEPTH})",
    )
    args = parser.parse_args()

    net = PoseErrorMLP(args.in_dim, hidden=args.hidden, depth=args.depth)
    n_params = sum(p.numel() for p in net.parameters())
    print(
        f"PoseErrorMLP(in_dim={args.in_dim}, hidden={args.hidden}, depth={args.depth}): "
        f"{n_params} params"
    )

    x = torch.randn(8, args.in_dim)
    assert net(x).shape == (8, OUT_DIM), net(x).shape

    y = torch.rand(64, OUT_DIM) * 3.0  # stand-in for physical (e_pos, e_rot), both non-negative
    transform = TargetTransform.fit(y)
    assert torch.allclose(transform.inverse(transform.forward(y)), y, atol=1e-5), "not invertible"
    assert (transform.inverse(torch.randn(64, OUT_DIM) * 5.0) >= 0).all(), "negative error escaped"

    net.target_transform = transform
    prediction = net.predict(x)
    assert prediction.shape == (8, OUT_DIM) and (prediction >= 0).all()
    print(f"forward/predict/round-trip ok; predict() range: "
          f"[{prediction.min():.4f}, {prediction.max():.4f}] ({', '.join(TARGET_NAMES)})")
