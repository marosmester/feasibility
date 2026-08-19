"""PyTorch dataset over comparator output HDF5 (comparator/provenance.py's write_comparison()
schema, produced by comparator/common.py's run_trial_comparison -- see TrialScenarioSpec/Trial).

A trial-style comparison file holds ONE shared heightmap and `n` rows that each vary the
commanded body twist and initial chassis pose -- exactly the (v, wz, x, y, yaw) -> (e_pos,
e_rot) mapping this dataset exposes:

    x = (v_drive, wz_drive, spawn_x, spawn_y, spawn_yaw)          [n, 5]
    y = (e_pos, e_rot)  -- final-pose SE(3) error, see pose_error [n, 2]

No generator for such files exists yet; this module only reads them. In the meantime it also
loads fine against a *sweep*-style file (compare_speed_bumps.py/compare_box_obstacles.py),
where spawn_pose/v_drive/wz_drive happen to be constant across rows instead of terrain --
useful for smoke-testing since those files already exist under outputs/.

    python -c "
    from feasibility.learning.dataset import make_dataloaders
    train, val, ds = make_dataloaders('outputs/compare_on_surface.h5', batch_size=2)
    xb, yb = next(iter(train))
    print(xb.shape, yb.shape)
    "
"""
from __future__ import annotations

import dataclasses
import pathlib

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.data import random_split

from feasibility.learning.pose_error import pose_to_se3
from feasibility.learning.pose_error import se3_error

FEATURE_NAMES_RAW = ("v", "wz", "x", "y", "yaw")
FEATURE_NAMES_SINCOS = ("v", "wz", "x", "y", "cos_yaw", "sin_yaw")
TARGET_NAMES = ("e_pos", "e_rot")


def final_pose_errors(path: pathlib.Path) -> np.ndarray:
    """Vectorized-over-rows counterpart to pose_error.final_pose_error: reads
    ostrich/pose[-1] and hstack/pose[-1] (each [n, 7]) from `path` and returns [n, 2]
    (e_pos, e_rot), reusing pose_to_se3/se3_error row by row so the metric can't drift
    from the single-variant CLI in pose_error.py."""
    with h5py.File(path, "r") as f:
        ostrich_final = f["ostrich/pose"][-1]
        hstack_final = f["hstack/pose"][-1]
    n = ostrich_final.shape[0]
    errors = np.empty((n, 2), dtype=np.float32)
    for i in range(n):
        errors[i] = se3_error(pose_to_se3(ostrich_final[i]), pose_to_se3(hstack_final[i]))
    return errors


@dataclasses.dataclass(frozen=True)
class Normalizer:
    """Per-column (x - mean) / std, with std floored so a constant column (e.g. v_drive in a
    pure-turn-in-place file) never divides by zero."""

    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def from_tensor(cls, x: torch.Tensor, eps: float = 1e-8) -> "Normalizer":
        mean = x.mean(dim=0)
        std = x.std(dim=0).clamp_min(eps)
        return cls(mean=mean, std=std)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


class PoseErrorDataset(Dataset):
    """One sample per row of a comparator output HDF5: x = (v, wz, x, y, yaw) commanded twist
    + spawn pose, y = (e_pos, e_rot) final-pose SE(3) error between ostrich and hstack.

    Reads the whole file into numpy eagerly in __init__ (n is tiny -- a handful to a few
    thousand rows -- and h5py file handles aren't fork-safe, so this sidesteps DataLoader
    num_workers>0 issues entirely rather than working around them).

    x_normalizer/y_normalizer are plain attributes, not baked into the stored tensors, so a
    train/val split can fit them on train rows only and then attach them to this shared
    instance afterwards -- see split_dataset()/make_dataloaders().
    """

    def __init__(
        self,
        path: pathlib.Path,
        *,
        x_normalizer: Normalizer | None = None,
        y_normalizer: Normalizer | None = None,
        yaw_encoding: str = "raw",
    ) -> None:
        if yaw_encoding not in ("raw", "sincos"):
            raise ValueError(f"yaw_encoding must be 'raw' or 'sincos', got {yaw_encoding!r}")

        self.source = pathlib.Path(path)
        self.yaw_encoding = yaw_encoding
        self.x_normalizer = x_normalizer
        self.y_normalizer = y_normalizer

        with h5py.File(self.source, "r") as f:
            v_drive = f["v_drive"][()].astype(np.float32)
            wz_drive = f["wz_drive"][()].astype(np.float32)
            spawn_pose = f["spawn_pose"][()].astype(np.float32)  # [n, 3] = (x, y, yaw)
            self.labels = [
                label.decode() if isinstance(label, bytes) else label
                for label in f["variant_label"][()]
            ]
            self.attrs = dict(f.attrs)
            self.git = dict(f["git"].attrs) if "git" in f else {}

        x, y, yaw = spawn_pose[:, 0], spawn_pose[:, 1], spawn_pose[:, 2]
        if yaw_encoding == "sincos":
            cols = [v_drive, wz_drive, x, y, np.cos(yaw), np.sin(yaw)]
            self.FEATURE_NAMES = FEATURE_NAMES_SINCOS
        else:
            cols = [v_drive, wz_drive, x, y, yaw]
            self.FEATURE_NAMES = FEATURE_NAMES_RAW
        self.TARGET_NAMES = TARGET_NAMES

        self.x = torch.from_numpy(np.stack(cols, axis=1))
        self.y = torch.from_numpy(final_pose_errors(self.source))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.x[idx], self.y[idx]
        if self.x_normalizer is not None:
            x = self.x_normalizer(x)
        if self.y_normalizer is not None:
            y = self.y_normalizer(y)
        return x, y


def split_dataset(
    ds: PoseErrorDataset, val_frac: float = 0.2, seed: int = 0
) -> tuple[Subset, Subset]:
    """Seeded, reproducible train/val split over `ds`'s rows."""
    n_val = max(1, round(len(ds) * val_frac))
    n_train = len(ds) - n_val
    generator = torch.Generator().manual_seed(seed)
    return random_split(ds, [n_train, n_val], generator=generator)


def make_dataloaders(
    path: pathlib.Path,
    *,
    batch_size: int = 32,
    val_frac: float = 0.2,
    seed: int = 0,
    normalize_targets: bool = False,
    num_workers: int = 0,
    **ds_kwargs: object,
) -> tuple[DataLoader, DataLoader, PoseErrorDataset]:
    """Build a PoseErrorDataset over `path`, split it, fit x/y Normalizers on the TRAIN rows
    only (fitting on all rows would leak val statistics into training), attach them to the
    (shared) underlying dataset, and return (train_loader, val_loader, dataset).

    normalize_targets defaults to False: e_pos (m) and e_rot (rad) are already on comparable
    scales for these scenarios, so leaving y raw keeps a loss directly readable in physical
    units. Set it when a bigger/mixed sweep makes one term dominate an MSE loss.
    """
    ds = PoseErrorDataset(path, **ds_kwargs)
    train_subset, val_subset = split_dataset(ds, val_frac=val_frac, seed=seed)

    ds.x_normalizer = Normalizer.from_tensor(ds.x[train_subset.indices])
    if normalize_targets:
        ds.y_normalizer = Normalizer.from_tensor(ds.y[train_subset.indices])

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, ds
