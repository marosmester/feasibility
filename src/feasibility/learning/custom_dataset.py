"""PyTorch dataset over comparator output HDF5 (comparator/provenance.py's write_comparison()
schema, produced by comparator/common.py's run_trial_comparison -- see TrialScenarioSpec/Trial).

A trial-style comparison file holds ONE shared heightmap and `n` rows that each vary the
commanded body twist and initial chassis pose -- exactly the (v, wz, x, y, yaw) -> (e_pos,
e_rot) mapping this dataset exposes:

    x = (v_drive, wz_drive, spawn_x, spawn_y, spawn_yaw)          [n, 5]
    y = (e_pos, e_rot)  -- final-pose SE(3) error, see pose_error [n, 2]

feasibility.learning.generate_init_pose_dataset generates exactly this: N randomized (spawn pose,
constant body twist) trials on one fixed centered-box heightmap (its sibling,
generate_dataset_body_centered_patch, writes the same schema plus a baked-in `patch` dataset --
see TERRAIN PATCH MODE below). This module also loads fine against a
*sweep*-style file (compare_speed_bumps.py/compare_box_obstacles.py), where
spawn_pose/v_drive/wz_drive happen to be constant across rows instead of terrain -- useful for
smoke-testing since those files already exist under outputs/.

    python -c "
    from feasibility.learning.custom_dataset import make_dataloaders
    train, val, ds, train_subset = make_dataloaders('outputs/compare_on_surface.h5', batch_size=2)
    xb, yb = next(iter(train))
    print(xb.shape, yb.shape)
    "

TERRAIN PATCH MODE (`use_patch=True`) swaps the spawn pose for the terrain itself:

    x = (v_drive, wz_drive) + flattened body-frame patch      [n, 2 + ny*nx]

The patch is read straight from the file's own `patch` dataset -- a feature of the FILE, baked in
once by feasibility.learning.generate_dataset_body_centered_patch.py, not recomputed here. See
terrain_patch.py for what the patch is and why (body-aligned, height relative to the wheel
contacts, scaled by WHEEL_RADIUS, forward-biased). Two consequences here:

* `include_pose` defaults to False in this mode, i.e. (x, y, yaw) are DROPPED. That is the
  point: the patch already says where the robot is relative to the terrain, in a form that
  transfers to terrain the model never saw, while absolute coordinates only mean anything on the
  one heightmap they were fitted to. Pass include_pose=True to keep both and ablate.
* The patch block is NOT run through x_normalizer. It is one physical field already in sane
  units, and per-column standardization would divide the float-noise std of the always-flat
  cells (most of the patch, on this terrain) up into O(1) noise. x_normalizer therefore covers
  only the leading SCALAR_FEATURE_NAMES columns -- build_input() is the single place that knows
  this, and both __getitem__ and inference (replay/test_nn.py) go through it.

use_patch=True on a file with no `patch` dataset (e.g. one written by
generate_init_pose_dataset.py) raises ValueError -- regenerate it with
generate_dataset_body_centered_patch.py instead.

    python -c "
    from feasibility.learning.custom_dataset import PoseErrorDataset
    ds = PoseErrorDataset('outputs/dataset_patch_box_h070cm_n128_cont.h5', use_patch=True)
    print(ds.x.shape, ds.patch.shape, len(ds.FEATURE_NAMES))
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
from feasibility.learning.terrain_patch import patch_feature_names
from feasibility.learning.terrain_patch import patch_spec_from_attrs

FEATURE_NAMES_TWIST = ("v", "wz")  # always present -- the commanded twist is a causal input, not
# a stand-in for terrain the way the spawn pose is
FEATURE_NAMES_POSE_RAW = ("x", "y", "yaw")
FEATURE_NAMES_POSE_SINCOS = ("x", "y", "cos_yaw", "sin_yaw")
FEATURE_NAMES_RAW = FEATURE_NAMES_TWIST + FEATURE_NAMES_POSE_RAW
FEATURE_NAMES_SINCOS = FEATURE_NAMES_TWIST + FEATURE_NAMES_POSE_SINCOS
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


def build_input(
    scalars: torch.Tensor, patch: torch.Tensor | None, x_normalizer: Normalizer | None
) -> torch.Tensor:
    """Assemble the model-ready feature row(s): normalize the scalar block, then concatenate the
    (already-scaled, deliberately un-normalized) patch block -- see the module docstring on why
    the patch is excluded from x_normalizer.

    The single source of truth for that layout, shared by __getitem__ and by inference-time
    callers (replay/test_nn.py), so a checkpoint can never be fed a row assembled a different
    way than the rows it was trained on. Shapes broadcast: [..., n_scalar] and [..., n_patch]."""
    x = scalars if x_normalizer is None else x_normalizer(scalars)
    return x if patch is None else torch.cat([x, patch], dim=-1)


class PoseErrorDataset(Dataset):
    """One sample per row of a comparator output HDF5: x = (v, wz) commanded twist plus either
    the spawn pose (x, y, yaw) or a body-frame terrain patch (see `use_patch`, and the module
    docstring on why they're alternatives rather than both by default), y = (e_pos, e_rot)
    final-pose SE(3) error between ostrich and hstack.

    Reads the whole file into numpy eagerly in __init__ (n is tiny -- a handful to a few
    thousand rows -- and h5py file handles aren't fork-safe, so this sidesteps DataLoader
    num_workers>0 issues entirely rather than working around them). With use_patch=True the
    file's own `patch` dataset is read here too, once, rather than per __getitem__ -- it was
    already baked in at generation time (generate_dataset_body_centered_patch.py), so there is no
    per-load computation left to repeat.

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
        use_patch: bool = False,
        include_pose: bool | None = None,
    ) -> None:
        if yaw_encoding not in ("raw", "sincos"):
            raise ValueError(f"yaw_encoding must be 'raw' or 'sincos', got {yaw_encoding!r}")
        # Defaulting rather than forcing: with a patch the pose is redundant AND terrain-specific
        # (module docstring), so dropping it is the right default -- but keeping both is the
        # ablation that says whether the patch actually carries the pose's information.
        if include_pose is None:
            include_pose = not use_patch

        self.source = pathlib.Path(path)
        self.yaw_encoding = yaw_encoding
        self.include_pose = include_pose
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
            if use_patch:
                if "patch" not in f:
                    raise ValueError(
                        f"{self.source} has no baked-in `patch` dataset -- generate it with "
                        f"generate_dataset_body_centered_patch.py, or pass use_patch=False to "
                        f"train on the spawn pose instead"
                    )
                patch_arr = f["patch"][()].astype(np.float32)  # [n, ny*nx], already flattened
                self.patch_spec = patch_spec_from_attrs(self.attrs)
            else:
                patch_arr = None
                self.patch_spec = None

        x, y, yaw = spawn_pose[:, 0], spawn_pose[:, 1], spawn_pose[:, 2]
        cols = [v_drive, wz_drive]
        scalar_names = FEATURE_NAMES_TWIST
        if include_pose:
            if yaw_encoding == "sincos":
                cols += [x, y, np.cos(yaw), np.sin(yaw)]
                scalar_names += FEATURE_NAMES_POSE_SINCOS
            else:
                cols += [x, y, yaw]
                scalar_names += FEATURE_NAMES_POSE_RAW

        self.SCALAR_FEATURE_NAMES = scalar_names
        self.FEATURE_NAMES = scalar_names + (
            () if self.patch_spec is None else patch_feature_names(self.patch_spec)
        )
        self.TARGET_NAMES = TARGET_NAMES

        x_arr = np.stack(cols, axis=1)
        y_arr = final_pose_errors(self.source)

        # A handful of rows can carry a non-finite target -- typically the ostrich dynamics
        # solver diverging on that one trial (see final_pose_errors/se3_error). A single such
        # row poisons every downstream torch.mean/std over the whole column (nanmean would only
        # fix that for stats we compute ourselves, not the loss on the row itself, and the row's
        # target is simply invalid) -- so drop it here, once, before anything else touches y.
        finite = np.isfinite(x_arr).all(axis=1) & np.isfinite(y_arr).all(axis=1)
        if patch_arr is not None:
            finite &= np.isfinite(patch_arr).all(axis=1)
        n_dropped = int((~finite).sum())
        if n_dropped:
            dropped_idx = np.flatnonzero(~finite).tolist()
            print(
                f"[dataset] {self.source.name}: dropping {n_dropped}/{len(finite)} row(s) with "
                f"non-finite x/y (indices {dropped_idx})"
            )
            x_arr, y_arr = x_arr[finite], y_arr[finite]
            if patch_arr is not None:
                patch_arr = patch_arr[finite]
            self.labels = [label for label, keep in zip(self.labels, finite) if keep]

        self.x = torch.from_numpy(x_arr)
        self.y = torch.from_numpy(y_arr)
        self.patch = None if patch_arr is None else torch.from_numpy(patch_arr)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        patch = None if self.patch is None else self.patch[idx]
        x = build_input(self.x[idx], patch, self.x_normalizer)
        y = self.y[idx]
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
) -> tuple[DataLoader, DataLoader, PoseErrorDataset, Subset]:
    """Build a PoseErrorDataset over `path`, split it, fit x/y Normalizers on the TRAIN rows
    only (fitting on all rows would leak val statistics into training), attach them to the
    (shared) underlying dataset, and return (train_loader, val_loader, dataset, train_subset).

    `ds_kwargs` forwards to PoseErrorDataset -- `yaw_encoding`, and `use_patch`/`include_pose`
    for terrain-patch mode. x_normalizer is fitted on `ds.x`, which in patch mode holds the
    SCALAR block alone; the patch block carries its own fixed scaling and is never standardized
    (see the module docstring and build_input()).

    train_subset is returned (not just consumed internally) so a caller can fit further
    TRAIN-only statistics of its own -- e.g. model.TargetTransform.fit(ds.y[train_subset.indices])
    -- without re-deriving the split itself.

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
    return train_loader, val_loader, ds, train_subset
