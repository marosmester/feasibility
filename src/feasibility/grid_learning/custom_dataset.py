"""PyTorch dataset over a grid_learning dataset_grid_*.h5 (generate_dataset.py's output): one
sample is a whole map plus one commanded yaw rate, and its label is the ostrich-vs-hstack final
pose error at EVERY cell of the spawn lattice --

    x = (heightmap [G_h, G_h], wz)                       -- terrain as a whole, CNN-ready
    y = (e_pos, e_rot)  -- SE(3) final-pose error         [G, G, 2]
    mask = which lattice cells are a real reading         [G, G] bool

This is the grid_learning counterpart of feasibility.learning.custom_dataset.PoseErrorDataset,
which exposes a per-TRIAL (v, wz, spawn_x, spawn_y, spawn_yaw) -> (e_pos, e_rot) row. Here the
label is a whole divergence FIELD over one map instead of a single number, because generate_dataset
already replays every lattice cell -- this module's only job is to turn its stored raw 14-vector
poses (`y[..., :7]` ostrich, `y[..., 7:14]` hstack) into the (e_pos, e_rot) error pair a model
actually regresses against, the same way learning.custom_dataset.final_pose_errors does for its
own schema.

Deliberately independent of feasibility.learning (same non-dependence stance as the rest of this
directory -- see generate_dataset.py's module docstring): the SE(3)-error formula is
reimplemented here, VECTORIZED over the whole [R, G, G] label tensor at once (learning/pose_error.py
and gl_replay_grid.py each already carry their own single-pose copy of the same formula; looping
that per lattice cell here would mean tens of thousands of python-level calls per file). Masked
cells store an exact-zero 14-vector for both poses (see generate_dataset.py's `y[~mask] = 0.0`),
which happens to satisfy T1 = T2 = identity and so already reduces to (e_pos, e_rot) = (0, 0)
under this formula -- verified by the self-check below -- so the explicit `y[~mask] = 0.0` here is
belt-and-braces for the same "detectably not a real reading" invariant the raw file keeps, not a
correction of anything the vectorized formula gets wrong.

    python src/feasibility/grid_learning/custom_dataset.py outputs/dataset_grid_box_random_M2_L5_g15.h5

Also importable as `feasibility.grid_learning.custom_dataset` (this is a plain package, per the
root CLAUDE.md) -- model.py does exactly that for `Normalizer`/`TARGET_NAMES`.
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.data import random_split

TARGET_NAMES = ("e_pos", "e_rot")


def poses_to_se3(poses: np.ndarray) -> np.ndarray:
    """[..., 7] (x, y, z, qx, qy, qz, qw) -> [..., 4, 4] SE(3) matrices, vectorized over any
    number of leading batch dims -- same per-pose formula as learning/pose_error.py's
    pose_to_se3 and gl_replay_grid.py's copy of it, batched rather than called in a python loop
    since a grid_learning label tensor is [R, G, G, 7] per robot (tens of thousands of poses)."""
    x, y, z = poses[..., 0], poses[..., 1], poses[..., 2]
    qx, qy, qz, qw = poses[..., 3], poses[..., 4], poses[..., 5], poses[..., 6]
    R = np.stack(
        [
            1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw),
            2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw),
            2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2),
        ],
        axis=-1,
    ).reshape(poses.shape[:-1] + (3, 3))
    T = np.zeros(poses.shape[:-1] + (4, 4), dtype=poses.dtype)
    T[..., :3, :3] = R
    T[..., 0, 3], T[..., 1, 3], T[..., 2, 3] = x, y, z
    T[..., 3, 3] = 1.0
    return T


def se3_errors(T1: np.ndarray, T2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[..., 4, 4], [..., 4, 4] -> (pos_error, rot_error), each [...] -- T_err = T1^-1 @ T2, same
    convention as learning/pose_error.py's se3_error, batched via numpy's stacked-matrix
    inv/matmul/trace instead of a per-pose python loop."""
    T_err = np.linalg.inv(T1) @ T2
    pos_error = np.linalg.norm(T_err[..., :3, 3], axis=-1)
    trace = np.clip((np.trace(T_err[..., :3, :3], axis1=-2, axis2=-1) - 1) / 2, -1.0, 1.0)
    rot_error = np.arccos(trace)
    return pos_error, rot_error


class GridPoseErrorDataset(Dataset):
    """One sample per (map, wz) row of a dataset_grid_*.h5: x = (heightmap [G_h, G_h], wz)
    commanded yaw rate, y = (e_pos, e_rot) [G, G, 2] SE(3) final-pose error field between ostrich
    and hstack over the whole spawn lattice, mask = [G, G] bool (True = a real reading; False =
    obstacle-blocked spawn or a diverged solve -- see generate_dataset.py's module docstring).

    Reads the whole file into memory eagerly and computes every row's error field at once via
    poses_to_se3/se3_errors -- R is at most a few hundred rows and h5py file handles aren't
    fork-safe, same rationale as learning/custom_dataset.py's PoseErrorDataset.

    `heightmap` is read from the file's own `grid/heightmap` [n_maps, G_h, G_h] (already resampled
    onto the fixed CNN grid by utils.heightmap_to_tensor at generation time) and indexed per row by
    `map_index` -- there is no per-row copy in the file, so this dataset makes that lookup once in
    __init__ rather than in every __getitem__.
    """

    def __init__(self, path: pathlib.Path, *, device: str | torch.device = "cpu") -> None:
        self.source = pathlib.Path(path)
        with h5py.File(self.source, "r") as f:
            self.attrs = dict(f.attrs)
            self.git = dict(f["git"].attrs) if "git" in f else {}
            wz = f["wz"][()].astype(np.float32)  # [R]
            map_index = f["map_index"][()].astype(np.int64)  # [R]
            self.map_path = [
                p.decode() if isinstance(p, bytes) else p for p in f["map_path"][()]
            ]
            y_raw = f["y"][()].astype(np.float32)  # [R, G, G, 14]
            mask = f["mask"][()]  # [R, G, G] bool
            heightmaps = f["grid/heightmap"][()].astype(np.float32)  # [n_maps, G_h, G_h]
            self.spawn_xy = torch.from_numpy(f["spawn_xy"][()].astype(np.float32))  # [G, G, 2]
            self.resolution = float(f["grid/resolution"][()])
            self.extent = float(f["grid/extent"][()])

        T_ostrich = poses_to_se3(y_raw[..., :7])
        T_hstack = poses_to_se3(y_raw[..., 7:])
        e_pos, e_rot = se3_errors(T_ostrich, T_hstack)  # each [R, G, G]
        y = np.stack([e_pos, e_rot], axis=-1).astype(np.float32)  # [R, G, G, 2]
        y[~mask] = 0.0  # belt-and-braces -- see module docstring, this is already what the
        # identity-pose masked rows reduce to

        self.wz = torch.from_numpy(wz).to(device)
        self.map_index = torch.from_numpy(map_index).to(device)
        self.heightmap = torch.from_numpy(heightmaps).to(device)
        self.y = torch.from_numpy(y).to(device)
        self.mask = torch.from_numpy(mask).to(device)
        self.TARGET_NAMES = TARGET_NAMES

    def __len__(self) -> int:
        return self.wz.shape[0]

    def __getitem__(
        self, idx: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        heightmap = self.heightmap[self.map_index[idx]]
        return (heightmap, self.wz[idx]), self.y[idx], self.mask[idx]


def split_dataset(
    ds: GridPoseErrorDataset, val_frac: float = 0.2, seed: int = 0
) -> tuple[Subset, Subset]:
    """Seeded, reproducible train/val split over `ds`'s rows -- same recipe as
    learning/custom_dataset.py's split_dataset. Splitting by ROW (map, wz) rather than by lattice
    cell: a row's whole [G, G] field is one sample, and multiple rows can share a map (different
    wz), so this is a row-level split, not a map-level one -- a val row's map may also appear in
    train under a different commanded wz."""
    n_val = max(1, round(len(ds) * val_frac))
    n_train = len(ds) - n_val
    generator = torch.Generator().manual_seed(seed)
    return random_split(ds, [n_train, n_val], generator=generator)


@dataclasses.dataclass(frozen=True)
class Normalizer:
    """Per-column (x - mean) / std, std floored so a constant column never divides by zero --
    restated from learning/custom_dataset.py's Normalizer rather than imported (see module
    docstring)."""

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


def make_dataloaders(
    path: pathlib.Path,
    *,
    batch_size: int = 4,
    val_frac: float = 0.2,
    seed: int = 0,
    num_workers: int = 0,
    **ds_kwargs: object,
) -> tuple[DataLoader, DataLoader, GridPoseErrorDataset, Subset]:
    """Build a GridPoseErrorDataset over `path`, split it row-wise, fit a `wz` Normalizer on the
    TRAIN rows only, and return (train_loader, val_loader, dataset, train_subset) -- mirrors
    learning/custom_dataset.py's make_dataloaders. The heightmap is NOT normalized here: it is one
    physical field already in sane units (world-frame z, meters), the same reasoning
    learning/custom_dataset.py's build_input applies to its terrain patch block.

    Batch size defaults low (4, not 32): each sample carries a [G_h, G_h] heightmap (e.g.
    100x100 float32 = 40 KB) plus a [G, G, 2] label field, an order of magnitude heavier per row
    than learning/'s scalar-feature rows.
    """
    ds = GridPoseErrorDataset(path, **ds_kwargs)
    train_subset, val_subset = split_dataset(ds, val_frac=val_frac, seed=seed)

    ds.wz_normalizer = Normalizer.from_tensor(ds.wz[train_subset.indices].unsqueeze(-1))

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, ds, train_subset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("path", type=pathlib.Path, help="dataset_grid_*.h5 path")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    # Self-check: an identity-pose pair (both raw 7-vectors zero, as generate_dataset.py fills
    # masked cells) must reduce to exactly (e_pos, e_rot) = (0, 0) under this module's vectorized
    # formula -- if this ever fails, the explicit `y[~mask] = 0.0` in __init__ stops being
    # belt-and-braces and starts being load-bearing.
    zero_pose = np.zeros((2, 3, 7), dtype=np.float32)
    T = poses_to_se3(zero_pose)
    pos_e, rot_e = se3_errors(T, T)
    assert np.allclose(pos_e, 0.0) and np.allclose(rot_e, 0.0), (pos_e, rot_e)
    print("[self-check] zero-pose pair -> (e_pos, e_rot) = (0, 0) ok")

    train_loader, val_loader, ds, train_subset = make_dataloaders(args.path, batch_size=args.batch_size)
    (heightmap, wz), y, mask = next(iter(train_loader))
    print(
        f"{len(ds)} rows ({len(train_subset)} train), heightmap {tuple(heightmap.shape)}, "
        f"wz {tuple(wz.shape)}, y {tuple(y.shape)}, mask {tuple(mask.shape)}"
    )
    print(f"batch mask valid fraction: {mask.float().mean().item():.3f}")
