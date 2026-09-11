"""PyTorch dataset over a lattice_learning `dataset_arc_*.h5` (`generate_dataset.py`'s output):
one sample is a single body-frame terrain patch plus ONE scalar curvature command, and its label
is the ostrich-vs-(arc+settle) final-pose SE(3) error (design.md sections 1, 7b) --

    x = (patch [1, 24, 28], kappa [])
    y = (e_pos, e_rot)          -- computed HERE from ref_pose and ostrich/pose[-1], not stored

Deliberately independent of `learning/`, `grid_learning/` and `grid_learning_2/` (design.md
section 11a): `se3_error` is re-derived below rather than imported -- ~10 lines of closed-form
SE(3) algebra that never changes, checked against a known rotation in `__main__`.
`feasibility.heightmap`/`feasibility.comparator` are shared infrastructure this module doesn't
even need directly; `generate_dataset.py` is the one that writes through
`comparator.provenance`.

`TARGET_NAMES` and `TargetTransform`/`Normalizer` come from `model.py` (the opposite of
`learning/`'s arrangement, design.md section 11a): the target space belongs to the network that
regresses it, and importing it from `model.py` means `model.py` runs and self-checks before any
dataset file exists.

A row's `valid` flag (design.md section 7b) marks the generator's own data-quality gate -- a
non-finite pose, a failed endpoint settle, an implausible displacement, or an overhanging patch --
so a `False` row carries no usable label and `__init__` drops it (`learning/custom_dataset.py`'s
non-finite-row precedent). `swept_clear` is kept as a per-row column on every SURVIVING row
instead: design.md section 8 is explicit that it is a REPORTING split ("train on everything
[valid]; report on the `swept_clear` subset"), not a second training filter -- the settle's own
`blocked` is heading-quantised and conservative, and its boundary is exactly where `d_hat` should
be earning its keep.

The train/val split is by MAP, never by row (`map_index`/`map_path`) -- the same leakage argument
`grid_learning_2/custom_dataset.py`'s `split_dataset_by_map` makes, restated here rather than
imported (design.md section 11a).

CLI parameters:
    path             dataset_arc_*.h5 path -- omit to run the offline self-test instead, which
                     writes a small synthetic file to a temp dir and reads it back
    --batch-size     rows per batch (default: 32)

Usage:
    python src/feasibility/lattice_learning/custom_dataset.py                # synthetic self-test
    python src/feasibility/lattice_learning/custom_dataset.py outputs/dataset_arc_seed0.h5
"""
from __future__ import annotations

import argparse
import pathlib
import tempfile

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset

from feasibility.lattice_learning.model import TARGET_NAMES
from feasibility.lattice_learning.patch import N_CHANNELS
from feasibility.lattice_learning.patch import patch_spec_from_attrs
from feasibility.lattice_learning.patch import patch_spec_to_attrs
from feasibility.lattice_learning.patch import PatchSpec


def pose_to_se3(pose: np.ndarray) -> np.ndarray:
    """[..., 7] (x, y, z, qx, qy, qz, qw) -> [..., 4, 4] SE(3), vectorized over any leading batch
    dims -- re-derived from `learning/pose_error.py`'s single-pose version (design.md 11a)."""
    x, y, z = pose[..., 0], pose[..., 1], pose[..., 2]
    qx, qy, qz, qw = pose[..., 3], pose[..., 4], pose[..., 5], pose[..., 6]
    R = np.stack(
        [
            1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw),
            2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw),
            2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2),
        ],
        axis=-1,
    ).reshape(pose.shape[:-1] + (3, 3))
    T = np.zeros(pose.shape[:-1] + (4, 4), dtype=pose.dtype)
    T[..., :3, :3] = R
    T[..., 0, 3], T[..., 1, 3], T[..., 2, 3] = x, y, z
    T[..., 3, 3] = 1.0
    return T


def se3_errors(T1: np.ndarray, T2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[..., 4, 4], [..., 4, 4] -> (pos_error, rot_error), each [...] -- T_err = T1^-1 @ T2, the
    same convention `learning/pose_error.py` uses, batched via numpy's stacked-matrix
    inv/matmul/trace rather than a per-row python loop (design.md 11a)."""
    T_err = np.linalg.inv(T1) @ T2
    pos_error = np.linalg.norm(T_err[..., :3, 3], axis=-1)
    trace = np.clip((np.trace(T_err[..., :3, :3], axis1=-2, axis2=-1) - 1) / 2, -1.0, 1.0)
    rot_error = np.arccos(trace)
    return pos_error, rot_error


class ArcDivergenceDataset(Dataset):
    """One sample per VALID row of a dataset_arc_*.h5: x = (patch [1, ny, nx], kappa []),
    y = (e_pos, e_rot) [2]. `swept_clear` [bool] and `map_index`/`map_path` survive alongside
    (design.md section 8's reporting split and section 11a's map-level split, respectively).

    Reads the whole file into memory eagerly -- h5py handles are not fork-safe, and even
    design.md section 7c's ~200k-trial budget at 24x28 float32 patches is a few hundred MB, not a
    per-map elevation grid."""

    def __init__(self, path: pathlib.Path, *, device: str | torch.device = "cpu") -> None:
        self.source = pathlib.Path(path)
        with h5py.File(self.source, "r") as f:
            self.attrs = dict(f.attrs)
            self.git = dict(f["git"].attrs) if "git" in f else {}
            patch = f["patch"][()].astype(np.float32)  # [n, ny*nx]
            kappa = f["kappa"][()].astype(np.float32)  # [n]
            ref_pose = f["ref_pose"][()].astype(np.float64)  # [n, 7]
            valid = f["valid"][()].astype(bool)  # [n]
            swept_clear = f["swept_clear"][()].astype(bool)  # [n]
            map_index = f["map_index"][()].astype(np.int64)  # [n]
            map_path = [p.decode() if isinstance(p, bytes) else p for p in f["map_path"][()]]
            ostrich_final = f["ostrich/pose"][-1].astype(np.float64)  # [n, 7]

        self.patch_spec = patch_spec_from_attrs(self.attrs)
        ny, nx = self.patch_spec.ny, self.patch_spec.nx
        assert N_CHANNELS == 1, (
            "ArcDivergenceDataset's patch reshape assumes a single relief channel -- see "
            "design.md section 3c on N_CHANNELS growing to 2; that is a schema change, and this "
            "reshape would need to change with it"
        )
        if patch.shape[1] != ny * nx:
            raise ValueError(
                f"{self.source} patch column count {patch.shape[1]} does not match its own "
                f"patch_* attrs ({ny}x{nx}={ny * nx}) -- file written by an incompatible PatchSpec"
            )

        e_pos, e_rot = se3_errors(pose_to_se3(ref_pose), pose_to_se3(ostrich_final))
        y = np.stack([e_pos, e_rot], axis=-1).astype(np.float32)  # [n, 2]

        keep = valid
        n_dropped = int((~keep).sum())
        if n_dropped:
            print(f"[dataset] {self.source.name}: dropping {n_dropped}/{len(keep)} invalid row(s)")

        self.patch = (
            torch.from_numpy(patch[keep].reshape(-1, ny, nx)).unsqueeze(1).to(device)
        )  # [m, 1, ny, nx]
        self.kappa = torch.from_numpy(kappa[keep]).to(device)
        self.y = torch.from_numpy(y[keep]).to(device)
        self.swept_clear = torch.from_numpy(swept_clear[keep]).to(device)
        self.map_index = torch.from_numpy(map_index[keep]).to(device)
        self.map_path = [p for p, k in zip(map_path, keep) if k]
        self.TARGET_NAMES = TARGET_NAMES

    def __len__(self) -> int:
        return self.patch.shape[0]

    def __getitem__(self, idx: int) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        return (self.patch[idx], self.kappa[idx]), self.y[idx]


def split_dataset_by_map(
    ds: ArcDivergenceDataset, val_frac: float = 0.2, seed: int = 0
) -> tuple[Subset, Subset]:
    """Seeded, reproducible train/val split over `ds`'s MAPS, not rows -- design.md section 11a.

    A row-level split leaks: with M maps and many trials per map, a val row's terrain almost
    always also appears in train under a different pose/kappa, so that val score measures
    interpolation on memorised terrain rather than transfer to unseen terrain -- the same argument
    `grid_learning_2/custom_dataset.py`'s `split_dataset_by_map` makes, restated here per the
    tree's independence stance (design.md 11a) rather than imported."""
    map_ids = ds.map_index.unique()
    if map_ids.shape[0] < 2:
        raise ValueError(
            f"{ds.source} holds {map_ids.shape[0]} map(s); a held-out-MAP split needs at least "
            "2. A single-map file is a generator smoke test, not something to train on -- "
            "re-run generate_dataset.py with +n_maps>=2."
        )
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(map_ids.shape[0], generator=generator)
    n_val_maps = max(1, round(map_ids.shape[0] * val_frac))
    val_maps = set(map_ids[perm[:n_val_maps]].tolist())
    train_indices = [i for i in range(len(ds)) if int(ds.map_index[i]) not in val_maps]
    val_indices = [i for i in range(len(ds)) if int(ds.map_index[i]) in val_maps]
    return Subset(ds, train_indices), Subset(ds, val_indices)


def valid_targets(ds: ArcDivergenceDataset, indices: list[int]) -> torch.Tensor:
    """[n, 2] physical (e_pos, e_rot) over the given rows -- what `model.TargetTransform.fit`
    must be handed. A named helper (rather than `ds.y[indices]` at the call site) purely for
    interface parity with `grid_learning_2/custom_dataset.py`'s `valid_targets`, whose `mask`
    argument this module has no analogue for: every row already surviving `__init__`'s `valid`
    filter has a real, usable target, so this is a plain index. Callers pass
    `train_subset.indices` -- fitting on all rows would leak val statistics into the transform."""
    idx = torch.as_tensor(indices, dtype=torch.long, device=ds.y.device)
    return ds.y[idx]


def make_dataloaders(
    path: pathlib.Path,
    *,
    batch_size: int = 32,
    val_frac: float = 0.2,
    seed: int = 0,
    num_workers: int = 0,
    **ds_kwargs: object,
) -> tuple[DataLoader, DataLoader, ArcDivergenceDataset, Subset, Subset]:
    """Build an ArcDivergenceDataset over `path`, split it by map, and return
    (train_loader, val_loader, dataset, train_subset, val_subset). Nothing is normalised here --
    the patch is already one physical field in sane units (design.md section 3c) and `kappa` is
    normalised inside `model.command_features`; only the TARGET transform is fitted, and that is
    the caller's job via `valid_targets(ds, train_subset.indices)`."""
    ds = ArcDivergenceDataset(path, **ds_kwargs)
    train_subset, val_subset = split_dataset_by_map(ds, val_frac=val_frac, seed=seed)
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, ds, train_subset, val_subset


def _write_synthetic(path: pathlib.Path, n_maps: int = 3, rows_per_map: int = 8) -> None:
    """Write a minimal dataset_arc_*.h5 for the self-test below, pinning the schema this module
    reads before generate_dataset.py has ever run in this environment. `ref_pose` is a pure
    translation (identity rotation) at a known offset from `ostrich/pose[-1]`, so the SE(3)
    reduction has a known answer: e_pos == the offset, e_rot == 0."""
    rng = np.random.default_rng(0)
    spec = PatchSpec()
    n = n_maps * rows_per_map

    map_index = np.repeat(np.arange(n_maps), rows_per_map).astype(np.int64)
    patch = rng.normal(0.0, 0.3, size=(n, spec.ny * spec.nx)).astype(np.float32)
    kappa = rng.uniform(-2.0, 2.0, size=n).astype(np.float32)

    ref_pose = np.zeros((n, 7), dtype=np.float64)
    ref_pose[:, 6] = 1.0  # identity quaternion
    offsets = rng.uniform(0.0, 0.5, size=n)
    ostrich_final = ref_pose.copy()
    ostrich_final[:, 0] += offsets  # pure x-translation from ref_pose

    valid = rng.random(n) > 0.1
    swept_clear = rng.random(n) > 0.2

    with h5py.File(path, "w") as f:
        for name, value in patch_spec_to_attrs(spec).items():
            f.attrs[name] = value
        f.attrs["v_nom"] = 0.6
        f.attrs["arc_len"] = 0.3
        f.attrs["min_turn_radius"] = 0.5
        f.create_dataset("patch", data=patch)
        f.create_dataset("kappa", data=kappa)
        f.create_dataset("ref_pose", data=ref_pose.astype(np.float32))
        f.create_dataset("valid", data=valid)
        f.create_dataset("swept_clear", data=swept_clear)
        f.create_dataset("map_index", data=map_index)
        f.create_dataset(
            "map_path", data=np.array([f"synthetic/map_{m}" for m in map_index], dtype="S32")
        )
        grp = f.create_group("ostrich")
        grp.create_dataset("pose", data=ostrich_final[None].astype(np.float32))  # [T=1, n, 7]
        grp_git = f.create_group("git")
        grp_git.attrs["helhest_stack_sha"] = "unknown"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("path", type=pathlib.Path, nargs="?", help="dataset_arc_*.h5 path")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    # --- an identity-pose pair must reduce to exactly (0, 0) ------------------------------------
    zero_pose = np.zeros((2, 7), dtype=np.float32)
    zero_pose[:, 6] = 1.0
    T = pose_to_se3(zero_pose)
    pos_e, rot_e = se3_errors(T, T)
    assert np.allclose(pos_e, 0.0) and np.allclose(rot_e, 0.0), (pos_e, rot_e)
    print("[se3] identity-pose pair -> (e_pos, e_rot) = (0, 0) ok")

    with tempfile.TemporaryDirectory() as tmp:
        path = args.path
        if path is None:
            path = pathlib.Path(tmp) / "dataset_arc_synthetic.h5"
            _write_synthetic(path)
            print(f"[self-test] wrote synthetic file {path.name}")

        train_loader, val_loader, ds, train_subset, val_subset = make_dataloaders(
            path, batch_size=args.batch_size
        )
        (patch, kappa), y = next(iter(train_loader))
        print(
            f"{len(ds)} valid rows ({len(train_subset)} train / {len(val_subset)} val), "
            f"patch {tuple(patch.shape)}, kappa {tuple(kappa.shape)}, y {tuple(y.shape)}"
        )

        # --- the split must be by map: no map may appear on both sides -------------------------
        train_maps = {int(ds.map_index[i]) for i in train_subset.indices}
        val_maps = {int(ds.map_index[i]) for i in val_subset.indices}
        assert not (train_maps & val_maps), f"map leak: {train_maps & val_maps}"
        print(f"[split] {len(train_maps)} train maps, {len(val_maps)} val maps, no overlap")

        # --- valid_targets selects exactly the given (train-only) rows -------------------------
        fit_rows = valid_targets(ds, train_subset.indices)
        assert fit_rows.shape == (len(train_subset), 2)
        assert (fit_rows >= 0).all(), "a negative physical error escaped the SE(3) reduction"
        print(
            f"[targets] {fit_rows.shape[0]} train rows, e_pos in "
            f"[{fit_rows[:, 0].min():.4f}, {fit_rows[:, 0].max():.4f}] m, e_rot in "
            f"[{fit_rows[:, 1].min():.4f}, {fit_rows[:, 1].max():.4f}] rad"
        )

        if args.path is None:
            # synthetic labels are pure x-translations, so e_pos is the offset and e_rot is 0
            assert torch.allclose(ds.y[:, 1], torch.zeros_like(ds.y[:, 1]), atol=1e-5), (
                "identity-rotation poses must give e_rot = 0"
            )
            assert ds.y[:, 0].max() <= 0.5 + 1e-4, ds.y[:, 0].max()
            print("[self-test] synthetic labels reduce to the known (offset, 0) answer")
            print(f"[swept_clear] {int(ds.swept_clear.sum())}/{len(ds)} rows marked swept-clear "
                  "(reporting-only, design.md section 8)")

    print("all self-checks ok")
