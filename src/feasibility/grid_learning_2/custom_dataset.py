"""PyTorch dataset over a grid_learning_2 dataset_grid2_*.h5 (generate_dataset.py's output): one
sample is a whole map plus a FIELD of commanded yaw rates, one per lattice cell, and its label is
the ostrich-vs-hstack final pose error at every cell of that same lattice --

    x = (heightmap [81, 81], wz [G, G])                  -- terrain + one command PER CELL
    y = (e_pos, e_rot)  -- SE(3) final-pose error         [G, G, 2]
    mask = which lattice cells are a real reading         [G, G] bool

The one schema difference from grid_learning's `dataset_grid_*.h5` is `wz`, which is [R, G, G]
here rather than [R] (design.md section 7a). That is the whole point of v2: in a v1 row all 225
cells share one command, so the command is perfectly confounded with the row and the effective
sample size for learning the CONTROL dependence is the number of rows, not the number of cells.

**v1 files are not supported.** A `wz` of shape [R] is rejected with a message rather than
broadcast to a constant field: a constant field is a measure-zero corner of the distribution v2
trains on, so silently accepting one would produce a dataset that looks fine and teaches the head
nothing about the command axis. Point generate_dataset.py at the same maps instead -- per-cell
commands cost exactly the same simulation (design.md section 7b).

Deliberately independent of feasibility.grid_learning and feasibility.learning (design.md section
11): the SE(3)-error formula is reimplemented here, VECTORIZED over the whole [R, G, G] label
tensor at once, since looping a per-pose formula over the lattice would mean tens of thousands of
python-level calls per file. Masked cells store an exact-zero 14-vector for both poses (the
generator's `y[~mask] = 0.0`), which satisfies T1 = T2 = identity and so already reduces to
(e_pos, e_rot) = (0, 0) under this formula -- verified by the self-check below -- so the explicit
`y[~mask] = 0.0` in __init__ is belt-and-braces for that invariant, not a correction.

`TARGET_NAMES` and the target transform come from model.py rather than being defined here (the
opposite of v1's direction, design.md section 11): the target space is a property of what the net
regresses, and keeping it next to the net lets model.py be run and checked before any dataset
exists.

CLI parameters:
    path             dataset_grid2_*.h5 path -- omit to run the offline self-test instead,
                      which writes a small synthetic file to a temp dir and reads it back
    --batch-size     rows per batch (default: 4)

Usage:
    python src/feasibility/grid_learning_2/custom_dataset.py          # synthetic self-test
    python src/feasibility/grid_learning_2/custom_dataset.py outputs/dataset_grid2_M2_L5_g15.h5
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

from feasibility.grid_learning_2.model import TARGET_NAMES
from feasibility.grid_learning_2.model import check_lattice_alignment


def poses_to_se3(poses: np.ndarray) -> np.ndarray:
    """[..., 7] (x, y, z, qx, qy, qz, qw) -> [..., 4, 4] SE(3) matrices, vectorized over any
    number of leading batch dims -- a label tensor here is [R, G, G, 7] per robot."""
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
    """[..., 4, 4], [..., 4, 4] -> (pos_error, rot_error), each [...] -- T_err = T1^-1 @ T2, the
    same convention learning/pose_error.py uses, batched via numpy's stacked-matrix
    inv/matmul/trace instead of a per-pose python loop."""
    T_err = np.linalg.inv(T1) @ T2
    pos_error = np.linalg.norm(T_err[..., :3, 3], axis=-1)
    trace = np.clip((np.trace(T_err[..., :3, :3], axis1=-2, axis2=-1) - 1) / 2, -1.0, 1.0)
    rot_error = np.arccos(trace)
    return pos_error, rot_error


class GridDivergenceDataset(Dataset):
    """One sample per row of a dataset_grid2_*.h5: x = (heightmap [G_h, G_h], wz [G, G]),
    y = (e_pos, e_rot) [G, G, 2], mask = [G, G] bool (True = a real reading; False = an
    obstacle-blocked spawn or a diverged solve).

    Reads the whole file into memory eagerly and computes every row's error field at once -- R is
    at most a few hundred rows, and h5py handles are not fork-safe.

    `heightmap` is read from `grid/heightmap` [n_maps, G_h, G_h] (already resampled onto the odd
    origin-centred CNN grid by utils.heightmap_to_tensor at generation time) and indexed per row
    by `map_index`; there is no per-row copy in the file.

    __init__ verifies the readout contract that replaces v1's `grid_sample`: the network's crop of
    the feature lattice must land exactly on the file's own `spawn_xy` (design.md section 3c).
    This is the right place for it -- it depends only on the file's grid geometry, so checking it
    once here means neither the model nor the training loop has to re-check per batch."""

    def __init__(self, path: pathlib.Path, *, device: str | torch.device = "cpu") -> None:
        self.source = pathlib.Path(path)
        with h5py.File(self.source, "r") as f:
            self.attrs = dict(f.attrs)
            self.git = dict(f["git"].attrs) if "git" in f else {}
            wz = f["wz"][()].astype(np.float32)
            map_index = f["map_index"][()].astype(np.int64)  # [R]
            self.map_path = [
                p.decode() if isinstance(p, bytes) else p for p in f["map_path"][()]
            ]
            y_raw = f["y"][()].astype(np.float32)  # [R, G, G, 14]
            mask = f["mask"][()]  # [R, G, G] bool
            heightmaps = f["grid/heightmap"][()].astype(np.float32)  # [n_maps, G_h, G_h]
            # [n_maps] map -> the map it derives from, so a mirrored map groups with its source
            # for the split. Absent unless a mirroring pass wrote it; then every map is its own
            # group, which is what a plain generate_dataset.py file means.
            map_source = (
                f["grid/map_source"][()].astype(np.int64)
                if "map_source" in f["grid"]
                else np.arange(heightmaps.shape[0], dtype=np.int64)
            )
            spawn_xy = f["spawn_xy"][()].astype(np.float32)  # [G, G, 2]
            self.resolution = float(f["grid/resolution"][()])
            self.extent = float(f["grid/extent"][()])

        if wz.ndim != 3:
            raise ValueError(
                f"{self.source} stores wz with shape {wz.shape}; grid_learning_2 needs a per-cell "
                f"command FIELD [R, G, G]. This looks like a grid_learning (v1) file, which is "
                "deliberately not supported -- see this module's docstring."
            )
        if wz.shape != mask.shape:
            raise ValueError(f"wz {wz.shape} and mask {mask.shape} must have the same shape")

        check_lattice_alignment(spawn_xy, n_input=heightmaps.shape[-1], resolution=self.resolution)

        T_ostrich = poses_to_se3(y_raw[..., :7])
        T_hstack = poses_to_se3(y_raw[..., 7:])
        e_pos, e_rot = se3_errors(T_ostrich, T_hstack)  # each [R, G, G]
        y = np.stack([e_pos, e_rot], axis=-1).astype(np.float32)  # [R, G, G, 2]
        y[~mask] = 0.0  # already what the identity-pose masked cells reduce to; see docstring

        self.wz = torch.from_numpy(wz).to(device)  # [R, G, G]
        self.map_index = torch.from_numpy(map_index).to(device)
        # [R] split key: equal to map_index unless a mirroring pass paired maps with their sources
        self.map_group = torch.from_numpy(map_source[map_index]).to(device)
        self.heightmap = torch.from_numpy(heightmaps).to(device)
        self.spawn_xy = torch.from_numpy(spawn_xy).to(device)
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


def split_dataset_by_map(
    ds: GridDivergenceDataset, val_frac: float = 0.2, seed: int = 0
) -> tuple[Subset, Subset]:
    """Seeded, reproducible train/val split over `ds`'s MAPS, not rows -- design.md section 7d.

    This is the only split offered. A row-level split leaks badly: with M maps x L rows, a val
    row's terrain almost always also appears in train under a different command field, so that
    val score measures interpolation in `wz` on memorised terrain rather than transfer to unseen
    terrain. It leaks MORE here than in v1, not less, because a v2 row's command content is richer
    and so a leaked map is an even easier row to fit.

    Splits on `ds.map_group`, not `ds.map_index`: a mirrored map is the SAME terrain as its
    source, so splitting them apart would leak just as badly. The two are identical on a file that
    was never mirrored."""
    map_ids = ds.map_group.unique()
    if map_ids.shape[0] < 2:
        raise ValueError(
            f"{ds.source} holds {map_ids.shape[0]} map group(s); a held-out-MAP split needs at "
            "least 2. A single-map file is a generator smoke test, not something to train on -- "
            "re-run generate_dataset.py with +n_maps>=2."
        )
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(map_ids.shape[0], generator=generator)
    n_val_maps = max(1, round(map_ids.shape[0] * val_frac))
    val_maps = set(map_ids[perm[:n_val_maps]].tolist())
    train_indices = [i for i in range(len(ds)) if int(ds.map_group[i]) not in val_maps]
    val_indices = [i for i in range(len(ds)) if int(ds.map_group[i]) in val_maps]
    return Subset(ds, train_indices), Subset(ds, val_indices)


def valid_targets(ds: GridDivergenceDataset, indices: list[int]) -> torch.Tensor:
    """[n, 2] physical (e_pos, e_rot) over the VALID cells of the given rows -- what
    `model.TargetTransform.fit` must be handed, and the reason it needs a helper rather than a
    one-liner: masked cells hold exact zeros by construction, and folding thousands of them into
    the mean/std would skew the standardisation toward "everything is zero" on a field that is
    ~94% flat ground anyway (design.md section 6). Callers pass `train_subset.indices` -- fitting
    on all rows would leak val statistics into the transform."""
    idx = torch.as_tensor(indices, dtype=torch.long, device=ds.y.device)
    return ds.y[idx][ds.mask[idx]]


def make_dataloaders(
    path: pathlib.Path,
    *,
    batch_size: int = 4,
    val_frac: float = 0.2,
    seed: int = 0,
    num_workers: int = 0,
    **ds_kwargs: object,
) -> tuple[DataLoader, DataLoader, GridDivergenceDataset, Subset, Subset]:
    """Build a GridDivergenceDataset over `path`, split it by map, and return
    (train_loader, val_loader, dataset, train_subset, val_subset).

    Nothing is normalised here. The heightmap is one physical field already in sane units
    (world-frame z in metres, which `model.relief` re-centres per sample anyway), and the command
    field is divided by `WZ_MAX` inside `model.command_features` -- a fitted normaliser for either
    would be a second, redundant source of truth. Only the TARGET transform is fitted, and that
    belongs to the caller, which is why both Subsets are returned: `train_subset.indices` feeds
    `valid_targets()` for a train-only fit, `val_subset.indices` scores baselines over exactly the
    held-out rows.

    Batch size defaults low (4): each sample carries an 81x81 heightmap plus a [G, G, 2] label
    field and a [G, G] command field, an order of magnitude heavier per row than a scalar-feature
    dataset. design.md section 10's recipe raises it to 16 for training."""
    ds = GridDivergenceDataset(path, **ds_kwargs)
    train_subset, val_subset = split_dataset_by_map(ds, val_frac=val_frac, seed=seed)
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, ds, train_subset, val_subset


def _write_synthetic(path: pathlib.Path, n_maps: int = 4, n_rows_per_map: int = 3) -> None:
    """Write a minimal dataset_grid2_*.h5 for the self-test below, pinning the schema this module
    reads before generate_dataset.py exists to write it. Labels are constructed so the SE(3)
    reduction has a known answer: both poses are pure translations with identity rotation and a
    known x-offset `d`, so e_pos == |d| and e_rot == 0 exactly."""
    from feasibility.grid_learning_2.utils import DEFAULT_N_CELLS
    from feasibility.grid_learning_2.utils import DEFAULT_RESOLUTION
    from feasibility.grid_learning_2.utils import extent_of
    from feasibility.grid_learning_2.utils import spawn_lattice

    rng = np.random.default_rng(0)
    lattice = spawn_lattice().astype(np.float32)
    G = lattice.shape[0]
    R = n_maps * n_rows_per_map

    map_index = np.repeat(np.arange(n_maps), n_rows_per_map).astype(np.int64)
    wz = rng.uniform(-1.0, 1.0, size=(R, G, G)).astype(np.float32)
    mask = rng.random((R, G, G)) > 0.1

    y = np.zeros((R, G, G, 14), dtype=np.float32)
    y[..., 3:7] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # identity quaternion, ostrich
    y[..., 10:14] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # ... and hstack
    y[..., 0] = lattice[None, ..., 0]  # ostrich x = the spawn cell's own x
    y[..., 7] = y[..., 0] + rng.uniform(0.0, 0.5, size=(R, G, G))  # hstack offset by d along x
    y[~mask] = 0.0

    with h5py.File(path, "w") as f:
        f.attrs["wz_per_cell"] = True
        f.create_dataset("wz", data=wz)
        f.create_dataset("map_index", data=map_index)
        f.create_dataset("map_path", data=np.array([f"synthetic/map_{m}" for m in range(n_maps)], dtype="S64"))
        f.create_dataset("y", data=y)
        f.create_dataset("mask", data=mask)
        f.create_dataset("spawn_xy", data=lattice)
        grid = f.create_group("grid")
        grid.create_dataset(
            "heightmap",
            data=rng.normal(0.0, 0.1, size=(n_maps, DEFAULT_N_CELLS, DEFAULT_N_CELLS)).astype(np.float32),
        )
        grid.create_dataset("resolution", data=DEFAULT_RESOLUTION)
        grid.create_dataset("extent", data=extent_of())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("path", type=pathlib.Path, nargs="?", help="dataset_grid2_*.h5 path")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    # --- an identity-pose pair (both raw 7-vectors zero, as the generator fills masked cells)
    # must reduce to exactly (0, 0). If this ever fails, `y[~mask] = 0.0` in __init__ stops being
    # belt-and-braces and starts being load-bearing. ------------------------------------------
    zero_pose = np.zeros((2, 3, 7), dtype=np.float32)
    T = poses_to_se3(zero_pose)
    pos_e, rot_e = se3_errors(T, T)
    assert np.allclose(pos_e, 0.0) and np.allclose(rot_e, 0.0), (pos_e, rot_e)
    print("[se3] zero-pose pair -> (e_pos, e_rot) = (0, 0) ok")

    with tempfile.TemporaryDirectory() as tmp:
        path = args.path
        if path is None:
            path = pathlib.Path(tmp) / "dataset_grid2_synthetic.h5"
            _write_synthetic(path)
            print(f"[self-test] wrote synthetic file {path.name}")

        train_loader, val_loader, ds, train_subset, val_subset = make_dataloaders(
            path, batch_size=args.batch_size
        )
        (heightmap, wz), y, mask = next(iter(train_loader))
        print(
            f"{len(ds)} rows ({len(train_subset)} train / {len(val_subset)} val), "
            f"heightmap {tuple(heightmap.shape)}, wz {tuple(wz.shape)}, y {tuple(y.shape)}, "
            f"mask {tuple(mask.shape)}"
        )
        print(f"[mask] batch valid fraction: {mask.float().mean().item():.3f}")

        # --- the command really is a FIELD, not a broadcast scalar: within a row, cells must
        # differ. This is the property v1 files lack and the reason they are rejected. ---------
        per_row_spread = (ds.wz.amax(dim=(1, 2)) - ds.wz.amin(dim=(1, 2)))
        assert (per_row_spread > 1e-6).all(), "some row has a constant command field"
        print(f"[command] per-row wz spread: min {per_row_spread.min():.3f}, "
              f"max {per_row_spread.max():.3f} rad/s -- a genuine field")

        # --- the split must be by map: no map may appear on both sides ------------------------
        train_maps = {int(ds.map_group[i]) for i in train_subset.indices}
        val_maps = {int(ds.map_group[i]) for i in val_subset.indices}
        assert not (train_maps & val_maps), f"map leak: {train_maps & val_maps}"
        print(f"[split] {len(train_maps)} train maps, {len(val_maps)} val maps, no overlap")

        # --- valid_targets must select exactly the unmasked cells, and only train rows --------
        fit_rows = valid_targets(ds, train_subset.indices)
        assert fit_rows.shape == (int(ds.mask[torch.as_tensor(train_subset.indices)].sum()), 2)
        assert (fit_rows >= 0).all(), "a negative physical error escaped the SE(3) reduction"
        print(f"[targets] {fit_rows.shape[0]} valid train cells, e_pos in "
              f"[{fit_rows[:, 0].min():.4f}, {fit_rows[:, 0].max():.4f}] m, e_rot in "
              f"[{fit_rows[:, 1].min():.4f}, {fit_rows[:, 1].max():.4f}] rad")

        if args.path is None:
            # The synthetic labels are pure x-translations, so e_pos is the offset and e_rot is 0
            assert torch.allclose(fit_rows[:, 1], torch.zeros_like(fit_rows[:, 1]), atol=1e-6), \
                "identity-rotation poses must give e_rot = 0"
            assert fit_rows[:, 0].max() <= 0.5 + 1e-5, fit_rows[:, 0].max()
            print("[self-test] synthetic labels reduce to the known (|d|, 0) answer")

        # --- a v1-shaped wz must be rejected, not silently broadcast --------------------------
        v1_path = pathlib.Path(tmp) / "dataset_grid_v1_shaped.h5"
        _write_synthetic(v1_path)
        with h5py.File(v1_path, "r+") as f:
            wz_v1 = f["wz"][()][:, 0, 0]
            del f["wz"]
            f.create_dataset("wz", data=wz_v1)
        try:
            GridDivergenceDataset(v1_path)
        except ValueError as exc:
            assert "per-cell command FIELD" in str(exc), exc
            print("[compat] a v1-shaped wz [R] is rejected with a message, not broadcast")
        else:
            raise AssertionError("a v1-shaped wz [R] must be rejected")

    print("all self-checks ok")
