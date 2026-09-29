"""PyTorch dataset over one or more mppi_learning `dataset_mppi_*.h5` files (generate_dataset.py's
output) -- design.md section 8 step 6. One sample is the body-frame patch at ostrich's realized
window start plus that window's command, and its label is how far helhest_stack's twin and ostrich
end up apart after the window:

    x = (patch [1, 24, 36], (v_mean, v_slope, wz_mean, wz_slope) [4])
    y = (e_pos, e_rot)                        -- label_mode="pos_rot" (default)
      = (e_pos, e_roll, e_pitch, e_yaw)       -- label_mode="pos_rpy"

`y` is computed HERE, from `twin_pose` and `ostrich/pose[-1]`, with lattice_learning's
`se3_errors` / `rpy_errors` (T_err = T_twin^-1 T_ostrich). The command is the file's `command`
column, `command.encode` of MPPI's own (uncompensated) wheel speeds -- what the net will be asked
about inside MPPI.

Rows: `valid == False` (the generator's data-quality gate) is always dropped. Two optional
filters, off by default:
  * `drop_endpoint_infeasible`: the static settle at the NOMINAL window end is infeasible. Edge
    and ramp trials drive into walls on purpose, so these rows are mostly the interesting ones.
  * `drop_twin_flagged`: the twin itself marks its rollout as fiction -- min clearance <= 0
    (high-centering) or a settle residual above helhest's `resid_tol`. MPPI rejects or penalizes
    such rollouts already, so the net may not need to learn them.

Several files train together. Their patch spec and every attr that defines the label must agree
(`LABEL_ATTRS`), or loading raises. Maps are identified by their `map_path` ACROSS files, not by
each file's `map_index`: two runs over one map dir share maps, and a per-file index would put the
same terrain on both sides of the split.

Everything else a report may want to split by is kept per row, beside the model's inputs:
`sampling`, `family`, `interact_dir`, `relief`, `endpoint_feasible`, `twin_min_clearance`,
`twin_max_residual`, `entry`, `jump` (the warm-up did NOT continue the window's own first command:
the `jump_frac` share, plus `rotate_in_place`, which is entered at rest, `entry` == 0), `t0_twist`
(design.md section 7's candidate fifth input) and `origin_drift`.

The split (`split_dataset_by_map`) and `valid_targets` are lattice_learning's, which only read
`map_index`, `y` and `source`.

Usage:
    python src/feasibility/mppi_learning/custom_dataset.py                  # synthetic self-test
    python src/feasibility/mppi_learning/custom_dataset.py outputs/dataset_mppi_<...>.h5 [more.h5 ...]
"""
from __future__ import annotations

import argparse
import pathlib
import tempfile
from collections.abc import Sequence

import h5py
import numpy as np
import torch
from helhest import dynamics
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset

from feasibility.lattice_learning.custom_dataset import pose_to_se3
from feasibility.lattice_learning.custom_dataset import rpy_errors
from feasibility.lattice_learning.custom_dataset import se3_errors
from feasibility.lattice_learning.custom_dataset import split_dataset_by_map
from feasibility.lattice_learning.custom_dataset import valid_targets
from feasibility.lattice_learning.patch import N_CHANNELS
from feasibility.lattice_learning.patch import patch_spec_from_attrs
from feasibility.lattice_learning.patch import patch_spec_to_attrs
from feasibility.mppi_learning.command import wheels_to_twist
from feasibility.mppi_learning.model import COMMAND_COLUMNS
from feasibility.mppi_learning.model import LABEL_MODES
from feasibility.mppi_learning.model import LABEL_NAMES
from feasibility.mppi_learning.spawn_sampling import PATCH_SPEC

__all__ = ["LABEL_ATTRS", "WindowDivergenceDataset", "make_dataloaders", "split_dataset_by_map", "valid_targets"]

# root attrs that change what a label MEANS; files that differ in any of them cannot train together
LABEL_ATTRS = ("window_s", "mppi_dt", "twin_solver", "k_turn", "ostrich_yaw_gain", "mu")
RESID_TOL = float(dynamics.robot_params().resid_tol)
JUMP_TOL = 1e-5  # a continued entry equals its window's first twist to float32 rounding (~1e-7)


def _strings(dataset: h5py.Dataset) -> list[str]:
    return [s.decode() if isinstance(s, bytes) else str(s) for s in dataset[()]]


def _same(a: object, b: object) -> bool:
    if isinstance(a, (str, bytes)) or isinstance(b, (str, bytes)):
        return a == b
    return bool(np.allclose(np.asarray(a, np.float64), np.asarray(b, np.float64), rtol=0.0, atol=1e-9))


class WindowDivergenceDataset(Dataset):
    """One sample per kept row of the given dataset_mppi_*.h5 file(s): x = (patch [1, ny, nx],
    command [4]), y [K] in `TARGET_NAMES` order. Reads everything into memory eagerly (h5py handles
    are not fork-safe; a 24 x 36 float32 patch is 3.5 kB, so 100k rows are ~350 MB)."""

    def __init__(
        self,
        paths: pathlib.Path | str | Sequence[pathlib.Path | str],
        *,
        device: str | torch.device = "cpu",
        label_mode: str = "pos_rot",
        drop_endpoint_infeasible: bool = False,
        drop_twin_flagged: bool = False,
    ) -> None:
        if label_mode not in LABEL_MODES:
            raise ValueError(f"label_mode must be one of {LABEL_MODES}, got {label_mode!r}")
        paths = [paths] if isinstance(paths, (str, pathlib.Path)) else list(paths)
        if not paths:
            raise ValueError("no dataset file given")
        self.sources = [pathlib.Path(p) for p in paths]
        self.source = self.sources[0] if len(self.sources) == 1 else pathlib.Path("+".join(p.name for p in self.sources))

        cols: dict[str, list[np.ndarray]] = {}
        map_paths: list[str] = []
        for path in self.sources:
            with h5py.File(path, "r") as f:
                attrs = dict(f.attrs)
                if path == self.sources[0]:
                    self.attrs = attrs
                    self.git = dict(f["git"].attrs) if "git" in f else {}
                    self.patch_spec = patch_spec_from_attrs(attrs)
                else:
                    if patch_spec_from_attrs(attrs) != self.patch_spec:
                        raise ValueError(f"{path.name}: patch spec {patch_spec_from_attrs(attrs)} != {self.patch_spec}")
                    for key in LABEL_ATTRS:
                        if not _same(attrs.get(key), self.attrs.get(key)):
                            raise ValueError(f"{path.name}: {key} = {attrs.get(key)!r} but {self.sources[0].name} "
                                             f"has {self.attrs.get(key)!r} -- the labels mean different things")
                for key in ("patch", "command", "twin_pose", "valid", "family", "interact_dir", "relief",
                            "endpoint_feasible", "twin_min_clearance", "twin_max_residual", "entry", "t0_twist",
                            "origin_drift"):
                    cols.setdefault(key, []).append(f[key][()])
                cols.setdefault("ostrich_end", []).append(f["ostrich/pose"][-1])
                cols.setdefault("first_twist", []).append(np.stack(wheels_to_twist(f["omega"][:, 0].astype(np.float64)), axis=-1))
                cols.setdefault("sampling", []).append(np.asarray(_strings(f["sampling"]), dtype=object))
                map_paths += _strings(f["map_path"])
        data = {k: np.concatenate(v) for k, v in cols.items()}

        ny, nx = self.patch_spec.ny, self.patch_spec.nx
        if data["patch"].shape[1] != ny * nx:
            raise ValueError(f"patch has {data['patch'].shape[1]} columns, its own patch_* attrs say {ny}x{nx}")
        if data["command"].shape[1] != len(COMMAND_COLUMNS):
            raise ValueError(f"command has {data['command'].shape[1]} columns, expected {COMMAND_COLUMNS}")

        T_twin = pose_to_se3(data["twin_pose"].astype(np.float64))
        T_ostrich = pose_to_se3(data["ostrich_end"].astype(np.float64))
        errors = dict(zip(("e_pos", "e_rot"), se3_errors(T_twin, T_ostrich)))
        if label_mode == "pos_rpy":
            errors.update(zip(("e_roll", "e_pitch", "e_yaw"), rpy_errors(T_twin, T_ostrich)))
        y = np.stack([errors[name] for name in LABEL_NAMES[label_mode]], axis=-1).astype(np.float32)

        _, map_index = np.unique(np.asarray(map_paths), return_inverse=True)  # one id per map path, across files

        keep = data["valid"].astype(bool)
        n_total = len(keep)
        if (~keep).any():
            print(f"[dataset] {self.source.name}: dropping {int((~keep).sum())}/{n_total} invalid row(s)")
        if drop_endpoint_infeasible:
            drop = keep & ~data["endpoint_feasible"].astype(bool)
            keep &= ~drop
            print(f"[dataset] {self.source.name}: dropping {int(drop.sum())} valid row(s) with an infeasible nominal end")
        if drop_twin_flagged:
            flagged = (data["twin_min_clearance"] <= 0.0) | (data["twin_max_residual"] > RESID_TOL)
            drop = keep & flagged
            keep &= ~drop
            print(f"[dataset] {self.source.name}: dropping {int(drop.sum())} valid row(s) the twin flags "
                  f"(clearance <= 0 or residual > {RESID_TOL:g})")
        if not keep.any():
            raise ValueError(f"{self.source.name}: no row left")

        def tensor(a: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
            return torch.as_tensor(np.ascontiguousarray(a[keep]), dtype=dtype, device=device)

        self.patch = tensor(data["patch"].reshape(-1, N_CHANNELS, ny, nx))  # [m, 1, ny, nx]
        self.command = tensor(data["command"])  # [m, 4]
        self.y = tensor(y)  # [m, K]
        self.map_index = tensor(map_index, torch.long)
        self.map_path = [p for p, k in zip(map_paths, keep) if k]
        self.sampling = [s for s, k in zip(data["sampling"], keep) if k]
        self.family = tensor(data["family"], torch.long)
        self.interact_dir = tensor(data["interact_dir"], torch.long)
        self.relief = tensor(data["relief"])
        self.endpoint_feasible = tensor(data["endpoint_feasible"], torch.bool)
        self.twin_min_clearance = tensor(data["twin_min_clearance"])
        self.twin_max_residual = tensor(data["twin_max_residual"])
        self.entry = tensor(data["entry"])  # [m, 2] (v, wz)
        self.jump = tensor(np.abs(data["entry"] - data["first_twist"]).max(axis=1) > JUMP_TOL, torch.bool)
        self.t0_twist = tensor(data["t0_twist"])  # [m, 3] (vx, vy, yaw_rate), realized
        self.origin_drift = tensor(data["origin_drift"])
        self.families = str(self.attrs.get("families", "wide,straight,narrow,spin")).split(",")
        self.label_mode = label_mode
        self.TARGET_NAMES = LABEL_NAMES[label_mode]

    def __len__(self) -> int:
        return self.patch.shape[0]

    def __getitem__(self, idx: int) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        return (self.patch[idx], self.command[idx]), self.y[idx]


def make_dataloaders(
    paths: pathlib.Path | str | Sequence[pathlib.Path | str],
    *,
    batch_size: int = 32,
    val_frac: float = 0.2,
    seed: int = 0,
    num_workers: int = 0,
    **ds_kwargs: object,
) -> tuple[DataLoader, DataLoader, WindowDivergenceDataset, Subset, Subset]:
    """(train_loader, val_loader, dataset, train_subset, val_subset), split by map. Only the target
    transform is ever fitted (`valid_targets(ds, train_subset.indices)`, the caller's job): the patch
    is already in wheel radii and the command is scaled inside the model."""
    ds = WindowDivergenceDataset(paths, **ds_kwargs)
    train_subset, val_subset = split_dataset_by_map(ds, val_frac=val_frac, seed=seed)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, ds, train_subset, val_subset


def _write_synthetic(path: pathlib.Path, maps: Sequence[str], rows_per_map: int, seed: int, **attrs: object) -> dict[str, np.ndarray]:
    """A minimal dataset_mppi_*.h5 with a known label: the twin ends at the identity and ostrich
    `offset` m ahead of it along x, so e_pos == offset and every rotation error is 0. Returns the
    per-row arrays the self-test checks against."""
    rng = np.random.default_rng(seed)
    n = len(maps) * rows_per_map
    twin = np.zeros((n, 7), np.float32)
    twin[:, 6] = 1.0
    offset = rng.uniform(0.0, 0.5, n)
    ostrich = twin.copy()
    omega = rng.uniform(0.0, 4.0, (n, 10, 3)).astype(np.float32)
    jump = rng.random(n) < 0.2
    entry = np.stack(wheels_to_twist(omega[:, 0].astype(np.float64)), axis=-1)
    entry[jump] += rng.uniform(0.1, 0.5, (int(jump.sum()), 2))
    ostrich[:, 0] += offset
    rows = dict(
        patch=rng.normal(0.0, 0.3, (n, PATCH_SPEC.ny * PATCH_SPEC.nx)).astype(np.float32),
        command=rng.normal(0.0, 1.0, (n, 4)).astype(np.float32),
        twin_pose=twin,
        valid=rng.random(n) > 0.1,
        family=rng.integers(0, 4, n).astype(np.int8),
        interact_dir=rng.integers(-1, 2, n).astype(np.int8),
        relief=rng.uniform(0.0, 0.3, n).astype(np.float32),
        endpoint_feasible=rng.random(n) > 0.3,
        twin_min_clearance=np.where(rng.random(n) > 0.1, 0.1, -0.01).astype(np.float32),
        twin_max_residual=np.where(rng.random(n) > 0.1, 0.0, 10 * RESID_TOL).astype(np.float32),
        entry=entry.astype(np.float32),
        omega=omega,
        t0_twist=rng.normal(0.0, 1.0, (n, 3)).astype(np.float32),
        origin_drift=rng.uniform(0.0, 0.1, n).astype(np.float32),
        map_index=np.repeat(np.arange(len(maps)), rows_per_map),
    )
    root = dict(window_s=1.0, mppi_dt=0.1, twin_solver="planning_solver", k_turn=1.0, ostrich_yaw_gain=1.15, mu=0.8,
                families="wide,straight,narrow,spin", **patch_spec_to_attrs(PATCH_SPEC))
    root.update(attrs)
    with h5py.File(path, "w") as f:
        for k, v in root.items():
            f.attrs[k] = v
        for k, v in rows.items():
            f.create_dataset(k, data=v)
        f.create_dataset("sampling", data=np.array(["edge", "uniform"] * (n // 2) + ["edge"] * (n % 2), dtype="S16"))
        f.create_dataset("map_path", data=np.array([maps[m] for m in rows["map_index"]], dtype="S64"))
        f.create_group("ostrich").create_dataset("pose", data=ostrich[None])  # [T = 1, n, 7]
    rows["offset"], rows["jump"] = offset, jump
    return rows


def _self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        a, b = pathlib.Path(tmp) / "dataset_mppi_a.h5", pathlib.Path(tmp) / "dataset_mppi_b.h5"
        rows_a = _write_synthetic(a, ["maps/m0", "maps/m1", "maps/m2"], 10, seed=0)
        rows_b = _write_synthetic(b, ["maps/m2", "maps/m3"], 10, seed=1)  # m2 is in both files
        valid = np.concatenate([rows_a["valid"], rows_b["valid"]])
        offset = np.concatenate([rows_a["offset"], rows_b["offset"]])[valid]

        for label_mode in LABEL_MODES:
            ds = WindowDivergenceDataset([a, b], label_mode=label_mode)
            assert len(ds) == int(valid.sum()) and ds.y.shape == (len(ds), len(LABEL_NAMES[label_mode]))
            assert ds.patch.shape[1:] == (1, 24, 36) and ds.command.shape[1] == 4
            assert np.allclose(ds.y[:, 0].numpy(), offset, atol=1e-5) and torch.allclose(ds.y[:, 1:], torch.zeros(1), atol=1e-5)
        print(f"[labels] twin identity vs ostrich offset -> e_pos = offset, rotation errors 0, in {LABEL_MODES}")

        jump = np.concatenate([rows_a["jump"], rows_b["jump"]])[valid]
        assert np.array_equal(ds.jump.numpy(), jump) and 0 < jump.sum() < len(jump)
        print(f"[jump] {int(jump.sum())}/{len(jump)} rows enter at an independent twist, recovered from entry vs omega[:, 0]")

        assert int(ds.map_index.max()) + 1 == 4, "m2 must be ONE map across the two files"
        for seed in range(5):
            train, val = split_dataset_by_map(ds, val_frac=0.25, seed=seed)
            train_maps = {ds.map_path[i] for i in train.indices}
            val_maps = {ds.map_path[i] for i in val.indices}
            assert not (train_maps & val_maps) and len(train) + len(val) == len(ds), (train_maps, val_maps)
        print("[split] 4 maps over 2 files (one shared), no map on both sides over 5 seeds")

        both = {k: np.concatenate([rows_a[k], rows_b[k]]) for k in ("endpoint_feasible", "twin_min_clearance", "twin_max_residual")}
        strict = WindowDivergenceDataset([a, b], drop_endpoint_infeasible=True)
        assert len(strict) == int((valid & both["endpoint_feasible"]).sum()) and bool(strict.endpoint_feasible.all())
        flagged = (both["twin_min_clearance"] <= 0) | (both["twin_max_residual"] > RESID_TOL)
        clean = WindowDivergenceDataset([a, b], drop_twin_flagged=True)
        assert len(clean) == int((valid & ~flagged).sum())
        assert bool((clean.twin_min_clearance > 0).all()) and bool((clean.twin_max_residual <= RESID_TOL).all())
        print(f"[filters] {len(ds)} valid rows; drop_endpoint_infeasible keeps {len(strict)}, drop_twin_flagged {len(clean)}")

        c = pathlib.Path(tmp) / "dataset_mppi_c.h5"
        _write_synthetic(c, ["maps/m9"], 4, seed=2, k_turn=0.6)
        try:
            WindowDivergenceDataset([a, c])
            raise AssertionError("files with different k_turn were merged")
        except ValueError as e:
            assert "k_turn" in str(e)
        print("[merge] files whose labels differ (k_turn 1.0 vs 0.6) are refused")

        train_loader, _, ds, train, _ = make_dataloaders([a, b], batch_size=4)
        (patch, command), y = next(iter(train_loader))
        assert patch.shape == (4, 1, 24, 36) and command.shape == (4, 4) and y.shape == (4, 2)
        assert valid_targets(ds, train.indices).shape == (len(train), 2)
        print("[loader] batches of (patch [B, 1, 24, 36], command [B, 4]), y [B, 2]")
    print("all self-checks ok")


def _report(paths: list[pathlib.Path], label_mode: str) -> None:
    from feasibility.mppi_learning.command import encode

    for path in paths:  # the stored command is command.encode of the stored wheel speeds
        with h5py.File(path, "r") as f:
            omega = f["omega"][()]  # [n, WINDOW_STEPS, 3]
            stored = f["command"][()]
        recomputed = encode(np.ascontiguousarray(omega.transpose(1, 0, 2)))
        assert np.allclose(stored, recomputed, atol=1e-5), np.abs(stored - recomputed).max()
    print(f"[command] stored command == command.encode(omega) in {len(paths)} file(s)")

    ds = WindowDivergenceDataset(paths, label_mode=label_mode)
    y = ds.y.numpy()
    print(f"{len(ds)} rows over {int(ds.map_index.max()) + 1} maps; y = ({', '.join(ds.TARGET_NAMES)})")
    header = "".join(f"{name + ' med':>12}{'q90':>7}" for name in ds.TARGET_NAMES)
    print(f"{'group':<22}{'n':>6}{header}")

    def row(name: str, mask: np.ndarray) -> None:
        if mask.any():
            stats = "".join(f"{np.median(y[mask, k]):12.3f}{np.quantile(y[mask, k], 0.9):7.3f}" for k in range(y.shape[1]))
            print(f"{name:<22}{int(mask.sum()):>6}{stats}")

    sampling = np.array(ds.sampling)
    for s in dict.fromkeys(ds.sampling):
        row(s, sampling == s)
    for d in (-1, 0, 1):
        row(f"interact_dir {d:+d}", ds.interact_dir.numpy() == d)
    for k, name in enumerate(ds.families):
        row(f"family {name}", ds.family.numpy() == k)
    row("all", np.ones(len(ds), bool))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", type=pathlib.Path, nargs="*", help="dataset_mppi_*.h5 file(s); none = synthetic self-test")
    parser.add_argument("--label-mode", type=str, default="pos_rot", choices=LABEL_MODES)
    args = parser.parse_args()
    if args.paths:
        _report(args.paths, args.label_mode)
    else:
        _self_test()
