"""Compares two `train.py` checkpoints' predictions on the SAME held-out rows -- written for the
original `model.WindowDivergenceNet` against `model_tensorRT_friendly.TrtFriendlyWindowDivergenceNet`
trained on the same data, but any two checkpoints of one dataset set work.

The two must share the split: the same set of dataset files, `val_frac`, `seed`, row filters and
`label_mode`, so one `WindowDivergenceDataset` and one held-out-MAP split serve both; anything else
is refused. Loading, the split and the scoring are `test_nn.py`'s (`load_dataset`, `resolve_split`,
lattice's `predict_over`). Per head it reports:

  * accuracy against the truth for each model, side by side: R^2 and mean |error|;
  * agreement between the two: mean / p95 / max |A - B|, Pearson r and Spearman rank r. The rank
    correlation is the one MPPI cares about, since its elite selection only sees the order of
    the costs;
  * all of it over every row, and split into rows that interact with terrain (`interact_dir` != 0)
    and those that do not.

CLI parameters:
    checkpoint_a, checkpoint_b  two .pt files (train.build_checkpoint schema); A is the reference
    --dataset PATH [PATH ...]   evaluate on these files instead of the checkpoints' own
    --split {train,val,all}     rows to score (default: val on the checkpoints' own files, else all)
    --batch-size INT            inference chunk size (default: 512)
    --device STR                torch device (default: cuda if available else cpu)
    --save-fig PATH             per-head scatter of B against A, coloured by the truth
    --self-test                 synthetic checkpoints + datasets in a temp dir, no GPU

Usage:
    python src/feasibility/mppi_learning/compare_checkpoints.py --self-test
    python src/feasibility/mppi_learning/compare_checkpoints.py outputs/checkpoints/a.pt outputs/checkpoints/a_trt.pt
"""
from __future__ import annotations

import argparse
import pathlib

import torch

from feasibility.lattice_learning.test_nn import predict_over
from feasibility.lattice_learning.train import r_squared
from feasibility.mppi_learning.custom_dataset import WindowDivergenceDataset
from feasibility.mppi_learning.test_nn import label_mismatches
from feasibility.mppi_learning.test_nn import load_dataset
from feasibility.mppi_learning.test_nn import resolve_split
from feasibility.mppi_learning.train import load_checkpoint

SPLIT_KEYS = ("val_frac", "seed")  # in ckpt["args"]
SPLIT_FIELDS = ("label_mode", "drop_endpoint_infeasible", "drop_twin_flagged")  # top-level ckpt keys


def split_mismatches(a: dict[str, object], b: dict[str, object]) -> list[str]:
    """Why the two checkpoints' held-out rows would differ, as printable lines (empty: they agree)."""
    out = []
    files_a = {pathlib.Path(p).resolve() for p in a["dataset_paths"]}
    files_b = {pathlib.Path(p).resolve() for p in b["dataset_paths"]}
    if files_a != files_b:
        out.append(f"dataset files: {sorted(map(str, files_a))} vs {sorted(map(str, files_b))}")
    out += [f"{k}: {a['args'][k]!r} vs {b['args'][k]!r}" for k in SPLIT_KEYS if a["args"][k] != b["args"][k]]
    out += [f"{k}: {a[k]!r} vs {b[k]!r}" for k in SPLIT_FIELDS if a[k] != b[k]]
    return out


def _ranks(x: torch.Tensor) -> torch.Tensor:
    """[n, K] -> per-column ranks (ties broken by order, fine for continuous predictions)."""
    return x.argsort(dim=0).argsort(dim=0).to(x.dtype)


def pearson(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """[n, K] x2 -> [K] Pearson correlation per column."""
    a, b = a - a.mean(dim=0), b - b.mean(dim=0)
    return (a * b).sum(dim=0) / (a.norm(dim=0) * b.norm(dim=0)).clamp_min(1e-12)


def compare(pred_a: torch.Tensor, pred_b: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-head accuracy of each model and their agreement, every value [K]."""
    diff = (pred_a - pred_b).abs()
    return {
        "r2_a": r_squared(pred_a, target), "r2_b": r_squared(pred_b, target),
        "mae_a": (pred_a - target).abs().mean(dim=0), "mae_b": (pred_b - target).abs().mean(dim=0),
        "diff_mean": diff.mean(dim=0), "diff_p95": diff.quantile(0.95, dim=0), "diff_max": diff.max(dim=0).values,
        "pearson": pearson(pred_a, pred_b), "spearman": pearson(_ranks(pred_a), _ranks(pred_b)),
    }


def print_table(title: str, stats: dict[str, torch.Tensor], names: tuple[str, ...], n: int) -> None:
    print(f"\n[{title}] {n} rows")
    print(f"  {'head':8s} {'R2 A':>7s} {'R2 B':>7s} {'|err| A':>8s} {'|err| B':>8s}   "
          f"{'|A-B| mean':>10s} {'p95':>7s} {'max':>7s} {'pearson':>8s} {'spearman':>8s}")
    for k, name in enumerate(names):
        s = {key: v[k].item() for key, v in stats.items()}
        print(f"  {name:8s} {s['r2_a']:7.3f} {s['r2_b']:7.3f} {s['mae_a']:8.4f} {s['mae_b']:8.4f}   "
              f"{s['diff_mean']:10.4f} {s['diff_p95']:7.4f} {s['diff_max']:7.4f} {s['pearson']:8.4f} {s['spearman']:8.4f}")


def subsets(ds: WindowDivergenceDataset, indices: list[int]) -> dict[str, torch.Tensor]:
    """Row masks over `indices`: all, interacting with terrain, not interacting."""
    interacting = (ds.interact_dir[torch.as_tensor(indices, device=ds.interact_dir.device)] != 0).cpu()
    return {"all": torch.ones_like(interacting), "interacting": interacting, "not interacting": ~interacting}


def make_figure(pred_a: torch.Tensor, pred_b: torch.Tensor, target: torch.Tensor, names: tuple[str, ...],
                labels: tuple[str, str], path: pathlib.Path | None) -> None:
    """One panel per head: B's prediction against A's, coloured by the truth, with y = x."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.5), squeeze=False)
    for k, (ax, name) in enumerate(zip(axes[0], names)):
        hi = float(torch.maximum(pred_a[:, k], pred_b[:, k]).max())
        points = ax.scatter(pred_a[:, k], pred_b[:, k], c=target[:, k], s=3, cmap="viridis", vmax=hi)
        ax.plot([0, hi], [0, hi], "k--", lw=0.8)
        ax.set(xlabel=f"A: {labels[0]}", ylabel=f"B: {labels[1]}", title=name, aspect="equal")
        fig.colorbar(points, ax=ax, label=f"true {name}")
    fig.tight_layout()
    if path is None:
        plt.show()
    else:
        fig.savefig(path, dpi=150)
        print(f"[figure] {path}")


def run(path_a: pathlib.Path, path_b: pathlib.Path, dataset: list[pathlib.Path] | None, split: str | None,
        device: torch.device, batch_size: int, save_fig: pathlib.Path | None,
        show: bool = True) -> dict[str, dict[str, torch.Tensor]]:
    model_a, ckpt_a = load_checkpoint(path_a, device)
    model_b, ckpt_b = load_checkpoint(path_b, device)
    problems = split_mismatches(ckpt_a, ckpt_b)
    if problems:
        raise ValueError("the checkpoints were not trained on the same split:\n  " + "\n  ".join(problems))
    paths = dataset or [pathlib.Path(p) for p in ckpt_a["dataset_paths"]]
    ds = load_dataset(ckpt_a, paths, device)
    for line in label_mismatches(ckpt_a, ds):
        print(f"[WARNING] labels differ from training -- {line}")
    indices, split = resolve_split(ckpt_a, paths, split, ds)
    for name, (model, ckpt) in (("A", (model_a, ckpt_a)), ("B", (model_b, ckpt_b))):
        model.eval()
        print(f"[{name}] {(path_a if name == 'A' else path_b).name}: {type(model).__name__}, trunk "
              f"{ckpt.get('trunk', 'original')}, epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f}")
    print(f"[dataset] {ds.source.name}: {len(indices)} rows (split={split})")

    pred_a, target = predict_over(model_a, ds, indices, device, batch_size)
    pred_b, _ = predict_over(model_b, ds, indices, device, batch_size)
    names = model_a.target_names
    results = {}
    for title, mask in subsets(ds, indices).items():
        if mask.sum() < 2:
            continue
        results[title] = compare(pred_a[mask], pred_b[mask], target[mask])
        print_table(title, results[title], names, int(mask.sum()))
    if save_fig is not None or show:
        make_figure(pred_a, pred_b, target, names, (path_a.stem, path_b.stem), save_fig)
    return results


def self_test() -> None:
    """Two synthetic checkpoints (original and trt_friendly, untrained) on synthetic files: the table
    runs, a model against itself agrees exactly, and a checkpoint on another split is refused."""
    import tempfile

    from feasibility.mppi_learning.custom_dataset import _write_synthetic
    from feasibility.mppi_learning.custom_dataset import split_dataset_by_map
    from feasibility.mppi_learning.custom_dataset import valid_targets
    from feasibility.mppi_learning.model import TargetTransform
    from feasibility.mppi_learning.model import WindowDivergenceNet
    from feasibility.mppi_learning.model_tensorRT_friendly import TrtFriendlyWindowDivergenceNet
    from feasibility.mppi_learning.train import build_checkpoint

    device = torch.device("cpu")
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = pathlib.Path(tmp_str)
        a = tmp / "dataset_mppi_a.h5"
        _write_synthetic(a, [f"maps/m{i}" for i in range(5)], 16, seed=0)
        ds = WindowDivergenceDataset([a])
        train_subset, _ = split_dataset_by_map(ds, val_frac=0.4, seed=0)
        transform = TargetTransform.fit(valid_targets(ds, train_subset.indices))

        def save(net_class: type, name: str, seed: int) -> pathlib.Path:
            torch.manual_seed(seed)
            model = net_class(target_transform=transform)
            fake_args = argparse.Namespace(base_width=32, embed_dim=64, head_fusion="film", blur_terrain=False,
                                           drop_endpoint_infeasible=False, drop_twin_flagged=False,
                                           dataset=[a], val_frac=0.4, seed=seed)
            path = tmp / f"{name}.pt"
            torch.save(build_checkpoint(model, torch.optim.AdamW(model.parameters()), 1, 0.0, ds, fake_args), path)
            return path

        original = save(WindowDivergenceNet, "original", 0)
        trt = save(TrtFriendlyWindowDivergenceNet, "trt", 0)
        results = run(original, trt, None, None, device, 16, tmp / "fig.png", show=False)
        assert (tmp / "fig.png").stat().st_size > 0 and set(results) >= {"all"}
        assert all(torch.isfinite(v).all() for v in results["all"].values())

        same = run(original, original, None, None, device, 16, None, show=False)["all"]
        assert torch.all(same["diff_max"] == 0) and torch.allclose(same["spearman"], torch.ones(2))
        print("[self-test] a checkpoint against itself: |A-B| = 0, rank r = 1")

        other = save(WindowDivergenceNet, "other_seed", 1)
        try:
            run(original, other, None, None, device, 16, None, show=False)
            raise AssertionError("checkpoints on different splits were compared")
        except ValueError as error:
            assert "seed" in str(error), error
        print("[self-test] checkpoints on different splits are refused")
    print("all self-checks ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint_a", type=pathlib.Path, nargs="?", help="reference checkpoint .pt")
    parser.add_argument("checkpoint_b", type=pathlib.Path, nargs="?", help="checkpoint .pt compared against A")
    parser.add_argument("--dataset", type=pathlib.Path, nargs="+", default=None, help="dataset_mppi_*.h5 file(s) (default: the checkpoints' own)")
    parser.add_argument("--split", type=str, default=None, choices=("train", "val", "all"), help="rows to score (default: val on the checkpoints' own files, else all)")
    parser.add_argument("--batch-size", type=int, default=512, help="inference chunk size (default: 512)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="torch device")
    parser.add_argument("--save-fig", type=pathlib.Path, default=None, help="write the figure here instead of showing it")
    parser.add_argument("--self-test", action="store_true", help="synthetic checkpoints + datasets; assert this module's logic")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.checkpoint_b is None:
        parser.error("two checkpoints are required unless --self-test is given")
    run(args.checkpoint_a, args.checkpoint_b, args.dataset, args.split, torch.device(args.device),
        args.batch_size, args.save_fig)


if __name__ == "__main__":
    main()
