"""Standalone checkpoint evaluator for `train.WindowDivergenceNet` checkpoints: loads a `.pt` plus
the dataset_mppi_*.h5 file(s) it trained on (or others), scores predictions against ground truth
and plots them -- `lattice_learning/test_nn.py` for this tree. Its scoring and plotting (batched
`predict_over`, R^2 and mean |error| per head, the bar chart of mean |error| over equal-population
bins of the TRUE value) are imported unchanged. What is local is what depends on the checkpoint and
dataset schema:

* a checkpoint may have trained on SEVERAL files (`ckpt["dataset_paths"]`), so `--dataset` takes
  several, and "the checkpoint's own data" means the same SET of files (order does not matter: map
  ids come from the sorted map paths, so the held-out maps are the same either way);
* the dataset is rebuilt with the checkpoint's own `drop_endpoint_infeasible` / `drop_twin_flagged`,
  since the held-out-MAP split was drawn over the rows that survived them;
* the data's `LABEL_ATTRS` (k_turn, ostrich yaw gain, mu, ...) are compared with the checkpoint's
  `label_attrs`, and a mismatch is printed as a WARNING: the net would be scored against labels
  that mean something else.

`train.py`'s `final_report` already owns the full battery (baselines, subsets, mirror check) at the
end of a run; this script is the quick re-score of a saved checkpoint.

CLI parameters:
    checkpoint                path to a .pt file (train.build_checkpoint schema)
    --dataset PATH [PATH ...] dataset_mppi_*.h5 file(s) to evaluate on (default: the checkpoint's
                              own ckpt["dataset_paths"])
    --split {train,val,all}   rows to score (default: "val" -- the checkpoint's held-out-MAP split,
                              rebuilt from its stored val_frac/seed -- when --dataset is the
                              checkpoint's own set of files, "all" otherwise)
    --batch-size INT          inference chunk size (default: 512)
    --device STR              torch device (default: cuda if available else cpu)
    --save-fig PATH           write the figure here (dpi=150) instead of showing it
    --self-test               synthetic checkpoint + datasets in a temp dir, no GPU

Usage:
    python src/feasibility/mppi_learning/test_nn.py --self-test
    python src/feasibility/mppi_learning/test_nn.py outputs/checkpoints/x.pt
    python src/feasibility/mppi_learning/test_nn.py outputs/checkpoints/x.pt \\
        --dataset outputs/dataset_mppi_other.h5 --save-fig outputs/test_nn_x.png
"""
from __future__ import annotations

import argparse
import pathlib
from collections.abc import Sequence

import torch

from feasibility.lattice_learning.test_nn import make_figure
from feasibility.lattice_learning.test_nn import predict_over
from feasibility.lattice_learning.test_nn import report
from feasibility.mppi_learning.custom_dataset import _same
from feasibility.mppi_learning.custom_dataset import LABEL_ATTRS
from feasibility.mppi_learning.custom_dataset import split_dataset_by_map
from feasibility.mppi_learning.custom_dataset import WindowDivergenceDataset
from feasibility.mppi_learning.train import label_attrs
from feasibility.mppi_learning.train import load_checkpoint


def load_dataset(
    ckpt: dict[str, object], paths: Sequence[pathlib.Path], device: torch.device
) -> WindowDivergenceDataset:
    """`paths` loaded with the checkpoint's label mode and row filters, so the rows (and hence the
    held-out-MAP split) are the ones training saw."""
    return WindowDivergenceDataset(
        paths, device=device, label_mode=ckpt["label_mode"],
        drop_endpoint_infeasible=bool(ckpt["drop_endpoint_infeasible"]),
        drop_twin_flagged=bool(ckpt["drop_twin_flagged"]),
    )


def label_mismatches(ckpt: dict[str, object], ds: WindowDivergenceDataset) -> list[str]:
    """The `LABEL_ATTRS` on which the data and the checkpoint disagree, as printable lines."""
    trained, data = ckpt["label_attrs"], label_attrs(ds)
    return [f"{k}: checkpoint {trained.get(k)!r}, data {data[k]!r}"
            for k in LABEL_ATTRS if not _same(trained.get(k), data[k])]


def resolve_split(
    ckpt: dict[str, object],
    paths: Sequence[pathlib.Path],
    requested_split: str | None,
    ds: WindowDivergenceDataset,
) -> tuple[list[int], str]:
    """"val" (the checkpoint's own held-out-MAP split, from its stored val_frac/seed) when `paths`
    is the checkpoint's own set of files, "all" otherwise -- a foreign file split by the same seed
    would carve out an arbitrary subset of ITS maps, not a leakage-safe one."""
    same_files = {p.resolve() for p in paths} == {pathlib.Path(p).resolve() for p in ckpt["dataset_paths"]}
    split = requested_split or ("val" if same_files else "all")
    if split == "all":
        return list(range(len(ds))), split
    train_subset, val_subset = split_dataset_by_map(
        ds, val_frac=float(ckpt["args"]["val_frac"]), seed=int(ckpt["args"]["seed"])
    )
    return (val_subset.indices if split == "val" else train_subset.indices), split


def self_test() -> None:
    """A synthetic pair of files and a hand-assembled checkpoint (no training -- train.py's own
    self-test covers fitting): the default split, the checkpoint's filters, the label-attr check and
    the figure. The scoring/binning helpers are lattice's and tested there."""
    import shutil
    import tempfile

    from feasibility.mppi_learning.custom_dataset import _write_synthetic
    from feasibility.mppi_learning.custom_dataset import valid_targets
    from feasibility.mppi_learning.model import TargetTransform
    from feasibility.mppi_learning.model import WindowDivergenceNet
    from feasibility.mppi_learning.train import build_checkpoint

    device = torch.device("cpu")
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = pathlib.Path(tmp_str)
        a, b = tmp / "dataset_mppi_a.h5", tmp / "dataset_mppi_b.h5"
        _write_synthetic(a, [f"maps/m{i}" for i in range(4)], 16, seed=0)
        _write_synthetic(b, ["maps/m3", "maps/m4", "maps/m5"], 16, seed=1)

        for drop in (False, True):
            ds = WindowDivergenceDataset([a, b], drop_endpoint_infeasible=drop, drop_twin_flagged=drop)
            train_subset, val_subset = split_dataset_by_map(ds, val_frac=0.2, seed=0)
            model = WindowDivergenceNet(target_transform=TargetTransform.fit(valid_targets(ds, train_subset.indices)))
            fake_args = argparse.Namespace(
                base_width=32, embed_dim=64, head_fusion="film", blur_terrain=False,
                drop_endpoint_infeasible=drop, drop_twin_flagged=drop, dataset=[a, b], val_frac=0.2, seed=0,
            )
            ckpt_path = tmp / f"checkpoint_drop{int(drop)}.pt"
            torch.save(build_checkpoint(model, torch.optim.AdamW(model.parameters()), 1, 0.0, ds, fake_args), ckpt_path)

            model2, ckpt = load_checkpoint(ckpt_path, device)
            paths = [pathlib.Path(p) for p in ckpt["dataset_paths"]]
            ds2 = load_dataset(ckpt, paths, device)
            assert len(ds2) == len(ds), (len(ds2), len(ds))  # the checkpoint's filters were applied
            indices, split = resolve_split(ckpt, paths, None, ds2)
            assert split == "val" and set(indices) == set(val_subset.indices), "val split did not reconstruct"

            # the same files in the other order: still "val", still the same held-out maps
            ds_rev = load_dataset(ckpt, paths[::-1], device)
            idx_rev, split_rev = resolve_split(ckpt, paths[::-1], None, ds_rev)
            assert split_rev == "val"
            assert {ds_rev.map_path[i] for i in idx_rev} == {ds.map_path[i] for i in val_subset.indices}
            assert not label_mismatches(ckpt, ds2)
            print(f"[self-test] filters={drop}: {len(ds2)} rows, default split -> 'val' ({len(indices)} rows), "
                  "file order irrelevant")

        # one of the files alone is not the checkpoint's data -> "all"
        copy = tmp / "dataset_mppi_copy.h5"
        shutil.copy2(a, copy)
        ds3 = load_dataset(ckpt, [copy], device)
        idx_all, split_all = resolve_split(ckpt, [copy], None, ds3)
        assert split_all == "all" and len(idx_all) == len(ds3)
        print(f"[self-test] a different file -> 'all' ({len(idx_all)} rows)")

        # data generated at another k_turn is flagged
        other = tmp / "dataset_mppi_k06.h5"
        _write_synthetic(other, ["maps/m9"], 8, seed=2, k_turn=0.6)
        mismatch = label_mismatches(ckpt, load_dataset(ckpt, [other], device))
        assert len(mismatch) == 1 and mismatch[0] == "k_turn: checkpoint 1.0, data 0.6", mismatch
        print(f"[self-test] label-attr mismatch reported: {mismatch[0]}")

        pred, target = predict_over(model2, ds2, indices, device, batch_size=16)
        stats = report(pred, target, model2.target_names)
        fig_path = tmp / "fig.png"
        make_figure(pred, target, model2.target_names, "self-test", stats, fig_path)
        assert fig_path.exists() and fig_path.stat().st_size > 0
        print("[self-test] make_figure() writes a PNG")
    print("all self-checks ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=pathlib.Path, nargs="?", help="checkpoint .pt path (train.build_checkpoint schema)")
    parser.add_argument("--dataset", type=pathlib.Path, nargs="+", default=None, help="dataset_mppi_*.h5 file(s) to evaluate on (default: the checkpoint's own)")
    parser.add_argument("--split", type=str, default=None, choices=("train", "val", "all"), help="rows to score (default: val on the checkpoint's own files, else all)")
    parser.add_argument("--batch-size", type=int, default=512, help="inference chunk size (default: 512)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="torch device (default: cuda if available else cpu)")
    parser.add_argument("--save-fig", type=pathlib.Path, default=None, help="write the figure here instead of showing it interactively")
    parser.add_argument("--self-test", action="store_true", help="synthetic checkpoint + datasets in a temp dir; assert this module's own logic")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if args.checkpoint is None:
        parser.error("checkpoint is required unless --self-test is given")

    device = torch.device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    model.eval()

    paths = args.dataset or [pathlib.Path(p) for p in ckpt["dataset_paths"]]
    ds = load_dataset(ckpt, paths, device)
    for line in label_mismatches(ckpt, ds):
        print(f"[WARNING] labels differ from training -- {line}")
    indices, split = resolve_split(ckpt, paths, args.split, ds)

    print(f"[checkpoint] {args.checkpoint.name}: epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f}, "
          f"label_mode={ckpt['label_mode']}, filters: endpoint-infeasible "
          f"{'dropped' if ckpt['drop_endpoint_infeasible'] else 'kept'}, twin-flagged "
          f"{'dropped' if ckpt['drop_twin_flagged'] else 'kept'}")
    print(f"[dataset]    {ds.source.name}: {len(indices)} rows scored (split={split})")

    pred, target = predict_over(model, ds, indices, device, args.batch_size)
    stats = report(pred, target, model.target_names)
    make_figure(pred, target, model.target_names, args.checkpoint.name, stats, args.save_fig)


if __name__ == "__main__":
    main()
