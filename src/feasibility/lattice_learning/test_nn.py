"""Standalone checkpoint evaluator for `train.ArcDivergenceNet` checkpoints: loads a `.pt` +
a `dataset_arc_*.h5`, scores predictions against ground truth, and plots them.

`train.py`'s own `final_report` already owns the full battery (three baselines, top-decile,
blocked/unblocked split, mirror-equivariance) at the end of a training run, and
`planning/tune_pivot_tau.py` already extracts one head (`e_pitch`) from a checkpoint's held-out
pivots to tune a planning threshold. This script generalizes that second recipe -- reload the
dataset the checkpoint trained on, rebuild its held-out-MAP val split from the checkpoint's own
stored `args`, run batched `model.predict` -- to every target head at once, and reports just the
two numbers requested: **R^2** and **mean |prediction error|** per head. The figure itself is a
bar chart per head: the x axis splits the TRUE target value into `n_bins` (default 3) equal-
population intervals (its own tertiles, so the split adapts to each head's units and its
right-skewed shape -- most rows near zero, a thin high-error tail -- instead of one hard-coded
set of thresholds that would only fit one head/checkpoint), and the y axis is the mean
|prediction error| of the rows whose true value falls in that interval -- where a head's error
actually concentrates, rather than a scatter of every row. R^2 stays in each subplot's title as
text; there is no scatter/y=x plot to show it visually anymore. No fresh simulation (unlike
`grid_learning_2/test_nn.py`, which has no pre-existing held-out file to read): every
`dataset_arc_*.h5` already holds real labels, so this is a plain reload + inference pass,
argparse like `train.py`, not Hydra.

CLI parameters:
    checkpoint        path to a .pt file (train.build_checkpoint schema)
    --dataset PATH    dataset_arc_*.h5 to evaluate on (default: the checkpoint's own
                      ckpt["dataset_path"])
    --split {train,val,all}
                      rows to score (default: resolved at runtime -- "val" when --dataset
                      resolves to the same file the checkpoint trained on, reconstructing the
                      held-out-MAP split from the checkpoint's own args; "all" when a different
                      file is given, since that file's "val split" by the same seed would just be
                      an arbitrary map subset of IT, not a leakage-safe one)
    --batch-size INT  inference chunk size (default: 512)
    --device STR      torch device (default: cuda if available else cpu)
    --save-fig PATH   write the figure here (dpi=150) instead of showing it interactively
    --self-test       build a synthetic checkpoint + dataset in a temp dir and assert this
                      module's own logic (no GPU, no real dataset needed)

Usage:
    python src/feasibility/lattice_learning/test_nn.py --self-test
    python src/feasibility/lattice_learning/test_nn.py outputs/checkpoints/x.pt
    python src/feasibility/lattice_learning/test_nn.py outputs/checkpoints/x.pt \\
        --dataset outputs/dataset_arc_other.h5 --save-fig outputs/test_nn_x.png
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch

from feasibility.lattice_learning.custom_dataset import ArcDivergenceDataset
from feasibility.lattice_learning.custom_dataset import split_dataset_by_map
from feasibility.lattice_learning.train import fmt
from feasibility.lattice_learning.train import load_checkpoint
from feasibility.lattice_learning.train import r_squared


@torch.no_grad()
def predict_over(
    model: torch.nn.Module,
    ds: ArcDivergenceDataset,
    indices: list[int],
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched `model.predict` over `ds` rows at `indices` -> (pred [n, K], true [n, K]), both CPU,
    physical units -- the same chunking loop `tune_pivot_tau.pivot_predictions` uses, generalized
    to every head instead of indexing one out."""
    idx = torch.as_tensor(indices, device=device)
    preds = [
        model.predict(ds.patch[idx[i : i + batch_size]], ds.command[idx[i : i + batch_size]])
        for i in range(0, len(idx), batch_size)
    ]
    pred = torch.cat(preds) if preds else torch.empty(0, len(model.target_names))
    return pred.cpu(), ds.y[idx].cpu()


def mean_abs_error_per_head(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """[K]: mean |pred - true| per head -- the "mean prediction error" half of the simplified
    report, alongside `train.r_squared` for R^2."""
    return (pred - target).abs().mean(dim=0)


# Physical units of each LABEL_NAMES column (custom_dataset.se3_errors/rpy_errors): e_pos is a
# Euclidean distance in meters, every rotation error is a radian angle -- used only for axis
# labels below, so a head this dict doesn't know about still plots, just unitless.
UNITS: dict[str, str] = {"e_pos": "m", "e_rot": "rad", "e_roll": "rad", "e_pitch": "rad", "e_yaw": "rad"}

# The y-axis label and the "R^2=..." title segment share this color so the two headline numbers
# stand out together when the figure is pasted small into a slide -- a darker red than matplotlib's
# plain "red", but not as dark as "darkred"/"maroon".
ACCENT_COLOR = "firebrick"

# make_figure()'s matplotlib rcParams, well above the library defaults so the figure is still
# readable pasted small into a slide (the original request this module's plot was built for).
FIGURE_STYLE: dict[str, float] = {
    "font.size": 16,
    "axes.titlesize": 19,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 15,
    "figure.titlesize": 22,
}


def quantile_bin_edges(values: np.ndarray, n_bins: int = 3) -> np.ndarray:
    """[n_bins + 1] bin edges over `values`' own quantiles (equal-population bins), so the split
    adapts to each head's actual distribution instead of one hard-coded set of thresholds -- an
    e_pos in meters and an e_rot in radians need different intervals, and even two checkpoints'
    e_pos don't share a scale. `edges[0]`/`edges[-1]` are exactly `values.min()`/`.max()` (what
    `np.quantile` already returns at q=0/1), so every row falls inside `[edges[0], edges[-1]]`."""
    return np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))


def binned_mean_abs_error(
    true: np.ndarray, pred: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(true, pred, edges [n_bins + 1]) -> (mean |pred - true| per bin [n_bins], row count per bin
    [n_bins]). Bin b covers [edges[b], edges[b+1]), except the last bin, which is closed on the
    right so `values.max()` (bin b's own upper edge) is never left out. An empty bin's mean is
    NaN, not 0 -- a 0 there would read as "the net nailed this range" instead of "no row landed in
    it", the same reasoning `train.rmse_per_head` uses for an empty row selector."""
    n_bins = len(edges) - 1
    idx = np.digitize(true, edges[1:-1], right=False).clip(0, n_bins - 1)
    mean_err = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        sel = idx == b
        counts[b] = int(sel.sum())
        if counts[b] > 0:
            mean_err[b] = np.abs(pred[sel] - true[sel]).mean()
    return mean_err, counts


def resolve_split(
    ckpt: dict[str, object],
    dataset_path: pathlib.Path,
    requested_split: str | None,
    ds: ArcDivergenceDataset,
) -> tuple[list[int], str]:
    """Applies the --split default-resolution rule: "val" (the checkpoint's own held-out-MAP
    split, reconstructed from its stored `args["val_frac"]`/`args["seed"]`) when `dataset_path`
    is the file the checkpoint trained on, "all" otherwise -- a foreign file split by the same
    seed would just carve out an arbitrary map subset of ITS OWN maps, not a leakage-safe one."""
    same_file = dataset_path.resolve() == pathlib.Path(ckpt["dataset_path"]).resolve()
    split = requested_split or ("val" if same_file else "all")
    if split == "all":
        return list(range(len(ds))), split
    train_subset, val_subset = split_dataset_by_map(
        ds, val_frac=float(ckpt["args"]["val_frac"]), seed=int(ckpt["args"]["seed"])
    )
    return (val_subset.indices if split == "val" else train_subset.indices), split


def report(
    pred: torch.Tensor, target: torch.Tensor, names: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    """Prints and returns {"r2": [K], "mae": [K]} -- the entire simplified report."""
    r2 = r_squared(pred, target)
    mae = mean_abs_error_per_head(pred, target)
    print(f"[test] {pred.shape[0]} rows")
    print(f"[test] R^2:          {fmt(r2, names)}")
    print(f"[test] mean|error|:  {fmt(mae, names)}")
    return {"r2": r2, "mae": mae}


def rainbow_title(ax, segments: list[tuple[str, str]], fontsize: float) -> None:
    """Place `segments` ([(text, color), ...]) as one horizontally-concatenated, centered title
    above `ax`, each segment in its own color. A matplotlib Text has one color for its whole
    string (mathtext has no working `\\color` command to fake it either -- checked, it raises
    ParseFatalException), so a title mixing colors needs one Text per segment, positioned by its
    predecessor's rendered width -- the matplotlib "rainbow text" cookbook recipe, centered here
    by measuring every segment first and shifting the whole run left by half its total width.
    Added via `ax.text` (not `ax.set_title`), so `fig.tight_layout()` still reserves room for it
    as long as it's called AFTER this."""
    fig = ax.figure
    texts = [
        ax.text(0.0, 1.0, text, transform=ax.transAxes, color=color, fontsize=fontsize,
                 va="bottom", ha="left")
        for text, color in segments
    ]
    renderer = fig.canvas.get_renderer()
    ax_width = ax.get_window_extent(renderer=renderer).width
    widths = [t.get_window_extent(renderer=renderer).width / ax_width for t in texts]
    x = 0.5 - sum(widths) / 2.0
    for t, w in zip(texts, widths):
        t.set_position((x, 1.0))
        x += w


def make_figure(
    pred: torch.Tensor,
    target: torch.Tensor,
    names: tuple[str, ...],
    ckpt_name: str,
    stats: dict[str, torch.Tensor],
    save_fig: pathlib.Path | None,
    n_bins: int = 3,
) -> None:
    """One row of bar charts, one per head: `n_bins` equal-population intervals of the TRUE target
    value on the x axis, mean |prediction error| of the rows in that interval on the y axis --
    `quantile_bin_edges`/`binned_mean_abs_error` do the binning. Each subplot's title still carries
    that head's overall R^2 and mean|error| (report()'s numbers) as text.

    Font sizes are set well above matplotlib's defaults (`FIGURE_STYLE`) so the figure survives
    being pasted into a slide and shrunk -- tick numbers, axis labels and titles all stay legible
    small, not just at native resolution. The y-axis label and the title's "R^2=..." segment are
    in `ACCENT_COLOR` so the two headline numbers stand out (`rainbow_title` for the mixed-color
    title, since a plain `ax.set_title` string can only be one color)."""
    import matplotlib

    if save_fig is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = len(names)
    with plt.rc_context(FIGURE_STYLE):
        fig, axes = plt.subplots(1, k, figsize=(6.5 * k, 6.0), squeeze=False)
        for i, name in enumerate(names):
            ax = axes[0, i]
            t, p = target[:, i].numpy(), pred[:, i].numpy()
            edges = quantile_bin_edges(t, n_bins)
            mean_err, counts = binned_mean_abs_error(t, p, edges)
            unit = UNITS.get(name, "")
            labels = [
                f"[{edges[b]:.3g}, {edges[b + 1]:.3g}]\n(n={counts[b]})" for b in range(n_bins)
            ]
            bars = ax.bar(range(n_bins), mean_err, color="tab:blue")
            ax.bar_label(bars, fmt="%.3f", fontsize=FIGURE_STYLE["font.size"], padding=4)
            ax.set_xticks(range(n_bins))
            ax.set_xticklabels(labels)
            ax.set_xlabel(f"true {name} interval" + (f" [{unit}]" if unit else ""))
            ax.set_ylabel(
                f"mean |error| ({name})" + (f" [{unit}]" if unit else ""), color=ACCENT_COLOR
            )
            # Plain y-tick labels, no "1e-N" scale offset: that offset text sits in the same
            # top-left corner as rainbow_title's manually-positioned Text objects (matplotlib only
            # dodges it for the standard ax.set_title mechanism, not for arbitrary added text), so
            # on a near-zero head it would silently overlap the title instead.
            ax.ticklabel_format(style="plain", axis="y")
            ax.margins(y=0.15)  # headroom so bar_label's text never clips the top axis
            rainbow_title(
                ax,
                [
                    (f"{name}: R^2=", "black"),
                    (f"{stats['r2'][i]:.3f}", ACCENT_COLOR),
                    (f", mean|err|={stats['mae'][i]:.4f}", "black"),
                ],
                fontsize=FIGURE_STYLE["axes.titlesize"],
            )
        fig.suptitle(ckpt_name)
        fig.tight_layout()
        if save_fig is not None:
            fig.savefig(save_fig, dpi=150)
            print(f"[figure] saved to {save_fig}")
        else:
            plt.show()


def self_test() -> None:
    """No GPU/real dataset needed, following train.py's own --self-test pattern: a synthetic
    dataset + a hand-assembled checkpoint (no training loop -- this script's job is to test
    loading/scoring/plotting, not fitting, which train.py's own self-test already covers)."""
    import shutil
    import tempfile

    from feasibility.lattice_learning.custom_dataset import _write_synthetic
    from feasibility.lattice_learning.custom_dataset import valid_targets
    from feasibility.lattice_learning.model import ArcDivergenceNet
    from feasibility.lattice_learning.model import DEFAULT_BASE_WIDTH
    from feasibility.lattice_learning.model import DEFAULT_EMBED_DIM
    from feasibility.lattice_learning.model import TargetTransform
    from feasibility.lattice_learning.train import build_checkpoint

    device = torch.device("cpu")
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = pathlib.Path(tmp_str)
        path = tmp / "dataset_arc_synthetic.h5"
        _write_synthetic(path, n_maps=6, rows_per_map=16)

        ds = ArcDivergenceDataset(path)
        train_subset, val_subset = split_dataset_by_map(ds, val_frac=0.2, seed=0)
        target_transform = TargetTransform.fit(valid_targets(ds, train_subset.indices))
        model = ArcDivergenceNet(
            patch_spec=ds.patch_spec, command_mode=ds.command_mode, label_mode=ds.label_mode,
            target_transform=target_transform,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        fake_args = argparse.Namespace(
            base_width=DEFAULT_BASE_WIDTH, embed_dim=DEFAULT_EMBED_DIM, head_fusion="film",
            blur_terrain=False, drop_blocked_endpoints=False, val_frac=0.2, seed=0,
        )
        ckpt = build_checkpoint(model, optimizer, epoch=1, val_loss=0.0, ds=ds, args=fake_args)
        ckpt_path = tmp / "checkpoint.pt"
        torch.save(ckpt, ckpt_path)

        # --- default split resolves to "val" when --dataset matches the checkpoint's own file --
        model2, ckpt2 = load_checkpoint(ckpt_path, device)
        dataset_path = pathlib.Path(ckpt2["dataset_path"])
        ds2 = ArcDivergenceDataset(
            dataset_path, label_mode=ckpt2["label_mode"], command_mode=ckpt2["command_mode"],
            device=device,
        )
        indices_val, split_val = resolve_split(ckpt2, dataset_path, None, ds2)
        assert split_val == "val", split_val
        assert set(indices_val) == set(val_subset.indices), "val split did not reconstruct exactly"
        print(f"[self-test] default split -> 'val' ({len(indices_val)} rows) for the checkpoint's own file")

        # --- a differently-pathed (but same-content) file defaults to "all" ---------------------
        path2 = tmp / "dataset_arc_synthetic_copy.h5"
        shutil.copy2(path, path2)
        ds3 = ArcDivergenceDataset(
            path2, label_mode=ckpt2["label_mode"], command_mode=ckpt2["command_mode"], device=device
        )
        indices_all, split_all = resolve_split(ckpt2, path2, None, ds3)
        assert split_all == "all" and len(indices_all) == len(ds3), (split_all, len(indices_all))
        print(f"[self-test] default split -> 'all' ({len(indices_all)} rows) for a differently-pathed file")

        # --- report(): perfect predictions -> R^2=1, mean|error|=0; noisy predictions differ ----
        K = len(model2.target_names)
        true = torch.rand(50, K) * 2.0
        perfect = report(true.clone(), true, model2.target_names)
        assert torch.allclose(perfect["r2"], torch.ones(K), atol=1e-4), perfect["r2"]
        assert torch.allclose(perfect["mae"], torch.zeros(K), atol=1e-6), perfect["mae"]
        noisy_pred = true + torch.randn(50, K) * 0.3
        noisy = report(noisy_pred, true, model2.target_names)
        assert (noisy["r2"] < 0.999).all() and (noisy["mae"] > 0.0).all(), noisy
        print("[self-test] report(): perfect predictions score (R^2=1, mean|error|=0), noisy ones don't")

        # --- quantile binning: edges span [min, max], bins partition every row, mean matches the
        # constant offset regardless of exactly where the boundaries fall -------------------------
        lin_true = torch.linspace(0.0, 3.0, 30).numpy()
        lin_pred = lin_true + 0.1
        edges = quantile_bin_edges(lin_true, n_bins=3)
        assert edges.shape == (4,)
        assert edges[0] == lin_true.min() and edges[-1] == lin_true.max()
        assert (np.diff(edges) >= 0).all(), edges
        mean_err, counts = binned_mean_abs_error(lin_true, lin_pred, edges)
        assert counts.sum() == 30, counts
        assert np.allclose(mean_err[np.isfinite(mean_err)], 0.1, atol=1e-6), mean_err
        print(f"[self-test] quantile binning: edges {edges.round(3).tolist()}, counts "
              f"{counts.tolist()}, mean|err| per bin {mean_err.round(4).tolist()}")

        # --- make_figure() writes a PNG without a display -----------------------------------------
        pred, target = predict_over(model2, ds2, indices_val, device, batch_size=16)
        stats = report(pred, target, model2.target_names)
        fig_path = tmp / "fig.png"
        make_figure(pred, target, model2.target_names, "self-test", stats, fig_path)
        assert fig_path.exists() and fig_path.stat().st_size > 0
        print("[self-test] make_figure() writes a PNG")

    print("all self-checks ok")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoint", type=pathlib.Path, nargs="?", help="checkpoint .pt path (train.build_checkpoint schema)")
    parser.add_argument("--dataset", type=pathlib.Path, default=None, help="dataset_arc_*.h5 to evaluate on (default: the checkpoint's own dataset_path)")
    parser.add_argument("--split", type=str, default=None, choices=("train", "val", "all"), help="rows to score (default: val if --dataset matches the checkpoint's own file, else all)")
    parser.add_argument("--batch-size", type=int, default=512, help="inference chunk size (default: 512)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="torch device (default: cuda if available else cpu)")
    parser.add_argument("--save-fig", type=pathlib.Path, default=None, help="write the figure here instead of showing it interactively")
    parser.add_argument("--self-test", action="store_true", help="build a synthetic checkpoint + dataset in a temp dir and assert this module's own logic")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if args.checkpoint is None:
        parser.error("checkpoint is required unless --self-test is given")

    device = torch.device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    model.eval()
    names = model.target_names

    dataset_path = args.dataset or pathlib.Path(ckpt["dataset_path"])
    ds = ArcDivergenceDataset(
        dataset_path, label_mode=ckpt["label_mode"], command_mode=ckpt["command_mode"], device=device
    )
    indices, split = resolve_split(ckpt, dataset_path, args.split, ds)

    print(
        f"[checkpoint] {args.checkpoint.name}: epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f}, "
        f"label_mode={ckpt['label_mode']}, command_mode={ckpt['command_mode']}"
    )
    print(f"[dataset]    {dataset_path.name}: {len(indices)} rows scored (split={split})")

    pred, target = predict_over(model, ds, indices, device, args.batch_size)
    stats = report(pred, target, names)
    make_figure(pred, target, names, args.checkpoint.name, stats, args.save_fig)


if __name__ == "__main__":
    main()
