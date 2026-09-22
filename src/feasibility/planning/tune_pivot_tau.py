"""Re-tune `planners.TAU_PIVOT_PITCH` against a v_wz checkpoint, on that checkpoint's own val split.

The point-turn gate compares a predicted e_pitch to a tolerance in radians. That only works while
the e_pitch head is calibrated for a PIVOT, and a head is calibrated per checkpoint -- retraining
the same config on the same dataset moves individual predictions by a few percent, which is enough
to flip a gate decision that sits near the line. This is the tool that re-derives the threshold
after a retrain, so the number in `planners.py` is never an artefact of weights nobody has any more.

The recipe, unchanged from the original calibration:

  * population: the PIVOT rows (v_drive == 0) of the checkpoint's HELD-OUT maps, reconstructed
    from its own stored `args` (val_frac, seed) and `dataset_path`. Never a map a demo plans on --
    tuning tau until a demo tells the story you wanted is how you end up with a constant that
    describes the demo instead of the robot.
  * label: a point turn is BAD when its TRUE e_pitch exceeds `--bad` (default 5 deg, the physical
    judgement -- past there a point turn is outside what flat ground produces). That is about the
    robot and does not move when the network does.
  * threshold: the PREDICTED value that best implements that label, by Youden's J (TPR - FPR).
    F1 and a mid-gap rule are printed beside it, and a bootstrap over the rows gives the spread --
    a few hundred pivots is not many, and the J curve is usually a plateau rather than a spike, so
    the band matters more than the argmax. Pick DOWN inside the band if a missed turn costs more
    than a lost path (the gate's TPR/FPR at every candidate is in the table).

CLI parameters:
    checkpoint        a v_wz pos_rpy checkpoint (planners.DEFAULT_CHECKPOINT_VWZ by default)
    --bad RAD         true e_pitch that counts as a bad point turn (default TAU_BAD, 5 deg)
    --device D        where inference runs (default cpu; a few hundred rows, so it hardly matters)
    --boot N          bootstrap resamples for the spread (default 2000, 0 to skip)

Usage:
    python src/feasibility/planning/tune_pivot_tau.py
    python src/feasibility/planning/tune_pivot_tau.py outputs/checkpoints/x_vwz_rpy.pt --bad 0.174
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch

from feasibility.lattice_learning.custom_dataset import ArcDivergenceDataset
from feasibility.lattice_learning.custom_dataset import split_dataset_by_map
from feasibility.lattice_learning.train import load_checkpoint
from feasibility.planning.planners import DEFAULT_CHECKPOINT_VWZ
from feasibility.planning.planners import TAU_PIVOT_PITCH

TAU_BAD = 0.0873  # [rad] = 5 deg, the physical "this point turn diverged" line (see planners.py)
# The candidates the table reports at, so a reader sees the whole operating curve and not only the
# argmax. The current constant is added to it, wherever it happens to sit.
REPORT_AT = (0.04, 0.05, 0.06, 0.07, 0.0873, 0.10, 0.113, 0.12, 0.15, 0.20)
GRID = np.round(np.linspace(0.01, 0.30, 291), 5)  # searched for the J / F1 optima


def pivot_predictions(
    checkpoint: pathlib.Path, device: torch.device
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """(predicted, true) e_pitch [n] over the checkpoint's held-out PIVOT rows, plus its own dict.

    The split is rebuilt from the checkpoint's stored `args`, not from defaults, so this reads the
    same held-out maps the training run did -- a different val_frac/seed would score rows the net
    was fitted on and quietly report a far better threshold than the gate will ever see.
    """
    model, ckpt = load_checkpoint(checkpoint, device)
    model.eval()
    if "e_pitch" not in model.target_names:
        raise SystemExit(
            f"{checkpoint.name} is a {ckpt['label_mode']} checkpoint with heads "
            f"{model.target_names} -- the pivot gate reads e_pitch, so it needs a pos_rpy one."
        )
    ds = ArcDivergenceDataset(
        pathlib.Path(ckpt["dataset_path"]), label_mode=ckpt["label_mode"],
        command_mode=ckpt["command_mode"], device=device,
    )
    _, val = split_dataset_by_map(
        ds, val_frac=float(ckpt["args"]["val_frac"]), seed=int(ckpt["args"]["seed"])
    )
    idx = torch.as_tensor(val.indices, device=device)
    k = list(model.target_names).index("e_pitch")
    with torch.no_grad():
        pred = torch.cat(
            [model.predict(ds.patch[idx[i:i + 512]], ds.command[idx[i:i + 512]])
             for i in range(0, len(idx), 512)]
        )[:, k].cpu().numpy()
    true = ds.y[idx][:, k].cpu().numpy()
    is_pivot = (ds.twist[idx][:, 0] == 0).cpu().numpy()  # v_drive == 0: the point turns
    return pred[is_pivot], true[is_pivot], ckpt


def rates(pred: np.ndarray, bad: np.ndarray, tau: float) -> tuple[float, float, float, float]:
    """(TPR, FPR, J, F1) of "prune when predicted > tau" against the true-error label."""
    hit = pred > tau
    tpr = float((hit & bad).sum() / max(bad.sum(), 1))
    fpr = float((hit & ~bad).sum() / max((~bad).sum(), 1))
    prec = float((hit & bad).sum() / max(hit.sum(), 1))
    f1 = 0.0 if prec + tpr == 0.0 else 2.0 * prec * tpr / (prec + tpr)
    return tpr, fpr, tpr - fpr, f1


def auc(pred: np.ndarray, bad: np.ndarray) -> float:
    """Rank AUC of the head against the label -- how well it ORDERS point turns, independently of
    where any threshold is put. This is what survives a retrain; the threshold is what does not."""
    order = np.argsort(pred)
    ranks = np.empty(len(pred), dtype=np.float64)
    ranks[order] = np.arange(1, len(pred) + 1)
    n1, n0 = bad.sum(), (~bad).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[bad].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def bootstrap_argmax(pred: np.ndarray, bad: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """[n] the J-optimal tau of n resamples of the rows, i.e. how much of the pick is the data and
    how much is the 200-odd rows that happened to land in the val split."""
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        s = rng.integers(0, len(pred), len(pred))
        ps, bs = pred[s], bad[s]
        j = [rates(ps, bs, float(x))[2] for x in GRID]
        out[i] = GRID[int(np.argmax(j))]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("checkpoint", type=pathlib.Path, nargs="?", default=DEFAULT_CHECKPOINT_VWZ)
    ap.add_argument("--bad", type=float, default=TAU_BAD,
                    help="[rad] true e_pitch above which a point turn counts as bad")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples (0 to skip)")
    args = ap.parse_args()

    pred, true, ckpt = pivot_predictions(args.checkpoint, torch.device(args.device))
    bad = true > args.bad
    print(f"\n{args.checkpoint.name}: epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f}, "
          f"dataset {pathlib.Path(ckpt['dataset_path']).name}")
    print(f"{len(pred)} held-out pivot rows, {int(bad.sum())} bad (true e_pitch > {args.bad:.4f} "
          f"rad = {np.degrees(args.bad):.1f} deg), {int((~bad).sum())} fine")
    if not bad.any() or bad.all():
        raise SystemExit("one class is empty -- nothing to tune against")

    j = np.array([rates(pred, bad, float(x))[2] for x in GRID])
    f1 = np.array([rates(pred, bad, float(x))[3] for x in GRID])
    tau_j, tau_f1 = float(GRID[int(j.argmax())]), float(GRID[int(f1.argmax())])
    # mid-gap: halfway between the facing quantiles of what the two populations predict
    mid = float(0.5 * (np.quantile(pred[~bad], 0.975) + np.quantile(pred[bad], 0.025)))

    print(f"\n  {'tau':>7} {'deg':>5}   {'TPR':>5} {'FPR':>5} {'J':>6} {'F1':>6}   "
          f"pruned / missed / false")
    for tau in sorted({*REPORT_AT, TAU_PIVOT_PITCH, tau_j, tau_f1, round(mid, 5)}):
        tpr, fpr, jj, ff = rates(pred, bad, tau)
        hit = pred > tau
        mark = " ".join(
            m for m, v in (("<- J", tau_j), ("<- F1", tau_f1), ("<- mid-gap", mid),
                           ("(current)", TAU_PIVOT_PITCH)) if abs(tau - v) < 1e-4
        )
        print(f"  {tau:7.4f} {np.degrees(tau):5.1f}   {tpr:5.2f} {fpr:5.2f} {jj:6.3f} {ff:6.3f}   "
              f"{int((hit & bad).sum()):3d} / {int((~hit & bad).sum()):3d} / "
              f"{int((hit & ~bad).sum()):3d}  {mark}")

    print(f"\n  ranking (threshold-free): AUC {auc(pred, bad):.3f}")
    upper = true >= np.median(true)
    print(f"  bias: predicted/true over the upper half, median "
          f"{np.median(pred[upper] / np.maximum(true[upper], 1e-9)):.3f}x  (1.0 = a tolerance in "
          f"radians means what it says)")
    print(f"  predicted on bad rows  median {np.median(pred[bad]):.4f}  q0.10 "
          f"{np.quantile(pred[bad], 0.10):.4f}")
    print(f"  predicted on fine rows median {np.median(pred[~bad]):.4f}  q0.90 "
          f"{np.quantile(pred[~bad], 0.90):.4f}  q0.975 {np.quantile(pred[~bad], 0.975):.4f}")

    band = ""
    if args.boot:
        b = bootstrap_argmax(pred, bad, args.boot)
        lo, hi = (float(v) for v in np.quantile(b, [0.05, 0.95]))
        band = (f", bootstrap 90% spread {lo:.4f}-{hi:.4f} "
                f"({np.degrees(lo):.1f}-{np.degrees(hi):.1f} deg)")
    print(f"\n  TAU_PIVOT_PITCH for this checkpoint: {tau_j:.4f} ({np.degrees(tau_j):.1f} deg) by "
          f"Youden J{band}")
    print(f"  F1 agrees at {tau_f1:.4f}; the mid-gap rule says {mid:.4f}. The J curve is a "
          f"plateau, so read the band,\n  not the argmax, and move DOWN inside it if a missed "
          f"point turn costs more than a longer path.")
    print(f"  planners.TAU_PIVOT_PITCH is currently {TAU_PIVOT_PITCH:.4f} "
          f"({np.degrees(TAU_PIVOT_PITCH):.1f} deg).")


if __name__ == "__main__":
    main()
