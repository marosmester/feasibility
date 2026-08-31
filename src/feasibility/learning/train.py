"""Trains model.PoseErrorMLP on custom_dataset.PoseErrorDataset -- the missing piece model.py's
own docstring names ("No optimizer, no scheduler, no training loop -- this module is
architecture only. The trainer constructs its own optimizer, matching how ostrich's PPOTrainer
takes one as an argument"): this module IS that trainer.

Two design points worth stating up front:

* The training loss is computed in MODEL SPACE (target_transform.forward(y) vs model(x)), never
  in physical metres/radians. model.TargetTransform.inverse() ends with a clamp_min(0.0) that
  enforces the network can't report a negative error -- but a clamp has zero gradient wherever
  it's active, so computing the loss AFTER that clamp would silently kill the gradient for any
  prediction currently on the wrong side of it. Model space has no clamp, so every prediction
  always has a usable gradient. inverse()/physical units are used only for the val-loop MAE
  printout and the optional --plot, never for backward().
* target_transform is fit on TRAIN rows only: make_dataloaders() returns train_subset alongside
  (train_loader, val_loader, ds) precisely so a caller can fit further train-only statistics --
  like this -- without re-deriving the split itself.

Usage:
    python src/feasibility/learning/train.py
    python src/feasibility/learning/train.py --dataset outputs/dataset_box_h070cm_n256.h5 --epochs 100
    python src/feasibility/learning/train.py --yaw-encoding sincos --patience 20
    python src/feasibility/learning/train.py --plot outputs/train_loss.png
    python -c "
    import torch
    from feasibility.learning.train import load_checkpoint
    model, x_normalizer, ckpt = load_checkpoint('outputs/checkpoints/dataset_box_h070cm_n128_cont.pt', torch.device('cpu'))
    print(model.predict(x_normalizer(torch.zeros(1, ckpt['in_dim']))))
    "
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from feasibility.comparator.common import OUT_DIR
from feasibility.comparator.provenance import git_provenance
from feasibility.learning.custom_dataset import make_dataloaders
from feasibility.learning.custom_dataset import Normalizer
from feasibility.learning.custom_dataset import PoseErrorDataset
from feasibility.learning.error_visual import E_POS_COLOR
from feasibility.learning.error_visual import E_ROT_COLOR
from feasibility.learning.model import DEFAULT_DEPTH
from feasibility.learning.model import DEFAULT_HIDDEN
from feasibility.learning.model import PoseErrorMLP
from feasibility.learning.model import TargetTransform

DEFAULT_DATASET = OUT_DIR / "dataset_box_h070cm_n128_cont.h5"  # continuous spawn mode preferred
# for learning -- see generate_dataset.py's module docstring on why "lattice" makes the spawn
# subspace effectively categorical.


def build_checkpoint(
    model: PoseErrorMLP,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    ds: PoseErrorDataset,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Self-describing checkpoint dict -- mirrors ostrich's
    examples/helhest/balance_ppo/train.py:_build_checkpoint(): everything load_checkpoint() (or
    any other future consumer) needs is IN the file, so it loads without this module's source or
    the dataset that trained it. target_transform lives on `model` (see model.py's docstring:
    "an ordinary attribute, not a buffer... must be saved alongside a checkpoint"), so its
    normalizer is pulled from there rather than threaded through as a separate argument."""
    target_transform = model.target_transform
    assert target_transform is not None  # train() always attaches one before calling this
    return dict(
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        epoch=epoch,
        val_loss=val_loss,
        in_dim=len(ds.FEATURE_NAMES),
        hidden=args.hidden,
        depth=args.depth,
        feature_names=ds.FEATURE_NAMES,
        target_names=ds.TARGET_NAMES,
        x_normalizer_mean=ds.x_normalizer.mean,
        x_normalizer_std=ds.x_normalizer.std,
        target_transform_mean=target_transform.normalizer.mean,
        target_transform_std=target_transform.normalizer.std,
        dataset_path=str(ds.source),
        dataset_git=ds.git,
        train_git=git_provenance(),
        args=vars(args),
    )


def load_checkpoint(
    path: pathlib.Path, device: torch.device
) -> tuple[PoseErrorMLP, Normalizer, dict[str, object]]:
    """Rebuilds (model, x_normalizer) from a build_checkpoint() dict -- the standalone
    counterpart to balance_ppo/train.py's load_policy(). Callers must apply x_normalizer to raw
    (v, wz, x, y, yaw[, ...]) inputs before calling model.predict() -- the model itself has no
    idea its inputs need normalizing, exactly like PoseErrorDataset.__getitem__ applies
    ds.x_normalizer, not the model, today."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    target_transform = TargetTransform(
        normalizer=Normalizer(
            mean=ckpt["target_transform_mean"].to(device), std=ckpt["target_transform_std"].to(device)
        )
    )
    model = PoseErrorMLP(
        ckpt["in_dim"], hidden=ckpt["hidden"], depth=ckpt["depth"], target_transform=target_transform
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    x_normalizer = Normalizer(
        mean=ckpt["x_normalizer_mean"].to(device), std=ckpt["x_normalizer_std"].to(device)
    )
    return model, x_normalizer, ckpt


def plot_history(history: list[dict[str, float]], out_path: pathlib.Path) -> None:
    """Two-panel loss curve: top = model-space MSE (train vs val, same units, one axis) so
    over/underfitting is directly readable; bottom = physical-unit val MAE per target head, one
    axis each like error_visual.py's e_pos/e_rot split -- reuses its exact colors so a reader
    who's seen that plot recognizes this one."""
    epochs = [row["epoch"] for row in history]

    fig, (ax_loss, ax_mae) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax_loss.plot(epochs, [row["train_loss"] for row in history], label="train (model space)")
    ax_loss.plot(epochs, [row["val_loss"] for row in history], label="val (model space)")
    ax_loss.set_ylabel("MSE (model space)")
    ax_loss.set_title("training loss")
    ax_loss.legend(loc="upper right")

    ax_mae.plot(
        epochs, [row["val_mae_e_pos"] for row in history], color=E_POS_COLOR, label="val e_pos MAE (m)"
    )
    ax_mae.set_xlabel("epoch")
    ax_mae.set_ylabel("e_pos MAE (m)", color=E_POS_COLOR)
    ax_mae.tick_params(axis="y", labelcolor=E_POS_COLOR)

    ax_rot = ax_mae.twinx()
    ax_rot.plot(
        epochs, [row["val_mae_e_rot"] for row in history], color=E_ROT_COLOR, label="val e_rot MAE (rad)"
    )
    ax_rot.set_ylabel("e_rot MAE (rad)", color=E_ROT_COLOR)
    ax_rot.tick_params(axis="y", labelcolor=E_ROT_COLOR)

    lines = ax_mae.get_lines() + ax_rot.get_lines()
    ax_mae.legend(lines, [line.get_label() for line in lines], loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved loss curve to {out_path}")


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    train_loader, val_loader, ds, train_subset = make_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        val_frac=args.val_frac,
        seed=args.seed,
        normalize_targets=False,  # TargetTransform (log1p + standardize) is the only y-transform
        # -- a second normalization here would double-standardize the targets.
        yaw_encoding=args.yaw_encoding,
    )
    target_transform = TargetTransform.fit(ds.y[train_subset.indices])

    checkpoint_path = args.checkpoint or OUT_DIR / "checkpoints" / f"{ds.source.stem}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    model = PoseErrorMLP(
        len(ds.FEATURE_NAMES), hidden=args.hidden, depth=args.depth, target_transform=target_transform
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(
        f"[data]  {ds.source.name}: {len(ds)} rows, {len(train_loader.dataset)} train / "
        f"{len(val_loader.dataset)} val, yaw_encoding={args.yaw_encoding}"
    )
    print(f"[model] in_dim={len(ds.FEATURE_NAMES)}, hidden={args.hidden}, depth={args.depth}, device={device}")

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_since_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = F.mse_loss(pred, target_transform.forward(yb))
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * xb.shape[0]
            train_n += xb.shape[0]
        train_loss = train_loss_sum / train_n

        model.eval()
        val_loss_sum, val_mae_sum, val_n = 0.0, torch.zeros(2), 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss_sum += F.mse_loss(pred, target_transform.forward(yb)).item() * xb.shape[0]
                physical_pred = target_transform.inverse(pred)
                val_mae_sum += (physical_pred - yb).abs().sum(dim=0).cpu()
                val_n += xb.shape[0]
        val_loss = val_loss_sum / val_n
        val_mae_e_pos, val_mae_e_rot = (val_mae_sum / val_n).tolist()

        history.append(
            dict(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_mae_e_pos=val_mae_e_pos,
                val_mae_e_rot=val_mae_e_rot,
            )
        )

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss, best_epoch = val_loss, epoch
            epochs_since_improve = 0
            torch.save(build_checkpoint(model, optimizer, epoch, val_loss, ds, args), checkpoint_path)
        else:
            epochs_since_improve += 1

        if epoch % args.log_every == 0 or epoch == args.epochs or improved:
            marker = " *" if improved else ""
            print(
                f"[{epoch:4d}/{args.epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_mae_e_pos={val_mae_e_pos:.4f}m val_mae_e_rot={val_mae_e_rot:.4f}rad{marker}"
            )

        if args.patience > 0 and epochs_since_improve >= args.patience:
            print(f"early stop at epoch {epoch}: no val_loss improvement in {args.patience} epochs")
            break

    print(f"[done]  best val_loss={best_val_loss:.4f} @ epoch {best_epoch}, checkpoint={checkpoint_path}")

    if args.plot is not None:
        plot_history(history, args.plot)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=pathlib.Path, default=DEFAULT_DATASET, help="dataset .h5 (custom_dataset.PoseErrorDataset schema)")
    parser.add_argument("--val-frac", type=float, default=0.2, help="fraction of rows held out for validation (default: 0.2)")
    parser.add_argument("--seed", type=int, default=0, help="seeds the train/val split and model init (default: 0)")
    parser.add_argument("--yaw-encoding", choices=("raw", "sincos"), default="raw", help="spawn yaw feature encoding (default: raw)")
    parser.add_argument("--batch-size", type=int, default=32, help="default: 32")
    parser.add_argument("--epochs", type=int, default=200, help="default: 200")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay (default: 0.0)")
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN, help=f"trunk width (default: {DEFAULT_HIDDEN})")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help=f"trunk layers (default: {DEFAULT_DEPTH})")
    parser.add_argument("--patience", type=int, default=0, help="epochs of no val_loss improvement before early stop; 0 disables (default: 0)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="torch device (default: cuda if available else cpu)")
    parser.add_argument("--checkpoint", type=pathlib.Path, default=None, help="save path (default: outputs/checkpoints/<dataset stem>.pt)")
    parser.add_argument("--plot", type=pathlib.Path, default=None, help="save a loss-curve PNG here (default: no plot)")
    parser.add_argument("--log-every", type=int, default=10, help="epochs between stdout progress lines (default: 10)")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
