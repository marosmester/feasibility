"""Trains model.WindowDivergenceNet on custom_dataset.WindowDivergenceDataset -- design.md section 8
step 8. `lattice_learning/train.py`'s recipe: TargetTransform fitted on train rows only, MSE in model
space, AdamW with cosine-to-0 after a linear warm-up, early stop on val loss, a held-out-MAP split,
the y-mirror augmentation, baselines computed before any training and a final report on the best
checkpoint. Its command-agnostic helpers (losses, metrics, blur, scheduler, `evaluate`, the verdict
line) are imported, not copied. What changes:

* **The command** is the window's `(v_mean, v_slope, wz_mean, wz_slope)`, and the mirror negates the
  two yaw-rate columns (`model.mirror_command`); the patch flips its rows (body +Y), labels unchanged.

* **Baselines** (fitted on train rows, scored on val, physical units). Lattice's kappa-binned mean and
  swept-envelope proxy assume a constant-kappa 0.3 m arc, so they are replaced:

      global mean        one constant per head -- a sanity floor
      per-family mean    one mean per command family (wide/straight/narrow/spin), shrunk toward the
                         global mean -- "uses the command at all"
      family + relief    y ~ a * relief + b_family per head by least squares: `relief` is the stored
                         max plane-relative relief along the window's path (spawn_sampling), the
                         geometric proxy the net must beat

* **Filters**: `--drop-endpoint-infeasible` and `--drop-twin-flagged` (custom_dataset's), both off by
  default like lattice's blocked endpoints; when kept, the report splits them out.

* **The report** is one table per head: rows are val subsets -- all, the top decile by true e_pos,
  interact_dir 0/+1/-1, each family, continued vs jump entry vs entry at rest (design.md section
  7's "does the jump share need the start twist" question), flat patches (the twin's pure kinematic
  offset, which is NOT zero here: spins carry ~0.1 m on flat ground) and the rows the two filters
  would drop -- and columns are the model, every baseline and the verdict against the best baseline
  for that head. R^2, mirror equivariance and a VERDICT banner from the "all" subset follow.

* **The checkpoint** carries what MPPI's hook must match before trusting the net: the patch spec, the
  command columns and the fixed feature scales (`V_SCALE`/`WZ_SCALE`, asserted on load), and the
  dataset's `LABEL_ATTRS` (window, dt, twin solver, k_turn, ostrich yaw gain, mu).

CLI parameters:
    --dataset PATH [PATH ...]    dataset_mppi_*.h5 file(s), trained together (required)
    --drop-endpoint-infeasible   drop rows whose nominal window end is settle-infeasible
    --drop-twin-flagged          drop rows the twin flags (clearance <= 0 or residual > resid_tol)
    --val-frac FLOAT             fraction of MAPS held out for validation (default: 0.2)
    --seed INT                   seeds the split, model init, and augmentation (default: 0)
    --base-width INT             trunk channel unit (default: 32)
    --embed-dim INT              command-embedding / FiLM width (default: 64)
    --head-fusion {film,concat}  (default: film)
    --label-mode {pos_rot,pos_rpy}  (e_pos, e_rot), or (e_pos, e_roll, e_pitch, e_yaw) (default: pos_rot)
    --trunk {original,trt_friendly}  the trunk blocks: replicate pad + channel LayerNorm, or zero pad +
                              BatchNorm (model_tensorRT_friendly.py, which TensorRT fuses); stored in the
                              checkpoint, which load_checkpoint rebuilds from (default: original)
    --no-augment                 disable the y-mirror augmentation (on by default)
    --blur-terrain               baseline: relief replaced by its per-sample mean, train AND eval
    --batch-size INT             default: 256
    --epochs INT                 default: 100
    --lr FLOAT                   AdamW learning rate (default: 3e-4)
    --weight-decay FLOAT         AdamW weight decay (default: 1e-4)
    --lr-schedule {cosine,none}  (default: cosine)
    --warmup-epochs INT          (default: 5)
    --patience INT               epochs without val_loss improvement before stopping; 0 disables (default: 20)
    --device STR                 torch device (default: cuda if available else cpu)
    --checkpoint PATH            save path (default: outputs/checkpoints/<first stem>[_plus<N>]<tags>.pt)
    --log-every INT              epochs between stdout progress lines (default: 10)
    --self-test                  a few epochs on synthetic files, wandb disabled, CPU -- no dataset/GPU
    --wandb-project STR          (default: "feasibility-mppi-window-divergence")
    --wandb-entity STR           (default: None)
    --wandb-mode {online,offline,disabled}  (default: online)
    --wandb-name STR             (default: None)

Usage:
    python src/feasibility/mppi_learning/train.py --self-test
    python src/feasibility/mppi_learning/train.py --dataset outputs/dataset_mppi_<...>.h5
    python src/feasibility/mppi_learning/train.py --dataset a.h5 b.h5 --drop-twin-flagged
    python src/feasibility/mppi_learning/train.py --dataset <...>.h5 --blur-terrain   # baseline: no terrain
"""
from __future__ import annotations

import argparse
import math
import pathlib
import tempfile

import numpy as np
import torch
import wandb

from feasibility.checkpoints.artifacts import log_checkpoint
from feasibility.comparator.common import OUT_DIR
from feasibility.comparator.provenance import git_provenance
from feasibility.lattice_learning.model import DEFAULT_BASE_WIDTH
from feasibility.lattice_learning.model import DEFAULT_EMBED_DIM
from feasibility.lattice_learning.model import HEAD_FUSIONS
from feasibility.lattice_learning.patch import patch_spec_from_attrs
from feasibility.lattice_learning.patch import patch_spec_to_attrs
from feasibility.lattice_learning.train import BASELINE_TIE_PCT
from feasibility.lattice_learning.train import blur
from feasibility.lattice_learning.train import build_scheduler
from feasibility.lattice_learning.train import compare_to_baselines
from feasibility.lattice_learning.train import evaluate
from feasibility.lattice_learning.train import FLAT_RELIEF
from feasibility.lattice_learning.train import fmt
from feasibility.lattice_learning.train import KAPPA_MEAN_PSEUDO_COUNTS
from feasibility.lattice_learning.train import mse_loss
from feasibility.lattice_learning.train import prepare_patch
from feasibility.lattice_learning.train import r_squared
from feasibility.lattice_learning.train import rmse_per_head
from feasibility.lattice_learning.train import top_decile_rows
from feasibility.mppi_learning.custom_dataset import LABEL_ATTRS
from feasibility.mppi_learning.custom_dataset import make_dataloaders
from feasibility.mppi_learning.custom_dataset import RESID_TOL
from feasibility.mppi_learning.custom_dataset import valid_targets
from feasibility.mppi_learning.custom_dataset import WindowDivergenceDataset
from feasibility.mppi_learning.model import COMMAND_COLUMNS
from feasibility.mppi_learning.model import LABEL_MODES
from feasibility.mppi_learning.model import mirror_command
from feasibility.mppi_learning.model import Normalizer
from feasibility.mppi_learning.model import TargetTransform
from feasibility.mppi_learning.model import V_SCALE
from feasibility.mppi_learning.model import WindowDivergenceNet
from feasibility.mppi_learning.model import WZ_SCALE
from feasibility.mppi_learning.model_tensorRT_friendly import TRUNK_STYLE
from feasibility.mppi_learning.model_tensorRT_friendly import TRUNK_STYLES
from feasibility.mppi_learning.model_tensorRT_friendly import TrtFriendlyWindowDivergenceNet

# Pseudo-observations pulling each family's mean toward the global mean (lattice's kappa-bin value).
FAMILY_MEAN_PSEUDO_COUNTS = KAPPA_MEAN_PSEUDO_COUNTS


# --- input transforms ----------------------------------------------------------------------------


def mirror_augment(
    patch: torch.Tensor, command: torch.Tensor, y: torch.Tensor, flip: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample, p=0.5: reflect the body-frame patch across y=0 (rows are body +Y) and negate
    wz_mean/wz_slope -- an exact symmetry of the robot, the twin and the command box. Labels are
    magnitudes and stay. `flip` overrides the coin, for the self-test."""
    if flip is None:
        flip = torch.rand(patch.shape[0], device=patch.device) < 0.5
    patch = torch.where(flip[:, None, None, None], torch.flip(patch, dims=[-2]), patch)
    command = torch.where(flip[:, None], mirror_command(command), command)
    return patch, command, y


# --- baselines -----------------------------------------------------------------------------------


def group_mean_table(groups: torch.Tensor, y: torch.Tensor, n_groups: int) -> torch.Tensor:
    """[n_groups, K] train mean label per group, each shrunk toward the global mean with
    `FAMILY_MEAN_PSEUDO_COUNTS` pseudo-observations, so an empty or sparse group is the global mean
    rather than noise."""
    global_mean = y.mean(dim=0)
    table = torch.empty(n_groups, y.shape[-1], dtype=y.dtype)
    m = FAMILY_MEAN_PSEUDO_COUNTS
    for g in range(n_groups):
        sel = groups == g
        table[g] = (y[sel].sum(dim=0) + m * global_mean) / (sel.sum() + m)
    return table


def compute_baselines(
    ds: WindowDivergenceDataset, train_idx: list[int], val_idx: list[int]
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """(val RMSE per baseline, val PREDICTIONS per baseline), CPU, physical units, fitted on train
    rows only. Predictions are returned so the report scores every baseline on the model's subsets."""
    y, family, relief = ds.y.cpu(), ds.family.cpu(), ds.relief.cpu()
    tr, va = torch.as_tensor(train_idx), torch.as_tensor(val_idx)
    n_families = len(ds.families)
    preds: dict[str, torch.Tensor] = {}

    preds["global mean"] = y[tr].mean(dim=0).expand(len(va), -1)
    preds["per-family mean"] = group_mean_table(family[tr], y[tr], n_families)[family[va]]

    def design(rows: torch.Tensor) -> torch.Tensor:  # [n, 1 + n_families]: relief, family one-hot
        onehot = torch.nn.functional.one_hot(family[rows], n_families)
        return torch.cat([relief[rows, None], onehot], dim=-1).double()

    # lstsq's minimum-norm solution copes with a family absent from the train rows
    coef = torch.linalg.lstsq(design(tr), y[tr].double(), driver="gelsd").solution
    preds["family + relief"] = (design(va) @ coef).float().clamp_min(0.0)

    rmse = {name: rmse_per_head(p, y[va]) for name, p in preds.items()}
    return rmse, preds


# --- checkpointing -------------------------------------------------------------------------------


def label_attrs(ds: WindowDivergenceDataset) -> dict[str, object]:
    """The dataset's `LABEL_ATTRS` as plain Python values -- what MPPI's hook must match."""
    plain = {k: ds.attrs.get(k) for k in LABEL_ATTRS}
    return {k: v.decode() if isinstance(v, bytes) else v.item() if isinstance(v, np.generic) else v
            for k, v in plain.items()}


def build_checkpoint(
    model: WindowDivergenceNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    ds: WindowDivergenceDataset,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Self-describing checkpoint: everything load_checkpoint() and MPPI's hook need is in the file."""
    target_transform = model.target_transform
    assert target_transform is not None  # train() always attaches one before calling this
    return dict(
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        epoch=epoch,
        val_loss=val_loss,
        base_width=args.base_width,
        embed_dim=args.embed_dim,
        head_fusion=args.head_fusion,
        trunk=TRUNK_STYLE if isinstance(model, TrtFriendlyWindowDivergenceNet) else "original",
        label_mode=model.label_mode,
        target_names=model.target_names,
        target_transform_mean=target_transform.normalizer.mean.cpu(),
        target_transform_std=target_transform.normalizer.std.cpu(),
        patch_spec=patch_spec_to_attrs(ds.patch_spec),
        command_columns=COMMAND_COLUMNS,
        v_scale=V_SCALE,
        wz_scale=WZ_SCALE,
        label_attrs=label_attrs(ds),
        blur_terrain=args.blur_terrain,
        drop_endpoint_infeasible=args.drop_endpoint_infeasible,
        drop_twin_flagged=args.drop_twin_flagged,
        dataset_paths=[str(p) for p in ds.sources],
        dataset_git=ds.git,
        train_git=git_provenance(),
        args={k: [str(p) for p in v] if k == "dataset" else v for k, v in vars(args).items()},
    )


def load_checkpoint(
    path: pathlib.Path, device: torch.device
) -> tuple[WindowDivergenceNet, dict[str, object]]:
    """Rebuilds a WindowDivergenceNet with its TargetTransform, so `.predict()` is physical. Refuses a
    checkpoint whose command columns or feature scales differ from this code's: the weights were
    trained on features the current `window_command_features` would no longer produce."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if tuple(ckpt["command_columns"]) != COMMAND_COLUMNS or not (
        math.isclose(ckpt["v_scale"], V_SCALE) and math.isclose(ckpt["wz_scale"], WZ_SCALE)
    ):
        raise ValueError(f"{path}: command {ckpt['command_columns']} at scales ({ckpt['v_scale']}, "
                         f"{ckpt['wz_scale']}), this code has {COMMAND_COLUMNS} at ({V_SCALE}, {WZ_SCALE})")
    target_transform = TargetTransform(
        normalizer=Normalizer(
            mean=ckpt["target_transform_mean"].to(device), std=ckpt["target_transform_std"].to(device)
        )
    )
    net_class = TrtFriendlyWindowDivergenceNet if ckpt.get("trunk") == TRUNK_STYLE else WindowDivergenceNet
    model = net_class(
        base_width=ckpt["base_width"],
        embed_dim=ckpt["embed_dim"],
        head_fusion=ckpt["head_fusion"],
        label_mode=ckpt["label_mode"],
        patch_spec=patch_spec_from_attrs(ckpt["patch_spec"]),
        target_transform=target_transform,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt


# --- evaluation ----------------------------------------------------------------------------------


@torch.no_grad()
def mirror_equivariance_check(
    model: WindowDivergenceNet,
    ds: WindowDivergenceDataset,
    val_idx: list[int],
    device: torch.device,
    blur_terrain: bool,
    n: int = 64,
) -> float:
    """max |f(patch, cmd) - f(flip(patch), mirror(cmd))| in model space over a few val rows -- a
    diagnostic of how well the augmentation took, not an assertion."""
    idx = torch.as_tensor(val_idx[:n], device=ds.patch.device)
    patch = prepare_patch(ds.patch[idx].to(device), blur_terrain)
    command = ds.command[idx].to(device)
    model.eval()
    a = model(patch, command)
    b = model(torch.flip(patch, dims=[-2]), mirror_command(command))
    return (a - b).abs().max().item()


def report_subsets(
    ds: WindowDivergenceDataset, val_idx: list[int], target: torch.Tensor
) -> dict[str, torch.Tensor | None]:
    """The val subsets the report scores, as [n_val] bool selectors (None = all). Empty ones are left
    out; the filter subsets only appear when their rows were kept."""
    vi = torch.as_tensor(val_idx)
    interact = ds.interact_dir.cpu()[vi]
    family = ds.family.cpu()[vi]
    jump = ds.jump.cpu()[vi]
    at_rest = (ds.entry.cpu()[vi] == 0.0).all(dim=1)  # rotate_in_place: entered from standstill
    flat = ds.patch.cpu()[vi].abs().amax(dim=(1, 2, 3)) < FLAT_RELIEF
    subsets: dict[str, torch.Tensor | None] = {
        "all": None,
        "top decile e_pos": top_decile_rows(target),
        "neutral": interact == 0,
        "climbing": interact > 0,
        "dropping": interact < 0,
        **{f"family {name}": family == k for k, name in enumerate(ds.families)},
        "continued entry": ~jump,
        "jump entry": jump & ~at_rest,
        "entry at rest": at_rest,
        "flat patch": flat,
    }
    infeasible = ~ds.endpoint_feasible.cpu()[vi]
    subsets["end infeasible"] = infeasible
    flagged = (ds.twin_min_clearance.cpu()[vi] <= 0.0) | (ds.twin_max_residual.cpu()[vi] > RESID_TOL)
    subsets["twin flagged"] = flagged
    return {k: v for k, v in subsets.items() if v is None or bool(v.any())}


def final_report(
    checkpoint_path: pathlib.Path,
    ds: WindowDivergenceDataset,
    val_loader: torch.utils.data.DataLoader,
    val_idx: list[int],
    baseline_preds: dict[str, torch.Tensor],
    device: torch.device,
    run: wandb.sdk.wandb_run.Run,
    blur_terrain: bool,
) -> dict[str, torch.Tensor]:
    """Reloads the BEST checkpoint and prints one table per head: every val subset scored for the
    model and every baseline, with the verdict against the best baseline of that subset and head.
    Returns the model's per-subset RMSE (used by the self-test)."""
    model, ckpt = load_checkpoint(checkpoint_path, device)
    names = model.target_names
    assert model.target_transform is not None
    _, pred, target = evaluate(model, val_loader, model.target_transform, device, blur_terrain)

    subsets = report_subsets(ds, val_idx, target)
    table = {"model": pred, **baseline_preds}
    rmse = {s: {name: rmse_per_head(p, target, rows) for name, p in table.items()} for s, rows in subsets.items()}
    verdicts = {s: compare_to_baselines(r["model"], {n: v for n, v in r.items() if n != "model"}, names)[1]
                for s, r in rmse.items()}

    print(f"\n[final report] checkpoint={checkpoint_path} (epoch {ckpt['epoch']}), "
          f"{len(val_idx)} val rows on held-out maps" + (", blur_terrain=ON" if blur_terrain else ""))
    summary: dict[str, float] = {}
    for k, head in enumerate(names):
        print(f"[final report] val RMSE of {head} (verdict: model vs best baseline, tie within "
              f"{BASELINE_TIE_PCT:g}%)")
        print(f"  {'subset':<18}{'n':>6}" + "".join(f"{name:>17}" for name in table) + "  verdict")
        for s, rows in subsets.items():
            n_rows = len(val_idx) if rows is None else int(rows.sum())
            cells = "".join(f"{rmse[s][name][k].item():17.4f}" for name in table)
            print(f"  {s:<18}{n_rows:>6}{cells}  {verdicts[s][k]}")
            for name in table:
                summary[f"final_rmse_{s}_{name}_{head}".replace(" ", "_").replace("+", "plus")] = rmse[s][name][k].item()
            summary[f"final_vs_baseline_{s}_{head}".replace(" ", "_")] = float(verdicts[s][k] == "WIN")

    r2 = r_squared(pred, target)
    print(f"[final report] val R^2 per head (model): {fmt(r2, names)}")
    summary.update({f"final_r2_{n}": v for n, v in zip(names, r2.tolist())})

    mirror_diff = mirror_equivariance_check(model, ds, val_idx, device, blur_terrain)
    print(f"[final report] mirror-equivariance max |diff| (model space): {mirror_diff:.4e}")
    summary["final_mirror_equivariance_max_diff"] = mirror_diff

    # Only the "all" subset decides the headline: the others are nested views of the same rows.
    won = [n for n, v in zip(names, verdicts["all"]) if v == "WIN"]
    lost = [n for n, v in zip(names, verdicts["all"]) if v != "WIN"]
    headline = ("BETTER THAN EVERY BASELINE" if not lost else "NO BETTER THAN THE BEST BASELINE" if not won
                else f"MIXED -- better on {', '.join(won)}, not on {', '.join(lost)}")
    all_verdicts = [v for vs in verdicts.values() for v in vs]
    n_won = sum(v == "WIN" for v in all_verdicts)
    summary["final_beats_baseline_checks"] = float(n_won)
    summary["final_beats_baseline_checks_total"] = float(len(all_verdicts))
    rule = "=" * 78
    print(f"\n{rule}\n  VERDICT: {headline}\n  (all {len(val_idx)} val rows; model wins {n_won}/"
          f"{len(all_verdicts)} (subset, head) checks against the best baseline of each)\n{rule}")

    run.summary.update(summary)
    return {s: r["model"] for s, r in rmse.items()}


# --- training ------------------------------------------------------------------------------------


def default_checkpoint_path(args: argparse.Namespace) -> pathlib.Path:
    """outputs/checkpoints/<first dataset stem>[_plus<N>]<tags>.pt"""
    stem = args.dataset[0].stem + (f"_plus{len(args.dataset) - 1}" if len(args.dataset) > 1 else "")
    tags = [("_blur", args.blur_terrain), (f"_{args.head_fusion}", args.head_fusion != "film"),
            ("_rpy", args.label_mode != "pos_rot"), ("_noinfeasible", args.drop_endpoint_infeasible),
            ("_noflagged", args.drop_twin_flagged), ("_noaug", args.no_augment),
            ("_trt", args.trunk == TRUNK_STYLE)]
    return OUT_DIR / "checkpoints" / f"{stem}{''.join(t for t, on in tags if on)}.pt"


def train(args: argparse.Namespace, run: wandb.sdk.wandb_run.Run) -> pathlib.Path | None:
    """Runs the whole recipe and returns the best checkpoint's path (None if nothing was saved)."""
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    train_loader, val_loader, ds, train_subset, val_subset = make_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        val_frac=args.val_frac,
        seed=args.seed,
        device=device,
        label_mode=args.label_mode,
        drop_endpoint_infeasible=args.drop_endpoint_infeasible,
        drop_twin_flagged=args.drop_twin_flagged,
    )

    # Fitted on TRAIN rows only -- fitting on all rows would leak val statistics.
    target_transform = TargetTransform.fit(valid_targets(ds, train_subset.indices))
    names = ds.TARGET_NAMES

    checkpoint_path = args.checkpoint or default_checkpoint_path(args)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    net_class = TrtFriendlyWindowDivergenceNet if args.trunk == TRUNK_STYLE else WindowDivergenceNet
    model = net_class(
        base_width=args.base_width,
        embed_dim=args.embed_dim,
        head_fusion=args.head_fusion,
        label_mode=ds.label_mode,
        patch_spec=ds.patch_spec,
        target_transform=target_transform,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (
        build_scheduler(optimizer, args.warmup_epochs, args.epochs, len(train_loader))
        if args.lr_schedule == "cosine"
        else None
    )

    n_maps = ds.map_index.unique().numel()
    n_infeasible = int((~ds.endpoint_feasible).sum())
    n_flagged = int(((ds.twin_min_clearance <= 0.0) | (ds.twin_max_residual > RESID_TOL)).sum())
    print(f"[data]  {ds.source.name}: {len(ds)} rows over {n_maps} maps, {len(train_subset)} train / "
          f"{len(val_subset)} val rows (held-out-MAP split); kept: {n_infeasible} end-infeasible, "
          f"{n_flagged} twin-flagged, {int(ds.jump.sum())} jump/at-rest entries")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] base_width={args.base_width}, embed_dim={args.embed_dim}, head_fusion={args.head_fusion}, "
          f"label_mode={args.label_mode} ({', '.join(names)}), {n_params} params, device={device}"
          + (", blur_terrain=ON (baseline)" if args.blur_terrain else "")
          + (", augmentation OFF" if args.no_augment else ""))

    baseline_rmse, baseline_preds = compute_baselines(ds, train_subset.indices, val_subset.indices)
    for name, value in baseline_rmse.items():
        print(f"[baseline] {name:<18s} RMSE: {fmt(value, names)}")
        run.summary.update(
            {f"baseline_{name.replace(' ', '_').replace('+', 'plus')}_{n}": v for n, v in zip(names, value.tolist())}
        )

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_since_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for (patch, command), y in train_loader:
            patch, command, y = patch.to(device), command.to(device), y.to(device)
            if not args.no_augment:
                patch, command, y = mirror_augment(patch, command, y)

            optimizer.zero_grad()
            y_hat = model(prepare_patch(patch, args.blur_terrain), command)
            loss = mse_loss(y_hat, target_transform.forward(y))
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            train_loss_sum += loss.item() * patch.shape[0]
            train_n += patch.shape[0]
        train_loss = train_loss_sum / max(train_n, 1)

        val_loss, val_pred, val_target = evaluate(model, val_loader, target_transform, device, args.blur_terrain)
        val_rmse = rmse_per_head(val_pred, val_target)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss, best_epoch = val_loss, epoch
            epochs_since_improve = 0
            torch.save(build_checkpoint(model, optimizer, epoch, val_loss, ds, args), checkpoint_path)
        else:
            epochs_since_improve += 1

        run.log(
            dict(
                train_loss=train_loss,
                val_loss=val_loss,
                val_minus_train_loss=val_loss - train_loss,
                lr=optimizer.param_groups[0]["lr"],
                **{f"val_rmse_{n}": v for n, v in zip(names, val_rmse.tolist())},
            ),
            step=epoch,
        )

        if epoch % args.log_every == 0 or epoch == args.epochs or improved:
            marker = " *" if improved else ""
            print(f"[{epoch:4d}/{args.epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"val_rmse=({fmt(val_rmse, names)}){marker}")

        if args.patience > 0 and epochs_since_improve >= args.patience:
            print(f"early stop at epoch {epoch}: no val_loss improvement in {args.patience} epochs")
            break

    print(f"[done]  best val_loss={best_val_loss:.4f} @ epoch {best_epoch}, checkpoint={checkpoint_path}")
    run.summary["best_val_loss"] = best_val_loss
    run.summary["best_epoch"] = best_epoch

    if best_epoch == -1:
        print("[done]  no checkpoint saved (val_loss never improved) -- skipping final report")
        return None

    run.save(str(checkpoint_path), base_path=str(checkpoint_path.parent), policy="now")
    final_report(checkpoint_path, ds, val_loader, val_subset.indices, baseline_preds, device, run,
                 blur_terrain=args.blur_terrain)
    return checkpoint_path


def self_test(args: argparse.Namespace) -> None:
    """The augmentation's exact form, the shrunk family table, and two short trains on
    custom_dataset's synthetic files (two files sharing a map) with a checkpoint round-trip each.
    CPU, wandb disabled."""
    from feasibility.mppi_learning.custom_dataset import _write_synthetic
    from feasibility.mppi_learning.spawn_sampling import PATCH_SPEC

    spec = PATCH_SPEC
    patch = torch.randn(2, 1, spec.ny, spec.nx)
    command = torch.tensor([[0.8, 0.1, 0.4, -0.2], [0.3, -0.1, -0.6, 0.3]])
    y = torch.rand(2, 2)
    flip = torch.tensor([True, False])
    a_patch, a_command, a_y = mirror_augment(patch, command, y, flip=flip)
    assert torch.equal(a_patch[0], torch.flip(patch[0], dims=[-2])) and torch.equal(a_patch[1], patch[1])
    assert torch.equal(a_command, torch.tensor([[0.8, 0.1, -0.4, 0.2], [0.3, -0.1, -0.6, 0.3]])) and torch.equal(a_y, y)
    twice = mirror_augment(a_patch, a_command, a_y, flip=flip)
    assert torch.equal(twice[0], patch) and torch.equal(twice[1], command), "mirroring twice is not the identity"
    print("[augment] row flip + wz_mean/wz_slope negated, v kept, labels unchanged, involution ok")

    groups = torch.tensor([0, 0, 1])
    table = group_mean_table(groups, torch.tensor([[1.0], [1.0], [4.0]]), 4)
    assert table.shape == (4, 1) and torch.allclose(table[2:], torch.full((2, 1), 2.0))  # empty -> global mean
    assert 1.0 < table[0].item() < 2.0 < table[1].item() < 4.0, table  # shrunk toward 2.0 from both sides
    print("[per-family mean] each family shrunk toward the global mean, an empty one IS the global mean")

    with tempfile.TemporaryDirectory() as tmp:
        a, b = pathlib.Path(tmp) / "dataset_mppi_a.h5", pathlib.Path(tmp) / "dataset_mppi_b.h5"
        _write_synthetic(a, [f"maps/m{i}" for i in range(4)], 16, seed=0)
        _write_synthetic(b, ["maps/m3", "maps/m4", "maps/m5"], 16, seed=1)  # m3 in both files
        args.dataset = [a, b]
        args.device = "cpu"
        args.epochs = min(args.epochs, 3)  # a smoke test, not a fit -- capped, never raised
        args.batch_size = 16
        args.warmup_epochs = 1
        args.patience = 0
        args.log_every = 1

        n_rows: dict[bool, int] = {}
        # the filters, label mode, head fusion and trunk are independent, so one second run covers them all
        for drop, label_mode, head_fusion, trunk in ((False, "pos_rot", "film", "original"),
                                                     (True, "pos_rpy", "concat", TRUNK_STYLE)):
            args.drop_endpoint_infeasible = args.drop_twin_flagged = drop
            args.label_mode, args.head_fusion, args.trunk = label_mode, head_fusion, trunk
            args.checkpoint = None
            expected = default_checkpoint_path(args)
            args.checkpoint = pathlib.Path(tmp) / expected.name
            assert expected.name == ("dataset_mppi_a_plus1_concat_rpy_noinfeasible_noflagged_trt.pt" if drop
                                     else "dataset_mppi_a_plus1.pt"), expected.name
            print(f"\n[self-test] training with filters={drop}, label_mode={label_mode}, head_fusion={head_fusion}, trunk={trunk}")
            run = wandb.init(mode="disabled", config=vars(args))
            try:
                checkpoint_path = train(args, run)
            finally:
                run.finish()
            assert checkpoint_path is not None and checkpoint_path.exists(), "no checkpoint saved"

            ds = WindowDivergenceDataset(args.dataset, label_mode=label_mode, drop_endpoint_infeasible=drop,
                                         drop_twin_flagged=drop)
            n_rows[drop] = len(ds)
            model, ckpt = load_checkpoint(checkpoint_path, torch.device("cpu"))
            assert model.label_mode == label_mode == ckpt["label_mode"] and model.head_fusion == head_fusion
            assert isinstance(model, TrtFriendlyWindowDivergenceNet) == (ckpt["trunk"] == trunk == TRUNK_STYLE)
            assert tuple(ckpt["target_names"]) == model.target_names == ds.TARGET_NAMES
            assert ckpt["label_attrs"] == label_attrs(ds) and ckpt["label_attrs"]["k_turn"] == 1.0
            assert all(type(v) in (str, float, int) for v in ckpt["label_attrs"].values()), ckpt["label_attrs"]
            assert ckpt["drop_endpoint_infeasible"] == ckpt["drop_twin_flagged"] == drop
            assert ckpt["dataset_paths"] == [str(a), str(b)] and math.isfinite(ckpt["val_loss"])
            prediction = model.predict(ds.patch[:4], ds.command[:4])
            assert prediction.shape == (4, len(ds.TARGET_NAMES)) and torch.isfinite(prediction).all()
            assert (prediction >= 0).all()
            print(f"[checkpoint] reloaded epoch {ckpt['epoch']}, label_attrs {ckpt['label_attrs']}, "
                  f"predict() range [{prediction.min():.4f}, {prediction.max():.4f}]")

        assert n_rows[True] < n_rows[False], n_rows
        print(f"\n[filters] {n_rows[False]} rows by default, {n_rows[True]} with both filters")

        ckpt = torch.load(checkpoint_path, weights_only=False)
        ckpt["wz_scale"] *= 2.0
        torch.save(ckpt, checkpoint_path)
        try:
            load_checkpoint(checkpoint_path, torch.device("cpu"))
            raise AssertionError("a checkpoint with other command scales was loaded")
        except ValueError:
            pass
        print("[checkpoint] a checkpoint trained at other command scales is refused")
    print("all self-checks ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=pathlib.Path, nargs="+", default=None, help="dataset_mppi_*.h5 file(s), trained together")
    parser.add_argument("--drop-endpoint-infeasible", action="store_true", help="drop rows whose nominal window end is settle-infeasible")
    parser.add_argument("--drop-twin-flagged", action="store_true", help=f"drop rows the twin flags (clearance <= 0 or residual > {RESID_TOL:g})")
    parser.add_argument("--val-frac", type=float, default=0.2, help="fraction of MAPS held out for validation (default: 0.2)")
    parser.add_argument("--seed", type=int, default=0, help="seeds the split, model init, and augmentation (default: 0)")
    parser.add_argument("--base-width", type=int, default=DEFAULT_BASE_WIDTH, help=f"trunk channel unit (default: {DEFAULT_BASE_WIDTH})")
    parser.add_argument("--embed-dim", type=int, default=DEFAULT_EMBED_DIM, help=f"command-embedding / FiLM width (default: {DEFAULT_EMBED_DIM})")
    parser.add_argument("--head-fusion", type=str, default="film", choices=HEAD_FUSIONS, help="(default: film)")
    parser.add_argument("--trunk", type=str, default="original", choices=TRUNK_STYLES, help="original (replicate pad + channel LayerNorm) or trt_friendly (zero pad + BatchNorm, model_tensorRT_friendly.py) (default: original)")
    parser.add_argument("--label-mode", type=str, default="pos_rot", choices=LABEL_MODES, help="(e_pos, e_rot), or (e_pos, e_roll, e_pitch, e_yaw) (default: pos_rot)")
    parser.add_argument("--no-augment", action="store_true", help="disable the y-mirror augmentation (on by default)")
    parser.add_argument("--blur-terrain", action="store_true", help="baseline: relief replaced by its per-sample mean, at train AND eval time")
    parser.add_argument("--batch-size", type=int, default=256, help="default: 256")
    parser.add_argument("--epochs", type=int, default=100, help="default: 100")
    parser.add_argument("--lr", type=float, default=3e-4, help="AdamW learning rate (default: 3e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay (default: 1e-4)")
    parser.add_argument("--lr-schedule", choices=("cosine", "none"), default="cosine", help="cosine-to-0 with linear warmup, or a constant LR (default: cosine)")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="linear-warmup length for --lr-schedule cosine (default: 5)")
    parser.add_argument("--patience", type=int, default=20, help="epochs of no val_loss improvement before early stop; 0 disables (default: 20)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="torch device (default: cuda if available else cpu)")
    parser.add_argument("--checkpoint", type=pathlib.Path, default=None, help="save path (default: outputs/checkpoints/<first stem>[_plus<N>]<tags>.pt)")
    parser.add_argument("--log-every", type=int, default=10, help="epochs between stdout progress lines (default: 10)")
    parser.add_argument("--self-test", action="store_true", help="train a few epochs on synthetic files with wandb disabled, then assert this module's invariants")
    parser.add_argument("--wandb-project", type=str, default="feasibility-mppi-window-divergence")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-name", type=str, default=None)
    args = parser.parse_args()

    if args.self_test:
        self_test(args)
        return
    if not args.dataset:
        parser.error("--dataset is required (or --self-test)")

    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name,
                     mode=args.wandb_mode, config={**vars(args), "dataset": [str(p) for p in args.dataset]})
    try:
        log_checkpoint(run, train(args, run))
    finally:
        run.finish()


if __name__ == "__main__":
    main()
