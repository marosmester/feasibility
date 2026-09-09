"""Trains model.GridPoseErrorNet on custom_dataset.GridPoseErrorDataset -- design.md
section 10 names this file's job precisely: "masked loss (5b), map-level split (6b),
augmentation (6a), baselines (7)". Structured after learning/train.py (same
build_checkpoint/load_checkpoint contract, same wandb logging shape), with the grid_learning
nuances substituted in:

* The loss is masked, not plain MSE -- design.md 5b. A dataset row's label is a whole
  [G, G, 2] field, and roughly half the lattice is masked (obstacle-blocked spawn footprint or a
  diverged solve), so every loss/metric here averages over valid (cell, head) entries only, never
  over the fixed cell count. Masked cells are exact zeros in the file (never NaN -- see
  custom_dataset.py's module docstring), so nothing here needs a NaN-safe torch.where.
* The train/val split is by MAP, not by row (design.md 6b, `by_map=True`): with e.g. 100
  maps x 10 commands, a row-level split leaks the terrain of almost every val row into train
  under a different wz, so the val score would measure wz-interpolation on memorized terrain
  rather than transfer to unseen terrain. `custom_dataset.split_dataset` (row-level) is still
  there and reachable via `--row-split`, purely as an ablation/debug knob.
* Mirror-symmetry augmentation (design.md 6a) is applied per-sample, p=0.5, by default: the
  robot and both grids are exactly y-symmetric, so it is a free 2x on data, not an approximation.
* Three non-parametric baselines from design.md section 7 (global mean, per-wz mean field)
  are computed once up front, before any training, and logged to wandb's summary so a run's own
  val RMSE has something to be judged against; the third (blur-the-terrain) needs the actual
  architecture trained on degraded input, so it is a `--blur-terrain` flag that reuses this same
  training loop rather than a separate code path -- run twice (with and without) to get that
  comparison. The flag is passed to `model.forward`, which flattens the RELIEF rather than the
  raw heightmap (see its docstring: blurring the input first would leave exactly zero, taking
  section 7's "is there a box somewhere" scalar with it), and it is threaded into every eval path
  too -- the val loop, the final report, and both sanity checks -- so a degraded-input run is
  never scored on undegraded terrain.
* Like model.py's TargetTransform docstring, the transform is fit on TRAIN rows' VALID
  (mask == True) cells only.
* End of run: a short battery of the free checks design.md section 7 calls out -- top-decile
  (by true e_pos) masked RMSE, per-head R^2, the wz~=0 sanity check, and the mirror-equivariance
  check -- against the BEST checkpoint (reloaded from disk, not just the last epoch's weights).

CLI parameters:
    --dataset PATH               dataset_grid_*.h5 (custom_dataset.GridPoseErrorDataset schema)
                                  (default: outputs/dataset_grid_box_random_M2_L5_g15.h5)
    --val-frac FLOAT             fraction of MAPS (or rows, with --row-split) held out for
                                  validation (default: 0.2)
    --row-split                  use the row-level split (custom_dataset.split_dataset) instead
                                  of the map-level one -- ablation/debug only, leaks terrain
                                  identity into val (see module docstring)
    --seed INT                   seeds the split, model init, and augmentation (default: 0)
    --base-width INT             trunk channel unit (default: 32, model.DEFAULT_BASE_WIDTH)
    --embed-dim INT              wz embedding / FiLM width (default: 32, model.DEFAULT_EMBED_DIM)
    --no-augment                 disable the y-mirror + wz-negation augmentation (on by default)
    --blur-terrain                the "blur-the-terrain" baseline: flatten the terrain to its own
                                  per-sample mean relief before every forward pass, everywhere
                                  (train AND val) -- same architecture, degraded input
    --batch-size INT             default: 16 (16 x 225 = 3600 labelled cells/step)
    --epochs INT                 default: 200
    --lr FLOAT                   AdamW learning rate (default: 3e-4)
    --weight-decay FLOAT         AdamW weight decay (default: 1e-4)
    --lr-schedule {cosine,none}  cosine-to-0 with linear warmup, or a constant LR (default: cosine)
    --warmup-epochs INT          linear-warmup length for --lr-schedule cosine (default: 5)
    --patience INT                epochs of no val_loss improvement before early stop; 0 disables
                                  (default: 20)
    --device STR                  torch device (default: cuda if available else cpu)
    --checkpoint PATH             save path (default: outputs/checkpoints/<dataset stem>.pt)
    --log-every INT               epochs between stdout progress lines (default: 10)
    --wandb-project STR           (default: "feasibility-grid-pose-error")
    --wandb-entity STR             (default: None)
    --wandb-mode {online,offline,disabled}  (default: online)
    --wandb-name STR               (default: None, wandb auto-generates a name)

Usage:
    python src/feasibility/grid_learning/train.py
    python src/feasibility/grid_learning/train.py --dataset outputs/dataset_grid_box_random_M2_L5_g15.h5 --epochs 100
    python src/feasibility/grid_learning/train.py --row-split          # ablation: the leaky split
    python src/feasibility/grid_learning/train.py --blur-terrain       # baseline: degraded input, same net
    python src/feasibility/grid_learning/train.py --wandb-mode offline
    python src/feasibility/grid_learning/train.py --wandb-mode disabled  # no network calls at all
    python -c "
    import torch
    from feasibility.grid_learning.train import load_checkpoint
    model, ckpt = load_checkpoint('outputs/checkpoints/dataset_grid_box_random_M2_L5_g15.pt', torch.device('cpu'))
    print(model.predict(torch.zeros(1, 100, 100), torch.zeros(1), ckpt['spawn_xy'], extent=ckpt['extent']))
    "
"""
from __future__ import annotations

import argparse
import math
import pathlib

import torch
import wandb

from feasibility.comparator.common import OUT_DIR
from feasibility.comparator.provenance import git_provenance
from feasibility.grid_learning.custom_dataset import GridPoseErrorDataset
from feasibility.grid_learning.custom_dataset import make_dataloaders
from feasibility.grid_learning.custom_dataset import Normalizer
from feasibility.grid_learning.custom_dataset import TARGET_NAMES
from feasibility.grid_learning.model import DEFAULT_BASE_WIDTH
from feasibility.grid_learning.model import DEFAULT_EMBED_DIM
from feasibility.grid_learning.model import GridPoseErrorNet
from feasibility.grid_learning.model import TargetTransform

DEFAULT_DATASET = OUT_DIR / "dataset_grid_box_random_M2_L5_g15.h5"


def fmt(values: torch.Tensor) -> str:
    """[K] per-head tensor -> "e_pos=..., e_rot=..." for the stdout lines."""
    return ", ".join(f"{n}={v:.4f}" for n, v in zip(TARGET_NAMES, values.tolist()))


# --- masked loss / masked metrics (design.md 5b) -----------------------------------------


def masked_mse_loss(y_hat: torch.Tensor, y_target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked MSE over valid (cell, head) entries only -- design.md section 5b. Both
    y_hat/y_target must already be in TargetTransform space; `mask` is [*, G, G] bool, broadcast
    over the trailing head dim since a cell's validity does not depend on which head is read.
    Normalizing by mask.sum() (not the fixed cell count) keeps the loss scale independent of how
    many cells a given map happens to block."""
    mask_f = mask.unsqueeze(-1).expand_as(y_hat).float()
    return (mask_f * (y_hat - y_target) ** 2).sum() / mask_f.sum().clamp_min(1.0)


def masked_rmse_per_head(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """[*, G, G, K] physical pred/target + [*, G, G] mask -> [K] RMSE per head over valid cells,
    reducing over every leading dim -- shared by the training-loop val metric and every baseline
    below so they are all scored identically."""
    mask_f = mask.unsqueeze(-1).expand_as(pred).float()
    reduce_dims = tuple(range(pred.dim() - 1))
    se = mask_f * (pred - target) ** 2
    mse = se.sum(dim=reduce_dims) / mask_f.sum(dim=reduce_dims).clamp_min(1.0)
    return mse.sqrt()


# --- mirror-symmetry augmentation (design.md 6a) ------------------------------------------


def mirror_augment(
    heightmap: torch.Tensor, wz: torch.Tensor, y: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample, p=0.5: reflect the map in Y and negate wz -- an EXACT symmetry of the
    data-generating process (the robot and both grids are y-symmetric), not an approximation, per
    design.md section 6a. e_pos/e_rot are magnitudes so their VALUES are unchanged, only
    which lattice cell they sit at -- hence y/mask flip along the row axis (world Y) rather than
    being renormalized."""
    B = heightmap.shape[0]
    flip = torch.rand(B, device=heightmap.device) < 0.5
    heightmap = torch.where(flip[:, None, None], torch.flip(heightmap, dims=[-2]), heightmap)
    wz = torch.where(flip, -wz, wz)
    y = torch.where(flip[:, None, None, None], torch.flip(y, dims=[-3]), y)
    mask = torch.where(flip[:, None, None], torch.flip(mask, dims=[-2]), mask)
    return heightmap, wz, y, mask


# --- non-parametric baselines (design.md 7) -----------------------------------------------


def masked_field_mean(y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """[n, G, G, K] targets + [n, G, G] mask -> [G, G, K] elementwise mean over the rows where
    each cell was valid -- the building block of the "per-wz mean field" baseline."""
    mask_f = mask.unsqueeze(-1).float()
    return (y * mask_f).sum(dim=0) / mask_f.sum(dim=0).clamp_min(1.0)


def baseline_global_mean(
    ds: GridPoseErrorDataset, train_idx: list[int], val_idx: list[int]
) -> torch.Tensor:
    """Predict one constant (e_pos, e_rot) -- the train-set valid-cell mean -- everywhere.
    design.md section 7's sanity floor: beating it proves almost nothing, but a run that
    can't beat it is broken, not just unimpressive."""
    y_train, mask_train = ds.y[train_idx], ds.mask[train_idx]
    mask_f = mask_train.unsqueeze(-1).float()
    mean_vec = (y_train * mask_f).sum(dim=(0, 1, 2)) / mask_f.sum(dim=(0, 1, 2)).clamp_min(1.0)
    y_val, mask_val = ds.y[val_idx], ds.mask[val_idx]
    pred = mean_vec.view(1, 1, 1, -1).expand_as(y_val)
    return masked_rmse_per_head(pred, y_val, mask_val)


def baseline_per_wz_mean(
    ds: GridPoseErrorDataset, train_idx: list[int], val_idx: list[int]
) -> torch.Tensor:
    """Look up the train-set mean FIELD for a val row's own wz (nearest match -- wz is sampled
    from a fixed, equidistant grid, see generate_dataset.py's module docstring). design.md
    section 7's real bar: beating this proves the model uses the TERRAIN, not just the commanded
    wz."""
    wz_train, y_train, mask_train = ds.wz[train_idx], ds.y[train_idx], ds.mask[train_idx]
    distinct_wz = wz_train.unique()
    fields = torch.stack(
        [masked_field_mean(y_train[wz_train == w], mask_train[wz_train == w]) for w in distinct_wz]
    )  # [K, G, G, out_dim]
    wz_val, y_val, mask_val = ds.wz[val_idx], ds.y[val_idx], ds.mask[val_idx]
    nearest = (wz_val.unsqueeze(1) - distinct_wz.unsqueeze(0)).abs().argmin(dim=1)
    pred = fields[nearest]
    return masked_rmse_per_head(pred, y_val, mask_val)


# --- checkpointing (mirrors learning/train.py's build_checkpoint/load_checkpoint) ---------------


def build_checkpoint(
    model: GridPoseErrorNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    ds: GridPoseErrorDataset,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Self-describing checkpoint -- everything load_checkpoint() needs is IN the file, so it
    loads without this module's source or the dataset that trained it. `spawn_xy`/`extent` are
    included because model.forward() takes them as call-time arguments rather than baking them
    in (see model.py's GridPoseErrorNet docstring), so an inference caller needs them from
    somewhere."""
    target_transform = model.target_transform
    assert target_transform is not None  # train() always attaches one before calling this
    return dict(
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        epoch=epoch,
        val_loss=val_loss,
        base_width=args.base_width,
        embed_dim=args.embed_dim,
        wheel_radius=model.wheel_radius,
        extent=ds.extent,
        spawn_xy=ds.spawn_xy,
        target_names=TARGET_NAMES,
        target_transform_mean=target_transform.normalizer.mean,
        target_transform_std=target_transform.normalizer.std,
        dataset_path=str(ds.source),
        dataset_git=ds.git,
        train_git=git_provenance(),
        args=vars(args),
    )


def load_checkpoint(
    path: pathlib.Path, device: torch.device
) -> tuple[GridPoseErrorNet, dict[str, object]]:
    """Rebuilds a GridPoseErrorNet from a build_checkpoint() dict. Unlike
    learning/train.py's load_checkpoint there is no x_normalizer to return alongside: the
    heightmap/wz inputs here are never normalized (see custom_dataset.make_dataloaders'
    docstring) -- only the targets are, via the returned model's own .target_transform."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    target_transform = TargetTransform(
        normalizer=Normalizer(
            mean=ckpt["target_transform_mean"].to(device),
            std=ckpt["target_transform_std"].to(device),
        )
    )
    model = GridPoseErrorNet(
        base_width=ckpt["base_width"],
        embed_dim=ckpt["embed_dim"],
        wheel_radius=ckpt["wheel_radius"],
        target_transform=target_transform,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt


# --- LR schedule (design.md section 9: cosine to 0, ~5 epoch linear warmup) ---------------


def build_scheduler(
    optimizer: torch.optim.Optimizer, warmup_epochs: int, total_epochs: int, steps_per_epoch: int
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    total_steps = max(warmup_steps + 1, total_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --- end-of-run report (design.md section 7's metrics + the two free sanity checks) -------


@torch.no_grad()
def collect_val_predictions(
    model: GridPoseErrorNet,
    val_loader: torch.utils.data.DataLoader,
    spawn_xy: torch.Tensor,
    extent: float,
    target_transform: TargetTransform,
    device: torch.device,
    blur_terrain: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Runs the whole val set once, returning (pred, target, mask) stacked over rows, in
    PHYSICAL units -- shared by every metric in final_report() so they all see the same
    predictions. `blur_terrain` has to be threaded through here (and into every other eval path
    below): a --blur-terrain run is the SAME degraded input at train and val time, so scoring its
    checkpoint on undegraded terrain would report a number no training step ever optimized."""
    model.eval()
    preds, targets, masks = [], [], []
    for (heightmap, wz), y, mask in val_loader:
        heightmap, wz = heightmap.to(device), wz.to(device)
        y_hat = model(heightmap, wz, spawn_xy, extent=extent, blur_terrain=blur_terrain)
        preds.append(target_transform.inverse(y_hat).cpu())
        targets.append(y.cpu())
        masks.append(mask.cpu())
    return torch.cat(preds), torch.cat(targets), torch.cat(masks)


def top_decile_rmse(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, decile: float = 0.1
) -> torch.Tensor:
    """Masked RMSE restricted to the hardest `decile` of valid cells by TRUE e_pos --
    design.md section 7: masked RMSE over ALL valid cells is dominated by the
    near-constant flat-ground background and is easy to score well on for the wrong reason; the
    collision cells in this tail are the entire point of the model."""
    e_pos_idx = TARGET_NAMES.index("e_pos")
    valid_e_pos = target[..., e_pos_idx][mask]
    if valid_e_pos.numel() == 0:
        return torch.full((len(TARGET_NAMES),), float("nan"))
    k = max(1, int(round(valid_e_pos.numel() * decile)))
    threshold = valid_e_pos.kthvalue(valid_e_pos.numel() - k + 1).values
    hard_mask = mask & (target[..., e_pos_idx] >= threshold)
    return masked_rmse_per_head(pred, target, hard_mask)


def r_squared(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """[K] R^2 per head over valid cells: 1 - SS_res/SS_tot, SS_tot against the valid-cell mean
    (not zero) -- design.md section 7."""
    mask_f = mask.unsqueeze(-1).expand_as(pred).float()
    reduce_dims = tuple(range(pred.dim() - 1))
    n = mask_f.sum(dim=reduce_dims).clamp_min(1.0)
    mean = (target * mask_f).sum(dim=reduce_dims) / n
    ss_res = (mask_f * (pred - target) ** 2).sum(dim=reduce_dims)
    ss_tot = (mask_f * (target - mean) ** 2).sum(dim=reduce_dims).clamp_min(1e-8)
    return 1.0 - ss_res / ss_tot


@torch.no_grad()
def wz_zero_check(
    model: GridPoseErrorNet,
    ds: GridPoseErrorDataset,
    val_idx: list[int],
    spawn_xy: torch.Tensor,
    extent: float,
    device: torch.device,
    blur_terrain: bool = False,
) -> torch.Tensor | None:
    """Rows with wz~=0 command no motion, so the true field is ~0 everywhere -- the prediction
    must be too. Free sanity check, needs no baseline at all (design.md section 7).
    Returns None if the val split happens to contain no such row."""
    idx = [i for i in val_idx if abs(float(ds.wz[i])) < 1e-6]
    if not idx:
        return None
    heightmap = ds.heightmap[ds.map_index[idx]].to(device)
    wz = ds.wz[idx].to(device)
    model.eval()
    pred = model.predict(heightmap, wz, spawn_xy, extent=extent, blur_terrain=blur_terrain).cpu()
    mask = ds.mask[idx].cpu()
    return masked_rmse_per_head(pred, torch.zeros_like(pred), mask)


@torch.no_grad()
def mirror_equivariance_check(
    model: GridPoseErrorNet,
    ds: GridPoseErrorDataset,
    val_idx: list[int],
    spawn_xy: torch.Tensor,
    extent: float,
    device: torch.device,
    blur_terrain: bool = False,
    n: int = 8,
) -> float:
    """f(flip(h), -wz) should equal flip(f(h, wz)) to numerical noise once trained -- the exact
    symmetry mirror_augment() trains on doubles as this test (design.md section 6a).
    Compared in MODEL space (pre-TargetTransform), so this checks the net's own equivariance
    rather than round-tripping through log1p/standardize. Returns the max abs difference."""
    idx = val_idx[:n]
    heightmap = ds.heightmap[ds.map_index[idx]].to(device)
    wz = ds.wz[idx].to(device)
    model.eval()
    y_hat = model(heightmap, wz, spawn_xy, extent=extent, blur_terrain=blur_terrain)
    y_hat_flip = model(
        torch.flip(heightmap, dims=[-2]), -wz, spawn_xy, extent=extent, blur_terrain=blur_terrain
    )
    return (torch.flip(y_hat, dims=[-3]) - y_hat_flip).abs().max().item()


def final_report(
    checkpoint_path: pathlib.Path,
    ds: GridPoseErrorDataset,
    val_loader: torch.utils.data.DataLoader,
    val_idx: list[int],
    spawn_xy: torch.Tensor,
    device: torch.device,
    run: wandb.sdk.wandb_run.Run,
    blur_terrain: bool = False,
) -> None:
    """Reloads the BEST checkpoint (not just the last epoch's in-memory weights) and prints the
    design.md section 7 battery: overall + top-decile masked RMSE, R^2 per head, the wz~=0
    check, and mirror equivariance."""
    model, ckpt = load_checkpoint(checkpoint_path, device)
    target_transform = model.target_transform
    assert target_transform is not None
    pred, target, mask = collect_val_predictions(
        model, val_loader, spawn_xy, ds.extent, target_transform, device, blur_terrain
    )

    overall = masked_rmse_per_head(pred, target, mask)
    decile = top_decile_rmse(pred, target, mask)
    r2 = r_squared(pred, target, mask)
    wz0 = wz_zero_check(model, ds, val_idx, spawn_xy, ds.extent, device, blur_terrain)
    mirror_diff = mirror_equivariance_check(model, ds, val_idx, spawn_xy, ds.extent, device, blur_terrain)

    print(f"\n[final report] checkpoint={checkpoint_path} (epoch {ckpt['epoch']})")
    print(f"[final report] val masked RMSE, all valid cells:    {fmt(overall)}")
    print(f"[final report] val masked RMSE, top-decile e_pos:   {fmt(decile)}")
    print(f"[final report] val R^2 per head:                    {fmt(r2)}")
    if wz0 is not None:
        print(f"[final report] wz~=0 sanity (should be ~0):         {fmt(wz0)}")
    print(f"[final report] mirror-equivariance max |diff| (model space): {mirror_diff:.4e}")

    run.summary.update(
        {
            **{f"final_rmse_all_{n}": v for n, v in zip(TARGET_NAMES, overall.tolist())},
            **{f"final_rmse_top_decile_{n}": v for n, v in zip(TARGET_NAMES, decile.tolist())},
            **{f"final_r2_{n}": v for n, v in zip(TARGET_NAMES, r2.tolist())},
            "final_mirror_equivariance_max_diff": mirror_diff,
        }
    )


def train(args: argparse.Namespace, run: wandb.sdk.wandb_run.Run) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    train_loader, val_loader, ds, train_subset, val_subset = make_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        val_frac=args.val_frac,
        seed=args.seed,
        by_map=not args.row_split,
        device=device,
    )
    spawn_xy = ds.spawn_xy.to(device)

    y_train, mask_train = ds.y[train_subset.indices], ds.mask[train_subset.indices]
    target_transform = TargetTransform.fit(y_train[mask_train])

    tag = "_blur" if args.blur_terrain else ("_rowsplit" if args.row_split else "")
    checkpoint_path = args.checkpoint or OUT_DIR / "checkpoints" / f"{ds.source.stem}{tag}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    model = GridPoseErrorNet(
        base_width=args.base_width, embed_dim=args.embed_dim, target_transform=target_transform
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (
        build_scheduler(optimizer, args.warmup_epochs, args.epochs, len(train_loader))
        if args.lr_schedule == "cosine"
        else None
    )

    n_maps = ds.map_index.unique().numel()
    print(
        f"[data]  {ds.source.name}: {len(ds)} rows over {n_maps} maps, "
        f"{len(train_loader.dataset)} train / {len(val_loader.dataset)} val rows "
        f"({'row' if args.row_split else 'map'}-level split)"
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] base_width={args.base_width}, embed_dim={args.embed_dim}, {n_params} params, device={device}")

    baseline_global = baseline_global_mean(ds, train_subset.indices, val_subset.indices)
    baseline_per_wz = baseline_per_wz_mean(ds, train_subset.indices, val_subset.indices)
    print(f"[baseline] global-mean RMSE: {fmt(baseline_global)}")
    print(f"[baseline] per-wz-mean RMSE: {fmt(baseline_per_wz)}")
    run.summary.update(
        {
            **{f"baseline_global_mean_{n}": v for n, v in zip(TARGET_NAMES, baseline_global.tolist())},
            **{f"baseline_per_wz_mean_{n}": v for n, v in zip(TARGET_NAMES, baseline_per_wz.tolist())},
        }
    )

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_since_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for (heightmap, wz), y, mask in train_loader:
            heightmap, wz, y, mask = heightmap.to(device), wz.to(device), y.to(device), mask.to(device)
            if not args.no_augment:
                heightmap, wz, y, mask = mirror_augment(heightmap, wz, y, mask)

            optimizer.zero_grad()
            y_hat = model(heightmap, wz, spawn_xy, extent=ds.extent, blur_terrain=args.blur_terrain)
            loss = masked_mse_loss(y_hat, target_transform.forward(y), mask)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            train_loss_sum += loss.item() * heightmap.shape[0]
            train_n += heightmap.shape[0]
        train_loss = train_loss_sum / train_n

        model.eval()
        val_loss_sum, val_n = 0.0, 0
        preds, targets, masks = [], [], []
        with torch.no_grad():
            for (heightmap, wz), y, mask in val_loader:
                heightmap, wz, y, mask = heightmap.to(device), wz.to(device), y.to(device), mask.to(device)
                y_hat = model(heightmap, wz, spawn_xy, extent=ds.extent, blur_terrain=args.blur_terrain)
                val_loss_sum += masked_mse_loss(y_hat, target_transform.forward(y), mask).item() * heightmap.shape[0]
                val_n += heightmap.shape[0]
                preds.append(target_transform.inverse(y_hat).cpu())
                targets.append(y.cpu())
                masks.append(mask.cpu())
        val_loss = val_loss_sum / val_n
        # Scored through masked_rmse_per_head, the same function every baseline and the final
        # report use, rather than re-deriving the masked reduction inline -- the whole val set is
        # already resident (GridPoseErrorDataset loads eagerly onto `device`), so stacking it
        # costs nothing and removes the one place these numbers could drift apart.
        val_rmse = masked_rmse_per_head(torch.cat(preds), torch.cat(targets), torch.cat(masks))

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
                lr=optimizer.param_groups[0]["lr"],
                **{f"val_rmse_{n}": v for n, v in zip(TARGET_NAMES, val_rmse.tolist())},
            ),
            step=epoch,
        )

        if epoch % args.log_every == 0 or epoch == args.epochs or improved:
            marker = " *" if improved else ""
            print(
                f"[{epoch:4d}/{args.epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_rmse=({fmt(val_rmse)}){marker}"
            )

        if args.patience > 0 and epochs_since_improve >= args.patience:
            print(f"early stop at epoch {epoch}: no val_loss improvement in {args.patience} epochs")
            break

    print(f"[done]  best val_loss={best_val_loss:.4f} @ epoch {best_epoch}, checkpoint={checkpoint_path}")
    run.summary["best_val_loss"] = best_val_loss
    run.summary["best_epoch"] = best_epoch

    if best_epoch == -1:
        # val_loss never improved on inf -- every epoch produced a non-finite loss (e.g. blown-up
        # gradients). Nothing was ever saved, so there is no checkpoint for final_report to load.
        print("[done]  no checkpoint saved (val_loss never improved) -- skipping final report")
        return

    run.save(str(checkpoint_path), base_path=str(checkpoint_path.parent), policy="now")
    final_report(
        checkpoint_path, ds, val_loader, val_subset.indices, spawn_xy, device, run,
        blur_terrain=args.blur_terrain,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=pathlib.Path, default=DEFAULT_DATASET, help="dataset_grid_*.h5 (custom_dataset.GridPoseErrorDataset schema)")
    parser.add_argument("--val-frac", type=float, default=0.2, help="fraction of maps (or rows, with --row-split) held out for validation (default: 0.2)")
    parser.add_argument("--row-split", action="store_true", help="use the row-level split (custom_dataset.split_dataset) instead of the map-level one -- ablation/debug only, leaks terrain identity into val")
    parser.add_argument("--seed", type=int, default=0, help="seeds the split, model init, and augmentation (default: 0)")
    parser.add_argument("--base-width", type=int, default=DEFAULT_BASE_WIDTH, help=f"trunk channel unit (default: {DEFAULT_BASE_WIDTH})")
    parser.add_argument("--embed-dim", type=int, default=DEFAULT_EMBED_DIM, help=f"wz embedding / FiLM width (default: {DEFAULT_EMBED_DIM})")
    parser.add_argument("--no-augment", action="store_true", help="disable the y-mirror + wz-negation augmentation (on by default)")
    parser.add_argument("--blur-terrain", action="store_true", help='the "blur-the-terrain" baseline: flatten the terrain to its own per-sample mean relief before every forward pass, at train AND eval time')
    parser.add_argument("--batch-size", type=int, default=16, help="default: 16")
    parser.add_argument("--epochs", type=int, default=200, help="default: 200")
    parser.add_argument("--lr", type=float, default=3e-4, help="AdamW learning rate (default: 3e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay (default: 1e-4)")
    parser.add_argument("--lr-schedule", choices=("cosine", "none"), default="cosine", help="cosine-to-0 with linear warmup, or a constant LR (default: cosine)")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="linear-warmup length for --lr-schedule cosine (default: 5)")
    parser.add_argument("--patience", type=int, default=20, help="epochs of no val_loss improvement before early stop; 0 disables (default: 20)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="torch device (default: cuda if available else cpu)")
    parser.add_argument("--checkpoint", type=pathlib.Path, default=None, help="save path (default: outputs/checkpoints/<dataset stem>.pt)")
    parser.add_argument("--log-every", type=int, default=10, help="epochs between stdout progress lines (default: 10)")
    parser.add_argument("--wandb-project", type=str, default="feasibility-grid-pose-error")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-name", type=str, default=None)
    args = parser.parse_args()

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        mode=args.wandb_mode,
        config=vars(args),
    )
    try:
        train(args, run)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
