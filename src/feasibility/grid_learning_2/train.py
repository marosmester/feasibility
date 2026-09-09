"""Trains model.GridDivergenceNet on custom_dataset.GridDivergenceDataset -- design.md section 11
names this file's job as "masked loss, map split, mirror augmentation, baselines", with the recipe
in section 10 and the baseline table in section 8. Structured after grid_learning/train.py (same
build_checkpoint/load_checkpoint contract, same wandb logging shape), with the four things a
per-cell command field changes substituted in:

* **The forward signature lost `spawn_xy`/`extent`.** v2's readout is an integer centre-crop, not
  an `F.grid_sample`, so the lattice's world coordinates are a property of the architecture rather
  than a per-batch lookup (design.md section 3). `custom_dataset` already asserts the crop lands on
  the file's own `spawn_xy` when it opens the file, so nothing here re-checks it; both are still
  carried in the checkpoint as provenance for an inference caller that wants to verify the same
  thing against a different file.

* **The mirror augmentation flips the command FIELD and negates it** -- `wz -> -flip(wz, dims=[-2])`
  (design.md section 7c), where v1 only negated a scalar. Both halves matter: the command belonging
  to cell (i, j) must follow that cell to row G-1-i, *and* its sign flips because a mirrored turn
  is the opposite turn. Getting only one half right would train the net on labels that belong to a
  different command, which is the kind of bug that shows up as "it just won't fit" rather than as a
  crash.

* **The per-`wz` baseline is now per CELL.** In v1 a row had one command, so the baseline was a
  mean field looked up by the row's `wz`. Here every cell has its own command, so the lookup table
  is `[bin, i, j]` and each val cell reads the train mean for *its own* command bin at *its own*
  lattice position. This is a much stronger baseline than v1's -- it already knows both "where on
  the lattice" and "how hard a turn" -- which is the point: beating it is what shows the net reads
  terrain STRUCTURE rather than the marginal statistics of the lattice.

* **The `wz ~ 0` sanity check is per cell too, and is no longer a lottery.** v1 needed a whole val
  row whose single command happened to be ~0. v2's generator forces `+wz_zero_frac` (5% by default)
  of cells to exactly 0 precisely so this check stays available (design.md section 7b), so it runs
  over thousands of scattered cells in the ordinary case. Reported alongside the TRUE label at
  those same cells, because "no command, no motion, error ~ 0" is an assumption about the physics
  worth seeing measured rather than assumed.

Baselines (design.md section 8): global mean and the per-`wz` mean field are non-parametric and
computed up front, before any training, so a run's val RMSE has something to be judged against
immediately. `--blur-terrain` and `--head-fusion concat` are the two that need the actual
architecture trained, so they are flags that reuse this same loop -- run twice and compare. The
table's fifth row (v1's `GridPoseErrorNet` on constant fields) is deliberately NOT implemented
here: it would mean importing `feasibility.grid_learning`, which design.md section 11's dependency
stance forbids, and it is a different dataset anyway. Run that tree's own `train.py` on a v1 file
and compare the reported numbers by hand.

CLI parameters:
    --dataset PATH               dataset_grid2_*.h5 (custom_dataset.GridDivergenceDataset schema)
                                  (default: outputs/dataset_grid2_large_box_random0_M2_L10_g15.h5)
    --val-frac FLOAT             fraction of MAPS held out for validation (default: 0.2)
    --seed INT                   seeds the split, model init, and augmentation (default: 0)
    --base-width INT             trunk channel unit (default: 32, model.DEFAULT_BASE_WIDTH)
    --embed-dim INT              command-embedding / FiLM width (default: 32)
    --head-fusion {film,concat}  design.md section 5d's ablation: FiLM's multiplicative gate, or
                                  the honest additive control (default: film)
    --no-augment                 disable the y-mirror + command-field flip/negate (on by default)
    --blur-terrain               the "blur-the-terrain" baseline: flatten the relief to its own
                                  per-sample mean before every forward pass, at train AND eval time
    --wz-bins INT                bins for the per-wz mean-field baseline (default: 8)
    --batch-size INT             default: 16 (16 x 225 = 3600 labelled cells/step)
    --epochs INT                 default: 200
    --lr FLOAT                   AdamW learning rate (default: 3e-4)
    --weight-decay FLOAT         AdamW weight decay (default: 1e-4)
    --lr-schedule {cosine,none}  cosine-to-0 with linear warmup, or a constant LR (default: cosine)
    --warmup-epochs INT          linear-warmup length for --lr-schedule cosine (default: 5)
    --patience INT               epochs of no val_loss improvement before early stop; 0 disables
                                  (default: 20)
    --device STR                 torch device (default: cuda if available else cpu)
    --checkpoint PATH            save path (default: outputs/checkpoints/<dataset stem>.pt)
    --log-every INT              epochs between stdout progress lines (default: 10)
    --self-test                  train a few epochs on a synthetic file in a temp dir, with wandb
                                  disabled, then assert the loss/augmentation/baseline/checkpoint
                                  invariants -- this module's smoke test, no dataset or GPU needed
    --wandb-project STR          (default: "feasibility-grid-divergence")
    --wandb-entity STR           (default: None)
    --wandb-mode {online,offline,disabled}  (default: online)
    --wandb-name STR             (default: None, wandb auto-generates a name)

Usage:
    python src/feasibility/grid_learning_2/train.py --self-test        # no dataset needed
    python src/feasibility/grid_learning_2/train.py
    python src/feasibility/grid_learning_2/train.py --dataset outputs/dataset_grid2_..._M2_L10_g15.h5
    python src/feasibility/grid_learning_2/train.py --head-fusion concat   # ablation: no FiLM
    python src/feasibility/grid_learning_2/train.py --blur-terrain         # baseline: no terrain
    python src/feasibility/grid_learning_2/train.py --wandb-mode disabled  # no network calls
"""
from __future__ import annotations

import argparse
import math
import pathlib
import tempfile

import torch
import wandb

from feasibility.comparator.common import OUT_DIR
from feasibility.comparator.provenance import git_provenance
from feasibility.grid_learning_2.custom_dataset import GridDivergenceDataset
from feasibility.grid_learning_2.custom_dataset import make_dataloaders
from feasibility.grid_learning_2.custom_dataset import valid_targets
from feasibility.grid_learning_2.model import DEFAULT_BASE_WIDTH
from feasibility.grid_learning_2.model import DEFAULT_EMBED_DIM
from feasibility.grid_learning_2.model import GridDivergenceNet
from feasibility.grid_learning_2.model import HEAD_FUSIONS
from feasibility.grid_learning_2.model import Normalizer
from feasibility.grid_learning_2.model import TARGET_NAMES
from feasibility.grid_learning_2.model import TargetTransform
from feasibility.grid_learning_2.model import WZ_MAX

DEFAULT_DATASET = OUT_DIR / "dataset_grid2_large_box_random0_M2_L10_g15.h5"
DEFAULT_WZ_BINS = 8

# A cell counts as "commanded zero" below this. The generator writes EXACT zeros for the
# `+wz_zero_frac` anchor (design.md section 7b), so this is a float-comparison guard, not a
# tolerance band around small commands.
WZ_ZERO_TOL = 1e-6

# Pseudo-observations pulling each (command bin, lattice cell) mean toward that bin's
# lattice-pooled mean -- see per_wz_mean_fields().
WZ_MEAN_PSEUDO_COUNTS = 4.0


def fmt(values: torch.Tensor) -> str:
    """[K] per-head tensor -> "e_pos=..., e_rot=..." for the stdout lines."""
    return ", ".join(f"{n}={v:.4f}" for n, v in zip(TARGET_NAMES, values.tolist()))


# --- masked loss / masked metrics (design.md section 6) -----------------------------------------


def masked_mse_loss(
    y_hat: torch.Tensor, y_target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Masked MSE over valid (cell, head) entries only. Both y_hat/y_target must already be in
    TargetTransform space -- the loss is computed in MODEL space so the transform's inverse clamp
    never sits between the prediction and the gradient. `mask` is [*, G, G] bool, broadcast over
    the trailing head dim since a cell's validity does not depend on which head is read.
    Normalizing by mask.sum() rather than the fixed cell count keeps the loss scale independent of
    how many cells a given map happens to block, which varies from ~10% to ~60% across maps.

    Masked cells hold exact zeros in the file, never NaN (`0 * NaN` is NaN and would poison the
    backward pass straight through a correct mask), so nothing here needs a NaN-safe torch.where."""
    mask_f = mask.unsqueeze(-1).expand_as(y_hat).float()
    return (mask_f * (y_hat - y_target) ** 2).sum() / mask_f.sum().clamp_min(1.0)


def masked_rmse_per_head(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """[*, G, G, K] physical pred/target + [*, G, G] mask -> [K] RMSE per head over valid cells,
    reducing over every leading dim -- shared by the training-loop val metric, every baseline and
    the final report so they are all scored identically."""
    mask_f = mask.unsqueeze(-1).expand_as(pred).float()
    reduce_dims = tuple(range(pred.dim() - 1))
    se = mask_f * (pred - target) ** 2
    mse = se.sum(dim=reduce_dims) / mask_f.sum(dim=reduce_dims).clamp_min(1.0)
    return mse.sqrt()


# --- mirror-symmetry augmentation (design.md section 7c) ----------------------------------------


def mirror_augment(
    heightmap: torch.Tensor,
    wz: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    flip: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample, p=0.5: reflect the map in world Y and mirror the command field -- an EXACT
    symmetry of the data-generating process (the robot, the odd origin-centred heightmap grid and
    the spawn lattice are all y-symmetric), not an approximation, so it is a free 2x on data.

    The command line is where this differs from v1, and it does two things at once:

        wz -> -flip(wz, dims=[-2])

    the flip moves cell (i, j)'s command to row G-1-i so it stays attached to the terrain that
    produced its label, and the negation turns a left turn into a right one. v1 needed only the
    negation because its `wz` was a scalar. e_pos/e_rot are magnitudes, so their VALUES are
    unchanged by a reflection -- only which lattice cell they sit at, hence y/mask flip along the
    row axis (world Y) rather than being recomputed.

    `flip` overrides the coin, so the self-check below can test the transform deterministically."""
    if flip is None:
        flip = torch.rand(heightmap.shape[0], device=heightmap.device) < 0.5
    heightmap = torch.where(flip[:, None, None], torch.flip(heightmap, dims=[-2]), heightmap)
    wz = torch.where(flip[:, None, None], -torch.flip(wz, dims=[-2]), wz)
    y = torch.where(flip[:, None, None, None], torch.flip(y, dims=[-3]), y)
    mask = torch.where(flip[:, None, None], torch.flip(mask, dims=[-2]), mask)
    return heightmap, wz, y, mask


# --- non-parametric baselines (design.md section 8) ---------------------------------------------


def baseline_global_mean(
    ds: GridDivergenceDataset, train_idx: list[int], val_idx: list[int]
) -> torch.Tensor:
    """Predict one constant (e_pos, e_rot) -- the train-set valid-cell mean -- everywhere.
    design.md section 8's sanity floor: beating it proves almost nothing, but a run that cannot
    beat it is broken, not merely unimpressive."""
    mean_vec = valid_targets(ds, train_idx).mean(dim=0)
    y_val, mask_val = ds.y[val_idx], ds.mask[val_idx]
    pred = mean_vec.view(1, 1, 1, -1).expand_as(y_val)
    return masked_rmse_per_head(pred, y_val, mask_val)


def wz_bin_index(wz: torch.Tensor, n_bins: int, lo: float, hi: float) -> torch.Tensor:
    """[*] commands -> [*] bin index in [0, n_bins), uniform over [lo, hi] with the ends clamped
    in rather than spilling into a phantom bin."""
    scaled = (wz - lo) / max(hi - lo, 1e-12) * n_bins
    return scaled.floor().long().clamp_(0, n_bins - 1)


def per_wz_mean_fields(
    ds: GridDivergenceDataset, train_idx: list[int], n_bins: int, lo: float, hi: float
) -> torch.Tensor:
    """[n_bins, G, G, K] train-set mean label for each (command bin, lattice cell) pair.

    v1's version of this baseline was indexed by the ROW's single command; with a per-cell field
    the natural counterpart is indexed by the CELL's own command, which makes it a considerably
    stronger opponent -- it knows both where on the lattice a cell sits (so it has absorbed
    "edge cells diverge more than centre cells") and how hard that cell was turning. That is
    deliberate: it is the baseline that isolates terrain STRUCTURE as the only thing left for the
    network to add.

    Each entry is SHRUNK toward that bin's mean pooled over the whole lattice, with
    `WZ_MEAN_PSEUDO_COUNTS` pseudo-observations, and a bin with no train cells at all falls back to
    the global valid-cell mean. That is not a nicety: with M maps and L rows a (bin, cell) entry
    holds about `M*L/n_bins` samples -- one or two on the small files this experiment starts with --
    and an unshrunk mean of two samples is mostly noise. A baseline that scores badly because it is
    noisy flatters the model instead of challenging it, which is the opposite of what section 8
    wants from this row. As M grows the shrinkage washes out and the entry becomes the plain
    per-(bin, cell) mean."""
    wz, y, mask = ds.wz[train_idx], ds.y[train_idx], ds.mask[train_idx]
    bins = wz_bin_index(wz, n_bins, lo, hi)
    G, K = wz.shape[-1], y.shape[-1]

    fields = torch.zeros(n_bins, G, G, K, dtype=y.dtype, device=y.device)
    global_mean = y[mask].mean(dim=0) if bool(mask.any()) else torch.zeros(K, device=y.device)
    for b in range(n_bins):
        selected = (bins == b) & mask  # [n, G, G]
        weight = selected.unsqueeze(-1).float()
        counts = weight.sum(dim=0)  # [G, G, 1]
        pooled = (
            y[selected].mean(dim=0) if bool(selected.any()) else global_mean
        )  # this bin over the whole lattice
        m = WZ_MEAN_PSEUDO_COUNTS
        fields[b] = ((y * weight).sum(dim=0) + m * pooled.view(1, 1, K)) / (counts + m)
    return fields


def lookup_per_wz_mean(fields: torch.Tensor, wz: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """[n_bins, G, G, K] table + [n, G, G] commands -> [n, G, G, K] predictions, each cell reading
    the entry for its OWN command bin at its OWN lattice position."""
    n_bins, G = fields.shape[0], fields.shape[1]
    bins = wz_bin_index(wz, n_bins, lo, hi)
    rows = torch.arange(G, device=wz.device).view(1, G, 1).expand_as(bins)
    cols = torch.arange(G, device=wz.device).view(1, 1, G).expand_as(bins)
    return fields[bins, rows, cols]


def baseline_per_wz_mean(
    ds: GridDivergenceDataset,
    train_idx: list[int],
    val_idx: list[int],
    n_bins: int,
    lo: float,
    hi: float,
) -> torch.Tensor:
    """Masked val RMSE of the per-(command bin, cell) train mean -- design.md section 8's real
    bar. Beating it proves the model uses the terrain at all."""
    fields = per_wz_mean_fields(ds, train_idx, n_bins, lo, hi)
    pred = lookup_per_wz_mean(fields, ds.wz[val_idx], lo, hi)
    return masked_rmse_per_head(pred, ds.y[val_idx], ds.mask[val_idx])


def wz_range_of(ds: GridDivergenceDataset) -> tuple[float, float]:
    """The command range the file was generated over, from its own root attrs (the generator writes
    `wz_min`/`wz_max`), falling back to the architecture's WZ_MAX for a hand-written file. Binning
    against the file's declared range rather than the observed min/max keeps the baseline's bins
    identical between a train and a val split of the same file."""
    lo = float(ds.attrs.get("wz_min", -WZ_MAX))
    hi = float(ds.attrs.get("wz_max", WZ_MAX))
    return lo, hi


# --- checkpointing (mirrors grid_learning/train.py's build/load contract) ------------------------


def build_checkpoint(
    model: GridDivergenceNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    ds: GridDivergenceDataset,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Self-describing checkpoint -- everything load_checkpoint() needs is IN the file, so it loads
    without this module's source or the dataset that trained it. The TargetTransform is fitted
    data, not a learned parameter, so it is not in state_dict() and has to be saved and re-attached
    explicitly (design.md section 10).

    `spawn_xy`/`extent`/`n_cells` are stored even though forward() no longer takes them: v2's
    readout is fixed integer geometry, so they are not needed to run the model, but they ARE what
    an inference caller checks a new heightmap's grid against (`model.check_lattice_alignment`)
    before trusting the output."""
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
        wheel_radius=model.wheel_radius,
        resolution=ds.resolution,
        extent=ds.extent,
        n_cells=int(ds.heightmap.shape[-1]),
        spawn_xy=ds.spawn_xy.cpu(),
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
) -> tuple[GridDivergenceNet, dict[str, object]]:
    """Rebuilds a GridDivergenceNet from a build_checkpoint() dict, with its TargetTransform
    re-attached so `.predict()` returns physical units. There is no x_normalizer to return
    alongside: the heightmap and command inputs are never normalized (see
    custom_dataset.make_dataloaders' docstring) -- only the targets are."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    target_transform = TargetTransform(
        normalizer=Normalizer(
            mean=ckpt["target_transform_mean"].to(device),
            std=ckpt["target_transform_std"].to(device),
        )
    )
    model = GridDivergenceNet(
        base_width=ckpt["base_width"],
        embed_dim=ckpt["embed_dim"],
        head_fusion=ckpt["head_fusion"],
        wheel_radius=ckpt["wheel_radius"],
        target_transform=target_transform,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt


# --- LR schedule (design.md section 10: cosine to 0, ~5 epoch linear warmup) ---------------------


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


# --- end-of-run report (design.md section 8's metrics + the free sanity checks) ------------------


@torch.no_grad()
def collect_val_predictions(
    model: GridDivergenceNet,
    val_loader: torch.utils.data.DataLoader,
    target_transform: TargetTransform,
    device: torch.device,
    blur_terrain: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Runs the whole val set once, returning (pred, target, mask) stacked over rows, in PHYSICAL
    units -- shared by every metric in final_report() so they all see the same predictions. The
    val loader does not shuffle, so row r of these tensors is `val_idx[r]` of the dataset, which is
    what lets the wz~0 check below reuse them instead of running a second forward pass.

    `blur_terrain` has to be threaded through here (and into every other eval path): a
    --blur-terrain run is the SAME degraded input at train and val time, so scoring its checkpoint
    on undegraded terrain would report a number no training step ever optimized."""
    model.eval()
    preds, targets, masks = [], [], []
    for (heightmap, wz), y, mask in val_loader:
        heightmap, wz = heightmap.to(device), wz.to(device)
        y_hat = model(heightmap, wz, blur_terrain=blur_terrain)
        preds.append(target_transform.inverse(y_hat).cpu())
        targets.append(y.cpu())
        masks.append(mask.cpu())
    return torch.cat(preds), torch.cat(targets), torch.cat(masks)


def top_decile_rmse(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, decile: float = 0.1
) -> torch.Tensor:
    """Masked RMSE restricted to the hardest `decile` of valid cells by TRUE e_pos -- design.md
    section 8: RMSE over ALL valid cells is dominated by the near-constant flat-ground background
    and is easy to score well on for the wrong reason; the collision cells in this tail are the
    entire point of the model."""
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
    (not zero) -- design.md section 8."""
    mask_f = mask.unsqueeze(-1).expand_as(pred).float()
    reduce_dims = tuple(range(pred.dim() - 1))
    n = mask_f.sum(dim=reduce_dims).clamp_min(1.0)
    mean = (target * mask_f).sum(dim=reduce_dims) / n
    ss_res = (mask_f * (pred - target) ** 2).sum(dim=reduce_dims)
    ss_tot = (mask_f * (target - mean) ** 2).sum(dim=reduce_dims).clamp_min(1e-8)
    return 1.0 - ss_res / ss_tot


def wz_zero_check(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, wz: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int] | None:
    """Cells commanded exactly 0 command no motion, so the true field there should be ~0 and the
    prediction must be too -- design.md section 8's free sanity check, kept available by the
    generator's zero anchor (section 7b).

    Returns (predicted RMSE against zero, TRUE RMSE against zero, cell count), or None if the val
    split holds no such valid cell. The true number is reported next to the predicted one on
    purpose: "no command, no motion, no divergence" is an assumption about the physics -- the
    settle step still moves both sims a little -- so a prediction should be judged against what the
    simulator actually produced there, not against a zero the data may not reach."""
    zero_cells = mask & (wz.abs() < WZ_ZERO_TOL)
    n = int(zero_cells.sum())
    if n == 0:
        return None
    zeros = torch.zeros_like(pred)
    return (
        masked_rmse_per_head(pred, zeros, zero_cells),
        masked_rmse_per_head(target, zeros, zero_cells),
        n,
    )


@torch.no_grad()
def mirror_equivariance_check(
    model: GridDivergenceNet,
    ds: GridDivergenceDataset,
    val_idx: list[int],
    device: torch.device,
    blur_terrain: bool = False,
    n: int = 8,
) -> float:
    """`f(flip(h), -flip(wz))` should equal `flip(f(h, wz))` to numerical noise once trained -- the
    exact symmetry mirror_augment() trains on doubles as this test (design.md section 7c). Note the
    command side is flipped AND negated here, matching the augmentation; comparing against `-wz`
    alone (v1's form, correct for a scalar) would measure a different, wrong symmetry.

    Compared in MODEL space (pre-TargetTransform), so this checks the net's own equivariance rather
    than round-tripping through log1p/standardize. Not an assertion: a conv stack is not
    reflection-equivariant at initialisation, so this number is a training diagnostic. What IS
    exact and weight-independent -- the lattice's antisymmetry and the mirrored receptive fields --
    is asserted in model.py's self-check instead. Returns the max abs difference."""
    idx = val_idx[:n]
    heightmap = ds.heightmap[ds.map_index[idx]].to(device)
    wz = ds.wz[idx].to(device)
    model.eval()
    y_hat = model(heightmap, wz, blur_terrain=blur_terrain)
    y_hat_flip = model(
        torch.flip(heightmap, dims=[-2]), -torch.flip(wz, dims=[-2]), blur_terrain=blur_terrain
    )
    return (torch.flip(y_hat, dims=[-3]) - y_hat_flip).abs().max().item()


def final_report(
    checkpoint_path: pathlib.Path,
    ds: GridDivergenceDataset,
    val_loader: torch.utils.data.DataLoader,
    val_idx: list[int],
    baselines: dict[str, torch.Tensor],
    device: torch.device,
    run: wandb.sdk.wandb_run.Run,
    blur_terrain: bool = False,
) -> None:
    """Reloads the BEST checkpoint (not just the last epoch's in-memory weights) and prints the
    design.md section 8 battery: overall + top-decile masked RMSE against the same non-parametric
    baselines the run started with, R^2 per head, the wz~0 check, and mirror equivariance."""
    model, ckpt = load_checkpoint(checkpoint_path, device)
    target_transform = model.target_transform
    assert target_transform is not None
    pred, target, mask = collect_val_predictions(
        model, val_loader, target_transform, device, blur_terrain
    )

    overall = masked_rmse_per_head(pred, target, mask)
    decile = top_decile_rmse(pred, target, mask)
    r2 = r_squared(pred, target, mask)
    zero = wz_zero_check(pred, target, mask, ds.wz[val_idx].cpu())
    mirror_diff = mirror_equivariance_check(model, ds, val_idx, device, blur_terrain)

    print(f"\n[final report] checkpoint={checkpoint_path} (epoch {ckpt['epoch']})")
    print("[final report] masked val RMSE over all valid cells -- model vs the section 8 baselines:")
    for name, value in {**baselines, "model": overall}.items():
        print(f"                 {name:<20s} {fmt(value)}")
    print(f"[final report] val masked RMSE, top-decile e_pos:   {fmt(decile)}")
    print(f"[final report] val R^2 per head:                    {fmt(r2)}")
    if zero is not None:
        pred_zero, true_zero, n_zero = zero
        print(f"[final report] wz==0 cells ({n_zero}), predicted vs 0:   {fmt(pred_zero)}")
        print(f"[final report] wz==0 cells ({n_zero}), TRUE vs 0:        {fmt(true_zero)}")
    else:
        print("[final report] wz==0 cells: none in val (was the file generated with wz_zero_frac=0?)")
    print(f"[final report] mirror-equivariance max |diff| (model space): {mirror_diff:.4e}")

    run.summary.update(
        {
            **{f"final_rmse_all_{n}": v for n, v in zip(TARGET_NAMES, overall.tolist())},
            **{f"final_rmse_top_decile_{n}": v for n, v in zip(TARGET_NAMES, decile.tolist())},
            **{f"final_r2_{n}": v for n, v in zip(TARGET_NAMES, r2.tolist())},
            "final_mirror_equivariance_max_diff": mirror_diff,
            **(
                {f"final_wz_zero_rmse_{n}": v for n, v in zip(TARGET_NAMES, zero[0].tolist())}
                if zero is not None
                else {}
            ),
        }
    )


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
    )

    # Fitted on TRAIN rows' VALID cells only (design.md section 6): masked cells are exact zeros by
    # construction, and a field that is ~94% flat ground would have its standardization dragged
    # toward "everything is zero" if they were folded in. valid_targets() does exactly that
    # selection, so the fit cannot drift from the dataset's own definition of a valid cell.
    target_transform = TargetTransform.fit(valid_targets(ds, train_subset.indices))

    tag = "_blur" if args.blur_terrain else ""
    if args.head_fusion != "film":
        tag += f"_{args.head_fusion}"
    checkpoint_path = args.checkpoint or OUT_DIR / "checkpoints" / f"{ds.source.stem}{tag}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    model = GridDivergenceNet(
        base_width=args.base_width,
        embed_dim=args.embed_dim,
        head_fusion=args.head_fusion,
        target_transform=target_transform,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (
        build_scheduler(optimizer, args.warmup_epochs, args.epochs, len(train_loader))
        if args.lr_schedule == "cosine"
        else None
    )

    n_maps = ds.map_group.unique().numel()
    n_valid = int(ds.mask[train_subset.indices].sum())
    print(
        f"[data]  {ds.source.name}: {len(ds)} rows over {n_maps} maps, "
        f"{len(train_subset)} train / {len(val_subset)} val rows (held-out-MAP split), "
        f"{n_valid} valid train cells = {n_valid} independent command draws "
        f"(a v1 file of this size would give {len(train_subset)})"
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[model] base_width={args.base_width}, embed_dim={args.embed_dim}, "
        f"head_fusion={args.head_fusion}, {n_params} params, device={device}"
        + (", blur_terrain=ON (baseline)" if args.blur_terrain else "")
    )

    lo, hi = wz_range_of(ds)
    baselines = {
        "global mean": baseline_global_mean(ds, train_subset.indices, val_subset.indices).cpu(),
        "per-wz mean field": baseline_per_wz_mean(
            ds, train_subset.indices, val_subset.indices, args.wz_bins, lo, hi
        ).cpu(),
    }
    for name, value in baselines.items():
        print(f"[baseline] {name:<20s} RMSE: {fmt(value)}")
    run.summary.update(
        {
            **{f"baseline_global_mean_{n}": v
               for n, v in zip(TARGET_NAMES, baselines["global mean"].tolist())},
            **{f"baseline_per_wz_mean_{n}": v
               for n, v in zip(TARGET_NAMES, baselines["per-wz mean field"].tolist())},
        }
    )

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_since_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for (heightmap, wz), y, mask in train_loader:
            heightmap, wz = heightmap.to(device), wz.to(device)
            y, mask = y.to(device), mask.to(device)
            if not args.no_augment:
                heightmap, wz, y, mask = mirror_augment(heightmap, wz, y, mask)

            optimizer.zero_grad()
            y_hat = model(heightmap, wz, blur_terrain=args.blur_terrain)
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
                heightmap, wz = heightmap.to(device), wz.to(device)
                y, mask = y.to(device), mask.to(device)
                y_hat = model(heightmap, wz, blur_terrain=args.blur_terrain)
                val_loss_sum += (
                    masked_mse_loss(y_hat, target_transform.forward(y), mask).item()
                    * heightmap.shape[0]
                )
                val_n += heightmap.shape[0]
                preds.append(target_transform.inverse(y_hat).cpu())
                targets.append(y.cpu())
                masks.append(mask.cpu())
        val_loss = val_loss_sum / val_n
        # Scored through masked_rmse_per_head, the same function every baseline and the final report
        # use, rather than re-deriving the masked reduction inline -- the val set is already
        # resident (GridDivergenceDataset loads eagerly onto `device`), so stacking it costs nothing
        # and removes the one place these numbers could drift apart.
        val_rmse = masked_rmse_per_head(torch.cat(preds), torch.cat(targets), torch.cat(masks))

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss, best_epoch = val_loss, epoch
            epochs_since_improve = 0
            torch.save(
                build_checkpoint(model, optimizer, epoch, val_loss, ds, args), checkpoint_path
            )
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

    print(
        f"[done]  best val_loss={best_val_loss:.4f} @ epoch {best_epoch}, "
        f"checkpoint={checkpoint_path}"
    )
    run.summary["best_val_loss"] = best_val_loss
    run.summary["best_epoch"] = best_epoch

    if best_epoch == -1:
        # val_loss never improved on inf -- every epoch produced a non-finite loss (e.g. blown-up
        # gradients). Nothing was ever saved, so there is no checkpoint for final_report to load.
        print("[done]  no checkpoint saved (val_loss never improved) -- skipping final report")
        return None

    run.save(str(checkpoint_path), base_path=str(checkpoint_path.parent), policy="now")
    final_report(
        checkpoint_path, ds, val_loader, val_subset.indices, baselines, device, run,
        blur_terrain=args.blur_terrain,
    )
    return checkpoint_path


def self_test(args: argparse.Namespace) -> None:
    """This module's smoke test (there is no pytest suite in this tree): train a few epochs on
    custom_dataset's synthetic file and assert the invariants that a silent bug here would break --
    the mask really excluding cells from the loss, the augmentation moving a cell's command with
    its label AND negating it, the baselines being finite and better than nothing, and a checkpoint
    round-tripping through load_checkpoint to identical predictions.

    Runs on CPU with wandb disabled, so it needs no dataset, no GPU and no network."""
    from feasibility.grid_learning_2.custom_dataset import _write_synthetic

    G, K = 15, len(TARGET_NAMES)

    # --- masked cells must not reach the loss: perturbing them cannot move it -------------------
    y_hat = torch.randn(3, G, G, K)
    y_true = torch.randn(3, G, G, K)
    mask = torch.rand(3, G, G) > 0.4
    before = masked_mse_loss(y_hat, y_true, mask)
    poisoned = y_true.clone()
    poisoned[~mask] += 1000.0
    assert torch.allclose(before, masked_mse_loss(y_hat, poisoned, mask)), "masked cells leaked"
    print(f"[loss] masked MSE ignores the {int((~mask).sum())} masked cells exactly")

    # --- the augmentation: cell (i, j) -> (G-1-i, j) for terrain, label and mask, and the command
    # goes with it NEGATED. This is the half v1 did not need and the easiest thing to get wrong. --
    heightmap = torch.randn(2, 81, 81)
    wz = torch.rand(2, G, G) * 2 - 1
    a_hm, a_wz, a_y, a_mask = mirror_augment(
        heightmap, wz, y_hat[:2], mask[:2], flip=torch.tensor([True, False])
    )
    assert torch.equal(a_hm[0], torch.flip(heightmap[0], dims=[-2])) and torch.equal(a_hm[1], heightmap[1])
    assert torch.allclose(a_wz[0], -torch.flip(wz[0], dims=[0])) and torch.equal(a_wz[1], wz[1])
    assert torch.equal(a_y[0], torch.flip(y_hat[0], dims=[-3])) and torch.equal(a_mask[0], torch.flip(mask[0], dims=[-2]))
    assert torch.allclose(a_wz[0, G - 1 - 3, 6], -wz[0, 3, 6]), "the command did not follow its cell"
    twice = mirror_augment(a_hm, a_wz, a_y, a_mask, flip=torch.tensor([True, False]))
    assert torch.allclose(twice[1], wz) and torch.equal(twice[0], heightmap), "mirroring twice is not the identity"
    print("[augment] flip+negate moves each command with its own cell, and is an involution")

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "dataset_grid2_synthetic.h5"
        _write_synthetic(path, n_maps=6, n_rows_per_map=4)
        args.dataset = path
        args.checkpoint = pathlib.Path(tmp) / "checkpoint.pt"
        args.device = "cpu"
        args.epochs = min(args.epochs, 5)  # a smoke test, not a fit -- capped, never raised
        args.batch_size = 4
        args.warmup_epochs = 1
        args.patience = 0
        args.log_every = 1

        run = wandb.init(mode="disabled", config=vars(args))
        try:
            checkpoint_path = train(args, run)
        finally:
            run.finish()
        assert checkpoint_path is not None and checkpoint_path.exists(), "no checkpoint was saved"

        # --- the checkpoint must reproduce the trained model's predictions exactly, including the
        # TargetTransform, which is fitted data and does NOT live in state_dict() ----------------
        ds = GridDivergenceDataset(path)
        model, ckpt = load_checkpoint(checkpoint_path, torch.device("cpu"))
        heightmap = ds.heightmap[ds.map_index[:2]]
        prediction = model.predict(heightmap, ds.wz[:2])
        assert prediction.shape == (2, G, G, K) and (prediction >= 0).all()
        assert ckpt["head_fusion"] == args.head_fusion and ckpt["n_cells"] == ds.heightmap.shape[-1]
        assert math.isfinite(ckpt["val_loss"]), f"val_loss was {ckpt['val_loss']}"
        print(f"[checkpoint] reloaded epoch {ckpt['epoch']}, predict() range "
              f"[{prediction.min():.4f}, {prediction.max():.4f}], transform re-attached")

    print("all self-checks ok")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", type=pathlib.Path, default=DEFAULT_DATASET, help="dataset_grid2_*.h5 (custom_dataset.GridDivergenceDataset schema)")
    parser.add_argument("--val-frac", type=float, default=0.2, help="fraction of MAPS held out for validation (default: 0.2)")
    parser.add_argument("--seed", type=int, default=0, help="seeds the split, model init, and augmentation (default: 0)")
    parser.add_argument("--base-width", type=int, default=DEFAULT_BASE_WIDTH, help=f"trunk channel unit (default: {DEFAULT_BASE_WIDTH})")
    parser.add_argument("--embed-dim", type=int, default=DEFAULT_EMBED_DIM, help=f"command-embedding / FiLM width (default: {DEFAULT_EMBED_DIM})")
    parser.add_argument("--head-fusion", type=str, default="film", choices=HEAD_FUSIONS, help="design.md section 5d's ablation: FiLM's multiplicative gate, or the additive concat control (default: film)")
    parser.add_argument("--no-augment", action="store_true", help="disable the y-mirror + command-field flip/negate augmentation (on by default)")
    parser.add_argument("--blur-terrain", action="store_true", help='the "blur-the-terrain" baseline: flatten the relief to its own per-sample mean before every forward pass, at train AND eval time')
    parser.add_argument("--wz-bins", type=int, default=DEFAULT_WZ_BINS, help=f"bins for the per-wz mean-field baseline (default: {DEFAULT_WZ_BINS})")
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
    parser.add_argument("--self-test", action="store_true", help="train a few epochs on a synthetic file with wandb disabled, then assert this module's invariants -- no dataset or GPU needed")
    parser.add_argument("--wandb-project", type=str, default="feasibility-grid-divergence")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-name", type=str, default=None)
    args = parser.parse_args()

    if args.self_test:
        self_test(args)
        return

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
