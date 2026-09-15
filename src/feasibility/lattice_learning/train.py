"""Trains model.ArcDivergenceNet on custom_dataset.ArcDivergenceDataset -- design.md section 9's
recipe, section 8's metrics/baselines, and section 11's "held-out-MAP split, mirror + kappa-negate
augmentation, the baseline battery, `--self-test`". Structured after grid_learning_2/train.py (same
build_checkpoint/load_checkpoint contract, same wandb logging shape), restated rather than imported
per design.md section 11a. What one-label-per-row changes:

* **No mask.** A grid_learning_2 row is a 15x15 field with blocked cells; a row here is one arc with
  one `(e_pos, e_rot)`, and every row the loader kept already has a usable label (custom_dataset
  drops `valid == False`). So the loss is a plain MSE and every metric reduces over rows.

* **Blocked endpoints are a FLAG, kept by default.** `endpoint_blocked` rows (helhest_stack's static
  settle at the arc end is infeasible) stay in unless `--drop-blocked-endpoints` is given. Their
  `ref_pose` z/pitch/roll come from a settle that did not converge to a feasible stance, so their
  labels are measured against a reference the planner would never use (it marks such poses
  `blocked` and never queries `d_hat` there, design.md section 7b). When they are kept, the final
  report splits RMSE into blocked vs. non-blocked rows so you can see how much they dominate.

* **The mirror augmentation is the scalar form**: `patch -> flip(patch, rows)`, `kappa -> -kappa`
  (or `(v, wz) -> (v, -wz)` under `--command-mode v_wz`), labels unchanged (design.md section 7c).
  The patch's rows are body +Y (`PatchSpec.ys()`, asserted mirror-symmetric in model.py), so a row
  flip is exactly a reflection in the body XZ plane.

* **`--command-mode v_wz`** feeds the net the stored `(v_drive, wz_drive)` twist instead of `kappa`
  (see model.COMMAND_MODES for why that is equivalent today). The kappa-binned baselines and the
  kappa~0 flat check keep using the file's `kappa` in either mode.

* **`--label-mode pos_rpy`** regresses (e_pos, e_roll, e_pitch, e_yaw) -- one head per column --
  instead of (e_pos, e_rot). Every metric, baseline and log key below is per head, so it follows
  the label mode (`val_rmse_e_roll`, ...). The loss is still a plain mean over heads, so rotation
  carries 3/4 of it here vs. 1/2 under pos_rot.

Baselines (design.md section 8). Non-parametric/closed-form ones are computed up front, before any
training, so a run's val RMSE has something to be judged against immediately:

    global mean          one constant (e_pos, e_rot) -- a sanity floor
    per-kappa mean       binned by kappa, each bin shrunk toward the global mean -- "uses terrain at all"
    max-relief-in-sweep  max |relief| over the patch cells the three wheels sweep along the nominal
                         arc, then `y ~ a*feat + b` per head by least squares -- the obvious
                         geometric proxy the net must beat

`--blur-terrain` (relief replaced by its per-sample mean, at train AND eval time) and
`--head-fusion concat` need the architecture trained, so they are flags on this same loop -- run
twice and compare. The `_step_gate_kernel` A/B (the table's decisive row) is NOT here: it needs
helhest_stack's Warp prominence kernel run over each row's source heightmap, which is a separate
piece of work.

Reporting is in physical units on held-out MAPS: RMSE over all val rows (model + every baseline),
RMSE over the top decile by true e_pos (the headline -- a 0.3 m arc has signal only where the
envelope meets terrain, section 7c), R^2 per head, RMSE on the `swept_clear` subset (the query
distribution the planner actually produces, section 8), blocked/non-blocked split when kept, the
kappa~0-on-flat-ground check, and mirror equivariance in model space (reported, not asserted).

CLI parameters:
    --dataset PATH               dataset_arc_*.h5 (custom_dataset.ArcDivergenceDataset schema)
                                  (default: outputs/dataset_arc_lattice_maps0_M30_R16_seed0.h5)
    --drop-blocked-endpoints     drop rows whose arc-end settle is infeasible (kept by default)
    --val-frac FLOAT             fraction of MAPS held out for validation (default: 0.2)
    --seed INT                   seeds the split, model init, and augmentation (default: 0)
    --base-width INT             trunk channel unit (default: 32, model.DEFAULT_BASE_WIDTH)
    --embed-dim INT              command-embedding / FiLM width (default: 64, model.DEFAULT_EMBED_DIM)
    --head-fusion {film,concat}  design.md section 5b's ablation (default: film)
    --command-mode {kappa,v_wz}  command input: curvature, or the (v_drive, wz_drive) twist
                                  (default: kappa)
    --label-mode {pos_rot,pos_rpy}  targets: (e_pos, e_rot), or (e_pos, e_roll, e_pitch, e_yaw)
                                  (default: pos_rot)
    --no-augment                disable the y-mirror + kappa-negate augmentation (on by default)
    --blur-terrain               baseline: relief replaced by its own per-sample mean, train AND eval
    --kappa-bins INT             bins for the per-kappa mean baseline (default: 8)
    --batch-size INT             default: 256
    --epochs INT                 default: 100
    --lr FLOAT                   AdamW learning rate (default: 3e-4)
    --weight-decay FLOAT         AdamW weight decay (default: 1e-4)
    --lr-schedule {cosine,none}  cosine-to-0 with linear warmup, or a constant LR (default: cosine)
    --warmup-epochs INT          linear-warmup length for --lr-schedule cosine (default: 5)
    --patience INT               epochs of no val_loss improvement before early stop; 0 disables
                                  (default: 20)
    --device STR                 torch device (default: cuda if available else cpu)
    --checkpoint PATH            save path (default: outputs/checkpoints/<dataset stem><tags>.pt)
    --log-every INT              epochs between stdout progress lines (default: 10)
    --self-test                  train a few epochs on a synthetic file in a temp dir, wandb
                                  disabled, and assert this module's invariants -- no dataset/GPU
    --wandb-project STR          (default: "feasibility-arc-divergence")
    --wandb-entity STR           (default: None)
    --wandb-mode {online,offline,disabled}  (default: online)
    --wandb-name STR             (default: None, wandb auto-generates a name)

Usage:
    python src/feasibility/lattice_learning/train.py --self-test        # no dataset needed
    python src/feasibility/lattice_learning/train.py
    python src/feasibility/lattice_learning/train.py --drop-blocked-endpoints
    python src/feasibility/lattice_learning/train.py --head-fusion concat   # ablation: no FiLM
    python src/feasibility/lattice_learning/train.py --command-mode v_wz    # (v, wz) input
    python src/feasibility/lattice_learning/train.py --label-mode pos_rpy   # per-axis rotation targets
    python src/feasibility/lattice_learning/train.py --blur-terrain         # baseline: no terrain
    python src/feasibility/lattice_learning/train.py --wandb-mode disabled  # no network calls
"""
from __future__ import annotations

import argparse
import math
import pathlib
import tempfile

import numpy as np
import torch
import wandb
from helhest.engine import RobotParams

from feasibility.comparator.common import OUT_DIR
from feasibility.comparator.provenance import git_provenance
from feasibility.lattice_learning.arc import ARC_LEN
from feasibility.lattice_learning.arc import integrate_arc
from feasibility.lattice_learning.arc import KAPPA_MAX
from feasibility.lattice_learning.arc import MIN_TURN_RADIUS
from feasibility.lattice_learning.arc import V_NOM
from feasibility.lattice_learning.custom_dataset import ArcDivergenceDataset
from feasibility.lattice_learning.custom_dataset import make_dataloaders
from feasibility.lattice_learning.custom_dataset import valid_targets
from feasibility.lattice_learning.model import ArcDivergenceNet
from feasibility.lattice_learning.model import COMMAND_MODES
from feasibility.lattice_learning.model import DEFAULT_BASE_WIDTH
from feasibility.lattice_learning.model import DEFAULT_EMBED_DIM
from feasibility.lattice_learning.model import HEAD_FUSIONS
from feasibility.lattice_learning.model import mirror_command
from feasibility.lattice_learning.model import Normalizer
from feasibility.lattice_learning.model import LABEL_MODES
from feasibility.lattice_learning.model import TargetTransform
from feasibility.lattice_learning.patch import patch_spec_from_attrs
from feasibility.lattice_learning.patch import patch_spec_to_attrs
from feasibility.lattice_learning.patch import PatchSpec

DEFAULT_DATASET = OUT_DIR / "dataset_arc_lattice_maps0_M30_R16_seed0.h5"
DEFAULT_KAPPA_BINS = 8

# Pseudo-observations pulling each kappa bin's mean toward the global mean -- see
# per_kappa_mean_table(). With a few hundred rows a bin holds a few dozen samples at best.
KAPPA_MEAN_PSEUDO_COUNTS = 4.0

# Points sampled along the nominal arc for the swept-envelope mask (both ends included).
SWEEP_SAMPLES = 6

# The kappa~0-on-flat-ground check (design.md section 8): |kappa| below this counts as straight,
# and patch max |relief| below FLAT_RELIEF counts as flat. Relief is already divided by
# WHEEL_RADIUS, so 0.02 is ~7 mm of real height.
KAPPA_ZERO_TOL = 0.1
FLAT_RELIEF = 0.02


def fmt(values: torch.Tensor, names: tuple[str, ...]) -> str:
    """[K] per-head tensor + its K target names -> "e_pos=..., e_rot=..." for the stdout lines."""
    return ", ".join(f"{n}={v:.4f}" for n, v in zip(names, values.tolist()))


# --- loss / metrics ------------------------------------------------------------------------------


def mse_loss(y_hat: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
    """MSE over (row, head). Both must already be in TargetTransform space -- the loss is computed
    in MODEL space so the inverse's clamp never sits between prediction and gradient (section 9)."""
    return ((y_hat - y_target) ** 2).mean()


def rmse_per_head(
    pred: torch.Tensor, target: torch.Tensor, rows: torch.Tensor | None = None
) -> torch.Tensor:
    """[n, K] physical pred/target (+ optional [n] bool row selector) -> [K] RMSE per head. Shared
    by the training loop, every baseline and the final report so they are all scored identically.
    NaN if the selector is empty."""
    if rows is not None:
        pred, target = pred[rows], target[rows]
    if pred.shape[0] == 0:
        return torch.full((pred.shape[-1],), float("nan"))
    return ((pred - target) ** 2).mean(dim=0).sqrt()


def top_decile_rows(target: torch.Tensor, decile: float = 0.1) -> torch.Tensor:
    """[n] bool: the hardest `decile` of rows by TRUE e_pos -- design.md section 8's headline
    subset. RMSE over all rows is dominated by the near-zero benign background."""
    e_pos = target[:, 0]  # column 0 in every label mode (model.LABEL_NAMES)
    k = max(1, int(round(e_pos.numel() * decile)))
    threshold = e_pos.kthvalue(e_pos.numel() - k + 1).values
    return e_pos >= threshold


def r_squared(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """[K] R^2 per head: 1 - SS_res/SS_tot, SS_tot against the target mean (not zero)."""
    ss_res = ((pred - target) ** 2).sum(dim=0)
    ss_tot = ((target - target.mean(dim=0)) ** 2).sum(dim=0).clamp_min(1e-8)
    return 1.0 - ss_res / ss_tot


# --- input transforms ----------------------------------------------------------------------------


def mirror_augment(
    patch: torch.Tensor,
    command: torch.Tensor,
    y: torch.Tensor,
    flip: torch.Tensor | None = None,
    command_mode: str = "kappa",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample, p=0.5: reflect the body-frame patch across y=0 and negate the turn --
    an EXACT symmetry of the robot and the arc (design.md section 7c), so a free 2x on data.

        patch   -> flip(patch, dims=[-2])    rows are body +Y
        command -> mirror_command(command)   kappa -> -kappa, (v, wz) -> (v, -wz)
        y       -> y                         every label is a magnitude (model.LABEL_NAMES)

    `flip` overrides the coin, so the self-check can test the transform deterministically."""
    if flip is None:
        flip = torch.rand(patch.shape[0], device=patch.device) < 0.5
    patch = torch.where(flip[:, None, None, None], torch.flip(patch, dims=[-2]), patch)
    command = torch.where(flip[:, None], mirror_command(command, command_mode), command)
    return patch, command, y


def blur(patch: torch.Tensor) -> torch.Tensor:
    """The blur-the-terrain baseline input: every cell replaced by its sample's mean relief, so the
    net keeps the patch's overall level but loses all structure (design.md section 8)."""
    return patch.mean(dim=(-2, -1), keepdim=True).expand_as(patch)


def prepare_patch(patch: torch.Tensor, blur_terrain: bool) -> torch.Tensor:
    return blur(patch) if blur_terrain else patch


# --- baselines (design.md section 8) -------------------------------------------------------------


def kappa_range_of(ds: ArcDivergenceDataset) -> tuple[float, float]:
    """The kappa range the file was generated over, from its root attrs (`kappa_min`/`kappa_max`),
    falling back to +-KAPPA_MAX. Binning against the declared range rather than the observed
    min/max keeps the bins identical between train and val."""
    return float(ds.attrs.get("kappa_min", -KAPPA_MAX)), float(ds.attrs.get("kappa_max", KAPPA_MAX))


def kappa_bin_index(kappa: torch.Tensor, n_bins: int, lo: float, hi: float) -> torch.Tensor:
    scaled = (kappa - lo) / max(hi - lo, 1e-12) * n_bins
    return scaled.floor().long().clamp_(0, n_bins - 1)


def per_kappa_mean_table(
    kappa: torch.Tensor, y: torch.Tensor, n_bins: int, lo: float, hi: float
) -> torch.Tensor:
    """[n_bins, K] train mean label per kappa bin, each SHRUNK toward the global mean with
    `KAPPA_MEAN_PSEUDO_COUNTS` pseudo-observations so a sparse bin is not mostly noise -- a noisy
    baseline flatters the model instead of challenging it."""
    bins = kappa_bin_index(kappa, n_bins, lo, hi)
    global_mean = y.mean(dim=0)
    table = torch.empty(n_bins, y.shape[-1], dtype=y.dtype, device=y.device)
    m = KAPPA_MEAN_PSEUDO_COUNTS
    for b in range(n_bins):
        sel = bins == b
        table[b] = (y[sel].sum(dim=0) + m * global_mean) / (sel.sum() + m)
    return table


def swept_envelope_mask(
    kappa: np.ndarray, spec: PatchSpec, robot: RobotParams | None = None
) -> np.ndarray:
    """[n] curvatures -> [n, ny, nx] bool: patch cells within `wheel_radius` of any of the three
    wheel contacts -- front `(0, +-half_track)`, rear `(-rear_offset, 0)` in the body frame -- at
    `SWEEP_SAMPLES` points along the nominal arc from the body origin. The "swept envelope" of
    design.md section 8's max-relief baseline, in the same body frame the patch is sampled in."""
    robot = robot or RobotParams()
    r, ht, ro = float(robot.wheel_radius), float(robot.half_track), float(robot.rear_offset)
    wheels = np.array([[0.0, ht], [0.0, -ht], [-ro, 0.0]])  # [3, 2] body frame
    kappa = np.asarray(kappa, dtype=np.float64)
    s = np.linspace(0.0, ARC_LEN, SWEEP_SAMPLES)  # [S]
    poses = integrate_arc(np.zeros((1, 1, 3)), kappa[:, None], s[None, :])  # [n, S, 3]
    c, sn = np.cos(poses[..., 2]), np.sin(poses[..., 2])
    wx = poses[..., 0, None] + c[..., None] * wheels[:, 0] - sn[..., None] * wheels[:, 1]
    wy = poses[..., 1, None] + sn[..., None] * wheels[:, 0] + c[..., None] * wheels[:, 1]
    wx, wy = wx.reshape(len(kappa), -1), wy.reshape(len(kappa), -1)  # [n, S*3]
    gx, gy = np.meshgrid(spec.xs(), spec.ys())  # [ny, nx], rows = body Y
    d2 = (gx[None, None] - wx[:, :, None, None]) ** 2 + (gy[None, None] - wy[:, :, None, None]) ** 2
    return (d2 <= r * r).any(axis=1)


def max_relief_in_sweep(ds: ArcDivergenceDataset) -> torch.Tensor:
    """[len(ds)] max |relief| over each row's swept envelope -- the geometric-proxy feature."""
    mask = swept_envelope_mask(ds.kappa.cpu().numpy(), ds.patch_spec)
    relief = ds.patch[:, 0].abs().cpu().numpy()
    return torch.from_numpy(np.where(mask, relief, 0.0).max(axis=(1, 2))).float()


def compute_baselines(
    ds: ArcDivergenceDataset, train_idx: list[int], val_idx: list[int], n_bins: int
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Returns (val RMSE per baseline, val PREDICTIONS per baseline), all on CPU in physical units,
    fitted on train rows only. Predictions are returned too so the final report can score every
    baseline on the same subsets (top decile, swept_clear, ...) as the model."""
    y = ds.y.cpu()
    kappa = ds.kappa.cpu()
    tr, va = torch.as_tensor(train_idx), torch.as_tensor(val_idx)
    preds: dict[str, torch.Tensor] = {}

    preds["global mean"] = y[tr].mean(dim=0).expand(len(va), -1)

    lo, hi = kappa_range_of(ds)
    table = per_kappa_mean_table(kappa[tr], y[tr], n_bins, lo, hi)
    preds["per-kappa mean"] = table[kappa_bin_index(kappa[va], n_bins, lo, hi)]

    feat = max_relief_in_sweep(ds)
    design = torch.stack([feat[tr], torch.ones(len(tr))], dim=-1).double()  # [n_train, 2]
    coef = torch.linalg.lstsq(design, y[tr].double()).solution  # [2, K]
    fitted = torch.stack([feat[va], torch.ones(len(va))], dim=-1).double() @ coef
    preds["max-relief-in-sweep"] = fitted.float().clamp_min(0.0)

    rmse = {name: rmse_per_head(p, y[va]) for name, p in preds.items()}
    return rmse, preds


# --- checkpointing -------------------------------------------------------------------------------


def build_checkpoint(
    model: ArcDivergenceNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    ds: ArcDivergenceDataset,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Self-describing checkpoint: everything load_checkpoint() needs is IN the file. Carries the
    fitted TargetTransform (fitted data, not in state_dict()), the full PatchSpec (a differently
    shaped/referenced patch at inference is silently wrong, section 9), and the three pinned
    constants `infer.py` must assert against before trusting the net (design.md section 4c) --
    read from the DATASET's attrs, i.e. what the labels were actually generated at."""
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
        command_mode=model.command_mode,
        blur_terrain=args.blur_terrain,
        drop_blocked_endpoints=args.drop_blocked_endpoints,
        patch_spec=patch_spec_to_attrs(ds.patch_spec),
        v_nom=ds.v_nom,
        arc_len=float(ds.attrs.get("arc_len", ARC_LEN)),
        min_turn_radius=float(ds.attrs.get("min_turn_radius", MIN_TURN_RADIUS)),
        label_mode=model.label_mode,
        target_names=model.target_names,
        target_transform_mean=target_transform.normalizer.mean.cpu(),
        target_transform_std=target_transform.normalizer.std.cpu(),
        dataset_path=str(ds.source),
        dataset_git=ds.git,
        train_git=git_provenance(),
        args=vars(args),
    )


def load_checkpoint(
    path: pathlib.Path, device: torch.device
) -> tuple[ArcDivergenceNet, dict[str, object]]:
    """Rebuilds an ArcDivergenceNet from a build_checkpoint() dict with its TargetTransform
    re-attached, so `.predict()` returns physical (e_pos m, e_rot rad)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    target_transform = TargetTransform(
        normalizer=Normalizer(
            mean=ckpt["target_transform_mean"].to(device),
            std=ckpt["target_transform_std"].to(device),
        )
    )
    model = ArcDivergenceNet(
        base_width=ckpt["base_width"],
        embed_dim=ckpt["embed_dim"],
        head_fusion=ckpt["head_fusion"],
        command_mode=ckpt.get("command_mode", "kappa"),  # checkpoints predating the option
        label_mode=ckpt.get("label_mode", "pos_rot"),
        patch_spec=patch_spec_from_attrs(ckpt["patch_spec"]),
        target_transform=target_transform,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt


# --- LR schedule (design.md section 9: cosine to 0, ~5 epoch linear warmup) ----------------------


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


# --- evaluation ----------------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: ArcDivergenceNet,
    loader: torch.utils.data.DataLoader,
    target_transform: TargetTransform,
    device: torch.device,
    blur_terrain: bool,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """One pass over `loader` -> (model-space loss, physical pred [n, K], target [n, K]) on CPU.
    The val loader does not shuffle, so row r is `val_subset.indices[r]`. `blur_terrain` must be
    threaded through every eval path: a --blur-terrain run scored on unblurred terrain would
    report a number no training step optimized."""
    model.eval()
    loss_sum, n = 0.0, 0
    preds, targets = [], []
    for (patch, command), y in loader:
        patch, command, y = patch.to(device), command.to(device), y.to(device)
        y_hat = model(prepare_patch(patch, blur_terrain), command)
        loss_sum += mse_loss(y_hat, target_transform.forward(y)).item() * patch.shape[0]
        n += patch.shape[0]
        preds.append(target_transform.inverse(y_hat).cpu())
        targets.append(y.cpu())
    return loss_sum / max(n, 1), torch.cat(preds), torch.cat(targets)


@torch.no_grad()
def mirror_equivariance_check(
    model: ArcDivergenceNet,
    ds: ArcDivergenceDataset,
    val_idx: list[int],
    device: torch.device,
    blur_terrain: bool,
    n: int = 64,
) -> float:
    """max |f(patch, cmd) - f(flip(patch), mirror(cmd))| in MODEL space over a few val rows. Not
    an assertion: a conv stack is not reflection-equivariant at init, so this is a training
    diagnostic of how well the augmentation took (design.md section 8)."""
    idx = torch.as_tensor(val_idx[:n], device=ds.patch.device)
    patch = prepare_patch(ds.patch[idx].to(device), blur_terrain)
    command = ds.command[idx].to(device)
    model.eval()
    a = model(patch, command)
    b = model(torch.flip(patch, dims=[-2]), mirror_command(command, model.command_mode))
    return (a - b).abs().max().item()


def final_report(
    checkpoint_path: pathlib.Path,
    ds: ArcDivergenceDataset,
    val_loader: torch.utils.data.DataLoader,
    val_idx: list[int],
    baseline_preds: dict[str, torch.Tensor],
    device: torch.device,
    run: wandb.sdk.wandb_run.Run,
    blur_terrain: bool,
) -> dict[str, torch.Tensor]:
    """Reloads the BEST checkpoint and prints design.md section 8's battery. Every row of the
    comparison table (model and baselines) is scored on the same subsets. Returns the model's
    per-subset RMSE dict (used by the self-test)."""
    model, ckpt = load_checkpoint(checkpoint_path, device)
    names = model.target_names
    assert model.target_transform is not None
    _, pred, target = evaluate(model, val_loader, model.target_transform, device, blur_terrain)

    vi = torch.as_tensor(val_idx)
    subsets: dict[str, torch.Tensor | None] = {
        "all": None,
        "top decile e_pos": top_decile_rows(target),
        "swept_clear": ds.swept_clear.cpu()[vi],
    }
    blocked = ds.endpoint_blocked.cpu()[vi]
    if bool(blocked.any()):
        subsets["endpoint OK"] = ~blocked
        subsets["endpoint BLOCKED"] = blocked

    table = {"model": pred, **baseline_preds}
    print(f"\n[final report] checkpoint={checkpoint_path} (epoch {ckpt['epoch']}), "
          f"{len(val_idx)} val rows on held-out maps"
          + (", blur_terrain=ON" if blur_terrain else ""))
    model_rmse: dict[str, torch.Tensor] = {}
    summary: dict[str, float] = {}
    for subset_name, rows in subsets.items():
        n_rows = len(val_idx) if rows is None else int(rows.sum())
        print(f"[final report] val RMSE, {subset_name} ({n_rows} rows):")
        for name, p in table.items():
            value = rmse_per_head(p, target, rows)
            print(f"                 {name:<22s} {fmt(value, names)}")
            key = f"final_rmse_{subset_name}_{name}".replace(" ", "_")
            summary.update({f"{key}_{n}": v for n, v in zip(names, value.tolist())})
            if name == "model":
                model_rmse[subset_name] = value

    r2 = r_squared(pred, target)
    print(f"[final report] val R^2 per head (model):      {fmt(r2, names)}")
    summary.update({f"final_r2_{n}": v for n, v in zip(names, r2.tolist())})

    flat = (
        (ds.kappa.cpu()[vi].abs() < KAPPA_ZERO_TOL)
        & (ds.patch.cpu()[vi].abs().amax(dim=(1, 2, 3)) < FLAT_RELIEF)
    )
    if bool(flat.any()):
        zeros = torch.zeros_like(pred)
        pred_zero, true_zero = rmse_per_head(pred, zeros, flat), rmse_per_head(target, zeros, flat)
        print(f"[final report] kappa~0 on flat ({int(flat.sum())} rows), predicted vs 0: {fmt(pred_zero, names)}")
        print(f"[final report] kappa~0 on flat ({int(flat.sum())} rows), TRUE vs 0:      {fmt(true_zero, names)}")
        summary.update({f"final_flat_zero_pred_{n}": v for n, v in zip(names, pred_zero.tolist())})
    else:
        print("[final report] kappa~0 on flat: no such val rows")

    mirror_diff = mirror_equivariance_check(model, ds, val_idx, device, blur_terrain)
    print(f"[final report] mirror-equivariance max |diff| (model space): {mirror_diff:.4e}")
    summary["final_mirror_equivariance_max_diff"] = mirror_diff
    run.summary.update(summary)
    return model_rmse


# --- training ------------------------------------------------------------------------------------


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
        drop_blocked_endpoints=args.drop_blocked_endpoints,
        command_mode=args.command_mode,
        label_mode=args.label_mode,
    )

    # Fitted on TRAIN rows only -- fitting on all rows would leak val statistics (section 9).
    target_transform = TargetTransform.fit(valid_targets(ds, train_subset.indices))
    names = ds.TARGET_NAMES

    tag = "_blur" if args.blur_terrain else ""
    if args.head_fusion != "film":
        tag += f"_{args.head_fusion}"
    if args.command_mode != "kappa":
        tag += f"_{args.command_mode.replace('_', '')}"
    if args.label_mode != "pos_rot":
        tag += "_rpy"
    if args.drop_blocked_endpoints:
        tag += "_noblocked"
    checkpoint_path = args.checkpoint or OUT_DIR / "checkpoints" / f"{ds.source.stem}{tag}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    model = ArcDivergenceNet(
        base_width=args.base_width,
        embed_dim=args.embed_dim,
        head_fusion=args.head_fusion,
        command_mode=ds.command_mode,
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
    n_blocked = int(ds.endpoint_blocked.sum())
    print(
        f"[data]  {ds.source.name}: {len(ds)} rows over {n_maps} maps, "
        f"{len(train_subset)} train / {len(val_subset)} val rows (held-out-MAP split), "
        + (
            "endpoint-blocked rows DROPPED"
            if args.drop_blocked_endpoints
            else f"{n_blocked} endpoint-blocked rows KEPT (--drop-blocked-endpoints to drop)"
        )
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[model] base_width={args.base_width}, embed_dim={args.embed_dim}, "
        f"head_fusion={args.head_fusion}, command_mode={args.command_mode}, "
        f"label_mode={args.label_mode} ({', '.join(names)}), {n_params} params, "
        f"device={device}"
        + (", blur_terrain=ON (baseline)" if args.blur_terrain else "")
    )

    baseline_rmse, baseline_preds = compute_baselines(
        ds, train_subset.indices, val_subset.indices, args.kappa_bins
    )
    for name, value in baseline_rmse.items():
        print(f"[baseline] {name:<22s} RMSE: {fmt(value, names)}")
        run.summary.update(
            {f"baseline_{name.replace(' ', '_')}_{n}": v for n, v in zip(names, value.tolist())}
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
                patch, command, y = mirror_augment(patch, command, y, command_mode=ds.command_mode)

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

        val_loss, val_pred, val_target = evaluate(
            model, val_loader, target_transform, device, args.blur_terrain
        )
        val_rmse = rmse_per_head(val_pred, val_target)

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
                val_minus_train_loss=val_loss - train_loss,
                lr=optimizer.param_groups[0]["lr"],
                **{f"val_rmse_{n}": v for n, v in zip(names, val_rmse.tolist())},
            ),
            step=epoch,
        )

        if epoch % args.log_every == 0 or epoch == args.epochs or improved:
            marker = " *" if improved else ""
            print(
                f"[{epoch:4d}/{args.epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_rmse=({fmt(val_rmse, names)}){marker}"
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
        # val_loss never improved on inf -- every epoch was non-finite. Nothing to report.
        print("[done]  no checkpoint saved (val_loss never improved) -- skipping final report")
        return None

    run.save(str(checkpoint_path), base_path=str(checkpoint_path.parent), policy="now")
    final_report(
        checkpoint_path, ds, val_loader, val_subset.indices, baseline_preds, device, run,
        blur_terrain=args.blur_terrain,
    )
    return checkpoint_path


def self_test(args: argparse.Namespace) -> None:
    """This module's smoke test (no pytest in this tree): the augmentation's exact form, the
    swept envelope's mirror symmetry, a short train on custom_dataset's synthetic file with and
    without --drop-blocked-endpoints, and a checkpoint round-trip. CPU, wandb disabled."""
    from feasibility.lattice_learning.custom_dataset import _write_synthetic

    spec = PatchSpec()
    K = 2  # augmentation checks only; labels pass through untouched whatever their width

    # --- augmentation: rows flip, kappa negates, labels untouched, and it is an involution ------
    patch = torch.randn(2, 1, spec.ny, spec.nx)
    kappa = torch.tensor([[0.7], [-1.3]])
    y = torch.rand(2, K)
    flip = torch.tensor([True, False])
    a_patch, a_kappa, a_y = mirror_augment(patch, kappa, y, flip=flip)
    assert torch.equal(a_patch[0], torch.flip(patch[0], dims=[-2])) and torch.equal(a_patch[1], patch[1])
    assert torch.equal(a_kappa, torch.tensor([[-0.7], [-1.3]])) and torch.equal(a_y, y)
    twice = mirror_augment(a_patch, a_kappa, a_y, flip=flip)
    assert torch.equal(twice[0], patch) and torch.equal(twice[1], kappa), "mirroring twice is not the identity"
    v_wz = torch.tensor([[0.6, 0.42], [0.6, -0.78]])
    _, a_vwz, _ = mirror_augment(patch, v_wz, y, flip=flip, command_mode="v_wz")
    assert torch.equal(a_vwz, torch.tensor([[0.6, -0.42], [0.6, -0.78]])), a_vwz
    print("[augment] row flip + kappa negate (wz negate, v kept for v_wz), labels unchanged, "
          "involution ok")

    # --- blur keeps the per-sample mean and removes all structure ------------------------------
    blurred = blur(patch)
    assert torch.allclose(blurred.mean(dim=(-2, -1)), patch.mean(dim=(-2, -1)), atol=1e-6)
    assert torch.allclose(blurred.std(dim=(-2, -1)), torch.zeros(2, 1), atol=1e-6)
    print("[blur] per-sample mean preserved, structure removed")

    # --- swept envelope: non-empty, covers the wheel contacts at s=0, mirrors under kappa -> -kappa
    kappas = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    env = swept_envelope_mask(kappas, spec)
    assert env.shape == (5, spec.ny, spec.nx) and env.any(axis=(1, 2)).all()
    assert np.array_equal(env[0], env[4][::-1]) and np.array_equal(env[1], env[3][::-1]), (
        "swept envelope is not mirror-symmetric under kappa -> -kappa"
    )
    assert np.array_equal(env[2], env[2][::-1]), "straight-ahead envelope is not y-symmetric"
    print(f"[sweep] envelope covers {env.sum(axis=(1, 2)).tolist()} cells for kappa={kappas.tolist()}, "
          "mirror-symmetric")

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "dataset_arc_synthetic.h5"
        _write_synthetic(path, n_maps=6, rows_per_map=16)
        args.dataset = path
        args.device = "cpu"
        args.epochs = min(args.epochs, 3)  # a smoke test, not a fit -- capped, never raised
        args.batch_size = 16
        args.warmup_epochs = 1
        args.patience = 0
        args.log_every = 1

        n_rows: dict[bool, int] = {}
        # drop, command_mode and label_mode are independent, so the second run covers all three
        # non-defaults without a third train
        for drop, command_mode, label_mode in ((False, "kappa", "pos_rot"), (True, "v_wz", "pos_rpy")):
            args.drop_blocked_endpoints = drop
            args.command_mode = command_mode
            args.label_mode = label_mode
            args.checkpoint = pathlib.Path(tmp) / f"checkpoint_drop{int(drop)}_{command_mode}.pt"
            print(f"\n[self-test] training with drop_blocked_endpoints={drop}, "
                  f"command_mode={command_mode}, label_mode={label_mode}")
            run = wandb.init(mode="disabled", config=vars(args))
            try:
                checkpoint_path = train(args, run)
            finally:
                run.finish()
            assert checkpoint_path is not None and checkpoint_path.exists(), "no checkpoint saved"

            ds = ArcDivergenceDataset(
                path, drop_blocked_endpoints=drop, command_mode=command_mode, label_mode=label_mode
            )
            n_rows[drop] = len(ds)
            if drop:
                assert not bool(ds.endpoint_blocked.any())

            # --- checkpoint must round-trip, TargetTransform and pinned constants included -----
            model, ckpt = load_checkpoint(checkpoint_path, torch.device("cpu"))
            assert model.command_mode == command_mode == ckpt["command_mode"]
            assert model.label_mode == label_mode == ckpt["label_mode"]
            assert tuple(ckpt["target_names"]) == model.target_names == ds.TARGET_NAMES
            prediction = model.predict(ds.patch[:4], ds.command[:4])
            assert prediction.shape == (4, len(ds.TARGET_NAMES)) and torch.isfinite(prediction).all()
            assert (prediction >= 0).all()
            assert ckpt["drop_blocked_endpoints"] == drop
            assert math.isclose(ckpt["arc_len"], ARC_LEN) and math.isclose(ckpt["v_nom"], V_NOM)
            assert math.isfinite(ckpt["val_loss"]), f"val_loss was {ckpt['val_loss']}"
            print(f"[checkpoint] reloaded epoch {ckpt['epoch']}, predict() range "
                  f"[{prediction.min():.4f}, {prediction.max():.4f}]")

        assert n_rows[True] < n_rows[False], n_rows
        print(f"\n[drop-blocked] {n_rows[False]} rows kept by default, {n_rows[True]} with the flag")

    print("all self-checks ok")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", type=pathlib.Path, default=DEFAULT_DATASET, help="dataset_arc_*.h5 (custom_dataset.ArcDivergenceDataset schema)")
    parser.add_argument("--drop-blocked-endpoints", action="store_true", help="drop rows whose arc-end settle is infeasible (kept by default)")
    parser.add_argument("--val-frac", type=float, default=0.2, help="fraction of MAPS held out for validation (default: 0.2)")
    parser.add_argument("--seed", type=int, default=0, help="seeds the split, model init, and augmentation (default: 0)")
    parser.add_argument("--base-width", type=int, default=DEFAULT_BASE_WIDTH, help=f"trunk channel unit (default: {DEFAULT_BASE_WIDTH})")
    parser.add_argument("--embed-dim", type=int, default=DEFAULT_EMBED_DIM, help=f"command-embedding / FiLM width (default: {DEFAULT_EMBED_DIM})")
    parser.add_argument("--head-fusion", type=str, default="film", choices=HEAD_FUSIONS, help="design.md section 5b's ablation (default: film)")
    parser.add_argument("--command-mode", type=str, default="kappa", choices=COMMAND_MODES, help="command input: kappa, or the (v_drive, wz_drive) twist (default: kappa)")
    parser.add_argument("--label-mode", type=str, default="pos_rot", choices=LABEL_MODES, help="regression targets: (e_pos, e_rot), or (e_pos, e_roll, e_pitch, e_yaw) (default: pos_rot)")
    parser.add_argument("--no-augment", action="store_true", help="disable the y-mirror + kappa-negate augmentation (on by default)")
    parser.add_argument("--blur-terrain", action="store_true", help="baseline: relief replaced by its per-sample mean, at train AND eval time")
    parser.add_argument("--kappa-bins", type=int, default=DEFAULT_KAPPA_BINS, help=f"bins for the per-kappa mean baseline (default: {DEFAULT_KAPPA_BINS})")
    parser.add_argument("--batch-size", type=int, default=256, help="default: 256")
    parser.add_argument("--epochs", type=int, default=100, help="default: 100")
    parser.add_argument("--lr", type=float, default=3e-4, help="AdamW learning rate (default: 3e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay (default: 1e-4)")
    parser.add_argument("--lr-schedule", choices=("cosine", "none"), default="cosine", help="cosine-to-0 with linear warmup, or a constant LR (default: cosine)")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="linear-warmup length for --lr-schedule cosine (default: 5)")
    parser.add_argument("--patience", type=int, default=20, help="epochs of no val_loss improvement before early stop; 0 disables (default: 20)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="torch device (default: cuda if available else cpu)")
    parser.add_argument("--checkpoint", type=pathlib.Path, default=None, help="save path (default: outputs/checkpoints/<dataset stem><tags>.pt)")
    parser.add_argument("--log-every", type=int, default=10, help="epochs between stdout progress lines (default: 10)")
    parser.add_argument("--self-test", action="store_true", help="train a few epochs on a synthetic file with wandb disabled, then assert this module's invariants")
    parser.add_argument("--wandb-project", type=str, default="feasibility-arc-divergence")
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
