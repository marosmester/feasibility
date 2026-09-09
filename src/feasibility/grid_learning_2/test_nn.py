"""Evaluates a trained `model.GridDivergenceNet` checkpoint against FRESH ostrich+helhest_stack
simulation trials on a set of heightmaps -- e.g. a whole batch folder such as
`assets/large_box_random/0/` -- rather than against rows already baked into one
`dataset_grid2_*.h5` file (which is all `train.py`'s own val split and `gl_replay_grid.py` can
show).

Runs the actual generation pipeline (`generate_dataset.py`'s own `simulate_map`, imported directly
rather than restated -- it is ~100 lines of Hydra/chunking/ostrich+hstack orchestration, not a
small formula worth duplicating per-script the way this package restates the SE(3)-error/
footprint-clear math elsewhere), computes ground-truth (e_pos, e_rot) from the real simulator poses
via custom_dataset.py's vectorized formula, runs the checkpoint's own inference on the same maps
and command fields, and reports how close the two are -- overall, split into flat-ground vs.
near-obstacle cells, and binned by |wz|.

**What differs from grid_learning's test_nn.py, and why this is a separate script.** Four things,
all downstream of design.md's two changes:

  * **Commands are FIELDS.** v1 evaluates `n_commands` scalar yaw rates per map
    (`linspace(*WZ_RANGE, n_commands)`), each tiled over all 225 cells. Here every cell draws its
    own, via `generate_dataset.sample_command_fields` -- the same function that wrote the training
    file, so the eval distribution is the training distribution by construction rather than by a
    restated `linspace`. The knob is `+rows_per_map` (command FIELDS per map), not `+n_commands`:
    a v2 row holds G*G commands, so the old name would be off by a factor of 225.

  * **The checkpoint is authoritative about geometry.** v1 must re-open its training .h5 to recover
    the `resolution` its checkpoint does not store, and falls back to a default (with a warning) if
    that file has moved. v2's `build_checkpoint` stores `resolution`, `n_cells`, `extent` and
    `spawn_xy`, so the CNN grid is fully determined by the checkpoint and the training file is
    needed for ONE thing only: the set of maps to exclude from the eval draw. A missing file
    therefore costs the exclusion, nothing else.

  * **The lattice check is an equality, not a comparison.** v1 compares its `spawn_lattice()`
    against the checkpoint's `spawn_xy` and bails on mismatch. This does that too, and then runs
    `model.check_lattice_alignment`, which is the stronger statement: the network's integer
    centre-crop must LAND on those coordinates (design.md section 3c). Same call
    `generate_dataset.py` makes before spending any GPU time, for the same reason -- a mis-sized
    grid otherwise produces numbers that look fine and are silently misregistered.

  * **`predict()` takes no `spawn_xy`/`extent`.** The readout is fixed integer geometry, so
    inference is `model.predict(heightmap, wz)` over the row's whole lattice at once.

**RMSE per |wz| bin** is the one diagnostic here that v1 structurally cannot produce. With one
command per row, filling 8 bins needs 8 simulated rows; with a per-cell field, a SINGLE map's 225
cells populate every bin. That axis -- how well the model tracks the command, as opposed to the
terrain -- is what the whole v2 reframing exists to improve (design.md section 1b), so this is the
number that says whether it did.

**Flat vs near-obstacle:** `near_obstacle_mask` flags a lattice cell if terrain ANYWHERE within
`near_obstacle_radius` m of its center clears `generate_dataset.obstacle_height_threshold` --
deliberately independent of `footprint_clear` (which only checks the three wheel contacts at rest):
a cell can be footprint-clear yet still sweep its rear wheel into the box during the commanded turn
in place, which is exactly the divergence this dataset is built to catch. The default radius
(1.5 m) is the same rear-wheel rim-reach padding utils.SPAWN_LIMIT's own docstring derives (1.101 m
axle offset + 0.35 m wheel radius, rounded up).

**Held-out maps:** there is no repo convention of a literal "validation/" directory -- a folder like
`assets/large_box_random/0/` is just another indexed batch. By default this script reads the
checkpoint's own self-described `dataset_path` (see train.build_checkpoint) and excludes any map
that file was trained on from the candidates drawn out of `+maps_dir`, so an eval run against the
SAME folder used for training still measures generalization rather than memorization.
`+exclude_training_maps=false` disables this.

Deliberately independent of feasibility.grid_learning and feasibility.learning (design.md section
11): `near_obstacle_mask` is reimplemented here rather than imported, and the CSV log is this
package's own `eval_log.py` writing its own schema (see that module's docstring for the three
columns that differ). feasibility.comparator and feasibility.heightmap are shared infrastructure
and ARE imported, as are this package's own siblings.

CLI parameters (Hydra overrides, `+` prefix required -- same convention as generate_dataset.py,
needed here too since running fresh trials means building the same sim_config/render_config/
engine_config/logging_config via hydra.utils.instantiate):
    +checkpoint=STR              trained GridDivergenceNet .pt path (required)
    +n_maps=INT                  maps drawn (without replacement) for evaluation (default: 5)
    +rows_per_map=INT            command FIELDS per map (default: 5). v1 calls this +n_commands;
                                  here a row is a field of G*G commands, not one.
    +maps_dir=STR                repo-root-relative or absolute map directory
                                  (default: assets/large_box_random/0)
    +seed=INT                    RNG seed for map selection AND the command fields (default: 0)
    +wz_zero_frac=FLOAT          fraction of cells forced to exactly wz = 0 (default: 0.05, the
                                  generator's own default -- keep it to match the training draw)
    +wz_grid=INT                 draw from K discrete levels instead of U(range) (default: unset)
    +wz_bins=INT                 bins for the per-|wz| RMSE breakdown (default: 8)
    +exclude_training_maps=BOOL  drop maps the checkpoint was trained on, if that dataset file is
                                  still reachable (default: true)
    +near_obstacle_radius=FLOAT  m, neighborhood radius for the flat/near-obstacle split (def: 1.5)
    +duration_s=FLOAT            command hold time (default: 2.4, must match training's)
    +chunk=INT                   worlds per ostrich model build (default: 128)
    +settle_steps=INT            ostrich settle steps, paid once per chunk (default: 12)
    +mu=FLOAT                    ground friction (default: 0.8)
    +k_turn=FLOAT                hstack ICR turning-rate gain (default: dynamics.K_TURN)
    +device=STR                  torch/warp device for the network, hstack, AND ostrich (via
                                  init_warp_device) (default: "cuda:0")
    +save_fig=STR                save the comparison figure here instead of showing it interactively
    +dry_run=BOOL                lattice/checkpoint/command-field self-checks only -- no simulation
                                  (default: false)
    +log_path=STR                CSV evaluation log to append this run's stats to, one row per run
                                  (default: outputs/eval_log2.csv, see eval_log.py -- deliberately
                                  NOT v1's eval_log.csv, the schemas differ)
    +no_log=BOOL                 skip the CSV log append entirely (default: false)
    Also accepts any standard Hydra config-group override against the "helhest" base config
    (engine=mujoco, simulation=..., logging=...); rendering is forced headless.

Usage:
    python src/feasibility/grid_learning_2/test_nn.py +checkpoint=outputs/checkpoints/dataset_grid2_large_box_random0_M100_L10_g15.pt +dry_run=true
    python src/feasibility/grid_learning_2/test_nn.py +checkpoint=outputs/checkpoints/dataset_grid2_large_box_random0_M100_L10_g15.pt +n_maps=1 +rows_per_map=1
    python src/feasibility/grid_learning_2/test_nn.py +checkpoint=outputs/checkpoints/dataset_grid2_large_box_random0_M100_L10_g15.pt +maps_dir=assets/large_box_random/1 +n_maps=3
    python src/feasibility/grid_learning_2/test_nn.py +checkpoint=outputs/checkpoints/dataset_grid2_large_box_random0_M100_L10_g15.pt +save_fig=outputs/eval_grid2_M100.png
"""
from __future__ import annotations

import pathlib

import h5py
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from helhest import dynamics
from omegaconf import DictConfig
from ostrich import EngineConfig
from ostrich import LoggingConfig
from ostrich import RenderingConfig
from ostrich import SimulationConfig

from feasibility.comparator.common import CONFIG_PATH
from feasibility.comparator.common import init_warp_device
from feasibility.grid_learning_2 import eval_log
from feasibility.grid_learning_2.custom_dataset import poses_to_se3
from feasibility.grid_learning_2.custom_dataset import se3_errors
from feasibility.grid_learning_2.generate_dataset import DEFAULT_CHUNK
from feasibility.grid_learning_2.generate_dataset import DEFAULT_DURATION_S
from feasibility.grid_learning_2.generate_dataset import DEFAULT_MAPS_DIR
from feasibility.grid_learning_2.generate_dataset import DEFAULT_SETTLE_STEPS
from feasibility.grid_learning_2.generate_dataset import DEFAULT_WZ_ZERO_FRAC
from feasibility.grid_learning_2.generate_dataset import footprint_clear
from feasibility.grid_learning_2.generate_dataset import MAP_GLOB
from feasibility.grid_learning_2.generate_dataset import obstacle_height_threshold
from feasibility.grid_learning_2.generate_dataset import resolve_path
from feasibility.grid_learning_2.generate_dataset import sample_command_fields
from feasibility.grid_learning_2.generate_dataset import simulate_map
from feasibility.grid_learning_2.generate_dataset import WZ_RANGE
from feasibility.grid_learning_2.model import check_lattice_alignment
from feasibility.grid_learning_2.model import TARGET_NAMES
from feasibility.grid_learning_2.model import WZ_MAX
from feasibility.grid_learning_2.train import DEFAULT_WZ_BINS
from feasibility.grid_learning_2.train import fmt
from feasibility.grid_learning_2.train import load_checkpoint
from feasibility.grid_learning_2.train import masked_rmse_per_head
from feasibility.grid_learning_2.train import r_squared
from feasibility.grid_learning_2.train import top_decile_rmse
from feasibility.grid_learning_2.train import wz_bin_index
from feasibility.grid_learning_2.utils import heightmap_to_tensor
from feasibility.grid_learning_2.utils import spawn_lattice
from feasibility.grid_learning_2.utils import SPAWN_LIMIT
from feasibility.grid_learning_2.utils import SPAWN_STEP
from feasibility.heightmap import HeightMapReader

DEFAULT_N_MAPS = 5
DEFAULT_ROWS_PER_MAP = 5
DEFAULT_SEED = 0

NEAR_OBSTACLE_RADIUS = 1.5  # m -- see module docstring: the rear-wheel rim-reach padding
# utils.SPAWN_LIMIT already derives (1.101 m axle offset + 0.35 m wheel radius)
NEAR_OBSTACLE_STEP = 0.25  # m, sampling pitch of the disc of offsets probed within the radius

E_POS_IDX = TARGET_NAMES.index("e_pos")
E_ROT_IDX = TARGET_NAMES.index("e_rot")


def near_obstacle_mask(
    terrain: HeightMapReader,
    lattice: np.ndarray,
    radius: float = NEAR_OBSTACLE_RADIUS,
    step: float = NEAR_OBSTACLE_STEP,
) -> np.ndarray:
    """[G, G] bool -- True where terrain ANYWHERE within `radius` m of a lattice cell's center
    clears obstacle_height_threshold. Independent of footprint_clear (which only checks the three
    wheel contacts at rest, at the cell's own position): a cell can be footprint-clear yet still
    sweep its rear wheel near the box during the commanded turn, which is the divergence this
    dataset targets. Uses a strict `>` (not `>=`): on perfectly flat terrain the threshold equals
    the (only) height present, so `>=` would mark every cell "near an obstacle" that trivially
    isn't -- see the dry-run self-check below, which catches exactly this."""
    threshold = obstacle_height_threshold(terrain)
    coords = np.arange(-radius, radius + 1e-9, step)
    OX, OY = np.meshgrid(coords, coords)
    keep = (OX**2 + OY**2) <= radius**2
    ox, oy = OX[keep], OY[keep]
    wx = lattice[..., 0, None] + ox
    wy = lattice[..., 1, None] + oy
    heights = np.asarray(terrain.sample(wx, wy), dtype=np.float64)
    return heights.max(axis=-1) > threshold


def resolve_training_maps(ckpt: dict[str, object]) -> set[str]:
    """The set of map paths the checkpoint was trained on, recovered by re-opening its
    self-described `dataset_path` (train.build_checkpoint), or an empty set (with a warning) if
    that file has since moved or been deleted.

    v1's counterpart also has to recover `resolution` here, because its checkpoint does not store
    it -- so a missing training file forces a guess at the CNN grid itself. v2's checkpoint carries
    `resolution`/`n_cells`/`extent`/`spawn_xy`, so this function's only job is the exclusion set
    and a missing file costs nothing but that."""
    dataset_path = pathlib.Path(str(ckpt["dataset_path"]))
    if dataset_path.exists():
        with h5py.File(dataset_path, "r") as f:
            train_maps = {p.decode() if isinstance(p, bytes) else p for p in f["map_path"][()]}
        print(f"[checkpoint] training dataset {dataset_path} found: {len(train_maps)} distinct "
              f"training map path(s) to exclude")
        return train_maps
    print(
        f"[checkpoint] WARNING: training dataset {dataset_path} not found on disk -- can't exclude "
        f"the maps it was trained on, so every selected map is evaluated. The CNN grid is "
        f"unaffected: this checkpoint stores its own resolution/n_cells/spawn_xy."
    )
    return set()


def select_eval_maps(
    maps_dir: pathlib.Path,
    n_maps: int,
    rng: np.random.Generator,
    exclude: set[str],
) -> list[pathlib.Path]:
    """`n_maps` extension-less heightmap stems drawn without replacement from every PNG directly in
    `maps_dir`, after dropping any stem in `exclude` (the checkpoint's own training maps, see
    resolve_training_maps) -- same sorted-candidates-then-seeded-draw recipe as
    generate_dataset.select_maps, restated here since it also needs the exclusion step that
    function doesn't have."""
    candidates = sorted(p.with_suffix("") for p in maps_dir.glob(MAP_GLOB))
    if exclude:
        before = len(candidates)
        candidates = [p for p in candidates if str(p) not in exclude]
        if before != len(candidates):
            print(f"[maps]     excluded {before - len(candidates)} map(s) already used to train "
                  f"this checkpoint")
    if len(candidates) < n_maps:
        raise ValueError(
            f"{maps_dir} holds {len(candidates)} eligible map(s) (after training-map exclusion), "
            f"need {n_maps}. Lower +n_maps, point +maps_dir elsewhere, or pass "
            f"+exclude_training_maps=false."
        )
    return [candidates[i] for i in rng.choice(len(candidates), size=n_maps, replace=False)]


def _paired_rmse(
    pred_pos: np.ndarray, pred_rot: np.ndarray, real_pos: np.ndarray, real_rot: np.ndarray
) -> torch.Tensor | None:
    """[K]=[e_pos, e_rot] RMSE over every entry (no further masking -- callers already pass only
    the subset they want scored). Returns None for an empty subset rather than 0.0, so a report can
    print "n/a" instead of a misleadingly perfect score for zero samples."""
    if pred_pos.size == 0:
        return None
    pred = torch.from_numpy(np.stack([pred_pos, pred_rot], axis=-1))
    target = torch.from_numpy(np.stack([real_pos, real_rot], axis=-1))
    mask = torch.ones(pred.shape[:-1], dtype=torch.bool)
    return masked_rmse_per_head(pred, target, mask)


def wz_bin_rmse(
    pred_pos: np.ndarray,
    pred_rot: np.ndarray,
    real_pos: np.ndarray,
    real_rot: np.ndarray,
    wz: np.ndarray,
    n_bins: int = DEFAULT_WZ_BINS,
) -> tuple[list[torch.Tensor | None], np.ndarray, np.ndarray]:
    """RMSE per |wz| bin over the valid cells -- returns (per-bin [K] RMSE with None for an empty
    bin, per-bin counts [n_bins], bin edges [n_bins + 1]).

    Binned on |wz| over [0, WZ_MAX] rather than on the signed command over [-WZ_MAX, WZ_MAX]: the
    mirror symmetry the model is trained on (design.md section 7c) makes +w and -w the same
    magnitude of turn on mirrored terrain, so folding the sign halves the bins' variance without
    discarding anything the model is supposed to distinguish. |wz| is also the quantity that sets
    how far the fixed-duration turn sweeps, i.e. the one the head actually needs to read.

    This is the diagnostic v1 structurally cannot produce: with ONE command per row it takes one
    simulated row per bin to fill the table, where a single v2 map's 225 per-cell draws populate
    every bin at once. It is also the axis design.md section 1b says the reframing exists to
    improve, so it is the number that says whether it did."""
    bins = wz_bin_index(torch.from_numpy(np.abs(wz)), n_bins, 0.0, WZ_MAX).numpy()
    edges = np.linspace(0.0, WZ_MAX, n_bins + 1)
    per_bin: list[torch.Tensor | None] = []
    counts = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        sel = bins == b
        counts[b] = int(sel.sum())
        per_bin.append(_paired_rmse(pred_pos[sel], pred_rot[sel], real_pos[sel], real_rot[sel]))
    return per_bin, counts, edges


def report(
    checkpoint_path: pathlib.Path,
    ckpt: dict[str, object],
    real_pos: np.ndarray,
    real_rot: np.ndarray,
    pred_pos: np.ndarray,
    pred_rot: np.ndarray,
    near: np.ndarray,
    wz: np.ndarray,
    n_maps: int,
    rows_per_map: int,
    maps_dir: pathlib.Path,
    n_blocked: int,
    n_diverged: int,
    n_bins: int,
) -> dict[str, object]:
    """Prints the stdout battery (totals/means + the same masked_rmse_per_head/top_decile_rmse/
    r_squared metrics train.py's own final_report prints, so numbers are directly comparable, plus
    the per-|wz| bin table) and returns everything make_figure and eval_log.build_row need, so the
    three cannot recompute a number differently."""
    n_valid = real_pos.size
    abs_err_pos = np.abs(pred_pos - real_pos)
    abs_err_rot = np.abs(pred_rot - real_rot)

    pred = torch.from_numpy(np.stack([pred_pos, pred_rot], axis=-1))
    target = torch.from_numpy(np.stack([real_pos, real_rot], axis=-1))
    mask_all = torch.ones(n_valid, dtype=torch.bool)

    overall_rmse = masked_rmse_per_head(pred, target, mask_all)
    flat_rmse = _paired_rmse(pred_pos[~near], pred_rot[~near], real_pos[~near], real_rot[~near])
    near_rmse = _paired_rmse(pred_pos[near], pred_rot[near], real_pos[near], real_rot[near])
    decile_rmse = top_decile_rmse(pred, target, mask_all)
    r2 = r_squared(pred, target, mask_all)
    bin_rmse, bin_counts, bin_edges = wz_bin_rmse(
        pred_pos, pred_rot, real_pos, real_rot, wz, n_bins
    )

    print(f"\n[eval] checkpoint={checkpoint_path} (trained epoch {ckpt['epoch']}, "
          f"val_loss={float(ckpt['val_loss']):.4f}, head_fusion={ckpt.get('head_fusion', '?')})")
    print(f"[eval] {n_maps} maps x {rows_per_map} command field(s) from {maps_dir}: "
          f"{n_valid} valid trials ({n_blocked} blocked, {n_diverged} diverged)")
    print(f"[eval] total accumulated error (sum |pred-real| over {n_valid} cells): "
          f"e_pos={abs_err_pos.sum():.3f} m   e_rot={abs_err_rot.sum():.3f} rad")
    print(f"[eval] mean error (|pred-real| per cell):                           "
          f"e_pos={abs_err_pos.mean():.4f} m   e_rot={abs_err_rot.mean():.4f} rad")
    print(f"[eval] RMSE, all valid cells:              {fmt(overall_rmse)}")
    print(f"[eval] RMSE, flat cells (n={int((~near).sum())}):           "
          f"{fmt(flat_rmse) if flat_rmse is not None else 'n/a'}")
    print(f"[eval] RMSE, near-obstacle cells (n={int(near.sum())}):     "
          f"{fmt(near_rmse) if near_rmse is not None else 'n/a'}")
    print(f"[eval] RMSE, top-decile true e_pos:        {fmt(decile_rmse)}")
    print(f"[eval] R^2 per head:                       {fmt(r2)}")
    # The per-|wz| table: what the reframing was for (design.md section 1b). A flat column here
    # means the command axis is tracked evenly; a column that climbs with |wz| means the model is
    # weakest exactly where the turn sweeps furthest, i.e. where a collision is most likely.
    print(f"[eval] RMSE by |wz| bin ({n_bins} bins over [0, {WZ_MAX}] rad/s) -- v1 needs one "
          f"simulated row per bin, a per-cell field fills them all from one map:")
    for b, value in enumerate(bin_rmse):
        span = f"[{bin_edges[b]:.3f}, {bin_edges[b + 1]:.3f})"
        body = fmt(value) if value is not None else "n/a (no valid cell)"
        print(f"         |wz| in {span}  n={bin_counts[b]:>5d}   {body}")

    return dict(
        overall_rmse=overall_rmse, flat_rmse=flat_rmse, near_rmse=near_rmse,
        decile_rmse=decile_rmse, r2=r2, wz_bin_rmse=bin_rmse, wz_bin_counts=bin_counts,
        wz_bin_edges=bin_edges, total_pos=abs_err_pos.sum(), total_rot=abs_err_rot.sum(),
        mean_pos=abs_err_pos.mean(), mean_rot=abs_err_rot.mean(), n_valid=n_valid,
        n_flat=int((~near).sum()), n_near=int(near.sum()), n_blocked=n_blocked,
        n_diverged=n_diverged, epoch=ckpt["epoch"],
    )


def make_figure(
    checkpoint_path: pathlib.Path,
    real_pos: np.ndarray,
    real_rot: np.ndarray,
    pred_pos: np.ndarray,
    pred_rot: np.ndarray,
    near: np.ndarray,
    stats: dict[str, object],
    save_fig: pathlib.Path | None,
) -> None:
    """One figure: RMSE bar chart by subset (overall/flat/near-obstacle/top-decile), the two
    predicted-vs-real scatters colored by flat/near-obstacle, the per-|wz| bin RMSE, and a text
    panel restating the stdout report so the figure is self-contained. `+save_fig=PATH` saves
    instead of showing interactively -- same convention learning/error_visual.py's `--out` uses.

    Laid out on a 2x3 gridspec rather than v1's 2x2: the |wz| breakdown is a fifth panel with its
    own x axis (command magnitude, not subset), so grouping it into the subset bar chart would put
    two different category axes on one plot."""
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3)
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_pos = fig.add_subplot(gs[0, 1])
    ax_rot = fig.add_subplot(gs[0, 2])
    ax_bins = fig.add_subplot(gs[1, 0])
    ax_text = fig.add_subplot(gs[1, 1:])

    def _grouped_bars(ax, labels, values, title, xlabel=None):
        """Two bars per category on twinned axes -- e_pos in metres on the left, e_rot in radians
        on the right. They share no unit, so one axis would make whichever is numerically smaller
        invisible."""
        e_pos_vals = [float(v[E_POS_IDX]) if v is not None else 0.0 for v in values]
        e_rot_vals = [float(v[E_ROT_IDX]) if v is not None else 0.0 for v in values]
        x = np.arange(len(labels))
        width = 0.35
        ax_twin = ax.twinx()
        ax.bar(x - width / 2, e_pos_vals, width, label="e_pos (m)", color="#1f77b4")
        ax_twin.bar(x + width / 2, e_rot_vals, width, label="e_rot (rad)", color="#d62728")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, fontsize=8)
        ax.set_ylabel("RMSE e_pos (m)", color="#1f77b4")
        ax_twin.set_ylabel("RMSE e_rot (rad)", color="#d62728")
        ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        lines = ax.get_legend_handles_labels()[0] + ax_twin.get_legend_handles_labels()[0]
        labs = ax.get_legend_handles_labels()[1] + ax_twin.get_legend_handles_labels()[1]
        ax.legend(lines, labs, loc="upper left", fontsize=8)

    _grouped_bars(
        ax_bar,
        ["overall", "flat", "near-obstacle", "top-decile"],
        [stats["overall_rmse"], stats["flat_rmse"], stats["near_rmse"], stats["decile_rmse"]],
        "RMSE by subset",
    )
    edges, counts = stats["wz_bin_edges"], stats["wz_bin_counts"]
    _grouped_bars(
        ax_bins,
        [f"{edges[b]:.2f}-{edges[b + 1]:.2f}\nn={counts[b]}" for b in range(len(counts))],
        stats["wz_bin_rmse"],
        "RMSE by |wz| bin (the axis v2 exists to improve)",
        xlabel="|commanded wz| (rad/s)",
    )

    for ax, real, pred, ylabel in (
        (ax_pos, real_pos, pred_pos, "e_pos"), (ax_rot, real_rot, pred_rot, "e_rot"),
    ):
        ax.scatter(real[~near], pred[~near], s=8, alpha=0.5, color="#2ca02c", label="flat")
        ax.scatter(real[near], pred[near], s=8, alpha=0.5, color="#d62728", label="near-obstacle")
        lim = max(float(real.max(initial=0.0)), float(pred.max(initial=0.0)), 1e-6)
        ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="y = x")
        ax.set_xlabel(f"real {ylabel}")
        ax.set_ylabel(f"pred {ylabel}")
        ax.set_title(f"predicted vs. real {ylabel}")
        ax.legend(loc="upper left", fontsize=8)

    ax_text.axis("off")
    text = (
        f"checkpoint: {checkpoint_path.name}\n"
        f"epoch {stats.get('epoch', '?')}\n\n"
        f"valid trials:  {stats['n_valid']}  (blocked {stats['n_blocked']}, "
        f"diverged {stats['n_diverged']})\n"
        f"flat cells:    {stats['n_flat']}\n"
        f"near-obstacle: {stats['n_near']}\n\n"
        f"total accumulated |pred-real| error:\n"
        f"  e_pos = {stats['total_pos']:.3f} m\n"
        f"  e_rot = {stats['total_rot']:.3f} rad\n\n"
        f"mean |pred-real| error:\n"
        f"  e_pos = {stats['mean_pos']:.4f} m\n"
        f"  e_rot = {stats['mean_rot']:.4f} rad\n\n"
        f"R^2:  e_pos = {float(stats['r2'][E_POS_IDX]):.3f}   "
        f"e_rot = {float(stats['r2'][E_ROT_IDX]):.3f}"
    )
    ax_text.text(0.0, 1.0, text, va="top", ha="left", family="monospace", fontsize=10,
                 transform=ax_text.transAxes)

    fig.suptitle(f"{checkpoint_path.name} -- evaluated on fresh simulation trials")
    fig.tight_layout()

    if save_fig is not None:
        fig.savefig(save_fig, dpi=150)
        print(f"[figure]   saved to {save_fig}")
    else:
        plt.show()


def evaluate(cfg: DictConfig) -> None:
    if "checkpoint" not in cfg:
        raise SystemExit("pass +checkpoint=<path to a trained GridDivergenceNet .pt file>")
    checkpoint_path = resolve_path(str(cfg["checkpoint"]))

    n_maps = int(cfg.get("n_maps", DEFAULT_N_MAPS))
    rows_per_map = int(cfg.get("rows_per_map", DEFAULT_ROWS_PER_MAP))
    seed = int(cfg.get("seed", DEFAULT_SEED))
    maps_dir = resolve_path(str(cfg.get("maps_dir", DEFAULT_MAPS_DIR)))
    wz_zero_frac = float(cfg.get("wz_zero_frac", DEFAULT_WZ_ZERO_FRAC))
    wz_grid = cfg.get("wz_grid", None)
    wz_grid = int(wz_grid) if wz_grid is not None else None
    n_bins = int(cfg.get("wz_bins", DEFAULT_WZ_BINS))
    exclude_training_maps = bool(cfg.get("exclude_training_maps", True))
    near_obstacle_radius = float(cfg.get("near_obstacle_radius", NEAR_OBSTACLE_RADIUS))
    duration_s = float(cfg.get("duration_s", DEFAULT_DURATION_S))
    chunk = int(cfg.get("chunk", DEFAULT_CHUNK))
    settle_steps = int(cfg.get("settle_steps", DEFAULT_SETTLE_STEPS))
    mu = float(cfg.get("mu", 0.8))
    k_turn = float(cfg.get("k_turn", dynamics.K_TURN))
    device_str = str(cfg.get("device", "cuda:0"))
    save_fig = cfg.get("save_fig", None)
    save_fig = resolve_path(str(save_fig)) if save_fig is not None else None
    dry_run = bool(cfg.get("dry_run", False))
    log_path = resolve_path(str(cfg.get("log_path", eval_log.DEFAULT_LOG_PATH)))
    no_log = bool(cfg.get("no_log", False))

    device = torch.device(device_str)
    model, ckpt = load_checkpoint(checkpoint_path, device)
    model.eval()
    n_cells = int(ckpt["n_cells"])
    resolution = float(ckpt["resolution"])
    # [G, G, 2]. build_checkpoint stores it on CPU, but load_checkpoint's torch.load maps the whole
    # file onto `device`, so it comes back wherever the model is -- hence the explicit .cpu().
    spawn_xy_ckpt = ckpt["spawn_xy"].cpu()

    lattice = spawn_lattice()
    G = lattice.shape[0]
    if not np.allclose(lattice, spawn_xy_ckpt.numpy()):
        raise SystemExit(
            "this script's spawn_lattice() does not match the checkpoint's own spawn_xy -- "
            "utils.SPAWN_STEP/SPAWN_LIMIT must have changed since this checkpoint was trained, "
            "so lattice cell (i, j) no longer means the same world position"
        )
    # The stronger statement, and the one v1 has no counterpart for: the network's integer
    # centre-crop must LAND on those coordinates. Cheap, and it fails in milliseconds rather than
    # after a GPU-hour of simulation -- same call generate_dataset.py makes for the same reason.
    err = check_lattice_alignment(lattice, n_input=n_cells, resolution=resolution)
    print(f"[lattice]  {G}x{G} = {G * G} cells, step {SPAWN_STEP} m, |x|,|y| <= {SPAWN_LIMIT} m")
    print(f"[readout]  {n_cells}x{n_cells} @ {resolution} m -> crop lands on spawn_xy, "
          f"max |diff| = {err:.2e} m (no interpolation)")
    print(f"[checkpoint] {checkpoint_path} (trained epoch {ckpt['epoch']}, "
          f"val_loss={float(ckpt['val_loss']):.4f}, head_fusion={ckpt.get('head_fusion', '?')}, "
          f"base_width={ckpt.get('base_width', '?')})")

    train_maps = resolve_training_maps(ckpt)
    rng = np.random.default_rng(seed)
    map_paths = select_eval_maps(
        maps_dir, n_maps, rng, train_maps if exclude_training_maps else set()
    )
    # Drawn with generate_dataset's OWN sampler, not a restated linspace: the eval command
    # distribution is then the training one by construction. RNG order matches the generator's
    # (maps first, then fields), so a seed reproduces a run exactly.
    wz_per_map = sample_command_fields(
        n_maps, rows_per_map, G, rng, wz_zero_frac=wz_zero_frac, wz_grid=wz_grid
    )  # [n_maps, rows_per_map, G, G]
    n_zero = int((wz_per_map == 0.0).sum())
    draw = f"U{WZ_RANGE}" if wz_grid is None else f"{wz_grid} discrete levels over {WZ_RANGE}"
    print(f"[maps]     {n_maps} from {maps_dir}, {rows_per_map} command field(s) each -> "
          f"{n_maps * rows_per_map} rows")
    print(f"[command]  per-cell wz ~ {draw}, zero anchor {100 * wz_zero_frac:.1f}% -> "
          f"{n_zero}/{wz_per_map.size} cells at exactly 0, "
          f"{n_maps * rows_per_map * G * G} independent draws "
          f"(a v1 eval of this size would give {n_maps * rows_per_map})")

    if dry_run:
        # Cheap CPU-only self checks -- same +dry_run convention generate_dataset.py uses. The
        # lattice/readout equality above already ran and is the strongest of them.
        flat = HeightMapReader.flat(xlim=(-8.0, 8.0), ylim=(-8.0, 8.0))
        assert footprint_clear(flat, lattice).all(), "flat ground must clear every spawn cell"
        assert not near_obstacle_mask(flat, lattice, radius=near_obstacle_radius).any(), (
            "flat ground must never be classified as near an obstacle"
        )
        # The command really is a FIELD: within a row, cells must differ. Same assert
        # generate_dataset.py and custom_dataset.py both make, for the property v1 files lack.
        flat_rows = wz_per_map.reshape(n_maps * rows_per_map, -1)
        spread = flat_rows.max(axis=1) - flat_rows.min(axis=1)
        assert (spread > 1e-6).all(), "some row has a constant command field"
        print("[dry-run]  lattice/readout/near-obstacle/command-field self-checks ok, "
              "nothing simulated")
        return

    init_warp_device(device_str)  # pins ostrich onto the same GPU `device` puts hstack/the model
    # on -- ostrich has no device config field of its own, see init_warp_device's docstring
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)
    render_config.vis_type = "null"  # headless, same reasoning as generate_dataset.py

    ostrich_dt = sim_config.target_timestep_seconds
    hstack_dt = dynamics.DT
    T_o = int(round(duration_s / ostrich_dt))
    T_h = int(round(duration_s / hstack_dt))
    if abs(T_o * ostrich_dt - T_h * hstack_dt) > 1e-6:
        raise ValueError(
            f"duration_s={duration_s} is not an exact multiple of both sims' dt (ostrich "
            f"{ostrich_dt}, hstack {hstack_dt}) -- the two rollouts would end at different times"
        )
    print(f"[ostrich]  {T_o} steps @ dt={ostrich_dt}, chunk={chunk}, settle={settle_steps} steps/chunk")
    print(f"[hstack]   {T_h} steps @ dt={hstack_dt}, device={device_str}")

    real_pos_all, real_rot_all, pred_pos_all, pred_rot_all, near_all, wz_all = [], [], [], [], [], []
    n_blocked_total = n_diverged_total = 0

    for m, p in enumerate(map_paths):
        print(f"[map {m + 1}/{n_maps}] {p.name}")
        terrain = HeightMapReader.load(p)
        clear = footprint_clear(terrain, lattice)
        near = near_obstacle_mask(terrain, lattice, radius=near_obstacle_radius)
        wz_field = wz_per_map[m]  # [L, G, G]

        y, mask = simulate_map(
            terrain, lattice, clear, wz_field,
            sim_config=sim_config, render_config=render_config, engine_config=engine_config,
            logging_config=logging_config, ostrich_dt=ostrich_dt, hstack_dt=hstack_dt,
            duration_s=duration_s, chunk=chunk, settle_steps=settle_steps, mu=mu,
            k_turn=k_turn, device=device_str,
        )  # y [L, G, G, 14], mask [L, G, G]

        # One heightmap tensor per map, repeated over the rows -- the trunk is control-free, so
        # every row of a map produces the SAME terrain code; only the head's command field differs.
        # (A planner would exploit that with terrain_code()/head_from_code(); here the repeat is
        # simpler and the cost is one map's worth of convs, not one per row.)
        heightmap = heightmap_to_tensor(
            terrain, n_cells=n_cells, resolution=resolution, device=device
        )
        wz_t = torch.from_numpy(wz_field).to(device)  # [L, G, G]
        pred = model.predict(
            heightmap.unsqueeze(0).expand(wz_field.shape[0], -1, -1), wz_t
        ).cpu().numpy()  # [L, G, G, 2]

        real_pos, real_rot = se3_errors(
            poses_to_se3(y[..., :7]), poses_to_se3(y[..., 7:])
        )  # each [L, G, G]
        near_b = np.broadcast_to(near[None, :, :], mask.shape)

        real_pos_all.append(real_pos[mask]); real_rot_all.append(real_rot[mask])
        pred_pos_all.append(pred[..., E_POS_IDX][mask]); pred_rot_all.append(pred[..., E_ROT_IDX][mask])
        near_all.append(near_b[mask]); wz_all.append(wz_field[mask])

        n_map_total = mask.size
        n_map_blocked = int((~clear).sum()) * rows_per_map
        n_map_valid = int(mask.sum())
        n_map_diverged = n_map_total - n_map_blocked - n_map_valid
        n_blocked_total += n_map_blocked
        n_diverged_total += n_map_diverged
        print(f"    -> {n_map_valid} valid, {n_map_blocked} blocked, {n_map_diverged} diverged")

    real_pos = np.concatenate(real_pos_all)
    real_rot = np.concatenate(real_rot_all)
    pred_pos = np.concatenate(pred_pos_all)
    pred_rot = np.concatenate(pred_rot_all)
    near = np.concatenate(near_all)
    wz = np.concatenate(wz_all)

    stats = report(
        checkpoint_path, ckpt, real_pos, real_rot, pred_pos, pred_rot, near, wz,
        n_maps, rows_per_map, maps_dir, n_blocked_total, n_diverged_total, n_bins,
    )

    if not no_log:
        row = eval_log.build_row(
            checkpoint_path, ckpt, stats, maps_dir=maps_dir, n_maps=n_maps,
            rows_per_map=rows_per_map, seed=seed, wz_zero_frac=wz_zero_frac, wz_grid=wz_grid,
            duration_s=duration_s, mu=mu, k_turn=k_turn, device_str=device_str,
            near_obstacle_radius=near_obstacle_radius,
            exclude_training_maps=exclude_training_maps,
        )
        eval_log.append_row(row, log_path)

    make_figure(checkpoint_path, real_pos, real_rot, pred_pos, pred_rot, near, stats, save_fig)


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest", version_base=None)
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
