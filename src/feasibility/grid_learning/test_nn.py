"""Evaluates a trained `model.GridPoseErrorNet` checkpoint against FRESH ostrich+helhest_stack
simulation trials on a set of heightmaps -- e.g. a whole batch folder such as
`assets/box_random/1/` -- rather than against rows already baked into one `dataset_grid_*.h5`
file (which is all `train.py`'s own val split and `gl_replay_grid.py` can show).

Runs the actual generation pipeline (`generate_dataset.py`'s own `simulate_map`, imported
directly rather than restated -- it is ~100 lines of Hydra/chunking/ostrich+hstack orchestration,
not a small formula worth duplicating per-script the way this package restates the SE(3)-error/
footprint-clear math elsewhere), computes ground-truth (e_pos, e_rot) from the real simulator
poses via custom_dataset.py's vectorized formula, runs the checkpoint's own inference on the same
maps/commands, and reports how close the two are -- overall, and split into flat-ground vs.
near-obstacle cells, since the near-obstacle case (lateral collision divergence helhest_stack
cannot represent) is the entire reason this dataset exists.

Held-out maps: there is no repo convention of a literal "validation/" directory -- a folder like
`assets/box_random/1/` is just another indexed batch. By default this script instead reads the
checkpoint's own self-described `dataset_path` (see train.build_checkpoint) and excludes any map
that file was trained on from the candidates drawn out of `+maps_dir`, so an eval run
against the SAME folder used for training still measures generalization rather than memorization.
`+exclude_training_maps=false` disables this. If the training dataset file no longer exists on
disk, a warning is printed and every selected map is evaluated instead of failing outright -- the
checkpoint's own `extent` is still authoritative for the CNN input, only `resolution` and the
exclusion set are best-effort.

Flat vs. near-obstacle: `near_obstacle_mask` flags a lattice cell if terrain ANYWHERE within
`near_obstacle_radius` m of its center clears `generate_dataset.obstacle_height_threshold` --
deliberately independent of `footprint_clear` (which only checks the three wheel contacts at
rest): a cell can be footprint-clear yet still sweep its rear wheel near the box during the
commanded turn in place, which is exactly the divergence this dataset is built to catch. The
default radius (1.5 m) is the same rear-wheel rim-reach padding generate_dataset.py's own
SPAWN_LIMIT docstring already derives (1.101 m axle offset + 0.35 m wheel radius, rounded up).

CLI parameters (Hydra overrides, `+` prefix required -- same convention as generate_dataset.py,
needed here too since running fresh trials means building the same sim_config/render_config/
engine_config/logging_config via hydra.utils.instantiate):
    +checkpoint=STR              trained GridPoseErrorNet .pt path (required)
    +n_maps=INT                  maps drawn (without replacement) for evaluation (default: 5)
    +n_commands=INT               wz commands per map, equidistant across WZ_RANGE (default: 5)
    +maps_dir=STR                 repo-root-relative or absolute map directory (default: assets/box_random)
    +seed=INT                      RNG seed for map selection (default: 0)
    +exclude_training_maps=BOOL    drop maps the checkpoint was trained on, if that dataset file
                                    is still reachable (default: true)
    +near_obstacle_radius=FLOAT    m, neighborhood radius for the flat/near-obstacle split (default: 1.5)
    +duration_s=FLOAT              command hold time (default: 2.4, must match training's)
    +chunk=INT                     worlds per ostrich model build (default: 128)
    +settle_steps=INT              ostrich settle steps, paid once per chunk (default: 12)
    +mu=FLOAT                      ground friction (default: 0.8)
    +k_turn=FLOAT                  hstack ICR turning-rate gain (default: dynamics.K_TURN)
    +device=STR                    torch/warp device for the network, hstack, AND ostrich (via
                                    init_warp_device) (default: "cuda:0")
    +save_fig=STR                  save the comparison figure here instead of showing it interactively
    +dry_run=BOOL                  lattice/checkpoint self-checks only -- no simulation (default: false)
    +log_path=STR                  CSV evaluation log to append this run's stats to, one row per
                                    run (default: outputs/eval_log.csv, see eval_log.py)
    +no_log=BOOL                    skip the CSV log append entirely (default: false)
    Also accepts any standard Hydra config-group override against the "helhest" base config
    (engine=mujoco, simulation=..., logging=...); rendering is forced headless.

Usage:
    python src/feasibility/grid_learning/test_nn.py +checkpoint=outputs/checkpoints/dataset_grid_box_random_M2_L5_g15.pt +dry_run=true
    python src/feasibility/grid_learning/test_nn.py +checkpoint=outputs/checkpoints/dataset_grid_box_random_M2_L5_g15.pt +n_maps=1 +n_commands=1
    python src/feasibility/grid_learning/test_nn.py +checkpoint=outputs/checkpoints/dataset_grid_box_random_M2_L5_g15.pt +maps_dir=assets/box_random/1 +n_maps=3
    python src/feasibility/grid_learning/test_nn.py +checkpoint=outputs/checkpoints/dataset_grid_box_random_M2_L5_g15.pt +save_fig=outputs/eval_box_random_1.png
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
from feasibility.grid_learning import eval_log
from feasibility.grid_learning.custom_dataset import poses_to_se3
from feasibility.grid_learning.custom_dataset import se3_errors
from feasibility.grid_learning.custom_dataset import TARGET_NAMES
from feasibility.grid_learning.generate_dataset import DEFAULT_CHUNK
from feasibility.grid_learning.generate_dataset import DEFAULT_DURATION_S
from feasibility.grid_learning.generate_dataset import DEFAULT_MAPS_DIR
from feasibility.grid_learning.generate_dataset import DEFAULT_SETTLE_STEPS
from feasibility.grid_learning.generate_dataset import footprint_clear
from feasibility.grid_learning.generate_dataset import MAP_GLOB
from feasibility.grid_learning.generate_dataset import obstacle_height_threshold
from feasibility.grid_learning.generate_dataset import resolve_path
from feasibility.grid_learning.generate_dataset import simulate_map
from feasibility.grid_learning.generate_dataset import spawn_lattice
from feasibility.grid_learning.generate_dataset import SPAWN_LIMIT
from feasibility.grid_learning.generate_dataset import SPAWN_STEP
from feasibility.grid_learning.generate_dataset import WZ_RANGE
from feasibility.grid_learning.train import fmt
from feasibility.grid_learning.train import load_checkpoint
from feasibility.grid_learning.train import masked_rmse_per_head
from feasibility.grid_learning.train import r_squared
from feasibility.grid_learning.train import top_decile_rmse
from feasibility.grid_learning.utils import DEFAULT_RESOLUTION
from feasibility.grid_learning.utils import heightmap_to_tensor
from feasibility.heightmap import HeightMapReader

DEFAULT_N_MAPS = 5
DEFAULT_N_COMMANDS = 5
DEFAULT_SEED = 0

NEAR_OBSTACLE_RADIUS = 1.5  # m -- see module docstring: same rear-wheel rim-reach padding
# generate_dataset.SPAWN_LIMIT already derives (1.101 m axle offset + 0.35 m wheel radius)
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


def resolve_training_context(ckpt: dict[str, object]) -> tuple[float, set[str]]:
    """Best-effort recovery of the (resolution, training map paths) the checkpoint was produced
    with, by re-opening its self-described `dataset_path` (train.build_checkpoint). Falls back to
    utils.DEFAULT_RESOLUTION and an empty exclusion set (with a warning) if that file has since
    moved or been deleted -- the checkpoint's own `extent` is unaffected either way."""
    dataset_path = pathlib.Path(str(ckpt["dataset_path"]))
    if dataset_path.exists():
        with h5py.File(dataset_path, "r") as f:
            resolution = float(f["grid/resolution"][()])
            train_maps = {p.decode() if isinstance(p, bytes) else p for p in f["map_path"][()]}
        print(
            f"[checkpoint] training dataset {dataset_path} found: resolution={resolution} m, "
            f"{len(train_maps)} distinct training map path(s)"
        )
        return resolution, train_maps
    print(
        f"[checkpoint] WARNING: training dataset {dataset_path} not found on disk -- can't "
        f"confirm resolution or exclude training maps; using resolution={DEFAULT_RESOLUTION} m "
        f"(utils.DEFAULT_RESOLUTION) and evaluating every selected map"
    )
    return DEFAULT_RESOLUTION, set()


def select_eval_maps(
    maps_dir: pathlib.Path,
    n_maps: int,
    rng: np.random.Generator,
    exclude: set[str],
) -> list[pathlib.Path]:
    """`n_maps` extension-less heightmap stems drawn without replacement from every PNG directly
    in `maps_dir`, after dropping any stem in `exclude` (the checkpoint's own training maps, see
    resolve_training_context) -- same sorted-candidates-then-seeded-draw recipe as
    generate_dataset.select_maps, restated here since it also needs the exclusion step that
    function doesn't have."""
    candidates = sorted(p.with_suffix("") for p in maps_dir.glob(MAP_GLOB))
    if exclude:
        before = len(candidates)
        candidates = [p for p in candidates if str(p) not in exclude]
        if before != len(candidates):
            print(f"[maps]     excluded {before - len(candidates)} map(s) already used to train this checkpoint")
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
    the subset they want scored). Returns None for an empty subset rather than 0.0, so a report
    can print "n/a" instead of a misleadingly perfect score for zero samples."""
    if pred_pos.size == 0:
        return None
    pred = torch.from_numpy(np.stack([pred_pos, pred_rot], axis=-1))
    target = torch.from_numpy(np.stack([real_pos, real_rot], axis=-1))
    mask = torch.ones(pred.shape[:-1], dtype=torch.bool)
    return masked_rmse_per_head(pred, target, mask)


def report(
    checkpoint_path: pathlib.Path,
    ckpt: dict[str, object],
    real_pos: np.ndarray,
    real_rot: np.ndarray,
    pred_pos: np.ndarray,
    pred_rot: np.ndarray,
    near: np.ndarray,
    n_maps: int,
    n_commands: int,
    maps_dir: pathlib.Path,
    n_blocked: int,
    n_diverged: int,
) -> dict[str, object]:
    """Prints the stdout battery (totals/means + the same masked_rmse_per_head/top_decile_rmse/
    r_squared metrics train.py's own final_report prints, so numbers are directly comparable) and
    returns everything make_figure needs, so the two don't have to recompute anything."""
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

    print(f"\n[eval] checkpoint={checkpoint_path} (trained epoch {ckpt['epoch']}, val_loss={float(ckpt['val_loss']):.4f})")
    print(f"[eval] {n_maps} maps x {n_commands} commands from {maps_dir}: {n_valid} valid trials "
          f"({n_blocked} blocked, {n_diverged} diverged)")
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

    return dict(
        overall_rmse=overall_rmse, flat_rmse=flat_rmse, near_rmse=near_rmse, decile_rmse=decile_rmse,
        r2=r2, total_pos=abs_err_pos.sum(), total_rot=abs_err_rot.sum(),
        mean_pos=abs_err_pos.mean(), mean_rot=abs_err_rot.mean(), n_valid=n_valid,
        n_flat=int((~near).sum()), n_near=int(near.sum()), n_blocked=n_blocked, n_diverged=n_diverged,
        epoch=ckpt["epoch"],
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
    """One 2x2 figure: RMSE bar chart (overall/flat/near-obstacle/top-decile), predicted-vs-real
    scatter per head colored by flat/near-obstacle, and a text panel restating the stdout report
    so the figure is self-contained. `+save_fig=PATH` saves instead of showing interactively --
    same convention as learning/error_visual.py's `--out`."""
    fig, ((ax_bar, ax_pos), (ax_text, ax_rot)) = plt.subplots(2, 2, figsize=(13, 10))

    categories = ["overall", "flat", "near-obstacle", "top-decile"]
    rmses = [stats["overall_rmse"], stats["flat_rmse"], stats["near_rmse"], stats["decile_rmse"]]
    e_pos_vals = [float(r[E_POS_IDX]) if r is not None else 0.0 for r in rmses]
    e_rot_vals = [float(r[E_ROT_IDX]) if r is not None else 0.0 for r in rmses]
    x = np.arange(len(categories))
    width = 0.35
    ax_bar.bar(x - width / 2, e_pos_vals, width, label="e_pos (m)", color="#1f77b4")
    ax_bar_rot = ax_bar.twinx()
    ax_bar_rot.bar(x + width / 2, e_rot_vals, width, label="e_rot (rad)", color="#d62728")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(categories, rotation=15)
    ax_bar.set_ylabel("RMSE e_pos (m)", color="#1f77b4")
    ax_bar_rot.set_ylabel("RMSE e_rot (rad)", color="#d62728")
    ax_bar.set_title("RMSE by subset")
    lines = ax_bar.get_legend_handles_labels()[0] + ax_bar_rot.get_legend_handles_labels()[0]
    labels = ax_bar.get_legend_handles_labels()[1] + ax_bar_rot.get_legend_handles_labels()[1]
    ax_bar.legend(lines, labels, loc="upper left")

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
        f"valid trials:  {stats['n_valid']}  (blocked {stats['n_blocked']}, diverged {stats['n_diverged']})\n"
        f"flat cells:    {stats['n_flat']}\n"
        f"near-obstacle: {stats['n_near']}\n\n"
        f"total accumulated |pred-real| error:\n"
        f"  e_pos = {stats['total_pos']:.3f} m\n"
        f"  e_rot = {stats['total_rot']:.3f} rad\n\n"
        f"mean |pred-real| error:\n"
        f"  e_pos = {stats['mean_pos']:.4f} m\n"
        f"  e_rot = {stats['mean_rot']:.4f} rad\n\n"
        f"R^2:  e_pos = {float(stats['r2'][E_POS_IDX]):.3f}   e_rot = {float(stats['r2'][E_ROT_IDX]):.3f}"
    )
    ax_text.text(0.0, 1.0, text, va="top", ha="left", family="monospace", fontsize=10, transform=ax_text.transAxes)

    fig.suptitle(f"{checkpoint_path.name} -- evaluated on fresh simulation trials")
    fig.tight_layout()

    if save_fig is not None:
        fig.savefig(save_fig, dpi=150)
        print(f"[figure]   saved to {save_fig}")
    else:
        plt.show()


def evaluate(cfg: DictConfig) -> None:
    if "checkpoint" not in cfg:
        raise SystemExit("pass +checkpoint=<path to a trained GridPoseErrorNet .pt file>")
    checkpoint_path = resolve_path(str(cfg["checkpoint"]))

    n_maps = int(cfg.get("n_maps", DEFAULT_N_MAPS))
    n_commands = int(cfg.get("n_commands", DEFAULT_N_COMMANDS))
    seed = int(cfg.get("seed", DEFAULT_SEED))
    maps_dir = resolve_path(str(cfg.get("maps_dir", DEFAULT_MAPS_DIR)))
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
    extent = float(ckpt["extent"])
    spawn_xy_ckpt = ckpt["spawn_xy"].to(device)  # [G, G, 2]

    lattice = spawn_lattice()
    G = lattice.shape[0]
    if not np.allclose(lattice, spawn_xy_ckpt.cpu().numpy()):
        raise SystemExit(
            "this script's spawn_lattice() does not match the checkpoint's own spawn_xy -- "
            "generate_dataset.SPAWN_STEP/SPAWN_LIMIT must have changed since this checkpoint "
            "was trained, so lattice cell (i, j) no longer means the same world position"
        )
    print(f"[lattice]  {G}x{G} = {G * G} cells, step {SPAWN_STEP} m, |x|,|y| <= {SPAWN_LIMIT} m")
    print(
        f"[checkpoint] {checkpoint_path} (trained epoch {ckpt['epoch']}, "
        f"val_loss={float(ckpt['val_loss']):.4f})"
    )

    resolution, train_maps = resolve_training_context(ckpt)
    rng = np.random.default_rng(seed)
    map_paths = select_eval_maps(
        maps_dir, n_maps, rng, train_maps if exclude_training_maps else set()
    )
    wz_values = np.linspace(*WZ_RANGE, n_commands, dtype=np.float32)
    print(f"[maps]     {n_maps} from {maps_dir}, {n_commands} wz command(s) each")

    if dry_run:
        # Cheap CPU-only self checks -- same +dry_run convention generate_dataset.py uses. On
        # perfectly flat terrain (no obstacle at all) footprint_clear must admit every cell and
        # near_obstacle_mask must flag none of them -- see that function's docstring for why a
        # naive `>=` would get this case wrong.
        flat = HeightMapReader.flat(xlim=(-8.0, 8.0), ylim=(-8.0, 8.0))
        assert footprint_clear(flat, lattice).all(), "flat ground must clear every spawn cell"
        assert not near_obstacle_mask(flat, lattice, radius=near_obstacle_radius).any(), (
            "flat ground must never be classified as near an obstacle"
        )
        print("[dry-run]  lattice/checkpoint/near-obstacle self-checks ok, nothing simulated")
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

    real_pos_all, real_rot_all, pred_pos_all, pred_rot_all, near_all = [], [], [], [], []
    n_blocked_total = n_diverged_total = 0

    for m, p in enumerate(map_paths):
        print(f"[map {m + 1}/{n_maps}] {p.name}")
        terrain = HeightMapReader.load(p)
        clear = footprint_clear(terrain, lattice)
        near = near_obstacle_mask(terrain, lattice, radius=near_obstacle_radius)

        y, mask = simulate_map(
            terrain, lattice, clear, wz_values,
            sim_config=sim_config, render_config=render_config, engine_config=engine_config,
            logging_config=logging_config, ostrich_dt=ostrich_dt, hstack_dt=hstack_dt,
            duration_s=duration_s, chunk=chunk, settle_steps=settle_steps, mu=mu,
            k_turn=k_turn, device=device_str,
        )  # y [L, G, G, 14], mask [L, G, G]

        heightmap = heightmap_to_tensor(terrain, resolution=resolution, extent=extent, device=device)
        wz_t = torch.from_numpy(wz_values).to(device)
        pred = model.predict(
            heightmap.unsqueeze(0).repeat(len(wz_values), 1, 1), wz_t, spawn_xy_ckpt, extent=extent,
        ).cpu().numpy()  # [L, G, G, 2]

        real_pos, real_rot = se3_errors(poses_to_se3(y[..., :7]), poses_to_se3(y[..., 7:]))  # [L, G, G]
        near_b = np.broadcast_to(near[None, :, :], mask.shape)

        real_pos_all.append(real_pos[mask]); real_rot_all.append(real_rot[mask])
        pred_pos_all.append(pred[..., E_POS_IDX][mask]); pred_rot_all.append(pred[..., E_ROT_IDX][mask])
        near_all.append(near_b[mask])

        n_map_total = clear.size * n_commands
        n_map_blocked = int((~clear).sum()) * n_commands
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

    stats = report(
        checkpoint_path, ckpt, real_pos, real_rot, pred_pos, pred_rot, near,
        n_maps, n_commands, maps_dir, n_blocked_total, n_diverged_total,
    )

    if not no_log:
        row = eval_log.build_row(
            checkpoint_path, stats, val_loss=float(ckpt["val_loss"]), maps_dir=maps_dir,
            n_maps=n_maps, n_commands=n_commands, seed=seed, duration_s=duration_s, mu=mu,
            k_turn=k_turn, device_str=device_str, near_obstacle_radius=near_obstacle_radius,
            exclude_training_maps=exclude_training_maps,
        )
        eval_log.append_row(row, log_path)

    make_figure(checkpoint_path, real_pos, real_rot, pred_pos, pred_rot, near, stats, save_fig)


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest", version_base=None)
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
