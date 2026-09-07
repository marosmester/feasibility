"""Appends one row per `test_nn.py` run to a persistent CSV evaluation log (default
`outputs/eval_log.csv`), so successive checkpoint evaluations accumulate into one
sortable/filterable table instead of living only in stdout and one-off figures.

One row = one `evaluate()` call. The statistics columns are read straight out of the same
dict `test_nn.report()` already builds for its stdout printout (see `build_row`), so the CSV
and the printed report can never disagree about a number.

Usage (from test_nn.py):
    row = build_row(checkpoint_path, stats, val_loss=..., maps_dir=..., n_maps=..., ...)
    append_row(row, log_path)
"""
from __future__ import annotations

import csv
import datetime
import pathlib

from feasibility.comparator.provenance import git_provenance
from feasibility.grid_learning.custom_dataset import TARGET_NAMES

DEFAULT_LOG_PATH = pathlib.Path("outputs/eval_log.csv")

E_POS_IDX = TARGET_NAMES.index("e_pos")
E_ROT_IDX = TARGET_NAMES.index("e_rot")

# Column order is the CSV's public contract -- append new fields at the end so old rows
# (read back with plain csv.DictReader, or in a spreadsheet) stay aligned under their header.
FIELDNAMES = [
    "timestamp",
    "checkpoint",
    "checkpoint_epoch",
    "checkpoint_val_loss",
    "maps_dir",
    "n_maps",
    "n_commands",
    "seed",
    "n_valid",
    "n_blocked",
    "n_diverged",
    "n_flat",
    "n_near",
    "total_e_pos",
    "total_e_rot",
    "mean_e_pos",
    "mean_e_rot",
    "rmse_overall_e_pos",
    "rmse_overall_e_rot",
    "rmse_flat_e_pos",
    "rmse_flat_e_rot",
    "rmse_near_e_pos",
    "rmse_near_e_rot",
    "rmse_decile_e_pos",
    "rmse_decile_e_rot",
    "r2_e_pos",
    "r2_e_rot",
    "duration_s",
    "mu",
    "k_turn",
    "device",
    "near_obstacle_radius",
    "exclude_training_maps",
    "ostrich_sha",
    "ostrich_dirty",
    "helhest_stack_sha",
    "helhest_stack_dirty",
]


def _rmse_component(rmse, idx: int) -> float | str:
    """A single (e_pos or e_rot) entry out of a [2]-shaped RMSE tensor/None -- report() returns
    None for an RMSE over an empty subset (e.g. no near-obstacle cells drawn this run), which
    would otherwise crash float(); the CSV records that as an empty cell rather than 0.0, since
    0.0 would misleadingly read as a perfect score. Rounded to 6 decimals -- these values start
    life as float32, and repr'ing that straight to float64 (e.g. 0.10000000149011612) is noise a
    bookkeeping CSV shouldn't carry."""
    return round(float(rmse[idx]), 6) if rmse is not None else ""


def build_row(
    checkpoint_path: pathlib.Path,
    stats: dict[str, object],
    *,
    val_loss: float,
    maps_dir: pathlib.Path,
    n_maps: int,
    n_commands: int,
    seed: int,
    duration_s: float,
    mu: float,
    k_turn: float,
    device_str: str,
    near_obstacle_radius: float,
    exclude_training_maps: bool,
) -> dict[str, object]:
    """Assembles one FIELDNAMES-shaped row from the dict `test_nn.report()` returns plus the run's
    own CLI/config parameters. Kept separate from `append_row` so a caller can inspect/modify the
    row (or skip logging) before it's written."""
    prov = git_provenance()
    return {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": stats["epoch"],
        "checkpoint_val_loss": val_loss,
        "maps_dir": str(maps_dir),
        "n_maps": n_maps,
        "n_commands": n_commands,
        "seed": seed,
        "n_valid": stats["n_valid"],
        "n_blocked": stats["n_blocked"],
        "n_diverged": stats["n_diverged"],
        "n_flat": stats["n_flat"],
        "n_near": stats["n_near"],
        "total_e_pos": round(float(stats["total_pos"]), 6),
        "total_e_rot": round(float(stats["total_rot"]), 6),
        "mean_e_pos": round(float(stats["mean_pos"]), 6),
        "mean_e_rot": round(float(stats["mean_rot"]), 6),
        "rmse_overall_e_pos": _rmse_component(stats["overall_rmse"], E_POS_IDX),
        "rmse_overall_e_rot": _rmse_component(stats["overall_rmse"], E_ROT_IDX),
        "rmse_flat_e_pos": _rmse_component(stats["flat_rmse"], E_POS_IDX),
        "rmse_flat_e_rot": _rmse_component(stats["flat_rmse"], E_ROT_IDX),
        "rmse_near_e_pos": _rmse_component(stats["near_rmse"], E_POS_IDX),
        "rmse_near_e_rot": _rmse_component(stats["near_rmse"], E_ROT_IDX),
        "rmse_decile_e_pos": _rmse_component(stats["decile_rmse"], E_POS_IDX),
        "rmse_decile_e_rot": _rmse_component(stats["decile_rmse"], E_ROT_IDX),
        "r2_e_pos": _rmse_component(stats["r2"], E_POS_IDX),
        "r2_e_rot": _rmse_component(stats["r2"], E_ROT_IDX),
        "duration_s": duration_s,
        "mu": mu,
        "k_turn": k_turn,
        "device": device_str,
        "near_obstacle_radius": near_obstacle_radius,
        "exclude_training_maps": exclude_training_maps,
        "ostrich_sha": prov["ostrich_sha"],
        "ostrich_dirty": prov["ostrich_dirty"],
        "helhest_stack_sha": prov["helhest_stack_sha"],
        "helhest_stack_dirty": prov["helhest_stack_dirty"],
    }


def append_row(row: dict[str, object], log_path: pathlib.Path = DEFAULT_LOG_PATH) -> None:
    """Appends `row` to `log_path`, writing the FIELDNAMES header first if the file doesn't exist
    yet. Uses `csv.DictWriter`'s `restval`/extrasaction defaults so a row missing a key (an older
    caller, a field added later) still writes -- missing cells come out empty rather than raising."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    with log_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, restval="", extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"[eval-log] appended row to {log_path}")
