"""Appends one row per `test_nn.py` run to a persistent CSV evaluation log (default
`outputs/eval_log2.csv`), so successive checkpoint evaluations accumulate into one
sortable/filterable table instead of living only in stdout and one-off figures.

One row = one `evaluate()` call. The statistics columns are read straight out of the same dict
`test_nn.report()` already builds for its stdout printout (see `build_row`), so the CSV and the
printed report can never disagree about a number.

**Why this is a separate file from grid_learning/eval_log.py.** Not only design.md section 11's
dependency stance -- the schemas genuinely differ, in three places:

  * `n_commands` -> `rows_per_map`, plus `wz_zero_frac`/`wz_grid`. In v1 the eval command set is
    fixed (`linspace` over WZ_RANGE, one scalar per row), so the run is fully described by how many
    of them there were. In v2 the commands are a per-cell FIELD drawn from a distribution the eval
    run chooses, so that distribution has to be logged or two rows are not comparable.
  * The per-|wz| bin columns, which v1 has no way to fill: with one command per row it would need
    one simulated row per bin, where v2 populates every bin from a single map.
  * `checkpoint_head_fusion` / `checkpoint_base_width`. v2 has design.md section 5d's
    `--head-fusion {film,concat}` ablation, and a log that cannot tell a FiLM run from a concat run
    is useless for filling in section 8's baseline table.

Writing v2 rows into v1's `outputs/eval_log.csv` would therefore misalign them under its header,
which is why `test_nn.py` defaults to a different filename rather than sharing one.

Usage (from test_nn.py):
    row = build_row(checkpoint_path, ckpt, stats, maps_dir=..., n_maps=..., rows_per_map=..., ...)
    append_row(row, log_path)

    python src/feasibility/grid_learning_2/eval_log.py    # this module's smoke test
"""
from __future__ import annotations

import csv
import datetime
import pathlib
import tempfile

from feasibility.comparator.provenance import git_provenance
from feasibility.grid_learning_2.model import TARGET_NAMES

DEFAULT_LOG_PATH = pathlib.Path("outputs/eval_log2.csv")  # NOT v1's eval_log.csv -- see the
# module docstring: the two schemas differ, so sharing one file would misalign the header

E_POS_IDX = TARGET_NAMES.index("e_pos")
E_ROT_IDX = TARGET_NAMES.index("e_rot")

# How many per-|wz| bins get their own pair of columns. FIELDNAMES is a fixed contract (see below),
# so this cannot follow `+wz_bins` at runtime; it matches train.DEFAULT_WZ_BINS, the only value
# anything defaults to. A run with more bins than this logs the first N here and records the real
# count in `wz_bins` / `wz_bin_edges`, so a truncated row is self-describing rather than silently
# wrong -- build_row warns when it happens.
N_WZ_BIN_COLUMNS = 8

# Column order is the CSV's public contract -- append new fields at the end so old rows (read back
# with plain csv.DictReader, or in a spreadsheet) stay aligned under their header.
FIELDNAMES = [
    "timestamp",
    "checkpoint",
    "checkpoint_epoch",
    "checkpoint_val_loss",
    "checkpoint_head_fusion",
    "checkpoint_base_width",
    "maps_dir",
    "n_maps",
    "rows_per_map",
    "seed",
    "wz_zero_frac",
    "wz_grid",
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
    "wz_bins",
    "wz_bin_edges",
    *[
        f"rmse_wz_bin{b}_{name}"
        for b in range(N_WZ_BIN_COLUMNS)
        for name in TARGET_NAMES
    ],
    *[f"n_wz_bin{b}" for b in range(N_WZ_BIN_COLUMNS)],
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
    None for an RMSE over an empty subset (e.g. no near-obstacle cells drawn this run), which would
    otherwise crash float(); the CSV records that as an empty cell rather than 0.0, since 0.0 would
    misleadingly read as a perfect score. Rounded to 6 decimals -- these values start life as
    float32, and repr'ing that straight to float64 (e.g. 0.10000000149011612) is noise a
    bookkeeping CSV shouldn't carry."""
    return round(float(rmse[idx]), 6) if rmse is not None else ""


def _bin_columns(stats: dict[str, object]) -> dict[str, object]:
    """The per-|wz| bin half of a row: `rmse_wz_bin{b}_{head}` and `n_wz_bin{b}` for the first
    N_WZ_BIN_COLUMNS bins, plus the run's real bin count and edges.

    `stats["wz_bin_rmse"]` is a list of per-bin [K] RMSE sequences with None for a bin no valid
    cell landed in (test_nn.wz_bin_rmse's own convention), so an empty bin comes out as an empty
    cell here for the same reason _rmse_component does it -- 0.0 would read as a perfect score on
    a bin that was never scored at all."""
    # Explicit `is None` rather than `or []`: report() hands the counts and edges over as numpy
    # arrays, and `array or default` raises ("truth value of an array is ambiguous") instead of
    # falling through the way it does for a list.
    def _as_list(key: str) -> list:
        value = stats.get(key)
        return [] if value is None else list(value)

    per_bin = _as_list("wz_bin_rmse")
    counts = _as_list("wz_bin_counts")
    edges = _as_list("wz_bin_edges")

    if len(per_bin) > N_WZ_BIN_COLUMNS:
        print(
            f"[eval-log] WARNING: this run used {len(per_bin)} |wz| bins but the CSV has columns "
            f"for {N_WZ_BIN_COLUMNS}; logging the first {N_WZ_BIN_COLUMNS}. The `wz_bins` and "
            f"`wz_bin_edges` columns record the real binning, and the stdout report printed every "
            f"bin -- lower +wz_bins to {N_WZ_BIN_COLUMNS} if you want them all in the CSV."
        )

    row: dict[str, object] = {
        "wz_bins": len(per_bin),
        # One string column rather than n_bins+1 numeric ones: the edge count follows +wz_bins,
        # which FIELDNAMES cannot, and this keeps a row readable on its own terms.
        "wz_bin_edges": ";".join(f"{e:.4f}" for e in edges),
    }
    for b in range(min(len(per_bin), N_WZ_BIN_COLUMNS)):
        row[f"rmse_wz_bin{b}_{TARGET_NAMES[E_POS_IDX]}"] = _rmse_component(per_bin[b], E_POS_IDX)
        row[f"rmse_wz_bin{b}_{TARGET_NAMES[E_ROT_IDX]}"] = _rmse_component(per_bin[b], E_ROT_IDX)
        row[f"n_wz_bin{b}"] = int(counts[b]) if b < len(counts) else ""
    return row


def build_row(
    checkpoint_path: pathlib.Path,
    ckpt: dict[str, object],
    stats: dict[str, object],
    *,
    maps_dir: pathlib.Path,
    n_maps: int,
    rows_per_map: int,
    seed: int,
    wz_zero_frac: float,
    wz_grid: int | None,
    duration_s: float,
    mu: float,
    k_turn: float,
    device_str: str,
    near_obstacle_radius: float,
    exclude_training_maps: bool,
) -> dict[str, object]:
    """Assembles one FIELDNAMES-shaped row from the dict `test_nn.report()` returns, the
    checkpoint's own self-description, and the run's CLI/config parameters. Kept separate from
    `append_row` so a caller can inspect/modify the row (or skip logging) before it's written.

    Takes the whole `ckpt` rather than a few unpacked fields: v2's build_checkpoint stores the
    architecture (`head_fusion`, `base_width`) alongside the training result, and those are exactly
    what distinguishes two rows of design.md section 8's ablation table from each other."""
    prov = git_provenance()
    return {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": ckpt["epoch"],
        "checkpoint_val_loss": round(float(ckpt["val_loss"]), 6),
        "checkpoint_head_fusion": ckpt.get("head_fusion", ""),
        "checkpoint_base_width": ckpt.get("base_width", ""),
        "maps_dir": str(maps_dir),
        "n_maps": n_maps,
        "rows_per_map": rows_per_map,
        "seed": seed,
        "wz_zero_frac": wz_zero_frac,
        "wz_grid": 0 if wz_grid is None else wz_grid,  # 0 = continuous, the same encoding
        # generate_dataset.py writes into a file's own `wz_grid` root attr
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
        **_bin_columns(stats),
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


if __name__ == "__main__":
    # This module's smoke test (there is no pytest suite in this tree): a synthetic stats dict must
    # round-trip through build_row/append_row into a well-formed CSV, appending rather than
    # rewriting, with an unscored subset landing as an EMPTY cell rather than a flattering 0.0.
    #
    # The bin counts/edges are numpy arrays here, NOT lists, because that is what test_nn.report()
    # actually hands over -- an earlier version of _bin_columns used `stats.get(k) or []`, which is
    # fine for a list and raises "truth value of an array is ambiguous" for an array, so the shapes
    # this test uses have to match the real caller's or it tests nothing.
    import numpy as np

    stats = dict(
        overall_rmse=[0.0897, 0.1595],
        flat_rmse=[0.0412, 0.0988],
        near_rmse=None,  # the empty-subset case -- must not become 0.0
        decile_rmse=[0.2718, 0.4591],
        r2=[0.8340, 0.8337],
        wz_bin_rmse=[[0.05, 0.11], None, [0.09, 0.17], [0.13, 0.22]],
        wz_bin_counts=np.array([220, 0, 198, 176]),
        wz_bin_edges=np.linspace(0.0, 1.0, 5),
        total_pos=53.1, total_rot=94.4, mean_pos=0.0641, mean_rot=0.1139,
        n_valid=828, n_flat=602, n_near=226, n_blocked=97, n_diverged=0,
    )
    ckpt = dict(epoch=70, val_loss=0.11397, head_fusion="film", base_width=32)

    row = build_row(
        pathlib.Path("outputs/checkpoints/example.pt"), ckpt, stats,
        maps_dir=pathlib.Path("assets/large_box_random/0"), n_maps=5, rows_per_map=5, seed=0,
        wz_zero_frac=0.05, wz_grid=None, duration_s=2.4, mu=0.8, k_turn=1.0,
        device_str="cuda:0", near_obstacle_radius=1.5, exclude_training_maps=True,
    )
    assert set(row) <= set(FIELDNAMES), sorted(set(row) - set(FIELDNAMES))
    assert row["rmse_near_e_pos"] == "" and row["rmse_wz_bin1_e_pos"] == "", (
        "an unscored subset must log as an empty cell, never 0.0"
    )
    assert row["rmse_wz_bin0_e_pos"] == 0.05 and row["n_wz_bin0"] == 220
    assert row["wz_bins"] == 4 and row["wz_bin_edges"].startswith("0.0000;0.2500")
    assert row["checkpoint_head_fusion"] == "film", "the ablation arm must be logged"
    print(f"[row] {len(FIELDNAMES)} columns, {len(row)} filled, empty subsets left blank")

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "eval_log2.csv"
        append_row(row, path)
        append_row(row, path)  # the header must be written once, not once per row
        with path.open(newline="") as f:
            read_back = list(csv.DictReader(f))
        assert len(read_back) == 2, f"expected 2 rows, got {len(read_back)}"
        assert list(read_back[0]) == FIELDNAMES, "header drifted from FIELDNAMES"
        assert read_back[0]["rmse_overall_e_pos"] == "0.0897"
        assert read_back[0]["rmse_near_e_pos"] == ""
        print(f"[csv] {len(read_back)} rows under one header, every FIELDNAMES key present")

        # A row from an older/newer caller (missing keys, extra keys) must still write -- that is
        # what restval/extrasaction buy, and the reason a schema change here is not a breaking one.
        append_row({"timestamp": "now", "not_a_column": 1}, path)
        with path.open(newline="") as f:
            assert len(list(csv.DictReader(f))) == 3
        print("[csv] a row with missing/extra keys still writes (restval/extrasaction)")

    print("all self-checks ok")
