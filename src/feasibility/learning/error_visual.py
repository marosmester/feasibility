"""Visualize per-sample pose error (e_pos, e_rot) from a generated dataset HDF5, loaded through
custom_dataset.PoseErrorDataset (see feasibility.learning.generate_dataset for how the file is
produced, custom_dataset.py for the schema).

Top subplot: the first --num-samples rows' e_pos (m) and e_rot (rad) against sample index,
e_pos on the left y-axis and e_rot on the right (different units, different colors) since a
single shared axis would flatten one series. Bottom subplot: a table with one column per
plotted sample (aligned to the same sample index on the x-axis above) and one row per input
feature -- x, y, yaw, v, omega -- so a spike in the error plot can be read off against the
commanded twist/spawn pose that produced it.

CLI parameters:
    --file PATH          dataset .h5 (custom_dataset.PoseErrorDataset schema)
                          (default: outputs/dataset_box_h070cm_n128.h5)
    --num-samples INT    M: number of leading samples to plot (default: 64)
    --out PATH           save to this path instead of showing interactively (default: show)

Usage:
    python src/feasibility/learning/error_visual.py --file outputs/dataset_box_h070cm_n128.h5
    python src/feasibility/learning/error_visual.py --file outputs/dataset_box_h070cm_n128.h5 --num-samples 40
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt

from feasibility.learning.custom_dataset import PoseErrorDataset

E_POS_COLOR = "#1f77b4"
E_ROT_COLOR = "#d62728"

# (row label, index into ds.x's raw feature columns (v, wz, x, y, yaw))
TABLE_ROWS = (("x", 2), ("y", 3), ("yaw", 4), ("v", 0), ("omega", 1))


def plot_errors(ds: PoseErrorDataset, num_samples: int, out_path: pathlib.Path | None) -> None:
    if ds.yaw_encoding != "raw":
        raise ValueError(f"error_visual expects yaw_encoding='raw', got {ds.yaw_encoding!r}")

    n = min(num_samples, len(ds))
    e_pos = ds.y[:n, 0].numpy()
    e_rot = ds.y[:n, 1].numpy()
    x = ds.x[:n].numpy()
    idx = range(n)

    fig, (ax_pos, ax_table) = plt.subplots(
        2, 1, figsize=(max(12, 0.3 * n), 8), gridspec_kw={"height_ratios": [2, 1]}
    )

    ax_pos.plot(idx, e_pos, color=E_POS_COLOR, marker="o", markersize=3, linewidth=1, label="e_pos (m)")
    ax_pos.set_xlabel("sample index")
    ax_pos.set_ylabel("e_pos (m)", color=E_POS_COLOR)
    ax_pos.tick_params(axis="y", labelcolor=E_POS_COLOR)
    ax_pos.set_xlim(-0.5, n - 0.5)

    ax_rot = ax_pos.twinx()
    ax_rot.plot(idx, e_rot, color=E_ROT_COLOR, marker="^", markersize=3, linewidth=1, label="e_rot (rad)")
    ax_rot.set_ylabel("e_rot (rad)", color=E_ROT_COLOR)
    ax_rot.tick_params(axis="y", labelcolor=E_ROT_COLOR)

    lines = ax_pos.get_lines() + ax_rot.get_lines()
    ax_pos.legend(lines, [line.get_label() for line in lines], loc="upper right")
    ax_pos.set_title(f"{ds.source.name}: first {n}/{len(ds)} samples")

    ax_table.axis("off")
    label_width = 0.05
    value_width = (1 - label_width) / n
    cell_text = [
        [label] + [f"{x[i, col]:.2f}" for i in range(n)] for label, col in TABLE_ROWS
    ]
    table = ax_table.table(
        cellText=cell_text,
        colLabels=[""] + [str(i) for i in idx],
        colWidths=[label_width] + [value_width] * n,
        cellLoc="center",
        loc="upper center",
    )
    table.auto_set_font_size(True)
    table.scale(1, 1.3)

    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=150)
        print(f"saved to {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=pathlib.Path, default=pathlib.Path("outputs/dataset_box_h070cm_n128.h5"), help="dataset .h5 (custom_dataset.PoseErrorDataset schema)")
    parser.add_argument("--num-samples", type=int, default=64, help="M: number of leading samples to plot (default: 64)")
    parser.add_argument("--out", type=pathlib.Path, default=None, help="save to this path instead of showing interactively")
    args = parser.parse_args()

    dataset = PoseErrorDataset(args.file)
    print(f"loaded {len(dataset)} samples from {args.file}")
    plot_errors(dataset, args.num_samples, args.out)
