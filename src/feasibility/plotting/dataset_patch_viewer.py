"""Viewer for a generate_dataset_body_centered_patch.py output: pick one sample by --id and plot
its baked-in body-frame terrain patch (learning.terrain_patch.sample_patches()) as a 2D heightmap,
color = dz/r_wheel, alongside the commanded (v, wz) twist and the ostrich-vs-hstack final-pose
error (e_pos, e_rot) -- not stored in the file, recomputed on the fly via
learning.pose_error.final_pose_error(), a single existing call, not worth skipping.

Body-centered-patch files only (see generate_dataset_body_centered_patch.py) -- a file with no
`patch` dataset (e.g. generate_init_pose_dataset.py's plain spawn-pose output) raises a clear
error rather than plotting nothing.

CLI parameters:
    --file PATH   dataset_patch_*.h5 path (required)
    --id INT      sample index to display (default: 0)

Usage:
    python src/feasibility/plotting/dataset_patch_viewer.py --file outputs/dataset_patch_box_centered_h070cm_n128.h5
    python src/feasibility/plotting/dataset_patch_viewer.py --file outputs/dataset_patch_box_centered_h070cm_n128.h5 --id 7
"""
from __future__ import annotations

import argparse
import pathlib

import h5py
import matplotlib.pyplot as plt
import numpy as np

from feasibility.learning.pose_error import final_pose_error
from feasibility.learning.terrain_patch import patch_spec_from_attrs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=pathlib.Path, required=True, help="dataset_patch_*.h5 path")
    ap.add_argument("--id", type=int, default=0, help="sample index to display (default: 0)")
    args = ap.parse_args()

    # Materialize everything to numpy inside the `with` -- h5py datasets are invalid once the
    # file closes, and plt.show() (a blocking call) must run outside it.
    with h5py.File(args.file, "r") as f:
        n = int(f.attrs["n"])
        if not (0 <= args.id < n):
            raise SystemExit(f"--id must be in [0, {n}), got {args.id}")
        if "patch" not in f:
            raise SystemExit(
                f"{args.file} has no `patch` dataset -- this viewer only reads files written by "
                f"generate_dataset_body_centered_patch.py (generate_init_pose_dataset.py's plain "
                f"spawn-pose files have no baked-in patch to show)"
            )
        spec = patch_spec_from_attrs(dict(f.attrs))
        patch = f["patch"][args.id].reshape(spec.ny, spec.nx)
        v = float(f["v_drive"][args.id])
        wz = float(f["wz_drive"][args.id])
        label = f["variant_label"].asstr()[args.id]

    pos_error, rot_error = final_pose_error(args.file, args.id)

    fig, ax = plt.subplots(figsize=(7, 6))
    # origin="lower": row 0 of `patch` is the most-negative-Y row (sample_patches' [ny, nx]
    # layout, row = body +Y), so drawing it at the bottom needs no flip, unlike terrain_patch.py's
    # text-mode _ascii() which prints top-down and does need one.
    im = ax.imshow(
        patch, origin="lower", cmap="terrain", aspect="equal",
        extent=(spec.x_min, spec.x_max, spec.y_min, spec.y_max),
    )
    fig.colorbar(im, ax=ax, label="dz / r_wheel")
    ax.set_xlabel("body +X (forward) [m]")
    ax.set_ylabel("body +Y (left) [m]")
    ax.set_title(
        f"sample {args.id}/{n - 1}: {label}\n"
        f"v={v:.3f} m/s, wz={wz:.3f} rad/s   |   "
        f"e_pos={pos_error:.4f} m, e_rot={np.degrees(rot_error):.2f} deg"
    )

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
