"""Viewer for one sample's baked-in body-frame terrain patch, plotted as a 2D heightmap
(color = dz/r_wheel) alongside its commanded curvature/twist and the ostrich-vs-reference
final-pose error (e_pos, e_rot) -- not stored in the file, recomputed on the fly.

Auto-detects (override with --dataset) which of two sibling generators wrote the file, since
each bakes in a different `PatchSpec` and a different error reference -- both trees are
deliberately independent (design.md section 11a in each), so this viewer is the one place that
knows about both:

* "patch" -- learning/generate_dataset_body_centered_patch.py: commanded (v_drive, wz_drive) +
  variant_label, error against an `hstack/` group's final pose
  (learning.pose_error.final_pose_error).
* "arc"   -- lattice_learning/generate_dataset.py: commanded kappa (v is pinned at v_nom, not
  stored per-row), `valid`/`swept_clear` flags, error against the arc-plus-settle `ref_pose`
  (there is no `hstack/` group -- that generator never runs the twin, see its own module
  docstring), via lattice_learning.custom_dataset.se3_errors/pose_to_se3.

A file with neither a `v_drive` nor a `kappa` dataset (e.g. generate_init_pose_dataset.py's plain
spawn-pose output, which has no baked-in patch at all) raises a clear error rather than plotting
nothing.

CLI parameters:
    --file PATH      dataset .h5 path (required)
    --id INT         sample index to display (default: 0)
    --dataset CHOICE  "auto" (default), "patch", or "arc" -- force the schema instead of
                      auto-detecting

Usage:
    python src/feasibility/plotting/dataset_patch_viewer.py --file outputs/dataset_patch_box_centered_h070cm_n128.h5
    python src/feasibility/plotting/dataset_patch_viewer.py --file outputs/dataset_patch_box_centered_h070cm_n128.h5 --id 7
    python src/feasibility/plotting/dataset_patch_viewer.py --file outputs/dataset_arc_rough_box0_M5_R16_seed0.h5 --id 3
"""
from __future__ import annotations

import argparse
import pathlib

import h5py
import matplotlib.pyplot as plt
import numpy as np

from feasibility.lattice_learning.custom_dataset import pose_to_se3
from feasibility.lattice_learning.custom_dataset import se3_errors
from feasibility.lattice_learning.patch import patch_spec_from_attrs as patch_spec_from_attrs_arc
from feasibility.learning.pose_error import final_pose_error
from feasibility.learning.terrain_patch import patch_spec_from_attrs as patch_spec_from_attrs_patch

DATASET_CHOICES = ("auto", "patch", "arc")


def detect_schema(f: h5py.File) -> str:
    """"patch" if the file has learning/generate_dataset_body_centered_patch.py's `v_drive`
    dataset, "arc" if it has lattice_learning/generate_dataset.py's `kappa` dataset instead --
    the two are mutually exclusive by construction, so this is unambiguous on any real file."""
    is_patch, is_arc = "v_drive" in f, "kappa" in f
    if is_patch and not is_arc:
        return "patch"
    if is_arc and not is_patch:
        return "arc"
    if "patch" not in f:
        raise SystemExit(
            f"{f.filename} has no `patch` dataset -- this viewer only reads files with a "
            "baked-in body-frame patch (generate_dataset_body_centered_patch.py's or "
            "lattice_learning/generate_dataset.py's output)"
        )
    raise SystemExit(
        f"{f.filename}: can't auto-detect the schema (has both/neither `v_drive` and `kappa`) "
        "-- pass --dataset patch|arc explicitly"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=pathlib.Path, required=True, help="dataset .h5 path")
    ap.add_argument("--id", type=int, default=0, help="sample index to display (default: 0)")
    ap.add_argument(
        "--dataset", choices=DATASET_CHOICES, default="auto",
        help="force the schema instead of auto-detecting (default: auto)",
    )
    args = ap.parse_args()

    # Materialize everything to numpy inside the `with` -- h5py datasets are invalid once the
    # file closes, and plt.show() (a blocking call) must run outside it.
    with h5py.File(args.file, "r") as f:
        schema = detect_schema(f) if args.dataset == "auto" else args.dataset
        n = int(f.attrs["n"])
        if not (0 <= args.id < n):
            raise SystemExit(f"--id must be in [0, {n}), got {args.id}")

        if schema == "patch":
            spec = patch_spec_from_attrs_patch(dict(f.attrs))
            patch = f["patch"][args.id].reshape(spec.ny, spec.nx)
            v = float(f["v_drive"][args.id])
            wz = float(f["wz_drive"][args.id])
            label = f["variant_label"].asstr()[args.id]
        else:
            spec = patch_spec_from_attrs_arc(dict(f.attrs))
            patch = f["patch"][args.id].reshape(spec.ny, spec.nx)
            v_nom = float(f.attrs["v_nom"])
            kappa = float(f["kappa"][args.id])
            valid = bool(f["valid"][args.id])
            swept_clear = bool(f["swept_clear"][args.id])
            ref_pose = f["ref_pose"][args.id].astype(np.float64)
            ostrich_final = f["ostrich/pose"][-1, args.id].astype(np.float64)

    if schema == "patch":
        pos_error, rot_error = final_pose_error(args.file, args.id)
        title = (
            f"sample {args.id}/{n - 1}: {label}\n"
            f"v={v:.3f} m/s, wz={wz:.3f} rad/s   |   "
            f"e_pos={pos_error:.4f} m, e_rot={np.degrees(rot_error):.2f} deg"
        )
    else:
        pos_error, rot_error = se3_errors(pose_to_se3(ref_pose), pose_to_se3(ostrich_final))
        pos_error, rot_error = float(pos_error), float(rot_error)
        flags = "valid" if valid else "INVALID"
        flags += ", swept-clear" if swept_clear else ""
        title = (
            f"sample {args.id}/{n - 1}: {flags}\n"
            f"v={v_nom:.3f} m/s (pinned), kappa={kappa:.3f} 1/m   |   "
            f"e_pos={pos_error:.4f} m, e_rot={np.degrees(rot_error):.2f} deg"
        )

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
    ax.set_title(title)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
