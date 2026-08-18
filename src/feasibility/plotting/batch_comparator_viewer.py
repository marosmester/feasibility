"""Viewer for a comparator/compare_*.py output (e.g. compare_speed_bumps.py,
compare_box_obstacles.py): pick one ostrich/helhest_stack trajectory pair by variant id and plot
it -- one 3D subplot (terrain heightmap + both (x,y,z) trajectories), one 2D subplot (the
wheel-velocity commands both sims were driven with, vs time).

Usage:
    python -m feasibility.plotting.batch_comparator_viewer            # variant 0
    python -m feasibility.plotting.batch_comparator_viewer --id 3
    python -m feasibility.plotting.batch_comparator_viewer --npz outputs/compare_box_obstacles.npz --id 3
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the '3d' projection)

from feasibility.comparator.provenance import terrain_from_npz
from feasibility.heightmap import HeightMapReader

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_NPZ = REPO_ROOT / "outputs" / "compare_speed_bumps.npz"

WHEEL_NAMES = ("left", "right", "rear")
WHEEL_COLORS = ("tab:red", "tab:green", "tab:blue")
# mplot3d has no GPU path or blitting: every drag frame re-sorts and re-rasterizes ALL faces in
# software
TERRAIN_MAX_CELLS = 40


def plot_terrain(ax, hmap: HeightMapReader) -> None:
    xs = hmap.x0 + (np.arange(hmap.nx) + 0.5) * hmap.cell
    ys = hmap.y0 + (np.arange(hmap.ny) + 0.5) * hmap.cell
    X, Y = np.meshgrid(xs, ys)  # [ny, nx], matching hmap.H's [i, j] cell-center convention
    s = max(1, (max(hmap.ny, hmap.nx) - 1) // TERRAIN_MAX_CELLS)
    ax.plot_surface(
        X[::s, ::s], Y[::s, ::s], hmap.H[::s, ::s],
        cmap="terrain", rstride=1, cstride=1,
        antialiased=False, alpha=1.0, linewidth=0,
        zorder=0,  # see ax.computed_zorder = False in main(): draw below the trajectories always
    )


def plot_trajectories(ax, ostrich_pose: np.ndarray, hstack_pose: np.ndarray) -> None:
    """pose is [T, 7] = (x, y, z, qx, qy, qz, qw); only the (x, y, z) columns are plotted."""
    ax.plot(*ostrich_pose[:, :3].T, color="tab:orange", linewidth=2, label="ostrich", zorder=10)
    ax.plot(*hstack_pose[:, :3].T, color="tab:purple", linewidth=2, label="hstack", zorder=10)
    ax.scatter(*ostrich_pose[0, :3], color="k", marker="o", s=30, zorder=11, label="start")


def plot_commanded_velocity(
    ax, ostrich_t: np.ndarray, ostrich_cmd: np.ndarray, hstack_t: np.ndarray, hstack_cmd: np.ndarray
) -> None:
    """cmd is [T, 3] per-wheel [left, right, rear] rad/s, piecewise-constant -> step plot.
    Both sims were built from the same command schedule at their own dt, so the two step
    curves should coincide (just resampled)."""
    for c, (name, color) in enumerate(zip(WHEEL_NAMES, WHEEL_COLORS)):
        ax.step(ostrich_t, ostrich_cmd[:, c], where="post", color=color, linestyle="-", label=f"{name} (ostrich)")
        ax.step(hstack_t, hstack_cmd[:, c], where="post", color=color, linestyle="--", label=f"{name} (hstack)")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("commanded wheel speed [rad/s]")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(alpha=0.3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--npz", type=pathlib.Path, default=DEFAULT_NPZ, help=f"compare_*.npz path (default {DEFAULT_NPZ})"
    )
    ap.add_argument("--id", type=int, default=0, help="variant index to display (default 0)")
    args = ap.parse_args()

    d = np.load(args.npz)
    n = int(d["n"])
    if not (0 <= args.id < n):
        raise SystemExit(f"--id must be in [0, {n}), got {args.id}")

    label = str(d["variant_label"][args.id])
    hmap = terrain_from_npz(d, args.id)

    fig = plt.figure(figsize=(13, 6))
    fig.suptitle(f"compare variant {args.id}/{n - 1}: {label}")

    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    # mplot3d's default per-redraw depth heuristic (computed_zorder=True) re-sorts artists by
    # projected depth every frame; from a near-top-down view that's degenerate (surface and
    # trajectory project to nearly the same depth) and it can flip which one wins, hiding the
    # line under the opaque surface depending on viewing angle. Disable it and use the explicit
    # zorder set in plot_terrain/plot_trajectories instead -- we already know the trajectories
    # always sit above the terrain, so a static order is both correct and cheaper.
    ax3d.computed_zorder = False
    plot_terrain(ax3d, hmap)
    plot_trajectories(ax3d, d["ostrich_pose"][:, args.id, :], d["hstack_pose"][:, args.id, :])
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    ax3d.legend()

    ax_cmd = fig.add_subplot(1, 2, 2)
    plot_commanded_velocity(
        ax_cmd,
        d["ostrich_t"], d["ostrich_cmd_wheel_omega"][:, args.id, :],
        d["hstack_t"], d["hstack_cmd_wheel_omega"][:, args.id, :],
    )

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
