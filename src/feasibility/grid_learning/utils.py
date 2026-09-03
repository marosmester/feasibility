"""Elevation heightmap -> fixed-resolution torch tensor, for feeding a saved
assets/box_random (or any HeightMapReader-loadable) terrain into an NN/CNN.

HeightMapReader stores elevation at its OWN native `cell` resolution (0.05 m for the
assets/ series) -- too fine and too irregularly sized across files to hand a model
directly. This module resamples (bilinear, via HeightMapReader.sample, which clamps
rather than raises outside the source grid) onto a fixed `resolution`-m grid spanning
an `extent` x `extent` square centered on the world origin, e.g. the defaults
(0.10 m, 10.0 m) give a 100x100 tensor -- matching create_box_obstacles.py's default
10x10 m grid (DEFAULT_CENTERED_EXTENT), so an assets/box_random map needs no cropping.

Same cell-center convention as HeightMapReader.H and learning.terrain_patch.PatchSpec:
grid point (i, j) sits at world (-extent/2 + (j+0.5)*resolution, -extent/2 +
(i+0.5)*resolution), row index along +Y, column along +X.

CLI parameters:
    path             heightmap path (with or without .png/.yaml), e.g.
                      assets/box_random/box_random_i0000_h070cm
    --resolution      tensor cell size in meters (default: 0.10)
    --extent          square extent to sample in meters (default: 10.0)

Usage:
    python src/feasibility/grid_learning/utils.py assets/box_random/box_random_i0000_h070cm
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch

from feasibility.heightmap import HeightMapReader

DEFAULT_RESOLUTION = 0.10  # m
DEFAULT_EXTENT = 10.0  # m -- matches create_box_obstacles.DEFAULT_CENTERED_EXTENT


def grid_coords(
    resolution: float = DEFAULT_RESOLUTION, extent: float = DEFAULT_EXTENT
) -> np.ndarray:
    """[n] world coordinate of each cell center along one axis of a `resolution`-m grid
    spanning `extent` m centered on 0 -- shared by both axes since the grid is square."""
    if resolution <= 0.0:
        raise ValueError(f"resolution must be > 0, got {resolution}")
    if extent <= 0.0:
        raise ValueError(f"extent must be > 0, got {extent}")
    if abs(extent / resolution - round(extent / resolution)) > 1e-6:
        raise ValueError(f"extent {extent} m is not an integer number of {resolution} m cells")
    n = int(round(extent / resolution))
    return -extent / 2.0 + (np.arange(n) + 0.5) * resolution


def heightmap_to_tensor(
    terrain: HeightMapReader,
    resolution: float = DEFAULT_RESOLUTION,
    extent: float = DEFAULT_EXTENT,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """[ny, nx] float32 tensor of absolute world z, resampled from `terrain` onto a
    `resolution`-m grid spanning `extent` x `extent` (see module docstring for the
    convention). Ready to feed a CNN as a single-channel input after `.unsqueeze(0)`.
    """
    coords = grid_coords(resolution, extent)
    X, Y = np.meshgrid(coords, coords)  # [ny, nx]
    Z = np.asarray(terrain.sample(X, Y), dtype=np.float32)
    return torch.from_numpy(Z).to(device)


def load_heightmap_tensor(
    path: str | pathlib.Path,
    resolution: float = DEFAULT_RESOLUTION,
    extent: float = DEFAULT_EXTENT,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """heightmap_to_tensor() straight from a saved <path>.png/.yaml pair."""
    terrain = HeightMapReader.load(path)
    return heightmap_to_tensor(terrain, resolution=resolution, extent=extent, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("path", type=str, help="heightmap path, e.g. assets/box_random/box_random_i0000_h070cm")
    parser.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION, help="tensor cell size (m)")
    parser.add_argument("--extent", type=float, default=DEFAULT_EXTENT, help="square extent to sample (m)")
    args = parser.parse_args()

    tensor = load_heightmap_tensor(args.path, resolution=args.resolution, extent=args.extent)
    print(
        f"{tensor.shape[0]}x{tensor.shape[1]} tensor, resolution {args.resolution} m, "
        f"extent {args.extent} m, z in [{tensor.min():.3f}, {tensor.max():.3f}]"
    )

    flat = HeightMapReader.flat(xlim=(-8.0, 8.0), ylim=(-8.0, 8.0))
    flat_tensor = heightmap_to_tensor(flat, resolution=0.1, extent=10.0)
    assert flat_tensor.shape == (100, 100), flat_tensor.shape
    assert torch.allclose(flat_tensor, torch.zeros_like(flat_tensor)), "flat ground must give an all-zero tensor"
    print("flat-ground self-check ok")
