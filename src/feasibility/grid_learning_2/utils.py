"""Elevation heightmap -> fixed-resolution torch tensor, on the ODD, ORIGIN-CENTRED grid
design.md section 3 is built around, plus the grid arithmetic that makes the network's readout an
integer crop instead of an interpolation.

This is grid_learning/utils.py's counterpart, and the difference is the whole point of v2. v1
samples an EVEN 100 x 100 grid at 0.10 m, whose cell centres sit at +-4.95, +-4.85, ... -- no
pixel centre on the world origin, and the 0.5 m spawn lattice therefore lands half a pixel off
forever, which is why v1's model needs `F.grid_sample` and a `readout_offset()` correction. Here
the grid is 81 x 81 at 0.125 m with centres at `0.125 * (i - 40)`, i.e. cell 40 IS the origin,
and the trunk's total stride (4) is exactly the lattice pitch in pixels (0.5 / 0.125), so the
label lattice is an integer centre-crop of the feature map. See design.md sections 3b-3d for why
0.125 m and not 0.10 m (pitch 4 = 2x2 ordinary stride-2 stages, versus a prime 5).

Same cell-center convention as HeightMapReader.H, learning.terrain_patch.PatchSpec and
grid_learning.utils: row index runs along +Y, column along +X, so a label cell (i, j) and a
heightmap pixel (i, j) overlay directly (different resolutions, not different conventions).

The grid's outer half-cell reaches 0.0625 m past the 10 x 10 m assets maps (81 * 0.125 = 10.125 m
of edge-to-edge extent). `HeightMapReader.sample` clamps outside its source grid rather than
raising, which is the same boundary condition the trunk's `replicate` padding uses -- so that
overhang is consistent with the rest of the stack, not a silent fabrication of new terrain.

Deliberately independent of feasibility.grid_learning (design.md section 11): the grid convention
genuinely differs, and a shared utils would have to serve both. SPAWN_STEP/SPAWN_LIMIT are
restated here rather than imported, same "restate rather than import" stance -- they live in
utils rather than in generate_dataset.py (where v1 keeps them) because in v2 the lattice geometry
is a CONTRACT between the dataset and the network's crop, checked by model.py, so both sides have
to read it from one place.

CLI parameters:
    path             heightmap path (with or without .png/.yaml), e.g.
                      assets/large_box_random/0/large_box_i0000_h070cm
    --resolution      tensor cell size in meters (default: 0.125)
    --cells           odd number of cells per side (default: 81)

Usage:
    python src/feasibility/grid_learning_2/utils.py assets/box_random/box_random_i0000_h070cm
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch

from feasibility.heightmap import HeightMapReader

DEFAULT_RESOLUTION = 0.125  # m -- design.md section 3d
DEFAULT_N_CELLS = 81  # odd, so a cell centre sits exactly on the world origin
DEFAULT_EXTENT = DEFAULT_N_CELLS * DEFAULT_RESOLUTION  # 10.125 m, edge to edge

SPAWN_STEP = 0.5  # m, (x, y) spawn-lattice pitch -- 4 input pixels at DEFAULT_RESOLUTION
SPAWN_LIMIT = 3.5  # m, |x|, |y| <= this -> a 15 x 15 lattice (1.5 m of padding on a 10 m map)


def grid_coords_centered(
    n_cells: int = DEFAULT_N_CELLS, resolution: float = DEFAULT_RESOLUTION
) -> np.ndarray:
    """[n_cells] world coordinate of each cell centre along one axis: `resolution * (i - c)` with
    `c = (n_cells - 1) / 2`, shared by both axes since the grid is square.

    `n_cells` must be ODD. That is not a stylistic constraint: an even count puts the origin
    between two pixels, and every strided conv then inherits a half-pixel offset that has to be
    undone at the readout (v1's `readout_offset()`). With an odd count the centre index is an
    integer, a stride-2 `kernel=3, padding=1` conv maps output index k to input index 2k, and so
    index 0 -> index 0: the origin is a fixed point of every downsample."""
    if resolution <= 0.0:
        raise ValueError(f"resolution must be > 0, got {resolution}")
    if n_cells <= 0:
        raise ValueError(f"n_cells must be > 0, got {n_cells}")
    if n_cells % 2 == 0:
        raise ValueError(
            f"n_cells must be odd so a cell centre lands on the world origin, got {n_cells} "
            "-- see this function's docstring and design.md section 3b"
        )
    return resolution * (np.arange(n_cells) - (n_cells - 1) / 2.0)


def extent_of(n_cells: int = DEFAULT_N_CELLS, resolution: float = DEFAULT_RESOLUTION) -> float:
    """Edge-to-edge extent (m) of the grid, i.e. what is written to a dataset file's
    `grid/extent`. Note this is a HALF CELL LARGER on each side than the span of the cell
    CENTRES (`resolution * (n_cells - 1)`) -- 10.125 vs 10.000 for the defaults."""
    return n_cells * resolution


def n_cells_for(extent: float, resolution: float = DEFAULT_RESOLUTION) -> int:
    """Inverse of extent_of(), for reconstructing the grid from a dataset file's stored
    `grid/extent` / `grid/resolution` instead of assuming the defaults."""
    n = extent / resolution
    if abs(n - round(n)) > 1e-6:
        raise ValueError(f"extent {extent} m is not an integer number of {resolution} m cells")
    return int(round(n))


def spawn_lattice(step: float = SPAWN_STEP, limit: float = SPAWN_LIMIT) -> np.ndarray:
    """[G, G, 2] world (x, y) of every spawn cell, G = 2*limit/step + 1 (15 for the defaults).
    Row index runs along +Y, column along +X -- the same orientation grid_coords_centered gives
    the heightmap tensor. Identical in form to grid_learning.generate_dataset.spawn_lattice; the
    label geometry is the one thing v2 does NOT change."""
    coords = np.arange(-limit, limit + 1e-9, step)
    X, Y = np.meshgrid(coords, coords)  # [G, G], row = y, col = x
    return np.stack([X, Y], axis=-1)


def conv_out_size(n: int, stride: int, kernel: int = 3, padding: int = 1) -> int:
    """Spatial size after one conv -- the standard `floor((n + 2p - k)/s) + 1`. Spelled out here
    rather than inlined in model.py so the grid arithmetic in design.md section 3c
    (81 -> 41 -> 21) is checkable without instantiating a network."""
    return (n + 2 * padding - kernel) // stride + 1


def downsampled_coords(coords: np.ndarray, total_stride: int, n_feat: int) -> np.ndarray:
    """[n_feat] world coordinates of a feature lattice produced from `coords` by convs with the
    given total stride: feature cell k draws on input cell `total_stride * k`, so its world
    coordinate is simply `coords[total_stride * k]`.

    That identity is the whole reason v2 exists, so it is asserted rather than assumed: it holds
    only when the input and feature grids are both odd and `len(coords) - 1 == total_stride *
    (n_feat - 1)`, i.e. the last feature cell lands on the last input cell. Any other combination
    means the feature lattice is NOT origin-centred and the readout would need an offset again."""
    n_input = len(coords)
    if n_input % 2 == 0 or n_feat % 2 == 0:
        raise ValueError(f"both grids must be odd, got n_input={n_input}, n_feat={n_feat}")
    if n_input - 1 != total_stride * (n_feat - 1):
        raise ValueError(
            f"feature lattice is not origin-centred: n_input-1 = {n_input - 1} != "
            f"total_stride * (n_feat-1) = {total_stride * (n_feat - 1)}"
        )
    return coords[total_stride * np.arange(n_feat)]


def crop_offset(n_feat: int, n_lattice: int) -> int:
    """Index of the first kept cell when centre-cropping an `n_feat` feature lattice down to the
    `n_lattice` spawn lattice -- design.md section 3c's `[3:18, 3:18]`, computed rather than
    hardcoded. Requires the difference to be even, which an odd/odd pair guarantees."""
    if n_feat < n_lattice:
        raise ValueError(f"feature lattice {n_feat} is smaller than the label lattice {n_lattice}")
    if (n_feat - n_lattice) % 2 != 0:
        raise ValueError(
            f"cannot centre-crop {n_feat} -> {n_lattice}: the difference must be even"
        )
    return (n_feat - n_lattice) // 2


def heightmap_to_tensor(
    terrain: HeightMapReader,
    n_cells: int = DEFAULT_N_CELLS,
    resolution: float = DEFAULT_RESOLUTION,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """[n_cells, n_cells] float32 tensor of absolute world z, resampled from `terrain` onto the
    odd origin-centred grid above. Ready to feed the CNN as a single-channel input after
    `.unsqueeze(0)`.

    Parameterised by (n_cells, resolution) rather than by (extent, resolution) as v1 is: for an
    odd grid the cell COUNT is the primitive quantity -- it is what has to be odd, and deriving
    it from an extent invites a rounding that silently produces an even count."""
    coords = grid_coords_centered(n_cells, resolution)
    X, Y = np.meshgrid(coords, coords)  # [n, n], row = y, col = x
    Z = np.asarray(terrain.sample(X, Y), dtype=np.float32)
    return torch.from_numpy(Z).to(device)


def load_heightmap_tensor(
    path: str | pathlib.Path,
    n_cells: int = DEFAULT_N_CELLS,
    resolution: float = DEFAULT_RESOLUTION,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """heightmap_to_tensor() straight from a saved <path>.png/.yaml pair."""
    terrain = HeightMapReader.load(path)
    return heightmap_to_tensor(terrain, n_cells=n_cells, resolution=resolution, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("path", type=str, help="heightmap path, e.g. assets/box_random/box_random_i0000_h070cm")
    parser.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION, help="tensor cell size (m)")
    parser.add_argument("--cells", type=int, default=DEFAULT_N_CELLS, help="odd cells per side")
    args = parser.parse_args()

    tensor = load_heightmap_tensor(args.path, n_cells=args.cells, resolution=args.resolution)
    print(
        f"{tensor.shape[0]}x{tensor.shape[1]} tensor, resolution {args.resolution} m, "
        f"extent {extent_of(args.cells, args.resolution):.4f} m, "
        f"z in [{tensor.min():.3f}, {tensor.max():.3f}]"
    )

    # --- grid convention: odd, origin-centred, symmetric about 0 ------------------------------
    coords = grid_coords_centered(DEFAULT_N_CELLS, DEFAULT_RESOLUTION)
    assert coords[DEFAULT_N_CELLS // 2] == 0.0, "no cell centre on the world origin"
    assert np.allclose(coords, -coords[::-1]), "grid is not symmetric about y = 0"
    assert abs(coords[0] + 5.0) < 1e-12 and abs(coords[-1] - 5.0) < 1e-12, (coords[0], coords[-1])
    assert abs(extent_of() - 10.125) < 1e-12, extent_of()
    assert n_cells_for(extent_of()) == DEFAULT_N_CELLS
    try:
        grid_coords_centered(80, DEFAULT_RESOLUTION)
    except ValueError:
        pass
    else:
        raise AssertionError("an even cell count must be rejected")
    print(f"[grid] {DEFAULT_N_CELLS} cells @ {DEFAULT_RESOLUTION} m, centres "
          f"{coords[0]:+.3f} .. {coords[-1]:+.3f}, origin at index {DEFAULT_N_CELLS // 2}")

    # --- the readout identity design.md section 3c claims: 81 -> 41 -> 21, and the 21-cell
    # feature lattice's coordinates ARE the 0.5 m lattice, so cropping it to 15 gives spawn_xy
    # exactly -- no interpolation, no offset. ---------------------------------------------------
    n1 = conv_out_size(DEFAULT_N_CELLS, stride=2)
    n2 = conv_out_size(n1, stride=2)
    assert (n1, n2) == (41, 21), (n1, n2)
    feat_coords = downsampled_coords(coords, total_stride=4, n_feat=n2)
    assert np.allclose(np.diff(feat_coords), SPAWN_STEP), "feature pitch != spawn lattice pitch"

    lattice = spawn_lattice()
    G = lattice.shape[0]
    assert G == 15, G
    off = crop_offset(n2, G)
    assert off == 3, off
    cropped = feat_coords[off:off + G]
    assert np.allclose(cropped, lattice[0, :, 0], atol=1e-12), (cropped, lattice[0, :, 0])
    assert np.allclose(cropped, lattice[:, 0, 1], atol=1e-12), (cropped, lattice[:, 0, 1])
    assert lattice[0, 0].tolist() == [-SPAWN_LIMIT, -SPAWN_LIMIT], lattice[0, 0]
    assert lattice[0, -1].tolist() == [SPAWN_LIMIT, -SPAWN_LIMIT], lattice[0, -1]
    assert lattice[-1, 0].tolist() == [-SPAWN_LIMIT, SPAWN_LIMIT], lattice[-1, 0]
    print(f"[readout] {DEFAULT_N_CELLS} -> {n1} -> {n2} @ {SPAWN_STEP} m, crop [{off}:{off + G}] "
          f"-> {cropped[0]:+.3f} .. {cropped[-1]:+.3f} m == spawn lattice, exactly")

    # --- the intermediate 41-cell stage is a valid readout too (design.md section 3e: any
    # power-of-two lattice pitch works with the same trunk) --------------------------------------
    mid_coords = downsampled_coords(coords, total_stride=2, n_feat=n1)
    assert np.allclose(np.diff(mid_coords), 2 * DEFAULT_RESOLUTION)
    assert np.allclose(mid_coords[::2], feat_coords), "block-5 slice[::2] is not co-located with block-7"
    print("[readout] block-5 slice[::2] is co-located with the 0.5 m lattice (design.md 5c)")

    # --- flat ground must still resample to an all-zero tensor ---------------------------------
    flat = HeightMapReader.flat(xlim=(-8.0, 8.0), ylim=(-8.0, 8.0))
    flat_tensor = heightmap_to_tensor(flat)
    assert flat_tensor.shape == (DEFAULT_N_CELLS, DEFAULT_N_CELLS), flat_tensor.shape
    assert torch.allclose(flat_tensor, torch.zeros_like(flat_tensor)), "flat ground must give an all-zero tensor"
    print("[flat] flat-ground self-check ok")
    print("all self-checks ok")
