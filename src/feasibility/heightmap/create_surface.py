"""Rasterize ostrich/examples/assets/surface.obj -- the hilly "Landscape" mesh Blender-exported
as a REGULAR 128x128 vertex grid -- into a single HeightMapReader asset (assets/surface/surface),
so feasibility.comparator.compare_on_surface can hand ostrich (Newton mesh) and helhest_stack
(elevation grid) bit-identical terrain, the same "one physical terrain, two adapters" philosophy
as create_speed_bumps.py/create_box_obstacles.py -- just a single terrain instead of a swept
series, since compare_on_surface batches over (initial condition, control sequence) instead of
over terrain.

The obj's raw vertices span x,y in [-2.5, 2.5] m, z in [0, ~0.37] m. Scaled/offset here by the
SAME (SCALE, Z_OFFSET) ostrich/examples/helhest/surface_drive.py applies at load time
(`mesh_points = points * scale + offset`), so this asset matches every other place in the repo
that already renders "the surface": x,y in +-15 m, z in [0.05, ~1.54] m.

CLI parameters: none.

Usage:
    python src/feasibility/heightmap/create_surface.py
"""
from __future__ import annotations

import pathlib

import numpy as np

from feasibility.heightmap import HeightMapReader

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
OBJ_PATH = REPO_ROOT / "ostrich" / "examples" / "assets" / "surface.obj"
ASSETS_DIR = REPO_ROOT / "assets" / "surface"

SCALE = np.array([6.0, 6.0, 4.0])
Z_OFFSET = 0.05


def surface_path() -> pathlib.Path:
    """assets/surface/surface -- no extension, HeightMapReader appends .png/.yaml. Single
    terrain, no series, so (unlike bump_path/box_path) no height-dependent naming needed."""
    return ASSETS_DIR / "surface"


def _load_obj_vertices(path: pathlib.Path) -> np.ndarray:
    """[V, 3] xyz for every "v x y z" line, in file order -- ignores faces/normals/mtllib;
    all that's needed to rebuild the elevation grid."""
    verts = []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            _, x, y, z = line.split()
            verts.append((float(x), float(y), float(z)))
    return np.array(verts, dtype=np.float64)


def build_surface(obj_path: pathlib.Path = OBJ_PATH) -> HeightMapReader:
    """surface.obj's vertices are a Blender grid export in regular row-major order (constant y
    per row, x descending left-to-right within a row -- checked below), so no interpolation is
    needed: reshape straight into a HeightMapReader grid."""
    verts = _load_obj_vertices(obj_path)
    n = int(round(len(verts) ** 0.5))
    if n * n != len(verts):
        raise ValueError(f"{obj_path} has {len(verts)} vertices, not a perfect square grid")
    x, y, z = verts[:, 0].reshape(n, n), verts[:, 1].reshape(n, n), verts[:, 2].reshape(n, n)
    if not (np.allclose(y[0], y[0, 0]) and np.allclose(x[:, 0], x[0, 0])):
        raise ValueError(f"{obj_path}'s vertices are not a regular row-major (y-per-row) grid")

    cell = abs(x[0, 0] - x[0, -1]) / (n - 1) * SCALE[0]
    # x descends along a row -> flip columns so H's column index ascends with world x, matching
    # HeightMapReader's convention (see its module docstring).
    H = (z * SCALE[2] + Z_OFFSET)[:, ::-1]
    x0 = x[0, -1] * SCALE[0] - cell / 2.0
    y0 = y[0, 0] * SCALE[1] - cell / 2.0
    return HeightMapReader(H, origin=(x0, y0), cell=cell)


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    hmap = build_surface()
    path = surface_path()
    hmap.save(path)
    print(
        f"saved {path}.png / {path}.yaml  "
        f"({hmap.nx}x{hmap.ny} cells, cell={hmap.cell:.4f} m, z in [{hmap.min_z:.3f}, {hmap.max_z:.3f}] m)"
    )


if __name__ == "__main__":
    main()
