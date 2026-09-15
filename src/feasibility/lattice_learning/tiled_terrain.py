"""Several heightmaps as ONE ostrich terrain, each shifted to its own tile in xy -- what lets
`generate_dataset.py +maps_per_build=K` run K maps' trials in a single ostrich model build.

`comparator.common.HelhestBatchSimulator` bakes its terrain into the model as one global mesh
shape, so a build's parallel worlds can never span two separately-built maps. `TiledTerrain`
sidesteps that without touching `comparator/`: each map is triangulated exactly as a per-map build
would (`HeightMapReader.to_ostrich_mesh()`, just with a shifted origin), and the K meshes are
concatenated into ONE `newton.Mesh` -- still one global shape, so `_build_world_starts`' leading-
global assumption and the explicit broadphase's pair count are unchanged. A trial spawned at
`(x, y) + offset` then sees exactly its own map's surface.

Off-map space stays EMPTY, exactly as in a per-map build (no padding, no edge fill), so a trial's
surroundings don't change; `TILE_GAP` only has to keep a robot near one map's border from reaching
the next map, and is far above the rear wheel's ~1.45 m rim reach.

Tiles are laid out on a square-ish grid centred on the world origin, so coordinates stay within a
few tens of metres (float32 positions keep ~micrometre resolution there).

Usage:
    python src/feasibility/lattice_learning/tiled_terrain.py   # self-test, no GPU
"""
from __future__ import annotations

import math

import newton
import numpy as np

from feasibility.heightmap import HeightMapReader

TILE_GAP = 5.0  # m of empty space between neighbouring map footprints


def _extent(terrain: HeightMapReader) -> tuple[float, float]:
    """(width_x, width_y) of a map's cell-center footprint's bounding cells, metres."""
    return terrain.nx * terrain.cell, terrain.ny * terrain.cell


def _center(terrain: HeightMapReader) -> tuple[float, float]:
    """World (x, y) of a map's own centre -- same `x0 + nx*cell/2` as `HeightMapReader.to_ostrich`."""
    w, h = _extent(terrain)
    return terrain.x0 + w / 2.0, terrain.y0 + h / 2.0


def tile_offsets(terrains: list[HeightMapReader], gap: float = TILE_GAP) -> np.ndarray:
    """[K, 2] xy offsets placing each map's centre on its own tile of a `ceil(sqrt(K))`-column
    grid centred on the origin, tile pitch = the largest map extent + `gap`. K = 1 returns a zero
    offset, so a single map keeps its own coordinates."""
    k = len(terrains)
    if k == 1:
        return np.zeros((1, 2), dtype=np.float64)
    cols = math.ceil(math.sqrt(k))
    rows = math.ceil(k / cols)
    pitch = max(max(_extent(t)) for t in terrains) + gap
    offsets = np.zeros((k, 2), dtype=np.float64)
    for i, t in enumerate(terrains):
        r, c = divmod(i, cols)
        tile_x = (c - (cols - 1) / 2.0) * pitch
        tile_y = (r - (rows - 1) / 2.0) * pitch
        cx, cy = _center(t)
        offsets[i] = (tile_x - cx, tile_y - cy)
    return offsets


def shifted(terrain: HeightMapReader, offset: np.ndarray) -> HeightMapReader:
    """The same grid with its origin moved by `offset` -- identical H, so identical triangulation."""
    return HeightMapReader(
        terrain.H, (terrain.x0 + float(offset[0]), terrain.y0 + float(offset[1])), terrain.cell,
        min_z=terrain.min_z, max_z=terrain.max_z,
    )


class TiledTerrain:
    """K maps at `offsets` [K, 2], duck-typed to the one terrain method
    `HelhestBatchSimulator.build_model` calls, `to_ostrich_mesh()`."""

    def __init__(self, terrains: list[HeightMapReader], offsets: np.ndarray):
        if len(terrains) != len(offsets):
            raise ValueError(f"{len(terrains)} terrains but {len(offsets)} offsets")
        self.terrains = list(terrains)
        self.offsets = np.asarray(offsets, dtype=np.float64)
        self._arrays: tuple[np.ndarray, np.ndarray] | None = None

    def to_ostrich_mesh(self) -> newton.Mesh:
        """One merged mesh. The triangulation is computed once and cached (a generator rebuilds
        the model once per chunk); every call still returns a fresh `newton.Mesh`."""
        if self._arrays is None:
            vertices, indices, n_verts = [], [], 0
            for terrain, offset in zip(self.terrains, self.offsets):
                mesh = shifted(terrain, offset).to_ostrich_mesh()
                vertices.append(mesh.vertices)
                indices.append(mesh.indices.astype(np.int64) + n_verts)
                n_verts += mesh.vertices.shape[0]
            self._arrays = (np.concatenate(vertices, axis=0), np.concatenate(indices).astype(np.int32))
        return newton.Mesh(*self._arrays, compute_inertia=False, is_solid=False)

    def sample(self, x, y):
        """Deliberately unsupported: `HelhestBatchSimulator` only samples the terrain for its
        default `terrain + 0.5` spawn, and every tiled caller must pass `spawn_zpr` instead."""
        raise NotImplementedError("TiledTerrain has no single height lookup; pass spawn_zpr")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    maps = [
        HeightMapReader(rng.uniform(0.0, 0.3, size=(40, 50)), (-2.5, -2.0), 0.1),
        HeightMapReader(rng.uniform(0.0, 0.5, size=(60, 30)), (1.0, -7.0), 0.1),
        HeightMapReader(np.zeros((20, 20)), (-1.0, -1.0), 0.1),
    ]
    offsets = tile_offsets(maps)
    assert np.allclose(tile_offsets(maps[:1]), 0.0)

    # footprints at least TILE_GAP apart (bounding boxes, axis-separated)
    boxes = []
    for t, o in zip(maps, offsets):
        s = shifted(t, o)
        w, h = _extent(s)
        boxes.append((s.x0, s.x0 + w, s.y0, s.y0 + h))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            sep = max(b[0] - a[1], a[0] - b[1], b[2] - a[3], a[2] - b[3])
            assert sep >= TILE_GAP - 1e-9, (i, j, sep)
    assert np.abs(np.array(boxes)).max() < 20.0, "tiles should stay near the origin"

    merged = TiledTerrain(maps, offsets).to_ostrich_mesh()
    singles = [t.to_ostrich_mesh() for t in maps]
    assert merged.vertices.shape[0] == sum(m.vertices.shape[0] for m in singles)
    assert merged.indices.size == sum(m.indices.size for m in singles)

    v0 = 0
    i0 = 0
    for m, o in zip(singles, offsets):
        nv, ni = m.vertices.shape[0], m.indices.size
        block = merged.vertices[v0 : v0 + nv].astype(np.float64)
        block[:, :2] -= o
        assert np.allclose(block, m.vertices, atol=1e-5), "shifted map mesh != standalone mesh"
        assert np.array_equal(merged.indices[i0 : i0 + ni] - v0, m.indices)
        v0 += nv
        i0 += ni

    try:
        TiledTerrain(maps, offsets).sample(0.0, 0.0)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("sample() must refuse")
    print(f"tiled_terrain self-test ok: {len(maps)} maps, offsets {offsets.round(2).tolist()}, "
          f"{merged.vertices.shape[0]} vertices")
