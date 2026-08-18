"""Simulator-agnostic heightmap: one elevation grid H[ny, nx], loaded from a
ROS map_server-style PNG+YAML pair, with adapter methods that hand each
simulator its own native terrain representation built from the SAME
underlying array -- so ostrich (Newton) and helhest_stack see physically
identical terrain, not two independently-approximated hills.

Grid convention (matches helhest_stack's helhest.heightmap.Heightmap and
helhest.engine.terrain._locate exactly, both bit-for-bit): H[i, j] sits at
CELL CENTERS, world (x0 + (j+0.5)*cell, y0 + (i+0.5)*cell), with (x0, y0) the
grid's min corner.

Newton's newton.Heightfield instead samples at grid VERTICES spanning
[-hx, +hx] x [-hy, +hy] around the shape's xform translation. to_ostrich()
reconciles the two conventions exactly, with no resampling: setting
hx = cell*(nx-1)/2, hy = cell*(ny-1)/2 and placing xform at this grid's
cell-center midpoint makes vertex (row, col) land on precisely the same
world (x, y) as this grid's cell-center (row, col), so the identical H array
feeds both simulators.
"""
from __future__ import annotations

import pathlib

import newton
import numpy as np
import warp as wp
import yaml
from helhest.engine import GridParams
from PIL import Image


class HeightMapReader:
    """Owns one physical heightmap; hands each simulator its native form."""

    def __init__(
        self,
        H: np.ndarray,
        origin: tuple[float, float],
        cell: float,
        min_z: float | None = None,
        max_z: float | None = None,
    ):
        self.H = np.asarray(H, dtype=np.float64)  # [ny, nx], cell centers
        self.ny, self.nx = self.H.shape
        self.x0, self.y0 = float(origin[0]), float(origin[1])
        self.cell = float(cell)
        self.min_z = float(self.H.min()) if min_z is None else float(min_z)
        self.max_z = float(self.H.max()) if max_z is None else float(max_z)

    @classmethod
    def flat(
        cls,
        xlim: tuple[float, float] = (-2.0, 6.0),
        ylim: tuple[float, float] = (-3.0, 3.0),
        cell: float = 0.05,
    ) -> "HeightMapReader":
        """Zero-elevation grid, no file needed -- keeps callers runnable with
        no terrain asset, matching both demos' historical flat-ground default."""
        nx = int(round((xlim[1] - xlim[0]) / cell)) + 1
        ny = int(round((ylim[1] - ylim[0]) / cell)) + 1
        H = np.zeros((ny, nx), dtype=np.float64)
        return cls(H, (xlim[0], ylim[0]), cell, min_z=0.0, max_z=0.0)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "HeightMapReader":
        """Load <path>.png (grayscale elevation) + <path>.yaml (resolution,
        origin: [x0, y0], min_z, max_z). `path` may be given with or without
        an extension."""
        path = pathlib.Path(path)
        meta = yaml.safe_load(path.with_suffix(".yaml").read_text())
        img = np.asarray(Image.open(path.with_suffix(".png")).convert("L"), dtype=np.float64) / 255.0
        min_z, max_z = float(meta["min_z"]), float(meta["max_z"])
        H = min_z + img * (max_z - min_z)
        x0, y0 = meta["origin"]
        return cls(H, (x0, y0), float(meta["resolution"]), min_z=min_z, max_z=max_z)

    def save(self, path: str | pathlib.Path) -> None:
        """Write <path>.png + <path>.yaml (inverse of load()). Elevation is
        quantized to 8 bits, i.e. (max_z-min_z)/255 resolution."""
        path = pathlib.Path(path)
        rng = self.max_z - self.min_z
        norm = np.zeros_like(self.H) if rng <= 0.0 else (self.H - self.min_z) / rng
        img = np.clip(np.round(norm * 255.0), 0, 255).astype(np.uint8)
        Image.fromarray(img, mode="L").save(path.with_suffix(".png"))
        meta = {
            "resolution": self.cell,
            "origin": [self.x0, self.y0],
            "min_z": self.min_z,
            "max_z": self.max_z,
        }
        path.with_suffix(".yaml").write_text(yaml.safe_dump(meta))

    def sample(self, x, y):
        """Bilinear height at world (x, y). Scalar or array. Same cell-center
        convention as helhest.heightmap.Heightmap.sample / helhest.engine's
        _locate -- used to place spawn poses on the loaded terrain."""
        fx = (np.asarray(x, dtype=np.float64) - self.x0) / self.cell - 0.5
        fy = (np.asarray(y, dtype=np.float64) - self.y0) / self.cell - 0.5
        ix = np.clip(np.floor(fx).astype(int), 0, self.nx - 2)
        iy = np.clip(np.floor(fy).astype(int), 0, self.ny - 2)
        tx = np.clip(fx - ix, 0.0, 1.0)
        ty = np.clip(fy - iy, 0.0, 1.0)
        H = self.H
        h00, h10 = H[iy, ix], H[iy, ix + 1]
        h01, h11 = H[iy + 1, ix], H[iy + 1, ix + 1]
        return (1 - tx) * (1 - ty) * h00 + tx * (1 - ty) * h10 + (1 - tx) * ty * h01 + tx * ty * h11

    def to_hstack(self, device: str) -> tuple[wp.array, GridParams]:
        """Elevation array + grid metadata for
        ForwardSimulator.set_terrain(...) / GridParams(...)."""
        elevation = wp.array(np.ascontiguousarray(self.H, np.float32), dtype=wp.float32, device=device)
        grid = GridParams(self.nx, self.ny, self.cell, self.x0, self.y0)
        return elevation, grid

    def to_ostrich(self) -> tuple[newton.Heightfield, wp.transform]:
        """Heightfield + placing xform for
        ModelBuilder.add_shape_heightfield(xform=..., heightfield=...)."""
        hx = self.cell * (self.nx - 1) / 2.0
        hy = self.cell * (self.ny - 1) / 2.0
        cx = self.x0 + self.cell * self.nx / 2.0
        cy = self.y0 + self.cell * self.ny / 2.0
        heightfield = newton.Heightfield(
            data=self.H.astype(np.float32),
            nrow=self.ny,
            ncol=self.nx,
            hx=hx,
            hy=hy,
            min_z=self.min_z,
            max_z=self.max_z,
        )
        xform = wp.transform(wp.vec3(cx, cy, 0.0), wp.quat_identity())
        return heightfield, xform

    def to_ostrich_mesh(self, stride: int = 1) -> newton.Mesh:
        """Same surface as to_ostrich(), as an explicit triangle mesh instead of a
        newton.Heightfield -- for A/B-ing heightfield collision against ostrich's
        mesh path (examples/helhest/surface_drive.py's terrain representation).

        Vertices sit at this grid's cell centers in WORLD coordinates, i.e. exactly
        where to_ostrich()'s heightfield vertices land, so the two adapters describe
        the same surface and the mesh needs no placing xform (add it at identity).
        Each cell quad is split into two CCW-wound triangles (+Z normals), matching
        how Newton's own heightfield collision triangulates cells -- so this is a
        representation swap, not a geometry change.

        `stride` subsamples the grid (every stride-th row/col, endpoints kept) to
        trade fidelity for triangle count: a full 801x601 grid is ~960k triangles,
        which is a heavy BVH. stride>1 loses ramp detail -- keep it at 1 unless the
        mesh path is too slow to iterate on.
        """
        rows = np.arange(0, self.ny, stride)
        cols = np.arange(0, self.nx, stride)
        if rows[-1] != self.ny - 1:
            rows = np.append(rows, self.ny - 1)
        if cols[-1] != self.nx - 1:
            cols = np.append(cols, self.nx - 1)

        xs = self.x0 + (cols + 0.5) * self.cell
        ys = self.y0 + (rows + 0.5) * self.cell
        X, Y = np.meshgrid(xs, ys)  # [nr, nc], row=y, col=x -- same layout as H
        Z = self.H[np.ix_(rows, cols)]
        points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)

        nr, nc = len(rows), len(cols)
        v = (np.arange(nr - 1)[:, None] * nc + np.arange(nc - 1)[None, :]).ravel()
        # (v, v+1, v+nc+1) and (v, v+nc+1, v+nc): +X along col, +Y along row, so this
        # winding gives an upward normal.
        tris = np.concatenate(
            [
                np.stack([v, v + 1, v + nc + 1], axis=-1),
                np.stack([v, v + nc + 1, v + nc], axis=-1),
            ]
        )
        # An open terrain sheet is not a closed solid: is_solid/compute_inertia would
        # be meaningless here, and the inertia integral over ~1M triangles is slow.
        return newton.Mesh(
            points,
            tris.ravel().astype(np.int32),
            compute_inertia=False,
            is_solid=False,
        )
