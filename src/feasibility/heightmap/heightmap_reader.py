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
        cell: float = 0.1,
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

    @staticmethod
    def _interval_reduce(A: np.ndarray, idx: np.ndarray, reduce) -> np.ndarray:
        """`reduce` (np.ptp / np.max) over each baseline interval [idx[k], idx[k+1]] along the LAST
        axis of A: [..., len(idx)-1]. With np.ptp on H this is the per-line rise, isolated per
        axis so a big X rise never forces Y refinement, and vice versa."""
        return np.stack([reduce(A[..., lo : hi + 1], axis=-1) for lo, hi in zip(idx[:-1], idx[1:])], axis=-1)

    @staticmethod
    def _zip_chains(a: list[int], ua: list[float], b: list[int], ub: list[float]) -> list[tuple[int, int, int]]:
        """Triangulate the strip between two parallel monotone chains -- `a` the lower one (its
        points at parameters ua, increasing), `b` the upper one -- walking both by parameter.
        Ties advance `b` first, which reproduces the unrefined quad's (0,0)-(1,1) diagonal."""
        tris = []
        i = j = 0
        while i < len(a) - 1 or j < len(b) - 1:
            if j < len(b) - 1 and (i == len(a) - 1 or ub[j + 1] <= ua[i + 1]):
                tris.append((a[i], b[j + 1], b[j]))
                j += 1
            else:
                tris.append((a[i], a[i + 1], b[j]))
                i += 1
        return tris

    @staticmethod
    def _zip_rings(outer: list[int], s_out: list[float], inner: list[int], s_in: list[float]) -> list[tuple[int, int, int]]:
        """Triangulate the annulus between two nested CCW rings, both starting at their (0,0)-side
        corner, each point tagged with a perimeter parameter s in [0, 4) (side index + fraction
        along that side). Corners of both rings share s = 0, 1, 2, 3, so the walk crosses every
        corner in lock-step and each side pair zips like two parallel chains."""
        tris = []
        no, ni = len(outer), len(inner)
        i = j = 0
        while i < no or j < ni:
            so = s_out[i + 1] if i + 1 < no else 4.0
            si = s_in[j + 1] if j + 1 < ni else 4.0
            if i < no and (j == ni or so <= si):
                tris.append((outer[i], outer[(i + 1) % no], inner[j % ni]))
                i += 1
            else:
                tris.append((outer[i % no], inner[(j + 1) % ni], inner[j]))
                j += 1
        return tris

    def to_ostrich_mesh(self, stride: int = 1, max_rise_per_tile: float | None = None) -> newton.Mesh:
        """Same surface as to_ostrich(), as an explicit triangle mesh instead of a
        newton.Heightfield -- needs no placing xform (identity), CCW-wound (+Z normal)
        triangles.

        `stride` subsamples the baseline grid to trade fidelity for triangle count in
        flat regions. `max_rise_per_tile` (meters, default `None` -> self.cell)
        refines steep cells so no output tile spans more than roughly this much
        elevation -- e.g. a near-90 deg speed-bump ramp (create_speed_bumps.py's
        --incline-deg) no longer collapses into one giant sliver. Newton's mesh
        collision uses input triangles VERBATIM (no engine-side re-tessellation), so
        triangle size directly sets contact-sampling density; refining only shrinks
        tile size along the source heightmap's already straight/collinear ramp, it
        doesn't invent new geometry.

        Refinement is LOCAL and crack-free. Every baseline cell edge gets its own
        subdivision count from the rise along it, so two cells sharing an edge always
        share its vertices; a cell is subdivided only if its own rise demands it, and
        then triangulated to match whatever counts its four edges carry (two-chain
        zipper when only one axis is refined, outer-edge ring zipped to an inner
        K x L grid when both are). Any cell whose edges and interior are within
        `max_rise_per_tile` -- flat ground, ramp plateaus, ordinary ramp faces -- stays
        the plain two-triangle baseline quad. (The previous tensor-grid scheme spread
        each steep interval's refinement across the whole map width.)
        """
        if max_rise_per_tile is None:
            max_rise_per_tile = self.cell
        if max_rise_per_tile <= 0.0:
            raise ValueError(f"max_rise_per_tile must be > 0, got {max_rise_per_tile}")

        rows = np.arange(0, self.ny, stride)
        cols = np.arange(0, self.nx, stride)
        if rows[-1] != self.ny - 1:
            rows = np.append(rows, self.ny - 1)
        if cols[-1] != self.nx - 1:
            cols = np.append(cols, self.nx - 1)
        nr, nc = len(rows), len(cols)
        xs = self.x0 + (cols + 0.5) * self.cell
        ys = self.y0 + (rows + 0.5) * self.cell

        def count(rise: np.ndarray) -> np.ndarray:
            return np.maximum(1, np.ceil(rise / max_rise_per_tile)).astype(np.int64)

        # Per-full-res-line rise within each baseline interval: X rise [ny, nc-1], Y rise [nr-1, nx].
        x_rise = self._interval_reduce(self.H, cols, np.ptp)
        y_rise = self._interval_reduce(self.H.T, rows, np.ptp).T
        ex = count(x_rise[rows])  # [nr, nc-1] subdivisions of the X-running edge on baseline row r
        ey = count(y_rise[:, cols])  # [nr-1, nc] subdivisions of the Y-running edge on baseline col c
        # Cell interior counts: worst line anywhere inside the stride block (== the two edges at
        # stride 1, since the only full-res lines in a 1-cell block are its edges).
        kx = count(self._interval_reduce(x_rise.T, rows, np.max).T)  # [nr-1, nc-1]
        ky = count(self._interval_reduce(y_rise, cols, np.max))  # [nr-1, nc-1]
        refined = (kx > 1) | (ky > 1)

        # Vertices: baseline corners first (index r*nc + c), then refined-edge interiors, then
        # refined-cell interiors. Positions are appended as (x, y); z is sampled once at the end.
        px = [np.repeat(xs[None, :], nr, axis=0).ravel()]
        py = [np.repeat(ys[:, None], nc, axis=1).ravel()]
        n_verts = nr * nc

        def add_points(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            nonlocal n_verts
            px.append(np.asarray(x, np.float64).ravel())
            py.append(np.asarray(y, np.float64).ravel())
            ids = np.arange(n_verts, n_verts + px[-1].size)
            n_verts += px[-1].size
            return ids

        def edge_chain(counts: np.ndarray, r: int, c: int, along_x: bool, cache: dict) -> list[int]:
            """Vertex ids of one baseline edge, start corner -> end corner, created on first use."""
            key = (r, c)
            if key not in cache:
                k = int(counts[r, c])
                t = np.arange(1, k) / k
                if along_x:
                    mid = add_points(xs[c] + t * (xs[c + 1] - xs[c]), np.full(k - 1, ys[r]))
                    cache[key] = [r * nc + c, *mid.tolist(), r * nc + c + 1]
                else:
                    mid = add_points(np.full(k - 1, xs[c]), ys[r] + t * (ys[r + 1] - ys[r]))
                    cache[key] = [r * nc + c, *mid.tolist(), (r + 1) * nc + c]
            return cache[key]

        x_edges: dict = {}
        y_edges: dict = {}
        tris: list[tuple[int, int, int]] = []
        for r, c in zip(*np.nonzero(refined)):
            r, c = int(r), int(c)
            bottom = edge_chain(ex, r, c, True, x_edges)
            top = edge_chain(ex, r + 1, c, True, x_edges)
            left = edge_chain(ey, r, c, False, y_edges)
            right = edge_chain(ey, r, c + 1, False, y_edges)
            ub = (np.arange(len(bottom)) / (len(bottom) - 1)).tolist()
            ut = (np.arange(len(top)) / (len(top) - 1)).tolist()
            vl = (np.arange(len(left)) / (len(left) - 1)).tolist()
            vr = (np.arange(len(right)) / (len(right) - 1)).tolist()
            K, L = int(kx[r, c]), int(ky[r, c])
            if L == 1:  # only X refined: left/right edges are single segments
                tris += self._zip_chains(bottom, ub, top, ut)
            elif K == 1:  # only Y refined: zip left (lower in X) to right, mirrored for CCW
                tris += [(a, c_, b) for a, b, c_ in self._zip_chains(left, vl, right, vr)]
            else:
                # Inner K x L grid at half-step offsets, strictly inside the cell.
                u = (np.arange(K) + 0.5) / K
                v = (np.arange(L) + 0.5) / L
                gx = xs[c] + u[None, :] * (xs[c + 1] - xs[c])
                gy = ys[r] + v[:, None] * (ys[r + 1] - ys[r])
                grid = add_points(np.broadcast_to(gx, (L, K)), np.broadcast_to(gy, (L, K))).reshape(L, K)
                g = grid[:-1, :-1].ravel()
                tris += list(zip(g, grid[:-1, 1:].ravel(), grid[1:, 1:].ravel()))
                tris += list(zip(g, grid[1:, 1:].ravel(), grid[1:, :-1].ravel()))
                # Outer ring CCW from (0,0): bottom, right, top (reversed), left (reversed).
                outer = bottom[:-1] + right[:-1] + top[::-1][:-1] + left[::-1][:-1]
                s_out = (
                    ub[:-1] + [1 + s for s in vr[:-1]]
                    + [3 - s for s in ut[::-1][:-1]] + [4 - s for s in vl[::-1][:-1]]
                )
                ring_u = np.arange(K) / max(K - 1, 1)
                ring_v = np.arange(L) / max(L - 1, 1)
                inner = (
                    grid[0, :-1].tolist() + grid[:-1, -1].tolist()
                    + grid[-1, ::-1][:-1].tolist() + grid[::-1, 0][:-1].tolist()
                )
                s_in = (
                    ring_u[:-1].tolist() + (1 + ring_v[:-1]).tolist()
                    + (3 - ring_u[::-1][:-1]).tolist() + (4 - ring_v[::-1][:-1]).tolist()
                )
                tris += self._zip_rings(outer, s_out, inner, s_in)

        # Every unrefined cell: (v, v+1, v+nc+1) and (v, v+nc+1, v+nc) -- +X along col, +Y along
        # row, so this winding gives an upward normal.
        v = (np.nonzero(~refined.ravel())[0] // (nc - 1)) * nc + np.nonzero(~refined.ravel())[0] % (nc - 1)
        plain = np.concatenate(
            [np.stack([v, v + 1, v + nc + 1], axis=-1), np.stack([v, v + nc + 1, v + nc], axis=-1)]
        )
        all_tris = np.concatenate([plain, np.asarray(tris, dtype=np.int64).reshape(-1, 3)])

        X = np.concatenate(px)
        Y = np.concatenate(py)
        Z = self.sample(X, Y)  # bilinear -- refined points generally aren't at original cell centers
        points = np.stack([X, Y, Z], axis=-1).astype(np.float32)
        # An open terrain sheet is not a closed solid: is_solid/compute_inertia would
        # be meaningless here, and the inertia integral over ~1M triangles is slow.
        return newton.Mesh(
            points,
            all_tris.ravel().astype(np.int32),
            compute_inertia=False,
            is_solid=False,
        )

    def to_ostrich_obstacle_mesh(
        self,
        threshold: float = 1e-3,
        z_offset: float = 0.003,
        stride: int = 1,
        max_rise_per_tile: float | None = None,
    ) -> newton.Mesh | None:
        """Visual-only twin of `to_ostrich_mesh` (same triangulation), kept only where a triangle
        clears `threshold` and raised `z_offset` [m] so it doesn't z-fight the real terrain mesh it
        sits on top of.

        Newton colours a mesh per SHAPE INSTANCE, not per triangle, so the single global terrain
        shape `comparator.common` builds is one flat colour in the GL viewer -- an obstacle (a
        curb, a wall, a box) reads by elevation on that shape's own colour, not as a distinct
        feature. Adding this as a SECOND, non-colliding shape (`has_shape_collision=False`) with
        its own `color` is what lets one be told apart from flat ground at a glance; the underlying
        collision mesh (and the physics) is untouched. Returns None if nothing clears `threshold`
        (an all-flat map), so the caller can skip adding a shape for it.
        """
        base = self.to_ostrich_mesh(stride=stride, max_rise_per_tile=max_rise_per_tile)
        verts, tris = base.vertices, base.indices.reshape(-1, 3)
        keep = verts[tris, 2].max(axis=1) > threshold
        if not keep.any():
            return None
        used = np.unique(tris[keep])
        remap = np.full(len(verts), -1, np.int64)
        remap[used] = np.arange(len(used))
        out_verts = verts[used].copy()
        out_verts[:, 2] += z_offset
        return newton.Mesh(
            out_verts,
            remap[tris[keep]].ravel().astype(np.int32),
            compute_inertia=False,
            is_solid=False,
        )
