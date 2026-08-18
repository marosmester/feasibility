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

    def _interval_subdivisions(self, lo: int, hi: int, axis: int, max_rise_per_tile: float) -> int:
        """Subdivision count for original-grid interval [lo, hi] along `axis`: the worst-case
        elevation change ALONG this axis within the interval, maximized over every line
        perpendicular to it. Must isolate the axis (per-line reduce, then max) rather than a flat
        block max-min over the whole interval -- otherwise a speed bump's huge X-driven rise would
        spuriously force refinement of the (perfectly flat) Y axis too, and vice versa."""
        if axis == 1:
            band = self.H[:, lo : hi + 1]
            per_line_rise = band.max(axis=1) - band.min(axis=1)
        else:
            band = self.H[lo : hi + 1, :]
            per_line_rise = band.max(axis=0) - band.min(axis=0)
        rise = float(per_line_rise.max())
        return max(1, int(np.ceil(rise / max_rise_per_tile)))

    def _refine_axis(self, idx: np.ndarray, origin: float, axis: int, max_rise_per_tile: float) -> np.ndarray:
        """World coordinates for one axis of to_ostrich_mesh's tensor grid. `idx` are the baseline
        (post-stride) original-grid indices; each interval [idx[i], idx[i+1]] is independently
        subdivided via _interval_subdivisions, so steep bands get denser sampling than flat ones."""
        coords = [origin + (idx[0] + 0.5) * self.cell]
        for i in range(len(idx) - 1):
            lo, hi = int(idx[i]), int(idx[i + 1])
            k = self._interval_subdivisions(lo, hi, axis, max_rise_per_tile)
            c_lo = origin + (lo + 0.5) * self.cell
            c_hi = origin + (hi + 0.5) * self.cell
            coords.extend(np.linspace(c_lo, c_hi, k + 1)[1:])  # drop shared left endpoint
        return np.asarray(coords, dtype=np.float64)

    def to_ostrich_mesh(self, stride: int = 1, max_rise_per_tile: float | None = None) -> newton.Mesh:
        """Same surface as to_ostrich(), as an explicit triangle mesh instead of a
        newton.Heightfield -- needs no placing xform (identity), CCW-wound (+Z normal)
        triangles.

        `stride` subsamples the baseline grid to trade fidelity for triangle count in
        flat regions. `max_rise_per_tile` (meters, default `None` -> self.cell)
        independently refines each baseline interval along X and Y so no output tile
        spans more than roughly this much elevation -- e.g. a near-90 deg speed-bump
        ramp (create_speed_bumps.py's --incline-deg) no longer collapses into one
        giant sliver. Newton's mesh collision uses input triangles VERBATIM (no
        engine-side re-tessellation), so triangle size directly sets contact-sampling
        density; refining only shrinks tile size along the source heightmap's already
        straight/collinear ramp, it doesn't invent new geometry.

        Refinement is a crack-free tensor grid (independent per-band X/Y subdivision),
        not a full 2D quadtree: a column band's X-refinement spans its whole
        Y-extent, and vice versa. Exact for this class's real use case (speed bumps
        span the full Y width); would over-refine a heightmap with sparse localized
        steep features elsewhere.
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

        xs = self._refine_axis(cols, self.x0, axis=1, max_rise_per_tile=max_rise_per_tile)
        ys = self._refine_axis(rows, self.y0, axis=0, max_rise_per_tile=max_rise_per_tile)
        X, Y = np.meshgrid(xs, ys)  # [nr, nc], row=y, col=x -- same layout as H
        Z = self.sample(X, Y)  # bilinear -- fine points generally aren't at original cell centers
        points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)

        nr, nc = len(ys), len(xs)
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
