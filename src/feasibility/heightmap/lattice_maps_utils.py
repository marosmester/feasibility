"""Shared plumbing for lattice_learning's map-generation family -- create_ramps.py,
create_curbs_and_walls.py and create_poles_and_walls.py, composed by
create_maps_for_lattice_learning.py. Lives here, not in one another, so none of those sibling
scripts ever imports from another: every piece they share has exactly one home.

`grid_axes` is the cell-center coordinate convention every builder's raster shares (also
create_large_box_obstacles.build_rect_obstacle's own, restated there rather than imported since
that module is shared infrastructure well beyond this family). `GROUND_EPS` is the "is this cell
covered by a feature" threshold every area/overlap computation below uses, and `PLACEMENT_MARGIN`
is how far a feature's own placement anchor must stay inside the grid edge for it to still be
reachable by lattice_learning's spawn/edge sampling (curbs/walls and poles/walls apply it to a
feature's CENTER; create_ramps.py's foot_edge_margin is the same idea applied to a ramp's rising
face FOOT instead, different enough in shape to stay its own constant there).

`place_features` is the rejection-sampling placement loop all three builders run: draw a feature,
keep it only if it lands fully inside the grid, more than `overlap_margin` clear of every earlier
feature (by distance transform) and without pushing the covered area past `max_area_fraction`; the
first feature that cannot land within `max_attempts` draws stops the map, so a caller's own
`n_target` is only ever an upper bound on how many features come back.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.ndimage import distance_transform_edt

GROUND_EPS = 1e-6  # m, a cell above this counts as covered by a feature
PLACEMENT_MARGIN = 1.1  # m -- module docstring; lattice_learning keeps spawn poses ~2.5 m from the
# map edge (patch reach + warm-up lead) and its edge strategy starts trials up to 1.5 m from an
# edge, so a feature anchored 1.1 m from the edge is still reachable

# (w, d, cx, cy, yaw) of one rectangle in world coordinates, w along its own rotated x axis --
# create_large_box_obstacles.build_rect_obstacle's parameters, bundled for feature placement.
Rect = tuple[float, float, float, float, float]

# One placement attempt: draws a candidate feature and returns its ([ny, nx] height layer, record)
# if it landed fully inside the grid, or None to retry -- see place_features.
Attempt = Callable[[], "tuple[np.ndarray, object] | None"]


def grid_axes(extent: float, cell: float) -> np.ndarray:
    """Cell-center coordinates along one axis, identical to build_rect_obstacle's and
    create_rough_terrain.build_rough_terrain's grid, so layers from all of them line up."""
    n = int(round(extent / cell)) + 1
    return -extent / 2.0 + (np.arange(n) + 0.5) * cell


def rect_inside_grid(rect: Rect, height: float, incline_deg: float, extent: float) -> bool:
    """True when the rectangle's footprint, including its sloped skirt, lies inside the grid."""
    w, d, cx, cy, yaw = rect
    skirt = height / np.tan(np.radians(incline_deg))
    hx, hy = w / 2.0 + skirt, d / 2.0 + skirt
    c, s = np.cos(yaw), np.sin(yaw)
    reach_x = abs(c) * hx + abs(s) * hy
    reach_y = abs(s) * hx + abs(c) * hy
    limit = extent / 2.0
    return abs(cx) + reach_x <= limit and abs(cy) + reach_y <= limit


def place_features(
    extent: float,
    cell: float,
    n_target: int,
    overlap_margin: float,
    max_area_fraction: float,
    make_attempt: Callable[[], Attempt],
    max_attempts: int = 50,
) -> tuple[np.ndarray, list]:
    """Rejection-sample up to `n_target` features onto flat ground, merging accepted layers with
    an elementwise MAXIMUM (union of solids).

    `make_attempt()` is called once per feature SLOT and must return an `Attempt`: a zero-arg
    callable that draws one candidate and returns its (layer, record) if it fits inside the grid,
    or None to signal a fresh draw is needed. Some builders redraw the whole feature on every
    attempt (curbs/walls, poles/walls: `make_attempt` just returns the same closure every slot);
    create_ramps.py instead draws a ramp's SHAPE once per slot -- so retrying placement never
    biases the angle distribution -- and returns a closure over that one shape, retried only on
    its placement. Both are exactly "call make_attempt() once per slot, then attempt() up to
    `max_attempts` times", which is why that shape/placement split lives in the caller, not here.

    Each candidate layer is then checked here, in order: every earlier feature's `overlap_margin`
    clearance (by distance transform), then the cumulative `max_area_fraction` budget. The first
    feature that cannot land within `max_attempts` draws stops the map, so `n_target` is only an
    upper bound on `len(records)`. Returns the merged [ny, nx] height layer and the accepted
    records. Callers rely on `attempt()`'s draws being the only rng consumption in this whole
    loop -- everything here is pure numpy on the resulting layer -- so a fixed rng keeps producing
    the same map across changes to this function's surroundings as long as a caller's own
    attempt() draws stay in the same order."""
    n_axis = grid_axes(extent, cell).size
    H = np.zeros((n_axis, n_axis), dtype=np.float64)
    clearance = np.full((n_axis, n_axis), np.inf)  # m, distance to the nearest existing footprint
    records: list = []
    for _ in range(n_target):
        attempt = make_attempt()
        for _ in range(max_attempts):
            drawn = attempt()
            if drawn is None:
                continue
            layer, record = drawn
            if (clearance[layer > GROUND_EPS] <= overlap_margin).any():
                continue
            candidate = np.maximum(H, layer)
            if np.count_nonzero(candidate > GROUND_EPS) / candidate.size > max_area_fraction:
                continue
            H = candidate
            clearance = distance_transform_edt(H <= GROUND_EPS) * cell
            records.append(record)
            break
        else:
            break
    return H, records
