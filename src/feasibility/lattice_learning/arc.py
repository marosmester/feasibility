"""The primitive contract, and the only place it is written down.

`lattice_solver._build_primitives` integrates the router's forward arcs numerically, in short
host-side segments -- convenient for a Warp model builder that also has to lay down pivots and
swept-collision cells, but not the definition of the curve. This module re-derives the same
curve in closed form: exact for any `kappa`, including the straight-ahead limit, and cheap to
evaluate on a batch of trials rather than one heading bin at a time.

Everything here is pinned, not carried as a runtime argument (design.md section 4): `V_NOM` and
`ARC_LEN` are constants of THIS dataset/checkpoint, matching `CostToGo`'s default `step` and a
router-typical speed, and `MIN_TURN_RADIUS`/`LEVER` are read off `RobotParams` rather than
re-guessed. A change to any of them invalidates a trained checkpoint silently unless the consumer
asserts against it -- see design.md section 4c; that assert lives in `infer.py`, not here.

`lattice_learning/` does not import from `learning/`, `grid_learning/` or `grid_learning_2/` (see
design.md section 11a) -- but `helhest` itself is an object of study, not a sibling experiment,
so `RobotParams` and (in `__main__` only) `_build_primitives` are imported directly.

Usage:
    python src/feasibility/lattice_learning/arc.py
"""
from __future__ import annotations

import numpy as np
from helhest.engine import RobotParams

_ROBOT = RobotParams()

V_NOM: float = 0.6  # m/s, pinned warm-start/travel speed -- design.md section 2
ARC_LEN: float = 0.3  # m, = CostToGo's default `step` -- design.md section 4a
MIN_TURN_RADIUS: float = _ROBOT.min_turn_radius  # m, the router's tightest forward arc
KAPPA_MAX: float = 1.0 / MIN_TURN_RADIUS  # 1/m
LEVER: float = _ROBOT.rear_offset + _ROBOT.wheel_radius  # m, rear-wheel lever arm -- design.md 7a


def primitive_kappas(min_turn_radius: float = MIN_TURN_RADIUS) -> tuple[float, float, float, float, float]:
    """The five curvatures (1/m) `_build_primitives`' `turns = [-step/R, -step/(2R), 0,
    step/(2R), step/R]` encode. `turns` is a list of HEADING CHANGES over one arc of length
    `step`, so kappa = dtheta/step is independent of `step` -- dividing it out here is what lets
    one primitive table serve every `step` the router is built with (design.md section 4a)."""
    r = float(min_turn_radius)
    return (-1.0 / r, -1.0 / (2.0 * r), 0.0, 1.0 / (2.0 * r), 1.0 / r)


def integrate_arc(pose: np.ndarray, kappa: np.ndarray | float, length: np.ndarray | float) -> np.ndarray:
    """Exact endpoint(s) of a constant-curvature arc: starting at body pose(s) `pose` = (x, y,
    yaw) [..., 3], turning at curvature `kappa` (1/m, +left) over `length` metres of travel.
    Broadcasts `kappa`/`length` against `pose[..., 0]`'s shape, so a single call handles one
    trial or a whole batch.

    This is the unicycle model integrated in closed form (theta(s) = yaw0 + kappa*s):
        x(L) = x0 + (sin(yaw0 + kappa*L) - sin(yaw0)) / kappa
        y(L) = y0 + (cos(yaw0) - cos(yaw0 + kappa*L)) / kappa
    with the `kappa -> 0` limit (x0 + L cos(yaw0), y0 + L sin(yaw0)) taken explicitly rather than
    left to divide-by-near-zero, since straight (`kappa = 0`) is one of the five primitives, not
    an edge case."""
    pose = np.asarray(pose, dtype=np.float64)
    x0, y0, yaw0 = pose[..., 0], pose[..., 1], pose[..., 2]
    kappa = np.asarray(kappa, dtype=np.float64)
    length = np.asarray(length, dtype=np.float64)
    dtheta = kappa * length
    yaw1 = yaw0 + dtheta
    straight = np.abs(kappa) < 1e-12
    safe_kappa = np.where(straight, 1.0, kappa)  # avoid 0-div; discarded by the where() below
    dx = np.where(straight, length * np.cos(yaw0), (np.sin(yaw1) - np.sin(yaw0)) / safe_kappa)
    dy = np.where(straight, length * np.sin(yaw0), (np.cos(yaw0) - np.cos(yaw1)) / safe_kappa)
    return np.stack([x0 + dx, y0 + dy, yaw1], axis=-1)


if __name__ == "__main__":
    from helhest.planning.lattice_solver import _build_primitives

    step = ARC_LEN
    r = MIN_TURN_RADIUS
    kappas = primitive_kappas(r)

    # Same formula _build_primitives uses for `turns`, re-derived independently of
    # primitive_kappas() -- this is the "do the two trees agree on what a primitive IS" check.
    turns_expected = (-step / r, -step / (2.0 * r), 0.0, step / (2.0 * r), step / r)
    assert np.allclose(np.array(kappas) * step, np.array(turns_expected)), (
        kappas, turns_expected,
    )

    # Fine resolution recovers _build_primitives' continuous (pre-rounding) endpoint to ~1e-6 m,
    # so comparing against integrate_arc's closed form isn't confounded by cell-rounding noise.
    n_theta = 16
    resolution = 1e-6
    n_prim, prim_dr, prim_dc, prim_heading, prim_cost, _, _, _ = _build_primitives(
        n_theta=n_theta,
        resolution=resolution,
        step=step,
        turn_radius=r,
        max_sweep=8,
        nseg=200,
        pivot_cost=0.0,
    )
    assert n_prim == 5

    dth = 2.0 * np.pi / n_theta
    max_err = 0.0
    for it in range(n_theta):
        th0 = (it + 0.5) * dth
        for p, kappa in enumerate(kappas):
            assert prim_cost[it, p] == step, "every forward arc shares length `step`"
            endpoint = integrate_arc(np.array([0.0, 0.0, th0]), kappa, step)
            x1, y1, yaw1 = endpoint
            x1_built, y1_built = prim_dc[it, p] * resolution, prim_dr[it, p] * resolution
            max_err = max(max_err, abs(x1 - x1_built), abs(y1 - y1_built))
            heading_expected = int(np.floor((yaw1 % (2.0 * np.pi)) / dth)) % n_theta
            assert prim_heading[it, p] == heading_expected, (it, p, prim_heading[it, p], heading_expected)

    assert max_err < 1e-4, f"arc.py disagrees with _build_primitives by {max_err:.2e} m"
    print(
        f"arc.py agrees with _build_primitives: max endpoint error {max_err:.2e} m over "
        f"{n_theta} headings x {n_prim} primitives"
    )
    print(f"V_NOM={V_NOM} m/s  ARC_LEN={ARC_LEN} m  MIN_TURN_RADIUS={MIN_TURN_RADIUS} m  "
          f"KAPPA_MAX={KAPPA_MAX} /m  LEVER={LEVER} m")
