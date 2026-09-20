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
ARC_DURATION_S: float = ARC_LEN / V_NOM  # 0.5 s -- the time one primitive takes, arc or pivot

# The in-place pivot primitive: `_build_primitives(pivot_cost > 0)` appends two point turns of +-1
# heading bin, same cell. Pinned the same way the arc is -- one trial turns exactly one bin in the
# same ARC_DURATION_S a forward arc takes, so OMEGA_NOM * ARC_DURATION_S == PIVOT_ANGLE mirrors
# V_NOM * ARC_DURATION_S == ARC_LEN.
N_THETA: int = 24  # the router's heading bin count (CostToGo's n_theta in every consumer)
PIVOT_ANGLE: float = 2.0 * np.pi / N_THETA  # rad, 15 deg -- one heading bin
OMEGA_NOM: float = PIVOT_ANGLE / ARC_DURATION_S  # rad/s, ~0.524 -- pinned pivot yaw rate


def primitive_kappas(min_turn_radius: float = MIN_TURN_RADIUS) -> tuple[float, float, float, float, float]:
    """The five curvatures (1/m) `_build_primitives`' `turns = [-step/R, -step/(2R), 0,
    step/(2R), step/R]` encode. `turns` is a list of HEADING CHANGES over one arc of length
    `step`, so kappa = dtheta/step is independent of `step` -- dividing it out here is what lets
    one primitive table serve every `step` the router is built with (design.md section 4a)."""
    r = float(min_turn_radius)
    return (-1.0 / r, -1.0 / (2.0 * r), 0.0, 1.0 / (2.0 * r), 1.0 / r)


def twist_from_kappa(kappa: np.ndarray, v: float = V_NOM) -> tuple[np.ndarray, np.ndarray]:
    """[n] curvature (1/m) -> the commanded body twist (v_drive [n] m/s, wz_drive [n] rad/s) for
    travel at speed `v` along it: wz = v * kappa. float32, the dataset's own column dtype."""
    kappa = np.asarray(kappa, dtype=np.float32)
    return np.full(kappa.shape, v, dtype=np.float32), (v * kappa).astype(np.float32)


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


def integrate_twist(
    pose: np.ndarray, v: np.ndarray | float, wz: np.ndarray | float, t: np.ndarray | float
) -> np.ndarray:
    """Exact endpoint(s) of a constant body twist (v m/s, wz rad/s) held for `t` seconds from
    `pose` = (x, y, yaw) [..., 3] -- the primitive-agnostic form of `integrate_arc`. `v > 0` is the
    arc of curvature wz/v over v*t metres; `v == 0` is a pivot about the body origin (the
    front-axle midpoint), i.e. the lattice's "same cell, heading over" point turn. Broadcasts like
    `integrate_arc`."""
    pose = np.asarray(pose, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    wz = np.asarray(wz, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    pivot = np.abs(v) < 1e-12
    kappa = np.where(pivot, 0.0, wz / np.where(pivot, 1.0, v))
    arc = integrate_arc(pose, kappa, v * t)
    spin = np.stack(np.broadcast_arrays(pose[..., 0], pose[..., 1], pose[..., 2] + wz * t), axis=-1)
    return np.where(pivot[..., None], spin, arc)


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
            endpoint = integrate_arc(np.array([0.0, 0.0, th0]), kappa, step)
            x1, y1, yaw1 = endpoint
            x1_built, y1_built = prim_dc[it, p] * resolution, prim_dr[it, p] * resolution
            # every primitive is billed its REALIZED chord (helhest_stack 373e6a6), not a flat `step`
            assert np.isclose(prim_cost[it, p], np.hypot(x1_built, y1_built), rtol=1e-5), (it, p)
            max_err = max(max_err, abs(x1 - x1_built), abs(y1 - y1_built))
            heading_expected = int(np.floor((yaw1 % (2.0 * np.pi)) / dth)) % n_theta
            assert prim_heading[it, p] == heading_expected, (it, p, prim_heading[it, p], heading_expected)

    assert max_err < 1e-4, f"arc.py disagrees with _build_primitives by {max_err:.2e} m"
    print(
        f"arc.py agrees with _build_primitives: max endpoint error {max_err:.2e} m over "
        f"{n_theta} headings x {n_prim} primitives"
    )

    # integrate_twist at (V_NOM, V_NOM * kappa) for ARC_DURATION_S IS the arc primitive
    rng = np.random.default_rng(0)
    poses = np.column_stack([rng.uniform(-3, 3, 64), rng.uniform(-3, 3, 64), rng.uniform(0, 6.3, 64)])
    kap = np.concatenate([rng.uniform(-KAPPA_MAX, KAPPA_MAX, 60), np.zeros(4)])
    assert np.allclose(integrate_twist(poses, V_NOM, V_NOM * kap, ARC_DURATION_S),
                       integrate_arc(poses, kap, ARC_LEN), atol=1e-12)

    # ... and at (0, +-OMEGA_NOM) the pivot primitive: same cell, heading exactly one bin over
    n_prim, prim_dr, prim_dc, prim_heading, _, _, _, _ = _build_primitives(
        n_theta=N_THETA, resolution=0.1, step=step, turn_radius=r, max_sweep=8, nseg=200,
        pivot_cost=1.0,
    )
    assert n_prim == 7
    dth = 2.0 * np.pi / N_THETA
    for it in range(N_THETA):
        th0 = (it + 0.5) * dth
        for p, sign in ((5, -1.0), (6, 1.0)):
            end = integrate_twist(np.array([0.0, 0.0, th0]), 0.0, sign * OMEGA_NOM, ARC_DURATION_S)
            assert end[0] == 0.0 and end[1] == 0.0 and prim_dr[it, p] == 0 and prim_dc[it, p] == 0
            assert prim_heading[it, p] == int(np.floor((end[2] % (2.0 * np.pi)) / dth)) % N_THETA
    print(f"integrate_twist agrees with the arc primitives and the +-1-bin pivot primitives")
    print(f"V_NOM={V_NOM} m/s  ARC_LEN={ARC_LEN} m  MIN_TURN_RADIUS={MIN_TURN_RADIUS} m  "
          f"KAPPA_MAX={KAPPA_MAX} /m  LEVER={LEVER} m  OMEGA_NOM={OMEGA_NOM:.4f} rad/s  "
          f"PIVOT_ANGLE={np.degrees(PIVOT_ANGLE):.1f} deg")
