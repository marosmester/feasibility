"""Pure-numpy scoring of a paired rollout: named behaviour metrics plus health flags.

Two different questions, deliberately kept apart:

  METRICS are what the baseline pins. They have no notion of correct -- they record what the two
  simulators did, and a later run differing from them is a prompt to look, not a bug report.

  FLAGS are physics violations: NaN, a pose off the map, a body teleporting. A flag makes the run
  FAIL regardless of the baseline, because no baseline value makes a NaN acceptable. Note the
  asymmetry -- updating the baseline is how you accept a metric change, but there is no way to
  accept a flag, which is the point.

METRIC_NAMES is the baseline's column contract: APPEND new metrics at the end, never reorder or
remove, so an old baseline stays readable under its own recorded names (the same rule
grid_learning_2/eval_log.py states for its CSV). run_check.py compares by name, so appending a
metric does not invalidate an existing baseline -- it just reports the new column as missing.

Everything here is a pure function of arrays, so `python src/feasibility/submodule_test/metrics.py`
self-tests every flag against fabricated rollouts in a second, with no GPU. That is this module's
smoke test.
"""

from __future__ import annotations

import numpy as np

from feasibility.comparator.common import _net_yaw_deg
from feasibility.comparator.common import _quat_yaw
from feasibility.heightmap import HeightMapReader
from feasibility.learning.pose_error import pose_to_se3
from feasibility.learning.pose_error import se3_error

# How far outside the terrain's own footprint / elevation band a final pose may sit before it is
# treated as a diverged solve rather than a large-but-real collision displacement. Same value and
# same justification as grid_learning/generate_dataset.py's POS_MARGIN, whose comment records the
# failure it was introduced for: a rear wheel catching a box ramp mid-turn flung the body to
# z = -23 m and still read "valid" under an older symmetric +-50 m check.
POS_MARGIN = 3.0

# A per-step body speed above `|v_drive| * FACTOR + FLOOR` (m/s) counts as an explosion. Catches a
# blow-up that happens mid-run and settles back inside the position bounds, which the final-pose
# checks cannot see.
#
# The floor is what makes the check meaningful for a turn in place, where commanded translation is
# zero, and it is set from measurement rather than taste. Peak per-step body speed over the smoke
# tier: 1.10-1.45 m/s on the v_drive=1.0 trials (i.e. ~1.45x commanded, the overshoot real), and
# 0.10-0.41 m/s on the turn-in-place trials. An earlier 1.0 m/s floor flagged box/turn_at_corner in
# 1 run out of 3 -- the rear wheel catching the box ramp mid-turn genuinely shoves the body past
# 1 m/s for a step, while displacement stayed at 0.23-0.27 m and every other check passed. That is
# contact physics, not divergence. 4.0 m/s sits ~3x above anything the commanded motion can
# produce and far below a diverged solve, which reaches tens of m/s (grid_learning's own
# turn-in-place tail runs to 8.6 m of displacement, and the `displaced` flag catches that too).
SPEED_SPIKE_FACTOR = 3.0
SPEED_SPIKE_FLOOR = 4.0

# helhest_stack's own quasi-static solve-quality gate: helhest/engine/robot.py's SolverParams
# resid_tol, applied at helhest/driver.py:71 and planning/costtogo.py to reject a rollout.
# run_hstack_batch already returns `residual` and nothing in this repo currently thresholds it.
HSTACK_RESID_TOL = 1e-2

# Chassis clearance below this counts as high-centered (helhest/engine/step.py's chassis_clearance
# is the signed height of the chassis bottom face over raw terrain, so negative == belly
# penetrating). Recorded as a METRIC, never a flag: helhest detects high-centering without
# resolving it, so negative clearance beside a box is expected behaviour, not a solver failure.
CLEAR_MARGIN = 0.0

METRIC_NAMES = (
    "o_final_x",
    "o_final_y",
    "o_final_z",
    "o_net_yaw_deg",
    "o_path_len",
    "o_peak_z",
    "o_max_pitch_deg",
    "o_max_roll_deg",
    "o_max_wheel_qd",
    "o_displacement",
    "h_final_x",
    "h_final_y",
    "h_final_z",
    "h_net_yaw_deg",
    "h_path_len",
    "h_peak_z",
    "h_max_pitch_deg",
    "h_max_roll_deg",
    "h_max_wheel_qd",
    "h_displacement",
    "e_pos",
    "e_rot_deg",
    "h_clearance_min",
    "h_clearance_below_frac",
    "h_residual_max",
)

FLAG_NAMES = ("nonfinite", "out_of_bounds", "displaced", "speed_spike", "hstack_residual")

# Which unit family each metric belongs to, so run_check.py can apply one tolerance per family
# rather than 25 hand-set numbers. Anything not listed is treated as dimensionless.
METRIC_UNITS = {
    name: (
        "deg"
        if name.endswith("_deg")
        else "rad_s"
        if name.endswith("_wheel_qd")
        else "frac"
        if name.endswith("_frac") or name.endswith("_residual_max")
        else "m"
    )
    for name in METRIC_NAMES
}


def quat_pitch_roll(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """q [..., 4] = (qx,qy,qz,qw) -> (pitch, roll) in radians, the Z-Y-X convention
    comparator/common.py's euler_zyx_to_quat_xyzw builds and _quat_yaw reads back."""
    qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    pitch = np.arcsin(np.clip(2.0 * (qy * qw - qx * qz), -1.0, 1.0))
    roll = np.arctan2(2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy))
    return pitch, roll


def plausible_bounds(terrain: HeightMapReader) -> np.ndarray:
    """[3, 2] per-axis (lo, hi) a pose may plausibly occupy on `terrain`: the map's own x/y
    footprint and its own (min_z, max_z) elevation band, each padded by POS_MARGIN.

    Re-derived here rather than imported from grid_learning/generate_dataset.py, which computes
    the same thing: that module is a dataset generator carrying Hydra and torch in its import
    path, and this one has to stay importable for a no-GPU self-test. Grounded in
    HeightMapReader.sample's documented behaviour -- querying outside these bounds does not raise,
    it clamps to the nearest edge cell and fabricates flat ground, so a physically real result can
    never legitimately land far outside them either.
    """
    x_lo, x_hi = terrain.x0, terrain.x0 + terrain.nx * terrain.cell
    y_lo, y_hi = terrain.y0, terrain.y0 + terrain.ny * terrain.cell
    return np.array(
        [
            [x_lo - POS_MARGIN, x_hi + POS_MARGIN],
            [y_lo - POS_MARGIN, y_hi + POS_MARGIN],
            [terrain.min_z - POS_MARGIN, terrain.max_z + POS_MARGIN],
        ]
    )


def _path_length(xy: np.ndarray) -> np.ndarray:
    """xy [T, n, 2] -> [n] cumulative path length. NaN-safe only in that a NaN anywhere makes the
    whole length NaN, which is what the `nonfinite` flag is for."""
    return np.linalg.norm(np.diff(xy, axis=0), axis=-1).sum(axis=0)


def _max_speed(xy: np.ndarray, dt: float) -> np.ndarray:
    """xy [T, n, 2] -> [n] peak per-step body speed (m/s)."""
    if xy.shape[0] < 2:
        return np.zeros(xy.shape[1])
    return np.linalg.norm(np.diff(xy, axis=0), axis=-1).max(axis=0) / dt


def _sim_block(pose: np.ndarray, wheel_qd: np.ndarray, spawn_xy: np.ndarray) -> list[np.ndarray]:
    """The ten per-simulator metrics, in METRIC_NAMES order, from pose [T, n, 7] and
    wheel_qd [T, n, 3]."""
    pitch, roll = quat_pitch_roll(pose[:, :, 3:7])
    return [
        pose[-1, :, 0],
        pose[-1, :, 1],
        pose[-1, :, 2],
        _net_yaw_deg(_quat_yaw(pose[:, :, 3:7])),
        _path_length(pose[:, :, :2]),
        pose[:, :, 2].max(axis=0),
        np.degrees(np.abs(pitch).max(axis=0)),
        np.degrees(np.abs(roll).max(axis=0)),
        np.abs(wheel_qd).max(axis=(0, 2)),
        np.linalg.norm(pose[-1, :, :2] - spawn_xy, axis=-1),
    ]


def compute_metrics(
    *,
    ostrich_pose: np.ndarray,
    ostrich_wheel_qd: np.ndarray,
    hstack_pose: np.ndarray,
    hstack_wheel_qd: np.ndarray,
    clearance: np.ndarray,
    residual: np.ndarray,
    spawn_xy: np.ndarray,
) -> np.ndarray:
    """[n_trials, len(METRIC_NAMES)] float64.

    `ostrich_pose` [T, n, 7] and `hstack_pose` [Th, n, 7] are both (x,y,z,qx,qy,qz,qw); the two
    sims run at different dt, so every reduction is over each one's own time axis and the two are
    only ever compared at the final step. `clearance`/`residual` [Th, n] come straight off
    run_hstack_batch.
    """
    n = ostrich_pose.shape[1]
    cols = _sim_block(ostrich_pose, ostrich_wheel_qd, spawn_xy)
    cols += _sim_block(hstack_pose, hstack_wheel_qd, spawn_xy)

    e_pos = np.empty(n)
    e_rot = np.empty(n)
    for i in range(n):
        o, h = ostrich_pose[-1, i], hstack_pose[-1, i]
        if not (np.isfinite(o).all() and np.isfinite(h).all()):
            e_pos[i] = e_rot[i] = np.nan  # se3_error would raise on a singular matrix
            continue
        p, r = se3_error(pose_to_se3(o), pose_to_se3(h))
        e_pos[i], e_rot[i] = p, np.degrees(r)
    cols += [e_pos, e_rot]

    cols += [
        clearance.min(axis=0),
        (clearance < CLEAR_MARGIN).mean(axis=0),
        residual.max(axis=0),
    ]

    out = np.stack(cols, axis=-1).astype(np.float64)
    assert out.shape == (n, len(METRIC_NAMES)), (
        f"built {out.shape[1]} columns but METRIC_NAMES has {len(METRIC_NAMES)} -- "
        "_sim_block and METRIC_NAMES have drifted apart"
    )
    return out


def health_flags(
    *,
    ostrich_pose: np.ndarray,
    hstack_pose: np.ndarray,
    residual: np.ndarray,
    spawn_xy: np.ndarray,
    max_displacement: np.ndarray,
    bounds: np.ndarray,
    ostrich_dt: float,
    hstack_dt: float,
    v_drive: np.ndarray,
) -> dict[str, np.ndarray]:
    """{flag_name: [n_trials] bool}. True means the run is not physically usable, independent of
    any baseline. Both sims are checked for every geometric flag, for symmetry -- though it is
    overwhelmingly ostrich, the one with a contact solver, that trips them."""
    n = ostrich_pose.shape[1]
    flags = {name: np.zeros(n, dtype=bool) for name in FLAG_NAMES}

    for pose, dt in ((ostrich_pose, ostrich_dt), (hstack_pose, hstack_dt)):
        # The WHOLE trajectory, not just the final pose: a run that goes NaN mid-flight and comes
        # back, or that flies out and returns, is still a broken run.
        flags["nonfinite"] |= ~np.isfinite(pose).all(axis=(0, 2))
        finite = np.isfinite(pose).all(axis=2)  # [T, n] -- ignore NaN steps here, flagged above
        for axis in range(3):
            v = np.where(finite, pose[:, :, axis], 0.0)
            flags["out_of_bounds"] |= (v < bounds[axis, 0]).any(axis=0)
            flags["out_of_bounds"] |= (v > bounds[axis, 1]).any(axis=0)
        travelled = np.linalg.norm(pose[-1, :, :2] - spawn_xy, axis=-1)
        flags["displaced"] |= travelled > max_displacement
        speed_cap = np.abs(v_drive) * SPEED_SPIKE_FACTOR + SPEED_SPIKE_FLOOR
        flags["speed_spike"] |= _max_speed(np.nan_to_num(pose[:, :, :2]), dt) > speed_cap

    flags["hstack_residual"] |= residual.max(axis=0) > HSTACK_RESID_TOL
    # A non-finite pose makes every derived comparison meaningless rather than merely large, so it
    # subsumes the others -- keep it the only flag set, so the report names the actual cause.
    for name in FLAG_NAMES:
        if name != "nonfinite":
            flags[name] &= ~flags["nonfinite"]
    return flags


def _flat_pose(n: int, T: int, x0: float = 0.0) -> np.ndarray:
    """[T, n, 7] identity-rotation poses creeping along +X from x0 -- the self-test's baseline."""
    pose = np.zeros((T, n, 7))
    pose[:, :, 0] = x0 + np.linspace(0.0, 1.0, T)[:, None]
    pose[:, :, 6] = 1.0
    return pose


if __name__ == "__main__":
    from feasibility.submodule_test.scenarios import flat_terrain

    n, T = 3, 20
    terrain = flat_terrain()
    bounds = plausible_bounds(terrain)
    spawn_xy = np.zeros((n, 2))
    common = dict(
        spawn_xy=spawn_xy,
        max_displacement=np.full(n, 5.0),
        bounds=bounds,
        ostrich_dt=0.03,
        hstack_dt=0.1,
        v_drive=np.ones(n),
    )

    good = _flat_pose(n, T)
    clean = health_flags(
        ostrich_pose=good, hstack_pose=good, residual=np.zeros((T, n)), **common
    )
    for name, flag in clean.items():
        assert not flag.any(), f"clean rollout tripped {name}: {flag}"
    print(f"  clean rollout          {'no flags':30s} OK")

    for label, mutate, expect in [
        ("nonfinite", lambda p: p.__setitem__((5, 0, 2), np.nan), "nonfinite"),
        ("off-map", lambda p: p.__setitem__((slice(None), 0, 1), 99.0), "out_of_bounds"),
        ("sunk", lambda p: p.__setitem__((slice(None), 0, 2), -23.0), "out_of_bounds"),
        ("teleport", lambda p: p.__setitem__((-1, 0, 0), 9.0), "displaced"),
        ("spike", lambda p: p.__setitem__((10, 0, 0), 3.0), "speed_spike"),
    ]:
        pose = _flat_pose(n, T)
        mutate(pose)
        got = health_flags(ostrich_pose=pose, hstack_pose=good, residual=np.zeros((T, n)), **common)
        assert got[expect][0], f"{label} did not trip {expect}"
        assert not got[expect][1:].any(), f"{label} tripped {expect} on an untouched trial"
        print(f"  {label:22s} {expect:30s} OK")

    resid = np.zeros((T, n))
    resid[3, 1] = HSTACK_RESID_TOL * 10
    got = health_flags(ostrich_pose=good, hstack_pose=good, residual=resid, **common)
    assert got["hstack_residual"].tolist() == [False, True, False]
    print(f"  {'bad hstack residual':22s} {'hstack_residual':30s} OK")

    # nonfinite must be the ONLY flag on a NaN run -- the report should name the cause, not five
    # downstream symptoms.
    pose = _flat_pose(n, T)
    pose[:, 0, :] = np.nan
    got = health_flags(ostrich_pose=pose, hstack_pose=good, residual=np.zeros((T, n)), **common)
    assert got["nonfinite"][0] and not any(got[f][0] for f in FLAG_NAMES if f != "nonfinite")
    print(f"  {'all-NaN trial':22s} {'nonfinite only':30s} OK")

    m = compute_metrics(
        ostrich_pose=good,
        ostrich_wheel_qd=np.full((T, n, 3), 2.0),
        hstack_pose=good,
        hstack_wheel_qd=np.full((T, n, 3), 2.0),
        clearance=np.full((T, n), 0.2),
        residual=np.zeros((T, n)),
        spawn_xy=spawn_xy,
    )
    assert m.shape == (n, len(METRIC_NAMES))
    named = dict(zip(METRIC_NAMES, m[0]))
    assert np.isclose(named["e_pos"], 0.0) and np.isclose(named["e_rot_deg"], 0.0)
    assert np.isclose(named["o_path_len"], 1.0) and np.isclose(named["o_displacement"], 1.0)
    assert np.isclose(named["h_clearance_below_frac"], 0.0)
    print(f"  {'identical rollouts':22s} {'e_pos = e_rot = 0':30s} OK")

    print(f"\nmetrics OK ({len(METRIC_NAMES)} metrics, {len(FLAG_NAMES)} flags)")
