"""Verdict on one ostrich world that followed a planned path with `pure_pursuit`: arrived / flipped /
off_map / stalled / nonfinite, peak climb pitch and |roll|, cross-track error to the planned path.

`python src/feasibility/planning/evaluation.py` is the smoke test (synthetic poses, no GPU).
"""
from __future__ import annotations

import math

import numpy as np

from feasibility.heightmap import HeightMapReader
from feasibility.planning.pure_pursuit import GOAL_TOLERANCE
from feasibility.planning.pure_pursuit import STOP_RADIUS

OFF_MAP_MARGIN = 0.5  # [m] from any grid edge


def pitch_roll(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """q [..., 4] (qx, qy, qz, qw) -> (pitch, roll, up_z), R = Rz@Ry@Rx. pitch is nose-up NEGATIVE
    (helhest_stack's convention); up_z is the chassis up-vector's world z (< 0 = upside down)."""
    qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r20 = 2.0 * (qx * qz - qw * qy)
    r21 = 2.0 * (qy * qz + qw * qx)
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
    return np.arcsin(np.clip(-r20, -1.0, 1.0)), np.arctan2(r21, r22), r22


def track_error(xy: np.ndarray, path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """xy [T, 2] -> (distance to the polyline [T], arc length of the closest point [T])."""
    a, b = path[:-1], path[1:]
    ab = b - a
    seg_len = np.linalg.norm(ab, axis=1)
    t = np.einsum("tsk,sk->ts", xy[:, None] - a[None], ab) / np.maximum(seg_len ** 2, 1e-12)
    t = np.clip(t, 0.0, 1.0)
    closest = a[None] + t[..., None] * ab[None]
    d = np.linalg.norm(xy[:, None] - closest, axis=-1)
    k = np.argmin(d, axis=1)
    s0 = np.concatenate([[0.0], np.cumsum(seg_len)])
    rows = np.arange(len(xy))
    return d[rows, k], s0[k] + t[rows, k] * seg_len[k]


def judge(pose: np.ndarray, dt: float, path: np.ndarray, terrain: HeightMapReader) -> dict:
    """pose [T, 7] of one world (settle already sliced off) -> verdict + metrics, measured up to
    arrival (or the first non-finite row). Arrival is the controller's own stop condition."""
    finite = np.isfinite(pose).all(axis=1)
    end = len(pose) if finite.all() else int(np.argmin(finite))
    p = pose[: max(end, 1)]
    goal = path[-1]
    d_goal = np.linalg.norm(p[:, :2] - goal, axis=1)
    at_end = np.argmin(np.linalg.norm(p[:, None, :2] - path[None], axis=-1), axis=1) == len(path) - 1
    arrived = np.nonzero((d_goal < STOP_RADIUS) | (at_end & (d_goal < GOAL_TOLERANCE)))[0]
    if len(arrived):
        p = p[: arrived[0] + 1]
    pitch, roll, up_z = pitch_roll(p[:, 3:7])
    flip = np.nonzero(up_z < 0.0)[0]
    if len(flip):  # metrics stop where the chassis goes past upside down; the tumble is not tracking
        p, pitch, roll, up_z = p[: flip[0] + 1], pitch[: flip[0] + 1], roll[: flip[0] + 1], up_z[: flip[0] + 1]
    x_lo, x_hi = terrain.x0 + OFF_MAP_MARGIN, terrain.x0 + terrain.nx * terrain.cell - OFF_MAP_MARGIN
    y_lo, y_hi = terrain.y0 + OFF_MAP_MARGIN, terrain.y0 + terrain.ny * terrain.cell - OFF_MAP_MARGIN
    off = (p[:, 0] < x_lo) | (p[:, 0] > x_hi) | (p[:, 1] < y_lo) | (p[:, 1] > y_hi)
    if (up_z < 0.0).any():
        status = "flipped"
    elif off.any():
        status = "off_map"
    elif len(arrived):
        status = "arrived"
    elif not finite.all():
        status = "nonfinite"
    else:
        status = "stalled"
    cte, along = track_error(p[:, :2].astype(np.float64), path)
    i_climb = int(np.argmax(-pitch))
    return dict(
        status=status,
        t_arrive=float(arrived[0] * dt) if status == "arrived" else math.nan,
        climb_deg=float(np.degrees(max(-pitch[i_climb], 0.0))),
        roll_deg=float(np.degrees(np.abs(roll).max())),
        cte_max=float(cte.max()),
        cte_mean=float(cte.mean()),
        cte_at_climb=float(cte[i_climb]),
        progress_m=float(along.max()),
        xy=p[:, :2],
    )


if __name__ == "__main__":
    terrain = HeightMapReader.flat(xlim=(-5.0, 5.0), ylim=(-5.0, 5.0))
    path = np.stack([np.linspace(-3.0, 3.0, 121), np.zeros(121)], axis=-1)
    level = np.array([0.0, 0.0, 0.0, 1.0])
    T = 200

    def poses(xs: np.ndarray, ys: np.ndarray, q: np.ndarray = level) -> np.ndarray:
        return np.column_stack([xs, ys, np.zeros(len(xs)), np.tile(q, (len(xs), 1))])

    r = judge(poses(np.linspace(-3.0, 3.0, T), np.full(T, 0.1)), 0.05, path, terrain)
    assert r["status"] == "arrived" and abs(r["cte_max"] - 0.1) < 1e-9, r
    # passes 0.2 m beside the goal: outside STOP_RADIUS, inside GOAL_TOLERANCE at the last waypoint
    r = judge(poses(np.linspace(-3.0, 3.5, T), np.full(T, 0.2)), 0.05, path, terrain)
    assert r["status"] == "arrived", r
    r = judge(poses(np.linspace(-3.0, 0.0, T), np.zeros(T)), 0.05, path, terrain)
    assert r["status"] == "stalled" and abs(r["progress_m"] - 3.0) < 1e-6, r
    upside_down = np.array([1.0, 0.0, 0.0, 0.0])  # 180 deg about x
    r = judge(poses(np.linspace(-3.0, 0.0, T), np.zeros(T), upside_down), 0.05, path, terrain)
    assert r["status"] == "flipped", r
    pitch, _, _ = pitch_roll(np.array([0.0, -math.sin(0.25), 0.0, math.cos(0.25)]))  # -0.5 rad about y
    assert abs(pitch + 0.5) < 1e-9, pitch  # nose up is negative
    print("all self-checks ok")
