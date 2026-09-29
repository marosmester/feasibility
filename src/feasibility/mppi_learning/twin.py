"""The kinematic twin's end poses under a sampled command profile (design.md section 8, step 2).

This is the `mppi_learning` label's helhest_stack half: `lattice_learning` compared ostrich to an
ideal arc + settle, here it is compared to what MPPI's rollouts actually run --
`ForwardSimulator` at `dynamics.DT` with the execution solver's ICR slip. It adds
nothing to `comparator.common.run_hstack_batch` but the window bookkeeping: an optional warm-up
that brings the twin to the entry speed, then the pose at the end of every window.

The twin starts at rest (`run_hstack_batch` zeroes its wheel state), so a warm-up profile is the
only way to give it an entry speed; the caller passes the SAME profile it gives ostrich, on the
twin's 0.1 s grid.

Usage:
    python src/feasibility/mppi_learning/twin.py            # flat-ground smoke test, needs a GPU
"""
from __future__ import annotations

import numpy as np
from helhest import dynamics

from feasibility.comparator.common import euler_zyx_to_quat_xyzw
from feasibility.comparator.common import run_hstack_batch
from feasibility.heightmap import HeightMapReader
from feasibility.mppi_learning.command import MPPI_DT
from feasibility.mppi_learning.command import WINDOW_STEPS


def run_twin(
    terrain: HeightMapReader,
    spawn_pose: np.ndarray,
    wheel_omega: np.ndarray,
    warmup_wheel_omega: np.ndarray | None = None,
    mu: float = 0.8,
    k_turn: float = dynamics.K_TURN,
    device: str = "cuda:0",
) -> np.ndarray:
    """`spawn_pose` [N, 3] (x, y, yaw); `wheel_omega` [n_windows * WINDOW_STEPS, N, 3] setpoints on
    the 0.1 s grid; `warmup_wheel_omega` [W, N, 3] the pre-roll driving up to the entry speed.
    Returns the twin's pose at the end of each window, [n_windows, N, 7] (x, y, z, qx, qy, qz, qw)
    -- the layout of ostrich's pose log, so `custom_dataset.pose_to_se3` reads both."""
    if wheel_omega.shape[0] % WINDOW_STEPS:
        raise ValueError(f"{wheel_omega.shape[0]} steps is not a whole number of windows")
    n_warmup = 0 if warmup_wheel_omega is None else warmup_wheel_omega.shape[0]
    setpoints = wheel_omega if n_warmup == 0 else np.concatenate([warmup_wheel_omega, wheel_omega])

    controlled, derived, *_ = run_hstack_batch(
        np.ascontiguousarray(setpoints, np.float32), terrain, MPPI_DT, k_turn, mu, device, spawn_pose
    )
    end = n_warmup + WINDOW_STEPS * (1 + np.arange(wheel_omega.shape[0] // WINDOW_STEPS)) - 1
    x, y, yaw = (controlled[end, :, k] for k in range(3))
    z, pitch, roll = (derived[end, :, k] for k in range(3))
    quat = euler_zyx_to_quat_xyzw(yaw, pitch, roll)  # [n_windows, N, 4]
    return np.concatenate([np.stack([x, y, z], axis=-1), quat], axis=-1)


if __name__ == "__main__":
    from feasibility.comparator.common import build_setpoints
    from feasibility.comparator.common import init_warp_device

    device = "cuda:0"
    init_warp_device(device)
    flat = HeightMapReader.flat(xlim=(-4.0, 8.0), ylim=(-4.0, 4.0), cell=0.1)

    # Straight, constant 0.5 m/s: the twin has no command delay by default, so each window
    # advances v * T, laterally exact, and the pose is level.
    v = 0.5
    steps = build_setpoints(MPPI_DT, v, 0.0, 2 * WINDOW_STEPS * MPPI_DT)[:, None, :]
    pose = run_twin(flat, np.zeros((1, 3)), steps, device=device)
    assert pose.shape == (2, 1, 7)
    assert np.allclose(pose[:, 0, 0], [v, 2 * v], atol=0.03), pose[:, 0, 0]
    assert abs(pose[:, 0, 1]).max() < 1e-3

    # A warm-up is prepended, not counted as a window: the window ends land at 3 s, not 1 s.
    warm = build_setpoints(MPPI_DT, v, 0.0, 2.0)[:, None, :]
    fast = run_twin(flat, np.zeros((1, 3)), steps[:WINDOW_STEPS], warmup_wheel_omega=warm, device=device)
    assert fast.shape == (1, 1, 7) and abs(fast[0, 0, 0] - 3 * v) < 0.03, fast[0, 0, 0]

    # A spin turns in place: the reference point (off the ICR) stays within a wheel-lever of the origin while the yaw changes.
    spin = np.tile(np.array([-3.0, 3.0, 0.0], np.float32), (WINDOW_STEPS, 1, 1))
    end = run_twin(flat, np.zeros((1, 3)), spin, device=device)[0, 0]
    yaw = 2 * np.arctan2(end[5], end[6])
    assert abs(yaw) > 0.5 and np.hypot(end[0], end[1]) < 0.5, (yaw, end[:2])
    print(f"OK  straight {pose[0, 0, 0]:.3f} m/window, with warm-up {fast[0, 0, 0]:.3f} m, spin yaw {np.degrees(yaw):.0f} deg")
