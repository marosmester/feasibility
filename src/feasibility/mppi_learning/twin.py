"""The kinematic twin's end poses under a sampled command profile (design.md section 8, step 2).

This is the `mppi_learning` label's helhest_stack half: `lattice_learning` compared ostrich to an
ideal arc + settle, here it is compared to what one MPPI rollout computes. So the twin is set up
exactly as `MppiGpu`'s `plan_sim` is on the robot:

  * `dynamics.planning_solver` (motor lag `MOTOR_TAU`, shallow settle), NOT the execution solver
    `comparator.common.run_hstack_batch` uses -- that one has no motor lag, so its rollouts are not
    the ones the cost will be charged on;
  * it starts from the REAL state at the window start -- ostrich's pose (x, y, yaw), wheel speeds
    and body twist after the warm-up -- the way the ROS node seeds `start_pose`,
    `set_initial_wheel_omega` and `set_initial_twist` before every replan. The warm-up is ostrich's
    alone: running the twin through it too would put the twin's own warm-up drift into the label,
    which no MPPI rollout ever has.

Usage:
    python src/feasibility/mppi_learning/twin.py            # flat-ground smoke test, needs a GPU
"""
from __future__ import annotations

import numpy as np
from helhest import dynamics
from helhest import friction as friction_mod
from helhest.engine import ForwardSimulator
from helhest.engine import SolverParams

from feasibility.comparator.common import euler_zyx_to_quat_xyzw
from feasibility.heightmap import HeightMapReader
from feasibility.mppi_learning.command import HALF_TRACK
from feasibility.mppi_learning.command import MPPI_DT
from feasibility.mppi_learning.command import WHEEL_RADIUS
from feasibility.mppi_learning.command import WINDOW_STEPS


def twist_from_wheels(wheel_omega: np.ndarray, alpha: float) -> np.ndarray:
    """[N, 3] realized wheel speeds -> [N, 3] body twist (vx, vy, yaw_rate): the batched
    `dynamics.twist_from_wheels`, the node's `init_twist` proxy when no measured twist is fresh."""
    wl, wr = wheel_omega[:, 0], wheel_omega[:, 1]
    vx = WHEEL_RADIUS * (wl + wr) / 2.0
    wz = WHEEL_RADIUS * (wr - wl) / (2.0 * HALF_TRACK * alpha)
    return np.stack([vx, np.zeros_like(vx), wz], axis=-1).astype(np.float32)


def run_twin(
    terrain: HeightMapReader,
    start_pose: np.ndarray,
    wheel_omega: np.ndarray,
    init_wheel_omega: np.ndarray | None = None,
    init_twist: np.ndarray | None = None,
    mu: float = 0.8,
    k_turn: float = dynamics.K_TURN,
    device: str = "cuda:0",
    solver: SolverParams | None = None,
    diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`start_pose` [N, 3] (x, y, yaw) at the window start; `wheel_omega` [n_windows *
    WINDOW_STEPS, N, 3] setpoints on the 0.1 s grid; `init_wheel_omega` [N, 3] the realized wheel
    speeds there (None = at rest); `init_twist` [N, 3] body (vx, vy, yaw_rate) there (None = derived
    from the wheels as the node does, `alpha = 1 + k_turn * mu`). `solver` None is MPPI's
    `planning_solver(k_turn=k_turn)`.

    Returns the twin's pose at the end of each window, [n_windows, N, 7] (x, y, z, qx, qy, qz,
    qw) -- the layout of ostrich's pose log, so `custom_dataset.pose_to_se3` reads both.
    `diagnostics` also returns the settle's min clearance and max residual over each window,
    [n_windows, N] each -- where the twin itself says its pose is fiction (high-centering, an
    unconverged settle)."""
    n_steps, n, _ = wheel_omega.shape
    if n_steps % WINDOW_STEPS:
        raise ValueError(f"{n_steps} steps is not a whole number of windows")
    solver = dynamics.planning_solver(k_turn=k_turn) if solver is None else solver
    if abs(solver.dt - MPPI_DT) > 1e-9:
        raise ValueError(f"solver dt {solver.dt} is not the MPPI step {MPPI_DT}")
    wheels = np.zeros((n, 3), np.float32) if init_wheel_omega is None else np.asarray(init_wheel_omega, np.float32)
    twist = twist_from_wheels(wheels, 1.0 + k_turn * mu) if init_twist is None else np.asarray(init_twist, np.float32)

    elevation, grid = terrain.to_hstack(device)
    mu_field = friction_mod.uniform(
        mu,
        xlim=(terrain.x0, terrain.x0 + (terrain.nx - 1) * terrain.cell),
        ylim=(terrain.y0, terrain.y0 + (terrain.ny - 1) * terrain.cell),
        cell=terrain.cell,
    )
    sim = ForwardSimulator(dynamics.robot_params(), solver, grid, batch_size=n, n_steps=n_steps, device=device)
    if sim.command_delay_steps > 0:
        # At DT 0.1 the measured 0.04 s delay rounds to 0 steps; a solver where it does not would
        # need the warm-up's in-flight commands here, as the node's `_load_command_history` does.
        raise NotImplementedError(f"command_delay_steps = {sim.command_delay_steps} needs a command history")
    sim.set_terrain(elevation)
    sim.set_friction(mu_field)
    sim.start_pose.assign(np.ascontiguousarray(start_pose, np.float32))
    sim.target_wheel_omega.assign(np.ascontiguousarray(wheel_omega, np.float32))
    sim.set_initial_wheel_omega(wheels)
    sim.set_initial_twist(twist)
    sim.rollout_launch()

    end = WINDOW_STEPS * (1 + np.arange(n_steps // WINDOW_STEPS))  # row 0 of the logs is the start
    controlled, derived = sim.controlled.numpy()[end], sim.derived.numpy()[end]
    quat = euler_zyx_to_quat_xyzw(controlled[..., 2], derived[..., 1], derived[..., 2])
    pose = np.concatenate([controlled[..., :1], controlled[..., 1:2], derived[..., :1], quat], axis=-1)
    if not diagnostics:
        return pose
    per_window = (n_steps // WINDOW_STEPS, WINDOW_STEPS, n)
    clearance = sim.clearance.numpy().reshape(per_window).min(axis=1)
    residual = sim.residual.numpy().reshape(per_window).max(axis=1)
    return pose, clearance, residual


if __name__ == "__main__":
    from feasibility.comparator.common import cmd_to_wheels
    from feasibility.comparator.common import init_warp_device

    device = "cuda:0"
    init_warp_device(device)
    flat = HeightMapReader.flat(xlim=(-4.0, 8.0), ylim=(-4.0, 4.0), cell=0.1)
    origin = np.zeros((2, 3))
    v = 0.5
    straight = np.tile(np.asarray(cmd_to_wheels(v, 0.0), np.float32), (2 * WINDOW_STEPS, 2, 1))

    # Straight at 0.5 m/s, row 0 from rest and row 1 already at speed. From rest the motor lag
    # costs about v * MOTOR_TAU in the first window; at speed each window is v * T, laterally
    # exact and level.
    pose = run_twin(flat, origin, straight, init_wheel_omega=np.stack([np.zeros(3), straight[0, 1]]), device=device)
    assert pose.shape == (2, 2, 7)
    moving = pose[:, 1, 0]
    assert np.allclose(moving, [v, 2 * v], atol=0.02), moving
    lag = v - pose[0, 0, 0]
    assert 0.5 * v * dynamics.MOTOR_TAU < lag < 2.0 * v * dynamics.MOTOR_TAU, lag
    assert np.abs(pose[..., 1]).max() < 1e-3 and np.abs(pose[..., 3:5]).max() < 1e-3
    again, clearance, residual = run_twin(flat, origin, straight, init_wheel_omega=np.stack([np.zeros(3), straight[0, 1]]), device=device, diagnostics=True)
    assert np.array_equal(again, pose) and clearance.shape == residual.shape == (2, 2)
    assert (clearance > 0).all() and (residual < dynamics.robot_params().resid_tol).all(), (clearance, residual)

    # A spin turns in place, slowed by the turn resistance alpha = 1 + k_turn * mu.
    mu = 0.8
    spin_w = np.array([-3.0, 3.0, 0.0], np.float32)
    spin = np.tile(spin_w, (WINDOW_STEPS, 1, 1))
    end = run_twin(flat, origin[:1], spin, init_wheel_omega=spin_w[None], mu=mu, device=device)[0, 0]
    yaw = 2 * np.arctan2(end[5], end[6])
    ideal = WHEEL_RADIUS * 6.0 / (2 * HALF_TRACK * (1 + dynamics.K_TURN * mu)) * WINDOW_STEPS * MPPI_DT
    assert abs(yaw - ideal) < 0.1 * ideal and np.hypot(end[0], end[1]) < 0.5, (yaw, ideal, end[:2])
    print(f"OK  at speed {moving[0]:.3f} m/window, from rest -{lag:.3f} m (motor lag), "
          f"spin {np.degrees(yaw):.0f} deg vs {np.degrees(ideal):.0f} ideal/alpha")
