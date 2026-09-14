"""Static settle of helhest_stack's kinematic twin, and the planner's feasibility test on it.

Shared by `generate_dataset.py` (the arc-endpoint reference and the swept_clear split) and
`spawn_sampling.py` (which spawn poses are valid at all), so both use one definition of "feasible".
Re-derives `costtogo.py`'s own `ForwardSimulator` settle convention rather than importing anything
from `helhest.planning`, since that machinery is embedded in `CostToGo`'s captured-graph setup and
isn't meant to be called standalone per trial. Works on CPU as well as CUDA.
"""
from __future__ import annotations

import numpy as np

from helhest import dynamics
from helhest import friction as friction_mod
from helhest.engine import ForwardSimulator
from helhest.engine import RobotParams

from feasibility.heightmap import HeightMapReader


def settle_batch(
    terrain: HeightMapReader, poses: np.ndarray, mu: float, device: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Static (zero-command) settle of the twin at `poses` [n, 3] = (x, y, yaw) -- design.md
    section 1b's producer of (z, pitch, roll). `n_steps=1` + a zeroed `target_wheel_omega`
    reproduces exactly `costtogo.py`'s own `settle_sim` convention: row 0 of `derived` (the
    pre-step state) IS the static settle, independent of n_steps or command (see costtogo.py's
    `_feasibility_kernel` comment "row 0 = the static settle").
    Returns (derived [n, 3] = (z, pitch, roll), residual [n], clearance [n])."""
    elevation, grid = terrain.to_hstack(device)
    mu_xlim = (terrain.x0, terrain.x0 + (terrain.nx - 1) * terrain.cell)
    mu_ylim = (terrain.y0, terrain.y0 + (terrain.ny - 1) * terrain.cell)
    mu_field = friction_mod.uniform(mu, xlim=mu_xlim, ylim=mu_ylim, cell=terrain.cell)
    n = poses.shape[0]
    sim = ForwardSimulator(
        dynamics.robot_params(), dynamics.execution_solver(), grid, batch_size=n, n_steps=1,
        device=device,
    )
    sim.set_terrain(elevation)
    sim.set_friction(mu_field)
    sim.start_pose.assign(np.ascontiguousarray(poses, np.float32))
    sim.target_wheel_omega.zero_()
    sim.rollout_launch()
    derived = sim.derived.numpy()[0]  # [n, 3]
    residual = sim.residual.numpy()[0]  # [n]
    clearance = sim.clearance.numpy()[0]  # [n]
    return derived, residual, clearance


def settle_feasible(
    derived: np.ndarray, residual: np.ndarray, clearance: np.ndarray, robot: RobotParams
) -> np.ndarray:
    """[n] bool -- reproduces costtogo.py's `_feasibility_kernel` OR, in numpy: direction-aware
    envelope (climb = nose-up = NEGATIVE pitch) plus residual/clearance thresholds."""
    pitch, roll = derived[:, 1], derived[:, 2]
    over_envelope = (
        (np.abs(roll) > robot.max_roll)
        | (pitch < -robot.max_pitch_up)
        | (pitch > robot.max_pitch_down)
    )
    return ~(over_envelope | (residual > robot.resid_tol) | (clearance < robot.clear_margin))
