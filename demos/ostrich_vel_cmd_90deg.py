"""Slip-COMPENSATED variant of ostrich_vel_cmd.py: same straight/arc/straight
schedule, robot/friction/gain setup, but the arc phase's duration is stretched
so the robot's ACTUAL heading (not the ideal no-slip one) ends up at 90 deg.

ostrich_vel_cmd.py is deliberately open-loop/uncompensated — it measures slip
as a quantity of interest. This script is the opposite use case: you don't
care about slip or about K_P/mu being realistic, you just want a command
sequence that drives the robot to a real 90-degree turn in this simulator, to
use as a canned trajectory (e.g. as an input for another script/comparison).

T_ARC below was found empirically by false-position search directly against
this file's K_P/friction/dt (bisecting duration against achieved yaw from
ostrich_vel_cmd.HelhestVelCmdSimulator.replay()), converging to 89.67 deg
(commanded 90) after 4 iterations. It is NOT derived from any formula and is
only valid for this exact (K_P, mu_front, mu_rear, dt, YAW_RATE, V_DRIVE)
combination — change any of those and T_ARC needs to be re-tuned the same way.

Usage:
    python demos/ostrich_vel_cmd_90deg.py                       # GL viewer
    python demos/ostrich_vel_cmd_90deg.py rendering=headless     # batch, no window
"""
import math
import pathlib

import examples
import hydra
import numpy as np
from omegaconf import DictConfig

from ostrich import EngineConfig
from ostrich import LoggingConfig
from ostrich import RenderingConfig
from ostrich import SimulationConfig

try:
    from demos.ostrich_vel_cmd import HelhestVelCmdSimulator, K_P, WHEEL_RADIUS, HALF_TRACK
    from demos.ostrich_vel_cmd import cmd_to_wheels, yaw_from_quat_xyzw
except ModuleNotFoundError:
    from ostrich_vel_cmd import HelhestVelCmdSimulator, K_P, WHEEL_RADIUS, HALF_TRACK
    from ostrich_vel_cmd import cmd_to_wheels, yaw_from_quat_xyzw

CONFIG_PATH = pathlib.Path(examples.__file__).parent.joinpath("conf")

# --- Command schedule: (duration_s, v [m/s], omega [rad/s], CCW+) ---
V_DRIVE = 1.0
YAW_RATE = 0.5
T_ARC = 5.0318  # empirically tuned, see module docstring — NOT (pi/2)/YAW_RATE
PHASES = [
    (2.0, V_DRIVE, 0.0),
    (T_ARC, V_DRIVE, YAW_RATE),
    (2.0, V_DRIVE, 0.0),
]


def build_setpoints(dt: float) -> np.ndarray:
    blocks = []
    for duration_s, v, wz in PHASES:
        n = int(round(duration_s / dt))
        blocks.append(np.tile(cmd_to_wheels(v, wz), (n, 1)))
    return np.concatenate(blocks, axis=0).astype(np.float32)


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest", version_base=None)
def ostrich_vel_cmd_90deg(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    setpoints = build_setpoints(sim_config.target_timestep_seconds)

    sim = HelhestVelCmdSimulator(
        sim_config, render_config, engine_config, logging_config, k_p=K_P
    )
    if sim_config.use_cuda_graph:
        poses, wheel_qd = sim.replay_graph(setpoints)
    else:
        poses, wheel_qd = sim.replay(setpoints)

    if render_config.vis_type == "gl":
        k = len(setpoints) - 1
        while sim.viewer.is_running():
            sim._maybe_render(k)

    achieved_yaw = yaw_from_quat_xyzw(poses[-1, 3:7]) - yaw_from_quat_xyzw(poses[0, 3:7])
    print(f"achieved yaw : {math.degrees(achieved_yaw):6.2f} deg  (target 90.0)")

    out = pathlib.Path(cfg.get("out", "/tmp/ostrich_vel_cmd_90deg.npz"))
    dt = sim_config.target_timestep_seconds
    np.savez_compressed(
        out,
        dt=np.float32(dt),
        t=np.arange(len(setpoints), dtype=np.float32) * dt,
        cmd_wheel_omega=setpoints,
        pose=poses,
        wheel_qd=wheel_qd,
        phases=np.array(PHASES, dtype=np.float32),
    )
    print(f"saved {out}")


if __name__ == "__main__":
    ostrich_vel_cmd_90deg()
