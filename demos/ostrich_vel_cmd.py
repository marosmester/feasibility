"""Open-loop velocity-command demo: Helhest Junior on flat ground in Ostrich.

Drives the robot with a fixed, scripted (non-interactive) sequence of body-twist
commands (v, omega) — straight, arc left, straight — nominally ending 90 degrees
rotated from the start heading. No obstacles: this isolates wheel-ground slip as
the only source of trajectory error, ahead of a later comparison against
helhest_stack's kinematic twin (that comparison is a separate script).

Commands are DELIBERATELY not slip-compensated: they're computed from ideal
no-slip differential-drive kinematics, so a perfectly-gripping robot would land
at exactly 90 degrees. Ostrich is a skid-steer and will under-rotate — that
shortfall is the quantity of interest (helhest_stack models the same effect as
`alpha = 1 + k_turn*mu` in engine/step.py, ~1.48 at mu=0.8), not a bug to hide.

Measured at mu_front=mu_rear=0.8 and dt=3e-2 (this file's defaults): 40.3 of the
commanded 90 deg, implied alpha 2.24, with the wheels tracking their setpoints to
within 3 percent — so that shortfall really is wheel-ground slip.

Getting there required K_P (see below). An earlier revision of this file ran at
replay_real.py's inherited k_p=250 and reported "only a few degrees of the
commanded 90"; that was a saturated velocity servo, not physics, and any
conclusion drawn from it about Ostrich disagreeing with helhest_stack is void.

Two caveats on comparing against helhest_stack's alpha = 1 + k_turn*mu (~1.48 at
mu=0.8), both open:

  * alpha here is NOT constant in the commanded yaw rate — ~2.24 at YAW_RATE=0.3
    but ~1.4 at YAW_RATE=1.0. Plausibly real skid-steer behaviour (the rear
    wheel's lateral Coulomb resistance is roughly yaw-rate independent, so it
    eats a larger fraction of a gentler turn's smaller differential thrust), but
    a single-constant alpha cannot represent it. Fix the yaw rate before quoting
    any calibrated number.
  * dt convergence is improved but not settled: at YAW_RATE=1.0, alpha runs
    1.64 / 1.47 / 1.88 / 1.95 for dt = 5e-2 / 3e-2 / 1e-2 / 5e-3. That is a
    physically sensible band (it was 15-40 at k_p=250) yet still drifting, so
    "achieved yaw" remains specific to the dt it was measured at.

Usage:
    python demos/ostrich_vel_cmd.py                       # GL viewer
    python demos/ostrich_vel_cmd.py rendering=headless     # batch, no window
    python demos/ostrich_vel_cmd.py +out=/tmp/run1.npz
    python demos/ostrich_vel_cmd.py +terrain_path=assets/ramp  # non-flat terrain
"""
import math
import pathlib
from typing import override

import examples
import hydra
import newton
import numpy as np
import warp as wp
from examples.helhest_junior.replay_real import HelhestJuniorReplaySimulator
from ostrich import EngineConfig
from ostrich import LoggingConfig
from ostrich import RenderingConfig
from ostrich import SimulationConfig
from omegaconf import DictConfig

# Works both as `python -m demos.ostrich_vel_cmd` (CWD on sys.path, `demos` resolves
# as a namespace package) and as `python demos/ostrich_vel_cmd.py` (only `demos/`
# itself is on sys.path, so the package-qualified name is not importable).
try:
    from demos.helhest_common import create_helhest_junior_model
except ModuleNotFoundError:
    from helhest_common import create_helhest_junior_model

from feasibility.heightmap import HeightMapReader

CONFIG_PATH = pathlib.Path(examples.__file__).parent.joinpath("conf")

# Robot constants (demos/helhest_common.py HelhestJuniorConfig).
WHEEL_RADIUS = 0.35  # [m]
HALF_TRACK = 0.365  # [m], (LEFT_WHEEL_POS.y - RIGHT_WHEEL_POS.y) / 2

# Wheel velocity-servo gain. Ostrich's TARGET_VELOCITY control constraint is a
# pure P law with no torque limit (control_constraint.py: alpha = 1/(dt*ke)
# => tau = k_p*(target - qd)), so k_p alone caps the torque the servo can ever
# apply: tau_max = k_p * (largest reachable velocity error), and that error
# saturates at the commanded wheel differential YAW_RATE*HALF_TRACK/WHEEL_RADIUS
# (the wheel simply keeps rolling at the straight-line speed v/WHEEL_RADIUS).
# replay_real's inherited k_p=250 is therefore far too soft to hold an arc: at
# YAW_RATE=1.0 the differential is 1.04 rad/s, ceilings the servo at ~260 Nm,
# and the robot yaws 2.3 deg of the commanded 90 while both front wheels sit at
# the straight-line speed. 1e4 is the knee of a k_p sweep at that yaw rate —
# achieved yaw 63.7 / 66.5 / 70.3 / 68.0 deg at k_p = 1e4 / 2.5e4 / 1e5 / 1e6,
# flat from here on — so the residual shortfall is ground slip, not motor droop.
#
# NOT a calibrated motor spec: no torque measurement for Helhest Junior exists in
# either repo. Note 1e4 lets the servo demand far more than the front wheels can
# transmit (friction budget mu*N*r = 0.8*39.1*9.81*0.35 ~ 107 Nm each), so this
# models a stiff velocity source, not a real drivetrain. Use it to isolate
# ground slip; a torque-limited actuator would be needed for motor realism.
K_P = 15000.0

# Terrain extent must cover the whole commanded path: add_shape_heightfield is
# a FINITE shape, unlike the add_ground_plane it replaces -- a query outside
# the extent gets pushed to "no contact" (sdf_contact.py's intentional
# ghost-contact-avoidance tradeoff at the footprint boundary), so a robot that
# drives past the edge falls through. PHASES below covers ~9 m forward with a
# 2 m-radius arc; these are generous, hand-tuned padding, not derived from it.
TERRAIN_XLIM_M = (-2.0, 12.0)
TERRAIN_YLIM_M = (-6.0, 6.0)
TERRAIN_CELL = 0.05

# Where the robot spawns in world XY -- independent of the terrain's own
# origin. PHASES below displaces the robot by roughly x:[0,4] y:[0,4.8]
# relative to this point, so keep it clear of the terrain's finite edges.
SPAWN_X_M = 1.0
SPAWN_Y_M = 3.0

# --- Command schedule: (duration_s, v [m/s], omega [rad/s], CCW+) ---
V_DRIVE = 1.0
YAW_RATE = 0.5  # -> 2.5 m nominal turn radius
T_ARC = 5.0318  # empirically tuned, instead of the (math.pi / 2.0) / YAW_RATE  
PHASES = [
    (0.5, V_DRIVE, 0.0),
    (T_ARC, V_DRIVE, YAW_RATE),
    (2.0, V_DRIVE, 0.0),
]


def nominal_endpoint(phases: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    """Zero-slip (x, y, yaw) reached by chaining straight/constant-curvature-arc
    phases, each integrated in the heading frame left by the previous phase.
    Used only for the console's commanded-vs-achieved comparison below."""
    x, y, psi = 0.0, 0.0, 0.0
    for duration_s, v, wz in phases:
        if abs(wz) < 1e-9:
            x += v * duration_s * math.cos(psi)
            y += v * duration_s * math.sin(psi)
        else:
            r = v / wz
            dpsi = wz * duration_s
            x += r * (math.sin(psi + dpsi) - math.sin(psi))
            y += r * (math.cos(psi) - math.cos(psi + dpsi))
            psi += dpsi
    return x, y, psi


NOMINAL_X, NOMINAL_Y, NOMINAL_YAW_RAD = nominal_endpoint(PHASES)


def cmd_to_wheels(v: float, wz: float) -> tuple[float, float, float]:
    """Ideal no-slip differential drive: body twist (v, omega) -> per-wheel rad/s
    [left, right, rear]. Rear wheel is on the centerline, so it just sees v."""
    v_l = (v - wz * HALF_TRACK) / WHEEL_RADIUS
    v_r = (v + wz * HALF_TRACK) / WHEEL_RADIUS
    v_rear = v / WHEEL_RADIUS
    return v_l, v_r, v_rear


def build_setpoints(dt: float) -> np.ndarray:
    """Sample PHASES onto the sim's dt grid. [T, 3] float32, dt-independent by
    construction so the same PHASES table replays identically at any dt."""
    blocks = []
    for duration_s, v, wz in PHASES:
        n = int(round(duration_s / dt))
        blocks.append(np.tile(cmd_to_wheels(v, wz), (n, 1)))
    return np.concatenate(blocks, axis=0).astype(np.float32)


def yaw_from_quat_xyzw(q: np.ndarray) -> float:
    """Yaw (rotation about world +Z) from a single quaternion [qx,qy,qz,qw]."""
    qx, qy, qz, qw = q
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


class HelhestVelCmdSimulator(HelhestJuniorReplaySimulator):
    """Same robot/actuator/friction setup as HelhestJuniorReplaySimulator, minus
    the box obstacle: bare ground (flat or loaded terrain) for an unobstructed
    open-loop rollout."""

    def __init__(self, *args, terrain: HeightMapReader, **kwargs):
        self.terrain = terrain
        super().__init__(*args, **kwargs)

    @override
    def build_model(self) -> newton.Model:
        self.builder.rigid_gap = 0.2

        ground_cfg = newton.ModelBuilder.ShapeConfig(mu=0.8, **self.ground_cfg_kwargs)
        heightfield, terrain_xform = self.terrain.to_ostrich()
        self.builder.add_shape_heightfield(xform=terrain_xform, heightfield=heightfield, cfg=ground_cfg)

        # Spawn 0.5 m above the local terrain height (was a bare literal 0.5
        # when ground was always flat at z=0).
        spawn_z = float(self.terrain.sample(SPAWN_X_M, SPAWN_Y_M)) + 0.5
        create_helhest_junior_model(
            self.builder,
            xform=wp.transform(wp.vec3(SPAWN_X_M, SPAWN_Y_M, spawn_z), wp.quat_identity()),
            control_mode=self.control_mode,
            k_p=self.k_p,
            k_d=self.k_d,
            friction_left_right=self.mu_front,
            friction_rear=self.mu_rear,
            mu_rolling=self.mu_rolling,
            ke=self.wheel_ke,
            kd=self.wheel_kd,
            kf=self.wheel_kf,
        )

        return self.builder.finalize_replicated(num_worlds=self.simulation_config.num_worlds)


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest", version_base=None)
def ostrich_vel_cmd(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    # We drive our own step loop (see replay() below) instead of run()'s segment
    # loop, so simulation.duration_seconds is unused — PHASES sets the length.
    # Only target_timestep_seconds (dt) is consumed.
    setpoints = build_setpoints(sim_config.target_timestep_seconds)

    terrain_path = cfg.get("terrain_path")
    terrain = (
        HeightMapReader.load(terrain_path)
        if terrain_path
        else HeightMapReader.flat(xlim=TERRAIN_XLIM_M, ylim=TERRAIN_YLIM_M, cell=TERRAIN_CELL)
    )

    sim = HelhestVelCmdSimulator(
        sim_config, render_config, engine_config, logging_config, k_p=K_P, terrain=terrain
    )
    # replay_graph() captures the per-step physics (control + solver.step + state
    # copy + pose/wheel logging) into one CUDA graph and replays it T times with
    # no Python in the loop — what simulation.use_cuda_graph is actually asking
    # for. replay() drives the same steps from Python instead: no graph capture,
    # plus a host<->device sync every step (.numpy() reads), so it's much slower.
    if sim_config.use_cuda_graph:
        poses, wheel_qd = sim.replay_graph(setpoints)
    else:
        poses, wheel_qd = sim.replay(setpoints)

    if render_config.vis_type == "gl":
        # Hold the window open so the final pose is inspectable instead of the
        # sim exiting the instant the schedule ends.
        k = len(setpoints) - 1
        while sim.viewer.is_running():
            sim._maybe_render(k)

    dt = sim_config.target_timestep_seconds
    t = np.arange(len(setpoints), dtype=np.float32) * dt
    start_xy, final_xy = poses[0, :2], poses[-1, :2]
    achieved_yaw = yaw_from_quat_xyzw(poses[-1, 3:7]) - yaw_from_quat_xyzw(poses[0, 3:7])
    commanded_deg, achieved_deg = math.degrees(NOMINAL_YAW_RAD), math.degrees(achieved_yaw)
    shortfall_pct = 100.0 * (1.0 - achieved_yaw / NOMINAL_YAW_RAD)
    nominal_x, nominal_y = start_xy[0] + NOMINAL_X, start_xy[1] + NOMINAL_Y

    print(
        f"commanded yaw : {commanded_deg:6.1f} deg   "
        f"(nominal endpoint x={nominal_x:.2f} y={nominal_y:.2f})"
    )
    print(
        f"achieved yaw  : {achieved_deg:6.1f} deg   "
        f"(actual  endpoint x={final_xy[0]:.2f} y={final_xy[1]:.2f})"
    )
    print(
        f"yaw shortfall : {shortfall_pct:5.1f} %     "
        f"-> implied alpha {NOMINAL_YAW_RAD / achieved_yaw:.2f}"
    )

    out = pathlib.Path(cfg.get("out", pathlib.Path(__file__).parent.parent / "outputs" /"ostrich_vel_cmd.npz"))
    np.savez_compressed(
        out,
        dt=np.float32(dt),
        t=t,
        cmd_wheel_omega=setpoints,
        pose=poses,
        wheel_qd=wheel_qd,
        phases=np.array(PHASES, dtype=np.float32),
    )
    terrain.save(out.with_suffix(""))
    print(f"saved {out}")


if __name__ == "__main__":
    ostrich_vel_cmd()
