"""Batch ostrich-vs-helhest_stack comparison: N wheel-velocity-command variants, each replayed
once in ostrich (dynamics) and once in helhest_stack (kinematic twin), on the same ramp terrain
and the same initial condition as demos/ostrich_vel_cmd.py / demos/hstack_vel_cmd.py.

Variants differ only in the commanded yaw rate of the middle (arc) phase of the
straight/arc/straight schedule used by those two demos -- a symmetric sweep from a hard right
turn to a hard left turn, so the N pairs cover the slip-comparison space those demos probe one
point of at a time.

Ostrich side: all N variants are ONE replicated model (`finalize_replicated`), so they run as a
single CUDA-graph-captured batch of N worlds -- N parallel dynamics rollouts, not N processes.
helhest_stack side: `ForwardSimulator` is natively batched (`batch_size=N`), so its N rollouts are
likewise one fused-kernel launch.

Usage:
    python -m feasibility.comparator.batch_compare                # N=10, ramp terrain
    python -m feasibility.comparator.batch_compare +n=4
    python -m feasibility.comparator.batch_compare +mu=0.5 +k_turn=1.0
    python -m feasibility.comparator.batch_compare +terrain_path=assets/ramp
"""

from __future__ import annotations

import math
import pathlib

import examples
import hydra
import newton
import numpy as np
import warp as wp
from examples.helhest_junior.common import create_helhest_junior_model
from examples.helhest_junior.common import HelhestJuniorConfig
from examples.helhest_junior.replay_real import HelhestJuniorReplaySimulator
from examples.helhest_junior.replay_real import WHEEL_DOF_OFFSET
from omegaconf import DictConfig
from ostrich import EngineConfig
from ostrich import LoggingConfig
from ostrich import RenderingConfig
from ostrich import SimulationConfig

from helhest import dynamics
from helhest import friction as friction_mod
from helhest.engine import ForwardSimulator

from feasibility.heightmap import HeightMapReader

CONFIG_PATH = pathlib.Path(examples.__file__).parent.joinpath("conf")
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_TERRAIN = REPO_ROOT / "assets" / "ramp"
OUT_NPZ = REPO_ROOT / "outputs" / "batch_compare.npz"

N_DEFAULT = 10

# Robot geometry, tied to Helhest Junior's own definition instead of re-hardcoded (see
# ostrich_vel_cmd.py, which duplicates these as WHEEL_RADIUS/HALF_TRACK literals).
WHEEL_RADIUS = HelhestJuniorConfig.WHEEL_RADIUS
HALF_TRACK = float(HelhestJuniorConfig.LEFT_WHEEL_POS[1])

# Same velocity-servo gain as ostrich_vel_cmd.py -- see that file for the k_p sweep that
# justifies it (soft enough to isolate ground slip, stiff enough not to be the bottleneck).
K_P = 15000.0

# Command schedule: straight / arc(yaw_rate) / straight, identical timing to ostrich_vel_cmd.py.
V_DRIVE = 1.0
YAW_RATE_MAX = 0.5
T_ARC = 5.0318
STRAIGHT_S = 2.0


def variant_yaw_rates(n: int) -> np.ndarray:
    """N commanded yaw rates, a symmetric sweep hard-right -> hard-left (n=1 -> straight only)."""
    if n == 1:
        return np.array([0.0], dtype=np.float32)
    return np.linspace(-YAW_RATE_MAX, YAW_RATE_MAX, n, dtype=np.float32)


def cmd_to_wheels(v: float, wz: float) -> tuple[float, float, float]:
    """Ideal no-slip differential drive: body twist (v, omega) -> per-wheel rad/s [left, right, rear]."""
    v_l = (v - wz * HALF_TRACK) / WHEEL_RADIUS
    v_r = (v + wz * HALF_TRACK) / WHEEL_RADIUS
    v_rear = v / WHEEL_RADIUS
    return v_l, v_r, v_rear


def build_setpoints(dt: float, yaw_rate: float) -> np.ndarray:
    """Sample the straight/arc(yaw_rate)/straight schedule onto the dt grid. [T, 3] float32."""
    phases = [(STRAIGHT_S, V_DRIVE, 0.0), (T_ARC, V_DRIVE, yaw_rate), (STRAIGHT_S, V_DRIVE, 0.0)]
    blocks = [np.tile(cmd_to_wheels(v, wz), (int(round(duration_s / dt)), 1)) for duration_s, v, wz in phases]
    return np.concatenate(blocks, axis=0).astype(np.float32)


def build_variant_setpoints(dt: float, yaw_rates: np.ndarray) -> np.ndarray:
    """[T, N, 3] float32: one setpoint schedule per variant, stacked on the batch axis."""
    return np.stack([build_setpoints(dt, w) for w in yaw_rates], axis=1).astype(np.float32)


def yaw_from_quat_xyzw(q: np.ndarray) -> np.ndarray:
    """Yaw (rotation about world +Z) from quaternion(s) [..., 4] = [qx,qy,qz,qw]."""
    qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def euler_zyx_to_quat_xyzw(yaw: np.ndarray, pitch: np.ndarray, roll: np.ndarray) -> np.ndarray:
    """(yaw, pitch, roll) [...] -> quaternion [..., 4] (qx,qy,qz,qw), R = Rz(yaw)@Ry(pitch)@Rx(roll)."""
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return np.stack([qx, qy, qz, qw], axis=-1).astype(np.float32)


# --- ostrich: N worlds, one replicated model, one CUDA-graph-captured batched rollout ----------


@wp.kernel
def _batch_control_kernel(
    setpoints: wp.array(dtype=wp.float32, ndim=3),  # [T, W, 3] in sim order [L, R, rear]
    step_buf: wp.array(dtype=wp.int32),
    T: int,
    joint_target_vel: wp.array(dtype=wp.float32),
    dofs_per_world: int,
    wheel_dof_offset: int,
):
    w = wp.tid()
    s = wp.min(step_buf[0], T - 1)
    base = w * dofs_per_world + wheel_dof_offset
    joint_target_vel[base + 0] = setpoints[s, w, 0]
    joint_target_vel[base + 1] = setpoints[s, w, 1]
    joint_target_vel[base + 2] = setpoints[s, w, 2]


@wp.kernel
def _batch_log_pose_kernel(
    body_q: wp.array(dtype=wp.transform),
    step_buf: wp.array(dtype=wp.int32),
    T: int,
    pose_log: wp.array(dtype=wp.float32, ndim=3),  # [T, W, 7]
    bodies_per_world: int,
    chassis_local_idx: int,
):
    w = wp.tid()
    s = wp.min(step_buf[0], T - 1)
    tf = body_q[w * bodies_per_world + chassis_local_idx]
    p = wp.transform_get_translation(tf)
    q = wp.transform_get_rotation(tf)
    pose_log[s, w, 0] = p[0]
    pose_log[s, w, 1] = p[1]
    pose_log[s, w, 2] = p[2]
    pose_log[s, w, 3] = q[0]
    pose_log[s, w, 4] = q[1]
    pose_log[s, w, 5] = q[2]
    pose_log[s, w, 6] = q[3]


@wp.kernel
def _batch_log_wheel_kernel(
    jqd: wp.array(dtype=wp.float32),
    step_buf: wp.array(dtype=wp.int32),
    T: int,
    wheel_log: wp.array(dtype=wp.float32, ndim=3),  # [T, W, 3]
    dofs_per_world: int,
    wheel_dof_offset: int,
):
    w = wp.tid()
    s = wp.min(step_buf[0], T - 1)
    base = w * dofs_per_world + wheel_dof_offset
    wheel_log[s, w, 0] = jqd[base + 0]
    wheel_log[s, w, 1] = jqd[base + 1]
    wheel_log[s, w, 2] = jqd[base + 2]


@wp.kernel
def _advance_kernel(step_buf: wp.array(dtype=wp.int32)):
    step_buf[0] = step_buf[0] + 1


class HelhestBatchSimulator(HelhestJuniorReplaySimulator):
    """HelhestJuniorReplaySimulator's robot/actuator/friction setup, replicated across
    `simulation_config.num_worlds` worlds via `finalize_replicated` so each world can be driven
    with its own wheel-velocity command sequence -- N ostrich rollouts in one CUDA-graph batch,
    on the same shared terrain and the same shared spawn pose."""

    def __init__(self, *args, terrain: HeightMapReader, **kwargs):
        self.terrain = terrain
        super().__init__(*args, **kwargs)
        num_worlds = self.simulation_config.num_worlds
        self.dofs_per_world = self.model.joint_dof_count // num_worlds
        self.bodies_per_world = self.model.body_count // num_worlds

    def build_model(self) -> newton.Model:
        self.builder.rigid_gap = 0.2

        ground_cfg = newton.ModelBuilder.ShapeConfig(mu=0.8, **self.ground_cfg_kwargs)
        heightfield, terrain_xform = self.terrain.to_ostrich()
        self.builder.add_shape_heightfield(xform=terrain_xform, heightfield=heightfield, cfg=ground_cfg)

        spawn_z = float(self.terrain.sample(0.0, 0.0)) + 0.5
        create_helhest_junior_model(
            self.builder,
            xform=wp.transform(wp.vec3(0.0, 0.0, spawn_z), wp.quat_identity()),
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

    def _batch_physics_step(self) -> None:
        """One physics step, all worlds at once (capturable) -- the batched analogue of
        HelhestJuniorReplaySimulator._graph_physics_step."""
        self.current_state.clear_forces()
        self.contacts = self.model.collide(self.current_state)
        wp.launch(
            kernel=_batch_control_kernel,
            dim=self.simulation_config.num_worlds,
            inputs=[
                self._setpoints_wp,
                self._step_buf,
                self._T,
                self.control.joint_target_vel,
                self.dofs_per_world,
                WHEEL_DOF_OFFSET,
            ],
            device=self.model.device,
        )
        self.solver.step(
            state_in=self.current_state,
            state_out=self.next_state,
            control=self.control,
            contacts=self.contacts,
            dt=self.clock.dt,
        )
        self._copy_state(self.current_state, self.next_state)
        wp.launch(
            _batch_log_pose_kernel,
            dim=self.simulation_config.num_worlds,
            inputs=[self.current_state.body_q, self._step_buf, self._T, self._pose_log, self.bodies_per_world, 0],
            device=self.model.device,
        )
        newton.eval_ik(self.model, self.current_state, self._jq, self._jqd)
        wp.launch(
            _batch_log_wheel_kernel,
            dim=self.simulation_config.num_worlds,
            inputs=[self._jqd, self._step_buf, self._T, self._wheel_log, self.dofs_per_world, WHEEL_DOF_OFFSET],
            device=self.model.device,
        )
        wp.launch(_advance_kernel, dim=1, inputs=[self._step_buf], device=self.model.device)

    def replay_graph_batch(
        self, setpoints: np.ndarray, settle_steps: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """CUDA-graph batched replay: settle at zero velocity (uncaptured), then capture one
        physics step covering ALL worlds and launch it T times with no Python in the loop.
        `setpoints` is [T, num_worlds, 3]. Returns pose [T, num_worlds, 7], wheel_qd [T, num_worlds, 3]."""
        T, num_worlds, _ = setpoints.shape
        assert num_worlds == self.simulation_config.num_worlds
        self._T = T
        self._setpoints_wp = wp.array(setpoints, dtype=wp.float32, device=self.model.device)
        self._step_buf = wp.zeros(1, dtype=wp.int32, device=self.model.device)
        self._pose_log = wp.zeros((T, num_worlds, 7), dtype=wp.float32, device=self.model.device)
        self._wheel_log = wp.zeros((T, num_worlds, 3), dtype=wp.float32, device=self.model.device)
        self._jq = wp.zeros_like(self.model.joint_q)
        self._jqd = wp.zeros_like(self.model.joint_qd)
        settle_steps = self._resolve_settle_steps(settle_steps)

        # Settle on the ground (uncaptured, zero command -- target_velocities only ever
        # addresses world 0's dofs, but joint_target_vel starts zero-initialized for every
        # world and is never otherwise written during settle, so all worlds settle at rest).
        self.target_velocities.zero_()
        for _ in range(settle_steps):
            self._single_physics_step(0)

        self._step_buf.zero_()
        with wp.ScopedCapture() as capture:
            self._batch_physics_step()
        graph = capture.graph

        for _ in range(T):
            wp.capture_launch(graph)
        wp.synchronize()
        return self._pose_log.numpy(), self._wheel_log.numpy()


def run_ostrich_batch(
    sim_config: SimulationConfig,
    render_config: RenderingConfig,
    engine_config: EngineConfig,
    logging_config: LoggingConfig,
    terrain: HeightMapReader,
    setpoints: np.ndarray,
    mu: float,
) -> tuple[np.ndarray, np.ndarray]:
    sim = HelhestBatchSimulator(
        sim_config, render_config, engine_config, logging_config,
        k_p=K_P, mu_front=mu, mu_rear=mu, terrain=terrain,
    )
    return sim.replay_graph_batch(setpoints)


# --- helhest_stack: natively batched ForwardSimulator -------------------------------------------


def run_hstack_batch(
    setpoints: np.ndarray, hmap: HeightMapReader, dt: float, k_turn: float, mu: float, device: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """setpoints [T, N, 3]. Returns controlled [T,N,3] (x,y,yaw), derived [T,N,3] (z,pitch,roll),
    clearance [T,N], residual [T,N], turning [T,N,2], wheel_qd [T,N,3] -- all with the row-0
    pre-command pose dropped, matching hstack_vel_cmd.py's single-rollout convention."""
    T, n, _ = setpoints.shape
    elevation, grid = hmap.to_hstack(device)
    mu_xlim = (hmap.x0, hmap.x0 + (hmap.nx - 1) * hmap.cell)
    mu_ylim = (hmap.y0, hmap.y0 + (hmap.ny - 1) * hmap.cell)
    mu_field = friction_mod.uniform(mu, xlim=mu_xlim, ylim=mu_ylim, cell=hmap.cell)

    solver = dynamics.execution_solver(dt=dt, k_turn=k_turn)
    sim = ForwardSimulator(dynamics.robot_params(), solver, grid, batch_size=n, n_steps=T, device=device)
    sim.set_terrain(elevation)
    sim.set_friction(mu_field)
    controlled, derived, clearance, residual = sim.rollout(
        np.ascontiguousarray(setpoints, np.float32), (0.0, 0.0, 0.0), np.zeros(3)
    )
    turning = sim.turning.numpy()
    wheel_qd = sim.current_wheel_omega.numpy()[1:]
    return controlled[1:], derived[1:], clearance, residual, turning, wheel_qd


def print_summary(yaw_rates: np.ndarray, ostrich_pose: np.ndarray, hstack_controlled: np.ndarray) -> None:
    ostrich_yaw = yaw_from_quat_xyzw(ostrich_pose[-1, :, 3:7]) - yaw_from_quat_xyzw(ostrich_pose[0, :, 3:7])
    hstack_yaw = hstack_controlled[-1, :, 2]
    print(f"{'cmd yaw_rate':>14}{'ostrich yaw':>14}{'hstack yaw':>14}")
    for w, oy, hy in zip(yaw_rates, ostrich_yaw, hstack_yaw):
        print(f"{w:14.3f}{math.degrees(oy):13.2f}°{math.degrees(hy):13.2f}°")


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest", version_base=None)
def batch_compare(cfg: DictConfig) -> None:
    wp.init()

    n = int(cfg.get("n", N_DEFAULT))
    mu = float(cfg.get("mu", 0.8))
    k_turn = float(cfg.get("k_turn", dynamics.K_TURN))
    device = str(cfg.get("device", "cuda:0"))
    terrain_path = str(cfg.get("terrain_path", DEFAULT_TERRAIN))

    yaw_rates = variant_yaw_rates(n)
    terrain = HeightMapReader.load(terrain_path)

    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)
    sim_config.num_worlds = n
    render_config.vis_type = "null"  # headless: no GL viewer

    ostrich_dt = sim_config.target_timestep_seconds
    ostrich_setpoints = build_variant_setpoints(ostrich_dt, yaw_rates)
    print(f"[ostrich]  {n} worlds x {ostrich_setpoints.shape[0]} steps @ dt={ostrich_dt}")
    ostrich_pose, ostrich_wheel_qd = run_ostrich_batch(
        sim_config, render_config, engine_config, logging_config, terrain, ostrich_setpoints, mu
    )

    hstack_dt = dynamics.DT
    hstack_setpoints = build_variant_setpoints(hstack_dt, yaw_rates)
    print(f"[hstack]   {n} worlds x {hstack_setpoints.shape[0]} steps @ dt={hstack_dt}")
    h_controlled, h_derived, h_clearance, h_residual, h_turning, h_wheel_qd = run_hstack_batch(
        hstack_setpoints, terrain, hstack_dt, k_turn, mu, device
    )

    print_summary(yaw_rates, ostrich_pose, h_controlled)

    h_quat = euler_zyx_to_quat_xyzw(h_controlled[..., 2], h_derived[..., 1], h_derived[..., 2])
    h_pose = np.concatenate([h_controlled[..., :2], h_derived[..., :1], h_quat], axis=-1).astype(np.float32)

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        n=np.int32(n),
        variant_yaw_rate=yaw_rates,
        variant_label=np.array([f"yaw_rate={w:+.2f}" for w in yaw_rates]),
        terrain_path=terrain_path,
        mu=np.float32(mu),
        k_turn=np.float32(k_turn),
        k_p=np.float32(K_P),
        ostrich_dt=np.float32(ostrich_dt),
        ostrich_t=np.arange(ostrich_setpoints.shape[0], dtype=np.float32) * ostrich_dt,
        ostrich_cmd_wheel_omega=ostrich_setpoints,
        ostrich_pose=ostrich_pose,
        ostrich_wheel_qd=ostrich_wheel_qd,
        hstack_dt=np.float32(hstack_dt),
        hstack_t=np.arange(hstack_setpoints.shape[0], dtype=np.float32) * hstack_dt,
        hstack_cmd_wheel_omega=hstack_setpoints,
        hstack_pose=h_pose,
        hstack_controlled=h_controlled,
        hstack_derived=h_derived,
        hstack_turning=h_turning,
        hstack_clearance=h_clearance,
        hstack_residual=h_residual,
        hstack_wheel_qd=h_wheel_qd,
    )
    print(f"saved {OUT_NPZ}")


if __name__ == "__main__":
    batch_compare()
