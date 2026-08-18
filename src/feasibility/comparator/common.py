"""Scenario-independent core shared by comparator/compare_*.py: the ostrich replicated-model
simulator (with its batched Warp logging kernels), the helhest_stack ForwardSimulator wrapper,
and the npz assembly -- everything that doesn't depend on WHICH obstacle series is being driven
over. A scenario module supplies a `ScenarioSpec` (which heightmap series, where the obstacle
sits, how far to drive) and calls `run_comparison(cfg, spec)`.

Originally all of this lived in one file (batch_compare.py, the speed-bump-only comparison) --
split out here once a second scenario (box obstacles) needed the exact same machinery on a
different heightmap series.

Each variant gets its own single-world build+rollout, not a fused GPU batch across variants --
see the historical note in HelhestBatchSimulator's docstring for why (each variant has its own
terrain, and ostrich's replicated model / helhest_stack's ForwardSimulator both take one terrain
per build).
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

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

from feasibility.comparator.provenance import git_provenance
from feasibility.comparator.provenance import terrain_fields
from feasibility.heightmap import HeightMapReader

CONFIG_PATH = pathlib.Path(examples.__file__).parent.joinpath("conf")
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "outputs"

# Robot geometry, tied to Helhest Junior's own definition instead of re-hardcoded (see
# ostrich_vel_cmd.py, which duplicates these as WHEEL_RADIUS/HALF_TRACK literals).
WHEEL_RADIUS = HelhestJuniorConfig.WHEEL_RADIUS
HALF_TRACK = float(HelhestJuniorConfig.LEFT_WHEEL_POS[1])

# Same velocity-servo gain as ostrich_vel_cmd.py -- see that file for the k_p sweep that
# justifies it (soft enough to isolate ground slip, stiff enough not to be the bottleneck).
K_P = 15000.0


@dataclass(frozen=True)
class ScenarioSpec:
    """Everything a comparison needs beyond the shared machinery below. `variants` comes
    straight from the generator module's `*_paths()` function (e.g.
    create_speed_bumps.speed_bump_paths(), create_box_obstacles.box_obstacle_paths()) -- the
    single source of truth for the height series stays in the heightmap package, not here."""

    name: str  # -> outputs/compare_<name>.npz
    variants: list[tuple[float, pathlib.Path]]  # (height, asset stem)
    obstacle_x: float  # feature X; anchors spawn and the replay camera
    value_name: str  # "bump_height" / "box_height" -- recorded in the npz
    value_header: str  # "bump h [m]" -- print_summary column title
    label_fmt: str  # "bump_h={:.2f}m" -> variant_label
    spawn_back: float = 3.0  # spawn_x = obstacle_x - spawn_back
    spawn_y: float = 0.0
    v_drive: float = 1.0
    drive_s: float = 6.0


def cmd_to_wheels(v: float, wz: float) -> tuple[float, float, float]:
    """Ideal no-slip differential drive: body twist (v, omega) -> per-wheel rad/s [left, right, rear]."""
    v_l = (v - wz * HALF_TRACK) / WHEEL_RADIUS
    v_r = (v + wz * HALF_TRACK) / WHEEL_RADIUS
    v_rear = v / WHEEL_RADIUS
    return v_l, v_r, v_rear


def build_setpoints(dt: float, v_drive: float, drive_s: float) -> np.ndarray:
    """Sample a straight-approach schedule (drive at `v_drive` for `drive_s` meters) onto the
    dt grid. [T, 3] float32."""
    n = int(round(drive_s / dt))
    return np.tile(cmd_to_wheels(v_drive, 0.0), (n, 1)).astype(np.float32)


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


# --- ostrich: single-world replicated model, CUDA-graph-captured rollout, one build per
# heightmap (see module docstring for why this can't batch across variants) ---------------------


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
    `simulation_config.num_worlds` worlds via `finalize_replicated`. Each compare_*.py driver
    builds one of these per heightmap in its scenario's series (num_worlds=1) since each variant
    needs its own terrain; the replication/batch-kernel machinery here is generic to
    num_worlds > 1 too (kept from an earlier same-terrain, different-command sweep) and still
    works, just unused at N>1 by the current drivers."""

    def __init__(self, *args, terrain: HeightMapReader, spawn_xy: tuple[float, float], **kwargs):
        self.terrain = terrain
        self.spawn_xy = spawn_xy
        super().__init__(*args, **kwargs)
        num_worlds = self.simulation_config.num_worlds
        self.dofs_per_world = self.model.joint_dof_count // num_worlds
        self.bodies_per_world = self.model.body_count // num_worlds

    def build_model(self) -> newton.Model:
        self.builder.rigid_gap = 0.2

        # Mesh goes in a separate builder so it gets shape_world=-1 (Newton's "global"
        # sentinel): stored once, broadphase-tested against every world instead of
        # duplicated per world. Same pattern as demos/ostrich_speed_bump.py.
        globals_builder = newton.ModelBuilder()
        globals_builder.add_shape_mesh(
            body=-1,
            mesh=self.terrain.to_ostrich_mesh(),
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.8, **self.ground_cfg_kwargs),
        )

        spawn_x, spawn_y = self.spawn_xy
        spawn_z = float(self.terrain.sample(spawn_x, spawn_y)) + 0.5
        create_helhest_junior_model(
            self.builder,
            xform=wp.transform(wp.vec3(spawn_x, spawn_y, spawn_z), wp.quat_identity()),
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

        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds, global_builder=globals_builder
        )

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
    spawn_xy: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    sim = HelhestBatchSimulator(
        sim_config, render_config, engine_config, logging_config,
        k_p=K_P, mu_front=mu, mu_rear=mu, terrain=terrain, spawn_xy=spawn_xy,
    )
    return sim.replay_graph_batch(setpoints)


# --- helhest_stack: natively batched ForwardSimulator -------------------------------------------


def run_hstack_batch(
    setpoints: np.ndarray,
    hmap: HeightMapReader,
    dt: float,
    k_turn: float,
    mu: float,
    device: str,
    spawn_xy: tuple[float, float],
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
        np.ascontiguousarray(setpoints, np.float32), (spawn_xy[0], spawn_xy[1], 0.0), np.zeros(3)
    )
    turning = sim.turning.numpy()
    wheel_qd = sim.current_wheel_omega.numpy()[1:]
    return controlled[1:], derived[1:], clearance, residual, turning, wheel_qd


def print_summary(values: np.ndarray, value_header: str, ostrich_pose: np.ndarray, hstack_controlled: np.ndarray) -> None:
    """ostrich_pose [T, n, 7] (x,y,z,qx,qy,qz,qw); hstack_controlled [T, n, 3] (x,y,yaw) -- z
    isn't in `controlled` (it's in `derived`), so compare final X reached (obstacle-induced slip
    shows up as a shortfall there) and ostrich's peak chassis Z (how far the dynamics model
    actually lifted going over the obstacle, which the kinematic twin's quasi-static settle can't
    represent the same way)."""
    ostrich_final_x = ostrich_pose[-1, :, 0]
    ostrich_peak_z = ostrich_pose[:, :, 2].max(axis=0)
    hstack_final_x = hstack_controlled[-1, :, 0]
    print(f"{value_header:>12}{'ostrich x':>12}{'ostrich peak z':>16}{'hstack x':>12}")
    for v, ox, oz, hx in zip(values, ostrich_final_x, ostrich_peak_z, hstack_final_x):
        print(f"{v:12.2f}{ox:12.3f}{oz:16.3f}{hx:12.3f}")


def run_comparison(cfg: DictConfig, spec: ScenarioSpec) -> None:
    """The full ostrich-vs-hstack sweep over `spec.variants`, writing
    outputs/compare_<spec.name>.npz. Shared by every comparator/compare_*.py driver -- see the
    module docstring for what varies per scenario (just `spec`)."""
    wp.init()

    mu = float(cfg.get("mu", 0.8))
    k_turn = float(cfg.get("k_turn", dynamics.K_TURN))
    device = str(cfg.get("device", "cuda:0"))

    variants = spec.variants
    variant_values = np.array([h for h, _ in variants], dtype=np.float32)
    n = len(variants)

    spawn_x = spec.obstacle_x - spec.spawn_back
    spawn_y = spec.spawn_y
    spawn_xy = (spawn_x, spawn_y)

    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)
    # One world per call below -- each variant has its OWN terrain, so (unlike an earlier
    # yaw-rate sweep on one shared ramp) they can't share a single replicated build.
    sim_config.num_worlds = 1
    render_config.vis_type = "null"  # headless: no GL viewer

    ostrich_dt = sim_config.target_timestep_seconds
    hstack_dt = dynamics.DT
    ostrich_setpoints_1 = build_setpoints(ostrich_dt, spec.v_drive, spec.drive_s)[:, None, :]  # [T, 1, 3]
    hstack_setpoints_1 = build_setpoints(hstack_dt, spec.v_drive, spec.drive_s)[:, None, :]
    print(f"[ostrich]  {n} heightmaps x {ostrich_setpoints_1.shape[0]} steps @ dt={ostrich_dt}")
    print(f"[hstack]   {n} heightmaps x {hstack_setpoints_1.shape[0]} steps @ dt={hstack_dt}")

    terrain_paths: list[str] = []
    terrain_entries: list[tuple[pathlib.Path, HeightMapReader]] = []
    ostrich_poses, ostrich_wheel_qds = [], []
    h_controlleds, h_deriveds, h_clearances, h_residuals, h_turnings, h_wheel_qds = [], [], [], [], [], []

    for height, path in variants:
        terrain = HeightMapReader.load(path)
        terrain_paths.append(str(path))
        terrain_entries.append((path, terrain))

        pose, wheel_qd = run_ostrich_batch(
            sim_config, render_config, engine_config, logging_config, terrain, ostrich_setpoints_1, mu, spawn_xy
        )
        ostrich_poses.append(pose[:, 0])
        ostrich_wheel_qds.append(wheel_qd[:, 0])

        controlled, derived, clearance, residual, turning, wheel_qd_h = run_hstack_batch(
            hstack_setpoints_1, terrain, hstack_dt, k_turn, mu, device, spawn_xy
        )
        h_controlleds.append(controlled[:, 0])
        h_deriveds.append(derived[:, 0])
        h_clearances.append(clearance[:, 0])
        h_residuals.append(residual[:, 0])
        h_turnings.append(turning[:, 0])
        h_wheel_qds.append(wheel_qd_h[:, 0])
        print(f"  {spec.value_name}={height:.2f} m  done")

    ostrich_pose = np.stack(ostrich_poses, axis=1)  # [T, n, 7]
    ostrich_wheel_qd = np.stack(ostrich_wheel_qds, axis=1)
    h_controlled = np.stack(h_controlleds, axis=1)  # [T, n, 3]
    h_derived = np.stack(h_deriveds, axis=1)
    h_clearance = np.stack(h_clearances, axis=1)
    h_residual = np.stack(h_residuals, axis=1)
    h_turning = np.stack(h_turnings, axis=1)
    h_wheel_qd = np.stack(h_wheel_qds, axis=1)

    print_summary(variant_values, spec.value_header, ostrich_pose, h_controlled)

    h_quat = euler_zyx_to_quat_xyzw(h_controlled[..., 2], h_derived[..., 1], h_derived[..., 2])
    h_pose = np.concatenate([h_controlled[..., :2], h_derived[..., :1], h_quat], axis=-1).astype(np.float32)

    out_npz = OUT_DIR / f"compare_{spec.name}.npz"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        n=np.int32(n),
        variant_name=np.array(spec.value_name),
        variant_value=variant_values,
        variant_label=np.array([spec.label_fmt.format(h) for h in variant_values]),
        obstacle_x=np.float32(spec.obstacle_x),
        # terrain_path stays as a provenance hint (where the terrain came from) but is no
        # longer load-bearing -- terrain_fields() embeds the grid+sidecar itself below, so a
        # reader never needs assets/ on disk (see comparator.provenance.terrain_from_npz).
        terrain_path=np.array(terrain_paths),
        **terrain_fields(terrain_entries),
        **git_provenance(),
        spawn_xy=np.array(spawn_xy, dtype=np.float32),
        mu=np.float32(mu),
        k_turn=np.float32(k_turn),
        k_p=np.float32(K_P),
        ostrich_dt=np.float32(ostrich_dt),
        ostrich_t=np.arange(ostrich_setpoints_1.shape[0], dtype=np.float32) * ostrich_dt,
        ostrich_cmd_wheel_omega=np.repeat(ostrich_setpoints_1, n, axis=1),
        ostrich_pose=ostrich_pose,
        ostrich_wheel_qd=ostrich_wheel_qd,
        hstack_dt=np.float32(hstack_dt),
        hstack_t=np.arange(hstack_setpoints_1.shape[0], dtype=np.float32) * hstack_dt,
        hstack_cmd_wheel_omega=np.repeat(hstack_setpoints_1, n, axis=1),
        hstack_pose=h_pose,
        hstack_controlled=h_controlled,
        hstack_derived=h_derived,
        hstack_turning=h_turning,
        hstack_clearance=h_clearance,
        hstack_residual=h_residual,
        hstack_wheel_qd=h_wheel_qd,
    )
    print(f"saved {out_npz}")
