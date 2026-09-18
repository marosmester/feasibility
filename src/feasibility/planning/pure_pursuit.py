"""Pure-pursuit path following for Helhest Junior in ostrich, as a Warp kernel INSIDE the captured
physics step (reads the chassis pose, writes wheel targets, no host round trip).

    * Waypoints: a planned polyline resampled every WAYPOINT_SPACING (`resample_polyline`). For a
      lattice plan these are the traced cell-corner poses (the front-axle base frame the settle and
      the network judge; ostrich's chassis body 0 is the same frame) plus the exact goal, joined by
      straight chords that zig-zag +-7.5 deg where the lattice has no 0 deg heading; the lookahead
      smooths that out.
    * Pure pursuit on the chassis' horizontal heading: a monotonic nearest-waypoint progress index
      (forward window of SEARCH_WINDOW only, so it cannot jump back), target = first waypoint at
      least `lookahead` away, kappa = 2 sin(alpha) / L clamped to `controller_kappa_max()`, constant
      forward speed v, wheel targets by the same ideal diff-drive map as comparator.common.cmd_to_wheels.
      The clamp is KAPPA_GAIN x the lattice's own turn limit (1 / min_turn_radius) because the
      skid-steer under-turns (slip): commanded at 2.0 1/m it achieves ~1.0-1.2 1/m, so a clamp at the
      planner's limit saturates the feedback and a tight planned hook swings wide.
    * A world stops (wheels zeroed, latched) once within STOP_RADIUS of the goal, or once its
      progress index is the last waypoint and it is within GOAL_TOLERANCE -- past the goal's
      perpendicular the last waypoint stays the pursuit target behind the robot, so a pass just
      outside STOP_RADIUS would otherwise never stop. `evaluation.judge` counts the same condition
      as arrived.

Every world follows its own path out of one flat waypoint buffer (`path_begin[w]`, `path_count[w]`),
so one build can drive many different paths -- or the same path in every world.
"""
from __future__ import annotations

import numpy as np
import warp as wp
from examples.helhest_junior.replay_real import WHEEL_DOF_OFFSET
from helhest import dynamics

from feasibility.comparator.common import HALF_TRACK
from feasibility.comparator.common import HelhestBatchSimulator
from feasibility.comparator.common import WHEEL_RADIUS

WAYPOINT_SPACING = 0.05  # [m] resampled path the controller walks
SEARCH_WINDOW = 40  # waypoints ahead the progress index may advance per step (2 m)
STOP_RADIUS = 0.15  # [m] from the goal: the world stops and counts as arrived
GOAL_TOLERANCE = 0.3  # [m] ... also once at the path's last waypoint within this of the goal
KAPPA_GAIN = 2.0  # controller curvature clamp over the planner's 1 / min_turn_radius (slip margin)
CHASSIS_LOCAL_IDX = 0  # body 0 of each world is the chassis -- the pose logger reads the same one


def controller_kappa_max() -> float:
    """Pure pursuit's curvature clamp [1/m]: KAPPA_GAIN x the planner's turn limit."""
    return KAPPA_GAIN / float(dynamics.robot_params().min_turn_radius)


def resample_polyline(xy: np.ndarray, spacing: float = WAYPOINT_SPACING) -> np.ndarray:
    """[n, 2] vertices -> [m, 2] points every `spacing` m along the chords, both ends kept."""
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    keep = np.concatenate([[True], seg > 1e-9])  # repeated vertices would break np.interp
    s, xy = s[keep], xy[keep]
    q = np.append(np.arange(0.0, s[-1], spacing), s[-1])
    return np.stack([np.interp(q, s, xy[:, 0]), np.interp(q, s, xy[:, 1])], axis=-1)


@wp.kernel
def _pure_pursuit_kernel(
    body_q: wp.array(dtype=wp.transform),
    path_flat: wp.array(dtype=wp.vec2),  # every world's waypoints back to back
    path_begin: wp.array(dtype=wp.int32),  # [W] first waypoint of world w in path_flat
    path_count: wp.array(dtype=wp.int32),  # [W] waypoints of world w
    progress: wp.array(dtype=wp.int32),  # [W] nearest waypoint so far, local index (monotonic)
    stopped: wp.array(dtype=wp.int32),  # [W] latched 1 once arrived (see the module docstring)
    step_buf: wp.array(dtype=wp.int32),
    settle_steps: int,
    lookahead: float,
    v: float,
    kappa_max: float,
    stop_radius: float,
    goal_tolerance: float,
    wheel_radius: float,
    half_track: float,
    joint_target_vel: wp.array(dtype=wp.float32),
    bodies_per_world: int,
    dofs_per_world: int,
    wheel_dof_offset: int,
):
    w = wp.tid()
    base = w * dofs_per_world + wheel_dof_offset
    tf = body_q[w * bodies_per_world + CHASSIS_LOCAL_IDX]
    p = wp.transform_get_translation(tf)
    b = path_begin[w]
    n = path_count[w]
    if step_buf[0] < settle_steps or stopped[w] == 1:
        joint_target_vel[base + 0] = 0.0
        joint_target_vel[base + 1] = 0.0
        joint_target_vel[base + 2] = 0.0
        return

    i0 = progress[w]
    best = i0
    q = path_flat[b + i0]
    best_d = (q[0] - p[0]) * (q[0] - p[0]) + (q[1] - p[1]) * (q[1] - p[1])
    for k in range(i0 + 1, wp.min(i0 + SEARCH_WINDOW, n)):
        q = path_flat[b + k]
        d = (q[0] - p[0]) * (q[0] - p[0]) + (q[1] - p[1]) * (q[1] - p[1])
        if d < best_d:
            best = k
            best_d = d
    progress[w] = best

    goal = path_flat[b + n - 1]
    d_goal = (goal[0] - p[0]) * (goal[0] - p[0]) + (goal[1] - p[1]) * (goal[1] - p[1])
    if d_goal < stop_radius * stop_radius or (best == n - 1 and d_goal < goal_tolerance * goal_tolerance):
        stopped[w] = 1
        joint_target_vel[base + 0] = 0.0
        joint_target_vel[base + 1] = 0.0
        joint_target_vel[base + 2] = 0.0
        return

    target = n - 1
    found = int(0)
    for k in range(best, n):
        if found == 0:
            q = path_flat[b + k]
            d = (q[0] - p[0]) * (q[0] - p[0]) + (q[1] - p[1]) * (q[1] - p[1])
            if d >= lookahead * lookahead:
                target = k
                found = 1

    # heading of the chassis x axis projected on the ground plane -- stays right when pitched
    fwd = wp.quat_rotate(wp.transform_get_rotation(tf), wp.vec3(1.0, 0.0, 0.0))
    t = path_flat[b + target]
    dx = t[0] - p[0]
    dy = t[1] - p[1]
    alpha = wp.atan2(dy, dx) - wp.atan2(fwd[1], fwd[0])
    alpha = wp.atan2(wp.sin(alpha), wp.cos(alpha))  # wrap to (-pi, pi]
    dist = wp.max(wp.sqrt(dx * dx + dy * dy), 1e-3)
    kappa = wp.clamp(2.0 * wp.sin(alpha) / dist, -kappa_max, kappa_max)
    wz = v * kappa
    joint_target_vel[base + 0] = (v - wz * half_track) / wheel_radius
    joint_target_vel[base + 1] = (v + wz * half_track) / wheel_radius
    joint_target_vel[base + 2] = v / wheel_radius


class PurePursuitSimulator(HelhestBatchSimulator):
    """HelhestBatchSimulator whose per-step control is the pure-pursuit kernel instead of a
    setpoint table: world w follows `paths[w]` ([P_w, 2] waypoints in the build's own coordinates,
    so already shifted onto its tile on a TiledTerrain)."""

    def __init__(
        self,
        *args,
        paths: list[np.ndarray],
        settle_steps: int,
        lookahead: float,
        v: float,
        kappa_max: float,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        device = self.model.device
        num_worlds = self.simulation_config.num_worlds
        assert len(paths) == num_worlds, f"{len(paths)} paths for {num_worlds} worlds"
        self.paths = [np.asarray(p, np.float32) for p in paths]
        counts = np.array([len(p) for p in self.paths], np.int32)
        begins = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int32)
        self._path_flat = wp.array(np.concatenate(self.paths), dtype=wp.vec2, device=device)
        self._path_begin = wp.array(begins, dtype=wp.int32, device=device)
        self._path_count = wp.array(counts, dtype=wp.int32, device=device)
        self._progress = wp.zeros(num_worlds, dtype=wp.int32, device=device)
        self._stopped = wp.zeros(num_worlds, dtype=wp.int32, device=device)
        self._settle_steps = int(settle_steps)
        self._lookahead, self._v, self._kappa_max = float(lookahead), float(v), float(kappa_max)

    def _batch_physics_step(self) -> None:
        # HelhestBatchSimulator._batch_physics_step with the setpoint kernel swapped for pursuit
        self.current_state.clear_forces()
        self.contacts = self.model.collide(self.current_state)
        wp.launch(
            kernel=_pure_pursuit_kernel,
            dim=self.simulation_config.num_worlds,
            inputs=[
                self.current_state.body_q, self._path_flat, self._path_begin, self._path_count,
                self._progress, self._stopped, self._step_buf, self._settle_steps, self._lookahead,
                self._v, self._kappa_max, STOP_RADIUS, GOAL_TOLERANCE, float(WHEEL_RADIUS), HALF_TRACK,
                self.control.joint_target_vel, self.bodies_per_world, self.dofs_per_world,
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
        self._log_step(self._step_buf, self._T, self._pose_log, self._wheel_log)

    def _maybe_render(self, step_idx: int) -> None:
        if self.viewer is None:
            return
        self.viewer.begin_frame(step_idx * self.clock.dt)
        self.viewer.log_state(self.current_state)
        self.viewer.log_lines("planned_path", self._path_starts, self._path_ends, (1.0, 0.2, 0.2))
        self.viewer.end_frame()

    def rollout(self, T: int, view: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Settle + drive as ONE captured step launched T times (the settle is the kernel's first
        `settle_steps` rows). Headless launches back to back; `view` renders between launches
        (replay_real.replay_graph's GL pattern), drawing every world's path, and holds the window on
        the last frame. Returns pose [T, W, 7], wheel_qd [T, W, 3] (fewer rows if the viewer was
        closed early)."""
        device = self.model.device
        num_worlds = self.simulation_config.num_worlds
        self._T = T
        self._step_buf = wp.zeros(1, dtype=wp.int32, device=device)
        self._pose_log = wp.zeros((T, num_worlds, 7), dtype=wp.float32, device=device)
        self._wheel_log = wp.zeros((T, num_worlds, 3), dtype=wp.float32, device=device)
        self._jq = wp.zeros_like(self.model.joint_q)  # _log_step's eval_ik buffers
        self._jqd = wp.zeros_like(self.model.joint_qd)
        with wp.ScopedCapture() as capture:
            self._batch_physics_step()
        graph = capture.graph

        if not view:
            for _ in range(T):
                wp.capture_launch(graph)
            wp.synchronize()
            return self._pose_log.numpy(), self._wheel_log.numpy()

        pts = [np.column_stack([p, self.terrain.sample(p[:, 0], p[:, 1]) + 0.05]).astype(np.float32)
               for p in self.paths]
        self._path_starts = wp.array(np.concatenate([q[:-1] for q in pts]), dtype=wp.vec3, device=device)
        self._path_ends = wp.array(np.concatenate([q[1:] for q in pts]), dtype=wp.vec3, device=device)
        step = 0
        while self.viewer.is_running() and step < T:
            if self.viewer.should_step():  # running, or "." pressed while paused
                wp.capture_launch(graph)
                step += 1
            self._maybe_render(step)
            wp.synchronize()
        while self.viewer.is_running():  # hold the final pose for inspection
            self._maybe_render(step)
        return self._pose_log.numpy()[:step], self._wheel_log.numpy()[:step]
