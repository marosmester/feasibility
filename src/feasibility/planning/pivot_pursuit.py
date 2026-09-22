"""Path following for a plan that TURNS IN PLACE, as a Warp kernel inside the captured physics step.

`pure_pursuit.py` cannot drive a point turn, and the limitation is structural rather than a missing
option: its waypoint buffer is `wp.vec2` (no yaw anywhere), its forward speed is one scalar fixed at
construction, and `wz = v * kappa` makes `v = 0` mean `wz = 0` identically. `resample_polyline` even
DELETES pivots before the controller could see them -- consecutive pivot poses share a cell exactly,
and repeated vertices are dropped so `np.interp` has a monotonic arc length to work with. Hand
`create_garage.py`'s map to that controller and the robot drives forward into the side wall 0.95 m
ahead, which is not a tuning failure: the garage is dimensioned so that no forward arc can get the
nose round (`clearances()`: a forward 90 deg throws a rim 1.13 m sideways, a point turn 0.71 m).

So this module is a SECOND controller, not a patch on that one -- `pure_pursuit.py` is untouched and
every existing caller keeps its behaviour. What is shared is imported from it (`WAYPOINT_SPACING`,
`SEARCH_WINDOW`, `STOP_RADIUS`, `GOAL_TOLERANCE`, `resample_polyline`, `controller_kappa_max`), so
the two stay in lockstep and `evaluation.judge` -- which imports the same two stop constants to
replicate the arrival test -- still describes both.

The controller walks a PHASE LIST built from the plan itself, so the robot turns where the planner
turned, by the amount it planned, rather than wherever a reactive heading rule happens to want to:

    PIVOT   v = 0, wz = +-omega, closed-loop on the chassis' measured yaw, one phase per 15 deg
            lattice bin, done when the remaining angle is within `yaw_tol`
    DRIVE   `pure_pursuit`'s law verbatim (monotonic nearest-waypoint progress, first waypoint at
            least `lookahead` away, kappa = 2 sin(alpha) / L clamped to `kappa_max`, constant v),
            restricted to that phase's own slice of the waypoint buffer

CLOSED-LOOP is the load-bearing word. An in-place skid-steer turn slips badly -- the rear wheel is
commanded 0 and dragged sideways at `mu_lat = MU_LAT_RATIO * mu` -- and `demos/ostrich_vel_cmd_90deg.py`
had to stretch a DRIVING arc to 5.0318 s to achieve the 90 deg whose no-slip time is 3.14 s, a 1.6x
under-rotation that is a floor for what a point turn does. Commanding `OMEGA_NOM` for
`ARC_DURATION_S` per bin open-loop would simply under-turn, so each phase measures the real yaw and
holds `wz` until the angle is actually there.

The per-phase TIMEOUT is not defensive either. A wheel jammed against a curb never reaches
`yaw_tol`, and that is the very case these plans are driven to investigate; timing out records it
(`phase_timeouts`) instead of leaving the world spinning its wheels for the rest of the rollout.

Every world follows its own phases and waypoints out of one flat buffer (`path_begin[w]` /
`phase_begin[w]`), so one build can drive many different plans -- or the same plan in every world.

`python src/feasibility/planning/pivot_pursuit.py` is the smoke test: the phase builder's asserts
run anywhere, and a flat-ground 90 deg turn is checked when CUDA is available.
"""
from __future__ import annotations

import math

import numpy as np
import warp as wp
from examples.helhest_junior.replay_real import WHEEL_DOF_OFFSET

from feasibility.comparator.common import HALF_TRACK
from feasibility.comparator.common import HelhestBatchSimulator
from feasibility.comparator.common import WHEEL_RADIUS
from feasibility.lattice_learning.arc import ARC_DURATION_S
from feasibility.lattice_learning.arc import OMEGA_NOM
from feasibility.lattice_learning.arc import PIVOT_ANGLE
from feasibility.planning.gated_lattice import N_PRIM_ARC
from feasibility.planning.gated_lattice import N_THETA
from feasibility.planning.pure_pursuit import controller_kappa_max  # noqa: F401  (re-exported)
from feasibility.planning.pure_pursuit import GOAL_TOLERANCE
from feasibility.planning.pure_pursuit import resample_polyline
from feasibility.planning.pure_pursuit import SEARCH_WINDOW
from feasibility.planning.pure_pursuit import STOP_RADIUS
from feasibility.planning.pure_pursuit import WAYPOINT_SPACING

KIND_DRIVE, KIND_PIVOT = 0, 1
# The rate the pivot rows of every `lattice_learning` dataset were generated at, so what gets driven
# here is the command the gating network was fitted on rather than a second opinion about how fast a
# point turn should be.
PIVOT_OMEGA = float(OMEGA_NOM)  # [rad/s] ~0.524 = one 15 deg bin per 0.5 s arc
YAW_TOL = math.radians(2.0)  # [rad] a phase is done this close to its target heading
# x the no-slip time for one bin. The 1.6x measured in ostrich_vel_cmd_90deg.py is for a DRIVING
# arc; an in-place turn drags the rear wheel, so this is deliberately generous -- it exists to catch
# a wheel that is jammed, not to trim a turn that is merely slow.
PIVOT_TIMEOUT_SLACK = 4.0
PHASE_RADIUS = 0.25  # [m] of its last waypoint: an intermediate DRIVE phase hands over
CHASSIS_LOCAL_IDX = 0  # body 0 of each world is the chassis -- the pose logger reads the same one


def bin_centre(yaw: float) -> float:
    """The heading `lattice_state` would quantise `yaw` to, as the lattice's own pose.

    `lattice_state` FLOORS into a bin and the planner then reasons about the bin's centre, so a
    start yaw of exactly 90 deg is planned for at 97.5 deg. See `plan_phases`' `anchor`.
    """
    step = 2.0 * math.pi / N_THETA
    return (math.floor((yaw % (2.0 * math.pi)) / step) + 0.5) * step


def plan_phases(
    result: dict,
    goal: tuple[float, float],
    spacing: float = WAYPOINT_SPACING,
    anchor: float = 0.0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """A `planners.plan_path` result -> (`path` [P, 2], `phases`), the two buffers the kernel walks.

    Pure numpy, no GPU. Primitives are classified by index rather than by comparing positions:
    `prims[i] >= N_PRIM_ARC` is a point turn, and the index also carries the DIRECTION (the
    lattice's pivot order is cw then ccw, matching `arc_network`'s `[[0, -OMEGA_NOM],
    [0, +OMEGA_NOM]]` command table), which `np.allclose` on xy could not tell us.

    One PIVOT phase per pivot primitive -- per 15 deg bin, not per run of them. Each phase
    re-measures the real yaw against an absolute target, so the tolerance does not accumulate over a
    six-bin exit, and no phase is ever asked for more than one bin, which keeps the kernel's
    overshoot test (a remaining angle that has wrapped past pi) unambiguous. Each maximal run of
    forward arcs becomes one DRIVE phase over its own resampled waypoints, and the exact `goal` is
    appended to the last of them, as `ostrich_follow_path.py` does.

    Yaw per traced pose is its bin centre, `(theta + 0.5) * 2 pi / N_THETA`.
    `demos/view_planned_path.py:path_poses` computes the same thing, but `src/feasibility` importing
    from `demos/` would be the wrong direction -- this one expression is the price of that.

    `anchor` [rad] is added to every pivot target. 0 (the default) drives the lattice's own bin
    centres, which are the poses the settle and the network actually judged; passing
    `start_yaw - bin_centre(start_yaw)` instead makes the executed turn the planned number of
    degrees, at the cost of no longer standing where the gate scored.

    A plan with no pivots yields exactly one DRIVE phase over every waypoint, i.e. plain pure
    pursuit over `resample_polyline`'s own output -- the smoke test pins that reduction.
    """
    xy = np.asarray(result["xy"], np.float64)
    prims = list(result["prims"])
    yaw = (np.asarray(result["states"])[:, 2] + 0.5) * 2.0 * np.pi / N_THETA
    kind, i0, i1, target, sign = [], [], [], [], []
    chunks: list[np.ndarray] = []
    n_way = 0

    def add_drive(lo: int, hi: int, last: bool) -> None:
        """One DRIVE phase over the traced positions xy[lo:hi+1] (plus the goal, if this is the
        last one). `nonlocal n_way` keeps the phase's waypoint slice in the flat buffer."""
        nonlocal n_way
        pts = xy[lo : hi + 1]
        if last:
            pts = np.vstack([pts, np.asarray(goal, np.float64)[None]])
        way = resample_polyline(pts, spacing)
        chunks.append(way)
        kind.append(KIND_DRIVE)
        i0.append(n_way)
        i1.append(n_way + len(way))
        target.append(0.0)
        sign.append(0.0)
        n_way += len(way)

    i, n = 0, len(prims)
    last_arc_run = max((k for k in range(n) if prims[k] < N_PRIM_ARC), default=-1)
    while i < n:
        if prims[i] >= N_PRIM_ARC:  # a point turn: one phase, one bin
            kind.append(KIND_PIVOT)
            i0.append(0)
            i1.append(0)
            target.append(float(yaw[i + 1]) + anchor)
            # prim N_PRIM_ARC is the cw (-1 bin) turn, N_PRIM_ARC + 1 the ccw (+1 bin) one
            sign.append(-1.0 if prims[i] == N_PRIM_ARC else 1.0)
            i += 1
        else:
            j = i
            while j < n and prims[j] < N_PRIM_ARC:
                j += 1
            add_drive(i, j, last=j - 1 == last_arc_run)
            i = j
    if last_arc_run < 0:  # an all-pivot plan still needs somewhere to stop
        add_drive(len(xy) - 1, len(xy) - 1, last=True)

    path = np.vstack(chunks).astype(np.float32)
    phases = dict(
        kind=np.array(kind, np.int32),
        i0=np.array(i0, np.int32),
        i1=np.array(i1, np.int32),
        target=np.array(target, np.float32),
        sign=np.array(sign, np.float32),
    )
    return path, phases


@wp.kernel
def _pivot_pursuit_kernel(
    body_q: wp.array(dtype=wp.transform),
    path_flat: wp.array(dtype=wp.vec2),  # every world's waypoints back to back
    path_begin: wp.array(dtype=wp.int32),  # [W] first waypoint of world w in path_flat
    phase_kind: wp.array(dtype=wp.int32),  # every world's phases back to back
    phase_i0: wp.array(dtype=wp.int32),  # [K] first waypoint of a DRIVE phase (world-local)
    phase_i1: wp.array(dtype=wp.int32),  # [K] one past its last
    phase_target: wp.array(dtype=wp.float32),  # [K] a PIVOT phase's absolute target yaw
    phase_sign: wp.array(dtype=wp.float32),  # [K] its direction, +1 ccw / -1 cw
    phase_begin: wp.array(dtype=wp.int32),  # [W] first phase of world w
    phase_count: wp.array(dtype=wp.int32),  # [W] phases of world w
    phase: wp.array(dtype=wp.int32),  # [W] phase in progress, LOCAL index
    phase_step: wp.array(dtype=wp.int32),  # [W] steps spent in it (the pivot timeout)
    timeouts: wp.array(dtype=wp.int32),  # [W] pivot phases that gave up -- reported, not fatal
    progress: wp.array(dtype=wp.int32),  # [W] nearest waypoint so far, local index (monotonic)
    stopped: wp.array(dtype=wp.int32),  # [W] latched 1 once arrived
    phase_log: wp.array(dtype=wp.int32),  # [T * W] the phase each logged step was driven under
    step_buf: wp.array(dtype=wp.int32),
    T: int,
    num_worlds: int,
    settle_steps: int,
    lookahead: float,
    v: float,
    kappa_max: float,
    omega: float,
    yaw_tol: float,
    pivot_timeout: int,
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
    step = step_buf[0]
    if step < T:
        phase_log[step * num_worlds + w] = phase[w]
    tf = body_q[w * bodies_per_world + CHASSIS_LOCAL_IDX]
    p = wp.transform_get_translation(tf)
    if step < settle_steps or stopped[w] == 1:
        joint_target_vel[base + 0] = 0.0
        joint_target_vel[base + 1] = 0.0
        joint_target_vel[base + 2] = 0.0
        return

    b = path_begin[w]
    g = phase_begin[w] + phase[w]
    last = int(0)
    if phase[w] == phase_count[w] - 1:
        last = 1
    # heading of the chassis x axis projected on the ground plane -- stays right when pitched
    fwd = wp.quat_rotate(wp.transform_get_rotation(tf), wp.vec3(1.0, 0.0, 0.0))
    heading = wp.atan2(fwd[1], fwd[0])
    v_cmd = float(0.0)
    wz = float(0.0)
    done = int(0)

    if phase_kind[g] == KIND_PIVOT:
        # Angle still to turn, measured the way we are turning: sign * (target - heading) folded
        # into [0, 2 pi). It starts at one bin and falls to 0; an overshoot wraps it to just under
        # 2 pi, which is why "past pi" counts as done rather than as almost a full turn to go.
        two_pi = 6.283185307179586
        rem = phase_sign[g] * (phase_target[g] - heading)
        rem = rem - two_pi * wp.floor(rem / two_pi)
        if rem < yaw_tol or rem > 3.141592653589793:
            done = 1
        elif phase_step[w] >= pivot_timeout:  # jammed: record it and move on
            done = 1
            timeouts[w] = timeouts[w] + 1
        else:
            wz = phase_sign[g] * omega
    else:
        i0 = phase_i0[g]
        i1 = phase_i1[g]
        best = progress[w]
        q = path_flat[b + best]
        best_d = (q[0] - p[0]) * (q[0] - p[0]) + (q[1] - p[1]) * (q[1] - p[1])
        for k in range(best + 1, wp.min(best + SEARCH_WINDOW, i1)):
            q = path_flat[b + k]
            d = (q[0] - p[0]) * (q[0] - p[0]) + (q[1] - p[1]) * (q[1] - p[1])
            if d < best_d:
                best = k
                best_d = d
        progress[w] = best

        end = path_flat[b + i1 - 1]
        d_end = (end[0] - p[0]) * (end[0] - p[0]) + (end[1] - p[1]) * (end[1] - p[1])
        if last == 1:
            # the last phase ends at the goal, so this is pure_pursuit's own arrival test
            if d_end < stop_radius * stop_radius or (
                best == i1 - 1 and d_end < goal_tolerance * goal_tolerance
            ):
                stopped[w] = 1
                joint_target_vel[base + 0] = 0.0
                joint_target_vel[base + 1] = 0.0
                joint_target_vel[base + 2] = 0.0
                return
        elif best == i1 - 1 and d_end < PHASE_RADIUS * PHASE_RADIUS:
            done = 1

        if done == 0:
            target = i1 - 1
            found = int(0)
            for k in range(best, i1):
                if found == 0:
                    q = path_flat[b + k]
                    d = (q[0] - p[0]) * (q[0] - p[0]) + (q[1] - p[1]) * (q[1] - p[1])
                    if d >= lookahead * lookahead:
                        target = k
                        found = 1
            t = path_flat[b + target]
            dx = t[0] - p[0]
            dy = t[1] - p[1]
            alpha = wp.atan2(dy, dx) - heading
            alpha = wp.atan2(wp.sin(alpha), wp.cos(alpha))  # wrap to (-pi, pi]
            dist = wp.max(wp.sqrt(dx * dx + dy * dy), 1e-3)
            v_cmd = v
            wz = v * wp.clamp(2.0 * wp.sin(alpha) / dist, -kappa_max, kappa_max)

    if done == 1:
        # hand over on the next step, from a standstill: with the phase advanced here, one settling
        # step between phases costs nothing and keeps each branch reading only its own state
        if last == 1:
            stopped[w] = 1
        else:
            phase[w] = phase[w] + 1
            phase_step[w] = 0
            g = phase_begin[w] + phase[w]
            if phase_kind[g] == KIND_DRIVE:
                progress[w] = phase_i0[g]
        joint_target_vel[base + 0] = 0.0
        joint_target_vel[base + 1] = 0.0
        joint_target_vel[base + 2] = 0.0
        return

    phase_step[w] = phase_step[w] + 1
    # ideal no-slip differential drive, the same map comparator.common.cmd_to_wheels applies
    joint_target_vel[base + 0] = (v_cmd - wz * half_track) / wheel_radius
    joint_target_vel[base + 1] = (v_cmd + wz * half_track) / wheel_radius
    joint_target_vel[base + 2] = v_cmd / wheel_radius


class PivotPursuitSimulator(HelhestBatchSimulator):
    """HelhestBatchSimulator whose per-step control is the phase-walking kernel above: world w
    follows `paths[w]` ([P_w, 2] waypoints in the build's own coordinates, so already shifted onto
    its tile on a TiledTerrain) through the phases of `phases[w]` (`plan_phases`' second return)."""

    def __init__(
        self,
        *args,
        paths: list[np.ndarray],
        phases: list[dict[str, np.ndarray]],
        settle_steps: int,
        lookahead: float,
        v: float,
        kappa_max: float,
        omega: float = PIVOT_OMEGA,
        yaw_tol: float = YAW_TOL,
        pivot_timeout_steps: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        device = self.model.device
        num_worlds = self.simulation_config.num_worlds
        assert len(paths) == num_worlds, f"{len(paths)} paths for {num_worlds} worlds"
        assert len(phases) == num_worlds, f"{len(phases)} phase lists for {num_worlds} worlds"
        self.paths = [np.asarray(p, np.float32) for p in paths]
        self.phases = phases
        counts = np.array([len(p) for p in self.paths], np.int32)
        begins = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int32)
        self._path_flat = wp.array(np.concatenate(self.paths), dtype=wp.vec2, device=device)
        self._path_begin = wp.array(begins, dtype=wp.int32, device=device)

        p_counts = np.array([len(f["kind"]) for f in phases], np.int32)
        p_begins = np.concatenate([[0], np.cumsum(p_counts)[:-1]]).astype(np.int32)
        def cat(key: str, np_dt, wp_dt) -> wp.array:
            flat = np.concatenate([f[key] for f in phases]).astype(np_dt)
            return wp.array(flat, dtype=wp_dt, device=device)

        self._phase_kind = cat("kind", np.int32, wp.int32)
        self._phase_i0 = cat("i0", np.int32, wp.int32)
        self._phase_i1 = cat("i1", np.int32, wp.int32)
        self._phase_target = cat("target", np.float32, wp.float32)
        self._phase_sign = cat("sign", np.float32, wp.float32)
        self._phase_begin = wp.array(p_begins, dtype=wp.int32, device=device)
        self._phase_count = wp.array(p_counts, dtype=wp.int32, device=device)
        self._phase = wp.zeros(num_worlds, dtype=wp.int32, device=device)
        self._phase_step = wp.zeros(num_worlds, dtype=wp.int32, device=device)
        self._timeouts = wp.zeros(num_worlds, dtype=wp.int32, device=device)
        # a world starting on a DRIVE phase must start at THAT phase's first waypoint
        first = np.array([f["i0"][0] if f["kind"][0] == KIND_DRIVE else 0 for f in phases], np.int32)
        self._progress = wp.array(first, dtype=wp.int32, device=device)
        self._stopped = wp.zeros(num_worlds, dtype=wp.int32, device=device)
        self._settle_steps = int(settle_steps)
        self._lookahead, self._v, self._kappa_max = float(lookahead), float(v), float(kappa_max)
        self._omega, self._yaw_tol = float(omega), float(yaw_tol)
        self._pivot_timeout = int(pivot_timeout_steps)

    def _batch_physics_step(self) -> None:
        # HelhestBatchSimulator._batch_physics_step with the setpoint kernel swapped for the phases
        self.current_state.clear_forces()
        self.contacts = self.model.collide(self.current_state)
        wp.launch(
            kernel=_pivot_pursuit_kernel,
            dim=self.simulation_config.num_worlds,
            inputs=[
                self.current_state.body_q, self._path_flat, self._path_begin, self._phase_kind,
                self._phase_i0, self._phase_i1, self._phase_target, self._phase_sign,
                self._phase_begin, self._phase_count, self._phase, self._phase_step,
                self._timeouts, self._progress, self._stopped, self._phase_log, self._step_buf,
                self._T, self.simulation_config.num_worlds, self._settle_steps, self._lookahead, self._v, self._kappa_max,
                self._omega, self._yaw_tol, self._pivot_timeout, STOP_RADIUS, GOAL_TOLERANCE,
                float(WHEEL_RADIUS), HALF_TRACK, self.control.joint_target_vel,
                self.bodies_per_world, self.dofs_per_world, WHEEL_DOF_OFFSET,
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

    def rollout(self, T: int, view: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Settle + drive as ONE captured step launched T times (the settle is the kernel's first
        `settle_steps` rows), exactly as `PurePursuitSimulator.rollout`. Returns pose [T, W, 7],
        wheel_qd [T, W, 3] and phase [T, W] -- row t of the last being the phase that PRODUCED
        pose[t], since the control kernel writes it before the solver steps. Fewer rows if the
        viewer was closed early."""
        device = self.model.device
        num_worlds = self.simulation_config.num_worlds
        self._T = T
        self._step_buf = wp.zeros(1, dtype=wp.int32, device=device)
        self._pose_log = wp.zeros((T, num_worlds, 7), dtype=wp.float32, device=device)
        self._wheel_log = wp.zeros((T, num_worlds, 3), dtype=wp.float32, device=device)
        self._phase_log = wp.zeros(T * num_worlds, dtype=wp.int32, device=device)
        self._jq = wp.zeros_like(self.model.joint_q)  # _log_step's eval_ik buffers
        self._jqd = wp.zeros_like(self.model.joint_qd)
        with wp.ScopedCapture() as capture:
            self._batch_physics_step()
        graph = capture.graph

        def phases_out(n: int) -> np.ndarray:
            return self._phase_log.numpy().reshape(T, num_worlds)[:n]

        if not view:
            for _ in range(T):
                wp.capture_launch(graph)
            wp.synchronize()
            return self._pose_log.numpy(), self._wheel_log.numpy(), phases_out(T)

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
        return self._pose_log.numpy()[:step], self._wheel_log.numpy()[:step], phases_out(step)

    def timeouts(self) -> np.ndarray:
        """[W] pivot phases that hit `pivot_timeout_steps` without reaching their heading."""
        return self._timeouts.numpy().copy()


def pivot_timeout_steps(dt: float, slack: float = PIVOT_TIMEOUT_SLACK, omega: float = PIVOT_OMEGA) -> int:
    """Step budget for one 15 deg phase: `slack` x the no-slip time for that bin."""
    return int(math.ceil(slack * PIVOT_ANGLE / omega / dt))


def _fake_result(prims: list[int], start_bin: int, step: float = 0.3) -> dict:
    """A `plan_path`-shaped result for the asserts below: straight-east arcs and in-place turns,
    with the `xy`/`states`/`prims` keys `plan_phases` reads and nothing else."""
    xy, states, x, b = [[0.0, 0.0]], [[0, 0, start_bin]], 0.0, start_bin
    for p in prims:
        if p >= N_PRIM_ARC:
            b = (b + (1 if p == N_PRIM_ARC + 1 else -1)) % N_THETA
        else:
            x += step
        xy.append([x, 0.0])
        states.append([0, int(round(x / step)), b])
    return dict(xy=np.array(xy), states=np.array(states), prims=prims)


def _check_phase_builder() -> None:
    goal = (2.0, 0.0)

    # 1. no pivots -> one DRIVE phase over every waypoint, and those waypoints are exactly what
    #    pure pursuit would have been handed. This is the "reduces to the old controller" claim.
    plain = _fake_result([0, 0, 0, 0], start_bin=0)
    path, ph = plan_phases(plain, goal)
    assert ph["kind"].tolist() == [KIND_DRIVE], ph["kind"]
    assert (ph["i0"][0], ph["i1"][0]) == (0, len(path)), (ph["i0"], ph["i1"], len(path))
    expect = resample_polyline(np.vstack([plain["xy"], np.array(goal)[None]]))
    assert np.allclose(path, expect.astype(np.float32)), "a pivot-free plan must be plain pursuit"

    # 2. pivots first, then a drive: one phase per bin, directions and targets off the END state
    turn = _fake_result([N_PRIM_ARC + 1] * 6 + [0, 0], start_bin=6)
    path, spin = plan_phases(turn, goal)
    ph = spin
    assert ph["kind"].tolist() == [KIND_PIVOT] * 6 + [KIND_DRIVE], ph["kind"]
    assert np.all(ph["sign"][:6] == 1.0), ph["sign"]
    bins = np.arange(7, 13) + 0.5
    assert np.allclose(ph["target"][:6], bins * 2.0 * np.pi / N_THETA), ph["target"]
    assert (ph["i0"][-1], ph["i1"][-1]) == (0, len(path)), (ph["i0"], ph["i1"])

    # 3. cw turns carry the other sign, and a run of arcs between two turns is its own phase
    mixed = _fake_result([0, N_PRIM_ARC, N_PRIM_ARC, 0, 0], start_bin=3)
    _, ph = plan_phases(mixed, goal)
    assert ph["kind"].tolist() == [KIND_DRIVE, KIND_PIVOT, KIND_PIVOT, KIND_DRIVE], ph["kind"]
    assert np.all(ph["sign"][1:3] == -1.0), ph["sign"]
    assert ph["i1"][0] == ph["i0"][3], "drive phases must partition the waypoint buffer"

    # 4. the anchor shifts every target and nothing else
    _, shifted = plan_phases(turn, goal, anchor=-0.1)
    assert np.allclose(shifted["target"][:6], spin["target"][:6] - 0.1), shifted["target"]
    assert shifted["kind"].tolist() == spin["kind"].tolist(), "the anchor moved more than the targets"
    # 90 deg sits in bin 6, whose centre is 97.5 -- the offset `anchor` exists to absorb
    assert math.isclose(bin_centre(math.pi / 2.0), math.radians(97.5)), bin_centre(math.pi / 2.0)
    print("phase builder: pivot-free plans reduce to pure pursuit, turns split per bin")


def _check_turn_on_flat() -> None:
    """The closed-loop claim, on ground that cannot be blamed: command the six-bin exit turn on
    flat terrain and check the robot really gets there. Open-loop at OMEGA_NOM would land ~40 deg
    short (see the module docstring), so this fails loudly if the feedback is ever lost."""
    from feasibility.comparator.common import friction_kwargs
    from feasibility.comparator.common import K_P
    from feasibility.heightmap import HeightMapReader
    from feasibility.lattice_learning.generate_dataset import compose_ostrich_config

    terrain = HeightMapReader.flat(xlim=(-5.0, 5.0), ylim=(-5.0, 5.0))
    start_bin = 6
    yaw0 = (start_bin + 0.5) * 2.0 * math.pi / N_THETA  # spawn ON the bin centre: one bin per phase
    result = _fake_result([N_PRIM_ARC + 1] * 6, start_bin=start_bin)
    path, phases = plan_phases(result, (0.0, 0.0))
    sim_config, render_config, engine_config, logging_config = compose_ostrich_config(())
    render_config.vis_type = "null"
    sim_config.num_worlds = 1
    dt = float(sim_config.target_timestep_seconds)
    budget = pivot_timeout_steps(dt)
    settle = 20
    sim = None
    try:
        sim = PivotPursuitSimulator(
            sim_config, render_config, engine_config, logging_config,
            k_p=K_P, **friction_kwargs(0.8), terrain=terrain,
            spawn_pose=np.array([[0.0, 0.0, yaw0]]), paths=[path], phases=[phases],
            settle_steps=settle, lookahead=0.4, v=0.6, kappa_max=controller_kappa_max(),
            pivot_timeout_steps=budget,
        )
        pose, _, phase_log = sim.rollout(settle + 6 * budget + 40)
        n_timeout = int(sim.timeouts()[0])
    finally:
        sim = None
    # Measure at the END OF THE TURN, not at the end of the rollout: the trailing DRIVE phase is
    # still chasing the goal, and the turn's own xy drift means it has somewhere to go.
    n_pivot = int((phases["kind"] == KIND_PIVOT).sum())
    turning = np.nonzero(phase_log[:, 0] < n_pivot)[0]
    assert len(turning), "the rollout never entered a pivot phase"
    t_end = int(turning[-1])
    q = pose[t_end, 0, 3:7]
    yaw = math.atan2(2.0 * (q[3] * q[2] + q[0] * q[1]), 1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2))
    want = float(phases["target"][n_pivot - 1])
    err = abs(math.atan2(math.sin(yaw - want), math.cos(yaw - want)))
    turned = math.atan2(math.sin(yaw - yaw0), math.cos(yaw - yaw0))
    took = (t_end - settle + 1) * dt
    ideal = n_pivot * PIVOT_ANGLE / PIVOT_OMEGA
    drift = float(np.linalg.norm(pose[t_end, 0, :2] - pose[settle, 0, :2]))
    print(f"flat-ground turn: commanded {math.degrees(want - yaw0):.1f} deg, achieved "
          f"{math.degrees(turned):.1f} deg, miss {math.degrees(err):.2f} deg, {n_timeout} timeouts")
    print(f"  took {took:.1f} s against {ideal:.1f} s of no-slip command ({took / ideal:.1f}x -- "
          f"which is why a pivot phase is closed-loop), drifted {drift:.2f} m off the spot")
    assert n_timeout == 0, f"{n_timeout} pivot phases timed out on FLAT ground"
    assert err < math.radians(6.0), f"ended {math.degrees(err):.1f} deg off the commanded heading"
    assert took < PIVOT_TIMEOUT_SLACK * ideal, "the turn is eating its whole timeout budget"


if __name__ == "__main__":
    wp.init()
    _check_phase_builder()
    if wp.is_cuda_available():
        _check_turn_on_flat()
    else:
        print("no CUDA: skipped the flat-ground turn (the phase builder's asserts ran)")
    print(f"omega {PIVOT_OMEGA:.4f} rad/s  bin {math.degrees(PIVOT_ANGLE):.1f} deg  "
          f"arc {ARC_DURATION_S} s  yaw_tol {math.degrees(YAW_TOL):.1f} deg")
