"""Plan on a heightmap with one of the uphill benchmark's planners, then have Helhest Junior FOLLOW that
path in ostrich -- the physics check on a planned path, not just on a straight drive.

`benchmarks/bench_uphill_nn.py` shows the NN-gated planners reaching the 60 deg plateau by crossing
the face obliquely, while `demos/ostrich_ramp_crossing.py` only ever drove straight at it. This
demo drives the planned line itself:

    1. load a heightmap (any PNG+YAML; start/goal from its sidecar or --start/--goal)
    2. plan start -> goal with --planner, via the benchmark's own solver + trace
       (vanilla-off: no `blocked`; vanilla-on: settle `blocked`; nn-gated-pos / nn-gated-pitch /
       nn-gated-rot: arcs pruned where the network's predicted e_pos / e_pitch / e_rot exceeds tau;
       nn-gated-fused: pruned where e_pos > --tau-fused-pos OR e_rot > --tau-fused-rot; e_rot and
       the fused gate's e_pos come from the pos_rot checkpoint --checkpoint-rot, the others from
       --checkpoint)
    3. drive it in ostrich with a pure-pursuit controller running INSIDE the captured physics step
       (a Warp kernel reading the chassis pose, no host round trip), --repeats worlds in one build
       because ostrich is nondeterministic on contact-rich driving
    4. report per repeat: arrived / flipped / off_map / stalled / nonfinite, peak climb pitch and
       |roll|, cross-track error to the planned path (max, mean, and at the peak-pitch moment)

Path and controller
    * Waypoints are the traced lattice states (cell-corner poses, the front-axle base frame the
      settle and the network judge; ostrich's chassis body 0 is the same frame), plus the exact
      goal, joined by straight chords and resampled every 5 cm. The chords zig-zag +-7.5 deg
      where the lattice has no 0 deg heading; the lookahead smooths that out.
    * Pure pursuit on the chassis' horizontal heading: a monotonic nearest-waypoint progress
      index (forward window only, so it cannot jump back), target = first waypoint at least
      --lookahead away, kappa = 2 sin(alpha) / L clamped to the lattice's own turn limit
      (1 / min_turn_radius), constant forward speed --v (0.6 m/s, the network's training speed),
      wheel targets by the same ideal diff-drive map as comparator.common.cmd_to_wheels.
    * A world stops (wheels zeroed, latched) once within STOP_RADIUS of the goal.

Outputs (<out-dir>/<map>_<planner>.*)
    .png  bird's-eye elevation, planned path dashed, each repeat's executed track coloured by outcome
    .h5   comparator/provenance's schema (poses after the settle, empty hstack/ group), plus the
          planned path as `planned_path`; replay one repeat K with
          python src/feasibility/replay/gl_replay.py --file <out-dir>/<map>_<planner>.h5 --id K --which ostrich

Run as a module (it imports benchmarks.bench_uphill_nn and demos.ostrich_ramp_crossing, which need
the repo root on sys.path -- see the root CLAUDE.md):
    python -m demos.ostrich_follow_path --map assets/uphill_series/uphill_a0600 --planner nn-gated-pos
    python -m demos.ostrich_follow_path --map assets/uphill_series/uphill_a0600 --planner vanilla-off --view
    python -m demos.ostrich_follow_path --map assets/uphill_series/uphill_a0650 --planner nn-gated-pitch \
        --tau-pitch 0.2 --repeats 5

CLI parameters:
    --map PATH            heightmap stem (PNG + YAML), required
    --planner NAME        vanilla-off | vanilla-on | nn-gated-pos | nn-gated-pitch | nn-gated-rot |
                          nn-gated-fused, required
    --start X Y YAW       start pose (default: the sidecar's `start`)
    --goal X Y            goal (default: the sidecar's `goal`)
    --tau-pos FLOAT       nn-gated-pos threshold [m] (default: bench_uphill_nn.TAU_POS)
    --tau-pitch FLOAT     nn-gated-pitch threshold [rad] (default: bench_uphill_nn.TAU_PITCH)
    --tau-rot FLOAT       nn-gated-rot threshold [rad] (default: TAU_ROT = 0.38)
    --tau-fused-pos FLOAT nn-gated-fused e_pos threshold [m] (default: TAU_FUSED_POS = 0.30)
    --tau-fused-rot FLOAT nn-gated-fused e_rot threshold [rad] (default: TAU_FUSED_ROT = 0.46)
    --checkpoint PATH     pos_rpy network checkpoint (default: bench_uphill_nn.DEFAULT_CHECKPOINT)
    --checkpoint-rot PATH pos_rot network checkpoint (default: ..._M200_R8_seed0.pt)
    --torch-device STR    network inference device (default: cpu)
    --chunk INT           patches per network batch (default: 4096)
    --v FLOAT             forward speed [m/s] (default: 0.6)
    --lookahead FLOAT     pure-pursuit lookahead [m] (default: 0.4)
    --repeats INT         worlds following the same path (default: 3; 1 with --view)
    --mu FLOAT            ground friction (default: 0.8)
    --slack FLOAT         drive time as a multiple of path length / v (default: 2.0)
    --settle-steps INT    zero-command drop onto the terrain before driving (default: 20)
    --view                live GL viewer instead of the headless rollout
    --out-dir PATH        output directory (default: outputs/follow_path)
    --override K=V ...    ostrich Hydra overrides, e.g. simulation.target_timestep_seconds=0.03
"""
from __future__ import annotations

import argparse
import gc
import math
import pathlib
import time

import numpy as np
import torch
import warp as wp
import yaml
from examples.helhest_junior.replay_real import WHEEL_DOF_OFFSET
from helhest import dynamics
from helhest.planning.costtogo import CostToGo

from benchmarks.bench_uphill_nn import arc_error_fields
from benchmarks.bench_uphill_nn import arm_result
from benchmarks.bench_uphill_nn import build_gated_solver
from benchmarks.bench_uphill_nn import DEFAULT_CHECKPOINT
from benchmarks.bench_uphill_nn import gated_solve
from benchmarks.bench_uphill_nn import lattice_state
from benchmarks.bench_uphill_nn import load_network
from benchmarks.bench_uphill_nn import N_THETA
from benchmarks.bench_uphill_nn import STEP
from benchmarks.bench_uphill_nn import TAU_PITCH
from benchmarks.bench_uphill_nn import TAU_POS
from demos.ostrich_ramp_crossing import pitch_roll
from feasibility.comparator.common import HALF_TRACK
from feasibility.comparator.common import HelhestBatchSimulator
from feasibility.comparator.common import K_P
from feasibility.comparator.common import WHEEL_RADIUS
from feasibility.comparator.provenance import write_comparison
from feasibility.heightmap import HeightMapReader
from feasibility.lattice_learning.generate_dataset import compose_ostrich_config

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PLANNERS = {  # CLI name -> network head it gates on (None = no network)
    "vanilla-off": None,
    "vanilla-on": None,
    "nn-gated-pos": "e_pos",
    "nn-gated-pitch": "e_pitch",
    "nn-gated-rot": "e_rot",
    "nn-gated-fused": "fused",
}
# e_rot lives in the pos_rot sibling of the default (pos_rpy) checkpoint, trained on the same data
DEFAULT_CHECKPOINT_ROT = REPO_ROOT / "outputs" / "checkpoints" / "dataset_arc_my_config_M200_R8_seed0.pt"
# Not calibrated like TAU_POS/TAU_PITCH on tau*, but on ostrich_follow_path_parallel_worlds runs of
# the uphill series: 0.50 and 0.45 plan straight through 60 deg, where pure pursuit arrives only
# 3/6; 0.38 is below 60 deg's bottleneck (0.391), so 60+ has no path, while 5-55 deg stay straight.
TAU_ROT = 0.38
# nn-gated-fused: an arc is pruned when EITHER of the pos_rot net's heads is over its own threshold,
# i.e. max(e_pos / TAU_FUSED_POS, e_rot / TAU_FUSED_ROT) > 1. An offline planner scan of the uphill
# series (straight / angled / no path per map, no ostrich): e_pos alone blocks 60 deg and up without
# ever admitting an angled path for 0.25-0.35, which widens e_rot's safe window from 0.36-0.38 to
# 0.36-0.56; both constants sit mid-window. Two thresholds fitted on one boundary -- unvalidated.
TAU_FUSED_POS = 0.30
TAU_FUSED_ROT = 0.46
WAYPOINT_SPACING = 0.05  # [m] resampled path the controller walks
SEARCH_WINDOW = 40  # waypoints ahead the progress index may advance per step (2 m)
STOP_RADIUS = 0.15  # [m] from the goal: the world stops and counts as arrived
OFF_MAP_MARGIN = 0.5  # [m] from any grid edge
CHASSIS_LOCAL_IDX = 0  # body 0 of each world is the chassis -- the pose logger reads the same one
STATUS_COLORS = {"arrived": "tab:green", "flipped": "tab:red", "off_map": "tab:orange",
                 "stalled": "white", "nonfinite": "magenta"}


# --- planning -------------------------------------------------------------------------------------


class PlanContext:
    """What planning can reuse across (map, planner) calls: the CostToGo + gated solver while the
    grid geometry is unchanged, each network loaded once, and the predicted error fields of the
    map last seen (one inference pass per network per map). Field keys: e_pos / e_pitch from the
    pos_rpy net, e_pos_rot / e_rot from the pos_rot net, and fused = max of the latter two over
    their fused thresholds, gated at tau 1."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.torch_device = torch.device(args.torch_device)
        self.taus = {"e_pos": float(args.tau_pos), "e_pitch": float(args.tau_pitch),
                     "e_rot": float(args.tau_rot), "fused": 1.0}
        self._grid_key = None
        self.ctg: CostToGo | None = None
        self.gated = None
        self._models: dict[str, object] = {}
        self._fields_key = None
        self._fields: dict[str, np.ndarray] = {}

    def planner_for(self, terrain: HeightMapReader) -> tuple[CostToGo, object]:
        elev, grid = terrain.to_hstack("cuda")
        key = (grid.cells_x, grid.cells_y, grid.cell_size, grid.origin_x, grid.origin_y)
        if key != self._grid_key:
            self.ctg = None  # drop the old buffers before allocating new ones (3 GiB GPU)
            self._models.clear()  # load_network checks against the lattice, so reload with it
            self._fields_key, self._fields = None, {}
            gc.collect()
            self.ctg = CostToGo(grid, dynamics.robot_params(), dynamics.planning_solver(),
                                n_theta=N_THETA, step=STEP, device="cuda")
            self.gated = build_gated_solver(self.ctg)
            self._grid_key = key
        self.elev = elev
        return self.ctg, self.gated

    def fields(self, terrain: HeightMapReader, head: str) -> dict[str, np.ndarray]:
        """Predicted error fields for `head` (plus whatever else that network outputs) on `terrain`."""
        if self._fields_key is not terrain:
            self._fields_key, self._fields = terrain, {}
        if head not in self._fields:
            mode = "pos_rot" if head in ("e_rot", "fused") else "pos_rpy"
            heads = ("e_pos", "e_rot") if mode == "pos_rot" else ("e_pos", "e_pitch")
            if mode not in self._models:
                path = self.args.checkpoint_rot if mode == "pos_rot" else self.args.checkpoint
                self._models[mode] = load_network(path, self.torch_device, self.ctg, label_mode=mode)
            t0 = time.time()
            out = arc_error_fields(self._models[mode], terrain, self.ctg, self.args.chunk,
                                   self.torch_device, heads=heads)
            print(f"network inference ({mode}): {time.time() - t0:.1f} s")
            if mode == "pos_rot":  # its e_pos is a different net's than nn-gated-pos gates on
                out["e_pos_rot"] = out.pop("e_pos")
                out["fused"] = np.maximum(out["e_pos_rot"] / float(self.args.tau_fused_pos),
                                          out["e_rot"] / float(self.args.tau_fused_rot))
            self._fields.update(out)
        return self._fields


def plan_path(
    terrain: HeightMapReader,
    start: tuple[float, float, float],
    goal: tuple[float, float],
    planner: str,
    ctx: PlanContext,
) -> dict:
    """The benchmark arm matching `planner`, solved and traced from `start`. Returns arm_result's
    dict plus `tau`, `head` (None for the vanilla planners) and `gate`, a printable description of
    the threshold ("" for the vanilla planners)."""
    ctg, gated = ctx.planner_for(terrain)
    v_on = ctg.compute(ctx.elev, goal).numpy().copy()
    blocked = ctg.blocked.numpy().copy()
    tilt = ctg.graded_tilt.numpy().copy()
    no_block = np.zeros_like(blocked)
    zeros = wp.zeros_like(ctg.blocked)
    start_rct = lattice_state(*start, ctg)
    grid = ctg.grid
    head = PLANNERS[planner]
    tau = math.inf

    if planner == "vanilla-on":
        result = arm_result(ctg, v_on, blocked, tilt, blocked, {}, start_rct)
    elif planner == "vanilla-off":
        v_off = ctg.solver._record_solve(zeros, ctg.graded_tilt, ctg._goal_rc, ctg.flatness_weight,
                                         False).numpy()
        result = arm_result(ctg, v_off, no_block, tilt, blocked, {}, start_rct)
    else:
        fields = ctx.fields(terrain, head)
        tau = ctx.taus[head]
        err = wp.array(fields[head], dtype=wp.float32, device=ctg.device)
        v = gated_solve(gated, ctg, zeros, ctg.graded_tilt, err, tau)
        result = arm_result(ctg, v, no_block, tilt, blocked, fields, start_rct, fields[head], tau)

    rct = result["states"]
    if head is None:
        gate = ""
    elif head == "fused":
        gate = (f"max(e_pos/{ctx.args.tau_fused_pos:.4f}, e_rot/{ctx.args.tau_fused_rot:.4f}) > 1")
    else:
        gate = f"{head} > {tau:.4f}"
    result.update(
        head=head, tau=tau, gate=gate,
        xy=np.stack([grid.origin_x + rct[:, 1] * grid.cell_size,
                     grid.origin_y + rct[:, 0] * grid.cell_size], axis=-1),
    )
    return result


def resample_polyline(xy: np.ndarray, spacing: float) -> np.ndarray:
    """[n, 2] vertices -> [m, 2] points every `spacing` m along the chords, both ends kept."""
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    keep = np.concatenate([[True], seg > 1e-9])  # repeated vertices would break np.interp
    s, xy = s[keep], xy[keep]
    q = np.append(np.arange(0.0, s[-1], spacing), s[-1])
    return np.stack([np.interp(q, s, xy[:, 0]), np.interp(q, s, xy[:, 1])], axis=-1)


# --- simulation -----------------------------------------------------------------------------------


@wp.kernel
def _pure_pursuit_kernel(
    body_q: wp.array(dtype=wp.transform),
    path: wp.array(dtype=wp.vec2),  # [P] waypoints, shared by every world
    progress: wp.array(dtype=wp.int32),  # [W] nearest waypoint so far (monotonic)
    stopped: wp.array(dtype=wp.int32),  # [W] latched 1 once within stop_radius of the goal
    step_buf: wp.array(dtype=wp.int32),
    settle_steps: int,
    lookahead: float,
    v: float,
    kappa_max: float,
    stop_radius: float,
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
    n = path.shape[0]
    goal = path[n - 1]
    if (goal[0] - p[0]) * (goal[0] - p[0]) + (goal[1] - p[1]) * (goal[1] - p[1]) < stop_radius * stop_radius:
        stopped[w] = 1
    if step_buf[0] < settle_steps or stopped[w] == 1:
        joint_target_vel[base + 0] = 0.0
        joint_target_vel[base + 1] = 0.0
        joint_target_vel[base + 2] = 0.0
        return

    i0 = progress[w]
    best = i0
    best_d = (path[i0][0] - p[0]) * (path[i0][0] - p[0]) + (path[i0][1] - p[1]) * (path[i0][1] - p[1])
    for k in range(i0 + 1, wp.min(i0 + SEARCH_WINDOW, n)):
        d = (path[k][0] - p[0]) * (path[k][0] - p[0]) + (path[k][1] - p[1]) * (path[k][1] - p[1])
        if d < best_d:
            best = k
            best_d = d
    progress[w] = best

    target = n - 1
    found = int(0)
    for k in range(best, n):
        if found == 0:
            d = (path[k][0] - p[0]) * (path[k][0] - p[0]) + (path[k][1] - p[1]) * (path[k][1] - p[1])
            if d >= lookahead * lookahead:
                target = k
                found = 1

    # heading of the chassis x axis projected on the ground plane -- stays right when pitched
    fwd = wp.quat_rotate(wp.transform_get_rotation(tf), wp.vec3(1.0, 0.0, 0.0))
    dx = path[target][0] - p[0]
    dy = path[target][1] - p[1]
    alpha = wp.atan2(dy, dx) - wp.atan2(fwd[1], fwd[0])
    alpha = wp.atan2(wp.sin(alpha), wp.cos(alpha))  # wrap to (-pi, pi]
    dist = wp.max(wp.sqrt(dx * dx + dy * dy), 1e-3)
    kappa = wp.clamp(2.0 * wp.sin(alpha) / dist, -kappa_max, kappa_max)
    wz = v * kappa
    joint_target_vel[base + 0] = (v - wz * half_track) / wheel_radius
    joint_target_vel[base + 1] = (v + wz * half_track) / wheel_radius
    joint_target_vel[base + 2] = v / wheel_radius


class PathFollowSimulator(HelhestBatchSimulator):
    """HelhestBatchSimulator whose per-step control is the pure-pursuit kernel instead of a
    setpoint table; every world follows the same `path_xy`."""

    def __init__(
        self,
        *args,
        path_xy: np.ndarray,
        settle_steps: int,
        lookahead: float,
        v: float,
        kappa_max: float,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        device = self.model.device
        num_worlds = self.simulation_config.num_worlds
        self.path_xy = path_xy
        self._path = wp.array(path_xy.astype(np.float32), dtype=wp.vec2, device=device)
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
                self.current_state.body_q, self._path, self._progress, self._stopped,
                self._step_buf, self._settle_steps, self._lookahead, self._v, self._kappa_max,
                STOP_RADIUS, float(WHEEL_RADIUS), HALF_TRACK, self.control.joint_target_vel,
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

    def rollout(self, T: int, view: bool) -> tuple[np.ndarray, np.ndarray]:
        """Settle + drive as ONE captured step launched T times (the settle is the kernel's first
        `settle_steps` rows). Headless launches back to back; --view renders between launches
        (replay_real.replay_graph's GL pattern) and holds the window on the last frame.
        Returns pose [T, W, 7], wheel_qd [T, W, 3]."""
        device = self.model.device
        num_worlds = self.simulation_config.num_worlds
        self._T = T
        self._step_buf = wp.zeros(1, dtype=wp.int32, device=device)
        self._pose_log = wp.zeros((T, num_worlds, 7), dtype=wp.float32, device=device)
        self._wheel_log = wp.zeros((T, num_worlds, 3), dtype=wp.float32, device=device)
        self._jq = wp.zeros_like(self.model.joint_q)
        self._jqd = wp.zeros_like(self.model.joint_qd)
        with wp.ScopedCapture() as capture:
            self._batch_physics_step()
        graph = capture.graph

        if not view:
            for _ in range(T):
                wp.capture_launch(graph)
            wp.synchronize()
            return self._pose_log.numpy(), self._wheel_log.numpy()

        z = self.terrain.sample(self.path_xy[:, 0], self.path_xy[:, 1]) + 0.05
        pts = np.column_stack([self.path_xy, z]).astype(np.float32)
        self._path_starts = wp.array(pts[:-1], dtype=wp.vec3, device=device)
        self._path_ends = wp.array(pts[1:], dtype=wp.vec3, device=device)
        step = 0
        while self.viewer.is_running() and step < T:
            if not self.viewer.is_paused():
                wp.capture_launch(graph)
                step += 1
            self._maybe_render(step)
            wp.synchronize()
        while self.viewer.is_running():  # hold the final pose for inspection
            self._maybe_render(step)
        return self._pose_log.numpy()[:step], self._wheel_log.numpy()[:step]


# --- verdict --------------------------------------------------------------------------------------


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
    arrival (or the first non-finite row)."""
    finite = np.isfinite(pose).all(axis=1)
    end = len(pose) if finite.all() else int(np.argmin(finite))
    p = pose[: max(end, 1)]
    goal = path[-1]
    arrived = np.nonzero(np.linalg.norm(p[:, :2] - goal, axis=1) < STOP_RADIUS)[0]
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


def plot_run(
    terrain: HeightMapReader,
    plan: dict,
    path: np.ndarray,
    results: list[dict],
    start: tuple[float, float, float],
    title: str,
    out: pathlib.Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = terrain
    extent = (t.x0, t.x0 + t.nx * t.cell, t.y0, t.y0 + t.ny * t.cell)
    fig, ax = plt.subplots(figsize=(12, 5.5), layout="constrained")
    im = ax.imshow(t.H, origin="lower", extent=extent, cmap="viridis", interpolation="nearest")
    ax.plot(plan["xy"][:, 0], plan["xy"][:, 1], "--", color="black", lw=2, label="planned path")
    for i, r in enumerate(results):
        ax.plot(r["xy"][:, 0], r["xy"][:, 1], "-", lw=1.2, color=STATUS_COLORS[r["status"]],
                label=f"repeat {i}: {r['status']}")
    ax.plot(start[0], start[1], "o", ms=9, color="white", mec="black", label="start")
    ax.plot(path[-1, 0], path[-1, 1], "*", ms=14, color="gold", mec="black", label="goal")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(im, ax=ax, label="elevation z [m]", shrink=0.8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"saved {out}")


# --- main -----------------------------------------------------------------------------------------


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    """Planner thresholds/networks and controller/physics flags, shared with
    ostrich_follow_path_parallel_worlds.py."""
    parser.add_argument("--tau-pos", type=float, default=TAU_POS)
    parser.add_argument("--tau-pitch", type=float, default=TAU_PITCH)
    parser.add_argument("--tau-rot", type=float, default=TAU_ROT)
    parser.add_argument("--tau-fused-pos", type=float, default=TAU_FUSED_POS)
    parser.add_argument("--tau-fused-rot", type=float, default=TAU_FUSED_ROT)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-rot", type=pathlib.Path, default=DEFAULT_CHECKPOINT_ROT)
    parser.add_argument("--torch-device", default="cpu")
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--v", type=float, default=0.6)
    parser.add_argument("--lookahead", type=float, default=0.4)
    parser.add_argument("--mu", type=float, default=0.8)
    parser.add_argument("--slack", type=float, default=2.0)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--override", nargs="*", default=[])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--map", type=pathlib.Path, required=True)
    parser.add_argument("--planner", choices=list(PLANNERS), required=True)
    parser.add_argument("--start", type=float, nargs=3, metavar=("X", "Y", "YAW"))
    parser.add_argument("--goal", type=float, nargs=2, metavar=("X", "Y"))
    add_shared_args(parser)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--out-dir", type=pathlib.Path, default=REPO_ROOT / "outputs" / "follow_path")
    args = parser.parse_args()

    map_path = args.map.with_suffix("")
    meta = yaml.safe_load(map_path.with_suffix(".yaml").read_text())
    start = tuple(args.start) if args.start is not None else meta.get("start")
    goal = tuple(args.goal) if args.goal is not None else meta.get("goal")
    if start is None or goal is None:
        raise SystemExit(f"{map_path.name}.yaml has no start/goal -- pass --start X Y YAW --goal X Y")
    start = tuple(float(v) for v in start)
    goal = (float(goal[0]), float(goal[1]))
    terrain = HeightMapReader.load(map_path)
    repeats = 1 if args.view else args.repeats
    run_name = f"{map_path.name}_{args.planner}"

    # --- 1. plan --------------------------------------------------------------------------------
    print(f"=== {map_path.name}: planning with {args.planner}, start {start} -> goal {goal} ===")
    ctx = PlanContext(args)
    plan = plan_path(terrain, start, goal, args.planner, ctx)
    del ctx
    gc.collect()  # CostToGo / gated solver buffers go before the ostrich build (3 GiB GPU)
    gate = f", gate {plan['gate']}" if plan["gate"] else ""
    if not plan["reached"]:
        why = "no feasible path" if not plan["reachable"] else "V* finite but the policy did not arrive"
        raise SystemExit(f"{args.planner}{gate}: {why} (V* = {plan['v_start']:.2f}) -- nothing to simulate")
    max_err = "  ".join(f"max predicted {k} {v:.4f}" for k, v in plan["max_err"].items())
    print(f"plan{gate}: {plan['path_m']:.2f} m, V* = {plan['v_start']:.2f}, {len(plan['xy'])} lattice "
          f"poses, {plan['n_settle_bad']} settle-infeasible  {max_err}")
    path = resample_polyline(np.vstack([plan["xy"], goal]), WAYPOINT_SPACING)
    path_len = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())

    # --- 2. simulate ----------------------------------------------------------------------------
    sim_config, render_config, engine_config, logging_config = compose_ostrich_config(tuple(args.override))
    render_config.vis_type = "gl" if args.view else "null"
    sim_config.num_worlds = repeats
    dt = float(sim_config.target_timestep_seconds)
    drive_steps = int(math.ceil(args.slack * path_len / args.v / dt))
    T = args.settle_steps + drive_steps
    kappa_max = 1.0 / float(dynamics.robot_params().min_turn_radius)
    print(f"ostrich: {repeats} world(s), v={args.v} m/s, lookahead {args.lookahead} m, mu={args.mu}, "
          f"dt={dt}, {args.settle_steps} settle + {drive_steps} drive steps ({drive_steps * dt:.0f} s)")
    t0 = time.time()
    sim = None
    try:
        sim = PathFollowSimulator(
            sim_config, render_config, engine_config, logging_config,
            k_p=K_P, mu_front=args.mu, mu_rear=args.mu, terrain=terrain,
            spawn_pose=np.tile(np.array(start, np.float64), (repeats, 1)),
            path_xy=path, settle_steps=args.settle_steps, lookahead=args.lookahead, v=args.v,
            kappa_max=kappa_max,
        )
        pose, wheel_qd = sim.rollout(T, args.view)
    finally:
        sim = None
        gc.collect()
    print(f"ostrich rollout done in {time.time() - t0:.1f} s")
    pose, wheel_qd = pose[args.settle_steps :], wheel_qd[args.settle_steps :]
    if len(pose) == 0:
        raise SystemExit("viewer closed during the settle -- nothing to judge")

    # --- 3. verdict -----------------------------------------------------------------------------
    results = [judge(pose[:, w], dt, path, terrain) for w in range(repeats)]
    print(f"\n{'id':>3} {'status':>9} {'t_arr':>6} {'climb':>6} {'roll':>5} {'cte_max':>7} "
          f"{'cte_mean':>8} {'cte@climb':>9} {'progress':>12}")
    for w, r in enumerate(results):
        print(f"{w:3d} {r['status']:>9} {r['t_arrive']:6.1f} {r['climb_deg']:6.1f} {r['roll_deg']:5.1f} "
              f"{r['cte_max']:7.3f} {r['cte_mean']:8.3f} {r['cte_at_climb']:9.3f} "
              f"{r['progress_m']:5.2f}/{path_len:<5.2f}")
    n_ok = sum(r["status"] == "arrived" for r in results)
    print(f"\n{n_ok}/{repeats} arrived  ({args.planner}{gate}, {map_path.name})")

    title = (f"{map_path.name}: {args.planner}{gate}\nplanned {plan['path_m']:.1f} m, "
             f"{n_ok}/{repeats} arrived in ostrich (v {args.v} m/s, lookahead {args.lookahead} m)")
    plot_run(terrain, plan, path, results, start, title, args.out_dir / f"{run_name}.png")
    write_comparison(
        args.out_dir / f"{run_name}.h5",
        root=dict(
            name=run_name, map=str(map_path), planner=args.planner, head=str(plan["head"]),
            tau=plan["tau"], gate=plan["gate"], n=repeats, v=args.v, lookahead=args.lookahead, mu=args.mu,
            slack=args.slack, settle_steps=args.settle_steps, path_m=path_len,
            obstacle_x=0.5 * (start[0] + goal[0]),
        ),
        per_variant=dict(
            spawn_pose=np.tile(np.array(start, np.float32), (repeats, 1)),
            v_drive=np.full(repeats, args.v, np.float32),
            wz_drive=np.zeros(repeats, np.float32),
            variant_value=np.arange(repeats, dtype=np.float32),
            variant_label=np.array([f"{run_name}_r{w}" for w in range(repeats)]),
            status=np.array([r["status"] for r in results]),
            planned_path=path.astype(np.float32),
            **{key: np.array([r[key] for r in results], np.float32)
               for key in ("t_arrive", "climb_deg", "roll_deg", "cte_max", "cte_mean",
                           "cte_at_climb", "progress_m")},
        ),
        terrain_entries=[(map_path, terrain)] * repeats,
        ostrich=dict(dt=dt, t=np.arange(len(pose), dtype=np.float32) * dt, pose=pose,
                     wheel_qd=wheel_qd),
        hstack=dict(dt=dt),
    )


if __name__ == "__main__":
    main()
