"""Closed-loop MPPI in ostrich, with and without the net's window cost -- mppi_learning/design.md
section 9d.6 and 9e.

MPPI plans with helhest_stack's twin, as on the robot; OSTRICH executes. Every `dynamics.DT` (0.1 s,
one replan per perception frame on the robot) each world:

  1. reads its REALIZED state from ostrich's logs: (x, y, yaw), wheel speeds, and the body twist over
     the last 0.1 s (`generate_dataset.realized_start`, the same estimate the dataset seeded the twin
     with);
  2. seeds its planner's rollouts with them (`set_initial_wheel_omega` / `set_initial_twist`, as the
     ROS node does from /joint_states and odometry), calls `WindowCost.update` on the `nn` arms, and
     replans (`--n-refine`, the node's 3 by default);
  3. blends the new plan with the previous one, shifted a step (the node's `plan_consistency` 0.3),
     and sends its first step to ostrich -- yaw-compensated by the checkpoint's `ostrich_yaw_gain`,
     exactly as the dataset commanded ostrich, since that gain is part of what the net's labels mean.

Two arms, each one replicated world of ONE ostrich build (replicated worlds do not collide with each
other) with its own `MppiGpu`, from the same start: `vanilla` (no hook) and ONE nn arm. The nn arm
charges `WindowCost` at `--nn-weights`, with `--baseline` "flat" (only the error predicted above the
same command on level ground, arm `nnflat[...]`) or "none" (the raw prediction, arm `nn[...]`), over
`--windows` windows (1 = window 0 only; 3 also charges windows 1 and 2 through a graph-captured
`WarpTrunk`, suffix `w3`), in the first `--nn-refines` refines of each replan (suffix `rK`; the rest
rank the rollouts by MPPI's own cost, `WindowCost.replan_split`, the hook behind a conditional graph
node, so no recapture). Ostrich is nondeterministic, so two runs of one arm differ too.

`--view` shows the run LIVE in Newton's ViewerGL as it is computed: each robot in its arm's colour
(the PNG's), its driven trail, its current MPPI plan (the line above the trail: the twin's rollout of
the nominal U as it entered the last refine, i.e. before the final reweight and the plan blend), and
the goal's arrival circle in red. It starts paused: SPACE runs, "." steps one frame, ESC quits (the
run ends there and is still saved). The run is paced to wall-clock real time when the planners are
faster than that; otherwise it runs as fast as it can, and the window title shows the real-time
factor. On the GTX 1050 even the window-0 pair cannot keep up: ostrich's own 4 steps take ~110 ms
per 0.1 s frame (0.8x real time headless), the viewer's ~40 FPS cap brings it to ~0.55x, and a
`--windows 3` arm adds ~100 ms of replan per frame (~70 ms on the 1650), ~0.45x.

Timing: every replan is timed on the wall clock (`WindowCost.update` + `replan` + reading the plan
back, which waits for the GPU), and every planner profiles its refine stages with CUDA events (the
hook runs inside "cost"). Reading those events syncs after every refine, which serializes the
host's launches with the GPU and adds a few percent to a vanilla replan's ~3 ms, and nothing
measurable to the ~100 ms of a 3-window one. After the run, a table per arm gives the median and 90th percentile
replan time and the mean ms per stage. Worlds replan one after another, so each replan has the GPU
to itself, as on the robot, but ostrich steps between frames and warms the GPU.

The planner is the node's (`elevation_node.py` defaults: 4096 rollouts, 3 refines, n_theta 24,
elite 0.01, straight prior 0.2, its cost weights, a cost-to-go field with a 0.3 m robust margin
coarsened to ~0.32 m cells), except for what the net requires: horizon 31 / 4 knots (the node runs
25), `wmin` 0, the twin's `k_turn` from the checkpoint's labels, and the sphere wheel envelope the
dataset's twin used (the node runs the 0.10 m cylinder). Not modelled: the node's command conditioner
(slew limit, goal brake, turn boost, yaw loop) and its terminal dock -- the first plan step goes to
ostrich unfiltered, and a world stops when it is within `--reach-radius` of the goal. The planner
sees the whole map, not the node's robot-centred 12 m window.

After the run, every executed 1 s window (10 consecutive frames driven by MPPI) is re-run through
the twin from ostrich's realized state at its start, with the commands MPPI actually sent: the
twin-vs-ostrich end-pose error ALONG THE DRIVEN PATH, which is the dataset's label and what the `nn`
cost tries to keep small. The net's prediction for the same windows is printed beside it.

Outputs `outputs/nn_mppi/<map>_<tag>.{h5,png}`: ostrich pose/wheel logs and commands per world, the
per-window errors and predictions, the replan and stage times, and a bird's-eye plot of every
world's path.

CLI parameters:
    --checkpoint PATH       mppi_learning train.py checkpoint (required)
    --map PATH|flat         heightmap (.png or stem) whose .yaml sidecar has `start`/`goal`, or
                            `flat` (a 12 m flat map, start (-4, 0, 0), goal (4, 0))
    --start X Y YAW         override the sidecar's start
    --goal X Y              override the sidecar's goal
    --nn-weights W          the nn arm's comma-separated weights in the checkpoint's target order
                            (e.g. 100,100 for e_pos [1/m], e_rot [1/rad]); default 100,100
    --baseline none|flat    the nn arm's WindowCost baseline (default flat)
    --no-vanilla            skip the vanilla arm (the nn arm alone)
    --no-nn                 skip the nn arm (vanilla alone; the checkpoint still sets k_turn and the yaw gain)
    --view                  live GL viewer (default: headless)
    --max-time S            per-run time limit (default 40)
    --spin-frac F           MPPI SPIN prior fraction (default: the sidecar's `spin_frac`, else 0, the node's)
    --reach-radius M        arrival radius (default 0.3, the node's plan_reach_radius)
    --batch INT             MPPI rollouts (default 4096)
    --n-refine INT          MPPI refines per replan (default 3)
    --windows N             windows the nn arm charges, 1 (window 0) to 3 (default 1)
    --nn-refines K          charge the nn cost in the first K refines of each replan only, 1 to
                            --n-refine (default: every refine)
    --routing-cell M        cost-to-go cell size, the map max-pooled to it (default 0.32)
    --pivot-cost C          cost-to-go point-turn primitives (default: the sidecar's `pivot_cost`, else 0,
                            the node's; a map whose only way out is a turn in place, like the garage,
                            needs > 0 and --spin-frac)
    --tag STR               output file suffix (default: the arms)

Usage:
    python src/feasibility/nn_mppi/closed_loop.py --checkpoint outputs/checkpoints/dataset_mppi_default_maps0_centered_M200_R10_seed0.pt \
    --map assets/curb_detour/curb_detour_h015_g04 --batch 512 --n-refine 1 --windows 3 --view
    python src/feasibility/nn_mppi/closed_loop.py --checkpoint outputs/checkpoints/<ckpt>.pt --map flat
    python src/feasibility/nn_mppi/closed_loop.py --checkpoint outputs/checkpoints/<ckpt>.pt \\
        --map assets/curb_detour/curb_detour_h015_g04 --batch 512 --n-refine 1 --windows 3 --view
    python src/feasibility/nn_mppi/closed_loop.py --checkpoint outputs/checkpoints/<ckpt>.pt \\
        --map assets/curb_detour/curb_detour_h015_g04 --batch 512 --n-refine 3 --windows 3 --nn-refines 1 \\
        --nn-weights 300,300
    python src/feasibility/nn_mppi/closed_loop.py --checkpoint outputs/checkpoints/<ckpt>.pt \\
        --map assets/garage/garage_b --pivot-cost 0.15 --spin-frac 0.1 --baseline none --nn-weights 10,10
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
import time
from collections.abc import Callable

import h5py
import newton
import numpy as np
import torch
import warp as wp
import yaml
from helhest import dynamics
from helhest import friction as friction_mod
from helhest.control.mppi import CostParams
from helhest.control.mppi import MppiGpu
from helhest.control.mppi import SamplingConfig
from helhest.engine import ForwardSimulator
from helhest.engine import GridParams
from helhest.planning.costtogo import CostToGo

from feasibility.comparator.common import friction_kwargs
from feasibility.comparator.common import HelhestBatchSimulator
from feasibility.comparator.common import init_warp_device
from feasibility.comparator.common import K_P
from feasibility.comparator.common import OUT_DIR
from feasibility.heightmap import HeightMapReader
from feasibility.lattice_learning.custom_dataset import pose_to_se3
from feasibility.lattice_learning.custom_dataset import se3_errors
from feasibility.lattice_learning.generate_dataset import compose_ostrich_config
from feasibility.lattice_learning.generate_dataset import OSTRICH_DT
from feasibility.lattice_learning.generate_dataset import SPAWN_CLEARANCE
from feasibility.lattice_learning.patch import sample_patches
from feasibility.lattice_learning.settle import settle_batch
from feasibility.mppi_learning.command import compensate
from feasibility.mppi_learning.command import encode
from feasibility.mppi_learning.command import MPPI_DT
from feasibility.mppi_learning.command import WINDOW_STEPS
from feasibility.mppi_learning.generate_dataset import _exact_steps
from feasibility.mppi_learning.generate_dataset import quat_to_yaw
from feasibility.mppi_learning.generate_dataset import realized_start
from feasibility.mppi_learning.spawn_sampling import PATCH_SPEC
from feasibility.mppi_learning.train import load_checkpoint
from feasibility.mppi_learning.twin import run_twin
from feasibility.nn_mppi.mppi_cost import HORIZON
from feasibility.nn_mppi.mppi_cost import N_KNOTS
from feasibility.nn_mppi.mppi_cost import WindowCost

OUT = OUT_DIR / "nn_mppi"
SETTLE_STEPS = 15  # ostrich steps dropping onto the terrain at zero command (mppi configs' settle_steps)
STEPS_PER_FRAME = _exact_steps(MPPI_DT, OSTRICH_DT)  # 4 ostrich steps per replan
N_REFINE = 3  # the node's plan_n_refine, --n-refine's default
STAGES = ("sample", "rollout", "cost", "reweight")  # MppiGpu's profiled refine stages; the hook is in "cost"
PLAN_CONSISTENCY = 0.3  # the node's EMA of the new plan toward the previous one, shifted a step
EDGE_MARGIN = 0.5  # [m] a world this close to the map edge is stopped as off_map
FLIP_DEG = 60.0  # |pitch| or |roll| past this: flipped
GOAL_COLOR = (1.0, 0.1, 0.1)  # --view: the arrival circle
PLAN_DZ = 0.10  # [m] --view: a plan line this far above its trail
# --view: the camera behind the start, looking along start -> goal from above
CAMERA_BACK = 4.0  # [m]
CAMERA_UP = 6.0  # [m] above the highest terrain
CAMERA_PITCH = -40.0


def node_planner(
    terrain: HeightMapReader, k_turn: float, mu: float, batch: int, spin_frac: float, seed: int, device: str
) -> MppiGpu:
    """The ROS node's MPPI (`elevation_node.py` defaults) at the net's horizon, wmin 0 and twin."""
    elevation, grid = terrain.to_hstack(device)
    sim = ForwardSimulator(dynamics.robot_params(), dynamics.planning_solver(k_turn=k_turn), grid, batch, HORIZON, device)
    sim.set_terrain(elevation)
    sim.set_uniform_friction(mu)
    cost = CostParams(goal_running=0.3, effort=1e-3, turn=0.03, smoothness=0.04, saturation=300.0)
    sampling = SamplingConfig(wmax=4.0, wmin=0.0, n_knots=N_KNOTS, straight_frac=0.2, spin_frac=spin_frac, elite_frac=0.01)
    # profile: CUDA events around each refine stage, read after every refine (the replan syncs anyway)
    planner = MppiGpu(sim, cost, sampling, n_theta=24, seed=seed, profile=True)
    planner.reset_nominal(1.5)  # plan_nominal_reset
    return planner


def routing_field(
    terrain: HeightMapReader, goal: tuple[float, float], k_turn: float, cell: float, pivot_cost: float, device: str
) -> tuple[wp.array, object, float]:
    """The cost-to-go V the goal cost samples, solved once (static map, fixed goal) on the map
    max-pooled to ~`cell` (the pool keeps thin walls). Returns (V, its Grid, the unreachable cap)."""
    k = max(1, round(cell / terrain.cell))
    ny, nx = terrain.ny // k, terrain.nx // k
    coarse = terrain.H[: ny * k, : nx * k].reshape(ny, k, nx, k).max(axis=(1, 3))
    grid = GridParams(nx, ny, terrain.cell * k, terrain.x0, terrain.y0)
    ctg = CostToGo(grid, dynamics.robot_params(), dynamics.planning_solver(k_turn=k_turn), n_theta=24,
                   robust_margin_m=0.3, pivot_cost=pivot_cost, device=device)
    V = ctg.compute(wp.array(np.ascontiguousarray(coarse, np.float32), dtype=wp.float32, device=device), goal)
    return V, grid.build(), float(ctg._vcap)


class OstrichStepper(HelhestBatchSimulator):
    """`HelhestBatchSimulator` driven one MPPI frame at a time: the host writes each world's wheel
    command for the next STEPS_PER_FRAME rows of the setpoint buffer, then the captured physics step
    runs that many times. The pose and wheel logs cover the whole run, settle included. World w's
    shapes are tinted `world_colors[w]`, which only the live viewer shows."""

    def __init__(self, *args, world_colors: list[tuple[float, float, float]], **kwargs) -> None:
        self._world_colors = world_colors  # before super().__init__, which calls build_model
        super().__init__(*args, **kwargs)

    def build_model(self) -> newton.Model:
        """Tint each world's shapes here: the viewer reads shape_color in set_model, right after."""
        model = super().build_model()
        world, color = model.shape_world.numpy(), model.shape_color.numpy()
        for w, rgb in enumerate(self._world_colors):
            color[world == w] = rgb  # the terrain (world -1) keeps its colour
        model.shape_color.assign(color)
        return model

    def begin(self, n_frames: int) -> None:
        worlds, dev = self.simulation_config.num_worlds, self.model.device
        self._T = SETTLE_STEPS + n_frames * STEPS_PER_FRAME
        self._setpoints_wp = wp.zeros((self._T, worlds, 3), dtype=wp.float32, device=dev)
        self._step_buf = wp.zeros(1, dtype=wp.int32, device=dev)
        self._pose_log = wp.zeros((self._T, worlds, 7), dtype=wp.float32, device=dev)
        self._wheel_log = wp.zeros((self._T, worlds, 3), dtype=wp.float32, device=dev)
        self._jq = wp.zeros_like(self.model.joint_q)
        self._jqd = wp.zeros_like(self.model.joint_qd)
        with wp.ScopedCapture(device=dev) as capture:
            self._batch_physics_step()
        self._graph = capture.graph
        self.step = 0
        self._launch(SETTLE_STEPS)  # zero setpoints: drop onto the terrain

    def _launch(self, n: int) -> None:
        for _ in range(n):
            wp.capture_launch(self._graph)
        self.step += n

    def drive(self, command: np.ndarray, after_step: Callable[[bool], None] | None = None) -> None:
        """[W, 3] ostrich wheel speeds, held for one MPPI frame; `after_step(last)` runs after every
        ostrich step, `last` on the frame's final one (the live view draws there)."""
        rows = self._setpoints_wp[self.step : self.step + STEPS_PER_FRAME]
        rows.assign(np.ascontiguousarray(np.broadcast_to(command, rows.shape), np.float32))
        for k in range(STEPS_PER_FRAME):
            self._launch(1)
            if after_step is not None:
                after_step(k == STEPS_PER_FRAME - 1)

    def state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Now, per world: (x, y, yaw) [W, 3], wheel speeds [W, 3], body twist [W, 3], pose [W, 7]."""
        lo = self.step - 1 - STEPS_PER_FRAME
        pose = self._pose_log[lo : self.step].numpy()
        wheels = self._wheel_log[lo : self.step].numpy()
        xy_yaw, wheel, twist = realized_start(pose, wheels, pose.shape[0])
        return xy_yaw, wheel, twist, pose[-1]

    def logs(self) -> tuple[np.ndarray, np.ndarray]:
        return self._pose_log[: self.step].numpy(), self._wheel_log[: self.step].numpy()


@dataclasses.dataclass
class Arm:
    name: str
    weights: dict[str, float] | None  # None = vanilla
    baseline: str = "none"  # WindowCost's baseline
    windows: int = 1  # WindowCost's n_windows
    nn_refines: int = 0  # charge the cost in the first nn_refines refines only (nn arms)


class LiveView:
    """`--view`: the ostrich build's own ViewerGL, drawn after the ostrich steps and paced to
    wall-clock real time per MPPI frame (never catching up: a slow frame just runs late, and a pause
    costs nothing). Draws each world's robot
    (tinted by `OstrichStepper`), driven trail, current MPPI plan and the goal's arrival circle."""

    def __init__(self, ostrich: OstrichStepper, terrain: HeightMapReader, start: np.ndarray, goal: tuple[float, float],
                 reach_radius: float, colors: list[tuple[float, float, float]]) -> None:
        from feasibility.mppi_learning.gl_dataset_browser import ground_polyline  # pyglet: only with a display

        self._ground = ground_polyline
        self.ostrich, self.terrain, self.colors = ostrich, terrain, colors
        self.viewer = ostrich.viewer
        # Newton spreads worlds apart for display, which would put the robots beside the one shared
        # terrain (demos/ostrich_follow_path_dual.py explains); here they belong on top of each other
        self.viewer.set_world_offsets((0.0, 0.0, 0.0))
        heading = np.arctan2(goal[1] - start[1], goal[0] - start[0])
        self.viewer.set_camera(
            pos=wp.vec3(start[0] - CAMERA_BACK * np.cos(heading), start[1] - CAMERA_BACK * np.sin(heading),
                        terrain.max_z + CAMERA_UP),
            pitch=CAMERA_PITCH, yaw=float(np.degrees(heading)),
        )
        self.viewer._paused = True  # start paused: SPACE runs, "." steps (no public setter)
        angle = np.linspace(0.0, 2.0 * np.pi, 33)
        self._goal = self._segments(ground_polyline(terrain, goal[0] + reach_radius * np.cos(angle),
                                                    goal[1] + reach_radius * np.sin(angle)))
        self.trails: list[list[np.ndarray]] = [[] for _ in colors]
        self.plans: list[np.ndarray | None] = [None] * len(colors)
        self.start_frame()

    def _segments(self, points: np.ndarray) -> tuple[wp.array, wp.array]:
        device = self.ostrich.model.device
        return wp.array(points[:-1], dtype=wp.vec3, device=device), wp.array(points[1:], dtype=wp.vec3, device=device)

    def record(self, w: int, xy: np.ndarray, plan: np.ndarray | None) -> None:
        """World w is at `xy` [2]; `plan` [H, 2] is its MPPI plan from there (None: it stopped)."""
        self.trails[w].append(np.asarray(xy[:2], np.float64))
        self.plans[w] = plan

    def render(self, title: str | None = None) -> None:
        if title is not None:
            self.viewer.renderer.set_title(title)
        self.viewer.begin_frame(self.ostrich.step * OSTRICH_DT)
        self.viewer.log_state(self.ostrich.current_state)
        self.viewer.log_lines("/goal", *self._goal, colors=GOAL_COLOR)
        for w, color in enumerate(self.colors):
            trail = np.array(self.trails[w])
            if len(trail) >= 2:
                self.viewer.log_lines(f"/trail_{w}", *self._segments(self._ground(self.terrain, trail[:, 0], trail[:, 1])),
                                      colors=color)
            plan = self.plans[w]
            if plan is not None and len(plan) >= 2:
                points = self._ground(self.terrain, plan[:, 0], plan[:, 1])
                points[:, 2] += PLAN_DZ
                self.viewer.log_lines(f"/plan_{w}", *self._segments(points), colors=tuple(0.5 + 0.5 * c for c in color))
            elif plan is None:
                self.viewer.log_lines(f"/plan_{w}", None, None, None)  # a stopped world's plan goes
        self.viewer.end_frame()

    def start_frame(self) -> None:
        """An MPPI frame starts now: its MPPI_DT of wall clock, planning included, runs from here."""
        self._frame_t0, self._steps = time.perf_counter(), 0

    def pace(self, last: bool) -> None:
        """After an ostrich step: draw it unless the frame is already late (the viewer caps at ~40
        FPS, so drawing every step of a late frame would double its time), always the frame's last
        step; then wait out the step's OSTRICH_DT of wall clock if it is early."""
        self._steps += 1
        deadline = self._frame_t0 + self._steps * OSTRICH_DT
        if last or time.perf_counter() < deadline:
            self.render()
        remaining = deadline - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)


def pitch_roll(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[..., 4] (qx, qy, qz, qw) -> (pitch, roll) [...], ZYX."""
    qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    pitch = np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
    roll = np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    return pitch, roll


def load_map(spec: str, start: list[float] | None, goal: list[float] | None
             ) -> tuple[HeightMapReader, str, np.ndarray, tuple[float, float], dict]:
    """(terrain, name, start, goal, sidecar) -- the sidecar may also carry `spin_frac`/`pivot_cost`."""
    if spec == "flat":
        terrain, name, meta = HeightMapReader.flat(xlim=(-6.0, 6.0), ylim=(-6.0, 6.0), cell=0.05), "flat", {"start": [-4.0, 0.0, 0.0], "goal": [4.0, 0.0]}
    else:
        path = pathlib.Path(spec).with_suffix("")
        terrain, name = HeightMapReader.load(path.with_suffix(".png")), path.name
        meta = yaml.safe_load(path.with_suffix(".yaml").read_text())
    start = start if start is not None else meta.get("start")
    goal = goal if goal is not None else meta.get("goal")
    if start is None or goal is None:
        raise SystemExit(f"{spec}: no start/goal in the sidecar -- pass --start X Y YAW --goal X Y")
    return terrain, name, np.asarray(start[:3], np.float64), (float(goal[0]), float(goal[1])), meta


def run(args: argparse.Namespace) -> None:
    device = "cuda:0"
    init_warp_device(device)
    terrain, map_name, start, goal, meta = load_map(args.map, args.start, args.goal)
    for key in ("spin_frac", "pivot_cost"):  # the CLI, else the map's sidecar, else the node's 0
        if getattr(args, key) is None:
            setattr(args, key, float(meta.get(key, 0.0)))
    if args.no_vanilla and args.no_nn:
        raise SystemExit("--no-vanilla and --no-nn leave nothing to run")
    net, ckpt = load_checkpoint(args.checkpoint, torch.device(device))
    net.eval()
    attrs = ckpt["label_attrs"]
    k_turn, mu, yaw_gain = float(attrs["k_turn"]), float(attrs["mu"]), float(attrs["ostrich_yaw_gain"])

    nn_refines = args.nn_refines or args.n_refine
    if not 1 <= nn_refines <= args.n_refine:
        raise SystemExit(f"--nn-refines {nn_refines}: must be in [1, --n-refine {args.n_refine}]")
    if not 1 <= args.windows <= N_KNOTS - 1:
        raise SystemExit(f"--windows {args.windows}: must be in [1, {N_KNOTS - 1}]")
    values = [float(v) for v in args.nn_weights.split(",")]
    if len(values) != len(net.target_names):
        raise SystemExit(f"--nn-weights {args.nn_weights}: need {len(net.target_names)} values for {net.target_names}")
    # window 0 alone, charged in every refine, keeps the plain arm name
    suffix = (f"w{args.windows}" if args.windows > 1 else "") + (f"r{nn_refines}" if nn_refines < args.n_refine else "")
    prefix = "nnflat" if args.baseline == "flat" else "nn"
    nn_arm = Arm(f"{prefix}[{args.nn_weights}]{suffix}", dict(zip(net.target_names, values)), args.baseline,
                 args.windows, nn_refines)
    arms = [nn_arm] if args.no_vanilla else [Arm("vanilla", None)] if args.no_nn else [Arm("vanilla", None), nn_arm]
    n_worlds, n_frames = len(arms), int(round(args.max_time / MPPI_DT))
    print(f"[map]      {map_name}: {terrain.nx}x{terrain.ny} @ {terrain.cell} m, start {start.tolist()}, goal {goal}")
    print(f"[labels]   k_turn {k_turn}, mu {mu}, ostrich yaw gain {yaw_gain} ({args.checkpoint.name})")
    print(f"[planner]  {args.batch} rollouts, {args.n_refine} refine(s) per replan, spin_frac {args.spin_frac}, "
          f"pivot_cost {args.pivot_cost}")
    print("[arms]     " + ", ".join(a.name for a in arms))

    t0 = time.perf_counter()
    V, lattice_grid, vcap = routing_field(terrain, goal, k_turn, args.routing_cell, args.pivot_cost, device)
    planners, costs = [], []
    for i, arm in enumerate(arms):
        planner = node_planner(terrain, k_turn, mu, args.batch, args.spin_frac, seed=i, device=device)
        planner.set_lattice(V, lattice_grid)
        planner.cw.lattice_cap = vcap
        cost = None
        if arm.weights is not None:
            cost = WindowCost(net, attrs, planner, arm.weights, blur_terrain=bool(ckpt["blur_terrain"]), baseline=arm.baseline,
                              n_windows=arm.windows, switchable=arm.nn_refines < args.n_refine)
            planner.set_cost_hook(cost)
        planners.append(planner)
        costs.append(cost)
    print(f"[setup]    cost-to-go + {n_worlds} planners in {time.perf_counter() - t0:.1f} s")

    spawn_zpr, _, _ = settle_batch(terrain, start[None], mu, device)
    spawn_zpr = spawn_zpr.astype(np.float64)
    spawn_zpr[:, 0] += SPAWN_CLEARANCE
    sim_config, render_config, engine_config, logging_config = compose_ostrich_config(())
    render_config.vis_type = "gl" if args.view else "null"
    sim_config.target_timestep_seconds = OSTRICH_DT
    sim_config.num_worlds = n_worlds
    colors = arm_colors([a.name for a in arms])
    ostrich = OstrichStepper(
        sim_config, render_config, engine_config, logging_config, k_p=K_P, **friction_kwargs(mu),
        terrain=terrain, spawn_pose=np.repeat(start[None], n_worlds, 0), spawn_zpr=np.repeat(spawn_zpr, n_worlds, 0),
        world_colors=colors,
    )
    view = LiveView(ostrich, terrain, start, goal, args.reach_radius, colors) if args.view else None
    ostrich.begin(n_frames)
    if view is not None:
        print("[view]     paused: SPACE runs, '.' steps one frame, ESC quits (the run is still saved)")

    x_lo, y_lo = terrain.x0 + EDGE_MARGIN, terrain.y0 + EDGE_MARGIN
    x_hi, y_hi = terrain.x0 + terrain.nx * terrain.cell - EDGE_MARGIN, terrain.y0 + terrain.ny * terrain.cell - EDGE_MARGIN
    status = np.array(["timeout"] * n_worlds, dtype=object)
    done = np.zeros(n_worlds, bool)
    end_frame = np.full(n_worlds, n_frames)
    commands = np.zeros((n_frames, n_worlds, 3), np.float32)  # MPPI convention, what the twin would get
    driving = np.zeros((n_frames, n_worlds), bool)  # the frame's command came from MPPI
    previous_plan = [None] * n_worlds
    plan_ms = np.full((n_frames, n_worlds), np.nan, np.float32)  # update + replan + the plan's readback
    t_run, rtf = time.perf_counter(), 1.0
    f = 0
    while f < n_frames:
        if view is not None:
            if not view.viewer.is_running():  # closed: end the run here, keep what was driven
                status[~done], end_frame[~done], done[:] = "closed", f, True
                n_frames = f
                break
            if not view.viewer.should_step():  # paused (SPACE runs, "." steps one frame)
                view.render()
                continue
            view.start_frame()
        t_frame = time.perf_counter()
        xy_yaw, wheels, twist, pose = ostrich.state()
        pitch, roll = pitch_roll(pose[:, 3:7])
        for w in range(n_worlds):
            if done[w]:
                continue
            reason = None
            if not np.isfinite(pose[w]).all():
                reason = "nonfinite"
            elif max(abs(pitch[w]), abs(roll[w])) > np.radians(FLIP_DEG):
                reason = "flipped"
            elif not (x_lo < xy_yaw[w, 0] < x_hi and y_lo < xy_yaw[w, 1] < y_hi):
                reason = "off_map"
            elif np.hypot(xy_yaw[w, 0] - goal[0], xy_yaw[w, 1] - goal[1]) < args.reach_radius:
                reason = "arrived"
            if reason is not None:
                status[w], done[w], end_frame[w] = reason, True, f
                if view is not None:
                    view.record(w, xy_yaw[w], None)
        for w in np.flatnonzero(~done):
            t_world = time.perf_counter()
            planner, cost = planners[w], costs[w]
            planner.sim.set_initial_wheel_omega(wheels[w])
            planner.sim.set_initial_twist(twist[w])
            if cost is None:
                planner.replan(xy_yaw[w], goal, args.n_refine)
            else:
                cost.update(xy_yaw[w])
                cost.replan_split(xy_yaw[w], goal, args.n_refine, arms[w].nn_refines)
            plan = planner.nominal()  # a device-to-host copy: waits for the replan's GPU work
            plan_ms[f, w] = (time.perf_counter() - t_world) * 1e3
            if view is not None:
                # candidate 0 is the nominal U: its twin rollout, [H + 1] poses, from the last refine
                view.record(w, xy_yaw[w], planner.sim.controlled.numpy()[:, 0, :2])
            if previous_plan[w] is not None:
                shifted = np.roll(previous_plan[w], -1, axis=0)
                shifted[-1] = previous_plan[w][-1]
                plan = (1.0 - PLAN_CONSISTENCY) * plan + PLAN_CONSISTENCY * shifted
                planner.set_nominal(plan)
            previous_plan[w] = plan.copy()
            commands[f, w] = (plan[0, 0], plan[0, 1], 0.5 * (plan[0, 0] + plan[0, 1]))
            driving[f, w] = True
        if done.all():
            n_frames = f
            break
        if view is not None:
            title = f"{map_name}  t {f * MPPI_DT:5.1f} s  {rtf:4.2f}x real time  |  " + "  ".join(
                f"{a.name}: {status[w] if done[w] else 'driving'}" for w, a in enumerate(arms))
            view.viewer.renderer.set_title(title)
        ostrich.drive(compensate(commands[f], yaw_gain), view.pace if view is not None else None)
        rtf = 0.8 * rtf + 0.2 * MPPI_DT / (time.perf_counter() - t_frame)  # smoothed, for the title
        if f % 50 == 0:
            print(f"  t={f * MPPI_DT:5.1f} s: " + ", ".join(f"{status[w] if done[w] else 'driving'}" for w in range(n_worlds)))
        f += 1
    n_run = min(n_frames, commands.shape[0])
    print(f"[run]      {n_run} frames, planning {np.nansum(plan_ms[:n_run]) / max(n_run, 1):.0f} ms per frame for {n_worlds} worlds")
    print(f"[run]      {n_run * MPPI_DT:.1f} s simulated in {time.perf_counter() - t_run:.1f} s of wall clock"
          + (" (pauses included)" if view is not None else ""))

    pose_log, wheel_log = ostrich.logs()
    windows = window_errors(terrain, pose_log, wheel_log, commands[:n_run], driving[:n_run], net, mu, k_turn, device)
    report(arms, status, end_frame, pose_log, windows)
    # per world, the mean ms of each refine stage over its replans, the first (cold) refine excluded
    stage_ms = np.array([[p.timing_stats()[s]["mean_ms"] for s in STAGES] for p in planners], np.float32)
    report_timing(arms, plan_ms[:n_run], stage_ms)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.tag or "_".join(a.name.replace("[", "").replace("]", "").replace(",", "-") for a in arms)
    stem = OUT / f"{map_name}_{tag}"
    save(stem.with_suffix(".h5"), args, map_name, terrain, start, goal, arms, status, end_frame, pose_log, wheel_log,
         commands[:n_run], driving[:n_run], windows, attrs, plan_ms[:n_run], stage_ms)
    plot(stem.with_suffix(".png"), terrain, start, goal, arms, status, pose_log, colors)
    if view is not None:
        print("[view]     run over: the viewer holds the last frame until closed")
        while view.viewer.is_running():
            view.render()


def frame_rows(frame: int) -> int:
    """The log row holding the state at the START of `frame` (the state `ostrich.state()` read)."""
    return SETTLE_STEPS + frame * STEPS_PER_FRAME


@torch.no_grad()
def window_errors(
    terrain: HeightMapReader, pose_log: np.ndarray, wheel_log: np.ndarray, commands: np.ndarray, driving: np.ndarray,
    net: torch.nn.Module, mu: float, k_turn: float, device: str,
) -> dict[str, np.ndarray]:
    """Every executed window (WINDOW_STEPS consecutive MPPI-driven frames) of every world: the twin
    from ostrich's realized state at its start with the commands MPPI sent, against ostrich's pose
    at its end; and the net's prediction for the same window."""
    n_frames, n_worlds = driving.shape
    starts = [(f, w) for w in range(n_worlds) for f in range(n_frames - WINDOW_STEPS + 1)
              if driving[f : f + WINDOW_STEPS, w].all() and frame_rows(f + WINDOW_STEPS) <= pose_log.shape[0]]
    if not starts:
        empty = np.zeros((0, 2))
        return {"frame": np.zeros(0, int), "world": np.zeros(0, int), "true": empty, "pred": empty, "pred_level": empty}
    frame, world = np.array(starts).T
    s0 = np.array([frame_rows(f) for f in frame])
    k = STEPS_PER_FRAME
    xy_yaw = np.zeros((len(s0), 3), np.float32)
    wheels, twist = np.zeros_like(xy_yaw), np.zeros_like(xy_yaw)
    for i, (s, w) in enumerate(zip(s0, world)):
        a, b, c = realized_start(pose_log[s - 1 - k : s, w : w + 1], wheel_log[s - 1 - k : s, w : w + 1], k + 1)
        xy_yaw[i], wheels[i], twist[i] = a[0], b[0], c[0]
    omega = np.stack([commands[f : f + WINDOW_STEPS, w] for f, w in zip(frame, world)], axis=1)  # [10, n, 3]
    twin_end = run_twin(terrain, xy_yaw, omega, init_wheel_omega=wheels, init_twist=twist, mu=mu, k_turn=k_turn, device=device)[0]
    ostrich_end = np.stack([pose_log[frame_rows(f + WINDOW_STEPS) - 1, w] for f, w in zip(frame, world)])
    e_pos, e_rot = se3_errors(pose_to_se3(twin_end.astype(np.float64)), pose_to_se3(ostrich_end.astype(np.float64)))
    patch = torch.from_numpy(sample_patches(terrain, xy_yaw, PATCH_SPEC)).to(device)[:, None]
    command = torch.from_numpy(encode(omega)).to(device)
    names = list(net.target_names)

    def predict(patch: torch.Tensor) -> np.ndarray:
        """The label pair (e_pos, e_rot); a pos_rpy net predicts no e_rot, which stays NaN."""
        pred = torch.cat([net.predict(patch[i : i + 512], command[i : i + 512]) for i in range(0, len(frame), 512)]).cpu().numpy()
        pred_rot = pred[:, names.index("e_rot")] if "e_rot" in names else np.full(len(pred), np.nan)
        return np.stack([pred[:, names.index("e_pos")], pred_rot], -1)

    # pred_level: the same commands on a level patch, what an nnflat arm subtracts
    return {"frame": frame, "world": world, "true": np.stack([e_pos, e_rot], -1), "pred": predict(patch),
            "pred_level": predict(torch.zeros_like(patch))}


def report(arms: list[Arm], status: np.ndarray, end_frame: np.ndarray, pose_log: np.ndarray,
           windows: dict[str, np.ndarray]) -> None:
    print(f"\n{'arm':>22} {'status':>9} {'time s':>7} {'path m':>7} {'|pitch|':>8} {'|roll|':>7} "
          f"{'e_pos true/pred':>16} {'e_rot true/pred':>16}")
    for w, arm in enumerate(arms):
        rows = pose_log[SETTLE_STEPS : frame_rows(end_frame[w]), w]
        path = np.linalg.norm(np.diff(rows[:, :2], axis=0), axis=1).sum()
        pitch, roll = pitch_roll(rows[:, 3:7])
        k = windows["world"] == w
        true, pred = windows["true"][k], windows["pred"][k]
        errors = (f"{true[:, 0].mean():6.3f} / {pred[:, 0].mean():6.3f}   {true[:, 1].mean():6.3f} / {pred[:, 1].mean():6.3f}"
                  if k.any() else "      (no full window)")
        tilt = (f"{np.degrees(np.abs(pitch).max()):7.1f}° {np.degrees(np.abs(roll).max()):6.1f}°" if len(rows)
                else f"{'-':>8} {'-':>7}")
        print(f"{arm.name:>22} {status[w]:>9} {end_frame[w] * MPPI_DT:7.1f} {path:7.2f} {tilt}   {errors}")
    if len(windows["frame"]) > 2:
        true, pred = windows["true"], windows["pred"]
        corr = [np.corrcoef(true[:, j], pred[:, j])[0, 1] if np.isfinite(pred[:, j]).all() else np.nan for j in range(2)]
        print(f"\n{len(true)} executed 1 s windows: mean over windows per arm above (true = twin re-run vs "
              f"ostrich, pred = the net); correlation true vs pred e_pos {corr[0]:.2f}, e_rot {corr[1]:.2f}")


def report_timing(arms: list[Arm], plan_ms: np.ndarray, stage_ms: np.ndarray) -> None:
    """Per arm: wall time per replan (median and 90th percentile, the first replan -- graph capture
    -- excluded) and the mean GPU time per refine stage. The arms replan one after another, so a
    replan has the GPU to itself, as on the robot."""
    print(f"\n{'arm':>22} {'replan ms':>10} {'p90':>6}   per refine, ms: " + "  ".join(f"{s:>8}" for s in STAGES))
    for w, arm in enumerate(arms):
        times = plan_ms[1:, w]
        times = times[np.isfinite(times)]
        if times.size == 0:
            continue
        print(f"{arm.name:>22} {np.median(times):10.1f} {np.percentile(times, 90):6.1f}                   "
              + "  ".join(f"{t:8.2f}" for t in stage_ms[w]))


def save(path: pathlib.Path, args: argparse.Namespace, map_name: str, terrain: HeightMapReader, start: np.ndarray,
         goal: tuple[float, float], arms: list[Arm], status: np.ndarray, end_frame: np.ndarray,
         pose_log: np.ndarray, wheel_log: np.ndarray, commands: np.ndarray, driving: np.ndarray,
         windows: dict[str, np.ndarray], label_attrs: dict[str, object], plan_ms: np.ndarray,
         stage_ms: np.ndarray) -> None:
    with h5py.File(path, "w") as f:
        f.attrs["map"] = str(args.map)
        f.attrs["map_name"] = map_name
        f.attrs["checkpoint"] = str(args.checkpoint)
        f.attrs["start"] = start
        f.attrs["goal"] = np.asarray(goal)
        f.attrs["settle_steps"] = SETTLE_STEPS
        f.attrs["ostrich_dt"] = OSTRICH_DT
        f.attrs["mppi_dt"] = MPPI_DT
        for key in ("batch", "n_refine", "spin_frac", "reach_radius", "routing_cell", "pivot_cost", "max_time"):
            f.attrs[key] = getattr(args, key)
        for key, value in label_attrs.items():
            f.attrs[f"label_{key}"] = value
        # one world per arm, in this order
        f["arm"] = np.array([a.name for a in arms], dtype=h5py.string_dtype())
        f["arm_baseline"] = np.array([a.baseline for a in arms], dtype=h5py.string_dtype())
        f["arm_windows"] = np.array([a.windows for a in arms])
        f["arm_nn_refines"] = np.array([a.nn_refines for a in arms])  # refines charged, 0 = vanilla
        f["plan_ms"] = plan_ms  # [frames, W] wall time of update + replan + readback, NaN when not planning
        f["stage_ms"] = stage_ms  # [W, stages] mean GPU ms per refine stage
        f["stage_ms"].attrs["stages"] = list(STAGES)
        f["status"] = np.array(list(status), dtype=h5py.string_dtype())
        f["end_frame"] = end_frame
        f["terrain"] = terrain.H.astype(np.float32)
        f["terrain"].attrs["origin"] = (terrain.x0, terrain.y0)
        f["terrain"].attrs["cell"] = terrain.cell
        f["ostrich_pose"] = pose_log  # [T, W, 7], settle included
        f["ostrich_wheel_qd"] = wheel_log
        f["command"] = commands  # [frames, W, 3] MPPI convention; ostrich got compensate(., yaw gain)
        f["driving"] = driving
        g = f.create_group("windows")
        for key, value in windows.items():
            g[key] = value
    print(f"saved {path}")


def arm_colors(names: list[str]) -> list[tuple[float, float, float]]:
    """One RGB per arm from matplotlib's tab10, in the PNG and the live view alike: vanilla purple,
    the nn arm orange (fixed by name, so --no-vanilla does not recolour it)."""
    import matplotlib

    tab10 = matplotlib.colormaps["tab10"]
    return [tuple(float(c) for c in tab10(4 if name == "vanilla" else 1)[:3]) for name in names]


def plot(path: pathlib.Path, terrain: HeightMapReader, start: np.ndarray, goal: tuple[float, float],
         arms: list[Arm], status: np.ndarray, pose_log: np.ndarray, colors: list[tuple[float, float, float]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    extent = (terrain.x0, terrain.x0 + terrain.nx * terrain.cell, terrain.y0, terrain.y0 + terrain.ny * terrain.cell)
    im = ax.imshow(terrain.H, origin="lower", extent=extent, cmap="gray")
    fig.colorbar(im, ax=ax, shrink=0.7, label="height [m]")
    for w, arm in enumerate(arms):
        xy = pose_log[SETTLE_STEPS:, w, :2]
        ax.plot(xy[:, 0], xy[:, 1], color=colors[w], lw=1.5, label=f"{arm.name}: {status[w]}")
    ax.plot(*start[:2], "go", ms=8)
    ax.plot(*goal, "r*", ms=14)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(path.stem)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True, help="mppi_learning checkpoint")
    parser.add_argument("--map", type=str, default="flat", help="heightmap .png/stem with a start/goal sidecar, or 'flat'")
    parser.add_argument("--start", type=float, nargs=3, default=None, metavar=("X", "Y", "YAW"))
    parser.add_argument("--goal", type=float, nargs=2, default=None, metavar=("X", "Y"))
    parser.add_argument("--nn-weights", type=str, default="100,100", help="the nn arm's weights in the checkpoint's target order")
    parser.add_argument("--baseline", choices=("none", "flat"), default="flat", help="the nn arm's WindowCost baseline")
    parser.add_argument("--no-vanilla", action="store_true", help="skip the vanilla arm")
    parser.add_argument("--no-nn", action="store_true", help="skip the nn arm (vanilla alone)")
    parser.add_argument("--view", action="store_true", help="live GL viewer")
    parser.add_argument("--max-time", type=float, default=40.0, help="per-run time limit [s]")
    parser.add_argument("--spin-frac", type=float, default=None, help="MPPI SPIN prior fraction (default: the sidecar's, else the node's 0)")
    parser.add_argument("--reach-radius", type=float, default=0.3, help="arrival radius [m]")
    parser.add_argument("--batch", type=int, default=4096, help="MPPI rollouts")
    parser.add_argument("--n-refine", type=int, default=N_REFINE, help="MPPI refines per replan (the node's 3)")
    parser.add_argument("--windows", type=int, default=1, help="windows the nn arm charges, 1 to 3")
    parser.add_argument("--nn-refines", type=int, default=None, help="charge the nn cost in the first K refines only (default: every refine)")
    parser.add_argument("--routing-cell", type=float, default=0.32, help="cost-to-go cell size [m]")
    parser.add_argument("--pivot-cost", type=float, default=None, help="cost-to-go point-turn cost (default: the sidecar's, else the node's 0 = off)")
    parser.add_argument("--tag", type=str, default=None, help="output file suffix")
    run(parser.parse_args())
