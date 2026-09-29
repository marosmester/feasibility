"""Closed-loop MPPI in ostrich, with and without the net's window-0 cost -- mppi_learning/design.md
section 9d.6.

MPPI plans with helhest_stack's twin, as on the robot; OSTRICH executes. Every `dynamics.DT` (0.1 s,
one replan per perception frame on the robot) each world:

  1. reads its REALIZED state from ostrich's logs: (x, y, yaw), wheel speeds, and the body twist over
     the last 0.1 s (`generate_dataset.realized_start`, the same estimate the dataset seeded the twin
     with);
  2. seeds its planner's rollouts with them (`set_initial_wheel_omega` / `set_initial_twist`, as the
     ROS node does from /joint_states and odometry), calls `WindowCost.update` on the `nn` arms, and
     replans (3 refines);
  3. blends the new plan with the previous one, shifted a step (the node's `plan_consistency` 0.3),
     and sends its first step to ostrich -- yaw-compensated by the checkpoint's `ostrich_yaw_gain`,
     exactly as the dataset commanded ostrich, since that gain is part of what the net's labels mean.

Every (arm, repeat) is one replicated world of ONE ostrich build (replicated worlds do not collide
with each other), each with its own `MppiGpu`. Arms: `vanilla` (no hook), one `nn` arm per
`--nn-weights` entry (the raw predicted error), and one `nnflat` arm per `--nn-flat-weights` entry
(`WindowCost`'s baseline "flat": only the error predicted above the same command on level ground).
Ostrich is nondeterministic, so repeats of one arm differ too.

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
per-window errors and predictions, and a bird's-eye plot of every world's path.

CLI parameters:
    --checkpoint PATH       mppi_learning train.py checkpoint (required)
    --map PATH|flat         heightmap (.png or stem) whose .yaml sidecar has `start`/`goal`, or
                            `flat` (a 12 m flat map, start (-4, 0, 0), goal (4, 0))
    --start X Y YAW         override the sidecar's start
    --goal X Y              override the sidecar's goal
    --nn-weights W [W ...]  one nn arm per entry, comma-separated weights in the checkpoint's
                            target order (e.g. 100,100 for e_pos [1/m], e_rot [1/rad]); default 100,100
    --nn-flat-weights W [W ...]  the same, one nnflat arm per entry (default none)
    --no-vanilla            skip the vanilla arm
    --repeats INT           worlds per arm (default 1)
    --max-time S            per-run time limit (default 40)
    --spin-frac F           MPPI SPIN prior fraction (default 0, the node's)
    --reach-radius M        arrival radius (default 0.3, the node's plan_reach_radius)
    --batch INT             MPPI rollouts (default 4096)
    --routing-cell M        cost-to-go cell size, the map max-pooled to it (default 0.32)
    --pivot-cost C          cost-to-go point-turn primitives (default 0, the node's; a map whose only
                            way out is a turn in place, like the garage, needs > 0 and --spin-frac)
    --tag STR               output file suffix (default: the arms)

Usage:
    python src/feasibility/nn_mppi/closed_loop.py --checkpoint outputs/checkpoints/<ckpt>.pt --map flat
    python src/feasibility/nn_mppi/closed_loop.py --checkpoint outputs/checkpoints/<ckpt>.pt \\
        --map assets/garage/garage_b --pivot-cost 0.15 --spin-frac 0.1 --nn-weights 3,3 10,10 --repeats 2
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
import time

import h5py
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
N_REFINE = 3
PLAN_CONSISTENCY = 0.3  # the node's EMA of the new plan toward the previous one, shifted a step
EDGE_MARGIN = 0.5  # [m] a world this close to the map edge is stopped as off_map
FLIP_DEG = 60.0  # |pitch| or |roll| past this: flipped


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
    planner = MppiGpu(sim, cost, sampling, n_theta=24, seed=seed)
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
    runs that many times. The pose and wheel logs cover the whole run, settle included."""

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

    def drive(self, command: np.ndarray) -> None:
        """[W, 3] ostrich wheel speeds, held for one MPPI frame."""
        rows = self._setpoints_wp[self.step : self.step + STEPS_PER_FRAME]
        rows.assign(np.ascontiguousarray(np.broadcast_to(command, rows.shape), np.float32))
        self._launch(STEPS_PER_FRAME)

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


def pitch_roll(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[..., 4] (qx, qy, qz, qw) -> (pitch, roll) [...], ZYX."""
    qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    pitch = np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
    roll = np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    return pitch, roll


def load_map(spec: str, start: list[float] | None, goal: list[float] | None) -> tuple[HeightMapReader, str, np.ndarray, tuple[float, float]]:
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
    return terrain, name, np.asarray(start[:3], np.float64), (float(goal[0]), float(goal[1]))


def run(args: argparse.Namespace) -> None:
    device = "cuda:0"
    init_warp_device(device)
    terrain, map_name, start, goal = load_map(args.map, args.start, args.goal)
    net, ckpt = load_checkpoint(args.checkpoint, torch.device(device))
    net.eval()
    attrs = ckpt["label_attrs"]
    k_turn, mu, yaw_gain = float(attrs["k_turn"]), float(attrs["mu"]), float(attrs["ostrich_yaw_gain"])

    arms = [] if args.no_vanilla else [Arm("vanilla", None)]
    for flag, prefix, baseline, entries in (("--nn-weights", "nn", "none", args.nn_weights),
                                            ("--nn-flat-weights", "nnflat", "flat", args.nn_flat_weights)):
        for text in entries:
            values = [float(v) for v in text.split(",")]
            if len(values) != len(net.target_names):
                raise SystemExit(f"{flag} {text}: need {len(net.target_names)} values for {net.target_names}")
            arms.append(Arm(f"{prefix}[{text}]", dict(zip(net.target_names, values)), baseline))
    worlds = [(arm, r) for arm in arms for r in range(args.repeats)]
    n_worlds, n_frames = len(worlds), int(round(args.max_time / MPPI_DT))
    print(f"[map]      {map_name}: {terrain.nx}x{terrain.ny} @ {terrain.cell} m, start {start.tolist()}, goal {goal}")
    print(f"[labels]   k_turn {k_turn}, mu {mu}, ostrich yaw gain {yaw_gain} ({args.checkpoint.name})")
    print(f"[worlds]   {n_worlds}: " + ", ".join(f"{a.name} x{args.repeats}" for a in arms))

    t0 = time.perf_counter()
    V, lattice_grid, vcap = routing_field(terrain, goal, k_turn, args.routing_cell, args.pivot_cost, device)
    planners, costs = [], []
    for i, (arm, _) in enumerate(worlds):
        planner = node_planner(terrain, k_turn, mu, args.batch, args.spin_frac, seed=i, device=device)
        planner.set_lattice(V, lattice_grid)
        planner.cw.lattice_cap = vcap
        cost = None
        if arm.weights is not None:
            cost = WindowCost(net, attrs, planner, arm.weights, blur_terrain=bool(ckpt["blur_terrain"]), baseline=arm.baseline)
            planner.set_cost_hook(cost)
        planners.append(planner)
        costs.append(cost)
    print(f"[setup]    cost-to-go + {n_worlds} planners in {time.perf_counter() - t0:.1f} s")

    spawn_zpr, _, _ = settle_batch(terrain, start[None], mu, device)
    spawn_zpr = spawn_zpr.astype(np.float64)
    spawn_zpr[:, 0] += SPAWN_CLEARANCE
    sim_config, render_config, engine_config, logging_config = compose_ostrich_config(())
    render_config.vis_type = "null"
    sim_config.target_timestep_seconds = OSTRICH_DT
    sim_config.num_worlds = n_worlds
    ostrich = OstrichStepper(
        sim_config, render_config, engine_config, logging_config, k_p=K_P, **friction_kwargs(mu),
        terrain=terrain, spawn_pose=np.repeat(start[None], n_worlds, 0), spawn_zpr=np.repeat(spawn_zpr, n_worlds, 0),
    )
    ostrich.begin(n_frames)

    x_lo, y_lo = terrain.x0 + EDGE_MARGIN, terrain.y0 + EDGE_MARGIN
    x_hi, y_hi = terrain.x0 + terrain.nx * terrain.cell - EDGE_MARGIN, terrain.y0 + terrain.ny * terrain.cell - EDGE_MARGIN
    status = np.array(["timeout"] * n_worlds, dtype=object)
    done = np.zeros(n_worlds, bool)
    end_frame = np.full(n_worlds, n_frames)
    commands = np.zeros((n_frames, n_worlds, 3), np.float32)  # MPPI convention, what the twin would get
    driving = np.zeros((n_frames, n_worlds), bool)  # the frame's command came from MPPI
    previous_plan = [None] * n_worlds
    t_plan = 0.0
    for f in range(n_frames):
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
        t = time.perf_counter()
        for w in np.flatnonzero(~done):
            planner, cost = planners[w], costs[w]
            planner.sim.set_initial_wheel_omega(wheels[w])
            planner.sim.set_initial_twist(twist[w])
            if cost is not None:
                cost.update(xy_yaw[w])
            planner.replan(xy_yaw[w], goal, N_REFINE)
            plan = planner.nominal()
            if previous_plan[w] is not None:
                shifted = np.roll(previous_plan[w], -1, axis=0)
                shifted[-1] = previous_plan[w][-1]
                plan = (1.0 - PLAN_CONSISTENCY) * plan + PLAN_CONSISTENCY * shifted
                planner.set_nominal(plan)
            previous_plan[w] = plan.copy()
            commands[f, w] = (plan[0, 0], plan[0, 1], 0.5 * (plan[0, 0] + plan[0, 1]))
            driving[f, w] = True
        t_plan += time.perf_counter() - t
        if done.all():
            n_frames = f
            break
        ostrich.drive(compensate(commands[f], yaw_gain))
        if f % 50 == 0:
            print(f"  t={f * MPPI_DT:5.1f} s: " + ", ".join(f"{status[w] if done[w] else 'driving'}" for w in range(n_worlds)))
    n_run = min(n_frames, commands.shape[0])
    print(f"[run]      {n_run} frames, planning {t_plan / max(n_run, 1) * 1e3:.0f} ms per frame for {n_worlds} worlds")

    pose_log, wheel_log = ostrich.logs()
    windows = window_errors(terrain, pose_log, wheel_log, commands[:n_run], driving[:n_run], net, mu, k_turn, device)
    report(worlds, status, end_frame, pose_log, windows, goal)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.tag or "_".join(a.name.replace("[", "").replace("]", "").replace(",", "-") for a in arms)
    stem = OUT / f"{map_name}_{tag}"
    save(stem.with_suffix(".h5"), args, map_name, terrain, start, goal, worlds, status, end_frame, pose_log, wheel_log,
         commands[:n_run], driving[:n_run], windows, attrs)
    plot(stem.with_suffix(".png"), terrain, start, goal, worlds, status, pose_log)


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


def report(worlds: list[tuple[Arm, int]], status: np.ndarray, end_frame: np.ndarray, pose_log: np.ndarray,
           windows: dict[str, np.ndarray], goal: tuple[float, float]) -> None:
    print(f"\n{'world':>18} {'status':>9} {'time s':>7} {'path m':>7} {'|pitch|':>8} {'|roll|':>7} "
          f"{'e_pos true/pred':>16} {'e_rot true/pred':>16}")
    for w, (arm, r) in enumerate(worlds):
        rows = pose_log[SETTLE_STEPS : frame_rows(end_frame[w]), w]
        path = np.linalg.norm(np.diff(rows[:, :2], axis=0), axis=1).sum()
        pitch, roll = pitch_roll(rows[:, 3:7])
        k = windows["world"] == w
        true, pred = windows["true"][k], windows["pred"][k]
        errors = (f"{true[:, 0].mean():6.3f} / {pred[:, 0].mean():6.3f}   {true[:, 1].mean():6.3f} / {pred[:, 1].mean():6.3f}"
                  if k.any() else "      (no full window)")
        print(f"{arm.name + f' #{r}':>18} {status[w]:>9} {end_frame[w] * MPPI_DT:7.1f} {path:7.2f} "
              f"{np.degrees(np.abs(pitch).max()):7.1f}° {np.degrees(np.abs(roll).max()):6.1f}°   {errors}")
    if len(windows["frame"]) > 2:
        true, pred = windows["true"], windows["pred"]
        corr = [np.corrcoef(true[:, j], pred[:, j])[0, 1] if np.isfinite(pred[:, j]).all() else np.nan for j in range(2)]
        print(f"\n{len(true)} executed 1 s windows: mean over windows per world above (true = twin re-run vs "
              f"ostrich, pred = the net); correlation true vs pred e_pos {corr[0]:.2f}, e_rot {corr[1]:.2f}")


def save(path: pathlib.Path, args: argparse.Namespace, map_name: str, terrain: HeightMapReader, start: np.ndarray,
         goal: tuple[float, float], worlds: list[tuple[Arm, int]], status: np.ndarray, end_frame: np.ndarray,
         pose_log: np.ndarray, wheel_log: np.ndarray, commands: np.ndarray, driving: np.ndarray,
         windows: dict[str, np.ndarray], label_attrs: dict[str, object]) -> None:
    with h5py.File(path, "w") as f:
        f.attrs["map"] = str(args.map)
        f.attrs["map_name"] = map_name
        f.attrs["checkpoint"] = str(args.checkpoint)
        f.attrs["start"] = start
        f.attrs["goal"] = np.asarray(goal)
        f.attrs["settle_steps"] = SETTLE_STEPS
        f.attrs["ostrich_dt"] = OSTRICH_DT
        f.attrs["mppi_dt"] = MPPI_DT
        for key in ("batch", "spin_frac", "reach_radius", "routing_cell", "pivot_cost", "max_time"):
            f.attrs[key] = getattr(args, key)
        for key, value in label_attrs.items():
            f.attrs[f"label_{key}"] = value
        f["arm"] = np.array([a.name for a, _ in worlds], dtype=h5py.string_dtype())
        f["arm_baseline"] = np.array([a.baseline for a, _ in worlds], dtype=h5py.string_dtype())
        f["repeat"] = np.array([r for _, r in worlds])
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


def plot(path: pathlib.Path, terrain: HeightMapReader, start: np.ndarray, goal: tuple[float, float],
         worlds: list[tuple[Arm, int]], status: np.ndarray, pose_log: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    extent = (terrain.x0, terrain.x0 + terrain.nx * terrain.cell, terrain.y0, terrain.y0 + terrain.ny * terrain.cell)
    im = ax.imshow(terrain.H, origin="lower", extent=extent, cmap="gray")
    fig.colorbar(im, ax=ax, shrink=0.7, label="height [m]")
    names = list(dict.fromkeys(a.name for a, _ in worlds))
    colors = plt.cm.tab10(np.arange(len(names)))
    for w, (arm, r) in enumerate(worlds):
        xy = pose_log[SETTLE_STEPS:, w, :2]
        ax.plot(xy[:, 0], xy[:, 1], color=colors[names.index(arm.name)], lw=1.5,
                label=f"{arm.name} #{r}: {status[w]}")
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
    parser.add_argument("--nn-weights", type=str, nargs="*", default=["100,100"], help="one nn arm per entry, weights in the checkpoint's target order")
    parser.add_argument("--nn-flat-weights", type=str, nargs="*", default=[], help="one nnflat arm (baseline 'flat') per entry")
    parser.add_argument("--no-vanilla", action="store_true", help="skip the vanilla arm")
    parser.add_argument("--repeats", type=int, default=1, help="worlds per arm")
    parser.add_argument("--max-time", type=float, default=40.0, help="per-run time limit [s]")
    parser.add_argument("--spin-frac", type=float, default=0.0, help="MPPI SPIN prior fraction (the node's 0)")
    parser.add_argument("--reach-radius", type=float, default=0.3, help="arrival radius [m]")
    parser.add_argument("--batch", type=int, default=4096, help="MPPI rollouts")
    parser.add_argument("--routing-cell", type=float, default=0.32, help="cost-to-go cell size [m]")
    parser.add_argument("--pivot-cost", type=float, default=0.0, help="cost-to-go point-turn cost (the node's 0 = off)")
    parser.add_argument("--tag", type=str, default=None, help="output file suffix")
    run(parser.parse_args())
