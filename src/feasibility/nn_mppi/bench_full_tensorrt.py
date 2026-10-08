"""Times the nn cost in a real MPPI replan as `WindowCost` (`WarpTrunk` + the Warp head) against the
FULL TensorRT network (patch + command -> physical errors in one engine), with every form running
as `MppiGpu`'s cost hook INSIDE the captured refine graph. Default: 512 rollouts, windows 0-2,
1 refine, baseline "none".

The forms differ in how much of the network they compute, which is what is being compared:

  * `WarpTrunk + Warp head`: the deployed `WindowCost(n_windows=3)`. Window 0 has one patch (every
    rollout starts at one pose); `update` runs a one-patch `WarpTrunk` on it once, OUTSIDE the graph,
    before the replan, and is timed as part of it. Inside the graph: every rollout's patches at
    steps 10 and 20, `WarpTrunk` over those `trunk_rows` patches, and the head per (window, rollout).
  * `TRT trunk + Warp head`: the same `WindowCost` with `WarpTrunk` swapped for a TensorRT trunk
    engine enqueued on Warp's stream, so it is captured with the rest.
  * `full TRT`: no `update`. Inside the graph a Warp kernel writes every (window, row)'s patch and
    command (window 0 per candidate, windows 1-2 per rollout: `n_cand + 2 * n_rollouts` rows), the
    engine runs the whole net on all of them, and a Warp kernel adds `sum_k weight_k * e_k` to `J`
    as `WindowCost` does. The full graph cannot share a terrain code, so window 0's single patch
    goes through the trunk once per candidate: 1536 trunk rows at 512 rollouts, against 1025.

Every engine is built here with a STATIC profile at exactly the batch its form runs
(`<stem>.<graph>-b<rows>.<precision>.engine` next to the ONNX export, reused if present; an fp16
build takes 5-10 min), so TensorRT tunes for that shape and reserves no activations beyond it.
TensorRT's first enqueue at a shape does setup that must not be captured, so each engine runs once
at construction; its buffers are fixed Warp arrays, so the captured graph stays valid.

Checks: all forms start from one seed and one nominal, so their first refine rolls out identical
trajectories (asserted); each form's per-(window, row) errors and its charge to `J` are compared
with `WarpTrunk + Warp head`'s, which `mppi_cost.py`'s self-test matches against `net.predict`.

Timing interleaves the forms (and a no-hook `vanilla`), `--reps` replans each per round, and keeps
each form's fastest round, because this laptop GPU drops its clock in bursts as it heats. A
replan is what `time_replan.py` times: `update` if any, `replan`, and the plan read back to the host.
Also reported: the mean GPU ms of MPPI's "cost" stage (cost kernel + hook) in that round, from
`MppiGpu`'s CUDA events.

Several `--checkpoint`s are timed in one interleaved run, each form named by the checkpoint's stem
suffix, so two nets compare on one GPU state. A net `WarpTrunk` refuses (`model_tensorRT_friendly`)
gets the TensorRT forms only, and its checks compare against its own first TRT-trunk form. Every
`WindowCost` trunk is also checked against `net.terrain_code` on the patches the refine sampled.

CLI parameters:
    --checkpoint PATH ...  mppi_learning train.py checkpoint(s) the ONNX files were exported from
    --onnx-dir PATH     ONNX export dir; engines are written there (default outputs/nn_mppi/onnx)
    --rollouts INT      MPPI rollouts (default 512)
    --windows INT       charged windows, 2 or 3 (default 3)
    --precision P ...   TensorRT precisions (default fp32 fp16)
    --rounds INT        interleaved timing rounds (default 7)
    --reps INT          replans per form per round (default 20)

Usage:
    python src/feasibility/nn_mppi/bench_full_tensorrt.py --checkpoint outputs/checkpoints/<ckpt>.pt
    python src/feasibility/nn_mppi/bench_full_tensorrt.py --checkpoint outputs/checkpoints/<a>.pt outputs/checkpoints/<b>.pt
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

import numpy as np
import tensorrt as trt
import torch
import warp as wp
from helhest.control.mppi import MppiGpu
from helhest.engine.terrain import Grid

from feasibility.comparator.common import init_warp_device
from feasibility.heightmap.create_box_obstacles import build_centered_box
from feasibility.lattice_learning.patch import HEIGHT_SCALE
from feasibility.lattice_learning.patch import WHEEL_CONTACTS_LOCAL
from feasibility.mppi_learning.command import HALF_TRACK
from feasibility.mppi_learning.command import WHEEL_RADIUS
from feasibility.mppi_learning.command import WINDOW_STEPS
from feasibility.nn_mppi.build_tensorRt_engine import build
from feasibility.nn_mppi.build_tensorRt_engine import LOGGER
from feasibility.nn_mppi.build_tensorRt_engine import onnx_metadata
from feasibility.nn_mppi.closed_loop import node_planner
from feasibility.mppi_learning.model import WindowDivergenceNet
from feasibility.nn_mppi.channel_layer_norm import _max_rel
from feasibility.nn_mppi.closed_loop import routing_field
from feasibility.nn_mppi.mppi_cost import _load
from feasibility.nn_mppi.mppi_cost import _relief
from feasibility.nn_mppi.mppi_cost import check_planner
from feasibility.nn_mppi.mppi_cost import trunk_rows
from feasibility.nn_mppi.mppi_cost import WindowCost
from feasibility.nn_mppi.warp_trunk import _check_trunk

DEVICE = "cuda:0"
ROUTING_CELL = 0.32  # closed_loop.py's --routing-cell default
STATE, GOAL = np.array([-1.6, 0.4, 0.3]), (4.0, 0.0)  # mppi_cost.py's bench: beside the box, facing it
JITTER = (0.05, 0.05, 0.05)  # per-replan start jitter (m, m, rad), as time_replan.py
WEIGHT = 100.0  # cost weight on every target (time_replan.py's default; does not change the timing)


@wp.kernel
def _full_inputs_kernel(
    elevation: wp.array2d(dtype=wp.float32),
    grid: Grid,
    controlled: wp.array2d(dtype=wp.vec3),  # [T + 1, B] every rollout's (x, y, yaw) per step
    target_wheel_omega: wp.array2d(dtype=wp.vec3),  # [T, B]
    window_steps: int,
    n_cand: int,
    n_rollouts: int,
    x_min: float,
    y_min: float,
    cell: float,
    contacts: wp.array(dtype=wp.vec2),
    height_scale: float,
    wheel_radius: float,
    half_track: float,
    patches: wp.array3d(dtype=wp.float32),  # [rows, ny, nx]
    command: wp.array2d(dtype=wp.float32),  # [rows, 4] command.encode: (v_mean, v_slope, wz_mean, wz_slope)
):
    # row r < n_cand: window 0 of candidate r (replica 0: every rollout starts at one pose); after
    # that, window w >= 1 of rollout c at n_cand + (w - 1) * n_rollouts + c
    row, cell_index = wp.tid()
    w = int(0)
    c = row
    if row >= n_cand:
        w = 1 + (row - n_cand) // n_rollouts
        c = (row - n_cand) - (w - 1) * n_rollouts
    first_step = w * window_steps
    nx = patches.shape[2]
    i = cell_index // nx
    j = cell_index - i * nx
    patches[row, i, j] = _relief(elevation, grid, controlled[first_step, c], i, j, x_min, y_min, cell, contacts,
                                 height_scale)
    if cell_index != 0:
        return
    centre = 0.5 * float(window_steps - 1)
    v_sum = float(0.0)
    wz_sum = float(0.0)
    v_moment = float(0.0)
    wz_moment = float(0.0)
    ic_sq = float(0.0)
    for t in range(window_steps):
        wheels = target_wheel_omega[first_step + t, c]
        v = wheel_radius * (wheels[0] + wheels[1]) / 2.0
        wz = wheel_radius * (wheels[1] - wheels[0]) / (2.0 * half_track)
        ic = float(t) - centre
        v_sum += v
        wz_sum += wz
        v_moment += ic * v
        wz_moment += ic * wz
        ic_sq += ic * ic
    n = float(window_steps)
    command[row, 0] = v_sum / n
    command[row, 1] = v_moment / ic_sq * n
    command[row, 2] = wz_sum / n
    command[row, 3] = wz_moment / ic_sq * n


@wp.kernel
def _charge_kernel(
    error: wp.array2d(dtype=wp.float32),  # [rows, K] physical errors, rows as _full_inputs_kernel
    cost_weight: wp.array(dtype=wp.float32),  # [K]
    n_windows: int,
    n_cand: int,
    n_rollouts: int,
    J: wp.array(dtype=wp.float32),  # [n_rollouts] MppiGpu.J, rollout = replica * n_cand + candidate
):
    r = wp.tid()
    cost = float(0.0)
    for k in range(error.shape[1]):
        cost += cost_weight[k] * error[r % n_cand, k]  # window 0: every replica pays its candidate's
        for w in range(1, n_windows):
            cost += cost_weight[k] * error[n_cand + (w - 1) * n_rollouts + r, k]
    J[r] += cost


class WarpEngine:
    """A TensorRT engine bound to fixed Warp buffers and enqueued on Warp's current stream, so a
    call is captured into whatever graph Warp is capturing. Runs once at construction: the first
    enqueue at a shape does setup that a capture must not see."""

    def __init__(self, plan: bytes, inputs: dict[str, tuple[wp.array, tuple[int, ...]]], output: wp.array,
                 device: wp.context.Device) -> None:
        self.engine = trt.Runtime(LOGGER).deserialize_cuda_engine(plan)
        self.context = self.engine.create_execution_context()
        self.device = device
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        (out_name,) = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        for name, (array, shape) in inputs.items():
            assert int(np.prod(shape)) * 4 == array.capacity, (name, shape, array.shape)
            self.context.set_input_shape(name, shape)
            self.context.set_tensor_address(name, array.ptr)
        out_shape = tuple(self.context.get_tensor_shape(out_name))
        assert int(np.prod(out_shape)) * 4 == output.capacity, (out_shape, output.shape)
        self.context.set_tensor_address(out_name, output.ptr)
        self()
        wp.synchronize_device(device)

    def bind(self, name: str, array: wp.array) -> None:
        self.context.set_tensor_address(name, array.ptr)

    def __call__(self) -> None:
        if not self.context.execute_async_v3(wp.get_stream(self.device).cuda_stream):
            raise RuntimeError("TensorRT execution failed")


class TrtTrunk:
    """A TensorRT trunk engine in `WarpTrunk`'s place inside `WindowCost`: same `max_batch`/`code`
    contract, for exactly `rows` patches (the static profile)."""

    def __init__(self, plan: bytes, rows: int, ny: int, nx: int, width: int, device: wp.context.Device) -> None:
        self.max_batch = rows
        self.code = wp.zeros((rows, width // 4), dtype=wp.vec4, device=device)
        warmup = wp.zeros((rows, ny, nx), dtype=wp.float32, device=device)
        self.engine = WarpEngine(plan, {"patch": (warmup, (rows, 1, ny, nx))}, self.code, device)

    def __call__(self, patches: wp.array) -> wp.array:
        if patches.shape[0] != self.max_batch:
            raise ValueError(f"{patches.shape[0]} patches, the engine's static profile is {self.max_batch}")
        self.engine.bind("patch", patches)
        self.engine()
        return self.code


class FullTrtCost:
    """The cost hook with the whole net in one TensorRT engine: inputs, engine and charge, all
    captured into the refine. No `update`: window 0's patch is sampled in the graph too."""

    def __init__(self, plan: bytes, net: object, label_attrs: dict[str, object], planner: MppiGpu, n_windows: int,
                 weights: dict[str, float]) -> None:
        check_planner(planner, label_attrs)
        self.device, self.n_windows = planner.device, n_windows
        self.rows = planner.n_cand + (n_windows - 1) * planner.n_rollouts
        self.spec = spec = net.patch_spec
        with wp.ScopedDevice(self.device):
            self.patches = wp.zeros((self.rows, spec.ny, spec.nx), dtype=wp.float32)
            self.command = wp.zeros((self.rows, 4), dtype=wp.float32)
            self.error = wp.zeros((self.rows, len(net.target_names)), dtype=wp.float32)
            self.contacts = wp.array(WHEEL_CONTACTS_LOCAL.astype(np.float32), dtype=wp.vec2)
            self.cost_weight = wp.array([weights[k] for k in net.target_names], dtype=wp.float32)
        self.engine = WarpEngine(plan, {"patch": (self.patches, (self.rows, 1, spec.ny, spec.nx)),
                                        "command": (self.command, (self.rows, 4))}, self.error, self.device)

    def __call__(self, planner: MppiGpu) -> None:
        spec, sim = self.spec, planner.sim
        wp.launch(
            _full_inputs_kernel, (self.rows, spec.ny * spec.nx),
            inputs=[sim.elevation, sim.grid, sim.controlled, sim.target_wheel_omega, WINDOW_STEPS, planner.n_cand,
                    planner.n_rollouts, spec.x_min, spec.y_min, spec.cell, self.contacts, HEIGHT_SCALE, WHEEL_RADIUS,
                    HALF_TRACK],
            outputs=[self.patches, self.command], device=self.device,
        )
        self.engine()
        wp.launch(_charge_kernel, planner.n_rollouts,
                  inputs=[self.error, self.cost_weight, self.n_windows, planner.n_cand, planner.n_rollouts],
                  outputs=[planner.J], device=self.device)

    def predictions(self) -> np.ndarray:
        return self.error.numpy()


def static_engine(onnx_dir: pathlib.Path, stem: str, graph: str, precision: str, rows: int) -> bytes:
    """The `graph` engine at exactly `rows`, built (and written with a sidecar) if not on disk."""
    path = onnx_dir / f"{stem}.{graph}-b{rows}.{precision}.engine"
    if path.exists():
        return path.read_bytes()
    onnx = onnx_dir / f"{stem}.{graph}.onnx"
    print(f"[build] {path.name} (static batch {rows}) ...", flush=True)
    start = time.perf_counter()
    plan = build(onnx, precision, (rows, rows, rows), workspace_mb=8192)
    seconds = time.perf_counter() - start
    path.write_bytes(plan)
    sidecar = dict(onnx_metadata(onnx), onnx=str(onnx), precision=precision, batch_min_opt_max=[rows] * 3,
                   tensorrt_version=trt.__version__, build_seconds=round(seconds, 1))
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    print(f"[build] {path.name}: {seconds:.0f} s", flush=True)
    return plan


def window_cost_predictions(cost: WindowCost) -> np.ndarray:
    """`WindowCost`'s errors in `FullTrtCost`'s row layout: window 0 per candidate, then the later windows."""
    later = cost.later.prediction.numpy()
    return np.concatenate([cost.window0.prediction.numpy()[0], *later], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=pathlib.Path, nargs="+", required=True, help="mppi_learning checkpoint(s)")
    parser.add_argument("--onnx-dir", type=pathlib.Path, default=pathlib.Path("outputs/nn_mppi/onnx"))
    parser.add_argument("--rollouts", type=int, default=512)
    parser.add_argument("--windows", type=int, default=3, choices=(2, 3))
    parser.add_argument("--precision", nargs="+", default=["fp32", "fp16"], choices=("fp32", "tf32", "fp16"))
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--reps", type=int, default=20)
    args = parser.parse_args()

    init_warp_device(DEVICE)
    device = wp.get_device(DEVICE)
    stems = [c.stem for c in args.checkpoint]
    common = os.path.commonprefix(stems) if len(stems) > 1 else None

    def tag(stem: str) -> str:
        return "" if common is None else f"[{stem[len(common):].lstrip('_') or 'base'}] "

    nets = []
    for checkpoint in args.checkpoint:
        net, attrs, blur = _load(checkpoint)
        if blur:
            raise SystemExit(f"{checkpoint}: blur_terrain checkpoints have no ONNX export")
        nets.append((checkpoint.stem, net, attrs))
    attrs = nets[0][2]
    k_turn, mu = float(attrs["k_turn"]), float(attrs["mu"])
    terrain = build_centered_box(0.3, 0.08, 70.0, 12.0)
    lattice, lattice_grid, vcap = routing_field(terrain, GOAL, k_turn, ROUTING_CELL, 0.0, DEVICE)

    def planner() -> MppiGpu:
        p = node_planner(terrain, k_turn, mu, args.rollouts, 0.0, seed=0, device=DEVICE)
        p.set_lattice(lattice, lattice_grid)
        p.cw.lattice_cap = vcap
        return p

    # form -> (planner, cost hook or None, the window-0 update or None); groups: (net, form names)
    # per checkpoint, its first form the reference its other forms are checked against
    forms: dict[str, tuple[MppiGpu, object, object]] = {"vanilla": (planner(), None, None)}
    groups: list[tuple[WindowDivergenceNet, list[str]]] = []
    n_trunk = trunk_rows(args.rollouts, args.windows)
    n_full = 0
    for stem, net, attrs in nets:
        weights = {k: WEIGHT for k in net.target_names}
        spec, width, names = net.patch_spec, net.geometry.out_channels, []
        try:
            _check_trunk(net)
        except ValueError:
            print(f"[skip] {tag(stem)}WarpTrunk + Warp head: WarpTrunk does not implement this trunk")
        else:
            p = planner()
            cost = WindowCost(net, attrs, p, weights, baseline="none", n_windows=args.windows)
            names.append(f"{tag(stem)}WarpTrunk + Warp head")
            forms[names[-1]] = (p, cost, cost.update)
        for precision in args.precision:
            p = planner()
            trunk = TrtTrunk(static_engine(args.onnx_dir, stem, "trunk", precision, n_trunk), n_trunk, spec.ny,
                             spec.nx, width, device)
            cost = WindowCost(net, attrs, p, weights, baseline="none", n_windows=args.windows, trunk=trunk)
            names.append(f"{tag(stem)}TRT trunk {precision} + Warp head")
            forms[names[-1]] = (p, cost, cost.update)
        for precision in args.precision:
            p = planner()
            n_full = p.n_cand + (args.windows - 1) * p.n_rollouts
            cost = FullTrtCost(static_engine(args.onnx_dir, stem, "full", precision, n_full), net, attrs, p,
                               args.windows, weights)
            names.append(f"{tag(stem)}full TRT {precision}")
            forms[names[-1]] = (p, cost, None)
        groups.append((net, names))
    for p, cost, _ in forms.values():
        if cost is not None:
            p.set_cost_hook(cost)
    print(f"{device.name}, TensorRT {trt.__version__}: {args.rollouts} rollouts x {args.windows} windows x 1 refine; "
          f"trunk rows: WindowCost {n_trunk} (+ 1 window-0 patch in update), full TRT {n_full}")

    # checks: one refine each from the same seed, nominal and start
    for p, cost, update in forms.values():
        if update is not None:
            update(STATE)
        p.replan(STATE, GOAL, 1)
    vanilla = forms["vanilla"][0]
    for p, _, _ in forms.values():
        assert np.array_equal(p.sim.controlled.numpy(), vanilla.sim.controlled.numpy())
    for net, names in groups:
        for name in names:  # every in-graph trunk against torch on the patches this refine sampled
            cost = forms[name][1]
            if isinstance(cost, WindowCost):
                with torch.no_grad():
                    expected = net.terrain_code(wp.to_torch(cost.patches).unsqueeze(1)).flatten(1)
                got = wp.to_torch(cost.terrain_trunk.code).reshape(cost.terrain_trunk.code.shape[0], -1)[:n_trunk]
                print(f"[check] {name}: trunk code max rel diff vs torch {_max_rel(got, expected):.1e}")
        reference_planner, reference_cost, _ = forms[names[0]]
        reference = window_cost_predictions(reference_cost)
        charge_ref = reference_planner.J.numpy() - vanilla.J.numpy()
        for name in names[1:]:
            p, cost, _ = forms[name]
            got = cost.predictions() if isinstance(cost, FullTrtCost) else window_cost_predictions(cost)
            err = np.abs(got - reference).max() / np.abs(reference).max()
            charge = p.J.numpy() - vanilla.J.numpy()
            err_j = np.abs(charge - charge_ref).max() / np.abs(charge_ref).max()
            print(f"[check] {name}: errors max rel diff {err:.1e}, charge to J max rel diff {err_j:.1e} "
                  f"(vs {names[0]})")
        print(f"[check] {names[0]}: errors {', '.join(net.target_names)} span "
              + ", ".join(f"[{reference[:, k].min():.3f}, {reference[:, k].max():.3f}]" for k in range(reference.shape[1])))
    print("[check] identical rollouts in every form")

    rng = np.random.default_rng(0)
    states = [STATE + rng.normal(0.0, JITTER) for _ in range(args.reps)]
    best = {name: (float("inf"), float("nan")) for name in forms}
    for _ in range(args.rounds):
        for name, (p, cost, update) in forms.items():
            p.reset_timing()
            wall = []
            for state in states:
                t0 = time.perf_counter()
                if update is not None:
                    update(state)
                p.replan(state, GOAL, 1)
                p.nominal()  # read back: waits for the GPU
                wall.append((time.perf_counter() - t0) * 1e3)
            median = float(np.median(wall))
            if median < best[name][0]:
                best[name] = (median, p.timing_stats()["cost"]["mean_ms"])
    print(f"[replan] wall ms per replan (median of {args.reps}, fastest of {args.rounds} interleaved rounds), "
          "and the GPU ms of MPPI's cost stage in that round")
    base = best["vanilla"][0]
    ref_name = groups[0][1][0]  # the first checkpoint's first form: WarpTrunk + Warp head where it exists
    ref = best[ref_name][0]
    for name, (wall, cost_ms) in best.items():
        extra = "" if name == "vanilla" else f"  nn {wall - base:6.1f} ms  ({ref / wall:.2f}x {ref_name}'s speed)"
        print(f"  {name:40s} {wall:7.1f} ms   cost stage {cost_ms:6.1f} ms{extra}")


if __name__ == "__main__":
    main()
