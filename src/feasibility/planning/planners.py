"""The uphill benchmark's planners as one call: `plan_path(terrain, start, goal, planner, ctx)`.

    vanilla-off     no `blocked` (the settle is ignored)
    vanilla-on      the shipped pipeline: the settle's per-pose `blocked` field
    nn-gated-pos    `blocked` zeros; an arc is pruned when the pos_rpy net's e_pos > tau_pos
    nn-gated-pitch  same, e_pitch > tau_pitch
    nn-gated-rot    same, the pos_rot net's e_rot > tau_rot
    nn-gated-fused  pruned when the pos_rot net's e_pos > tau_fused_pos OR e_rot > tau_fused_rot,
                    i.e. max(e_pos / tau_fused_pos, e_rot / tau_fused_rot) > 1
    nn-report       vanilla-on's plan (settle `blocked`, no gate), then the v_wz net's predicted
                    error for every primitive the traced path takes -- pivots included, the one net
                    that can describe them -- one column per head the checkpoint carries (pos_rpy
                    by default). Prunes nothing.
    nn-gated-pivot  the settle's `blocked` AND a gate on the IN-PLACE TURNS only: a point turn is
                    pruned when the v_wz net's e_pitch > tau_pivot_pitch, the five forward arcs are
                    left open (tau inf). Needs pivot_cost > 0, or there is nothing to gate.

All share one `CostToGo` (n_theta=24, 0.3 m arcs) and its `graded_tilt` soft cost, so they differ
only in what makes a transition infeasible. The thresholds are defaults for the two checkpoints
below; a new checkpoint needs a new calibration.

`nn-gated-pivot` is the odd one out in keeping `blocked`: the other nn-gated planners were built to
ask whether the network can REPLACE the settle, so they zero it. This one adds to it. The settle
already refuses the walls it can see; the point turn it waves through is the one dragging a wheel
sideways into a low curb, which tilts the body well inside the envelope and is exactly what the
kinematic twin cannot represent. Two different instruments on two different features, not a
substitution.
"""
from __future__ import annotations

import dataclasses
import gc
import math
import pathlib
import time

import numpy as np
import torch
import warp as wp
from helhest.planning.costtogo import CostToGo

from feasibility.heightmap import HeightMapReader
from feasibility.planning.arc_network import arc_error_fields
from feasibility.planning.arc_network import load_network
from feasibility.planning.arc_network import load_network_vwz
from feasibility.planning.arc_network import path_arc_errors
from feasibility.planning.arc_network import vwz_error_fields
from feasibility.planning.gated_lattice import arm_result
from feasibility.planning.gated_lattice import build_gated_solver
from feasibility.planning.gated_lattice import EdgeGatedLatticeSolver
from feasibility.planning.gated_lattice import gated_solve
from feasibility.planning.gated_lattice import lattice_state
from feasibility.planning.gated_lattice import make_cost_to_go
from feasibility.planning.gated_lattice import N_PRIM_ARC
from feasibility.planning.gated_lattice import N_THETA

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
PLANNERS = {  # name -> network head it gates on (None = no network)
    "vanilla-off": None,
    "vanilla-on": None,
    "nn-gated-pos": "e_pos",
    "nn-gated-pitch": "e_pitch",
    "nn-gated-rot": "e_rot",
    "nn-gated-fused": "fused",
    "nn-report": "vwz",
    "nn-gated-pivot": "pivot",  # the v_wz net's e_pitch, on the point turns only
}
# pos_rpy net (e_pos, e_pitch) and its pos_rot sibling (e_pos, e_rot), trained on the same data
DEFAULT_CHECKPOINT = REPO_ROOT / "outputs" / "checkpoints" / "dataset_arc_my_config_M200_R8_seed0_rpy.pt"
DEFAULT_CHECKPOINT_ROT = REPO_ROOT / "outputs" / "checkpoints" / "dataset_arc_my_config_M200_R8_seed0.pt"
# v_wz net (arcs AND pivots), what nn-report queries. Trained separately: pass --checkpoint-vwz.
# A pos_rpy checkpoint by default, not its pos_rot sibling: both describe a pivot, but the rotation
# axes carry very different amounts of the signal. On the 1080 pivot rows of the dataset both were
# fit on, predicted interacting/near-miss medians separate 70.7x on pos_rpy's e_pitch against 9.2x
# on pos_rot's e_rot -- e_rot sums the one informative axis with e_roll and e_yaw, which a pivot
# barely moves (true separation 1.3x on e_yaw), so it arrives diluted. See `load_network_vwz`.
DEFAULT_CHECKPOINT_VWZ = (
    REPO_ROOT / "outputs" / "checkpoints" / "dataset_arc_my_config_M300_R8_seed0_vwz_rpy.pt"
)
TAU_PIVOT_PITCH = 0.06  # [rad] = 3.4 deg -- mid-window of the garage A/B split.
# Previous values, both on weights of this same checkpoint NAME: 0.0873 (5 deg), the tolerance
# shipped until 2026-09-22, fitted against the pre-retrain weights that no longer exist; 0.1130
# (6.5 deg), what `tune_pivot_tau.py` picks by Youden's J on the current weights' own held-out
# pivots. Both keep a route on garage_b, which is why neither is the shipped value.
# One-time tau* calibration for DEFAULT_CHECKPOINT, see THRESHOLDS in benchmarks/bench_uphill_nn.py
TAU_POS = 0.1741  # [m]
TAU_PITCH = 0.0933  # [rad]
# Not calibrated like TAU_POS/TAU_PITCH on tau*, but on ostrich_follow_path_parallel_worlds runs of
# the uphill series: 0.50 and 0.45 plan straight through 60 deg, where pure pursuit arrives only
# 3/6; 0.38 is below 60 deg's bottleneck (0.391), so 60+ has no path, while 5-55 deg stay straight.
TAU_ROT = 0.38
# nn-gated-fused: an arc is pruned when EITHER of the pos_rot net's heads is over its own threshold,
# i.e. max(e_pos / TAU_FUSED_POS, e_rot / TAU_FUSED_ROT) > 1. TAU_FUSED_POS from
# ostrich_follow_path_parallel_worlds runs of the uphill series (3 start/goal pairs per map, the
# sidecar's + 2 random, 2 repeats): at 0.26-0.30 the gated search finds a slightly angled crest
# approach on 60 deg whose worst arc the net under-predicts (0.258-0.280), and it flips 6/6; at 0.25
# 60 deg has no path while 5-55 deg keep one (55 deg is marginal in ostrich itself, ~70% arrive);
# at 0.24 55 deg re-routes to an angled climb (0.230, 0.456) that flips 6/6. So this is a one-point
# window, not a margin: the search exploits the net's blind spots whatever tau is. TAU_FUSED_ROT is
# the offline scan's mid-window value; it is what closes 65 deg at tau_pos 0.30 (e_rot 0.556) but binds
# nothing on this series at 0.25.
TAU_FUSED_POS = 0.25
TAU_FUSED_ROT = 0.46


@dataclasses.dataclass
class PlannerConfig:
    """Thresholds, checkpoints and inference settings of the gated planners. Field names match the
    demos' CLI dests, so `PlannerConfig.from_args(args)` picks them out of an argparse Namespace
    (a field a demo has no flag for keeps its default)."""

    tau_pos: float = TAU_POS
    tau_pitch: float = TAU_PITCH
    tau_rot: float = TAU_ROT
    tau_fused_pos: float = TAU_FUSED_POS
    tau_fused_rot: float = TAU_FUSED_ROT
    tau_pivot_pitch: float = TAU_PIVOT_PITCH
    checkpoint: pathlib.Path = DEFAULT_CHECKPOINT
    checkpoint_rot: pathlib.Path = DEFAULT_CHECKPOINT_ROT
    checkpoint_vwz: pathlib.Path = DEFAULT_CHECKPOINT_VWZ
    torch_device: str = "cpu"
    chunk: int = 4096  # patches per network batch
    # [m-equiv per 15 deg heading bin] > 0 adds helhest_stack's two in-place point turns to the
    # lattice, so a route may turn in place (see make_cost_to_go). 0 = the forward-only lattice
    # TAU_POS/TAU_PITCH/TAU_ROT/TAU_FUSED_* were calibrated on, and the four kappa-gated planners
    # REQUIRE it, since their error fields are indexed by arc curvature and a pivot has none.
    # nn-gated-pivot is the other way round and requires pivot_cost > 0.
    pivot_cost: float = 0.0

    @classmethod
    def from_args(cls, args) -> PlannerConfig:
        return cls(**{f.name: getattr(args, f.name) for f in dataclasses.fields(cls)
                      if hasattr(args, f.name)})


class PlanContext:
    """What planning can reuse across (map, planner) calls: the CostToGo + gated solver while the
    grid geometry is unchanged, each network loaded once, and the predicted error fields of the
    map last seen (one inference pass per network per map). Field keys: e_pos / e_pitch from the
    pos_rpy net, e_pos_rot / e_rot from the pos_rot net, and fused = max of the latter two over
    their fused thresholds, gated at tau 1."""

    def __init__(self, config: PlannerConfig) -> None:
        self.config = config
        self.torch_device = torch.device(config.torch_device)
        self.taus = {"e_pos": float(config.tau_pos), "e_pitch": float(config.tau_pitch),
                     "e_rot": float(config.tau_rot), "fused": 1.0}
        self._grid_key = None
        self.ctg: CostToGo | None = None
        self.gated: EdgeGatedLatticeSolver | None = None
        self._models: dict[str, object] = {}
        self._fields_key = None
        self._fields: dict[str, np.ndarray] = {}
        self._vwz_key = None  # the v_wz FIELD is a separate (and far more expensive) cache
        self._vwz: dict[str, np.ndarray] = {}

    def planner_for(self, terrain: HeightMapReader) -> tuple[CostToGo, EdgeGatedLatticeSolver]:
        elev, grid = terrain.to_hstack("cuda")
        key = (grid.cells_x, grid.cells_y, grid.cell_size, grid.origin_x, grid.origin_y)
        if key != self._grid_key:
            self.ctg = None  # drop the old buffers before allocating new ones (3 GiB GPU)
            self._models.clear()  # load_network checks against the lattice, so reload with it
            self._fields_key, self._fields = None, {}
            self._vwz_key, self._vwz = None, {}  # its last plane is indexed by primitive
            gc.collect()
            self.ctg = make_cost_to_go(grid, pivot_cost=self.config.pivot_cost)
            self.gated = build_gated_solver(self.ctg)
            self._grid_key = key
        self.elev = elev
        return self.ctg, self.gated

    def fields(self, terrain: HeightMapReader, head: str) -> dict[str, np.ndarray]:
        """Predicted error fields for `head` (plus whatever else that network outputs) on `terrain`.
        `planner_for(terrain)` must have been called first."""
        if self._fields_key is not terrain:
            self._fields_key, self._fields = terrain, {}
        if head not in self._fields:
            cfg = self.config
            mode = "pos_rot" if head in ("e_rot", "fused") else "pos_rpy"
            heads = ("e_pos", "e_rot") if mode == "pos_rot" else ("e_pos", "e_pitch")
            if mode not in self._models:
                path = cfg.checkpoint_rot if mode == "pos_rot" else cfg.checkpoint
                self._models[mode] = load_network(path, self.torch_device, self.ctg, label_mode=mode)
            t0 = time.time()
            out = arc_error_fields(self._models[mode], terrain, self.ctg, cfg.chunk,
                                   self.torch_device, heads=heads)
            print(f"network inference ({mode}): {time.time() - t0:.1f} s")
            if mode == "pos_rot":  # its e_pos is a different net's than nn-gated-pos gates on
                out["e_pos_rot"] = out.pop("e_pos")
                out["fused"] = np.maximum(out["e_pos_rot"] / float(cfg.tau_fused_pos),
                                          out["e_rot"] / float(cfg.tau_fused_rot))
            self._fields.update(out)
        return self._fields

    def vwz_model(self):
        """The (v_drive, wz_drive)-commanded network, loaded once per lattice. Unlike the kappa
        nets it is allowed on a pivot lattice -- describing a point turn is what it is for."""
        if "v_wz" not in self._models:
            self._models["v_wz"] = load_network_vwz(self.config.checkpoint_vwz, self.torch_device,
                                                    self.ctg)
        return self._models["v_wz"]

    def vwz_fields(self, terrain: HeightMapReader) -> dict[str, np.ndarray]:
        """{head: [ny, nx, n_theta, n_prim]} from the v_wz net, every primitive of the lattice --
        what `nn-gated-pivot` gates on. `planner_for(terrain)` must have been called first.

        One inference pass over every lattice pose, which is minutes on CPU for a map that is not
        row-invariant (see `arc_network.vwz_error_fields`), so it is cached per map for the life of
        this context -- plan every arm of a comparison in ONE process and it is paid once.
        """
        if self._vwz_key is not terrain:
            model = self.vwz_model()
            t0 = time.time()
            self._vwz = vwz_error_fields(model, terrain, self.ctg, self.config.chunk,
                                         self.torch_device, heads=tuple(model.target_names))
            n = self.ctg.grid.cells_y * self.ctg.grid.cells_x * N_THETA
            print(f"network inference (v_wz field, {n} lattice poses x {self.ctg.solver.n_prim} "
                  f"primitives): {time.time() - t0:.1f} s")
            self._vwz_key = terrain
        return self._vwz

    def vwz_path_errors(self, terrain: HeightMapReader, result: dict) -> dict[str, np.ndarray]:
        """The v_wz net's {e_pos, e_rot} [n_steps] for the primitives of a traced `result` (see
        `arc_network.path_arc_errors`). `planner_for(terrain)` must have been called first."""
        t0 = time.time()
        out = path_arc_errors(self.vwz_model(), terrain, self.ctg, result["states"],
                              result["prims"], self.config.chunk, self.torch_device)
        print(f"network inference (v_wz, {len(result['prims'])} path primitives): "
              f"{time.time() - t0:.2f} s")
        return out


def plan_path(
    terrain: HeightMapReader,
    start: tuple[float, float, float],
    goal: tuple[float, float],
    planner: str,
    ctx: PlanContext,
    audit: bool = False,
) -> dict:
    """The planner `planner` (a PLANNERS key), solved and traced from `start`. Returns arm_result's
    dict plus `tau`, `head` (None for the vanilla planners), `gate`, a printable description of
    the threshold ("" for the vanilla planners), and `xy` [n, 2], the traced lattice poses in world
    coordinates. `nn-report` and `nn-gated-pivot` also carry `arc_errors` (one entry per checkpoint
    head, per primitive taken, so index i is the step from pose i to i+1); for `nn-gated-pivot`
    `tau` is the point turns' threshold alone, the forward arcs being ungated.
    `audit` also fills a vanilla plan's `max_err` from the pos_rot net's fields
    (e_pos_rot / e_rot / fused), so ungated paths can calibrate the gate."""
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

    # nn-report and nn-gated-pivot are the v_wz planners: they describe every primitive of any
    # lattice, so only the kappa-indexed ones are refused here
    if ((head not in (None, "vwz", "pivot")) or audit) and ctg.solver.n_prim != N_PRIM_ARC:
        raise NotImplementedError(
            f"{planner} needs a predicted error per primitive, but this lattice has "
            f"{ctg.solver.n_prim} primitives (pivot_cost > 0) and arc_network's fields cover only "
            f"the {N_PRIM_ARC} forward arcs -- a pivot has no curvature to index them by. Gating a "
            "pivot lattice needs v_wz-commanded fields; until then use a vanilla planner, or "
            "pivot_cost 0."
        )
    audit_fields = ctx.fields(terrain, "fused") if audit and head is None else {}
    if planner in ("vanilla-on", "nn-report"):
        result = arm_result(ctg, v_on, blocked, tilt, blocked, audit_fields, start_rct)
        if planner == "nn-report":
            result["arc_errors"] = ctx.vwz_path_errors(terrain, result)
    elif planner == "vanilla-off":
        v_off = ctg.solver._record_solve(zeros, ctg.graded_tilt, ctg._goal_rc, ctg.flatness_weight,
                                         False).numpy()
        result = arm_result(ctg, v_off, no_block, tilt, blocked, audit_fields, start_rct)
    elif planner == "nn-gated-pivot":
        if ctg.solver.n_prim == N_PRIM_ARC:
            raise ValueError(
                "nn-gated-pivot gates the in-place point turns, and pivot_cost 0 builds a lattice "
                f"with only the {N_PRIM_ARC} forward arcs -- nothing to gate. Re-run with "
                "pivot_cost > 0 (0.15 is the demos' default), or pick another planner."
            )
        fields = ctx.vwz_fields(terrain)
        tau = float(ctx.config.tau_pivot_pitch)
        # The gate, per primitive: the five forward arcs keep inf (no finite prediction exceeds it,
        # so they are untouched) and the two point turns get the tolerance calibrated for THEM.
        # One array rather than one number because the two classes are different populations of the
        # same head -- see TAU_PIVOT_PITCH.
        taus = np.full(ctg.solver.n_prim, math.inf, np.float32)
        taus[N_PRIM_ARC:] = tau
        err = wp.array(fields["e_pitch"], dtype=wp.float32, device=ctg.device)
        # `ctg.blocked`, where the other gated planners pass `zeros`: they were built to ask
        # whether the net can REPLACE the settle, this one adds to it. The settle still stops the
        # walls; the net only adds the low curb it tolerates (see the module docstring).
        v = gated_solve(gated, ctg, ctg.blocked, ctg.graded_tilt, err, taus)
        result = arm_result(ctg, v, blocked, tilt, blocked, fields, start_rct, fields["e_pitch"],
                            taus)
        # the field is already in hand, so the path's own predictions cost nothing -- this is what
        # `view_planned_path.py` prints per pose, the same columns nn-report gives
        taken, prims = result["states"][:-1], result["prims"]
        result["arc_errors"] = {
            name: f[taken[:, 0], taken[:, 1], taken[:, 2], prims] if prims
            else np.empty(0, np.float32)
            for name, f in fields.items()
        }
    else:
        fields = ctx.fields(terrain, head)
        tau = ctx.taus[head]
        err = wp.array(fields[head], dtype=wp.float32, device=ctg.device)
        v = gated_solve(gated, ctg, zeros, ctg.graded_tilt, err, tau)
        result = arm_result(ctg, v, no_block, tilt, blocked, fields, start_rct, fields[head], tau)

    rct = result["states"]
    if head is None:
        gate = ""
    elif head == "vwz":
        gate = "none (report only)"
    elif head == "pivot":
        gate = f"in-place turns only: e_pitch > {tau:.4f} rad, forward arcs ungated"
    elif head == "fused":
        cfg = ctx.config
        gate = f"max(e_pos/{cfg.tau_fused_pos:.4f}, e_rot/{cfg.tau_fused_rot:.4f}) > 1"
    else:
        gate = f"{head} > {tau:.4f}"
    result.update(
        head=head, tau=tau, gate=gate,
        xy=np.stack([grid.origin_x + rct[:, 1] * grid.cell_size,
                     grid.origin_y + rct[:, 0] * grid.cell_size], axis=-1),
    )
    return result
