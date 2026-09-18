"""The uphill benchmark's planners as one call: `plan_path(terrain, start, goal, planner, ctx)`.

    vanilla-off     no `blocked` (the settle is ignored)
    vanilla-on      the shipped pipeline: the settle's per-pose `blocked` field
    nn-gated-pos    `blocked` zeros; an arc is pruned when the pos_rpy net's e_pos > tau_pos
    nn-gated-pitch  same, e_pitch > tau_pitch
    nn-gated-rot    same, the pos_rot net's e_rot > tau_rot
    nn-gated-fused  pruned when the pos_rot net's e_pos > tau_fused_pos OR e_rot > tau_fused_rot,
                    i.e. max(e_pos / tau_fused_pos, e_rot / tau_fused_rot) > 1

All share one `CostToGo` (n_theta=24, 0.3 m arcs) and its `graded_tilt` soft cost, so they differ
only in what makes a transition infeasible. The thresholds are defaults for the two checkpoints
below; a new checkpoint needs a new calibration.
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
from feasibility.planning.gated_lattice import arm_result
from feasibility.planning.gated_lattice import build_gated_solver
from feasibility.planning.gated_lattice import EdgeGatedLatticeSolver
from feasibility.planning.gated_lattice import gated_solve
from feasibility.planning.gated_lattice import lattice_state
from feasibility.planning.gated_lattice import make_cost_to_go

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
PLANNERS = {  # name -> network head it gates on (None = no network)
    "vanilla-off": None,
    "vanilla-on": None,
    "nn-gated-pos": "e_pos",
    "nn-gated-pitch": "e_pitch",
    "nn-gated-rot": "e_rot",
    "nn-gated-fused": "fused",
}
# pos_rpy net (e_pos, e_pitch) and its pos_rot sibling (e_pos, e_rot), trained on the same data
DEFAULT_CHECKPOINT = REPO_ROOT / "outputs" / "checkpoints" / "dataset_arc_my_config_M200_R8_seed0_rpy.pt"
DEFAULT_CHECKPOINT_ROT = REPO_ROOT / "outputs" / "checkpoints" / "dataset_arc_my_config_M200_R8_seed0.pt"
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
    demos' CLI dests, so `PlannerConfig.from_args(args)` picks them out of an argparse Namespace."""

    tau_pos: float = TAU_POS
    tau_pitch: float = TAU_PITCH
    tau_rot: float = TAU_ROT
    tau_fused_pos: float = TAU_FUSED_POS
    tau_fused_rot: float = TAU_FUSED_ROT
    checkpoint: pathlib.Path = DEFAULT_CHECKPOINT
    checkpoint_rot: pathlib.Path = DEFAULT_CHECKPOINT_ROT
    torch_device: str = "cpu"
    chunk: int = 4096  # patches per network batch

    @classmethod
    def from_args(cls, args) -> PlannerConfig:
        return cls(**{f.name: getattr(args, f.name) for f in dataclasses.fields(cls)})


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

    def planner_for(self, terrain: HeightMapReader) -> tuple[CostToGo, EdgeGatedLatticeSolver]:
        elev, grid = terrain.to_hstack("cuda")
        key = (grid.cells_x, grid.cells_y, grid.cell_size, grid.origin_x, grid.origin_y)
        if key != self._grid_key:
            self.ctg = None  # drop the old buffers before allocating new ones (3 GiB GPU)
            self._models.clear()  # load_network checks against the lattice, so reload with it
            self._fields_key, self._fields = None, {}
            gc.collect()
            self.ctg = make_cost_to_go(grid)
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
    coordinates. `audit` also fills a vanilla plan's `max_err` from the pos_rot net's fields
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

    audit_fields = ctx.fields(terrain, "fused") if audit and head is None else {}
    if planner == "vanilla-on":
        result = arm_result(ctg, v_on, blocked, tilt, blocked, audit_fields, start_rct)
    elif planner == "vanilla-off":
        v_off = ctg.solver._record_solve(zeros, ctg.graded_tilt, ctg._goal_rc, ctg.flatness_weight,
                                         False).numpy()
        result = arm_result(ctg, v_off, no_block, tilt, blocked, audit_fields, start_rct)
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
