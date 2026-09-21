"""helhest_stack's lattice planner (`CostToGo`) with a per-ARC gate, and tracing its policy.

The gate is a local copy of `lattice_solver._relax_lattice_pose_kernel` with one extra check
(`EdgeGatedLatticeSolver`): primitive p out of lattice pose (row, col, heading) is pruned when a
predicted error field says arc_error[row, col, heading, p] > tau[p]. helhest_stack is not modified.
`trace_states` / `arm_result` follow the solved value function from a start pose and audit the arcs
actually taken.

The threshold is PER PRIMITIVE (`primitive_taus` broadcasts a scalar, so a one-number gate is
unchanged). That is what lets a planner gate the two in-place point turns and leave the five
forward arcs open: the same head means different things on the two, and `planners.TAU_PIVOT_PITCH`
is calibrated for pivots alone.

`python src/feasibility/planning/gated_lattice.py` is the smoke test: an open gate reproduces the stock
relaxation bit for bit, a shut gate makes the goal unreachable, `pivot_cost` > 0 appends the two
point turns and makes a goal BEHIND the robot reachable in a corridor too narrow to loop in, and a
per-primitive gate that shuts only those two takes that route away again (CUDA only).
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import warp as wp
from helhest import dynamics
from helhest.planning.costtogo import CostToGo
from helhest.planning.lattice_solver import LatticeValueSolver

N_THETA = 24
STEP = 0.3  # [m] CostToGo's default arc length -- asserted against a network checkpoint's arc_len
N_PRIM_ARC = 5  # forward arcs in _build_primitives' `turns`; pivot_cost > 0 appends 2 point turns


@wp.kernel
def _relax_gated_kernel(
    dist_in: wp.array(dtype=wp.float32, ndim=3),
    blocked: wp.array(dtype=wp.float32, ndim=3),
    tilt: wp.array(dtype=wp.float32, ndim=3),
    prim_dr: wp.array(dtype=wp.int32, ndim=2),
    prim_dc: wp.array(dtype=wp.int32, ndim=2),
    prim_heading: wp.array(dtype=wp.int32, ndim=2),
    prim_cost: wp.array(dtype=wp.float32, ndim=2),
    sweep_dr: wp.array(dtype=wp.int32, ndim=3),
    sweep_dc: wp.array(dtype=wp.int32, ndim=3),
    sweep_n: wp.array(dtype=wp.int32, ndim=2),
    n_prim: wp.int32,
    tilt_weight: wp.float32,
    inf: wp.float32,
    arc_error: wp.array(dtype=wp.float32, ndim=4),  # [h, w, n_theta, n_prim] predicted error
    tau: wp.array(dtype=wp.float32, ndim=1),  # [n_prim] primitive p pruned when its error > tau[p]
    dist_out: wp.array(dtype=wp.float32, ndim=3),
    changed: wp.array(dtype=wp.int32),
):
    """lattice_solver._relax_lattice_pose_kernel, verbatim, plus ONE gate: primitive p out of pose
    (r, c, t) is skipped when arc_error[r, c, t, p] > tau[p]. Indexed at the source pose, like every
    other per-arc table the relaxation reads.

    `tau` is PER PRIMITIVE, not one number: the point turns and the forward arcs are different
    populations of the same predicted error (a pivot's median true e_pitch is 0.043 rad against an
    arc's 0.009), so a planner may want to gate one and leave the other open -- `planners.py`'s
    `nn-gated-pivot` sets the five arcs to inf and the two pivots to TAU_PIVOT_PITCH. `set_gate`
    still takes a scalar and broadcasts it, which is every earlier caller."""
    r, c, t = wp.tid()
    h = dist_in.shape[0]
    w = dist_in.shape[1]
    if blocked[r, c, t] > 0.5:
        dist_out[r, c, t] = inf
        return
    best = dist_in[r, c, t]
    for p in range(n_prim):
        ok = int(1)
        if arc_error[r, c, t, p] > tau[p]:
            ok = 0
        ns = sweep_n[t, p]
        tsum = float(0.0)
        for s in range(ns):
            sr = r + sweep_dr[t, p, s]
            sc = c + sweep_dc[t, p, s]
            inb = int(0)
            if sr >= 0 and sr < h and sc >= 0 and sc < w:
                inb = 1
            scr = wp.clamp(sr, 0, h - 1)
            scc = wp.clamp(sc, 0, w - 1)
            if inb == 0 or blocked[scr, scc, t] > 0.5:
                ok = 0
            tsum += tilt[scr, scc, t]
        if ok == 1:
            nr = r + prim_dr[t, p]
            nc = c + prim_dc[t, p]
            if nr >= 0 and nr < h and nc >= 0 and nc < w:
                arc = prim_cost[t, p]
                if ns > 0:
                    arc = arc * (1.0 + tilt_weight * tsum / float(ns))
                best = wp.min(best, arc + dist_in[nr, nc, prim_heading[t, p]])
    dist_out[r, c, t] = best
    if best < dist_in[r, c, t]:
        changed[0] = 1


def primitive_taus(tau: float | Sequence[float] | np.ndarray, n_prim: int) -> np.ndarray:
    """[n_prim] float32 thresholds from either one number (the same tolerance for every primitive,
    what every gated planner before `nn-gated-pivot` wants) or one per primitive. `math.inf` leaves
    a primitive ungated, since no finite predicted error can exceed it."""
    out = np.asarray(tau, dtype=np.float32)
    if out.ndim == 0:
        out = np.full(n_prim, out, dtype=np.float32)
    assert out.shape == (n_prim,), f"tau must be a scalar or {n_prim} values, got {out.shape}"
    return out


class EdgeGatedLatticeSolver(LatticeValueSolver):
    """LatticeValueSolver whose relaxation also prunes individual ARCS by a predicted error field.
    Built with the same arguments CostToGo uses, so its primitive tables are identical (asserted in
    build_gated_solver). `set_gate` must be called before `_record_solve`."""

    def set_gate(self, arc_error: wp.array, tau: float | Sequence[float] | np.ndarray) -> None:
        assert arc_error.shape == (self.height, self.width, self.n_theta, self.n_prim)
        self._arc_error = arc_error
        self._tau = wp.array(primitive_taus(tau, self.n_prim), dtype=wp.float32, device=self.device)

    def _relax(
        self,
        dist_in: wp.array,
        dist_out: wp.array,
        blocked: wp.array,
        tilt: wp.array,
        tilt_weight: float,
    ) -> None:
        wp.launch(
            _relax_gated_kernel,
            dim=(self.height, self.width, self.n_theta),
            inputs=[
                dist_in,
                blocked,
                tilt,
                self._prim_dr,
                self._prim_dc,
                self._prim_heading,
                self._prim_cost,
                self._sweep_dr,
                self._sweep_dc,
                self._sweep_n,
                self.n_prim,
                float(tilt_weight),
                self._inf,
                self._arc_error,
                self._tau,
            ],
            outputs=[dist_out, self._changed],
            device=self.device,
        )


def make_cost_to_go(grid, device: str = "cuda", pivot_cost: float = 0.0) -> CostToGo:
    """CostToGo on `grid` with Helhest's robot/solver parameters and this lattice (N_THETA, STEP).

    `pivot_cost` > 0 [m-equivalent per heading bin] appends helhest_stack's two POINT-TURN
    primitives (same cell, heading +-1 bin) to the five forward arcs, so a route may turn in place
    instead of looping -- and a goal behind the robot becomes reachable at all in a corridor too
    narrow for a min_turn_radius U-turn. The lattice pivot IS one `lattice_learning` pivot trial:
    N_THETA = 24 makes a bin `arc.PIVOT_ANGLE`, turned in `arc.ARC_DURATION_S` at `arc.OMEGA_NOM`.

    It defaults to 0 -- the forward-only lattice every threshold in `planners.py` was calibrated on,
    and the only one `arc_network`'s kappa-indexed error fields can describe (a pivot has no
    curvature; gating one needs a v_wz checkpoint). Raising it changes `solver.n_prim` 5 -> 7, which
    `plan_path` refuses to gate rather than mis-index.
    """
    return CostToGo(grid, dynamics.robot_params(), dynamics.planning_solver(), n_theta=N_THETA,
                    step=STEP, pivot_cost=pivot_cost, device=device)


def pivot_cost_of(solver: LatticeValueSolver) -> float:
    """The `pivot_cost` a BUILT solver carries: 0 with only the N_PRIM_ARC forward arcs, else the
    point turns' own cost read back off the primitive table (`_build_primitives` writes the same
    value into every heading row). Read back rather than passed around so `build_gated_solver` can
    keep reconstructing CostToGo's lattice from the CostToGo alone."""
    if solver.n_prim == N_PRIM_ARC:
        return 0.0
    return float(solver._prim_cost.numpy()[0, N_PRIM_ARC])


def build_gated_solver(ctg: CostToGo) -> EdgeGatedLatticeSolver:
    """An EdgeGatedLatticeSolver with CostToGo's own lattice, primitive tables asserted identical --
    trace_states reads ctg.solver's tables for every planner."""
    grid = ctg.grid
    solver = EdgeGatedLatticeSolver(
        grid.cell_size,
        grid.cells_y,
        grid.cells_x,
        n_theta=N_THETA,
        turn_radius=ctg.robot.min_turn_radius,
        step=STEP,
        pivot_cost=pivot_cost_of(ctg.solver),
        device=ctg.device,
    )
    for name in ("_prim_dr", "_prim_dc", "_prim_heading", "_prim_cost", "_sweep_dr", "_sweep_dc",
                 "_sweep_n"):
        assert np.array_equal(getattr(solver, name).numpy(), getattr(ctg.solver, name).numpy()), name
    return solver


def gated_solve(
    solver: EdgeGatedLatticeSolver,
    ctg: CostToGo,
    zeros: wp.array,
    tilt: wp.array,
    arc_error: wp.array,
    tau: float | Sequence[float] | np.ndarray,
) -> np.ndarray:
    solver.set_gate(arc_error, tau)
    return solver._record_solve(zeros, tilt, ctg._goal_rc, ctg.flatness_weight, False).numpy()


def lattice_state(x: float, y: float, yaw: float, ctg: CostToGo) -> tuple[int, int, int]:
    """World pose -> (row, col, heading bin), floor mapping as _goal_cell_kernel. The small epsilon
    keeps a start sitting exactly on a cell boundary from flooring into the previous cell."""
    grid = ctg.grid
    return (
        int(math.floor((y - grid.origin_y) / grid.cell_size + 1e-6)),
        int(math.floor((x - grid.origin_x) / grid.cell_size + 1e-6)),
        int(math.floor((yaw % (2.0 * math.pi)) / (2.0 * math.pi / N_THETA))) % N_THETA,
    )


def trace_states(
    ctg: CostToGo,
    V: np.ndarray,
    blocked: np.ndarray,
    tilt: np.ndarray,
    start_rct: tuple[int, int, int],
    arc_error: np.ndarray | None = None,
    tau: float | Sequence[float] | np.ndarray = math.inf,
    max_steps: int = 800,
) -> tuple[list[tuple[int, int, int]], list[int], float, bool]:
    """Follow the lattice's own policy from the start -> (states, primitive taken per step, billed
    m, reached). helhest_stack/scripts/bench_ramp_series.py's trace_states restated (scripts/ is not
    importable) with the arc gate added, so a gated plan traces the policy it was solved with --
    including a per-primitive `tau`, which the trace has to apply exactly as the kernel did or it
    would follow a policy through arcs the solve had pruned."""
    s = ctg.solver
    taus = primitive_taus(tau, s.n_prim)
    pdr, pdc = s._prim_dr.numpy(), s._prim_dc.numpy()
    pheading, pcost = s._prim_heading.numpy(), s._prim_cost.numpy()
    sdr, sdc, sn = s._sweep_dr.numpy(), s._sweep_dc.numpy(), s._sweep_n.numpy()
    tilt_weight = float(ctg.flatness_weight)
    gr, gc = (int(v) for v in ctg._goal_rc.numpy())
    ny, nx, _ = V.shape
    vcap = float(ctg._vcap)

    r, c, t = start_rct
    states, prims, arc_len = [(r, c, t)], [], 0.0
    for _ in range(max_steps):
        if abs(r - gr) <= 1 and abs(c - gc) <= 1:
            return states, prims, arc_len, True
        best_p, best_val = -1, np.inf
        for p in range(s.n_prim):
            if arc_error is not None and arc_error[r, c, t, p] > taus[p]:
                continue
            ns, ok, tsum = int(sn[t, p]), True, 0.0
            for si in range(ns):
                sr, sc = r + int(sdr[t, p, si]), c + int(sdc[t, p, si])
                if not (0 <= sr < ny and 0 <= sc < nx) or blocked[sr, sc, t] > 0.5:
                    ok = False
                    break
                tsum += tilt[sr, sc, t]
            if not ok:
                continue
            nr, nc, nt = r + int(pdr[t, p]), c + int(pdc[t, p]), int(pheading[t, p])
            if not (0 <= nr < ny and 0 <= nc < nx):
                continue
            arc = float(pcost[t, p]) * (1.0 + tilt_weight * tsum / ns if ns > 0 else 1.0)
            val = arc + V[nr, nc, nt]
            if val < best_val:
                best_val, best_p = val, p
        if best_p < 0 or best_val >= vcap * 0.9:
            return states, prims, arc_len, False
        arc_len += float(pcost[t, best_p])
        prims.append(best_p)
        r, c, t = r + int(pdr[t, best_p]), c + int(pdc[t, best_p]), int(pheading[t, best_p])
        states.append((r, c, t))
    return states, prims, arc_len, False


def arm_result(
    ctg: CostToGo,
    V: np.ndarray,
    arm_blocked: np.ndarray,
    tilt: np.ndarray,
    settle_blocked: np.ndarray,
    fields: dict[str, np.ndarray],
    start_rct: tuple[int, int, int],
    arc_error: np.ndarray | None = None,
    tau: float | Sequence[float] | np.ndarray = math.inf,
) -> dict:
    """V at the start, the traced path, and an audit of that path against the settle's `blocked`
    and against every error field in `fields` (the worst arc actually taken, `max_err`)."""
    V = np.minimum(V, ctg._vcap)
    states, prims, arc_len, reached = trace_states(
        ctg, V, arm_blocked, tilt, start_rct, arc_error, tau
    )
    rct = np.array(states)
    taken = rct[:-1]
    max_err = {
        name: float(field[taken[:, 0], taken[:, 1], taken[:, 2], prims].max()) if prims else math.nan
        for name, field in fields.items()
    }
    return dict(
        v_start=float(V[start_rct]),
        reachable=bool(V[start_rct] < ctg._vcap * 0.9),
        reached=reached,
        path_m=arc_len,
        n_poses=len(states),
        n_settle_bad=int((settle_blocked[rct[:, 0], rct[:, 1], rct[:, 2]] > 0.5).sum()),
        max_err=max_err,
        states=rct,
        prims=prims,
    )


if __name__ == "__main__":
    from feasibility.heightmap.create_uphill_series import uphill_ramp

    wp.init()
    if not wp.is_cuda_available():
        raise SystemExit("CUDA not available -- the lattice solves are Warp CUDA kernels.")
    terrain = uphill_ramp(30.0, x0=-3.0, nx=60, extent_y=2.0)  # 60 x 20 cells
    elev, grid = terrain.to_hstack("cuda")
    ctg = make_cost_to_go(grid)
    gated = build_gated_solver(ctg)
    ctg.compute(elev, (2.0, 0.0))
    zeros = wp.zeros_like(ctg.blocked)
    shape = (grid.cells_y, grid.cells_x, N_THETA, ctg.solver.n_prim)
    v_off = ctg.solver._record_solve(zeros, ctg.graded_tilt, ctg._goal_rc, ctg.flatness_weight,
                                     False).numpy().copy()
    open_gate = wp.zeros(shape, dtype=wp.float32, device=ctg.device)
    v_gated = gated_solve(gated, ctg, zeros, ctg.graded_tilt, open_gate, math.inf)
    assert np.array_equal(v_off, v_gated), "gated solver with an open gate differs from the stock one"
    print("[gate] open gate == stock LatticeValueSolver, bit for bit")
    start_rct = lattice_state(-2.0, 0.0, 0.0, ctg)
    shut = gated_solve(gated, ctg, zeros, ctg.graded_tilt, open_gate, -1.0)
    assert shut[start_rct] >= 0.5 * float(gated._inf), "a shut gate still reaches the goal"
    print("[gate] shut gate (every arc pruned) == goal unreachable")
    del ctg, gated

    # --- pivots: pivot_cost > 0 appends the two point turns, and a goal BEHIND the robot becomes
    # reachable in a 1.1 m corridor, where a min_turn_radius U-turn does not fit -----------------
    from feasibility.heightmap.heightmap_reader import HeightMapReader

    corridor = HeightMapReader.flat(xlim=(-2.0, 2.0), ylim=(-0.5, 0.5), cell=0.1)
    c_elev, c_grid = corridor.to_hstack("cuda")
    behind, pivot_ctg = {}, None
    for pc in (0.0, 0.15):
        p_ctg = make_cost_to_go(c_grid, pivot_cost=pc)
        n_expect = N_PRIM_ARC + (2 if pc > 0.0 else 0)
        assert p_ctg.solver.n_prim == n_expect, (pc, p_ctg.solver.n_prim, n_expect)
        assert math.isclose(pivot_cost_of(p_ctg.solver), pc, rel_tol=1e-6), pivot_cost_of(p_ctg.solver)
        build_gated_solver(p_ctg)  # asserts the gated lattice matches this one, pivots included
        p_ctg.compute(c_elev, (-1.0, 0.0))  # goal behind a robot that starts facing +x
        v_start = float(p_ctg.V.numpy()[lattice_state(1.0, 0.0, 0.0, p_ctg)])
        behind[pc] = (v_start, v_start < float(p_ctg._vcap) * 0.9)
        if pc > 0.0:
            pivot_ctg = p_ctg
        else:
            del p_ctg
    assert not behind[0.0][1], f"forward-only lattice reached a goal behind it: V {behind[0.0][0]}"
    assert behind[0.15][1], f"pivot lattice could not reach the goal behind it: V {behind[0.15][0]}"
    print(f"[pivot] goal behind: pivot_cost 0 -> unreachable (V {behind[0.0][0]:.3f}), "
          f"0.15 -> V {behind[0.15][0]:.3f}")

    # --- a PER-PRIMITIVE tau shutting only the two point turns takes that route away again, while
    # the five forward arcs stay open: the gate `planners.nn-gated-pivot` is built on -------------
    p_gated = build_gated_solver(pivot_ctg)
    p_zeros = wp.zeros_like(pivot_ctg.blocked)
    p_shape = (c_grid.cells_y, c_grid.cells_x, N_THETA, pivot_ctg.solver.n_prim)
    p_start = lattice_state(1.0, 0.0, 0.0, pivot_ctg)
    zero_err = wp.zeros(p_shape, dtype=wp.float32, device=pivot_ctg.device)  # every error exactly 0
    taus = np.full(pivot_ctg.solver.n_prim, math.inf, np.float32)
    taus[N_PRIM_ARC:] = -1.0  # ... so -1 prunes a pivot and inf keeps every arc
    v_pivots_shut = gated_solve(p_gated, pivot_ctg, p_zeros, pivot_ctg.graded_tilt, zero_err, taus)
    v_all_open = gated_solve(p_gated, pivot_ctg, p_zeros, pivot_ctg.graded_tilt, zero_err, math.inf)
    assert v_all_open[p_start] < float(pivot_ctg._vcap) * 0.9, "open per-primitive gate lost the route"
    assert v_pivots_shut[p_start] >= float(pivot_ctg._vcap) * 0.9, (
        f"gating only the pivots still reached the goal behind (V {v_pivots_shut[p_start]:.3f}) -- "
        "the per-primitive tau is not reaching the kernel"
    )
    # and the arcs really are untouched: gating pivots changes nothing a forward-only lattice could do
    fwd_only = np.full(pivot_ctg.solver.n_prim, math.inf, np.float32)
    assert np.array_equal(
        gated_solve(p_gated, pivot_ctg, p_zeros, pivot_ctg.graded_tilt, zero_err, fwd_only), v_all_open
    ), "an all-inf per-primitive tau differs from a scalar inf"
    print(f"[gate] per-primitive tau: arcs open + pivots shut -> goal behind unreachable "
          f"(V {v_pivots_shut[p_start]:.3f} vs {v_all_open[p_start]:.3f} open)")
    del pivot_ctg, p_gated
    print("all self-checks ok")
