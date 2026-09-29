"""Trial sampling for `mppi_learning`: each trial's window origin, entry twist and command,
chosen together by a strategy tailored to the map's category (design.md sections 5, 8 step 3).

`lattice_learning/spawn_sampling.py` places every trial for a 0.3 m arc at `V_NOM`. A 1 s MPPI
window travels up to ~5x further along a curve that is no constant-kappa arc, so here every
strategy measures along the window's own PATH: the ideal no-slip integration of the sampled wheel
speeds (`window_poses`, 0.05 s apart). Placement is geometry only and runs on thousands of
candidates per round, so the twin is not involved; the label's twin comes later.

A trial is: a window ORIGIN (the pose the patch is taken at), an ENTRY twist `(v, wz)` held for
`warmup_s` before it (the spawn is the origin backed out along that constant twist), and one
window's wheel-speed profile from `command.sample_window_commands`. The entry is mostly the window's
OWN first command: a knot is shared by the two windows that meet at it, so every rollout window but
the first starts with the command continuing. `EntrySpec.jump_frac` of the trials instead enter at
an independent twist -- window 0, whose WIDE/STRAIGHT knot 0 ignores the measured state.

The origin is NOMINAL: the entry twist integrated from the spawn with its yaw rate scaled by
`EntrySpec.yaw_ratio`, the share of MPPI's commanded yaw rate ostrich realizes under its
`CommandSpec.ostrich_yaw_gain` compensation (measured on flat ground). Ostrich's warm-up (terrain,
its stick-slip while turning) never reproduces it exactly, and placement only needs it to be close.
Every entry is inside the box after that compensation too (`entry_in_box`), as every window is.
The label and the patch must use ostrich's REALIZED state at the end of the warm-up: the
patch is sampled there, and `twin.run_twin` starts from it (pose, wheel speeds, body twist), as
MPPI starts every replan from the measured state.

What carries over from `lattice_learning`, unchanged in meaning:

* **Validity** -- helhest_stack's static settle feasible at the spawn (`settle.settle_feasible`)
  and the patch at the origin on mapped terrain. `require_endpoint` also demands it at the nominal
  window end; `uniform`/`targeted` default to it, the wall-driving strategies do not, and every
  row stores `endpoint_feasible` either way.
* **Interaction** -- plane-relative relief along the path against the contact plane at the origin,
  so a uniform slope scores ~0. `interact_dir` is +1 climbing / -1 driving down. The lattice
  measured contact points along an arc and rim sweeps only for pivots; a window has both arcs and
  spins, so ONE measure covers them: the terrain rising into any wheel CYLINDER (`rim_penetration`,
  which already contains the lookahead an arc needed) or dropping under the three contacts and the
  body centre. It is not numerically the lattice's `arc_relief`.
* **Strategies** -- `uniform`/`targeted`, `edge`, `ramp_up`/`ramp_down`, `rotate_in_place`, the
  same placement logic (distance to an edge, head-on to a face, exactly `interact_frac`
  interacting), stratified by `_fill` with the same fallbacks and `shortfall` accounting.

What differs:

* `rotate_in_place` draws the SPIN family only and enters at rest. The lattice entered already
  spinning, but a window spin turns up to ~220 deg, so a warm-up at that rate would sweep the same
  rim circle before the window does and reject exactly the fast spins.
* Ramp heading is fixed at MID-WINDOW (0.5 s), the ramps' commands are STRAIGHT/NARROW only
  (`straight_frac` and the rest), and a candidate whose wheels leave the face's width or its own
  surface anywhere from spawn to window end is rejected.
* Origins for every other strategy are spread with `sampling_bounds`, sized for the longest warm-up
  (`entry.v_max * warmup_s`) instead of the arc's `lead`.

Deliberately independent of `learning`, `grid_learning` and `planning` (design.md section 8's import
rule); `lattice_learning`'s geometry helpers are imported, not copied.

Usage (smoke test -- synthetic maps, no assets; `--device cuda:0` for the GPU settle):
    python src/feasibility/mppi_learning/spawn_sampling.py
"""
from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np
from helhest.engine import RobotParams

from feasibility.heightmap import HeightMapReader
from feasibility.heightmap.create_ramps import Ramp
from feasibility.lattice_learning.arc import integrate_twist
from feasibility.lattice_learning.patch import patch_overhangs
from feasibility.lattice_learning.patch import PatchSpec
from feasibility.lattice_learning.patch import WHEEL_CONTACTS_LOCAL
from feasibility.lattice_learning.settle import settle_batch
from feasibility.lattice_learning.settle import settle_feasible
from feasibility.lattice_learning.spawn_sampling import body_points
from feasibility.lattice_learning.spawn_sampling import cell_centers
from feasibility.lattice_learning.spawn_sampling import contact_plane
from feasibility.lattice_learning.spawn_sampling import edge_field
from feasibility.lattice_learning.spawn_sampling import EDGE_FACING_CONE_DEG
from feasibility.lattice_learning.spawn_sampling import EdgeField
from feasibility.lattice_learning.spawn_sampling import interaction_dir
from feasibility.lattice_learning.spawn_sampling import MAX_PROPOSAL_ROUNDS
from feasibility.lattice_learning.spawn_sampling import ramp_alone
from feasibility.lattice_learning.spawn_sampling import ramp_faces
from feasibility.lattice_learning.spawn_sampling import RAMP_SIDE_MARGIN
from feasibility.lattice_learning.spawn_sampling import RAMP_SURFACE_TOL
from feasibility.lattice_learning.spawn_sampling import RampFace
from feasibility.lattice_learning.spawn_sampling import rim_penetration
from feasibility.lattice_learning.spawn_sampling import sampling_bounds
from feasibility.lattice_learning.spawn_sampling import SETTLE_SLACK
from feasibility.lattice_learning.spawn_sampling import TOUCH_TOL
from feasibility.mppi_learning.command import CommandSpec
from feasibility.mppi_learning.command import HALF_TRACK
from feasibility.mppi_learning.command import MPPI_DT
from feasibility.mppi_learning.command import sample_window_commands
from feasibility.mppi_learning.command import SPIN
from feasibility.mppi_learning.command import WINDOW_S
from feasibility.mppi_learning.command import WINDOW_STEPS
from feasibility.mppi_learning.command import WHEEL_RADIUS
from feasibility.mppi_learning.command import wheels_to_twist

PATCH_SPEC = PatchSpec(x_max=3.0)  # design.md section 4: 24 x 36 cells at 0.125 m

PATH_SUBSTEPS = 2  # path poses per MPPI step (0.05 s): ~7 cm of travel at v_max, ~0.2 m of rim at the fastest spin
PROPOSAL_BATCH = 2048  # candidates per round; each costs a 21-pose rim sweep, ~8x a lattice arc's
MID_WINDOW = WINDOW_STEPS * PATH_SUBSTEPS // 2  # index into `window_poses`: t = 0.5 s
SPIN_SPEED_BINS = 5  # rotate_in_place stratifies spin speed x interaction, see spin_speed_bin
RAMP_MIN_TRAVEL = 0.3  # m, a face trial's origin is at least this far short of the crest going up


@dataclasses.dataclass(frozen=True)
class EntrySpec:
    """The twist commanded during the warm-up. `1 - jump_frac` of the trials continue into the
    window (the ideal twist of its first step's wheel speeds); the rest jump, drawn uniform on
    [0, v_max] x [-wz_max, wz_max] AND inside the command box (`entry_in_box`), so it is a state
    MPPI could have commanded and ostrich's compensated warm-up command stays in the box."""

    jump_frac: float = 0.2  # share of window-0-like trials, entry independent of the window
    v_max: float = 1.4  # m/s, = the command box's v_max
    wz_max: float = 2.0  # rad/s; the box caps it further, to 1.67 at v = 0.7 m/s with gain 1.15
    yaw_ratio: float = 0.56  # placement only: realized / MPPI-commanded yaw rate of the warm-up


@dataclasses.dataclass(frozen=True)
class TrialContext:
    """Everything a strategy shares, so each takes `(ctx, n, rng, **its_own_params)`."""

    terrain: HeightMapReader
    mu: float
    device: str
    interact_relief: float  # m, see the module docstring
    warmup_s: float = 0.3  # a multiple of both MPPI_DT and ostrich's dt; ostrich settles in ~0.2 s
    spec: PatchSpec = PATCH_SPEC
    command: CommandSpec = CommandSpec()
    entry: EntrySpec = EntrySpec()
    robot: RobotParams = dataclasses.field(default_factory=RobotParams)

    @property
    def max_warmup_distance(self) -> float:
        return self.entry.v_max * self.warmup_s


@dataclasses.dataclass
class Candidates:
    """m proposals: where each window starts and what it does. `spawn` is derived from the origin."""

    origin: np.ndarray  # [m, 3] (x, y, yaw) of the window start
    omega: np.ndarray  # [m, WINDOW_STEPS, 3] float32 wheel speeds (left, right, rear), rad/s
    family: np.ndarray  # [m] int8, `command.FAMILIES` index
    entry: np.ndarray  # [m, 2] (v, wz) held for the warm-up
    spawn: np.ndarray  # [m, 3]
    ramp_deg: np.ndarray  # [m] float32, NaN off a face
    ramp_s: np.ndarray  # [m] float32, NaN off a face

    def __len__(self) -> int:
        return len(self.origin)

    def take(self, idx: np.ndarray) -> "Candidates":
        return Candidates(*(getattr(self, f.name)[idx] for f in dataclasses.fields(self)))


def make_candidates(
    ctx: TrialContext, origin: np.ndarray, omega: np.ndarray, family: np.ndarray, entry: np.ndarray,
    ramp_deg: np.ndarray | None = None, ramp_s: np.ndarray | None = None,
) -> Candidates:
    """`omega` is `command.sample_window_commands`' [WINDOW_STEPS, m, 3]; stored row-major."""
    nan = np.full(len(origin), np.nan, dtype=np.float32)
    spawn = integrate_twist(origin, entry[:, 0], ctx.entry.yaw_ratio * entry[:, 1], -ctx.warmup_s)
    return Candidates(
        origin=origin, omega=np.ascontiguousarray(omega.transpose(1, 0, 2)), family=family,
        entry=entry, spawn=spawn, ramp_deg=nan if ramp_deg is None else ramp_deg,
        ramp_s=nan if ramp_s is None else ramp_s,
    )


@dataclasses.dataclass
class WindowBatch:
    """One map's trials. Row fields mirror `lattice_learning`'s `SpawnBatch` where they mean the same."""

    pose: np.ndarray  # [n, 3] float64 spawn (x, y, yaw) -- where ostrich and the twin start
    origin: np.ndarray  # [n, 3] float64 NOMINAL window start (ideal kinematics); label/patch use ostrich's realized one
    omega: np.ndarray  # [n, WINDOW_STEPS, 3] float32 wheel speeds on the MPPI grid
    family: np.ndarray  # [n] int8
    entry: np.ndarray  # [n, 2] float32 (v, wz)
    relief: np.ndarray  # [n] float32 m, see the module docstring
    interact_dir: np.ndarray  # [n] int8 +1 up / -1 down / 0 not interacting
    endpoint_feasible: np.ndarray  # [n] bool, static settle at the nominal window end
    targeted: np.ndarray  # [n] bool, whether the strategy stratified by interaction
    strategy: np.ndarray  # [n] str
    shortfall: int  # stratified trials requested but substituted from a fallback stratum
    proposal_interact_rate: float  # natural share of interacting proposals, NaN once mixed
    ramp_deg: np.ndarray  # [n] float32, NaN off a face
    ramp_s: np.ndarray  # [n] float32 m, front axle along the face from its entry point


def concat_batches(batches: list[WindowBatch], rng: np.random.Generator) -> WindowBatch:
    """Several strategies' trials for one map, in a random row order; `shortfall` sums."""
    batches = [b for b in batches if len(b.pose)]
    if len(batches) == 1:
        return batches[0]
    order = rng.permutation(sum(len(b.pose) for b in batches))
    rows = {
        f.name: np.concatenate([getattr(b, f.name) for b in batches])[order]
        for f in dataclasses.fields(WindowBatch)
        if f.name not in ("shortfall", "proposal_interact_rate")
    }
    return WindowBatch(**rows, shortfall=sum(b.shortfall for b in batches), proposal_interact_rate=float("nan"))


# --- paths and relief -------------------------------------------------------------------------


def window_poses(origin: np.ndarray, omega: np.ndarray) -> list[np.ndarray]:
    """[WINDOW_STEPS * PATH_SUBSTEPS + 1] poses [m, 3] along the window, from `origin` [m, 3],
    each MPPI step's wheel speeds (`omega` [m, WINDOW_STEPS, >=2]) held as an ideal no-slip twist."""
    v, wz = wheels_to_twist(omega)  # [m, S]
    poses, pose = [origin], origin
    for step in range(WINDOW_STEPS):
        for _ in range(PATH_SUBSTEPS):
            pose = integrate_twist(pose, v[:, step], wz[:, step], MPPI_DT / PATH_SUBSTEPS)
            poses.append(pose)
    return poses


def warmup_poses(ctx: TrialContext, cand: Candidates) -> list[np.ndarray]:
    """Poses [m, 3] from the spawn to the origin along the entry twist, ~5 cm apart at v_max."""
    times = np.linspace(0.0, ctx.warmup_s, int(round(ctx.warmup_s / (MPPI_DT / PATH_SUBSTEPS))) + 1)
    wz = ctx.entry.yaw_ratio * cand.entry[:, 1]
    return [integrate_twist(cand.spawn, cand.entry[:, 0], wz, t) for t in times]


def path_relief_signed(
    terrain: HeightMapReader, origin: np.ndarray, poses: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """([m] m, [m] int8) -- max relief along `poses` against the contact plane at `origin`, and +1/-1
    for whether it is above (a wheel cylinder pushed into) or below (a drop under the three contacts
    or the body centre) that plane. The pivot twin of `lattice_learning`'s `spin_relief_signed`,
    over an arbitrary path."""
    above = np.maximum(rim_penetration(terrain, origin, poses), 0.0)
    local = np.vstack([WHEEL_CONTACTS_LOCAL, [0.0, 0.0]])
    plane = contact_plane(terrain, origin)
    below = np.zeros(len(origin))
    for pose in poses:
        px, py = body_points(pose, local)
        below = np.maximum(below, (plane(px, py) - np.asarray(terrain.sample(px, py))).max(axis=1))
    return np.maximum(above, below), np.where(above >= below, 1, -1).astype(np.int8)


def trials_feasible(ctx: TrialContext, cand: Candidates) -> tuple[np.ndarray, np.ndarray]:
    """([m] bool, [m] bool) -- (settle feasible at the spawn and the patch at the origin on the map,
    settle feasible at the nominal window end)."""
    k = len(cand)
    end = window_poses(cand.origin, cand.omega)[-1]
    derived, residual, clearance = settle_batch(ctx.terrain, np.concatenate([cand.spawn, end]), ctx.mu, ctx.device)
    feasible = settle_feasible(derived, residual, clearance, ctx.robot)
    return feasible[:k] & ~patch_overhangs(ctx.terrain, cand.origin, ctx.spec), feasible[k:]


def entry_in_box(entry: np.ndarray, command: CommandSpec) -> np.ndarray:
    """[m] bool -- both ideal wheel speeds of the twists `entry` [m, 2], as ostrich is commanded
    them (yaw rate * `ostrich_yaw_gain`), inside [wmin, wmax]; MPPI's own wheels then are too."""
    v, wz = entry[:, 0], entry[:, 1] * command.ostrich_yaw_gain
    wheels = np.stack([v - wz * HALF_TRACK, v + wz * HALF_TRACK], axis=-1) / WHEEL_RADIUS
    tol = 1e-5  # the continuation entries come from float32 windows clamped to the box edge
    return ((wheels >= command.wmin - tol) & (wheels <= command.wmax + tol)).all(axis=-1)


def sample_entry(ctx: TrialContext, omega: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """[m, 2] (v, wz) for the windows `omega` [WINDOW_STEPS, m, 3]: each window's own first
    command, except `jump_frac` of them drawn by `sample_jump_entry`."""
    m = omega.shape[1]
    entry = np.stack(wheels_to_twist(omega[0].astype(np.float64)), axis=-1)
    jump = rng.random(m) < ctx.entry.jump_frac
    entry[jump] = sample_jump_entry(ctx, int(jump.sum()), rng)
    return entry


def sample_jump_entry(ctx: TrialContext, m: int, rng: np.random.Generator) -> np.ndarray:
    """[m, 2] (v, wz), uniform over the part of [0, v_max] x [-wz_max, wz_max] the command box
    allows, by rejection."""
    out = np.empty((0, 2))
    while len(out) < m:
        draw = np.column_stack(
            [rng.uniform(0.0, ctx.entry.v_max, 2 * m), rng.uniform(-ctx.entry.wz_max, ctx.entry.wz_max, 2 * m)]
        )
        out = np.concatenate([out, draw[entry_in_box(draw, ctx.command)]])
    return out[:m]


# --- the shared stratified fill ---------------------------------------------------------------


@dataclasses.dataclass
class _Filled:
    cand: Candidates
    relief: np.ndarray
    sign: np.ndarray
    end_ok: np.ndarray
    shortfall: int
    rate: float


def _fill(
    ctx: TrialContext,
    need: dict[int, int],
    fallback: list[tuple[int, int]],
    propose: Callable[[int], Candidates],
    label: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    admit: Callable[[Candidates], np.ndarray] | None = None,
    substratum: tuple[Callable[[Candidates], np.ndarray], int] | None = None,
    require_endpoint: bool,
) -> _Filled:
    """The loop every stratified strategy shares (`lattice_learning`'s `stratified_fill`, over
    window paths). `propose(m)` -> candidates; `admit(cand)` -> [m] bool geometry pre-filter;
    `label(relief, sign)` -> [m] stratum; `need` maps stratum -> trials wanted, and each (short,
    into) pair of `fallback` moves an unfilled stratum's deficit after MAX_PROPOSAL_ROUNDS.
    `substratum` (fn, k) crosses the label with fn(cand) in [0, k): the stratum becomes
    `label * k + fn(cand)`, and `need`/`fallback` are keyed on that."""
    need = dict(need)
    n = sum(need.values())
    got: dict[int, list[tuple]] = {k: [] for k in need}
    counts = {k: 0 for k in need}
    n_proposed = n_interacting = 0

    def fill(rounds: int) -> None:
        nonlocal n_proposed, n_interacting
        for _ in range(rounds):
            missing = {k: need[k] - counts[k] for k in need}
            if all(m <= 0 for m in missing.values()):
                return
            cand = propose(PROPOSAL_BATCH)
            if admit is not None:
                cand = cand.take(np.flatnonzero(admit(cand)))
            if not len(cand):
                continue
            relief, sign = path_relief_signed(ctx.terrain, cand.origin, window_poses(cand.origin, cand.omega))
            labels = label(relief, sign)
            if substratum is not None:
                labels = labels * substratum[1] + substratum[0](cand)
            n_proposed += len(cand)
            n_interacting += int((relief > ctx.interact_relief).sum())

            picked = [
                (key, np.flatnonzero(labels == key)[: int(SETTLE_SLACK * miss) + 16])
                for key, miss in missing.items() if miss > 0
            ]
            sel = np.concatenate([idx for _, idx in picked])
            if not len(sel):
                continue
            base_ok, end_ok = trials_feasible(ctx, cand.take(sel))
            ok = base_ok & end_ok if require_endpoint else base_ok
            start = 0
            for key, idx in picked:
                part = slice(start, start + len(idx))
                start += len(idx)
                keep = np.flatnonzero(ok[part])[: need[key] - counts[key]]
                rows = idx[keep]
                got[key].append((cand.take(rows), relief[rows], sign[rows], end_ok[part][keep]))
                counts[key] += len(rows)

    fill(MAX_PROPOSAL_ROUNDS)
    shortfall = 0
    for short, into in fallback:
        if counts[short] < need[short]:
            deficit = need[short] - counts[short]
            shortfall += deficit
            need[short], need[into] = counts[short], need[into] + deficit
            fill(MAX_PROPOSAL_ROUNDS)
    total = sum(counts.values())
    if total < n:
        raise ValueError(
            f"only found {total}/{n} settle-feasible trials after {n_proposed} proposals -- the "
            "map has too little drivable ground where this strategy proposes"
        )
    parts = [p for v in got.values() for p in v]
    cand = Candidates(*(np.concatenate([getattr(p[0], f.name) for p in parts]) for f in dataclasses.fields(Candidates)))
    return _Filled(
        cand, np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts]),
        np.concatenate([p[3] for p in parts]), shortfall, n_interacting / max(n_proposed, 1),
    )


def _to_batch(
    ctx: TrialContext, filled: _Filled, rng: np.random.Generator, *, strategy: str, targeted: bool
) -> WindowBatch:
    """The fill's rows as a WindowBatch, shuffled so chunk order never tracks the stratum."""
    c, n = filled.cand, len(filled.cand)
    order = rng.permutation(n)
    return WindowBatch(
        pose=c.spawn[order], origin=c.origin[order], omega=c.omega[order], family=c.family[order],
        entry=c.entry[order].astype(np.float32), relief=filled.relief[order].astype(np.float32),
        interact_dir=interaction_dir(filled.relief, filled.sign, ctx.interact_relief)[order],
        endpoint_feasible=filled.end_ok[order], targeted=np.full(n, targeted),
        strategy=np.full(n, strategy), shortfall=filled.shortfall,
        proposal_interact_rate=filled.rate, ramp_deg=c.ramp_deg[order], ramp_s=c.ramp_s[order],
    )


def _check_frac(name: str, value: float | None, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if value is None or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]{' or None' if optional else ''}, got {value}")


# --- strategies -------------------------------------------------------------------------------


def sample_trials(
    ctx: TrialContext, n: int, rng: np.random.Generator, *, interact_frac: float | None,
    require_endpoint: bool = True,
) -> WindowBatch:
    """`n` trials, origins uniform over the map at a uniform heading, commands from `ctx.command`.
    `interact_frac` None samples plainly (`uniform`); a float stratifies by `relief >
    interact_relief` (`targeted`)."""
    _check_frac("interact_frac", interact_frac, optional=True)
    x_lo, x_hi, y_lo, y_hi = sampling_bounds(ctx.terrain, ctx.spec, ctx.max_warmup_distance)

    def propose(m: int) -> Candidates:
        omega, family = sample_window_commands(m, rng, ctx.command)
        origin = np.column_stack([rng.uniform(x_lo, x_hi, m), rng.uniform(y_lo, y_hi, m), rng.uniform(0.0, 2.0 * np.pi, m)])
        return make_candidates(ctx, origin, omega, family, sample_entry(ctx, omega, rng))

    if interact_frac is None:
        need, fallback = {0: n}, []
        label = lambda relief, sign: np.zeros(len(relief), dtype=int)  # noqa: E731
    else:
        n_int = round(n * interact_frac)
        need, fallback = {1: n_int, 0: n - n_int}, [(1, 0)]
        label = lambda relief, sign: (relief > ctx.interact_relief).astype(int)  # noqa: E731
    filled = _fill(ctx, need, fallback, propose, label, require_endpoint=require_endpoint)
    return _to_batch(
        ctx, filled, rng, strategy="targeted" if interact_frac is not None else "uniform",
        targeted=interact_frac is not None,
    )


def edge_cells_in_reach(ctx: TrialContext, cell_reach: float, field: EdgeField | None = None) -> np.ndarray:
    """Flat indices of the cells inside `sampling_bounds` within `cell_reach` of a height edge --
    where `edge`/`rotate_in_place` origins are drawn. Empty when every feature lies outside."""
    field = edge_field(ctx.terrain) if field is None else field
    x_lo, x_hi, y_lo, y_hi = sampling_bounds(ctx.terrain, ctx.spec, ctx.max_warmup_distance)
    CX, CY = cell_centers(ctx.terrain)
    return np.flatnonzero((field.dist <= cell_reach) & (CX >= x_lo) & (CX <= x_hi) & (CY >= y_lo) & (CY <= y_hi))


def rotate_cell_reach(ctx: TrialContext, band: float) -> float:
    """m, `rotate_in_place`'s `cell_reach`: the farthest a wheel rim reaches from the body origin,
    plus `band`."""
    return float(np.hypot(WHEEL_CONTACTS_LOCAL[:, 0], WHEEL_CONTACTS_LOCAL[:, 1]).max() + ctx.robot.wheel_radius) + band


def spin_speed_bin(command: CommandSpec, omega: np.ndarray, k: int) -> np.ndarray:
    """[m] which of `k` equal-width bins of the SPIN prior's |wheel speed| (MPPI's command, over
    [spin_min, wmax / ostrich_yaw_gain]) each window `omega` [m, WINDOW_STEPS, 3] falls in."""
    lo, hi = command.spin_min, command.wmax / command.ostrich_yaw_gain
    frac = (np.abs(omega[:, 0, 1].astype(np.float64)) - lo) / (hi - lo)
    return np.clip((frac * k).astype(int), 0, k - 1)


def _balanced_need(n: int, n_int: int | None, k: int, rng: np.random.Generator) -> tuple[dict[int, int], list]:
    """`need`/`fallback` for `_fill` over (interacting, speed bin) strata keyed `label * k + bin`:
    the `n` rows spread evenly over the `k` bins, exactly `n_int` of them interacting, dealt at
    random so neither the remainders nor the interacting rows favour a bin. A bin that cannot fill
    its interacting rows moves them into ITS OWN non-interacting ones, so the speed marginal stays
    uniform (a spin too slow to reach anything in one window does not interact, and is recorded
    in `shortfall`). `n_int` None: bins only, every label 0."""
    bins = rng.permutation(rng.permutation(k)[np.arange(n) % k])  # the n % k leftovers go to random bins
    inter = rng.permutation(np.arange(n) < (n_int or 0)).astype(int)
    keys, counts = np.unique(inter * k + bins, return_counts=True)
    need = {int(key): int(c) for key, c in zip(keys, counts)}
    fallback = [(k + b, b) for b in range(k) if k + b in need] if n_int is not None else []
    for _, into in fallback:
        need.setdefault(into, 0)
    return need, fallback


def _sample_near_edges(
    ctx: TrialContext, n: int, rng: np.random.Generator, *, strategy: str, interact_frac: float | None,
    cell_reach: float, facing_frac: float, down_frac: float | None, command: CommandSpec,
    at_rest: bool, min_clearance: float | None, require_endpoint: bool, speed_bins: int = 1,
) -> WindowBatch:
    """`edge` and `rotate_in_place`: origins on cells within `cell_reach` of a height edge (found
    from the heightmap alone), heading at the nearest edge +- EDGE_FACING_CONE_DEG with
    probability `facing_frac`. `down_frac` None stratifies binary (interacting or not); a float
    splits the interacting share into driving-down and climbing-up, as `lattice_learning`'s `edge`.
    `min_clearance` rejects candidates whose wheels come within it of the terrain above the contact
    plane anywhere in the warm-up, so the feature is met, if at all, in the window. `speed_bins` > 1
    (SPIN-only commands, `down_frac` None) also stratifies the spin speed, `_balanced_need`."""
    _check_frac("interact_frac", interact_frac, optional=True)
    if speed_bins > 1 and (down_frac is not None or command.mix[:3] != (0.0, 0.0, 0.0)):
        raise ValueError("speed_bins > 1 needs SPIN-only commands and down_frac None")
    field = edge_field(ctx.terrain)
    CX, CY = cell_centers(ctx.terrain)
    cells = edge_cells_in_reach(ctx, cell_reach, field)
    if len(cells) == 0:
        if strategy == "rotate_in_place":
            raise ValueError("no map cell lies within reach of a height edge for sample_rotate_in_place_trials "
                             "(features outside sampling_bounds' square -- regenerate the maps with "
                             "create_maps_for_lattice_learning.py --center-limit 2.38)")
        rest = sample_trials(ctx, n, rng, interact_frac=interact_frac, require_endpoint=require_endpoint)
        return dataclasses.replace(rest, shortfall=n, strategy=np.full(n, strategy))

    cone = np.radians(EDGE_FACING_CONE_DEG)
    dist, near_x, near_y = field.dist.ravel(), field.nearest_x.ravel(), field.nearest_y.ravel()

    def propose(m: int) -> Candidates:
        cell = cells[rng.integers(0, len(cells), m)]
        x = CX.ravel()[cell] + rng.uniform(-0.5, 0.5, m) * ctx.terrain.cell
        y = CY.ravel()[cell] + rng.uniform(-0.5, 0.5, m) * ctx.terrain.cell
        to_edge = np.arctan2(near_y[cell] - y, near_x[cell] - x)
        facing = (rng.uniform(size=m) < facing_frac) & (dist[cell] > 0.0)
        yaw = np.where(facing, to_edge + rng.uniform(-cone, cone, m), rng.uniform(0.0, 2.0 * np.pi, m))
        omega, family = sample_window_commands(m, rng, command)
        entry = np.zeros((m, 2)) if at_rest else sample_entry(ctx, omega, rng)
        return make_candidates(ctx, np.column_stack([x, y, yaw]), omega, family, entry)

    def admit(cand: Candidates) -> np.ndarray:
        clear = rim_penetration(ctx.terrain, cand.origin, warmup_poses(ctx, cand), margin=min_clearance)
        return clear <= TOUCH_TOL

    substratum = None
    if speed_bins > 1:
        n_int = None if interact_frac is None else round(n * interact_frac)
        need, fallback = _balanced_need(n, n_int, speed_bins, rng)
        substratum = (lambda cand: spin_speed_bin(command, cand.omega, speed_bins), speed_bins)
    elif interact_frac is None:
        need, fallback = {0: n}, []
    elif down_frac is None:
        n_int = round(n * interact_frac)
        need, fallback = {1: n_int, 0: n - n_int}, [(1, 0)]
    else:
        n_int = round(n * interact_frac)
        n_down = round(n_int * down_frac)
        need, fallback = {-1: n_down, 1: n_int - n_down, 0: n - n_int}, [(-1, 1), (1, 0)]

    def label(relief: np.ndarray, sign: np.ndarray) -> np.ndarray:
        if interact_frac is None:
            return np.zeros(len(relief), dtype=int)
        if down_frac is None:
            return (relief > ctx.interact_relief).astype(int)
        return interaction_dir(relief, sign, ctx.interact_relief).astype(int)

    filled = _fill(
        ctx, need, fallback, propose, label, require_endpoint=require_endpoint,
        admit=admit if min_clearance is not None else None, substratum=substratum,
    )
    return _to_batch(ctx, filled, rng, strategy=strategy, targeted=interact_frac is not None)


def sample_edge_trials(
    ctx: TrialContext, n: int, rng: np.random.Generator, *, interact_frac: float | None, band: float,
    facing_frac: float, down_frac: float, require_endpoint: bool = False,
) -> WindowBatch:
    """`n` trials whose window origins lie within `band` of a height edge; exactly
    `round(n * interact_frac)` interact, `down_frac` of them driving down. A map with no edge cell in
    reach falls back to `sample_trials` (all of `n` counted in `shortfall`)."""
    _check_frac("facing_frac", facing_frac)
    _check_frac("down_frac", down_frac)
    return _sample_near_edges(
        ctx, n, rng, strategy="edge", interact_frac=interact_frac, cell_reach=band,
        facing_frac=facing_frac, down_frac=down_frac, command=ctx.command, at_rest=False,
        min_clearance=None, require_endpoint=require_endpoint,
    )


def sample_rotate_in_place_trials(
    ctx: TrialContext, n: int, rng: np.random.Generator, *, interact_frac: float | None, band: float,
    min_clearance: float, require_endpoint: bool = False,
) -> WindowBatch:
    """`n` spin windows (`command.SPIN` family only, entered at rest) placed so a wheel can sweep into
    a pole or wall: origins within `band` of the farthest a wheel rim reaches from the body origin
    of a height edge. No wheel comes within `min_clearance` of the terrain above the contact plane
    at the spawn, and exactly `round(n * interact_frac)` sweep a wheel into it during the window.
    The spin speed is stratified too (SPIN_SPEED_BINS equal bins, interacting rows dealt across them
    at random), or
    the interacting half would fill with fast spins -- a slow one rarely reaches anything -- and
    speed would stand in for contact in the data; bins a slow spin cannot interact in give up their
    interacting rows to their own non-interacting ones (`shortfall`)."""
    if min_clearance < 0.0 or band <= 0.0:
        raise ValueError(f"need min_clearance >= 0 and band > 0, got {min_clearance}, {band}")
    return _sample_near_edges(
        ctx, n, rng, strategy="rotate_in_place", interact_frac=interact_frac, cell_reach=rotate_cell_reach(ctx, band),
        facing_frac=0.0, down_frac=None, command=dataclasses.replace(ctx.command, mix=(0.0, 0.0, 0.0, 1.0)),
        at_rest=True, min_clearance=min_clearance, require_endpoint=require_endpoint,
        speed_bins=SPIN_SPEED_BINS,
    )


PLATFORM_MARGIN = 0.1  # m, heading jitter and curvature swing the rear wheel off the face axis


def platform_behind(ctx: TrialContext) -> float:
    """m of plateau a `ramp_down` trial occupies BEHIND its origin's front axle: the warm-up at the
    fastest entry, the wheelbase back to the rear axle, the rear wheel's back edge and a margin."""
    return ctx.max_warmup_distance + float(ctx.robot.rear_offset) + float(ctx.robot.wheel_radius) + PLATFORM_MARGIN


def required_platform_length(ctx: TrialContext) -> float:
    """m -- the shortest plateau a `ramp_down` face can be sampled on: the whole robot on it, any
    entry speed, with the origin's front wheel at the crest (front axle `wheel_radius` short of it).
    A longer plateau lets the origin start further back (`_sample_face_trials`), up to
    `reach + wheel_radius` before the crest -- where the front wheel just reaches it at window end,
    which takes `reach + wheel_radius + platform_behind` (3.37 m at the defaults)."""
    return float(ctx.robot.wheel_radius) + platform_behind(ctx)


def _sample_face_trials(
    ctx: TrialContext, faces: list[RampFace], n: int, rng: np.random.Generator, *, direction: int,
    straight_frac: float, yaw_jitter_deg: float, fallback_interact_frac: float | None,
    require_endpoint: bool = False,
) -> WindowBatch:
    """The ramp sampler for both directions: `direction` +1 drives UP a face from its foot, -1 DOWN it
    from its crest, `s` measured from that entry point along the direction of travel. The origin's
    front axle sits at `s` in [-(reach + wheel_radius), run - RAMP_MIN_TRAVEL] (never before the
    event `-direction * wheel_radius`), anywhere across the top width, the heading at MID-WINDOW within
    +- `yaw_jitter_deg` of the face axis. The lattice's `s_event` argument carries over: going down a
    short face it is what puts the drop inside the window.

    Going DOWN, the lower bound is also capped per face by its plateau: the robot and its warm-up
    must fit on it (`platform_behind`), so on a plateau P the origin starts at most
    `P - platform_behind` before the crest. On the lattice maps (P 2.1-3.0 m) that is 0.5-1.4 m
    instead of the full 1.75 m, so a fast approach reaches the crest early in the window; every
    entry speed stays equally likely at every start."""
    _check_frac("straight_frac", straight_frac)
    r = float(ctx.robot.wheel_radius)
    jitter = np.radians(yaw_jitter_deg)
    command = dataclasses.replace(ctx.command, mix=(0.0, straight_frac, 1.0 - straight_frac, 0.0))
    s_min = -ctx.command.v_max * WINDOW_S - r
    s_event = -direction * r
    surfaces = {f.ramp_index: ramp_alone(ctx.terrain, f.ramp) for f in faces}
    foot_x, foot_y, face_yaw, face_run, face_half_w, face_slope = (
        np.array([getattr(f, k) for f in faces]) for k in ("foot_x", "foot_y", "yaw", "run", "half_width", "slope_deg")
    )
    face_s_min = np.full(len(faces), s_min)
    if direction < 0:  # the robot and its warm-up stand on the plateau behind the crest
        plateau = np.array([f.ramp.plateau for f in faces])
        face_s_min = np.maximum(s_min, np.minimum(platform_behind(ctx) - plateau, s_event))
    entry_frac = 0.0 if direction > 0 else 1.0  # entry point: the foot going up, the crest going down
    turn = 0.0 if direction > 0 else np.pi

    got: list[Candidates] = []
    count = 0
    for _ in range(MAX_PROPOSAL_ROUNDS if faces else 0):
        if count >= n:
            break
        m = PROPOSAL_BATCH
        face_idx = rng.integers(0, len(faces), m)
        fx, fy, fyaw, run = foot_x[face_idx], foot_y[face_idx], face_yaw[face_idx], face_run[face_idx]
        half_w = face_half_w[face_idx]
        c, s = np.cos(fyaw), np.sin(fyaw)  # the face's uphill axis
        ex, ey = fx + entry_frac * run * c, fy + entry_frac * run * s
        travel = fyaw + turn

        s0 = rng.uniform(face_s_min[face_idx], np.maximum(run - RAMP_MIN_TRAVEL, s_event))
        t0 = rng.uniform(-half_w, half_w)
        omega, family = sample_window_commands(m, rng, command)
        # heading at mid-window: the path's own turning in the body frame, read off a yaw-0 origin
        mid_yaw = window_poses(np.zeros((m, 3)), omega.transpose(1, 0, 2))[MID_WINDOW][:, 2]
        yaw0 = travel + rng.uniform(-jitter, jitter, m) - mid_yaw
        tc, ts = np.cos(travel), np.sin(travel)
        origin = np.column_stack([ex + s0 * tc - t0 * ts, ey + s0 * ts + t0 * tc, yaw0])
        cand = make_candidates(ctx, origin, omega, family, sample_entry(ctx, omega, rng), face_slope[face_idx], s0.astype(np.float32))

        # every wheel contact from spawn to window end inside the top width and on this ramp's surface
        on_face = np.ones(m, dtype=bool)
        for pose in warmup_poses(ctx, cand) + window_poses(cand.origin, cand.omega):
            wx, wy = body_points(pose, WHEEL_CONTACTS_LOCAL)  # [m, 3]
            lateral = -s[:, None] * (wx - fx[:, None]) + c[:, None] * (wy - fy[:, None])
            on_face &= (np.abs(lateral) <= half_w[:, None] - RAMP_SIDE_MARGIN).all(axis=1)
            z = np.asarray(ctx.terrain.sample(wx, wy), dtype=np.float64)
            for j, face in enumerate(faces):
                rows = np.flatnonzero((face_idx == j) & on_face)
                surface = np.asarray(surfaces[face.ramp_index].sample(wx[rows], wy[rows]))
                on_face[rows] &= (np.abs(z[rows] - surface) <= RAMP_SURFACE_TOL).all(axis=1)

        idx = np.flatnonzero(on_face)[: int(SETTLE_SLACK * (n - count)) + 16]
        if not len(idx):
            continue
        sub = cand.take(idx)
        base_ok, end_ok = trials_feasible(ctx, sub)
        ok = base_ok & end_ok if require_endpoint else base_ok
        got.append(sub.take(np.flatnonzero(ok)[: n - count]))
        count += len(got[-1])

    fields = dataclasses.fields(Candidates)
    cand = Candidates(*(np.concatenate([getattr(g, f.name) for g in got]) for f in fields)) if got else None
    shortfall = n - (len(cand) if cand is not None else 0)
    parts: list[WindowBatch] = []
    if cand is not None:
        relief, sign = path_relief_signed(ctx.terrain, cand.origin, window_poses(cand.origin, cand.omega))
        end_ok = trials_feasible(ctx, cand)[1]
        parts.append(_to_batch(
            ctx, _Filled(cand, relief, sign, end_ok, 0, float("nan")), rng,
            strategy="ramp_up" if direction > 0 else "ramp_down", targeted=True,
        ))
    if shortfall:
        rest = sample_trials(ctx, shortfall, rng, interact_frac=fallback_interact_frac, require_endpoint=require_endpoint)
        parts.append(dataclasses.replace(rest, strategy=np.full(shortfall, "ramp_up" if direction > 0 else "ramp_down"), targeted=np.ones(shortfall, dtype=bool)))
    batch = concat_batches(parts, rng)
    return dataclasses.replace(batch, shortfall=shortfall, proposal_interact_rate=float("nan"))


def sample_ramp_up_trials(ctx: TrialContext, faces: list[RampFace], n: int, rng: np.random.Generator, **kw: object) -> WindowBatch:
    """`n` trials driving head-on UP the given faces (`_sample_face_trials`); any remainder the faces
    cannot supply is filled by `sample_trials` and counted in `shortfall`."""
    return _sample_face_trials(ctx, faces, n, rng, direction=1, **kw)


def sample_ramp_down_trials(ctx: TrialContext, faces: list[RampFace], n: int, rng: np.random.Generator, **kw: object) -> WindowBatch:
    """The mirror: DOWN the faces from their crest, drops included. The earliest spawn stands on a
    plateau that must be `required_platform_length(ctx)` long."""
    return _sample_face_trials(ctx, faces, n, rng, direction=-1, **kw)


if __name__ == "__main__":
    import argparse
    import time

    import warp as wp

    from feasibility.heightmap.create_ramps import ramp_layer
    from feasibility.mppi_learning.command import FAMILIES

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    wp.init()
    rng = np.random.default_rng(0)
    xs = np.arange(-8.0, 8.0, 0.05) + 0.025
    X, Y = np.meshgrid(xs, xs)

    def context(terrain: HeightMapReader) -> TrialContext:
        return TrialContext(terrain, mu=0.8, device=args.device, interact_relief=0.05)

    # Section 4's lateral-extent assert, from RobotParams and the sampling box rather than the
    # design's estimate: every wheel rim point along any sampled window, in the origin's body frame,
    # stays inside the patch.
    from feasibility.lattice_learning.spawn_sampling import wheel_footprint

    probe = context(HeightMapReader.flat(xlim=(-8.0, 8.0), ylim=(-8.0, 8.0)))
    om, fam = sample_window_commands(20_000, np.random.default_rng(1), probe.command)
    local, _ = wheel_footprint()
    lo = np.full(2, np.inf)
    hi = np.full(2, -np.inf)
    for pose in window_poses(np.zeros((20_000, 3)), om.transpose(1, 0, 2)):
        px, py = body_points(pose, local)
        lo = np.minimum(lo, [px.min(), py.min()])
        hi = np.maximum(hi, [px.max(), py.max()])
    assert lo[0] >= PATCH_SPEC.x_min and hi[0] <= PATCH_SPEC.x_max, (lo, hi)
    assert lo[1] >= PATCH_SPEC.y_min and hi[1] <= PATCH_SPEC.y_max, (lo, hi)
    print(f"patch extent: window rims span x [{lo[0]:.2f}, {hi[0]:.2f}] y [{lo[1]:.2f}, {hi[1]:.2f}] "
          f"inside x [{PATCH_SPEC.x_min}, {PATCH_SPEC.x_max}] y [{PATCH_SPEC.y_min}, {PATCH_SPEC.y_max}]")

    # Entries are twists the command box can produce, spread over it.
    ent = sample_jump_entry(probe, 20_000, np.random.default_rng(2))
    assert entry_in_box(ent, probe.command).all() and ent[:, 0].min() >= 0.0
    assert ent[:, 0].max() > 0.9 * probe.entry.v_max and np.abs(ent[:, 1]).max() > 0.9 * 1.4 / HALF_TRACK / 2
    # ... and the rest continue into their window: its first step's twist, exactly.
    ent = sample_entry(probe, om, np.random.default_rng(2))
    first = np.stack(wheels_to_twist(om[0].astype(np.float64)), axis=-1)
    same = np.isclose(ent, first).all(axis=1)
    assert abs(same.mean() - (1 - probe.entry.jump_frac)) < 0.02, same.mean()
    # With ostrich's yaw compensation, every entry -- continued or jumped -- still fits once compensated.
    gained = dataclasses.replace(probe, command=CommandSpec(ostrich_yaw_gain=1.15))
    om_g, fam_g = sample_window_commands(20_000, np.random.default_rng(1), gained.command)
    ent = sample_entry(gained, om_g, np.random.default_rng(2))
    spin = fam_g == FAMILIES.index("spin")  # a spin's continuation entry spins, one wheel reversed
    assert entry_in_box(ent[~spin], gained.command).all()
    spin_wheels = (np.abs(ent[spin, 1]) * gained.command.ostrich_yaw_gain * HALF_TRACK
                   + np.abs(ent[spin, 0])) / WHEEL_RADIUS
    assert spin_wheels.max() <= gained.command.wmax + 1e-5
    assert not entry_in_box(np.array([[0.7, 1.9]]), gained.command).any()  # in the plain box, not after x1.15

    # Uniform slope scores ~0 relief along any path.
    slope = HeightMapReader(np.tan(np.radians(10.0)) * (X + 8.0), origin=(-8.0, -8.0), cell=0.05)
    org = np.column_stack([rng.uniform(-4, 4, 256), rng.uniform(-4, 4, 256), rng.uniform(0, 6.28, 256)])
    om, _ = sample_window_commands(256, rng, probe.command)
    assert path_relief_signed(slope, org, window_poses(org, om.transpose(1, 0, 2)))[0].max() < 5e-3

    # Flat: nothing interacts, a targeted draw falls back and reports it.
    flat = HeightMapReader.flat(xlim=(-8.0, 8.0), ylim=(-8.0, 8.0))
    fb = sample_trials(context(flat), 32, rng, interact_frac=0.5)
    assert fb.pose.shape == (32, 3) and fb.omega.shape == (32, WINDOW_STEPS, 3)
    assert fb.shortfall == 16 and (fb.relief == 0).all() and (fb.interact_dir == 0).all()
    warm_wz = probe.entry.yaw_ratio * fb.entry[:, 1]
    assert np.allclose(integrate_twist(fb.pose, fb.entry[:, 0], warm_wz, probe.warmup_s), fb.origin, atol=1e-6)

    # 0.2 m box: exact stratification, patch on the map, spawns AND window ends settle-feasible.
    box = HeightMapReader(np.where((np.abs(X) < 1.5) & (np.abs(Y) < 1.5), 0.2, 0.0), origin=(-8.0, -8.0), cell=0.05)
    ctx = context(box)
    t0 = time.perf_counter()
    b = sample_trials(ctx, 200, rng, interact_frac=0.5)
    assert (b.relief > 0.05).sum() == 100 and b.shortfall == 0 and b.endpoint_feasible.all()
    assert not patch_overhangs(box, b.origin, PATCH_SPEC).any()
    end = np.stack([window_poses(b.origin[i:i + 1], b.omega[i:i + 1])[-1][0] for i in range(len(b.pose))])
    d, r, c = settle_batch(box, np.concatenate([b.pose, end]), 0.8, args.device)
    assert settle_feasible(d, r, c, ctx.robot).all()
    print(f"box: 100/200 interacting (natural {b.proposal_interact_rate:.1%}), "
          f"families {dict(zip(FAMILIES, np.bincount(b.family, minlength=4)))}, {time.perf_counter() - t0:.1f}s")
    same = [sample_trials(ctx, 16, np.random.default_rng(3), interact_frac=0.25) for _ in range(2)]
    assert np.array_equal(same[0].pose, same[1].pose) and np.array_equal(same[0].omega, same[1].omega)

    # Edges: a 0.6 m box (drivable top) and a 1.0 m wall -- exact interacting share and down share,
    # origins within the band, some window ends blocked (no endpoint requirement).
    walls = np.where((np.abs(X + 2.5) < 1.25) & (np.abs(Y) < 1.25), 0.6, 0.0)
    walls = np.maximum(walls, np.where((np.abs(X - 2.5) < 0.1) & (np.abs(Y) < 2.0), 1.0, 0.0))
    wall_map = HeightMapReader(walls, origin=(-8.0, -8.0), cell=0.05)
    ctx = context(wall_map)
    t0 = time.perf_counter()
    eb = sample_edge_trials(ctx, 200, rng, interact_frac=0.5, band=1.5, facing_frac=0.7, down_frac=0.5)
    n_up, n_down = int((eb.interact_dir > 0).sum()), int((eb.interact_dir < 0).sum())
    assert eb.shortfall == 0 and (eb.strategy == "edge").all() and n_up + n_down == 100 and n_down == 50, (n_up, n_down, eb.shortfall)
    field = edge_field(wall_map)
    col = np.clip(((eb.origin[:, 0] - wall_map.x0) / wall_map.cell).astype(int), 0, wall_map.nx - 1)
    row = np.clip(((eb.origin[:, 1] - wall_map.y0) / wall_map.cell).astype(int), 0, wall_map.ny - 1)
    assert (field.dist[row, col] <= 1.5 + 1e-9).all()
    assert (~eb.endpoint_feasible).any(), "spawn-only edge trials should include blocked window ends"
    print(f"edges: {n_up} up / {n_down} down, {int((~eb.endpoint_feasible).sum())} blocked ends, {time.perf_counter() - t0:.1f}s")

    # Rotate in place: only spins, entered at rest, no wheel touching the feature at the spawn,
    # exactly interact_frac sweep into it, and the stored relief re-derives from the origin.
    thin = np.maximum(np.where((np.abs(X - 2.5) < 0.1) & (np.abs(Y) < 2.0), 1.0, 0.0),
                      np.where((np.abs(X + 2.0) < 0.15) & (np.abs(Y - 1.0) < 0.15), 0.5, 0.0))
    thin_map = HeightMapReader(thin, origin=(-8.0, -8.0), cell=0.05)
    ctx = context(thin_map)
    t0 = time.perf_counter()
    rb = sample_rotate_in_place_trials(ctx, 100, rng, interact_frac=0.5, band=0.3, min_clearance=0.05)
    assert (rb.family == SPIN).all() and (rb.entry == 0).all() and rb.shortfall == 0
    assert np.allclose(rb.omega[..., 0], -rb.omega[..., 1]) and (rb.interact_dir != 0).sum() == 50
    assert (rim_penetration(thin_map, rb.origin, [rb.pose]) <= TOUCH_TOL).all()
    relief, _ = path_relief_signed(thin_map, rb.origin, window_poses(rb.origin, rb.omega))
    assert np.allclose(relief, rb.relief, atol=1e-5)
    turned = np.abs(np.degrees(wheels_to_twist(rb.omega)[1].mean(axis=1) * WINDOW_S))
    print(f"rotate_in_place: 100 spins ({turned.min():.0f}-{turned.max():.0f} deg), 50 sweep a wheel in "
          f"(natural {rb.proposal_interact_rate:.1%}), {time.perf_counter() - t0:.1f}s")

    # Spins from standstill: the speed marginal stays exactly even over the bins even where a slow
    # spin cannot reach the feature, and speed does not stand in for contact.
    slow = dataclasses.replace(ctx, command=dataclasses.replace(ctx.command, spin_min=0.0))
    t0 = time.perf_counter()
    sb = sample_rotate_in_place_trials(slow, 200, rng, interact_frac=0.5, band=0.3, min_clearance=0.05)
    k = spin_speed_bin(slow.command, sb.omega, SPIN_SPEED_BINS)
    per_bin = np.bincount(k, minlength=SPIN_SPEED_BINS)
    assert (per_bin == 200 // SPIN_SPEED_BINS).all(), per_bin
    inter_bin = np.bincount(k, weights=sb.interact_dir != 0, minlength=SPIN_SPEED_BINS) / per_bin
    assert (sb.interact_dir != 0).sum() == 100 - sb.shortfall
    assert inter_bin[1:].min() > 0.25 and inter_bin.max() < 0.75, inter_bin
    print(f"rotate_in_place from 0: {per_bin.tolist()} per speed bin, interacting "
          f"{np.round(inter_bin, 2).tolist()}, shortfall {sb.shortfall}, {time.perf_counter() - t0:.1f}s")

    # Ramps: head-on along a face at mid-window, contacts on that face's own surface, no shortfall.
    ramps = [
        Ramp(cx=-3.0, cy=-2.0, yaw=0.3, up_deg=12.0, height=0.5, plateau=2.1, down_deg=80.0, width=1.6, side_deg=80.0),
        Ramp(cx=2.5, cy=2.5, yaw=2.0, up_deg=20.0, height=0.4, plateau=2.1, down_deg=8.0, width=2.0, side_deg=80.0),
    ]
    ramp_map = HeightMapReader(np.maximum(*(ramp_layer(r_, 16.0, 0.05) for r_ in ramps)), origin=(-8.0, -8.0), cell=0.05)
    ctx = context(ramp_map)
    faces = ramp_faces({"ramps": [dataclasses.asdict(r_) for r_ in ramps]})
    assert required_platform_length(ctx) <= 2.1, "the lattice maps' plateau floor must admit ramp_down"
    print(f"ramp_down: plateau floor {required_platform_length(ctx):.2f} m (these maps have 2.1), "
          f"origins from {2.1 - platform_behind(ctx):.2f} m before the crest")
    ramp_kw = dict(straight_frac=0.75, yaw_jitter_deg=5.0, fallback_interact_frac=0.5)
    for sampler, name, turn in ((sample_ramp_up_trials, "ramp_up", 0.0), (sample_ramp_down_trials, "ramp_down", np.pi)):
        t0 = time.perf_counter()
        rmb = sampler(ctx, faces, 200, rng, **ramp_kw)
        assert (rmb.strategy == name).all() and np.isfinite(rmb.ramp_deg).all(), (name, rmb.shortfall)
        mid = np.stack([window_poses(rmb.origin[i:i + 1], rmb.omega[i:i + 1])[MID_WINDOW][0] for i in range(200)])
        axis = np.array([next(f.yaw for f in faces if np.isclose(f.slope_deg, d_)) for d_ in rmb.ramp_deg])
        off = np.degrees(np.abs(np.angle(np.exp(1j * (mid[:, 2] - axis - turn)))))
        assert off.max() <= ramp_kw["yaw_jitter_deg"] + 1e-6, (name, off.max())
        assert set(np.unique(rmb.family[np.isfinite(rmb.ramp_s)])) <= {1, 2}, "ramp commands are STRAIGHT/NARROW"
        if name == "ramp_down":  # robot + warm-up on the 2.1 m plateau
            assert np.nanmin(rmb.ramp_s) >= platform_behind(ctx) - 2.1 - 1e-5, np.nanmin(rmb.ramp_s)
        print(f"{name}: 200 trials, shortfall {rmb.shortfall}, mid-window heading off-axis <= {off.max():.1f} deg, "
              f"interact {int((rmb.interact_dir > 0).sum())} up / {int((rmb.interact_dir < 0).sum())} down, {time.perf_counter() - t0:.1f}s")

    mixed = concat_batches([fb, eb], rng)
    assert len(mixed.pose) == 232 and mixed.shortfall == fb.shortfall + eb.shortfall
    print("all spawn_sampling checks ok")
