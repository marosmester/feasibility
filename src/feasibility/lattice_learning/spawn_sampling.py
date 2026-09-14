"""Trial sampling for `generate_dataset.py`: each trial's spawn pose AND its curvature, chosen
together, by a strategy tailored to the map's category.

Three decisions, all independent of any per-map height threshold:

* **Validity -- the planner's own static settle.** A trial needs helhest_stack's settle to be
  feasible (`settle.settle_feasible`: pitch/roll envelope, residual, chassis clearance) at the
  spawn, and the patch at the arc origin to stay on mapped terrain. Whether the NOMINAL arc end
  must be settle-feasible too depends on the strategy (`SamplingPolicy.spawn_only_strategies`):
  - `uniform`/`targeted` keep the arc-end check: without it, most arcs aimed at a tall obstacle
    end where the planner says `blocked` (measured 25% end-feasible among interacting trials vs
    92% for the rest), which is not those maps' regime.
  - `ramp`/`edge` drop it: those maps exist to show what ostrich does when the robot drives at a
    wall-like ramp or tries to climb up / drive down a 1.0 m edge, which is exactly where the
    settle says `blocked`. generate_dataset.py records every row's `endpoint_blocked`, so a
    consumer can still drop them (custom_dataset.py's `drop_blocked_endpoints`).
  `SpawnBatch.endpoint_feasible` carries the nominal check for every trial either way. The spawn
  check replaced an older "footprint below median + 15% of the relief" filter, so trials can start
  on ramps, box tops, curbs and rough ground.

* **Interaction -- by what the arc meets, not by where the robot stands.** `arc_relief` measures
  how far the terrain under the robot departs, over the recorded arc, from the plane through its
  three wheel contacts at the arc's origin. A constant slope scores ~0 (the static settle already
  handles it); a step, wall, kink, drop or bump under the wheels or the body scores its height.
  A trial `interacts` when `arc_relief > interact_relief`, and its `interact_dir` is +1 when the
  largest departure is ABOVE the plane (climbing up: a step, a wall, the foot of a ramp) and -1
  when BELOW (driving down: off a box top, over a crest), 0 when it does not interact.

* **Strategy per map category -- `sample_map_trials`.** The map's sidecar `category` (written by
  heightmap/create_maps_for_lattice_learning.py) picks how its trials are drawn:

  - `ramps` (`ramp_categories`) -> `sample_ramp_trials`: every trial drives UP a ramp face, head
    on, so divergence can be read against a continuous slope angle (`SpawnBatch.ramp_deg`). The
    faces come from the sidecar's `ramps` list: each rising face and each far side (a down-ramp,
    or a steep drop -- a wall-like face driven at from its foot). A trial picks a face, puts the
    arc origin's front axle at `ramp_s` ~ U(-ARC_LEN - wheel_radius, run - ARC_LEN) metres along
    it from the foot -- from where the front wheel just reaches the foot at the arc end, up to the
    crest; a 0.3 m arc after a 0.225 m warm-up cannot climb a long ramp from its foot, so to see
    the face at all the arc must also be allowed to START on it -- anywhere across its top width,
    with the heading at MID-arc along the face +- `ramp_yaw_jitter_deg`. A `ramp_straight_frac`
    share drives straight (kappa = 0), the rest on kappa ~ U(-kappa_max, kappa_max). Every wheel
    contact from spawn to arc end must stay inside the face's top width and on that ramp's own
    surface (compared against the ramp rasterised on the terrain's grid, so bilinear interpolation
    of an 80 deg face matches exactly and a second ramp overlapping it rejects the trial). Spawns
    past the climb envelope fail the settle, so on a steep face trials start before its foot.
  - `curbs_and_walls` (`edge_categories`, also the retired `walls`/`boxes`) -> `sample_edge_trials`:
    arc origins uniform among cells within `edge_band` of a height EDGE (a cell whose neighbour is
    steeper than EDGE_MIN_SLOPE_DEG -- found from the heightmap alone, no sidecar needed), heading
    at the nearest edge +- EDGE_FACING_CONE_DEG with probability `edge_facing_frac`, else uniform.
    Exactly `round(n * interact_frac)` trials interact, `edge_down_frac` of them driving down and
    the rest climbing up; the remainder are near an edge but do not meet it.
  - `untargeted_categories` (default `rough`) -> `sample_trials(interact_frac=None)`: uniform
    (pose, kappa) over the whole map.
  - anything else (and maps with no sidecar category) -> targeted `sample_trials`: uniform
    proposals, exactly `round(n * interact_frac)` interacting.

  Every stratified strategy is exact conditional sampling from its proposal: candidates are
  labelled by `arc_relief`, and each stratum is filled uniformly from the candidates carrying its
  label. If a stratum cannot be filled within `MAX_PROPOSAL_ROUNDS`, its deficit moves to a
  fallback stratum (down -> up -> non-interacting) and is counted in `SpawnBatch.shortfall`; only
  a map with too few valid spawns at all raises. A ramp map whose faces cannot supply enough
  trials falls back to `sample_trials` for the rest.

Where the arc starts: every trial is entered at speed after a warm-up of `lead` metres along the
same curvature (generate_dataset.py's `warmup_s * V_NOM`), so relief and the overhang check are
evaluated from the NOMINAL arc origin `integrate_arc(spawn, kappa, lead)`, not from the spawn.

Deliberately independent of `feasibility.learning`, `feasibility.grid_learning` and
`feasibility.grid_learning_2` (design.md section 11a).

Usage (smoke test -- synthetic maps on CPU, no assets; the first run compiles the settle kernels):
    python src/feasibility/lattice_learning/spawn_sampling.py
    python src/feasibility/lattice_learning/spawn_sampling.py --maps-dir assets/lattice_maps/0
"""
from __future__ import annotations

import dataclasses
import pathlib
from typing import Callable

import numpy as np
import yaml
from helhest.engine import RobotParams
from scipy.ndimage import distance_transform_edt

from feasibility.heightmap import HeightMapReader
from feasibility.heightmap.create_ramps import Ramp
from feasibility.heightmap.create_ramps import ramp_height
from feasibility.lattice_learning.arc import ARC_LEN
from feasibility.lattice_learning.arc import integrate_arc
from feasibility.lattice_learning.patch import patch_overhangs
from feasibility.lattice_learning.patch import PatchSpec
from feasibility.lattice_learning.patch import WHEEL_CONTACTS_LOCAL
from feasibility.lattice_learning.settle import settle_batch
from feasibility.lattice_learning.settle import settle_feasible

DEFAULT_INTERACT_FRAC: float | None = 0.5  # share of a stratified map's trials whose arc meets
# terrain; None = no interaction strata anywhere (plain draws from each strategy's proposal)
DEFAULT_INTERACT_RELIEF = 0.05  # m -- low enough that the lowest curb counts as an interaction
DEFAULT_UNTARGETED_CATEGORIES = ("rough",)  # sidecar categories sampled uniformly: flat and
# low-amplitude rough ground are negatives, there is nothing on them to aim at

DEFAULT_RAMP_CATEGORIES = ("ramps",)  # sidecar categories sampled by sample_ramp_trials
DEFAULT_RAMP_STRAIGHT_FRAC = 0.75  # share of ramp trials driven straight (kappa = 0)
DEFAULT_RAMP_YAW_JITTER_DEG = 5.0  # mid-arc heading off the face's uphill axis, uniform +-

DEFAULT_EDGE_CATEGORIES = ("curbs_and_walls", "walls", "boxes")  # sampled by sample_edge_trials;
# walls/boxes are create_maps_for_lattice_learning.py's retired categories, so old maps get it too
DEFAULT_EDGE_BAND = 1.5  # m, arc origins at most this far from an edge
DEFAULT_EDGE_FACING_FRAC = 0.7  # share of edge proposals heading at the nearest edge
DEFAULT_EDGE_DOWN_FRAC = 0.5  # share of an edge map's interacting trials that drive DOWN

DEFAULT_SPAWN_ONLY_STRATEGIES = ("ramp", "edge")  # no arc-end settle check, see module docstring

RAMP_MAX_FACE_DEG = 90.0  # every face is driven at, including 80 deg drops
RAMP_SIDE_MARGIN = 0.05  # m, wheel contacts stay this far inside the face's top width
RAMP_SURFACE_TOL = 0.03  # m, |terrain - that ramp alone| at a contact; catches a second ramp
# overlapping the face (and a rough base layer, which would reject every trial if enabled)

EDGE_MIN_SLOPE_DEG = 45.0  # a cell is an edge when a neighbour rises/falls steeper than this
EDGE_FACING_CONE_DEG = 45.0  # heading at the nearest edge, uniform +-

ARC_SAMPLES = 11  # poses along the relief window, ~5 cm apart (one heightmap cell)
PROPOSAL_BATCH = 8192  # (pose, kappa) candidates per round -- geometry only, cheap
MAX_PROPOSAL_ROUNDS = 64  # ~520k candidates before a stratum is declared short
SETTLE_SLACK = 2.0  # settle this many times the still-missing count per round (~70% feasible)
EDGE_SLACK = 0.2  # m, beyond patch reach + lead: covers generate_dataset's xy_jitter (0.12 m)

_ROBOT = RobotParams()


@dataclasses.dataclass
class SpawnBatch:
    pose: np.ndarray  # [n, 3] float64 spawn (x, y, yaw)
    kappa: np.ndarray  # [n] float32 arc curvature, 1/m
    arc_relief: np.ndarray  # [n] float32 m, see arc_relief()
    targeted: bool  # whether this map's trials were stratified by interaction
    shortfall: int  # stratified trials requested but substituted from a fallback stratum
    proposal_interact_rate: float  # natural share of interacting proposal candidates
    strategy: str = "targeted"  # "uniform" | "targeted" | "ramp" | "edge"
    ramp_deg: np.ndarray | None = None  # [n] float32 slope of the face driven, NaN = not a ramp trial
    ramp_s: np.ndarray | None = None  # [n] float32 m, arc origin's front axle along the face from
    # its foot (negative = still before the foot), NaN = not a ramp trial
    interact_dir: np.ndarray | None = None  # [n] int8 +1 up / -1 down / 0 not interacting
    endpoint_feasible: np.ndarray | None = None  # [n] bool, static settle at the NOMINAL arc end

    def __post_init__(self) -> None:
        n = len(self.pose)
        nan = np.full(n, np.nan, dtype=np.float32)
        self.ramp_deg = nan.copy() if self.ramp_deg is None else self.ramp_deg
        self.ramp_s = nan.copy() if self.ramp_s is None else self.ramp_s
        self.interact_dir = np.zeros(n, np.int8) if self.interact_dir is None else self.interact_dir
        if self.endpoint_feasible is None:
            self.endpoint_feasible = np.ones(n, dtype=bool)


@dataclasses.dataclass(frozen=True)
class SamplingPolicy:
    """How `sample_map_trials` picks a strategy from a map's sidecar category, and each strategy's
    knobs. generate_dataset.py builds one from its Hydra overrides."""

    interact_frac: float | None = DEFAULT_INTERACT_FRAC
    interact_relief: float = DEFAULT_INTERACT_RELIEF
    untargeted_categories: tuple[str, ...] = DEFAULT_UNTARGETED_CATEGORIES
    ramp_categories: tuple[str, ...] = DEFAULT_RAMP_CATEGORIES
    ramp_straight_frac: float = DEFAULT_RAMP_STRAIGHT_FRAC
    ramp_yaw_jitter_deg: float = DEFAULT_RAMP_YAW_JITTER_DEG
    edge_categories: tuple[str, ...] = DEFAULT_EDGE_CATEGORIES
    edge_band: float = DEFAULT_EDGE_BAND
    edge_facing_frac: float = DEFAULT_EDGE_FACING_FRAC
    edge_down_frac: float = DEFAULT_EDGE_DOWN_FRAC
    spawn_only_strategies: tuple[str, ...] = DEFAULT_SPAWN_ONLY_STRATEGIES


def map_metadata(stem: str | pathlib.Path) -> dict:
    """A map's whole .yaml sidecar (create_maps_for_lattice_learning.py adds `category` and the
    builder's params, e.g. the `ramps` list, to HeightMapReader's own keys)."""
    return yaml.safe_load(pathlib.Path(stem).with_suffix(".yaml").read_text()) or {}


def map_category(stem: str | pathlib.Path) -> str | None:
    """The `category` key a create_maps_for_lattice_learning.py sidecar carries, or None for maps
    from other generators (which are then targeted)."""
    return map_metadata(stem).get("category")


def map_strategy(meta: dict, policy: SamplingPolicy) -> str:
    """"ramp" | "edge" | "uniform" | "targeted" for a map with sidecar `meta`, checked in that
    order -- see the module docstring."""
    category = meta.get("category")
    if category in policy.ramp_categories:
        return "ramp"
    if category in policy.edge_categories:
        return "edge"
    if policy.interact_frac is None or category in policy.untargeted_categories:
        return "uniform"
    return "targeted"


def relief_lookahead(interact_relief: float) -> float:
    """m -- how far ahead of its contact point a wheel of radius r first touches a step of height
    h: sqrt(2 r h - h^2). Extends the relief window past the arc's end so a step the front wheels
    are about to hit counts, without counting ones they never reach."""
    r, h = float(_ROBOT.wheel_radius), min(interact_relief, float(_ROBOT.wheel_radius))
    return float(np.sqrt(2.0 * r * h - h * h))


def arc_relief_signed(
    terrain: HeightMapReader, spawn: np.ndarray, kappa: np.ndarray, lead: float, lookahead: float
) -> tuple[np.ndarray, np.ndarray]:
    """([n] m, [n] int8) -- max |terrain - plane| over the arc, and +1/-1 for whether that largest
    departure is above/below the plane. The plane passes through the three wheel contacts at the
    nominal arc origin `integrate_arc(spawn, kappa, lead)`; the terrain is sampled at the three
    wheel contacts plus the body center at ARC_SAMPLES poses from that origin to
    `ARC_LEN + lookahead` further along the same curvature. Plane-relative, so a uniform slope is
    ~0 while steps, kinks, drops and bumps under wheels or body register their height."""
    local = np.vstack([WHEEL_CONTACTS_LOCAL, [0.0, 0.0]])  # [4, 2]: 3 wheels + body center

    def contacts(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c, s = np.cos(pose[:, 2:3]), np.sin(pose[:, 2:3])
        return (pose[:, 0:1] + c * local[:, 0] - s * local[:, 1],
                pose[:, 1:2] + s * local[:, 0] + c * local[:, 1])  # [n, 4] each

    origin = integrate_arc(spawn, kappa, lead)
    ox, oy = contacts(origin)
    oz = np.asarray(terrain.sample(ox[:, :3], oy[:, :3]), dtype=np.float64)
    x0, y0 = origin[:, 0:1], origin[:, 1:2]
    A = np.stack([np.ones_like(oz), ox[:, :3] - x0, oy[:, :3] - y0], axis=-1)  # [n, 3, 3]
    coef = np.linalg.solve(A, oz[..., None])[..., 0]  # [n, 3] z = a + b dx + c dy

    above = np.zeros(len(spawn))
    below = np.zeros(len(spawn))
    for s in np.linspace(lead, lead + ARC_LEN + lookahead, ARC_SAMPLES):
        px, py = contacts(integrate_arc(spawn, kappa, s))
        z = np.asarray(terrain.sample(px, py), dtype=np.float64)
        plane = coef[:, :1] + coef[:, 1:2] * (px - x0) + coef[:, 2:3] * (py - y0)
        above = np.maximum(above, (z - plane).max(axis=1))
        below = np.maximum(below, (plane - z).max(axis=1))
    return np.maximum(above, below), np.where(above >= below, 1, -1).astype(np.int8)


def arc_relief(
    terrain: HeightMapReader, spawn: np.ndarray, kappa: np.ndarray, lead: float, lookahead: float
) -> np.ndarray:
    """[n] m -- `arc_relief_signed`'s magnitude only."""
    return arc_relief_signed(terrain, spawn, kappa, lead, lookahead)[0]


def interaction_dir(relief: np.ndarray, sign: np.ndarray, interact_relief: float) -> np.ndarray:
    """[n] int8: +1 climbing up, -1 driving down, 0 not interacting."""
    return np.where(relief > interact_relief, sign, 0).astype(np.int8)


def trials_feasible(
    terrain: HeightMapReader,
    spec: PatchSpec,
    spawn: np.ndarray,
    kappa: np.ndarray,
    lead: float,
    mu: float,
    device: str,
    robot: RobotParams,
) -> tuple[np.ndarray, np.ndarray]:
    """([n] bool, [n] bool) -- (static settle feasible at the spawn and the patch at the nominal
    arc origin on mapped terrain, static settle feasible at the nominal arc end). Every strategy
    requires the first; `uniform`/`targeted` also require the second."""
    k = len(spawn)
    origin = integrate_arc(spawn, kappa, lead)
    arc_end = integrate_arc(origin, kappa, ARC_LEN)
    derived, residual, clearance = settle_batch(terrain, np.concatenate([spawn, arc_end]), mu, device)
    feasible = settle_feasible(derived, residual, clearance, robot)
    return feasible[:k] & ~patch_overhangs(terrain, origin, spec), feasible[k:]


def sampling_bounds(terrain: HeightMapReader, spec: PatchSpec, lead: float) -> tuple[float, ...]:
    """(x_lo, x_hi, y_lo, y_hi) of the square uniform proposals draw spawns from."""
    margin = spec.reach + lead + EDGE_SLACK
    x_lo, x_hi = terrain.x0 + margin, terrain.x0 + terrain.nx * terrain.cell - margin
    y_lo, y_hi = terrain.y0 + margin, terrain.y0 + terrain.ny * terrain.cell - margin
    if x_hi <= x_lo or y_hi <= y_lo:
        raise ValueError(
            f"terrain is too small for a patch reach of {spec.reach:.2f} m plus a {lead:.2f} m "
            f"lead (extent {terrain.nx * terrain.cell:.2f} x {terrain.ny * terrain.cell:.2f} m)"
        )
    return x_lo, x_hi, y_lo, y_hi


def stratified_fill(
    terrain: HeightMapReader,
    spec: PatchSpec,
    n: int,
    need: dict[int, int],
    fallback: list[tuple[int, int]],
    propose: Callable[[int], np.ndarray],
    label: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    lead: float,
    interact_relief: float,
    mu: float,
    device: str,
    robot: RobotParams,
    require_endpoint: bool,
) -> tuple[np.ndarray, int, float]:
    """The loop every stratified strategy shares. `propose(m)` -> [m, 4] candidate (spawn x, y,
    yaw, kappa); `label(relief, dir)` -> [m] int stratum per candidate; `need` maps stratum ->
    trials wanted (in fill order). After MAX_PROPOSAL_ROUNDS, each (short, into) pair of
    `fallback`, in order, moves `short`'s deficit to `into` and fills again.
    Returns ([n, 7] rows (spawn x, y, yaw, kappa, relief, dir, endpoint_ok), shortfall, natural
    interacting share of the proposals)."""
    need = dict(need)
    lookahead = relief_lookahead(interact_relief)
    got: dict[int, list[np.ndarray]] = {k: [] for k in need}
    counts = {k: 0 for k in need}
    n_proposed = n_interacting = 0

    def fill(rounds: int) -> None:
        nonlocal n_proposed, n_interacting
        for _ in range(rounds):
            missing = {k: need[k] - counts[k] for k in need}
            if all(m <= 0 for m in missing.values()):
                return
            cand = propose(PROPOSAL_BATCH)
            m = len(cand)
            relief, sign = arc_relief_signed(terrain, cand[:, :3], cand[:, 3], lead, lookahead)
            labels = label(relief, sign)
            n_proposed += m
            n_interacting += int((relief > interact_relief).sum())

            picked = []
            for key, miss in missing.items():
                if miss <= 0:
                    continue
                idx = np.flatnonzero(labels == key)
                picked.append((key, idx[: int(SETTLE_SLACK * miss) + 16]))
            all_idx = np.concatenate([idx for _, idx in picked])
            if len(all_idx) == 0:
                continue
            base_ok, end_ok = trials_feasible(
                terrain, spec, cand[all_idx, :3], cand[all_idx, 3], lead, mu, device, robot
            )
            ok_all = base_ok & end_ok if require_endpoint else base_ok
            ok_by_index = dict(zip(all_idx.tolist(), ok_all.tolist()))
            end_by_index = dict(zip(all_idx.tolist(), end_ok.tolist()))
            for key, idx in picked:
                keep = idx[[ok_by_index[i] for i in idx.tolist()]][: need[key] - counts[key]]
                end = np.array([end_by_index[i] for i in keep.tolist()], dtype=np.float64)
                rows = np.column_stack([cand[keep], relief[keep], sign[keep], end])
                got[key].append(rows)
                counts[key] += len(rows)

    fill(MAX_PROPOSAL_ROUNDS)
    shortfall = 0
    for short, into in fallback:
        if counts[short] < need[short]:
            deficit = need[short] - counts[short]
            shortfall += deficit
            need[short] = counts[short]
            need[into] += deficit
            fill(MAX_PROPOSAL_ROUNDS)
    total = sum(counts.values())
    if total < n:
        raise ValueError(
            f"only found {total}/{n} settle-feasible trials after {n_proposed} proposals -- the "
            "map has too little drivable ground where this strategy proposes"
        )
    rows = np.concatenate([np.concatenate(v) for v in got.values() if v])
    return rows, shortfall, n_interacting / max(n_proposed, 1)


def batch_from_rows(
    rows: np.ndarray, rng: np.random.Generator, interact_relief: float, **fields: object
) -> SpawnBatch:
    """SpawnBatch from `stratified_fill`'s rows, shuffled so chunk order never correlates with
    which stratum a trial came from."""
    rng.shuffle(rows)
    return SpawnBatch(
        pose=rows[:, :3],
        kappa=rows[:, 3].astype(np.float32),
        arc_relief=rows[:, 4].astype(np.float32),
        interact_dir=interaction_dir(rows[:, 4], rows[:, 5], interact_relief),
        endpoint_feasible=rows[:, 6] > 0.5,
        **fields,
    )


def sample_trials(
    terrain: HeightMapReader,
    spec: PatchSpec,
    n: int,
    rng: np.random.Generator,
    *,
    kappa_max: float,
    lead: float,
    mu: float,
    device: str,
    interact_frac: float | None = DEFAULT_INTERACT_FRAC,
    interact_relief: float = DEFAULT_INTERACT_RELIEF,
    require_endpoint: bool = True,
    robot: RobotParams | None = None,
) -> SpawnBatch:
    """`n` (spawn pose, kappa) trials drawn uniformly over `terrain` -- see the module docstring.
    `interact_frac` None samples plainly (`uniform`); a float in [0, 1] stratifies by
    `arc_relief > interact_relief` (`targeted`)."""
    if interact_frac is not None and not 0.0 <= interact_frac <= 1.0:
        raise ValueError(f"interact_frac must be in [0, 1] or None, got {interact_frac}")
    robot = robot or RobotParams()
    x_lo, x_hi, y_lo, y_hi = sampling_bounds(terrain, spec, lead)

    def propose(m: int) -> np.ndarray:
        return np.column_stack([
            rng.uniform(x_lo, x_hi, m), rng.uniform(y_lo, y_hi, m),
            rng.uniform(0.0, 2.0 * np.pi, m), rng.uniform(-kappa_max, kappa_max, m),
        ])

    if interact_frac is None:
        need, fallback = {0: n}, []
        label = lambda relief, sign: np.zeros(len(relief), dtype=int)  # noqa: E731
    else:
        n_int = round(n * interact_frac)
        need, fallback = {1: n_int, 0: n - n_int}, [(1, 0)]
        label = lambda relief, sign: (relief > interact_relief).astype(int)  # noqa: E731
    rows, shortfall, rate = stratified_fill(
        terrain, spec, n, need, fallback, propose, label, lead=lead,
        interact_relief=interact_relief, mu=mu, device=device, robot=robot,
        require_endpoint=require_endpoint,
    )
    return batch_from_rows(
        rows, rng, interact_relief, targeted=interact_frac is not None, shortfall=shortfall,
        proposal_interact_rate=rate, strategy="targeted" if interact_frac is not None else "uniform",
    )


@dataclasses.dataclass(frozen=True)
class EdgeField:
    """Per-cell distance to, and position of, the nearest height edge of a map."""

    edge: np.ndarray  # [ny, nx] bool
    dist: np.ndarray  # [ny, nx] m, 0 on an edge cell
    nearest_x: np.ndarray  # [ny, nx] m, world x of the nearest edge cell's center
    nearest_y: np.ndarray  # [ny, nx] m


def edge_field(terrain: HeightMapReader, min_slope_deg: float = EDGE_MIN_SLOPE_DEG) -> EdgeField:
    """Edge cells: a 4-neighbour differs in height by more than cell * tan(min_slope_deg). Found
    from the heightmap alone, so it works on any map (no sidecar geometry)."""
    H = np.asarray(terrain.H, dtype=np.float64)
    dz_max = terrain.cell * np.tan(np.radians(min_slope_deg))
    edge = np.zeros(H.shape, dtype=bool)
    for axis in (0, 1):
        step = np.abs(np.diff(H, axis=axis)) > dz_max
        lo = [slice(None)] * 2
        hi = [slice(None)] * 2
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        edge[tuple(lo)] |= step
        edge[tuple(hi)] |= step
    if not edge.any():
        inf = np.full(H.shape, np.inf)
        return EdgeField(edge, inf, inf, inf)
    dist, (iy, ix) = distance_transform_edt(~edge, return_indices=True)
    return EdgeField(
        edge=edge,
        dist=dist * terrain.cell,
        nearest_x=terrain.x0 + (ix + 0.5) * terrain.cell,
        nearest_y=terrain.y0 + (iy + 0.5) * terrain.cell,
    )


def sample_edge_trials(
    terrain: HeightMapReader,
    spec: PatchSpec,
    n: int,
    rng: np.random.Generator,
    *,
    kappa_max: float,
    lead: float,
    mu: float,
    device: str,
    interact_frac: float | None = DEFAULT_INTERACT_FRAC,
    interact_relief: float = DEFAULT_INTERACT_RELIEF,
    band: float = DEFAULT_EDGE_BAND,
    facing_frac: float = DEFAULT_EDGE_FACING_FRAC,
    down_frac: float = DEFAULT_EDGE_DOWN_FRAC,
    require_endpoint: bool = False,
    robot: RobotParams | None = None,
) -> SpawnBatch:
    """`n` trials whose arc origins lie within `band` of a height edge -- see the module
    docstring. A map without a single edge cell in its sampling square falls back to
    `sample_trials` (all of `n` counted in `shortfall`)."""
    for name, value in (("facing_frac", facing_frac), ("down_frac", down_frac)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
    if interact_frac is not None and not 0.0 <= interact_frac <= 1.0:
        raise ValueError(f"interact_frac must be in [0, 1] or None, got {interact_frac}")
    robot = robot or RobotParams()
    field = edge_field(terrain)
    x_lo, x_hi, y_lo, y_hi = sampling_bounds(terrain, spec, 0.0)  # origins: the patch is there
    cx = terrain.x0 + (np.arange(terrain.nx) + 0.5) * terrain.cell
    cy = terrain.y0 + (np.arange(terrain.ny) + 0.5) * terrain.cell
    CX, CY = np.meshgrid(cx, cy)
    cells = np.flatnonzero(
        (field.dist <= band) & (CX >= x_lo) & (CX <= x_hi) & (CY >= y_lo) & (CY <= y_hi)
    )
    if len(cells) == 0:
        rest = sample_trials(
            terrain, spec, n, rng, kappa_max=kappa_max, lead=lead, mu=mu, device=device,
            interact_frac=interact_frac, interact_relief=interact_relief,
            require_endpoint=require_endpoint, robot=robot,
        )
        return dataclasses.replace(rest, shortfall=n, strategy="edge")

    cone = np.radians(EDGE_FACING_CONE_DEG)
    dist, near_x, near_y = field.dist.ravel(), field.nearest_x.ravel(), field.nearest_y.ravel()

    def propose(m: int) -> np.ndarray:
        cell = cells[rng.integers(0, len(cells), m)]
        x = CX.ravel()[cell] + rng.uniform(-0.5, 0.5, m) * terrain.cell
        y = CY.ravel()[cell] + rng.uniform(-0.5, 0.5, m) * terrain.cell
        to_edge = np.arctan2(near_y[cell] - y, near_x[cell] - x)
        facing = (rng.uniform(size=m) < facing_frac) & (dist[cell] > 0.0)
        yaw = np.where(facing, to_edge + rng.uniform(-cone, cone, m), rng.uniform(0.0, 2.0 * np.pi, m))
        kappa = rng.uniform(-kappa_max, kappa_max, m)
        spawn = integrate_arc(np.column_stack([x, y, yaw]), kappa, -lead)
        return np.column_stack([spawn, kappa])

    if interact_frac is None:
        need, fallback = {0: n}, []
    else:
        n_int = round(n * interact_frac)
        n_down = round(n_int * down_frac)
        need, fallback = {-1: n_down, 1: n_int - n_down, 0: n - n_int}, [(-1, 1), (1, 0)]

    def label(relief: np.ndarray, sign: np.ndarray) -> np.ndarray:
        if interact_frac is None:
            return np.zeros(len(relief), dtype=int)
        return interaction_dir(relief, sign, interact_relief).astype(int)

    rows, shortfall, rate = stratified_fill(
        terrain, spec, n, need, fallback, propose, label, lead=lead,
        interact_relief=interact_relief, mu=mu, device=device, robot=robot,
        require_endpoint=require_endpoint,
    )
    return batch_from_rows(
        rows, rng, interact_relief, targeted=interact_frac is not None, shortfall=shortfall,
        proposal_interact_rate=rate, strategy="edge",
    )


@dataclasses.dataclass(frozen=True)
class RampFace:
    """One drivable face, in world coordinates: the robot drives along `yaw` (uphill) from `foot`."""

    ramp: Ramp  # the whole ramp, for its surface
    ramp_index: int  # position in the sidecar's `ramps` list
    foot_x: float
    foot_y: float
    yaw: float  # rad, uphill direction
    slope_deg: float
    run: float  # m, horizontal length of the face, foot to crest
    half_width: float  # m, half the ramp's TOP width (the skirt below it is a side drop)


def ramp_faces(meta: dict) -> list[RampFace]:
    """Every face in a create_ramps.py sidecar's `ramps` list no steeper than RAMP_MAX_FACE_DEG:
    each ramp's rising face and its far side (down-ramp or drop)."""
    faces = []
    for i, r in enumerate(meta.get("ramps") or []):
        ramp = Ramp(**{f.name: float(r[f.name]) for f in dataclasses.fields(Ramp)})
        dx, dy = np.cos(ramp.yaw), np.sin(ramp.yaw)
        half = ramp.length / 2.0
        for sign, slope_deg in ((-1.0, ramp.up_deg), (1.0, ramp.down_deg)):
            if slope_deg > RAMP_MAX_FACE_DEG:
                continue
            faces.append(RampFace(
                ramp=ramp,
                ramp_index=i,
                foot_x=ramp.cx + sign * half * dx,
                foot_y=ramp.cy + sign * half * dy,
                yaw=float(ramp.yaw if sign < 0 else ramp.yaw + np.pi),
                slope_deg=float(slope_deg),
                run=float(ramp.height / np.tan(np.radians(slope_deg))),
                half_width=ramp.width / 2.0,
            ))
    return faces


def ramp_alone(terrain: HeightMapReader, ramp: Ramp) -> HeightMapReader:
    """`ramp` on otherwise flat ground, rasterised on `terrain`'s own grid -- so sampling both at
    the same point interpolates identically wherever the ramp is the only thing there."""
    xs = terrain.x0 + (np.arange(terrain.nx) + 0.5) * terrain.cell
    ys = terrain.y0 + (np.arange(terrain.ny) + 0.5) * terrain.cell
    X, Y = np.meshgrid(xs, ys)
    return HeightMapReader(ramp_height(ramp, X, Y), origin=(terrain.x0, terrain.y0), cell=terrain.cell)


def sample_ramp_trials(
    terrain: HeightMapReader,
    faces: list[RampFace],
    spec: PatchSpec,
    n: int,
    rng: np.random.Generator,
    *,
    kappa_max: float,
    lead: float,
    mu: float,
    device: str,
    straight_frac: float = DEFAULT_RAMP_STRAIGHT_FRAC,
    yaw_jitter_deg: float = DEFAULT_RAMP_YAW_JITTER_DEG,
    interact_relief: float = DEFAULT_INTERACT_RELIEF,
    fallback_interact_frac: float | None = DEFAULT_INTERACT_FRAC,
    require_endpoint: bool = False,
    robot: RobotParams | None = None,
) -> SpawnBatch:
    """`n` trials driving head-on up the given ramp faces -- see the module docstring. Faces are
    picked uniformly per candidate. If too few candidates survive (no faces, or every face too
    narrow or too crowded for a feasible spawn), the remainder is filled by `sample_trials` with
    `fallback_interact_frac` and counted in `shortfall`."""
    if not 0.0 <= straight_frac <= 1.0:
        raise ValueError(f"straight_frac must be in [0, 1], got {straight_frac}")
    robot = robot or RobotParams()
    lookahead = relief_lookahead(interact_relief)
    jitter = np.radians(yaw_jitter_deg)
    local = WHEEL_CONTACTS_LOCAL
    window = np.linspace(-lead, ARC_LEN, ARC_SAMPLES)  # spawn ... arc end, relative to the origin
    s_min = -ARC_LEN - float(robot.wheel_radius)  # front wheel just reaches the foot at arc end
    surfaces = {f.ramp_index: ramp_alone(terrain, f.ramp) for f in faces}

    got: list[np.ndarray] = []
    count = 0
    for _ in range(MAX_PROPOSAL_ROUNDS if faces else 0):
        if count >= n:
            break
        m = PROPOSAL_BATCH
        face_idx = rng.integers(0, len(faces), m)
        fx = np.array([f.foot_x for f in faces])[face_idx]
        fy = np.array([f.foot_y for f in faces])[face_idx]
        fyaw = np.array([f.yaw for f in faces])[face_idx]
        run = np.array([f.run for f in faces])[face_idx]
        half_w = np.array([f.half_width for f in faces])[face_idx]

        s0 = rng.uniform(s_min, np.maximum(run - ARC_LEN, s_min))
        t0 = rng.uniform(-half_w, half_w)
        kappa = np.where(rng.uniform(size=m) < straight_frac, 0.0, rng.uniform(-kappa_max, kappa_max, m))
        # heading along the face at MID-arc, so a curved arc bends symmetrically about the axis
        yaw0 = fyaw + rng.uniform(-jitter, jitter, m) - kappa * ARC_LEN / 2.0
        c, s = np.cos(fyaw), np.sin(fyaw)
        origin = np.column_stack([fx + s0 * c - t0 * s, fy + s0 * s + t0 * c, yaw0])

        # every wheel contact from spawn to arc end inside the top width and on this ramp's surface
        on_face = np.ones(m, dtype=bool)
        for dist in window:
            p = integrate_arc(origin, kappa, dist)
            pc, ps = np.cos(p[:, 2:3]), np.sin(p[:, 2:3])
            wx = p[:, 0:1] + pc * local[:, 0] - ps * local[:, 1]  # [m, 3]
            wy = p[:, 1:2] + ps * local[:, 0] + pc * local[:, 1]
            lateral = -s[:, None] * (wx - fx[:, None]) + c[:, None] * (wy - fy[:, None])
            on_face &= (np.abs(lateral) <= half_w[:, None] - RAMP_SIDE_MARGIN).all(axis=1)
            z = np.asarray(terrain.sample(wx, wy), dtype=np.float64)
            for j, face in enumerate(faces):
                rows = np.flatnonzero((face_idx == j) & on_face)
                surface = np.asarray(surfaces[face.ramp_index].sample(wx[rows], wy[rows]))
                on_face[rows] &= (np.abs(z[rows] - surface) <= RAMP_SURFACE_TOL).all(axis=1)

        idx = np.flatnonzero(on_face)[: int(SETTLE_SLACK * (n - count)) + 16]
        if len(idx) == 0:
            continue
        spawn = integrate_arc(origin[idx], kappa[idx], -lead)
        base_ok, end_ok = trials_feasible(terrain, spec, spawn, kappa[idx], lead, mu, device, robot)
        ok = base_ok & end_ok if require_endpoint else base_ok
        keep = np.flatnonzero(ok)[: n - count]
        slope = np.array([f.slope_deg for f in faces])[face_idx[idx[keep]]]
        got.append(np.column_stack([spawn[keep], kappa[idx[keep]], slope, s0[idx[keep]], end_ok[keep]]))
        count += len(keep)

    rows = np.concatenate(got) if got else np.zeros((0, 7))
    shortfall = n - len(rows)
    pose, kap = rows[:, :3], rows[:, 3].astype(np.float32)
    ramp_deg, ramp_s = rows[:, 4].astype(np.float32), rows[:, 5].astype(np.float32)
    end_feasible = rows[:, 6] > 0.5
    if len(rows):
        relief, sign = arc_relief_signed(terrain, pose, rows[:, 3], lead, lookahead)
        direction = interaction_dir(relief, sign, interact_relief)
    else:
        relief, direction = np.zeros(0), np.zeros(0, dtype=np.int8)
    if shortfall:
        rest = sample_trials(
            terrain, spec, shortfall, rng, kappa_max=kappa_max, lead=lead, mu=mu, device=device,
            interact_frac=fallback_interact_frac, interact_relief=interact_relief,
            require_endpoint=require_endpoint, robot=robot,
        )
        nan = np.full(shortfall, np.nan, dtype=np.float32)
        pose = np.concatenate([pose, rest.pose])
        kap = np.concatenate([kap, rest.kappa])
        relief = np.concatenate([relief, rest.arc_relief])
        direction = np.concatenate([direction, rest.interact_dir])
        end_feasible = np.concatenate([end_feasible, rest.endpoint_feasible])
        ramp_deg, ramp_s = np.concatenate([ramp_deg, nan]), np.concatenate([ramp_s, nan])
    order = rng.permutation(n)
    return SpawnBatch(
        pose=pose[order],
        kappa=kap[order],
        arc_relief=relief[order].astype(np.float32),
        targeted=True,
        shortfall=shortfall,
        proposal_interact_rate=float("nan"),
        strategy="ramp",
        ramp_deg=ramp_deg[order],
        ramp_s=ramp_s[order],
        interact_dir=direction[order],
        endpoint_feasible=end_feasible[order],
    )


def sample_map_trials(
    terrain: HeightMapReader,
    meta: dict,
    spec: PatchSpec,
    n: int,
    rng: np.random.Generator,
    *,
    kappa_max: float,
    lead: float,
    mu: float,
    device: str,
    policy: SamplingPolicy = SamplingPolicy(),
    robot: RobotParams | None = None,
) -> SpawnBatch:
    """`n` trials on one map, drawn by the strategy its sidecar `meta` selects (`map_strategy`)."""
    strategy = map_strategy(meta, policy)
    kw = dict(kappa_max=kappa_max, lead=lead, mu=mu, device=device,
              interact_relief=policy.interact_relief, robot=robot,
              require_endpoint=strategy not in policy.spawn_only_strategies)
    if strategy == "ramp":
        return sample_ramp_trials(
            terrain, ramp_faces(meta), spec, n, rng, straight_frac=policy.ramp_straight_frac,
            yaw_jitter_deg=policy.ramp_yaw_jitter_deg, fallback_interact_frac=policy.interact_frac,
            **kw,
        )
    if strategy == "edge":
        return sample_edge_trials(
            terrain, spec, n, rng, interact_frac=policy.interact_frac, band=policy.edge_band,
            facing_frac=policy.edge_facing_frac, down_frac=policy.edge_down_frac, **kw,
        )
    frac = None if strategy == "uniform" else policy.interact_frac
    return sample_trials(terrain, spec, n, rng, interact_frac=frac, **kw)


if __name__ == "__main__":
    import argparse
    import time

    import warp as wp

    from feasibility.lattice_learning.arc import KAPPA_MAX
    from feasibility.lattice_learning.arc import V_NOM

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps-dir", default=None, help="also report per-map stats for every *.png here")
    ap.add_argument("--n", type=int, default=64, help="trials per map for --maps-dir (default: 64)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    wp.init()
    spec = PatchSpec()
    lead = 0.375 * V_NOM  # generate_dataset.py's DEFAULT_WARMUP_S
    kw = dict(kappa_max=KAPPA_MAX, lead=lead, mu=0.8, device=args.device)
    rng = np.random.default_rng(0)

    xs = np.arange(-8.0, 8.0, 0.05) + 0.025
    X, Y = np.meshgrid(xs, xs)

    # Uniform slope: the plane-relative relief must ignore it entirely.
    slope = HeightMapReader(np.tan(np.radians(10.0)) * (X + 8.0), origin=(-8.0, -8.0), cell=0.05)
    poses = np.column_stack([rng.uniform(-4, 4, 256), rng.uniform(-4, 4, 256), rng.uniform(0, 6.28, 256)])
    kap = rng.uniform(-KAPPA_MAX, KAPPA_MAX, 256)
    assert arc_relief(slope, poses, kap, lead, 0.2).max() < 5e-3, "uniform slope must score ~0 relief"

    # Flat: nothing interacts, a targeted draw falls back to non-interacting and reports it.
    flat = HeightMapReader.flat(xlim=(-8.0, 8.0), ylim=(-8.0, 8.0))
    fb = sample_trials(flat, spec, 32, rng, interact_frac=0.5, **kw)
    assert fb.pose.shape == (32, 3) and fb.shortfall == 16 and (fb.arc_relief == 0).all()
    assert (fb.interact_dir == 0).all() and fb.endpoint_feasible.all()

    # 0.2 m box: exact stratification, patch at the arc origin on the map, spawns AND arc ends
    # settle-feasible (including some spawns on the box top).
    box_h = 0.2
    box = HeightMapReader(np.where((np.abs(X) < 1.5) & (np.abs(Y) < 1.5), box_h, 0.0),
                          origin=(-8.0, -8.0), cell=0.05)
    t0 = time.perf_counter()
    b = sample_trials(box, spec, 200, rng, interact_frac=0.5, **kw)
    interacts = b.arc_relief > DEFAULT_INTERACT_RELIEF
    assert interacts.sum() == 100 and b.shortfall == 0, (interacts.sum(), b.shortfall)
    assert ((b.interact_dir != 0) == interacts).all() and b.endpoint_feasible.all()
    assert not patch_overhangs(box, integrate_arc(b.pose, b.kappa, lead), spec).any()
    arc_end = integrate_arc(integrate_arc(b.pose, b.kappa, lead), b.kappa, ARC_LEN)
    d, r, c = settle_batch(box, np.concatenate([b.pose, arc_end]), 0.8, args.device)
    assert settle_feasible(d, r, c, RobotParams()).all()
    on_top = (np.abs(b.pose[:, 0]) < 1.0) & (np.abs(b.pose[:, 1]) < 1.0)
    assert on_top.any(), "a 0.2 m box top is drivable ground and must be spawnable"
    print(f"box: 100/200 interacting (natural rate {b.proposal_interact_rate:.1%}; "
          f"{int((b.interact_dir > 0).sum())} up / {int((b.interact_dir < 0).sum())} down), "
          f"{int(on_top.sum())} spawns on the box top, {time.perf_counter() - t0:.2f}s")

    same = [sample_trials(box, spec, 16, np.random.default_rng(3), interact_frac=0.25, **kw) for _ in range(2)]
    assert np.array_equal(same[0].pose, same[1].pose) and np.array_equal(same[0].kappa, same[1].kappa)
    print("slope/flat/box/reproducibility checks ok")

    # Edges: a 0.6 m box (drivable top, so trials can drive DOWN) and a 1.0 m thin wall. Exact
    # interacting share, the requested down share, origins within the band, spawns feasible, and
    # -- no arc-end check -- some arcs end where the settle says blocked.
    walls = np.where((np.abs(X + 2.5) < 1.25) & (np.abs(Y) < 1.25), 0.6, 0.0)
    walls = np.maximum(walls, np.where((np.abs(X - 2.5) < 0.1) & (np.abs(Y) < 2.0), 1.0, 0.0))
    wall_map = HeightMapReader(walls, origin=(-8.0, -8.0), cell=0.05)
    t0 = time.perf_counter()
    policy = SamplingPolicy()
    eb = sample_map_trials(wall_map, {"category": "curbs_and_walls"}, spec, 200, rng, policy=policy, **kw)
    n_up, n_down = int((eb.interact_dir > 0).sum()), int((eb.interact_dir < 0).sum())
    assert eb.strategy == "edge" and eb.shortfall == 0, (eb.strategy, eb.shortfall)
    assert n_up + n_down == 100 and n_down == 50, (n_up, n_down)
    field = edge_field(wall_map)
    origin = integrate_arc(eb.pose, eb.kappa, lead)
    col = np.clip(((origin[:, 0] - wall_map.x0) / wall_map.cell).astype(int), 0, wall_map.nx - 1)
    row = np.clip(((origin[:, 1] - wall_map.y0) / wall_map.cell).astype(int), 0, wall_map.ny - 1)
    assert (field.dist[row, col] <= policy.edge_band + 1e-9).all()
    base_ok, end_ok = trials_feasible(wall_map, spec, eb.pose, eb.kappa, lead, 0.8, args.device, RobotParams())
    assert base_ok.all() and np.array_equal(end_ok, eb.endpoint_feasible)
    assert (~eb.endpoint_feasible).any(), "spawn-only edge trials should include blocked arc ends"
    down_on_box = (np.abs(eb.pose[eb.interact_dir < 0, 0] + 2.5) < 1.25).mean()
    print(f"edges: {n_up} up / {n_down} down / {200 - n_up - n_down} near-miss (natural interacting "
          f"rate {eb.proposal_interact_rate:.1%}), {int((~eb.endpoint_feasible).sum())} blocked arc "
          f"ends, {down_on_box:.0%} of down spawns on the box, {time.perf_counter() - t0:.2f}s")
    print("edge checks ok")

    # Ramps: every trial head-on up a face, straight share ~ ramp_straight_frac, mid-arc heading
    # within the jitter of the uphill axis, all wheel contacts on that face's surface.
    from feasibility.heightmap.create_ramps import ramp_layer

    ramps = [
        Ramp(cx=-2.0, cy=0.0, yaw=0.3, up_deg=12.0, height=0.5, plateau=1.0, down_deg=80.0,
             width=1.6, side_deg=80.0),
        Ramp(cx=2.5, cy=1.0, yaw=2.0, up_deg=20.0, height=0.4, plateau=1.0, down_deg=8.0,
             width=2.0, side_deg=80.0),
    ]
    ramp_map = HeightMapReader(np.maximum(*(ramp_layer(r, 16.0, 0.05) for r in ramps)),
                               origin=(-8.0, -8.0), cell=0.05)
    meta = {"category": "ramps", "ramps": [dataclasses.asdict(r) for r in ramps]}
    faces = ramp_faces(meta)
    assert sorted(round(f.slope_deg) for f in faces) == [8, 12, 20, 80], "drop is a face too"
    t0 = time.perf_counter()
    rb = sample_map_trials(ramp_map, meta, spec, 400, rng, policy=policy, **kw)
    assert rb.strategy == "ramp" and rb.shortfall == 0 and np.isfinite(rb.ramp_deg).all()
    straight = (rb.kappa == 0).mean()
    assert abs(straight - policy.ramp_straight_frac) < 0.1, straight
    assert set(np.round(rb.ramp_deg).tolist()) == {8.0, 12.0, 20.0, 80.0}
    mid = integrate_arc(rb.pose, rb.kappa, lead + ARC_LEN / 2.0)
    uphill = np.array([next(f.yaw for f in faces if np.isclose(f.slope_deg, d)) for d in rb.ramp_deg])
    off = np.degrees(np.abs(np.angle(np.exp(1j * (mid[:, 2] - uphill)))))
    assert off.max() <= policy.ramp_yaw_jitter_deg + 1e-6, off.max()
    assert (rb.ramp_s >= -ARC_LEN - _ROBOT.wheel_radius - 1e-6).all()
    base_ok, end_ok = trials_feasible(ramp_map, spec, rb.pose, rb.kappa, lead, 0.8, args.device, RobotParams())
    assert base_ok.all() and np.array_equal(end_ok, rb.endpoint_feasible)
    # a wall-like 80 deg face: trials start before the foot and run into it, no shortfall
    steep = [dataclasses.replace(ramps[0], up_deg=80.0, height=0.7)]
    steep_map = HeightMapReader(ramp_layer(steep[0], 16.0, 0.05), origin=(-8.0, -8.0), cell=0.05)
    sb = sample_ramp_trials(steep_map, ramp_faces({"ramps": [dataclasses.asdict(r) for r in steep]}),
                            spec, 32, rng, **kw)
    assert sb.shortfall == 0 and (sb.interact_dir > 0).mean() > 0.5, (sb.shortfall, sb.interact_dir)
    print(f"ramps: 400 trials, {straight:.0%} straight, max mid-arc heading off-axis {off.max():.1f} deg, "
          f"ramp_s p10/p90 {np.percentile(rb.ramp_s, 10):.2f}/{np.percentile(rb.ramp_s, 90):.2f} m, "
          f"{int((~rb.endpoint_feasible).sum())} blocked ends, {time.perf_counter() - t0:.2f}s; "
          f"80 deg face: 32/32, {int((sb.interact_dir > 0).sum())} reach it, "
          f"{int((~sb.endpoint_feasible).sum())} blocked ends")
    print("ramp checks ok")

    if args.maps_dir:
        maps_dir = pathlib.Path(args.maps_dir)
        maps_dir = maps_dir if maps_dir.is_absolute() else pathlib.Path(__file__).resolve().parents[3] / maps_dir
        print(f"\n{'map':24s} {'strategy':8s} {'up':>3s} {'down':>4s} {'short':>5s} {'blocked':>7s} "
              f"{'relief p50/p90':>15s}  time")
        for png in sorted(maps_dir.glob("*.png")):
            stem = png.with_suffix("")
            terrain = HeightMapReader.load(stem)
            meta = map_metadata(stem)
            t0 = time.perf_counter()
            s = sample_map_trials(terrain, meta, spec, args.n, rng, **kw)
            p50, p90 = np.percentile(s.arc_relief, [50, 90])
            extra = ""
            if s.strategy == "ramp" and np.isfinite(s.ramp_deg).any():
                extra = f"  slopes {np.nanmin(s.ramp_deg):.0f}-{np.nanmax(s.ramp_deg):.0f} deg"
            print(f"{stem.name:24s} {s.strategy:8s} {int((s.interact_dir > 0).sum()):3d} "
                  f"{int((s.interact_dir < 0).sum()):4d} {s.shortfall:5d} "
                  f"{int((~s.endpoint_feasible).sum()):7d} {p50:7.3f}/{p90:.3f}  "
                  f"{time.perf_counter() - t0:.1f}s{extra}")
