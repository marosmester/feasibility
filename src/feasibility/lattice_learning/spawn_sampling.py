"""Trial sampling for `generate_dataset.py`: each trial's spawn pose AND its curvature, chosen
together, by a strategy tailored to the map's category.

Three decisions, all independent of any per-map height threshold:

* **Validity -- the planner's own static settle.** A trial needs helhest_stack's settle to be
  feasible (`settle.settle_feasible`: pitch/roll envelope, residual, chassis clearance) at the
  spawn, and the patch at the arc origin to stay on mapped terrain. Whether the NOMINAL arc end
  must be settle-feasible too depends on the strategy (`dataset_config.StrategySpec.spawn_only`):
  - `uniform`/`targeted` keep the arc-end check: without it, most arcs aimed at a tall obstacle
    end where the planner says `blocked` (measured 25% end-feasible among interacting trials vs
    92% for the rest), which is not those maps' regime.
  - `ramp_up`/`ramp_down`/`edge` drop it: those maps exist to show what ostrich does when the robot drives at a
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

* **Strategies.** Which strategy runs on which map category, with which params and on what share
  of the dataset, is NOT decided here: generate_dataset.py's YAML config lists a `mix`, and
  dataset_config.py (its registry `STRATEGIES`) validates it and calls one of these per map:

  - `ramp_up` -> `sample_ramp_up_trials` (maps with a create_ramps.py sidecar `ramps` list): every
    trial drives UP a ramp face, head on, so divergence can be read against a continuous slope
    angle (`SpawnBatch.ramp_deg`). The faces come from the sidecar's `ramps` list: each rising
    face and each far side (a down-ramp, or a steep drop -- a wall-like face driven at from its
    foot). A trial picks a face, puts the arc origin's front axle at `ramp_s` ~ U(-ARC_LEN -
    wheel_radius, max(run - ARC_LEN, -direction * wheel_radius)) metres along it from the foot --
    from where the front wheel just reaches the foot at the arc end, up to the crest (that second
    bound is never the binding one going UP, see `s_event`); a 0.3 m arc after a 0.225 m warm-up
    cannot climb a long ramp from its foot, so to see the face at all the arc must also be allowed
    to START on it -- anywhere across its top width, with the heading at MID-arc along the face +-
    `yaw_jitter_deg`. A `straight_frac` share drives straight (kappa = 0), the rest on kappa ~
    U(-kappa_max, kappa_max). Every wheel contact from spawn to arc end must stay inside the
    face's top width and on that ramp's own surface (compared against the ramp rasterised on the
    terrain's grid, so bilinear interpolation of an 80 deg face matches exactly and a second ramp
    overlapping it rejects the trial). Spawns past the climb envelope fail the settle, so on a
    steep face trials start before its foot.
  - `ramp_down` -> `sample_ramp_down_trials`: the exact mirror. The entry point is the face's
    CREST and the robot heads downhill, so `ramp_s` runs from where the front wheel just reaches
    the crest at the arc end (the robot standing on the plateau) to where the arc ends at the
    foot -- or, on a face too short for that to reach the moment the front wheel rolls off the
    lip, to one wheel radius past the crest (`s_event`). Drops are faces too, so some trials drive
    off an 80 deg edge, and on those it is `s_event` that puts the drop inside the recorded window
    instead of just beyond it. The earliest spawn needs `required_platform_length` of plateau
    behind the crest; create_ramps.py's plateau floor is sized for it and
    dataset_config.check_maps refuses maps whose sidecar plateaus are shorter.
  - `edge` -> `sample_edge_trials`: arc origins uniform among cells within `band` of a height
    EDGE (a cell whose neighbour is steeper than EDGE_MIN_SLOPE_DEG -- found from the heightmap
    alone, no sidecar needed), heading at the nearest edge +- EDGE_FACING_CONE_DEG with
    probability `facing_frac`, else uniform. Exactly `round(n * interact_frac)` trials interact,
    `down_frac` of them driving down and the rest climbing up; the remainder are near an edge but
    do not meet it.
  - `uniform` -> `sample_trials(interact_frac=None)`: uniform (pose, kappa) over the whole map.
  - `targeted` -> `sample_trials(interact_frac)`: uniform proposals, exactly
    `round(n * interact_frac)` interacting.
  - `rotate_in_place` -> `sample_rotate_in_place_trials` (poles_and_walls maps, or any map with a
    height edge): the router's PIVOT primitive instead of an arc -- `v = 0, wz = +-OMEGA_NOM`
    (50/50), so the recorded ARC_DURATION_S turns exactly PIVOT_ANGLE (one heading bin), entered
    already spinning after a warm-up at the same yaw rate. The spin's origin is placed so that the
    CLOSEST wheel is within `min_clearance + band` beyond contact of the nearest pole/wall edge
    (`edge_field`, heightmap-only, same as `edge`) while every wheel stays >= `min_clearance` clear
    of it through the whole warm-up -- so the feature is met, if at all, in the recorded window,
    never before and never by a wheel resting on it. `arc_relief`/`interact_dir` are measured over
    the recorded spin (`spin_relief_signed`), `endpoint_feasible` is the settle at the pivot's end
    heading, `kappa` is NaN (no curvature) and `SpawnBatch.v`/`wz` carry the command.

  Several strategies can share one map; `concat_batches` merges their trials and SpawnBatch
  carries `strategy`/`targeted` per row.

  Every stratified strategy is exact conditional sampling from its proposal: candidates are
  labelled by `arc_relief`, and each stratum is filled uniformly from the candidates carrying its
  label. If a stratum cannot be filled within `MAX_PROPOSAL_ROUNDS`, its deficit moves to a
  fallback stratum (down -> up -> non-interacting) and is counted in `SpawnBatch.shortfall`; only
  a map with too few valid spawns at all raises. A ramp map whose faces cannot supply enough
  trials falls back to `sample_trials` (`fallback_interact_frac`) for the rest, still labelled
  with the ramp strategy but with NaN `ramp_deg`/`ramp_s`.

Where the arc starts: every trial is entered at speed after a warm-up of `lead` metres along the
same curvature (generate_dataset.py's `warmup_s * V_NOM`), so relief and the overhang check are
evaluated from the NOMINAL arc origin `integrate_arc(spawn, kappa, lead)`, not from the spawn. A
pivot's warm-up turns `OMEGA_NOM * lead / V_NOM` in place instead, so its origin is the spawn
rotated by that much.

Every SpawnBatch row carries its commanded twist `(v, wz)` -- `(V_NOM, V_NOM * kappa)` for every
arc strategy (filled in by default), `(0, +-OMEGA_NOM)` for a pivot.

Deliberately independent of `feasibility.learning`, `feasibility.grid_learning` and
`feasibility.grid_learning_2` (design.md section 11a).

Usage (smoke test -- synthetic maps on CPU, no assets; the first run compiles the settle kernels;
per-map stats on real maps come from generate_dataset.py's `run.dry_run: true`):
    python src/feasibility/lattice_learning/spawn_sampling.py
"""
from __future__ import annotations

import dataclasses
import pathlib
from typing import Callable
from typing import Iterable

import numpy as np
import yaml
from helhest.engine import RobotParams
from scipy.ndimage import distance_transform_edt

from feasibility.heightmap import HeightMapReader
from feasibility.heightmap.create_ramps import Ramp
from feasibility.heightmap.create_ramps import ramp_height
from feasibility.lattice_learning.arc import ARC_LEN
from feasibility.lattice_learning.arc import integrate_arc
from feasibility.lattice_learning.arc import OMEGA_NOM
from feasibility.lattice_learning.arc import PIVOT_ANGLE
from feasibility.lattice_learning.arc import V_NOM
from feasibility.lattice_learning.patch import patch_overhangs
from feasibility.lattice_learning.patch import PatchSpec
from feasibility.lattice_learning.patch import WHEEL_CONTACTS_LOCAL
from feasibility.lattice_learning.settle import settle_batch
from feasibility.lattice_learning.settle import settle_feasible

RAMP_MAX_FACE_DEG = 90.0  # every face is driven at, including 80 deg drops
RAMP_SIDE_MARGIN = 0.05  # m, wheel contacts stay this far inside the face's top width
RAMP_SURFACE_TOL = 0.03  # m, |terrain - that ramp alone| at a contact; catches a second ramp
# overlapping the face (and a rough base layer, which would reject every trial if enabled)
PLATFORM_MARGIN = 0.1  # m, added to the footprint in required_platform_length: heading jitter and
# curvature swing the rear wheel off the face axis

EDGE_MIN_SLOPE_DEG = 45.0  # a cell is an edge when a neighbour rises/falls steeper than this
EDGE_FACING_CONE_DEG = 45.0  # heading at the nearest edge, uniform +-

ARC_SAMPLES = 11  # poses along the relief window, ~5 cm apart (one heightmap cell)
PROPOSAL_BATCH = 8192  # (pose, kappa) candidates per round -- geometry only, cheap
MAX_PROPOSAL_ROUNDS = 64  # ~520k candidates before a stratum is declared short
SETTLE_SLACK = 2.0  # settle this many times the still-missing count per round (~70% feasible)
EDGE_SLACK = 0.2  # m, beyond patch reach + lead: covers generate_dataset's xy_jitter (0.12 m)

FOOTPRINT_ALONG = 15  # wheel_footprint samples along the wheel plane, ~5 cm apart over 0.7 m
FOOTPRINT_ACROSS = 3  # ... and across the 0.10 m tread (both sides + center)
TOUCH_TOL = 0.01  # m of rim_penetration still counted as "not touching" (bilinear noise on flat)

_ROBOT = RobotParams()


@dataclasses.dataclass
class SpawnBatch:
    pose: np.ndarray  # [n, 3] float64 spawn (x, y, yaw)
    kappa: np.ndarray  # [n] float32 arc curvature, 1/m
    arc_relief: np.ndarray  # [n] float32 m, see arc_relief()
    targeted: np.ndarray  # [n] bool, whether the trial's strategy stratified by interaction
    shortfall: int  # stratified trials requested but substituted from a fallback stratum
    proposal_interact_rate: float  # natural share of interacting proposal candidates
    strategy: np.ndarray  # [n] str: uniform | targeted | ramp_up | ramp_down | edge
    ramp_deg: np.ndarray | None = None  # [n] float32 slope of the face driven, NaN = not a ramp trial
    ramp_s: np.ndarray | None = None  # [n] float32 m, arc origin's front axle along the face from
    # its entry point in the direction of travel -- the foot for ramp_up, the crest for ramp_down
    # (negative = not on the face yet), NaN = not a ramp trial
    interact_dir: np.ndarray | None = None  # [n] int8 +1 up / -1 down / 0 not interacting.
    # The SIGN comes from a plane through three terrain samples at the arc origin, so it is
    # ill-posed once one of those samples has passed a lip -- a `ramp_down` trial already over
    # a short face's crest (`ramp_s` > 0 with `run` < ARC_LEN + wheel_radius) fits its plane
    # partly on the ground below and can read +1. Magnitude and the `interact_dir != 0` test
    # stay usable; measured 20 of 159 `ramp_down` rows on 60-81 deg faces.
    endpoint_feasible: np.ndarray | None = None  # [n] bool, static settle at the NOMINAL arc end
    v: np.ndarray | None = None  # [n] float32 m/s commanded body speed; default V_NOM (every arc)
    wz: np.ndarray | None = None  # [n] float32 rad/s commanded yaw rate; default V_NOM * kappa

    def __post_init__(self) -> None:
        n = len(self.pose)
        if self.v is None:
            self.v = np.full(n, V_NOM, dtype=np.float32)
        if self.wz is None:
            self.wz = (V_NOM * np.asarray(self.kappa, dtype=np.float64)).astype(np.float32)
        nan = np.full(n, np.nan, dtype=np.float32)
        self.ramp_deg = nan.copy() if self.ramp_deg is None else self.ramp_deg
        self.ramp_s = nan.copy() if self.ramp_s is None else self.ramp_s
        self.interact_dir = np.zeros(n, np.int8) if self.interact_dir is None else self.interact_dir
        if self.endpoint_feasible is None:
            self.endpoint_feasible = np.ones(n, dtype=bool)


def map_metadata(stem: str | pathlib.Path) -> dict:
    """A map's whole .yaml sidecar (create_maps_for_lattice_learning.py adds `category` and the
    builder's params, e.g. the `ramps` list, to HeightMapReader's own keys)."""
    return yaml.safe_load(pathlib.Path(stem).with_suffix(".yaml").read_text()) or {}


def concat_batches(batches: list[SpawnBatch], rng: np.random.Generator) -> SpawnBatch:
    """One map's trials from several strategies, as one batch in a random row order (so chunk order
    never correlates with strategy). `shortfall` sums; `proposal_interact_rate` is NaN when more
    than one batch contributes, since the rates come from different proposals."""
    batches = [b for b in batches if len(b.pose)]
    if len(batches) == 1:
        return batches[0]
    order = rng.permutation(sum(len(b.pose) for b in batches))
    rows = {
        f.name: np.concatenate([getattr(b, f.name) for b in batches])[order]
        for f in dataclasses.fields(SpawnBatch)
        if f.name not in ("shortfall", "proposal_interact_rate")
    }
    return SpawnBatch(**rows, shortfall=sum(b.shortfall for b in batches), proposal_interact_rate=float("nan"))


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
    origin = integrate_arc(spawn, kappa, lead)
    plane = contact_plane(terrain, origin)

    above = np.zeros(len(spawn))
    below = np.zeros(len(spawn))
    for s in np.linspace(lead, lead + ARC_LEN + lookahead, ARC_SAMPLES):
        px, py = body_points(integrate_arc(spawn, kappa, s), local)
        z = np.asarray(terrain.sample(px, py), dtype=np.float64)
        above = np.maximum(above, (z - plane(px, py)).max(axis=1))
        below = np.maximum(below, (plane(px, py) - z).max(axis=1))
    return np.maximum(above, below), np.where(above >= below, 1, -1).astype(np.int8)


def body_points(pose: np.ndarray, local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """([n, P] world x, [n, P] world y) of body-frame points `local` [P, 2] at `pose` [n, 3]."""
    c, s = np.cos(pose[:, 2:3]), np.sin(pose[:, 2:3])
    return (pose[:, 0:1] + c * local[:, 0] - s * local[:, 1],
            pose[:, 1:2] + s * local[:, 0] + c * local[:, 1])


def contact_plane(
    terrain: HeightMapReader, origin: np.ndarray
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """The plane through the three wheel contacts at `origin` [n, 3], as a function of world
    ([n, P] x, [n, P] y) -> [n, P] z -- the reference every relief measure is taken against."""
    ox, oy = body_points(origin, WHEEL_CONTACTS_LOCAL)
    oz = np.asarray(terrain.sample(ox, oy), dtype=np.float64)
    x0, y0 = origin[:, 0:1], origin[:, 1:2]
    A = np.stack([np.ones_like(oz), ox - x0, oy - y0], axis=-1)  # [n, 3, 3]
    coef = np.linalg.solve(A, oz[..., None])[..., 0]  # [n, 3] z = a + b dx + c dy
    return lambda px, py: coef[:, :1] + coef[:, 1:2] * (px - x0) + coef[:, 2:3] * (py - y0)


def wheel_footprint(margin: float = 0.0, robot: RobotParams | None = None) -> tuple[np.ndarray, np.ndarray]:
    """([3 * P, 2] body-frame points, [3 * P] rim height above the contact plane) sampling each
    wheel's cylinder -- the same r = wheel_radius, `wheel_width` tread both ostrich's collision
    cylinder and helhest_stack's cylinder envelope use -- as its ground-projected rectangle:
    FOOTPRINT_ALONG points along the wheel plane (body x), FOOTPRINT_ACROSS across the tread. A
    feature at along-offset u touches the rim once it rises above r - sqrt(r^2 - u^2). `margin`
    inflates the cylinder by that much in every direction (the rim profile shifted outward), so
    `rim_penetration(..., margin) <= 0` means "at least `margin` clear of the wheel"."""
    robot = robot or _ROBOT
    r = float(robot.wheel_radius)
    hw = r if robot.wheel_width is None else float(robot.wheel_width) / 2.0  # None: the sphere
    u = np.linspace(-(r + margin), r + margin, FOOTPRINT_ALONG)
    w = np.linspace(-(hw + margin), hw + margin, FOOTPRINT_ACROSS)
    uu, ww = (a.ravel() for a in np.meshgrid(u, w))
    rim = r - np.sqrt(r * r - np.minimum(np.maximum(np.abs(uu) - margin, 0.0), r) ** 2)
    points = np.concatenate([np.column_stack([cx + uu, cy + ww]) for cx, cy in WHEEL_CONTACTS_LOCAL])
    return points, np.tile(rim, len(WHEEL_CONTACTS_LOCAL))


def rim_penetration(
    terrain: HeightMapReader, origin: np.ndarray, poses: Iterable[np.ndarray], margin: float = 0.0
) -> np.ndarray:
    """[n] m -- how far the terrain rises into any wheel's (`margin`-inflated) cylinder over
    `poses` (each [n, 3]), relative to the contact plane at `origin`; <= 0 when no wheel touches."""
    local, rim = wheel_footprint(margin)
    plane = contact_plane(terrain, origin)
    worst = np.full(len(origin), -np.inf)
    for pose in poses:
        px, py = body_points(pose, local)
        z = np.asarray(terrain.sample(px, py), dtype=np.float64)
        worst = np.maximum(worst, (z - plane(px, py) - rim).max(axis=1))
    return worst


def spin_poses(origin: np.ndarray, direction: np.ndarray, angles: np.ndarray) -> Iterable[np.ndarray]:
    """`origin` [n, 3] turned in place by each of `angles` in `direction` [n] (+1 CCW / -1 CW)."""
    for a in angles:
        yield origin + np.column_stack([np.zeros((len(origin), 2)), direction * a])


def spin_relief_signed(
    terrain: HeightMapReader, origin: np.ndarray, direction: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """([n] m, [n] int8) -- `arc_relief_signed`'s pivot twin over the RECORDED spin (`origin` [n, 3]
    turning PIVOT_ANGLE in `direction` [n]), against the same contact plane at `origin`. It cannot
    reuse the arc's contact-point sampling: during a pivot the front wheels roll only ~0.1 m and
    the rear one skids ~0.2 m sideways, so a contact point never reaches a pole its wheel is
    already hitting -- what hits it is the front rim or the rear tread's side. So "above" is
    `rim_penetration` of the whole wheel cylinders (a pole the wheel sweeps into scores ~its
    height), "below" is the drop under the three contacts plus the body center, as for an arc."""
    angles = np.linspace(0.0, PIVOT_ANGLE, ARC_SAMPLES)
    above = np.maximum(rim_penetration(terrain, origin, spin_poses(origin, direction, angles)), 0.0)
    local = np.vstack([WHEEL_CONTACTS_LOCAL, [0.0, 0.0]])
    plane = contact_plane(terrain, origin)
    below = np.zeros(len(origin))
    for pose in spin_poses(origin, direction, angles):
        px, py = body_points(pose, local)
        below = np.maximum(below, (plane(px, py) - np.asarray(terrain.sample(px, py))).max(axis=1))
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
    rows: np.ndarray, rng: np.random.Generator, interact_relief: float, *, strategy: str,
    targeted: bool, **fields: object
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
        strategy=np.full(len(rows), strategy),
        targeted=np.full(len(rows), targeted),
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
    interact_frac: float | None,
    interact_relief: float,
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


def cell_centers(terrain: HeightMapReader) -> tuple[np.ndarray, np.ndarray]:
    """([ny, nx] world x, [ny, nx] world y) of every cell center, `EdgeField`'s own grid -- shared
    by `sample_edge_trials` and `sample_rotate_in_place_trials`, both of which restrict candidate
    origins to cells within reach of an edge."""
    cx = terrain.x0 + (np.arange(terrain.nx) + 0.5) * terrain.cell
    cy = terrain.y0 + (np.arange(terrain.ny) + 0.5) * terrain.cell
    return np.meshgrid(cx, cy)


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
    interact_frac: float | None,
    interact_relief: float,
    band: float,
    facing_frac: float,
    down_frac: float,
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
    CX, CY = cell_centers(terrain)
    cells = np.flatnonzero(
        (field.dist <= band) & (CX >= x_lo) & (CX <= x_hi) & (CY >= y_lo) & (CY <= y_hi)
    )
    if len(cells) == 0:
        rest = sample_trials(
            terrain, spec, n, rng, kappa_max=kappa_max, lead=lead, mu=mu, device=device,
            interact_frac=interact_frac, interact_relief=interact_relief,
            require_endpoint=require_endpoint, robot=robot,
        )
        return dataclasses.replace(rest, shortfall=n, strategy=np.full(n, "edge"))

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


def sample_rotate_in_place_trials(
    terrain: HeightMapReader,
    spec: PatchSpec,
    n: int,
    rng: np.random.Generator,
    *,
    interact_frac: float | None,
    band: float,
    min_clearance: float,
    lead: float,
    mu: float,
    device: str,
    interact_relief: float,
    require_endpoint: bool = False,
    robot: RobotParams | None = None,
    **_: object,
) -> SpawnBatch:
    """`n` pivot trials (`v = 0, wz = +-OMEGA_NOM`, direction 50/50) -- see the module docstring's
    `rotate_in_place` entry. Proposes the ORIGIN of the recorded spin (the pose the patch is taken
    at, `t0_pose` in generate_dataset.py) at a random heading, from cells within `band` of the
    farthest a wheel's rim can reach from the body origin (rear_offset + wheel_radius) of a height
    edge, and backs the spawn out by the warm-up's own rotation, `OMEGA_NOM * lead / V_NOM` (`lead`
    is the forward warm-up distance, so `lead / V_NOM` is exactly `trial.warmup_s`).

    A candidate is kept only if no wheel cylinder comes within `min_clearance` of the terrain
    above the contact plane anywhere in the warm-up (`rim_penetration`, spawn -> origin) -- the
    feature is met, if at all, in the recorded window, and no wheel starts on it -- and passes the
    checks every strategy shares: settle-feasible at the spawn, patch on the map at the origin;
    `require_endpoint` also demands a feasible settle at the pivot's end heading. Candidates are
    labelled by `spin_relief_signed > interact_relief` and, like `edge`, exactly
    `round(n * interact_frac)` interact (None: no strata); a short interacting stratum is filled
    from the non-interacting one and counted in `shortfall`. `kappa` is NaN (a pure spin has no
    curvature) -- consumers read `v`/`wz`. Unused common keywords (`kappa_max`) are swallowed by
    `**_`, so `dataset_config.sample_map_mix` can call every sampler alike."""
    if min_clearance < 0.0:
        raise ValueError(f"min_clearance must be >= 0, got {min_clearance}")
    if band <= 0.0:
        raise ValueError(f"band must be > 0, got {band}")
    if interact_frac is not None and not 0.0 <= interact_frac <= 1.0:
        raise ValueError(f"interact_frac must be in [0, 1] or None, got {interact_frac}")
    robot = robot or RobotParams()

    field = edge_field(terrain)
    x_lo, x_hi, y_lo, y_hi = sampling_bounds(terrain, spec, 0.0)
    CX, CY = cell_centers(terrain)
    rim_reach = float(np.hypot(WHEEL_CONTACTS_LOCAL[:, 0], WHEEL_CONTACTS_LOCAL[:, 1]).max()
                      + robot.wheel_radius)
    cells = np.flatnonzero(
        (field.dist <= rim_reach + band) & (CX >= x_lo) & (CX <= x_hi) & (CY >= y_lo) & (CY <= y_hi)
    )
    if len(cells) == 0:
        raise ValueError(
            "no map cell lies within reach of a height edge for sample_rotate_in_place_trials"
        )

    warmup_angles = np.linspace(0.0, OMEGA_NOM * lead / V_NOM, ARC_SAMPLES)  # spawn -> origin
    if interact_frac is None:
        need = {0: n}
    else:
        n_int = round(n * interact_frac)
        need = {1: n_int, 0: n - n_int}
    got: dict[int, list[np.ndarray]] = {k: [] for k in need}
    counts = {k: 0 for k in need}
    n_proposed = n_interacting = 0

    def fill() -> None:
        nonlocal n_proposed, n_interacting
        for _ in range(MAX_PROPOSAL_ROUNDS):
            missing = {k: need[k] - counts[k] for k in need}
            if all(miss <= 0 for miss in missing.values()):
                return
            m = PROPOSAL_BATCH
            cell = cells[rng.integers(0, len(cells), m)]
            x = CX.ravel()[cell] + rng.uniform(-0.5, 0.5, m) * terrain.cell
            y = CY.ravel()[cell] + rng.uniform(-0.5, 0.5, m) * terrain.cell
            direction = np.where(rng.uniform(size=m) < 0.5, -1.0, 1.0)
            origin = np.column_stack([x, y, rng.uniform(0.0, 2.0 * np.pi, m)])
            spawn = origin - np.column_stack([np.zeros((m, 2)), direction * warmup_angles[-1]])

            clear = rim_penetration(
                terrain, origin, spin_poses(spawn, direction, warmup_angles), margin=min_clearance
            ) <= TOUCH_TOL
            cand = np.flatnonzero(clear & ~patch_overhangs(terrain, origin, spec))
            relief, sign = spin_relief_signed(terrain, origin[cand], direction[cand])
            interacting = relief > interact_relief
            label = interacting.astype(int) if interact_frac is not None else np.zeros(len(cand), int)
            n_proposed += len(cand)
            n_interacting += int(interacting.sum())

            picked = [(k, np.flatnonzero(label == k)[: int(SETTLE_SLACK * miss) + 16])
                      for k, miss in missing.items() if miss > 0]
            sel = np.concatenate([p for _, p in picked])  # positions into `cand`
            if len(sel) == 0:
                continue
            idx = cand[sel]
            end = origin[idx] + np.column_stack([np.zeros((len(idx), 2)), direction[idx] * PIVOT_ANGLE])
            derived, residual, clearance = settle_batch(
                terrain, np.concatenate([spawn[idx], end]), mu, device
            )
            feasible = settle_feasible(derived, residual, clearance, robot)
            spawn_ok, end_ok = feasible[: len(idx)], feasible[len(idx):]
            ok = spawn_ok & end_ok if require_endpoint else spawn_ok
            row_of = {int(s): j for j, s in enumerate(sel)}
            for k, positions in picked:
                j = np.array([row_of[int(p)] for p in positions], dtype=int)
                j = j[ok[j]][: need[k] - counts[k]]
                p = sel[j]
                got[k].append(np.column_stack(
                    [spawn[idx[j]], direction[idx[j]], relief[p], sign[p], end_ok[j]]
                ))
                counts[k] += len(j)

    fill()
    shortfall = 0
    if interact_frac is not None and counts[1] < need[1]:
        shortfall = need[1] - counts[1]
        need[1], need[0] = counts[1], need[0] + shortfall
        fill()
    if sum(counts.values()) < n:
        raise ValueError(
            f"only found {sum(counts.values())}/{n} settle-feasible pivots near a height edge and "
            f"clear of it through the warm-up after {n_proposed} candidates -- the map has too "
            "little qualifying ground for sample_rotate_in_place_trials"
        )
    rows = np.concatenate([np.concatenate(v) for v in got.values() if v])
    rng.shuffle(rows)
    return SpawnBatch(
        pose=rows[:, :3],
        kappa=np.full(n, np.nan, dtype=np.float32),
        arc_relief=rows[:, 4].astype(np.float32),
        interact_dir=interaction_dir(rows[:, 4], rows[:, 5], interact_relief),
        endpoint_feasible=rows[:, 6] > 0.5,
        targeted=np.full(n, interact_frac is not None),
        shortfall=shortfall,
        proposal_interact_rate=n_interacting / max(n_proposed, 1),
        strategy=np.full(n, "rotate_in_place"),
        v=np.zeros(n, dtype=np.float32),
        wz=(rows[:, 3] * OMEGA_NOM).astype(np.float32),
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


def required_platform_length(lead: float, robot: RobotParams | None = None) -> float:
    """m -- the plateau length a `ramp_down` trial needs for the robot's whole footprint to stand on
    it at the earliest spawn. That spawn puts the arc origin's front axle ARC_LEN + wheel_radius
    before the crest (the front wheel just reaches the crest at the arc end), `lead` further back
    for the warm-up; the rear wheel's back edge is rear_offset + wheel_radius behind the front axle.
    create_ramps.RampsConfig.plateau's floor is sized from this at the default lead."""
    robot = robot or _ROBOT
    r = float(robot.wheel_radius)
    return ARC_LEN + r + lead + float(robot.rear_offset) + r + PLATFORM_MARGIN


def _sample_face_trials(
    terrain: HeightMapReader,
    faces: list[RampFace],
    spec: PatchSpec,
    n: int,
    rng: np.random.Generator,
    *,
    direction: int,
    kappa_max: float,
    lead: float,
    mu: float,
    device: str,
    straight_frac: float,
    yaw_jitter_deg: float,
    interact_relief: float,
    fallback_interact_frac: float | None,
    require_endpoint: bool = False,
    robot: RobotParams | None = None,
) -> SpawnBatch:
    """The ramp sampler for both directions. `direction` +1 drives UP a face from its foot, -1
    drives DOWN it from its crest; `s` below is measured from that entry point along the direction
    of travel, so the whole proposal is the same expression either way."""
    if not 0.0 <= straight_frac <= 1.0:
        raise ValueError(f"straight_frac must be in [0, 1], got {straight_frac}")
    robot = robot or _ROBOT
    lookahead = relief_lookahead(interact_relief)
    jitter = np.radians(yaw_jitter_deg)
    local = WHEEL_CONTACTS_LOCAL
    window = np.linspace(-lead, ARC_LEN, ARC_SAMPLES)  # spawn ... arc end, relative to the origin
    s_min = -ARC_LEN - float(robot.wheel_radius)  # front wheel just reaches the entry at arc end
    # ... and `s_event` is where the thing a face trial exists to record actually happens: driving
    # UP, the front wheel meets the face a wheel radius BEFORE the entry (its rim touches it);
    # driving DOWN, its contact rolls off the lip a wheel radius AFTER the entry. `run - ARC_LEN`
    # ("the arc ends no further than the face's far end") is never below -wheel_radius, so going up
    # it always binds and this floor changes nothing. Going down a face shorter than ARC_LEN +
    # wheel_radius -- steeper than ~47 deg on a 0.7 m ramp -- it collapses past the event and pins
    # every trial to the approach: measured over 78 such rows, the terrain under the body moved
    # 0.000 m from t0 to arc end and |e_pitch| stayed at 0.0006 rad, i.e. flat-ground negatives
    # carrying an interact_dir of -1. The spawn is still `lead` further back and still has to pass
    # the settle, so widening this cannot start the robot beyond the lip.
    s_event = -direction * float(robot.wheel_radius)
    surfaces = {f.ramp_index: ramp_alone(terrain, f.ramp) for f in faces}
    foot_x, foot_y, face_yaw, face_run, face_half_w, face_slope = (
        np.array([getattr(f, k) for f in faces])
        for k in ("foot_x", "foot_y", "yaw", "run", "half_width", "slope_deg")
    )
    entry = 0.0 if direction > 0 else 1.0  # entry point: the foot going up, the crest going down
    turn = 0.0 if direction > 0 else np.pi  # heading of travel relative to the uphill axis

    got: list[np.ndarray] = []
    count = 0
    for _ in range(MAX_PROPOSAL_ROUNDS if faces else 0):
        if count >= n:
            break
        m = PROPOSAL_BATCH
        face_idx = rng.integers(0, len(faces), m)
        fx, fy, fyaw, run = foot_x[face_idx], foot_y[face_idx], face_yaw[face_idx], face_run[face_idx]
        half_w = face_half_w[face_idx]
        c, s = np.cos(fyaw), np.sin(fyaw)  # the face's uphill axis
        ex, ey = fx + entry * run * c, fy + entry * run * s
        travel = fyaw + turn

        s0 = rng.uniform(s_min, np.maximum(run - ARC_LEN, s_event))
        t0 = rng.uniform(-half_w, half_w)
        kappa = np.where(rng.uniform(size=m) < straight_frac, 0.0, rng.uniform(-kappa_max, kappa_max, m))
        # heading along the face at MID-arc, so a curved arc bends symmetrically about the axis
        yaw0 = travel + rng.uniform(-jitter, jitter, m) - kappa * ARC_LEN / 2.0
        tc, ts = np.cos(travel), np.sin(travel)
        origin = np.column_stack([ex + s0 * tc - t0 * ts, ey + s0 * ts + t0 * tc, yaw0])

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
        slope = face_slope[face_idx[idx[keep]]]
        got.append(np.column_stack([spawn[keep], kappa[idx[keep]], slope, s0[idx[keep]], end_ok[keep]]))
        count += len(keep)

    rows = np.concatenate(got) if got else np.zeros((0, 7))
    shortfall = n - len(rows)
    pose, kap = rows[:, :3], rows[:, 3].astype(np.float32)
    ramp_deg, ramp_s = rows[:, 4].astype(np.float32), rows[:, 5].astype(np.float32)
    end_feasible = rows[:, 6] > 0.5
    if len(rows):
        relief, sign = arc_relief_signed(terrain, pose, rows[:, 3], lead, lookahead)
        interact = interaction_dir(relief, sign, interact_relief)
    else:
        relief, interact = np.zeros(0), np.zeros(0, dtype=np.int8)
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
        interact = np.concatenate([interact, rest.interact_dir])
        end_feasible = np.concatenate([end_feasible, rest.endpoint_feasible])
        ramp_deg, ramp_s = np.concatenate([ramp_deg, nan]), np.concatenate([ramp_s, nan])
    order = rng.permutation(n)
    return SpawnBatch(
        pose=pose[order],
        kappa=kap[order],
        arc_relief=relief[order].astype(np.float32),
        targeted=np.ones(n, dtype=bool),
        shortfall=shortfall,
        proposal_interact_rate=float("nan"),
        strategy=np.full(n, "ramp_up" if direction > 0 else "ramp_down"),
        ramp_deg=ramp_deg[order],
        ramp_s=ramp_s[order],
        interact_dir=interact[order],
        endpoint_feasible=end_feasible[order],
    )


def sample_ramp_up_trials(
    terrain: HeightMapReader, faces: list[RampFace], spec: PatchSpec, n: int,
    rng: np.random.Generator, **kw: object,
) -> SpawnBatch:
    """`n` trials driving head-on UP the given faces -- see the module docstring. Faces are picked
    uniformly per candidate. If too few candidates survive (no faces, or every face too narrow or
    too crowded for a feasible spawn), the remainder is filled by `sample_trials` with
    `fallback_interact_frac` and counted in `shortfall`. Keywords: `_sample_face_trials`'s."""
    return _sample_face_trials(terrain, faces, spec, n, rng, direction=1, **kw)


def sample_ramp_down_trials(
    terrain: HeightMapReader, faces: list[RampFace], spec: PatchSpec, n: int,
    rng: np.random.Generator, **kw: object,
) -> SpawnBatch:
    """`n` trials driving head-on DOWN the given faces from their crest -- the mirror of
    `sample_ramp_up_trials`, drops included (driving off an 80 deg drop). The earliest spawn stands
    on the plateau, which must be `required_platform_length` long for the whole robot to fit."""
    return _sample_face_trials(terrain, faces, spec, n, rng, direction=-1, **kw)


if __name__ == "__main__":
    import argparse
    import time

    import warp as wp

    from feasibility.lattice_learning.arc import KAPPA_MAX
    from feasibility.lattice_learning.arc import V_NOM

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    wp.init()
    spec = PatchSpec()
    lead = 0.375 * V_NOM  # configs/default.yaml's trial.warmup_s
    relief_thr = 0.05  # configs/default.yaml's trial.interact_relief
    kw = dict(kappa_max=KAPPA_MAX, lead=lead, mu=0.8, device=args.device, interact_relief=relief_thr)
    # configs/default.yaml's mix params
    edge_kw = dict(interact_frac=0.5, band=1.5, facing_frac=0.7, down_frac=0.5)
    ramp_kw = dict(straight_frac=0.75, yaw_jitter_deg=5.0, fallback_interact_frac=0.5)
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
    assert fb.strategy.shape == (32,) and (fb.strategy == "targeted").all() and fb.targeted.all()

    # 0.2 m box: exact stratification, patch at the arc origin on the map, spawns AND arc ends
    # settle-feasible (including some spawns on the box top).
    box_h = 0.2
    box = HeightMapReader(np.where((np.abs(X) < 1.5) & (np.abs(Y) < 1.5), box_h, 0.0),
                          origin=(-8.0, -8.0), cell=0.05)
    t0 = time.perf_counter()
    b = sample_trials(box, spec, 200, rng, interact_frac=0.5, **kw)
    interacts = b.arc_relief > relief_thr
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
    eb = sample_edge_trials(wall_map, spec, 200, rng, **edge_kw, **kw)
    n_up, n_down = int((eb.interact_dir > 0).sum()), int((eb.interact_dir < 0).sum())
    assert (eb.strategy == "edge").all() and eb.shortfall == 0, eb.shortfall
    assert n_up + n_down == 100 and n_down == 50, (n_up, n_down)
    field = edge_field(wall_map)
    origin = integrate_arc(eb.pose, eb.kappa, lead)
    col = np.clip(((origin[:, 0] - wall_map.x0) / wall_map.cell).astype(int), 0, wall_map.nx - 1)
    row = np.clip(((origin[:, 1] - wall_map.y0) / wall_map.cell).astype(int), 0, wall_map.ny - 1)
    assert (field.dist[row, col] <= edge_kw["band"] + 1e-9).all()
    base_ok, end_ok = trials_feasible(wall_map, spec, eb.pose, eb.kappa, lead, 0.8, args.device, RobotParams())
    assert base_ok.all() and np.array_equal(end_ok, eb.endpoint_feasible)
    assert (~eb.endpoint_feasible).any(), "spawn-only edge trials should include blocked arc ends"
    down_on_box = (np.abs(eb.pose[eb.interact_dir < 0, 0] + 2.5) < 1.25).mean()
    print(f"edges: {n_up} up / {n_down} down / {200 - n_up - n_down} near-miss (natural interacting "
          f"rate {eb.proposal_interact_rate:.1%}), {int((~eb.endpoint_feasible).sum())} blocked arc "
          f"ends, {down_on_box:.0%} of down spawns on the box, {time.perf_counter() - t0:.2f}s")
    print("edge checks ok")

    # Rotate-in-place: a thin wall + a small pole stand-in -- poles_and_walls features never have
    # an interior point farther than ~0.15 m from their own edge, unlike the box above (half-
    # extent 1.25 m), so this needs its own map to test the "never on the feature" guarantee for
    # real. The command is the pivot primitive in both directions (kappa NaN); no wheel cylinder
    # comes within min_clearance of the feature anywhere in the warm-up (checked on a finer yaw grid
    # than the sampler's); exactly interact_frac of the recorded spins sweep a wheel into it, and
    # their stored relief re-derives from the spin origin; spawns are settle-feasible; reproduces.
    t0 = time.perf_counter()
    thin_wall = np.where((np.abs(X - 2.5) < 0.1) & (np.abs(Y) < 2.0), 1.0, 0.0)  # 0.2 m thick
    thin_pole = np.where((np.abs(X + 2.0) < 0.15) & (np.abs(Y - 1.0) < 0.15), 0.5, 0.0)  # ~0.3 m square
    thin_map = HeightMapReader(np.maximum(thin_wall, thin_pole), origin=(-8.0, -8.0), cell=0.05)
    rot_kw = dict(interact_frac=0.5, band=0.3, min_clearance=0.05, lead=lead, mu=0.8,
                  device=args.device, interact_relief=relief_thr,
                  kappa_max=KAPPA_MAX)  # kappa_max: swallowed, as sample_map_mix passes it
    rb2 = sample_rotate_in_place_trials(thin_map, spec, 200, rng, **rot_kw)
    assert (rb2.strategy == "rotate_in_place").all() and rb2.shortfall == 0, rb2.shortfall
    assert np.isnan(rb2.kappa).all() and (rb2.v == 0).all()
    direction = np.sign(rb2.wz)
    assert np.allclose(np.abs(rb2.wz), OMEGA_NOM) and 0 < (direction > 0).sum() < 200
    assert int((rb2.interact_dir != 0).sum()) == 100
    warmup_yaw = OMEGA_NOM * lead / V_NOM
    origin = rb2.pose + np.column_stack([np.zeros((200, 2)), direction * warmup_yaw])
    fine = np.linspace(0.0, warmup_yaw, 4 * ARC_SAMPLES)
    warm = rim_penetration(thin_map, origin, spin_poses(rb2.pose, direction, fine), margin=0.0)
    assert (warm <= TOUCH_TOL).all(), f"a wheel touches the feature in warm-up by {warm.max():.3f} m"
    for pose in spin_poses(rb2.pose, direction, fine):
        wx, wy = body_points(pose, WHEEL_CONTACTS_LOCAL)
        assert (np.asarray(thin_map.sample(wx, wy)) < 1e-6).all(), "a wheel contact on the feature"
    relief, _ = spin_relief_signed(thin_map, origin, direction)
    assert np.allclose(relief, rb2.arc_relief, atol=1e-5)
    d, r, c = settle_batch(thin_map, rb2.pose, 0.8, args.device)
    assert settle_feasible(d, r, c, RobotParams()).all()
    same = [sample_rotate_in_place_trials(thin_map, spec, 16, np.random.default_rng(5), **rot_kw)
            for _ in range(2)]
    assert np.array_equal(same[0].pose, same[1].pose) and np.array_equal(same[0].wz, same[1].wz)
    mixed_rot = concat_batches([rb2, fb], rng)
    assert np.isnan(mixed_rot.kappa).sum() == 200 and (mixed_rot.v == 0).sum() == 200
    assert np.allclose(mixed_rot.wz[mixed_rot.v > 0], V_NOM * mixed_rot.kappa[mixed_rot.v > 0])
    meet = rb2.interact_dir != 0
    print(f"rotate_in_place: 200 pivots ({int((direction > 0).sum())} CCW), 100 sweep a wheel into "
          f"the feature (natural rate {rb2.proposal_interact_rate:.1%}, median relief "
          f"{np.median(rb2.arc_relief[meet]):.2f} m), max warm-up penetration {warm.max():+.3f} m, "
          f"{int((~rb2.endpoint_feasible).sum())} blocked end headings, {time.perf_counter() - t0:.2f}s")
    print("rotate-in-place checks ok")

    # Ramps, both directions: every trial head-on along a face, straight share ~ straight_frac,
    # mid-arc heading within the jitter of the face axis (uphill / downhill), wheel contacts on
    # that face's surface. Plateaus are create_ramps.RampsConfig's standing-platform floor.
    from feasibility.heightmap.create_ramps import ramp_layer

    platform = required_platform_length(lead)
    assert 2.0 < platform <= 2.1, platform  # create_ramps.RampsConfig.plateau[0] = 2.1 is sized on it
    ramps = [
        Ramp(cx=-3.0, cy=-2.0, yaw=0.3, up_deg=12.0, height=0.5, plateau=2.1, down_deg=80.0,
             width=1.6, side_deg=80.0),
        Ramp(cx=2.5, cy=2.5, yaw=2.0, up_deg=20.0, height=0.4, plateau=2.1, down_deg=8.0,
             width=2.0, side_deg=80.0),
    ]
    ramp_map = HeightMapReader(np.maximum(*(ramp_layer(r, 16.0, 0.05) for r in ramps)),
                               origin=(-8.0, -8.0), cell=0.05)
    meta = {"category": "ramps", "ramps": [dataclasses.asdict(r) for r in ramps]}
    faces = ramp_faces(meta)
    assert sorted(round(f.slope_deg) for f in faces) == [8, 12, 20, 80], "drop is a face too"
    for sampler, name, turn in ((sample_ramp_up_trials, "ramp_up", 0.0),
                                (sample_ramp_down_trials, "ramp_down", np.pi)):
        t0 = time.perf_counter()
        rb = sampler(ramp_map, faces, spec, 400, rng, **ramp_kw, **kw)
        assert (rb.strategy == name).all() and rb.shortfall == 0 and np.isfinite(rb.ramp_deg).all(), \
            (name, rb.shortfall)
        straight = (rb.kappa == 0).mean()
        assert abs(straight - ramp_kw["straight_frac"]) < 0.1, (name, straight)
        assert set(np.round(rb.ramp_deg).tolist()) == {8.0, 12.0, 20.0, 80.0}, name
        mid = integrate_arc(rb.pose, rb.kappa, lead + ARC_LEN / 2.0)
        axis = np.array([next(f.yaw for f in faces if np.isclose(f.slope_deg, d)) for d in rb.ramp_deg])
        off = np.degrees(np.abs(np.angle(np.exp(1j * (mid[:, 2] - axis - turn)))))
        assert off.max() <= ramp_kw["yaw_jitter_deg"] + 1e-6, (name, off.max())
        assert (rb.ramp_s >= -ARC_LEN - _ROBOT.wheel_radius - 1e-6).all()
        base_ok, end_ok = trials_feasible(ramp_map, spec, rb.pose, rb.kappa, lead, 0.8, args.device,
                                          RobotParams())
        assert base_ok.all() and np.array_equal(end_ok, rb.endpoint_feasible)
        n_up, n_down = int((rb.interact_dir > 0).sum()), int((rb.interact_dir < 0).sum())
        if name == "ramp_down":
            # the crest drops away below the wheel-contact plane; only arcs starting ON a face
            # and reaching its foot read +1 there
            assert n_down > n_up, (n_up, n_down)
            before_crest = rb.ramp_s < 0
            assert before_crest.any() and (rb.interact_dir[before_crest] <= 0).mean() > 0.9
        print(f"{name}: 400 trials, {straight:.0%} straight, max mid-arc heading off-axis "
              f"{off.max():.1f} deg, ramp_s p10/p90 {np.percentile(rb.ramp_s, 10):.2f}/"
              f"{np.percentile(rb.ramp_s, 90):.2f} m, interact {n_up} up / {n_down} down, "
              f"{int((~rb.endpoint_feasible).sum())} blocked ends, {time.perf_counter() - t0:.2f}s")
    # a wall-like 80 deg face: trials start before the foot and run into it, no shortfall; driven
    # down, the same face is an 0.7 m drop off a standing platform
    steep = [dataclasses.replace(ramps[0], cx=0.0, cy=0.0, up_deg=80.0, height=0.7)]
    steep_map = HeightMapReader(ramp_layer(steep[0], 16.0, 0.05), origin=(-8.0, -8.0), cell=0.05)
    steep_faces = ramp_faces({"ramps": [dataclasses.asdict(r) for r in steep]})
    sb = sample_ramp_up_trials(steep_map, steep_faces, spec, 32, rng, **ramp_kw, **kw)
    assert sb.shortfall == 0 and (sb.interact_dir > 0).mean() > 0.5, (sb.shortfall, sb.interact_dir)
    # Driven down, the same face is an 0.7 m drop off a standing platform. `s_event` is what
    # lets the arc reach the moment the front wheel rolls off it, so trials past the crest must
    # exist -- and on those rows interact_dir's SIGN is ill-posed (see the field's comment), so
    # assert it only where the whole contact plane is still on the platform.
    db = sample_ramp_down_trials(steep_map, steep_faces, spec, 32, rng, **ramp_kw, **kw)
    over = db.ramp_s > 0.0
    assert db.shortfall == 0 and over.any(), (db.shortfall, db.ramp_s.max())
    assert (db.interact_dir[~over] <= 0).mean() > 0.9, db.interact_dir[~over]
    print(f"80 deg face: up 32/32, {int((sb.interact_dir > 0).sum())} reach it, "
          f"{int((~sb.endpoint_feasible).sum())} blocked ends; down 32/32, "
          f"{int(over.sum())} start past the crest, "
          f"{int((db.interact_dir < 0).sum())} read a drop, "
          f"{int((~db.endpoint_feasible).sum())} blocked ends")

    mixed = concat_batches([fb, eb], rng)
    assert len(mixed.pose) == 232 and set(mixed.strategy) == {"targeted", "edge"}
    assert (mixed.strategy == "edge").sum() == 200 and mixed.shortfall == fb.shortfall + eb.shortfall
    print("ramp / concat checks ok")
