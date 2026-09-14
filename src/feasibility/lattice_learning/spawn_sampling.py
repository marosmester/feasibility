"""Trial sampling for `generate_dataset.py`: each trial's spawn pose AND its curvature, chosen
together so a set share of trials has its arc actually meet terrain.

Two decisions, both independent of any per-map height threshold:

* **Validity -- the planner's own static settle.** A trial is valid iff helhest_stack's settle is
  feasible (`settle.settle_feasible`: pitch/roll envelope, residual, chassis clearance) both at the
  spawn and at the NOMINAL arc end, and the patch at the arc origin stays on mapped terrain. The
  arc-end check anticipates generate_dataset.py's own `valid` filter (an infeasible endpoint is
  `blocked` for the planner, so its row is never a training target): without it, most arcs aimed
  at a tall wall were simulated only to be dropped (measured 25% valid among interacting trials vs
  92% for the rest). What remains is exactly the dataset's regime -- terrain the settle calls
  fine that the arc still meets. The spawn check replaces the older
  "footprint below median + 15% of the relief" filter, which forbade every ramp face and plateau
  (so a ramp's top kink and side drop-offs were unreachable in a 0.3 m arc) and blocked ~30% of a
  rough map over centimetre bumps. Trials can now start on ramps, box tops, curbs, rough ground.

* **Targeting -- by what the arc meets, not by where the robot stands.** `arc_relief` measures how
  far the terrain under the robot departs, over the recorded arc, from the plane through its three
  wheel contacts at the arc's origin. A constant slope scores ~0 (the static settle already
  handles it); a step, wall, kink, drop or bump under the wheels or the body scores its height.
  A trial `interacts` when `arc_relief > interact_relief`. On a targeted map, exactly
  `round(n * interact_frac)` trials are drawn uniformly from the interacting (pose, kappa) pairs
  and the rest uniformly from the non-interacting ones -- exact conditional sampling from the
  uniform proposal, so there is no band or yaw heuristic to tune, and the knob reads the same on
  boxes, walls and ramps.

  Untargeted maps (`interact_frac=None`, or a map whose sidecar `category` is in
  `untargeted_categories`, default `rough` -- flat and low-amplitude rough ground) keep plain
  uniform (pose, kappa) draws, still settle-validated and still labelled with `arc_relief`.

  If a targeted map cannot supply enough interacting trials within `MAX_PROPOSAL_ROUNDS`
  (a map with almost no features), the shortfall is filled with non-interacting trials and
  reported in `SpawnBatch.shortfall`; only a map with too few valid spawns at all raises.

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

import numpy as np
import yaml
from helhest.engine import RobotParams

from feasibility.heightmap import HeightMapReader
from feasibility.lattice_learning.arc import ARC_LEN
from feasibility.lattice_learning.arc import integrate_arc
from feasibility.lattice_learning.patch import patch_overhangs
from feasibility.lattice_learning.patch import PatchSpec
from feasibility.lattice_learning.patch import WHEEL_CONTACTS_LOCAL
from feasibility.lattice_learning.settle import settle_batch
from feasibility.lattice_learning.settle import settle_feasible

DEFAULT_INTERACT_FRAC: float | None = 0.5  # share of a targeted map's trials whose arc meets
# terrain; None = untargeted everywhere (plain uniform draws)
DEFAULT_INTERACT_RELIEF = 0.05  # m -- the lower end of create_maps_for_lattice_learning.py's
# obstacle heights (0.05-0.5 m), so the lowest curb it generates counts as an interaction
DEFAULT_UNTARGETED_CATEGORIES = ("rough",)  # sidecar categories sampled uniformly: flat and
# low-amplitude rough ground are negatives, there is nothing on them to aim at

ARC_SAMPLES = 11  # poses along the relief window, ~5 cm apart (one heightmap cell)
PROPOSAL_BATCH = 8192  # uniform (pose, kappa) candidates per round -- geometry only, cheap
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
    shortfall: int  # interacting trials requested but not found (filled non-interacting)
    proposal_interact_rate: float  # natural share of interacting uniform candidates


def map_category(stem: str | pathlib.Path) -> str | None:
    """The `category` key a create_maps_for_lattice_learning.py sidecar carries, or None for maps
    from other generators (which are then targeted)."""
    meta = yaml.safe_load(pathlib.Path(stem).with_suffix(".yaml").read_text())
    return meta.get("category")


def relief_lookahead(interact_relief: float) -> float:
    """m -- how far ahead of its contact point a wheel of radius r first touches a step of height
    h: sqrt(2 r h - h^2). Extends the relief window past the arc's end so a step the front wheels
    are about to hit counts, without counting ones they never reach."""
    r, h = float(_ROBOT.wheel_radius), min(interact_relief, float(_ROBOT.wheel_radius))
    return float(np.sqrt(2.0 * r * h - h * h))


def arc_relief(
    terrain: HeightMapReader, spawn: np.ndarray, kappa: np.ndarray, lead: float, lookahead: float
) -> np.ndarray:
    """[n] m -- max |terrain - plane| over the arc, where the plane passes through the three wheel
    contacts at the nominal arc origin `integrate_arc(spawn, kappa, lead)` and the terrain is
    sampled at the three wheel contacts plus the body center at ARC_SAMPLES poses from that origin
    to `ARC_LEN + lookahead` further along the same curvature. Plane-relative, so a uniform slope
    is ~0 while steps, kinks, drops and bumps under wheels or body register their height."""
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

    relief = np.zeros(len(spawn))
    for s in np.linspace(lead, lead + ARC_LEN + lookahead, ARC_SAMPLES):
        px, py = contacts(integrate_arc(spawn, kappa, s))
        z = np.asarray(terrain.sample(px, py), dtype=np.float64)
        plane = coef[:, :1] + coef[:, 1:2] * (px - x0) + coef[:, 2:3] * (py - y0)
        relief = np.maximum(relief, np.abs(z - plane).max(axis=1))
    return relief


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
    robot: RobotParams | None = None,
) -> SpawnBatch:
    """`n` (spawn pose, kappa) trials on `terrain` -- see the module docstring. `interact_frac`
    None samples uniformly; a float in [0, 1] stratifies by `arc_relief > interact_relief`.
    Per-category opt-out is the caller's decision (pass None), see `map_category`."""
    if interact_frac is not None and not 0.0 <= interact_frac <= 1.0:
        raise ValueError(f"interact_frac must be in [0, 1] or None, got {interact_frac}")
    robot = robot or RobotParams()
    lookahead = relief_lookahead(interact_relief)
    margin = spec.reach + lead + EDGE_SLACK
    x_lo, x_hi = terrain.x0 + margin, terrain.x0 + terrain.nx * terrain.cell - margin
    y_lo, y_hi = terrain.y0 + margin, terrain.y0 + terrain.ny * terrain.cell - margin
    if x_hi <= x_lo or y_hi <= y_lo:
        raise ValueError(
            f"terrain is too small for a patch reach of {spec.reach:.2f} m plus a {lead:.2f} m "
            f"lead (extent {terrain.nx * terrain.cell:.2f} x {terrain.ny * terrain.cell:.2f} m)"
        )

    # stratum -> trials still needed; None = "any", True/False = interacting or not
    if interact_frac is None:
        need: dict[bool | None, int] = {None: n}
    else:
        n_int = round(n * interact_frac)
        need = {True: n_int, False: n - n_int}
    got: dict[bool | None, list[np.ndarray]] = {k: [] for k in need}
    counts = {k: 0 for k in need}
    n_proposed = n_interacting = 0

    def fill(rounds: int) -> None:
        nonlocal n_proposed, n_interacting
        for _ in range(rounds):
            missing = {k: need[k] - counts[k] for k in need}
            if all(m <= 0 for m in missing.values()):
                return
            m = PROPOSAL_BATCH
            cand = np.column_stack([
                rng.uniform(x_lo, x_hi, m), rng.uniform(y_lo, y_hi, m),
                rng.uniform(0.0, 2.0 * np.pi, m), rng.uniform(-kappa_max, kappa_max, m),
            ])
            relief = arc_relief(terrain, cand[:, :3], cand[:, 3], lead, lookahead)
            interacts = relief > interact_relief
            n_proposed += m
            n_interacting += int(interacts.sum())

            picked = []
            for key, miss in missing.items():
                if miss <= 0:
                    continue
                idx = np.arange(m) if key is None else np.flatnonzero(interacts == key)
                picked.append((key, idx[: int(SETTLE_SLACK * miss) + 16]))
            all_idx = np.concatenate([idx for _, idx in picked])
            if len(all_idx) == 0:
                continue
            k = len(all_idx)
            origin = integrate_arc(cand[all_idx, :3], cand[all_idx, 3], lead)
            arc_end = integrate_arc(origin, cand[all_idx, 3], ARC_LEN)
            derived, residual, clearance = settle_batch(
                terrain, np.concatenate([cand[all_idx, :3], arc_end]), mu, device
            )
            feasible = settle_feasible(derived, residual, clearance, robot)
            ok_all = feasible[:k] & feasible[k:] & ~patch_overhangs(terrain, origin, spec)
            ok_by_index = dict(zip(all_idx.tolist(), ok_all.tolist()))
            for key, idx in picked:
                keep = idx[[ok_by_index[i] for i in idx.tolist()]][: need[key] - counts[key]]
                rows = np.column_stack([cand[keep], relief[keep]])
                got[key].append(rows)
                counts[key] += len(rows)

    fill(MAX_PROPOSAL_ROUNDS)
    shortfall = 0
    if interact_frac is not None and counts[True] < need[True]:
        shortfall = need[True] - counts[True]
        need[True] = counts[True]
        need[False] += shortfall
        fill(MAX_PROPOSAL_ROUNDS)
    total = sum(counts.values())
    if total < n:
        raise ValueError(
            f"sample_trials: only found {total}/{n} settle-feasible trials after "
            f"{n_proposed} proposals -- the map has too little drivable ground in its "
            f"{x_hi - x_lo:.1f} x {y_hi - y_lo:.1f} m sampling square"
        )

    rows = np.concatenate([np.concatenate(v) for v in got.values() if v])
    rng.shuffle(rows)  # so chunk order never correlates with which stratum a trial came from
    return SpawnBatch(
        pose=rows[:, :3],
        kappa=rows[:, 3].astype(np.float32),
        arc_relief=rows[:, 4].astype(np.float32),
        targeted=interact_frac is not None,
        shortfall=shortfall,
        proposal_interact_rate=n_interacting / max(n_proposed, 1),
    )


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

    # 0.2 m box: exact stratification, patch at the arc origin on the map, spawns settle-feasible
    # (including some on the box top, which the old height-threshold filter forbade).
    box_h = 0.2
    box = HeightMapReader(np.where((np.abs(X) < 1.5) & (np.abs(Y) < 1.5), box_h, 0.0),
                          origin=(-8.0, -8.0), cell=0.05)
    t0 = time.perf_counter()
    b = sample_trials(box, spec, 200, rng, interact_frac=0.5, **kw)
    interacts = b.arc_relief > DEFAULT_INTERACT_RELIEF
    assert interacts.sum() == 100 and b.shortfall == 0, (interacts.sum(), b.shortfall)
    assert not patch_overhangs(box, integrate_arc(b.pose, b.kappa, lead), spec).any()
    arc_end = integrate_arc(integrate_arc(b.pose, b.kappa, lead), b.kappa, ARC_LEN)
    d, r, c = settle_batch(box, np.concatenate([b.pose, arc_end]), 0.8, args.device)
    assert settle_feasible(d, r, c, RobotParams()).all()
    on_top = (np.abs(b.pose[:, 0]) < 1.0) & (np.abs(b.pose[:, 1]) < 1.0)
    assert on_top.any(), "a 0.2 m box top is drivable ground and must be spawnable"
    print(f"box: 100/200 interacting (natural rate {b.proposal_interact_rate:.1%}), "
          f"{int(on_top.sum())} spawns on the box top, {time.perf_counter() - t0:.2f}s")

    same = [sample_trials(box, spec, 16, np.random.default_rng(3), interact_frac=0.25, **kw) for _ in range(2)]
    assert np.array_equal(same[0].pose, same[1].pose) and np.array_equal(same[0].kappa, same[1].kappa)
    print("slope/flat/box/reproducibility checks ok")

    if args.maps_dir:
        maps_dir = pathlib.Path(args.maps_dir)
        maps_dir = maps_dir if maps_dir.is_absolute() else pathlib.Path(__file__).resolve().parents[3] / maps_dir
        # uniform = interacting share of a plain settle-validated uniform draw, i.e. what targeting
        # changes; interact = the share actually delivered with the default policy
        print(f"\n{'map':14s} {'uniform':>7s} {'interact':>8s} {'short':>5s} {'relief p50/p90':>15s}  time")
        for png in sorted(maps_dir.glob("*.png")):
            stem = png.with_suffix("")
            terrain = HeightMapReader.load(stem)
            cat = map_category(stem)
            frac = None if cat in DEFAULT_UNTARGETED_CATEGORIES else DEFAULT_INTERACT_FRAC
            u = sample_trials(terrain, spec, args.n, rng, interact_frac=None, **kw)
            t0 = time.perf_counter()
            s = sample_trials(terrain, spec, args.n, rng, interact_frac=frac, **kw)
            p50, p90 = np.percentile(s.arc_relief, [50, 90])
            print(f"{stem.name:14s} {(u.arc_relief > DEFAULT_INTERACT_RELIEF).mean():7.1%} "
                  f"{(s.arc_relief > DEFAULT_INTERACT_RELIEF).mean():8.1%} {s.shortfall:5d} "
                  f"{p50:7.3f}/{p90:.3f}  {time.perf_counter() - t0:.1f}s{'' if s.targeted else '  (untargeted)'}")
