"""What the regression harness simulates: three terrains x a handful of (spawn, twist) trials.

Terrain is BUILT IN MEMORY, never loaded from assets/ -- `assets` is gitignored, so a baseline
recorded against an asset file could not be reproduced from a fresh clone (and a regenerated
asset series would silently shift the baseline's meaning). Every builder used here returns a
HeightMapReader directly and is a pure function of its arguments, so the same tier always gets
bit-identical terrain.

The three scenarios probe different failure modes:

  flat   both sims should agree closely -- it is the control. Divergence here is a red flag, not
         a measurement.
  box    the lateral-collision case comparator/compare_box_obstacles.py exists to measure: the
         rear wheel sweeps into the box's side, which helhest_stack cannot represent at all.
         Divergence is EXPECTED and large; what the baseline pins is that it stays the size it is.
  rough  the solver stress case -- band-limited random ground is where ostrich's contact solve is
         most likely to struggle.

Two tiers: `smoke` (short, three trials each -- run it after every submodule bump) and `full`
(longer, more trials, a second box height and a second rough seed -- run it before trusting a
pointer update).

Run `python src/feasibility/submodule_test/scenarios.py` to print the scenario table and each
terrain's grid shape and z-range. No GPU needed; it is this module's smoke test.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Callable

import numpy as np

from examples.helhest_junior.common import HelhestJuniorConfig

from feasibility.comparator.common import Trial
from feasibility.heightmap import HeightMapReader
from feasibility.heightmap.create_box_obstacles import build_centered_box
from feasibility.heightmap.create_box_obstacles import BOX_SIZE
from feasibility.heightmap.create_box_obstacles import DEFAULT_CELL
from feasibility.heightmap.create_box_obstacles import DEFAULT_INCLINE_DEG
from feasibility.heightmap.create_rough_terrain import build_rough_terrain

# Grid extents. Sized so the longest trial (full tier: v_drive=1.0 held for FULL_DURATION_S)
# finishes well inside the mapped area -- HeightMapReader.sample CLAMPS outside the grid rather
# than raising, so an under-sized map fabricates flat ground with no error and the run silently
# stops testing what it claims to.
FLAT_EXTENT = 12.0  # m, square, origin-centered -- smoke tier
FLAT_EXTENT_FULL = 16.0  # m -- the full tier drives twice as long, so it needs twice the room
BOX_EXTENT = 12.0  # m -- wider than create_box_obstacles' own DEFAULT_CENTERED_EXTENT (10.0) for
# exactly the reason above: a straight-drive trial that is NOT stopped by the box needs the room.
BOX_EXTENT_FULL = 16.0  # m
ROUGH_EXTENT = 16.0  # m, create_rough_terrain's own default -- already big enough for both tiers

SMOKE_DURATION_S = 4.0
FULL_DURATION_S = 8.0

BOX_HEIGHT = 0.50  # m, the smoke tier's box -- mid-range of create_box_obstacles' BOX_HEIGHTS
BOX_HEIGHT_TALL = 0.70  # m, the full tier's second box
ROUGH_SEED = 0
ROUGH_SEED_ALT = 7  # full tier's second roughness realization

# Turn-in-place rate and drive speed, matching the comparator scenarios these trials are modeled
# on (compare_box_obstacles.py's wz_drive=-0.8, compare_on_surface.py's V_DRIVE=1.0).
V_DRIVE = 1.0
WZ_TURN = 0.8

# Clearance between the robot's spawn and the box footprint's corner, so it starts beside the box
# rather than on its ramp -- compare_box_obstacles.py's CORNER_MARGIN, same value and same intent.
CORNER_MARGIN = 0.35
_BOX_HALF = BOX_SIZE / 2.0

# How far past its commanded travel a final pose may sit before the run counts as blown up rather
# than merely surprising. 1.5x leaves room for a real downhill run-out or a collision shove; the
# +1.0 m floor is what makes it meaningful for a turn in place, where commanded travel is ZERO --
# the same reasoning (and roughly the same number) as grid_learning/generate_dataset.py's
# MAX_SPAWN_DISPLACEMENT, whose comment records ostrich turn-in-place displacement as median
# 0.11 m / p90 0.20 m with a 2.75% tail running out to 8.6 m.
DISPLACEMENT_SLACK = 1.5
DISPLACEMENT_FLOOR = 1.0

# How far outside the body origin the robot's own geometry reaches, used when checking that a
# trial stays on mapped terrain. grid_learning/generate_dataset.py sizes its spawn padding off
# the rear wheel's ~1.45 m rim reach during a turn in place; 1.6 m adds slack for the settle drop
# and a collision shove.
ROBOT_REACH = 1.6

# Terrain height spread across the robot's footprint above which a spawn is unsupportable -- see
# footprint_relief. One wheel radius: at that spread a wheel is a full radius off the ground while
# its neighbours are down, so there is no rest pose for ostrich to settle into.
MAX_SPAWN_RELIEF = float(HelhestJuniorConfig.WHEEL_RADIUS)

# [3, 2] body-frame (x, y) of the three wheel contacts, taken from the robot's own config rather
# than hardcoded so the spawn check cannot drift from the model it is checking for.
WHEEL_CONTACTS_LOCAL = np.array(
    [
        [float(p[0]), float(p[1])]
        for p in (
            HelhestJuniorConfig.LEFT_WHEEL_POS,
            HelhestJuniorConfig.RIGHT_WHEEL_POS,
            HelhestJuniorConfig.REAR_WHEEL_POS,
        )
    ]
)


@dataclass(frozen=True)
class Scenario:
    """One terrain plus every (spawn pose, commanded twist) replayed on it. All trials share
    `duration_s` so their rollouts stack into the [T, N, ...] arrays both batch runners and
    comparator/provenance.py's HDF5 schema expect."""

    name: str
    terrain_fn: Callable[[], HeightMapReader]
    trials: list[Trial]
    duration_s: float

    def max_displacement(self) -> list[float]:
        """[n_trials] how far each trial's final pose may legitimately sit from its own spawn."""
        return [
            abs(t.v_drive) * self.duration_s * DISPLACEMENT_SLACK + DISPLACEMENT_FLOOR
            for t in self.trials
        ]


def flat_terrain(extent: float = FLAT_EXTENT) -> HeightMapReader:
    half = extent / 2.0
    return HeightMapReader.flat(xlim=(-half, half), ylim=(-half, half), cell=DEFAULT_CELL)


def box_terrain(height: float = BOX_HEIGHT, extent: float = BOX_EXTENT) -> HeightMapReader:
    return build_centered_box(height, DEFAULT_CELL, DEFAULT_INCLINE_DEG, extent=extent)


def rough_terrain(seed: int = ROUGH_SEED, extent: float = ROUGH_EXTENT) -> HeightMapReader:
    return build_rough_terrain(extent=extent, cell=DEFAULT_CELL, seed=seed)


def _flat_trials(full: bool) -> list[Trial]:
    trials = [
        Trial(label="straight", spawn_x=-3.0, spawn_y=0.0, spawn_yaw=0.0, v_drive=V_DRIVE),
        Trial(label="turn_in_place", spawn_x=0.0, spawn_y=0.0, spawn_yaw=0.0, v_drive=0.0, wz_drive=WZ_TURN),
        Trial(label="arc", spawn_x=-2.0, spawn_y=-2.0, spawn_yaw=0.0, v_drive=0.8, wz_drive=0.4),
    ]
    if full:
        trials += [
            # Yawed spawn: the wheel layout is not symmetric about the body Y axis (the rear wheel
            # is on the centerline), so driving along +X and along a diagonal are different tests.
            Trial(label="straight_yawed", spawn_x=-3.0, spawn_y=1.0, spawn_yaw=0.6, v_drive=V_DRIVE),
            Trial(label="turn_in_place_ccw", spawn_x=2.0, spawn_y=2.0, spawn_yaw=0.0, v_drive=0.0, wz_drive=-WZ_TURN),
            Trial(label="arc_tight", spawn_x=-1.0, spawn_y=3.0, spawn_yaw=0.0, v_drive=0.5, wz_drive=0.9),
            Trial(label="slow_creep", spawn_x=-3.0, spawn_y=-3.0, spawn_yaw=0.0, v_drive=0.2),
            Trial(label="reverse", spawn_x=3.0, spawn_y=-2.0, spawn_yaw=0.0, v_drive=-0.5),
        ]
    return trials


def _box_trials(full: bool) -> list[Trial]:
    # Beside the box's (-X, -Y) corner, facing +X: the same geometry compare_box_obstacles.py
    # spawns for, expressed against the origin-centered footprint this module builds.
    corner_x = -(_BOX_HALF + CORNER_MARGIN)
    corner_y = -(_BOX_HALF + CORNER_MARGIN)
    trials = [
        Trial(label="approach_face", spawn_x=-3.5, spawn_y=0.0, spawn_yaw=0.0, v_drive=V_DRIVE),
        Trial(
            label="turn_at_corner",
            spawn_x=corner_x, spawn_y=corner_y, spawn_yaw=0.0, v_drive=0.0, wz_drive=-WZ_TURN,
        ),
        Trial(label="skirt_side", spawn_x=-3.5, spawn_y=-1.6, spawn_yaw=0.0, v_drive=V_DRIVE),
    ]
    if full:
        trials += [
            Trial(label="approach_corner", spawn_x=-3.0, spawn_y=-3.0, spawn_yaw=0.785, v_drive=V_DRIVE),
            Trial(
                label="turn_at_corner_ccw",
                spawn_x=corner_x, spawn_y=corner_y, spawn_yaw=0.0, v_drive=0.0, wz_drive=WZ_TURN,
            ),
            Trial(label="arc_into_box", spawn_x=-3.5, spawn_y=-2.0, spawn_yaw=0.0, v_drive=0.8, wz_drive=0.3),
            Trial(label="skirt_far", spawn_x=-3.5, spawn_y=2.2, spawn_yaw=0.0, v_drive=V_DRIVE),
            # Clear of the box's far corner by the ramp toe (BOX_SIZE/2 + ramp width) plus the
            # robot's own half-width -- at 1.1 m its wheels sat on the ramp and the spawn had no
            # rest pose, which ostrich turned into a diverged solve (see footprint_relief).
            Trial(label="depart", spawn_x=1.8, spawn_y=1.8, spawn_yaw=0.785, v_drive=0.8),
        ]
    return trials


def _rough_trials(full: bool) -> list[Trial]:
    trials = [
        Trial(label="straight", spawn_x=-4.0, spawn_y=0.0, spawn_yaw=0.0, v_drive=V_DRIVE),
        Trial(label="diagonal", spawn_x=-4.0, spawn_y=-4.0, spawn_yaw=0.785, v_drive=V_DRIVE),
        Trial(label="turn_in_place", spawn_x=1.0, spawn_y=1.0, spawn_yaw=0.0, v_drive=0.0, wz_drive=WZ_TURN),
    ]
    if full:
        trials += [
            Trial(label="straight_offset", spawn_x=-4.0, spawn_y=3.0, spawn_yaw=0.0, v_drive=V_DRIVE),
            Trial(label="diagonal_back", spawn_x=4.0, spawn_y=4.0, spawn_yaw=-2.356, v_drive=V_DRIVE),
            Trial(label="arc", spawn_x=-3.0, spawn_y=2.0, spawn_yaw=0.0, v_drive=0.8, wz_drive=0.4),
            Trial(label="turn_in_place_ccw", spawn_x=-2.0, spawn_y=-2.0, spawn_yaw=0.0, v_drive=0.0, wz_drive=-WZ_TURN),
            Trial(label="slow_creep", spawn_x=-4.0, spawn_y=-2.0, spawn_yaw=0.0, v_drive=0.3),
        ]
    return trials


def scenario_set(tier: str = "smoke") -> list[Scenario]:
    """Every Scenario in `tier`, in a fixed order -- the baseline is keyed by scenario name, so
    order is cosmetic, but keeping it stable keeps two runs' printed reports diffable."""
    if tier not in ("smoke", "full"):
        raise ValueError(f"tier must be 'smoke' or 'full', got {tier!r}")
    full = tier == "full"
    duration = FULL_DURATION_S if full else SMOKE_DURATION_S
    flat_extent = FLAT_EXTENT_FULL if full else FLAT_EXTENT
    box_extent = BOX_EXTENT_FULL if full else BOX_EXTENT

    scenarios = [
        Scenario("flat", functools.partial(flat_terrain, flat_extent), _flat_trials(full), duration),
        Scenario(
            "box",
            functools.partial(box_terrain, BOX_HEIGHT, box_extent),
            _box_trials(full),
            duration,
        ),
        Scenario("rough", functools.partial(rough_terrain, ROUGH_SEED), _rough_trials(full), duration),
    ]
    if full:
        scenarios += [
            Scenario(
                "box_tall",
                functools.partial(box_terrain, BOX_HEIGHT_TALL, box_extent),
                _box_trials(full=False),
                duration,
            ),
            Scenario(
                "rough_alt",
                functools.partial(rough_terrain, ROUGH_SEED_ALT),
                _rough_trials(full=False),
                duration,
            ),
        ]
    return scenarios


def path_bbox(trial: Trial, duration_s: float, reach: float = ROBOT_REACH) -> tuple[float, float, float, float]:
    """(x_lo, x_hi, y_lo, y_hi) the trial's ideal no-slip path sweeps, dilated by `reach`.

    Bounds the ACTUAL commanded arc rather than a disc of radius v*duration around the spawn: the
    robot only ever moves along its own heading, so a disc bound would force maps several times
    larger than the trials need. Constant (v, wz) makes the path a circular arc, sampled densely
    here instead of solved in closed form -- the extremum of an arc's bbox falls at a cardinal
    tangent point, which is fiddly to case-split and not worth it for a sizing check.
    """
    t = np.linspace(0.0, duration_s, 256)
    yaw = trial.spawn_yaw + trial.wz_drive * t
    if abs(trial.wz_drive) < 1e-9:
        x = trial.spawn_x + trial.v_drive * t * np.cos(trial.spawn_yaw)
        y = trial.spawn_y + trial.v_drive * t * np.sin(trial.spawn_yaw)
    else:
        r = trial.v_drive / trial.wz_drive
        x = trial.spawn_x + r * (np.sin(yaw) - np.sin(trial.spawn_yaw))
        y = trial.spawn_y - r * (np.cos(yaw) - np.cos(trial.spawn_yaw))
    return float(x.min() - reach), float(x.max() + reach), float(y.min() - reach), float(y.max() + reach)


def footprint_relief(terrain: HeightMapReader, trial: Trial) -> float:
    """Height spread across the robot's footprint (three wheel contacts plus body center) at
    `trial`'s spawn pose.

    A spawn straddling a step taller than a wheel cannot rest on the terrain at all: ostrich drops
    the robot in interpenetrating the mesh and the contact solve diverges, so such a trial tests
    nothing and just produces a blown-up rollout (caught exactly this way on an earlier `depart`
    spawn placed 0.2 m off a box's ramp toe -- close enough that its wheels sat on the ramp).

    Deliberately measured as spread across the footprint rather than height above the map's median,
    which is how grid_learning/generate_dataset.py's footprint_clear does it: that rule assumes a
    BIMODAL map (flat background plus one tall obstacle) and misfires on continuous ground, where
    it flagged an ordinary rough-terrain spawn. Spread is terrain-agnostic and is the quantity that
    actually matters -- whether the three wheels can touch down together.
    """
    c, s = np.cos(trial.spawn_yaw), np.sin(trial.spawn_yaw)
    local_x = np.append(WHEEL_CONTACTS_LOCAL[:, 0], 0.0)  # + body center
    local_y = np.append(WHEEL_CONTACTS_LOCAL[:, 1], 0.0)
    wx = trial.spawn_x + c * local_x - s * local_y
    wy = trial.spawn_y + s * local_x + c * local_y
    h = np.asarray(terrain.sample(wx, wy), dtype=np.float64)
    return float(h.max() - h.min())


def select_scenarios(scenarios: list[Scenario], names: list[str] | None) -> list[Scenario]:
    """`names=None` passes everything through; otherwise keeps only the named scenarios, raising
    on an unknown name rather than silently running a smaller set. Mirrors
    comparator/common.py's select_variants/select_trials."""
    if names is None:
        return scenarios
    known = {s.name: s for s in scenarios}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(f"unknown scenario(s) {unknown}; have {sorted(known)}")
    return [known[n] for n in names]


if __name__ == "__main__":
    for tier in ("smoke", "full"):
        scenarios = scenario_set(tier)
        n_trials = sum(len(s.trials) for s in scenarios)
        print(f"\n=== tier {tier}: {len(scenarios)} scenario(s), {n_trials} trial(s) ===")
        for sc in scenarios:
            terrain = sc.terrain_fn()
            span_x = terrain.nx * terrain.cell
            span_y = terrain.ny * terrain.cell
            print(
                f"\n  {sc.name:10s} grid {terrain.ny}x{terrain.nx} @ {terrain.cell} m "
                f"({span_x:.1f} x {span_y:.1f} m, origin {terrain.x0:.2f},{terrain.y0:.2f})  "
                f"z in [{terrain.min_z:.3f}, {terrain.max_z:.3f}]  duration {sc.duration_s} s"
            )
            for trial, cap in zip(sc.trials, sc.max_displacement()):
                reach = abs(trial.v_drive) * sc.duration_s
                print(
                    f"      {trial.label:20s} spawn=({trial.spawn_x:6.2f},{trial.spawn_y:6.2f},"
                    f"{trial.spawn_yaw:5.2f})  v={trial.v_drive:5.2f} wz={trial.wz_drive:5.2f}  "
                    f"reach={reach:4.1f} m  cap={cap:4.1f} m"
                )
                # The commanded path must stay inside the mapped area, or sample()'s clamping
                # fabricates flat ground and the trial stops testing the terrain it names.
                gx_hi = terrain.x0 + terrain.nx * terrain.cell
                gy_hi = terrain.y0 + terrain.ny * terrain.cell
                bx_lo, bx_hi, by_lo, by_hi = path_bbox(trial, sc.duration_s)
                assert terrain.x0 <= bx_lo and bx_hi <= gx_hi, (
                    f"{sc.name}/{trial.label} sweeps x [{bx_lo:.2f}, {bx_hi:.2f}] "
                    f"outside the grid's [{terrain.x0:.2f}, {gx_hi:.2f}]"
                )
                assert terrain.y0 <= by_lo and by_hi <= gy_hi, (
                    f"{sc.name}/{trial.label} sweeps y [{by_lo:.2f}, {by_hi:.2f}] "
                    f"outside the grid's [{terrain.y0:.2f}, {gy_hi:.2f}]"
                )
                relief = footprint_relief(terrain, trial)
                assert relief <= MAX_SPAWN_RELIEF, (
                    f"{sc.name}/{trial.label} spawns straddling a {relief:.2f} m step (limit "
                    f"{MAX_SPAWN_RELIEF:.2f} m) -- no rest pose exists there, so ostrich "
                    "interpenetrates the mesh and the solve diverges"
                )
    print("\nscenarios OK")
