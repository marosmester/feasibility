"""Generates a grid_learning_2 divergence dataset: one sample is a whole heightmap plus a FIELD of
commanded yaw rates -- one command per spawn cell -- and its label is the ostrich-vs-hstack final
pose pair at every cell of that same lattice:

    x = (heightmap [81, 81] @ 0.125 m, wz [15, 15])   -- terrain as a whole + one command PER CELL
    y = [15, 15, 14]  = [ostrich_pose(7), hstack_pose(7)] per lattice cell
    mask = [15, 15] bool                               -- which cells are a real reading

The physics is byte-for-byte grid_learning/generate_dataset.py's: the robot spawns facing +X (yaw
locked at SPAWN_YAW) on a SPAWN_STEP lattice within +-SPAWN_LIMIT and is commanded a pure yaw rate
(V_DRIVE = 0), i.e. compare_box_obstacles.py's lateral-collision scenario swept over the whole map.
Two things differ, both from design.md:

  * **The command is per cell** (section 7b). v1 tiles ONE wz over all 225 cells of a row, so the
    command is perfectly confounded with the row and the effective sample size for learning the
    CONTROL dependence is the number of rows (order 100), not the number of labelled cells (order
    225k). Here `wz[l,i,j] ~ U(WZ_RANGE)` i.i.d. -- and it costs exactly the same simulation,
    because `simulate_map` already flattens (lattice cell, command) into one trial list before
    chunking and both sims take per-world spawns [N,3] and per-world setpoints [T,N,3]. The change
    is one line of index arithmetic:

        v1:   wz_drive = wz_values[trial_l]                      # L values, tiled over cells
        v2:   wz_drive = wz_field[trial_l, trial_i, trial_j]     # one value per trial

  * **The heightmap grid is odd and origin-centred** (section 3): 81 x 81 @ 0.125 m rather than
    100 x 100 @ 0.10 m, so the 0.5 m label lattice is an exact integer centre-crop of the network's
    feature map and the model needs no `F.grid_sample`. The generator's stake in that is small but
    load-bearing: it writes the grid the crop assumes, so it CHECKS the contract
    (`model.check_lattice_alignment`) before spending any GPU time, rather than letting a
    mis-sized file fail at training time.

Continuous sampling covers the command axis far better than v1's `linspace(-1, 1, L)` grid -- that
is half the point of the change -- but it puts zero mass on `wz = 0`, which is where the one
command-side sanity check with a known answer lives (no command, no motion, error ~ 0). Hence
`+wz_zero_frac` (default 0.05): a small fraction of cells is forced to exactly 0 so that check
stays available. `+wz_grid=K` restricts the draws to v1's discrete levels for a controlled
comparison; note it does NOT recreate v1's one-command-per-row layout, which is the thing v2 exists
to remove.

Masked (non-clear) cells get a command draw too, so `wz` is never structurally zero-filled in a way
a reader could confuse with a real "no command" trial. It is never read for those cells: the loss
is masked.

RNG consumption, in order: map selection, then the command fields. A seed therefore reproduces a
file exactly, and changing `+n_maps` does not reshuffle the commands of the maps that stayed.

Deliberately independent of feasibility.grid_learning and feasibility.learning (design.md section
11): the wheel geometry is re-derived from HelhestJuniorConfig, and the spawn filter / plausibility
bounds are re-implemented here rather than imported. feasibility.comparator (the batch-rollout
core) and feasibility.heightmap are shared infrastructure and ARE imported; so is this package's
own utils.py, which owns the lattice geometry because in v2 that geometry is a CONTRACT between the
dataset and the network's crop.

CLI parameters (Hydra overrides; `+` prefix required since none exist in the base "helhest"
config):
    +n_maps=INT        maps drawn (without replacement) from maps_dir  (default: 10)
    +rows_per_map=INT  command fields per map -- R = n_maps * this            (default: 10)
    +maps_dir=STR      repo-root-relative or absolute map directory
                       (default: assets/large_box_random/0 -- note the per-seed subdirectory)
    +seed=INT          RNG seed for map selection AND command fields   (default: 0)
    +wz_zero_frac=F    fraction of cells forced to exactly wz = 0      (default: 0.05)
    +wz_grid=INT       draw from K discrete levels instead of U(range) (default: unset/continuous)
    +duration_s=FLOAT  command hold time                               (default: 2.4)
    +chunk=INT         worlds per ostrich model build                  (default: 128)
    +settle_steps=INT  ostrich settle steps, paid once per chunk       (default: 12)
    +mu=FLOAT          ground friction                                 (default: 0.8)
    +k_turn=FLOAT      hstack ICR turning-rate gain                    (default: dynamics.K_TURN)
    +device=STR        torch/warp device for BOTH sims                 (default: "cuda:0")
    +resolution=FLOAT  heightmap tensor cell size, m                   (default: 0.125)
    +cells=INT         heightmap tensor cells per side, ODD            (default: 81)
    +dry_run=BOOL      lattice/filter/count only -- no simulation, no output (default: false)
    Also accepts any standard Hydra config-group override against the "helhest" base config
    (engine=mujoco, simulation=..., logging=...); rendering is forced headless.

Usage:
    python src/feasibility/grid_learning_2/generate_dataset.py +dry_run=true   # cheap CPU check
    python src/feasibility/grid_learning_2/generate_dataset.py +n_maps=1 +rows_per_map=1
    python src/feasibility/grid_learning_2/generate_dataset.py                 # M=10, L=10
    python src/feasibility/grid_learning_2/generate_dataset.py +chunk=64       # if the GPU OOMs
"""
from __future__ import annotations

import pathlib
import time

import h5py
import hydra
import numpy as np
from examples.helhest_junior.common import HelhestJuniorConfig
from helhest import dynamics
from omegaconf import DictConfig
from ostrich import EngineConfig
from ostrich import LoggingConfig
from ostrich import RenderingConfig
from ostrich import SimulationConfig

from feasibility.comparator.common import build_setpoints
from feasibility.comparator.common import CONFIG_PATH
from feasibility.comparator.common import euler_zyx_to_quat_xyzw
from feasibility.comparator.common import init_warp_device
from feasibility.comparator.common import K_P
from feasibility.comparator.common import OUT_DIR
from feasibility.comparator.common import run_hstack_batch
from feasibility.comparator.common import run_ostrich_batch
from feasibility.comparator.provenance import git_provenance
from feasibility.comparator.provenance import terrain_fields
from feasibility.grid_learning_2.model import check_lattice_alignment
from feasibility.grid_learning_2.utils import DEFAULT_N_CELLS
from feasibility.grid_learning_2.utils import DEFAULT_RESOLUTION
from feasibility.grid_learning_2.utils import extent_of
from feasibility.grid_learning_2.utils import heightmap_to_tensor
from feasibility.grid_learning_2.utils import spawn_lattice
from feasibility.grid_learning_2.utils import SPAWN_LIMIT
from feasibility.grid_learning_2.utils import SPAWN_STEP
from feasibility.heightmap import HeightMapReader

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

# --- hyperparameters (module constants -- override any of them with a Hydra `+key=value`) -------

# SPAWN_STEP / SPAWN_LIMIT live in utils.py, not here: in v2 the lattice is a contract between the
# dataset and the model's integer crop, so the generator and the network must read it from ONE
# place (design.md section 11). Everything else about the trial is generation-only and lives here.
SPAWN_YAW = 0.0  # rad, locked -- every spawn faces +X, so the lattice varies in position only

V_DRIVE = 0.0  # m/s, forward body velocity -- always zero here: this dataset is about yaw-rate-
# only commands (turn in place), where helhest_stack's missing lateral collision response shows up
WZ_RANGE = (-1.0, 1.0)  # rad/s -- restated locally rather than imported (see module docstring).
# model.WZ_MAX is the same number seen from the network's side, where it is a normalisation
# constant; keep the two in step if this range ever changes.

DEFAULT_N_MAPS = 10
DEFAULT_ROWS_PER_MAP = 10  # command FIELDS per map -- the file holds n_maps * this rows. v1 calls
# the same knob n_commands, because there a row IS one command; here a row holds G*G of them, so
# that name would be off by a factor of 225. The stored root attr `n_rows` is the TOTAL, and
# `rows_per_map` is this -- deliberately different words for the two different counts.
DEFAULT_SEED = 0
DEFAULT_WZ_ZERO_FRAC = 0.05  # see the module docstring: continuous sampling puts zero mass on
# wz = 0, and that is the only command with a known answer worth checking against
DEFAULT_DURATION_S = 2.4  # s -- an exact multiple of both sims' dt (ostrich 3e-2, hstack 0.1),
# giving T_o = 80 / T_h = 24 steps, and the same hold time every dataset in this repo uses
DEFAULT_CHUNK = 128  # worlds per ostrich model build -- a tuning knob, not a hard limit. It also
# sizes comparator.common's collision pipeline (max_triangle_pairs = max(1e6, worlds * 12_000)),
# so lower it before anything else if the GPU runs out of memory.
DEFAULT_SETTLE_STEPS = 12  # ostrich steps spent dropping the robot onto the terrain at zero
# command before recording starts, paid once PER CHUNK (the chassis is measurably at rest by step
# 7, so 12 is that with ~1.7x margin).

DEFAULT_MAPS_DIR = "assets/large_box_random/0"  # heightmap/create_large_box_obstacles.py's series
# rather than the small fixed 1.5 x 1.5 m boxes: a whole-map divergence FIELD starves for signal on
# a map that is ~94% flat ground, and that series exists precisely to fix it (independently sized
# rectangles under an area cap, placed to maximise obstacle edge within the spawn lattice's reach).
# The `<seed>` subdirectory is part of the path -- that generator writes one per placement seed.
MAP_GLOB = "*.png"  # every heightmap PNG directly in maps_dir is a candidate; nothing downstream
# cares how many obstacles a map has or where they are (see footprint_clear), only that every map
# in one run shares a grid shape.

OBSTACLE_MARGIN_FRACTION = 0.15  # a footprint point counts as "on an obstacle" once its height
# clears this fraction of the way from the terrain's median height (background, assumed to cover
# most of the map) to its max height. Scales with each map's own relief instead of a hardcoded
# absolute, so it needs no per-map tuning. Deliberately low, not a midpoint: the box generators'
# ramps are steep (75 deg incline), so a wheel resting even a third of the way up one already sits
# on a near-vertical local slope -- ostrich spawning a robot there interpenetrates the mesh and the
# contact solver diverges.

POS_MARGIN = 3.0  # m -- how far a final pose may legitimately land beyond the terrain's own
# footprint (x, y) or elevation band (z) before it counts as a diverged, non-physical solve rather
# than a real but large collision displacement. Per-map bounds (see plausible_bounds()) rather than
# a flat "distance from the origin" cutoff, which -- set loose enough to admit every real result --
# was loose enough to also admit genuine explosions.

MAX_SPAWN_DISPLACEMENT = 2.0  # m -- how far a final pose may legitimately sit from its OWN spawn
# cell. POS_MARGIN above is a per-map absolute box, so on a 10 m map it still admits a body flung
# 8 m sideways; this is the per-trial companion that catches those. It can be this tight because
# V_DRIVE == 0: every trial is a turn in place, so the body has no commanded translation and only
# contact interaction moves it. Measured over a 100-map run, ostrich's displacement has median
# 0.11 m and 90th percentile 0.20 m, yet a small tail runs out to 8.6 m -- contact-solver
# explosions, not large-but-real collisions, and being unlearnable noise of enormous magnitude they
# carried most of the e_pos label variance. Applied to both sims for symmetry, though it is
# overwhelmingly ostrich (the one with a contact solver) that trips it.

WHEEL_CONTACTS_LOCAL = np.array(
    [
        [float(p[0]), float(p[1])]
        for p in (
            HelhestJuniorConfig.LEFT_WHEEL_POS,
            HelhestJuniorConfig.RIGHT_WHEEL_POS,
            HelhestJuniorConfig.REAR_WHEEL_POS,
        )
    ]
)  # [3, 2] body-frame (x, y) of the three wheel contacts, derived from the robot config rather
# than hardcoded so the footprint filter can't drift from the model it is filtering for


def resolve_path(arg: str) -> pathlib.Path:
    """Turns a `+maps_dir=` CLI value into a real path: absolute paths pass through, everything
    else resolves against the repo root, so `+maps_dir=assets/box_random` works regardless of the
    invoking cwd (Hydra changes it) and matches every heightmap/create_*.py generator's own asset
    path convention."""
    p = pathlib.Path(arg)
    return p if p.is_absolute() else REPO_ROOT / p


def select_maps(
    maps_dir: pathlib.Path, n_maps: int, rng: np.random.Generator
) -> list[pathlib.Path]:
    """`n_maps` extension-less heightmap stems drawn WITHOUT replacement from every PNG directly
    in `maps_dir`. The candidate list is sorted before drawing so a given seed always picks the
    same maps regardless of filesystem ordering."""
    candidates = sorted(p.with_suffix("") for p in maps_dir.glob(MAP_GLOB))
    if len(candidates) < n_maps:
        raise ValueError(
            f"{maps_dir} holds {len(candidates)} map(s) matching {MAP_GLOB}, need {n_maps}. "
            f"Generate more with: python src/feasibility/heightmap/create_large_box_obstacles.py "
            f"--n {n_maps} (note the series is written to a per-seed SUBDIRECTORY, so "
            f"+maps_dir must name it)"
        )
    return [candidates[i] for i in rng.choice(len(candidates), size=n_maps, replace=False)]


def sample_command_fields(
    n_maps: int,
    rows_per_map: int,
    grid_n: int,
    rng: np.random.Generator,
    *,
    wz_zero_frac: float = DEFAULT_WZ_ZERO_FRAC,
    wz_grid: int | None = None,
) -> np.ndarray:
    """[n_maps, rows_per_map, G, G] float32 commanded yaw rates -- ONE PER LATTICE CELL, i.i.d.
    across cells, rows and maps (design.md section 7b).

    This is the entirety of v2's data-side change. Independence across cells is what decorrelates
    the command from map identity: v1's `linspace` tiled over a row makes all G*G gradients of a
    row flow through a single wz value, so the command axis is learned from `M * L` effective
    samples rather than `M * L * G * G`.

    `wz_zero_frac` of the cells are then forced to exactly 0 -- the continuous draw puts zero
    probability mass on the one command whose answer is known a priori (no command, no motion, so
    e_pos ~ e_rot ~ 0), and that anchor is what keeps the eval-time `wz ~ 0` check honest. Set it
    to 0 to disable.

    `wz_grid=K` draws from `linspace(*WZ_RANGE, K)` instead of the continuous range, for a
    controlled comparison against v1's discrete command set. It restricts the VALUES only -- every
    cell still draws independently, so this is not v1's one-command-per-row layout."""
    shape = (n_maps, rows_per_map, grid_n, grid_n)
    if wz_grid is None:
        wz = rng.uniform(WZ_RANGE[0], WZ_RANGE[1], size=shape)
    else:
        if wz_grid < 2:
            raise ValueError(f"+wz_grid must be >= 2 discrete levels, got {wz_grid}")
        levels = np.linspace(WZ_RANGE[0], WZ_RANGE[1], wz_grid)
        wz = levels[rng.integers(0, wz_grid, size=shape)]
    if not 0.0 <= wz_zero_frac <= 1.0:
        raise ValueError(f"+wz_zero_frac must be in [0, 1], got {wz_zero_frac}")
    if wz_zero_frac > 0.0:
        wz[rng.random(shape) < wz_zero_frac] = 0.0
    return wz.astype(np.float32)


def obstacle_height_threshold(terrain: HeightMapReader) -> float:
    """Height above which a footprint point counts as "on an obstacle" rather than background
    terrain: OBSTACLE_MARGIN_FRACTION of the way from the map's median height (the background is
    assumed to cover most of the map) to its max height (the top of the tallest obstacle).
    Expressed relative to the map's own height range rather than as a hardcoded absolute, so it
    needs no per-map tuning."""
    baseline = float(np.median(terrain.H))
    return baseline + OBSTACLE_MARGIN_FRACTION * (terrain.max_z - baseline)


def footprint_clear(terrain: HeightMapReader, xy: np.ndarray) -> np.ndarray:
    """[...] bool over `xy` [..., 2] -- True where a robot spawned at (x, y) facing SPAWN_YAW has
    its whole footprint (three wheel contacts plus body center) below obstacle_height_threshold,
    i.e. does not spawn on top of / inside an elevated obstacle. Spawning in a box makes ostrich's
    contact solver diverge rather than produce a usable trial, so those cells are never simulated
    at all -- they come back masked.

    Note this filter depends only on the MAP, never on the command: which cells are blocked is
    identical for every row of a map, which is why the blocked count below scales exactly with
    rows_per_map."""
    threshold = obstacle_height_threshold(terrain)
    c, s = np.cos(SPAWN_YAW), np.sin(SPAWN_YAW)
    local_x = np.append(WHEEL_CONTACTS_LOCAL[:, 0], 0.0)  # [4], + body center
    local_y = np.append(WHEEL_CONTACTS_LOCAL[:, 1], 0.0)
    wx = xy[..., 0, None] + c * local_x - s * local_y
    wy = xy[..., 1, None] + s * local_x + c * local_y
    heights = np.asarray(terrain.sample(wx, wy), dtype=np.float64)
    return heights.max(axis=-1) <= threshold


def plausible_bounds(terrain: HeightMapReader) -> np.ndarray:
    """[3, 2] per-axis (lo, hi) a final pose may plausibly occupy on `terrain`: x/y bounded by the
    map's own footprint (terrain.x0/y0 + nx/ny*cell -- its REAL extent, not the fixed-resolution
    tensor extent used elsewhere for the CNN input) padded by POS_MARGIN, z bounded by the
    terrain's own elevation band (min_z, max_z) padded the same way. Grounded in
    HeightMapReader.sample's documented behavior: querying outside these x/y bounds doesn't raise,
    it silently clamps to the nearest edge cell and fabricates flat ground -- so a physically real
    result can never legitimately land far outside them either."""
    x_lo, x_hi = terrain.x0, terrain.x0 + terrain.nx * terrain.cell
    y_lo, y_hi = terrain.y0, terrain.y0 + terrain.ny * terrain.cell
    z_lo, z_hi = terrain.min_z, terrain.max_z
    return np.array(
        [[x_lo - POS_MARGIN, x_hi + POS_MARGIN],
         [y_lo - POS_MARGIN, y_hi + POS_MARGIN],
         [z_lo - POS_MARGIN, z_hi + POS_MARGIN]]
    )


def simulate_map(
    terrain: HeightMapReader,
    lattice: np.ndarray,
    clear: np.ndarray,
    wz_field: np.ndarray,
    *,
    sim_config: SimulationConfig,
    render_config: RenderingConfig,
    engine_config: EngineConfig,
    logging_config: LoggingConfig,
    ostrich_dt: float,
    hstack_dt: float,
    duration_s: float,
    chunk: int,
    settle_steps: int,
    mu: float,
    k_turn: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Replays every (row, clear lattice cell) trial on ONE terrain, `chunk` worlds at a time, and
    returns (y [L, G, G, 14], mask [L, G, G]) for `wz_field` [L, G, G].

    Rows and cells are flattened into a single trial list before chunking -- both sims take a
    per-world spawn ([N, 3]) and per-world setpoints ([T, N, 3]), so one chunk can mix them freely,
    and only ONE short tail chunk is paid per map instead of one per row. What a chunk cannot mix
    is terrains: HelhestBatchSimulator puts the heightmap in the model's globals builder, shared by
    all worlds, hence one call to this function per map.

    The ONLY difference from grid_learning's version is that `wz_drive` gathers the command with
    the trial's full (row, i, j) index instead of the row index alone. Trial count, chunking, GPU
    cost and settle steps are all identical -- a per-cell command field is a re-bundling of exactly
    the same simulations (design.md sections 1c and 7b).

    Only the FINAL timestep of each rollout is kept. The full [T, N, 7] trajectories that
    comparator's HDF5 schema stores would be ~80x this dataset's size for ~22k worlds and are not
    what a divergence-field model consumes."""
    L, G = wz_field.shape[0], lattice.shape[0]
    if wz_field.shape != (L, G, G):
        raise ValueError(f"wz_field {wz_field.shape} does not match the {G}x{G} lattice")
    bounds = plausible_bounds(terrain)  # [3, 2] (x, y, z) lo/hi -- see plausible_bounds' docstring

    cell_i, cell_j = np.nonzero(clear)  # [P] lattice cells actually worth simulating
    n_cells = cell_i.size
    trial_l = np.repeat(np.arange(L), n_cells)
    trial_i = np.tile(cell_i, L)
    trial_j = np.tile(cell_j, L)
    n = trial_l.size

    spawn_pose = np.column_stack(
        [lattice[trial_i, trial_j, 0], lattice[trial_i, trial_j, 1], np.full(n, SPAWN_YAW)]
    ).astype(np.float64)
    wz_drive = wz_field[trial_l, trial_i, trial_j]  # <-- v1: wz_values[trial_l]

    y = np.zeros((L, G, G, 14), dtype=np.float32)
    mask = np.zeros((L, G, G), dtype=bool)

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        b = end - start
        t0 = time.time()
        sim_config.num_worlds = b  # mutated in place per chunk; HelhestBatchSimulator.build_model
        # cross-checks it against spawn_pose's row count, so the two must stay in step
        spawn_chunk = spawn_pose[start:end]
        wz_chunk = wz_drive[start:end]

        ostrich_setpoints = np.stack(
            [build_setpoints(ostrich_dt, V_DRIVE, wz_chunk[i], duration_s) for i in range(b)], axis=1
        )  # [T_o, b, 3]
        hstack_setpoints = np.stack(
            [build_setpoints(hstack_dt, V_DRIVE, wz_chunk[i], duration_s) for i in range(b)], axis=1
        )  # [T_h, b, 3]

        pose, _ = run_ostrich_batch(
            sim_config, render_config, engine_config, logging_config, terrain,
            ostrich_setpoints, mu, spawn_chunk, settle_steps,
        )
        controlled, derived, _, _, _, _ = run_hstack_batch(
            hstack_setpoints, terrain, hstack_dt, k_turn, mu, device, spawn_chunk
        )

        o_final = pose[-1]  # [b, 7] (px,py,pz,qx,qy,qz,qw)
        # hstack reports (x, y, yaw) + (z, pitch, roll) separately; reassemble the same 7-vector
        # form ostrich logs so both halves of `y` are directly comparable.
        h_quat = euler_zyx_to_quat_xyzw(controlled[-1, :, 2], derived[-1, :, 1], derived[-1, :, 2])
        h_final = np.concatenate(
            [controlled[-1, :, :2], derived[-1, :, :1], h_quat], axis=-1
        ).astype(np.float32)

        both = np.concatenate([o_final, h_final], axis=1)  # [b, 14]
        ok = np.isfinite(both).all(axis=1)
        for final in (o_final, h_final):  # both sims' final pose must independently land inside
            # the terrain's own plausible x/y/z bounds -- see plausible_bounds()
            ok &= (final[:, :3] >= bounds[:, 0]).all(axis=1)
            ok &= (final[:, :3] <= bounds[:, 1]).all(axis=1)
            # ...and within MAX_SPAWN_DISPLACEMENT of the trial's OWN spawn cell, which the
            # map-wide box above cannot express -- see MAX_SPAWN_DISPLACEMENT
            travelled = np.linalg.norm(final[:, :2] - spawn_chunk[:, :2], axis=1)
            ok &= travelled <= MAX_SPAWN_DISPLACEMENT

        sl = slice(start, end)
        y[trial_l[sl], trial_i[sl], trial_j[sl]] = both
        mask[trial_l[sl], trial_i[sl], trial_j[sl]] = ok

        n_bad = int((~ok).sum())
        bad = f", {n_bad} diverged" if n_bad else ""
        print(f"    trials {start}..{end - 1} ({b}) done in {time.time() - t0:.1f}s{bad}")

    # Masked cells carry zeros, never NaN: a NaN target poisons the backward pass even through a
    # correctly masked loss (torch.where zeroes the VALUE, but the chain rule still evaluates
    # d/dpred = 2*(pred - NaN) = NaN on the masked branch and then multiplies it by zero, and
    # 0 * NaN is NaN). Zeros are inert under any reduction, and an all-zero quaternion is not a
    # valid rotation, so the fill stays detectably-not-a-pose for anyone who ignores the mask.
    # custom_dataset.py's SE(3) reduction turns exactly this fill into (0, 0), by construction.
    y[~mask] = 0.0
    return y, mask


def write_grid_dataset(
    path: pathlib.Path,
    *,
    root: dict[str, object],
    per_row: dict[str, np.ndarray],
    grid: dict[str, np.ndarray],
    terrain_entries: list[tuple[pathlib.Path, HeightMapReader]],
) -> None:
    """Writes the dataset's HDF5. A local writer rather than
    comparator.provenance.write_comparison: that schema is built around per-variant time-series
    arrays in `ostrich/`/`hstack/` groups, which this dataset (final poses only, shaped as a
    lattice) does not have. Both self-description mechanisms are reused verbatim, though --
    terrain_fields() embeds the raw elevation grids + yaml sidecars so a stored dataset never has
    to trust assets/ again, and git_provenance() records both submodules' HEAD SHA and dirty state
    -- so provenance cannot drift between writers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for name, value in root.items():
            f.attrs[name] = value

        for group, fields in ((f, per_row), (f.create_group("grid"), grid),
                              (f.create_group("terrain"), terrain_fields(terrain_entries))):
            for name, arr in fields.items():
                arr = np.asarray(arr)
                if arr.dtype.kind == "U":  # h5py needs an explicit vlen string dtype, and gzip
                    # buys nothing on a handful of path/yaml strings
                    group.create_dataset(name, data=arr.astype(object), dtype=h5py.string_dtype())
                elif arr.shape == ():  # h5py rejects chunking/compression on scalar datasets
                    group.create_dataset(name, data=arr)
                else:
                    group.create_dataset(name, data=arr, compression="gzip", compression_opts=4)

        grp_git = f.create_group("git")
        for name, value in git_provenance().items():
            grp_git.attrs[name] = value

    print(f"saved {path}")


def generate(cfg: DictConfig) -> None:
    n_maps = int(cfg.get("n_maps", DEFAULT_N_MAPS))
    rows_per_map = int(cfg.get("rows_per_map", DEFAULT_ROWS_PER_MAP))
    seed = int(cfg.get("seed", DEFAULT_SEED))
    maps_dir = resolve_path(str(cfg.get("maps_dir", DEFAULT_MAPS_DIR)))
    wz_zero_frac = float(cfg.get("wz_zero_frac", DEFAULT_WZ_ZERO_FRAC))
    wz_grid = cfg.get("wz_grid", None)
    wz_grid = int(wz_grid) if wz_grid is not None else None
    duration_s = float(cfg.get("duration_s", DEFAULT_DURATION_S))
    chunk = int(cfg.get("chunk", DEFAULT_CHUNK))
    settle_steps = int(cfg.get("settle_steps", DEFAULT_SETTLE_STEPS))
    mu = float(cfg.get("mu", 0.8))
    k_turn = float(cfg.get("k_turn", dynamics.K_TURN))
    device = str(cfg.get("device", "cuda:0"))
    resolution = float(cfg.get("resolution", DEFAULT_RESOLUTION))
    n_cells = int(cfg.get("cells", DEFAULT_N_CELLS))
    dry_run = bool(cfg.get("dry_run", False))

    lattice = spawn_lattice()
    G = lattice.shape[0]

    # RNG consumption, in this order: map selection, then the command fields (design.md 7b). Both
    # come off the SAME generator, so a seed reproduces a file exactly; drawing the fields for all
    # maps in one call keeps map m's commands independent of how many maps came before it only in
    # the sense that the array is sliced, not re-drawn -- change n_maps and the earlier maps keep
    # their fields, change the seed and everything moves.
    rng = np.random.default_rng(seed)
    map_paths = select_maps(maps_dir, n_maps, rng)
    wz_per_map = sample_command_fields(
        n_maps, rows_per_map, G, rng, wz_zero_frac=wz_zero_frac, wz_grid=wz_grid
    )  # [n_maps, rows_per_map, G, G]

    n_zero = int((wz_per_map == 0.0).sum())  # cells AT zero, which for an odd `+wz_grid` includes
    # the grid's own 0.0 level as well as the wz_zero_frac draws -- hence no "target" in the print
    print(f"[lattice]  {G}x{G} = {G * G} cells, step {SPAWN_STEP} m, |x|,|y| <= {SPAWN_LIMIT} m, "
          f"yaw locked at {SPAWN_YAW} rad")
    print(f"[maps]     {n_maps} from {maps_dir}, {rows_per_map} command field(s) each -> "
          f"{n_maps * rows_per_map} rows")
    draw = f"U{WZ_RANGE}" if wz_grid is None else f"{wz_grid} discrete levels over {WZ_RANGE}"
    print(f"[command]  per-cell wz ~ {draw}, zero anchor {100 * wz_zero_frac:.1f}% -> "
          f"{n_zero}/{wz_per_map.size} cells at exactly 0 "
          f"({100 * n_zero / wz_per_map.size:.1f}%)")

    terrains = [HeightMapReader.load(p) for p in map_paths]
    clears = [footprint_clear(t, lattice) for t in terrains]
    heightmaps = np.stack(
        [heightmap_to_tensor(t, n_cells=n_cells, resolution=resolution).numpy() for t in terrains]
    ).astype(np.float32)

    # The readout contract, checked BEFORE any GPU time is spent: the network's integer centre-crop
    # of the feature lattice must land exactly on this file's spawn_xy (design.md section 3c). This
    # is what replaces v1's grid_sample, and it is the one way a mis-sized `+cells`/`+resolution`
    # could produce a file that loads fine and trains on silently misregistered labels.
    err = check_lattice_alignment(lattice.astype(np.float32), n_input=n_cells, resolution=resolution)
    print(f"[readout]  {n_cells}x{n_cells} @ {resolution} m -> crop lands on spawn_xy, "
          f"max |diff| = {err:.2e} m")

    total = sum(int(c.sum()) for c in clears) * rows_per_map
    for p, c in zip(map_paths, clears):
        print(f"  {p.name}: {int(c.sum())}/{G * G} spawn cells clear of the obstacle")
    print(f"[trials]   {total} per simulator ({total / max(chunk, 1):.0f} chunks of <= {chunk})")

    if dry_run:
        # Cheap CPU-only self check -- the lattice and the heightmap tensor must agree on
        # orientation and extent, or a label grid would be silently transposed relative to the
        # terrain a CNN sees. (The readout check above already ran, and is the stronger statement.)
        assert heightmaps.shape[1:] == (n_cells, n_cells), heightmaps.shape
        assert n_cells % 2 == 1, "the CNN grid must be odd -- design.md section 3b"
        assert lattice[0, 0].tolist() == [-SPAWN_LIMIT, -SPAWN_LIMIT], lattice[0, 0]
        assert lattice[0, -1].tolist() == [SPAWN_LIMIT, -SPAWN_LIMIT], lattice[0, -1]
        assert lattice[-1, 0].tolist() == [-SPAWN_LIMIT, SPAWN_LIMIT], lattice[-1, 0]
        flat = HeightMapReader.flat(xlim=(-8.0, 8.0), ylim=(-8.0, 8.0))
        assert footprint_clear(flat, lattice).all(), "flat ground must clear every spawn cell"
        # The command really is a FIELD: within a row, cells must differ. This is the property v1
        # files lack and custom_dataset.py rejects them for.
        flat_rows = wz_per_map.reshape(n_maps * rows_per_map, -1)
        spread = flat_rows.max(axis=1) - flat_rows.min(axis=1)
        assert (spread > 1e-6).all(), "some row has a constant command field"
        print("[dry-run]  lattice/heightmap convention + filter + command-field self-checks ok, "
              "nothing simulated")
        return

    init_warp_device(device)  # pins ostrich onto the same GPU `device` puts hstack on -- ostrich
    # has no device config field of its own, see init_warp_device's docstring

    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)
    render_config.vis_type = "null"  # headless -- see HelhestBatchSimulator's docstring for why a
    # GL viewer's set_world_offsets would scatter N robots off the one shared terrain anyway

    ostrich_dt = sim_config.target_timestep_seconds
    hstack_dt = dynamics.DT
    T_o = int(round(duration_s / ostrich_dt))
    T_h = int(round(duration_s / hstack_dt))
    if abs(T_o * ostrich_dt - T_h * hstack_dt) > 1e-6:
        raise ValueError(
            f"duration_s={duration_s} is not an exact multiple of both sims' dt "
            f"(ostrich {ostrich_dt}, hstack {hstack_dt}) -- the two rollouts would end at "
            f"different times ({T_o * ostrich_dt} vs {T_h * hstack_dt} s), so their final poses "
            f"would not be comparable"
        )
    print(f"[ostrich]  {T_o} steps @ dt={ostrich_dt}, chunk={chunk}, settle={settle_steps} steps/chunk")
    print(f"[hstack]   {T_h} steps @ dt={hstack_dt}, device={device}")

    ys, masks = [], []
    for m, (p, terrain, clear) in enumerate(zip(map_paths, terrains, clears)):
        print(f"[map {m + 1}/{n_maps}] {p.name}")
        y, mask = simulate_map(
            terrain, lattice, clear, wz_per_map[m],
            sim_config=sim_config, render_config=render_config, engine_config=engine_config,
            logging_config=logging_config, ostrich_dt=ostrich_dt, hstack_dt=hstack_dt,
            duration_s=duration_s, chunk=chunk, settle_steps=settle_steps, mu=mu,
            k_turn=k_turn, device=device,
        )
        ys.append(y)
        masks.append(mask)
        n_map_total = clear.size * rows_per_map
        n_map_blocked = int((~clear).sum()) * rows_per_map
        n_map_valid = int(mask.sum())
        n_map_diverged = n_map_total - n_map_blocked - n_map_valid
        print(
            f"    -> {n_map_valid} valid, {n_map_blocked} blocked, {n_map_diverged} diverged "
            f"({n_map_total} cells)"
        )

    y = np.concatenate(ys, axis=0)  # [R, G, G, 14], R = n_maps * rows_per_map
    mask = np.concatenate(masks, axis=0)  # [R, G, G]
    map_index = np.repeat(np.arange(n_maps, dtype=np.int64), rows_per_map)

    # create_large_box_obstacles.py writes into a per-seed SUBDIRECTORY, so `maps_dir.name` alone
    # would tag the file with a bare "0"; fold the parent in when the leaf is just that index.
    tag = f"{maps_dir.parent.name}{maps_dir.name}" if maps_dir.name.isdigit() else maps_dir.name
    out_path = OUT_DIR / f"dataset_grid2_{tag}_M{n_maps}_L{rows_per_map}_g{G}.h5"
    write_grid_dataset(
        out_path,
        root=dict(
            n_maps=n_maps, rows_per_map=rows_per_map, n_rows=len(y), grid_n=G,  # n_rows is the
            # TOTAL row count (n_maps * rows_per_map); rows_per_map is the CLI knob of that name
            wz_per_cell=True,  # the schema flag design.md section 7a adds -- a reader that cares
            # about the v1/v2 distinction can branch on this instead of on wz.ndim
            wz_zero_frac=wz_zero_frac, wz_grid=0 if wz_grid is None else wz_grid,
            spawn_step=SPAWN_STEP, spawn_limit=SPAWN_LIMIT, spawn_yaw=SPAWN_YAW,
            v_drive=V_DRIVE, wz_min=WZ_RANGE[0], wz_max=WZ_RANGE[1],
            duration_s=duration_s, resolution=resolution, extent=extent_of(n_cells, resolution),
            n_cells=n_cells, seed=seed, mu=mu, k_turn=k_turn, k_p=K_P, chunk=chunk,
            settle_steps=settle_steps, maps_dir=str(maps_dir),
        ),
        per_row=dict(
            wz=wz_per_map.reshape(n_maps * rows_per_map, G, G),  # [R, G, G] -- v1 writes [R]
            map_index=map_index,
            map_path=np.array([str(map_paths[i]) for i in map_index]),
            y=y,
            mask=mask,
            spawn_xy=lattice.astype(np.float32),  # [G, G, 2], shared by every row
        ),
        grid=dict(
            heightmap=heightmaps,  # [n_maps, n_cells, n_cells] -- stored per MAP, indexed by
            # map_index, the same dedup philosophy terrain_fields() applies to the raw grids
            resolution=np.float64(resolution),
            extent=np.float64(extent_of(n_cells, resolution)),
        ),
        terrain_entries=[(map_paths[i], terrains[i]) for i in map_index],
    )
    n_blocked = sum(int((~c).sum()) for c in clears) * rows_per_map  # never simulated, same for every
    # row since footprint_clear only depends on the map, not on the command
    n_valid = int(mask.sum())
    n_total = mask.size
    n_diverged = n_total - n_blocked - n_valid  # footprint-clear (simulated) but failed the
    # finite/plausible-bounds/displacement check in simulate_map -- see its `ok` computation
    print("=" * 60)
    print(f"[summary]  {n_total} lattice cells across {len(y)} rows ({n_maps} maps x {rows_per_map} command fields)")
    print(f"  valid     {n_valid:>7d}  ({100 * n_valid / n_total:5.1f}%)")
    print(f"  blocked   {n_blocked:>7d}  ({100 * n_blocked / n_total:5.1f}%)  -- obstacle footprint, never simulated")
    print(f"  diverged  {n_diverged:>7d}  ({100 * n_diverged / n_total:5.1f}%)  -- simulated but failed the finite/plausible-bounds/displacement check")
    print(f"  commands  {n_valid} independent wz draws contribute to the loss "
          f"(v1 would give {len(y)})")
    print("=" * 60)


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest", version_base=None)
def main(cfg: DictConfig) -> None:
    generate(cfg)


if __name__ == "__main__":
    main()
