"""Generates a GRID-style ostrich-vs-helhest_stack divergence dataset: one sample is a whole
heightmap plus one commanded yaw rate, and its label is the final-pose pair at EVERY point of a
regular spawn lattice over that map.

    x = (heightmap [100, 100], wz)                 -- terrain as a whole, CNN-ready
    y = [15, 15, 14]  = [ostrich_pose(7), hstack_pose(7)] per lattice cell
    mask = [15, 15] bool                            -- which cells are real (see below)

This is the CNN reframing of feasibility.learning's per-trial datasets. There, terrain reaches
the model either as a spawn-pose coordinate code (which only means anything on the one heightmap
it was fitted to) or as a small body-frame patch; here it reaches the model whole, resampled onto
the fixed grid utils.heightmap_to_tensor produces, and the label is a divergence FIELD rather
than a single number -- "given this terrain and this spin command, where on the map do the two
simulators disagree, and how".

The robot spawns facing +X (yaw locked to 0) on a SPAWN_STEP lattice within +-SPAWN_LIMIT, and is
commanded a pure yaw rate: V_DRIVE is 0, so `cmd_to_wheels(0, wz)` drives the two front wheels at
opposite signs and the rear wheel at exactly 0 -- a turn in place about the front-axle midpoint.
That is the same scenario comparator/compare_box_obstacles.py probes at one hand-picked pose:
LATERAL body collision, which helhest_stack cannot represent at all (its quasi-static settle only
resolves vertical support, so it rotates straight through a box), while ostrich jams against it.
Sweeping the whole lattice turns that single measurement into a map of where on this terrain the
kinematic twin is trustworthy.

SPAWN_LIMIT leaves 1.5 m of padding on a 10 m map, which is what keeps the robot on mapped
terrain: the spawn (x, y) is the front-axle midpoint, and a turn in place sweeps the rear wheel's
rim to ~1.45 m from it (1.101 m axle offset + 0.35 m wheel radius). Without that padding a corner
spawn would rotate its rear wheel off the edge of the grid, where HeightMapReader.sample clamps
rather than raises -- silently fabricating flat ground instead of erroring.

Deliberately independent of feasibility.learning (the two trees stay non-dependent): the wheel
geometry is re-derived here from HelhestJuniorConfig, and the on-obstacle spawn filter is
re-implemented locally rather than imported. feasibility.comparator (the scenario-independent
batch-rollout core) and feasibility.heightmap are shared infrastructure and ARE imported.

Unlike learning/generate_init_pose_dataset.py, which samples many random poses on ONE fixed map,
this sweeps M maps -- so the map loop has to be the OUTER one. comparator.common's
HelhestBatchSimulator bakes the terrain into the model's globals builder (one mesh shared by all
worlds), so a chunk of parallel worlds can never span two heightmaps; within one map, though, a
chunk freely mixes lattice cells AND yaw rates, since `setpoints` is per-world [T, N, 3].

CLI parameters (Hydra overrides; `+` prefix required since none exist in the base "helhest"
config):
    +n_maps=INT        maps drawn (without replacement) from maps_dir  (default: 10)
    +n_commands=INT    wz commands per map, equidistant across WZ_RANGE (default: 10)
    +maps_dir=STR      repo-root-relative or absolute map directory    (default: assets/box_random)
    +seed=INT          RNG seed for map selection                     (default: 0)
    +duration_s=FLOAT  command hold time                               (default: 2.4)
    +chunk=INT         worlds per ostrich model build                  (default: 128)
    +settle_steps=INT  ostrich settle steps, paid once per chunk       (default: 12)
    +mu=FLOAT          ground friction                                 (default: 0.8)
    +k_turn=FLOAT      hstack ICR turning-rate gain                    (default: dynamics.K_TURN)
    +device=STR        torch/warp device for BOTH sims (via init_warp_device) (default: "cuda:0")
    +resolution=FLOAT  heightmap tensor cell size, m                   (default: 0.10)
    +extent=FLOAT      heightmap tensor extent, m                      (default: 10.0)
    +dry_run=BOOL      lattice/filter/count only -- no simulation, no output file (default: false)
    Also accepts any standard Hydra config-group override against the "helhest" base config
    (engine=mujoco, simulation=..., logging=...); rendering is forced headless.

Usage:
    python src/feasibility/grid_learning/generate_dataset.py +dry_run=true   # cheap CPU-only check
    python src/feasibility/grid_learning/generate_dataset.py +n_maps=1 +n_commands=1
    python src/feasibility/grid_learning/generate_dataset.py                 # M=10, L=10
    python src/feasibility/grid_learning/generate_dataset.py +chunk=64       # if the GPU OOMs
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
from feasibility.grid_learning.utils import DEFAULT_EXTENT
from feasibility.grid_learning.utils import DEFAULT_RESOLUTION
from feasibility.grid_learning.utils import heightmap_to_tensor
from feasibility.heightmap import HeightMapReader

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

# --- hyperparameters (module constants -- override any of them with a Hydra `+key=value`) -------

SPAWN_STEP = 0.5  # m, (x, y) lattice pitch
SPAWN_LIMIT = 3.5  # m, |x|, |y| <= this -> a 15 x 15 lattice, i.e. 1.5 m of padding on a 10 m
# map. See the module docstring for why 1.5 m specifically (rear-wheel rim reach under a turn in
# place), and why sample()'s silent clamping makes under-padding a correctness bug, not a crash.
SPAWN_YAW = 0.0  # rad, locked -- every spawn faces +X, so the lattice varies in position only

V_DRIVE = 0.0  # m/s, forward body velocity -- always zero here: this dataset is about yaw-rate-
# only commands (turn in place), where helhest_stack's missing lateral collision response shows up
WZ_RANGE = (-1.0, 1.0)  # rad/s, yaw-rate command -- same limits learning/ uses, restated locally
# rather than imported so the two trees stay independent (see module docstring)

DEFAULT_N_MAPS = 10
DEFAULT_N_COMMANDS = 10
DEFAULT_SEED = 0
DEFAULT_DURATION_S = 2.4  # s -- an exact multiple of both sims' dt (ostrich 3e-2, hstack 0.1),
# giving T_o = 80 / T_h = 24 steps, and the same hold time learning/'s datasets use
DEFAULT_CHUNK = 128  # worlds per ostrich model build -- a tuning knob, not a hard limit. It also
# sizes comparator.common's collision pipeline (max_triangle_pairs = max(1e6, worlds * 12_000)),
# so lower it before anything else if the GPU runs out of memory.
DEFAULT_SETTLE_STEPS = 12  # ostrich steps spent dropping the robot onto the terrain at zero
# command before recording starts, paid once PER CHUNK (see run_ostrich_batch's docstring: the
# chassis is measurably at rest by step 7, so 12 is that with ~1.7x margin).

DEFAULT_MAPS_DIR = "assets/box_random"
MAP_GLOB = "*.png"  # every heightmap PNG in maps_dir is a candidate; nothing downstream cares
# how many obstacles a map has or where they are (see footprint_clear), only that every map in
# one run shares a grid shape, so no naming-pattern filter is needed -- just enough of them.

OBSTACLE_MARGIN_FRACTION = 0.15  # a footprint point counts as "on an obstacle" once its height
# clears this fraction of the way from the terrain's median height (background, assumed to cover
# most of the map) to its max height (the top of the tallest obstacle). Scales with each map's own
# relief instead of a hardcoded absolute, so it needs no per-map tuning. Deliberately low, not a
# midpoint: create_box_obstacles.py's ramps are steep (75 deg incline, ramp_width ~= 0.27*height),
# so a wheel resting even a third of the way up one already sits on a near-vertical local slope --
# ostrich spawning a robot there interpenetrates the mesh and the contact solver diverges.

POS_MARGIN = 3.0  # m -- how far a final pose may legitimately land beyond the terrain's own
# footprint (x, y) or elevation band (z) before it's treated as a diverged, non-physical solve
# rather than a real but large collision displacement. Per-map bounds (see plausible_bounds())
# replace a flat "distance from the origin" cutoff, which -- set loose enough to admit every real
# result -- was loose enough to also admit genuine explosions: a rear wheel catching a box's ramp
# mid-turn flung to z=-23 m still read as "valid" under a symmetric +-50 m check on every axis.
# 3 m covers real cases with room to spare (the worst observed real collision divergence, i.e. the
# gap between the two sims' final poses on an otherwise-sane trial, was ~7 m -- see the
# grid_learning viewer's cell (10,12) investigation for the exploded case this replaces).

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
            f"{maps_dir} holds {len(candidates)} map(s), need {n_maps}. "
            f"Generate more with: python src/feasibility/heightmap/create_box_obstacles.py "
            f"--batch --n {n_maps}"
        )
    return [candidates[i] for i in rng.choice(len(candidates), size=n_maps, replace=False)]


def spawn_lattice(step: float = SPAWN_STEP, limit: float = SPAWN_LIMIT) -> np.ndarray:
    """[G, G, 2] world (x, y) of every spawn cell, G = 2*limit/step + 1. Row index runs along +Y
    and column index along +X -- the SAME orientation utils.grid_coords gives the heightmap
    tensor, so label cell (i, j) and heightmap pixel (i, j) can be overlaid directly (they are
    different resolutions, not different conventions)."""
    coords = np.arange(-limit, limit + 1e-9, step)
    X, Y = np.meshgrid(coords, coords)  # [G, G], row=y, col=x
    return np.stack([X, Y], axis=-1)


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
    at all -- they come back masked."""
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
    tensor `extent` used elsewhere for the CNN input) padded by POS_MARGIN, z bounded by the
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
    wz_values: np.ndarray,
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
    """Replays every (lattice cell, wz) trial on ONE terrain, `chunk` worlds at a time, and
    returns (y [L, G, G, 14], mask [L, G, G]).

    Cells and yaw rates are flattened into a single trial list before chunking -- both sims take a
    per-world spawn ([N, 3]) and per-world setpoints ([T, N, 3]), so one chunk can mix them
    freely, and only ONE short tail chunk is paid per map instead of one per command. What a chunk
    cannot mix is terrains: HelhestBatchSimulator puts the heightmap in the model's globals
    builder, shared by all worlds, hence one call to this function per map.

    Only the FINAL timestep of each rollout is kept. The full [T, N, 7] trajectories that
    comparator's HDF5 schema stores would be ~80x this dataset's size for ~22k worlds and are not
    what a divergence-field model consumes."""
    L, G = len(wz_values), lattice.shape[0]
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
    wz_drive = wz_values[trial_l]

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
    """Writes the grid dataset's HDF5. A local writer rather than
    comparator.provenance.write_comparison: that schema is built around per-variant time-series
    arrays in `ostrich/`/`hstack/` groups, which this dataset (final poses only, shaped as a
    lattice) does not have. Both self-description mechanisms are reused verbatim, though --
    terrain_fields() embeds the raw elevation grids + yaml sidecars so a stored dataset never has
    to trust assets/ again, and git_provenance() records both submodules' HEAD SHA and dirty
    state -- so provenance cannot drift between the two writers."""
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
    n_commands = int(cfg.get("n_commands", DEFAULT_N_COMMANDS))
    seed = int(cfg.get("seed", DEFAULT_SEED))
    maps_dir = resolve_path(str(cfg.get("maps_dir", DEFAULT_MAPS_DIR)))
    duration_s = float(cfg.get("duration_s", DEFAULT_DURATION_S))
    chunk = int(cfg.get("chunk", DEFAULT_CHUNK))
    settle_steps = int(cfg.get("settle_steps", DEFAULT_SETTLE_STEPS))
    mu = float(cfg.get("mu", 0.8))
    k_turn = float(cfg.get("k_turn", dynamics.K_TURN))
    device = str(cfg.get("device", "cuda:0"))
    resolution = float(cfg.get("resolution", DEFAULT_RESOLUTION))
    extent = float(cfg.get("extent", DEFAULT_EXTENT))
    dry_run = bool(cfg.get("dry_run", False))

    # RNG consumption is map selection only: wz commands are an equidistant grid across WZ_RANGE
    # (below), not drawn, so `seed` no longer affects them and every map gets the identical
    # n_commands-length command set -- results stay directly comparable across maps/seeds and a
    # regeneration at the same n_commands reproduces the same commands even with a different seed.
    rng = np.random.default_rng(seed)
    map_paths = select_maps(maps_dir, n_maps, rng)

    wz_values = np.linspace(*WZ_RANGE, n_commands, dtype=np.float32)  # e.g. n_commands=5 ->
    # [-1, -0.5, 0, 0.5, 1] -- equidistant coverage of the command range rather than a random
    # sample of it, so the lattice-of-poses x command-sweep together tile the (position, wz) input
    # space evenly instead of leaving gaps some regenerations would happen to miss
    wz_per_map = np.tile(wz_values, (n_maps, 1))

    lattice = spawn_lattice()
    G = lattice.shape[0]
    print(f"[lattice]  {G}x{G} = {G * G} cells, step {SPAWN_STEP} m, |x|,|y| <= {SPAWN_LIMIT} m, "
          f"yaw locked at {SPAWN_YAW} rad")
    print(f"[maps]     {n_maps} from {maps_dir}, {n_commands} wz command(s) each -> "
          f"{n_maps * n_commands} rows")

    terrains = [HeightMapReader.load(p) for p in map_paths]
    clears = [footprint_clear(t, lattice) for t in terrains]
    heightmaps = np.stack(
        [heightmap_to_tensor(t, resolution=resolution, extent=extent).numpy() for t in terrains]
    ).astype(np.float32)

    total = sum(int(c.sum()) for c in clears) * n_commands
    for p, c in zip(map_paths, clears):
        print(f"  {p.name}: {int(c.sum())}/{G * G} spawn cells clear of the obstacle")
    print(f"[trials]   {total} per simulator ({total / max(chunk, 1):.0f} chunks of <= {chunk})")

    if dry_run:
        # Cheap CPU-only self check -- the lattice and the heightmap tensor must agree on
        # orientation and extent, or a label grid would be silently transposed relative to the
        # terrain a CNN sees.
        assert heightmaps.shape[1:] == (round(extent / resolution),) * 2, heightmaps.shape
        assert lattice[0, 0].tolist() == [-SPAWN_LIMIT, -SPAWN_LIMIT], lattice[0, 0]
        assert lattice[0, -1].tolist() == [SPAWN_LIMIT, -SPAWN_LIMIT], lattice[0, -1]
        assert lattice[-1, 0].tolist() == [-SPAWN_LIMIT, SPAWN_LIMIT], lattice[-1, 0]
        flat = HeightMapReader.flat(xlim=(-8.0, 8.0), ylim=(-8.0, 8.0))
        assert footprint_clear(flat, lattice).all(), "flat ground must clear every spawn cell"
        print("[dry-run]  lattice/heightmap convention + filter self-checks ok, nothing simulated")
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
        n_map_total = clear.size * n_commands
        n_map_blocked = int((~clear).sum()) * n_commands
        n_map_valid = int(mask.sum())
        n_map_diverged = n_map_total - n_map_blocked - n_map_valid
        print(
            f"    -> {n_map_valid} valid, {n_map_blocked} blocked, {n_map_diverged} diverged "
            f"({n_map_total} cells)"
        )

    y = np.concatenate(ys, axis=0)  # [R, G, G, 14], R = n_maps * n_commands
    mask = np.concatenate(masks, axis=0)  # [R, G, G]
    map_index = np.repeat(np.arange(n_maps, dtype=np.int64), n_commands)

    out_path = OUT_DIR / f"dataset_grid_{maps_dir.name}_M{n_maps}_L{n_commands}_g{G}.h5"
    write_grid_dataset(
        out_path,
        root=dict(
            n_maps=n_maps, n_commands=n_commands, n_rows=len(y), grid_n=G,
            spawn_step=SPAWN_STEP, spawn_limit=SPAWN_LIMIT, spawn_yaw=SPAWN_YAW,
            v_drive=V_DRIVE, wz_min=WZ_RANGE[0], wz_max=WZ_RANGE[1],
            duration_s=duration_s, resolution=resolution, extent=extent, seed=seed,
            mu=mu, k_turn=k_turn, k_p=K_P, chunk=chunk, settle_steps=settle_steps,
            maps_dir=str(maps_dir),
        ),
        per_row=dict(
            wz=wz_per_map.reshape(-1),
            map_index=map_index,
            map_path=np.array([str(map_paths[i]) for i in map_index]),
            y=y,
            mask=mask,
            spawn_xy=lattice.astype(np.float32),  # [G, G, 2], shared by every row
        ),
        grid=dict(
            heightmap=heightmaps,  # [n_maps, G_h, G_h] -- stored per MAP, indexed by map_index,
            # the same dedup philosophy terrain_fields() applies to the raw grids
            resolution=np.float64(resolution),
            extent=np.float64(extent),
        ),
        terrain_entries=[(map_paths[i], terrains[i]) for i in map_index],
    )
    n_blocked = sum(int((~c).sum()) for c in clears) * n_commands  # never simulated, same for
    # every command since footprint_clear only depends on the map, not on wz
    n_valid = int(mask.sum())
    n_total = mask.size
    n_diverged = n_total - n_blocked - n_valid  # footprint-clear (simulated) but failed the
    # finite/plausible-bounds check in simulate_map -- see that function's `ok` computation
    print("=" * 60)
    print(f"[summary]  {n_total} lattice cells across {len(y)} rows ({n_maps} maps x {n_commands} commands)")
    print(f"  valid     {n_valid:>7d}  ({100 * n_valid / n_total:5.1f}%)")
    print(f"  blocked   {n_blocked:>7d}  ({100 * n_blocked / n_total:5.1f}%)  -- obstacle footprint, never simulated")
    print(f"  diverged  {n_diverged:>7d}  ({100 * n_diverged / n_total:5.1f}%)  -- simulated but failed the finite/plausible-bounds check")
    print("=" * 60)


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest", version_base=None)
def main(cfg: DictConfig) -> None:
    generate(cfg)


if __name__ == "__main__":
    main()
