"""Applies the world y-mirror symmetry to an ALREADY GENERATED dataset_grid_*.h5, doubling it
without re-simulating anything.

The Helhest tripod is exactly symmetric about its own centerline plane -- LEFT_WHEEL_POS
(0, +0.365, 0) and RIGHT_WHEEL_POS (0, -0.365, 0) are a mirror pair, REAR_WHEEL_POS sits on the
centerline, both CHASSIS_BOXES are y-centred (so CoM y = 0 and I_xy = I_yz = 0), wheel inertia is
diagonal, friction is isotropic, and gravity is along -z. Reflecting the whole scene about the
WORLD plane y = 0 therefore maps a valid trial onto another valid trial:

    heightmap   ->  flipped about y = 0        (row axis; see utils.grid_coords)
    wz          ->  -wz                        (cmd_to_wheels(0, -wz) swaps v_l and v_r exactly)
    spawn cell  ->  lattice row i -> G-1-i     (spawn_xy itself is symmetric, so it does NOT move)
    pose        ->  (x, -y, z), (-qx, qy, -qz, qw)

That last line is the whole geometric content of this module. Conjugating a rotation by
M = diag(1, -1, 1) gives R' = M R M, i.e. R'_ij = s_i s_j R_ij with s = (1, -1, 1), which in
quaternion form is exactly "negate qx and qz". Sanity: a pure yaw q = (0, 0, sin, cos) becomes
(0, 0, -sin, cos) = yaw negated; a pure pitch about y is untouched; a roll about x flips sign --
which is what the pseudovector rule omega -> (-wx, wy, -wz) demands.

The world plane y = 0 is the ONLY plane this can use. One row holds 225 spawn poses sharing one
terrain, each with its own body centerline, and no single reflection can be about all of them; but
the heightmap tensor is centred on the origin and the spawn lattice spans +-SPAWN_LIMIT, so y = 0
is the one plane that maps both onto themselves. That the robot need not sit at y = 0 is what
makes it work anyway: reflecting a robot at (x0, y0, yaw=0) about y = 0 relocates it to (x0, -y0)
and swaps its left/right wheels, and the wheel swap is precisely what -wz commands.

WHY IT IS EXACT AND NOT APPROXIMATE. The tensor is an even 100 cells with centres at +-4.95 ...,
so flipping the row axis is a true mirror about y = 0 rather than a half-cell-off one; the lattice
{-3.5 ... +3.5} step 0.5 is its own mirror; and the labels custom_dataset computes -- e_pos, a
norm, and e_rot, an arccos -- are non-negative SCALARS, hence mirror-INVARIANT. Nothing about the
target has to transform, so there is no reflected quantity to get wrong. The masked-cell zero fill
survives too: mirroring an all-zero pose gives an all-zero pose, so "masked cells carry zeros,
never NaN" still holds (see generate_dataset.simulate_map's closing comment).

ONLY the y-flip. Do not add an x-flip, a transpose or a 90-degree rotation: the robot is strongly
fore-aft asymmetric (rear wheel at x = -0.75, CoM at x = -0.198, and v_rear is undriven in a turn
in place), so it has exactly ONE mirror plane and this is a 2x augmentation, not the 8x dihedral
group a symmetric robot would give. The yaw lock is load-bearing as well -- a world y-flip sends
yaw -> -yaw, which is a no-op only because generate_dataset pins SPAWN_YAW = 0. If spawn yaw is
ever unlocked, mirror_poses() still holds but the spawn lattice grows a yaw column that must be
negated alongside it; this module refuses a file whose spawn_yaw attr is non-zero rather than
silently mislabelling one.

MAP PAIRING AND SPLIT LEAKAGE. A mirrored map is the same terrain as its source, so letting the
two land on opposite sides of a train/val split leaks as badly as the row-level split
custom_dataset.split_dataset_by_map already exists to avoid. Append mode therefore writes a
`grid/map_source` index -- mirrored map n_maps+m points back at m -- and split_dataset_by_map
groups by it when present. Files written before this field existed read back as map_source ==
arange(n_maps), i.e. the old behaviour exactly.

Unlike remask_dataset.py this imports nothing from generate_dataset: a reflection is pure
geometry, with no physics threshold to keep in step, so the tool stays numpy+h5py only and runs in
any environment rather than needing the one a generation run wants. It does import
utils.grid_coords for the self-check, so the "row axis is +Y" convention is verified against the
module that defines it rather than restated here.

Writing is a fresh serialization rather than remask_dataset.py's copy-then-mutate: row counts
change in append mode, so h5py cannot resize in place. The originals' arrays are copied through
verbatim -- including the embedded terrain grids and the git provenance of the run that actually
produced the physics -- and only the mirrored half is synthesized, so no original byte is
recomputed. Unrecognized datasets are a hard error, not a silent pass-through: this tool cannot
know how to double a field it has never seen.

CLI parameters:
    --file PATH      dataset_grid_*.h5 to mirror (required unless --self-test)
    --out PATH       output path (default: "<stem>_mirror.h5" beside --file)
    --mirror-only    write ONLY the mirrored copy (same row count) instead of original + mirror
    --dry-run        report what would be written, write nothing
    --self-test      run the geometry asserts on synthetic arrays and exit (no file needed)

Usage:
    python src/feasibility/grid_learning/mirror_dataset.py --self-test
    python src/feasibility/grid_learning/mirror_dataset.py --file outputs/dataset_grid_box_random_M100_L10_g15.h5 --dry-run
    python src/feasibility/grid_learning/mirror_dataset.py --file outputs/dataset_grid_box_random_M100_L10_g15.h5
"""
from __future__ import annotations

import argparse
import pathlib
import tempfile

import h5py
import numpy as np

from feasibility.grid_learning.utils import grid_coords

# (x, y, z, qx, qy, qz, qw) under reflection about the world plane y = 0. Position negates y;
# the rotation conjugates as R -> M R M with M = diag(1, -1, 1), which is "negate qx and qz".
# See the module docstring for the derivation and its three degenerate-case checks.
POSE_MIRROR_SIGNS = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0], dtype=np.float64)

LATTICE_Y_AXIS = 1  # axis of a [R, G, G, ...] label field that runs along +Y (spawn_lattice's
# meshgrid puts the row index on y); the same axis holds y in `grid/heightmap` [n_maps, G_h, G_h]
# (utils.grid_coords) and in `terrain/H` [U, ny, nx] (HeightMapReader's row = y0 + (i+0.5)*cell)

ROOT_DATASETS = ("wz", "map_index", "map_path", "y", "mask", "spawn_xy")
GRID_DATASETS = ("heightmap", "resolution", "extent", "map_source")  # map_source is written by
# this tool, not by generate_dataset, so it is optional on read and always present on write
TERRAIN_DATASETS = ("H", "cell", "origin", "min_z", "max_z", "yaml", "path", "variant_to_terrain")


def mirror_poses(y: np.ndarray) -> np.ndarray:
    """[..., 14] (ostrich pose(7) + hstack pose(7)) -> the same poses reflected about y = 0.

    Applies POSE_MIRROR_SIGNS to each 7-block. Deliberately NOT combined with the lattice flip:
    a pose reflection and a lattice re-indexing are independent operations and mirror_field()
    handles the second, so each can be tested on its own."""
    if y.shape[-1] % 7 != 0:
        raise ValueError(f"expected a whole number of 7-vectors in the last axis, got {y.shape}")
    signs = np.tile(POSE_MIRROR_SIGNS, y.shape[-1] // 7).astype(y.dtype)
    return y * signs


def mirror_field(a: np.ndarray, axis: int = LATTICE_Y_AXIS) -> np.ndarray:
    """Re-index a lattice-shaped array along its +Y axis: the mirrored world's cell at y belongs
    to the original world's cell at -y, and the lattice is symmetric, so that is row G-1-i.

    Applies equally to `y`, `mask` and (with the same axis) the heightmap/terrain grids, since
    every one of them puts +Y on the axis after the leading batch dimension."""
    return np.flip(a, axis=axis).copy()  # copy: h5py will not write a negative-stride view


def mirror_wz(wz: np.ndarray) -> np.ndarray:
    """Commanded yaw rate under the reflection. cmd_to_wheels(0, -wz) returns exactly the left and
    right wheel speeds of cmd_to_wheels(0, wz) swapped, and the rear wheel is 0 either way, so the
    mirrored command drives the mirrored robot."""
    return -wz


def mirror_origin(origin: np.ndarray, n_rows: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """[U, 2] (x0, y0) of the `terrain/` group's raw grids after the reflection.

    A HeightMapReader grid spans y in [y0, y0 + ny*cell], so its mirror spans
    [-(y0 + ny*cell), -y0] and the new y0 is the lower end of that. Flipping H's rows WITHOUT this
    would mirror each grid about its OWN centre rather than about the world plane y = 0 -- which
    is the same mirror only for a grid that happens to be origin-centred, and silently wrong for
    the ones that are not. x0 is untouched: the reflection preserves x."""
    out = np.array(origin, dtype=np.float64, copy=True)
    out[:, 1] = -(origin[:, 1] + n_rows * cell)
    return out


def read_source(f: h5py.File) -> dict[str, np.ndarray]:
    """Pulls every dataset this tool knows how to transform out of an open source file, and
    rejects anything it does not recognize. A schema addition should make this raise rather than
    be dropped on the floor or copied through under an assumption that may not hold."""
    for group, expected in ((f, ROOT_DATASETS), (f["grid"], GRID_DATASETS),
                            (f["terrain"], TERRAIN_DATASETS)):
        unknown = [k for k in group.keys() if k not in expected and not isinstance(group[k], h5py.Group)]
        if unknown:
            raise ValueError(
                f"{group.name} holds dataset(s) {unknown} this tool does not know how to mirror. "
                f"Add them to the *_DATASETS tuples (and to mirror_file) before using it."
            )
    data = {name: f[name][()] for name in ROOT_DATASETS}
    data.update({
        f"grid/{name}": f["grid"][name][()] for name in GRID_DATASETS if name in f["grid"]
    })
    data.update({
        f"terrain/{name}": f["terrain"][name][()]
        for name in TERRAIN_DATASETS if name in f["terrain"]
    })
    return data


def _as_str(a: np.ndarray) -> np.ndarray:
    """h5py hands back vlen strings as bytes; normalize to a numpy unicode array so the mirrored
    half can be built with ordinary string concatenation."""
    return np.array([s.decode() if isinstance(s, bytes) else s for s in a])


def mirrored_arrays(data: dict[str, np.ndarray], *, append: bool) -> dict[str, np.ndarray]:
    """Every dataset of the output file, built from the source's. `append` True concatenates the
    original and its mirror (2x the rows and maps); False returns the mirror alone, which is a
    strict involution -- mirroring twice reproduces the input, asserted in self_test()."""
    n_maps = int(data["grid/heightmap"].shape[0])
    n_terrains = int(data["terrain/H"].shape[0])

    # `variant_to_terrain` is written only when the source had something to dedup (see
    # provenance.terrain_fields); its absence means row i IS terrain row i, so normalize to the
    # explicit form here and decide again on write whether the output still needs it.
    vtt = data.get("terrain/variant_to_terrain")
    vtt = np.arange(len(data["wz"]), dtype=np.int64) if vtt is None else vtt.astype(np.int64)

    # `map_source` groups a map with the map it mirrors, so a by-map split cannot put the two on
    # opposite sides (see the module docstring). Absent on a file generate_dataset wrote, where
    # every map is its own group; present once this tool has run, and preserved from there on so
    # re-mirroring an already-mirrored file keeps pairing back to the ORIGINAL source map.
    map_source = data.get("grid/map_source")
    map_source = np.arange(n_maps, dtype=np.int64) if map_source is None else map_source.astype(np.int64)

    mirror = {
        "wz": mirror_wz(data["wz"]),
        "map_index": data["map_index"] + (n_maps if append else 0),
        # the mirrored terrain is not the asset on disk any more; keep the provenance trail but
        # make it unmistakable, and never loadable by accident
        "map_path": np.char.add(_as_str(data["map_path"]), "#mirrored"),
        "y": mirror_field(mirror_poses(data["y"])),
        "mask": mirror_field(data["mask"]),
        "grid/heightmap": mirror_field(data["grid/heightmap"]),
        "terrain/H": mirror_field(data["terrain/H"]),
        "terrain/origin": mirror_origin(
            data["terrain/origin"], np.asarray(data["terrain/H"].shape[1]), data["terrain/cell"]
        ),
        # cell / min_z / max_z are invariant: the reflection touches neither resolution nor the
        # elevation band, and z is untouched by a y-mirror
        "terrain/cell": data["terrain/cell"],
        "terrain/min_z": data["terrain/min_z"],
        "terrain/max_z": data["terrain/max_z"],
        # "" for both, which terrain_fields documents as "built in memory rather than loaded from
        # assets/": true of a mirrored grid, and better than a sidecar whose `origin` would now
        # contradict the `origin` dataset beside it. terrain_from_h5 reads neither.
        "terrain/yaml": np.array([""] * n_terrains),
        "terrain/path": np.array([""] * n_terrains),
        "terrain/variant_to_terrain": vtt + (n_terrains if append else 0),
    }
    # spawn_xy is already its own mirror (a symmetric lattice), and the grid geometry scalars
    # describe a window centred on the origin, so a reflection leaves all three alone
    mirror["spawn_xy"] = data["spawn_xy"]
    mirror["grid/resolution"] = data["grid/resolution"]
    mirror["grid/extent"] = data["grid/extent"]

    if not append:
        mirror["grid/map_source"] = map_source  # the mirror of map m is still terrain m
        return mirror

    out: dict[str, np.ndarray] = {}
    for key in ("wz", "map_index", "y", "mask", "terrain/H", "terrain/origin", "terrain/cell",
                "terrain/min_z", "terrain/max_z", "grid/heightmap", "terrain/variant_to_terrain"):
        out[key] = np.concatenate([data[key], mirror[key]], axis=0)
    for key in ("map_path", "terrain/yaml", "terrain/path"):
        out[key] = np.concatenate([_as_str(data[key]), mirror[key]])
    for key in ("spawn_xy", "grid/resolution", "grid/extent"):
        out[key] = data[key]
    # map n_maps+m is the mirror of map m -- see the module docstring on split leakage
    out["grid/map_source"] = np.tile(map_source, 2)
    return out


def write_arrays(dst: pathlib.Path, src: h5py.File, arrays: dict[str, np.ndarray],
                 attrs: dict[str, object]) -> None:
    """Serializes `arrays` into `dst` with write_grid_dataset's storage policy (gzip-4 except on
    scalars, explicit vlen dtype for strings), carrying the source's root attrs and `git/` group
    across so the physics run that produced the labels stays recorded."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(dst, "w") as f:
        for name, value in src.attrs.items():
            f.attrs[name] = value
        for name, value in attrs.items():
            f.attrs[name] = value

        for key, arr in arrays.items():
            group = f
            name = key
            if "/" in key:
                gname, name = key.split("/", 1)
                group = f.require_group(gname)
            arr = np.asarray(arr)
            if arr.dtype.kind == "U":
                group.create_dataset(name, data=arr.astype(object), dtype=h5py.string_dtype())
            elif arr.shape == ():
                group.create_dataset(name, data=arr)
            else:
                group.create_dataset(name, data=arr, compression="gzip", compression_opts=4)

        grp_git = f.require_group("git")
        for name, value in src["git"].attrs.items():
            grp_git.attrs[name] = value


def mirror_file(src: pathlib.Path, dst: pathlib.Path, *, append: bool = True,
                dry_run: bool = False) -> None:
    """Reads `src`, builds the mirrored (and by default also the original) rows, and writes
    `dst`."""
    with h5py.File(src, "r") as f:
        spawn_yaw = float(f.attrs.get("spawn_yaw", 0.0))
        if spawn_yaw != 0.0:
            raise ValueError(
                f"{src} was generated with spawn_yaw={spawn_yaw}, not 0. A world y-mirror sends "
                f"yaw -> -yaw, which is only a no-op at yaw 0; this file's spawn poses would need "
                f"their yaw negated too (see the module docstring)."
            )
        data = read_source(f)
        arrays = mirrored_arrays(data, append=append)

        n_maps_in = int(data["grid/heightmap"].shape[0])
        n_rows_in = int(len(data["wz"]))
        n_maps_out = int(arrays["grid/heightmap"].shape[0])
        n_rows_out = int(len(arrays["wz"]))
        wz_min, wz_max = float(f.attrs.get("wz_min", -1.0)), float(f.attrs.get("wz_max", 1.0))

        print("=" * 78)
        print(f"[mirror]   {src}")
        print(f"  mode      {'append (original + mirror)' if append else 'mirror only'}")
        print(f"  maps      {n_maps_in} -> {n_maps_out}")
        print(f"  rows      {n_rows_in} -> {n_rows_out}")
        print(f"  valid     {int(data['mask'].sum())} -> {int(arrays['mask'].sum())} cells")
        print(f"  wz        [{wz_min:+.3f}, {wz_max:+.3f}] -> "
              f"[{min(wz_min, -wz_max):+.3f}, {max(wz_max, -wz_min):+.3f}] rad/s")
        print("=" * 78)
        if dry_run:
            print("[dry-run]  nothing written")
            return

        write_arrays(
            dst, f, arrays,
            attrs=dict(
                n_maps=n_maps_out,
                n_rows=n_rows_out,
                # the command range's symmetric hull: a no-op for the symmetric WZ_RANGE
                # generate_dataset uses, but correct if a future run sweeps an asymmetric one
                wz_min=min(wz_min, -wz_max),
                wz_max=max(wz_max, -wz_min),
                mirror_source=str(src),
                mirror_mode="append" if append else "mirror_only",
                mirror_rows_added=n_rows_out - n_rows_in,
            ),
        )
    print(f"saved {dst}")


def self_test() -> None:
    """Synthetic asserts, this package's stand-in for a pytest suite (see the repo CLAUDE.md: a
    module's __main__ block IS its test). Covers the reflection's geometry, the label invariance
    that makes the augmentation legal at all, the terrain-origin bookkeeping, and one end-to-end
    file round trip."""
    rng = np.random.default_rng(0)

    # --- the pose reflection is an involution, and negates yaw while preserving pitch ----------
    poses = rng.normal(size=(5, 14))
    assert np.allclose(mirror_poses(mirror_poses(poses)), poses), "pose mirror is not involutive"
    for name, axis, flips in (("yaw", 2, True), ("pitch", 1, False), ("roll", 0, True)):
        q = np.zeros(7)
        q[3 + axis], q[6] = np.sin(0.3), np.cos(0.3)
        got = mirror_poses(q[None])[0, 3 + axis]
        assert np.isclose(got, -q[3 + axis] if flips else q[3 + axis]), name
    print("[self-test] pose mirror: involutive, yaw/roll negate, pitch preserved")

    # --- THE property the augmentation rests on: the (e_pos, e_rot) labels custom_dataset
    # computes are non-negative scalars, so they must come out IDENTICAL under the reflection.
    # Checked against the real se3_errors rather than a local copy -- if that formula ever
    # changes, this augmentation's validity has to be re-argued, and this assert is where it
    # would surface. -----------------------------------------------------------------------------
    from feasibility.grid_learning.custom_dataset import poses_to_se3, se3_errors

    raw = rng.normal(size=(3, 4, 4, 14))
    raw[..., 3:7] /= np.linalg.norm(raw[..., 3:7], axis=-1, keepdims=True)  # unit quaternions
    raw[..., 10:14] /= np.linalg.norm(raw[..., 10:14], axis=-1, keepdims=True)
    before = se3_errors(poses_to_se3(raw[..., :7]), poses_to_se3(raw[..., 7:]))
    after = se3_errors(*(poses_to_se3(p) for p in (mirror_poses(raw)[..., :7],
                                                   mirror_poses(raw)[..., 7:])))
    for name, b, a in zip(("e_pos", "e_rot"), before, after):
        assert np.allclose(b, a, atol=1e-10), f"{name} is not mirror-invariant: {np.abs(b - a).max()}"
    print("[self-test] se3_errors: (e_pos, e_rot) invariant under the reflection")

    # --- registration: a bump at +y must land at -y, verified against utils.grid_coords rather
    # than against this module's own idea of which axis is which -------------------------------
    coords = grid_coords(resolution=0.1, extent=10.0)
    n = len(coords)
    hm = np.zeros((1, n, n), dtype=np.float32)
    row = int(np.argmin(np.abs(coords - 2.0)))  # the cell whose centre is nearest y = +2.0 m
    hm[0, row, :] = 1.0
    flipped_row = int(np.argmax(mirror_field(hm)[0, :, 0]))
    assert np.isclose(coords[flipped_row], -coords[row]), (coords[flipped_row], -coords[row])
    print(f"[self-test] registration: bump at y={coords[row]:+.2f} m -> "
          f"y={coords[flipped_row]:+.2f} m")

    # --- terrain origin: an OFF-CENTRE raw grid must mirror about the world plane, not its own
    # centre. Row i of the mirrored grid has to sit at minus the world y of row ny-1-i. --------
    origin = np.array([[-1.0, 0.5]])  # y0 = +0.5: deliberately not origin-centred
    ny, cell = 4, np.array([0.25])
    new_origin = mirror_origin(origin, np.array(ny), cell)
    for i in range(ny):
        assert np.isclose(new_origin[0, 1] + (i + 0.5) * cell[0],
                          -(origin[0, 1] + (ny - 1 - i + 0.5) * cell[0])), i
    assert new_origin[0, 0] == origin[0, 0], "x0 must not move under a y-mirror"
    print(f"[self-test] terrain origin: y0 {origin[0, 1]:+.3f} -> {new_origin[0, 1]:+.3f} m")

    # --- end-to-end: a synthetic file, mirrored twice in --mirror-only mode, must reproduce
    # itself exactly (the reflection is an involution on the whole schema, not just on poses) --
    with tempfile.TemporaryDirectory() as tmp:
        a, b, c = (pathlib.Path(tmp) / f"{s}.h5" for s in ("a", "b", "c"))
        _write_synthetic(a, rng)
        mirror_file(a, b, append=False)
        mirror_file(b, c, append=False)
        with h5py.File(a, "r") as fa, h5py.File(c, "r") as fc:
            for key in ("wz", "y", "mask", "spawn_xy", "map_index", "grid/heightmap",
                        "terrain/H", "terrain/origin"):
                assert np.allclose(fa[key][()], fc[key][()]), f"round trip changed {key}"
        _write_synthetic(a, rng)
        mirror_file(a, b, append=True)
        with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
            r = len(fa["wz"])
            assert len(fb["wz"]) == 2 * r
            assert np.allclose(fb["wz"][r:], -fa["wz"][()]), "appended half is not the mirror"
            assert np.allclose(fb["grid/map_source"][()], np.tile(np.arange(2), 2)), "map_source"
    print("[self-test] file round trip: mirror-only is involutive, append doubles and pairs")
    print("all self-checks ok")


def _write_synthetic(path: pathlib.Path, rng: np.random.Generator) -> None:
    """A minimal file in write_grid_dataset's schema, for the round-trip check above."""
    n_maps, n_commands, G, Gh, ny = 2, 3, 5, 8, 6
    r = n_maps * n_commands
    y = rng.normal(size=(r, G, G, 14)).astype(np.float32)
    mask = rng.random((r, G, G)) > 0.3
    y[~mask] = 0.0
    coords = np.linspace(-1.0, 1.0, G)
    X, Y = np.meshgrid(coords, coords)
    with h5py.File(path, "w") as f:
        f.attrs.update(dict(n_maps=n_maps, n_commands=n_commands, n_rows=r, grid_n=G,
                            spawn_yaw=0.0, wz_min=-1.0, wz_max=1.0))
        f.create_dataset("wz", data=rng.uniform(-1, 1, r).astype(np.float32))
        f.create_dataset("map_index", data=np.repeat(np.arange(n_maps), n_commands))
        f.create_dataset("map_path", data=np.array(["m"] * r, dtype=object),
                         dtype=h5py.string_dtype())
        f.create_dataset("y", data=y)
        f.create_dataset("mask", data=mask)
        f.create_dataset("spawn_xy", data=np.stack([X, Y], -1).astype(np.float32))
        g = f.create_group("grid")
        g.create_dataset("heightmap", data=rng.normal(size=(n_maps, Gh, Gh)).astype(np.float32))
        g.create_dataset("resolution", data=np.float64(0.1))
        g.create_dataset("extent", data=np.float64(Gh * 0.1))
        t = f.create_group("terrain")
        t.create_dataset("H", data=rng.normal(size=(n_maps, ny, ny)).astype(np.float32))
        t.create_dataset("cell", data=np.full(n_maps, 0.05))
        t.create_dataset("origin", data=np.tile([-0.15, 0.35], (n_maps, 1)))
        t.create_dataset("min_z", data=np.zeros(n_maps))
        t.create_dataset("max_z", data=np.ones(n_maps))
        t.create_dataset("yaml", data=np.array(["y"] * n_maps, dtype=object),
                         dtype=h5py.string_dtype())
        t.create_dataset("path", data=np.array(["p"] * n_maps, dtype=object),
                         dtype=h5py.string_dtype())
        t.create_dataset("variant_to_terrain", data=np.repeat(np.arange(n_maps), n_commands))
        f.create_group("git").attrs["ostrich_sha"] = "deadbeef"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--file", type=pathlib.Path, default=None, help="dataset_grid_*.h5 to mirror")
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help='output path (default: "<stem>_mirror.h5" beside --file)')
    ap.add_argument("--mirror-only", action="store_true",
                    help="write only the mirrored rows instead of original + mirror")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument("--self-test", action="store_true", help="run the geometry asserts and exit")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.self_test:
        self_test()
    elif args.file is None:
        raise SystemExit("--file is required (or use --self-test)")
    else:
        suffix = "_mirroronly" if args.mirror_only else "_mirror"
        out = args.out or args.file.with_name(f"{args.file.stem}{suffix}.h5")
        if out == args.file:
            raise SystemExit("refusing to overwrite --file; pass a different --out")
        mirror_file(args.file, out, append=not args.mirror_only, dry_run=args.dry_run)
