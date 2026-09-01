"""Self-description schema for comparator/compare_*.py's output HDF5 (via comparator.common's
run_comparison) -- writer and reader side kept together so they can't drift apart.

Two independent gaps `batch_compare.py`'s npz used to have (neither closed by just saving
`terrain_path`, a path string into assets/):

1. `terrain_path` is a REFERENCE, not the terrain itself. Regenerating
   `feasibility.heightmap.create_speed_bumps` with different heights/extent/cell overwrites the
   same asset filenames, so an old file would silently start replaying against a terrain it was
   never actually run on, with no error. `terrain_fields()`/`terrain_from_h5()` embed the actual
   elevation grid + yaml sidecar per variant, so a viewer never has to touch assets/ again.
2. Neither `ostrich` nor `helhest_stack` (both actively-developed git submodules, see root
   CLAUDE.md) had any commit recorded, so a stored comparison couldn't be tied to the engine
   version that produced it. `git_provenance()` records both submodules' HEAD SHA and whether
   their tracked files were dirty at run time.

HDF5 layout written by write_comparison(): scalars as root attrs, per-variant/time-series arrays
as datasets, grouped `terrain/`, `git/`, `ostrich/`, `hstack/` -- matching the grouping+attrs
convention ostrich's own loggers use (ostrich/src/ostrich/logging/*_logger.py).

write_run()/read_run() are the single-run counterpart used by the demos/ scripts: one simulator
and one trajectory, so the arrays are flat [T, ...] and the two sim groups are absent, but the
same embedded `terrain/` + `git/` groups close both gaps above.
"""
from __future__ import annotations

import pathlib
import subprocess

import h5py
import numpy as np

from feasibility.heightmap import HeightMapReader

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

SUBMODULES = {"ostrich": REPO_ROOT / "ostrich", "helhest_stack": REPO_ROOT / "helhest_stack"}


def _git(repo: pathlib.Path, *args: str) -> str | None:
    """Run `git -C repo *args`, return stripped stdout or None on any failure (git missing,
    repo path not a git checkout, timeout, ...). Provenance is a nice-to-have, never worth
    aborting a multi-minute comparison run over."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def git_provenance() -> dict[str, object]:
    """{<name>_sha, <name>_dirty} for each submodule in SUBMODULES -- written into the file's
    `git/` group attrs by write_comparison(). sha is "unknown" when it can't be determined (git
    missing / not a checkout) -- that value doubles as the signal that the paired *_dirty=False
    is "not determined", not "verified clean".

    dirty uses `status --porcelain --untracked-files=no`: untracked files (e.g. this repo's own
    outputs/, or scratch files under a submodule) must NOT make a comparison look
    unreproducible -- only modifications to tracked files should. For ostrich this also flags a
    moved third_party/newton gitlink as dirty, which is correct: that really does change the
    physics."""
    fields: dict[str, object] = {}
    for name, repo in SUBMODULES.items():
        sha = _git(repo, "rev-parse", "HEAD")
        status = _git(repo, "status", "--porcelain", "--untracked-files=no")
        fields[f"{name}_sha"] = sha if sha is not None else "unknown"
        fields[f"{name}_dirty"] = bool(status)
    return fields


def terrain_fields(
    entries: list[tuple[pathlib.Path | None, HeightMapReader]]
) -> dict[str, np.ndarray]:
    """entries: (path, terrain) pairs in variant order, `path` the assets/ stem the terrain was
    loaded from (used both as an archival reference and to re-read its yaml sidecar verbatim;
    HeightMapReader itself keeps no raw yaml text -- load() parses it into a local dict and
    discards it). Returns the terrain block to splice into the file's `terrain/` group by
    write_comparison()/write_run().

    `path` may be None for a terrain that was built in memory rather than loaded from assets/
    (e.g. HeightMapReader.flat() in demos/ostrich_vel_cmd.py) -- yaml/path record "" for that
    variant. The elevation grid itself is embedded either way, so a None path costs only the
    archival reference, never the ability to replay.

    Deduplicated by HeightMapReader identity, not stacked one-per-variant: callers whose terrain
    is genuinely shared across every variant (generate_dataset.py's one terrain reused for every
    sample, run_trial_comparison's one terrain shared by every Trial -- both pass the SAME
    HeightMapReader object n times) previously had terrain_fields() np.stack() n identical
    copies of H, e.g. ~4GB of duplicate host RAM at n=10000 on the default 16m/0.05m grid, before
    gzip discarded the redundancy on write. Deduping up front means that RAM spike never happens.
    A `variant_to_terrain` [n] int index is only written when it's non-trivial (some variants
    actually share a terrain) -- run_comparison's per-height variants each load() their own
    HeightMapReader, so nothing is deduped there and the field is omitted; terrain_from_h5()
    treats a missing index as the old direct-index behavior, so already-written files (which
    never had this field) still read back correctly."""
    shapes = {(t.ny, t.nx) for _, t in entries}
    if len(shapes) > 1:
        bad = next((p, t) for p, t in entries if (t.ny, t.nx) != next(iter(shapes)))
        raise ValueError(f"terrain grids differ in shape across variants: e.g. {bad[0]} is {(bad[1].ny, bad[1].nx)}")

    unique_entries: list[tuple[pathlib.Path | None, HeightMapReader]] = []
    seen: dict[int, int] = {}  # id(terrain) -> index into unique_entries
    variant_to_terrain = np.empty(len(entries), dtype=np.int64)
    for i, (p, t) in enumerate(entries):
        j = seen.setdefault(id(t), len(unique_entries))
        if j == len(unique_entries):
            unique_entries.append((p, t))
        variant_to_terrain[i] = j

    fields = {
        # H: float32 -- load() already quantizes elevation to 8-bit png levels, so this loses
        # nothing, and the grid is near-constant across genuinely-different variants (e.g. a
        # bump/box series) too, so gzip shrinks it further still. cell/origin/min_z/max_z stay
        # float64: they're a handful of scalars per unique terrain (storage cost is noise), and
        # unlike H's 255-level quantization, a value like cell=0.05 has no exact float32
        # representation -- downcasting it would make a round-tripped HeightMapReader.cell
        # silently not bit-match the yaml sidecar's.
        "H": np.stack([t.H for _, t in unique_entries], axis=0).astype(np.float32),
        "cell": np.array([t.cell for _, t in unique_entries], dtype=np.float64),
        "origin": np.array([[t.x0, t.y0] for _, t in unique_entries], dtype=np.float64),
        "min_z": np.array([t.min_z for _, t in unique_entries], dtype=np.float64),
        "max_z": np.array([t.max_z for _, t in unique_entries], dtype=np.float64),
        "yaml": np.array(
            [
                "" if p is None else pathlib.Path(p).with_suffix(".yaml").read_text()
                for p, _ in unique_entries
            ]
        ),
        "path": np.array(["" if p is None else str(p) for p, _ in unique_entries]),
    }
    if len(unique_entries) < len(entries):
        fields["variant_to_terrain"] = variant_to_terrain
    return fields


def _terrain_row(grp: h5py.Group, j: int) -> HeightMapReader:
    """Build the HeightMapReader for unique-terrain row `j` of an already-opened `terrain/`
    group -- the shared body of terrain_from_h5()/unique_terrains_from_h5(), kept in one place
    so a schema change to the terrain/ group only has to be applied once."""
    x0, y0 = grp["origin"][j]
    return HeightMapReader(
        grp["H"][j],
        (float(x0), float(y0)),
        float(grp["cell"][j]),
        min_z=float(grp["min_z"][j]),
        max_z=float(grp["max_z"][j]),
    )


def terrain_from_h5(f: h5py.File, i: int) -> HeightMapReader:
    """Inverse of terrain_fields(): rebuild variant i's terrain straight from the file, no
    assets/ files needed. `variant_to_terrain` (see terrain_fields()) indirects i to the
    underlying unique-terrain row when present; its absence (every file written before
    deduplication, or a file with nothing to dedupe) means i already IS that row, matching the
    old direct-index layout."""
    grp = f["terrain"]
    j = int(grp["variant_to_terrain"][i]) if "variant_to_terrain" in grp else i
    return _terrain_row(grp, j)


def unique_terrains_from_h5(f: h5py.File, n: int) -> tuple[list[HeightMapReader], np.ndarray]:
    """Bulk counterpart to terrain_from_h5(), for a reader that needs EVERY variant's terrain at
    once: the deduplicated terrain list plus the [n] variant -> terrain index into it.

    terrain_from_h5(f, i) in a loop would re-materialize the same shared grid n times -- for
    generate_dataset.py's files, where all n variants point at one terrain (see terrain_fields()
    on deduplication), that is n copies of a 321x321 grid to read one. Returning the unique set
    plus the index lets a caller do its per-terrain work once and scatter the result, which is
    what learning/custom_dataset.py's patch sampling does.

    `variant_to_terrain`'s absence means nothing was deduped, i.e. variant i IS terrain row i --
    the same convention terrain_from_h5() applies, kept in step with it here."""
    grp = f["terrain"]
    index = (
        np.asarray(grp["variant_to_terrain"][()], dtype=np.int64)
        if "variant_to_terrain" in grp
        else np.arange(n, dtype=np.int64)
    )
    terrains = [_terrain_row(grp, j) for j in range(grp["H"].shape[0])]
    return terrains, index


def _write_group(grp: h5py.Group, fields: dict[str, np.ndarray]) -> None:
    """Write each array in `fields` as a compressed dataset of `grp` (gzip/4, matching every
    ostrich HDF5 logger). Scalar (shape-()) arrays and string arrays skip compression -- h5py
    rejects chunking/compression on scalar datasets, and gzip buys nothing on the handful of
    label/path/yaml strings here."""
    for name, arr in fields.items():
        arr = np.asarray(arr)
        if arr.dtype.kind == "U":
            grp.create_dataset(name, data=arr.astype(object), dtype=h5py.string_dtype())
        elif arr.shape == ():
            grp.create_dataset(name, data=arr)
        else:
            grp.create_dataset(name, data=arr, compression="gzip", compression_opts=4)


def write_comparison(
    path: pathlib.Path,
    *,
    root: dict[str, object],
    per_variant: dict[str, np.ndarray],
    terrain_entries: list[tuple[pathlib.Path, HeightMapReader]],
    ostrich: dict[str, np.ndarray],
    hstack: dict[str, np.ndarray],
) -> None:
    """The single writer shared by run_comparison/run_trial_comparison. `root` scalars become
    file-level attrs; `per_variant` arrays (spawn_pose, v_drive, wz_drive, variant_value,
    variant_label, ...) become file-level datasets; `terrain_entries` populates `terrain/` via
    terrain_fields(); git_provenance() populates `git/`; `ostrich`/`hstack` populate their own
    groups, with each dict's `dt` key lifted to a group attr (a scalar, so it doesn't belong
    alongside the [T, n, ...] time-series datasets)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for name, value in root.items():
            f.attrs[name] = value

        _write_group(f, per_variant)

        grp_terrain = f.create_group("terrain")
        _write_group(grp_terrain, terrain_fields(terrain_entries))

        grp_git = f.create_group("git")
        for name, value in git_provenance().items():
            grp_git.attrs[name] = value

        for group_name, fields in (("ostrich", ostrich), ("hstack", hstack)):
            grp = f.create_group(group_name)
            fields = dict(fields)
            grp.attrs["dt"] = fields.pop("dt")
            _write_group(grp, fields)

    print(f"saved {path}")


def write_run(
    path: pathlib.Path,
    *,
    attrs: dict[str, object],
    arrays: dict[str, np.ndarray],
    terrain: HeightMapReader,
    terrain_path: pathlib.Path | None = None,
) -> None:
    """Single-run counterpart to write_comparison(), for the demos/ scripts: ONE simulator, ONE
    trajectory, so the arrays are flat [T, ...] with no per-variant axis and there are no
    `ostrich/`/`hstack/` groups. Everything else matches -- `attrs` become root attrs, `arrays`
    root datasets, and the same embedded `terrain/` + `git/` groups make the file self-describing
    (see this module's docstring for why a `terrain_path` string alone is not enough).

    `terrain_path` is the assets/ stem the terrain came from, or None when it was built in
    memory (HeightMapReader.flat()) -- see terrain_fields()."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for name, value in attrs.items():
            f.attrs[name] = value

        _write_group(f, arrays)

        grp_terrain = f.create_group("terrain")
        # One-entry variant list: the terrain/ group keeps its leading axis of length 1, so
        # terrain_from_h5(f, 0) reads a single-run file and a comparison file identically.
        _write_group(grp_terrain, terrain_fields([(terrain_path, terrain)]))

        grp_git = f.create_group("git")
        for name, value in git_provenance().items():
            grp_git.attrs[name] = value

    print(f"saved {path}")


def read_run(
    path: pathlib.Path,
) -> tuple[dict[str, object], dict[str, np.ndarray], HeightMapReader]:
    """Inverse of write_run(): (attrs, arrays, terrain). Everything is materialized to numpy
    inside the `with` -- h5py datasets are invalid once the file closes."""
    with h5py.File(path, "r") as f:
        attrs = dict(f.attrs)
        arrays = {name: f[name][()] for name in f if not isinstance(f[name], h5py.Group)}
        terrain = terrain_from_h5(f, 0)
    return attrs, arrays, terrain
