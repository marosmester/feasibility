"""Self-description schema for comparator/batch_compare.py's output npz -- writer and reader
side kept together so they can't drift apart.

Two independent gaps `batch_compare.py`'s npz used to have (neither closed by just saving
`terrain_path`, a path string into assets/):

1. `terrain_path` is a REFERENCE, not the terrain itself. Regenerating
   `feasibility.heightmap.create_speed_bumps` with different heights/extent/cell overwrites the
   same asset filenames, so an old npz would silently start replaying against a terrain it was
   never actually run on, with no error. `terrain_fields()`/`terrain_from_npz()` embed the actual
   elevation grid + yaml sidecar per variant, so a viewer never has to touch assets/ again.
2. Neither `ostrich` nor `helhest_stack` (both actively-developed git submodules, see root
   CLAUDE.md) had any commit recorded, so a stored comparison couldn't be tied to the engine
   version that produced it. `git_provenance()` records both submodules' HEAD SHA and whether
   their tracked files were dirty at run time.
"""
from __future__ import annotations

import pathlib
import subprocess

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
    """{git_<name>_sha, git_<name>_dirty} for each submodule in SUBMODULES. sha is "unknown"
    when it can't be determined (git missing / not a checkout) -- that value doubles as the
    signal that the paired *_dirty=False is "not determined", not "verified clean".

    dirty uses `status --porcelain --untracked-files=no`: untracked files (e.g. this repo's own
    outputs/, or scratch files under a submodule) must NOT make a comparison look
    unreproducible -- only modifications to tracked files should. For ostrich this also flags a
    moved third_party/newton gitlink as dirty, which is correct: that really does change the
    physics."""
    fields: dict[str, object] = {}
    for name, repo in SUBMODULES.items():
        sha = _git(repo, "rev-parse", "HEAD")
        status = _git(repo, "status", "--porcelain", "--untracked-files=no")
        fields[f"git_{name}_sha"] = sha if sha is not None else "unknown"
        fields[f"git_{name}_dirty"] = bool(status)
    return fields


def terrain_fields(entries: list[tuple[pathlib.Path, HeightMapReader]]) -> dict[str, np.ndarray]:
    """entries: (path, terrain) pairs in variant order, `path` the assets/ stem the terrain was
    loaded from (used only to re-read its yaml sidecar verbatim; HeightMapReader itself keeps no
    raw yaml text -- load() parses it into a local dict and discards it). Returns the per-variant
    terrain block to splice into batch_compare's np.savez_compressed(...) call."""
    shapes = {(t.ny, t.nx) for _, t in entries}
    if len(shapes) > 1:
        bad = next((p, t) for p, t in entries if (t.ny, t.nx) != next(iter(shapes)))
        raise ValueError(f"terrain grids differ in shape across variants: e.g. {bad[0]} is {(bad[1].ny, bad[1].nx)}")

    return {
        # H: float32 -- load() already quantizes elevation to 8-bit png levels, so this loses
        # nothing, and the grid is near-constant across variants so savez_compressed shrinks it
        # to almost nothing. cell/origin/min_z/max_z stay float64: they're a handful of scalars
        # per variant (storage cost is noise), and unlike H's 255-level quantization, a value
        # like cell=0.05 has no exact float32 representation -- downcasting it would make a
        # round-tripped HeightMapReader.cell silently not bit-match the yaml sidecar's.
        "terrain_H": np.stack([t.H for _, t in entries], axis=0).astype(np.float32),
        "terrain_cell": np.array([t.cell for _, t in entries], dtype=np.float64),
        "terrain_origin": np.array([[t.x0, t.y0] for _, t in entries], dtype=np.float64),
        "terrain_min_z": np.array([t.min_z for _, t in entries], dtype=np.float64),
        "terrain_max_z": np.array([t.max_z for _, t in entries], dtype=np.float64),
        "terrain_yaml": np.array(
            [pathlib.Path(p).with_suffix(".yaml").read_text() for p, _ in entries]
        ),
    }


def terrain_from_npz(d, i: int) -> HeightMapReader:
    """Inverse of terrain_fields(): rebuild variant i's terrain straight from the npz, no
    assets/ files needed. Falls back to HeightMapReader.load(terrain_path[i]) against a
    pre-provenance npz that has no terrain_H key, so old outputs keep working."""
    if "terrain_H" not in d.files:
        return HeightMapReader.load(str(d["terrain_path"][i]))
    x0, y0 = d["terrain_origin"][i]
    return HeightMapReader(
        d["terrain_H"][i],
        (float(x0), float(y0)),
        float(d["terrain_cell"][i]),
        min_z=float(d["terrain_min_z"][i]),
        max_z=float(d["terrain_max_z"][i]),
    )
