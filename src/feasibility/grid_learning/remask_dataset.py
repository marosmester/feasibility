"""Re-applies generate_dataset.py's per-trial displacement filter to an ALREADY GENERATED
dataset_grid_*.h5, without re-simulating anything.

generate_dataset.simulate_map decides a cell's validity at generation time, so a file written
before MAX_SPAWN_DISPLACEMENT existed still carries exploded contact solves as mask=True -- and
re-running train.py changes nothing, since custom_dataset.GridPoseErrorDataset reads that stored
mask verbatim. Re-simulating a 100-map run to fix it would be hours of GPU time for information
the file already has: the check needs only each cell's final poses (`y`) and its spawn position
(`spawn_xy`), both stored, and a cell that was valid had its real poses written rather than the
zero fill. So the mask can simply be recomputed.

    keep = mask & (|o_final_xy - spawn_xy| <= max_disp) & (|h_final_xy - spawn_xy| <= max_disp)

`mask &` is load-bearing, not belt-and-braces: an already-masked cell holds the all-zero fill, so
its "displacement" reads as |spawn_xy| (up to ~4.95 m at a lattice corner) and an unguarded
distance test would keep or drop it for a reason that has nothing to do with its solve. Newly
dropped cells get that same zero fill, preserving the file's "masked cells carry zeros, never
NaN" invariant (see generate_dataset.simulate_map's closing comment for why NaN is not an option).

Only the mask tightens -- a cell this rejects can never come back, and no cell it keeps changes
value -- so the operation is idempotent and safe to re-run at a lower threshold.

The threshold is imported from generate_dataset rather than restated here, unlike the
re-derivations elsewhere in grid_learning/ (footprint_clear, the SE(3) error formula): those keep
independent TREES independent, whereas this tool exists precisely to reproduce that module's
decision on old files, and a drifted copy would silently mint datasets inconsistent with newly
generated ones. The cost is that this otherwise numpy-only utility inherits generate_dataset's
ostrich/warp import, so it wants the same environment a generation run does.

Writing is copy-then-mutate (shutil.copy2 + h5py "r+"), not a re-serialization: `mask` and `y`
keep their shapes, so every other dataset, group and root attr -- the embedded terrain grids, the
git provenance of the run that actually produced the physics -- survives byte-for-byte instead of
being rebuilt by a second writer that could drift from write_grid_dataset. Three `remask_*` root
attrs record what was done, so a re-masked file stays as self-describing as a generated one.

CLI parameters:
    --file PATH        dataset_grid_*.h5 to re-mask (required)
    --out PATH         output path (default: alongside --file, "<stem>_d<max_disp>.h5")
    --in-place         mutate --file itself instead of writing a copy (refuses without --force)
    --force            confirm --in-place (there is no undo -- dropped cells are overwritten)
    --max-disp FLOAT   metres a final pose may sit from its spawn cell
                       (default: generate_dataset.MAX_SPAWN_DISPLACEMENT)
    --dry-run          report what would be dropped, write nothing
    --list-dropped INT print this many ready-to-paste gl_replay_grid.py commands for the dropped
                       cells, worst displacement first, so the threshold can be eyeballed before
                       it is committed to (default: 10; 0 disables)
    --self-test        run the filter's own asserts on synthetic arrays and exit (no file needed)

Usage:
    python src/feasibility/grid_learning/remask_dataset.py --self-test
    python src/feasibility/grid_learning/remask_dataset.py --file outputs/dataset_grid_0_M100_L10_g15.h5 --dry-run
    python src/feasibility/grid_learning/remask_dataset.py --file outputs/dataset_grid_0_M100_L10_g15.h5
"""
from __future__ import annotations

import argparse
import pathlib
import shutil

import h5py
import numpy as np

from feasibility.grid_learning.generate_dataset import MAX_SPAWN_DISPLACEMENT


def spawn_displacement(y: np.ndarray, spawn_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(ostrich, hstack) horizontal distance [R, G, G] from each cell's spawn to its final pose.

    `y` is [R, G, G, 14] = ostrich pose(7) + hstack pose(7), so the two (x, y) pairs live at
    columns 0:2 and 7:9; `spawn_xy` is the file's [G, G, 2] lattice, broadcast across rows since
    every row shares it. Only the horizontal distance is tested -- z is already bounded per map by
    generate_dataset.plausible_bounds, and a turn in place has no legitimate vertical travel to
    confuse it with."""
    spawn = spawn_xy[None]  # [1, G, G, 2] -> broadcasts over R
    return (
        np.linalg.norm(y[..., 0:2] - spawn, axis=-1),
        np.linalg.norm(y[..., 7:9] - spawn, axis=-1),
    )


def tightened_mask(
    y: np.ndarray, mask: np.ndarray, spawn_xy: np.ndarray, max_disp: float
) -> np.ndarray:
    """[R, G, G] bool: the cells of `mask` whose BOTH final poses stayed within `max_disp` of
    their spawn. Never promotes a cell -- see the module docstring on why the `mask &` guard is
    required rather than merely conservative."""
    o_disp, h_disp = spawn_displacement(y, spawn_xy)
    return mask & (o_disp <= max_disp) & (h_disp <= max_disp)


def report(y: np.ndarray, mask: np.ndarray, spawn_xy: np.ndarray, max_disp: float) -> np.ndarray:
    """Prints the displacement distribution over currently-valid cells and what `max_disp` costs,
    and returns the tightened mask. Percentiles are the point of the printout: the threshold is a
    judgement call, and the gap between the bulk of the distribution and its tail is the evidence
    for where to put it."""
    keep = tightened_mask(y, mask, spawn_xy, max_disp)
    o_disp, h_disp = spawn_displacement(y, spawn_xy)
    n_valid = int(mask.sum())
    n_drop = n_valid - int(keep.sum())

    print("=" * 78)
    print(f"[remask]   {mask.size} cells, {n_valid} currently valid, threshold {max_disp} m")
    if n_valid:
        qs = (50, 90, 99, 99.9)
        for name, d in (("ostrich", o_disp[mask]), ("hstack ", h_disp[mask])):
            p = np.percentile(d, qs)
            cols = "  ".join(f"q{q:<4g} {v:6.3f}" for q, v in zip(qs, p))
            print(f"  {name} displacement (m)  {cols}   max {d.max():7.3f}")
        o_only = int((mask & (o_disp > max_disp)).sum())
        h_only = int((mask & (h_disp > max_disp)).sum())
        print(f"  dropped   {n_drop:>7d}  ({100 * n_drop / n_valid:5.2f}% of valid)"
              f"  -- ostrich {o_only}, hstack {h_only}")
        print(f"  remaining {int(keep.sum()):>7d}")
    print("=" * 78)
    return keep


def print_dropped(
    y: np.ndarray, mask: np.ndarray, keep: np.ndarray, spawn_xy: np.ndarray,
    n_commands: int, file: pathlib.Path, limit: int,
) -> None:
    """Prints gl_replay_grid.py invocations for the `limit` worst dropped cells. Rows are laid out
    map-major (generate_dataset writes n_commands consecutive rows per map), so a row index splits
    into --map-index/--command-index by divmod. Freezing the viewer on one of these is the only
    real way to confirm a dropped cell is an exploded solve and not a violent but genuine jam."""
    dropped = mask & ~keep
    if not dropped.any() or limit <= 0:
        return
    o_disp, h_disp = spawn_displacement(y, spawn_xy)
    worst = np.maximum(o_disp, h_disp)
    rows, iis, jjs = np.nonzero(dropped)
    order = np.argsort(-worst[rows, iis, jjs])[:limit]
    print(f"[inspect]  {len(order)} of {len(rows)} dropped cells, worst displacement first:")
    for k in order:
        r, i, j = int(rows[k]), int(iis[k]), int(jjs[k])
        m, c = divmod(r, n_commands)
        print(f"  {worst[r, i, j]:8.2f} m  python src/feasibility/grid_learning/gl_replay_grid.py"
              f" --file {file} --map-index {m} --command-index {c} --cell {i} {j}")


def remask_file(
    src: pathlib.Path, dst: pathlib.Path, max_disp: float, *, list_dropped: int = 10
) -> None:
    """Copies `src` to `dst` (or mutates it, when they are the same path) with the tightened mask
    applied and the newly dropped cells zero-filled."""
    with h5py.File(src, "r") as f:
        y = f["y"][:]
        mask = f["mask"][:].astype(bool)
        spawn_xy = f["spawn_xy"][:]
        n_commands = int(f.attrs["n_commands"])

    keep = report(y, mask, spawn_xy, max_disp)
    print_dropped(y, mask, keep, spawn_xy, n_commands, src, list_dropped)
    if keep.sum() == mask.sum():
        print(f"[remask]   nothing to drop at {max_disp} m -- {dst} would be a plain copy")

    y[~keep] = 0.0  # the file's invariant: masked cells carry zeros, never NaN
    if dst != src:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)  # preserves every dataset/attr this tool does not touch, including
        # the terrain grids and the git provenance of the run that produced the physics
    with h5py.File(dst, "r+") as f:
        f["y"][...] = y  # same shapes, so h5py rewrites in place -- no schema rebuild
        f["mask"][...] = keep
        f.attrs["remask_max_spawn_displacement"] = float(max_disp)
        f.attrs["remask_source"] = str(src)
        f.attrs["remask_dropped"] = int(mask.sum() - keep.sum())
    print(f"saved {dst}")


def self_test() -> None:
    """Synthetic-array asserts, this package's stand-in for a pytest suite (see the repo CLAUDE.md:
    a module's __main__ block IS its test). Covers the three behaviours the filter has to get
    right -- keep the sane cell, drop the exploded one, and leave an already-masked cell masked
    even though its zero fill sits far from a corner spawn."""
    spawn_xy = np.array([[[3.0, 4.0], [0.0, 0.0]]], dtype=np.float32)  # [1, 2, 2]; |[3,4]| == 5
    y = np.zeros((1, 1, 2, 14), dtype=np.float32)
    y[0, 0, 0, 0:2] = [3.1, 4.0]  # ostrich moved 0.1 m from spawn (3, 4)
    y[0, 0, 0, 7:9] = [3.0, 4.2]  # hstack   moved 0.2 m
    y[0, 0, 1, 0:2] = [5.0, 0.0]  # ostrich flung 5 m from spawn (0, 0) -- an exploded solve
    mask = np.ones((1, 1, 2), dtype=bool)

    keep = tightened_mask(y, mask, spawn_xy, 1.0)
    assert keep.tolist() == [[[True, False]]], keep

    masked_corner = tightened_mask(y, np.zeros_like(mask), spawn_xy, 1.0)
    assert not masked_corner.any(), "a masked cell's zero fill must not be re-tested as a pose"

    assert tightened_mask(y, keep, spawn_xy, 1.0).tolist() == keep.tolist(), "not idempotent"
    print("[self-test] tightened_mask: keep/drop/masked-corner/idempotence ok")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=pathlib.Path, default=None, help="dataset_grid_*.h5 to re-mask")
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help='output path (default: "<stem>_d<max_disp>.h5" beside --file)')
    ap.add_argument("--in-place", action="store_true", help="mutate --file itself (needs --force)")
    ap.add_argument("--force", action="store_true", help="confirm --in-place; there is no undo")
    ap.add_argument("--max-disp", type=float, default=MAX_SPAWN_DISPLACEMENT,
                    help=f"metres from spawn a final pose may sit (default: {MAX_SPAWN_DISPLACEMENT})")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument("--list-dropped", type=int, default=10,
                    help="print this many gl_replay_grid.py commands for dropped cells (default: 10)")
    ap.add_argument("--self-test", action="store_true", help="run the filter's asserts and exit")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.file is None:
        raise SystemExit("--file is required (or use --self-test)")
    if args.in_place and not args.force:
        raise SystemExit("--in-place overwrites the dataset irreversibly; pass --force to confirm")

    if args.dry_run:
        with h5py.File(args.file, "r") as f:
            y, mask, spawn_xy = f["y"][:], f["mask"][:].astype(bool), f["spawn_xy"][:]
            n_commands = int(f.attrs["n_commands"])
        keep = report(y, mask, spawn_xy, args.max_disp)
        print_dropped(y, mask, keep, spawn_xy, n_commands, args.file, args.list_dropped)
        print("[dry-run]  nothing written")
        return

    if args.in_place:
        out = args.file
    elif args.out is not None:
        out = args.out
    else:  # "d0.5" rather than "d0_5": the threshold reads back at a glance and the shell is fine
        # with a dot in a filename, same spirit as the M/L/g tags generate_dataset already encodes
        out = args.file.with_name(f"{args.file.stem}_d{args.max_disp:g}.h5")
    remask_file(args.file, out, args.max_disp, list_dropped=args.list_dropped)


if __name__ == "__main__":
    main()
