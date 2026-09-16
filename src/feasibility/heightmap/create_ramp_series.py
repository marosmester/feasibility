"""The helhest_stack lattice-planner steepness benchmark's ramp series, in this repo's PNG+YAML
convention -- so ostrich can drive the exact maps `helhest_stack/scripts/bench_ramp_series.py`
plans on.

Each map is flat ground crossed by ONE symmetric full-width ridge: ramp up at `up_deg`, a
`plateau`-long flat top at `height`, ramp down at the same angle. Height is fixed, only the face
angle sweeps, and every map shares one grid sized from the shallowest angle. Start and goal sit on
opposite X edges, so a full-width ridge admits no detour: crossing it tests the climb AND the
descent in one straight drive.

The geometry is NOT re-derived here. `helhest.planning.rampmaps` (`bump_ridge`, `ridge_extent`,
`start_goal`) is imported and called with the same defaults `helhest_stack/scripts/
make_ramp_series.py` uses, so the two repos cannot drift apart -- this script only changes the
file format. `helhest` is an object of study here, the same stance `lattice_learning/arc.py` takes.

The one thing the format does change: `HeightMapReader.save` quantizes elevation to 8 bits, i.e.
`height / 255` = 2.9 mm levels for the default 0.75 m rise. That is why helhest_stack writes these
maps as float32 .npz instead. For ostrich, a 3 mm staircase under a 0.35 m wheel changes nothing
physically, and fitted over a whole face the angle is still accurate (the 5 deg face rises 8.7 mm
per cell, so it is the most affected). This script checks both claims on the saved files rather
than assuming them: the worst per-cell height error must stay within half a quantum, and the
face angle fitted to the loaded grid is printed next to the requested one and stored in the
sidecar as `fitted_deg`.

Output: `<out-dir>/ramp_a<tenths of deg, 4 digits>.png/.yaml`, the same angle-in-tenths naming as
helhest_stack's `bump_a0050.npz`, so a sorted glob returns sweep order. Each .yaml sidecar also
carries `up_deg`, `measured_deg` (the float grid's steepest rendered rise, rampmaps'
`face_angle_deg`), `fitted_deg`, `height`, `plateau`, `start` [x, y, yaw] and `goal` [x, y].
`HeightMapReader.load` ignores the extra keys. `manifest.yaml` records the sweep.

CLI parameters:
    --angles FLOAT...   face angles in deg (default: 5 10 ... 75, helhest_stack's DEFAULT_ANGLES)
    --out-dir PATH      output directory (default: assets/ramp_series)

Usage:
    python src/feasibility/heightmap/create_ramp_series.py
    python src/feasibility/heightmap/create_ramp_series.py --angles 20 40 60 --out-dir /tmp/ramps
"""
from __future__ import annotations

import argparse
import math
import pathlib

import numpy as np
import yaml
from helhest.planning.rampmaps import bump_ridge
from helhest.planning.rampmaps import DEFAULT_HEIGHT
from helhest.planning.rampmaps import DEFAULT_MARGIN
from helhest.planning.rampmaps import DEFAULT_PLATEAU
from helhest.planning.rampmaps import face_angle_deg
from helhest.planning.rampmaps import ramp_run
from helhest.planning.rampmaps import ridge_extent
from helhest.planning.rampmaps import start_goal

from feasibility.heightmap import HeightMapReader

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
ASSETS_DIR = REPO_ROOT / "assets" / "ramp_series"

# deg -- helhest_stack/scripts/make_ramp_series.py's DEFAULT_ANGLES. Restated rather than imported:
# scripts/ is not an importable package. Brackets the robot's settle envelope (descend 15, climb 25).
DEFAULT_ANGLES = tuple(float(d) for d in range(5, 76, 5))


def ramp_path(out_dir: pathlib.Path, up_deg: float) -> pathlib.Path:
    """<out_dir>/ramp_a0125 -- tenths of a degree, zero-padded, no dot (HeightMapReader appends
    .png/.yaml via with_suffix, which would eat a decimal point)."""
    return out_dir / f"ramp_a{round(up_deg * 10):04d}"


def ramp_series_paths(out_dir: pathlib.Path = ASSETS_DIR) -> list[pathlib.Path]:
    """Every saved map stem in `out_dir`, in sweep order -- what a consumer globs."""
    return [p.with_suffix("") for p in sorted(out_dir.glob("ramp_a*.png"))]


def fitted_face_deg(terrain: HeightMapReader, up_deg: float, height: float, plateau: float) -> float:
    """Least-squares slope of the rising face's interior cells along +X, in deg -- the angle the
    8-bit grid actually holds on average, as opposed to its steepest single-cell step."""
    run = ramp_run(up_deg, height)
    xs = terrain.x0 + (np.arange(terrain.nx) + 0.5) * terrain.cell
    foot, top = -run - 0.5 * plateau, -0.5 * plateau
    # stay one cell clear of both kinks so neither bilinear corner leaks into the fit
    face = (xs > foot + terrain.cell) & (xs < top - terrain.cell)
    if face.sum() < 2:  # steepest faces span under two interior cells; nothing to fit
        return float("nan")
    slope = np.polyfit(xs[face], terrain.H[0, face], 1)[0]
    return math.degrees(math.atan(slope))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--angles", type=float, nargs="+", default=list(DEFAULT_ANGLES))
    parser.add_argument("--out-dir", type=pathlib.Path, default=ASSETS_DIR)
    args = parser.parse_args()

    angles = sorted(args.angles)
    height, plateau, margin = DEFAULT_HEIGHT, DEFAULT_PLATEAU, DEFAULT_MARGIN
    # ONE grid for the whole series, sized from the shallowest (longest) ridge -- as helhest_stack does
    extent_x = ridge_extent(angles[0], height, plateau, margin)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    quantum = height / 255.0
    print(f"{'deg':>5} {'measured':>8} {'fitted':>7} {'max|dH| mm':>10} {'grid':>9}  file")
    for up_deg in angles:
        hm = bump_ridge(up_deg, extent_x, height, plateau)
        (sx, sy, syaw), (gx, gy) = start_goal(hm, margin)
        path = ramp_path(args.out_dir, up_deg)
        HeightMapReader(hm.H, (hm.x0, hm.y0), hm.cell, min_z=0.0, max_z=height).save(path)

        loaded = HeightMapReader.load(path)
        err = float(np.abs(loaded.H - hm.H).max())
        assert (loaded.nx, loaded.ny, loaded.cell) == (hm.nx, hm.ny, hm.cell), "grid changed on save"
        assert (loaded.x0, loaded.y0) == (hm.x0, hm.y0), "origin changed on save"
        assert err <= 0.5 * quantum + 1e-9, f"{path.name}: {err} m exceeds half an 8-bit quantum"
        measured = face_angle_deg(hm)
        fitted = fitted_face_deg(loaded, up_deg, height, plateau)

        meta = yaml.safe_load(path.with_suffix(".yaml").read_text())
        meta.update(
            up_deg=float(up_deg), measured_deg=float(measured), fitted_deg=float(fitted),
            height=float(height), plateau=float(plateau),
            start=[float(sx), float(sy), float(syaw)], goal=[float(gx), float(gy)],
        )
        path.with_suffix(".yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
        print(f"{up_deg:5.1f} {measured:8.2f} {fitted:7.2f} {1e3 * err:10.2f} "
              f"{hm.nx:4d}x{hm.ny:<4d}  {path.name}")

    manifest = {
        "source": "helhest.planning.rampmaps (bump_ridge / ridge_extent / start_goal)",
        "angles_deg": [float(a) for a in angles],
        "height": float(height), "plateau": float(plateau), "margin": float(margin),
        "extent_x": float(extent_x), "cell": float(hm.cell), "extent_y": float(hm.ny * hm.cell),
        "start": [float(sx), float(sy), float(syaw)], "goal": [float(gx), float(gy)],
        "quantum_m": float(quantum),
    }
    (args.out_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    print(f"\nwrote {len(angles)} maps + manifest.yaml to {args.out_dir}  "
          f"({extent_x:.2f} x {hm.ny * hm.cell:.1f} m at {hm.cell} m; start ({sx:.2f}, {sy:.2f}) "
          f"-> goal ({gx:.2f}, {gy:.2f}); 8-bit quantum {1e3 * quantum:.2f} mm)")


if __name__ == "__main__":
    main()
