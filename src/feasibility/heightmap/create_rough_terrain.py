"""Generate a band-limited random rough-terrain heightmap: ROUGH ground, not a HILLY one --
small-amplitude, short-wavelength texture the wheels feel individually, with no feature large
enough for the whole robot to just tilt and ride over (that's create_surface.py's Blender
"Landscape" mesh, which is exactly the failure mode this script avoids: its dominant wavelength
is ~5-10 m against ~1.5 m of relief, i.e. gentle 8-15 deg hills the kinematic twin tracks just
fine).

Method: spectral synthesis, band-limited on BOTH ends. White noise -> rFFT -> multiply by a
radial amplitude filter k**(-beta/2) for wavenumber k=1/wavelength (i.e. a power spectrum
S(k) ~ k**-beta, the same power-law family road-roughness standards like ISO 8608 use), zeroed
outside [1/--cutoff-wavelength, 1/--min-wavelength] -> irFFT -> rescale to the target RMS. Both
cutoffs are HARD ZEROS, not damping, so widening either one can't silently regrow the thing it
excludes. FFT synthesis is also exactly periodic over the grid, so the field tiles with no seam
if it's ever repeated -- unlike diamond-square or raw fBm/Perlin octaves, which are hill-shaped
by construction (their lowest octave IS a hill spanning the whole domain) and would need those
low octaves discarded to do the same job less precisely.

The high-frequency cutoff is NOT optional polish -- a low-pass-only filter (--min-wavelength
disabled) diverges. In 2D, height variance integrates S(k)*k dk (mode count grows as k) but
SLOPE variance integrates k^2*S(k)*k dk = k^{3-beta} dk, which only converges for beta > 4; the
beta~2-3 range that makes height look naturally rough leaves slope dominated by whatever the
finest grid scale contains. Try it: drop --min-wavelength and even the "rms=0.035 m" default
below produces 60+ deg single-cell spikes -- textbook aliasing, not terrain. --min-wavelength
caps that: nothing narrower survives, so slope statistics are set by --min-wavelength itself,
not by grid resolution.

Three independent knobs set the physical character, not the visual one:
* --cutoff-wavelength (m) -- longest feature let through. Below the robot's own footprint
  (wheelbase 0.75 m, track 0.73 m) a feature can no longer just tilt the chassis; it forces the
  three wheels into independently conflicting contact constraints, which is where ostrich
  (exact contact) and the kinematic twin (quasi-static 3x3 settle) actually diverge. The
  default (3.0 m) sits well above that so most of the spectrum is available; tighten it for a
  visibly rougher, less hilly field.
* --min-wavelength (m) -- shortest feature let through -- see the divergence note above. The
  default (0.6 m) is comfortably above both the grid's Nyquist wavelength (2*cell = 0.1 m) and
  the wheel width (0.10 m), so it bounds slope statistics without filtering out anything a
  0.35 m-radius wheel would notice anyway.
* --rms (m) -- target root-mean-square elevation. GROUND_CLEARANCE below (derived from
  HelhestJuniorConfig, ~0.25 m) is the hard ceiling on PEAK-TO-PEAK relief before the chassis
  starts high-centering -- which helhest_stack's engine only detects (valid=False), so an
  oversized --rms doesn't fail loudly, it just quietly drops rows out of a downstream dataset.
  main() prints achieved peak-to-peak against this ceiling every run.

--beta sets the spectral slope (S(k) ~ k**-beta): higher looks smoother/rounder within the
passband, lower pushes energy toward the short-wavelength end (grittier). 2.0-3.0 is the
natural-terrain range quoted by roughness standards; it's a texture knob, not a safety one.

Grid, cell, and seed round out the knobs: --extent/--cell size the square grid (must resolve
--cutoff-wavelength, i.e. cutoff-wavelength >= 2*cell, the grid's Nyquist wavelength, or the
whole passband would be empty); --seed selects the random realization -- call this script
repeatedly with different seeds for variety, same physical limits every time. Default output is
named by seed (rough_seed<NNNN>); pass --out for a custom path, e.g. when sweeping --rms itself
into a named series (mirroring create_box_obstacles.py's height series) for a dataset generator.

The generation params (including achieved RMS, for comparison against the --rms target) are
appended into the saved .yaml sidecar alongside HeightMapReader's own resolution/origin/min_z/
max_z keys -- harmless extra keys HeightMapReader.load() ignores -- so a generated terrain is
self-describing the same way comparator/provenance.py's HDF5 runs are.

CLI parameters:
    --extent FLOAT             full width/height of the square grid, in meters (default: 16.0)
    --cell FLOAT                grid resolution in meters (default: 0.05)
    --cutoff-wavelength FLOAT   longest wavelength let through, in meters -- longer features are
                                 hills and are suppressed entirely (default: 3.0)
    --min-wavelength FLOAT      shortest wavelength let through, in meters -- caps slope
                                 statistics; see module docstring's divergence note (default: 0.6)
    --beta FLOAT                power-spectrum slope S(k) ~ k^-beta (default: 2.5)
    --rms FLOAT                 target RMS elevation, in meters (default: 0.035)
    --seed INT                  RNG seed selecting the random realization (default: 0)
    --out PATH                  output path stem (no extension), overriding the default
                                 assets/rough/rough_seed<seed> naming

Usage:
    python src/feasibility/heightmap/create_rough_terrain.py
    python src/feasibility/heightmap/create_rough_terrain.py --seed 7 --rms 0.08
    python src/feasibility/heightmap/create_rough_terrain.py --cutoff-wavelength 1.5 --min-wavelength 0.4
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import yaml
from examples.helhest_junior.common import HelhestJuniorConfig

from feasibility.heightmap import HeightMapReader

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
ASSETS_DIR = REPO_ROOT / "assets" / "rough"

DEFAULT_EXTENT = 16.0  # m, square grid -- matches create_box_obstacles.py's centered series
DEFAULT_CELL = 0.05  # m, grid resolution -- matches every other create_*.py default
DEFAULT_CUTOFF_WAVELENGTH = 3.0  # m, longest wavelength let through -- see module docstring
DEFAULT_MIN_WAVELENGTH = 0.6  # m, shortest wavelength let through -- caps slope divergence
DEFAULT_BETA = 2.5  # power-spectrum slope S(k) ~ k^-beta
DEFAULT_RMS = 0.035  # m, target RMS elevation -- keeps default peak-to-peak under GROUND_CLEARANCE
DEFAULT_SEED = 0

# Chassis underside sits 0.10 m below the axle (CHASSIS_BOXES center z=0, z-size 0.20), so ground
# clearance = wheel radius (axle height above ground) minus that 0.10 m -- 0.25 m for the current
# config. Derived rather than hardcoded so it tracks HelhestJuniorConfig if the chassis geometry
# changes, same rationale as generate_dataset.py's ROBOT_X_MIN/ROBOT_X_MAX.
_CHASSIS_MIN_Z = min(pos[2] - size[2] / 2.0 for pos, size, _ in HelhestJuniorConfig.CHASSIS_BOXES.values())
GROUND_CLEARANCE = HelhestJuniorConfig.WHEEL_RADIUS + _CHASSIS_MIN_Z  # m


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extent", type=float, default=DEFAULT_EXTENT, help="full width/height of the square grid, in meters")
    parser.add_argument("--cell", type=float, default=DEFAULT_CELL, help="grid resolution in meters")
    parser.add_argument(
        "--cutoff-wavelength",
        type=float,
        default=DEFAULT_CUTOFF_WAVELENGTH,
        help="longest wavelength let through, in meters -- longer features are hills and are suppressed entirely",
    )
    parser.add_argument(
        "--min-wavelength",
        type=float,
        default=DEFAULT_MIN_WAVELENGTH,
        help="shortest wavelength let through, in meters -- caps slope statistics, see module docstring",
    )
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA, help="power-spectrum slope S(k) ~ k^-beta")
    parser.add_argument("--rms", type=float, default=DEFAULT_RMS, help="target RMS elevation, in meters")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed selecting the random realization")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="output path stem (no extension), overriding the default assets/rough/rough_seed<seed> naming",
    )
    args = parser.parse_args()
    if args.min_wavelength < 2.0 * args.cell:
        parser.error(
            f"--min-wavelength {args.min_wavelength} m is below the grid's Nyquist wavelength "
            f"{2.0 * args.cell} m (=2*cell) -- lower --cell or raise --min-wavelength"
        )
    if args.cutoff_wavelength <= args.min_wavelength:
        parser.error(
            f"--cutoff-wavelength {args.cutoff_wavelength} m must be greater than "
            f"--min-wavelength {args.min_wavelength} m -- the passband would be empty"
        )
    return args


def rough_path(seed: int) -> pathlib.Path:
    """assets/rough/rough_seed<NNNN> -- no extension, HeightMapReader appends .png/.yaml. Keyed
    on seed (the "which realization" knob), same as box_path/bump_path keying on height (the
    "which series member" knob) -- other params (rms, cutoff, beta, ...) just change the content
    of that seed's file rather than forking the filename, matching those scripts' convention
    that non-identity knobs (--incline-deg, --cell) silently affect the same-named output."""
    return ASSETS_DIR / f"rough_seed{seed:04d}"


def build_rough_terrain(
    extent: float = DEFAULT_EXTENT,
    cell: float = DEFAULT_CELL,
    cutoff_wavelength: float = DEFAULT_CUTOFF_WAVELENGTH,
    min_wavelength: float = DEFAULT_MIN_WAVELENGTH,
    beta: float = DEFAULT_BETA,
    rms: float = DEFAULT_RMS,
    seed: int = DEFAULT_SEED,
) -> HeightMapReader:
    """Band-limited random rough terrain on a square grid centered at the origin -- see module
    docstring for the synthesis method and why min_wavelength is a hard requirement, not a
    nicety. rfft2/irfft2 (real-input FFT) rather than fft2/ifft2: the output is guaranteed real
    with no manual Hermitian-symmetry bookkeeping, at half the work."""
    half = extent / 2.0
    n = int(round(extent / cell)) + 1

    rng = np.random.default_rng(seed)
    white = rng.standard_normal((n, n))
    spectrum = np.fft.rfft2(white)  # [n, n//2+1]

    kx = np.fft.rfftfreq(n, d=cell)  # cycles/m, length n//2+1
    ky = np.fft.fftfreq(n, d=cell)  # cycles/m, length n
    k = np.hypot(kx[None, :], ky[:, None])  # [n, n//2+1]

    k_cutoff, k_max = 1.0 / cutoff_wavelength, 1.0 / min_wavelength
    with np.errstate(divide="ignore"):
        amplitude = np.where((k >= k_cutoff) & (k <= k_max), k ** (-beta / 2.0), 0.0)
    amplitude[0, 0] = 0.0  # no DC term -- the rms rescale below re-centers/re-scales anyway

    H = np.fft.irfft2(spectrum * amplitude, s=(n, n))
    H -= H.mean()
    achieved_rms = float(H.std())
    if achieved_rms > 0.0:
        H *= rms / achieved_rms

    return HeightMapReader(H, origin=(-half, -half), cell=cell)


def _write_params(path: pathlib.Path, **params: float) -> None:
    """Merges generation params into the .yaml sidecar HeightMapReader.save() just wrote --
    extra keys are inert to HeightMapReader.load() (it only reads resolution/origin/min_z/
    max_z), so this makes the file self-describing without touching the shared reader class."""
    yaml_path = path.with_suffix(".yaml")
    meta = yaml.safe_load(yaml_path.read_text())
    meta.update(params)
    yaml_path.write_text(yaml.safe_dump(meta))


def main() -> None:
    args = parse_args()
    hmap = build_rough_terrain(
        args.extent, args.cell, args.cutoff_wavelength, args.min_wavelength, args.beta, args.rms, args.seed
    )
    path = pathlib.Path(args.out) if args.out else rough_path(args.seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    hmap.save(path)

    achieved_rms = float(hmap.H.std())
    peak_to_peak = float(hmap.H.max() - hmap.H.min())
    gy, gx = np.gradient(hmap.H, args.cell)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
    _write_params(
        path,
        extent=args.extent,
        cell=args.cell,
        cutoff_wavelength=args.cutoff_wavelength,
        min_wavelength=args.min_wavelength,
        beta=args.beta,
        rms_target=args.rms,
        rms_achieved=achieved_rms,
        seed=args.seed,
        peak_to_peak=peak_to_peak,
    )

    print(
        f"saved {path}.png / {path}.yaml  "
        f"({hmap.nx}x{hmap.ny} cells, cell={args.cell} m, seed={args.seed}, "
        f"wavelength band=[{args.min_wavelength}, {args.cutoff_wavelength}] m, beta={args.beta}, "
        f"rms target/achieved={args.rms:.3f}/{achieved_rms:.3f} m, peak-to-peak={peak_to_peak:.3f} m, "
        f"slope median/95th/max={np.median(slope_deg):.1f}/{np.percentile(slope_deg, 95):.1f}/{slope_deg.max():.1f} deg)"
    )
    if peak_to_peak > GROUND_CLEARANCE:
        print(
            f"WARNING: peak-to-peak {peak_to_peak:.3f} m exceeds the {GROUND_CLEARANCE:.2f} m "
            "chassis ground clearance (HelhestJuniorConfig) -- expect high-centering on the "
            "tallest features; lower --rms or raise --cutoff-wavelength"
        )


if __name__ == "__main__":
    main()
