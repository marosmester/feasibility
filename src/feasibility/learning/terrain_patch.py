"""Robot-centric terrain patch: the elevation grid around one spawn pose, resampled into the
BODY frame, so a learned model sees terrain GEOMETRY rather than an absolute (x, y, yaw) that
only means anything on the single heightmap it was fitted to.

custom_dataset.PoseErrorDataset's original features were (v, wz, spawn_x, spawn_y, spawn_yaw).
On a fixed centered-box terrain that IS a sufficient description -- spawn pose plus one known
obstacle determines the whole interaction -- but it is a COORDINATE code, not a geometry one:
the network has to internally reconstruct "where is the box, how tall, which wheel hits it"
from three numbers through a learned rotation, and its weights then encode that one heightmap.
Swap the terrain and the model is worthless. This module replaces those three columns with a
patch of the actual terrain, which makes the input terrain-general and drops `yaw` outright.

Four choices, each of which matters more than the patch's exact size:

* BODY-ALIGNED, not world-aligned. The patch is rotated by -yaw into the frame where +X is
  forward and +Y is left, so a scenario and its rotated copy produce the SAME patch. That is a
  real symmetry of the problem (the terrain has no privileged direction), and it removes `yaw`
  from the feature vector entirely rather than asking the network to learn invariance to it.
* RELATIVE heights, never absolute z. Every sample is offset by the terrain height under the
  robot (see `reference`), so what the network reads is the local RISE. Absolute elevation is
  meaningless -- a box 0.7 m tall is the same obstacle whether the ground under it sits at 0 m
  or 5 m -- and subtracting the reference is what lets one model span the whole
  create_box_obstacles.py height series instead of memorizing one of them.
* Divided by WHEEL_RADIUS. `dz / r_wheel` is the dimensionless quantity that actually decides
  whether a step is climbable, and it puts the patch on a sane numeric scale WITHOUT a fitted
  per-cell normalizer. That last part is not cosmetic: on this terrain most patch cells are flat
  ground in every single row, so a per-column standardization (custom_dataset.Normalizer) would
  divide their float-noise-level std into O(1) garbage. The patch is ONE physical field with one
  natural scale, not `nx*ny` independent features -- so custom_dataset deliberately leaves the
  patch block out of its x_normalizer and relies on this fixed scaling instead.
* FORWARD-BIASED, not a symmetric disk. generate_dataset.V_RANGE is (0.0, 1.5) -- the robot
  never reverses -- so DEFAULT_X_RANGE reaches 4.5 m ahead but only 1.5 m behind (just past the
  rear wheel at x=-0.75 minus its 0.35 m radius). At the 2.4 s default duration a v=1.5 trial
  covers 3.6 m, so the forward extent is sized to keep the swept path inside the patch.

DEFAULT_CELL is 0.25 m, five times coarser than the 0.05 m heightmaps in assets/. That is
deliberate: the finest real feature in the box series is the ramp bevel, ramp_width =
height/tan(75 deg) = 0.268*height <= 0.21 m at the tallest box, and everything the dynamics
actually cares about (edge position, step height) survives at 0.25 m. Sampling at the source
resolution would mean 120x120 = 14400 inputs into an MLP fitted on a few thousand rows; 24x24 =
576 is already the aggressive end of what model.PoseErrorMLP can carry. Read the resolution as
a knob on that tradeoff, not as a fidelity setting.

Sampling uses HeightMapReader.sample(), which CLAMPS outside the grid rather than raising, so a
patch that overhangs the heightmap edge silently repeats the border value. On the default
16m x 16m centered grid with generate_dataset.SPAWN_LIMIT=2.0 the farthest corner a patch can
reach is ~2 + sqrt(4.5^2 + 3^2) = 7.4 m from the origin, inside the 8 m half-extent, so nothing
clamps -- widen the terrain (create_box_obstacles.py --extent) before widening the patch.

CLI parameters:
    --box-height FLOAT   which centered_box_paths() height to sample (default: 0.70)
    --pose X Y YAW       body pose to sample the patch at, yaw in radians (default: -3 0 0)
    --cell FLOAT         patch resolution in meters (default: 0.25)
    --reference STR      height reference: wheels|center|none (default: wheels)

Usage:
    python src/feasibility/learning/terrain_patch.py                       # self-check + ASCII
    python src/feasibility/learning/terrain_patch.py --pose 0 -3 1.5708    # same box, rotated
    python src/feasibility/learning/terrain_patch.py --cell 0.1 --box-height 0.4
"""
from __future__ import annotations

import argparse
import dataclasses

import numpy as np
from examples.helhest_junior.common import HelhestJuniorConfig

from feasibility.heightmap import HeightMapReader
from feasibility.heightmap.create_box_obstacles import centered_box_path

# Body-frame XY of the three wheel contacts (X forward, Y left, origin = front-wheel axle
# midpoint) -- the support triangle whose mean terrain height is the default patch reference.
# Read from HelhestJuniorConfig rather than re-hardcoded, same as generate_dataset.py's footprint.
WHEEL_CONTACTS_LOCAL = np.array(
    [
        [float(p[0]), float(p[1])]
        for p in (
            HelhestJuniorConfig.LEFT_WHEEL_POS,
            HelhestJuniorConfig.RIGHT_WHEEL_POS,
            HelhestJuniorConfig.REAR_WHEEL_POS,
        )
    ]
)  # [3, 2]

HEIGHT_SCALE = float(HelhestJuniorConfig.WHEEL_RADIUS)  # 0.35 m -- see module docstring

DEFAULT_X_RANGE = (-1.5, 4.5)  # m, body frame: rear-wheel rim behind, ~1 rollout of travel ahead
DEFAULT_Y_RANGE = (-3.0, 3.0)  # m, body frame: symmetric, the robot turns either way
DEFAULT_CELL = 0.25  # m -- see module docstring on why not the terrain's own 0.05
DEFAULT_REFERENCE = "wheels"
REFERENCES = ("wheels", "center", "none")


@dataclasses.dataclass(frozen=True)
class PatchSpec:
    """Geometry of the body-frame sampling grid: a `x_max-x_min` by `y_max-y_min` metre
    rectangle tiled by `cell`-metre square cells, sampled at CELL CENTERS.

    The extent must tile exactly -- nx = (x_max-x_min)/cell with no `+1` and no remainder, unlike
    HeightMapReader/create_box_obstacles.py's `int(round(extent/cell)) + 1`, whose extra row/col
    overhangs the requested limits by one cell. Here the extent IS the contract (it is what the
    docstring's "reaches 4.5 m ahead" means), so a cell that doesn't divide it is an error rather
    than something to round away.

    Frozen and round-trippable through to_dict()/from_dict() because a trained checkpoint has to
    record the exact geometry its inputs were built from -- a model fed a differently-shaped or
    differently-referenced patch at inference is silently wrong, not broken."""

    x_min: float = DEFAULT_X_RANGE[0]
    x_max: float = DEFAULT_X_RANGE[1]
    y_min: float = DEFAULT_Y_RANGE[0]
    y_max: float = DEFAULT_Y_RANGE[1]
    cell: float = DEFAULT_CELL
    reference: str = DEFAULT_REFERENCE

    def __post_init__(self) -> None:
        if self.cell <= 0.0:
            raise ValueError(f"cell must be > 0, got {self.cell}")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError(
                f"patch extent must be positive, got x=({self.x_min}, {self.x_max}), "
                f"y=({self.y_min}, {self.y_max})"
            )
        if self.reference not in REFERENCES:
            raise ValueError(f"reference must be one of {REFERENCES}, got {self.reference!r}")
        for name, span in (("x", self.x_max - self.x_min), ("y", self.y_max - self.y_min)):
            if abs(span / self.cell - round(span / self.cell)) > 1e-6:
                raise ValueError(
                    f"{name} extent {span} m is not an integer number of {self.cell} m cells"
                )

    @property
    def nx(self) -> int:
        return int(round((self.x_max - self.x_min) / self.cell))

    @property
    def ny(self) -> int:
        return int(round((self.y_max - self.y_min) / self.cell))

    @property
    def size(self) -> int:
        """Flattened length -- the number of input columns this patch contributes."""
        return self.nx * self.ny

    def xs(self) -> np.ndarray:
        """[nx] body-frame X of each column's cell center."""
        return self.x_min + (np.arange(self.nx) + 0.5) * self.cell

    def ys(self) -> np.ndarray:
        """[ny] body-frame Y of each row's cell center."""
        return self.y_min + (np.arange(self.ny) + 0.5) * self.cell

    def to_dict(self) -> dict[str, float | str]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, float | str]) -> "PatchSpec":
        return cls(**d)  # type: ignore[arg-type]


PATCH_ATTR_PREFIX = "patch_"  # see patch_spec_to_attrs()/patch_spec_from_attrs()


def patch_spec_to_attrs(spec: PatchSpec) -> dict[str, float | str]:
    """to_dict(), flattened with a `patch_` prefix so it splices straight into a comparator
    HDF5's flat root attrs (comparator.provenance.write_comparison's `root=` dict, alongside
    scalars like mu/k_turn/spawn_mode) instead of needing its own nested group -- written by
    generate_dataset_body_centered_patch.py so the file records the exact geometry its `patch`
    dataset was sampled with. Inverse of patch_spec_from_attrs()."""
    return {f"{PATCH_ATTR_PREFIX}{k}": v for k, v in spec.to_dict().items()}


def patch_spec_from_attrs(attrs: dict[str, object]) -> PatchSpec:
    """Inverse of patch_spec_to_attrs(): rebuilds a PatchSpec from a comparator HDF5's root attrs
    (h5py.File(...).attrs, already materialized to a plain dict). Raises KeyError if `attrs` has
    no patch_* entries, i.e. the file wasn't written by generate_dataset_body_centered_patch.py."""
    keys = ("x_min", "x_max", "y_min", "y_max", "cell", "reference")
    kwargs: dict[str, float | str] = {
        k: (str(attrs[f"{PATCH_ATTR_PREFIX}{k}"]) if k == "reference" else float(attrs[f"{PATCH_ATTR_PREFIX}{k}"]))
        for k in keys
    }
    return PatchSpec(**kwargs)  # type: ignore[arg-type]


def patch_feature_names(spec: PatchSpec) -> tuple[str, ...]:
    """One name per flattened patch cell, row-major (row = body +Y, col = body +X), matching
    sample_patches()'s [ny, nx] layout and its reshape order. Exists so
    `len(ds.FEATURE_NAMES) == in_dim` stays true with a patch attached -- train.py sizes the
    model and writes the checkpoint off that identity."""
    return tuple(f"h_r{i:02d}c{j:02d}" for i in range(spec.ny) for j in range(spec.nx))


def _body_to_world(
    x: np.ndarray, y: np.ndarray, c: np.ndarray, s: np.ndarray, local_x: np.ndarray, local_y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Body-frame point(s) (local_x, local_y) rotated by yaw (c=cos, s=sin) and translated by
    (x, y) -- the standard planar rotate-then-translate, shared by sample_patches' grid and
    _reference_heights' wheel contacts so the two don't carry independent copies of the same
    formula. Callers pre-broadcast x/y/c/s (extra trailing None axes) to local_x/local_y's rank."""
    return x + c * local_x - s * local_y, y + s * local_x + c * local_y


def _reference_heights(
    terrain: HeightMapReader, x: np.ndarray, y: np.ndarray, c: np.ndarray, s: np.ndarray, mode: str
) -> np.ndarray:
    """[n] height each row's patch is measured against. "wheels" (the default) is the mean
    terrain height under the three wheel contacts -- the cheapest stand-in for the plane the
    robot actually rests on, and exactly 0 on flat ground; "center" is the height at the body
    origin, which is cheaper but sits between the wheels and can fall INSIDE a box footprint the
    wheels straddle; "none" leaves absolute elevation in, only useful as an ablation."""
    if mode == "none":
        return np.zeros_like(x)
    if mode == "center":
        return np.asarray(terrain.sample(x, y), dtype=np.float64)
    local_x, local_y = WHEEL_CONTACTS_LOCAL[:, 0], WHEEL_CONTACTS_LOCAL[:, 1]
    wx, wy = _body_to_world(x[:, None], y[:, None], c[:, None], s[:, None], local_x, local_y)  # [n, 3]
    return np.asarray(terrain.sample(wx, wy), dtype=np.float64).mean(axis=1)


def sample_patches(
    terrain: HeightMapReader, spawn_pose: np.ndarray, spec: PatchSpec | None = None
) -> np.ndarray:
    """[n, ny, nx] float32 body-frame terrain patches for `spawn_pose` [n, 3] = (x, y, yaw) on
    one `terrain`, in units of WHEEL_RADIUS relative to `spec.reference` (see module docstring).

    Row index runs along body +Y (left), column index along body +X (forward) -- the same
    row=y/col=x layout HeightMapReader.H uses, so a patch prints the right way up under
    np.flipud() and indexes like the source grid.

    Vectorized over every row at once rather than looped: the whole point of the dataset is n in
    the thousands, and HeightMapReader.sample() is already array-shaped. Peak intermediate cost
    is a few n*ny*nx float64 arrays (46 MB at n=10000 on the default 24x24 patch), which is why
    custom_dataset calls this once per UNIQUE terrain rather than once per row."""
    spec = spec or PatchSpec()
    pose = np.asarray(spawn_pose, dtype=np.float64).reshape(-1, 3)
    x, y, yaw = pose[:, 0], pose[:, 1], pose[:, 2]
    c, s = np.cos(yaw), np.sin(yaw)

    body_x, body_y = np.meshgrid(spec.xs(), spec.ys())  # [ny, nx]
    # Body -> world per row: broadcasting the [n] pose against the [ny, nx] grid gives
    # [n, ny, nx] in one shot.
    wx, wy = _body_to_world(
        x[:, None, None], y[:, None, None], c[:, None, None], s[:, None, None], body_x, body_y
    )

    heights = np.asarray(terrain.sample(wx, wy), dtype=np.float64)  # [n, ny, nx]
    reference = _reference_heights(terrain, x, y, c, s, spec.reference)
    return ((heights - reference[:, None, None]) / HEIGHT_SCALE).astype(np.float32)


def _ascii(patch: np.ndarray) -> str:
    """One patch [ny, nx] as text, forward (+X) to the right and left (+Y) UP -- np.flipud,
    since row 0 is the most-negative Y. Levels are dz/r_wheel, so '#' is roughly a wheel-radius
    step. Debug aid for the CLI below, not used by the dataset."""
    levels = " .:-=+*#"
    rows = []
    for row in np.flipud(patch):
        idx = np.clip((np.abs(row) * 4.0).astype(int), 0, len(levels) - 1)
        rows.append("".join(levels[k] for k in idx))
    return "\n".join(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--box-height", type=float, default=0.70, help="centered box height (m)")
    parser.add_argument(
        "--pose", type=float, nargs=3, default=(-3.0, 0.0, 0.0), metavar=("X", "Y", "YAW"),
        help="body pose to sample at, yaw in radians (default: -3 0 0)",
    )
    parser.add_argument("--cell", type=float, default=DEFAULT_CELL, help="patch resolution (m)")
    parser.add_argument(
        "--reference", choices=REFERENCES, default=DEFAULT_REFERENCE, help="height reference"
    )
    args = parser.parse_args()

    spec = PatchSpec(cell=args.cell, reference=args.reference)
    print(f"PatchSpec: {spec.ny}x{spec.nx} = {spec.size} inputs, {spec.cell} m cells, "
          f"x=[{spec.x_min}, {spec.x_max}], y=[{spec.y_min}, {spec.y_max}], ref={spec.reference}")
    assert len(patch_feature_names(spec)) == spec.size

    flat = HeightMapReader.flat(xlim=(-8.0, 8.0), ylim=(-8.0, 8.0))
    flat_patch = sample_patches(flat, np.array([[0.5, -1.0, 0.7]]), spec)
    assert flat_patch.shape == (1, spec.ny, spec.nx), flat_patch.shape
    assert np.allclose(flat_patch, 0.0), "flat ground must give an all-zero patch"

    terrain = HeightMapReader.load(centered_box_path(args.box_height))
    # The centered box is 4-fold symmetric about the origin, so a robot 3 m out along -X facing
    # +X and one 3 m out along -Y facing +Y see the SAME thing. Body-aligning the patch is
    # exactly the claim that those two rows are one sample, so they must agree numerically.
    poses = np.array([[-3.0, 0.0, 0.0], [0.0, -3.0, np.pi / 2.0]])
    rotated = sample_patches(terrain, poses, spec)
    assert np.allclose(rotated[0], rotated[1], atol=2e-3), (
        f"body-frame patch is not rotation invariant: max diff "
        f"{np.abs(rotated[0] - rotated[1]).max():.4f}"
    )

    patch = sample_patches(terrain, np.array([args.pose]), spec)[0]
    print(
        f"\nbox h={args.box_height} m at pose {tuple(args.pose)}: "
        f"dz/r in [{patch.min():.3f}, {patch.max():.3f}]  "
        f"(peak {patch.max() * HEIGHT_SCALE:.3f} m)\n"
    )
    print(_ascii(patch))
    print("\nflat/rotation-invariance checks ok")
