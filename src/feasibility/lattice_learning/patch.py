"""The input patch: a body-frame relief grid sized for a 0.3-0.6 m lattice arc, not for a
2.4 s / up-to-3.6 m rollout.

Local re-derivation of `learning/terrain_patch.py` (see design.md section 11a -- `lattice_learning`
does not import from `learning/`, `grid_learning/` or `grid_learning_2/`). The four choices that
module got right all carry over unchanged: body-aligned (yaw drops out by construction), relief
relative to the wheel contacts, divided by the wheel radius (a dimensionless step height, no
fitted per-cell normaliser), explicitly not translation-invariant. What changes is the geometry --
finer, and much smaller, because a lattice arc is a fraction of that rollout's length (design.md
section 3):

    PatchSpec(x_min=-1.5, x_max=2.0, y_min=-1.5, y_max=1.5, cell=0.125, reference="wheels")
    -> 28 x 24 = 672 cells spanning 3.5 m x 3.0 m

The channel count is a module constant (`N_CHANNELS`), not a literal 1: design.md section 3c
defers a second "is this cell real data" channel to a later dataset revision, and pinning the
count here means that revision is a data change, not an architecture change.
"""
from __future__ import annotations

import argparse
import dataclasses

import numpy as np
from helhest.engine import RobotParams

from feasibility.heightmap import HeightMapReader

_ROBOT = RobotParams()

# Body-frame XY of the three wheel contacts (X forward, Y left), the support triangle whose mean
# terrain height is the default patch reference -- same role as terrain_patch.py's
# WHEEL_CONTACTS_LOCAL, re-derived from RobotParams rather than ostrich's HelhestJuniorConfig
# since this tree's object of study is helhest_stack's own robot model.
WHEEL_CONTACTS_LOCAL = np.array(
    [
        [0.0, _ROBOT.half_track],
        [0.0, -_ROBOT.half_track],
        [-_ROBOT.rear_offset, 0.0],
    ]
)  # [3, 2]

HEIGHT_SCALE = float(_ROBOT.wheel_radius)  # 0.35 m -- see module docstring

N_CHANNELS = 1  # relief only; see module docstring and design.md section 3c

# design.md section 3a: causal-support extent for step<=0.6 m, |kappa|<=2/m arcs.
DEFAULT_X_RANGE = (-1.5, 2.0)  # m, body frame: rear-wheel rim behind, arc + settle ahead
DEFAULT_Y_RANGE = (-1.5, 1.5)  # m, body frame: symmetric, the robot turns either way
DEFAULT_CELL = 0.125  # m -- see module docstring on why finer than learning/terrain_patch.py's
DEFAULT_REFERENCE = "wheels"
REFERENCES = ("wheels", "center", "none")


@dataclasses.dataclass(frozen=True)
class PatchSpec:
    """Geometry of the body-frame sampling grid: a `x_max-x_min` by `y_max-y_min` metre
    rectangle tiled by `cell`-metre square cells, sampled at CELL CENTERS. The extent must tile
    exactly -- see `learning/terrain_patch.py`'s `PatchSpec` docstring for why that's an error
    rather than something to round away.

    Frozen and round-trippable through `patch_spec_to_attrs`/`patch_spec_from_attrs` because a
    trained checkpoint has to record the exact geometry its inputs were built from -- a model fed
    a differently-shaped or differently-referenced patch at inference is silently wrong, not
    broken (`terrain_patch.py`'s own warning)."""

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
    def reach(self) -> float:
        """Farthest a patch corner sits from the body origin, at any yaw -- the radius
        `patch_overhangs` needs (design.md section 7b)."""
        return max(abs(self.x_min), abs(self.x_max), abs(self.y_min), abs(self.y_max))

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
    """`to_dict()`, flattened with a `patch_` prefix so it splices straight into a comparator
    HDF5's flat root attrs (`comparator.provenance.write_comparison`'s `root=` dict) instead of
    needing its own nested group. Inverse of `patch_spec_from_attrs`."""
    return {f"{PATCH_ATTR_PREFIX}{k}": v for k, v in spec.to_dict().items()}


def patch_spec_from_attrs(attrs: dict[str, object]) -> PatchSpec:
    """Inverse of `patch_spec_to_attrs`: rebuilds a `PatchSpec` from a comparator HDF5's root
    attrs (`h5py.File(...).attrs`, already materialized to a plain dict). Raises `KeyError` if
    `attrs` has no `patch_*` entries, i.e. the file wasn't written by `generate_dataset.py`."""
    keys = ("x_min", "x_max", "y_min", "y_max", "cell", "reference")
    kwargs: dict[str, float | str] = {
        k: (
            str(attrs[f"{PATCH_ATTR_PREFIX}{k}"])
            if k == "reference"
            else float(attrs[f"{PATCH_ATTR_PREFIX}{k}"])
        )
        for k in keys
    }
    return PatchSpec(**kwargs)  # type: ignore[arg-type]


def _body_to_world(
    x: np.ndarray, y: np.ndarray, c: np.ndarray, s: np.ndarray, local_x: np.ndarray, local_y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Body-frame point(s) (local_x, local_y) rotated by yaw (c=cos, s=sin) and translated by
    (x, y). Callers pre-broadcast x/y/c/s (extra trailing None axes) to local_x/local_y's rank."""
    return x + c * local_x - s * local_y, y + s * local_x + c * local_y


def _reference_heights(
    terrain: HeightMapReader, x: np.ndarray, y: np.ndarray, c: np.ndarray, s: np.ndarray, mode: str
) -> np.ndarray:
    """[n] height each row's patch is measured against. "wheels" (the default, and the only mode
    design.md's dataset uses) is the mean terrain height under the three wheel contacts; "center"
    and "none" are ablations carried over from `terrain_patch.py` for parity."""
    if mode == "none":
        return np.zeros_like(x)
    if mode == "center":
        return np.asarray(terrain.sample(x, y), dtype=np.float64)
    local_x, local_y = WHEEL_CONTACTS_LOCAL[:, 0], WHEEL_CONTACTS_LOCAL[:, 1]
    wx, wy = _body_to_world(x[:, None], y[:, None], c[:, None], s[:, None], local_x, local_y)  # [n, 3]
    return np.asarray(terrain.sample(wx, wy), dtype=np.float64).mean(axis=1)


def sample_patches(
    terrain: HeightMapReader, pose: np.ndarray, spec: PatchSpec | None = None
) -> np.ndarray:
    """[n, ny, nx] float32 body-frame relief patches for `pose` [n, 3] = (x, y, yaw) on one
    `terrain`, in units of `HEIGHT_SCALE` (the wheel radius) relative to `spec.reference`.

    Row index runs along body +Y (left), column index along body +X (forward) -- same row=y/
    col=x layout `HeightMapReader.H` uses. Vectorized over every row at once, matching
    `terrain_patch.py`'s reasoning: the dataset is many thousands of rows, and
    `HeightMapReader.sample` is already array-shaped."""
    spec = spec or PatchSpec()
    pose = np.asarray(pose, dtype=np.float64).reshape(-1, 3)
    x, y, yaw = pose[:, 0], pose[:, 1], pose[:, 2]
    c, s = np.cos(yaw), np.sin(yaw)

    body_x, body_y = np.meshgrid(spec.xs(), spec.ys())  # [ny, nx]
    wx, wy = _body_to_world(
        x[:, None, None], y[:, None, None], c[:, None, None], s[:, None, None], body_x, body_y
    )  # [n, ny, nx]

    heights = np.asarray(terrain.sample(wx, wy), dtype=np.float64)  # [n, ny, nx]
    reference = _reference_heights(terrain, x, y, c, s, spec.reference)
    return ((heights - reference[:, None, None]) / HEIGHT_SCALE).astype(np.float32)


def patch_overhangs(terrain: HeightMapReader, pose: np.ndarray, spec: PatchSpec | None = None) -> np.ndarray:
    """[n] bool, True where sampling `spec`'s patch at body `pose` would reach outside
    `terrain`'s mapped extent. `HeightMapReader.sample` CLAMPS out-of-grid queries rather than
    raising, so an overhanging patch would silently fabricate flat ground from the border value
    (design.md section 3c) instead of failing loudly -- this is the cheap, conservative stand-in
    for a `measured` mask: a circle of radius `spec.reach` around the body origin, independent of
    yaw, so it doesn't need the per-row rotation `sample_patches` does (design.md section 7b)."""
    spec = spec or PatchSpec()
    pose = np.asarray(pose, dtype=np.float64).reshape(-1, 3)
    x, y = pose[:, 0], pose[:, 1]
    reach = spec.reach
    x_lo, y_lo = terrain.x0, terrain.y0
    x_hi, y_hi = terrain.x0 + terrain.cell * terrain.nx, terrain.y0 + terrain.cell * terrain.ny
    return (x - reach < x_lo) | (x + reach > x_hi) | (y - reach < y_lo) | (y + reach > y_hi)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--pose", type=float, nargs=3, default=(-1.0, 0.0, 0.0), metavar=("X", "Y", "YAW"),
        help="body pose to sample at, yaw in radians (default: -1 0 0)",
    )
    parser.add_argument("--cell", type=float, default=DEFAULT_CELL, help="patch resolution (m)")
    parser.add_argument(
        "--reference", choices=REFERENCES, default=DEFAULT_REFERENCE, help="height reference"
    )
    args = parser.parse_args()

    spec = PatchSpec(cell=args.cell, reference=args.reference)
    print(
        f"PatchSpec: {spec.ny}x{spec.nx} = {spec.ny * spec.nx} cells, {spec.cell} m cells, "
        f"x=[{spec.x_min}, {spec.x_max}], y=[{spec.y_min}, {spec.y_max}], ref={spec.reference}"
    )
    assert spec.nx == 28 and spec.ny == 24, (spec.nx, spec.ny)  # design.md section 3a
    assert patch_spec_from_attrs(patch_spec_to_attrs(spec)) == spec

    flat = HeightMapReader.flat(xlim=(-8.0, 8.0), ylim=(-8.0, 8.0))
    flat_patch = sample_patches(flat, np.array([[0.5, -1.0, 0.7]]), spec)
    assert flat_patch.shape == (1, spec.ny, spec.nx), flat_patch.shape
    assert np.allclose(flat_patch, 0.0), "flat ground must give an all-zero patch"
    assert not patch_overhangs(flat, np.array([[0.5, -1.0, 0.7]]), spec)[0]
    assert patch_overhangs(flat, np.array([[7.9, 0.0, 0.0]]), spec)[0]

    # A box centered on the world origin is 4-fold symmetric, so a robot 3 m out along -X facing
    # +X and one 3 m out along -Y facing +Y see the SAME terrain. Body-aligning the patch is
    # exactly the claim that those two rows are one sample -- built in-line (not loaded from
    # assets/) so this self-check has no generated-file dependency.
    xs = np.arange(-8.0, 8.0, 0.05) + 0.025
    ys = np.arange(-8.0, 8.0, 0.05) + 0.025
    X, Y = np.meshgrid(xs, ys)
    H = np.where((np.abs(X) < 0.75) & (np.abs(Y) < 0.75), 0.7, 0.0)
    box = HeightMapReader(H, origin=(-8.0, -8.0), cell=0.05)
    poses = np.array([[-3.0, 0.0, 0.0], [0.0, -3.0, np.pi / 2.0]])
    rotated = sample_patches(box, poses, spec)
    assert np.allclose(rotated[0], rotated[1], atol=2e-3), (
        f"body-frame patch is not rotation invariant: max diff "
        f"{np.abs(rotated[0] - rotated[1]).max():.4f}"
    )

    patch = sample_patches(box, np.array([args.pose]), spec)[0]
    print(
        f"\npose {tuple(args.pose)}: dz/r in [{patch.min():.3f}, {patch.max():.3f}] "
        f"(peak {patch.max() * HEIGHT_SCALE:.3f} m)"
    )
    print("flat/rotation-invariance/overhang checks ok")
