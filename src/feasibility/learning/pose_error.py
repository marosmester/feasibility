"""Final-pose SE(3) error between ostrich and hstack trajectories saved by
comparator/common.py's run_comparison (see comparator/provenance.py for the HDF5 schema).

CLI parameters:
    --file PATH   comparator output .h5 (required)
    --id INT      variant index (default: 0)

Usage:
    python src/feasibility/learning/pose_error.py --file outputs/compare_box_obstacles.h5 --id 3

For variant `id`, loads `ostrich/pose[-1, id]` and `hstack/pose[-1, id]` -- each (x, y, z, qx,
qy, qz, qw) -- builds their SE(3) matrices, and reports the two scalars a single quaternion/
position diff can't give you together: pos_error (m) and rot_error (rad), via
T_err = T1^{-1} @ T2.
"""
from __future__ import annotations

import argparse
import pathlib

import h5py
import numpy as np


def pose_to_se3(pose: np.ndarray) -> np.ndarray:
    """(x, y, z, qx, qy, qz, qw) -> 4x4 SE(3) matrix."""
    x, y, z, qx, qy, qz, qw = pose
    R = np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ]
    )
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = (x, y, z)
    return T


def se3_error(T1: np.ndarray, T2: np.ndarray) -> tuple[float, float]:
    """T_err = T1^-1 @ T2 -> (pos_error, rot_error): pos_error = ||translation part of T_err||
    (m); rot_error = arccos((tr(R_err) - 1) / 2) (rad), clamped for float round-off."""
    T_err = np.linalg.inv(T1) @ T2
    pos_error = float(np.linalg.norm(T_err[:3, 3]))
    trace = np.clip((np.trace(T_err[:3, :3]) - 1) / 2, -1.0, 1.0)
    rot_error = float(np.arccos(trace))
    return pos_error, rot_error


def final_pose_error(path: pathlib.Path, variant_id: int) -> tuple[float, float]:
    """Loads variant `variant_id`'s last ostrich/hstack poses from `path` and returns their
    (pos_error, rot_error) via se3_error."""
    with h5py.File(path, "r") as f:
        ostrich_pose = f["ostrich/pose"][-1, variant_id]
        hstack_pose = f["hstack/pose"][-1, variant_id]
    return se3_error(pose_to_se3(ostrich_pose), pose_to_se3(hstack_pose))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=pathlib.Path, required=True, help="comparator output .h5")
    parser.add_argument("--id", type=int, default=0, help="variant index (default: 0)")
    args = parser.parse_args()

    pos_error, rot_error = final_pose_error(args.file, args.id)
    print(f"variant {args.id}: pos_error={pos_error:.4f} m, rot_error={rot_error:.4f} rad ({np.degrees(rot_error):.2f} deg)")
