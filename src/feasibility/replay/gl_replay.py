"""GL replay of a single variant from comparator/batch_compare.py's output npz: loads that
variant's terrain heightfield and plays back the ostrich and/or hstack trajectory over it in
Newton's interactive GL viewer, in real time, rendering the real Helhest Junior mesh
(create_helhest_junior_model) -- chassis + 3 wheels -- rather than a box stand-in.

Pose-only playback, not physics: each frame we write `joint_q` directly -- the chassis's
free-joint 7 slots (px,py,pz,qx,qy,qz,qw) from the recorded chassis pose, and each wheel's
revolute-joint 1 slot from a wheel angle WE integrate ourselves (the npz only logs wheel
angular *velocity* -- ostrich_wheel_qd/hstack_wheel_qd -- not angle) -- then call
`newton.eval_fk` to propagate `joint_q` -> `state.body_q` for every body, so the wheels render
attached to the chassis and spinning. No solver step, no control targets; `model.collide()` is
only there to feed the viewer's contact-point overlay.

Usage:
    python -m feasibility.replay.gl_replay --id 3                 # ostrich, variant 3
    python -m feasibility.replay.gl_replay --id 3 --which hstack
    python -m feasibility.replay.gl_replay --id 3 --which both --speed 0.25 --loop
"""

from __future__ import annotations

import argparse
import pathlib
import time

import newton
import numpy as np
import warp as wp
from examples.helhest_junior.common import create_helhest_junior_model
from ostrich.core.model_builder import OstrichModelBuilder

from feasibility.comparator.provenance import terrain_from_npz
from feasibility.heightmap import HeightMapReader
from feasibility.heightmap.create_speed_bumps import BUMP_X0

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_NPZ = REPO_ROOT / "outputs" / "batch_compare.npz"

# create_helhest_junior_model always adds exactly 4 joints per robot, in this fixed order:
# base_joint (free: 7 joint_q slots / 6 joint_qd slots), then left/right/rear wheel_j (revolute:
# 1 slot each) -- see ostrich/examples/helhest_junior/common.py:172-299. Building N robots into
# one shared ModelBuilder therefore puts robot k's joints at indices [4k, 4k+4).
JOINTS_PER_ROBOT = 4
WHEEL_NAMES = ("left", "right", "rear")  # order matches create_helhest_junior_model's wheel_j joints

# Only used to tell the two meshes apart in --which both -- a single robot keeps its default
# mesh color, nothing to distinguish.
ROBOT_COLOR = {"ostrich": (0.9, 0.55, 0.1), "hstack": (0.55, 0.2, 0.85)}

# Default camera pose: parked beside the speed bump, a few meters off to the side (-Y) and
# slightly elevated, facing +Y (yaw=90 in this Z-up viewer's convention -- see
# ostrich/third_party/newton/newton/_src/viewer/camera.py's get_front()) -- gives a side-profile
# view of the whole run as the robot(s) cross the bump. Not a CLI flag on purpose -- edit these
# constants directly to change the default view.
CAMERA_POS = wp.vec3(BUMP_X0, -4.0-2, 1.5)
CAMERA_PITCH = -5.0
CAMERA_YAW = 90.0


def build_model(terrain: HeightMapReader, which: tuple[str, ...]) -> tuple[newton.Model, dict[str, dict[str, int]]]:
    # create_helhest_junior_model's wheel joints use the "joint_dof_mode" custom attribute
    # (see common.py's mode=JointMode.TARGET_VELOCITY/POSITION), which only a plain
    # newton.ModelBuilder doesn't declare -- it's registered by OstrichModelBuilder.
    builder = OstrichModelBuilder()
    ground_cfg = newton.ModelBuilder.ShapeConfig(mu=0.8)
    builder.add_shape_mesh(body=-1, mesh=terrain.to_ostrich_mesh(), cfg=ground_cfg)

    robots: dict[str, dict[str, int]] = {}
    for i, name in enumerate(which):
        shape_start = builder.shape_count
        create_helhest_junior_model(builder, xform=wp.transform_identity())
        j_base = i * JOINTS_PER_ROBOT
        robots[name] = {
            "base": j_base, "left": j_base + 1, "right": j_base + 2, "rear": j_base + 3,
            "shapes": (shape_start, builder.shape_count),
        }

    model = builder.finalize()
    if len(which) > 1:
        colors = model.shape_color.numpy()
        for name in which:
            start, end = robots[name]["shapes"]
            colors[start:end] = ROBOT_COLOR[name]
        model.shape_color.assign(colors)
    return model, robots


def integrate_wheel_angle(t: np.ndarray, wheel_qd: np.ndarray) -> np.ndarray:
    """wheel_qd is [T, 3] rad/s [left, right, rear]; the npz logs wheel VELOCITY, not angle, so
    trapezoidal integration over the recorded time axis is the only way to get a spin angle for
    joint_q -- exact for the piecewise-linear qd samples we have, and the integration constant
    (theta starts at 0) is invisible on the wheel's rotationally-symmetric mesh."""
    dt = np.diff(t)
    dtheta = 0.5 * (wheel_qd[:-1] + wheel_qd[1:]) * dt[:, None]
    return np.concatenate([np.zeros((1, 3)), np.cumsum(dtheta, axis=0)], axis=0)


def interp_series(t: np.ndarray, values: np.ndarray, sim_t: float) -> np.ndarray:
    """Linear interpolation of values [T, K] sampled at times t, at query time sim_t (clamped
    to [t[0], t[-1]])."""
    sim_t = min(max(sim_t, t[0]), t[-1])
    k = int(np.searchsorted(t, sim_t, side="right") - 1)
    k = min(max(k, 0), len(t) - 2)
    a = (sim_t - t[k]) / max(t[k + 1] - t[k], 1e-9)
    return values[k] * (1 - a) + values[k + 1] * a


def interp_pose(t: np.ndarray, pose: np.ndarray, sim_t: float) -> np.ndarray:
    """pose is [T, 7] = (x,y,z,qx,qy,qz,qw). Same caveat as interp_series -- not a slerp, but at
    the npz's native dt (ostrich/hstack physics rate, always finer than a sensible playback fps)
    the per-frame rotation step is small enough that the un-normalized lerp is visually
    indistinguishable."""
    p = interp_series(t, pose, sim_t).copy()
    p[3:] /= max(np.linalg.norm(p[3:]), 1e-9)  # renormalize the lerped quaternion
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", type=pathlib.Path, default=DEFAULT_NPZ, help=f"batch_compare.npz path (default {DEFAULT_NPZ})")
    ap.add_argument("--id", type=int, default=0, help="variant index to replay (default 0)")
    ap.add_argument("--which", choices=("ostrich", "hstack", "both"), default="both")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier (default 1.0 = real time)")
    ap.add_argument("--loop", action="store_true", help="loop playback instead of freezing on the last frame")
    args = ap.parse_args()

    d = np.load(args.npz)
    n = int(d["n"])
    if not (0 <= args.id < n):
        raise SystemExit(f"--id must be in [0, {n}), got {args.id}")

    which = ("ostrich", "hstack") if args.which == "both" else (args.which,)
    label = str(d["variant_label"][args.id])
    terrain = terrain_from_npz(d, args.id)

    tracks = {}
    for name in which:
        t = d[f"{name}_t"]
        pose = d[f"{name}_pose"][:, args.id, :]
        wheel_theta = integrate_wheel_angle(t, d[f"{name}_wheel_qd"][:, args.id, :])
        tracks[name] = (t, pose, wheel_theta)
    t_end = max(float(t[-1]) for t, _, _ in tracks.values())

    wp.init()
    model, robots = build_model(terrain, which)
    viewer = newton.viewer.ViewerGL()
    viewer.set_model(model)
    viewer.set_camera(pos=CAMERA_POS, pitch=CAMERA_PITCH, yaw=CAMERA_YAW)
    state = model.state()

    q_start = model.joint_q_start.numpy()
    joint_q = model.joint_q.numpy().copy()
    joint_q_wp = wp.array(joint_q, dtype=wp.float32, device=model.device)
    joint_qd_wp = wp.zeros_like(model.joint_qd)  # eval_fk only needs joint_q for body_q -- see module docstring

    print(f"replaying variant {args.id}/{n - 1}: {label}  [{', '.join(which)}]  duration {t_end:.2f}s @ speed={args.speed}")

    t0 = time.perf_counter()
    while viewer.is_running():
        real_t = (time.perf_counter() - t0) * args.speed
        sim_t = (real_t % t_end) if (args.loop and t_end > 0) else min(real_t, t_end)

        for name in which:
            t, pose, wheel_theta = tracks[name]
            joints = robots[name]
            base_q = q_start[joints["base"]]
            joint_q[base_q:base_q + 7] = interp_pose(t, pose, sim_t)
            theta = interp_series(t, wheel_theta, sim_t)
            for wheel_name, angle in zip(WHEEL_NAMES, theta):
                joint_q[q_start[joints[wheel_name]]] = angle
        joint_q_wp.assign(joint_q)

        newton.eval_fk(model, joint_q_wp, joint_qd_wp, state)
        contacts = model.collide(state)
        viewer.begin_frame(sim_t)
        viewer.log_state(state)
        viewer.log_contacts(contacts, state)
        viewer.end_frame()
        wp.synchronize()


if __name__ == "__main__":
    main()
