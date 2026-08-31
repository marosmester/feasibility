"""GL replay of a single variant from a comparator/learning output HDF5 (compare_*.py or
learning/generate_dataset.py's dataset_*.h5) -- same real-time Newton ViewerGL playback as
gl_replay.py (see its docstring for the pose-only/no-physics mechanics) -- but also loads a
trained learning.model.PoseErrorMLP checkpoint, runs inference on this variant's commanded twist
and spawn pose, and once playback finishes prints both the REAL final-pose (e_pos, e_rot) --
computed directly from the file's ostrich/pose[-1] vs hstack/pose[-1], the exact
learning.pose_error.se3_error math -- and the network's PREDICTED (e_pos, e_rot) to the terminal,
so the two can be compared by eye.

The network only ever sees (v_drive, wz_drive, spawn_x, spawn_y, spawn_yaw) -- never terrain --
so --file need not match --nn-checkpoint's training terrain for inference to run, but the
"real" error will only be in-distribution for the checkpoint if it does.

CLI parameters:
    --file PATH             comparator/generate_dataset output .h5 (required)
    --id INT                variant index to replay (default: 0)
    --nn-checkpoint PATH    trained PoseErrorMLP checkpoint (required), e.g.
                            outputs/checkpoints/dataset_box_h070cm_n10000.pt
    --which {ostrich,hstack,both}   which trajectory/trajectories to render (default: both)
    --speed FLOAT           playback speed multiplier (default: 1.0 = real time)
    --loop                  loop playback instead of freezing on the last frame
    --device STR            torch device (default: cuda if available else cpu)

Usage:
    python src/feasibility/replay/test_nn.py --file outputs/dataset_box_h070cm_n10000.h5 \
        --nn-checkpoint outputs/checkpoints/dataset_box_h070cm_n10000.pt --id 0
    python src/feasibility/replay/test_nn.py --file outputs/dataset_box_h070cm_n10000.h5 \
        --id 3 --nn-checkpoint outputs/checkpoints/dataset_box_h070cm_n10000.pt --which hstack --loop
"""

from __future__ import annotations

import argparse
import pathlib
import time

import h5py
import newton
import numpy as np
import torch
import warp as wp

from feasibility.comparator.provenance import terrain_from_h5
from feasibility.learning.custom_dataset import FEATURE_NAMES_SINCOS
from feasibility.learning.pose_error import pose_to_se3
from feasibility.learning.pose_error import se3_error
from feasibility.learning.train import load_checkpoint
from feasibility.replay.gl_replay import CAMERA_PITCH
from feasibility.replay.gl_replay import CAMERA_YAW
from feasibility.replay.gl_replay import CAMERA_Y_OFFSET
from feasibility.replay.gl_replay import CAMERA_Z
from feasibility.replay.gl_replay import WHEEL_NAMES
from feasibility.replay.gl_replay import build_model
from feasibility.replay.gl_replay import integrate_wheel_angle
from feasibility.replay.gl_replay import interp_pose
from feasibility.replay.gl_replay import interp_series


def build_feature_row(ckpt: dict[str, object], v_drive: float, wz_drive: float, x: float, y: float, yaw: float) -> torch.Tensor:
    """Assembles the [1, in_dim] raw feature row a checkpoint expects, keyed off its own
    ckpt["feature_names"] rather than assuming raw yaw -- a checkpoint trained with
    --yaw-encoding sincos (custom_dataset.FEATURE_NAMES_SINCOS) needs (cos_yaw, sin_yaw) instead
    of yaw, same as PoseErrorDataset.__init__ builds it."""
    if tuple(ckpt["feature_names"]) == FEATURE_NAMES_SINCOS:
        row = [v_drive, wz_drive, x, y, np.cos(yaw), np.sin(yaw)]
    else:
        row = [v_drive, wz_drive, x, y, yaw]
    return torch.tensor([row], dtype=torch.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=pathlib.Path, required=True, help="comparator/generate_dataset output .h5")
    ap.add_argument("--id", type=int, default=0, help="variant index to replay (default 0)")
    ap.add_argument("--nn-checkpoint", type=pathlib.Path, required=True, help="trained PoseErrorMLP checkpoint")
    ap.add_argument("--which", choices=("ostrich", "hstack", "both"), default="both")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier (default 1.0 = real time)")
    ap.add_argument("--loop", action="store_true", help="loop playback instead of freezing on the last frame")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="torch device (default: cuda if available else cpu)")
    args = ap.parse_args()

    which = ("ostrich", "hstack") if args.which == "both" else (args.which,)

    # Materialize everything to numpy inside the `with` -- h5py datasets are invalid once the
    # file closes, and the render loop below runs long after this returns.
    with h5py.File(args.file, "r") as f:
        n = int(f.attrs["n"])
        if not (0 <= args.id < n):
            raise SystemExit(f"--id must be in [0, {n}), got {args.id}")

        label = f["variant_label"].asstr()[args.id]
        terrain = terrain_from_h5(f, args.id)
        obstacle_x = float(f.attrs["obstacle_x"])

        tracks = {}
        for name in which:
            t = f[name]["t"][:]
            pose = f[name]["pose"][:, args.id, :]
            wheel_theta = integrate_wheel_angle(t, f[name]["wheel_qd"][:, args.id, :])
            tracks[name] = (t, pose, wheel_theta)

        v_drive = float(f["v_drive"][args.id])
        wz_drive = float(f["wz_drive"][args.id])
        spawn_x, spawn_y, spawn_yaw = f["spawn_pose"][args.id]
        real_e_pos, real_e_rot = se3_error(
            pose_to_se3(f["ostrich"]["pose"][-1, args.id]), pose_to_se3(f["hstack"]["pose"][-1, args.id])
        )

    device = torch.device(args.device)
    model, x_normalizer, ckpt = load_checkpoint(args.nn_checkpoint, device)
    model.eval()
    x_raw = build_feature_row(ckpt, v_drive, wz_drive, spawn_x, spawn_y, spawn_yaw).to(device)
    pred_e_pos, pred_e_rot = model.predict(x_normalizer(x_raw))[0].tolist()

    camera_pos = wp.vec3(obstacle_x, CAMERA_Y_OFFSET, CAMERA_Z)
    t_end = max(float(t[-1]) for t, _, _ in tracks.values())

    wp.init()
    model_, robots = build_model(terrain, which)
    viewer = newton.viewer.ViewerGL()
    viewer.set_model(model_)
    viewer.set_camera(pos=camera_pos, pitch=CAMERA_PITCH, yaw=CAMERA_YAW)
    state = model_.state()

    q_start = model_.joint_q_start.numpy()
    joint_q = model_.joint_q.numpy().copy()
    joint_q_wp = wp.array(joint_q, dtype=wp.float32, device=model_.device)
    joint_qd_wp = wp.zeros_like(model_.joint_qd)  # eval_fk only needs joint_q for body_q -- see gl_replay's docstring

    print(f"replaying variant {args.id}/{n - 1}: {label}  [{', '.join(which)}]  duration {t_end:.2f}s @ speed={args.speed}")
    print(f"nn-checkpoint: {args.nn_checkpoint}  device={device}")

    t0 = time.perf_counter()
    reported = False
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

        newton.eval_fk(model_, joint_q_wp, joint_qd_wp, state)
        contacts = model_.collide(state)
        viewer.begin_frame(sim_t)
        viewer.log_state(state)
        viewer.log_contacts(contacts, state)
        viewer.end_frame()
        wp.synchronize()

        if not reported and real_t >= t_end:
            print(
                f"[variant {args.id}] REAL   e_pos={real_e_pos:.4f} m  "
                f"e_rot={real_e_rot:.4f} rad ({np.degrees(real_e_rot):.2f} deg)"
            )
            print(
                f"[variant {args.id}] PRED   e_pos={pred_e_pos:.4f} m  "
                f"e_rot={pred_e_rot:.4f} rad ({np.degrees(pred_e_rot):.2f} deg)"
            )
            reported = True


if __name__ == "__main__":
    main()
