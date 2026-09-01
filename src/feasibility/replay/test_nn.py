"""GL replay of a single variant from a comparator/learning output HDF5 (compare_*.py or
learning/generate_dataset.py's dataset_*.h5) -- same real-time Newton ViewerGL playback as
gl_replay.py (see its docstring for the pose-only/no-physics mechanics) -- but also loads a
trained learning.model.PoseErrorMLP checkpoint, runs inference on this variant's commanded twist
and spawn pose, and once playback finishes prints both the REAL final-pose (e_pos, e_rot) --
computed directly from the file's ostrich/pose[-1] vs hstack/pose[-1], the exact
learning.pose_error.se3_error math -- and the network's PREDICTED (e_pos, e_rot) to the terminal,
so the two can be compared by eye.

What the network sees is whatever its checkpoint says it was trained on, rebuilt here by
build_feature_row(): either the pose-only (v_drive, wz_drive, spawn_x, spawn_y, spawn_yaw), or --
for a checkpoint trained with train.py --patch -- (v_drive, wz_drive) plus a body-frame terrain
patch sampled from THIS file's embedded terrain at this variant's spawn pose (see
learning/terrain_patch.py). Either way --file need not match the checkpoint's training terrain
for inference to run, but a pose-only checkpoint is meaningless off its own terrain, whereas a
patch checkpoint is exactly the thing that is supposed to survive the swap -- which makes
running one against a different --file the interesting experiment rather than a mistake.

For a --patch checkpoint, the viewer also draws the patch's body-frame footprint as a bright
line-outline rectangle on the terrain, at the spawn pose it was actually sampled at (see
patch_rectangle_world). It's an outline rather than a filled translucent quad because ViewerGL's
solid-shape pipeline has no alpha channel to blend with; the line pipeline does support blending,
so this is the closest thing to a translucent overlay the renderer can actually do. Nothing is
drawn for a pose-only checkpoint.

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
from feasibility.heightmap import HeightMapReader
from feasibility.learning.custom_dataset import build_input
from feasibility.learning.custom_dataset import Normalizer
from feasibility.learning.pose_error import pose_to_se3
from feasibility.learning.pose_error import se3_error
from feasibility.learning.terrain_patch import PatchSpec
from feasibility.learning.terrain_patch import sample_patches
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


def patch_rectangle_world(
    spec: PatchSpec, terrain: HeightMapReader, x: float, y: float, yaw: float, z_offset: float = 0.03
) -> np.ndarray:
    """[4, 3] world-frame corners of the patch's body-frame bounding rectangle, in order, at the
    pose the patch was actually sampled at (spawn pose -- see build_feature_row/main). Each
    corner is lifted `z_offset` m above the terrain height directly under it so the outline
    doesn't z-fight with the terrain mesh; the corners therefore follow the ground rather than
    sitting on one flat plane, which matters on the sloped/box terrains this patch is sampled on.

    Same body->world rotation as terrain_patch._body_to_world, kept as a local copy rather than
    imported: that function is underscore-private to terrain_patch (a sampling detail), while this
    one only exists to feed a debug-line overlay."""
    c, s = np.cos(yaw), np.sin(yaw)
    corners_body = np.array(
        [
            [spec.x_min, spec.y_min],
            [spec.x_max, spec.y_min],
            [spec.x_max, spec.y_max],
            [spec.x_min, spec.y_max],
        ]
    )
    wx = x + c * corners_body[:, 0] - s * corners_body[:, 1]
    wy = y + s * corners_body[:, 0] + c * corners_body[:, 1]
    wz = np.asarray(terrain.sample(wx, wy), dtype=np.float64) + z_offset
    return np.stack([wx, wy, wz], axis=1).astype(np.float32)


def build_feature_row(
    ckpt: dict[str, object],
    x_normalizer: Normalizer,
    terrain: HeightMapReader,
    v_drive: float,
    wz_drive: float,
    x: float,
    y: float,
    yaw: float,
) -> torch.Tensor:
    """Assembles the [1, in_dim] MODEL-READY feature row a checkpoint expects -- normalized and
    concatenated, ready to hand straight to model.predict().

    Keyed off the checkpoint's own metadata rather than assuming a layout, because there are now
    three of them: pose with raw yaw, pose with (cos_yaw, sin_yaw) (--yaw-encoding sincos), and
    (v, wz) + a flattened terrain patch (--patch). ckpt["n_scalar_features"] splits the scalar
    block x_normalizer applies to from the patch block it must NOT touch, and
    ckpt["patch_spec"] carries the exact sampling geometry -- both absent on checkpoints written
    before patch mode existed, whose every column is a scalar, which is what the fallbacks mean.

    Returns the row through custom_dataset.build_input() rather than assembling it here, so
    inference and PoseErrorDataset.__getitem__ cannot drift apart."""
    names = tuple(ckpt["feature_names"])
    n_scalar = int(ckpt.get("n_scalar_features", len(names)))
    scalar_names = names[:n_scalar]

    row = [v_drive, wz_drive]
    if "cos_yaw" in scalar_names:
        row += [x, y, np.cos(yaw), np.sin(yaw)]
    elif "yaw" in scalar_names:
        row += [x, y, yaw]
    if len(row) != n_scalar:
        raise SystemExit(
            f"checkpoint's scalar features {scalar_names} don't match anything this script can "
            f"build (got {len(row)} values for {n_scalar} columns)"
        )
    # load_checkpoint() put x_normalizer's statistics on the model's device, so the row has to
    # be there before it is normalized, not after.
    device = x_normalizer.mean.device
    scalars = torch.tensor([row], dtype=torch.float32, device=device)

    patch_spec = ckpt.get("patch_spec")
    patch = None
    if patch_spec is not None:
        spec = PatchSpec.from_dict(patch_spec)
        patch = torch.from_numpy(
            sample_patches(terrain, np.array([[x, y, yaw]]), spec).reshape(1, spec.size)
        ).to(device)
    return build_input(scalars, patch, x_normalizer)


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
    x_in = build_feature_row(
        ckpt, x_normalizer, terrain, v_drive, wz_drive, spawn_x, spawn_y, spawn_yaw
    )
    pred_e_pos, pred_e_rot = model.predict(x_in)[0].tolist()

    camera_pos = wp.vec3(obstacle_x, CAMERA_Y_OFFSET, CAMERA_Z)
    t_end = max(float(t[-1]) for t, _, _ in tracks.values())

    wp.init()
    model_, robots = build_model(terrain, which)
    viewer = newton.viewer.ViewerGL()
    viewer.set_model(model_)
    viewer.set_camera(pos=camera_pos, pitch=CAMERA_PITCH, yaw=CAMERA_YAW)
    state = model_.state()

    # Draw the patch footprint iff this checkpoint's input includes one -- ckpt["patch_spec"] is
    # only non-None for a train.py --patch checkpoint (see load_checkpoint/build_feature_row).
    # A filled translucent quad isn't achievable here: ViewerGL's solid-shape pipeline never
    # enables GL_BLEND and its per-instance colors are vec3 (no alpha channel) -- see
    # newton's viewer_gl.py _render_scene / gl/opengl.py. log_lines DOES blend, so a bright
    # outline is the closest supported stand-in -- static (spawn pose, once), since that's the
    # one pose the patch was actually sampled at, not something that tracks the robot per frame.
    ckpt_patch_spec = ckpt.get("patch_spec")
    if ckpt_patch_spec is not None:
        corners = patch_rectangle_world(
            PatchSpec.from_dict(ckpt_patch_spec), terrain, spawn_x, spawn_y, spawn_yaw
        )
        starts = wp.array(corners, dtype=wp.vec3, device=model_.device)
        ends = wp.array(np.roll(corners, -1, axis=0), dtype=wp.vec3, device=model_.device)
        viewer.log_lines("/patch_footprint", starts, ends, colors=(0.2, 0.9, 1.0))

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
