"""GL viewer for a lattice_learning dataset_arc_*.h5 (generate_dataset.py's output): replays ONE
trial's real ostrich trajectory (`ostrich/pose`, a genuine T_RECORD_OSTRICH-step rollout -- unlike
grid_learning's dataset, which stores only frozen final poses) over its real terrain in Newton's
ViewerGL, alongside a FROZEN "ghost" robot parked at `ref_pose` -- the arc-plus-settle target the
training label (`custom_dataset.py`'s `se3_errors`) is actually computed against -- and a drawn
polyline of the geometric arc itself (`t0_pose` -> `arc_end_pose`, via `arc.integrate_arc`). This
lets a person SEE how far the real rollout's endpoint diverged from the label's target: the visual
counterpart of the `(e_pos, e_rot)` numbers this script also prints, and that
`plotting/dataset_patch_viewer.py` already prints for this same schema.

Pose-only playback, not physics -- same mechanism as `replay/gl_replay.py`, whose
`build_model`/`integrate_wheel_angle`/`interp_pose`/`interp_series` are imported and reused
verbatim here (the way `replay/test_nn.py` already reuses them): each frame writes `joint_q`
directly (chassis free-joint 7 slots from `interp_pose`, 3 wheel revolute slots from a wheel angle
integrated from the file's `ostrich/wheel_qd`, since the file logs wheel VELOCITY, not angle), then
`newton.eval_fk` propagates it to `state.body_q`. `model.collide()` only feeds the viewer's contact
overlay -- no solver step, no control targets. `pose_to_se3`/`se3_errors` come from this package's
own `custom_dataset.py` (intra-package reuse, not a cross-tree import -- the same choice
`plotting/dataset_patch_viewer.py` already makes for the identical formula); `terrain_from_h5` and
`HeightMapReader` are shared infrastructure (design.md section 11a). `replay/` is not on any tree's
"does not import from" list -- that only names sibling dataset-generation trees (`learning/`,
`grid_learning/`, `grid_learning_2/`) -- so reusing its generic Newton/GL playback machinery here
stays within this package's independence policy.

Unlike `grid_learning/gl_replay_grid.py`'s lattice stepper (one dataset row = one frozen keyframe,
no trajectory to interpolate), `dataset_arc_*.h5` gives each trial a real time series, so this
viewer is a continuous single-trajectory replay like `gl_replay.py`'s own (`--id`/`--speed`/
`--loop`), not a stepper. There is no `--which` either: this schema has no `hstack/` group (the
generator never runs the dynamic twin, see its own module docstring), so there is only ever one
real trajectory (ostrich, orange) plus one static reference marker (cyan) and one static arc guide
line (gold) -- nothing to choose between.

`build_model` colors robots by name via `replay.gl_replay.ROBOT_COLOR`, which only knows
`"ostrich"`/`"hstack"` out of the box -- this script registers a `"reference"` entry into that same
dict before calling it, rather than forking `build_model` to take an explicit color map.

The camera is framed locally (not imported): `gl_replay.py`'s own camera is positioned relative to
a single `f.attrs["obstacle_x"]`, which doesn't exist in this schema (maps vary in extent, and an
obstacle -- if any -- moves per map). Instead this uses the terrain-center-relative framing
`grid_learning/gl_replay_grid.py` already uses for the same "no single fixed obstacle" reason.

CLI parameters:
    --file PATH      dataset_arc_*.h5 path (required -- filenames encode M/R/seed, no single
                     sensible default)
    --id INT         trial index into the flat [0, n) range (default: 0)
    --speed FLOAT    playback speed multiplier (default: 1.0 = real time)
    --loop           loop playback instead of freezing on the last frame
    --dry-run        load + validate the trial and print its status line, no GL viewer -- the
                     only piece of this script not needing a DISPLAY (it still needs a Warp
                     device for the `valid` breakdown below), mirroring generate_dataset.py's own
                     +dry_run and gl_replay_grid.py's --dry-run
    --device STR     device for re-settling the arc endpoint when breaking down `valid` into its
                     four component checks (see below) (default: cuda:0)

Every invocation also prints a breakdown of the stored `valid` flag into the four conditions
generate_dataset.py's simulate_map ANDs together (design.md section 7b) -- `finite`, `settle_ok`
(a real settle re-run AT THE ARC'S OWN ENDPOINT, the only one of the four needing a device),
`displacement_ok`, `overhang_ok` -- so an INVALID row's actual cause is visible instead of just
the combined flag. If the recomputed AND doesn't match the file's stored `valid`, that's flagged
as a WARNING rather than trusted silently, since it would mean generation-time inputs (mu, robot
params, terrain) drifted from what this script re-derives them as.

Usage:
    python src/feasibility/lattice_learning/gl_replay_arc.py --file outputs/dataset_arc_rough_box0_M5_R16_seed0.h5 --dry-run
    python src/feasibility/lattice_learning/gl_replay_arc.py --file outputs/dataset_arc_rough_box0_M5_R16_seed0.h5
    python src/feasibility/lattice_learning/gl_replay_arc.py --file outputs/dataset_arc_rough_box0_M5_R16_seed0.h5 --id 42
    python src/feasibility/lattice_learning/gl_replay_arc.py --file outputs/dataset_arc_rough_box0_M5_R16_seed0.h5 --id 42 --speed 0.25 --loop
"""
from __future__ import annotations

import argparse
import pathlib
import time

import h5py
import newton
import numpy as np
import warp as wp
from helhest.engine import RobotParams

from feasibility.comparator.provenance import terrain_from_h5
from feasibility.heightmap import HeightMapReader
from feasibility.lattice_learning.arc import ARC_LEN
from feasibility.lattice_learning.arc import integrate_arc
from feasibility.lattice_learning.custom_dataset import pose_to_se3
from feasibility.lattice_learning.custom_dataset import se3_errors
from feasibility.lattice_learning.generate_dataset import MAX_SPAWN_DISPLACEMENT
from feasibility.lattice_learning.generate_dataset import settle_batch
from feasibility.lattice_learning.generate_dataset import settle_feasible
from feasibility.lattice_learning.patch import patch_overhangs
from feasibility.lattice_learning.patch import patch_spec_from_attrs
from feasibility.replay.gl_replay import build_model
from feasibility.replay.gl_replay import integrate_wheel_angle
from feasibility.replay.gl_replay import interp_pose
from feasibility.replay.gl_replay import interp_series
from feasibility.replay.gl_replay import ROBOT_COLOR
from feasibility.replay.gl_replay import WHEEL_NAMES

REFERENCE_COLOR = (0.2, 0.9, 1.0)  # cyan -- "target/reference", same convention test_nn.py's own
# static patch-footprint outline uses. Registered into gl_replay.ROBOT_COLOR (imported above)
# rather than forking build_model, since build_model looks colors up by robot name and only knows
# "ostrich"/"hstack" out of the box.
ROBOT_COLOR["reference"] = REFERENCE_COLOR

ARC_LINE_COLOR = (1.0, 0.9, 0.2)  # gold -- the geometric arc guide line, distinct from both robots
ARC_LINE_Z_OFFSET = 0.03  # m, lifts the guide line just above the terrain to avoid z-fighting
N_ARC_POINTS = 20

# Terrain-center-relative camera framing (see module docstring on why gl_replay.py's own
# obstacle_x-relative camera doesn't fit this schema) -- same convention/constants
# grid_learning/gl_replay_grid.py uses for its own "no single fixed obstacle" terrain.
CAMERA_YAW = 90.0
CAMERA_PITCH = -25.0
CAMERA_BACK = 0.7  # camera distance behind terrain center, as a fraction of the larger extent
CAMERA_UP = 0.4  # camera height above max_z, as a fraction of the larger extent


def arc_polyline_world(
    terrain: HeightMapReader, t0_pose: np.ndarray, kappa: float, n_points: int = N_ARC_POINTS,
    z_offset: float = ARC_LINE_Z_OFFSET,
) -> np.ndarray:
    """[n_points, 3] world (x, y, z) points along the geometric arc from `t0_pose` at curvature
    `kappa`, via `arc.integrate_arc(t0_pose, kappa, s)` with `s = linspace(0, ARC_LEN, n_points)`
    -- one call, no loop (`integrate_arc` broadcasts `kappa`/`length` against a single pose). `z`
    is the terrain height directly under each (x, y) plus `z_offset`, so the guide line follows
    the ground rather than floating on one flat plane -- same per-point ground-following
    convention `replay/test_nn.py`'s patch-footprint outline uses."""
    s = np.linspace(0.0, ARC_LEN, n_points)
    xy_yaw = integrate_arc(t0_pose, kappa, s)  # [n_points, 3]
    x, y = xy_yaw[:, 0], xy_yaw[:, 1]
    z = np.asarray(terrain.sample(x, y), dtype=np.float64) + z_offset
    return np.stack([x, y, z], axis=-1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=pathlib.Path, required=True, help="dataset_arc_*.h5 path")
    ap.add_argument("--id", type=int, default=0, help="trial index into the flat [0, n) range (default: 0)")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier (default 1.0 = real time)")
    ap.add_argument("--loop", action="store_true", help="loop playback instead of freezing on the last frame")
    ap.add_argument("--dry-run", action="store_true", help="validate + print stats only, no GL viewer")
    ap.add_argument("--device", type=str, default="cuda:0", help="torch/warp device for re-settling the arc endpoint to break down `valid` (default: cuda:0, matching generate_dataset.py)")
    args = ap.parse_args()

    # Materialize everything to numpy inside the `with` -- h5py datasets are invalid once the
    # file closes, and the render loop below runs long after this returns.
    with h5py.File(args.file, "r") as f:
        n = int(f.attrs["n"])
        if not (0 <= args.id < n):
            raise SystemExit(f"--id must be in [0, {n}), got {args.id}")

        attrs = dict(f.attrs)
        map_index = int(f["map_index"][args.id])
        map_path = f["map_path"].asstr()[args.id]
        kappa = float(f["kappa"][args.id])
        valid = bool(f["valid"][args.id])
        swept_clear = bool(f["swept_clear"][args.id])
        t0_pose = f["t0_pose"][args.id].astype(np.float64)  # [3]
        belief_pose = f["belief_pose"][args.id].astype(np.float64)  # [3]
        arc_end_pose = f["arc_end_pose"][args.id].astype(np.float64)  # [3]
        ref_pose = f["ref_pose"][args.id].astype(np.float64)  # [7]

        terrain = terrain_from_h5(f, args.id)
        t = f["ostrich/t"][:].astype(np.float64)  # [T]
        ostrich_pose = f["ostrich/pose"][:, args.id, :].astype(np.float64)  # [T, 7]
        wheel_theta = integrate_wheel_angle(
            t, f["ostrich/wheel_qd"][:, args.id, :].astype(np.float64)
        )  # [T, 3]

    e_pos, e_rot = se3_errors(pose_to_se3(ref_pose), pose_to_se3(ostrich_pose[-1]))
    e_pos, e_rot = float(e_pos), float(e_rot)

    flags = "valid" if valid else "INVALID"
    flags += "  swept_clear" if swept_clear else ""
    print(f"[trial {args.id}/{n - 1}]  map={map_path} (map_index={map_index})  kappa={kappa:+.3f} 1/m  {flags}")
    print(
        f"  e_pos={e_pos:.4f} m  e_rot={e_rot:.4f} rad ({np.degrees(e_rot):.2f} deg)   "
        f"ostrich duration={t[-1]:.3f}s ({len(t)} steps @ dt={t[1] - t[0]:.4f}s)"
    )

    # `valid`'s own four ANDed conditions (generate_dataset.py's simulate_map, design.md section
    # 7b), recomputed from the file's stored fields so an INVALID row's actual cause is visible
    # instead of just the combined flag. Three are cheap (pure numpy over already-loaded arrays);
    # `settle_ok` needs a real settle re-run at the arc's own endpoint (the same one generation
    # did), so it is the only one that needs Warp initialized and a device.
    wp.init()
    finite = bool(
        np.isfinite(ostrich_pose[-1]).all() and np.isfinite(t0_pose).all() and np.isfinite(ref_pose).all()
    )
    displacement = float(np.linalg.norm(ostrich_pose[-1, :2] - t0_pose[:2]))
    displacement_ok = displacement <= MAX_SPAWN_DISPLACEMENT
    spec = patch_spec_from_attrs(attrs)
    overhang_ok = not bool(patch_overhangs(terrain, belief_pose[None], spec)[0])
    derived, residual, clearance = settle_batch(terrain, arc_end_pose[None], float(attrs["mu"]), args.device)
    settle_ok = bool(settle_feasible(derived, residual, clearance, RobotParams())[0])
    recomputed_valid = finite and settle_ok and displacement_ok and overhang_ok

    print(
        f"  valid breakdown: finite={finite}  settle_ok={settle_ok}  "
        f"displacement_ok={displacement_ok} ({displacement:.4f} m <= {MAX_SPAWN_DISPLACEMENT:.4f} m)  "
        f"overhang_ok={overhang_ok}"
    )
    if not settle_ok:
        z, pitch, roll = derived[0]
        print(
            f"    settle @ arc_end_pose: pitch={np.degrees(pitch):.2f} deg roll={np.degrees(roll):.2f} deg  "
            f"residual={residual[0]:.4g} (tol {RobotParams().resid_tol:.4g})  "
            f"clearance={clearance[0]:.4f} m (margin {RobotParams().clear_margin:.4f} m)"
        )
    if recomputed_valid != valid:
        print(
            f"  WARNING: recomputed valid={recomputed_valid} does not match the file's stored "
            f"valid={valid} -- mu/robot params/terrain may have drifted since generation, or "
            f"this is a genuine bug in this breakdown"
        )

    if args.dry_run:
        print("[dry-run]  loaded + validated, nothing rendered")
        return

    points = arc_polyline_world(terrain, t0_pose, kappa)

    which = ("ostrich", "reference")
    model, robots = build_model(terrain, which)
    viewer = newton.viewer.ViewerGL()
    viewer.set_model(model)
    cx = terrain.x0 + terrain.nx * terrain.cell / 2.0
    cy = terrain.y0 + terrain.ny * terrain.cell / 2.0
    extent = max(terrain.nx, terrain.ny) * terrain.cell
    viewer.set_camera(
        pos=wp.vec3(cx, cy - CAMERA_BACK * extent, terrain.max_z + CAMERA_UP * extent),
        pitch=CAMERA_PITCH, yaw=CAMERA_YAW,
    )
    state = model.state()

    q_start = model.joint_q_start.numpy()
    joint_q = model.joint_q.numpy().copy()

    base_ref = q_start[robots["reference"]["base"]]
    joint_q[base_ref:base_ref + 7] = ref_pose  # frozen, one-time -- never touched again below
    base_ost = q_start[robots["ostrich"]["base"]]

    joint_q_wp = wp.array(joint_q, dtype=wp.float32, device=model.device)
    joint_qd_wp = wp.zeros_like(model.joint_qd)  # eval_fk only needs joint_q for body_q

    starts = wp.array(points[:-1].astype(np.float32), dtype=wp.vec3, device=model.device)
    ends = wp.array(points[1:].astype(np.float32), dtype=wp.vec3, device=model.device)
    viewer.log_lines("/arc_guide", starts, ends, colors=ARC_LINE_COLOR)

    t_end = float(t[-1])
    print(
        f"replaying trial {args.id}  duration {t_end:.3f}s @ speed={args.speed}  "
        f"(orange = real ostrich rollout, cyan = frozen arc+settle reference target, "
        f"gold = commanded arc)"
    )

    t0 = time.perf_counter()
    while viewer.is_running():
        real_t = (time.perf_counter() - t0) * args.speed
        sim_t = (real_t % t_end) if (args.loop and t_end > 0) else min(real_t, t_end)

        joint_q[base_ost:base_ost + 7] = interp_pose(t, ostrich_pose, sim_t)
        theta = interp_series(t, wheel_theta, sim_t)
        for wheel_name, angle in zip(WHEEL_NAMES, theta):
            joint_q[q_start[robots["ostrich"][wheel_name]]] = angle
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
