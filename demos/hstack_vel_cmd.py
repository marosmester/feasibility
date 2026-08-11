"""Replay demos/ostrich_vel_cmd.py's saved wheel commands through helhest_stack's
kinematic twin, on the same flat ground / mu / robot geometry, for a direct
ostrich-(dynamics) vs helhest_stack-(kinematic) comparison.

ostrich_vel_cmd.py drives Helhest Junior with an open-loop straight/arc-left/straight
wheel-velocity schedule and saves outputs/ostrich_vel_cmd.npz. Its docstring flags the
missing half of the experiment: a helhest_stack replay of the *same* commands, to see
how well the kinematic model's skid-steer slip law (alpha = 1 + k_turn*mu,
engine/step.py) reproduces ostrich's dynamic under-rotation. This is that replay.

Robot geometry, mass, and control-input convention (per-wheel [left, right, rear]
rad/s) are identical between the two repos -- see root CLAUDE.md -- so the saved
`cmd_wheel_omega` array is fed to helhest_stack VERBATIM, no reorder or sign flip.

The one thing that has to be dealt with is timestep: ostrich ran at dt=3e-2,
helhest_stack's canonical control rate is dynamics.DT=0.1. The command schedule is
piecewise-constant (only two distinct rows: straight, arc), so it zero-order-holds
onto any dt losslessly; helhest_stack's own step is a single explicit-Euler update
with no substepping, so unlike ostrich's dt sweep (alpha drifted 1.64->1.95 over
5e-2..5e-3) the achieved yaw here is flat to <1% across dt in [0.01, 0.1] -- run
`--dt-sweep` to confirm.

Usage:
    python demos/hstack_vel_cmd.py                          # GL viewer
    python demos/hstack_vel_cmd.py --no-view                # headless, batch
    python demos/hstack_vel_cmd.py --no-view --k-turn 1.0    # outdoor turn gain
    python demos/hstack_vel_cmd.py --no-view --dt-sweep      # dt-invariance check
"""

from __future__ import annotations

import argparse
import math
import pathlib
import time
from types import SimpleNamespace

import numpy as np
import warp as wp

from helhest import dynamics
from helhest import friction as friction_mod
from helhest import heightmap as hm_mod
from helhest.engine import ForwardSimulator
from helhest.engine import GridParams

IN_NPZ = pathlib.Path(__file__).parent.parent / "outputs" / "ostrich_vel_cmd.npz"
OUT_NPZ = pathlib.Path(__file__).parent.parent / "outputs" / "hstack_vel_cmd.npz"

DEFAULT_DT = 0.1  # [s] helhest_stack's canonical control rate (dynamics.DT)
GRID_MARGIN = 3.0  # [m] padding around the ostrich trajectory bbox
GRID_CELL = 0.05  # [m]


def load_ostrich_commands(
    npz_path: pathlib.Path, dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Zero-order-hold ostrich's [N,3] cmd_wheel_omega onto a dt grid.

    Returns (setpoints [T,3] float32, t_new [T], ostrich_pose [N,7], phases [3,3])."""
    d = np.load(npz_path)
    t_src, cmd_src, dt_src = d["t"], d["cmd_wheel_omega"], float(d["dt"])
    T = int(round((t_src[-1] + dt_src) / dt))
    t_new = np.arange(T) * dt
    idx = np.clip(np.searchsorted(t_src, t_new, side="right") - 1, 0, len(t_src) - 1)
    return cmd_src[idx].astype(np.float32), t_new.astype(np.float32), d["pose"], d["phases"]


def grid_extent(
    ostrich_pose: np.ndarray, margin: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Bounding box of the ostrich trajectory (relative to its settled start), padded."""
    xy = ostrich_pose[:, :2] - ostrich_pose[0, :2]
    xlim = (float(xy[:, 0].min()) - margin, float(xy[:, 0].max()) + margin)
    ylim = (float(xy[:, 1].min()) - margin, float(xy[:, 1].max()) + margin)
    return xlim, ylim


def euler_zyx_to_quat_xyzw(yaw: np.ndarray, pitch: np.ndarray, roll: np.ndarray) -> np.ndarray:
    """(yaw, pitch, roll) [T] -> quaternion [T,4] (qx,qy,qz,qw), R = Rz(yaw)@Ry(pitch)@Rx(roll)."""
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return np.stack([qx, qy, qz, qw], axis=-1).astype(np.float32)


def rollout(
    setpoints: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    dt: float,
    k_turn: float,
    mu: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Runs one ForwardSimulator rollout on flat, uniform-mu ground.

    Returns controlled [T,3] (x,y,yaw), derived [T,3] (z,pitch,roll), clearance [T],
    residual [T], turning [T,2] (alpha, x_icr), wheel_qd [T,3] realized wheel speed
    (== setpoints since tau_motor=0, kept for schema parity with ostrich) -- all
    sliced to drop the row-0 pre-command pose, so index k lines up with
    setpoints[k] the same way ostrich's pose[k] does."""
    T = len(setpoints)
    scene = hm_mod.flat(xlim=xlim, ylim=ylim, cell=GRID_CELL)
    mu_field = friction_mod.uniform(mu, xlim=xlim, ylim=ylim, cell=GRID_CELL)
    grid = GridParams(scene.nx, scene.ny, scene.cell, scene.x0, scene.y0)

    solver = dynamics.execution_solver(dt=dt, k_turn=k_turn)
    sim = ForwardSimulator(
        dynamics.robot_params(), solver, grid, batch_size=1, n_steps=T, device=device
    )
    sim.set_terrain(
        wp.array(np.ascontiguousarray(scene.H, np.float32), dtype=wp.float32, device=device)
    )
    sim.set_friction(mu_field)
    controlled, derived, clearance, residual = sim.rollout(
        np.ascontiguousarray(setpoints[:, None, :], np.float32), (0.0, 0.0, 0.0), np.zeros(3)
    )
    turning = sim.turning.numpy()[:, 0, :]
    wheel_qd = sim.current_wheel_omega.numpy()[1:, 0, :]
    return (
        controlled[1:, 0, :],
        derived[1:, 0, :],
        clearance[:, 0],
        residual[:, 0],
        turning,
        wheel_qd,
    )


def print_summary(
    dt: float, controlled: np.ndarray, ostrich_pose: np.ndarray, mu: float, k_turn: float
) -> None:
    yaw = float(controlled[-1, 2])

    def yaw_q(q: np.ndarray) -> float:
        return math.atan2(2.0 * (q[3] * q[2] + q[0] * q[1]), 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]))

    ostrich_yaw = yaw_q(ostrich_pose[-1, 3:7]) - yaw_q(ostrich_pose[0, 3:7])
    print(f"dt={dt:.3f}  T={len(controlled)}  k_turn={k_turn}  mu={mu}")
    print(
        f"hstack yaw : {math.degrees(yaw):6.2f} deg   "
        f"end=({controlled[-1, 0]:.2f}, {controlled[-1, 1]:.2f})   "
        f"alpha_implied~{(1.0 + k_turn * mu):.3f}"
    )
    print(
        f"ostrich yaw: {math.degrees(ostrich_yaw):6.2f} deg   "
        f"end=({ostrich_pose[-1, 0] - ostrich_pose[0, 0]:.2f}, "
        f"{ostrich_pose[-1, 1] - ostrich_pose[0, 1]:.2f})"
    )


def run_headless(args: argparse.Namespace) -> None:
    wp.init()
    setpoints, t_new, ostrich_pose, phases = load_ostrich_commands(IN_NPZ, args.dt)
    xlim, ylim = grid_extent(ostrich_pose, GRID_MARGIN)

    if args.dt_sweep:
        for dt in (0.1, 0.05, 0.03, 0.01):
            sp, _, _, _ = load_ostrich_commands(IN_NPZ, dt)
            controlled, *_ = rollout(sp, xlim, ylim, dt, args.k_turn, args.mu, args.device)
            print_summary(dt, controlled, ostrich_pose, args.mu, args.k_turn)
        return

    controlled, derived, clearance, residual, turning, wheel_qd = rollout(
        setpoints, xlim, ylim, args.dt, args.k_turn, args.mu, args.device
    )
    print_summary(args.dt, controlled, ostrich_pose, args.mu, args.k_turn)

    quat = euler_zyx_to_quat_xyzw(controlled[:, 2], derived[:, 1], derived[:, 2])
    pose = np.concatenate([controlled[:, :2], derived[:, :1], quat], axis=1).astype(np.float32)

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        dt=np.float32(args.dt),
        t=t_new,
        cmd_wheel_omega=setpoints,
        pose=pose,
        wheel_qd=wheel_qd,
        phases=phases,
        controlled=controlled,
        derived=derived,
        turning=turning,
        clearance=clearance,
        residual=residual,
        k_turn=np.float32(args.k_turn),
        mu=np.float32(args.mu),
    )
    print(f"saved {OUT_NPZ}")


# --- GL viewer -----------------------------------------------------------------
def _pose_to_st(pose3: np.ndarray, der3: np.ndarray) -> SimpleNamespace:
    x, y, yaw = float(pose3[0]), float(pose3[1]), float(pose3[2])
    z, pitch, roll = float(der3[0]), float(der3[1]), float(der3[2])
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], np.float32)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], np.float32)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], np.float32)
    R = (Rz @ Ry @ Rx).astype(np.float32)
    return SimpleNamespace(x=x, y=y, yaw=yaw, valid=True, place={"z": z, "R": R})


def run_view(args: argparse.Namespace) -> None:
    import glfw
    from OpenGL import GL as gl
    from OpenGL import GLU as glu
    from helhest.viz.render import _draw
    from helhest.viz.render import _init_gl
    from helhest.viz.render import build_robot
    from helhest.viz.render import build_terrain

    wp.init()
    setpoints, t_new, ostrich_pose, _phases = load_ostrich_commands(IN_NPZ, args.dt)
    xlim, ylim = grid_extent(ostrich_pose, GRID_MARGIN)
    controlled, derived, *_ = rollout(
        setpoints, xlim, ylim, args.dt, args.k_turn, args.mu, args.device
    )
    T = len(controlled)
    print_summary(args.dt, controlled, ostrich_pose, args.mu, args.k_turn)

    scene = hm_mod.flat(xlim=xlim, ylim=ylim, cell=GRID_CELL)
    terrain = build_terrain(scene)
    robot = build_robot()

    ghost_xy = ostrich_pose[:, :2] - ostrich_pose[0, :2]

    if not glfw.init():
        raise RuntimeError("glfw.init() failed")
    win = glfw.create_window(1100, 820, "hstack_vel_cmd", None, None)
    if win is None:
        raise RuntimeError("Failed to create GLFW window")
    glfw.set_window_pos(win, 100, 80)
    glfw.make_context_current(win)
    _init_gl()

    cam = [-2.1, 0.85, 12.0]
    ms = {"down": False, "x": 0.0, "y": 0.0}

    def on_button(w_, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            ms["down"] = action == glfw.PRESS
            ms["x"], ms["y"] = glfw.get_cursor_pos(w_)

    def on_cursor(w_, x, y):
        if ms["down"]:
            cam[0] -= (x - ms["x"]) * 0.01
            cam[1] = float(np.clip(cam[1] + (y - ms["y"]) * 0.01, 0.05, 1.5))
            ms["x"], ms["y"] = x, y

    def on_scroll(w_, dx, dy):
        cam[2] = float(np.clip(cam[2] - dy * 0.8, 2.0, 60.0))

    glfw.set_mouse_button_callback(win, on_button)
    glfw.set_cursor_pos_callback(win, on_cursor)
    glfw.set_scroll_callback(win, on_scroll)

    def draw_trail(points: list, color_rgb: tuple[float, float, float]) -> None:
        if len(points) < 2:
            return
        gl.glDisable(gl.GL_LIGHTING)
        gl.glColor3f(*color_rgb)
        gl.glLineWidth(3.0)
        gl.glBegin(gl.GL_LINE_STRIP)
        for p in points:
            gl.glVertex3f(*p)
        gl.glEnd()
        gl.glEnable(gl.GL_LIGHTING)

    def draw_robot(st: SimpleNamespace) -> None:
        V, N, C, _ = robot
        R4 = np.eye(4, dtype=np.float32)
        R4[:3, :3] = st.place["R"]
        gl.glPushMatrix()
        gl.glTranslatef(st.x, st.y, st.place["z"])
        gl.glMultMatrixf(np.ascontiguousarray(R4.T))
        _draw(V, N, C)
        gl.glPopMatrix()

    t = 0
    trail: list = []
    frame_dt = args.dt / max(args.speed, 1e-3)
    t_next = time.monotonic()

    while not glfw.window_should_close(win):
        glfw.poll_events()
        if glfw.get_key(win, glfw.KEY_ESCAPE) == glfw.PRESS or glfw.get_key(win, glfw.KEY_Q) == glfw.PRESS:
            break

        now = time.monotonic()
        if now >= t_next:
            t += 1
            t_next = now + frame_dt
            if t >= T:
                t = 0
                trail.clear()

        st = _pose_to_st(controlled[t], derived[t])
        trail.append((st.x, st.y, st.place["z"] + 0.03))

        glfw.set_window_title(
            win, f"hstack_vel_cmd  [step {t}/{T}]  yaw={math.degrees(st.yaw):.1f} deg"
        )

        w, h = glfw.get_framebuffer_size(win)
        az, el, dist = cam
        tgt = np.array([st.x, st.y, st.place["z"]])
        eye = tgt + dist * np.array(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
        )
        gl.glViewport(0, 0, w, h)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        glu.gluPerspective(50.0, w / max(h, 1), 0.1, 100.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        glu.gluLookAt(*eye, *tgt, 0, 0, 1)

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        _draw(*terrain[:3], terrain[3])
        if not args.no_ghost:
            draw_trail(
                list(zip(ghost_xy[:, 0], ghost_xy[:, 1], np.full(len(ghost_xy), 0.03))),
                (0.86, 0.08, 0.24),
            )
        draw_trail(trail, (0.27, 0.51, 0.71))
        draw_robot(st)
        glfw.swap_buffers(win)

    glfw.terminate()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--dt", type=float, default=DEFAULT_DT, help=f"control timestep [s] (default {DEFAULT_DT})"
    )
    ap.add_argument(
        "--k-turn",
        type=float,
        default=dynamics.K_TURN,
        help=f"turn gain (default {dynamics.K_TURN}, indoor)",
    )
    ap.add_argument(
        "--mu", type=float, default=0.8, help="uniform ground friction (default 0.8, matches ostrich)"
    )
    ap.add_argument("--device", default="cuda:0", help="Warp device (default cuda:0)")
    ap.add_argument("--no-view", action="store_true", help="headless: skip the GL viewer, save npz")
    ap.add_argument("--no-ghost", action="store_true", help="viewer: hide the ostrich trail overlay")
    ap.add_argument(
        "--dt-sweep",
        action="store_true",
        help="headless: print yaw at dt=0.1/0.05/0.03/0.01 and exit",
    )
    ap.add_argument(
        "--speed", type=float, default=1.0, help="viewer playback speed multiplier (default 1.0)"
    )
    args = ap.parse_args()

    if not IN_NPZ.exists():
        raise SystemExit(f"missing {IN_NPZ} -- run demos/ostrich_vel_cmd.py first")

    if args.no_view:
        run_headless(args)
    else:
        run_view(args)


if __name__ == "__main__":
    main()
