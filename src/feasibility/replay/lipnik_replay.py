"""GL replay of a real-robot Lipnik recording: the lidar heightmap plus the odometry/wheel log of
the same drive, rendered with the real Helhest Junior mesh in Newton's interactive viewer.
Pose-only playback like gl_replay.py (joint_q written directly, newton.eval_fk, no physics).

Inputs, all under assets/lipnik/ (gitignored -- see the module-level constants for the layout):
    <scene>/odometry.txt           one row per pose: t, 3x4 row-major [R|t], 0 0 0 1 (Unix seconds)
    <scene>/odometry_segments.txt  the odometry restarts at identity after a gap/teleport; each
                                   segment is its OWN frame, and its own heightmap
    <scene>/joint_states.npy       [T, 10] = t, 3 wheel angles [rad], 3 rates, 3 efforts, in the
                                   order of <scene>/joint_names.txt (left, rear, right)
    full_heightmaps/lipnik_<map>_seg<k>.npy
                                   [N, 3] x y z points on an exact 0.1 m lattice, ~20-60% of the
                                   bounding box (a swath along the track), NOT a dense grid

What had to be inferred (none of it is written down in the data):
  * Map <-> scene. The map files are numbered differently from the scenes (`MAP_FOR_SCENE`),
    and `lipnik_<map>_seg<k>` is the heightmap of odometry segment k of its scene, in that
    segment's own start frame, gravity-aligned (see "Gravity" below). Not the scene's number:
    GPS puts the scene folders sharing a map's number ~240 km from Lipnik, on a different route.
    `check_pairing` re-derives this from the data on every run -- share of the segment's poses
    within 3 m of map data, and of the map's points within 15 m of the track -- and refuses a
    pairing below `MIN_COVERAGE`, so a wrong table entry fails loudly instead of replaying a
    robot over someone else's terrain.
  * Pose frame. Odometry body +x is the direction of travel (cos ~ 1 against the velocity) and
    positive wheel rate is forward on all three wheels, so the pose is used as the robot's
    chassis frame as-is (the true lidar-to-base offset, ~0.1 m, is ignored).
  * Gravity. The raw odometry frame is tilted 2-4 deg against the map, which reads as metres of
    z "drift" over a run. The map is built in that frame rotated by <scene>/gravity_align.npy's
    matrix at the segment's FIRST frame, applied once to the whole segment (`p' = G0 p`,
    `R' = G0 R`): that leaves < 0.2 deg of tilt and 2-6 cm of z residual against the map on every
    pairing. The per-frame matrices are something else (they swing up to 27 deg over a run) and
    applying them per pose makes the fit far worse.
  * Height. Even aligned, the pose is the sensor's, not the wheels', so the chassis is still
    lowered until the wheels sit on the map under them (`ground_z`), keeping the aligned
    roll/pitch/yaw; the dry-run reports how far that moved it. Poses with no map under any wheel
    take z linearly interpolated from the neighbouring poses that have one.
  * Wheel spin. The logged angles are used directly (clock shared with the odometry); before
    the first joint_states sample the wheels hold still.

Holes: the maps are swaths, so there is no terrain beside/behind them. The mesh is built from
the cells that exist (triangles only where all three corners are real data) after closing gaps of
at most `--fill` metres (default 0.3, i.e. dotted lidar rings); nothing is invented beyond that.

CLI parameters:
    --scene {0003,0004,...}  odometry scene (default 0003)
    --segment INT            odometry/map segment within the scene (default 1)
    --map-scene STR          override which map file number holds this scene's heightmaps
    --speed FLOAT            playback speed multiplier (default 1.0)
    --loop                   loop instead of freezing on the last frame
    --fill FLOAT             close terrain gaps up to this many metres (default 0.3)
    --stride INT             mesh vertex stride in cells (default 1; raise for slow GPUs)
    --camera {follow,free}   chase camera behind the robot, or leave the mouse in charge
    --dry-run                load, verify the pairing, print a summary; no viewer

Usage:
    python src/feasibility/replay/lipnik_replay.py --dry-run
    python src/feasibility/replay/lipnik_replay.py --scene 0003 --speed 4
    python src/feasibility/replay/lipnik_replay.py --scene 0004 --segment 3 --camera free --loop
"""

from __future__ import annotations

import argparse
import pathlib
import time
from dataclasses import dataclass

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
LIPNIK_DIR = REPO_ROOT / "assets" / "lipnik"
CELL = 0.1  # heightmap lattice [m]
WHEEL_RADIUS = 0.35  # == HelhestJuniorConfig.WHEEL_RADIUS
# Wheel centres in the chassis frame == HelhestJuniorConfig.{LEFT,RIGHT,REAR}_WHEEL_POS; repeated
# here so the pairing/dry-run path needs no ostrich import.
WHEEL_LOCAL = {"left": (0.0, 0.365, 0.0), "right": (0.0, -0.365, 0.0), "rear": (-0.75, 0.0, 0.0)}
# map file number holding each scene's heightmaps (see module docstring; verified, not trusted)
MAP_FOR_SCENE = {"0003": "0005", "0004": "0006"}
MIN_COVERAGE = 0.8
JOINTS_PER_ROBOT = 4  # base (free), left, right, rear -- see replay/gl_replay.py
WHEEL_ORDER = ("left", "right", "rear")  # joint order create_helhest_junior_model adds

CAMERA_BACK = 5.0
CAMERA_UP = 2.5
CAMERA_PITCH = -22.0
CAMERA_YAW_TAU = 0.6  # s, low-pass on the chase heading so a point turn doesn't whip the view


@dataclass
class Terrain:
    H: np.ndarray  # [ny, nx], NaN where there is no data; cell centres at (x0+(j+.5)c, y0+(i+.5)c)
    x0: float
    y0: float


@dataclass
class Run:
    t: np.ndarray  # [T] seconds since the segment's first pose
    pos: np.ndarray  # [T, 3] with z already re-grounded on the map
    quat: np.ndarray  # [T, 4] (x, y, z, w), sign-continuous
    wheel_t: np.ndarray  # [W] seconds since the segment start
    wheel_theta: np.ndarray  # [W, 3] angle in WHEEL_ORDER, zero at the first sample
    odo_z: np.ndarray  # [T] the gravity-aligned odometry z, kept for the dry-run report
    grounded: np.ndarray  # [T] bool: some wheel had map under it


def load_points(map_scene: str, segment: int) -> np.ndarray:
    path = LIPNIK_DIR / "full_heightmaps" / f"lipnik_{map_scene}_seg{segment}.npy"
    if not path.exists():
        raise SystemExit(f"no heightmap {path}")
    return np.load(path).astype(np.float64)


def points_to_grid(pts: np.ndarray) -> Terrain:
    """Scatter the lattice points into a NaN-filled [ny, nx] grid; asserts they ARE on the lattice."""
    ix = np.round((pts[:, 0] - pts[:, 0].min()) / CELL).astype(int)
    iy = np.round((pts[:, 1] - pts[:, 1].min()) / CELL).astype(int)
    off = max(np.abs(pts[:, 0] - (pts[:, 0].min() + ix * CELL)).max(), np.abs(pts[:, 1] - (pts[:, 1].min() + iy * CELL)).max())
    if off > 1e-3:
        raise SystemExit(f"heightmap points are not on a {CELL} m lattice (worst {off:.4f} m)")
    H = np.full((iy.max() + 1, ix.max() + 1), np.nan)
    H[iy, ix] = pts[:, 2]
    return Terrain(H, pts[:, 0].min() - CELL / 2, pts[:, 1].min() - CELL / 2)


def close_gaps(H: np.ndarray, cells: int) -> np.ndarray:
    """Fill NaN cells with the mean of their valid 4-neighbours, `cells` rounds -- grows every
    data edge by at most `cells` cells, so gaps up to 2*cells wide close and nothing else does."""
    H = H.copy()
    for _ in range(cells):
        P = np.pad(H, 1, constant_values=np.nan)
        nb = np.stack([P[:-2, 1:-1], P[2:, 1:-1], P[1:-1, :-2], P[1:-1, 2:]])
        cnt = np.isfinite(nb).sum(0)
        fill = np.where(cnt > 0, np.nansum(nb, 0) / np.maximum(cnt, 1), np.nan)
        H = np.where(np.isnan(H), fill, H)
    return H


def terrain_height(T: Terrain, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Nearest-cell height, NaN outside the grid or on a hole."""
    j = np.floor((x - T.x0) / CELL).astype(int)
    i = np.floor((y - T.y0) / CELL).astype(int)
    ok = (i >= 0) & (i < T.H.shape[0]) & (j >= 0) & (j < T.H.shape[1])
    h = np.full(x.shape, np.nan)
    h[ok] = T.H[i[ok], j[ok]]
    return h


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """[T,3,3] -> [T,4] (x,y,z,w), Shepperd's method, vectorised."""
    m = R
    tr = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = np.zeros((len(m), 4))
    for k in range(len(m)):
        a = m[k]
        if tr[k] > 0:
            s = 2 * np.sqrt(tr[k] + 1)
            q[k] = ((a[2, 1] - a[1, 2]) / s, (a[0, 2] - a[2, 0]) / s, (a[1, 0] - a[0, 1]) / s, s / 4)
        else:
            d = int(np.argmax([a[0, 0], a[1, 1], a[2, 2]]))
            i, j, kk = d, (d + 1) % 3, (d + 2) % 3
            s = 2 * np.sqrt(1 + a[i, i] - a[j, j] - a[kk, kk])
            v = np.zeros(4)
            v[i] = s / 4
            v[j] = (a[j, i] + a[i, j]) / s
            v[kk] = (a[kk, i] + a[i, kk]) / s
            v[3] = (a[kk, j] - a[j, kk]) / s
            q[k] = v
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    for k in range(1, len(q)):  # keep neighbours in one hemisphere so nlerp never takes the long way
        if q[k] @ q[k - 1] < 0:
            q[k] = -q[k]
    return q


def load_odometry(scene: str, segment: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(t [T] absolute, pos [T,3], R [T,3,3]) of one odometry segment."""
    rows = [l.split() for l in (LIPNIK_DIR / scene / "odometry_segments.txt").read_text().splitlines() if not l.startswith("#")]
    segs = {int(r[0]): (float(r[1]), float(r[2]), r[4]) for r in rows}
    if segment not in segs:
        raise SystemExit(f"scene {scene} has segments {sorted(segs)}, not {segment}")
    t0, t1, _ = segs[segment]
    o = np.loadtxt(LIPNIK_DIR / scene / "odometry.txt")
    o = o[(o[:, 0] >= t0 - 1e-6) & (o[:, 0] <= t1 + 1e-6)]
    return o[:, 0], o[:, [4, 8, 12]], o[:, [1, 2, 3, 5, 6, 7, 9, 10, 11]].reshape(-1, 3, 3)


def gravity_frame(scene: str, t0: float) -> np.ndarray:
    """gravity_align.npy's matrix at the camera frame nearest the segment's first pose -- the
    rotation that takes this segment's odometry frame into the map's (module docstring)."""
    ts = np.loadtxt(LIPNIK_DIR / scene / "data_ts.txt")
    k = int(np.argmin(np.abs(ts - t0)))
    if abs(ts[k] - t0) > 0.2:
        raise SystemExit(f"scene {scene}: no camera frame within 0.2 s of the segment start (nearest {ts[k] - t0:+.2f} s)")
    return np.load(LIPNIK_DIR / scene / "gravity_align.npy")[k]


def check_pairing(pos: np.ndarray, pts: np.ndarray) -> tuple[float, float]:
    """(forward, reverse) coverage on a 0.5 m raster: share of poses with map data within 3 m,
    share of map points within 15 m of a pose. Both must be high for a real pairing -- forward
    alone is fooled by a bigger map of the same route, reverse alone by a bigger track."""
    r = 0.5

    def mask(xy: np.ndarray, rad: float, org: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        g = np.zeros(shape, bool)
        ix = np.floor((xy[:, 0] - org[0]) / r).astype(int)
        iy = np.floor((xy[:, 1] - org[1]) / r).astype(int)
        ok = (ix >= 0) & (ix < shape[1]) & (iy >= 0) & (iy < shape[0])
        g[iy[ok], ix[ok]] = True
        n = int(np.ceil(rad / r))
        out = np.zeros_like(g)
        for dy in range(-n, n + 1):  # square dilation, cheap and conservative enough for a check
            for dx in range(-n, n + 1):
                out |= np.roll(np.roll(g, dy, 0), dx, 1)
        return out

    allxy = np.concatenate([pos[:, :2], pts[:, :2]])
    org = allxy.min(0) - 20
    shape = (int((allxy[:, 1].max() - org[1]) / r) + 41, int((allxy[:, 0].max() - org[0]) / r) + 41)
    map_near = mask(pts[::5, :2], 3.0, org, shape)
    trk_near = mask(pos[::3, :2], 15.0, org, shape)

    def at(g: np.ndarray, xy: np.ndarray) -> float:
        ix = np.floor((xy[:, 0] - org[0]) / r).astype(int)
        iy = np.floor((xy[:, 1] - org[1]) / r).astype(int)
        return float(g[iy, ix].mean())

    return at(map_near, pos[:, :2]), at(trk_near, pts[:, :2])


def ground_z(T: Terrain, pos: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Chassis z that puts each wheel's tyre on the map under its centre, per pose; the odometry's
    own rotation is kept, so a wheel at (R @ w)_z above the chassis origin needs
    z = ground + WHEEL_RADIUS - (R @ w)_z. Mean over the wheels with map under them."""
    est = []
    for w in WHEEL_LOCAL.values():
        wl = np.asarray(w)
        wx = pos[:, 0] + R[:, 0, :] @ wl
        wy = pos[:, 1] + R[:, 1, :] @ wl
        est.append(terrain_height(T, wx, wy) + WHEEL_RADIUS - R[:, 2, :] @ wl)
    est = np.stack(est, 1)
    have = np.isfinite(est).any(1)
    z = np.full(len(pos), np.nan)
    z[have] = np.nanmean(est[have], axis=1)
    if not have.any():
        raise SystemExit("no pose has map under any wheel -- wrong map for this segment?")
    idx = np.arange(len(z))
    z = np.interp(idx, idx[have], z[have])  # holes: interpolate, hold at the ends
    k = 5  # moving average over ~0.35 s: the map is 0.1 m cells, the chassis shouldn't buzz on them
    zp = np.pad(z, k // 2, mode="edge")
    return np.convolve(zp, np.ones(k) / k, mode="valid"), have


def load_wheels(scene: str, t_ref: float) -> tuple[np.ndarray, np.ndarray]:
    names = (LIPNIK_DIR / scene / "joint_names.txt").read_text().split()
    cols = {n.replace("_wheel_j", ""): 1 + i for i, n in enumerate(names)}
    j = np.load(LIPNIK_DIR / scene / "joint_states.npy")
    theta = np.stack([j[:, cols[w]] for w in WHEEL_ORDER], 1)
    return j[:, 0] - t_ref, theta - theta[0]


def load_run(scene: str, segment: int, map_scene: str, fill: float) -> tuple[Run, Terrain, np.ndarray, tuple[float, float]]:
    t_abs, pos, R = load_odometry(scene, segment)
    G0 = gravity_frame(scene, t_abs[0])
    pos, R = pos @ G0.T, G0 @ R
    pts = load_points(map_scene, segment)
    cov = check_pairing(pos, pts)
    if min(cov) < MIN_COVERAGE:
        raise SystemExit(
            f"scene {scene} segment {segment} does not fit map {map_scene} seg{segment}: coverage "
            f"{cov[0]:.0%} of poses on map, {cov[1]:.0%} of map near track (< {MIN_COVERAGE:.0%}). "
            "Pass --map-scene if the file numbering differs."
        )
    terr = points_to_grid(pts)
    terr.H = close_gaps(terr.H, int(round(fill / CELL / 2)))
    z, have = ground_z(terr, pos, R)
    wt, wth = load_wheels(scene, t_abs[0])
    run = Run(t_abs - t_abs[0], np.column_stack([pos[:, :2], z]), mat_to_quat(R), wt, wth, pos[:, 2], have)
    return run, terr, pts, cov


def terrain_mesh(T: Terrain, stride: int):
    """Triangles over the cells that have data. Vertices sit at cell centres; a quad contributes
    two triangles only if all four corners are real, so holes stay holes."""
    import newton

    H = T.H[::stride, ::stride]
    ny, nx = H.shape
    valid = np.isfinite(H)
    quad = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, 1:] & valid[1:, :-1]
    vid = np.full(H.shape, -1, np.int64)
    vid[valid] = np.arange(valid.sum())
    # keep only vertices some quad uses, so isolated points don't pile up as unused vertices
    used = np.zeros(H.shape, bool)
    for di, dj in ((0, 0), (0, 1), (1, 1), (1, 0)):
        used[di : di + ny - 1, dj : dj + nx - 1] |= quad
    remap = np.full(H.shape, -1, np.int64)
    remap[used] = np.arange(used.sum())
    ii, jj = np.nonzero(used)
    xs = T.x0 + (jj * stride + 0.5) * CELL
    ys = T.y0 + (ii * stride + 0.5) * CELL
    verts = np.stack([xs, ys, H[ii, jj]], 1).astype(np.float32)
    qi, qj = np.nonzero(quad)
    a, b, c, d = remap[qi, qj], remap[qi, qj + 1], remap[qi + 1, qj + 1], remap[qi + 1, qj]
    tris = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)]).astype(np.int32)  # CCW, +Z normal
    return newton.Mesh(verts, tris.ravel(), compute_inertia=False, is_solid=False), len(verts), len(tris)


def interp(t: np.ndarray, v: np.ndarray, q: float) -> np.ndarray:
    q = min(max(q, t[0]), t[-1])
    k = min(max(int(np.searchsorted(t, q, side="right") - 1), 0), len(t) - 2)
    a = (q - t[k]) / max(t[k + 1] - t[k], 1e-9)
    return v[k] * (1 - a) + v[k + 1] * a


def summarize(scene: str, segment: int, map_scene: str, run: Run, terr: Terrain, cov: tuple[float, float]) -> None:
    H = terr.H
    print(f"scene {scene} segment {segment}  <->  map {map_scene} seg{segment}")
    print(f"  pairing coverage: {cov[0]:.0%} of poses within 3 m of map data, {cov[1]:.0%} of map within 15 m of track")
    print(f"  {len(run.t)} poses over {run.t[-1]:.0f} s, {len(run.wheel_t)} joint samples")
    print(f"  map {H.shape[1]}x{H.shape[0]} cells, {np.isfinite(H).mean():.0%} filled after closing gaps, z in [{np.nanmin(H):.2f}, {np.nanmax(H):.2f}] m")
    print(f"  wheels on map for {run.grounded.mean():.0%} of poses (rest: z interpolated)")
    dz = run.odo_z - run.pos[:, 2]
    print(f"  aligned odometry z minus wheel-grounded z: median {np.median(dz):+.3f} m, sd {np.std(dz):.3f} m (median: where the pose origin sits vs the wheel axles; sd: what re-grounding corrects)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="0003")
    ap.add_argument("--segment", type=int, default=1)
    ap.add_argument("--map-scene", default=None, help="map file number (default: MAP_FOR_SCENE)")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--fill", type=float, default=0.3, help="close terrain gaps up to this many metres")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--camera", choices=("follow", "free"), default="follow")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    map_scene = args.map_scene or MAP_FOR_SCENE.get(args.scene)
    if map_scene is None:
        raise SystemExit(f"no map known for scene {args.scene} (MAP_FOR_SCENE has {sorted(MAP_FOR_SCENE)}); pass --map-scene")
    run, terr, pts, cov = load_run(args.scene, args.segment, map_scene, args.fill)
    summarize(args.scene, args.segment, map_scene, run, terr, cov)
    if args.dry_run:
        return

    import newton
    import warp as wp
    from examples.helhest_junior.common import create_helhest_junior_model
    from ostrich.core.model_builder import OstrichModelBuilder

    mesh, nv, nt = terrain_mesh(terr, args.stride)
    print(f"  terrain mesh: {nv} vertices, {nt} triangles")

    wp.init()
    builder = OstrichModelBuilder()  # registers joint_dof_mode, which the wheel joints use
    builder.add_shape_mesh(body=-1, mesh=mesh, cfg=newton.ModelBuilder.ShapeConfig(mu=0.8))
    create_helhest_junior_model(builder, xform=wp.transform_identity())
    model = builder.finalize()

    viewer = newton.viewer.ViewerGL()
    viewer.set_model(model)
    state = model.state()
    q_start = model.joint_q_start.numpy()
    joint_q = model.joint_q.numpy().copy()
    joint_q_wp = wp.array(joint_q, dtype=wp.float32, device=model.device)
    joint_qd_wp = wp.zeros_like(model.joint_qd)
    base_q = q_start[0]
    wheel_q = [q_start[1 + i] for i in range(JOINTS_PER_ROBOT - 1)]  # left, right, rear

    # the whole driven path as a line, just above the ground it was driven on
    p = run.pos.copy()
    p[:, 2] -= WHEEL_RADIUS - 0.05
    viewer.log_lines(
        "path",
        wp.array(p[:-1], dtype=wp.vec3),
        wp.array(p[1:], dtype=wp.vec3),
        (1.0, 0.2, 0.2),
    )

    t_end = float(run.t[-1])
    print(f"replaying {t_end:.0f} s @ speed={args.speed}  (camera: {args.camera})")
    yaw_f = None
    last = time.perf_counter()
    t0 = last
    while viewer.is_running():
        now = time.perf_counter()
        real_t = (now - t0) * args.speed
        sim_t = (real_t % t_end) if args.loop else min(real_t, t_end)

        pos = interp(run.t, run.pos, sim_t)
        quat = interp(run.t, run.quat, sim_t)
        quat = quat / np.linalg.norm(quat)
        joint_q[base_q : base_q + 3] = pos
        joint_q[base_q + 3 : base_q + 7] = quat
        theta = interp(run.wheel_t, run.wheel_theta, sim_t)
        for q_i, ang in zip(wheel_q, theta):
            joint_q[q_i] = ang
        joint_q_wp.assign(joint_q)
        newton.eval_fk(model, joint_q_wp, joint_qd_wp, state)

        if args.camera == "follow":
            x, y, z, w = quat
            yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            if yaw_f is None:
                yaw_f = yaw
            else:
                a = 1 - np.exp(-(now - last) / CAMERA_YAW_TAU)
                yaw_f += a * np.arctan2(np.sin(yaw - yaw_f), np.cos(yaw - yaw_f))
            cam = wp.vec3(pos[0] - CAMERA_BACK * np.cos(yaw_f), pos[1] - CAMERA_BACK * np.sin(yaw_f), pos[2] + CAMERA_UP)
            viewer.set_camera(pos=cam, pitch=CAMERA_PITCH, yaw=float(np.degrees(yaw_f)))
        last = now

        viewer.begin_frame(sim_t)
        viewer.log_state(state)
        viewer.end_frame()
        wp.synchronize()


if __name__ == "__main__":
    main()
