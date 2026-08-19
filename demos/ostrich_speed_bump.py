"""Single-run ostrich speed-bump demo: Helhest Junior driving into one bump from
feasibility.heightmap.create_speed_bumps' series.

Recreates the ostrich half of feasibility.comparator.compare_speed_bumps for exactly
one bump height and one simulator, as a normal GL-viewer-or-headless demos/ script
instead of a batch comparison sweep -- useful for eyeballing a single case (e.g.
after changing create_speed_bumps.py's ramp geometry) without running the full
ostrich+helhest_stack sweep across BUMP_HEIGHTS. Spawn pose and command defaults come
from compare_speed_bumps.spec()'s ScenarioSpec, and the wheel-servo gain from
comparator.common, rather than being re-derived -- so this stays in lockstep with the
batch comparison's own ostrich case.

The robot is commanded with a body twist (v, wz) -- forward velocity and yaw rate --
via the PHASES table below, which comparator.common.cmd_to_wheels converts to the
per-wheel [left, right, rear] rad/s setpoints ostrich's joint servo actually consumes.
Edit PHASES to change the manoeuvre; it is the same (duration_s, v, wz) form
demos/ostrich_vel_cmd.py uses, so one row is a constant twist and several rows are a
time-varying schedule. The command lives in the script rather than on the CLI because a
schedule is a table, not a flag.

That conversion is the IDEAL no-slip differential drive, so a commanded wz is what a
perfectly gripping robot would achieve, not what this one will: ostrich under-rotates
through real skid, and that shortfall is the thing worth measuring (see
demos/ostrich_vel_cmd.py's alpha estimate).

Usage:
    python demos/ostrich_speed_bump.py                     # GL viewer
    python demos/ostrich_speed_bump.py rendering=headless   # batch, no window
    python demos/ostrich_speed_bump.py +out=/tmp/run1.h5
    python demos/ostrich_speed_bump.py +mu=0.5               # override ground friction
    python demos/ostrich_speed_bump.py +bump=0.30            # another height in the series
    python demos/ostrich_speed_bump.py +mesh_stride=4         # coarser mesh
    python demos/ostrich_speed_bump.py +mesh_max_rise=0.02    # finer ramp tiles
    python demos/ostrich_speed_bump.py +terrain=heightfield   # newton.Heightfield instead

`+terrain` selects the collision representation of the *same* surface: `mesh` (default),
which triangulates the grid and adds it via add_shape_mesh the way ostrich's
examples/helhest/surface_drive.py does its terrain, or `heightfield`, Newton's native
newton.Heightfield -- an A/B to tell heightfield-collision artefacts apart from real
dynamics.
"""
import math
import pathlib
from typing import override

import examples
import hydra
import newton
import numpy as np
import warp as wp
from examples.helhest_junior.replay_real import HelhestJuniorReplaySimulator
from ostrich import EngineConfig
from ostrich import LoggingConfig
from ostrich import RenderingConfig
from ostrich import SimulationConfig
from omegaconf import DictConfig

# Works both as `python -m demos.ostrich_speed_bump` (CWD on sys.path, `demos`
# resolves as a namespace package) and as `python demos/ostrich_speed_bump.py`
# (only `demos/` itself is on sys.path, so the package-qualified name is not
# importable) -- see root CLAUDE.md.
try:
    from demos.helhest_common import create_helhest_junior_model
except ModuleNotFoundError:
    from helhest_common import create_helhest_junior_model

from feasibility.comparator.common import cmd_to_wheels
from feasibility.comparator.common import K_P
from feasibility.comparator.compare_speed_bumps import spec
from feasibility.comparator.provenance import write_run
from feasibility.heightmap import HeightMapReader
from feasibility.heightmap.create_speed_bumps import BUMP_HEIGHTS
from feasibility.heightmap.create_speed_bumps import bump_path

CONFIG_PATH = pathlib.Path(examples.__file__).parent.joinpath("conf")

BUMP_HEIGHT = BUMP_HEIGHTS[4]  # smallest variant in the series -- 0.10 m

# The very ScenarioSpec that drives compare_speed_bumps.py -- the spawn pose is read off it
# rather than restated here, so this demo and the batch comparison can't drift into spawning
# the robot in different places.
SPEC = spec()

# --- Command schedule: (duration_s, v [m/s], omega [rad/s], CCW+) ---
# Same shape as demos/ostrich_vel_cmd.py's PHASES: edit this table to change what the robot
# does. ONE row is a constant twist; add rows for a time-varying schedule (e.g. approach the
# bump straight, then arc away once past it). The default single row reproduces
# compare_speed_bumps' straight approach, so out of the box this still spot-checks the batch
# comparison's ostrich case.
PHASES = [
    (SPEC.duration_s, SPEC.v_drive, SPEC.wz_drive),
]


def build_setpoints(dt: float) -> np.ndarray:
    """Sample PHASES onto the sim's dt grid. [T, 3] float32 in sim order [left, right, rear],
    dt-independent by construction so the same table replays identically at any dt.

    comparator.common.build_setpoints is the constant-twist equivalent -- it takes a single
    (v, wz) because a ScenarioSpec holds one; this one exists so the demo can express a
    schedule the batch comparison has no way to represent."""
    blocks = []
    for duration_s, v, wz in PHASES:
        n = int(round(duration_s / dt))
        blocks.append(np.tile(cmd_to_wheels(v, wz), (n, 1)))
    return np.concatenate(blocks, axis=0).astype(np.float32)


def yaw_from_quat_xyzw(q: np.ndarray) -> float:
    """Yaw (rotation about world +Z) from a single quaternion [qx,qy,qz,qw]."""
    qx, qy, qz, qw = q
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


class OstrichSpeedBumpSimulator(HelhestJuniorReplaySimulator):
    """Same robot/actuator/friction setup as HelhestJuniorReplaySimulator, spawned
    upstream of a speed-bump heightfield instead of the base class's bare ground +
    box obstacle -- the single-run counterpart to comparator.common's
    HelhestBatchSimulator, minus its multi-world CUDA-graph batching machinery."""

    def __init__(self, *args, terrain: HeightMapReader,
                 spawn_pose: tuple[float, float, float], terrain_repr: str = "mesh",
                 mesh_stride: int = 1, mesh_max_rise: float | None = None, **kwargs):
        self.terrain = terrain
        self.spawn_pose = spawn_pose
        self.terrain_repr = terrain_repr
        self.mesh_stride = mesh_stride
        self.mesh_max_rise = mesh_max_rise
        super().__init__(*args, **kwargs)

    @override
    def build_model(self) -> newton.Model:
        self.builder.rigid_gap = 0.20

        ground_cfg = newton.ModelBuilder.ShapeConfig(mu=0.8, **self.ground_cfg_kwargs)
        # Same physical surface either way (see HeightMapReader.to_ostrich_mesh) --
        # only the collision representation differs, which is the point of the switch.
        globals_builder = None
        if self.terrain_repr == "mesh":
            # Mesh goes in a separate builder so it gets shape_world=-1 (Newton's
            # "global" sentinel): stored once, broadphase-tested against every world
            # instead of duplicated per world. Same pattern as ostrich's
            # examples/helhest/surface_drive.py.
            globals_builder = newton.ModelBuilder()
            globals_builder.add_shape_mesh(
                body=-1,
                mesh=self.terrain.to_ostrich_mesh(
                    stride=self.mesh_stride, max_rise_per_tile=self.mesh_max_rise
                ),
                cfg=newton.ModelBuilder.ShapeConfig(
                    density=0.0, mu=0.8, **self.ground_cfg_kwargs
                ),
            )
        else:
            heightfield, terrain_xform = self.terrain.to_ostrich()
            self.builder.add_shape_heightfield(
                xform=terrain_xform, heightfield=heightfield, cfg=ground_cfg
            )

        spawn_x, spawn_y, spawn_yaw = self.spawn_pose
        spawn_z = float(self.terrain.sample(spawn_x, spawn_y)) + 0.5
        spawn_q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), spawn_yaw)
        create_helhest_junior_model(
            self.builder,
            xform=wp.transform(wp.vec3(spawn_x, spawn_y, spawn_z), spawn_q),
            control_mode=self.control_mode,
            k_p=self.k_p,
            k_d=self.k_d,
            friction_left_right=self.mu_front,
            friction_rear=self.mu_rear,
            mu_rolling=self.mu_rolling,
            ke=self.wheel_ke,
            kd=self.wheel_kd,
            kf=self.wheel_kf,
        )

        return self.builder.finalize_replicated(
            num_worlds=self.simulation_config.num_worlds, global_builder=globals_builder
        )


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest", version_base=None)
def ostrich_speed_bump(cfg: DictConfig):
    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)

    mu = float(cfg.get("mu", 0.8))  # matches comparator.common's own default
    # "mesh" (default) = the surface triangulated and fed through add_shape_mesh;
    # "heightfield" = add_shape_heightfield, Newton's native representation (what the
    # batch comparison used before it also switched to the mesh path) -- kept as an
    # opt-in A/B to tell heightfield-collision artefacts apart from real dynamics.
    terrain_repr = str(cfg.get("terrain", "mesh"))
    mesh_stride = int(cfg.get("mesh_stride", 1))
    mesh_max_rise_cfg = cfg.get("mesh_max_rise", None)
    mesh_max_rise = None if mesh_max_rise_cfg is None else float(mesh_max_rise_cfg)

    # We drive our own step loop (see replay()/replay_graph() below) instead of
    # run()'s segment loop, so simulation.duration_seconds is unused -- PHASES
    # sets the length.
    setpoints = build_setpoints(sim_config.target_timestep_seconds)

    bump_height = float(cfg.get("bump", BUMP_HEIGHT))
    terrain_asset = bump_path(bump_height)
    terrain = HeightMapReader.load(terrain_asset)

    sim = OstrichSpeedBumpSimulator(
        sim_config, render_config, engine_config, logging_config,
        k_p=K_P, mu_front=mu, mu_rear=mu, terrain=terrain,
        spawn_pose=(SPEC.spawn_x, SPEC.spawn_y, SPEC.spawn_yaw),
        terrain_repr=terrain_repr, mesh_stride=mesh_stride, mesh_max_rise=mesh_max_rise,
    )
    # replay_graph() captures the per-step physics into one CUDA graph and
    # replays it T times with no Python in the loop -- what
    # simulation.use_cuda_graph is actually asking for. replay() drives the same
    # steps from Python instead: no graph capture, plus a host<->device sync
    # every step (.numpy() reads), so it's much slower.
    if sim_config.use_cuda_graph:
        poses, wheel_qd = sim.replay_graph(setpoints)
    else:
        poses, wheel_qd = sim.replay(setpoints)

    if render_config.vis_type == "gl":
        # Hold the window open so the final pose is inspectable instead of the
        # sim exiting the instant the schedule ends.
        k = len(setpoints) - 1
        while sim.viewer.is_running():
            sim._maybe_render(k)

    dt = sim_config.target_timestep_seconds
    t = np.arange(len(setpoints), dtype=np.float32) * dt
    start_xy, final_xy = poses[0, :2], poses[-1, :2]
    peak_z = float(poses[:, 2].max())
    yaw_drift_deg = math.degrees(
        yaw_from_quat_xyzw(poses[-1, 3:7]) - yaw_from_quat_xyzw(poses[0, 3:7])
    )

    print(f"bump height   : {bump_height:.2f} m   mu={mu}")
    print(
        f"final endpoint: x={final_xy[0]:.2f} y={final_xy[1]:.2f}   "
        f"(started x={start_xy[0]:.2f} y={start_xy[1]:.2f})"
    )
    print(f"peak chassis z: {peak_z:.3f} m   yaw drift: {yaw_drift_deg:.2f} deg")

    out = pathlib.Path(
        cfg.get("out", pathlib.Path(__file__).parent.parent / "outputs" / "ostrich_speed_bump.h5")
    )
    write_run(
        out,
        attrs={"dt": float(dt), "bump_height": bump_height, "mu": mu},
        arrays={
            "t": t,
            "cmd_wheel_omega": setpoints,
            "pose": poses,
            "wheel_qd": wheel_qd,
            # The commanded twist schedule, not just the wheel setpoints it expands to -- so a
            # saved run says what was asked of the robot, not only what the servo was fed.
            "phases": np.array(PHASES, dtype=np.float32),
        },
        terrain=terrain,
        terrain_path=terrain_asset,
    )


if __name__ == "__main__":
    ostrich_speed_bump()
