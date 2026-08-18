"""Single-run ostrich speed-bump demo: Helhest Junior driving straight into the
smallest bump in feasibility.heightmap.create_speed_bumps' series (0.10 m).

Recreates the ostrich half of feasibility.comparator.batch_compare for exactly one
bump height and one simulator, as a normal GL-viewer-or-headless demos/ script
instead of a batch/npz-comparison sweep -- useful for eyeballing a single case (e.g.
after changing create_speed_bumps.py's ramp geometry) without running the full
ostrich+helhest_stack sweep across BUMP_HEIGHTS. Spawn point, command schedule
(straight approach, no turning), and wheel-servo gain are imported from
batch_compare.py rather than re-derived, so this stays in lockstep with the batch
comparison's own ostrich case.

Usage:
    python demos/ostrich_speed_bump.py                     # GL viewer
    python demos/ostrich_speed_bump.py rendering=headless   # batch, no window
    python demos/ostrich_speed_bump.py +out=/tmp/run1.npz
    python demos/ostrich_speed_bump.py +mu=0.5               # override ground friction
    python demos/ostrich_speed_bump.py +bump=0.30            # another height in the series
    python demos/ostrich_speed_bump.py +terrain=mesh         # triangle mesh terrain
    python demos/ostrich_speed_bump.py +terrain=mesh +mesh_stride=4   # coarser mesh

`+terrain` selects the collision representation of the *same* surface: `heightfield`
(default, what batch_compare uses) or `mesh`, which triangulates the identical grid
and adds it via add_shape_mesh the way ostrich's examples/helhest/surface_drive.py
does its terrain -- an A/B to tell heightfield-collision artefacts apart from real
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

from feasibility.comparator.batch_compare import build_setpoints
from feasibility.comparator.batch_compare import K_P
from feasibility.comparator.batch_compare import SPAWN_X
from feasibility.comparator.batch_compare import SPAWN_Y
from feasibility.heightmap import HeightMapReader
from feasibility.heightmap.create_speed_bumps import BUMP_HEIGHTS
from feasibility.heightmap.create_speed_bumps import bump_path

CONFIG_PATH = pathlib.Path(examples.__file__).parent.joinpath("conf")

BUMP_HEIGHT = BUMP_HEIGHTS[1]  # smallest variant in the series -- 0.10 m


def yaw_from_quat_xyzw(q: np.ndarray) -> float:
    """Yaw (rotation about world +Z) from a single quaternion [qx,qy,qz,qw]."""
    qx, qy, qz, qw = q
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


class OstrichSpeedBumpSimulator(HelhestJuniorReplaySimulator):
    """Same robot/actuator/friction setup as HelhestJuniorReplaySimulator, spawned
    upstream of a speed-bump heightfield instead of the base class's bare ground +
    box obstacle -- the single-run counterpart to batch_compare's
    HelhestBatchSimulator, minus its multi-world CUDA-graph batching machinery."""

    def __init__(self, *args, terrain: HeightMapReader, terrain_repr: str = "heightfield",
                 mesh_stride: int = 1, **kwargs):
        self.terrain = terrain
        self.terrain_repr = terrain_repr
        self.mesh_stride = mesh_stride
        super().__init__(*args, **kwargs)

    @override
    def build_model(self) -> newton.Model:
        self.builder.rigid_gap = 0.10

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
                mesh=self.terrain.to_ostrich_mesh(stride=self.mesh_stride),
                cfg=newton.ModelBuilder.ShapeConfig(
                    density=0.0, mu=0.8, **self.ground_cfg_kwargs
                ),
            )
        else:
            heightfield, terrain_xform = self.terrain.to_ostrich()
            self.builder.add_shape_heightfield(
                xform=terrain_xform, heightfield=heightfield, cfg=ground_cfg
            )

        spawn_z = float(self.terrain.sample(SPAWN_X, SPAWN_Y)) + 0.5
        create_helhest_junior_model(
            self.builder,
            xform=wp.transform(wp.vec3(SPAWN_X, SPAWN_Y, spawn_z), wp.quat_identity()),
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

    mu = float(cfg.get("mu", 0.8))  # matches batch_compare's own default
    # "heightfield" (default) = add_shape_heightfield, as batch_compare does;
    # "mesh" = the same surface triangulated and fed through add_shape_mesh, to test
    # whether observed artefacts come from Newton's heightfield collision path.
    terrain_repr = str(cfg.get("terrain", "heightfield"))
    mesh_stride = int(cfg.get("mesh_stride", 1))

    # We drive our own step loop (see replay()/replay_graph() below) instead of
    # run()'s segment loop, so simulation.duration_seconds is unused -- DRIVE_S
    # (baked into build_setpoints, imported from batch_compare) sets the length.
    setpoints = build_setpoints(sim_config.target_timestep_seconds)

    bump_height = float(cfg.get("bump", BUMP_HEIGHT))
    terrain = HeightMapReader.load(bump_path(bump_height))

    sim = OstrichSpeedBumpSimulator(
        sim_config, render_config, engine_config, logging_config,
        k_p=K_P, mu_front=mu, mu_rear=mu, terrain=terrain,
        terrain_repr=terrain_repr, mesh_stride=mesh_stride,
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
        cfg.get("out", pathlib.Path(__file__).parent.parent / "outputs" / "ostrich_speed_bump.npz")
    )
    np.savez_compressed(
        out,
        dt=np.float32(dt),
        t=t,
        cmd_wheel_omega=setpoints,
        pose=poses,
        wheel_qd=wheel_qd,
        bump_height=np.float32(bump_height),
        mu=np.float32(mu),
    )
    terrain.save(out.with_suffix(""))
    print(f"saved {out}")


if __name__ == "__main__":
    ostrich_speed_bump()
