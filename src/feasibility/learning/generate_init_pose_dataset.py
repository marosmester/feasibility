"""Generates a trial-style ostrich-vs-helhest_stack comparison HDF5 for
feasibility.learning.custom_dataset.PoseErrorDataset -- N randomized (spawn pose, constant body
twist) initial conditions on ONE heightmap loaded via `+map=` (default: the fixed centered-box
terrain this used to hardcode via box_height), each replayed once in ostrich (dynamics) and once
in helhest_stack (kinematic twin), producing exactly the mapping custom_dataset.py expects:

    x = (v_drive, wz_drive, spawn_x, spawn_y, spawn_yaw)   [n, 5]
    y = (e_pos, e_rot)                                      [n, 2]  -- computed by custom_dataset
                                                                        from the saved poses, not
                                                                        stored here

See generate_dataset_utils.py for the shared spawn-pose sampling
(sample_dataset()/legal_spawn_poses()/continuous_spawn_poses()), map-path resolution
(resolve_map_path()), and the chunked ostrich/hstack batch rollout (simulate_dataset_rollout())
this module is built on -- this file only decides what gets written to the output HDF5's
`per_variant`/`root` fields, everything else is shared with the sibling generator,
generate_dataset_body_centered_patch.py (same spawn sampling and simulator rollout, but writing a
body-frame terrain patch instead of the raw spawn pose as the model's spatial feature -- see
terrain_patch.py).

Unlike comparator/compare_box_obstacles.py (one fixed spawn+twist, swept across a heightmap
SERIES) or compare_on_surface.py (a handful of hand-picked Trials on one terrain), this samples
MANY random (spawn, twist) trials on one terrain, and runs them as N robots in N replicated
ostrich worlds sharing that one terrain rather than rebuilding the model once per sample -- see
generate_dataset_utils.simulate_dataset_rollout()'s docstring for how. This is what makes N in
the hundreds-to-thousands practical; run_trial_comparison's one-build-per-trial loop would not be.

CLI parameters: see generate_dataset_utils.py's module docstring (+n_samples=, +seed=, +map=,
+duration_s=, +chunk=, +settle_steps=, +spawn_mode=, +mu=, +k_turn=, +device=, plus any standard
Hydra config-group override).

Usage:
    python src/feasibility/learning/generate_init_pose_dataset.py                       # DEFAULT_N, lattice
    python src/feasibility/learning/generate_init_pose_dataset.py +n_samples=2000 +seed=1
    python src/feasibility/learning/generate_init_pose_dataset.py +map=assets/speed_bumps/speed_bump_h010cm +chunk=64
    python src/feasibility/learning/generate_init_pose_dataset.py +spawn_mode=continuous +n_samples=5000
    python src/feasibility/learning/generate_init_pose_dataset.py +duration_s=1.2       # exact multiple of
                                                                                          # both sims' dt
"""
from __future__ import annotations

import hydra
import numpy as np
import warp as wp
from omegaconf import DictConfig

from feasibility.comparator.common import CONFIG_PATH
from feasibility.comparator.common import K_P
from feasibility.comparator.common import OUT_DIR
from feasibility.comparator.provenance import write_comparison
from feasibility.heightmap import HeightMapReader
from feasibility.learning.generate_dataset_utils import DEFAULT_DURATION_S
from feasibility.learning.generate_dataset_utils import DEFAULT_MAP
from feasibility.learning.generate_dataset_utils import DEFAULT_N
from feasibility.learning.generate_dataset_utils import DEFAULT_SEED
from feasibility.learning.generate_dataset_utils import DEFAULT_SPAWN_MODE
from feasibility.learning.generate_dataset_utils import resolve_map_path
from feasibility.learning.generate_dataset_utils import sample_dataset
from feasibility.learning.generate_dataset_utils import simulate_dataset_rollout
from feasibility.learning.generate_dataset_utils import SPAWN_MODE_TAGS


def generate(cfg: DictConfig) -> None:
    wp.init()

    n = int(cfg.get("n_samples", DEFAULT_N))
    seed = int(cfg.get("seed", DEFAULT_SEED))
    spawn_mode = str(cfg.get("spawn_mode", DEFAULT_SPAWN_MODE))

    terrain_path = resolve_map_path(str(cfg.get("map", DEFAULT_MAP)))
    terrain = HeightMapReader.load(terrain_path)
    print(f"[terrain]  {terrain_path}")

    spawn_pose, v_drive, wz_drive = sample_dataset(n, seed, spawn_mode)
    labels = np.array([f"s{i:05d}" for i in range(n)])
    print(f"[spawn]    mode={spawn_mode}, {len(np.unique(spawn_pose, axis=0))} distinct poses")

    ostrich_fields, hstack_fields, mu, k_turn = simulate_dataset_rollout(
        cfg, terrain, spawn_pose, v_drive, wz_drive
    )

    tag = SPAWN_MODE_TAGS[spawn_mode]
    out_path = OUT_DIR / f"dataset_{terrain_path.stem}_n{n}{tag}.h5"
    write_comparison(
        out_path,
        root=dict(
            n=n,
            variant_name="sample",
            obstacle_x=0.0,
            duration_s=float(cfg.get("duration_s", DEFAULT_DURATION_S)),
            mu=mu,
            k_turn=k_turn,
            k_p=K_P,
            spawn_mode=spawn_mode,  # so a saved file says how its poses were drawn, not just
            # what they were -- see comparator/provenance.py on self-describing runs
            map=str(terrain_path),  # so a saved file records which heightmap it was generated
            # on, alongside the terrain data itself in `terrain_entries` below
        ),
        per_variant=dict(
            # no single scalar parameter here -- sample index, purely so the field stays
            # populated for any generic consumer that expects it (see run_trial_comparison)
            variant_value=np.arange(n, dtype=np.float32),
            variant_label=labels,
            spawn_pose=spawn_pose.astype(np.float32),
            v_drive=v_drive,
            wz_drive=wz_drive,
        ),
        terrain_entries=[(terrain_path, terrain)] * n,
        ostrich=ostrich_fields,
        hstack=hstack_fields,
    )


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest", version_base=None)
def main(cfg: DictConfig) -> None:
    generate(cfg)


if __name__ == "__main__":
    main()
