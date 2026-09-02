"""Generates a trial-style ostrich-vs-helhest_stack comparison HDF5, like
generate_init_pose_dataset.py, but bakes the model's spatial input in as a body-frame terrain
patch (terrain_patch.sample_patches()) instead of the raw spawn pose -- so the patch is a feature
of the DATASET FILE, computed once here, rather than something custom_dataset.PoseErrorDataset
recomputes on every load:

    x = (v_drive, wz_drive) + flattened body-frame patch     [n, 2 + ny*nx]
    y = (e_pos, e_rot)                                        [n, 2]  -- computed by
                                                                          custom_dataset from the
                                                                          saved poses, not stored
                                                                          here

spawn_pose/v_drive/wz_drive are still recorded in the output file (per_variant, same as
generate_init_pose_dataset.py) -- not as a training feature, but for provenance and so
replay/gl_replay.py, replay/test_nn.py can still place the robot and re-derive a patch at
inference time against a DIFFERENT terrain (see terrain_patch.py's module docstring on why a
patch is meant to transfer).

See generate_dataset_utils.py for the shared spawn-pose sampling (sample_dataset()) and the
chunked ostrich/hstack batch rollout (simulate_dataset_rollout()) -- identical to
generate_init_pose_dataset.py, since both scripts run the exact same simulator; they only differ
in what non-simulator feature array ends up in the output file.

CLI parameters: everything generate_dataset_utils.py's module docstring documents (+n_samples=,
+seed=, +map=, +duration_s=, +chunk=, +settle_steps=, +spawn_mode=, +mu=, +k_turn=, +device=),
plus the patch geometry (see terrain_patch.py's PatchSpec for what each means):
    +patch_cell=FLOAT        patch resolution in meters (default: terrain_patch.DEFAULT_CELL)
    +patch_x_min=FLOAT       patch body-frame X min, meters (default: terrain_patch.DEFAULT_X_RANGE[0])
    +patch_x_max=FLOAT       patch body-frame X max, meters (default: terrain_patch.DEFAULT_X_RANGE[1])
    +patch_y_min=FLOAT       patch body-frame Y min, meters (default: terrain_patch.DEFAULT_Y_RANGE[0])
    +patch_y_max=FLOAT       patch body-frame Y max, meters (default: terrain_patch.DEFAULT_Y_RANGE[1])
    +patch_reference=STR     wheels|center|none (default: terrain_patch.DEFAULT_REFERENCE)

Usage:
    python src/feasibility/learning/generate_dataset_body_centered_patch.py                       # DEFAULT_N, lattice
    python src/feasibility/learning/generate_dataset_body_centered_patch.py +n_samples=2000 +seed=1
    python src/feasibility/learning/generate_dataset_body_centered_patch.py +map=assets/speed_bumps/speed_bump_h010cm +chunk=64
    python src/feasibility/learning/generate_dataset_body_centered_patch.py +spawn_mode=continuous +n_samples=5000
    python src/feasibility/learning/generate_dataset_body_centered_patch.py +patch_cell=0.5
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
from feasibility.learning.terrain_patch import DEFAULT_CELL
from feasibility.learning.terrain_patch import DEFAULT_REFERENCE
from feasibility.learning.terrain_patch import DEFAULT_X_RANGE
from feasibility.learning.terrain_patch import DEFAULT_Y_RANGE
from feasibility.learning.terrain_patch import patch_spec_to_attrs
from feasibility.learning.terrain_patch import PatchSpec
from feasibility.learning.terrain_patch import sample_patches


def generate(cfg: DictConfig) -> None:
    wp.init()

    n = int(cfg.get("n_samples", DEFAULT_N))
    seed = int(cfg.get("seed", DEFAULT_SEED))
    spawn_mode = str(cfg.get("spawn_mode", DEFAULT_SPAWN_MODE))

    patch_spec = PatchSpec(
        x_min=float(cfg.get("patch_x_min", DEFAULT_X_RANGE[0])),
        x_max=float(cfg.get("patch_x_max", DEFAULT_X_RANGE[1])),
        y_min=float(cfg.get("patch_y_min", DEFAULT_Y_RANGE[0])),
        y_max=float(cfg.get("patch_y_max", DEFAULT_Y_RANGE[1])),
        cell=float(cfg.get("patch_cell", DEFAULT_CELL)),
        reference=str(cfg.get("patch_reference", DEFAULT_REFERENCE)),
    )

    terrain_path = resolve_map_path(str(cfg.get("map", DEFAULT_MAP)))
    terrain = HeightMapReader.load(terrain_path)
    print(f"[terrain]  {terrain_path}")

    spawn_pose, v_drive, wz_drive = sample_dataset(n, seed, spawn_mode, terrain)
    labels = np.array([f"s{i:05d}" for i in range(n)])
    print(f"[spawn]    mode={spawn_mode}, {len(np.unique(spawn_pose, axis=0))} distinct poses")

    patch = sample_patches(terrain, spawn_pose, patch_spec).reshape(n, -1)  # [n, ny*nx]
    print(
        f"[patch]    {patch_spec.ny}x{patch_spec.nx} = {patch_spec.size} cells @ "
        f"{patch_spec.cell} m, x=[{patch_spec.x_min}, {patch_spec.x_max}], "
        f"y=[{patch_spec.y_min}, {patch_spec.y_max}], ref={patch_spec.reference}"
    )

    ostrich_fields, hstack_fields, mu, k_turn = simulate_dataset_rollout(
        cfg, terrain, spawn_pose, v_drive, wz_drive
    )

    tag = SPAWN_MODE_TAGS[spawn_mode]
    out_path = OUT_DIR / f"dataset_patch_{terrain_path.stem}_n{n}{tag}.h5"
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
            **patch_spec_to_attrs(patch_spec),  # so the file records the exact patch geometry
            # its `patch` dataset was sampled with -- see custom_dataset.PoseErrorDataset,
            # which reconstructs a PatchSpec from these attrs rather than taking one as an
            # argument.
        ),
        per_variant=dict(
            # no single scalar parameter here -- sample index, purely so the field stays
            # populated for any generic consumer that expects it (see run_trial_comparison)
            variant_value=np.arange(n, dtype=np.float32),
            variant_label=labels,
            # spawn_pose/v_drive/wz_drive are kept for provenance/replay even though the patch,
            # not the pose, is the training feature here -- see module docstring.
            spawn_pose=spawn_pose.astype(np.float32),
            v_drive=v_drive,
            wz_drive=wz_drive,
            patch=patch,
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
