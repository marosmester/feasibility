"""Learning glue shared between the two submodules' demos: pose-error metrics
(pose_error.py), PyTorch data loading over comparator/ output HDF5 (custom_dataset.py), two
generators for that HDF5 -- spawn-pose feature (generate_init_pose_dataset.py) and body-centered
terrain-patch feature (generate_dataset_body_centered_patch.py), sharing spawn sampling and the
simulator rollout via generate_dataset_utils.py -- a per-sample error viewer over one such file
(error_visual.py), the MLP that regresses those errors (model.py), and the trainer that fits it
(train.py)."""
