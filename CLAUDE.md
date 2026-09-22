# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`feasibility` is an **umbrella repo with no source of its own** — it exists to pin two
independent research repos as git submodules so they can be worked on together:

| submodule | upstream | role |
|---|---|---|
| [ostrich/](ostrich/) | `aleskucera/ostrich` | GPU rigid-body simulator — **accurate but slow**. Non-smooth contact, exact friction cone, maximal coordinates, thousands of parallel worlds. Built on NVIDIA Warp + [Newton](https://github.com/newton-physics/newton) (itself a nested submodule at `ostrich/third_party/newton`). |
| [helhest_stack/](helhest_stack/) | `aleskucera/helhest_stack` | The on-robot GPU navigation stack — perception → localization → planning — around a **fast, differentiable, purely kinematic** twin of the robot (no dynamics). |

Research context (`context.txt`): ground wheeled robotics; the platform is **Helhest**, a
three-wheeled skid-steer robot that can flip itself upright. Ostrich is the precise/slow
simulator, the kinematic twin in `helhest_stack` is the fast/less-accurate one. Ostrich also
models **Marv** (four-flipper tracked robot) and other platforms.

**Each submodule has its own `CLAUDE.md` and they are binding when you work inside it** —
`helhest_stack/CLAUDE.md` in particular carries non-negotiable project rules (type hints
everywhere, black line-length 100 + reorder-python-imports, and **device-native Warp arrays
rather than numpy for any bulk/data-parallel stage**). Read it before editing that tree.

**`src/feasibility/` is an installed package** — root `pyproject.toml` has a
`[build-system]` (hatchling) packaging `src/feasibility`, so `import feasibility...` works from
the shared env like `ostrich`/`helhest`. It holds glue code shared between the two submodules'
demos that doesn't belong inside either one:

| subpackage | role |
|---|---|
| `heightmap/` | simulator-agnostic elevation-grid I/O (`HeightMapReader`, PNG+YAML); `.to_ostrich()`/`.to_hstack()` so both sims see bit-identical terrain. `create_speed_bumps.py` generates a swept series of bump-height heightmaps (`assets/speed_bumps/`). `create_box_obstacles.py` builds a box obstacle across `BOX_HEIGHTS`, three CLI modes: default — one random position anywhere its ramp fits inside `--extent`, shared across the whole height series (`assets/box_random/`); `--center` — origin-centered, same series (`assets/box_centered/`); `--batch` — `--n` (default 100) independent maps at one fixed `--height`, each its own random position (`assets/box_random/`, index-prefixed filenames), and `--n-boxes K` puts K independently-placed boxes on each such map, rejection-sampled to keep `--min-gap` (default 1.6 m, two signal-ring widths) between ramp footprints. K > 1 writes `box_random_k<K>_i*_h*cm` — the K tag precedes the index so the single-box glob `box_random_i*_h*` cannot match a multi-box map, and K=1 keeps the original un-tagged name so that series regenerates byte-identically. Multi-box maps exist because a grid-shaped label (see `grid_learning/`) is ~94% open flat ground with one box; K boxes multiply the divergence signal ~K-fold while making the map *cheaper* to simulate, and stay valid until obstacles cover half the map (~14 boxes) since spawn filters estimate the background as the median height. `build_multi_box` sums one `build_centered_box` layer per center — exact overlay, since box maps are relative bump layers zero outside their footprint, asserted by a no-overlap check on the summed max. A legacy off-center series (`assets/box/`) is no longer CLI-reachable but its constants/builder stay importable — `compare_box_obstacles.py` depends on them. `create_large_box_obstacles.py` generalizes the fixed 1.5m x 1.5m box for `grid_learning`, whose whole-map divergence-field labeling starves for signal on a single small box: obstacle dimensions are independently-sampled rectangles (0.5-6.0m per side, height fixed at 0.7m, one or several, allowed to touch/overlap), placed under a hard `--max-area-fraction` cap (default 0.5) so plenty of the map stays genuinely flat, and `--n-boxes` is only an UPPER bound — sizing stops once that area budget runs out, so a map can end up with fewer. Because every obstacle shares one fixed height, layers combine via an elementwise MAXIMUM rather than `build_multi_box`'s sum, so touching/overlapping footprints merge for free with none of `min_gap`'s non-overlap bookkeeping. Placement is chosen by a best-of-N random search (`--position-trials`) maximizing `accessible_frontier_score` — obstacle/flat-ground boundary cells within the spawn lattice's own reach (`DEFAULT_ACCESS_LIMIT`, deliberately == `grid_learning.generate_dataset.SPAWN_LIMIT`, re-derived rather than imported per that module's own independence stance) — so more of the 15x15 spawn lattice lands near a divergence-producing edge instead of on empty flat ground. Written to `assets/large_box_random/<seed>/large_box_i*_h070cm`, a filename prefix deliberately distinct from `box_random_*` so `generate_dataset.py`'s default glob never picks these up by accident. `create_pivot_pocket.py` writes ONE map whose only route out needs an in-place turn (`assets/pivot_pocket/`), for the pivot work: a dead-end pocket of TALL walls (0.70 m, which the settle does block) that the robot starts inside facing the closed end, with the goal outside behind it, at a clear width between the pivot's own swept radius (`rear_offset + wheel_radius` = 1.10 m, so 2.2 m) and a `min_turn_radius` U-turn's (3.2 m) -- so `pivot_cost` 0 finds no route at all and `pivot_cost` > 0 turns on the spot. Running the full length of the pocket, inside the rear wheel's pivot orbit, are LOW curbs (0.15 m) that the settle is blind to at every heading, so the forced point turn drags a wheel sideways through a vertical face -- the divergence helhest_stack cannot represent. The two heights are the design: the tall walls are respected, the low curb is not. `wheel_xy` (the three wheel centres at a body pose) backs its design asserts and `demos/view_planned_path.py`'s audit. `create_garage.py` writes the PAIR that `demos/garage_gate.py` compares: a U-shaped garage of ≥ 1 m walls with the robot parked inside facing a SIDE wall, empty (`garage_a`) and with one 0.20 m curb on the floor (`garage_b`) — identical but for that feature, so a difference in the plan is attributable to it. The curb is a rib across the floor parallel to the door plus a SPUR off it reaching back towards the robot, because a rib alone is reached only by the PLANNED turn: an in-place skid-steer turn drifts ~0.22 m in ostrich, so the wheel that the plan puts over the curb goes somewhere else and `demos/ostrich_follow_plan_turning.py` measures nothing. The spur exploits the fact that the wheels which pin the rib (the front pair, at the spawn) and the wheel that must reach it (the rear one, mid-turn) are in different places along y. The exit is a 90° point turn, and 90° is what forces the garage to be ASYMMETRIC: over that turn the wheel rims stay 0.71 m ahead of the reference point but 1.09 m behind it, while the tightest FORWARD 90° turn throws a rim 1.13 m sideways, so the faced wall has to sit in (0.71, 1.13) — above the 1.09 m the same turn needs behind, which a centred robot cannot satisfy at any width. `clearances()` measures all of that off `RobotParams` rather than hard-coding it. How far the curb may reach towards the robot is a WINDOW two cells wide, and `write_map` asserts both edges rather than arguing them. Below: the start footprint — the settle dilates the heightmap by `wheel_radius`, so a curb whose ramp foot is within 0.35 m of a start wheel CENTRE blocks the spawn (checked at the start yaw *and* at the bin centre the lattice floors it to). Above: the settle is blind to a TYRE hanging over the curb but not to a wheel CENTRE on it — a centre on it lifts the wheel the full height and the pose is blocked, at which point map B plans around it and stops being map A's plan over a second terrain. A third assert displaces the planned turn by the drift ostrich really has and requires a `CONTACT_MARGIN` of tyre still over the curb, which is the physics-side premise the geometry cannot otherwise state. The height is chosen so the REAR wheel on it pitches inside `max_pitch_down` while a front wheel would roll outside `max_roll` — the settle is blind to the one and blocks the other; at 0.20 m that is 14.9° against a 15° envelope, and it is not the margin it looks like (the blocked-centre case above is the same knife edge). The premise is a reachability claim, not a per-pose one, and `reach_closure` walks the lattice three times to check it: forward arcs alone must NOT get out, point turns must, and point turns restricted to the curb-free ones must not again. That third run is what caught the first draft — turns near the door are genuinely curb-free and the robot could shuffle to them, which only `WALL_GAP` closes. `create_uphill_series.py` writes `benchmarks/bench_uphill_nn.py`'s maps (`assets/uphill_series/uphill_a*`, 5–75°): a full-width 0.75 m face with NO descent, crest pinned at x = 0 so start (−8.8, 0) and goal (2.0, 0, on the plateau, 2 m past the crest, 2 m more plateau beyond for the patch's forward reach) are shared by the whole series; geometry constants from `helhest.planning.rampmaps`. `create_surface.py` rasterizes ostrich's Blender "Landscape" mesh into one hilly terrain (`assets/surface/`). `create_rough_terrain.py` synthesizes band-limited random rough ground via spectral synthesis (`assets/rough/`). `mend_two_meshes.py` adds two heightmaps together — e.g. a box obstacle riding on rough terrain (`assets/mended/`). `create_maps_for_lattice_learning.py` is the orchestrator for `lattice_learning/`'s maps: a CONFIG block (ratios, default 1/4 each, `--ratios` overrides) mixes four categories — `ramps` (`create_ramps.py`: closed-form finite-width ramps, always 0.7 m tall, rising face uniform 5–80° from gentle to wall-like, sharp foot/top kinks, side drop-offs, plateau ≥ 2.1 m so the whole robot can stand on it before a `ramp_down` trial (`spawn_sampling.required_platform_length`), on 14 m maps — a per-category CONFIG `extent`, the others stay 12 m, `--extent` overrides all; each ramp's shape is drawn once and only its placement retried, so long gentle ramps are not under-represented — asserted by a KS test — and footprints never touch), `curbs_and_walls` (`create_curbs_and_walls.py`: rotated curbs, thin walls ≥ 0.15 m so the 0.125 m patch always sees them, L-corners and boxes with both sides ≥ 1.5 m so the robot fits on top, each 0.2–1.0 m tall with 80° sides; exactly two per map, footprints kept > 1.0 m apart) `poles_and_walls` (`create_poles_and_walls.py`: the sparse, narrow counterpart — up to two features per map, each a pole (a thin vertical cylinder, radius ≥ the 0.125 m patch cell's half-diagonal so a sample always lands on its top) or one thin wall (≥ 0.15 m thick, 1.5–5 m long), 0.1–1.0 m tall with 80° sides, footprints > 1.0 m apart, ≤ 30% of the map's area) and `rough` (`build_rough_terrain`, some exactly flat) — into ONE flat dir `assets/lattice_maps/<seed>/<category>_i<NNNN>` + `manifest.yaml`, which a dataset config's `maps.dir` points at (`dataset_config.check_maps` refuses a dir for `ramp_down` unless every sidecar ramp plateau clears `spawn_sampling.required_platform_length` — a dir generated before that floor has to be regenerated). Category ids 0 (`boxes`) and 1 (`walls`, the old 0.05–0.5 m series) are retired, so a dir generated before that still carries them. The helpers are pure builders returning `(HeightMapReader, params)`; all IO is in the orchestrator. Heights with different values merge by elementwise max (union of solids). Per-map seeding `SeedSequence([seed, category_id, index])` keeps existing maps byte-identical when ratios or `--n` change; it refuses to write into a dir holding PNGs it would not produce. `build_rect_obstacle` gained `yaw` (0 default, bit-identical). `lattice_learning/generate_dataset.py` takes ONE argument, a dataset config (a name in `lattice_learning/configs/` or a path; `configs/default.yaml` documents every key, all required, no code defaults) — seed, map dir/counts, warm-up, friction, jitter, chunking, device, `dry_run`, `ostrich_overrides` (Hydra overrides composed via the compose API, no `@hydra.main`), and the sampling `mix`: entries `{map: <sidecar category>, strategy, percent, params}` summing to 100. `lattice_learning/dataset_config.py` validates the file (unknown/missing keys, strategy eligible for the category via the `STRATEGIES` registry, params ranges, duplicate pairs), checks the map dir (`check_maps`), and allocates: category map counts by largest remainder of its summed percent, entries on one category SHARE its maps and are dealt round-robin so every map has exactly `trials_per_map` rows; the h5 embeds the file as `config_yaml` and stores `sampling`/`map_category`/`targeted` per row. Strategies (`spawn_sampling.py`): every trial needs helhest_stack's static settle feasible at the spawn (so trials start on ramps, box tops and rough ground too); interaction is plane-relative relief under the wheels/body > `trial.interact_relief` (0.05 m), so a uniform slope doesn't count, and its sign gives `interact_dir` (+1 climbing up, −1 driving down). `ramp_up` / `ramp_down` (ramps) → `sample_ramp_up_trials` / `sample_ramp_down_trials`, one face sampler mirrored: using the sidecar's `ramps` geometry every trial drives head-on along a face (rising face or far side, 80° drops included) — up from before its foot, or down from its crest with the robot standing on the plateau — 75% straight, mid-arc heading within ±5° of the face axis, wheels kept on that ramp's own rasterised surface; rows store `ramp_deg`/`ramp_s` (position onto the face from its entry point, NaN elsewhere). `edge` (curbs_and_walls, retired walls/boxes) → `sample_edge_trials`: arc origins within `band` (1.5 m) of a height edge found from the heightmap alone (distance transform), mostly heading at it, exactly `interact_frac` interacting, `down_frac` of those driving down. `uniform` / `targeted` (any category) → `sample_trials`. `rotate_in_place` (curbs_and_walls, poles_and_walls) → `sample_rotate_in_place_trials`: helhest_stack's PIVOT primitive instead of an arc — `v = 0, wz = ±arc.OMEGA_NOM` (one heading bin, `arc.PIVOT_ANGLE` = 15°, in the same 0.5 s; warm-up spins too), placed beside a pole/wall so that no wheel CYLINDER (r 0.35, 0.10 m tread — contact points never reach the feature in a 15° pivot) touches it during the warm-up and exactly `interact_frac` sweep a wheel into it in the recorded window; `kappa` is NaN on those rows, so `SpawnBatch`/the h5 carry the per-row twist (`v`/`wz`, `v_drive`/`wz_drive`), `arc.integrate_twist` gives every primitive's end, and such files train only with `--command-mode v_wz` (`custom_dataset` refuses kappa mode); `configs/pivot.yaml` is the worked example (its `maps.dir` needs a `poles_and_walls` category, which `check_maps` reports if missing). `ramp_*`/`edge`/`rotate_in_place` do NOT require the nominal arc end to be settle-feasible (they deliberately drive into walls); `uniform`/`targeted` do. `generate_dataset.py`'s `valid` no longer includes the arc-end settle at all — it is stored per row as `endpoint_blocked` (root attr `valid_excludes_endpoint_settle`), and `custom_dataset.py`'s `drop_blocked_endpoints` / `train.py --drop-blocked-endpoints` removes those rows. `run.maps_per_build: K` (1 = one ostrich build per map) runs K maps in one build: `lattice_learning/tiled_terrain.py` merges their meshes into one global mesh, each shifted to its own tile ≥ 5 m apart, spawns are offset and logged poses shifted back, and `comparator/provenance.terrain_fields` NaN-pads mixed grid shapes with a per-terrain `shape` row; the rng is consumed identically (jitter is drawn in `prepare_map`, before the rollout), so K only changes ostrich's run-to-run noise, not the trials |
| `comparator/` | `common.py` — the scenario-independent Hydra-driven core (config path anchored under `ostrich/examples/`) that spawns the robot at a pose and holds a constant commanded body twist `(v_drive, wz_drive)` once in ostrich (dynamics) and once in helhest_stack (kinematic twin), saving poses/wheel velocities/terrain/git-provenance to `outputs/compare_<name>.h5`. Two ways to batch a run, sharing that same per-trial (spawn pose, twist) → (ostrich pose, hstack pose) contract: a `ScenarioSpec` holds spawn+twist fixed and sweeps *terrain* across a `variants` height series (`compare_speed_bumps.py` drives straight over the bump — vertical-clearance divergence; `compare_box_obstacles.py` turns in place beside the box so the rear wheel sweeps into its side — lateral-collision divergence helhest_stack can't represent at all — each reading its height series from `heightmap.create_speed_bumps`/`create_box_obstacles`); a `TrialScenarioSpec` holds one terrain fixed and sweeps a list of `Trial`s (spawn pose + twist) instead, via `run_trial_comparison` (`compare_on_surface.py` runs five hand-picked trials — flat baseline, uphill, downhill, contour traverse, in-place turn on a slope — over one fixed hilly terrain from `heightmap.create_surface`). `provenance.py` is the HDF5 schema (embeds git SHA/dirty state of both submodules so a saved run is self-describing), grouped/attrs-based like ostrich's own `ostrich.logging` loggers |
| `plotting/` | `batch_comparator_viewer.py` — matplotlib 3D terrain+trajectory and 2D wheel-velocity viewer for one saved variant; `dataset_patch_viewer.py` — plots one `generate_dataset_body_centered_patch.py` sample's baked-in body-frame terrain patch alongside its (v, wz) command and a recomputed ostrich-vs-hstack pose error; `terrain_visualizer.py` — Newton GL viewer for a saved heightmap with a static Helhest Junior model parked at its center for scale, no physics; `terrain_browser.py <maps_dir>` — the same scene for every `*.png` in a directory, RIGHT/N and LEFT/P cycle maps (model rebuilt and re-`set_model`'d in the render loop, camera kept across switches) |
| `replay/` | `gl_replay.py` — Newton `ViewerGL` real-time playback of a saved trajectory pair on the real Helhest Junior mesh (pose-only, no physics stepping); `test_nn.py` — same playback plus a loaded `learning.model.PoseErrorMLP` checkpoint, printing the real final-pose (e_pos, e_rot) alongside the network's predicted one for the same variant |
| `learning/` | Learns to predict ostrich-vs-hstack final-pose divergence from the commanded twist plus **either** the spawn pose **or** a robot-centric terrain patch (`terrain_patch.py`, `train.py --patch`) — the pose is a coordinate code that only means anything on the one heightmap it was fitted to, the patch is terrain geometry that transfers. Two Hydra generator scripts, sharing spawn-pose sampling and the chunked ostrich/hstack batch rollout via `generate_dataset_utils.py`: `generate_init_pose_dataset.py` samples N random (spawn pose, twist) trials on one fixed centered-box heightmap and writes `spawn_pose`/`v_drive`/`wz_drive` per row (`x = (v, wz, spawn_x, spawn_y, spawn_yaw)`); `generate_dataset_body_centered_patch.py` writes the same rows plus a baked-in `patch` dataset (`terrain_patch.sample_patches()`, computed once at generation time — `x = (v, wz) + flattened patch`). Both replay each trial once in ostrich/once in hstack as N parallel worlds (`comparator.common.HelhestBatchSimulator`/`run_hstack_batch` take a per-world `[N, 3]` spawn pose), writing `outputs/dataset_*.h5`/`outputs/dataset_patch_*.h5` in the same schema `comparator/provenance.py` uses; `pose_error.py` — the (e_pos, e_rot) SE(3)-error metric shared by every consumer below; `custom_dataset.py` — `PoseErrorDataset`/`make_dataloaders`, a torch `Dataset` over that schema (also loads `ScenarioSpec`-style sweep files), dropping NaN rows, reading a file's baked-in `patch` dataset directly when `use_patch=True`; `terrain_patch.py` — `PatchSpec`/`sample_patches`, the body-frame elevation patch (yaw-aligned so `yaw` drops out, heights relative to the wheel contacts and scaled by `WHEEL_RADIUS` so box height generalizes, forward-biased since `V_RANGE` is non-negative), plus `patch_spec_to_attrs`/`patch_spec_from_attrs` to round-trip a `PatchSpec` through a dataset file's flat root attrs; `model.py` — `PoseErrorMLP`, one shared trunk with two heads (e_pos, e_rot correlate 0.72–0.83) regressing in log1p/standardized space via `TargetTransform`; `train.py` — the training loop (target transform fit on train rows only, loss computed in model space so `TargetTransform`'s clamp never kills a gradient), logging to Weights & Biases and writing a self-describing checkpoint to `outputs/checkpoints/`; `error_visual.py` — plots a dataset's per-sample (e_pos, e_rot) against the input features that produced them |
| `grid_learning/` | New and still growing (branch `experiment/grid-learning`) — fixed-resolution heightmap → torch tensor prep, for feeding a saved heightmap into an NN/CNN. `utils.py`: `heightmap_to_tensor`/`load_heightmap_tensor` resample a `HeightMapReader` (bilinear) onto a `resolution`-m grid (default 0.10 m) spanning an `extent`-m square centered on the world origin (default 10.0 m → 100×100), same cell-center convention as `HeightMapReader.H`/`learning.terrain_patch.PatchSpec`. `generate_dataset.py` is the CNN reframing of `learning/`'s per-trial datasets: one sample is a whole map plus one commanded yaw rate (`x = (heightmap [100,100], wz)`), and the label is a divergence **field** — the ostrich/hstack final-pose pair at every cell of a 15×15 spawn lattice (`y = [15,15,14]`, `mask = [15,15]`). The robot spawns yaw-locked at 0 on a 0.5 m lattice within ±3.5 m (1.5 m of padding, sized so a turn in place keeps the rear wheel's ~1.45 m rim reach on mapped terrain — `HeightMapReader.sample` clamps outside the grid rather than raising, so under-padding fabricates flat ground silently) and is commanded `v=0, wz~U(-1,1)`, i.e. `compare_box_obstacles.py`'s lateral-collision scenario swept over the whole map. Masked cells (spawn footprint on the obstacle, or a diverged solve) carry **zeros, not NaN** — a NaN target poisons the backward pass even through a correctly masked loss, since `0 * NaN` is NaN. **Deliberately does not import from `learning/`** (the two trees stay independent): wheel geometry is re-derived from `HelhestJuniorConfig`, the on-obstacle spawn filter re-implemented locally; `comparator`/`heightmap` are shared infrastructure and are imported. Because `HelhestBatchSimulator` bakes the terrain into the model's globals builder, a chunk of parallel worlds can never span two heightmaps — so the map loop is the outer one, and within a map a chunk freely mixes lattice cells *and* yaw rates. Writes `outputs/dataset_grid_*.h5` (own writer — final poses only, no time series — reusing `provenance.terrain_fields`/`git_provenance` so provenance can't drift). **Plain importable package** (`feasibility.grid_learning`, `__init__.py` present) — `model.py` imports `custom_dataset.py`/`utils.py` siblings via ordinary `from feasibility.grid_learning.x import y`, and every module still also runs standalone as a script (`python src/feasibility/grid_learning/utils.py <path>`). `gl_replay_grid.py` is the QA viewer for a `dataset_grid_*.h5`'s output: unlike `replay/gl_replay.py` there is no trajectory to interpolate (only one final-pose keyframe per lattice cell), so it steps through a row's G x G cells instead, freezing the real ostrich/hstack meshes at each cell's stored final pose and drawing the whole lattice as static colored spheres (green = valid, gray = obstacle-blocked, red = footprint-clear but diverged) so a masked cell's reason is visible at a glance; `--cell I J` freezes on one trial for close inspection, `--dry-run` validates a row (including a local `footprint_clear()` recompute cross-checked against the file's own `mask`) without opening the viewer. Deliberately re-derives its footprint filter and `pose_error`-style SE(3) error formula locally rather than importing them (from `generate_dataset.py`/`learning/pose_error.py` respectively), same independence stance as the rest of this directory. `remask_dataset.py` re-applies `generate_dataset.py`'s per-trial `MAX_SPAWN_DISPLACEMENT` filter (both sims' final pose must stay within 1.0 m of the cell's own spawn — `V_DRIVE` is 0, so a turn in place has no commanded translation and ostrich's real displacement is median 0.11 m / q90 0.20 m, while a 2.75% tail runs to 8.6 m and carried ~90% of the e_pos label variance) to an ALREADY GENERATED file, no re-simulation: the check needs only the stored `y` and `spawn_xy`, so the mask can be recomputed post hoc, and re-running `train.py` alone would change nothing since `GridPoseErrorDataset` reads the stored mask verbatim. Copy-then-mutate via `shutil.copy2` + h5py `"r+"` rather than a second writer, so terrain grids and the physics run's git provenance survive byte-for-byte; three `remask_*` root attrs record what was done. The mask only ever tightens (idempotent), and the `mask &` guard is load-bearing — an already-masked cell holds the zero fill, whose "displacement" reads as |spawn_xy|. This one DOES import its threshold from `generate_dataset.py` rather than re-deriving it: the independence stance separates independent trees, but this tool exists to reproduce that module's decision on old files, so a drifted copy would silently mint inconsistent datasets. `--self-test` is its smoke test, `--dry-run` reports the displacement percentiles and prints ready-to-paste `gl_replay_grid.py --cell I J` commands for the worst dropped cells (the 1.0 m threshold is a judgement call — eyeball a few before committing to it) |
| `planning/` | Lattice planning + path following, shared by `benchmarks/` and `demos/` (neither imports the other's code for it). `gated_lattice.py` — helhest_stack's `CostToGo` (`make_cost_to_go`, n_theta 24, 0.3 m arcs, `pivot_cost` 0 by default — > 0 appends helhest_stack's two in-place point turns, so `solver.n_prim` goes 5 (`N_PRIM_ARC`) -> 7 and a goal behind the robot routes as pivot-then-drive instead of a loop it may have no room for; the lattice pivot IS one `lattice_learning` pivot trial, since n_theta 24 makes a bin `arc.PIVOT_ANGLE`. `pivot_cost_of` reads it back off a built solver so `build_gated_solver` still reconstructs the lattice from the `CostToGo` alone) with a per-ARC gate: `EdgeGatedLatticeSolver` is a local copy of the relax kernel that also prunes primitive p whose predicted error > `tau[p]` (helhest_stack untouched), plus `lattice_state`/`trace_states`/`arm_result` (trace the solved policy, audit the arcs taken: `max_err`); its `__main__` asserts an open gate == the stock solver bit for bit. The threshold is PER PRIMITIVE — `primitive_taus` broadcasts a scalar, so every one-number caller is unchanged, but a planner can gate the two point turns at a finite tau and leave the five forward arcs at inf, which is what `nn-gated-pivot` does and what the `__main__` pivot check covers. `arc_network.py` — per-arc error fields `[ny, nx, n_theta, n_prim]` from a `lattice_learning` `ArcDivergenceNet` at exactly the settle's lattice poses, with a checked row-invariance shortcut for CPU inference. One `_error_fields` body, two command tables: `arc_error_fields` passes `arc.primitive_kappas` (5 forward arcs, and `load_network` refuses a pivot lattice — a pivot has no curvature to index by) and `vwz_error_fields` passes `primitive_commands` (every primitive, point turns included). The field is the expensive object — ny·nx·n_theta poses, ~120k on a 7 m map at 0.1 m, ~17 s on a GTX 1650 and minutes on CPU — but a gate needs it, since value iteration must know the error of arcs the final path never takes. `planners.py` — `PLANNERS` (vanilla-off/on, nn-gated-pos/pitch/rot/fused, nn-report, nn-gated-pivot), every threshold and default checkpoint (`TAU_POS`/`TAU_PITCH` tau*-calibrated in `bench_uphill_nn.py`, `TAU_ROT`, `TAU_FUSED_POS`/`TAU_FUSED_ROT`, `TAU_PIVOT_PITCH` set to the middle of the window that splits `demos/garage_gate.py`'s A/B, cross-checked against the v_wz checkpoint's own held-out pivots by `tune_pivot_tau.py` — which must be re-run after EVERY retrain, since a threshold is a property of weights and training the same config twice moves individual predictions enough to flip a gate decision near the line; the constant's comment records every value it has held and why, so the drift is visible instead of silent), `PlannerConfig` (field names = the demos' CLI dests, `from_args`), `PlanContext` (reuses CostToGo/solver per grid, networks, kappa fields and the v_wz field per map) and `plan_path`. `PlannerConfig.pivot_cost` > 0 rules the four kappa-gated planners out (`plan_path` raises rather than mis-index a curvature table with a pivot) and is REQUIRED by `nn-gated-pivot`, which is the other way round. That planner is also the only gated one that keeps the settle's `blocked` instead of zeroing it: the others were built to ask whether the net can REPLACE the settle, this one adds to it — the settle stops the walls, the net adds the low curb it tolerates. `tune_pivot_tau.py` — re-derives `TAU_PIVOT_PITCH` for a given v_wz checkpoint: the PIVOT rows (`v_drive` 0) of that checkpoint's own held-out maps, rebuilt from its stored `args` (val_frac/seed) rather than from defaults, labelled BAD by TRUE e_pitch above a physical tolerance (5°, a judgement about the robot that does not move when the net does) and thresholded by Youden's J, with F1/mid-gap and a bootstrap band beside it because the J curve is a plateau and a few hundred pivots is not many. Prints the whole TPR/FPR curve, so the operating point can be moved DOWN inside the band deliberately when a missed point turn costs more than a longer path. Never tuned on a demo's map. `pure_pursuit.py` — `PurePursuitSimulator`, a `HelhestBatchSimulator` whose control is ONE pure-pursuit Warp kernel inside the captured step, world w following its own path out of a flat waypoint buffer (same path in every world for the single demo); `KAPPA_GAIN`/`STOP_RADIUS`/`GOAL_TOLERANCE` and their rationale live in its docstring. `pivot_pursuit.py` — the follower for a plan that TURNS IN PLACE, which that one structurally cannot drive: its `v` is a scalar fixed at construction and `wz = v * kappa`, so `v = 0` means `wz = 0`, its waypoints are `wp.vec2` with no yaw in them, and `resample_polyline` DELETES pivots (consecutive pivot poses share a cell exactly, and repeated vertices are dropped so `np.interp` has a monotonic arc length). A SECOND kernel rather than a flag on that one, so every existing caller is untouched; everything shared is imported from it, `evaluation.judge` included, so the two cannot drift. `plan_phases` turns a `plan_path` result into the phase list the kernel walks — one PIVOT phase per pivot PRIMITIVE (`prims[i] >= N_PRIM_ARC`, which unlike comparing positions also gives the direction), so each 15° bin re-measures the real yaw against an absolute target and the tolerance cannot accumulate over a six-bin turn, and one DRIVE phase per maximal run of forward arcs, running `pure_pursuit`'s law verbatim over its own slice of the buffer. A pivot-free plan yields exactly one DRIVE phase over every waypoint — the reduction to plain pure pursuit, which the `__main__` asserts. A PIVOT phase is CLOSED-LOOP on the measured heading because an in-place skid-steer turn is ~2.2x slower than its no-slip command (measured by that same `__main__` on flat ground: 6.5 s for a 90° turn commanded at `arc.OMEGA_NOM` for 3.0 s, landing within 1.7°), so an open-loop `omega x duration` would simply under-turn — the 1.6x in `demos/ostrich_vel_cmd_90deg.py` is for a DRIVING arc and is a floor. The per-phase timeout is load-bearing, not defensive: a wheel jammed against a curb never reaches `yaw_tol`, and that is the case these plans are driven to investigate, so it is recorded (`timeouts()`) rather than left spinning. `rollout` also returns a per-step `phase` log, which is what lets a caller isolate the turn window in the pose log. Targets are the lattice's own BIN CENTRES, and `lattice_state` FLOORS into a bin, so a start yaw of exactly 90° is planned for at 97.5° — a fixed offset, not an accumulating one; `anchor` trades driving the poses the settle and the net actually judged for turning the planned number of degrees. `evaluation.py` — `judge` (arrived/flipped/off_map/stalled/nonfinite + climb, roll, cross-track error; arrival = the controller's own stop condition), `track_error`, `pitch_roll`; `__main__` is a GPU-free smoke test |
| `submodule_test/` | Regression harness for **submodule bumps** (`git submodule update --remote` can land 50+ ostrich commits at once). Replays a fixed scenario set in both simulators and compares against a committed baseline — a *characterization* test: the baseline has no notion of correct, it records what the two sims did on the day it was written, so a reported change is a prompt to look and accepting one is a deliberate `+update=true` commit. `scenarios.py` — three terrains (flat / centered box / band-limited rough) x a few (spawn, twist) `Trial`s, in `smoke` and `full` tiers. Terrain is BUILT IN MEMORY (`HeightMapReader.flat`, `build_centered_box`, `build_rough_terrain`), never loaded from `assets/` — that directory is gitignored, so an asset-backed baseline could not be reproduced from a fresh clone; `path_bbox` asserts each trial's commanded arc stays on mapped terrain, since `sample` clamps rather than raising. `metrics.py` — pure numpy, no GPU: 25 named behaviour metrics (`METRIC_NAMES` is the baseline's append-only column contract) plus health FLAGS (`nonfinite`, `out_of_bounds`, `displaced`, `speed_spike`, `hstack_residual`) that fail regardless of the baseline — there is no way to accept a NaN. `run_check.py` — the Hydra entry point; runs each scenario's trials as N parallel worlds (one terrain, one model build, per `generate_dataset.py:simulate_map`) and also writes `outputs/submodule_test_<scenario>.h5` in `comparator/provenance.py`'s schema, so anything the report flags is one `replay/gl_replay.py` command from being watched. **The two sims get unequal authority, by measurement**: helhest_stack is bit-exact across identical runs (every `h_*` spread exactly 0.0) and is asserted tightly, while ostrich is intrinsically nondeterministic — 12.7–31.2 deg of net-yaw scatter across repeats, confirmed NOT a batching artifact (a `num_worlds=1` rerun is equally non-reproducible) — because pivoting on near-tangential contacts is a real bifurcation the Newton solve amplifies. So `+update` runs several repeats, stores the measured noise pooled per scenario, and cells whose tolerance is set by that scatter are REPORTED, NOT ASSERTED: a verdict resting on them is a coin flip, and a randomly-failing test gets ignored. A green run therefore means nothing blew up, helhest_stack is unchanged, and ostrich did not change grossly — read the "moved" lines for the rest. Detection power verified by running `+mu=0.6` against a `mu=0.8` baseline: caught on the hstack side at 0.28 m of final-x against a ±0.008 m tolerance |

`assets/` and `outputs/` are gitignored, so nothing in this file may depend on what happens to
be in them. An `assets/...` path above is only a generator script's default output — a fact
about the code, reproducible by running that script. Never record a specific seeded map dir
(`assets/lattice_maps/<seed>`) or a particular saved run here: those exist on one machine, and
a claim about which one has which property is unverifiable from a fresh clone and goes stale
silently. Name the check that decides it instead (`dataset_config.check_maps`), or the config
that points at it.

Each entry point runs as `python src/feasibility/<pkg>/<module>.py` (e.g.
`python src/feasibility/comparator/compare_speed_bumps.py`,
`python src/feasibility/comparator/compare_box_obstacles.py`,
`python src/feasibility/learning/generate_init_pose_dataset.py`,
`python src/feasibility/learning/train.py --dataset outputs/dataset_<...>.h5`,
`python src/feasibility/grid_learning/generate_dataset.py +dry_run=true`,
`python src/feasibility/lattice_learning/generate_dataset.py up_down`,
`python src/feasibility/grid_learning/gl_replay_grid.py --file outputs/dataset_grid_<...>.h5 --map-index 0 --command-index 0`,
`python src/feasibility/submodule_test/run_check.py +tier=smoke`,
`python src/feasibility/submodule_test/run_check.py +tier=smoke +update=true`,
`python src/feasibility/replay/gl_replay.py --id 3 --which both --speed 0.25 --loop`) and documents
its own CLI in a module-top docstring — there is no README. There is also no pytest suite for
this tree: every module's `if __name__ == "__main__":` block doubles as its smoke test (e.g.
`terrain_patch.py`'s flat-ground and rotation-invariance asserts), so running a module directly
*is* how you test it. Running `gl_replay.py` (or
`test_nn.py`) writes an `imgui.ini` window-layout file to the repo root (gitignored).
`train.py` logs to Weights & Biases and writes checkpoints under `outputs/checkpoints/`;
both `outputs/` and `wandb/` are gitignored.

`benchmarks/bench_uphill_nn.py` runs helhest_stack's lattice planner (`CostToGo`) on the uphill
series in four arms: no `blocked`, settle `blocked`, and arcs gated by the `lattice_learning`
network on e_pos or on e_pitch (`feasibility.planning`'s `EdgeGatedLatticeSolver`, a local copy of the relax kernel with a
per-arc `err > tau` check, so the submodule isn't touched). The two global thresholds are fixed
constants (`TAU_POS` 0.1741 m, `TAU_PITCH` 0.0933 rad, CLI-overridable) calibrated once for the
default checkpoint: halfway between `tau*` (the worst arc on the best path) on the 60° map, which
ostrich climbs, and the 65° map, which it fails. That boundary comes from
`demos/ostrich_ramp_crossing.py +series=uphill`: 5–60° cross in 3/3 repeats, 65–75° flip at the
foot in 3/3. A new checkpoint needs a new calibration. `--plot-dir` writes one 2×2 bird's-eye PNG
per map (one panel per arm, with the planned path). Inference defaults to CPU (`--torch-device`);
each map is row-invariant, so only one row gets evaluated (checked). On the ThinkPad-T15p-Gen-1
machine only, torch from the root `.venv` can't use the GTX 1050 — for GPU torch there, activate
`.venv-cu126` instead (`source .venv-cu126/bin/activate`, then plain `python`, not `uv run`). `--self-test` covers the gate and the inference shortcut.

`demos/` is smaller and less settled — still worth treating as scratch, not architecture.
`helhest_common.py`, `ostrich_keyboard.py`, and `ostrich_vel_cmd*.py` mirror
`ostrich/examples/helhest_junior/{common,control}.py`, running an ostrich example from the
shared root env instead of `ostrich`'s own; `ostrich_speed_bump.py` recreates one bump height of
`comparator.compare_speed_bumps` as a single-run GL-viewer-or-headless demo, reusing that
module's `ScenarioSpec` and `comparator.common`'s wheel-servo gain rather than re-deriving them;
`hstack_vel_cmd.py` is the helhest_stack-side counterpart.
`view_planned_path.py` plans on one heightmap with a `feasibility.planning` planner and draws the traced policy in the Newton GL viewer -- one RED ARROW per planned pose (x, y, yaw) over the terrain mesh, no physics stepped, so it shows what was PLANNED. Consecutive poses sharing a cell (a point turn) are lifted `ARROW_DZ` each, since flat arrows would stack into a star at one point; a pivot therefore reads as a spiral and ordinary driving stays flat. Its per-pose table takes the HIGHEST ground within one `wheel_radius` of each wheel centre -- the settle's own spherical envelope, and the tyre's -- and flags relief above `--relief`, which is how a curb the settle tolerates becomes visible next to a `blocked` count of 0; sampling the centre alone, which is what it did first, reads a wheel dragging its rim along a curb face as standing on flat ground. `--dry-run` is the display-free smoke test, `--torch-device cuda` is worth passing for `nn-gated-pivot` (it needs a field over every lattice pose). `garage_gate.py` is the A/B experiment for that planner on `heightmap/create_garage.py`'s pair: three arms per map (`vanilla-on` at `pivot_cost` 0, which must find no route; `vanilla-on` with point turns; `nn-gated-pivot`), then the predicted `e_pitch` of every point turn the lattice can place inside the garage split by whether that turn's swept TYRE reaches the curb (geometric ground truth from `create_garage.feature_distances`), and with `--tau-sweep` which tolerances still leave a route. Reads as intended -- the curb costs the gated planner its route on map B and not on map A -- but that verdict is what `planners.TAU_PIVOT_PITCH` is SET to produce, not independent evidence for it: after the 2026-09-22 retrain the demo's own `--tau-sweep` put the separating window at 0.050-0.070 rad, and the shipped 0.06 is its middle (the lower edge of the 0.062-0.113 band `planning/tune_pivot_tau.py` derives from the same weights' held-out pivots, so aggressive but inside what the data supports). The verdict used to ride on a few percent of one prediction -- the decisive point turn grazed the curb with 0.05 m of rim and its wheel CENTRE 0.30 m clear -- which is why a retrain flipped it once already; `create_garage.py`'s spur now puts 0.25 m of rim over it, and the turns whose tyre reaches the curb sit at 0.085 rad against 0.037 for the ones that do not. Driving a wheel CENTRE through the curb, which earlier versions of this file named as the standing fix, turns out NOT to be available: the settle blocks that pose, and then map B no longer plans map A's route. It also reports its own caveat, since roughly a third of map A's curb-free turns already score over tau -- the head reads the 1 m walls too, so map A's gated route can come out longer rather than identical, and the demo says which of the two it got from the run's own numbers instead of claiming one. `--view` then draws the outcome in the same Newton GL viewer, reusing `view_planned_path.py`'s `build_model`/`arrow_segments` rather than re-deriving them: terrain mesh, the robot parked at the start pose, and one red heading arrow per planned pose of the arm `--view-arm` picks (the gated one by default). An arm that found NO ROUTE -- map B under the gate, the result the demo exists for -- still gets its scene, so the robot is visible boxed in, but nothing is drawn over it and the console says so; with both maps loaded it is ONE viewer cycling between them on RIGHT/N and LEFT/P (`terrain_browser.py`'s pattern, the rebuild in the render loop rather than in pyglet's key callback). `ostrich_follow_path.py` plans on one heightmap with a `feasibility.planning` planner
(`--planner vanilla-off|vanilla-on|nn-gated-pos|nn-gated-pitch|nn-gated-rot|nn-gated-fused`; e_rot comes from the
pos_rot sibling checkpoint `--checkpoint-rot`, `TAU_ROT` 0.38 was picked from
parallel-worlds runs of the uphill series -- straight paths to 55°, no path from 60°, where pure
pursuit arrives only half the time -- not a tau* calibration; `nn-gated-fused` prunes an arc when
the pos_rot net's e_pos > `TAU_FUSED_POS` 0.25 OR e_rot > `TAU_FUSED_ROT` 0.46; tau_pos from parallel-worlds runs with
random start/goal pairs -- a one-point window, the gated search finds angled 60°/55° climbs the net
under-predicts at 0.26+ and 0.24 -- e_rot mid-window from an offline planner scan, uncalibrated.
Pure pursuit clamps curvature at `KAPPA_GAIN` 2x the planner's (skid-steer slip) and also stops at the
path's last waypoint within `GOAL_TOLERANCE` 0.3 m, which `judge` counts as arrived), then drives that path in ostrich
with `planning.pure_pursuit`'s kernel inside the captured step (`--repeats` worlds, or `--view` for the
live GL viewer, camera starting at the corner behind-right of the start; `--start-side`/`--goal-side`
left|center|right shift the sidecar's start/goal perpendicular to the start -> goal line, to 1.5 m of the map edge), writing a verdict table, a bird's-eye PNG and a replayable h5 to
`outputs/follow_path/`. The script itself keeps only the CLI (`add_shared_args`), `plot_run` and the
output writing; it runs as `python demos/ostrich_follow_path.py` or `python -m demos.ostrich_follow_path`. `ostrich_follow_path_parallel_worlds.py` is its batch
form (headless, no `--view`; imports the demo's CLI/plot helpers, so `python -m` only): plans every (`--maps` × `--pairs` × `--planners`), then runs every
(map, start/goal pair, planner, repeat) as a replicated world of ONE build — replicated worlds collide only with
global shapes, never each other, so pairs on the same map share a footprint, and different maps sit
on `lattice_learning.tiled_terrain` tiles with offsets added at spawn/path and subtracted from the
logs; worlds sorted by path length, split into builds by `--worlds-per-build` (GPU memory). Pair 0
is the sidecar's start/goal; the other `--pairs` − 1 are sampled per map (seeded by map name) from
the heightmap alone -- start with random yaw on the map's lowest level with the terrain around each
wheel contact under that wheel's rim, goal with a wheelbase-radius disc on its highest level.
`<map>_p<K>_<planner>` PNG/h5 plus `summary.yaml` in `outputs/follow_path_parallel/`.
`ostrich_follow_plan_turning.py` is the same shape (plan -> follow -> verdict table, PNG, replayable h5,
reusing that script's `add_shared_args`/`plot_run`/`corner_camera`/`show_parked` rather than copying them)
for a route whose decisive manoeuvre is a POINT TURN, which pure pursuit cannot drive -- so it follows with
`planning.pivot_pursuit` instead, and defaults `--pivot-cost` to 0.15 and `--torch-device` to cuda, both of
which the uphill demo defaults the other way. `--map A|B` names `create_garage.py`'s pair (`--map-dir`,
default `assets/garage`); any other value is a heightmap stem, so `create_pivot_pocket.py`'s map works too.
It prints the PLANNED turn before driving it (bins, degrees, and `view_planned_path.wheel_relief` at the
turning poses) and then a TURN TABLE over the phase log's pivot window: pitch BOTH ways round -- `judge`'s
`climb_deg` is `max(-pitch, 0)`, nose-up only, and the garage curb is sized against the REAR wheel, so the
existing metric cannot see the one feature the map exists for -- plus |roll|, achieved yaw, drift, timeouts,
and the relief the wheels REALLY reached. That last one is what keeps a null result honest, and it earned its
keep: a point turn drifts (the reference point is the front axle, the rear wheel is dragged), so where the
plan put a wheel is not where the wheel went, the first garage_b measured nothing at all, and the script says
so when the plan grazed something no repeat touched. `create_garage.py`'s curb spur is sized against that
drift, so the pair now separates: over the same 45° turn map B costs ~1.3 s and 5° of heading, drifts 0.14 m
further and times a pivot out in every repeat, with the rear wheel on 0.20 m of relief and ~0.8° nose-down.
The pitch signal stays small because a 0.35 m wheel shoved sideways into a 0.20 m face jams rather than
climbs -- the timeout is the measurement, not a controller fault. The arms worth running are a controlled
pair: `vanilla-on` plans the SAME route on both maps (the settle cannot see the curb, so there is nothing to
plan around -- worth re-checking after any change to the map, since a curb close enough to block a pivot pose
ends the pair), so A and B drive one plan over two terrains differing by one feature; `nn-gated-pivot` then
finds no route on B at all, which is the result rather than an error and
which `--view` shows as the robot parked where the plan gave up. (`helhest_in_ostrich.py`, referenced
in earlier versions of this file, no longer exists.)

Two things that pattern has to get right, and that any further root-level `demos/` script will
hit too — `src/feasibility` already gets both right and is the template to copy instead of the
older `demos/` scripts:

* **Hydra configs and mesh assets live under `ostrich/examples/`, not at the repo root.** A
  `__file__`-relative `parent.parent/"conf"` (correct in its original home) resolves to a
  nonexistent `<root>/conf` here. Anchor on the `examples` package instead —
  `pathlib.Path(examples.__file__).parent` — which the shared editable install resolves to
  `ostrich/examples/`.
* **`python demos/foo.py` puts `demos/` on `sys.path`, not the repo root**, so a
  `from demos.x import ...` self-reference raises `ModuleNotFoundError: No module named 'demos'`.
  Either run it as `python -m demos.foo` (root on `sys.path`; `demos` resolves as a PEP 420
  namespace package, no `__init__.py` needed) or keep the try/except fallback to the bare
  `from x import ...` that ostrich's originals use.

## Python environment

The root [pyproject.toml](pyproject.toml) resolves one shared `.venv` at the repo root in which
both submodules are editable installs — `import ostrich`, `import helhest`, and ostrich's
`examples.*` all work from the same interpreter — and also builds/installs `src/feasibility`
itself (see above) via a hatchling `[build-system]`. Deliberately *not* a uv workspace: ostrich already declares its own
`[tool.uv.workspace]`, and uv rejects nested workspaces, so a workspace root here would force
a permanent local diff inside a submodule.

```bash
git -C ostrich submodule update --init --recursive   # third_party/newton — required first
uv sync                                              # several GB (torch/cu128); builds openmesh
```

**`tool.uv.sources` is not transitive** — uv honours it only for the project being resolved,
never for a path dependency's own manifest. Every source ostrich and newton rely on
internally is therefore restated at the root (`newton` → the nested submodule path, else uv
fetches an unrelated PyPI package; `warp-lang` → nvidia index; `torch` → cu128 index, which
also forces `torch` to be a direct dependency so the pin applies). Keep that in mind before
editing the root manifest.

The joint resolve does **not** reuse either submodule's lock, so transitive versions drift
from what each repo tests against (ostrich locked numpy 2.3.0 / warp 1.13; helhest_stack numpy
2.4.6 / warp 1.14 — no real conflict, ostrich's lock is just older). To reproduce an ostrich
experiment exactly, run it under `ostrich/`'s own `uv sync` instead.

**ROS is exempt.** [ros/colcon-build.sh](helhest_stack/ros/colcon-build.sh) hard-requires
`helhest_stack/.venv` created with `--system-site-packages`, because colcon stamps
`sys.executable` into the generated node shebangs. The root `.venv` will not satisfy it.

## The two simulators, and which one is current

`ostrich/kinematic_helhest/` is the **original prototype** of the kinematic twin (phases 0–4,
last touched 2026-06 along with the rest of ostrich). It was superseded by
`helhest_stack/src/helhest/engine/`, which carries the same phase plan forward through
implicit gradients, benchmarks, and the planning demo. **New kinematic work belongs in
`helhest_stack`**; treat `ostrich/kinematic_helhest/` as history unless asked otherwise.
Ostrich remains the full-dynamics reference the kinematic model is calibrated/compared against.

Frame convention throughout: X-forward, Y-left, Z-up, meters/radians.

## Working in `helhest_stack/`

One importable package, `helhest`, spanning the whole loop. Key seam to understand before
touching either half: perception emits a **`GridMap`** (`perception/gridmap.py`, deliberately
minimal contract), and the planner consumes it through `perception/grid_adapter.py` →
the engine's `GridParams`, **zero-copy when the elevation is already a `wp.array`**.
`localization/` (GPU ICP over a device-resident rolling accumulator) keeps that map
registered to the world as the robot drives.

The motion core resolves the robot as a rigid tripod: `(x, y, yaw)` from no-slip
differential-drive kinematics with friction-dependent ICR parameters; `(z, pitch, roll)` from
a quasi-static analytic 3×3 settle against the heightmap. Two simulators share `BaseSimulator`
in `engine/simulator.py` — `ForwardSimulator` (fused graph-capturable rollout, no gradients,
the planner's workhorse) and `DifferentiableSimulator` (taped, CUDA-only, gradients w.r.t. the
**raw** heightmap and friction field via a hand-written IFT adjoint through the settle).
Chassis high-centering is **detection-only by design** — it sets `valid=False` and rejects a
trajectory rather than resolving the pose. See the README's "Known limitations" before
"fixing" that or the isotropic (too-fat) spherical wheel envelope.

Standalone env (the shared root env above already covers this — use these only when you want
this repo's own pinned lock, or for ROS):

```bash
cd helhest_stack
uv sync                              # core: numpy + warp-lang
uv sync --extra viz --extra data     # + viewers (glfw/PyOpenGL/matplotlib) + rosbag loader (h5py)
```

Tests are **not pytest** — they are self-checking script modules run individually:

```bash
python -m tests.engine.step          # full physics vs the numpy oracle (reference/)
python -m tests.engine.gpu_check     # forward / adjoint / VJP parity + throughput
python -m tests.engine.gradients     # implicit gradients vs finite differences
python -m tests.perception.test_rasterize
python -m benchmarks.forward         # also: .differentiable, .planning, .control (CUDA)
```

`demos/` and `scripts/` are the runnable entry points (`python demos/eval.py --stress`,
`python scripts/example.py`, …) — the README lists them grouped by subsystem.

ROS lives in [ros/](helhest_stack/ros/), outside the package. **Read
[ros/README.md](helhest_stack/ros/README.md) before debugging any "node drops LiDAR frames /
map is sparse / ICP rejects" symptom** — the usual cause is transport, not compute (6 MB
Ouster clouds silently dropped by Fast DDS; inert under Zenoh, which the live robot uses).
Build with `ros/colcon-build.sh`, which runs colcon under the repo `.venv` python so the
generated console-script shebang can find `warp`.

## Working in `ostrich/`

The engine is a Newton-Raphson solve over a maximal-coordinate constraint system:
`src/ostrich/core/` (engine, residual, linear system, line search, model builder) with
`constraints/` (contact, friction, joint, dynamics, control), `collision/`, `mechanics/`,
`optim/` (PCR solver + preconditioners), `adjoint/` (hand-written friction adjoints), and
`learning/` (differentiable Newton step, torch bridges, warm-start nets). `simulation/` wraps
these into `InteractiveSimulator` / `DifferentiableSimulator` / `DatasetSimulator`.

Examples are **Hydra-driven**: each script in `examples/<robot>/` pairs with configs in
`examples/conf/` (composed from `engine/`, `simulation/`, `control/`, `logging/`,
`rendering/` groups), so the engine backend is swappable per run
(`engine=ostrich|mujoco|featherstone|xpbd|semi_implicit`).

```bash
cd ostrich
git submodule update --init --recursive   # pulls third_party/newton
uv sync --extra sim                       # standalone env; the shared root env also covers this
# system CMake must be < 4.0 (openmesh); otherwise prefix a 3.x on PATH — see README
PATH=~/.local/opt/cmake-3.27.0-linux-x86_64/bin:$PATH uv sync --extra sim

python examples/helhest/surface_drive.py rendering=headless   # Hydra: append key=value overrides
pytest tests/test_joint_constraints.py -k revolute   # top-level tests DO use pytest
bash tests/differentiable_simulator/run_all.sh       # gradient suite — NOT via pytest
```

Don't run `pytest tests/` wholesale: `tests/differentiable_simulator/` has its own
`run_all.sh` that launches **one process per test** "to avoid CUDA state contamination
between models with different topologies", which pytest's single process defeats. Use pytest
for the four top-level test files only.

`experiments/` holds the paper-facing studies (sim-to-real, dt stability, gradient quality,
scalability, terrain traversal); `test_scripts/` is exploratory probe/diagnostic code, not a
test suite — don't treat failures there as regressions.

Both repos publish mkdocs sites from `docs/` via GitHub Actions on push to `main`
(ostrich builds an API reference too; helhest_stack's job only fires on `docs/**` changes).

## Submodule workflow

Commits go **inside** the submodule first, then the superproject records the new pointer:

```bash
git -C helhest_stack commit -am "..."   # work happens here (its own branch/remote)
git add helhest_stack && git commit     # superproject only stores the SHA
```

By design the superproject should track nothing but `.gitmodules`, those two gitlinks, and
`src/feasibility/` — avoid adding permanent *simulator* source at the root; a submodule's own
tree (or its `examples/`) is almost always the right home for that. `src/feasibility/` is the
sanctioned exception, for glue that belongs to neither submodule (see above); `demos/` remains
scratch and shouldn't grow further without being folded into `src/feasibility/` or a submodule.
`*.txt` and `.claude/` are gitignored here, so `context.txt` is local-only.

## File reading and shell commands
When reading or inspecting files, prefer the dedicated `Read` tool.

When using shell commands for operations that should be analyzable and auto-approvable, use literal path arguments rather than wrapping the operation inside a quoted `-c` script.

Prefer:
* `Read` for reading files.
* `cat path/to/file`
* `python -m py_compile path/to/file.py`
* Other shell commands with literal file/path arguments.

Avoid unnecessarily wrapping simple file operations in commands such as:
* `bash -c "cat path/to/file"`
* `sh -c "python ..."`
* `python -c "..."` when a direct command can accomplish the same operation.
The goal is to keep file operations transparent, directly analyzable, and easy for Claude Code's permission/approval system to recognize.

## Reporting back

Keep the wrap-up short. After a long task the report should be a few sentences or a small table —
what changed, what was measured, what is still open — not a multi-section write-up that takes ten
minutes to read. Lead with the result, give the numbers that back it, and stop. Detail belongs in
the code and its docstrings, where it stays useful; if something needs a long explanation, say so
in one line and let the reader ask.
