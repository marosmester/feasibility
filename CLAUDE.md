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
| `heightmap/` | simulator-agnostic elevation-grid I/O (`HeightMapReader`, PNG+YAML); `.to_ostrich()`/`.to_hstack()` so both sims see bit-identical terrain. `create_speed_bumps.py` generates a swept series of bump-height heightmaps (`assets/speed_bumps/`). `create_box_obstacles.py` builds a box obstacle across `BOX_HEIGHTS`, three CLI modes: default — one random position anywhere its ramp fits inside `--extent`, shared across the whole height series (`assets/box_random/`); `--center` — origin-centered, same series (`assets/box_centered/`); `--batch` — `--n` (default 100) independent maps at one fixed `--height`, each its own random position (`assets/box_random/`, index-prefixed filenames), and `--n-boxes K` puts K independently-placed boxes on each such map, rejection-sampled to keep `--min-gap` (default 1.6 m, two signal-ring widths) between ramp footprints. K > 1 writes `box_random_k<K>_i*_h*cm` — the K tag precedes the index so the single-box glob `box_random_i*_h*` cannot match a multi-box map, and K=1 keeps the original un-tagged name so that series regenerates byte-identically. Multi-box maps exist because a grid-shaped label (see `grid_learning/`) is ~94% open flat ground with one box; K boxes multiply the divergence signal ~K-fold while making the map *cheaper* to simulate, and stay valid until obstacles cover half the map (~14 boxes) since spawn filters estimate the background as the median height. `build_multi_box` sums one `build_centered_box` layer per center — exact overlay, since box maps are relative bump layers zero outside their footprint, asserted by a no-overlap check on the summed max. A legacy off-center series (`assets/box/`) is no longer CLI-reachable but its constants/builder stay importable — `compare_box_obstacles.py` depends on them. `create_large_box_obstacles.py` generalizes the fixed 1.5m x 1.5m box for `grid_learning`, whose whole-map divergence-field labeling starves for signal on a single small box: obstacle dimensions are independently-sampled rectangles (0.5-6.0m per side, height fixed at 0.7m, one or several, allowed to touch/overlap), placed under a hard `--max-area-fraction` cap (default 0.5) so plenty of the map stays genuinely flat, and `--n-boxes` is only an UPPER bound — sizing stops once that area budget runs out, so a map can end up with fewer. Because every obstacle shares one fixed height, layers combine via an elementwise MAXIMUM rather than `build_multi_box`'s sum, so touching/overlapping footprints merge for free with none of `min_gap`'s non-overlap bookkeeping. Placement is chosen by a best-of-N random search (`--position-trials`) maximizing `accessible_frontier_score` — obstacle/flat-ground boundary cells within the spawn lattice's own reach (`DEFAULT_ACCESS_LIMIT`, deliberately == `grid_learning.generate_dataset.SPAWN_LIMIT`, re-derived rather than imported per that module's own independence stance) — so more of the 15x15 spawn lattice lands near a divergence-producing edge instead of on empty flat ground. Written to `assets/large_box_random/<seed>/large_box_i*_h070cm`, a filename prefix deliberately distinct from `box_random_*` so `generate_dataset.py`'s default glob never picks these up by accident. `create_surface.py` rasterizes ostrich's Blender "Landscape" mesh into one hilly terrain (`assets/surface/`). `create_rough_terrain.py` synthesizes band-limited random rough ground via spectral synthesis (`assets/rough/`). `mend_two_meshes.py` adds two heightmaps together — e.g. a box obstacle riding on rough terrain (`assets/mended/`) |
| `comparator/` | `common.py` — the scenario-independent Hydra-driven core (config path anchored under `ostrich/examples/`) that spawns the robot at a pose and holds a constant commanded body twist `(v_drive, wz_drive)` once in ostrich (dynamics) and once in helhest_stack (kinematic twin), saving poses/wheel velocities/terrain/git-provenance to `outputs/compare_<name>.h5`. Two ways to batch a run, sharing that same per-trial (spawn pose, twist) → (ostrich pose, hstack pose) contract: a `ScenarioSpec` holds spawn+twist fixed and sweeps *terrain* across a `variants` height series (`compare_speed_bumps.py` drives straight over the bump — vertical-clearance divergence; `compare_box_obstacles.py` turns in place beside the box so the rear wheel sweeps into its side — lateral-collision divergence helhest_stack can't represent at all — each reading its height series from `heightmap.create_speed_bumps`/`create_box_obstacles`); a `TrialScenarioSpec` holds one terrain fixed and sweeps a list of `Trial`s (spawn pose + twist) instead, via `run_trial_comparison` (`compare_on_surface.py` runs five hand-picked trials — flat baseline, uphill, downhill, contour traverse, in-place turn on a slope — over one fixed hilly terrain from `heightmap.create_surface`). `provenance.py` is the HDF5 schema (embeds git SHA/dirty state of both submodules so a saved run is self-describing), grouped/attrs-based like ostrich's own `ostrich.logging` loggers |
| `plotting/` | `batch_comparator_viewer.py` — matplotlib 3D terrain+trajectory and 2D wheel-velocity viewer for one saved variant; `dataset_patch_viewer.py` — plots one `generate_dataset_body_centered_patch.py` sample's baked-in body-frame terrain patch alongside its (v, wz) command and a recomputed ostrich-vs-hstack pose error; `terrain_visualizer.py` — Newton GL viewer for a saved heightmap with a static Helhest Junior model parked at its center for scale, no physics |
| `replay/` | `gl_replay.py` — Newton `ViewerGL` real-time playback of a saved trajectory pair on the real Helhest Junior mesh (pose-only, no physics stepping); `test_nn.py` — same playback plus a loaded `learning.model.PoseErrorMLP` checkpoint, printing the real final-pose (e_pos, e_rot) alongside the network's predicted one for the same variant |
| `learning/` | Learns to predict ostrich-vs-hstack final-pose divergence from the commanded twist plus **either** the spawn pose **or** a robot-centric terrain patch (`terrain_patch.py`, `train.py --patch`) — the pose is a coordinate code that only means anything on the one heightmap it was fitted to, the patch is terrain geometry that transfers. Two Hydra generator scripts, sharing spawn-pose sampling and the chunked ostrich/hstack batch rollout via `generate_dataset_utils.py`: `generate_init_pose_dataset.py` samples N random (spawn pose, twist) trials on one fixed centered-box heightmap and writes `spawn_pose`/`v_drive`/`wz_drive` per row (`x = (v, wz, spawn_x, spawn_y, spawn_yaw)`); `generate_dataset_body_centered_patch.py` writes the same rows plus a baked-in `patch` dataset (`terrain_patch.sample_patches()`, computed once at generation time — `x = (v, wz) + flattened patch`). Both replay each trial once in ostrich/once in hstack as N parallel worlds (`comparator.common.HelhestBatchSimulator`/`run_hstack_batch` take a per-world `[N, 3]` spawn pose), writing `outputs/dataset_*.h5`/`outputs/dataset_patch_*.h5` in the same schema `comparator/provenance.py` uses; `pose_error.py` — the (e_pos, e_rot) SE(3)-error metric shared by every consumer below; `custom_dataset.py` — `PoseErrorDataset`/`make_dataloaders`, a torch `Dataset` over that schema (also loads `ScenarioSpec`-style sweep files), dropping NaN rows, reading a file's baked-in `patch` dataset directly when `use_patch=True`; `terrain_patch.py` — `PatchSpec`/`sample_patches`, the body-frame elevation patch (yaw-aligned so `yaw` drops out, heights relative to the wheel contacts and scaled by `WHEEL_RADIUS` so box height generalizes, forward-biased since `V_RANGE` is non-negative), plus `patch_spec_to_attrs`/`patch_spec_from_attrs` to round-trip a `PatchSpec` through a dataset file's flat root attrs; `model.py` — `PoseErrorMLP`, one shared trunk with two heads (e_pos, e_rot correlate 0.72–0.83) regressing in log1p/standardized space via `TargetTransform`; `train.py` — the training loop (target transform fit on train rows only, loss computed in model space so `TargetTransform`'s clamp never kills a gradient), logging to Weights & Biases and writing a self-describing checkpoint to `outputs/checkpoints/`; `error_visual.py` — plots a dataset's per-sample (e_pos, e_rot) against the input features that produced them |
| `grid_learning/` | New and still growing (branch `experiment/grid-learning`) — fixed-resolution heightmap → torch tensor prep, for feeding a saved heightmap into an NN/CNN. `utils.py`: `heightmap_to_tensor`/`load_heightmap_tensor` resample a `HeightMapReader` (bilinear) onto a `resolution`-m grid (default 0.10 m) spanning an `extent`-m square centered on the world origin (default 10.0 m → 100×100), same cell-center convention as `HeightMapReader.H`/`learning.terrain_patch.PatchSpec`. `generate_dataset.py` is the CNN reframing of `learning/`'s per-trial datasets: one sample is a whole map plus one commanded yaw rate (`x = (heightmap [100,100], wz)`), and the label is a divergence **field** — the ostrich/hstack final-pose pair at every cell of a 15×15 spawn lattice (`y = [15,15,14]`, `mask = [15,15]`). The robot spawns yaw-locked at 0 on a 0.5 m lattice within ±3.5 m (1.5 m of padding, sized so a turn in place keeps the rear wheel's ~1.45 m rim reach on mapped terrain — `HeightMapReader.sample` clamps outside the grid rather than raising, so under-padding fabricates flat ground silently) and is commanded `v=0, wz~U(-1,1)`, i.e. `compare_box_obstacles.py`'s lateral-collision scenario swept over the whole map. Masked cells (spawn footprint on the obstacle, or a diverged solve) carry **zeros, not NaN** — a NaN target poisons the backward pass even through a correctly masked loss, since `0 * NaN` is NaN. **Deliberately does not import from `learning/`** (the two trees stay independent): wheel geometry is re-derived from `HelhestJuniorConfig`, the on-obstacle spawn filter re-implemented locally; `comparator`/`heightmap` are shared infrastructure and are imported. Because `HelhestBatchSimulator` bakes the terrain into the model's globals builder, a chunk of parallel worlds can never span two heightmaps — so the map loop is the outer one, and within a map a chunk freely mixes lattice cells *and* yaw rates. Writes `outputs/dataset_grid_*.h5` (own writer — final poses only, no time series — reusing `provenance.terrain_fields`/`git_provenance` so provenance can't drift). **Plain importable package** (`feasibility.grid_learning`, `__init__.py` present) — `model.py` imports `custom_dataset.py`/`utils.py` siblings via ordinary `from feasibility.grid_learning.x import y`, and every module still also runs standalone as a script (`python src/feasibility/grid_learning/utils.py <path>`). `gl_replay_grid.py` is the QA viewer for a `dataset_grid_*.h5`'s output: unlike `replay/gl_replay.py` there is no trajectory to interpolate (only one final-pose keyframe per lattice cell), so it steps through a row's G x G cells instead, freezing the real ostrich/hstack meshes at each cell's stored final pose and drawing the whole lattice as static colored spheres (green = valid, gray = obstacle-blocked, red = footprint-clear but diverged) so a masked cell's reason is visible at a glance; `--cell I J` freezes on one trial for close inspection, `--dry-run` validates a row (including a local `footprint_clear()` recompute cross-checked against the file's own `mask`) without opening the viewer. Deliberately re-derives its footprint filter and `pose_error`-style SE(3) error formula locally rather than importing them (from `generate_dataset.py`/`learning/pose_error.py` respectively), same independence stance as the rest of this directory. `remask_dataset.py` re-applies `generate_dataset.py`'s per-trial `MAX_SPAWN_DISPLACEMENT` filter (both sims' final pose must stay within 1.0 m of the cell's own spawn — `V_DRIVE` is 0, so a turn in place has no commanded translation and ostrich's real displacement is median 0.11 m / q90 0.20 m, while a 2.75% tail runs to 8.6 m and carried ~90% of the e_pos label variance) to an ALREADY GENERATED file, no re-simulation: the check needs only the stored `y` and `spawn_xy`, so the mask can be recomputed post hoc, and re-running `train.py` alone would change nothing since `GridPoseErrorDataset` reads the stored mask verbatim. Copy-then-mutate via `shutil.copy2` + h5py `"r+"` rather than a second writer, so terrain grids and the physics run's git provenance survive byte-for-byte; three `remask_*` root attrs record what was done. The mask only ever tightens (idempotent), and the `mask &` guard is load-bearing — an already-masked cell holds the zero fill, whose "displacement" reads as |spawn_xy|. This one DOES import its threshold from `generate_dataset.py` rather than re-deriving it: the independence stance separates independent trees, but this tool exists to reproduce that module's decision on old files, so a drifted copy would silently mint inconsistent datasets. `--self-test` is its smoke test, `--dry-run` reports the displacement percentiles and prints ready-to-paste `gl_replay_grid.py --cell I J` commands for the worst dropped cells (the 1.0 m threshold is a judgement call — eyeball a few before committing to it) |

Each entry point runs as `python src/feasibility/<pkg>/<module>.py` (e.g.
`python src/feasibility/comparator/compare_speed_bumps.py`,
`python src/feasibility/comparator/compare_box_obstacles.py`,
`python src/feasibility/learning/generate_init_pose_dataset.py`,
`python src/feasibility/learning/train.py --dataset outputs/dataset_box_h070cm_n256.h5`,
`python src/feasibility/grid_learning/generate_dataset.py +dry_run=true`,
`python src/feasibility/grid_learning/gl_replay_grid.py --file outputs/dataset_grid_box_random_M2_L5_g15.h5 --map-index 0 --command-index 0`,
`python src/feasibility/replay/gl_replay.py --id 3 --which both --speed 0.25 --loop`) and documents
its own CLI in a module-top docstring — there is no README. There is also no pytest suite for
this tree: every module's `if __name__ == "__main__":` block doubles as its smoke test (e.g.
`terrain_patch.py`'s flat-ground and rotation-invariance asserts), so running a module directly
*is* how you test it. Running `gl_replay.py` (or
`test_nn.py`) writes an `imgui.ini` window-layout file to the repo root (gitignored).
`train.py` logs to Weights & Biases and writes checkpoints under `outputs/checkpoints/`;
both `outputs/` and `wandb/` are gitignored.

`demos/` is smaller and less settled — still worth treating as scratch, not architecture.
`helhest_common.py`, `ostrich_keyboard.py`, and `ostrich_vel_cmd*.py` mirror
`ostrich/examples/helhest_junior/{common,control}.py`, running an ostrich example from the
shared root env instead of `ostrich`'s own; `ostrich_speed_bump.py` recreates one bump height of
`comparator.compare_speed_bumps` as a single-run GL-viewer-or-headless demo, reusing that
module's `ScenarioSpec` and `comparator.common`'s wheel-servo gain rather than re-deriving them;
`hstack_vel_cmd.py` is the helhest_stack-side counterpart. (`helhest_in_ostrich.py`, referenced
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
