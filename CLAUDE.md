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

## Python environment

The root [pyproject.toml](pyproject.toml) is a **dependency manifest only** (`package = false`)
that resolves one shared `.venv` at the repo root in which both submodules are editable
installs — `import ostrich`, `import helhest`, and ostrich's `examples.*` all work from the
same interpreter. Deliberately *not* a uv workspace: ostrich already declares its own
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

python examples/helhest/surface_drive.py            # Hydra: append key=value overrides
pytest tests/                                        # ostrich tests DO use pytest
pytest tests/test_joint_constraints.py -k revolute   # single test (it's parametrized by joint type)
bash tests/differentiable_simulator/run_all.sh       # gradient-correctness suite
```

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

The superproject tracks nothing but `.gitmodules` and those two gitlinks — never add source
files at the root. `*.txt` and `.claude/` are gitignored here, so `context.txt` is local-only.
