# `mppi_learning` — a learned model-error cost for helhest_stack's MPPI

Plan, not a spec. Where this says "same as `lattice_learning`", read `lattice_learning/design.md`.

## 1. The task

Predict how far helhest_stack's **kinematic twin** diverges from **ostrich** over one short
window of driving, from the local terrain, the robot's entry speed and the window's command.
MPPI's rollouts *are* the twin, so the prediction answers "where is this rollout fiction?", and
the controller can pay a cost for driving through it.

What changes relative to `lattice_learning`:

| | `lattice_learning` | `mppi_learning` |
|---|---|---|
| label | ideal arc + settle ↔ ostrich (the *router's* model error) | **twin ↔ ostrich** (the *executor's* model error) |
| command | `kappa` or constant `(v, wz)`, pinned at `V_NOM` / `OMEGA_NOM` | **`(v_mean, v_slope, wz_mean, wz_slope)`** over MPPI's sampling box |
| window | 0.3 m arc = 0.5 s | **T = 1.0 s**, aligned with MPPI's knots (§3) |
| patch | x ∈ [−1.5, 2.0] m, y ∈ [−1.5, 1.5] m | **x ∈ [−1.5, 3.0] m**, y and x_min unchanged (§4) |
| motion | forward arcs, pivots at one rate | **forward motion and in-place pivots only** — no reverse |

The architecture is `ArcDivergenceNet` as is: command-free terrain trunk, command encoder, FiLM
head. Only the command encoder's input width (4) and the patch shape change.

## Chosen values

| what | value | why |
|---|---|---|
| network window T | **1.0 s** = 10 MPPI steps (DT 0.1 s) = 40 ostrich steps (dt 2.5e-2) | one window per MPPI knot interval (§3) |
| command | **`(v_mean, v_slope, wz_mean, wz_slope)`** per window | linear-in-time assumption (§2) |
| patch x | **[−1.5, 3.0] m** | `x_max` 2.0 + (1.4 − 0.6) m of extra travel, rounded up to a multiple of the trunk stride (§4) |
| patch y | **[−1.5, 1.5] m** (unchanged) | pivots and 1 s arcs stay inside it (§4) |
| patch cell / reference | **0.125 m / `"wheels"`** (unchanged) | same as `lattice_learning` |
| patch grid | **24 × 36** (ny × nx); trunk output 6 × 9, so the geometry layer's kernel is (6, 9) | follows from the above |
| MPPI horizon `H` | **31 steps** (3.0 s of rollout, three windows + the final knot) | `H = 10 · (n_knots − 1) + 1` |
| MPPI `n_knots` | **4**, at steps 0, 10, 20, 30 | knot spacing = exactly one window |
| MPPI wheel-speed box | **`wmin` = 0, `wmax` = 4 rad/s** | forward only; v_max = 4 · 0.35 = 1.4 m/s |
| MPPI pivots | **spin prior on** (`spin_frac` > 0, `spin_min` 2 rad/s); `pivot_frac` = 0 | the spin prior is exempt from the `wmin` clamp; the pivot prior is not, so with `wmin` = 0 it degrades to arcs |

## 2. The command — and the assumption it rests on

**Assumption: within one window, the commanded `v(t)` and `wz(t)` are each close to linear in
time.** So each is encoded as its mean and slope over the window, four numbers in total.

It holds because every MPPI sampling prior (`_sample_target_wheel_omega_kernel`) interpolates
linearly between control knots that are shared by all candidates, and the per-step jitter on top
is small (σ = 0.5 rad/s) and filtered by the motor lag. It is exact for the noise samples if a
window never straddles a knot (§3). It is only approximate for the nominal `U` once the
`plan_consistency` one-step shift has moved its kinks off the knot grid.

Training commands are drawn **from MPPI's own sampler**: spline knots + jitter, restricted to
forward motion (`v ≥ 0`) and pivots (`wl = −wr`, magnitudes up to the spin prior's). The data
then covers exactly what the net will be asked about at control time.

## 3. The window length

T = 1.0 s, chosen mainly so that **window boundaries fall on MPPI's knots**. The command is then
linear in time inside each window, and the mean + slope encoding loses nothing but jitter. MPPI's
knot spacing is `(H − 1) / (n_knots − 1)` steps, so we set it to exactly one window (10 steps):
`H` = 31, `n_knots` = 4, knots at steps 0, 10, 20, 30. Window k covers steps [10k, 10k + 10),
which is exactly the span between knots k and k + 1. The final step (knot 3) is in no window.

This changes MPPI's current settings. Neither aligns today: the ROS node's `plan_horizon` 25
gives 8-step spacing, and the demos' `H` = 70 gives 23. A longer horizon keeps the alignment as
long as `H = 10 · (n_knots − 1) + 1`, e.g. `H` = 61 with `n_knots` = 7. The 1 s spacing is about
as smooth as what the robot runs now (0.8 s at `plan_horizon` 25).

T is capped from above by the patch (§4) and by how precisely the cost can place a problem in
time. A longer window mixes more events into one label.

## 4. The patch

Same construction as `lattice_learning/patch.py`: body frame, relief relative to the wheel
contacts, divided by `wheel_radius`, 0.125 m cells. Only the extent changes.

* **Forward:** the patch must reach `v_max · T` of travel ahead of the robot, where
  `v_max = wmax · wheel_radius` = 4 · 0.35 = 1.4 m/s. That is ~1.4 m at T = 1 s, against the
  ≤ 0.6 m the current `x_max` = 2.0 m was sized for. That gives 2.8 m, rounded **up** to
  **`x_max` = 3.0 m** (36 cells). 2.875 m (35 cells) would tile, but `ArcDivergenceNet` requires
  both patch sides to be multiples of the trunk's total stride 4.
* **Lateral: unchanged at ±1.5 m.** A pivot's rear-wheel rim reaches 1.10 m
  (`rear_offset + wheel_radius`). On a rough estimate, a 1 s forward arc keeps every rim within
  about 1.25 m: the tightest arcs are slow, and the fast ones are wide. `patch.py`'s self-test
  should assert this from `RobotParams` and the sampling box rather than trust the estimate. If
  the check fails, widen y, not T.
* **Backward:** unchanged. The integration below allows only forward motion and pivots, so
  nothing behind the current `x_min` is ever driven onto.

## 5. Data and training

Very close to `lattice_learning`: the same config-driven `generate_dataset.py`, map categories,
`spawn_sampling` strategies, `tiled_terrain` batching and `train.py`. The differences:

* Each trial records **twin vs ostrich** under the same time-varying command profile.
* Each trial starts with a warm-up that brings the robot to a sampled entry speed rather than the
  window's own command. That speed is not one of the four command inputs; whether it becomes a
  fifth input or stays a data-only randomisation is open (§7).
* Everything else carries over: the label heads (`pos_rot` / `pos_rpy`), `TargetTransform`, the
  mirror augmentation and the self-checks.

## 6. Integration into MPPI

* Split each rollout into its windows (knot-aligned, §3). For window k, take the patch at the
  twin's pose at the window's start, and the command's mean + slope from `target_wheel_omega`.
* Charge each rollout a **weighted sum** of its per-window predictions, **earlier windows weighted
  more** (they are closer to the real state and harder to escape). The exact weighting, and how
  many windows to evaluate at all, are left for later.
* MPPI's sampling is restricted to forward motion and pivots (`wmin` = 0 plus the spin prior,
  see Chosen values), which is the net's training envelope.
* helhest_stack gets a **torch-free hook** in `MppiGpu`: a stable device buffer the cost kernel
  reads, like `set_lattice`. The network and everything that fills that buffer stay here, in
  `src/feasibility/`, since helhest_stack must not import this tree.

## 7. Later

* **Real-time inference.** Once the net is trained: cache the command-free trunk as a code field
  at routing cadence, run only the head per rollout window, and move inference into the CUDA
  graph (e.g. by exporting to TensorRT). Nothing above depends on how this is done.
* **Entry speed as an input.** Decide from data whether the twin's realized twist at the window
  start must be a net input or whether warm-up randomisation is enough.
* **Complete maps only.** Like `lattice_learning` (§3c there), there is no measured-cell channel,
  so a partially observed map is outside the training distribution.

## 8. Implementation plan

**Import rule.** `mppi_learning` may import from `lattice_learning`, `comparator`, `heightmap`
and helhest_stack. It must not import from `learning`, `grid_learning` or `planning`. Changes to
`lattice_learning` are additive only (new optional arguments whose defaults keep it
bit-identical), so its checkpoints and datasets stay valid.

### Reused as is (imported)

| module | what we take |
|---|---|
| `lattice_learning/patch.py` | `PatchSpec` (built with `x_max` 3.0), `sample_patches`, `patch_overhangs`, attrs round-trip. Stays parameterised, so no edit. |
| `lattice_learning/settle.py` | `settle_batch` / `settle_feasible`: spawn feasibility |
| `lattice_learning/tiled_terrain.py` | `TiledTerrain`, `tile_offsets`: several maps per ostrich build |
| `lattice_learning/model.py` | `ArcDivergenceNet` trunk/head, `TargetTransform`, `Normalizer`, `LABEL_NAMES` |
| `lattice_learning/custom_dataset.py` | `pose_to_se3`, `se3_errors`, `rpy_errors`, `split_dataset_by_map` |
| `lattice_learning/dataset_config.py` | the YAML parsing helpers, `MapsConfig` / `RunConfig`, `check_maps`, `allocate` |
| `comparator/common.py`, `provenance.py` | `run_ostrich_batch` (it already takes per-step, per-world setpoints `[T, W, 3]`, so a time-varying profile needs no change), `cmd_to_wheels`, `git_provenance`, `terrain_fields` |
| `heightmap/` | the map generators and `assets/lattice_maps` as they are |

### Reused only in part

`spawn_sampling.py` is **not** reusable wholesale, although it looks it. Every strategy places
its trial geometry for a **0.3 m arc at `V_NOM`**: `ARC_LEN`, `integrate_arc` and `OMEGA_NOM`
are baked into the face and edge placement, `required_platform_length` and the relief look-ahead.
A 1 s window at up to 1.4 m/s travels ~5× further, along a curve that is not a constant-κ arc.

* Imported as is: `map_metadata`, `SpawnBatch` (plus new columns), `concat_batches`,
  `edge_field`, `ramp_faces`, `body_points`, `contact_plane`, `wheel_footprint`,
  `rim_penetration`, `stratified_fill`.
* Rewritten in `mppi_learning`: the strategy functions (`uniform`, `edge`, `ramp_up/down`,
  `rotate_in_place`). They keep the same placement logic (distance to an edge, head-on to a face,
  exact `interact_frac`), but measure along the window's **sampled twin path** instead of a
  nominal arc. If the change is just "path length / path integrator as an argument", prefer
  adding that argument to `lattice_learning` over copying.

### New in `mppi_learning`, in order

1. **`command.py`**: the command sampler and its encoding. It draws per-trial wheel-speed
   profiles from MPPI's own priors (spline knots + jitter, forward and spin), restricted to
   `wmin` = 0 plus pivots. It converts them to `(v_mean, v_slope, wz_mean, wz_slope)` and to
   per-step ostrich setpoints. The self-test checks the linear-in-window assumption (§2): the
   residual of the linear fit stays within the jitter.
2. **`twin.py`**: runs helhest_stack's `ForwardSimulator` batched over the same profiles, from
   the same spawn and entry state, and returns the twin pose at each window end. This replaces
   `lattice_learning`'s "ideal arc + settle" label. The self-test checks that on flat ground and
   at low speed twin ≈ ostrich.
3. **`spawn_sampling.py`**: the rewritten strategies above (warm-up to a sampled entry speed,
   then one window), with the same self-tests as `lattice_learning`'s. The one extra self-test is
   the §4 lateral-extent assert.
4. **`dataset_config.py` + `configs/default.yaml`**: `lattice_learning`'s schema plus a
   `command:` block (knot count, jitter σ, spin fraction/min, entry-speed range). Parsing,
   `check_maps` and `allocate` are imported.
5. **`generate_dataset.py`**: same skeleton as `lattice_learning`'s (config → allocate → tiled
   ostrich builds → h5 with provenance). Per row it stores the patch, the 4-number command, entry
   twist, twin and ostrich end poses. `dry_run` first on one map.
6. **`custom_dataset.py`**: a `Dataset` over that h5 (4 command columns, labels from twin vs
   ostrich via the imported `se3_errors` / `rpy_errors`), map-level split.
7. **`model.py`**: a thin subclass of `ArcDivergenceNet` that swaps the command features for the
   4-number encoding (`(v_mean, v_slope)` scaled by `v_max`, `wz` split into value/|value|/sign
   as today). Mirror: `wz_mean`, `wz_slope` → negated. If a subclass is awkward because
   `command_features` / `COMMAND_COLUMNS` are module-level, add a `"mean_slope"` mode to
   `lattice_learning/model.py` instead (additive).
8. **`train.py`**: `lattice_learning`'s loop, importing its checkpoint/metric helpers where they
   are command-agnostic. It drops the kappa baselines, logs to W&B, and writes to
   `outputs/checkpoints/`.
9. **`test_nn.py` / a replay viewer**: held-out evaluation, then a GL replay of twin vs ostrich
   over one window (adapted from `gl_replay_arc.py`, which imports `replay/`: allowed, since
   `replay` is not in the forbidden list).
10. **MPPI hook** (§6): first the torch-free buffer + cost term in helhest_stack's `MppiGpu`,
    then `mppi_learning/mppi_cost.py`, which fills it from the net (eager torch first, TensorRT
    later, §7).

Steps 1–3 are independent of training and can be checked on their own. Step 5 is the first
GPU-expensive one.
