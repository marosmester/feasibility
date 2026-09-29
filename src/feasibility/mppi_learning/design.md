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

One deliberate exception: the dataset's spins start at `command.spin_min` 0, below the 2 rad/s
MPPI floors its spin prior at (the real robot does not break loose under it). The data is a
superset of what MPPI samples, so the net does not have to extrapolate below the floor if MPPI's
`spin_min` is ever lowered. Ostrich has no breakaway either: on flat, non-interacting spins of the
first spin_min-0 dataset it realized a median 0.52-0.58 of MPPI's commanded yaw in every speed band
down to |wz| < 0.25 rad/s, which is the same ratio as above 2 rad/s and close to the twin's `1/α`. So
a slow spin here is NOT the real robot's stall, and the net learns ostrich's version of it. `entry.yaw_ratio` was measured
on spins ≥ 2 rad/s; it only places the spawn, and `rotate_in_place` enters at rest, so a slow spin
changes where an `edge` trial's origin lands by at most its warm-up's yaw error.

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
* Each trial starts with a short warm-up (0.3 s) that gets ostrich moving; ostrich cannot be
  spawned at speed. Measured on flat ground from rest, straight driving settles in ≤ 0.2 s. Turning
  reaches its mean yaw rate in ~0.2 s and then never settles, because ostrich stick-slips, so a
  longer warm-up buys nothing. The warm-up is mostly commanded at the window's **own first
  command**: MPPI's knots are shared by adjacent windows, so every rollout window after the first
  starts with its command continuing. A share (`entry.jump_frac`, 0.2) enters at an independent
  twist in the wheel box instead, like window 0, whose WIDE/STRAIGHT knot 0 ignores the measured
  state. Only ostrich runs the warm-up. The
  twin starts at ostrich's **realized** state at the window start (pose, wheel speeds, body
  twist), with MPPI's `planning_solver`, the way every MPPI replan starts from the measured
  state. The patch is taken at that same pose.
* **`ramp_down` on the lattice maps.** A trial could start up to one window's reach (1.4 m + a wheel
  radius) before the crest. Add the warm-up and the robot's length, and full coverage would need
  3.37 m plateaus, while the maps have 2.1–3.0 m. So each face caps the start by its own plateau:
  the robot and its warm-up must fit on it (`platform_behind`, 1.62 m), and origins start at most
  `plateau − 1.62` m (0.5–1.4 m) before the crest. `check_maps` requires plateaus ≥ 1.97 m. Every
  entry speed and the whole descent are covered, but a fast approach reaches the crest early in
  the window, so late drops are under-represented. Longer-plateau maps would fix that.
* **The net is for outdoors**, and ostrich is commanded a slightly faster turn than the twin is.
  See "Turning" below.
* Everything else carries over: the label heads (`pos_rot` / `pos_rpy`), `TargetTransform`, the
  mirror augmentation and the self-checks.

### Turning: the yaw ratio and the 1.15 yaw gain

**Yaw ratio.** Take a pair of left/right wheel speeds. If the wheels rolled without slipping, they
would turn the body at the *ideal* yaw rate `wz = r (wr − wl) / (2 · half_track)`. A skid-steer
robot cannot turn without slipping: its wheels are dragged sideways, so the body turns slower.
The **yaw ratio** is realized ÷ ideal yaw rate, and each "robot" has its own:

| robot | yaw ratio | source |
|---|---|---|
| ideal kinematics | 1.00 | definition |
| real Helhest, outdoors | **0.55** | helhest's ICP-truth fit of real drives: α ≈ 1.82, ratio 1/α |
| twin (`k_turn` 1.0, mu 0.8) | 0.56 | α = 1 + k_turn · mu = 1.80 |
| ostrich, same command | **0.48** | measured, flat ground |
| ostrich, yaw rate × 1.15 | **0.56** | measured: arcs 0.51–0.55, spins 0.57–0.63 |

**Why a gain.** The label is twin vs ostrich, and ostrich stands in for the real robot. With the
same command, ostrich turns less than the real robot (0.48 against 0.55). Part of every label
would then be *ostrich's* error, not the twin's, and the net would learn to distrust turns that
are fine on the robot. So ostrich alone is commanded MPPI's yaw rate × `trial.ostrich_yaw_gain`.
Ostrich turns at 0.48 / 0.55 ≈ 0.87 of the robot, so the gain is 1/0.87 ≈ **1.15**: ostrich asks
for 1.15× the yaw rate and realizes
about what the robot would. It applies in both the warm-up and the window. The forward speed is
untouched, and the twin gets MPPI's command unchanged.

**Why not something else.**
* *Retune ostrich's friction instead.* This doesn't work: a lateral/longitudinal ratio
  (`MU_LAT_RATIO`) of 0.2–0.5 only reaches 0.48–0.52. Only a longitudinal mu of 1.6 reaches the
  target, and that would let ostrich climb slopes the robot can't.
* *lattice_learning's 0.49.* That value is ostrich vs *ideal* kinematics. Dividing by it would
  make ostrich turn at ~1.0, well past the real robot.

**The sampling box.** Compensation widens the wheel spread: `m ± d` becomes `m ± 1.15·d`. Every
command family is therefore drawn in ostrich's compensated wheel space, inside `[wmin, wmax]`,
and mapped back to MPPI's command (`command.contract`). MPPI's spins top out at wmax / 1.15, and
the widest arcs shrink the same way.

**`entry.yaw_ratio`** (0.56) is the realized ÷ MPPI-commanded yaw rate under this gain. It is
used only to place the spawn so that the warm-up ends near the window origin. Re-measure it
whenever the gain or friction changes.

**Effect.** On flat ground the spin label's median offset fell from 0.17 m / 0.28 rad to
0.10 m / 0.15 rad. The turning noise stayed the same (cv 0.39 vs 0.37).

## 6. Integration into MPPI

* Split each rollout into its windows (knot-aligned, §3). For window k, take the patch at the
  twin's pose at the window's start, and the command's mean + slope from `target_wheel_omega`.
* Charge each rollout a **weighted sum** of its per-window predictions, **earlier windows weighted
  more** (they are closer to the real state and harder to escape). The exact weighting, and how
  many windows to evaluate at all, are left for later.
* MPPI's sampling is restricted to forward motion and pivots (`wmin` = 0 plus the spin prior,
  see Chosen values), which is the net's training envelope.
* helhest_stack gets a **torch-free hook** in `MppiGpu`. The network and everything that fills
  it stay here, in `src/feasibility/`, since helhest_stack must not import this tree. How the
  net runs inside MPPI's CUDA graph is §9.

## 7. Later

* **Real-time inference** of every window, not just the first: §9e.
* **Entry speed as an input.** With the entry mostly continuing into the window, the start speed
  is readable from the command (`v_mean − v_slope/2`). Decide from data whether the jump share
  still needs the realized start twist as a fifth input.
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
2. **`twin.py`**: runs helhest_stack's `ForwardSimulator` with `planning_solver` (motor lag
   included) batched over the same profiles. It starts from ostrich's state at the window start
   (§5) and returns the twin pose at each window end. This replaces `lattice_learning`'s
   "ideal arc + settle" label. The flat-ground twin ≈ ostrich check needs ostrich, so it belongs
   to `generate_dataset.py`'s `dry_run`.
3. **`spawn_sampling.py`**: the rewritten strategies above (warm-up to a sampled entry speed,
   then one window), with the same self-tests as `lattice_learning`'s. The one extra self-test is
   the §4 lateral-extent assert.
4. **`dataset_config.py` + `configs/default.yaml`**: `lattice_learning`'s schema plus a
   `command:` block (knot count, jitter σ, spin fraction/min, entry-speed range). Parsing,
   `check_maps` and `allocate` are imported.
5. **`generate_dataset.py`**: same skeleton as `lattice_learning`'s (config → allocate → tiled
   ostrich builds → h5 with provenance). Per row it stores the patch, the 4-number command, entry
   twist, twin and ostrich end poses. `dry_run` first on one map.
6. **`custom_dataset.py`** (done): `WindowDivergenceDataset` over one or more h5 files, with the
   4 command columns and labels from twin vs ostrich via the imported `se3_errors` /
   `rpy_errors`. Files whose label-defining attrs differ are refused. The split is by map, with
   maps identified by path across files. `drop_endpoint_infeasible` and `drop_twin_flagged` are
   opt-in filters.
7. **`model.py`** (done): `WindowDivergenceNet`, a subclass of `ArcDivergenceNet` through class
   hooks added to lattice without changing its numbers. It turns the 4-number command into 7 features:
   `(v_mean, v_slope)` / `v_max`; `wz_mean` as value / |value| / sign; `wz_slope`; and
   `wz_slope · sign(wz_mean)`, which says whether the turn tightens or eases and is
   mirror-invariant. The scales are fixed by the wheel box, not fitted. Mirror: `wz_mean` and
   `wz_slope` are negated.
8. **`train.py`** (done): `lattice_learning`'s loop, importing its command-agnostic helpers
   (losses, metrics, blur, scheduler, `evaluate`, the verdict line). The kappa baselines are
   replaced by a per-family mean and `family + relief` (least squares on the stored path relief
   with a per-family intercept). The report scores every baseline on the same val subsets:
   interaction, family, continued / jump / at-rest entry (for §7's fifth-input question), flat
   patches, and the rows the two filters would drop. The checkpoint stores the dataset's
   `LABEL_ATTRS` and the command scales, which load asserts. Logs to W&B and writes to
   `outputs/checkpoints/`.
9. **`test_nn.py` / a replay viewer** (done): `test_nn.py` re-scores a checkpoint with
   `lattice_learning/test_nn.py`'s scoring and figure. It rebuilds the checkpoint's own files,
   row filters and held-out-MAP split, and warns when the data's `LABEL_ATTRS` differ from the
   checkpoint's. The viewer is `gl_dataset_browser.py`.
10. **MPPI hook** (§6, planned in §9): the cost hook in helhest_stack's `MppiGpu` (done:
    `set_cost_hook`), then `nn_mppi/mppi_cost.py`, which charges the first window exactly from
    the net (done, checks 9d.1–5), then a closed-loop MPPI-in-ostrich check
    (`nn_mppi/closed_loop.py`; driver done, the experiments open). The scripts
    that put the net into MPPI live in `src/feasibility/nn_mppi/`, beside this tree.

Steps 1–3 are independent of training and can be checked on their own. Step 5 is the first
GPU-expensive one.

## 9. Running the net inside MPPI

The plan for §8 step 10. The numbers below were measured 2026-09-29 on the development GPU (a GTX
1050, fp32, torch 2.11 in `.venv-cu126`) with the first 200-map checkpoint. The robot has a Jetson
Orin and torch; nothing has been measured on it yet.

### 9a. What the net costs

| what | time |
|---|---|
| full net, one patch | **214 µs** (the same from 512 to 36,864 patches) |
| the same, captured in a CUDA graph | no faster (105 vs 108 ms for 512 patches): the GPU is busy, not waiting on launches |
| the same in fp16 | slower (138 ms for 512): Pascal has no fast fp16. The Orin does, so this does not transfer |
| head only (terrain code given), 12,288 rows | **6.8 ms** |
| trunk over one rotated 12 m map (24 × 24 codes at 0.5 m), 24 headings | 105 ms |

Most of the full net's time is not convolution. The profiler gives `ChannelLayerNorm`'s
LayerNorm 57 %, the permutes around it ~16 %, replicate padding 11 % and the convolutions ~15 %.
Computing that LayerNorm channels-first gives the same output to 5e-6 and cuts a patch to
141 µs; the convolutions alone, padding included, take 72 µs. A fused runtime would land
somewhere between those.

At the deployed batch this rules out the full net on every window: 4096 rollouts × 3 windows × 3
refines is 36,864 patches, **7.9 s per frame**. Even 512 rollouts on the first window only is
1,536 patches, 324 ms.

### 9b. The first window is exact and cheap

`MppiGpu.replan` starts every rollout at the same pose, so **window 0 has one patch, shared by
all rollouts**; only the command differs. The trunk therefore runs once per replan, and only the
head runs per rollout. No approximation, no retraining, and the head for 4096 rollouts × 3
refines is about 7 ms on the 1050. Windows 1 and 2 start at each rollout's own pose, so they are
not covered by this; they wait for §9e.

The horizon is the one the net was trained for: `H` = 31, `n_knots` = 4 (Chosen values, §3), so
window 0 is exactly steps [0, 10), between knots 0 and 1.

### 9c. Design

**helhest_stack** (additive; a planner without a hook runs exactly as before):

* `MppiGpu.set_cost_hook(fn)`. `_refine` calls `fn(planner)` right after `_cost_kernel` and before
  the robust reduction. `fn` may only launch device work (no host sync, no allocation, no torch),
  so it is captured into the refine's graph, and it adds its term to `planner.J` in place. All
  `n_mu` replicas of a candidate share its command, so the term is the same in each and the
  reduction's worst/mean split does not matter for it. Setting a hook drops the captured graph,
  so the next `replan` recaptures.

**`nn_mppi/mppi_cost.py`** (torch outside the graph, Warp inside it):

* `WindowCost.from_checkpoint(path, planner, weights)` loads the net with `train.load_checkpoint`
  and refuses a planner it was not trained for: `H` ≠ 31 or `n_knots` ≠ 4, a twin `k_turn` other than the
  checkpoint's `label_attrs`, or `wmin` ≠ 0. It copies the head's weights and the
  `TargetTransform` into Warp arrays once.
* `update(state)`, called before each `planner.replan`: the patch at the start pose, sampled by a
  Warp kernel from the planner's own elevation grid with helhest's `sample_field` (the same
  cell-centre bilinear and edge clamping as `patch.sample_patches`, wheel-contact reference,
  divided by `wheel_radius`), then the torch trunk on it zero-copy, then the 256-number terrain
  code, copied into a fixed Warp buffer. torch runs on Warp's stream, so the graph never reads a
  half-written code.
* The hook, in Warp, per rollout:
  * window 0's command from `target_wheel_omega[0:10]`, with the same least-squares mean + slope as
    `command.encode`;
  * the 7 features, the FiLM head and `TargetTransform`'s inverse;
  * the result, `w_pos · e_pos + w_rot · e_rot`, added to `J`.

  The weights live in device scalars, so tuning them needs no recapture. The head runs over
  `n_cand` rows (the mu replicas share a command) and its term is added to every replica.

  **What it costs** (GTX 1050, 4096 rollouts, 3 refines): 31 ms per replan without the hook,
  44 ms with it — about 3.9 ms per refine for the head plus 1.3 ms for `update`. The planned
  "one thread per (rollout, neuron)" layer took 30–40 ms for one 256 × 256 layer (uncoalesced
  weight reads, 13 GFLOP/s). Transposed weights with 8 rows × 4 outputs per thread take ~2 ms,
  about cuBLAS's speed on this GPU. `baseline="flat"` (9d.6) adds ~10 ms (54 ms), because the
  LayerNorm and the two trunk layers run twice. The LayerNorm runs one tile block per row (0.33 ms instead
  of 2.3 ms for a thread per row). `wp.tile_matmul` does not compile for the 1050 (sm_61:
  MathDx fails to build its kernel), so it cannot be tried here. On the Orin (sm_87) it should,
  and it is the thing to try if the Orin misses its budget.

### 9d. Checks, in order

1. **Patch:** the Warp sampler matches `sample_patches` on the same grid and pose.
2. **Command:** the kernel's encoding matches `command.encode` on sampled `target_wheel_omega`.
3. **Head:** the Warp head matches `net.predict` for the same code and commands, to float tolerance.
4. **No hook, no change:** the refine's `U` is bit-identical to the unhooked planner's.
5. **Timing:** a refine at 4096 rollouts with and without the hook.

Checks 1–5 are `nn_mppi/mppi_cost.py`'s self-test (`--checkpoint` for a trained net, `--bench`
for 5); check 4 is also helhest_stack's `tests/control/test_mppi.py` (`selftest_cost_hook`).
6. **Closed loop in ostrich:** MPPI plans with the twin, ostrich executes, replanning at the ROS
   node's cadence. Run it with and without the cost, on maps where the twin is wrong (edges,
   ramps, pivots beside walls) and on flat ground, where the cost must change little. The driver
   is `nn_mppi/closed_loop.py`. It also re-runs the twin over every executed 1 s window and
   reports the actual twin-vs-ostrich error along the driven path next to the net's prediction.
   This run decides the cost's shape and weights, and whether window 0 alone is enough.

   **First result, flat ground, the first 200-map checkpoint (2026-09-29):** the cost does NOT
   change little. Arrival over 8 m: vanilla 9.7 s; weights (e_pos, e_rot) 3/3: 10.7 s; 10/10:
   12.6 s; 30/30 and 100/100: not in 25 s (100/100 also veers 2 m off the line). On flat ground
   the net's prediction grows with speed, so `w · e` acts as a speed penalty. It also
   over-predicts there: 0.06 m predicted against 0.02 m actual, correlation 0.3 over ~500
   windows. Two ways out: charge only the error above the net's own prediction on a
   FLAT patch for the same command (the terrain's share; a second, all-zero terrain code, and
   the head run twice), or a better flat-ground fit from more data.

   **The level baseline (`WindowCost(..., baseline="flat")`, closed_loop's `nnflat` arms),
   2026-09-29.** Each candidate pays `sum_k w_k · max(e_k(patch) − e_k(level), 0)`. The level code
   is computed once, because an all-zero patch is the same at every pose; the trunk layers then run
   over twice the rows. It costs 54 ms per replan against 44 ms for the raw cost and 29–31 ms with
   no hook. On a level map the excess is exactly 0, so `U` is bit-identical to vanilla's (self-test).

   * **Flat, 8 m, 2 repeats:** fixed. `nnflat` 10/10, 30/30 and 100/100 arrive in 9.2–9.6 s, the
     same as vanilla's 9.7 s. The raw cost at 10/10 still takes 12.7–13.3 s.
   * **Uphill 30°, 3 repeats:** vanilla, raw 3/3, `nnflat` 3/3 and `nnflat` 10/10 all time out
     at 30 s, 0.8–1.2 m short of the goal. They take the same route, veering ~1.4 m sideways on the
     face. They all CRAWL the face: ~0.35 rad/s wheels, ~16 s from foot to crest, vanilla included,
     so it is MPPI's own cost (probably `saturation`; not checked). `nnflat` 30/30 arrived twice
     (20 s), because it did not slow at the foot and climbed in ~2 s at ~3 rad/s.
   * **That is not the cost working as meant.** Re-scoring the executed windows shows the net
     ranks the two climbs correctly: true e_pos 0.18 m climbing fast against 0.02 m crawling,
     predicted excess 0.19 against 0.08. The cost asks for the crawl, so the fast climbs happened
     despite it. The excess is ~0 on the flat approach and on the plateau (≤ 0.006), and the
     arrival order matters less than that.
   * **Still needed:** a map where the vanilla route itself is the one the twin gets wrong, so
     that avoiding the error means taking a DIFFERENT route (an edge or curb beside a clear
     path), rather than a map with one way up.

   **Curb detour (`heightmap/create_curb_detour.py`), 2026-09-29: window 0 cannot pick a route.**
   The map has a 0.15 m curb across the straight 10 m line and a clear way round its end. At that
   height the settle accepts every crossing pose (|pitch| ≤ 11.5°; 0.20 m is already blocked),
   and the cost-to-go routes straight over it: V(start) is the same as for a curb with no gap,
   against +1.6 (curb end at y = 1.0) or +1.1 (y = 0.4) for a forced detour.
   * **What happened:** every arm crossed the curb at every weight (paths 9.7 m, 2 repeats each).
     `nnflat` only slowed down: 10/10 and 30/30 took 10.2–10.8 s, 100/100 took 15–16 s, against
     vanilla's 9.6–10.2 s. 300/300 drove up to the curb face and stopped there for the rest of the
     25 s.
   * **The net is not the problem.** Crossing windows really do diverge more: true e_pos 0.06
     slow and 0.10 fast, against 0.01 before the curb. The charged excess ranks them the same
     way: 0.026 slow, 0.045 fast, 0 away from the curb.
   * **The horizon is the problem.** The charge appears only once the curb is inside the next
     1 s. By then a sideways shift of 1.2–1.8 m is out of reach, so slowing or stopping is the
     cheapest way to cut it. The cost-to-go, which does see the whole route, still points over
     the curb. Once the robot is at the face, `wmin` 0 (no reverse) leaves it no move that avoids
     the curb.
   * **So route choice needs the error before the robot reaches the obstacle.** Either charge
     windows 1–2 (9e), or put the net's error into the cost-to-go, as `planning/`'s gated lattice
     does with `lattice_learning`'s net, and keep window 0 for the local speed choice it does well.

### 9e. Later windows, only if 9d.6 asks for them

Decided with numbers measured on the Orin (`bench_inference`-style: the full net per patch,
eager and fused):

* **Full net per rollout window.** Exact. Eager torch is probably too slow even on the Orin,
  because what dominates here is memory-bound (the LayerNorm, the permutes, the padding). A fused
  runtime could fit at a reduced batch: TensorRT in fp16, enqueued on Warp's stream inside the
  capture, or `torch.compile` (Triton runs on the Orin but not on the 1050).
* **A smaller trunk.** `base_width` 16 has about a quarter of the convolution work. It needs a
  retrain and costs some accuracy.
* **A cached terrain code.** The trunk runs over the map rotated to each of n heading bins, and
  each rollout's code is interpolated from the grid at its pose. It is approximate, so it is the
  last resort. It also needs the trunk to ignore a uniform height offset, which it does not: the
  patch is relative to the wheel contacts at its own pose, which a map-wide pass cannot
  reproduce, and raising a patch by 5 cm moves the current net's prediction by 0.030 m /
  0.068 rad. A zero-sum first convolution would make it exact, at the cost of a retrain.

**The robot.** The ROS node lives in helhest_stack and cannot import this tree. How it loads the
hook and calls `update` is decided once step 10 works in simulation.
