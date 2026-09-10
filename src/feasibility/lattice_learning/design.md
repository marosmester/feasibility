# `ArcDivergenceNet` — what the lattice's arcs cost the robot that the planner cannot see

What we are building, in plain language, before any code exists. Sibling document to
`grid_learning_2/design.md`; read that one for the arguments about relief normalisation,
`ChannelLayerNorm`, the target transform and the mirror symmetry — they carry over and are not
re-derived here. This document is about the three things that are different: **the reference the
label is measured against**, **the input geometry**, and **the fact that the consumer is a value
iteration with 5–7 edges per node rather than a dense field.**

```
    learning/ (patch)   IN : (v, wz) + body patch 24x24 @ 0.25 m over [-1.5,4.5]x[-3.0,3.0]
                        OUT: (e_pos, e_rot)  --  ostrich vs. the KINEMATIC TWIN
                        how: flatten -> 3x256 MLP

    lattice_learning    IN : kappa (ONE scalar) + body patch 28x24 @ 0.125 m, 1 channel (relief),
                             over [-1.5,2.0]x[-1.5,1.5]
                             -- v = 0.6 m/s and L = 0.3 m are PINNED constants, not inputs
                        OUT: (e_pos, e_rot)  --  ostrich vs. the PLANNER'S OWN ARC
                        how: command-free conv trunk -> Conv2d geometry layer -> FiLM -> per-arc head
```

**Scope.** This is an idealistic first prototype and it is scoped as one deliberately: complete maps
only (§3c), one speed, one arc length, forward arcs only. The question it exists to answer is narrow
— *does a learned `d_hat` beat the hand-tuned `_step_gate_kernel` it would replace, on terrain where
both are well-defined* (§8) — and every simplification below is chosen to reach that answer, with the
cost of each one stated where it is taken.

The consumer is `helhest_stack/src/helhest/planning/lattice_solver.py`'s
`_relax_lattice_pose_kernel`, which today has **no term at all describing what happens during a
transition** — it checks whether every swept cell is an acceptable place to *stand*, at the source
heading, and charges arc length times a tilt penalty. `d_hat` would be the first quantity anywhere
in that planner that says something about the transit itself.

---

## 1. The reference: the arc, settled

### 1a. Why the arc and not the twin

`lattice_solver.py` imports `math`, `numpy` and `warp`. Nothing else. There is no simulator in it,
and the kinematic twin is never run to produce an edge. `_build_primitives` integrates a
constant-curvature curve on the host:

```python
    cth = th + dth_p * (float(s) - 0.5) / float(nseg - 1)
    x += (step / float(nseg - 1)) * math.cos(cth)
    y += (step / float(nseg - 1)) * math.sin(cth)
```

That curve **is** the planner's belief about the edge. So the error we want to charge the edge is
the error of *that* curve, not the error of `helhest_stack`'s kinematic twin. Labelling against the
twin (what every dataset in `learning/`, `grid_learning/` and `grid_learning_2/` currently does)
would measure a gap the planner never pays: the twin's grip-weighted `x_icr` and gravity drift
(`engine/step.py`) are real physics the arc does not have, but they are also physics the *router*
does not have, so the twin sits between the two things we care about and belongs in neither.

Three propagators, and we must be explicit about which pair the label spans:

| | what it is | who runs it |
|---|---|---|
| **arc** | ideal constant-curvature curve, pure host numpy | the router — `_build_primitives` |
| **twin** | `helhest_stack` kinematic model: grip-weighted ICR, slope drift, settle feedback | the executor (MPPI), and every existing dataset's `hstack` side |
| **ostrich** | full dynamics, exact friction cone, non-smooth contact | the reference |

**The label spans arc ↔ ostrich.** The twin is still recorded per trial (it is nearly free), but
only as a diagnostic that splits `d` into `arc→twin` and `twin→ostrich`, which tells us whether the
router's simplification or the twin's simplification dominates — a number we do not currently have
for any configuration.

### 1b. Why the settle still plays a role

The arc gives `(x, y, yaw)` and nothing else. But the planner's belief about the robot at the end of
an edge is not planar — `CostToGo` settles the robot at **every** lattice pose (`nx * ny * n_theta`
worlds in one `ForwardSimulator` batch, `target_wheel_omega.zero_()`, headings at bin centres) and
reads `(z, pitch, roll)` out of `sim.derived`. That settle is what `_feasibility_kernel` turns into
`blocked` and `graded_tilt`.

So the planner's full 6-DOF belief about an edge's endpoint is exactly:

```
    planar   (x, y, yaw)        <- the exact arc, integrated from the source pose
    vertical (z, pitch, roll)   <- helhest_stack ForwardSimulator, static settle at that planar pose
```

and that composite is the reference the label is measured against. This matters: a pure SE(2) label
would be blind to the single most informative failure mode the settle *can* express and the arc
cannot — the robot ending up pitched or rolled somewhere the arc thinks is flat. Measuring in SE(3)
against the settled reference charges exactly the residual: everything ostrich does that the arc
plus the settle together fail to predict.

> **Label.** `(e_pos, e_rot) = pose_error.se3_error(T_ref, T_ostrich(t1))`, with
> `T_ref = SE3(arc_endpoint_xy, settle_z, settle_pitch, settle_roll, arc_endpoint_yaw)`.

The error metric is the same one `learning/pose_error.py` uses — `T_err = T1^-1 @ T2`,
`pos = ||t_err||`, `rot = arccos((tr(R_err)-1)/2)` — **re-derived here, not imported** (§11a).

The settle is a **training-time** computation. At inference the network predicts `(e_pos, e_rot)`
straight from the patch and the command; nothing about the reference is needed at plan time.

### 1c. The arc origin, and what the quantisation is allowed to do

**Why this section exists.** §1a and §1b fixed *what* the label measures; this fixes *where the
measurement starts from*, which matters because the generator and the deployed planner know different
things about that starting pose. At generation time we can read ostrich's exact pose to machine
precision; at plan time `CostToGo` evaluates a lattice, so every state it can ask about is a **cell
centre** at a **bin-centre heading** — up to `cell/2` of position and `pi/n_theta` of heading error
from wherever the robot physically is. Train on exact poses, query on rounded ones, and that gap is a
silent distribution shift nothing in the validation loss will report.

The fix is to decide, per quantity, which pose it is anchored to — and the answer differs between the
label and the input. Three poses are in play per trial and they must not be conflated:

```
    p_spawn    where the robot is dropped                        (nominal)
    p_t0       ostrich's ACTUAL (x, y, yaw) when the arc begins  (after settle + warm-up)
    p_belief   the pose the planner THINKS it is at              (cell centre, bin-centre heading)
```

* **The arc is integrated from `p_t0`.** The label is then pure dynamics divergence over the arc,
  with no quantisation bias baked in.
* **The patch is sampled at `p_belief`.** At deployment the planner can only offer a cell centre and
  a bin-centre heading; feeding the true pose at training and a quantised one at inference is a
  distribution shift we would not detect. So the generator jitters:

  ```
      p_belief = p_t0 + (dx, dy, dyaw),   dx, dy ~ U(-cell/2, cell/2),  dyaw ~ U(-pi/n_theta, pi/n_theta)
  ```

  with `cell` and `n_theta` the router's, not the patch's. The network therefore learns
  `E[divergence | what the planner can observe]` — the quantisation enters as **input noise**, which
  is what it physically is, rather than as a bias in the target.
* **Cell quantisation of the endpoint is NOT learned.** `prim_dc`/`prim_dr` round the arc endpoint to
  an integer cell and `prim_heading` floor-bins the heading; that discrepancy is deterministic,
  closed-form, and computable by the planner in one line. Spending network capacity on it would be
  fitting arithmetic. If the planner wants it charged, it can add `|exact_endpoint - cell_centre|`
  itself.

In one line: **the label is anchored to the true pose, the input to the pose the planner believes
in.** Quantisation is then modelled as what it actually is — uncertainty about where the robot is
when the arc starts — and the network's prediction is an expectation over that uncertainty, which is
exactly the quantity a lattice edge cost should carry.

---

## 2. Warm start: pinning `v`, not ignoring it

Every existing dataset starts from complete rest — `comparator/common.py:418`
(`self.target_velocities.zero_()`, "all worlds settle at rest") and `common.py:505`
(`sim.init_current_wheel_omega.zero_()`). Every label is therefore `d(pose, u | v0 = 0, omega0 = 0)`:
one slice of a velocity-dependent function, and the slice where much of the measured error is
**acceleration lag** — a terrain-independent bias that would swamp the collision signal this network
exists to find.

The lattice has no velocity dimension and adding one multiplies a dense `[ny, nx, n_theta]` structure
by `n_v`. So we do not add one; we **pin** it. Every arc in the primitive table shares one length
(`prim_cost` is initialised to `step` for all five), so with a nominal speed `v_nom` every edge is
both *traversed* and *entered* at that speed. `v_nom` becomes a design constant of the router in
exactly the way `mu = 0.8` already is in `costtogo.py`.

**How the warm start is implemented, with no change to `HelhestBatchSimulator`.**
`replay_graph_batch` settles at zero command for `settle_steps` uncaptured steps, then launches one
captured step `T` times reading `self._setpoints_wp[step, world]`. So prepending `W` warm-up rows to
the setpoint array and slicing the first `W` rows off the returned logs gives a warm start for free —
the extra steps are captured and batched, i.e. cheap.

```
    setpoints = [ warmup (W rows at the trial's own twist) | record (T rows, same twist) ]
    t0 = row W-1's pose,   t1 = last row's pose,   T = round(L / (v * dt_ostrich))
```

### 2a. The ostrich timestep is re-pinned to `2.5e-2`

`generate_dataset_utils.py:304` warns when the command hold is not an exact multiple of *both*
timesteps, and it is right to: a hold that misses an integer step in either sim biases every label by
up to one step of travel. Ostrich's default `dt = 3e-2` (`examples/conf/simulation/helhest.yaml`) and
the twin's `DT = 0.1` (`helhest_stack/src/helhest/dynamics.py:19`) admit only multiples of **0.3 s** —
which is why `learning/` uses 2.4 s, and why **this dataset's `ARC_LEN / V_NOM = 0.5 s` does not
fit**: ostrich would round to 17 steps = 0.51 s = 0.306 m of nominal travel against an arc endpoint
drawn at 0.300 m — a **+6 mm floor on every row**, sitting exactly where §8 asserts flat ground must
read zero.

Fix the timestep to suit the arc, not the arc to suit the timestep. Any `dt = 0.1 / k` divides both
0.5 s and the twin's 0.1 s exactly; near the current value that leaves `1/30` (coarser,
non-terminating) and **`2.5e-2`** (finer, 20 steps) — take the latter, since finer is the safe
direction in a Newton contact solve and 20% more steps is nothing against what the short arc already
saved.

```
    OSTRICH_DT = 2.5e-2 s
    arc   0.5 s / 0.025 = 20 ostrich steps   exact
          0.5 s / 0.1   =  4 twin steps      exact
    L_eff = V_NOM * T * OSTRICH_DT = 0.300 m == ARC_LEN,  residual exactly zero
```

More generally this makes every admissible arc length a multiple of `V_NOM * OSTRICH_DT = 0.015 m`,
so a future change to the router's `step` stays on the grid as long as it is a multiple of 15 mm.

**`examples/conf/simulation/helhest.yaml` is NOT edited** — it is shared with `comparator/`,
`learning/`, `grid_learning/` and, decisively, `submodule_test/`, whose committed baseline is a
*characterization* test: changing ostrich's dt underneath it would make every scenario report "moved"
at once, destroying its ability to distinguish a real submodule regression from this change. The
generator overrides it in code instead, records it in the root attrs, and asserts the divisibility it
depends on:

```python
OSTRICH_DT = 2.5e-2  # s. Chosen so ARC_LEN/V_NOM = 0.5 s is an exact multiple (20 steps) AND
# helhest_stack's DT = 0.1 is too (4 steps); helhest.yaml's own 3e-2 divides neither. Overridden
# here rather than there so submodule_test's baseline and the existing learning/ datasets stay
# reproducible -- see design.md section 2a.
cfg.simulation.target_timestep_seconds = OSTRICH_DT
```

**Two consequences.** `W` and `settle_steps` are *durations*, not step counts, so both are re-expressed
at the new dt (~0.36 s each → ~15 steps, with `W` still measured per §2 rather than converted
blindly). And §8's ostrich repeat-scatter figure must be re-measured **at `2.5e-2`**: the 12.7–31.2 deg
`submodule_test/` recorded is a `3e-2` number, and the scatter comes from a contact bifurcation the
timestep directly conditions.

`W` is a **measured** quantity, not a guessed one, and is fixed the same way `DEFAULT_SETTLE_STEPS`
was: run a pilot batch, find the step by which body speed is within 2% of `v_nom` and yaw rate within
2% of `v_nom * kappa`, take that with ~1.7x margin. Expect order 12–18 steps at
`OSTRICH_DT = 2.5e-2` (§2a), i.e. ~0.3–0.45 s; record the measurement in the constant's comment. On the twin side, `sim.init_current_wheel_omega` is
assigned the steady wheel speeds instead of zeroed, and the same prefix is applied and sliced, so
both sims enter the arc in the same regime.

---

## 3. The input patch — re-derived, not inherited

`learning/terrain_patch.py`'s four choices are all correct and all kept: **body-aligned** (so yaw
drops out by construction, with a literal rotation-invariance assert), **relief relative to the wheel
contacts**, **divided by `WHEEL_RADIUS`** (a dimensionless step height, no fitted per-cell
normaliser), **explicitly not translation-invariant**. What changes is the extent, because
`learning/`'s extent was sized for a 2.4 s / up-to-3.6 m rollout and a lattice arc is 0.3–0.6 m.

### 3a. Extent from the causal support

`RobotParams`: `half_track = 0.365`, `rear_offset = 0.75`, `wheel_radius = 0.35`,
`min_turn_radius = 0.5`. `CostToGo` defaults to `step = 0.3`; `demos/costfield.py` uses
`step = max(0.3, 1.6 * ccell)`, which at `cell = 0.06` and `lat_coarsen = 6` gives `0.576`. Take
`L <= 0.6 m` and `|kappa| <= 1/0.5 = 2 /m`, so `dtheta = L*kappa <= 1.2 rad`.

Worst-case reach of the wheel envelope, measured in the **source** body frame:

| direction | worst case | value |
|---|---|---|
| behind | rear wheel at t0: `rear_offset + wheel_radius` | **-1.10 m** |
| ahead | endpoint origin `sin(dth)/kappa = 0.466`, + front wheel `0.365*sin(dth)`, + `wheel_radius` | **+1.16 m** |
| lateral | endpoint origin `(1-cos dth)/kappa = 0.319`, + outer front wheel `0.365*cos(dth)`, + `wheel_radius` | **+-0.80 m** |

(The rear wheel swings to `y = 0.319 - 0.75*sin(1.2) = -0.38` on a left turn — outward, on the
*opposite* side from the turn. That is the lateral-collision mode `compare_box_obstacles.py` exists
to produce, and it is why the patch cannot be forward-only.)

> **`PatchSpec(x_min=-1.5, x_max=2.0, y_min=-1.5, y_max=1.5, cell=0.125, reference="wheels")`**
> → **28 x 24 = 672 cells** spanning 3.5 m x 3.0 m.

Margins: 0.40 m behind, 0.84 m ahead, 0.70 m lateral — enough for `L` up to ~1.0 m and for the
settle's support to reach a little past the contacts.

The extent is capped deliberately, on `grid_learning_2` §5b's argument ("one more block starts
admitting a second, causally unrelated obstacle"). Terrain outside this box cannot influence a 0.6 m
arc, so every cell of it is capacity spent on something that can only produce spurious correlation.

**But the receptive-field cap does not carry over.** In `grid_learning_2` the label was a *field* and
each cell's label was causally local, so the trunk's RF had to be capped at 27 px. Here one patch
produces one label that legitimately depends on the whole patch — there is no locality constraint
inside the patch, and the trunk's final RF (23 px = 2.875 m, §5b) is allowed to approach the patch
size. **The extent, not the RF, is what enforces causality in this design.** That is the single most
important difference from the sibling document and it is easy to get wrong by analogy.

### 3b. Resolution: 0.125 m

Same argument as `grid_learning_2` §3d, one step finer than the arc needs: 0.125 m resolves the
0.35 m wheel at 2.8 px and localises a box edge to ±6 cm, while making 28 and 24 both divisible by 4
so two ordinary stride-2 stages work. `learning/`'s 0.25 m is too coarse here — the phenomenon is
whether a rim clips an edge, and the arc is only ~4 cells long at 0.125 m as it is.

Sampling is bilinear point sampling of the source grid (`HeightMapReader.sample`, or in the planner
the device elevation), which **clamps outside the grid rather than raising** — same caveat
`terrain_patch.py` carries.

### 3c. One channel, and the assumption that buys

```
    ch 0   relief   (h - h_ref) / WHEEL_RADIUS,  h_ref = mean height under the three wheel contacts
```

That is the whole input tensor: `[B, 1, 24, 28]`.

**The assumption it rests on: the map is complete.** A single-channel patch says "every cell of this
tensor is real terrain," and the prototype only ever sees maps for which that is true — the
`assets/` heightmaps at generation time, and a fully-observed `GridMap` at evaluation time.

That is a real restriction and it should be named rather than discovered. At deployment the patch is
gathered from a perception `GridMap` built incrementally from LiDAR, where cells behind an obstacle,
beyond sensor range, or not yet driven past **have no height at all**, and whatever constant is
written into them is a lie the network cannot detect. The same problem already bit the hand-written
gate this design would replace — `costtogo.py`'s `_local_step_kernel` carries the post-mortem: a
blind-cell fill (a constant, e.g. 0.0) *"reads as a real step wherever the ground sits away from that
constant, and the map frontier gates off as a closed ring."*

A convolution is more susceptible than that prominence kernel, not less: on ground at −0.4 m, a 0.0
fill across half a patch presents as a clean 0.4 m wall with a straight edge — the feature a trained
trunk fires hardest on. The predicted failure is a large `d_hat` along every map frontier, i.e. a
planner that refuses to route into unexplored ground. `CostToGo.compute` already threads a `measured`
mask through for exactly this reason, and `demos/navigate_partial_view.py` is where it would show up.

**Deferred, with the cost stated.** The fix is a second binary channel (`1` = real data, `0` = fill,
with relief written as exactly `0.0` there — never NaN, per `grid_learning`'s "`0 * NaN` is NaN"
lesson) plus synthetic sensor-shadow dropout as a training augmentation. Adding it later costs a
**full dataset regeneration**, which is the expensive step in this pipeline — so this is a real debt,
not a free deferral. It is the right debt for a prototype whose purpose is to answer "does `d_hat`
beat `_step_gate_kernel` on complete maps at all", and it comes due before any partial-map evaluation.

Two consequences meanwhile: the input channel count is a module constant rather than a literal 1, so
the schema change is a data change and not an architecture change; and **poses whose patch overhangs
the map are excluded at generation** (§7b's `valid` flag) rather than silently clamped, so the border
case cannot leak into training as fake flat ground.

---

## 4. The command: one scalar, `kappa`

### 4a. What actually varies per edge — only curvature

Read `_build_primitives` carefully before designing the input, because it is easy to assume more
freedom than the lattice has:

```python
    turns = [-step/R, -step/(2R), 0.0, step/(2R), step/R]     # dtheta over the step
    prim_cost = np.full((n_theta, n_prim), step, np.float32)  # <- EVERY arc, one length
```

`turns` is a list of **heading changes**, not of lengths, and the integrator lays down exactly
`nseg-1` segments of `step/(nseg-1)` each. So:

> **All five forward primitives have identical length `L = step`. They differ only in curvature.**
> `kappa_p = dtheta_p / step`, taking `{-1/R, -1/(2R), 0, +1/(2R), +1/R}` = `{-2, -1, 0, 1, 2}` /m at
> `R = min_turn_radius = 0.5`. The two pivots are not arcs at all — zero translation, and their
> `prim_cost` is a hand-set m-equivalent penalty, not a length.

What varies is `step` **between solver instances**: `CostToGo` defaults to 0.3, `LatticeValueSolver`
to `2.0 * resolution`, and `demos/costfield.py` computes `max(0.3, 1.6 * ccell)` → 0.384 at
`lat_coarsen = 4`, 0.576 at 6. And `v` does not appear in the lattice at all; §2 pins it.

Hence the query the planner actually makes:

```
    d_hat( patch , kappa )          kappa: 5 values, the only thing that varies per edge
              with  v = V_NOM       a router constant, like mu = 0.8 in costtogo.py
                    L = step        a per-RUN constant, fixed when CostToGo is constructed
```

This is a deliberately idealistic prototype, so both constants are **pinned rather than carried**.

**So the command input is one scalar.** `v` and `L` are constants of the configuration, not features:
they are pinned at generation, recorded in the checkpoint, and **asserted at load** (§4c). The
network's command port is `kappa` alone.

```
    V_NOM   = 0.6    m/s    every trial generated at this speed; the router assumes it
    ARC_LEN = 0.3    m      = CostToGo's default `step`, which navigate_partial_view.py uses
    kappa   ~ U(-2.0, 2.0)  1/m   the only sampled command axis
```

### 4b. Encoding

The command encoder is fed `(kappa/KAPPA_MAX, |kappa|/KAPPA_MAX, sign(kappa))` — three numbers from
one. `|kappa|` matters for the same reason `grid_learning_2` carried `|wz|` (the magnitude sets how
far the rear rim sweeps, the sign sets which way), and handing the net the split explicitly saves it
from having to build a V-shape out of a monotone input.

`kappa`, not `omega`, because `kappa` is the geometry invariant: at inference it is read straight off
the primitive table as `dtheta_p / step` with no unit conversion, and it stays meaningful if `V_NOM`
is ever re-pinned. The commanded body twist the generator actually issues is `omega = V_NOM * kappa`,
held for `T = ARC_LEN / V_NOM = 0.5 s`.

**Sample `kappa` continuously, not at the five lattice values.** `grid_learning_2` §7b's argument:
five discrete levels confound the command with everything else in the row, and a continuous draw is
free. It is also what lets the same checkpoint survive a change to `min_turn_radius` or `n_theta`,
neither of which changes the physics the net learned.

**Check before generating:** at `v = 0.6, kappa = 2.0` (`omega = 1.2 rad/s`) the commanded wheel
speeds are `v +- omega*half_track = 0.6 +- 0.438`, i.e. 0.16 and 1.04 m/s → 0.46 and 2.97 rad/s.
Confirm the wheel servo (`K_P` in `comparator/common.py`) tracks that before committing GPU time.

**Pivots are out of scope for the prototype.** `CostToGo` defaults to `pivot_cost = 0.0`, so the
point-turn primitives do not exist unless asked for. They are a different regime anyway — `v = 0`,
zero translation, `prim_cost` a hand-set m-equivalent rather than a length — and the `v = 0` case is
what `grid_learning`/`grid_learning_2` already cover. If `pivot_cost > 0` is enabled later, those two
primitives get `d_hat = 0` (charged nothing) until a `v = 0` sub-population is added to the dataset,
and that exemption is recorded in the checkpoint so it cannot be forgotten.

### 4c. What pinning costs, and the assert that contains it

Two things are given up, and both are real:

* **A change to `step` silently invalidates the checkpoint.** `step` is a runtime argument, ranging
  0.3–0.576 m across the call sites listed in §4a. A net trained at `L = 0.3` and queried on a 0.576 m
  arc keeps predicting, at the wrong arc length, with no error.
* **`∂d/∂v` is not measured.** The sharpest objection to this whole design is that a quasi-static
  router is asking a dynamics question, and pinning `v` at generation means we never get a number for
  how much that costs. §10 keeps that as an unquantified limitation rather than a bounded one.

The first is contained cheaply and must be:

> **`infer.py` refuses to load a checkpoint whose `arc_len` does not match `solver._step`, or whose
> `min_turn_radius` does not match the robot's, to within 1e-6.** Same for `V_NOM` against the
> router's assumed speed. A hard failure at setup, not a wrong number at every edge.

That assert is what makes pinning safe rather than merely simpler, and it is the reason the constants
belong in the checkpoint rather than in a module-level literal on both sides.

The second is not containable, only deferrable: recovering it means regenerating with a `v` band. Do
that only if the prototype survives §8's `_step_gate_kernel` comparison — there is no point measuring
the velocity sensitivity of a signal that turns out not to beat the incumbent.

---

## 5. The network

### 5a. Answering the architecture question directly

> **Convolution for the terrain features. A single full-window `Conv2d` for the geometry.
> 1x1 / fully-connected for the command. No global pooling anywhere.**

Three reasons, each of which rules out one of the obvious alternatives:

* **Not a plain MLP on the flattened patch.** A box edge looks the same wherever it appears in the
  patch; an MLP has to learn a detector for it separately at each of 672 positions. `learning/
  model.py`'s own docstring already concedes this: *"a flattened patch through a plain MLP is the
  deliberately simplest terrain encoding, not the most sample-efficient one"*, and *"at 578 inputs
  the first Linear alone is 148k params against a few thousand training rows."* Weight sharing is the
  whole point of using a conv here.

* **Not a fully-convolutional net ending in a global pool.** The label is **not** translation-invariant
  in the body frame: the body frame has a privileged origin and the wheels sit at fixed places in it.
  An obstacle 0.4 m behind-left decides whether the rear rim clips on a left turn; the same obstacle
  1.2 m ahead is irrelevant to a 0.5 m arc. Global average pooling would dilute a 5%-of-patch box by
  20x and destroy the position information entirely; global max pooling would keep the magnitude and
  throw away the position. So the spatial collapse must be **position-aware**, which means a
  full-window kernel, i.e. a flatten-then-Linear written as a convolution (§5c).

  This is the opposite of `grid_learning_2`, whose *output* was a field and whose head had to stay
  spatial and per-cell. Here the output is a scalar pair and the collapse is the whole job — do not
  carry that document's head design across by analogy.

* **The command must not enter the trunk.** Not for `grid_learning_2` §2's causal reason — there is
  one label per trial here, so there is no cross-cell leak to prevent — but for **caching**. All 5–7
  primitives at a lattice pose share one source pose, hence one patch, hence one trunk pass. A
  command-free trunk makes the marginal cost of an additional primitive `0.42 MMAC` against the
  trunk's `21.6 MMAC` — a **50x** saving on the per-edge query, and the difference between "one pass
  per pose" and "seven". This is the single most consequential decision in this document, and it is
  the same factorisation `N(H) -> c_hat(s)`, `d_hat(s,u) = head(c_hat(s), phi(u))` that
  `grid_learning_2` §1a argues for, arrived at from a different direction.

### 5b. Top level

```
  patch [B,1,24,28]                                   command kappa [B]
  ch0 = relief                                                 |
        |                                                      v
        v                                          +---------------------------+
  +===========================================+    | (k/K, |k|/K, sign k) [B,3]|
  |  TERRAIN TRUNK -- 6 conv blocks           |    | Linear 3->64 + SiLU       |
  |  NO conditioning of any kind              |    | Linear 64->64             |
  |  replicate pad, ChannelLayerNorm, SiLU    |    +---------------------------+
  +===========================================+
        |  [B,96,6,7]   (stride 4, RF 23 px = 2.875 m)         |  e [B,64]
        v                                                      |
  +-------------------------------+                            |
  | Conv1x1 96 -> 24              |   channel squeeze before   |
  +-------------------------------+   the full-window kernel   |
        |  [B,24,6,7]                                          |
        v                                                      |
  +-------------------------------+                            |
  | Conv2d 24 -> 256, k=(6,7)     |   THE GEOMETRY LAYER --    |
  | valid pad                     |   position-aware collapse  |
  +-------------------------------+   (= flatten + Linear)     |
        |  c [B,256,1,1]                                       |
        |  the terrain code; control-free, cached per pose     |
        +------------------- FiLM -----------------------------+
        |   gamma, beta = Linear(64 -> 512)
        |   h = c * (1 + gamma) + beta
        v
  +-------------------------------+
  | LayerNorm + SiLU              |
  | Linear 256->256 + SiLU        |
  | Linear 256->256 + SiLU        |
  | {head_e_pos, head_e_rot}      |
  +-------------------------------+
        |
        v
  y_hat [B,2] in log1p / standardised space
```

Trunk, block by block. `s` = stride; RF in input pixels, x0.125 for metres.

| # | block | s | C_in -> C_out | out (h x w) | m/cell | RF (px) | RF (m) |
|---|-------|---|---------------|-------------|--------|---------|--------|
| 1 | conv3 | 1 | 1 -> 32       | 24 x 28     | 0.125  |  3      | 0.375  |
| 2 | conv3 | 1 | 32 -> 32      | 24 x 28     | 0.125  |  5      | 0.625  |
| 3 | conv3 | **2** | 32 -> 64  | 12 x 14     | 0.250  |  7      | 0.875  |
| 4 | conv3 | 1 | 64 -> 64      | 12 x 14     | 0.250  | 11      | 1.375  |
| 5 | conv3 | **2** | 64 -> 96  | 6 x 7       | 0.500  | 15      | 1.875  |
| 6 | conv3 | 1 | 96 -> 96      | 6 x 7       | 0.500  | **23**  | **2.875** |

`ChannelLayerNorm` (normalise over channels at each position), **not** `GroupNorm` / `BatchNorm` /
`InstanceNorm`, for `grid_learning_2` §5b's reason and for a second one that is specific to this
design — see §5d. Keep that document's autograd RF probe; it is what caught the substitution there.

`(1 + gamma)` rather than `gamma` so an untrained net starts as a plain terrain→error map and learns
the command dependence from there. FiLM rather than concatenation for `grid_learning_2` §5d's
arguments (a concat head's first layer is *additive* in the command, but "how much does this terrain
feature matter" is *gated* by how far the rim sweeps; and **one** command scalar against 256 terrain
channels is a gradient the concat head can quietly drop — the imbalance is worse here, not better).
`--head-fusion concat` stays as the honest control.

Parameter count:

| part | params |
|---|---|
| conv trunk blocks 1-6 | ~206k |
| Conv1x1 96 -> 24 | 2.3k |
| geometry layer `Conv2d(24, 256, k=(6,7))` | 258k |
| command encoder `Linear(3->64->64)` | 4.4k |
| FiLM `Linear(64 -> 512)` | 33k |
| head (256->256->256->2) | 132k |
| **total** | **~636k** |

Larger than `grid_learning_2`'s 290k, and that is fine: this dataset can be an order of magnitude
bigger (§7c) because one trial is ~35 captured steps rather than ~80, and because `chunk` can be
raised far past 128. `base_width` (32) remains the single scaling knob.

### 5c. Why the geometry layer is written as a `Conv2d`, not a `Linear`

`nn.Linear(6*7*24 -> 256)` on the flattened feature map and `nn.Conv2d(24, 256, kernel_size=(6,7))`
with valid padding are **the same arithmetic**. Writing it as the convolution costs nothing and buys
the whole of §6b: applied to a patch it produces a `1x1` output and behaves exactly like the MLP
formulation; applied to a *whole rotated elevation map* it produces a dense field of 256-d terrain
codes, one per output position, with the same weights. The network is fully convolutional by
construction, and patch mode is the single-window case of map mode.

Write it this way from the first commit. Retrofitting it later means re-deriving the index order of
the flatten, which is exactly the class of bug `grid_learning_2` §3 spent a section eliminating.

### 5d. The norm choice is what makes the two modes agree

In patch mode a norm with spatial extent pools over the 24x28 patch. In map mode the same norm pools
over the whole map. Those are different numbers, so the patch↔map equivalence — the entire basis for
§6b's 40x speedup — **holds if and only if every normalisation is per-position**. `ChannelLayerNorm`
is per-position over channels and is identical in both modes; `GroupNorm`, `BatchNorm` and
`InstanceNorm` are not.

This gives a free, exact unit test: for a random map and a random pose,
`map_mode(map)[i, j]` must equal `patch_mode(patch_at(i, j))` to float tolerance. It is stronger than
`grid_learning_2`'s RF probe because it is an equality rather than a bound, and it fails loudly the
moment someone "improves" a norm.

---

## 6. Batched inference inside `CostToGo`

### 6a. Phase 1 — per-pose patches

One `d_hat` per `(row, col, heading, primitive)`. The trunk runs once per `(row, col, heading)`; the
head runs `n_prim` times on a 256-vector.

Sizing against `demos/navigate_partial_view.py` (`lat_coarsen = 4`, `n_theta = 24`, `cell = 0.06` →
router cell 0.24 m) on a 64 x 64 router grid:

```
    poses          64 * 64 * 24                        = 98,304
    trunk          98,304 * 21.6 MMAC                  = 2.12 TMAC  = 4.2 TFLOP
    heads          98,304 * 7 * 0.42 MMAC              = 0.29 TMAC  = 0.6 TFLOP
    patch tensor   98,304 * 672 * 1ch * 2B (fp16)      = 132 MB     <- tile it
    d_hat output   64*64*24*7 * 4B                     = 2.75 MB
```

Honest statement: **at fp32 this is the dominant cost of the whole planner**, order 200 ms against
single-digit milliseconds for the settle plus value iteration. It is not free and the design must not
pretend otherwise. What makes it survivable:

* It runs at **routing cadence**, not control cadence, and **outside** `CostToGo`'s captured graph.
* fp16 / TF32 takes it to roughly 50 ms.
* `blocked` is already computed before the divergence pass, and a blocked pose holds `V = +inf`
  regardless of `d_hat` — **skip those poses entirely.** Typically 30–50% of the lattice.
* Process in tiles of ~16k poses so the patch tensor stays under ~25 MB.
* Fallback if it still does not fit: compute `d_hat` on a 2x coarser `(row, col)` sub-lattice and
  broadcast. Divergence is smooth at the 0.24 m scale; this is a 4x cut for a defensible loss of
  spatial detail, and it is a knob rather than a redesign.

**The patch gather must be a Warp kernel, not numpy.** `helhest_stack/CLAUDE.md` §6 is unambiguous —
"bulk/data-parallel stages are `wp.array` + `@wp.kernel`, never numpy" — and a per-replan
device→host→device round trip of 132 MB is precisely the failure it names. Rotate-and-bilinear-gather
straight from the device elevation into a `wp.array`, then `wp.to_torch` (zero-copy) into the net,
then `wp.from_torch` back.

### 6b. Phase 2 — the map-level pass

Because the trunk is command-free and fully convolutional (§5c, §5d), the terrain code for *every*
pose at a *fixed heading* is one convolutional pass over the whole map rotated by `-theta_t`. That
replaces 98,304 patch passes with 24 map passes:

```
    per heading, on a 256x256 relief map:   ~2.1 GMAC
    x 24 headings                           = 50 GMAC = 100 GFLOP     ~40x cheaper
```

The catch is registration: the trunk's total stride is 4, so at 0.125 m input the code lattice has a
0.5 m pitch while the router runs at 0.24–0.36 m. Two ways out, both standard, both deferred:
dilate the last stages (a trous) for a dense output, or accept a 0.5 m divergence lattice and
broadcast — which is fallback (c) of §6a arriving by a different road. Either way the **weights are
the same weights**; map mode is a deployment optimisation of a network trained on patches, and §5d's
equality test is what certifies it.

Phase 1 first, because it trains and infers with identical geometry and therefore has no
train/deploy mismatch to debug. Phase 2 only if Phase 1's latency turns out to matter.

### 6c. Not in the prototype: a quantile head

Named so it is a decision rather than an omission. One extra output channel trained with pinball loss
at `q = 0.9` would give a conservative `d_hat` — the divergence exceeded 10% of the time over the
entry-state distribution — which is what a planner should charge when it cannot know `v0`. It is out
of scope here because with `v` pinned there is no entry-state *band* left to take a quantile over: the
residual spread is ostrich's own non-determinism plus §1c's belief jitter, a noise floor rather than a
risk distribution. It becomes meaningful the moment the `v` band comes back (§4c), and not before.

---

## 7. Integration and data

### 7a. The change to `lattice_solver.py` — about fifteen lines

`_relax_lattice_pose_kernel` gains one array and one scalar:

```python
    divergence: wp.array(dtype=wp.float32, ndim=4),   # [h, w, n_theta, n_prim], metres
    divergence_weight: wp.float32,                    # dimensionless planner gain, sibling of tilt_weight
```

and the cost line becomes

```python
    arc = prim_cost[t, p]
    if ns > 0:
        arc = arc * (1.0 + tilt_weight * tsum / float(ns))
    arc = arc + divergence_weight * divergence[r, c, t, p]
    best = wp.min(best, arc + dist_in[nr, nc, prim_heading[t, p]])
```

Four things to note:

* **Indexed at the source `(r, c, t)`**, like `prim_dr`/`prim_cost`/`sweep_*` and like the
  `blocked`/`tilt` gathers, which the kernel already takes at the pose's own heading `t`. Consistent
  with everything else in the relaxation; no new convention.
* **Units are metres**, so it adds to arc length. The producer collapses the two heads into one
  scalar before writing: `rho = e_pos + LEVER * e_rot`, with
  `LEVER = rear_offset + wheel_radius = 1.10 m`, the rear-wheel lever arm — derived from
  `RobotParams`, not a magic constant. Collapsing on the producer side keeps the kernel a single
  extra load.
* **Grade, do not prune.** A hard `d_hat > gate -> blocked` can disconnect the graph, and a
  disconnected value iteration fails *silently*: `V = +inf` over a region, indistinguishable from
  "the goal is genuinely unreachable". The gate flag can exist; it must default to off. (This is the
  opposite of the right answer for a Hybrid A* consumer, where dropping a successor is benign and
  total infeasibility surfaces as an explicit empty-open-set failure. Do not carry the reasoning
  across.)
* **`divergence_weight = 0.0` must be bit-identical to today.** Default it to zero and the feature
  ships dark.

`CostToGo` gains a `set_divergence(d: wp.array)` that copies into a stable owned buffer the captured
graph reads, exactly as `_elev_in` / `_measured_in` are handled today. **helhest_stack stays
torch-free**: it consumes a `wp.array` it is handed and knows nothing about how it was produced. The
producer lives in `feasibility/lattice_learning/`, which is where torch already is.

### 7b. Dataset schema

`outputs/dataset_arc_*.h5`, written through `comparator.provenance.write_comparison` so the terrain
grids and both submodules' git SHA/dirty state come along and cannot drift.

Root attrs, of which the first three are the **pinned constants `infer.py` asserts against** (§4c):
`v_nom`, `arc_len`, `min_turn_radius`; then the `patch_*` spec (via a local `patch_spec_to_attrs`),
`kappa_range`, `warmup_steps`, `settle_steps`, `xy_jitter`, `yaw_jitter`, `router_cell`, `n_theta`,
`map_glob`, `mu`, `k_turn`.

Per-variant:

| field | shape | what |
|---|---|---|
| `spawn_pose` | [n,3] | nominal drop pose |
| `t0_pose` | [n,3] | ostrich's actual (x,y,yaw) at warm-up end — the arc origin |
| `belief_pose` | [n,3] | the jittered pose the patch was sampled at (§1c) |
| `patch` | [n, 24*28] | baked in at generation, as `learning/generate_dataset_body_centered_patch.py` does |
| `kappa` | [n] | the command — the only sampled one |
| `arc_end_pose` | [n,3] | exact arc endpoint integrated from `t0_pose` |
| `ref_pose` | [n,7] | `arc_end_pose` + the settle's (z, pitch, roll), as SE(3) x,y,z,qx,qy,qz,qw |
| `valid` | [n] bool | see below |
| `swept_clear` | [n] bool | would `_relax_lattice_pose_kernel`'s sweep test have passed |

`y = (e_pos, e_rot)` is **computed by the loader** from `ref_pose` and `ostrich/pose[-1]`, not stored
— the same stance `learning/custom_dataset.py` takes, so a change to the error metric does not
invalidate generated files. `hstack/*` is written as usual and used only for the arc→twin /
twin→ostrich diagnostic split of §1a.

`valid = False` for: a non-finite ostrich pose; a settle at the arc endpoint that itself failed
(`residual > resid_tol` or `clearance < clear_margin`) — those poses are `blocked` at inference and
`d_hat` is never queried there; a spawn footprint on an obstacle
(`generate_dataset_utils.obstacle_height_threshold`'s filter, re-derived locally); an implausible
displacement, the analogue of `grid_learning/remask_dataset.py`'s `MAX_SPAWN_DISPLACEMENT`, here
`||t1 - t0|| > 3 * ARC_LEN`; and — per §3c, since there is no `measured` channel to say so — **any
pose whose patch would overhang the heightmap**, i.e. within `max(|x|, |y|)` of the patch corner from
the grid edge. That last one is a filter the two-channel version would not need, and it is the
cheapest possible stand-in for it.

**`swept_clear` is a reporting split, not a training filter.** Train on everything; report on the
`swept_clear` subset, because that is the query distribution the planner actually produces and
reporting on the wrong subset flatters the model. The rows that fail it are still useful — the
settle's `blocked` is heading-quantised and conservative, and its boundary is exactly where `d_hat`
should be earning its keep.

### 7c. Generation

Maps: `assets/large_box_random/<seed>/` as the bulk (many independently-placed rectangles, built for
exactly this starvation problem), plus `assets/rough/`, `assets/mended/` (a box riding on rough
ground) and `assets/surface/` for slope-and-turn cases the box series cannot produce. Per
`grid_learning_2` §1c, **terrain diversity is the binding constraint on generalisation** — more
trials on the same maps buy less than more maps.

Poses: continuous, not on a lattice. `spawn_mode="continuous"`-style rejection sampling against the
obstacle-height filter and against the patch-overhang filter above, with the sampling square sized
against the loaded map rather than a fixed `SPAWN_LIMIT` (these arcs are short, so a much larger
fraction of each map is usable than `learning/` assumed).

Cost per trial, at `OSTRICH_DT = 2.5e-2` (§2a): `settle_steps = 15` amortised per chunk (0.375 s,
uncaptured), `W ≈ 15` warm-up (0.375 s), and `T = ARC_LEN/(V_NOM*OSTRICH_DT) = 20` recorded steps
(0.500 s exactly) — about **35 captured steps / 1.25 s of simulated time per trial**, against
`learning/`'s 80 captured steps at 2.4 s (92 steps / 2.76 s all in), i.e. roughly 2.3x cheaper per
trial despite the finer timestep. **Raise `chunk` hard.** `learning/`'s 128 was chosen when the settle was
~35% of ostrich time; here the settle is amortised over a much larger batch and ostrich is documented
at 65536 worlds with an adjoint tape. At `chunk = 2048`, 200k trials is ~100 chunks of ~29 steps.
That is what makes a ~636k-parameter network defensible.

Budget: order **200k trials over >= 200 maps**, one `kappa` per trial, drawn i.i.d.

**Note the SNR consequence of a short arc.** 0.5 s at 0.6 m/s is 0.3 m of travel, so on benign
terrain the true `(e_pos, e_rot)` will be small and the distribution heavily concentrated near zero —
the informative rows are the minority where the envelope actually meets an obstacle. That is the
signal we want, but it means: report the top-decile RMSE (§8) as the headline rather than the
aggregate, and if the labels turn out to be dominated by ostrich's own run-to-run scatter rather than
by terrain, the first knob is to raise `step` in the **router and the dataset together** — not to
lengthen the trial while leaving the planner's arc where it is.

Augmentation: the y-mirror is exact, by `grid_learning_2` §7c's argument (robot and world geometry,
not command shape). At p = 0.5:

```
    patch  ->  flip(patch, dims=[-2])      # rows = body +Y
    kappa  ->  -kappa                      # a mirrored turn is the opposite turn
    e_pos, e_rot  ->  unchanged            # both are magnitudes
```

Free 2x. RNG consumption order (map, pose, command, jitter) is documented in the generator so a seed
reproduces a file exactly.

---

## 8. Self-checks and baselines

Every module's `if __name__ == "__main__":` block is its test; there is no pytest suite in this tree.

**In `model.py`:**

* **Patch↔map equivalence** — §5d. `map_mode(map)[i,j] == patch_mode(patch_at(i,j))` to float
  tolerance. Exact, weight-independent, and the check that makes Phase 2 possible.
* **Command independence of the trunk** — autograd: `d c / d u == 0` for every trunk output. This is
  the caching guarantee of §5a made testable, and it is what the 50x per-edge saving rests on.
* **Receptive field** — autograd probe on the trunk output must measure exactly 23 px. Carried over
  from `grid_learning_2` because it is the check that catches a substituted norm.
* **Flat ground** — relief of a constant heightmap is exactly zero. `learning/terrain_patch.py`'s
  flat assert, copied into `patch.py` (§11a).
* **Pinned-constant guard** — `infer.py`'s §4c assert, exercised both ways: a checkpoint whose
  `arc_len` / `min_turn_radius` / `v_nom` match the router loads; one that differs by 1e-3 raises.
  This is the check that makes pinning `v` and `L` safe rather than merely simpler, so it is a test,
  not a code comment.
* **Rotation invariance of the patch** — `learning/terrain_patch.py`'s centred-box assert, copied
  into `patch.py` (a robot 3 m out along -X facing +X and one 3 m out along -Y facing +Y must produce
  the same patch). It is the only thing that justifies dropping `yaw` from the feature vector.
* **Mirror registration** asserted (row `i` and row `H-1-i` draw on mirrored input rows); mirror
  **equivariance** only *reported* on a trained model — a conv stack at random init is not
  reflection-equivariant, and conflating the two is the mistake `grid_learning_2` §8 calls out.
* **`TargetTransform` round-trip** and non-negativity of the inverse.

**At eval time, in `train.py`** — reported in physical units on **held-out maps**, split by map, never
by row:

| baseline | what it is | what beating it proves |
|---|---|---|
| global mean | one constant | a sanity floor |
| per-`kappa` mean | binned mean, shrunk toward the global mean | the model uses terrain at all |
| blur-the-terrain | same net, relief replaced by its own per-sample mean | the model uses terrain *structure* |
| max-relief-in-sweep | `max(relief)` over the arc's swept envelope, one fitted scalar | the model beats the obvious geometric proxy |
| **`_step_gate_kernel`** | the incumbent: max 3x3 prominence within `foot_r` cells, thresholded | **the falsifiable A/B — that a learned `d_hat` earns its place against the hand-tuned heuristic it would replace** |
| concat head | `--head-fusion concat` | that FiLM earns its place |

The last-but-one is the one that matters. `costtogo.py`'s `_local_step_kernel` + `_step_gate_kernel`
are a hand-tuned prominence heuristic that exists precisely because *"the settle straddles"* thin
vertical obstacles. If `d_hat` cannot out-predict it, the honest conclusion is that this network is
not worth its 50 ms.

Metrics: **RMSE over the top decile by true `e_pos`** as the headline (the collision rows — the entire
point, and the only place a 0.3 m arc has much signal, per §7c); masked RMSE over all valid rows
(m, rad); R^2 per head; and the `kappa ~ 0` on flat ground check (must predict ~0).

One more, reported rather than optimised, and it is the label-quality floor everything else is
measured against:

* **Ostrich's own repeat scatter on identical trials.** `submodule_test/` measured 12.7–31.2 deg of
  net-yaw spread across repeats of a pivot, confirmed not a batching artifact — but that is a
  `dt = 3e-2` pivot number and does not transfer. Re-measure for *this* trial shape (0.3 m forward arc,
  warm-started) **at `OSTRICH_DT = 2.5e-2`** (§2a) on a few hundred duplicated rows. If the scatter is
  comparable to the terrain-driven spread of `d`, the label is mostly noise and no architecture will
  fix it — a stop-and-rethink result, much cheaper to find now than after a 200k-trial generation.

**The experiment that actually decides it.** None of the above measures whether the *plan* is better.
`helhest_stack`'s own demos execute with the kinematic twin, so a plan-vs-execution A/B inside that
tree is zero by construction — the twin cannot disagree with itself. The honest evaluation is:

> plan in `helhest_stack` with `divergence_weight ∈ {0, ...}`, **execute in ostrich**, measure
> goal-reach rate, collision rate, and time-to-goal over a held-out map set.

`comparator/common.py` and `submodule_test/` already have the machinery to run ostrich on an
arbitrary heightmap from a scripted setpoint stream. That harness is a deliverable of this design,
not an afterthought.

---

## 9. Training recipe (starting point)

```
    optimiser     AdamW, lr 3e-4, weight_decay 1e-4
    schedule      cosine to 0, ~5 epochs linear warmup
    batch size    256 trials
    epochs        ~100, early stopping on held-out-MAP val loss
    targets       log1p -> standardise (TargetTransform), fitted on TRAIN + valid rows only;
                  loss computed in model space so the inverse's clamp never kills a gradient
    augmentation  y-mirror + kappa negate, p = 0.5
    precision     fp32 for training; export fp16 for inference
    logging       Weights & Biases, as learning/train.py and grid_learning_2/train.py
    checkpoint    outputs/checkpoints/, self-describing, WITH the fitted TargetTransform and the
                  full PatchSpec (a model fed a differently-shaped or differently-referenced patch
                  at inference is silently wrong, not broken -- terrain_patch.py's own warning)
```

---

## 10. What this can and cannot claim

Stated up front so nobody has to discover it from a disappointing result.

**It can claim:** that a given edge, entered at `v_nom` from a pose the planner believes it occupies,
is one whose real endpoint the ideal arc plus a static settle will predict badly. That is a
**corridor-selection signal** — it steers the value iteration away from regions where the router's
own model is fiction, and it is strictly more transit information than the planner has today, which
is none.

**It cannot claim:** that the resulting plan is executable.

* **No velocity state, and no bound on what that costs.** The lattice is `(x, y, yaw)`; `v` is pinned
  by convention (§2) and by generation (§4), not tracked. A `d_hat` trained entirely at 0.6 m/s says
  nothing about the same edge taken at 1.4 m/s, and because the prototype never samples a `v` band it
  cannot even say how fast the claim decays. §4c names the regeneration that would fix this.
* **One arc length.** Everything is trained at `L = 0.3 m`. A different `step` is a different network,
  and §4c's load-time assert is what keeps that from failing silently.
* **Complete maps only.** §3c: no `measured` channel, so a partially-observed `GridMap` is outside the
  training distribution in a way the network cannot detect. Do not evaluate on
  `demos/navigate_partial_view.py`'s incremental map without adding the channel first.
* **Forward arcs only.** Pivots are excluded (§4b) and charged nothing.
* **The seams are invisible.** Curvature is discontinuous between consecutive primitives — straight
  into hard-left is an instantaneous `omega` step. A per-edge `d_hat` cannot see that a *concatenation*
  is infeasible even when each edge is fine. Solving that in the lattice needs
  `nx*ny*n_theta*n_v` settle worlds and curvature-continuous primitives; it is MPPI's job and this
  design deliberately does not attempt it.
* **The ground truth is a simulator.** Everything here is conditional on ostrich being right. The
  claim only becomes a claim about reality when an ostrich-labelled `d_hat` is shown to predict
  failures of the real Helhest that the twin did not anticipate.
* **The heading input is quantised.** §1c handles it by training under the noise rather than
  pretending it away, but the residual noise floor it sets is real and bounds how sharp `d_hat` can
  ever be at a box edge.

---

## 11. Files this implies

`src/feasibility/lattice_learning/`, a plain importable package (`__init__.py`), every module also
runnable standalone as a script — same convention as `learning/`, `grid_learning/`, `grid_learning_2/`.

### 11a. This tree does not import from its siblings

**`lattice_learning/` imports nothing from `learning/`, `grid_learning/` or `grid_learning_2/`, and
they import nothing from it** — the same stance those trees already hold (`grid_learning/` re-derives
wheel geometry from `HelhestJuniorConfig` and re-implements the on-obstacle spawn filter locally
rather than importing `learning/`'s). These are parallel *experiments*, not layers of one library:
each must be readable end to end on its own and free to change its own assumptions without silently
invalidating a sibling's committed datasets and checkpoints. A shared `terrain_patch.py` would mean
retuning this design's patch extent (§3a) changes what `learning/`'s loader reconstructs from an old
file's attrs.

What that costs here, stated plainly, is three deliberate duplications:

| duplicated | from | why the copy is the right call |
|---|---|---|
| `se3_error` | `learning/pose_error.py` | ~10 lines of closed-form SE(3) algebra that will never change; §8 asserts it against a known rotation |
| `PatchSpec` / `sample_patches` + its two asserts | `learning/terrain_patch.py` | the geometry is **different** (§3a/§3b: 0.125 m not 0.25 m, a different extent, `n_channels` as a constant) — this is a re-derivation, not a copy, and sharing would have forced a parameterisation neither tree wants |
| `TargetTransform` | `learning/model.py` | lives in `model.py` so the network self-checks before any dataset exists |

**What *is* imported:** `feasibility.comparator` (`provenance`, `common`) and `feasibility.heightmap`
— **shared infrastructure**, not sibling experiments. The HDF5 schema and the elevation-grid reader are
the interfaces every tree writes and reads through, so duplicating *those* is what would actually cause
drift; `grid_learning/` draws the line in the same place. Likewise `helhest` and `ostrich`: objects of
study. The one load-bearing cross-tree agreement, `arc.py` ↔ `_build_primitives`, is handled by
assertion rather than import — see the table below.

| file | role |
|---|---|
| `design.md` | this document |
| `arc.py` | the primitive contract, and the ONLY place it is written down: `kappa`/`arc_len` from `(step, min_turn_radius)` matching `_build_primitives`; `integrate_arc(pose, kappa, L)` → exact endpoint; `LEVER = rear_offset + wheel_radius`. Re-derived from `helhest_stack`'s constants rather than imported, and its `__main__` asserts it reproduces `_build_primitives`' five `turns` and their endpoints to float tolerance — the one place the two trees must agree |
| `patch.py` | `PatchSpec` (§3a geometry, 1 channel — `n_channels` a constant, not a literal, so §3c's deferred `measured` plane is a data change and not an architecture one) + `sample_patches` + `patch_spec_to_attrs`/`from_attrs` + the overhang predicate §7b's `valid` filter uses. Local re-derivation of `learning/terrain_patch.py`; keeps its flat-ground and rotation-invariance asserts verbatim |
| `custom_dataset.py` | `ArcDivergenceDataset` — reads `dataset_arc_*.h5`, computes `y` from `ref_pose`/`ostrich pose[-1]` via a **local** `se3_error` (§11a), map-level split only, `valid`/`swept_clear` masks, `valid_targets()` for the train-only transform fit. Imports `TargetTransform`/`Normalizer` **from `model.py`**, so `model.py` runs and self-checks before any dataset exists |
| `model.py` | `ArcDivergenceNet` (§5) + `TargetTransform` + the §8 self-check battery, including the patch↔map equality and the trunk command-independence probe. `forward(patch, kappa)` for patch mode, `encode_map(relief)` for §6b |
| `generate_dataset.py` | Hydra-driven, multi-map, continuous poses, `kappa` sampling at the pinned `V_NOM`/`ARC_LEN`, the warm-up prefix + slice of §2, the arc reference of §1, the endpoint settle batch, the belief jitter of §1c, and §7b's overhang + obstacle filters. Writes `dataset_arc_*.h5` through `comparator.provenance`, with the three pinned constants in the root attrs. `+dry_run=true` validates geometry and prints the trial budget without touching a GPU; `+repeat_trials=N` duplicates rows for §8's ostrich-scatter measurement |
| `train.py` | §9's recipe, held-out-MAP split, mirror + `kappa`-negate augmentation, the §8 baseline battery (`--head-fusion concat`, `--blur-terrain`, and the `_step_gate` comparison), `--self-test` on a synthetic file with wandb disabled |
| `infer.py` | the planner-side producer: §4c's pinned-constant assert at load, then Warp rotate-and-gather kernel → `wp.to_torch` → net → `rho = e_pos + LEVER*e_rot` → a `[ny, nx, n_theta, n_prim]` `wp.array`, tiled per §6a. Loads a `CostToGo` and calls `set_divergence`. This is where torch lives; `helhest_stack` never sees it |
| `eval_in_ostrich.py` | §8's deciding experiment: plan in `helhest_stack` at several `divergence_weight`, execute the resulting setpoint stream in ostrich on held-out maps, report goal-reach / collision / time-to-goal |

**In `helhest_stack` (a separate, small, torch-free commit):** the `divergence` array + `divergence_weight`
scalar in `_relax_lattice_pose_kernel` and `LatticeValueSolver._relax`, and `CostToGo.set_divergence()`
with a stable owned buffer. Default `divergence_weight = 0.0`, bit-identical to today. Per that tree's
`CLAUDE.md`: type hints on every signature (not on `@wp.kernel` functions), black at line-length 100,
one symbol per import line, comments that give the units and say *why*. Commit inside the submodule
first, then record the pointer in the superproject.
