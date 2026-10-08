# The MPPI Controller in `helhest_stack`

*A walk-through of how the Helhest navigation stack turns a goal and a terrain map into wheel
commands, written as a lecture note. Source: `helhest_stack/src/helhest/control/mppi.py` (the
sampler/optimiser), `.../planning/costtogo.py` (the global routing field it follows),
`.../control/command.py` + `terminal.py` (output conditioning), and
`ros/.../elevation_node.py::_plan` (the loop that ties it together).*

Numbers below are tagged **[lib]** (default in `mppi.py`), **[node]** (default declared in the ROS
node, which is what actually runs), or **[yaml]** (override in `ros/odin/odin_elevation.params.yaml`,
the on-robot config). Where they differ, [yaml] wins on the robot, then [node], then [lib].

---

## 1. The big picture

Helhest is a three-wheeled skid-steer robot. Its controller is a **sampling-based model-predictive
controller**: every cycle it

1. imagines thousands of possible short futures (wheel-speed trajectories),
2. simulates each one forward in time on the *current terrain map* with a fast kinematic model,
3. scores each future with a cost,
4. combines the best few into a new plan, and repeats that a few times,
5. executes only the **first step** of the resulting plan, then starts over with fresh sensor data
   (receding horizon).

Two things make this planner distinctive:

* **Routing is separated from local control.** MPPI only looks 2.5 s ahead. The long-range question
  *"which way around the wall?"* is answered by a **cost-to-go field** `V(x, y, θ)` — a value
  function over position **and heading**, solved on the whole local map. MPPI's goal cost is simply
  "where does this rollout end up on `V`?". The sampler is the *local optimiser*, the lattice is the
  *global router*.
* **It is not textbook MPPI.** The weights are not `exp(-J/λ)`; the update is a **CEM-style elite
  mean** (§7). Treat the name as a family resemblance: sample → roll out → cost → re-fit.

```
 elevation map ──┬──► cost-to-go V(x,y,θ)  (global router, once per frame)
                 │              │
 robot state ────┼──► MPPI: sample ─► roll out ─► cost ─► elite mean ─┐ ×3 refines
 goal ───────────┘          ▲                                          │
                            └───────── nominal plan U ◄────────────────┘
                                          │  U[0] only
                                          ▼
                        output conditioning ─► /cmd_joints
```

---

## 2. Notation and what is being optimised

| symbol | meaning | value |
|---|---|---|
| `Δt` | rollout/control timestep | **0.1 s** (`dynamics.DT`) |
| `T` | horizon steps | **25** [node] → 2.5 s lookahead (`plan_horizon`) |
| `B` | rollouts simulated per refine | **4096** [node] (`plan_batch`) |
| `n_μ` | friction replicas per candidate | **1** [lib, node]; 3 [yaml] |
| `C = B / n_μ` | distinct candidate control sequences | 4096 (or 1365 with `n_μ`=3) |
| `U ∈ ℝ^{T×2}` | the **nominal plan**: left/right wheel speed per step | persistent |
| `(w_L, w_R)` | commanded left/right wheel angular speed [rad/s] | box `[w_min, w_max]` |
| `s_t = (x, y, ψ)` | planar pose of the robot at step `t` | |
| `V(x,y,θ)` | cost-to-go (metre-equivalents to the goal) | from the lattice (§6.1) |

**The decision variable is wheel speed, not body velocity.** Forward speed is the mean of the two
wheels, yaw rate is proportional to their difference. The rear wheel is not controlled — it is
commanded to the mean of left and right (a "follower").

* `w_max` = **4.0** rad/s [lib, node]; **6.0** [yaml].
* `w_min` = **0.0** [lib, node, yaml] → **forward-only**. Negative values enable reverse and point
  turns, but reverse is additionally gated on map coverage behind the robot (§9.3).

The unknown being solved for is `U`, a sequence of 25 pairs `(w_L, w_R)`.

---

## 3. The algorithm in pseudocode

### 3.1 One *refine* (the inner iteration)

```text
function REFINE(U):                       # runs on the GPU, captured as one graph
    # 1. SAMPLE  C candidate wheel-speed sequences  (§4)
    for c in 0 … C-1:
        W[c] ← SAMPLE_CANDIDATE(c, U)     # a [T × 2] sequence, clamped to the wheel box
    replicate each candidate n_μ times    # replica k of candidate c shares W[c] exactly

    # 2. ROLL OUT all B = C·n_μ sequences from the SAME start state  (§5)
    for r in 0 … B-1, in parallel:
        trajectory[r] ← SIMULATE(start_state, W[r mod C], terrain, μ_scale[r])

    # 3. COST each trajectory  (§6)
    for r in 0 … B-1:
        J[r], J_safe[r] ← COST(trajectory[r], W[r mod C])

    # 4. COLLAPSE the friction replicas  (§7)
    for c in 0 … C-1:
        Jc[c] ← max_k J_safe[k·C+c]  +  mean_k (J[k·C+c] − J_safe[k·C+c])

    # 5. RE-FIT: CEM elite mean  (§8)
    τ  ← cost threshold such that ≈ K_elite candidates satisfy Jc ≤ τ
    E  ← { c : Jc[c] ≤ τ  and c is "mode-compatible" with the best candidate }
    U  ← clamp( mean_{c ∈ E} W[c] , w_min, w_max )          # plain, unweighted mean
    return U
```

### 3.2 One *replan* (what happens per perception frame)

```text
function REPLAN(state, goal, n_refine = 3):
    seed rollouts with the measured state                    # §9.1
    solve the cost-to-go field V for this frame's map & goal  # §6.1
    for i in 1 … n_refine:                                    # = 3 [node]
        U ← REFINE(U)                                         # fresh random numbers each time
    return U
```

### 3.3 The full control loop

```text
U ← constant(1.5 rad/s)                         # plan_nominal_reset
loop on every accumulated LiDAR/elevation frame:
    build local terrain (12 m window) and routing map (16 m window)
    if within 0.3 m of goal:   command STOP; continue
    U_new ← REPLAN(state, goal, 3)
    U     ← 0.7·U_new + 0.3·shift(U_prev)           # plan consistency (§9.2)
    if within 1.5 m of goal:   (wl,wr) ← DOCK_CONTROL(state, goal)
    else:                      (wl,wr) ← U[0]       # first committed step only
    cmd ← CONDITION(wl, wr, previous_cmd)           # §10
    publish cmd
```

The rest of this note unpacks each numbered step.

---

## 4. How the rollouts are sampled

All `C` candidates are *not* drawn from one distribution. The population is a **mixture of priors**,
laid out by candidate index so each GPU thread knows which prior it belongs to:

```
index:   0      1 … n_wide-1     [spin]       [straight]        [pivot]      rest
prior:  NOMINAL  WIDE            SPIN         STRAIGHT          PIVOT        NARROW
```

With the **[node]** settings and `C = 4096`:

| prior | share | count | what it is | purpose |
|---|---|---|---|---|
| **NOMINAL** | 1 candidate | 1 | `U` itself, **no noise** | keeps the previous best in the race; also the trajectory that is published/visualised |
| **WIDE** | `wide_frac` = **0.25** | ≈ 1023 | wheel speeds drawn **uniformly over the whole box** `[w_min, w_max]`, independently for each wheel and each knot; independent of `U` | global search — lets the elite escape local minima and find a totally different manoeuvre |
| **SPIN** | `spin_frac` = **0** (off) | 0 | `w_L = −m, w_R = +m`, `m ~ U[spin_min, w_max]` (`spin_min` = **2.0** rad/s), random sign, held constant over the horizon | turn on the spot; only meaningful with reverse |
| **STRAIGHT** | `straight_frac` = **0.2** [node] (0 [lib]) | 819 | `w_L = w_R = v(t)` with `v` interpolated between uniformly drawn knot speeds | gives straight-ahead its own explicit chance, so a clear shot is not a noisy wobble |
| **PIVOT** | 0.05 if reverse enabled else 0 | 0 | `w_L = −w_R`, one signed rate per knot | point turns (reverse only) |
| **NARROW** | the rest | ≈ 2253 | `U` + smooth noise + jitter (below) | local refinement around the current plan |

### 4.1 The NARROW prior (the "classic MPPI" part)

For every wheel independently:

```text
w[t] = U[t] + σ_knot · interp_t( ε_knot[0..3] )  +  σ · ε_step[t]
```

* **Spline-knot noise:** `n_knots` = **4** random numbers `ε ~ N(0,1)` per wheel are drawn at
  evenly spaced knots across the horizon, then **linearly interpolated** to the 25 steps (with
  `T=25`, knots sit 8 steps apart). Scale `σ_knot` = **1.0** rad/s. Because adjacent steps share
  knots, a candidate is a *smooth, committed manoeuvre*, not white noise — the robot cannot execute
  white noise anyway, and smooth candidates explore far more of the useful space per sample.
* **Step jitter:** a light i.i.d. term `σ · ε`, `σ` = **0.5** rad/s, on top. The code draws it from a
  single random stream shared by the two wheels, so in practice it perturbs the common mode
  (forward speed) rather than the left/right difference.
* Candidate noise is **deterministic given the candidate index and a seed counter** that is bumped
  every refine. Same index → same random numbers within a refine, new numbers on the next one.

### 4.2 Clamping

Every sample is clipped to `[w_min, w_max]` (so with `w_min = 0` all wheel speeds are ≥ 0). The one
exception is the SPIN band, which is allowed down to `−w_max`: its whole point is one wheel running
backwards, and clamping it would silently turn every spin into a one-wheel-stopped arc.

### 4.3 Why mix priors at all?

A Gaussian cloud around `U` can only refine what it already roughly knows. The mixture splits the
two jobs: **NARROW** exploits (≈ 55% of samples), **WIDE** explores (25%), **STRAIGHT** protects the
most common good answer from sampling noise (20%). The elite selection in §8 then decides who wins,
with no tuning of a temperature.

---

## 5. How a rollout is simulated

Each candidate sequence is pushed through a **kinematic twin of the robot** (the `ForwardSimulator`
in `helhest/engine`), *not* a force-level physics engine. Per 0.1 s step:

1. **Actuator model.** Wheels do not reach the commanded speed instantly: a first-order lag with
   `τ_motor` = **0.19 s** (measured), i.e. each step the wheel closes `Δt/τ ≈ 53 %` of the gap. A
   small transport delay `command_delay` = 0.04 s exists but rounds to **0 steps** at `Δt = 0.1`.
   The **commands already published but not yet acted on** are loaded into the rollout
   (`command_history`), and the **measured** wheel speeds and body twist are used as the initial
   state — so the plan starts from what the robot is *really* doing, not from rest.
2. **Body twist from wheels (skid-steer model).** Forward speed `v = R·(ω_L+ω_R)/2`; yaw rate
   `ω_z = R·(ω_R − ω_L) / (2·half_track·α)`. The divisor `α = 1 + k_turn·(grip/weight)` is the
   **turn resistance**: skid-steer wheels must scrub sideways to turn, so the robot turns less than
   ideal. `k_turn` is a calibrated constant (indoor 0.6, outdoor 1.0, **1.27** [yaml]); rollout
   friction `μ` = **0.8** (`plan_friction`) scales the grip. A momentum model additionally makes
   body speed grip-limited so braking and launch distances depend on `μ`.
3. **Pose update.** Integrate `(v, ω_z)` to advance `(x, y, ψ)`.
4. **Terrain settle.** The robot is a rigid tripod resting on the heightmap: a short (6 Newton
   iterations, loose tolerance) analytic settle gives height `z`, `pitch`, `roll`, plus a per-step
   **clearance** (wheel/chassis to terrain) and a **residual** (how badly the rigid tripod fails to
   fit the ground — a violation flag), and **wheel normal loads**. Wheel-terrain contact uses a
   0.10 m-wide cylinder envelope (`plan_wheel_width`).

All `B` rollouts run in parallel on the GPU from the *same* initial pose, so one replan evaluates
`B × T` = 4096 × 25 ≈ 100 k simulated steps per refine. The terrain they see is the single-scan
**local** elevation (12 m window, 0.08 m cells).

---

## 6. The cost function

The cost of one rollout is a weighted sum over its `T` steps. Each term is a *penalty*, and many are
**graded** — they grow with the *size* and *earliness* of a violation rather than being a 0/1
flag, so the ranking still works when *every* sample violates something.

Let `early(t) = (T − t)/T` (violations right now hurt more than ones at the end of the horizon).

### 6.1 Goal term — the cost-to-go field

Instead of "distance to goal", MPPI reads the lattice value function `V(x, y, θ)` at every rollout
pose (trilinear interpolation: bilinear in position, linear in the wrapped heading).

**How `V` is made** (`planning/costtogo.py`, once per frame, GPU-resident):

1. *Feasibility by settling.* The robot is virtually placed at **every** lattice pose
   (cell × 24 headings) on the routing map. A pose is **blocked** if the settle residual is too
   large, clearance too small, `|roll| > max_roll`, or pitch exceeds the asymmetric climb/descend
   limits. Because heading is part of the pose, a side-slope can be fine to climb head-on and
   forbidden to traverse sideways.
2. *Disturbance tube.* Blocked set eroded by `robust_margin_m` = **0.3 m** [node] (0.20 m [yaml])
   and `robust_margin_deg` = 0° [node] (15° [yaml]).
3. *Value iteration.* Starting from `V = 0` at the goal cell, Bellman-style relaxation over
   **forward-arc motion primitives** (arc length `step` = **0.3 m**, curvature capped by the robot's
   minimum turn radius, swept cells collision-checked). Arc cost =
   `arc_length · (1 + flatness_weight · mean(roll_w·|roll| + pitch_w·|pitch|))` with
   `flatness_weight` = **2.0**, so flatter ground is preferred. Optional `pivot_cost`
   (0.0 [node]; 0.3 [yaml]) adds two in-place-turn primitives (±1 heading bin) so a goal behind the
   robot is "turn, then drive" rather than a wide loop.
4. Heading bins: `n_theta` = **24** (15° each). Routing grid is the map coarsened by
   `plan_lat_coarsen` = **4** [node] (3 [yaml]); routing window **16 m**.
5. Unreachable cells (`+∞`, e.g. a forward-only robot facing away from a goal it cannot loop to) are
   clamped to a finite cap
   `V_cap = 1.5·(nx+ny)·cell·(1 + flatness_weight)` so interpolation never blends with infinity.

**How MPPI uses it.** For each step the goal cost is `g_t = V(x_t, y_t, ψ_t)²`:

* **Squared**, so large detours are punished super-linearly.
* **Reversing steps** are scored at heading `ψ + π` (a reversing robot progresses like a
  forward-only robot facing the other way).
* **Explore fallback:** if `V ≥ 0.9·V_cap` (goal unreachable in-window, the field is flat) the term
  becomes `V_cap² + 1.0·‖p − goal‖²` — a straight-line pull so the robot explores toward the goal
  rather than idling.

```text
goal_cost = 3.0 · g_T                       # terminal: where the plan ENDS      (goal_terminal)
          + 0.3 · (1/T) Σ_t g_t             # running mean: progress every step  (goal_running)
```

### 6.2 Control-shaping terms (raw sums over the horizon)

| term | formula | weight |
|---|---|---|
| effort | `Σ (w_L² + w_R²)` | **1e-3** [node] (lib 2e-3) |
| smoothness | `Σ (Δw_L² + Δw_R²)` step-to-step change | **0.04** [node] (0.1 [yaml], lib 2e-3) |
| turn | `Σ (w_R − w_L)²` — prefers straight when the free-heading goal is indifferent | **0.03** [node] (0.2 [yaml], lib 0) |
| reverse | `Σ |v|·Δt` while reversing — makes reverse an *escape*, not a route | 75 |

Note these are **sums**, not means: their effective strength scales with horizon length (the weights
were tuned at `T=25`).

### 6.3 Safety terms (graded penalties — the "do not hurt the robot" group)

| term | what is penalised | weight |
|---|---|---|
| infeasible | `early·[clearance below margin + residual above tol + roll beyond `max_roll` + pitch beyond climb/descend limit]` (limits flip sign when reversing) | **1e5** |
| saturation | friction demand over grip budget (below) | **300** |
| tip | negative wheel load (CoM outside support triangle), as a fraction of robot weight, `early`-weighted | **2e4** |
| unknown | while reversing, `early·(1 − measured(x,y))` — unseen cells behind the robot must hard-lose | **1e4** |
| out of bounds | depth beyond a 0.4 m soft wall inside the map edge (`V` is clamped off-grid, so the goal term alone would not stop it) | **50** |

**Friction-saturation certificate.** The commanded manoeuvre demands friction both along the slope
and sideways (centripetal + side slope):

```text
d_long = m·g·|sin(pitch)|
d_lat  = m·( |v·ω_z| + g·|sin(roll)| )
grip   = max( (α − 1)·m·g / k_turn , 1 )              # recovered from the model's own α
sat    = √(d_long² + d_lat²) / grip
penalty += early · max(sat − 1, 0)
```

Past 1.0 the commanded motion is not achievable on that surface. This is what makes the planner slow
down on low-grip or sloped ground — the kinematic model itself would otherwise get *more* optimistic
as `μ` falls.

### 6.4 Total

```text
J = goal_cost + effort + smoothness + turn + reverse + J_safe
J_safe = oob + infeasible + saturation + tip + unknown        # kept separately (needed by §7)
```

The magnitudes are deliberate: `1e5` on a collision-ish violation dwarfs any plausible goal cost, so
a candidate that touches anything effectively cannot win — yet because it is *graded*, a candidate
that violates by a centimetre still ranks above one that violates by a metre.

---

## 7. Robustness to friction (`n_μ` replicas)

Friction is the largest model uncertainty, so each candidate can be rolled out `n_μ` times with
different friction scales spread evenly over `[centre − span, centre + span]`
(`span` = **0.25** [node, yaml]; scales floored at 0.05). The `n_μ` copies share *identical*
controls, so their costs differ only through the model. The collapse to one number per candidate is:

```text
Jc[c] = max over replicas of J_safe       (worst case for safety)
      + mean over replicas of (J − J_safe) (average for everything else)
```

**Why split it?** Worst-casing *everything* makes "slow or still" look safest (measured: +53% on
turny ground at ±40%, +232% at ±80% traversal time). Worst-casing only the terms that must hold under
every hypothesis keeps the guarantee "no friction hypothesis may crash" without the slowness tax.
`n_μ` = 1 collapses to a copy. `B` must be divisible by `n_μ` (see the note at the end). Optionally
an online estimator recentres/narrows the band from gyro feedback (`plan_mu_adapt`, on in [yaml]).

---

## 8. Updating the plan: the elite mean

This is where the algorithm departs from textbook MPPI. Textbook: `U ← U + Σ_k w_k ε_k` with
`w_k ∝ exp(−J_k/λ)`. Here:

1. **Find the elite threshold `τ`.** The elite is the best `K_elite = ⌊elite_frac · C⌋`
   candidates; `elite_frac` = **0.01** [node] (lib 0.02), so **K = 40 of 4096**. It is a *rank*
   statistic, so a huge penalty on a bad sample cannot distort the weights — bad samples just do not
   make the cut. `τ` is found on the GPU by **bisection** on the cost range
   (`⌈log₂ C⌉ + 5` = 17 steps; a fixed count, so no sort/readback is needed and the whole refine
   stays inside one CUDA graph).
2. **Mode-coherent mean.** Averaging a "pass left" elite with a "pass right" elite produces a plan
   that drives *into* the obstacle. So each candidate gets two keys — **direction** (sign of total
   wheel speed: forward/reverse) and **turn side** (sign of summed differential, with a deadband
   `turn_mode_th` = **0.5** rad/s per step; inside it the candidate is "neutral"). Only elites that
   match the *best* candidate's direction and turn side (neutral is compatible with either side,
   unless the best itself is neutral) are averaged. During forward cruising every key is `(+1, 0)`
   and this reduces to the plain elite mean.
3. **Average and clamp.** `U[t, wheel] ← clamp(mean over kept elites of W[c][t, wheel])`. The mean
   is **unweighted** and **replaces** `U`; there is no step-size, no learning rate.

Consequences worth noting:

* A small `elite_frac` gives a *peaky* update (average the few best) — measured to drive straighter
  (0.02 → 0.01 cut lateral wander ~20%) but too small (< ~0.003) starves the mean.
* Because the new `U` is the centre of the next refine's NARROW prior, **the 3 refines per frame are
  3 CEM iterations**: each tightens around the previous elite while WIDE and STRAIGHT keep injecting
  new hypotheses (and can erase the progress if they win outright).
* There is no temperature to tune; the inverse role is played by `elite_frac`.

---

## 9. The loop around the optimiser

### 9.1 Seeding the rollouts (per frame)

* Terrain: the rollout simulator's elevation is replaced by this frame's local map.
* Start pose `(x, y, ψ)`: the localised robot pose in the map window; identical for all rollouts.
* Realised wheel speeds: measured `/joint_states` if fresher than 0.3 s, else the last conditioned
  command. Body twist: odometry if fresher than 0.3 s, else derived from wheels through the turn
  model (lateral velocity unobservable → 0).
* In-flight commands (`command_history`) filled in so latency is modelled.
* Goal, and the freshly solved cost-to-go `V`.

### 9.2 Warm start and plan consistency

`U` persists across frames — the last plan is the starting guess. The node additionally blends
(`plan_consistency` = **0.3**):

```text
U ← (1 − 0.3)·U_new + 0.3·shift(U_prev, by one step; last step repeated)
```

so the committed manoeuvre does not jitter on open ground (≈ 35% less cruise churn in sim; too high
→ lag reacting to new obstacles). The blended `U` is what seeds the next frame.

### 9.3 Reverse gate (only if `plan_wmin < 0`)

The effective lower bound `wlo` is a device scalar flipped per frame: reverse is unlocked only if a
robot-width strip 0.3 … 1.5 m directly behind the robot is ≥ 95 % *measured* in the map (no rear
sensor, so reverse may only use remembered ground). Off by default (`plan_wmin` = 0).

### 9.4 Replan cadence

Planning runs once per **accumulated LiDAR/elevation frame** — there is no separate fixed timer —
with **3** refines (`plan_n_refine`). The plan's time axis (Δt = 0.1 s) is fixed regardless of how
often it is recomputed.

---

## 10. From plan to motor command

Only `U[0]` leaves the planner. Then, in order:

1. **Stop / dock / drive switch.**
   * distance to goal `< plan_reach_radius` = **0.3 m** → command `(0,0)` (a stop ramp), latched
     "reached" until a new goal;
   * `< plan_dock_radius` = **1.5 m** → bypass MPPI and use `dock_control`: forward speed
     `2.0·min(1, d/1.5)·max(0, cos(bearing))`, differential `3.0·bearing·0.5`. Reason: a
     horizon-limited, forward-only MPPI never commands deceleration and orbits the goal;
   * otherwise → `U[0]`.
2. **Output conditioning** (`condition_command`, the single place for actuator safety):
   * rear wheel = mean of left and right;
   * **goal brake:** forward speed scaled by `min(1, d/2.0 m)`;
   * **turn boost:** differential multiplied by `turn_boost` (1.0 [node], 1.36 [yaml]) to
     compensate a measured drivetrain loss;
   * optional **turn brake** (`plan_turn_brake_a_max`, 0.0 [node], 0.6 [yaml]) — scales both
     `mean` and `diff` by the same factor so the arc radius is preserved while lateral acceleration
     `a_lat = lat_gain·mean·diff` stays under the ceiling;
   * asymmetric **slew limit** on each joint: accel `max_slew` = **6** rad/s², decel `max_decel` =
     **12** [node] (8 [yaml]) rad/s²;
   * hard clamp `|ω| ≤ max_omega` = **5.0** [node] (7.5 [yaml]).
3. Optional **inner yaw-rate loop** (`plan_yaw_track`, off [node], on [yaml]): a PI controller
   (`kp` 0.4, `ki` 1.0) on the gyro that nudges the differential so the realised yaw rate matches
   what the plan asked for.

---

## 11. Parameter cheat-sheet

| group | parameter | default | notes |
|---|---|---|---|
| **Budget** | `plan_batch` `B` | 4096 | rollouts per refine |
| | `plan_horizon` `T` | 25 | steps → 2.5 s; short on purpose (the lattice does the routing) |
| | `Δt` | 0.1 s | |
| | `plan_n_refine` | 3 | CEM iterations per frame |
| | `plan_n_mu` | 1 (3 in yaml) | friction replicas; must divide `B` |
| **Sampling** | `sigma` | 0.5 | per-step jitter [rad/s] |
| | `sigma_knot` | 1.0 | knot noise [rad/s] |
| | `n_knots` | 4 | spline knots over the horizon |
| | `wide_frac` | 0.25 | global-search share |
| | `straight_frac` | 0.2 (lib 0) | |
| | `spin_frac` / `pivot_frac` | 0 / 0.05 only if reverse | |
| | `spin_min` | 2.0 rad/s | robot will not break loose slower |
| | `w_min`, `w_max` | 0.0, 4.0 (yaml 6.0) | rad/s per wheel |
| **Elite** | `elite_frac` | 0.01 (lib 0.02) | K = 40 |
| | `turn_mode_th` | 0.5 | turn-side deadband |
| **Cost** | `goal_terminal` / `goal_running` | 3.0 / 0.3 | on `V²` |
| | `explore_fallback` | 1.0 | |
| | `effort`, `smoothness`, `turn` | 1e-3, 0.04, 0.03 | node values |
| | `infeasible`, `tip`, `unknown`, `saturation` | 1e5, 2e4, 1e4, 300 | |
| | `out_of_bounds`, `reverse` | 50, 75 | |
| **Model** | `τ_motor` | 0.19 s | |
| | `k_turn` | 0.6 / 1.0 / 1.27 | indoor / outdoor / yaml |
| | `plan_friction` μ | 0.8 | |
| | `plan_mu_span` | 0.25 | |
| **Routing** | `n_theta`, `step` | 24, 0.3 m | |
| | `flatness_weight` | 2.0 | |
| | `lat_coarsen` | 4 (yaml 3) | |
| | `robust_margin` | 0.3 m / 0° (yaml 0.2 m / 15°) | |
| **Loop** | `plan_consistency` | 0.3 | |
| | `reach` / `dock` / `goal_brake` radii | 0.3 / 1.5 / 2.0 m | |
| | `max_slew` / `max_decel` / `max_omega` | 6 / 12 / 5.0 | yaml: 6 / 8 / 7.5 |

---

## 12. What to take away

1. **Sampling is structured, not just Gaussian:** one nominal, ~25 % uniform global search, ~20 %
   straight-ahead, the remaining ~55 % smooth spline noise around the plan.
2. **Rollouts are not averaged into the answer.** Each rollout is scored individually; only the
   ≈ 1 % best (40 of 4096) are averaged, and only if they belong to the best candidate's manoeuvre
   mode. The result is the *wheel-speed plan*, not a trajectory average.
3. **No exponential weighting, no temperature** — rank-based elite selection (CEM). Robust to
   enormous penalty magnitudes, which are used deliberately.
4. **Global intent comes from `V(x, y, θ)`,** local safety from graded penalties; the horizon can
   therefore be short (2.5 s).
5. **Only `U[0]` is executed,** after a stage of deterministic output conditioning, and the whole
   thing is re-solved from fresh sensor data next frame, warm-started from a smoothed shift of the
   previous plan.
6. **Reduce friction risk by worst-casing only safety** across `μ` replicas.

> **Config caveat spotted while writing this:** `odin_elevation.params.yaml` sets `plan_n_mu: 3`,
> but `plan_batch` stays at the node default 4096, which is not divisible by 3. `MppiGpu.__init__`
> raises `ValueError` in that case, so either the batch is overridden at launch or that yaml/default
> pairing fails to start. Worth confirming before relying on the `n_μ = 3` numbers above.
