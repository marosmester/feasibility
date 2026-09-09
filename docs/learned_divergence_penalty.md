# A learned divergence penalty for the Helhest planner

How the `grid_learning` CNN's output is defined mathematically, and how the two halves of
`helhest_stack`'s motion stack — the local trajectory optimizer and the global routing field —
would consume it. Written model-first and code-agnostic; see
[helhest_stack_motion_planning.md](helhest_stack_motion_planning.md) for the stack it plugs into,
and the status note at the bottom for what exists today versus what has to be built.

---

## The setup

Two things propagate a robot state forward over a fixed horizon:

```
state       s = (x, y, θ)          pose on the plane
control     u = (v, ω)             forward speed, yaw rate
terrain     H(x, y)                a height field

Φ_ref(s, u; H)     accurate, expensive   — the dynamics simulator
Φ_twin(s, u; H)    cheap, approximate    — the model the planner uses
```

Their disagreement is a scalar:

```
d(s, u; H) = ρ( Φ_ref(s,u;H) , Φ_twin(s,u;H) )

where   ρ = ‖Δposition‖ + L · ‖Δheading‖
```

`L` is a lever arm (a characteristic body length) that converts radians into the metres of
displacement they cause — so `d` comes out in metres. It is ≥ 0, and `d ≡ 0` would mean the twin
is exact.

**This `d` is the whole object of interest.** It is a deterministic, systematic function of
terrain, pose and control — not noise. That is exactly why it is learnable.

---

## What the CNN is

Evaluating `d` at one point costs two rollouts, one of them expensive. The network amortizes
that. It is a map from a whole terrain to a **vector-valued field over pose space**:

```
    N(H)  →  ĉ(s) ∈ ℝᴷ           one K-vector per pose s
```

and the divergence at any pose *and any control* is recovered by an inner product against a small
fixed basis `φ(u)`:

```
    d̂(s, u) = ⟨ ĉ(s), φ(u) ⟩ = Σ_k  ĉ_k(s) · φ_k(v, ω)
```

Three properties matter:

**Amortization.** One evaluation of `N` gives `d̂` over the *entire* map, for *every* control, at
once. Pointwise evaluation of the true `d` would cost thousands of expensive rollouts.

**Locality.** `d(s,u;H)` depends on `H` only through the region the robot sweeps out over the
horizon — a bounded neighbourhood of `s`. So `N` can be a convolution with bounded support. This
is the inductive bias that lets it learn from few terrains: it *cannot* memorise which map it is
looking at, only local geometry.

**Separability in control.** Factoring the control dependence into `φ` keeps the output a field
over *pose only*. Otherwise you would need a field over the joint pose–control space, too large to
store or query.

If you regress an upper quantile instead of the mean, everything below still holds and gets a
stronger interpretation.

---

## Consumption 1 — the trajectory optimizer

It minimizes, over control sequences `U = (u₀ … u_{T-1})`:

```
    J_twin(U) = Σ_t  ℓ(s_t, u_t)        with   s_{t+1} = Φ_twin(s_t, u_t)
```

**The pathology is structural.** Every state in that sum is produced by the twin. So the optimizer
is minimizing the wrong objective wherever twin ≠ ref — and it does so *adversarially*: an argmin
actively seeks regions where the surrogate is optimistic. Model bias does not average out under
optimization, it gets exploited.

Concretely: if the twin believes it can rotate through the side of an obstacle, that manoeuvre
isn't merely unpunished — it looks **cheap**, and the optimizer prefers it.

The correction is one extra term:

```
    J′(U) = J_twin(U) + λ · Σ_t  w_t · d̂(s_t, u_t)
```

with `w_t` decaying in `t` (near-term errors matter more; replanning corrects later ones).

There's a clean justification. If the stage cost is Lipschitz in state with constant `L_ℓ`:

```
    | J_true(U) − J_twin(U) |  ≤  L_ℓ · Σ_t d(s_t, u_t)
```

so for `λ ≥ L_ℓ`:

```
    J_twin(U) + λ·Σ d̂   ≳   J_true(U)
```

**Minimizing `J′` minimizes an upper bound on the true cost.** That is the standard robust-control
move; the network is what makes the bound *spatially resolved* rather than one global constant.
With a quantile head it becomes an upper *confidence* bound.

In model-based RL language: `λ·Σ d̂` is a **pessimism penalty** confining the optimizer to where
its surrogate is trustworthy — a learned trust region rather than an assumed one.

---

## Consumption 2 — the routing field

The global router solves a Bellman equation over a pose lattice with a small finite action set `A`:

```
    V(s) = min over u ∈ A of  [ g(s,u) + V( F(s,u) ) ]

    V(goal) = 0 ,   V = +∞ on infeasible poses
```

where `g` is a stage cost (path length weighted by terrain quality). Modify it:

```
    g′(s,u) = g(s,u) + μ · d̂(s,u)
```

Then `V′` is the **accumulated minimum, over all routes to the goal, of length plus distrust** —
the cost of the cheapest route that also stays where the model is valid. Since `V` is the dominant
term in the local optimizer's objective, this propagates automatically: a distrust-aware `V′`
reshapes local planning without modifying it.

But note what is lost. `V′` is:

- **path-integrated** — one untrustworthy pose is smeared into a route total
- **marginalized over `A`** — reflects the router's coarse action set, not the control actually
  chosen
- **fixed at routing cadence**

So `V′` separates *corridors*; the explicit penalty in `J′` separates *trajectories within a
corridor*. Complementary, not redundant.

---

## Why this is an improvement, not a tuning knob

**It reaches an error of omission.** The twin's existing self-reported failures — a pose it can't
resolve, a tilt past limits — are errors of *commission*: the model knows something is wrong. Its
dangerous failures are errors of *omission*: it confidently reports a valid pose for a physically
impossible configuration. A model cannot penalize what it cannot represent, so no reweighting of
existing terms reaches this. Only an external reference can, and `d̂` is that reference compressed
into something evaluable in real time.

**It makes unmodeled error a first-class, tradeable cost.** Today model error is an unmodeled
disturbance — the planner has no representation of it and cannot reason about it. After this it is
a scalar field traded off against path length, exactly as tilt already is. That is a qualitative
change in what the planner can express.

**It inherits the asymmetry of the real failure.** `d` is large precisely where the twin's specific
structural simplification breaks — zero on open flat ground, large only near the geometry that
defeats the approximation. Nothing about it is generic conservatism.

---

## The honest limits

- `d̂` is a *learned* approximation of `d`, so the bound argument holds only as far as the network
  generalizes — valid on the terrain distribution it was fit to, and nowhere else.
- It predicts divergence from the *reference simulator*, so its worth is capped by that simulator's
  own fidelity to reality.
- It is a soft preference. It steers away from where the plan is fiction; it does not make the plan
  correct there.

---

## Status: what exists, what has to be built

### Already in place

**Producing `d` (`src/feasibility/`).** `grid_learning/generate_dataset.py` already runs both
simulators over a spawn lattice per map and stores the final-pose pair per cell — that *is* `d`,
before the `ρ` collapse. `grid_learning/model.py` is a capped-receptive-field fully-convolutional
trunk with a per-cell readout, i.e. already the `H → field over pose` operator shape;
`train.py` has the masked loss, map-level split, mirror augmentation and baselines;
`gl_replay_grid.py` is the QA viewer. `learning/pose_error.py` computes the SE(3) error the `ρ`
metric is built from.

**Consuming a field (`helhest_stack/`).** Both consumption slots already exist structurally:

- The MPPI cost kernel already samples a `[ny, nx, n_theta]` device field trilinearly at every
  rollout pose, and `set_lattice` already uploads one into the buffer the captured graph reads. It
  also already accumulates *graded, early-weighted* penalties — the `w_t` shape in `J′`.
- The cost-to-go's feasibility kernel already produces a continuous per-pose penalty
  (`graded_tilt`) and its value iteration already weights each motion primitive's arc by the mean
  of that penalty over the cells the arc sweeps — the `g′` slot, ready for a `+ μ·d̂` term. Its
  motion primitives have known `(v, ω)`, so `φ(u)` can be evaluated per primitive on the host at
  table-build time.

### Needed for either consumption path

1. **Regenerate the dataset over planner-reachable commands.** This is the blocker. The current
   grid dataset is `v = 0` (pure in-place spin) with spawn yaw locked at 0. The planner's wheel box
   is non-negative, so `v = 0 ⟺ ω = 0` — every trial in the dataset lies outside the control set
   the planner can execute. Resample `v ∈ [0, v_max]`, `ω` inside the wheel box, and put spawn yaw
   on the routing lattice's heading bins (under forward motion the swept region is a directed arc,
   so yaw stops being a nuisance dimension and becomes dominant).
2. **Move the command out of the trunk and into the readout.** Today `ω` enters via FiLM through
   the conv trunk, so one forward pass yields the field for *one* command value — unusable when
   every rollout step has its own `(v, ω)`. Emit `K` basis coefficients `ĉ(s)` per cell instead,
   and evaluate `⟨ĉ(s), φ(u)⟩` at the consumer. A first basis: `φ = (1, |ω|, v, v|ω|, ω², v²)`.
3. **Widen and de-symmetrize the receptive field.** The current 3.5 m symmetric cap was sized for a
   spin. Forward travel over the horizon reaches several metres ahead, so the support must become
   asymmetric — long forward, short behind.
4. **Collapse the two heads to one scalar.** Train on (or reduce to) `d = e_pos + L·e_rot` with
   `L ≈ 1.101 m` (the rear-wheel lever arm), so the consumer gets one number in metres,
   commensurate with the existing violation terms.
5. **Optional but recommended: a quantile/σ head.** One extra output channel and a pinball loss;
   upgrades `J′` from a mean penalty to an upper confidence bound.
6. **Inference-side field construction.** Run the net at routing cadence (never inside the captured
   graph) on the planner's own window, and emit `ĉ` on the consumer's grid and frame. Cell size is
   a hard contract — a fully-convolutional trunk generalizes over extent but *not* over resolution,
   since the receptive field in metres changes. Assert it, resample if it differs.
7. **Close the perception gap.** The net is trained on exact synthetic heightmaps; the planner sees
   occlusion shadows, optimistically-inpainted unknown cells, and ICP drift. Decide whether to
   train on the dilated wheel envelope (what the settle actually samples) rather than raw
   elevation, and whether to augment generation with the occlusion pipeline.

### Additionally needed for consumption 2 (routing)

8. A `d̂` field on the router's **coarse** grid with its `n_theta` heading axis, and the
   per-primitive `φ(u)` evaluation folded into the primitive cost table.
9. A `μ` weight, and a decision on whether distrust also *hard-blocks* above a threshold or only
   grades. (Grading is the safer default and matches the existing design.)

### Additionally needed for consumption 1 (trajectory optimizer)

10. An extra device array `[ny, nx, n_theta, K]` uploaded alongside the routing field, plus the
    constant frame-offset bookkeeping the routing field already does (the fine and coarse windows
    share a snap grid, so the offset is precomputable).
11. A basis evaluation and one new weight `λ` inside the cost kernel — array reads only, so it
    stays graph-capturable.

### Needed to show any of it works

12. **A closed loop whose executor is the reference simulator, not the twin.** The current eval
    harness drives a single instance of the twin as "ground truth", so the plan→reality gap is zero
    by construction and a divergence penalty can only ever look like a pure cost. Demonstrating
    benefit requires executing the planned command in the dynamics simulator and measuring
    task success — otherwise there is no way to distinguish "avoided a real failure" from "took an
    unnecessary detour".
