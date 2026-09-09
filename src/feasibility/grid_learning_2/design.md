# `GridDivergenceNet` — a per-cell command field, and an exact readout

What we are building, in plain language, before any code exists. This is the `grid_learning_2`
counterpart of `grid_learning/design.md`; read that file first — everything it says about
*why* the divergence problem is local, about relief normalisation, about the masked loss and the
mirror symmetry, still holds here and is not repeated in full. This document is about the two
things that change.

```
    grid_learning     IN : heightmap [100,100]  +  wz  (ONE scalar for the whole map)
                      OUT: error field [15,15,2]
                      how: wz FiLM'd into every trunk block; F.grid_sample readout

    grid_learning_2   IN : heightmap [81,81]    +  wz FIELD [15,15]  (one command per cell)
                      OUT: error field [15,15,2]
                      how: pure terrain trunk; command enters ONLY through 1x1 ops;
                           readout is an integer centre-crop, no interpolation
```

The prediction target is unchanged: `(e_pos, e_rot)`, the SE(3) final-pose error between ostrich
(dynamics) and helhest_stack (kinematic twin), at every cell of the 15x15 spawn lattice.

---

## 1. Why a command *field* rather than a command *scalar*

### 1a. It is the shape `docs/learned_divergence_penalty.md` actually asks for

That document defines the object being learned as a map from terrain to a field over **pose
only**, with the control folded in afterwards:

```
    N(H)  ->  c_hat(s)                    a K-vector per pose s
    d_hat(s, u)  =  < c_hat(s) , phi(u) >
```

and calls out *amortization* — one evaluation of `N` giving the divergence over the entire map,
for **every** control at once — as the property that makes it worth training at all.

`grid_learning`'s `GridPoseErrorNet` does not have that property. Its trunk is FiLM-conditioned on
`wz` in all 8 blocks, so the terrain encoder is control-dependent: asking "what if I turned at
-0.4 instead of +0.7 here" re-runs the whole CNN. Evaluating a planner's action set `A` at every
pose costs `|A|` full forward passes.

Moving the command out of the trunk and into a per-cell head restores exactly the factorisation
the doc assumes. The trunk becomes `N(H) -> c_hat`, a control-free terrain code computed once;
the head is `phi`, generalised from a fixed inner product to a small learned MLP. As a bonus, at
inference the head can be evaluated for `K` commands per cell at negligible cost (broadcast the
cell features against `K` command vectors), so the "field over pose x control" query the routing
field in that doc wants is one trunk pass plus a batched 1x1 head.

### 1b. It fixes a statistics problem in the v1 dataset

In a v1 row, all 225 cells share one `wz`. The command is therefore perfectly confounded with the
row: the effective sample size for learning the *control* dependence is the number of rows
(`M * L`, order 100 in the runs so far), not the number of labelled cells (order 225k). Every
cell in a row contributes a gradient through the same single `wz` value, so those 225 gradients
are correlated through the command path even though they are independent through the terrain path.

Drawing `wz` independently per cell decorrelates the command from map identity and gives
`M * L * 225` independent command draws — the same simulation cost, better-conditioned learning
of the one axis v1 is worst at.

### 1c. What it does NOT do

It does not add information. `generate_dataset.py` already flattens `(lattice cell, wz)` into a
single trial list before chunking, so a per-cell command field is a **re-bundling of exactly the
same trials**, not more of them. Terrain diversity is still bounded by the number of maps `M`,
and that remains the binding constraint on generalisation. Nothing here substitutes for
generating more maps.

---

## 2. The constraint this creates: the command may only enter through 1x1 ops

The label at lattice cell `(i, j)` is produced by one simulated trial: spawn at `spawn_xy[i,j]`,
command `wz[i,j]`, roll out. It depends on the terrain near that cell and on **that cell's own
command**. The commands at other cells are causally irrelevant — they belong to other worlds.

> **Therefore the command field may only ever be consumed by operations with a 1x1 spatial
> footprint.** Any convolution with kernel > 1, any pooling, any normalisation with spatial
> extent, applied *after* the command field is injected, lets cell A's prediction depend on cell
> B's command.

This is not pedantry, it is a deployment failure mode. Train on i.i.d. command fields, let the
net leak across cells, and then have the planner query a *constant* field (all cells the same
`wz` — the most natural query there is, and the one v1 files represent) and the net is being
asked to extrapolate to a command-field distribution it never saw. The failure would be silent
and would look like "the model doesn't transfer".

It also gives a free, exact unit test: perturbing `wz[i,j]` must leave every other cell's output
bit-identical, and `d y_hat[p,q] / d wz[i,j] == 0` for `(p,q) != (i,j)` under autograd. See
section 8.

**Corollary: there is no legal place to put the command except after the trunk.** "No FiLM in the
trunk" is not a stylistic preference — it is forced by the causal structure. (Note the asymmetry
with v1, where a *scalar* command is a legal trunk input precisely because there is only one of
it: FiLM's per-channel broadcast adds no cross-position dependence when the whole map shares one
value. Turn that scalar into a field and the same block becomes illegal.)

---

## 3. Grid geometry — the readout is an integer crop

### 3a. Why v1 needed `grid_sample` at all

v1's heightmap tensor is 100 cells @ 0.10 m — an **even** count, so cell centres sit at
`+-4.95, +-4.85, ...` and no pixel centre lies on the world origin. The spawn lattice is at
multiples of 0.5 m. Half a pixel apart, forever: lattice centre `-3.5` is pixel coordinate `14.5`.
Hence `F.grid_sample`, and hence `readout_offset()`, whose whole job is to undo a -0.15 m
registration error introduced by strided convs on an even grid.

That misalignment is a property of the grid convention, not of the problem. Fix the convention
and the readout becomes a slice.

### 3b. The rule

> **Make the input grid odd and origin-centred, and let the trunk's total stride equal the
> lattice pitch measured in input pixels.** Then the label lattice is an exact integer
> centre-crop of the final feature map.

A stride-2 `kernel=3, padding=1` conv maps output index `k` to input index `2k`. So index 0 stays
index 0, and an origin-centred odd grid stays origin-centred through every downsample — the
alignment is preserved by construction rather than corrected afterwards.

### 3c. The concrete numbers

Input resolution **0.125 m**, **81 x 81** cells, centres at `0.125 * (i - 40)` for `i = 0..80`:

```
   world extent covered:  cell centres  -5.000 .. +5.000 m
                          cell edges    -5.0625 .. +5.0625 m   (81 * 0.125 = 10.125 m)
```

The outer half-cell reaches 0.0625 m past the 10 x 10 m assets maps; `HeightMapReader.sample`
clamps rather than raises, which is the same boundary condition the trunk's `replicate` padding
uses, so this is consistent, not a bug. (Recorded in the file as `grid/extent = 10.125`.)

```
  input          81 x 81  @ 0.125 m   centres -5.000 .. +5.000, origin at i = 40
    | conv k3 s1  x2                      81 x 81  @ 0.125
    | conv k3 s2                          41 x 41  @ 0.250   centres -5.0 .. +5.0, origin at k = 20
    | conv k3 s1  x2                      41 x 41  @ 0.250
    | conv k3 s2                          21 x 21  @ 0.500   centres -5.0 .. +5.0, origin at k = 10
    v
  feature lattice 21 x 21 @ 0.50 m   ==   the spawn lattice pitch, exactly
  crop [3:18, 3:18]              ->   15 x 15 spanning -3.5 .. +3.5   ==  spawn_xy, exactly
```

Size arithmetic: `floor((81 + 2 - 3)/2) + 1 = 41`, `floor((41 + 2 - 3)/2) + 1 = 21`.

`spawn_xy` is still read from the dataset file — but it is now used as an **assertion** (the
cropped feature lattice's world coordinates must equal it to within 1e-6) rather than as
interpolation coordinates. The geometry is still data-driven; it is just also exact.

The crop offset is computed, never hardcoded: `offset = (n_feat - G) // 2`, with a check that
`n_feat - G` is even and that the coordinates match.

### 3d. Why 0.125 m and not 0.10 m

At 0.10 m the lattice pitch is 5 pixels, and 5 is prime — the trunk would need a single stride-5
stage (which makes the receptive-field arithmetic jump in 1.0 m steps and forces the cheap
early convs to run at full 101^2 resolution) instead of two ordinary stride-2 stages. At 0.125 m
the pitch is 4 = 2 x 2 and everything is a standard conv stack.

The cost of the coarser grid is nil for this problem: 0.125 m still resolves the 0.35 m wheel
(2.8 px) and localises a box edge to +-6 cm, while the *label* lattice pitch is 0.5 m, so terrain
detail finer than ~0.1 m cannot be attributed to a label cell anyway. And 81^2 is 34% fewer
pixels than 100^2.

The resampling is bilinear point sampling of a 0.05 m native asset, not area averaging, so a
feature thinner than 0.125 m could alias. Box obstacles are >= 0.5 m wide; if a future terrain
series has thin features this is the line to revisit.

### 3e. The rule generalises

Any lattice pitch that is a power-of-two multiple of the input resolution works with the same
trunk. A 0.25 m lattice (denser labels, proportionally more simulation) is the crop of the 41x41
stage instead of the 21x21 one; nothing else changes.

The odd grid also keeps `mirror_dataset.py`'s exactness argument intact: 81 centres at
`0.125*(i-40)` are symmetric about `y = 0`, so flipping the row axis is a true mirror, exactly as
the even 100-cell grid was (`+-4.95 ...` is symmetric too). The odd grid additionally has a cell
*on* the axis, which is fixed by the flip — harmless.

---

## 4. Input preprocessing

### 4a. Heightmap -> relief in wheel radii

Unchanged from `design.md` section 3a:

```
    h_rel  =  (h - median(h)) / WHEEL_RADIUS          # WHEEL_RADIUS = 0.35 m
```

Per-sample median as the background-ground estimate (a single scalar per map, so it recenters
every pixel identically and leaks no position information); a 0.70 m box reads as 2.0 wheel radii;
flat ground reads as exactly 0.

### 4b. Command field -> a 2-channel field

```
    cmd[:, 0, i, j] = wz[i,j] / WZ_MAX
    cmd[:, 1, i, j] = |wz[i,j]| / WZ_MAX             # WZ_MAX = 1.0
```

Same two numbers as v1, for the same reason (`|wz|` sets how far the fixed-duration turn sweeps,
the sign sets which way the rear wheel goes), now carried per cell as a `[B, 2, 15, 15]` tensor.

The channel count is a parameter (`N_CMD_FEATURES`), not a literal 2, so adding forward velocity
`v` later is a data-schema change rather than an architecture change. **For now the field carries
`wz` only** — `V_DRIVE` is 0 in every trial this dataset contains.

---

## 5. The network

### 5a. Top level

```
  heightmap [B,1,81,81]                             wz field [B,15,15]
  (absolute world z, m)                             (rad/s, one per lattice cell)
        |                                                   |
        v                                                   v
  +--------------------+                       +-----------------------------+
  | relief prep        |                       | (wz/1.0, |wz|/1.0)          |
  | (h - median)/0.35  |                       +-----------------------------+
  +--------------------+                                    |  [B,2,15,15]
        |                                                   v
        v                                       +-----------------------------+
  +===========================================+ | Conv1x1 2  -> 32   + SiLU   |
  |   TERRAIN TRUNK -- 7 conv blocks          | | Conv1x1 32 -> 32            |
  |   NO conditioning of any kind             | +-----------------------------+
  |   replicate pad, ChannelLayerNorm, SiLU   |               |  e [B,32,15,15]
  |   receptive field capped at 27 px = 3.375m|               |  (per-cell command
  +===========================================+               |   embedding)
        |                    |                                |
        |  L1 [B,64,41,41]   |  L2 [B,96,21,21]               |
        |  (RF 1.875 m)      |  (RF 3.375 m)                  |
        v                    v                                |
   slice [::2]               |                                |
   [B,64,21,21]              |                                |
        +--------- concat ---+                                |
                  |                                           |
          [B,160,21,21]                                       |
                  |                                           |
             crop [3:18,3:18]                                 |
                  |                                           |
          c(s) [B,160,15,15]  <-- the terrain code; control-free, section 1a               |
                  |                                           |
                  +---------------- FiLM ---------------------+
                  |     gamma, beta = Conv1x1(e): [B,320,15,15]
                  |     h = c * (1 + gamma) + beta      <-- PER CELL, 1x1, no mixing
                  v
          +--------------------------------+
          | SiLU                           |
          | Conv1x1 160 -> 128  + SiLU     |     a per-cell MLP, weights shared across cells
          | Conv1x1 128 -> 128  + SiLU     |
          | Conv1x1 128 ->   2             |
          +--------------------------------+
                  |
                  v
          y_hat [B,2,15,15] -> permute -> [B,15,15,2]
          in log1p / standardised space (section 6)
```

Every operation after `e` is injected is 1x1. That is the section 2 constraint, discharged
architecturally.

### 5b. The trunk, block by block

`s` is the conv stride. RF is in input pixels; multiply by 0.125 for metres.

| # | block | s | C_in -> C_out | out size | m/cell | RF (px) | RF (m) |
|---|-------|---|---------------|----------|--------|---------|--------|
| 1 | conv3 | 1 | 1 -> 32       | 81x81    | 0.125  |  3      | 0.375  |
| 2 | conv3 | 1 | 32 -> 32      | 81x81    | 0.125  |  5      | 0.625  |
| 3 | conv3 | **2** | 32 -> 64  | 41x41    | 0.250  |  7      | 0.875  |
| 4 | conv3 | 1 | 64 -> 64      | 41x41    | 0.250  | 11      | 1.375  |
| 5 | conv3 | 1 | 64 -> 64      | 41x41    | 0.250  | **15**  | **1.875** |
| 6 | conv3 | **2** | 64 -> 96  | 21x21    | 0.500  | 19      | 2.375  |
| 7 | conv3 | 1 | 96 -> 96      | 21x21    | 0.500  | **27**  | **3.375** |

Final RF = 27 px = **3.375 m diameter = 1.69 m radius**, against the 1.45 m rear-wheel rim sweep
(1.101 m axle offset + 0.35 m wheel radius) — 0.24 m of margin, the same budget v1 chose
(1.75 m vs 1.5 m). One more block would take it to 35 px = 4.375 m, which starts admitting a
second, causally unrelated obstacle. As in v1, **the block count is chosen to hit this number**;
the stride/width table is load-bearing, not a free tuning knob.

`replicate` padding and `ChannelLayerNorm` carry over verbatim from `design.md` section 4b,
including the reason `ChannelLayerNorm` is not negotiable: `GroupNorm`/`BatchNorm`/`InstanceNorm`
pool statistics over `(H, W)`, which silently makes every block's receptive field the whole map
regardless of the table above. v1's autograd RF probe caught exactly that, and this design keeps
the probe.

### 5c. The multi-scale readout (rows 5 and 7)

In v1, `|wz|` could tell the trunk *which radius matters* — a slow turn sweeps a small disc, a
fast one the whole 1.45 m rim — because FiLM gated the trunk's features. A control-free trunk
cannot know that, so it must hand the head enough to decide: features at two receptive fields,
1.875 m (block 5) and 3.375 m (block 7).

The block-5 map is 41x41 @ 0.25 m; **slice `[::2]`, do not pool.** The slice keeps indices
`0, 2, ..., 40`, i.e. world `0.5*(m-10)` for `m = 0..20` — exactly the 21x21 lattice. An
`avg_pool2d(2)` would average adjacent cells and land the result half a cell off, reintroducing
precisely the misregistration section 3 exists to remove. Discarding half the block-5 cells is
fine: this is a readout, not a bottleneck, and each retained cell's feature already summarises its
own 1.875 m neighbourhood.

### 5d. Why FiLM *and* a per-cell MLP

A fair question, since a concatenate-then-MLP head can approximate anything FiLM can. The
difference is parameterisation and optimisation, not expressive class:

* **The first layer of a concat head is additive in the command.** `Conv1x1([c; e])` computes
  `W_c @ c + W_e @ e + b`: the command can only *shift* the pre-activation. Every multiplicative
  interaction has to be reconstructed out of the downstream nonlinearities. But the physical
  relationship is close to multiplicative — "how much does terrain feature k (obstacle mass in
  this angular sector, at this radius) matter" is *gated* by how far the turn sweeps. FiLM makes
  that gating a primitive: `c * (1 + gamma(e))` is a learned per-channel gate.
* **Scale.** The head sees 160 terrain channels and one command. Under concat, the command is one
  input among 161 and its gradient is easily swamped early in training; a net that ignores `wz`
  and predicts the terrain-marginal mean is a real local optimum here (it is what v1's "per-`wz`
  mean field" baseline measures). FiLM makes the command act on *every* channel by construction,
  so it cannot be quietly dropped.
* **`(1 + gamma)`, not `gamma`,** initialises the head to an identity modulation, so an untrained
  net starts as a plain terrain->error map and learns the command dependence from there.
* **Cost.** FiLM is one `Conv1x1(32 -> 320)` = ~10.6k params. Reaching comparable command
  sensitivity by widening a concat head costs more.

So: FiLM supplies the multiplicative interaction, the MLP after it supplies the general nonlinear
mixing. Neither subsumes the other in practice. This is worth an ablation
(`--head-fusion {film,concat}`) rather than an argument — it is a cheap flag, and the concat head
is the honest control.

### 5e. Parameter count (estimate)

| part | params |
|---|---|
| conv trunk (weights + biases) | ~240k |
| ChannelLayerNorm | ~0.9k |
| command encoder (2 x Conv1x1) | ~1.2k |
| FiLM projection `Conv1x1(32 -> 320)` | ~10.6k |
| per-cell head (160->128->128->2) | ~37k |
| **total** | **~290k** |

Below v1's 368k despite the extra head, because 7 blocks of unconditioned conv are cheaper than 8
blocks with a `Linear(32 -> 2C)` each. `base_width` (32) remains the single scaling knob.

---

## 6. Target space, loss and mask — unchanged

Reused verbatim in spirit from `design.md` sections 5a-5c:

* **`TargetTransform`**: `log1p` then standardise; inverse clamps at 0 before `expm1` so a
  physically impossible negative error cannot escape. Fitted on **train rows only**, and only on
  cells with `mask == True` (masked cells are exact zeros by construction; folding thousands of
  them into the mean/std would wreck the standardisation).
* **Masked MSE**, normalised by `mask.sum()` rather than the cell count, computed in model space
  so the transform's clamp never kills a gradient.
* **Masked cells store exact zeros, never NaN** — `0 * NaN = NaN` poisons the backward pass even
  through a correct mask.
* The mask still means two things: *blocked* (spawn footprint on an obstacle, never simulated —
  a deterministic function of the map and independent of `wz`) and *diverged* (simulated but
  failed the finite / plausible-bounds / `MAX_SPAWN_DISPLACEMENT` checks — this one **does**
  depend on `wz`).

---

## 7. Data

### 7a. Schema: one field changes

`outputs/dataset_grid2_*.h5`, identical to `generate_dataset.py`'s schema except:

```
    wz          [R]           float32        ->   wz  [R, G, G]  float32
    (root attr) wz_per_cell = True                                  <-- new
    (root attrs) wz_zero_frac, wz_grid, n_rows_per_map, n_cells     <-- new, provenance for 7b
    (root attr) n_commands                   ->   dropped: a row is a FIELD of commands now
```

`grid/resolution` and `grid/extent` carry 0.125 and 10.125 rather than 0.10 and 10.0, and
`grid/heightmap` is `[n_maps, 81, 81]` — the section 3 grid, written by `utils.heightmap_to_tensor`
at generation time exactly as v1 writes its own.

`y [R,G,G,14]`, `mask [R,G,G]`, `spawn_xy [G,G,2]`, `map_index [R]`, `map_path`, `grid/heightmap`,
`grid/resolution`, `grid/extent`, `grid/map_source`, and the `git/` provenance group are all
unchanged, and still written through `comparator.provenance`'s `terrain_fields`/`git_provenance`
so provenance cannot drift.

**v1 files are not read.** An earlier draft had the loader broadcast `wz[:, None, None]` to a
constant field as a smoke-test path; that is gone. A constant field is a measure-zero corner of
the distribution v2 trains on, so accepting one silently would produce a dataset that looks fine
and teaches the head nothing about the command axis — and per-cell commands cost exactly the same
simulation (section 7b), so re-generating is the cheap option anyway. A `wz` of shape `[R]` is
rejected with a message naming the reason. Run with no path argument, `custom_dataset.py` writes a
small synthetic file instead, which pins the schema and exercises the whole loader before the
generator exists.

### 7b. Generation: per-cell sampling, same cost

`simulate_map` already flattens `(lattice cell, wz)` into one trial list before chunking, and both
sims take per-world spawn `[N,3]` and per-world setpoints `[T,N,3]`. So per-cell commands are a
change to *one line* of index arithmetic:

```
    v1:   wz_drive = wz_values[trial_l]                      # L values, tiled over cells
    v2:   wz_drive = wz_field[trial_l, trial_i, trial_j]     # R x G x G values, one per trial
```

Trial count per map is unchanged: `n_clear * L`. GPU cost, chunking, settle steps, all unchanged.

Sampling, per row `l` and clear cell `(i,j)`:

```
    wz[l,i,j] ~ U(WZ_RANGE)          i.i.d.,  WZ_RANGE = (-1.0, 1.0)
```

with `+wz_zero_frac` (default 0.05) of cells forced to exactly 0. Continuous sampling covers the
command axis far better than v1's `linspace(-1, 1, L)` grid — that is half the point of the
change — but it puts zero mass on `wz = 0`, which is where the one command-side sanity check with
a known answer lives (no command, no motion, error ~ 0). A 5% anchor keeps that check honest for
the price of a few easy cells. `+wz_zero_frac=0` disables it; `+wz_grid=K` restricts the draws to
v1's discrete levels for a controlled comparison — note that it constrains the *values* only, so
each cell still draws independently and it does **not** reproduce v1's one-command-per-row layout,
which is the thing this change exists to remove.

The rows-per-map knob is `+n_rows`, not v1's `+n_commands`: a v2 row is a command *field* holding
`G*G` commands, so the old name would be off by a factor of 225.

Masked (non-clear) cells get a command too — an arbitrary in-range draw — so the `wz` array is
never structurally NaN or zero-filled in a way a reader could confuse with a real command. It is
never read for those cells, since the loss is masked.

RNG consumption is documented in the generator: map selection, then the command field, in that
order, so a seed reproduces a file exactly.

### 7c. The mirror symmetry, with one extra line

Still exact (`mirror_dataset.py`'s argument is about robot and world geometry, not about the
command's shape), and still worth a free 2x. As a train-time augmentation with p = 0.5:

```
    heightmap ->  flip(heightmap, dims=[-2])         # rows = world Y
    wz field  ->  -flip(wz, dims=[-2])               # flip the FIELD, then negate  <-- new
    y         ->  flip(y,    dims=[-3])
    mask      ->  flip(mask, dims=[-2])
```

Both halves matter: the command belonging to cell `(i,j)` must follow that cell to row `G-1-i`,
*and* its sign flips because a mirrored turn is the opposite turn.

### 7d. The split is by map, not by row

As in v1: group rows by `map_source` (which handles mirrored maps grouping with their source),
split the *maps* 80/20, send all rows of a map to the same side. A row-level split would put a
val row's terrain in train under a different command field and measure interpolation in `wz` on
memorised terrain.

This matters *more* here than in v1, not less: per-cell commands make each row's command content
richer, so a leaked map is an even easier row to fit.

---

## 8. Self-checks and baselines

Every module's `if __name__ == "__main__":` block is its test (there is no pytest suite in this
tree). The checks that must exist:

**In `model.py`:**

* **Receptive field** — autograd probe on the trunk output must measure exactly 27 px, matching
  the formula. (This is the check that caught `GroupNorm` in v1; keep it.)
* **Crop alignment** — the cropped feature lattice's world coordinates must equal `spawn_xy` to
  < 1e-6. Replaces v1's `readout_offset` probe; it is now an equality, not a correction.
* **Per-cell command independence** — `d y_hat[p,q] / d wz[i,j] == 0` for all `(p,q) != (i,j)`,
  by autograd. This is section 2's constraint made testable, and it is exact, not approximate.
* **Flat ground** — `relief()` of a constant heightmap is exactly zero.
* **Mirror geometry, and mirror equivariance.** These are two different things and only the first
  is exact at initialisation — a conv stack with random weights is *not* reflection-equivariant,
  so `f(flip(h), -flip(wz)) == flip(f(h, wz))` is a property training on the augmentation
  produces, not one the odd grid confers. What the odd grid *does* confer, exactly and
  weight-independently, is **registration**: the readout lattice is antisymmetric about `y = 0`
  (`coords[G-1-i] == -coords[i]`), and output row `p` and output row `G-1-p` draw on exactly
  mirrored input rows (checkable by comparing autograd support masks). Both are asserted. The
  model-space equivariance number is *reported* — as v1's `train.py` reports it, on a trained
  model — not asserted.
* **`TargetTransform` round-trip** and non-negativity of the inverse.

**At eval time, in `train.py`:**

| baseline | what it is | what beating it proves |
|---|---|---|
| global mean | one constant over the dataset | a sanity floor |
| per-`wz` mean | binned mean field looked up by the cell's own `wz` | the model uses terrain at all |
| blur-the-terrain | same net, relief replaced by its per-sample mean | the model uses terrain *structure*, not just "is there a box somewhere" |
| concat head | `--head-fusion concat` | that FiLM earns its place (section 5d) |
| v1 net, constant fields | `GridPoseErrorNet` on the same maps | that the reframing helps |

Reported in physical units on **held-out maps**: masked RMSE over all valid cells (m, rad), masked
RMSE over the top decile by true `e_pos` (the collision cells, the entire point), R^2 per head.
Plus the `wz ~ 0` check (cells commanded ~0 must predict ~0) which section 7b's zero anchor keeps
available.

Note the `blur-the-terrain` baseline applies to the *relief*, not the raw heightmap — blurring the
raw input gives identically zero relief and collapses the baseline into the per-`wz` mean. Same
reasoning as v1.

---

## 9. Alternatives considered and rejected

**Inject the command field early, as extra input channels.** Upsample `wz` to 81x81 and
concatenate to the relief. Simple, and standard for conditioning a dense-prediction net — but it
violates section 2 outright: the very first 3x3 conv mixes neighbouring cells' commands. Rejected
on causal grounds, not empirical ones.

**Inject the command field at the last trunk stage.** The 21x21 stage is already at lattice pitch,
so the shapes line up — but any conv after it with kernel > 1 has the same leak. Restricting those
convs to 1x1 *is* the head. So this is not a different design, it is this one.

**Strict bilinear head `d_hat = <c(s), phi(u)>`.** The literal form in
`docs/learned_divergence_penalty.md`, and the cheapest possible thing for the planner to consume
(precompute `c` per cell, then every control query is a dot product). Kept as an ablation flag,
not the default: it forces the control dependence to be linear in a fixed basis, and there is no
evidence yet that the true dependence on `|wz|` is that simple. If the ablation ties the MLP head,
prefer it for deployment.

**Remix v1 files into per-cell fields.** For each new row, pick a random command index per cell and
gather `y`/`mask` with it. Because every trial is an independent parallel world, this yields
*genuinely valid* labels at zero simulation cost, and would have been a fast bootstrap. Not on the
critical path now that path 3 (native generation) is the plan — it adds no information, only
re-bundling, and the command values stay on v1's coarse `linspace` grid. Worth remembering if GPU
time gets tight.

**Keep 0.10 m and one stride-5 stage.** Section 3d.

**Avg-pool the block-5 map to 21x21 instead of slicing.** Section 5c — reintroduces a half-cell
offset.

**U-Net / global bottleneck, flatten-then-MLP.** Rejected for the same reasons as in
`design.md` section 8; the locality argument is unchanged.

---

## 10. Training recipe (starting point)

```
    optimiser     AdamW, lr 3e-4, weight_decay 1e-4
    schedule      cosine to 0, ~5 epochs linear warmup
    batch size    16 rows  (= 16 x 225 labelled cells per step)
    epochs        ~200, early stopping on held-out-MAP val loss
    augmentation  y-mirror + command-field flip/negate, p = 0.5
    logging       Weights & Biases, as learning/train.py and grid_learning/train.py
    checkpoint    outputs/checkpoints/, self-describing, WITH the fitted TargetTransform
                  (fitted data, not a learned parameter -- not in state_dict(), must be
                   saved and re-attached explicitly)
```

---

## 11. Files this implies

`src/feasibility/grid_learning_2/`, a plain importable package (`__init__.py`), every module also
runnable standalone as a script — same convention as `grid_learning/`.

| file | role |
|---|---|
| `design.md` | this document |
| `utils.py` | odd origin-centred grid (`grid_coords_centered`), `heightmap_to_tensor`, `load_heightmap_tensor`, and the readout arithmetic (`conv_out_size`, `downsampled_coords`, `crop_offset`) + `spawn_lattice`/`SPAWN_STEP`/`SPAWN_LIMIT`. The lattice constants live here rather than in `generate_dataset.py` (where v1 keeps them) because in v2 the lattice geometry is a *contract* between the dataset and the network's crop, so both sides must read it from one place |
| `custom_dataset.py` | `GridDivergenceDataset` — per-cell `wz` (a v1-shaped `[R]` is rejected, not broadcast), map-level split only, vectorised SE(3) error, `valid_targets()` for the train-only transform fit. Imports `TARGET_NAMES`/`Normalizer`/`TargetTransform` **from `model.py`** (the opposite of v1's direction), so `model.py` has no sibling imports beyond `utils.py`'s pure geometry and can be run and checked before any dataset exists |
| `model.py` | `GridDivergenceNet` (sections 4, 5) + `TargetTransform` + the section 8 self-checks. `forward(heightmap, wz)` only — no `spawn_xy`, no `extent`: the readout is fixed integer geometry, and `check_lattice_alignment(spawn_xy)` verifies it once at setup |
| `generate_dataset.py` | per-cell command sampling (section 7b), Hydra-driven, writes `dataset_grid2_*.h5`. Physics identical to v1's generator (same trials, same chunking, same cost); what differs is `sample_command_fields` and the one-line gather in `simulate_map`, plus the odd 81-cell grid it writes. Calls `model.check_lattice_alignment` **before** simulating, so a mis-sized `+cells`/`+resolution` fails in milliseconds rather than producing a file that loads fine and trains on misregistered labels. Defaults to `assets/large_box_random/<seed>/` — a whole-map divergence field starves for signal on the small fixed box series |
| `train.py` | masked loss, map split, mirror augmentation, baselines |
| `gl_replay_grid.py` | QA viewer for a generated file: steps a row's cells, freezing both sims' meshes at each cell's stored final pose, lattice drawn as valid/blocked/diverged spheres. `--dry-run` needs no display and is the useful half — label percentiles plus the worst cells listed as ready-to-paste `--cell` arguments |

Later, if the experiment survives: `mirror_dataset.py`, `eval_log.py`, `test_nn.py`.

An earlier draft of this table claimed v1's `gl_replay_grid.py` runs on a v2 file "except for the
`wz` shape — a two-line read change". Measured against the file the generator writes, that was too
optimistic, which is why v2 has its own copy. Three reads break — `f.attrs["n_commands"]` (v2
writes `rows_per_map`; this is where v1's dies), `float(f["wz"][row])` (a `[G, G]` field is not a
scalar), and the status line that prints one `wz` for the whole row instead of `wz[i,j]` per cell
— and the whole `--nn-checkpoint` branch is v1-model-specific: it loads a `GridPoseErrorNet`
through `grid_learning.train.load_checkpoint` and calls `predict(heightmap, wz, spawn_xy,
extent=...)`, a signature `GridDivergenceNet` deliberately does not have, so v2's copy simply omits
it until `train.py` defines a checkpoint format. Everything else transfers untouched (row ordering,
`spawn_xy`/`y`/`mask` layout, `terrain_from_h5`, the footprint filter, the SE(3) formula, all the
rendering) — including the terrain mesh, which comes from the embedded raw asset grid rather than
from `grid/heightmap`, so the 81 @ 0.125 m change is invisible to the viewer.

**Dependency stance:** self-contained, in the same spirit as `grid_learning/` is towards
`learning/`. `feasibility.heightmap` and `feasibility.comparator` are shared infrastructure and
are imported; `feasibility.grid_learning` is **not**. The grid convention genuinely differs (odd
vs even, 0.125 vs 0.10 m) and a shared `utils.py` would have to serve both, which is exactly how
two experiments start silently constraining each other. `WHEEL_RADIUS`, `WZ_RANGE`, `SPAWN_STEP`,
`SPAWN_LIMIT` are restated locally with the same "restate rather than import" rationale
`grid_learning` gives — importing them from `examples.helhest_junior.common` would drag
ostrich/warp into every training run for the sake of a few floats.
