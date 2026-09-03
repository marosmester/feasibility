# `GridPoseErrorNet` — the divergence-field architecture

What we are building, in plain language, before any code exists.

**The job.** Given one whole heightmap and one commanded yaw rate, predict how badly the fast
kinematic twin (helhest_stack) will disagree with the accurate dynamics simulator (ostrich) —
*at every point of a 15x15 lattice of possible spawn positions at once*.

```
    IN :  heightmap [100, 100]  (world z in metres)   +   wz  (scalar, rad/s)
    OUT:  error field [15, 15, 2]                     =   (e_pos in m, e_rot in rad) per cell
```

This is the model that consumes `custom_dataset.GridPoseErrorDataset`, which is fed by
`generate_dataset.py`. Read those two module docstrings first; this file assumes them.

---

## 1. The one fact that drives every design choice: the physics is LOCAL

In this dataset `V_DRIVE = 0`. Every trial is a **turn in place** for 2.4 s at some
`wz` in [-1, 1] — so at most +-2.4 rad (+-137 deg) of rotation, and almost no translation.

The divergence being measured has exactly one cause: helhest_stack's quasi-static settle only
resolves *vertical* support, so it rotates straight through the side of a box, while ostrich
jams against it. So the error at a spawn cell is decided by **whether the robot's swept
footprint hits an obstacle**, and by nothing else.

That swept footprint is a disc of radius ~1.5 m around the spawn point. (`SPAWN_LIMIT = 3.5` on
a 10 m map leaves exactly 1.5 m of padding for precisely this reason — see
`generate_dataset.py`'s module docstring.)

```
        3.5 m   =   35 heightmap pixels   =   the network's receptive field
    <--------------------------------------------------->
    +---------------------------------------------------+
    |                                                   |
    |             .-''''''''''''''''''-.                |
    |           .'                      '.              |
    |          /    robot's swept disc    \             |
    |         |         r = 1.5 m          |            |
    |         |                            |            |
    |         |             X              |            |   X = the ONE spawn cell
    |         |          (spawn)           |            |       being predicted here
    |         |                            |            |
    |          \                          /             |
    |           '.                      .'              |
    |             '-..................-'                |
    |                                                   |
    +---------------------------------------------------+
              0.25 m of margin on every side
```

**Consequence:** ~97% of the map is causally irrelevant to any single lattice cell. A network
that can see the whole map (a U-Net bottleneck, a transformer, a flatten-then-MLP) is not more
powerful here — it just has a memorisation pathway it does not need. So:

> The trunk is **fully convolutional with a deliberately capped receptive field of 35 pixels
> (3.5 m)**. It is architecturally incapable of using map-global context.

A convolutional trunk with receptive field R applied densely *is* a shared patch model over
RxR windows — the same hypothesis class as `learning/terrain_patch.py`, but computed once with
shared arithmetic instead of 225 times per map.

---

## 2. Grid geometry — how the two grids line up

The input grid and the output grid are different resolutions but the **same convention**
(row index runs along +Y, column along +X, cell-centre sampling), so they overlay directly.

```
        world Y
           ^
    +5.0 m |  +-------------------------------------------+  <- heightmap edge
           |  |                                           |
           |  |     .  .  .  .  .  .  .  .  .  .  .  .    |
    +3.5 m |  |     .  .  .  .  .  .  .  .  .  .  .  .    |  <- lattice row i = 14
           |  |     .  .  .  .  .  .  .  .  .  .  .  .    |
           |  |     .  .  .  .  .  .  .  .  .  .  .  .    |
       0.0 |  |     .  .  .  .  .  o  .  .  .  .  .  .    |     o = world origin
           |  |     .  .  .  .  .  .  .  .  .  .  .  .    |
           |  |     .  .  .  .  .  .  .  .  .  .  .  .    |
    -3.5 m |  |     .  .  .  .  .  .  .  .  .  .  .  .    |  <- lattice row i = 0
           |  |                                           |
    -5.0 m |  +-------------------------------------------+
           +----------------------------------------------------> world X
             -5.0        -3.5        0.0        +3.5      +5.0

           heightmap : 100 x 100  @ 0.10 m/cell, spans +-5.0 m
           lattice   :  15 x  15  @ 0.50 m/cell, spans +-3.5 m
```

Zooming in on one axis, the lattice pitch is exactly **5 heightmap pixels**, but offset by
**half a pixel**:

```
  heightmap pixel:    12     13     14  |  15     16   ...    84  |  85     86
  world x (m):      -3.75  -3.65  -3.55 | -3.45  -3.35        3.45|  3.55   3.65
                                        |                        |
  lattice cell j:                       0                       14
  lattice world x:                    -3.50                   +3.50
```

Lattice centres fall on pixel *corners* (pixel coordinate `14.5 + 5j`), not pixel centres.

> **This is why the readout is `F.grid_sample`, not a slice or a stride-5 pool.** Sampling at
> the world coordinates stored in the file (`spawn_xy`) is exact, needs no padding/stride
> gymnastics, and keeps working if `SPAWN_STEP`, `SPAWN_LIMIT`, `resolution` or `extent` ever
> change.

---

## 3. Input preprocessing

### 3a. Heightmap -> relief in wheel-radii

The raw tensor is absolute world z in metres. Two problems: (1) the absolute baseline is a free
*map-identity* cue the net can memorise, and (2) metres are not the natural scale for
"can a wheel climb this".

```
    h_rel  =  (h - median(h))  /  WHEEL_RADIUS          # WHEEL_RADIUS = 0.35 m
```

The median is the background-ground estimate — the same one
`generate_dataset.obstacle_height_threshold` uses, and valid for the same reason (background
covers most of the map; stays valid up to ~14 boxes). After this, a 0.70 m box reads as `2.0`
"wheel radii", flat ground reads as `0.0`, and box *height* generalises across the series —
exactly the normalisation `learning/terrain_patch.py` already validated.

This is a fixed transform, not a learned or fitted one. It is computed per sample inside
`forward()`, so nothing has to be stored in a checkpoint.

### 3b. wz -> a 2-vector

```
    wz_feat  =  ( wz / WZ_MAX ,  |wz| / WZ_MAX )        # WZ_MAX = 1.0
```

Two numbers because they mean different things physically: `|wz|` sets **how far** the robot
turns in the fixed 2.4 s (so how much of the disc gets swept), and the sign sets **which way**
the rear wheel goes. Handing the net both saves it from having to carve `|.|` out of a linear
layer.

---

## 4. The network

### 4a. Top-level

```
   heightmap [B,1,100,100]                     wz [B]
   (absolute world z, m)                       (rad/s)
          |                                       |
          v                                       v
   +-----------------+                +------------------------+
   | relief prep     |                | (wz/1.0, |wz|/1.0)     |
   | (h - median)    |                +------------------------+
   |   / 0.35        |                            |
   +-----------------+                            v
          |                              +------------------+
          |                              | Linear 2  -> 32  |
          |                              |      SiLU        |
          |                              | Linear 32 -> 32  |
          |                              +------------------+
          |                                       |
          |                                       |  e  [B,32]
          |                                       |  (the command embedding —
          |                                       |   ONE vector, broadcast to
          |                                       |   every conv block below)
          v                                       |
   +=========================================================================+
   |                        CONV TRUNK  (8 FiLM blocks)                      |
   |     fully convolutional, receptive field capped at 35 px = 3.5 m        |
   +=========================================================================+
          |
          v
   feature map [B,96,25,25]  @ 0.40 m/cell, spanning world +-5.0 m
          |
          v
   +----------------------------------------------+
   | F.grid_sample  at the 15x15 spawn_xy coords  |
   +----------------------------------------------+
          |
          v
   per-cell features [B,96,15,15]
          |
          v
   +--------------------------------+
   | Conv1x1 96 -> 64  +  SiLU      |    (a per-cell MLP, shared across cells)
   | Conv1x1 64 ->  2               |
   +--------------------------------+
          |
          v
   y_hat [B,2,15,15]  ->  permute  ->  [B,15,15,2]
   in log1p / standardised space (see section 5)
```

### 4b. One FiLM block, in detail

Every block is the same shape. FiLM = "feature-wise linear modulation": the command `wz` is not
concatenated as an input channel (the first conv could then only use it *additively*); instead
it produces a per-channel gain and bias that rescale the terrain features. This lets `wz`
genuinely gate *which terrain features matter*, which is what it physically does.

```
   x [B,C_in,H,W]
        |
        v
   Conv2d(C_in -> C_out, kernel 3, stride s, padding 1, padding_mode='replicate')
        |
        v
   ChannelLayerNorm(C_out)     <- normalizes over channels ONLY, per spatial position
        |                                   e [B,32]
        |                                       |
        |                                       v
        |                          Linear(32 -> 2*C_out)
        |                                       |
        |                              split into gamma, beta   (each [B,C_out])
        |                                       |
        v                                       v
        +-------->  h * (1 + gamma)[:,:,None,None] + beta[:,:,None,None]
                                     |
                                     v
                                   SiLU
                                     |
                                     v
                               out [B,C_out,H',W']
```

Notes on the choices:

* **`padding_mode='replicate'`, never zeros.** Zero padding stamps a distinctive constant
  signature at the border, which is a known route for a CNN to encode absolute position — a
  memorisation pathway we do not want. Replicate padding instead extends the edge height
  outward, which is *exactly the boundary condition the ground truth used*:
  `HeightMapReader.sample` clamps to the nearest edge cell rather than raising (see
  `utils.py`'s docstring). So this is not a hack, it is the physically correct extension.
* **`ChannelLayerNorm`, not `GroupNorm`/`BatchNorm`/`InstanceNorm`.** This one is load-bearing,
  not stylistic, and was caught by the implementation's own receptive-field self-check: GroupNorm
  (and InstanceNorm, and BatchNorm) compute their mean/std over the **whole spatial extent**
  (H, W) jointly with the channels in a group, per sample. That means a block's output at any one
  position depends on statistics pooled from *every* position in the feature map — silently
  making the receptive field the whole 100x100 map from the very first block, no matter what the
  conv kernel/stride table says. `ChannelLayerNorm` instead normalizes only across the channel
  dimension, independently at each `(h, w)` position (the ConvNeXt-style "LayerNorm2d") — no
  spatial pooling, so it cannot leak position information, and the conv geometry alone determines
  the RF. Verified empirically in `model.py`'s self-check: an autograd probe on the trunk's output
  now shows a nonzero-gradient region of exactly 35 px, matching the table below.
* **`SiLU`, not `ReLU`.** Matches `learning/model.py`; the field is smooth away from the
  grazing-contact boundary, and dead ReLU units waste a small network.
* **`(1 + gamma)`, not `gamma`.** Initialises the block to an identity modulation, so an
  untrained net starts as a plain CNN and learns the `wz` dependence from there.

### 4c. The trunk, block by block

`s` is the conv stride. Receptive field (RF) is in heightmap pixels; multiply by 0.10 m for
metres. Nothing here is padded away — every block keeps the full +-5.0 m world extent.

| # | block | s | C_in -> C_out | out size | m/cell | RF (px) | RF (m) |
|---|-------|---|---------------|----------|--------|---------|--------|
| 1 | conv3 | 1 | 1 -> 32       | 100x100  | 0.10   |  3      | 0.3    |
| 2 | conv3 | 1 | 32 -> 32      | 100x100  | 0.10   |  5      | 0.5    |
| 3 | conv3 | **2** | 32 -> 64  | 50x50    | 0.20   |  7      | 0.7    |
| 4 | conv3 | 1 | 64 -> 64      | 50x50    | 0.20   | 11      | 1.1    |
| 5 | conv3 | 1 | 64 -> 64      | 50x50    | 0.20   | 15      | 1.5    |
| 6 | conv3 | **2** | 64 -> 96  | 25x25    | 0.40   | 19      | 1.9    |
| 7 | conv3 | 1 | 96 -> 96      | 25x25    | 0.40   | 27      | 2.7    |
| 8 | conv3 | 1 | 96 -> 96      | 25x25    | 0.40   | **35**  | **3.5**|

The final **RF = 35 px = 3.5 m diameter = 1.75 m radius** — the 1.5 m swept disc plus 0.25 m of
margin. That number is the whole point of the table: *the block count is chosen to hit it*.
Adding a 9th 3x3 block at stride 4 would push RF to 43 px (4.3 m), which starts letting the net
see a second, causally-unrelated obstacle.

Resolution bookkeeping: 100 -> 50 -> 25 under stride-2 convs with `padding=1`
(`floor((100 + 2 - 3)/2) + 1 = 50`). We stop at 25x25 @ 0.40 m — slightly finer than the 0.50 m
lattice pitch, so `grid_sample` interpolates rather than extrapolates at every lattice point.

> Implementation note: `nn.Conv2d(padding='same')` does **not** accept `stride > 1` in PyTorch.
> Use explicit `padding=1` everywhere — for `kernel_size=3` it is identical to `'same'` at
> stride 1, and gives the exact halving above at stride 2.

### 4d. The readout

```
   feature map [B,96,25,25]      spans world x,y in [-5.0, +5.0]
                 |
                 |    grid[b,i,j] = ( spawn_x[i,j] / 5.0 ,  spawn_y[i,j] / 5.0 )
                 |                 = normalised to [-1,1] over the map extent
                 |                   -> lattice edge cells land at +-0.7
                 |
                 |    F.grid_sample(feat, grid, mode='bilinear',
                 |                  padding_mode='border', align_corners=False)
                 v
   [B,96,15,15]
```

`spawn_xy` comes straight out of the dataset file, so the geometry is read from the data rather
than hardcoded in the model. `grid_sample`'s last dim is `(x, y)` where `x` indexes the **width**
(our columns = world X) and `y` the **height** (our rows = world Y) — which matches the
row=+Y / col=+X convention both grids already use.

### 4e. Parameter count

| part | params |
|---|---|
| conv trunk (weights) | ~323k |
| ChannelLayerNorm | ~1.1k |
| wz embedding MLP | ~1.2k |
| FiLM projections (8 x `Linear(32 -> 2C)`) | ~35k |
| per-cell head | ~6.3k |
| **total** | **~368k** (measured: `model.py` prints 367,874) |

Against a dataset of 100 maps x 10 commands = 1000 rows x 225 cells ~= 225k labelled cells.
Comfortable. The base width (32) is the single knob if that needs to move.

---

## 5. Target space, loss, and the mask

### 5a. Predict in log1p / standardised space

Reuse `learning/model.py`'s `TargetTransform` verbatim in spirit:

```
    forward:  y (m, rad)  --log1p-->  standardise  -->  what the loss sees
    inverse:  net output  --unstandardise-->  clamp at 0  --expm1-->  y (m, rad)
```

`e_pos` and `e_rot` are non-negative and heavily right-skewed — on a single-box map ~94% of the
field is open flat ground sitting at a near-constant small background error, with a handful of
collision cells far out in the tail. A plain MSE on raw metres lets those few cells dominate
every gradient, and lets the net predict physically impossible negative errors.

**Fit the transform on TRAIN rows only, and only on cells where `mask == True`.** Masked cells
store exact zeros, and folding thousands of spurious zeros into the mean/std would wreck the
standardisation.

### 5b. The masked loss

```
             sum_over_cells[ mask * (y_hat - y_target)^2 ]
    L  =    -----------------------------------------------      summed over both heads
                       sum_over_cells[ mask ]
```

Both `y_hat` and `y_target` are in transform space. Normalising by `mask.sum()` (not by the
cell count) keeps the loss scale independent of how many cells a given map happens to block.

Masked cells are already **exact zeros, never NaN**, in the file — `generate_dataset.py` is
explicit about why (`0 * NaN = NaN` would poison the backward pass even through a correct
mask). Nothing in the model may reintroduce a NaN: in particular do **not** compute the loss as
`torch.where(mask, se, 0)` on a tensor that had NaNs in it.

### 5c. What the mask actually means

Two different things, both excluded from the loss:

* **blocked** — the spawn footprint sits on an obstacle, so the trial was never simulated.
* **diverged** — simulated, but ostrich's solve failed the finite / plausible-bounds check.

Worth keeping in mind when reading results: blocked cells are a *deterministic function of the
map*, so a model could in principle learn to predict the mask itself. We are not asking it to.

---

## 6. Two things that matter as much as the architecture

### 6a. The exact mirror symmetry — a free 2x on data

The robot is precisely symmetric about its own x-axis (`LEFT_WHEEL_POS = (0, +0.365, 0)`,
`RIGHT_WHEEL_POS = (0, -0.365, 0)`, `REAR_WHEEL_POS` on the axis), and yaw is locked to 0 for
every spawn. Both grids are symmetric about `y = 0`. Therefore:

```
   reflect the terrain in y   +   negate wz   ==   reflect the label field in y
```

`e_pos` and `e_rot` are magnitudes, so their *values* are unchanged — only their positions move.
Concretely, as a training augmentation applied with probability 0.5:

```
    heightmap  ->  torch.flip(heightmap, dims=[-2])      # rows = world Y
    wz         ->  -wz
    y          ->  torch.flip(y,    dims=[-3])           # [B,15,15,2], rows = world Y
    mask       ->  torch.flip(mask, dims=[-2])
```

This is an **exact** symmetry of the data-generating process, not an approximation. It also
doubles as a test: at eval time, `f(flip(h), -wz)` should equal `flip(f(h, wz))` to within
numerical noise once trained.

### 6b. Split by MAP, not by row — `split_dataset` currently leaks

`custom_dataset.split_dataset` splits by **row**, and its docstring says so plainly: "a val
row's map may also appear in train under a different commanded wz."

With 100 maps x 10 commands that means a validation row will almost always share its terrain
with 9 training rows. The val score then measures *interpolation in wz on memorised terrain*,
which is not the question. The claim we care about is **"does this transfer to a map it has
never seen"**, which needs a map-level split:

```
    group rows by `map_index`  ->  split the MAPS 80/20  ->  all rows of a map go
    to the same side of the split
```

This needs a small addition alongside `split_dataset`; it is a dataset-side change, not a model
change, but the architecture is unfalsifiable without it.

---

## 7. Baselines it has to beat, and how we will know

Because the label field is dominated by a near-constant background, masked MSE is easy to
score well on for the wrong reason. Fix the comparisons **before** training:

| baseline | what it is | what beating it proves |
|---|---|---|
| global mean | one constant for the whole dataset | almost nothing; a sanity floor |
| per-`wz` mean field | mean field over train rows, looked up by `wz` | the model uses the *terrain* at all |
| blur-the-terrain | same net, heightmap replaced by its per-map mean | the model uses terrain *structure*, not just "is there a box somewhere" |

And report metrics in physical units, split by regime:

* masked RMSE over **all** valid cells, in m and rad,
* masked RMSE over the **top-decile** cells by true `e_pos` — the collision cells, which are
  the entire point,
* R^2 per head,
* on held-out **maps**, per 6b.

Two free sanity checks that need no baseline at all:

* **`wz = 0` rows** (present whenever `n_commands` is odd) command no motion, so the true field
  is ~0 everywhere. The prediction must be ~0 everywhere too.
* **mirror equivariance**, per 6a.

---

## 8. Alternatives considered and rejected

**Per-cell patch CNN** (crop a 33x33 window at each lattice cell, run a shared small CNN).
Mathematically almost the same hypothesis class as the trunk above — a capped-RF fully
convolutional net *is* a shared patch model — but computes each window separately, 225x
redundantly, and needs its own crop-indexing bookkeeping. Its one real advantage, per-patch
relative-height normalisation, has been folded into section 3a instead. It is also close to a
re-run of `learning/terrain_patch.py`, which would waste the reframing this dataset exists for.

**U-Net / encoder-decoder with a global bottleneck.** The standard dense-prediction workhorse,
and the wrong tool here: its selling point is global context, which section 1 argues the physics
does not have. ~1-2M params, a bottleneck that sees the whole map (i.e. can identify *which*
map it is looking at), and an awkward output shape (15 is not a clean fraction of 100). Revisit
only if evidence turns up that divergence has genuinely non-local structure.

**Flatten-the-map MLP / vision transformer.** No translation equivariance at all, so every
box position has to be learned independently. At 100 maps this cannot work, and it throws away
the strongest inductive bias available.

---

## 9. Training recipe (starting point, not gospel)

```
    optimiser     AdamW, lr 3e-4, weight_decay 1e-4
    schedule      cosine to 0 over the run, ~5 epochs linear warmup
    batch size    16 rows  (= 16 x 225 = 3600 labelled cells per step)
    epochs        ~200 on 1000 rows, early stopping on held-out-map val loss
    augmentation  y-mirror + wz negation, p = 0.5
    logging       Weights & Biases, same as learning/train.py
    checkpoint    outputs/checkpoints/, self-describing, WITH the fitted TargetTransform
                  (it is fitted data, not a learned parameter — it does not live in
                   state_dict() and must be saved and re-attached explicitly)
```

## 10. Files this implies

| file | role |
|---|---|
| `model.py` | `GridPoseErrorNet` (sections 3, 4) + the reused `TargetTransform` |
| `train.py` | masked loss (5b), map-level split (6b), augmentation (6a), baselines (7) |
| `custom_dataset.py` | **+ a map-level split** alongside the existing row-level one |

Nothing above requires regenerating the dataset. Everything the model needs — `heightmap`,
`wz`, `spawn_xy`, `y`, `mask`, `grid/resolution`, `grid/extent` — is already in the file.
