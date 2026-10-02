# Terrain trunk speed (2026-09-30)

**Why this matters.** `WindowCost` charges only MPPI window 0. All rollouts start at one pose, so window 0 needs one patch and one trunk pass. Charging the later windows needs a terrain code for every rollout's pose at steps 10 and 20, recomputed in every refine and inside the captured refine graph. The trunk (`ArcDivergenceNet.terrain_code`: 6 conv blocks, then squeeze, then geometry) is therefore the cost to beat.

- At H 31 / 4 knots there are 3 windows, so 2 extra patches per rollout.
- The existing window-0 hook (4096 rollouts, 3 refines) costs 30 ms per replan without it, 44 ms with it, and 54 ms with `baseline="flat"`.

All timings: GTX 1050 (sm_61) laptop, trained mppi_learning checkpoint. The GPU throttles as it heats, so absolute numbers vary by about ±15% between runs.

## What was tried

| step | 512 patches |
|---|---|
| torch `terrain_code` | 119 ms |
| + fp16 (no fast fp16 on sm_61) | 140 ms |
| + `NCHWChannelLayerNorm` (`channel_layer_norm.py`) | 73 ms |
| `HybridTrunk` | 48 ms |
| `WarpTrunk` | 47 ms |

Where torch's time went at 512 patches (profiler):

| component | share |
|---|---|
| `ChannelLayerNorm` (permute + LayerNorm over tiny rows) | 50% |
| replicate padding (a separate copy before every conv) | 10% |
| permutes and SiLU | ~15% |
| the convolutions themselves | ~25% |

`NCHWChannelLayerNorm` computes the same maths over dim 1 without the permutes. It reuses the same `nn.LayerNorm`, so the state_dict is unchanged.

## The two rewrites (`warp_trunk.py`)

| | `HybridTrunk` | `WarpTrunk` |
|---|---|---|
| convs | cuDNN (`F.conv2d`, padding 0) | Warp, direct, register-blocked (9 pixels × 4 or 8 channels per thread) |
| norm + SiLU | one Warp kernel per block, which also writes the next conv's replicate-padded input | Warp, in place, one thread per pixel |
| squeeze + geometry | torch conv + `F.linear` | `mppi_cost._dense_kernel` |
| layout | NCHW, torch buffers | NHWC vec4, preallocated for `max_batch` |
| graph-capturable | **no** (cuDNN on torch's stream) | **yes** (`WarpTrunk.captured`) |
| matches `terrain_code` | ~2e-6 relative | ~2e-6 relative |

Whole trunk:

| patches | torch | torch + NCHW norm | hybrid | Warp | Warp graph |
|---|---|---|---|---|---|
| 512 | 118.7 | 74.1 | 48.1 | 47.1 | 48.8 |
| 3584 (back to back) | – | – | **298** | 338 | – |

- **Graph capture** adds no speed. The trunk makes about 15 launches, which cost nothing next to the kernels. Its value is that the trunk can run inside the refine graph.
- **Large batches:** cuDNN gets relatively faster, while the Warp conv scales linearly.

Per block at 512 patches, in ms:

| block | pad + cuDNN | cuDNN alone | Warp conv | torch norm | Warp norm |
|---|---|---|---|---|---|
| 0: 1→32 @ 24×36 | 2.5 | 2.4 | 1.8 | 7.2 | 1.6 |
| 1: 32→32 @ 24×36 | 9.5 | 6.7 | 11.6 | 7.1 | 1.6 |
| 2: 32→64 s2 | 6.2 | 3.5 | 6.9 | 3.6 | 1.0 |
| 3: 64→64 @ 12×18 | 13.9 | 12.1 | 10.5 | 3.6 | 0.9 |
| 4: 64→96 s2 | 4.4 | 3.4 | 4.4 | 1.3 | 1.1 |
| 5: 96→96 @ 6×9 | 5.2 | 4.3 | 6.0 | 1.3 | 1.0 |

## Lessons for Warp kernels here
- **`vec4` loads:** Warp's `vec4` is 4-byte aligned, so `x[i]` compiles to 4 scalar loads. A 128-bit `__ldg` through `wp.func_native` took the Warp trunk from 122 to 61 ms.
- **Indexing:** multi-dimensional indices cost integer multiplies per load, which Pascal emulates on the FMA cores. Flat indices hoisted out of the inner loop took it from 61 to 50 ms.
- **Column reuse** across the 3 kx taps (3× fewer input loads) was no faster, so the conv is no longer load-bound. It sits at about 40% of FMA peak.

## Where this leaves the goal
Windows 1 and 2 at 512 rollouts mean 1024 patches per refine, which is about 94 ms per refine, or about 280 ms with 3 refines. Faster kernels won't close that gap on this GPU. It needs fewer or cheaper patches: a smaller trunk, fewer refines, pose-sharing, or the dense code map (`encode_map`).

## Wired in
`WindowCost(..., n_windows=3)` runs `WarpTrunk` inside the refine graph. It samples each rollout's patch at steps 10 and 20. At 512 rollouts × 1 refine, a replan takes 3.0 ms without the hook, 4.9 ms with window 0 only (baseline "flat"), and 99.7 ms with windows 0–2 (`mppi_cost.py --bench`). `closed_loop.py` measures the same on a short run: 99 ms per replan, 96 ms of it in the cost stage. Over 25 s runs with 14 worlds the GPU heats up: median 101–126 ms, p90 109–139 ms. The closed-loop results are in `mppi_learning/design.md` §9e.

## Open
- `mppi_cost._dense_kernel`, which runs the head, has the same scalar-`vec4`-load problem, so the same fix should speed it up.
