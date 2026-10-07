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

## TensorRT vs `WarpTrunk` (2026-10-07)

Same trunk through TensorRT 10.16 engines (`export_torch_model_to_onnx.py` -> `build_tensorRt_engine.py`), timed by `bench_trunk_tensorrt.py`. Different machine from the rest of this file: **GTX 1650 Max-Q (sm_75, 4 GB)**, since TensorRT 10 cannot run on the 1050. TensorRT installed into the root `.venv` with `uv pip install "tensorrt-cu12<11"` (not in `pyproject.toml`; ~6 GB, and a `uv sync` removes it). Checkpoint `dataset_mppi_default_maps0_centered_M200_R10_seed0`.

Workload: 512 rollouts × 3 windows = `trunk_rows(512, 3)` = **1025 patches** per refine (windows 1–2 plus the level patch). Fastest of 7 interleaved rounds, ms per call; three separate runs:

| form | run 1 | run 2 | run 3 | vs `WarpTrunk` graph | max rel diff vs torch |
|---|---|---|---|---|---|
| torch eager | 124 | 163 | 125 | 0.4× | – |
| torch + NCHW norm | 80 | 88 | 79 | 0.7× | – |
| `HybridTrunk` | 55 | 63 | 56 | ~1.0× | – |
| `WarpTrunk` (eager = graph) | 54.5 | 66.6 | 54.4 | 1.00× | 2.9e-6 |
| TensorRT fp32 | 69 | 77 | 69 | 0.8–0.87× | 2.7e-6 |
| TensorRT fp16 | **46.5** | **61.8** | **46.0** | 1.08–1.18× | 1.0e-2 |

Run 2 was a warm GPU; the ranking held in all three. Runs 1–2 launched TensorRT eagerly on the default stream; run 3 used a side stream and also replayed each engine from a captured torch CUDA graph. Neither changed TensorRT's time (eager and graph agree to 0.3 ms), as with `WarpTrunk`: at this batch the kernels dominate, not launches.

- **fp32 TensorRT is slower than `WarpTrunk`.** Its autotuned kernels do not beat the hand-written conv + fused norm at these shapes.
- **fp16 TensorRT is only ~15% faster**, at ~1% error in the code (~0.5% in the physical errors after the head). The 1650 Max-Q's fp16 rate is not much above its fp32 one. The Orin has fast fp16 tensor cores, so the result there may differ and should be measured there.
- **Integration cost.** `WarpTrunk` already runs inside `MppiGpu`'s captured refine graph. A TensorRT engine runs on a torch/CUDA stream; it can be captured (as above), but splicing that into the Warp graph is extra work for a 15% saving on this GPU.
- **Memory.** An execution context reserves activations for the profile's max batch: ~1.4 GB at 4097, ~2.7 GB at 8193. The 8193 profile (4096 rollouts) does not fit next to a torch reference on 4 GB, so the laptop engines were built with `--max-batch 4097`. One context per engine is all the card holds alongside anything else.
- **Build time.** fp32 engines build in 25–40 s, fp16 ones in 5–9 min (tactic search). The timings `build_tensorRt_engine.py` itself printed (324 / 79 ms) were medians through clock bursts and are not comparable with this table.

## Full TensorRT net vs `WarpTrunk` + Warp head, in the replan (2026-10-07)

The table above times the trunk alone. `bench_full_tensorrt.py` times the whole nn cost the way MPPI runs it: a real replan (`closed_loop.node_planner`, 512 rollouts, windows 0–2, 1 refine, baseline "none"). Every form is `MppiGpu`'s cost hook, captured in the refine graph, with its TensorRT engine enqueued on Warp's stream. Each engine is built with a static profile at exactly the batch its form runs. Same GTX 1650 Max-Q and checkpoint.

- **`WarpTrunk` + Warp head**: the deployed `WindowCost`. One torch trunk pass for window 0 in `update`, then 1025 trunk rows in the graph.
- **TRT trunk + Warp head**: the same `WindowCost` with a TensorRT trunk engine in `WarpTrunk`'s place.
- **full TRT**: patches and commands written in the graph, then one engine (patch + command → errors), then the charge to `J`. It cannot share a terrain code, so window 0's single patch goes through the trunk once per candidate: **1536 trunk rows** instead of 1025.

All forms roll out identical trajectories (asserted). Against `WarpTrunk` + Warp head, the max relative diff is 1.3e-6 in errors at fp32, and 6e-3 in errors and 4e-3 in the charge to `J` at fp16. Wall ms per replan (median of 20, fastest of 7 interleaved rounds), two runs:

| form | run 1 | run 2 | vs `WarpTrunk` + head |
|---|---|---|---|
| vanilla (no hook) | 2.2 | 2.8 | – |
| `WarpTrunk` + Warp head | 72.1 | 73.4 | 1.00× |
| TRT trunk fp32 + Warp head | 81.8 | 81.7 | 0.88–0.90× |
| TRT trunk fp16 + Warp head | **57.2** | **57.1** | **1.26–1.29×** |
| full TRT fp32 | 115.3 | 130.7 | 0.56–0.63× |
| full TRT fp16 | 78.8 | 89.5 | 0.82–0.92× |

- **The full net loses at both precisions.** Its 50% more trunk rows cost more than fusing the head saves, and the head is only a few ms. Even fp16 is slower than all-fp32 `WarpTrunk`.
- **The fastest form is a TensorRT fp16 trunk with the Warp head**, saving ~15 ms per refine (~26%). It runs inside the refine graph with no Warp changes: an engine enqueued on Warp's stream during `ScopedCapture` is captured like any kernel, after one uncaptured warm-up enqueue at the shape. Its cost is ~0.6% error in the errors and a GPU-specific engine.
- In this test, ~18 ms of the `WarpTrunk` arm's ~70 ms is outside the trunk's ~54 ms: the head, the patch sampling and the torch window-0 trunk. That leaves the head (`_dense_kernel`, below) as the next target.

## Potential speed-ups (2026-10-07)

Where one refine goes with the TensorRT fp16 trunk + Warp head at 512 × 3 on the GTX 1650 Max-Q. Each stage was timed alone, on a cooler GPU than the table above, so the totals are lower:

| stage | ms |
|---|---|
| **TensorRT fp16 trunk, 1025 patches** | **42–44** |
| MPPI itself (vanilla replan) | ~2.5 |
| Warp head, windows 1–2 (1024 rows) | 0.8 |
| Warp head, window 0 (512 rows) | 0.5 |
| `update()`: window-0 patch + torch trunk | 0.8 |
| patch sampling, 1024 patches | 0.3 |

The head, `update()` and sampling together are ~2.5 ms, so speeding them up (the `_dense_kernel` fix below) is worth ~1 ms. Halving the replan has to come from the trunk. TensorRT's per-layer profile (`IProfiler`, eager) shows only ~40% of the trunk is convolution arithmetic:

| share | what |
|---|---|
| ~40% | the 7 convolutions |
| ~20% | layout-conversion copies between the convolutions and the LayerNorms |
| ~17% | separate replicate-pad ops, one before every conv |
| ~12% | the channel LayerNorm, as transpose → mean → … → transpose → SiLU |

TensorRT cannot fuse replicate padding or a per-pixel channel LayerNorm into its convolutions, so every block makes several extra trips through memory. Ideas, cheapest first (estimates, not measurements):

1. ~~**Export with `--nchw-norm`**~~ **Tried, slower.** The fp16 static engine at 1025 patches took 71.1 ms against 50.8 ms for the current export (fastest of 9 interleaved rounds on a warm GPU; `WarpTrunk` 67.7 ms in the same run). Both match torch to 1.0e-2 on the code. The profile shows why. The NCHW form removed ~6 ms of layout copies as intended (14.7 → 8.6 ms), but TensorRT splits a LayerNorm over dim 1 into many separate layers (46 → 85 in total; norm and other 9.8 → 26.0 ms), and the convs got slower (24.6 → 31.0 ms). The permute-based form is the one TensorRT fuses into its own norm kernels, the opposite of torch, where the NCHW form is 1.6× faster. Reshaping the graph does not remove the glue: the replicate pads and the channel LayerNorm themselves are the problem, which leaves 2 and 3.
2. **Evaluate the nn cost only on the top-K candidates by vanilla cost** (no retraining). MPPI keeps ~5 elites of 512 (`elite_frac` 0.01). The nn cost is ≥ 0, so a rollout outside the top K with a vanilla cost above the 5th-best total inside it cannot become an elite. That gives an exactness check each refine can run on the GPU and count failures of. K = 256 halves the trunk rows, K = 128 quarters them. It needs a top-K selection inside the graph, which can bisect for its cutoff as `_cem_reweight` already does. This is the one idea that halves the time on its own.
3. **Retrain with TensorRT-friendly blocks: zero padding and BatchNorm** instead of replicate padding and channel LayerNorm. BatchNorm folds into the conv at inference, so each block becomes one fused conv + bias + SiLU kernel. That removes the ~50% glue, roughly 20–25 ms. Slimming the two heavy blocks (1: 32→32 at full resolution, 3: 64→64 at half) would cut it further. The cost is a retrain, re-validation, recalibrated cost weights and matching `WarpTrunk` changes.
   **Speed measured with random weights; not retrained yet.** The trunk is `mppi_learning/model_tensorRT_friendly.py`. Its static fp16 engine at 1025 patches takes **23.2 ms against 50.3 ms** for the current one (2.16×; `WarpTrunk` 69.6 ms in the same run, fastest of 9 interleaved rounds). TensorRT builds it as 13 layers instead of 46: one fused conv + SiLU per block, with 96% of the time in convolution and no pads or norms left. The two heaviest blocks are still 1 (32→32 at full resolution, 6.8 ms) and 3 (64→64 at half resolution, 5.8 ms), so slimming them is the next lever. Whether the retrained net is as accurate as the original is still open: `train.py --trunk trt_friendly`, then `mppi_learning/compare_checkpoints.py <original.pt> <trt.pt>`.
4. **INT8 TensorRT.** The 1650 has no tensor cores, but its int8 instructions run up to 4× the fp32 rate. It needs calibration data and an accuracy check. It matters more on the Orin, which has int8 tensor cores.
5. **Cache terrain codes on a pose lattice** (x, y, yaw bins), filled lazily and kept across replans. The map is static within a replan and the robot moves ~0.15 m per replan, so most codes would be reused. The cost is pose-quantization error, which has to be measured against the net's own error.

With 1 ruled out, 2 is the remaining lever without retraining: K = 256 would bring the trunk from ~43 to ~22 ms. 3 is the structural fix if the net is retrained anyway.

## Open
- `mppi_cost._dense_kernel`, which runs the head, has the same scalar-`vec4`-load problem, so the same fix should speed it up.
