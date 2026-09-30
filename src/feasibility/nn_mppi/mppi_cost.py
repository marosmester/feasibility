"""`WindowCost`: the `mppi_learning` net's predicted twin-vs-ostrich error of MPPI's windows, charged
to every rollout inside helhest_stack's captured refine (mppi_learning/design.md section 9). Window
0 by default; `n_windows` 3 also charges windows 1 and 2 (below).

`MppiGpu.replan` starts every rollout at the same pose, so window 0 (steps [0, 10), knot 0 to
knot 1) has ONE terrain patch, shared by all rollouts; only the command differs. The work is
therefore split at the terrain code:

  * `update(state)`, called before each `planner.replan`, OUTSIDE the graph: a Warp kernel samples
    the body-frame patch at `state` from the planner's own elevation buffer (`sample_patches`'s
    cell centres, bilinear, edge clamping, wheel-contact reference, / wheel radius -- helhest's
    `sample_field` has the same cell-centre convention as `HeightMapReader.sample`), torch runs
    the trunk on it zero-copy, and the 256-number code is copied into a fixed Warp buffer. All on
    Warp's stream, so the graph launched next reads the finished code.
  * the hook (`MppiGpu.set_cost_hook(self)`), INSIDE the graph, pure Warp: window 0's command from
    `target_wheel_omega[0:10]` with `command.encode`'s least-squares mean + slope, the 7 features,
    the FiLM head, `TargetTransform`'s inverse, and `sum_k weight_k * e_k` added to `planner.J`.
    The mu replicas of a candidate share its command, so the head runs over `n_cand` rows and
    the cost is added to every replica.

`baseline="flat"` charges only the error the TERRAIN adds: the head also runs on a LEVEL patch (all
zeros, whose code is computed once at construction), and each candidate pays
`sum_k weight_k * max(e_k(patch) - e_k(level), 0)`. The raw prediction includes the twin's kinematic
error on flat ground, which grows with speed, so without the baseline the cost is also a speed brake
(design.md section 9d.6). The trunk layers then run over twice the rows.

The weights are a device array (`set_weights`), so tuning them needs no recapture. The planner
must be the one the net was trained for (`check_planner`): H 31 / 4 knots (window 0 = one knot
interval), `wmin` 0, and the twin's `k_turn` and step from the checkpoint's `label_attrs`.

Windows 1 and 2 (`n_windows` > 1, design.md section 9e) start where each rollout is at steps 10 and
20, so they need a patch and a terrain code PER ROLLOUT, and those poses exist only after the
rollout, inside the graph. The hook therefore samples every rollout's patch there with the same
sampler, runs `warp_trunk.WarpTrunk` (the graph-capturable trunk) over all of them plus one level
patch, and runs the head per (window, rollout) with that window's command from
`target_wheel_omega[10 k : 10 k + 10]`. Each rollout pays for its own later windows, with the same
weights and baseline as window 0; the robust cost then averages that term over the mu replicas.
The trunk is the cost: ~47 ms per 512 patches on the GTX 1050 (`trunk_speed.md`).

`switchable=True` wraps the hook in a conditional graph node (`wp.capture_if`) on a device flag, so
`replan_split(state, goal, n_refine, k)` can charge the cost in the first k refines only and let the
rest rank the rollouts by the planner's own cost. Both replay the same captured graph, with no
recapture, and the skipped refines cost what a vanilla refine does.

CLI parameters:
    --checkpoint PATH   a mppi_learning train.py checkpoint (default: a random-weight net, which
                        checks the plumbing but not a trained net's numbers)
    --batch INT         MPPI rollouts for the checks (default 1024)
    --bench             also time a replan at 4096 rollouts x 3 refines without the hook, with it,
                        and with it at baseline "flat"; at 512 rollouts x 1 refine without the hook
                        and at baseline "flat" charging 1 and 3 windows; and at 512 x 3 refines
                        without the hook and charging 3 windows in every refine or in the first only

Usage:
    python src/feasibility/nn_mppi/mppi_cost.py
    python src/feasibility/nn_mppi/mppi_cost.py --checkpoint outputs/checkpoints/<ckpt>.pt --bench
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import math
import pathlib
import time
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp
from helhest.control.mppi import MppiGpu
from helhest.engine.terrain import Grid
from helhest.engine.terrain import sample_field

from feasibility.lattice_learning.patch import HEIGHT_SCALE
from feasibility.lattice_learning.patch import WHEEL_CONTACTS_LOCAL
from feasibility.lattice_learning.train import prepare_patch
from feasibility.mppi_learning.command import HALF_TRACK
from feasibility.mppi_learning.command import WHEEL_RADIUS
from feasibility.mppi_learning.command import WINDOW_STEPS
from feasibility.mppi_learning.model import V_SCALE
from feasibility.mppi_learning.model import WindowDivergenceNet
from feasibility.mppi_learning.model import WZ_SCALE
from feasibility.mppi_learning.train import load_checkpoint

if TYPE_CHECKING:
    from feasibility.nn_mppi.warp_trunk import WarpTrunk

HORIZON = 31  # design.md section 3: H = WINDOW_STEPS * (N_KNOTS - 1) + 1
N_KNOTS = 4
ROWS_PER_THREAD = wp.constant(8)  # batch rows per `_dense_kernel` thread; the head's rows are padded to it


@wp.func
def _relief(
    elevation: wp.array2d(dtype=wp.float32),
    grid: Grid,
    p: wp.vec3,  # (x, y, yaw) of the patch's body frame
    i: int,
    j: int,
    x_min: float,
    y_min: float,
    cell: float,
    contacts: wp.array(dtype=wp.vec2),  # [3] body-frame wheel contacts, the height reference
    height_scale: float,
):
    c = wp.cos(p[2])
    s = wp.sin(p[2])
    reference = float(0.0)
    for k in range(3):
        q = contacts[k]
        reference += sample_field(elevation, grid, p[0] + c * q[0] - s * q[1], p[1] + s * q[0] + c * q[1])
    reference = reference / 3.0
    bx = x_min + (float(j) + 0.5) * cell
    by = y_min + (float(i) + 0.5) * cell
    height = sample_field(elevation, grid, p[0] + c * bx - s * by, p[1] + s * bx + c * by)
    return (height - reference) / height_scale


@wp.kernel
def _patch_kernel(
    elevation: wp.array2d(dtype=wp.float32),
    grid: Grid,
    pose: wp.array(dtype=wp.vec3),  # [1] (x, y, yaw) the rollouts start from
    x_min: float,
    y_min: float,
    cell: float,
    contacts: wp.array(dtype=wp.vec2),
    height_scale: float,
    patch: wp.array2d(dtype=wp.float32),  # [ny, nx] rows along body +y, columns along body +x
):
    i, j = wp.tid()
    patch[i, j] = _relief(elevation, grid, pose[0], i, j, x_min, y_min, cell, contacts, height_scale)


@wp.kernel
def _rollout_patch_kernel(
    elevation: wp.array2d(dtype=wp.float32),
    grid: Grid,
    controlled: wp.array2d(dtype=wp.vec3),  # [T + 1, B] every rollout's (x, y, yaw) per step
    window_steps: int,
    rows: int,  # patch rows per window (B padded to whole ROWS_PER_THREAD groups)
    x_min: float,
    y_min: float,
    cell: float,
    contacts: wp.array(dtype=wp.vec2),
    height_scale: float,
    patches: wp.array3d(dtype=wp.float32),  # [n_later * rows + 1, ny, nx]: window w + 1, rollout r at w * rows + r
):
    w, r, cell_index = wp.tid()
    nx = patches.shape[2]
    i = cell_index // nx
    j = cell_index - i * nx
    p = controlled[(w + 1) * window_steps, r]  # the rollout's pose where window w + 1 starts
    patches[w * rows + r, i, j] = _relief(elevation, grid, p, i, j, x_min, y_min, cell, contacts, height_scale)


@wp.func
def _sign(x: float):
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    return 0.0  # torch.sign(0) = 0: an exactly straight window has no turn side


@wp.func
def _silu(x: float):
    return x / (1.0 + wp.exp(-x))


@wp.func
def _silu4(v: wp.vec4):
    return wp.vec4(_silu(v[0]), _silu(v[1]), _silu(v[2]), _silu(v[3]))


@wp.func
def _command_features(
    target_wheel_omega: wp.array2d(dtype=wp.vec3),  # [T, B] MPPI's commanded wheel speeds
    first_step: int,
    column: int,
    window_steps: int,
    wheel_radius: float,
    half_track: float,
    v_scale: float,
    wz_scale: float,
):
    """window_command_features of steps [first_step, first_step + window_steps) of one rollout, as
    two vec4 (the 7 features + a 0 pad)."""
    # command.encode: least-squares mean + slope over the window, the slope times window_steps
    centre = 0.5 * float(window_steps - 1)
    v_sum = float(0.0)
    wz_sum = float(0.0)
    v_moment = float(0.0)
    wz_moment = float(0.0)
    ic_sq = float(0.0)
    for t in range(window_steps):
        w = target_wheel_omega[first_step + t, column]
        v = wheel_radius * (w[0] + w[1]) / 2.0
        wz = wheel_radius * (w[1] - w[0]) / (2.0 * half_track)
        ic = float(t) - centre
        v_sum += v
        wz_sum += wz
        v_moment += ic * v
        wz_moment += ic * wz
        ic_sq += ic * ic
    n = float(window_steps)
    wz_mean = wz_sum / n / wz_scale
    wz_slope = wz_moment / ic_sq * n / wz_scale
    sign = _sign(wz_mean)
    return (wp.vec4(v_sum / n / v_scale, v_moment / ic_sq * n / v_scale, wz_mean, wp.abs(wz_mean)),
            wp.vec4(sign, wz_slope, wz_slope * sign, 0.0))


@wp.kernel
def _features_kernel(
    target_wheel_omega: wp.array2d(dtype=wp.vec3),
    window_steps: int,
    wheel_radius: float,
    half_track: float,
    v_scale: float,
    wz_scale: float,
    features: wp.array2d(dtype=wp.vec4),  # [rows, 2]: window 0 of candidate c at row c
):
    c = wp.tid()
    f0, f1 = _command_features(target_wheel_omega, 0, c, window_steps, wheel_radius, half_track, v_scale, wz_scale)
    features[c, 0] = f0
    features[c, 1] = f1


@wp.kernel
def _rollout_features_kernel(
    target_wheel_omega: wp.array2d(dtype=wp.vec3),
    window_steps: int,
    rows: int,
    wheel_radius: float,
    half_track: float,
    v_scale: float,
    wz_scale: float,
    features: wp.array2d(dtype=wp.vec4),  # [n_later * rows, 2]: window w + 1 of rollout r at row w * rows + r
):
    w, r = wp.tid()
    f0, f1 = _command_features(target_wheel_omega, (w + 1) * window_steps, r, window_steps, wheel_radius,
                               half_track, v_scale, wz_scale)
    features[w * rows + r, 0] = f0
    features[w * rows + r, 1] = f1


@wp.kernel
def _dense_kernel(
    x: wp.array2d(dtype=wp.vec4),  # [rows, n_in / 4]
    weight: wp.array2d(dtype=wp.vec4),  # [n_in, n_out / 4]: nn.Linear's weight TRANSPOSED
    bias: wp.array(dtype=wp.vec4),  # [n_out / 4]
    silu: int,
    y: wp.array2d(dtype=wp.vec4),  # [rows, n_out / 4]
):
    # One thread per (ROWS_PER_THREAD rows, 4 outputs), so a weight load serves every row and
    # consecutive threads read consecutive weights. A thread per output with nn.Linear's layout ran
    # a 4096 x 256 x 256 layer in 40 ms on the GTX 1050; this runs it in ~2, about cuBLAS's speed.
    g, q = wp.tid()
    r0 = g * ROWS_PER_THREAD
    acc = wp.matrix(shape=(ROWS_PER_THREAD, 4), dtype=float)
    for kq in range(x.shape[1]):
        xs = wp.matrix(shape=(ROWS_PER_THREAD, 4), dtype=float)
        for i in range(ROWS_PER_THREAD):
            xs[i] = x[r0 + i, kq]
        for s in range(4):
            w = weight[kq * 4 + s, q]
            for i in range(ROWS_PER_THREAD):
                xv = xs[i, s]
                acc[i, 0] += xv * w[0]
                acc[i, 1] += xv * w[1]
                acc[i, 2] += xv * w[2]
                acc[i, 3] += xv * w[3]
    b = bias[q]
    for i in range(ROWS_PER_THREAD):
        out = wp.vec4(acc[i, 0], acc[i, 1], acc[i, 2], acc[i, 3]) + b
        if silu != 0:
            out = _silu4(out)
        y[r0 + i, q] = out


_FILM_LAYER_NORM_KERNELS: dict[int, wp.Kernel] = {}


def _film_layer_norm_silu_kernel(width: int) -> wp.Kernel:
    """FiLM (`code * (1 + gamma) + beta`, as ArcDivergenceNet._head), LayerNorm and SiLU, one tile
    block per row: 0.33 ms at 4096 x 256 on the GTX 1050, against 2.3 ms for a thread per row
    (whose reads of a row are strided across the warp). A tile's shape is a compile-time constant,
    so the kernel is built per width."""
    if width in _FILM_LAYER_NORM_KERNELS:
        return _FILM_LAYER_NORM_KERNELS[width]
    n = wp.constant(width)

    def film_layer_norm_silu(
        film: wp.array2d(dtype=float),  # [rows, 2 * width]: gamma, then beta
        code: wp.array2d(dtype=float),  # the terrain code(s), [n_codes, width] or, per_row, [rows + 1, width]
        rows: int,
        per_row: int,
        weight: wp.array(dtype=float),  # LayerNorm affine
        bias: wp.array(dtype=float),
        eps: float,
        y: wp.array2d(dtype=float),  # [n_codes * rows, width]: row r is film row r % rows under the code below
    ):
        r = wp.tid()
        # shared codes (window 0): code r // rows. per_row (the later windows, a patch per rollout):
        # film row r's own code r, then, for a level half r >= rows, the level code at row `rows`
        c = r // rows
        if per_row != 0:
            c = wp.min(r, rows)
        gamma = wp.tile_load(film[r % rows], shape=n, offset=0)
        beta = wp.tile_load(film[r % rows], shape=n, offset=n)
        one_plus_gamma = wp.tile_map(wp.add, gamma, wp.tile_ones(shape=n, dtype=float))
        h = wp.tile_map(wp.add, wp.tile_map(wp.mul, wp.tile_load(code[c], shape=n), one_plus_gamma), beta)
        mean = wp.tile_sum(h)[0] / float(n)
        centred = wp.tile_map(wp.sub, h, wp.tile_full(shape=n, value=mean, dtype=float))
        var = wp.tile_sum(wp.tile_map(wp.mul, centred, centred))[0] / float(n)  # biased, as nn.LayerNorm
        normed = wp.tile_map(wp.mul, centred * (1.0 / wp.sqrt(var + eps)), wp.tile_load(weight, shape=n))
        wp.tile_store(y[r], wp.tile_map(_silu, wp.tile_map(wp.add, normed, wp.tile_load(bias, shape=n))))

    kernel = wp.Kernel(film_layer_norm_silu, key=f"film_layer_norm_silu_{width}")
    _FILM_LAYER_NORM_KERNELS[width] = kernel
    return kernel


def _as_floats(a: wp.array) -> wp.array:
    """A vec4 [..., m] array seen as float [..., 4 m], sharing its memory (`a` must outlive it)."""
    return wp.array(ptr=a.ptr, dtype=float, shape=(*a.shape[:-1], 4 * a.shape[-1]), device=a.device)


@wp.func
def _head_error(
    h: wp.array2d(dtype=wp.vec4),
    weight: wp.array2d(dtype=wp.vec4),
    bias: wp.array(dtype=float),
    target_mean: wp.array(dtype=float),
    target_std: wp.array(dtype=float),
    k: int,
    row: int,
):
    out = bias[k]
    for q in range(weight.shape[1]):
        out += wp.dot(weight[k, q], h[row, q])
    # TargetTransform.inverse: unstandardize, clamp at 0, expm1
    return wp.exp(wp.max(out * target_std[k] + target_mean[k], 0.0)) - 1.0


@wp.kernel
def _output_kernel(
    h: wp.array2d(dtype=wp.vec4),  # [n_codes * rows, head_width / 4]
    weight: wp.array2d(dtype=wp.vec4),  # [K, head_width / 4], one row per target name
    bias: wp.array(dtype=float),
    target_mean: wp.array(dtype=float),  # [K] TargetTransform, log1p space
    target_std: wp.array(dtype=float),
    cost_weight: wp.array(dtype=float),  # [K]
    n_cand: int,
    n_mu: int,
    level_row: int,  # 0 = no baseline, else the first row of h under the level code
    prediction: wp.array2d(dtype=float),  # [n_cand, K] physical (m, rad)
    level_prediction: wp.array2d(dtype=float),  # [n_cand, K] the same command on a level patch
    J: wp.array(dtype=float),  # [n_mu * n_cand] MppiGpu.J, added to in place
):
    c = wp.tid()
    cost = float(0.0)
    for k in range(weight.shape[0]):
        error = _head_error(h, weight, bias, target_mean, target_std, k, c)
        prediction[c, k] = error
        if level_row > 0:
            level = _head_error(h, weight, bias, target_mean, target_std, k, level_row + c)
            level_prediction[c, k] = level
            error = wp.max(error - level, 0.0)
        cost += cost_weight[k] * error
    for replica in range(n_mu):
        J[replica * n_cand + c] += cost


@wp.kernel
def _rollout_output_kernel(
    h: wp.array2d(dtype=wp.vec4),  # [n_codes * n_later * rows, head_width / 4]
    weight: wp.array2d(dtype=wp.vec4),
    bias: wp.array(dtype=float),
    target_mean: wp.array(dtype=float),
    target_std: wp.array(dtype=float),
    cost_weight: wp.array(dtype=float),
    n_later: int,  # windows after window 0
    rows: int,  # h rows per window
    level_row: int,  # 0 = no baseline, else the first row of h under the level code
    prediction: wp.array3d(dtype=float),  # [n_later, B, K] physical (m, rad)
    level_prediction: wp.array3d(dtype=float),
    J: wp.array(dtype=float),  # [B] MppiGpu.J: each rollout pays for its own later windows
):
    r = wp.tid()
    cost = float(0.0)
    for w in range(n_later):
        row = w * rows + r
        for k in range(weight.shape[0]):
            error = _head_error(h, weight, bias, target_mean, target_std, k, row)
            prediction[w, r, k] = error
            if level_row > 0:
                level = _head_error(h, weight, bias, target_mean, target_std, k, level_row + row)
                level_prediction[w, r, k] = level
                error = wp.max(error - level, 0.0)
            cost += cost_weight[k] * error
    J[r] += cost


def _padded_rows(n: int) -> int:
    return -(-n // ROWS_PER_THREAD) * ROWS_PER_THREAD


def trunk_rows(n_rollouts: int, n_windows: int) -> int:
    """Patches per refine for windows 1 .. n_windows - 1: each window's rollouts padded to whole
    ROWS_PER_THREAD groups, plus the level patch. A `WarpTrunk` shared by several `WindowCost`s
    needs at least this `max_batch`."""
    return (n_windows - 1) * _padded_rows(n_rollouts) + 1


def check_planner(planner: MppiGpu, label_attrs: dict[str, object]) -> None:
    """Raises unless `planner` rolls out what the net was trained on: window 0 must be exactly one
    knot interval, the twin's k_turn and step must be the label's, and wmin must be 0 (the data
    has no reverse)."""
    problems = []
    if planner.horizon != HORIZON or planner.sampling.n_knots != N_KNOTS:
        problems.append(f"horizon {planner.horizon} / {planner.sampling.n_knots} knots, trained on {HORIZON} / {N_KNOTS}")
    if planner.sampling.wmin != 0.0:
        problems.append(f"wmin {planner.sampling.wmin}, trained on 0 (no reverse)")
    for name, value in (("k_turn", planner.sim.solver.k_turn), ("mppi_dt", planner.sim.solver.dt)):
        if not math.isclose(float(value), float(label_attrs[name]), rel_tol=1e-6):
            problems.append(f"{name} {value}, trained on {label_attrs[name]}")
    if problems:
        raise ValueError("planner does not match the net's training: " + "; ".join(problems))


class WindowCost:
    """The window cost hook for one `MppiGpu`: window 0, and windows 1 .. n_windows - 1 if
    `n_windows` > 1. Construct, `set_weights` if needed, then `planner.set_cost_hook(cost)`, and call
    `cost.update(state)` before every `planner.replan(state, ...)`. `trunk` may be a `WarpTrunk` of
    the same net shared with other `WindowCost`s whose planners replan one after another on the same
    stream (its buffers are reused per call); its `max_batch` must be at least `trunk_rows`.
    `switchable` puts the hook's work behind a conditional graph node on a device flag
    (`set_enabled`), so some refines can skip it without a recapture (`replan_split`)."""

    def __init__(
        self,
        net: WindowDivergenceNet,
        label_attrs: dict[str, object],
        planner: MppiGpu,
        weights: dict[str, float],
        blur_terrain: bool = False,
        baseline: str = "none",
        n_windows: int = 1,
        trunk: WarpTrunk | None = None,
        switchable: bool = False,
    ) -> None:
        if baseline not in ("none", "flat"):
            raise ValueError(f"baseline {baseline!r}: 'none' or 'flat'")
        if not 1 <= n_windows <= N_KNOTS - 1:
            raise ValueError(f"n_windows {n_windows}: the horizon holds 1 to {N_KNOTS - 1} windows")
        if n_windows > 1 and blur_terrain:
            raise ValueError("the blur-terrain baseline is implemented for window 0 only")
        if net.head_fusion != "film":
            raise ValueError(f"only the FiLM head is implemented in Warp, the net has {net.head_fusion!r}")
        if net.target_transform is None:
            raise ValueError("the net needs its TargetTransform to give physical errors")
        check_planner(planner, label_attrs)
        self.device = planner.device
        self.torch_device = wp.device_to_torch(self.device)
        self.net = net.to(self.torch_device).eval()
        self.blur_terrain = blur_terrain
        self.baseline = baseline
        self.switchable = switchable
        if switchable and not wp.is_conditional_graph_supported():
            raise ValueError("switchable needs conditional CUDA graph nodes, which this Warp/driver lacks")
        self.enabled = wp.ones(1, dtype=wp.int32, device=self.device)  # read by the conditional node
        n_codes = 2 if baseline == "flat" else 1
        self.target_names = net.target_names
        self.n_cand, self.n_mu = planner.n_cand, planner.n_mu
        spec = net.patch_spec
        self.patch_spec = spec

        def quads(t: torch.Tensor) -> wp.array:
            """[..., n] -> vec4 [..., n / 4] on the planner's device."""
            a = t.detach().float().cpu().numpy()
            return wp.array(np.ascontiguousarray(a.reshape(*a.shape[:-1], -1, 4)), dtype=wp.vec4, device=self.device)

        def linear(layer: torch.nn.Linear) -> tuple[wp.array, wp.array]:
            """nn.Linear -> (weight [n_in rounded up to 4, n_out / 4], bias [n_out / 4]), zero rows for the pad."""
            weight = layer.weight.detach()
            weight = torch.nn.functional.pad(weight, (0, -weight.shape[1] % 4)).T
            return quads(weight), quads(layer.bias)

        rows = -(-self.n_cand // ROWS_PER_THREAD) * ROWS_PER_THREAD
        embed_dim, width = net.film.in_features, net.geometry.out_channels
        head_width = net.head_trunk[0].out_features
        assert embed_dim % 4 == 0 and width % 4 == 0 and head_width % 4 == 0
        with wp.ScopedDevice(self.device):
            self.pose = wp.zeros(1, dtype=wp.vec3)
            self.contacts = wp.array(WHEEL_CONTACTS_LOCAL.astype(np.float32), dtype=wp.vec2)
            self.patch = wp.zeros((spec.ny, spec.nx), dtype=wp.float32)
            self.code = wp.zeros((n_codes, width // 4), dtype=wp.vec4)  # row 0 the patch's, row 1 the level one
            self.features = wp.zeros((rows, 2), dtype=wp.vec4)  # the 7 command features + a 0 pad
            self.embedding_hidden = wp.zeros((rows, embed_dim // 4), dtype=wp.vec4)
            self.embedding = wp.zeros((rows, embed_dim // 4), dtype=wp.vec4)
            self.film_out = wp.zeros((rows, 2 * width // 4), dtype=wp.vec4)
            self.normed = wp.zeros((n_codes * rows, width // 4), dtype=wp.vec4)
            self.hidden = [wp.zeros((n_codes * rows, head_width // 4), dtype=wp.vec4) for _ in range(2)]
            self.prediction = wp.zeros((self.n_cand, len(self.target_names)), dtype=float)
            self.level_prediction = wp.zeros((self.n_cand, len(self.target_names)), dtype=float)
            self.cost_weight = wp.zeros(len(self.target_names), dtype=float)
        # (layer, SiLU after it) in order, up to the FiLM; then the LayerNorm, then the head trunk
        self.encoder = [(*linear(net.command_encoder[0]), 1), (*linear(net.command_encoder[2]), 0)]
        self.film = linear(net.film)
        layer_norm = net.head_pre[0]
        self.layer_norm = tuple(wp.array(t.detach().cpu().numpy(), dtype=float, device=self.device)
                                for t in (layer_norm.weight, layer_norm.bias)) + (float(layer_norm.eps),)
        self._film_layer_norm = _film_layer_norm_silu_kernel(width)
        self.trunk = [(*linear(net.head_trunk[i]), 1) for i in (0, 2)]
        heads = [getattr(net, f"head_{name}") for name in self.target_names]
        self.head = (quads(torch.cat([h.weight for h in heads])),
                     wp.array(torch.cat([h.bias for h in heads]).detach().cpu().numpy(), dtype=float, device=self.device))
        normalizer = net.target_transform.normalizer
        self.target = tuple(wp.array(t.reshape(-1).float().cpu().numpy(), dtype=float, device=self.device)
                            for t in (normalizer.mean, normalizer.std))
        self.set_weights(weights)

        # sim.elevation is a stable buffer (set_terrain copies into it), so it is read at update time
        self._planner = planner
        # torch and float views of the Warp buffers, which own the memory
        self._code_torch = wp.to_torch(self.code).view(n_codes, -1)
        self._film_out_floats, self._code_floats, self._normed_floats = (
            _as_floats(a) for a in (self.film_out, self.code, self.normed))
        self._patch_torch = wp.to_torch(self.patch).view(1, 1, spec.ny, spec.nx)
        if baseline == "flat":  # a level patch is all zeros at any pose, so its code is a constant
            with torch.no_grad():
                level = torch.zeros_like(self._patch_torch)
                self._code_torch[1].copy_(self.net.terrain_code(prepare_patch(level, blur_terrain)).reshape(-1))

        # windows 1 .. n_windows - 1: a patch per rollout, so the trunk runs inside the graph (Warp)
        self.n_windows, self.n_rollouts = n_windows, planner.n_rollouts
        if n_windows == 1:
            return
        from feasibility.nn_mppi.warp_trunk import WarpTrunk  # warp_trunk imports this module's kernels

        self.window_rows = _padded_rows(self.n_rollouts)
        later_rows = (n_windows - 1) * self.window_rows
        needed = trunk_rows(self.n_rollouts, n_windows)
        self.terrain_trunk = trunk if trunk is not None else WarpTrunk(net, needed, self.device)
        if self.terrain_trunk.max_batch < needed:
            raise ValueError(f"the shared trunk takes {self.terrain_trunk.max_batch} patches, this cost needs {needed}")
        with wp.ScopedDevice(self.device):
            # the last patch row is never written: the all-zero level patch, whose code the level
            # half of the head reads (baseline "flat")
            self.patches = wp.zeros((needed, spec.ny, spec.nx), dtype=wp.float32)
            self.rollout_features = wp.zeros((later_rows, 2), dtype=wp.vec4)
            self.rollout_embedding_hidden = wp.zeros((later_rows, embed_dim // 4), dtype=wp.vec4)
            self.rollout_embedding = wp.zeros((later_rows, embed_dim // 4), dtype=wp.vec4)
            self.rollout_film_out = wp.zeros((later_rows, 2 * width // 4), dtype=wp.vec4)
            self.rollout_normed = wp.zeros((n_codes * later_rows, width // 4), dtype=wp.vec4)
            self.rollout_hidden = [wp.zeros((n_codes * later_rows, head_width // 4), dtype=wp.vec4) for _ in range(2)]
            shape = (n_windows - 1, self.n_rollouts, len(self.target_names))
            self.rollout_prediction = wp.zeros(shape, dtype=float)
            self.rollout_level_prediction = wp.zeros(shape, dtype=float)
        self._rollout_film_out_floats, self._trunk_code_floats, self._rollout_normed_floats = (
            _as_floats(a) for a in (self.rollout_film_out, self.terrain_trunk.code, self.rollout_normed))

    @classmethod
    def from_checkpoint(
        cls, path: pathlib.Path, planner: MppiGpu, weights: dict[str, float], baseline: str = "none",
        n_windows: int = 1,
    ) -> "WindowCost":
        net, ckpt = load_checkpoint(path, wp.device_to_torch(planner.device))
        return cls(net, ckpt["label_attrs"], planner, weights, blur_terrain=bool(ckpt["blur_terrain"]), baseline=baseline,
                   n_windows=n_windows)

    def set_weights(self, weights: dict[str, float]) -> None:
        """Cost weight per target name (1/m for e_pos, 1/rad for the angles). Graph-safe."""
        if set(weights) != set(self.target_names):
            raise ValueError(f"weights {sorted(weights)} must name exactly the targets {self.target_names}")
        self.cost_weight.assign(np.array([weights[k] for k in self.target_names], np.float32))

    def set_enabled(self, enabled: bool) -> None:
        """Run (True) or skip (False) the hook's work in the refines launched after this call. A
        stream-ordered device write, so it takes effect between two graph launches. Switchable only."""
        if not self.switchable:
            raise ValueError("set_enabled needs a WindowCost built with switchable=True")
        self.enabled.fill_(int(enabled))

    def replan_split(self, state: np.ndarray, goal: tuple[float, float], n_refine: int, cost_refines: int) -> wp.array:
        """`planner.replan(state, goal, n_refine)` with this cost charged in the FIRST `cost_refines`
        refines only; the rest rank the rollouts by the planner's own cost. Two replans of the same
        captured graph, which together are exactly one replan of n_refine."""
        if not 1 <= cost_refines <= n_refine:
            raise ValueError(f"cost_refines {cost_refines} must be in [1, {n_refine}]")
        self.set_enabled(True)
        U = self._planner.replan(state, goal, cost_refines)
        if n_refine > cost_refines:
            self.set_enabled(False)
            U = self._planner.replan(state, goal, n_refine - cost_refines)
        return U

    def _stream_context(self) -> contextlib.AbstractContextManager:
        if not self.device.is_cuda:
            return contextlib.nullcontext()
        return torch.cuda.stream(wp.stream_to_torch(wp.get_stream(self.device)))

    @torch.no_grad()
    def update(self, state: np.ndarray) -> None:
        """The terrain code at the start pose `state` (x, y, yaw), for the next replan."""
        self.pose.assign(np.asarray(state[:3], np.float32).reshape(1, 3))
        spec, sim = self.patch_spec, self._planner.sim
        wp.launch(
            _patch_kernel, (spec.ny, spec.nx),
            inputs=[sim.elevation, sim.grid, self.pose, spec.x_min, spec.y_min, spec.cell, self.contacts, HEIGHT_SCALE],
            outputs=[self.patch], device=self.device,
        )
        with self._stream_context():
            code = self.net.terrain_code(prepare_patch(self._patch_torch, self.blur_terrain))
            self._code_torch[0].copy_(code.reshape(-1))

    def __call__(self, planner: MppiGpu) -> None:
        """The hook: launches only, on the planner's buffers, so it is captured into the refine;
        switchable, inside a conditional node that runs only while `enabled` is 1."""
        if self.switchable:
            wp.capture_if(self.enabled, on_true=self._charge, planner=planner)
        else:
            self._charge(planner)

    def _charge(self, planner: MppiGpu) -> None:
        dev, rows = self.device, self.features.shape[0]

        def dense(x: wp.array, weight: wp.array, bias: wp.array, silu: int, y: wp.array) -> None:
            groups = x.shape[0] // ROWS_PER_THREAD
            wp.launch(_dense_kernel, (groups, y.shape[1]), inputs=[x, weight, bias, silu], outputs=[y], device=dev)

        wp.launch(
            _features_kernel, self.n_cand,
            inputs=[planner.sim.target_wheel_omega, WINDOW_STEPS, WHEEL_RADIUS, HALF_TRACK, V_SCALE, WZ_SCALE],
            outputs=[self.features], device=dev,
        )
        dense(self.features, *self.encoder[0], self.embedding_hidden)
        dense(self.embedding_hidden, *self.encoder[1], self.embedding)
        dense(self.embedding, *self.film, 0, self.film_out)
        wp.launch_tiled(
            self._film_layer_norm, dim=self.normed.shape[0], block_dim=32,
            inputs=[self._film_out_floats, self._code_floats, rows, 0, *self.layer_norm],
            outputs=[self._normed_floats], device=dev,
        )
        dense(self.normed, *self.trunk[0], self.hidden[0])
        dense(self.hidden[0], *self.trunk[1], self.hidden[1])
        wp.launch(
            _output_kernel, self.n_cand,
            inputs=[self.hidden[1], *self.head, *self.target, self.cost_weight, self.n_cand, self.n_mu,
                    rows if self.baseline == "flat" else 0],
            outputs=[self.prediction, self.level_prediction, planner.J], device=dev,
        )
        if self.n_windows == 1:
            return

        # windows 1 .. n_windows - 1: every ROLLOUT (not candidate: the mu replicas of a candidate
        # end window 0 at different poses) from its own pose at the window's first step
        spec, sim, n_later, later_rows = self.patch_spec, planner.sim, self.n_windows - 1, self.rollout_features.shape[0]
        wp.launch(
            _rollout_patch_kernel, (n_later, self.n_rollouts, spec.ny * spec.nx),
            inputs=[sim.elevation, sim.grid, sim.controlled, WINDOW_STEPS, self.window_rows, spec.x_min, spec.y_min,
                    spec.cell, self.contacts, HEIGHT_SCALE],
            outputs=[self.patches], device=dev,
        )
        self.terrain_trunk(self.patches)
        wp.launch(
            _rollout_features_kernel, (n_later, self.n_rollouts),
            inputs=[sim.target_wheel_omega, WINDOW_STEPS, self.window_rows, WHEEL_RADIUS, HALF_TRACK, V_SCALE, WZ_SCALE],
            outputs=[self.rollout_features], device=dev,
        )
        dense(self.rollout_features, *self.encoder[0], self.rollout_embedding_hidden)
        dense(self.rollout_embedding_hidden, *self.encoder[1], self.rollout_embedding)
        dense(self.rollout_embedding, *self.film, 0, self.rollout_film_out)
        wp.launch_tiled(
            self._film_layer_norm, dim=self.rollout_normed.shape[0], block_dim=32,
            inputs=[self._rollout_film_out_floats, self._trunk_code_floats, later_rows, 1, *self.layer_norm],
            outputs=[self._rollout_normed_floats], device=dev,
        )
        dense(self.rollout_normed, *self.trunk[0], self.rollout_hidden[0])
        dense(self.rollout_hidden[0], *self.trunk[1], self.rollout_hidden[1])
        wp.launch(
            _rollout_output_kernel, self.n_rollouts,
            inputs=[self.rollout_hidden[1], *self.head, *self.target, self.cost_weight, n_later, self.window_rows,
                    later_rows if self.baseline == "flat" else 0],
            outputs=[self.rollout_prediction, self.rollout_level_prediction, planner.J], device=dev,
        )


# --- self-test -----------------------------------------------------------------------------------


def _test_planner(terrain: object, batch: int, k_turn: float, n_mu: int, seed: int = 0) -> MppiGpu:
    """An MppiGpu on `terrain` configured as design.md section 3 (H 31, 4 knots, wmin 0), with
    straight and spin candidates so sign(wz) = 0 and |wz| near the box edge are both exercised."""
    from helhest import dynamics
    from helhest import friction
    from helhest.control.mppi import CostParams
    from helhest.control.mppi import SamplingConfig
    from helhest.engine import ForwardSimulator

    elevation, grid = terrain.to_hstack("cuda:0")
    sim = ForwardSimulator(dynamics.robot_params(), dynamics.planning_solver(k_turn=k_turn), grid, batch, HORIZON, "cuda:0")
    sim.set_terrain(elevation)
    extent = dict(xlim=(terrain.x0, terrain.x0 + (terrain.nx - 1) * terrain.cell),
                  ylim=(terrain.y0, terrain.y0 + (terrain.ny - 1) * terrain.cell), cell=terrain.cell)
    sim.set_friction(friction.uniform(0.8, **extent))
    sampling = SamplingConfig(n_knots=N_KNOTS, wmin=0.0, wide_frac=0.25, straight_frac=0.1, spin_frac=0.1, n_mu=n_mu)
    planner = MppiGpu(sim, CostParams(), sampling, n_theta=16, seed=seed)
    planner.reset_nominal(1.5)
    return planner


def _load(checkpoint: pathlib.Path | None) -> tuple[WindowDivergenceNet, dict[str, object], bool]:
    """(net, label_attrs, blur_terrain) of `checkpoint`; None = a random-weight net whose
    TargetTransform keeps the outputs off the clamp, with the design's twin as its labels."""
    from feasibility.lattice_learning.model import Normalizer
    from feasibility.lattice_learning.model import TargetTransform

    if checkpoint is not None:
        net, ckpt = load_checkpoint(checkpoint, torch.device("cuda:0"))
        return net.eval(), ckpt["label_attrs"], bool(ckpt["blur_terrain"])
    torch.manual_seed(0)
    net = WindowDivergenceNet(target_transform=TargetTransform(Normalizer(mean=torch.tensor([0.3, 0.4]), std=torch.tensor([0.2, 0.3]))))
    with torch.no_grad():  # a random init leaves the FiLM near identity; make the command matter
        net.film.weight.mul_(20.0)
    return net.to("cuda:0").eval(), {"k_turn": 1.0, "mppi_dt": 0.1}, False


def self_test(checkpoint: pathlib.Path | None, batch: int) -> None:
    from feasibility.heightmap.create_box_obstacles import build_centered_box
    from feasibility.heightmap.create_rough_terrain import build_rough_terrain
    from feasibility.heightmap.heightmap_reader import HeightMapReader
    from feasibility.lattice_learning.patch import sample_patches
    from feasibility.mppi_learning.command import encode
    from feasibility.mppi_learning.model import window_command_features

    wp.init()
    net, label_attrs, blur = _load(checkpoint)
    k_turn = float(label_attrs["k_turn"])
    weights = {name: 1.0 for name in net.target_names}
    terrain = build_centered_box(0.3, 0.08, 70.0, 12.0)
    state, goal = np.array([-1.6, 0.4, 0.3]), (4.0, 0.0)

    # a planner whose twin turns differently from the label's is refused
    try:
        WindowCost(net, label_attrs, _test_planner(terrain, 64, k_turn + 0.5, n_mu=1), weights)
        raise AssertionError("a planner at another k_turn was accepted")
    except ValueError as e:
        print(f"[refuse] {e}")

    planner = _test_planner(terrain, batch, k_turn, n_mu=2)
    cost = WindowCost(net, label_attrs, planner, weights, blur_terrain=blur)
    spec = net.patch_spec

    # 1. patch: the Warp sampler vs sample_patches, on the box face and hanging off the edge of a
    # rough map (both clamp there)
    rough = build_rough_terrain(extent=12.0)
    rough_cost = WindowCost(net, label_attrs, _test_planner(rough, 64, k_turn, n_mu=1), weights)
    for name, where, samples, pose in (("box", terrain, cost, state), ("rough, map edge", rough, rough_cost, np.array([-5.2, 4.9, 2.4]))):
        samples.update(pose)
        ref = sample_patches(where, pose[None], spec)[0]
        err = np.abs(samples.patch.numpy() - ref).max()
        assert err < 1e-5, err
        print(f"[patch] {name} {pose.tolist()}: relief [{ref.min():.3f}, {ref.max():.3f}] wheel radii, max |diff| {err:.1e}")

    # 2 + 3. command and head: one refine, then the candidates' window 0 through the torch net
    cost.update(state)
    planner.set_cost_hook(cost)
    planner.replan(state, goal, 1)
    omega = planner.sim.target_wheel_omega.numpy()[:WINDOW_STEPS, : planner.n_cand]
    command = torch.from_numpy(encode(omega)).cuda()
    features = window_command_features(command).cpu().numpy()
    err_features = np.abs(cost.features.numpy().reshape(-1, 8)[: planner.n_cand, :7] - features).max()
    assert err_features < 1e-5, err_features
    assert (features[:, 4] == 0).any() and (features[:, 4] != 0).any()
    with torch.no_grad():
        patch = torch.from_numpy(sample_patches(terrain, state[None], spec)).cuda()[:, None]
        expected = net.predict(prepare_patch(patch, blur).expand(len(command), -1, -1, -1), command).cpu().numpy()
    got = cost.prediction.numpy()
    err_head = np.abs(got - expected).max() / max(1e-3, np.abs(expected).max())
    assert err_head < 1e-4, err_head
    print(f"[command] features of {planner.n_cand} candidates vs command.encode: max |diff| {err_features:.1e} "
          f"({int((features[:, 4] == 0).sum())} straight)")
    print(f"[head] Warp vs net.predict: max rel diff {err_head:.1e}; "
          + ", ".join(f"{n} [{got[:, k].min():.3f}, {got[:, k].max():.3f}]" for k, n in enumerate(net.target_names)))

    # the term reaches J: the same seed without the hook samples the same candidates
    plain = _test_planner(terrain, batch, k_turn, n_mu=2)
    plain.replan(state, goal, 1)
    plain_jc = plain.Jc.numpy()
    added = (got @ np.ones(len(weights), np.float32))
    dJc = planner.Jc.numpy() - plain_jc
    err_j = np.abs(dJc - added).max() / max(1.0, np.abs(plain.Jc.numpy()).max())
    assert err_j < 1e-5, err_j
    print(f"[J] the hook's term is in the robust cost Jc, max rel diff {err_j:.1e}")

    # 4. zero weights: the hook adds exactly 0, so U is bit-identical to the unhooked planner's
    zero, plain = _test_planner(terrain, batch, k_turn, n_mu=2), _test_planner(terrain, batch, k_turn, n_mu=2)
    zero_cost = WindowCost(net, label_attrs, zero, weights)
    zero_cost.set_weights({name: 0.0 for name in net.target_names})
    zero.set_cost_hook(zero_cost)
    for _ in range(3):
        zero_cost.update(state)
        assert np.array_equal(zero.replan(state, goal, 3).numpy(), plain.replan(state, goal, 3).numpy())
    print("[no-op] zero weights: U bit-identical to the unhooked planner over 3 replans x 3 refines")

    # 5. baseline "flat": the level head matches net.predict on a zero patch, the terrain head is
    # unchanged, and J gets sum_k w_k * max(e_k - level_k, 0)
    flat_planner = _test_planner(terrain, batch, k_turn, n_mu=2)
    flat_cost = WindowCost(net, label_attrs, flat_planner, weights, blur_terrain=blur, baseline="flat")
    flat_cost.update(state)
    flat_planner.set_cost_hook(flat_cost)
    flat_planner.replan(state, goal, 1)
    with torch.no_grad():
        level_expected = net.predict(torch.zeros_like(patch).expand(len(command), -1, -1, -1), command).cpu().numpy()
    got_flat, level = flat_cost.prediction.numpy(), flat_cost.level_prediction.numpy()
    err_level = np.abs(level - level_expected).max() / max(1e-3, np.abs(level_expected).max())
    err_same = np.abs(got_flat - got).max()
    assert err_level < 1e-4 and err_same == 0.0, (err_level, err_same)
    added = np.maximum(got_flat - level, 0.0) @ np.ones(len(weights), np.float32)
    err_j = np.abs(flat_planner.Jc.numpy() - plain_jc - added).max() / max(1.0, np.abs(plain_jc).max())
    assert err_j < 1e-5, err_j
    print(f"[flat] level head vs net.predict: max rel diff {err_level:.1e}; excess over level in Jc: max rel diff "
          f"{err_j:.1e}; {100 * (added > 0).mean():.0f}% of candidates charged, mean excess {added.mean():.4f}")

    # 6. baseline "flat" on level ground: patch == level, the excess is exactly 0, U bit-identical
    level_map = HeightMapReader.flat(xlim=(-6.0, 6.0), ylim=(-6.0, 6.0), cell=0.05)
    hooked, plain = _test_planner(level_map, batch, k_turn, n_mu=2), _test_planner(level_map, batch, k_turn, n_mu=2)
    level_cost = WindowCost(net, label_attrs, hooked, weights, blur_terrain=blur, baseline="flat")
    hooked.set_cost_hook(level_cost)
    for _ in range(3):
        level_cost.update(state)
        assert np.array_equal(hooked.replan(state, goal, 3).numpy(), plain.replan(state, goal, 3).numpy())
    print("[flat] level map: excess exactly 0, U bit-identical to the unhooked planner over 3 replans x 3 refines")
    if blur:
        print("all self-checks ok (windows 1-2 skipped: the blur-terrain baseline is window 0 only)")
        return

    # 7. windows 1 and 2 (n_windows 3, baseline "flat"): every rollout's patch at its OWN pose at
    # steps 10 and 20, its command over the window, both heads per window, and the term in Jc:
    # window 0's excess plus the mean over a candidate's mu replicas of theirs (the robust cost
    # averages J over the replicas)
    later = _test_planner(terrain, batch, k_turn, n_mu=2)
    later_cost = WindowCost(net, label_attrs, later, weights, baseline="flat", n_windows=3)
    later_cost.update(state)
    later.set_cost_hook(later_cost)
    later.replan(state, goal, 1)
    poses, omega_all = later.sim.controlled.numpy(), later.sim.target_wheel_omega.numpy()
    n, rows = later.n_rollouts, later_cost.window_rows
    got_later, level_later = later_cost.rollout_prediction.numpy(), later_cost.rollout_level_prediction.numpy()
    patches, features_later = later_cost.patches.numpy(), later_cost.rollout_features.numpy().reshape(-1, 8)
    assert not patches[-1].any()  # the level patch
    for w in range(2):
        first = (w + 1) * WINDOW_STEPS
        ref = sample_patches(terrain, poses[first].astype(np.float64), spec)
        err_patch = np.abs(patches[w * rows : w * rows + n] - ref).max()
        command_w = torch.from_numpy(encode(omega_all[first : first + WINDOW_STEPS])).cuda()
        err_features = np.abs(features_later[w * rows : w * rows + n, :7]
                              - window_command_features(command_w).cpu().numpy()).max()
        with torch.no_grad():
            ref_torch = torch.from_numpy(ref).cuda()[:, None]
            expected_w = net.predict(ref_torch, command_w).cpu().numpy()
            expected_level = net.predict(torch.zeros_like(ref_torch), command_w).cpu().numpy()
        err_head = np.abs(got_later[w] - expected_w).max() / max(1e-3, np.abs(expected_w).max())
        err_level = np.abs(level_later[w] - expected_level).max() / max(1e-3, np.abs(expected_level).max())
        assert err_patch < 1e-5 and err_features < 1e-5 and err_head < 1e-4 and err_level < 1e-4, (
            w + 1, err_patch, err_features, err_head, err_level)
        spread = np.ptp(poses[first, :, :2], axis=0)
        print(f"[window {w + 1}] {n} rollouts from their own step-{first} poses (spread {spread[0]:.2f} x {spread[1]:.2f} m): "
              f"patch max |diff| {err_patch:.1e}, features {err_features:.1e}, head / level head vs net.predict "
              f"max rel diff {err_head:.1e} / {err_level:.1e}")
    ones = np.ones(len(weights), np.float32)
    window0 = np.maximum(later_cost.prediction.numpy() - later_cost.level_prediction.numpy(), 0.0) @ ones
    per_rollout = np.maximum(got_later - level_later, 0.0).sum(axis=0) @ ones
    added = window0 + per_rollout.reshape(later.n_mu, later.n_cand).mean(axis=0)
    err_j = np.abs(later.Jc.numpy() - plain_jc - added).max() / max(1.0, np.abs(plain_jc).max())
    assert err_j < 1e-5, err_j
    print(f"[windows] window 0 + windows 1-2 excess in Jc: max rel diff {err_j:.1e}; windows 1-2 charge "
          f"{100 * (per_rollout > 0).mean():.0f}% of rollouts, mean {per_rollout.mean():.4f}")

    # 8. the no-op guarantees with windows 1-2, on a trunk shared with the cost above
    for name, where, value, baseline in (("zero weights", terrain, 0.0, "none"),
                                         ("baseline 'flat' on a level map", level_map, 1.0, "flat")):
        hooked, plain = _test_planner(where, batch, k_turn, n_mu=2), _test_planner(where, batch, k_turn, n_mu=2)
        hooked_cost = WindowCost(net, label_attrs, hooked, {k: value for k in net.target_names}, baseline=baseline,
                                 n_windows=3, trunk=later_cost.terrain_trunk)
        hooked.set_cost_hook(hooked_cost)
        for _ in range(3):
            hooked_cost.update(state)
            assert np.array_equal(hooked.replan(state, goal, 3).numpy(), plain.replan(state, goal, 3).numpy()), name
        print(f"[no-op] windows 1-2, {name}: U bit-identical to the unhooked planner over 3 replans x 3 refines")

    # 9. switchable (the hook behind a conditional graph node): enabled it is the plain hook, and
    # disabled it is no hook, U bit-identical either way; replan_split(3, 1) is the same as one
    # hooked refine, then a recapture WITHOUT the hook and two more (what it saves is the recapture).
    # The split MAY end bit-identical to vanilla: only candidate 0 and the NARROW band are drawn
    # around U, so when a vanilla refine's elites all come from the U-independent priors (WIDE,
    # STRAIGHT, SPIN), the hooked refine leaves no trace. Printed, not asserted.
    names = ("hooked", "on", "off", "plain", "split", "reference")
    planners = {name: _test_planner(terrain, batch, k_turn, n_mu=2) for name in names}
    # at weight 1 a trained net's excess is too small to move U at all; 300 is closed_loop's weight
    strong = {k: 300.0 for k in net.target_names}
    costs = {name: WindowCost(net, label_attrs, planners[name], strong, baseline="flat", n_windows=3,
                              trunk=later_cost.terrain_trunk, switchable=name in ("on", "off", "split"))
             for name in names if name != "plain"}
    for name in ("hooked", "on", "off", "split"):
        planners[name].set_cost_hook(costs[name])
    costs["off"].set_enabled(False)
    split_is_vanilla = []
    for _ in range(2):
        U = {}
        for name, planner in planners.items():
            if name in costs:
                costs[name].update(state)
            if name == "split":
                costs[name].replan_split(state, goal, 3, 1)
            elif name == "reference":
                planner.set_cost_hook(costs[name])
                planner.replan(state, goal, 1)
                planner.set_cost_hook(None)
                planner.replan(state, goal, 2)
            else:
                planner.replan(state, goal, 3)
            U[name] = planner.nominal()
        assert np.array_equal(U["on"], U["hooked"]) and np.array_equal(U["off"], U["plain"])
        assert np.array_equal(U["split"], U["reference"])
        assert not np.array_equal(U["hooked"], U["plain"])  # else the checks above prove nothing
        split_is_vanilla.append(bool(np.array_equal(U["split"], U["plain"])))
    print("[switch] conditional hook: enabled = the plain hook, disabled = no hook, replan_split(3, 1) = a hooked "
          f"refine + 2 unhooked after a recapture, U bit-identical over 2 replans (split == vanilla: {split_is_vanilla})")
    print("all self-checks ok")


def bench(checkpoint: pathlib.Path | None) -> None:
    """9d.5: a replan at the deployed 4096 rollouts x 3 refines (no hook, the hook, the hook at
    baseline "flat"), and at the window study's 512 rollouts x 1 refine (no hook, baseline "flat"
    on window 0, on windows 0-2). Each is the fastest of 3 trials of 10 replans: this laptop GPU
    drops its clock in bursts."""
    from feasibility.heightmap.create_box_obstacles import build_centered_box

    net, label_attrs, blur = _load(checkpoint)
    terrain = build_centered_box(0.3, 0.08, 70.0, 12.0)
    state, goal = np.array([-1.6, 0.4, 0.3]), (4.0, 0.0)
    weights = {name: 1.0 for name in net.target_names}
    # (batch, n_refine, [(baseline or None = no hook, n_windows, refines charged or None = all)])
    for batch, n_refine, arms in ((4096, 3, ((None, 1, None), ("none", 1, None), ("flat", 1, None))),
                                  (512, 1, ((None, 1, None), ("flat", 1, None), ("flat", 3, None))),
                                  (512, 3, ((None, 1, None), ("flat", 3, None), ("flat", 3, 1)))):
        for baseline, n_windows, cost_refines in arms:
            # a hooked planner and its cost point at each other, so only the cycle collector frees
            # the previous arm's GPU buffers (a 3-window cost's trunk is ~0.2 GB)
            gc.collect()
            planner = _test_planner(terrain, batch, float(label_attrs["k_turn"]), n_mu=1)
            cost = None if baseline is None else WindowCost(
                net, label_attrs, planner, weights, blur_terrain=blur, baseline=baseline, n_windows=n_windows,
                switchable=cost_refines is not None)
            if cost is not None:
                planner.set_cost_hook(cost)

            def step() -> None:
                if cost is not None:
                    cost.update(state)
                if cost_refines is not None:
                    cost.replan_split(state, goal, n_refine, cost_refines)
                else:
                    planner.replan(state, goal, n_refine)

            for _ in range(3):
                step()
            best = float("inf")
            for _ in range(3):
                wp.synchronize()
                start = time.perf_counter()
                for _ in range(10):
                    step()
                wp.synchronize()
                best = min(best, (time.perf_counter() - start) / 10 * 1e3)
            label = "without the hook" if baseline is None else f"with the hook, baseline {baseline!r}, {n_windows} window(s)"
            if cost_refines is not None:
                label += f", in the first {cost_refines} refine(s) only"
            print(f"[bench] {batch} rollouts, {n_refine} refine(s), {label}: {best:.2f} ms per replan")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=None, help="mppi_learning checkpoint (default: random weights)")
    parser.add_argument("--batch", type=int, default=1024, help="MPPI rollouts for the checks (default 1024)")
    parser.add_argument("--bench", action="store_true", help="also time replans without/with the hook (4096 x 3 refines, 512 x 1)")
    args = parser.parse_args()
    self_test(args.checkpoint, args.batch)
    if args.bench:
        bench(args.checkpoint)
