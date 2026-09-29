"""The command sampler for `mppi_learning`, and its 4-number encoding (design.md sections 2, 3).

One window is `WINDOW_STEPS` MPPI steps (10 x 0.1 s = 1 s), knot-aligned, so the commanded wheel
speeds are linear in time inside it. This module draws such windows from the same priors as
helhest_stack's `_sample_target_wheel_omega_kernel` -- restricted to what the net is trained on
(forward motion, `wmin` = 0, and spins) -- and turns them into

  * `wheel_omega` [WINDOW_STEPS, N, 3]: per-step setpoints on the twin's own 0.1 s grid,
  * `ostrich_setpoints`: the same profile held over ostrich's finer steps (`upsample`),
  * `encode`: `(v_mean, v_slope, wz_mean, wz_slope)`, the net's command input.

Everything is numpy and host-side: a window is 10 x 3 numbers per trial, and this file must
import without warp (the twin, which needs the GPU, lives in `twin.py`).

Mirrors the kernel rather than importing it: its priors are a Warp kernel keyed on device RNG
streams, which cannot be called for a batch of independent one-window trials.
    * WIDE / STRAIGHT / SPIN are noise-free, exactly linear in the window (SPIN constant);
    * only NARROW carries knot noise and per-step jitter, and the kernel draws ONE jitter per
      (step, candidate) and adds it to BOTH wheels -- a common-mode wobble that moves `v` and never
      `wz`. Copied on purpose: an independent-per-wheel jitter would be a different distribution.

Usage:
    python src/feasibility/mppi_learning/command.py
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from helhest.engine import RobotParams

_ROBOT = RobotParams()

MPPI_DT: float = 0.1  # s, helhest_stack's `dynamics.DT` (the twin's step), pinned here to stay torch/warp-free
WINDOW_STEPS: int = 10  # MPPI steps per window = the knot spacing (H = 10 * (n_knots - 1) + 1)
WINDOW_S: float = WINDOW_STEPS * MPPI_DT  # 1.0 s
WHEEL_RADIUS: float = _ROBOT.wheel_radius  # m
HALF_TRACK: float = _ROBOT.half_track  # m

FAMILIES: tuple[str, ...] = ("wide", "straight", "narrow", "spin")
WIDE, STRAIGHT, NARROW, SPIN = range(len(FAMILIES))  # the `family` codes


@dataclass(frozen=True)
class CommandSpec:
    """MPPI's sampling box and priors (`MppiConfig` defaults, with the design's `wmin` = 0)."""

    wmin: float = 0.0  # rad/s, no reverse
    wmax: float = 4.0  # rad/s, v_max = wmax * wheel_radius = 1.4 m/s
    sigma: float = 0.5  # rad/s, NARROW per-step jitter
    sigma_knot: float = 1.0  # rad/s, NARROW knot noise
    spin_min: float = 2.0  # rad/s, the real robot will not break loose below ~2 rad/s
    # sampling weight per family, in `FAMILIES` order; normalised on use
    mix: tuple[float, float, float, float] = (0.3, 0.2, 0.3, 0.2)

    @property
    def v_max(self) -> float:
        return self.wmax * WHEEL_RADIUS


def wheels_to_twist(wheel_omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[..., >=2] wheel speeds (left, right, ...) rad/s -> (v, wz) [...] via the ideal no-slip
    differential drive -- the inverse of `comparator.common.cmd_to_wheels`."""
    wl, wr = wheel_omega[..., 0], wheel_omega[..., 1]
    return WHEEL_RADIUS * (wl + wr) / 2.0, WHEEL_RADIUS * (wr - wl) / (2.0 * HALF_TRACK)


def sample_window_commands(
    n: int, rng: np.random.Generator, spec: CommandSpec = CommandSpec()
) -> tuple[np.ndarray, np.ndarray]:
    """N independent one-window commands -> (`wheel_omega` [WINDOW_STEPS, N, 3] float32
    (left, right, rear) rad/s on the MPPI grid, `family` [N] int8 index into `FAMILIES`).

    Step i of a window sits at knot fraction i / WINDOW_STEPS, as in the kernel's `_knot_bracket`:
    the far knot is reached only at the NEXT window's step 0, so it is not in this window."""
    p = np.asarray(spec.mix, dtype=np.float64)
    family = rng.choice(len(FAMILIES), size=n, p=p / p.sum()).astype(np.int8)
    frac = (np.arange(WINDOW_STEPS) / WINDOW_STEPS)[:, None]  # [S, 1]
    span = spec.wmax - spec.wmin

    def uniform(*shape: int) -> np.ndarray:
        return spec.wmin + span * rng.random(shape)

    # WIDE: independent uniform knots per wheel over the whole box
    knots = uniform(2, 2, n)  # [knot lo/hi, wheel l/r, N]
    wheel = (1 - frac[:, :, None]) * knots[0] + frac[:, :, None] * knots[1]  # [S, 2, N]
    wide = wheel.transpose(0, 2, 1)  # [S, N, 2]

    # STRAIGHT: one common forward speed per knot, so wl == wr all along
    v_knots = uniform(2, n)
    v = (1 - frac) * v_knots[0] + frac * v_knots[1]  # [S, N]
    straight = np.stack([v, v], axis=-1)

    # SPIN: wl = -wr, magnitude floored at spin_min, held constant across the window
    mag = spec.spin_min + (spec.wmax - spec.spin_min) * rng.random(n)
    mag = np.where(rng.random(n) < 0.5, -mag, mag)
    spin = np.broadcast_to(np.stack([-mag, mag], axis=-1), (WINDOW_STEPS, n, 2))

    # NARROW: spline noise around a constant nominal, plus a jitter shared by both wheels
    nominal = uniform(n, 2)  # [N, 2]
    knot_noise = spec.sigma_knot * rng.standard_normal((2, n, 2))
    spline = nominal + (1 - frac[:, :, None]) * knot_noise[0] + frac[:, :, None] * knot_noise[1]
    jitter = spec.sigma * rng.standard_normal((WINDOW_STEPS, n, 1))
    narrow = spline + jitter

    f = family[None, :, None]
    out = np.select([f == WIDE, f == STRAIGHT, f == NARROW, f == SPIN], [wide, straight, narrow, spin])
    return _finish(out, family, spec)


def _finish(profile: np.ndarray, family: np.ndarray, spec: CommandSpec) -> tuple[np.ndarray, np.ndarray]:
    """Clamp to the box (the spin band keeps its reversed wheel) and append the rear wheel."""
    lo = np.where(family == SPIN, -spec.wmax, spec.wmin)[None, :, None]
    profile = np.clip(profile, lo, spec.wmax)
    rear = profile.mean(axis=-1, keepdims=True)  # cmd_to_wheels: v_rear = v / r = mean(wl, wr)
    return np.concatenate([profile, rear], axis=-1).astype(np.float32), family


def upsample(wheel_omega: np.ndarray, dt_to: float, dt_from: float = MPPI_DT) -> np.ndarray:
    """Hold each `dt_from` setpoint for `dt_from / dt_to` finer steps -- the controller's zero-order
    hold, which is what ostrich (dt 2.5e-2) sees of a profile the twin reads on 0.1 s steps."""
    k = round(dt_from / dt_to)
    if abs(k * dt_to - dt_from) > 1e-9:
        raise ValueError(f"dt_to={dt_to} does not divide dt_from={dt_from}")
    return np.repeat(wheel_omega, k, axis=0)


def encode(wheel_omega: np.ndarray) -> np.ndarray:
    """[WINDOW_STEPS, N, >=2] -> [N, 4] = (v_mean, v_slope, wz_mean, wz_slope).

    `*_mean` is the window average; `*_slope` is the least-squares change ACROSS the whole window
    (fitted slope per step x WINDOW_STEPS), so it is in the same units as the mean and, for a
    knot-aligned window, is exactly the knot-to-knot difference. Mirror symmetry (a left/right
    flip of the terrain) negates `wz_mean` and `wz_slope` and leaves the other two."""
    v, wz = wheels_to_twist(wheel_omega)  # each [S, N]
    i = np.arange(WINDOW_STEPS, dtype=np.float64)[:, None]
    ic = i - i.mean()

    def mean_slope(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return x.mean(axis=0), (ic * (x - x.mean(axis=0))).sum(axis=0) / (ic**2).sum() * WINDOW_STEPS

    v_mean, v_slope = mean_slope(v)
    wz_mean, wz_slope = mean_slope(wz)
    return np.stack([v_mean, v_slope, wz_mean, wz_slope], axis=-1).astype(np.float32)


def linear_residual(wheel_omega: np.ndarray) -> np.ndarray:
    """[N] RMS over the window of (left, right) wheel speed minus its own least-squares line,
    rad/s -- the size of what the mean + slope encoding throws away."""
    x = wheel_omega[..., :2].astype(np.float64)  # [S, N, 2]
    i = np.arange(WINDOW_STEPS, dtype=np.float64)[:, None, None]
    ic = i - i.mean()
    slope = (ic * (x - x.mean(axis=0))).sum(axis=0) / (ic**2).sum()
    fit = x.mean(axis=0) + slope * ic
    return np.sqrt(((x - fit) ** 2).mean(axis=(0, 2)))


if __name__ == "__main__":
    spec = CommandSpec()
    rng = np.random.default_rng(0)
    n = 20_000
    w, fam = sample_window_commands(n, rng, spec)
    assert w.shape == (WINDOW_STEPS, n, 3) and fam.shape == (n,)
    assert set(np.unique(fam)) == set(range(len(FAMILIES)))
    assert np.isfinite(w).all()

    # Forward-only except the spin band, whose wheels are exactly opposite.
    forward = fam != SPIN
    assert w[:, forward, :].min() >= spec.wmin - 1e-6 and w.max() <= spec.wmax + 1e-6
    spin = w[:, fam == SPIN]
    assert np.allclose(spin[..., 0], -spin[..., 1]) and np.abs(spin[..., 1]).min() >= spec.spin_min - 1e-5
    assert np.allclose(spin[..., 1], spin[:1, :, 1]), "a spin is held, not ramped"
    straight = w[:, fam == STRAIGHT]
    assert np.allclose(straight[..., 0], straight[..., 1])
    assert np.allclose(w[..., 2], w[..., :2].mean(axis=-1), atol=1e-6)

    # The section-2 assumption: the mean + slope encoding is lossless for the noise-free priors
    # (only a clamp could break it, and their knots are inside the box), and for NARROW what it
    # drops is jitter-sized.
    resid = linear_residual(w)
    for code in (WIDE, STRAIGHT, SPIN):
        assert resid[fam == code].max() < 1e-5, FAMILIES[code]
    narrow_rms = resid[fam == NARROW].mean()
    assert narrow_rms < spec.sigma, f"NARROW residual {narrow_rms:.3f} exceeds the jitter {spec.sigma}"

    # And the slope is the knot difference: reconstruct a WIDE window from (mean, slope) alone.
    e = encode(w)
    v, wz = wheels_to_twist(w)
    i = np.arange(WINDOW_STEPS)[:, None]
    wide = fam == WIDE
    v_fit = e[wide, 0] + e[wide, 1] * (i - (WINDOW_STEPS - 1) / 2) / WINDOW_STEPS
    assert np.abs(v_fit - v[:, wide]).max() < 1e-5
    assert np.abs(e[:, 0]).max() <= spec.v_max + 1e-5

    # Ostrich sees the same profile, held.
    o = upsample(w[:, :5], dt_to=2.5e-2)
    assert o.shape == (4 * WINDOW_STEPS, 5, 3) and np.array_equal(o[::4], w[:, :5])

    print(f"OK  window {WINDOW_S:.1f} s, families {dict(zip(FAMILIES, np.bincount(fam) / n))}")
    print(f"    NARROW residual {narrow_rms:.3f} rad/s vs jitter sigma {spec.sigma}")
