"""Replay every scenario in both simulators and compare the result against a committed baseline.

This is a CHARACTERIZATION test, not a correctness test: the baseline has no notion of right, it
records what ostrich and helhest_stack did on the day it was written. A reported change is a
prompt to look, not a bug report -- and accepting one is a deliberate, separately-committed act
(`+update=true`), so `git log` on baselines/ becomes the record of which submodule bump changed
the physics and by how much. Health flags (see metrics.py) are the exception: a NaN or a body off
the map fails regardless of the baseline, and there is no way to accept one.

WHAT THE VERDICT ACTUALLY RESTS ON. The two simulators are not equally reproducible, so they do
not get equal authority here (the FLOOR_ATOL comment carries the measurements):

  helhest_stack   bit-exact across identical runs. Every h_* metric is asserted at a tight floor,
                  so any change to the kinematic twin is caught.
  ostrich         intrinsically nondeterministic -- a pivot on near-tangential contacts is a real
                  bifurcation, and repeats scatter by tens of degrees of yaw. Its well-conditioned
                  cells are asserted; the cells whose tolerance is set by that scatter are REPORTED
                  AND NOT ASSERTED, because a verdict resting on them is a coin flip, and a test
                  that fails at random gets ignored, which costs more than the coverage is worth.

So a green run means: nothing blew up, helhest_stack is unchanged, and ostrich did not change
grossly. It does not mean ostrich is unchanged -- read the "moved" lines for that.

CLI parameters (Hydra overrides -- note the leading `+`, these are new keys, not config edits):
    +tier=smoke|full      scenario set (default: smoke). Baselines are per-tier.
    +update=true          rewrite this tier's baseline instead of checking it. Runs
                          DEFAULT_UPDATE_REPEATS passes and stores the measured run-to-run noise
                          alongside the values, because that noise IS the tolerance (see below).
    +scenarios=[box,...]  run a subset by name
    +repeat=N             run everything N times and report the spread instead of comparing --
                          how to inspect the noise floor without touching the baseline
    +device=cuda:0        Warp device for BOTH sims (see comparator.common.init_warp_device)
    +mu=0.8 +k_turn=0.6   friction / turn gain, as in every comparator driver
    +no_h5=true           skip writing the per-scenario replay files

Usage:
    python src/feasibility/submodule_test/run_check.py +tier=smoke
    python src/feasibility/submodule_test/run_check.py +tier=smoke +update=true
    python src/feasibility/submodule_test/run_check.py +tier=full +repeat=3

Alongside the check it writes outputs/submodule_test_<scenario>.h5 in comparator/provenance.py's
schema, so anything the report flags is one command from being looked at:

    python src/feasibility/replay/gl_replay.py --file outputs/submodule_test_box.h5 --id 1 --which both
    python src/feasibility/plotting/batch_comparator_viewer.py --file outputs/submodule_test_box.h5 --id 1

Exit status is non-zero when the check fails, so this is usable from a shell script.
"""

from __future__ import annotations

import datetime
import pathlib
from dataclasses import dataclass

import hydra
import numpy as np
import warp as wp
from omegaconf import DictConfig
from ostrich import EngineConfig
from ostrich import LoggingConfig
from ostrich import RenderingConfig
from ostrich import SimulationConfig

from helhest import dynamics

from feasibility.comparator.common import build_setpoints
from feasibility.comparator.common import CONFIG_PATH
from feasibility.comparator.common import euler_zyx_to_quat_xyzw
from feasibility.comparator.common import init_warp_device
from feasibility.comparator.common import K_P
from feasibility.comparator.common import OUT_DIR
from feasibility.comparator.common import run_hstack_batch
from feasibility.comparator.common import run_ostrich_batch
from feasibility.comparator.provenance import git_provenance
from feasibility.comparator.provenance import write_comparison
from feasibility.submodule_test.metrics import compute_metrics
from feasibility.submodule_test.metrics import FLAG_NAMES
from feasibility.submodule_test.metrics import health_flags
from feasibility.submodule_test.metrics import METRIC_NAMES
from feasibility.submodule_test.metrics import METRIC_UNITS
from feasibility.submodule_test.metrics import plausible_bounds
from feasibility.submodule_test.scenarios import Scenario
from feasibility.submodule_test.scenarios import scenario_set
from feasibility.submodule_test.scenarios import select_scenarios

BASELINE_DIR = pathlib.Path(__file__).parent / "baselines"

# Keyframes kept per trajectory. The baseline's job is to say WHEN two runs started to differ, not
# to reproduce a rollout -- 32 evenly-spaced poses do that at a few KB per scenario, where the
# full [T, n, 7] would be the size of a dataset file and would drift on every duration change.
TRAJ_KEYFRAMES = 32

# Explicit rather than run_ostrich_batch's settle_steps=None default (which resolves to
# max(60, 0.5s/dt) inside ostrich): a baseline should not silently shift because a default in a
# submodule changed. 24 steps is ~0.7 s at ostrich's dt=3e-2, comfortably past the ~7 steps that
# run_ostrich_batch's docstring measures the chassis taking to come to rest from the +0.5 m spawn.
SETTLE_STEPS = 24

# Tolerance is |current - baseline| <= floor_atol + NOISE_FACTOR * noise + REL_TOL * |baseline|,
# where `noise` is the per-(trial, metric) spread MEASURED when the baseline was recorded and
# stored alongside it.
#
# Measured, not chosen, because THE TWO SIMULATORS SIT AT OPPOSITE EXTREMES and no single
# tolerance can serve both. Measured on a GTX 1650 / warp 1.14 smoke tier, over three separate
# calibration passes (+repeat=3, 3, 8):
#
#   helhest_stack   every h_* metric spread EXACTLY 0.0, in all three passes. Its fused kernel is
#                   bit-reproducible, which is why its own tests/engine/golden.py can demand
#                   np.array_equal. Anything that moves an h_* metric is a real change.
#   ostrich         nondeterministic run to run, and not subtly: net yaw spread 12.7-31.2 deg and
#                   final xy 0.07-0.21 m across identical repeats. Confirmed intrinsic, NOT an
#                   artifact of batching trials as parallel worlds -- a num_worlds=1 rerun of one
#                   trial is equally non-reproducible (5.2e-2 m of trajectory spread, never
#                   bit-identical). The physical cause is real: a wheel grazing a 75 deg ramp
#                   corner, or pivoting on near-tangential contacts, is a bifurcation the Newton
#                   solve amplifies from float noise.
#
# The awkward part is that the noise ESTIMATE is itself unstable -- the same metric measured
# 31.2 / 18.8 / 12.7 deg on three passes. A per-cell estimate from a handful of repeats therefore
# UNDERestimates about as often as it overestimates, and a baseline built on one unlucky-low draw
# fails on the very next run (observed: a cell measured at 1.9 deg saw a 6.4 deg deviation next
# run). Two things address that: pool the estimate across a scenario's trials before storing it
# (see _pool_noise), and allow generous headroom over it.
#
# Consequence worth being explicit about: an ostrich cell with any real scatter is NOT asserted on
# at all (see NOISE_ASSERT_MULTIPLE) -- it is reported, so a human still sees it move. The stored
# noise therefore sets how wide the reported tolerance is, not whether the run passes.
FLOOR_ATOL = {
    "m": 5e-3,
    "deg": 1e-1,
    "rad_s": 1e-2,
    "frac": 1e-3,
}
TRAJ_FLOOR_ATOL = 5e-3  # positions and quaternion components are both order-1 here
REL_TOL = 1e-2

# Headroom over the pooled measured spread. 3x covers the worst observed pass (31.2 deg) from the
# median estimate (~12.7 deg) with room to spare.
NOISE_FACTOR = 3.0

# `+update=true` runs this many repeats unless told otherwise: recording a baseline without
# measuring its noise produces one that cries wolf on the very next run, and three repeats proved
# too few to estimate a tail this heavy.
DEFAULT_UPDATE_REPEATS = 5

# A cell is ASSERTED only if its measured run-to-run noise fits inside the floor tolerance, i.e.
# it is effectively reproducible. Everything else is reported and excluded from the verdict.
#
# This bar is deliberately absolute rather than relative to the metric's own value. A relative bar
# ("noise < half the value") kept asserting on large-but-bistable quantities and the harness stayed
# flaky: box_tall/approach_face's o_peak_z swung 1.73 -> 1.19 m between identical runs, because
# driving a 1 m/s robot into a 0.7 m box either rears it up or bounces it off, and five repeats
# measured only a 0.09 m spread. No amount of widening fixes a bimodal outcome -- the honest move
# is to stop pretending such a cell can support a pass/fail decision.
#
# What remains asserted is still the useful part: every helhest_stack metric (bit-exact), plus the
# ostrich cells that genuinely do reproduce. Verified to retain real detection power -- a run at
# +mu=0.6 against a mu=0.8 baseline is caught on h_final_x at 0.28 m against a +-0.008 m tolerance.
NOISE_ASSERT_MULTIPLE = 1.0


@dataclass
class ScenarioResult:
    """One scenario's scored rollout: what goes into the baseline, plus the flags that fail
    independently of it."""

    name: str
    trial_labels: np.ndarray
    metrics: np.ndarray  # [n_trials, len(METRIC_NAMES)]
    flags: dict[str, np.ndarray]  # {name: [n_trials] bool}
    traj_ostrich: np.ndarray  # [n_trials, K, 7]
    traj_hstack: np.ndarray  # [n_trials, K, 7]

    def flagged(self) -> list[tuple[str, str]]:
        """[(trial_label, flag_name)] for every flag that fired, in trial order."""
        return [
            (str(self.trial_labels[i]), name)
            for i in range(len(self.trial_labels))
            for name in FLAG_NAMES
            if self.flags[name][i]
        ]


def _keyframes(a: np.ndarray, k: int = TRAJ_KEYFRAMES) -> np.ndarray:
    """[T, n, D] -> [n, k, D], evenly sampled along time including both endpoints. The two sims
    run at different dt and therefore different T, so this also puts them on a common axis."""
    idx = np.linspace(0, a.shape[0] - 1, k).round().astype(int)
    return np.ascontiguousarray(a[idx].transpose(1, 0, 2))


def simulate_scenario(
    scenario: Scenario,
    *,
    sim_config: SimulationConfig,
    render_config: RenderingConfig,
    engine_config: EngineConfig,
    logging_config: LoggingConfig,
    mu: float,
    k_turn: float,
    device: str,
    write_h5: bool,
) -> ScenarioResult:
    """Replay every trial of `scenario` as N parallel worlds -- one terrain, one model build.

    Both batch runners take a per-world [N, 3] spawn and per-world [T, N, 3] setpoints, so the
    whole scenario is one call each; this is the batching grid_learning/generate_dataset.py's
    simulate_map established, and the reason a scenario costs one build rather than one per trial.
    What a batch cannot mix is TERRAIN (HelhestBatchSimulator bakes the heightmap into the model's
    globals builder, shared by every world), hence one call to this function per scenario.
    """
    terrain = scenario.terrain_fn()
    trials = scenario.trials
    n = len(trials)

    sim_config.num_worlds = n  # cross-checked against spawn_pose's row count in build_model
    ostrich_dt = sim_config.target_timestep_seconds
    hstack_dt = dynamics.DT

    spawn_pose = np.array(
        [[t.spawn_x, t.spawn_y, t.spawn_yaw] for t in trials], dtype=np.float64
    )
    ostrich_setpoints = np.stack(
        [build_setpoints(ostrich_dt, t.v_drive, t.wz_drive, scenario.duration_s) for t in trials],
        axis=1,
    )
    hstack_setpoints = np.stack(
        [build_setpoints(hstack_dt, t.v_drive, t.wz_drive, scenario.duration_s) for t in trials],
        axis=1,
    )

    print(
        f"  [{scenario.name}] {n} trial(s) x {ostrich_setpoints.shape[0]}/"
        f"{hstack_setpoints.shape[0]} steps (ostrich/hstack), grid {terrain.ny}x{terrain.nx}"
    )

    o_pose, o_wheel_qd = run_ostrich_batch(
        sim_config, render_config, engine_config, logging_config, terrain,
        ostrich_setpoints, mu, spawn_pose, SETTLE_STEPS,
    )
    h_controlled, h_derived, h_clearance, h_residual, h_turning, h_wheel_qd = run_hstack_batch(
        hstack_setpoints, terrain, hstack_dt, k_turn, mu, device, spawn_pose
    )

    # hstack reports (x, y, yaw) and (z, pitch, roll) separately; reassemble ostrich's 7-vector
    # pose form so both sims are scored by exactly the same code.
    h_quat = euler_zyx_to_quat_xyzw(h_controlled[..., 2], h_derived[..., 1], h_derived[..., 2])
    h_pose = np.concatenate(
        [h_controlled[..., :2], h_derived[..., :1], h_quat], axis=-1
    ).astype(np.float32)

    spawn_xy = spawn_pose[:, :2]
    metrics = compute_metrics(
        ostrich_pose=o_pose,
        ostrich_wheel_qd=o_wheel_qd,
        hstack_pose=h_pose,
        hstack_wheel_qd=h_wheel_qd,
        clearance=h_clearance,
        residual=h_residual,
        spawn_xy=spawn_xy,
    )
    flags = health_flags(
        ostrich_pose=o_pose,
        hstack_pose=h_pose,
        residual=h_residual,
        spawn_xy=spawn_xy,
        max_displacement=np.array(scenario.max_displacement()),
        bounds=plausible_bounds(terrain),
        ostrich_dt=ostrich_dt,
        hstack_dt=hstack_dt,
        v_drive=np.array([t.v_drive for t in trials]),
    )

    if write_h5:
        _write_scenario_h5(
            scenario, terrain, trials, spawn_pose, mu, k_turn,
            ostrich_dt, hstack_dt, ostrich_setpoints, hstack_setpoints,
            o_pose, o_wheel_qd, h_pose, h_controlled, h_derived, h_turning,
            h_clearance, h_residual, h_wheel_qd,
        )

    return ScenarioResult(
        name=scenario.name,
        trial_labels=np.array([t.label for t in trials]),
        metrics=metrics,
        flags=flags,
        traj_ostrich=_keyframes(o_pose),
        traj_hstack=_keyframes(h_pose),
    )


def _write_scenario_h5(
    scenario, terrain, trials, spawn_pose, mu, k_turn,
    ostrich_dt, hstack_dt, ostrich_setpoints, hstack_setpoints,
    o_pose, o_wheel_qd, h_pose, h_controlled, h_derived, h_turning,
    h_clearance, h_residual, h_wheel_qd,
) -> None:
    """Write the rollout in comparator/provenance.py's schema so the existing viewers and the GL
    replay work on it unchanged. terrain_path is None -- the terrain was built in memory, and
    terrain_fields embeds the grid itself either way, so only the archival path string is lost."""
    n = len(trials)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"submodule_test_{scenario.name}.h5"
    write_comparison(
        path,
        root=dict(
            n=n,
            variant_name="trial",
            obstacle_x=0.0,  # every scenario's terrain is origin-centered
            duration_s=float(scenario.duration_s),
            mu=mu,
            k_turn=k_turn,
            k_p=K_P,
        ),
        per_variant=dict(
            variant_value=np.arange(n, dtype=np.float32),
            variant_label=np.array([t.label for t in trials]),
            spawn_pose=spawn_pose.astype(np.float32),
            v_drive=np.array([t.v_drive for t in trials], dtype=np.float32),
            wz_drive=np.array([t.wz_drive for t in trials], dtype=np.float32),
        ),
        # The same HeightMapReader object n times -- terrain_fields dedups by identity, so the
        # grid is stored once and a variant_to_terrain index is written.
        terrain_entries=[(None, terrain)] * n,
        ostrich=dict(
            dt=ostrich_dt,
            t=np.arange(ostrich_setpoints.shape[0], dtype=np.float32) * ostrich_dt,
            cmd_wheel_omega=ostrich_setpoints.astype(np.float32),
            pose=o_pose,
            wheel_qd=o_wheel_qd,
        ),
        hstack=dict(
            dt=hstack_dt,
            t=np.arange(hstack_setpoints.shape[0], dtype=np.float32) * hstack_dt,
            cmd_wheel_omega=hstack_setpoints.astype(np.float32),
            pose=h_pose,
            controlled=h_controlled,
            derived=h_derived,
            turning=h_turning,
            clearance=h_clearance,
            residual=h_residual,
            wheel_qd=h_wheel_qd,
        ),
    )
    print(f"    wrote {path}")


# --- baseline I/O ------------------------------------------------------------------------------


def baseline_path(tier: str) -> pathlib.Path:
    return BASELINE_DIR / f"baseline_{tier}.npz"


def _median_and_spread(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[R, ...] repeated measurements -> (median, max - min). The median is the recorded value
    (robust to one outlying run in a way the mean is not); the spread is the measured noise every
    tolerance is then built from."""
    return np.median(stack, axis=0), np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)


def _pool_noise(spread: np.ndarray) -> np.ndarray:
    """Raise every cell's noise estimate to the worst seen for that metric anywhere in the
    scenario, by taking the max over every axis except the last (metric / pose component).

    A handful of repeats estimates a heavy tail badly -- the same metric measured 31.2, 18.8 and
    12.7 deg on three calibration passes -- so a per-cell estimate is as likely to be an
    unlucky-low draw as a representative one, and a low draw makes the baseline fail on the next
    run for no reason. Pooling trades sensitivity on the well-behaved trials for an estimate
    stable enough to be worth having. It also matches the physics: within one scenario, whether a
    given trial happens to be the chaotic one is not something to bet the tolerance on.
    """
    return np.broadcast_to(
        spread.max(axis=tuple(range(spread.ndim - 1)), keepdims=True), spread.shape
    ).copy()


def write_baseline(
    path: pathlib.Path, runs: list[list[ScenarioResult]], *, tier: str, device: str
) -> None:
    """np.savez_compressed with '<scenario>.<array>' keys -- the layout of
    helhest_stack/tests/engine/golden_fixture.npz, for the same reason: one self-describing file
    next to the checker that reads it.

    `runs` is every repeat of the recording pass. Each scenario stores the median across them plus
    the measured spread, so the file carries its own tolerances instead of relying on constants
    that would have to be re-tuned per machine (see the FLOOR_ATOL comment).
    """
    arrays: dict[str, np.ndarray] = {
        "metric_names": np.array(METRIC_NAMES),
        "tier": np.array(tier),
        "device": np.array(device),
        "warp_version": np.array(wp.config.version),
        "created": np.array(datetime.datetime.now().isoformat(timespec="seconds")),
        "repeats": np.array(len(runs)),
        "scenario_names": np.array([r.name for r in runs[0]]),
    }
    # Which submodule SHAs produced these numbers, so a stale baseline is self-identifying.
    # sha == "unknown" doubles as "not determined", and the paired *_dirty is then not a
    # verified-clean claim -- see git_provenance's docstring.
    for key, value in git_provenance().items():
        arrays[f"git.{key}"] = np.array(value)
    for s, r in enumerate(runs[0]):
        arrays[f"{r.name}.trial_labels"] = r.trial_labels
        for field in ("metrics", "traj_ostrich", "traj_hstack"):
            stack = np.stack([getattr(run[s], field) for run in runs])
            median, spread = _median_and_spread(stack)
            arrays[f"{r.name}.{field}"] = median
            arrays[f"{r.name}.{field}_noise"] = _pool_noise(spread)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _describe_baseline(ref: np.lib.npyio.NpzFile) -> str:
    def get(key: str) -> str:
        return str(ref[key]) if key in ref.files else "?"

    dirty = [n for n in ("ostrich", "helhest_stack") if get(f"git.{n}_dirty") == "True"]
    suffix = f", DIRTY: {'+'.join(dirty)}" if dirty else ""
    return (
        f"recorded {get('created')} on {get('device')} / warp {get('warp_version')}\n"
        f"    ostrich {get('git.ostrich_sha')[:12]}  "
        f"helhest_stack {get('git.helhest_stack_sha')[:12]}{suffix}"
    )


# --- comparison --------------------------------------------------------------------------------


def _tolerance(ref: np.ndarray, noise: np.ndarray, floor_atol: float) -> np.ndarray:
    """Elementwise allowance: a floor, plus headroom over the noise measured when the baseline was
    recorded, plus a relative term. See the FLOOR_ATOL comment for why the noise term carries most
    of the weight."""
    return floor_atol + NOISE_FACTOR * np.nan_to_num(noise) + REL_TOL * np.abs(ref)


def _within(current: np.ndarray, ref: np.ndarray, tol: np.ndarray) -> np.ndarray:
    """Elementwise |current - ref| <= tol. NaN on either side is never within tolerance --
    including NaN == NaN, since a run that reproduces a previous NaN is still a broken run, and
    the health flags are what should report it."""
    delta = np.abs(current - ref)
    return np.isfinite(delta) & (delta <= tol)


def compare(results: list[ScenarioResult], ref: np.lib.npyio.NpzFile) -> bool:
    """Print the per-scenario report and return whether everything matched. Metrics are
    aggregated to one row each (worst trial named) rather than one row per trial x metric --
    25 metrics x 9 trials of all-OK rows is noise that hides the one line that matters."""
    ok = True
    ref_names = [str(s) for s in ref["metric_names"]]

    for r in results:
        print(f"\n  --- {r.name} ---")
        key = f"{r.name}.metrics"
        if key not in ref.files:
            print("    MISSING from baseline -- regenerate with +update=true")
            ok = False
            continue

        ref_labels = [str(s) for s in ref[f"{r.name}.trial_labels"]]
        cur_labels = [str(s) for s in r.trial_labels]
        if ref_labels != cur_labels:
            print(f"    TRIALS CHANGED  baseline {ref_labels}\n                    current  {cur_labels}")
            ok = False
            continue

        ref_metrics = ref[key]
        if ref_metrics.shape != r.metrics.shape:
            print(f"    SHAPE {r.metrics.shape} != baseline {ref_metrics.shape}")
            ok = False
            continue

        noise_key = f"{r.name}.metrics_noise"
        # A baseline recorded before noise was measured (or with +repeat=1) has none; zeros fall
        # back to the floor tolerance, which is exactly the cries-wolf behaviour that motivates
        # +update defaulting to several repeats.
        metric_noise = ref[noise_key] if noise_key in ref.files else np.zeros_like(ref_metrics)
        changed, unchecked, moved = 0, [], []
        for j, name in enumerate(METRIC_NAMES):
            if name not in ref_names:
                print(f"    {name:24s} NEW -- not in baseline, not checked")
                continue
            k = ref_names.index(name)
            col_ref, col_cur = ref_metrics[:, k], r.metrics[:, j]
            floor = FLOOR_ATOL[METRIC_UNITS[name]]
            tol = _tolerance(col_ref, metric_noise[:, k], floor)
            # Noise-dominated means the run-to-run SCATTER, not the floor, sets the tolerance --
            # the metric cannot resolve a change smaller than its own jitter. hstack's noise is
            # exactly 0.0, so its floor-sized tolerance is a real check and it is never dominated.
            dominated = metric_noise[:, k] > NOISE_ASSERT_MULTIPLE * floor
            if dominated.any():
                i = int(np.argmax(np.where(dominated, tol, -np.inf)))
                unchecked.append((float(tol[i]), f"{name}[{cur_labels[i]}] +-{tol[i]:.3g}"))
            good = _within(col_cur, col_ref, tol)
            if good.all():
                continue

            def _worst(mask: np.ndarray) -> int:
                """Index of the largest deviation among the trials `mask` selects."""
                return int(np.nanargmax(np.where(mask, np.abs(col_cur - col_ref), -np.inf)))

            # A cell whose tolerance is set by ostrich's own scatter cannot support a verdict:
            # asserting on it makes the whole harness flaky (observed: o_max_pitch_deg on flat
            # turn_in_place moved 0.55 -> 3.44 deg between identical runs, four times the spread
            # five repeats had measured). Report those, keep them out of the pass/fail decision,
            # and judge the rest -- split per trial, because within one metric a reproducible
            # trial must still be able to fail even when a noisier sibling deviates further.
            drifted = ~good & dominated
            failed = ~good & ~dominated
            if drifted.any():
                i = _worst(drifted)
                moved.append(
                    f"{name}[{cur_labels[i]}] {col_ref[i]:+.4f} -> {col_cur[i]:+.4f} "
                    f"(d={col_cur[i] - col_ref[i]:+.4f}, tol +-{tol[i]:.3g})"
                )
            if failed.any():
                changed += 1
                ok = False
                i = _worst(failed)
                print(
                    f"    {name:24s} CHANGED on {int(failed.sum())}/{len(good)} trial(s), worst "
                    f"{cur_labels[i]}: {col_ref[i]:+.4f} -> {col_cur[i]:+.4f} "
                    f"(d={col_cur[i] - col_ref[i]:+.4f}, tol +-{tol[i]:.3g})"
                )
        if not changed:
            print(f"    {len(METRIC_NAMES)} metrics x {len(cur_labels)} trials  OK")
        if moved:
            print(f"    {len(moved)} noise-dominated metric(s) moved (reported, not asserted):")
            for entry in moved:
                print(f"      {entry}")
        if unchecked:
            worst = sorted(unchecked, reverse=True)[:3]
            print(
                f"    {len(unchecked)}/{len(METRIC_NAMES)} metrics not asserted (ostrich scatter "
                f"exceeds the floor, so they cannot support a verdict); widest:"
            )
            for _, entry in worst:
                print(f"      {entry}")

        for which, cur in (("ostrich", r.traj_ostrich), ("hstack", r.traj_hstack)):
            ref_traj = ref[f"{r.name}.traj_{which}"]
            if ref_traj.shape != cur.shape:
                print(f"    traj_{which:8s} SHAPE {cur.shape} != baseline {ref_traj.shape}")
                ok = False
                continue
            nk = f"{r.name}.traj_{which}_noise"
            traj_noise = ref[nk] if nk in ref.files else np.zeros_like(ref_traj)
            tol = _tolerance(ref_traj, traj_noise, TRAJ_FLOOR_ATOL)
            # Same rule as the metrics above: a keyframe whose tolerance is set by ostrich's own
            # scatter is reported, never asserted on.
            dominated = traj_noise > NOISE_ASSERT_MULTIPLE * TRAJ_FLOOR_ATOL
            good = _within(cur, ref_traj, tol) | dominated
            if good.all():
                continue
            ok = False
            # Which keyframe first drifted says WHEN the runs parted, which the final pose can't.
            bad_k = np.nonzero(~good.all(axis=(0, 2)))[0]
            worst = np.unravel_index(
                int(np.nanargmax(np.where(good, -np.inf, np.abs(cur - ref_traj)))), cur.shape
            )
            print(
                f"    traj_{which:8s} CHANGED from keyframe {bad_k[0]}/{cur.shape[1]}, worst "
                f"{cur_labels[worst[0]]} k={worst[1]} component {worst[2]}: "
                f"{ref_traj[worst]:+.4f} -> {cur[worst]:+.4f} (tol +-{tol[worst]:.3g})"
            )
    return ok


def report_flags(runs: list[list[ScenarioResult]]) -> bool:
    """Print every health flag that fired in ANY repeat, and return whether the rollouts are
    usable at all. Checked BEFORE the baseline comparison, because metrics derived from a NaN
    rollout say nothing about whether the physics changed.

    Aggregated across repeats rather than reported per run: a flag that fires in one repeat out of
    three is still a flag -- ostrich's nondeterminism means an unstable trial can pass and fail
    the same check on consecutive runs, and that is exactly the case worth surfacing."""
    fired: dict[tuple[str, str, str], int] = {}
    for run in runs:
        for r in run:
            for label, flag in r.flagged():
                fired[(r.name, label, flag)] = fired.get((r.name, label, flag), 0) + 1
    n = sum(len(r.trial_labels) for r in runs[0])
    if not fired:
        print(f"\n  health: {n} trial(s) x {len(runs)} run(s), no flags")
        return True
    print(f"\n  health: {len(fired)} FLAG(S) -- these fail regardless of the baseline")
    for (scenario, label, flag), count in fired.items():
        seen = f"  ({count}/{len(runs)} runs)" if len(runs) > 1 else ""
        print(f"    {scenario}/{label:22s} {flag}{seen}")
    return False


def report_spread(runs: list[list[ScenarioResult]]) -> None:
    """Per-metric spread across repeated identical runs -- the noise floor the tolerances have to
    clear. Anything above zero here is nondeterminism (ostrich's Newton solve and contact ordering
    under CUDA graphs), and a tolerance below it will cry wolf on every run."""
    print(f"\n=== spread over {len(runs)} identical runs (max |max - min| per metric) ===")
    print(f"  {'metric':24s} {'unit':6s} {'max spread':>12s}  worst scenario/trial")
    for j, name in enumerate(METRIC_NAMES):
        worst_val, worst_where = 0.0, "-"
        for s, _ in enumerate(runs[0]):
            stack = np.stack([run[s].metrics[:, j] for run in runs])  # [R, n_trials]
            spread = np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)
            i = int(np.nanargmax(spread))
            if spread[i] > worst_val:
                worst_val = float(spread[i])
                worst_where = f"{runs[0][s].name}/{runs[0][s].trial_labels[i]}"
        print(f"  {name:24s} {METRIC_UNITS[name]:6s} {worst_val:12.3e}  {worst_where}")

    traj = 0.0
    for s, _ in enumerate(runs[0]):
        for which in ("traj_ostrich", "traj_hstack"):
            stack = np.stack([getattr(run[s], which) for run in runs])
            traj = max(traj, float(np.nanmax(np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0))))
    print(f"  {'(trajectory keyframes)':24s} {'m/q':6s} {traj:12.3e}")
    print(
        "\nA spread of exactly 0.0 means that metric is bit-reproducible and is asserted at the\n"
        "floor tolerance. Non-zero is ostrich's intrinsic scatter: pooled per scenario and scaled\n"
        f"by NOISE_FACTOR={NOISE_FACTOR:g}, it becomes that metric's tolerance when +update writes\n"
        "the baseline, and the widest such cells are reported rather than asserted on."
    )


@hydra.main(config_path=str(CONFIG_PATH), config_name="helhest", version_base=None)
def main(cfg: DictConfig) -> None:
    device = str(cfg.get("device", "cuda:0"))
    init_warp_device(device)

    tier = str(cfg.get("tier", "smoke"))
    update = bool(cfg.get("update", False))
    # Recording a baseline inherently means measuring its noise -- ostrich's run-to-run scatter is
    # what every tolerance is built from, so a one-shot +update would produce a baseline that
    # fails on the very next run.
    repeat = int(cfg.get("repeat", DEFAULT_UPDATE_REPEATS if update else 1))
    write_h5 = not bool(cfg.get("no_h5", False))
    mu = float(cfg.get("mu", 0.8))
    k_turn = float(cfg.get("k_turn", dynamics.K_TURN))

    names = cfg.get("scenarios", None)
    scenarios = select_scenarios(scenario_set(tier), list(names) if names is not None else None)

    sim_config: SimulationConfig = hydra.utils.instantiate(cfg.simulation)
    render_config: RenderingConfig = hydra.utils.instantiate(cfg.rendering)
    engine_config: EngineConfig = hydra.utils.instantiate(cfg.engine)
    logging_config: LoggingConfig = hydra.utils.instantiate(cfg.logging)
    render_config.vis_type = "null"  # headless

    print(
        f"=== submodule_test tier={tier} scenarios={[s.name for s in scenarios]} "
        f"device={device} mu={mu} k_turn={k_turn} ==="
    )

    runs: list[list[ScenarioResult]] = []
    for rep in range(repeat):
        if repeat > 1:
            print(f"\n--- run {rep + 1}/{repeat} ---")
        runs.append(
            [
                simulate_scenario(
                    sc,
                    sim_config=sim_config,
                    render_config=render_config,
                    engine_config=engine_config,
                    logging_config=logging_config,
                    mu=mu,
                    k_turn=k_turn,
                    device=device,
                    # Only the first run's replay files are worth keeping; later ones would just
                    # overwrite them with the same thing.
                    write_h5=write_h5 and rep == 0,
                )
                for sc in scenarios
            ]
        )
    results = runs[0]

    healthy = report_flags(runs)
    path = baseline_path(tier)

    if update:
        if not healthy:
            # Refusing here is the whole point: the moment a baseline can absorb a broken run,
            # it starts tracking the bug instead of the physics.
            raise SystemExit("refusing to update the baseline: health flags fired (see above)")
        report_spread(runs)
        write_baseline(path, runs, tier=tier, device=device)
        print(f"\nwrote baseline {path} ({len(runs)} repeat(s), noise measured)")
        print("Commit it naming the submodule SHAs it was recorded at.")
        return

    if repeat > 1:
        report_spread(runs)
        assert healthy, "health flags fired -- see above"
        return

    if not path.exists():
        raise SystemExit(
            f"no baseline at {path} -- record one with:\n"
            f"  python src/feasibility/submodule_test/run_check.py +tier={tier} +update=true"
        )

    with np.load(path, allow_pickle=False) as ref:
        print(f"\n  baseline: {_describe_baseline(ref)}")
        matched = compare(results, ref)

    verdict = "OK" if (matched and healthy) else "REVIEW -- see above"
    print(f"\nsubmodule_test {tier}: {verdict}")
    if not healthy:
        raise SystemExit("health flags fired: at least one rollout is not physically usable")
    assert matched, (
        "simulator behaviour differs from the baseline. If the change is expected (a submodule "
        f"bump you intend to accept), re-record with +tier={tier} +update=true and commit it."
    )


if __name__ == "__main__":
    main()
