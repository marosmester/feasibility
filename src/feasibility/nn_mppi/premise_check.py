"""Does a `heightmap/mppi_eval` map family hurt VANILLA MPPI in ostrich? The check to run before any
nn-MPPI data is made for a family.

Every instance of a family is a pair, `<stem>` (with its low features) and `<stem>_ctrl` (without).
For each pair this runs `closed_loop.py --no-nn` on both maps (one subprocess per run, so each gets
a fresh CUDA context; a run whose h5 already exists is reused unless --rerun), then scores the two
ostrich logs against the features the sidecar lists:

    status, time     closed_loop's verdict (arrived / timeout / flipped / off_map / ...) and when
    contact s        time with a wheel CENTRE within wheel_radius of a curb's footprint, i.e. the
                     tyre on or against it (the settle's spherical envelope reaching it)
    stuck s          time the body neither translates (< 0.05 m/s) nor turns (< 0.1 rad/s)
    wall m           closest wheel-centre approach to a tall wall, minus wheel_radius (< 0: touching)
    |roll| |pitch|   maxima over the run
    turn x           turn_spot only: x where the heading first passes 135 deg off the start's, i.e.
                     where the robot turned round (at the start ~0, in the bay > curb_fwd)
    turn d           simple_pivot_trap only: how far the robot has moved from the start when its
                     heading first passes half the goal bearing -- ~0 for a turn in place (the
                     "red" manoeuvre), > 0.5 m for driving forward first (the "green" one)

A with-map whose sidecar names a `ctrl` map is paired with that one (simple_pivot_trap's sweep
shares one ctrl map per bearing, run once); otherwise the ctrl map is `<stem>_ctrl`.

An instance PASSES -- the family hurts vanilla there -- when the ctrl run arrived, the with run
touched a curb, and the with run did worse beyond ostrich's run-to-run noise: it did not arrive, or
it was stuck > STUCK_MARGIN s longer, or it arrived > TIME_MARGIN s later. One table per family is
printed and written to `outputs/nn_mppi/premise_<family>.yaml`.

Run it with the GPU torch env on the GTX 1050 machine (`.venv-cu126`), as closed_loop.py itself.

CLI parameters:
    --maps DIR          a family's map dir (pairs found by their `_ctrl` sidecars)
    --checkpoint PATH   passed to closed_loop.py (its k_turn and yaw gain; the nn arm is not run)
    --batch INT         MPPI rollouts (default 512)
    --n-refine INT      refines per replan (default 1)
    --max-time S        per-run time limit (default 40)
    --rerun             re-run maps whose h5 already exists
    --score-only        only score existing h5 files

Usage:
    python src/feasibility/nn_mppi/premise_check.py --maps assets/mppi_eval/turn_spot \\
        --checkpoint outputs/checkpoints/dataset_mppi_default_maps0_centered_M200_R10_seed0.pt
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

import h5py
import numpy as np
import yaml
from helhest import dynamics

from feasibility.heightmap.create_garage import wheels_of
from feasibility.heightmap.mppi_eval.common import distance_field
from feasibility.heightmap.mppi_eval.common import sample_field
from feasibility.mppi_learning.generate_dataset import quat_to_yaw
from feasibility.nn_mppi.closed_loop import OUT
from feasibility.nn_mppi.closed_loop import pitch_roll
from feasibility.nn_mppi.closed_loop import SETTLE_STEPS
from feasibility.nn_mppi.closed_loop import STEPS_PER_FRAME

TAG = "premise"
STUCK_V = 0.05  # [m/s]
STUCK_WZ = 0.1  # [rad/s]
STUCK_MARGIN = 1.0  # [s] more stuck time than ctrl that counts as harm
TIME_MARGIN = 2.0  # [s] later arrival that counts as harm (vanilla's own arrival varies ~ +-0.5 s)
TURNED = np.radians(135.0)
CLOSED_LOOP = pathlib.Path(__file__).with_name("closed_loop.py")


def run_map(stem: pathlib.Path, args: argparse.Namespace) -> pathlib.Path:
    h5 = OUT / f"{stem.name}_{TAG}.h5"
    if h5.exists() and not args.rerun:
        return h5
    cmd = [sys.executable, "-u", str(CLOSED_LOOP), "--checkpoint", str(args.checkpoint), "--map", str(stem),
           "--no-nn", "--batch", str(args.batch), "--n-refine", str(args.n_refine), "--max-time", str(args.max_time),
           "--tag", TAG]
    print(f"[run] {stem.name}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not h5.exists():
        print(result.stdout[-3000:], result.stderr[-3000:])
        raise SystemExit(f"closed_loop.py failed on {stem}")
    return h5


def score(h5: pathlib.Path, meta: dict) -> dict:
    r = float(dynamics.robot_params().wheel_radius)
    with h5py.File(h5, "r") as f:
        status, end_frame = str(f["status"][0].decode() if isinstance(f["status"][0], bytes) else f["status"][0]), int(f["end_frame"][0])
        pose = f["ostrich_pose"][:, 0]
        shape, (x0, y0), cell = f["terrain"].shape, f["terrain"].attrs["origin"], float(f["terrain"].attrs["cell"])
        dt = float(f.attrs["ostrich_dt"])
    extent = (shape[0] - 1) * cell
    rows = pose[SETTLE_STEPS : SETTLE_STEPS + end_frame * STEPS_PER_FRAME + 1]
    xy, yaw = rows[:, :2], quat_to_yaw(rows[:, 3:7])
    wheels = wheels_of(np.concatenate([xy, yaw[:, None]], -1))  # [T, 3, 2]
    # the CURB footprint is rasterised from the with map's rectangles for both runs, so the ctrl run
    # reports how often vanilla drove where the curb would have been
    d_curb = distance_field(meta["curbs"], meta["curb_height"], extent, cell)
    d_wall = distance_field(meta["walls"], meta["wall_height"], extent, cell)
    contact = (sample_field(d_curb, x0, y0, cell, wheels).min(axis=1) < r).sum() * dt
    wall = float(sample_field(d_wall, x0, y0, cell, wheels).min()) - r
    frames = rows[::STEPS_PER_FRAME]  # one per 0.1 s replan
    fdt = STEPS_PER_FRAME * dt
    v = np.linalg.norm(np.diff(frames[:, :2], axis=0), axis=1) / fdt
    fyaw = np.unwrap(quat_to_yaw(frames[:, 3:7]))
    wz = np.abs(np.diff(fyaw)) / fdt
    stuck = float(((v < STUCK_V) & (wz < STUCK_WZ)).sum() * fdt)
    pitch, roll = pitch_roll(rows[:, 3:7])
    out = dict(status=status, time=end_frame * fdt, contact_s=float(contact), stuck_s=stuck, wall_m=wall,
               max_roll_deg=float(np.degrees(np.abs(roll).max())), max_pitch_deg=float(np.degrees(np.abs(pitch).max())))
    if meta["family"] == "turn_spot":
        turned = np.flatnonzero(np.abs(fyaw - fyaw[0]) > TURNED)
        out["turn_x"] = float(frames[turned[0], 0]) if len(turned) else None
    if meta["family"] == "simple_pivot_trap":
        turned = np.flatnonzero(fyaw - fyaw[0] > np.radians(meta["bearing_deg"]) / 2.0)
        out["turn_d"] = float(np.linalg.norm(frames[turned[0], :2] - frames[0, :2])) if len(turned) else None
    return out


def verdict(w: dict, c: dict) -> tuple[bool, str]:
    if c["status"] != "arrived":
        return False, f"ctrl {c['status']}"
    if w["contact_s"] <= 0.0:
        return False, "no curb contact"
    if w["status"] != "arrived":
        return True, w["status"]
    if w["stuck_s"] - c["stuck_s"] > STUCK_MARGIN:
        return True, f"stuck +{w['stuck_s'] - c['stuck_s']:.1f} s"
    if w["time"] - c["time"] > TIME_MARGIN:
        return True, f"+{w['time'] - c['time']:.1f} s"
    return False, "no harm"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--maps", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--n-refine", type=int, default=1)
    parser.add_argument("--max-time", type=float, default=40.0)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    pairs = []
    for path in sorted(args.maps.glob("*.yaml")):
        meta = yaml.safe_load(path.read_text())
        if isinstance(meta, dict) and meta.get("variant") == "with":
            pairs.append((path.stem, meta.get("ctrl", f"{path.stem}_ctrl"), meta))
    if not pairs:
        raise SystemExit(f"{args.maps}: no with-map sidecars")
    results, family = [], None
    for stem, ctrl, meta in pairs:
        family = meta["family"]
        scored = {}
        for variant, name in (("with", stem), ("ctrl", ctrl)):
            h5 = OUT / f"{name}_{TAG}.h5" if args.score_only else run_map(args.maps / name, args)
            scored[variant] = score(h5, meta)
        ok, why = verdict(scored["with"], scored["ctrl"])
        results.append(dict(stem=stem, passed=ok, why=why, **scored))
        w, c = scored["with"], scored["ctrl"]
        print(f"[{stem}] {'PASS' if ok else 'fail'} ({why}): with {w['status']} {w['time']:.1f} s, contact "
              f"{w['contact_s']:.1f} s, stuck {w['stuck_s']:.1f} s | ctrl {c['status']} {c['time']:.1f} s, "
              f"stuck {c['stuck_s']:.1f} s", flush=True)

    keys = ["status", "time", "contact_s", "stuck_s", "wall_m", "max_roll_deg", "max_pitch_deg"]
    keys += ["turn_x"] if family == "turn_spot" else ["turn_d"] if family == "simple_pivot_trap" else []
    print(f"\n{family}: vanilla, with curbs | ctrl (no curbs)")
    print(f"{'instance':>18} {'verdict':>22} " + " ".join(f"{k:>16}" for k in keys))
    fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)
    for row in results:
        cells = [f"{fmt(row['with'][k])} | {fmt(row['ctrl'][k])}" for k in keys]
        print(f"{row['stem']:>18} {('PASS ' if row['passed'] else 'fail ') + row['why']:>22} "
              + " ".join(f"{c:>16}" for c in cells))
    n_pass = sum(r["passed"] for r in results)
    print(f"\n{n_pass}/{len(results)} instances hurt vanilla")
    path = OUT / f"premise_{family}.yaml"
    path.write_text(yaml.safe_dump(dict(family=family, maps=str(args.maps), batch=args.batch, n_refine=args.n_refine,
                                        n_pass=n_pass, instances=results), sort_keys=False))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
