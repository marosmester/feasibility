"""Plan TWO paths with TWO planners on one heightmap, then drive them SIMULTANEOUSLY in one ostrich
build -- two Helhest Juniors spawned at the identical start pose, each following its own planner's
path, so the planning difference is watched live rather than read off two separate runs.

Same physics/controller for both (one --v, one --lookahead, one --mu): only the planned path differs,
which is the point. Robot A is RED and robot B is GREEN -- each robot's own mesh and its own path
overlay wear its colour (ROBOT_COLORS, ColoredPursuitSimulator), so no legend is needed to see who
followed what. At the spawn the two meshes interpenetrate, so the colour is also the only thing
telling them apart until the routes diverge. The viewer is the point of this demo and runs by
DEFAULT; --no-view is the headless form, for a verdict table without a display (it is also the only
way to use --repeats, since a viewer showing six robots on two paths shows nothing a pair does not).

Spawning two robots on top of each other needs no special handling -- ostrich
builds every world via `finalize_replicated`, and replicated worlds only ever collide with the
GLOBAL shapes (the terrain), never with each other (comparator/common.py), so the two robots simply
pass through one another at the start and separate onto their own routes.

What the viewer DOES need: its per-world display offsets zeroed. Newton spreads worlds apart to be
looked at side by side (ostrich's InteractiveSimulator sets a 20 m spacing, centred), which at two
worlds renders the robots 10 m either side of where they physically are while the terrain and the
path overlay stay put -- one robot off the map, the other apparently driving past the ramp. Single-
world viewer demos never see it (one world's offset is 0). See the `set_world_offsets` call below.

Plans are CACHED to <out-dir>/plan_cache (keyed on the map's bytes, the endpoints, the planner, its
thresholds and its checkpoints' mtimes; --replan forces a recompute). A gated arm costs one network
inference over every lattice pose -- 360k poses on the ramp_detour map, ~137 s on a GTX 1650, 247 s
on CPU -- and watching the same comparison again should not pay it twice. A full cache hit skips the
CostToGo build as well, so a re-run goes straight to the ostrich rollout.

    1. load a heightmap (any PNG+YAML; start/goal from its sidecar or --start/--goal)
    2. plan start -> goal TWICE, with --planner-a and --planner-b, sharing one PlanContext
       (feasibility.planning.planners.plan_path; same planner names as ostrich_follow_path.py)
    3. drive both paths in one ostrich build: 2 * --repeats worlds (repeats per planner), worlds
       0..repeats-1 on planner A's path, repeats..2*repeats-1 on planner B's, via
       feasibility.planning.pure_pursuit.PurePursuitSimulator (ragged per-world paths, each world
       stops independently once it reaches its own goal)
    4. report per planner, per repeat: arrived / flipped / off_map / stalled / nonfinite, peak climb
       pitch and |roll|, cross-track error to that planner's own path

See demos/ostrich_follow_path.py for the single-planner version this generalizes, and
demos/ostrich_follow_path_parallel_worlds.py for the N-worlds-of-one-build precedent this reuses.

Output: <out-dir>/<map>_<planner_a>_vs_<planner_b>.h5, comparator/provenance's schema, world axis
0..repeats-1 = planner A, repeats..2*repeats-1 = planner B; both planned paths saved as
`planned_path_a` / `planned_path_b`; replay one world K with
    python src/feasibility/replay/gl_replay.py --file <out-dir>/<...>.h5 --id K --which ostrich
There is no plot: what this demo has to show is two robots moving at once, which a still cannot
carry -- ostrich_follow_path.py's `plot_run` is there for a bird's-eye of one planner's run.

Run (imports demos.ostrich_follow_path, so -- unlike that script -- this one needs `-m`; see
CLAUDE.md's `demos/` sys.path note):
    python -m demos.ostrich_follow_path_dual \\
        --map assets/ramp_detour/ramp_detour_s0650_g0200_a040 \\
        --planner-a vanilla-off --planner-b nn-gated-pos
    # vanilla-off climbs the 65 deg face and flips; nn-gated-pos detours by the 20 deg ramp --
    # both robots start together and visibly part ways
    python -m demos.ostrich_follow_path_dual \\
        --map assets/uphill_series/uphill_a0650 --planner-a vanilla-on --planner-b nn-gated-pitch \\
        --tau-pitch 0.2 --no-view --repeats 5

CLI parameters:
    --map PATH            heightmap stem (PNG + YAML), required
    --planner-a NAME      planner for robot A (default vanilla-off)
    --planner-b NAME      planner for robot B (default nn-gated-pos)
    --start X Y YAW       start pose (default: the sidecar's `start`), shared by both robots
    --goal X Y            goal (default: the sidecar's `goal`), shared by both robots
    --start-side SIDE     left | center | right: shift the start sideways (default center)
    --goal-side SIDE      same for the goal (see ostrich_follow_path.py for the exact shift)
    <planner thresholds / networks / controller / physics flags: see add_shared_args in
     ostrich_follow_path.py -- --tau-pos/--tau-pitch/--tau-rot/--tau-fused-pos/--tau-fused-rot,
     --checkpoint/--checkpoint-rot, --torch-device, --chunk, --pivot-cost, --v, --lookahead, --mu,
     --slack, --settle-steps, --override>
    --repeats INT         worlds PER planner (default 3, and only with --no-view; the viewer always
                          drives one pair) -- total worlds = 2*repeats
    --replan              recompute the plans instead of reusing <out-dir>/plan_cache (see below)
    --no-view             headless: no GL window, just the verdict table and the .h5. Without it the
                          viewer opens PAUSED -- SPACE runs/pauses, "." steps once while paused
    --out-dir PATH        output directory (default: outputs/follow_path_dual)
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import pathlib
import time

import numpy as np
import warp as wp
import yaml

from demos.ostrich_follow_path import add_shared_args
from demos.ostrich_follow_path import corner_camera
from demos.ostrich_follow_path import REPO_ROOT
from demos.ostrich_follow_path import shift_sideways
from demos.ostrich_follow_path import SIDES
from feasibility.comparator.common import friction_kwargs
from feasibility.comparator.common import K_P
from feasibility.comparator.provenance import write_comparison
from feasibility.heightmap import HeightMapReader
from feasibility.lattice_learning.generate_dataset import compose_ostrich_config
from feasibility.planning.evaluation import judge
from feasibility.planning.planners import plan_path
from feasibility.planning.planners import PlanContext
from feasibility.planning.planners import PlannerConfig
from feasibility.planning.planners import PLANNERS
from feasibility.planning.pure_pursuit import controller_kappa_max
from feasibility.planning.pure_pursuit import PurePursuitSimulator
from feasibility.planning.pure_pursuit import resample_polyline

# One colour per robot, worn by its CHASSIS+WHEELS and by its path overlay -- so "the red robot is
# driving the red line" needs no legend. matplotlib tab10 values (the palette nn_mppi's arm_colors
# hands its arms), named so the console can say which robot is which. Both stand off Newton's
# default shape colour, which the GLOBAL terrain mesh keeps -- a blue-grey (0.27, 0.47, 0.67) that a
# blue or cyan robot sinks into.
ROBOT_COLORS = {"a": (0.8392, 0.1529, 0.1569), "b": (0.1725, 0.6275, 0.1725)}
ROBOT_COLOR_NAMES = {"a": "red", "b": "green"}

# Everything about the PLANNER that changes a plan. The thresholds are all of them, not just the
# one head in use, because which head a planner reads is its own business -- keying on the lot
# costs nothing and cannot under-key.
PLAN_CACHE_TAUS = ("tau_pos", "tau_pitch", "tau_rot", "tau_fused_pos", "tau_fused_rot",
                   "tau_pivot_pitch", "pivot_cost")
PLAN_CACHE_KEYS = ("reached", "reachable", "v_start", "path_m", "n_settle_bad", "gate", "head",
                   "tau", "max_err")


# --- plan cache -----------------------------------------------------------------------------------
# A gated plan costs one network inference over every lattice pose -- 360k poses on a 13 m map,
# ~137 s on a GTX 1650 and over 4 minutes on CPU (arc_network's row-invariance shortcut cannot fire
# on a map with a detour beside the face). It depends only on the map, the endpoints, the planner
# and its checkpoint, so re-running the same comparison to watch it again pays that for nothing.
# The result is two short polylines, so cache them on disk and key on everything that moves them --
# checkpoint MTIME included, since a retrain moves predictions and so moves the gate.


def plan_cache_key(
    map_path: pathlib.Path,
    start: tuple[float, float, float],
    goal: tuple[float, float],
    planner: str,
    config: PlannerConfig,
) -> str:
    png = map_path.with_suffix(".png")
    stat = png.stat()
    parts = [str(map_path), str(stat.st_mtime_ns), str(stat.st_size), repr(start), repr(goal), planner]
    parts += [f"{name}={getattr(config, name, None)!r}" for name in PLAN_CACHE_TAUS]
    for attr in ("checkpoint", "checkpoint_rot", "checkpoint_vwz"):
        ckpt = pathlib.Path(getattr(config, attr, "") or "")
        stamp = str(ckpt.stat().st_mtime_ns) if ckpt.exists() else "missing"
        parts.append(f"{attr}={ckpt}:{stamp}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def load_plan(cache_dir: pathlib.Path, key: str) -> dict | None:
    path = cache_dir / f"{key}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    plan = json.loads(str(data["meta"]))
    plan["xy"] = data["xy"]
    return plan


def save_plan(cache_dir: pathlib.Path, key: str, plan: dict) -> None:
    meta = {k: plan[k] for k in PLAN_CACHE_KEYS}
    meta["max_err"] = {k: float(v) for k, v in meta["max_err"].items()}
    for k in ("v_start", "path_m", "tau"):
        meta[k] = float(meta[k])  # json chokes on np.float32; inf round-trips as Infinity
    meta["n_settle_bad"] = int(meta["n_settle_bad"])
    for k in ("reached", "reachable"):
        meta[k] = bool(meta[k])
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache_dir / f"{key}.npz", xy=np.asarray(plan["xy"], np.float64), meta=json.dumps(meta))


class ColoredPursuitSimulator(PurePursuitSimulator):
    """PurePursuitSimulator that paints world w -- its robot and its planned-path overlay -- in
    `world_colors[w]`, instead of one shape colour for every robot and one red for every path.

    Two robots at the same start pose are otherwise told apart only once their routes diverge (and
    at the spawn their meshes interpenetrate, so even then it is the colour that says which is
    which). Done here rather than in `planning/pure_pursuit.py` because every other caller drives
    ONE path and has nothing to distinguish."""

    def __init__(self, *args, world_colors: list[tuple[float, float, float]], **kwargs) -> None:
        # Before super().__init__, which calls build_model() below.
        self._world_colors = [tuple(float(c) for c in rgb) for rgb in world_colors]
        self._line_colors: wp.array | None = None
        super().__init__(*args, **kwargs)
        assert len(self._world_colors) == self.simulation_config.num_worlds

    def build_model(self):
        """Tint each world's shapes. Has to happen HERE, not after construction: ostrich's
        InteractiveSimulator calls viewer.set_model(self.model) as soon as this returns, and
        set_model reads model.shape_color then to build its instance batches."""
        model = super().build_model()
        world = model.shape_world.numpy()  # -1 on the global terrain shapes, which keep their colour
        color = model.shape_color.numpy()
        for w, rgb in enumerate(self._world_colors):
            color[world == w] = rgb
        model.shape_color.assign(color)
        return model

    def _maybe_render(self, step_idx: int) -> None:
        if self.viewer is None:
            return
        if self._line_colors is None:
            # one colour per SEGMENT, matching rollout's concatenation of the per-world polylines
            per_world = [np.tile(rgb, (len(p) - 1, 1)) for rgb, p in zip(self._world_colors, self.paths)]
            self._line_colors = wp.array(np.concatenate(per_world), dtype=wp.vec3, device=self.model.device)
        self.viewer.begin_frame(step_idx * self.clock.dt)
        self.viewer.log_state(self.current_state)
        self.viewer.log_lines("planned_path", self._path_starts, self._path_ends, self._line_colors)
        self.viewer.end_frame()


def print_table(planner: str, gate: str, path_len: float, results: list[dict]) -> None:
    print(f"\n--- {planner}{gate} ---")
    print(f"{'id':>3} {'status':>9} {'t_arr':>6} {'climb':>6} {'roll':>5} {'cte_max':>7} "
          f"{'cte_mean':>8} {'cte@climb':>9} {'progress':>12}")
    for w, r in enumerate(results):
        print(f"{w:3d} {r['status']:>9} {r['t_arrive']:6.1f} {r['climb_deg']:6.1f} {r['roll_deg']:5.1f} "
              f"{r['cte_max']:7.3f} {r['cte_mean']:8.3f} {r['cte_at_climb']:9.3f} "
              f"{r['progress_m']:5.2f}/{path_len:<5.2f}")
    n_ok = sum(r["status"] == "arrived" for r in results)
    print(f"{n_ok}/{len(results)} arrived  ({planner}{gate})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--map", type=pathlib.Path, required=True)
    parser.add_argument("--planner-a", choices=list(PLANNERS), default="vanilla-off")
    parser.add_argument("--planner-b", choices=list(PLANNERS), default="nn-gated-pos")
    parser.add_argument("--start", type=float, nargs=3, metavar=("X", "Y", "YAW"))
    parser.add_argument("--goal", type=float, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--start-side", choices=SIDES, default="center")
    parser.add_argument("--goal-side", choices=SIDES, default="center")
    add_shared_args(parser)
    # A gated arm needs a field over EVERY lattice pose, so CPU inference dominates the run
    # (measured on the ramp_detour map: 247 s CPU vs 137 s cuda). Same reasoning, same default as
    # demos/ostrich_follow_plan_turning.py. Pass --torch-device cpu where torch has no usable GPU.
    parser.set_defaults(torch_device="cuda")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--replan", action="store_true",
                        help="ignore the cached plans and recompute them (they are re-cached)")
    # The viewer is what this demo is for, so it is the default; --no-view is the headless escape
    # (and the only way --repeats does anything -- see `repeats` below).
    parser.add_argument("--no-view", dest="view", action="store_false",
                        help="headless: verdict table and .h5 only, no GL window")
    parser.add_argument("--out-dir", type=pathlib.Path, default=REPO_ROOT / "outputs" / "follow_path_dual")
    args = parser.parse_args()

    map_path = args.map.with_suffix("")
    meta = yaml.safe_load(map_path.with_suffix(".yaml").read_text())
    start = tuple(args.start) if args.start is not None else meta.get("start")
    goal = tuple(args.goal) if args.goal is not None else meta.get("goal")
    if start is None or goal is None:
        raise SystemExit(f"{map_path.name}.yaml has no start/goal -- pass --start X Y YAW --goal X Y")
    if (args.start is not None and args.start_side != "center") or (
        args.goal is not None and args.goal_side != "center"
    ):
        raise SystemExit("--start-side/--goal-side shift the sidecar's start/goal; drop --start/--goal")
    start = tuple(float(v) for v in start)
    goal = (float(goal[0]), float(goal[1]))
    terrain = HeightMapReader.load(map_path)
    start, goal = shift_sideways(start, goal, args.start_side, args.goal_side, terrain)
    repeats = 1 if args.view else args.repeats  # one pair on screen; repeats are for the table
    sides = "" if args.start_side == args.goal_side == "center" else f"_s{args.start_side[0]}_g{args.goal_side[0]}"
    run_name = f"{map_path.name}{sides}_{args.planner_a}_vs_{args.planner_b}"

    # --- 1. plan both paths, sharing one PlanContext (same pivot_cost for both) -----------------
    print(f"=== {map_path.name}: planning {args.planner_a} vs {args.planner_b}, "
          f"start {start} -> goal {goal} ===")
    config = PlannerConfig.from_args(args)
    cache_dir = args.out_dir / "plan_cache"
    names = [args.planner_a, args.planner_b]
    keys = [plan_cache_key(map_path, start, goal, name, config) for name in names]
    plans = [None if args.replan else load_plan(cache_dir, key) for key in keys]
    for name, plan in zip(names, plans):
        if plan is not None:
            print(f"{name}: plan from cache (--replan to recompute)")
    if any(plan is None for plan in plans):
        # Only now is the lattice worth building -- a full cache hit skips the CostToGo too.
        ctx = PlanContext(config)
        for i, (name, key) in enumerate(zip(names, keys)):
            if plans[i] is None:
                plans[i] = plan_path(terrain, start, goal, name, ctx)
                save_plan(cache_dir, key, plans[i])
        del ctx
        gc.collect()  # CostToGo / gated solver buffers go before the ostrich build (3 GiB GPU)
    plan_a, plan_b = plans

    failed = [name for name, plan in ((args.planner_a, plan_a), (args.planner_b, plan_b)) if not plan["reached"]]
    if failed:
        for name, plan in ((args.planner_a, plan_a), (args.planner_b, plan_b)):
            why = "no feasible path" if not plan["reachable"] else "V* finite but the policy did not arrive"
            status = "OK" if plan["reached"] else f"FAILED -- {why} (V* = {plan['v_start']:.2f})"
            print(f"{name}: {status}")
        raise SystemExit(f"{', '.join(failed)} found no route -- nothing to simulate")

    for name, plan in ((args.planner_a, plan_a), (args.planner_b, plan_b)):
        gate = f", gate {plan['gate']}" if plan["gate"] else ""
        max_err = "  ".join(f"max predicted {k} {v:.4f}" for k, v in plan["max_err"].items())
        print(f"{name}{gate}: {plan['path_m']:.2f} m, V* = {plan['v_start']:.2f}, {len(plan['xy'])} "
              f"lattice poses, {plan['n_settle_bad']} settle-infeasible  {max_err}")
    path_a = resample_polyline(np.vstack([plan_a["xy"], goal]))
    path_b = resample_polyline(np.vstack([plan_b["xy"], goal]))
    path_len_a = float(np.linalg.norm(np.diff(path_a, axis=0), axis=1).sum())
    path_len_b = float(np.linalg.norm(np.diff(path_b, axis=0), axis=1).sum())

    # --- 2. simulate: 2*repeats worlds in one build, 0..repeats-1 = A, repeats..2*repeats-1 = B --
    sim_config, render_config, engine_config, logging_config = compose_ostrich_config(tuple(args.override))
    render_config.vis_type = "gl" if args.view else "null"
    n_worlds = 2 * repeats
    sim_config.num_worlds = n_worlds
    dt = float(sim_config.target_timestep_seconds)
    drive_steps = int(math.ceil(args.slack * max(path_len_a, path_len_b) / args.v / dt))
    T = args.settle_steps + drive_steps
    kappa_max = controller_kappa_max()
    print(f"\nostrich: {n_worlds} world(s) ({repeats} x {args.planner_a} [{ROBOT_COLOR_NAMES['a']}], "
          f"{repeats} x {args.planner_b} [{ROBOT_COLOR_NAMES['b']}]), "
          f"v={args.v} m/s, lookahead {args.lookahead} m, mu={args.mu}, dt={dt}, "
          f"{args.settle_steps} settle + {drive_steps} drive steps ({drive_steps * dt:.0f} s)")
    t0 = time.time()
    sim = None
    try:
        sim = ColoredPursuitSimulator(
            sim_config, render_config, engine_config, logging_config,
            k_p=K_P, **friction_kwargs(args.mu), terrain=terrain,
            spawn_pose=np.tile(np.array(start, np.float64), (n_worlds, 1)),
            paths=[path_a] * repeats + [path_b] * repeats,
            world_colors=[ROBOT_COLORS["a"]] * repeats + [ROBOT_COLORS["b"]] * repeats,
            settle_steps=args.settle_steps, lookahead=args.lookahead, v=args.v, kappa_max=kappa_max,
        )
        if args.view:
            # Newton's viewer spreads worlds apart for DISPLAY: ostrich's InteractiveSimulator calls
            # set_world_offsets((20, 20, 0)), and compute_world_offsets centres that grid, so at two
            # worlds the robots render at x-10 and x+10 while the terrain (a GLOBAL shape, world -1)
            # and the log_lines path overlay stay at their true coordinates -- one robot lands off
            # the map out of frame, the other drives along 10 m past the ramp. At num_worlds=1 the
            # offset is 0, which is why every other viewer demo here never had to care. We WANT the
            # worlds coincident (they cannot collide -- see this module's docstring), so zero it.
            sim.viewer.set_world_offsets((0.0, 0.0, 0.0))
            pos, pitch, yaw, dist = corner_camera(terrain, start, np.vstack([path_a, path_b]), sim.viewer.camera.fov)
            sim.viewer.set_camera(pos=wp.vec3(*pos), pitch=pitch, yaw=yaw)
            sim.viewer.camera.sync_pivot_to_view(dist)  # mouse orbit turns around both paths' midpoint
            sim.viewer._paused = True  # start paused: SPACE runs, "." steps (no public setter)
        pose, wheel_qd = sim.rollout(T, args.view)
    finally:
        sim = None
        gc.collect()
    print(f"ostrich rollout done in {time.time() - t0:.1f} s")
    pose, wheel_qd = pose[args.settle_steps :], wheel_qd[args.settle_steps :]
    if len(pose) == 0:
        raise SystemExit("viewer closed during the settle -- nothing to judge")

    # --- 3. verdict, one table per planner -------------------------------------------------------
    results_a = [judge(pose[:, w], dt, path_a, terrain) for w in range(repeats)]
    results_b = [judge(pose[:, repeats + w], dt, path_b, terrain) for w in range(repeats)]
    gate_a = f", gate {plan_a['gate']}" if plan_a["gate"] else ""
    gate_b = f", gate {plan_b['gate']}" if plan_b["gate"] else ""
    print_table(args.planner_a, gate_a, path_len_a, results_a)
    print_table(args.planner_b, gate_b, path_len_b, results_b)

    write_comparison(
        args.out_dir / f"{run_name}.h5",
        root=dict(
            name=run_name, map=str(map_path), planner_a=args.planner_a, planner_b=args.planner_b,
            head_a=str(plan_a["head"]), head_b=str(plan_b["head"]), tau_a=plan_a["tau"],
            tau_b=plan_b["tau"], gate_a=plan_a["gate"], gate_b=plan_b["gate"], n=n_worlds,
            repeats=repeats, v=args.v, lookahead=args.lookahead, mu=args.mu, slack=args.slack,
            settle_steps=args.settle_steps, path_m_a=path_len_a, path_m_b=path_len_b,
            obstacle_x=0.5 * (start[0] + goal[0]),
        ),
        per_variant=dict(
            spawn_pose=np.tile(np.array(start, np.float32), (n_worlds, 1)),
            v_drive=np.full(n_worlds, args.v, np.float32),
            wz_drive=np.zeros(n_worlds, np.float32),
            variant_value=np.arange(n_worlds, dtype=np.float32),
            variant_label=np.array(
                [f"{run_name}_a_r{w}" for w in range(repeats)] + [f"{run_name}_b_r{w}" for w in range(repeats)]
            ),
            planner=np.array([args.planner_a] * repeats + [args.planner_b] * repeats),
            status=np.array([r["status"] for r in results_a + results_b]),
            planned_path_a=path_a.astype(np.float32),
            planned_path_b=path_b.astype(np.float32),
            **{key: np.array([r[key] for r in results_a + results_b], np.float32)
               for key in ("t_arrive", "climb_deg", "roll_deg", "cte_max", "cte_mean",
                           "cte_at_climb", "progress_m")},
        ),
        terrain_entries=[(map_path, terrain)] * n_worlds,
        ostrich=dict(dt=dt, t=np.arange(len(pose), dtype=np.float32) * dt, pose=pose, wheel_qd=wheel_qd),
        hstack=dict(dt=dt),
    )


if __name__ == "__main__":
    main()
