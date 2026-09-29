"""The YAML config `mppi_learning/generate_dataset.py` consumes (design.md section 8, step 4).

`lattice_learning/dataset_config.py`'s schema and rules -- ONE file is the generator's only input,
every key is required, the h5 embeds its text -- with the lattice-only `trial` keys (kappa, belief
jitter) replaced and two sections added:

  * `command`: MPPI's wheel-speed box and sampling priors -> `command.CommandSpec`;
  * `entry`: the twist the robot is already driving when the window starts -> `EntrySpec`.

What is imported rather than restated: the `maps`/`run` sections, the section parser
(`build_section`/`exact_keys`), each strategy's params dataclass (the params mean the same here),
`check_maps` (given this package's own ramp_down platform length) and `allocate`. The strategy
registry is this package's own, pointing at `mppi_learning.spawn_sampling`'s window samplers.

`configs/default.yaml` documents every key.

Usage (smoke test -- no assets, no GPU):
    python src/feasibility/mppi_learning/dataset_config.py
"""
from __future__ import annotations

import dataclasses
import pathlib
from typing import Callable

import numpy as np
import yaml
from helhest.engine import RobotParams

from feasibility.heightmap import HeightMapReader
from feasibility.lattice_learning.dataset_config import allocate
from feasibility.lattice_learning.dataset_config import Allocation
from feasibility.lattice_learning.dataset_config import build_section
from feasibility.lattice_learning.dataset_config import check_maps
from feasibility.lattice_learning.dataset_config import ConfigError
from feasibility.lattice_learning.dataset_config import EDGE_CATEGORIES
from feasibility.lattice_learning.dataset_config import EdgeParams
from feasibility.lattice_learning.dataset_config import exact_keys
from feasibility.lattice_learning.dataset_config import MapsConfig
from feasibility.lattice_learning.dataset_config import PERCENT_TOL
from feasibility.lattice_learning.dataset_config import RAMP_CATEGORIES
from feasibility.lattice_learning.dataset_config import RampParams
from feasibility.lattice_learning.dataset_config import RotateParams
from feasibility.lattice_learning.dataset_config import RunConfig
from feasibility.lattice_learning.dataset_config import StrategySpec
from feasibility.lattice_learning.dataset_config import TargetedParams
from feasibility.lattice_learning.dataset_config import UniformParams
from feasibility.lattice_learning.spawn_sampling import ramp_faces
from feasibility.mppi_learning.command import CommandSpec
from feasibility.mppi_learning.command import FAMILIES
from feasibility.mppi_learning.command import MPPI_DT
from feasibility.mppi_learning.spawn_sampling import concat_batches
from feasibility.mppi_learning.spawn_sampling import edge_cells_in_reach
from feasibility.mppi_learning.spawn_sampling import EntrySpec
from feasibility.mppi_learning.spawn_sampling import required_platform_length
from feasibility.mppi_learning.spawn_sampling import rotate_cell_reach
from feasibility.mppi_learning.spawn_sampling import sample_edge_trials
from feasibility.mppi_learning.spawn_sampling import sample_ramp_down_trials
from feasibility.mppi_learning.spawn_sampling import sample_ramp_up_trials
from feasibility.mppi_learning.spawn_sampling import sample_rotate_in_place_trials
from feasibility.mppi_learning.spawn_sampling import sample_trials
from feasibility.mppi_learning.spawn_sampling import TrialContext
from feasibility.mppi_learning.spawn_sampling import WindowBatch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
CONFIGS_DIR = pathlib.Path(__file__).resolve().parent / "configs"


def _check(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _multiple_of(value: float, step: float) -> bool:
    return abs(value / step - round(value / step)) < 1e-6


# --- sections ------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TrialConfig:
    warmup_s: float
    settle_steps: int
    mu: float
    interact_relief: float
    k_turn: float
    ostrich_yaw_gain: float

    def __post_init__(self) -> None:
        # >= 2 MPPI steps: the twin's initial body twist is ostrich's pose difference over the
        # warm-up's last MPPI step, which must not reach back into the settle
        _check(self.warmup_s >= 2 * MPPI_DT - 1e-9 and _multiple_of(self.warmup_s, MPPI_DT),
               f"warmup_s must be a multiple of {MPPI_DT} s and >= {2 * MPPI_DT:g}, got {self.warmup_s}")
        _check(self.settle_steps >= 0, f"settle_steps must be >= 0, got {self.settle_steps}")
        _check(self.mu > 0.0, f"mu must be > 0, got {self.mu}")
        _check(self.interact_relief > 0.0, f"interact_relief must be > 0, got {self.interact_relief}")
        _check(self.k_turn >= 0.0, f"k_turn must be >= 0, got {self.k_turn}")
        # a gain < 1 would slow ostrich's turning; the compensation only ever speeds it up
        _check(self.ostrich_yaw_gain >= 1.0,
               f"ostrich_yaw_gain must be >= 1, got {self.ostrich_yaw_gain}")


@dataclasses.dataclass(frozen=True)
class CommandBox:
    """`command:` minus `family_mix`; with it, one `CommandSpec`."""

    wmin: float
    wmax: float
    sigma: float
    sigma_knot: float
    spin_min: float

    def __post_init__(self) -> None:
        # the samplers and design.md assume forward motion outside the spin family
        _check(self.wmin >= 0.0, f"wmin must be >= 0 (no reverse), got {self.wmin}")
        _check(self.wmax > self.wmin, f"wmax must be > wmin, got {self.wmax} <= {self.wmin}")
        _check(self.sigma >= 0.0, f"sigma must be >= 0, got {self.sigma}")
        _check(self.sigma_knot >= 0.0, f"sigma_knot must be >= 0, got {self.sigma_knot}")
        # 0 draws spins from standstill, below the real robot's ~2 rad/s breakaway (design.md section 2)
        _check(0.0 <= self.spin_min <= self.wmax, f"spin_min must be in [0, wmax], got {self.spin_min}")


@dataclasses.dataclass(frozen=True)
class FamilyMix:
    wide: float
    straight: float
    narrow: float
    spin: float

    def __post_init__(self) -> None:
        weights = dataclasses.astuple(self)
        _check(min(weights) >= 0.0 and sum(weights) > 0.0,
               f"weights must be >= 0 and not all 0, got {weights}")


assert tuple(f.name for f in dataclasses.fields(FamilyMix)) == FAMILIES


@dataclasses.dataclass(frozen=True)
class EntryConfig:
    jump_frac: float
    v_max: float
    wz_max: float
    yaw_ratio: float

    def __post_init__(self) -> None:
        _check(0.0 <= self.jump_frac <= 1.0, f"jump_frac must be in [0, 1], got {self.jump_frac}")
        _check(0.0 < self.yaw_ratio <= 1.0, f"yaw_ratio must be in (0, 1], got {self.yaw_ratio}")
        _check(self.v_max > 0.0, f"v_max must be > 0, got {self.v_max}")
        _check(self.wz_max >= 0.0, f"wz_max must be >= 0, got {self.wz_max}")


def _parse_command(data: object) -> CommandSpec:
    names = [f.name for f in dataclasses.fields(CommandBox)]
    data = exact_keys(data, names + ["family_mix"], "command")
    box = build_section(CommandBox, {k: data[k] for k in names}, "command")
    mix = build_section(FamilyMix, data["family_mix"], "command.family_mix")
    return CommandSpec(**dataclasses.asdict(box), mix=dataclasses.astuple(mix))


# --- strategies ----------------------------------------------------------------------------------

STRATEGIES: dict[str, StrategySpec] = {
    s.name: s
    for s in (
        StrategySpec("uniform", None, UniformParams, False, sample_trials, fixed={"interact_frac": None}),
        StrategySpec("targeted", None, TargetedParams, False, sample_trials),
        StrategySpec("ramp_up", RAMP_CATEGORIES, RampParams, True, sample_ramp_up_trials, needs_faces=True),
        StrategySpec("ramp_down", RAMP_CATEGORIES, RampParams, True, sample_ramp_down_trials, needs_faces=True),
        StrategySpec("edge", EDGE_CATEGORIES, EdgeParams, True, sample_edge_trials),
        StrategySpec("rotate_in_place", EDGE_CATEGORIES, RotateParams, True, sample_rotate_in_place_trials),
    )
}


def strategies_for(category: str) -> list[str]:
    return [s.name for s in STRATEGIES.values() if s.categories is None or category in s.categories]


# --- the whole file ------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MixEntry:
    map: str
    strategy: str
    percent: float
    params: object  # the strategy's params dataclass

    @property
    def spec(self) -> StrategySpec:
        return STRATEGIES[self.strategy]

    @property
    def label(self) -> str:
        return f"{self.strategy}@{self.map}"


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    seed: int
    maps: MapsConfig
    trial: TrialConfig
    command: CommandSpec
    entry: EntrySpec
    run: RunConfig
    ostrich_overrides: tuple[str, ...]
    mix: tuple[MixEntry, ...]
    name: str  # the config file's stem, tags the output file
    path: pathlib.Path | None  # None when parsed from a dict
    raw_yaml: str  # the file text, comments included, embedded in the h5

    @property
    def n(self) -> int:
        return self.maps.n_maps * self.maps.trials_per_map

    @property
    def categories(self) -> list[str]:
        """Categories in first-appearance order of the mix."""
        return list(dict.fromkeys(e.map for e in self.mix))


def _parse_mix(mix: object) -> tuple[MixEntry, ...]:
    if not isinstance(mix, list) or not mix:
        raise ValueError("mix: expected a non-empty list of {map, strategy, percent, params} entries")
    entries, seen = [], set()
    for i, item in enumerate(mix):
        where = f"mix[{i}]"
        item = exact_keys(item, ["map", "strategy", "percent", "params"], where)
        category, strategy, percent = item["map"], item["strategy"], item["percent"]
        if not isinstance(category, str) or not isinstance(strategy, str):
            raise ValueError(f"{where}: map and strategy must be strings")
        if strategy not in STRATEGIES:
            raise ValueError(f"{where}.strategy: unknown strategy {strategy!r} (known: {', '.join(STRATEGIES)})")
        spec = STRATEGIES[strategy]
        if spec.categories is not None and category not in spec.categories:
            raise ValueError(
                f"{where}: strategy {strategy!r} cannot run on map category {category!r} -- it needs "
                f"one of {', '.join(spec.categories)}; {category!r} allows: {', '.join(strategies_for(category))}"
            )
        if isinstance(percent, bool) or not isinstance(percent, (int, float)) or percent <= 0.0:
            raise ValueError(f"{where}.percent: expected a number > 0, got {percent!r}")
        if (category, strategy) in seen:
            raise ValueError(f"{where}: duplicate entry {strategy!r} on {category!r}")
        seen.add((category, strategy))
        entries.append(MixEntry(category, strategy, float(percent), build_section(spec.params, item["params"], f"{where}.params")))
    total = sum(e.percent for e in entries)
    if abs(total - 100.0) > PERCENT_TOL:
        raise ValueError(f"mix: percents sum to {total:g}, must be 100")
    return tuple(entries)


def _parse(data: object, name: str, path: pathlib.Path | None, raw: str) -> DatasetConfig:
    top = exact_keys(data, ["seed", "maps", "trial", "command", "entry", "run", "ostrich_overrides", "mix"], "")
    if isinstance(top["seed"], bool) or not isinstance(top["seed"], int):
        raise ValueError(f"seed: expected int, got {top['seed']!r}")
    overrides = top["ostrich_overrides"]
    if not isinstance(overrides, list) or not all(isinstance(o, str) for o in overrides):
        raise ValueError(f"ostrich_overrides: expected a list of strings, got {overrides!r}")
    trial = build_section(TrialConfig, top["trial"], "trial")
    # the sampler draws in ostrich's (compensated) wheel space, so the gain is part of the box
    command = dataclasses.replace(_parse_command(top["command"]), ostrich_yaw_gain=trial.ostrich_yaw_gain)
    if command.spin_min * command.ostrich_yaw_gain > command.wmax + 1e-9:
        raise ValueError(f"command.spin_min {command.spin_min} leaves no spin band: compensated by "
                         f"trial.ostrich_yaw_gain {command.ostrich_yaw_gain} it exceeds wmax {command.wmax}")
    entry = build_section(EntryConfig, top["entry"], "entry")
    if entry.v_max > command.v_max + 1e-9:
        raise ValueError(f"entry.v_max {entry.v_max} exceeds the command box's {command.v_max:.3f} m/s")
    return DatasetConfig(
        seed=top["seed"],
        maps=build_section(MapsConfig, top["maps"], "maps"),
        trial=trial,
        command=command,
        entry=EntrySpec(**dataclasses.asdict(entry)),
        run=build_section(RunConfig, top["run"], "run"),
        ostrich_overrides=tuple(overrides),
        mix=_parse_mix(top["mix"]),
        name=name,
        path=path,
        raw_yaml=raw,
    )


def resolve_config_path(name_or_path: str) -> pathlib.Path:
    """A path (absolute, cwd- or repo-root-relative) if it exists, else `configs/<name>[.yaml]`."""
    p = pathlib.Path(name_or_path)
    for c in (p, REPO_ROOT / p, CONFIGS_DIR / p, CONFIGS_DIR / f"{name_or_path}.yaml"):
        if c.is_file():
            return c.resolve()
    known = ", ".join(sorted(c.stem for c in CONFIGS_DIR.glob("*.yaml")))
    raise ConfigError(f"no config {name_or_path!r}: not a file, and not one of configs/ ({known})")


def parse_config(data: object, name: str = "<dict>", path: pathlib.Path | None = None, raw: str = "") -> DatasetConfig:
    """`DatasetConfig` from already-loaded YAML data; every problem raises ConfigError."""
    try:
        return _parse(data, name, path, raw)
    except ValueError as err:
        raise ConfigError(f"{path or name}: {err}") from None


def load_config(name_or_path: str) -> DatasetConfig:
    path = resolve_config_path(name_or_path)
    raw = path.read_text()
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as err:
        raise ConfigError(f"{path}: not valid YAML: {err}") from None
    return parse_config(data, path.stem, path, raw)


# --- maps and sampling ---------------------------------------------------------------------------


def trial_context(
    cfg: DatasetConfig, terrain: HeightMapReader | None, device: str, robot: RobotParams | None = None
) -> TrialContext:
    """The samplers' shared context for one map. `terrain` None is only for map-independent
    queries (`required_platform_length`)."""
    return TrialContext(
        terrain, mu=cfg.trial.mu, device=device, interact_relief=cfg.trial.interact_relief,
        warmup_s=cfg.trial.warmup_s, command=cfg.command, entry=cfg.entry, robot=robot or RobotParams(),
    )


def platform_length(cfg: DatasetConfig) -> float:
    """m, the shortest plateau ramp_down can sample on at this config's warm-up: the robot and its
    warm-up on it (longer plateaus let trials start further back, `spawn_sampling`)."""
    return required_platform_length(trial_context(cfg, None, "cpu"))


def check_mppi_maps(cfg: DatasetConfig) -> dict[str, list[pathlib.Path]]:
    """`lattice_learning.dataset_config.check_maps` with this package's ramp_down platform."""
    return check_maps(cfg, lead=float("nan"), platform_length=platform_length(cfg))


def check_edge_reach(allocation: Allocation) -> None:
    """Refuse, before anything is simulated, an allocation holding a `rotate_in_place` map whose
    features all lie outside `sampling_bounds`' square (that sampler raises on it; `edge` falls back
    to uniform instead). Such a dir was generated without `--center-limit`; costs one distance
    transform per affected map."""
    cfg = allocation.cfg
    bad = []
    for m in allocation.maps:
        entries = [cfg.mix[e] for e in m.counts if cfg.mix[e].strategy == "rotate_in_place"]
        if not entries:
            continue
        ctx = trial_context(cfg, HeightMapReader.load(m.path), "cpu")
        if any(len(edge_cells_in_reach(ctx, rotate_cell_reach(ctx, e.params.band))) == 0 for e in entries):
            bad.append(m.path.name)
    if bad:
        raise ConfigError(
            f"{cfg.path or cfg.name}: {len(bad)} rotate_in_place map(s) in {cfg.maps.path} have no height edge "
            f"within reach of the origin square ({', '.join(bad[:5])}{', ...' if len(bad) > 5 else ''}) -- "
            f"regenerate the dir with create_maps_for_lattice_learning.py --center-limit 2.38"
        )


def sample_map_mix(
    terrain: HeightMapReader,
    meta: dict,
    counts: dict[int, int],
    cfg: DatasetConfig,
    rng: np.random.Generator,
    *,
    device: str,
    robot: RobotParams | None = None,
) -> WindowBatch:
    """One map's trials: each mix entry's sampler for its share of `counts`, merged by
    `spawn_sampling.concat_batches` into one batch in random row order."""
    ctx = trial_context(cfg, terrain, device, robot)
    batches = []
    for e in sorted(counts):
        entry = cfg.mix[e]
        strategy = entry.spec
        faces = (ramp_faces(meta),) if strategy.needs_faces else ()
        batches.append(strategy.sampler(
            ctx, *faces, counts[e], rng, **strategy.fixed, **dataclasses.asdict(entry.params),
            require_endpoint=not strategy.spawn_only,
        ))
    return concat_batches(batches, rng)


if __name__ == "__main__":
    import copy

    def raises(data: dict, fragment: str) -> None:
        try:
            parse_config(data)
        except ConfigError as err:
            assert fragment in str(err), (fragment, str(err))
            return
        raise AssertionError(f"expected a ConfigError containing {fragment!r}")

    configs = sorted(CONFIGS_DIR.glob("*.yaml"))
    assert {c.stem for c in configs} >= {"default", "smoke"}, configs
    for c in configs:
        cfg = load_config(c.stem)
        assert load_config(str(c)) == cfg, "a name and its path must load the same config"
        print(f"[load] {c.stem}: n={cfg.n}, mix {', '.join(f'{e.label} {e.percent:g}%' for e in cfg.mix)}")

    base = yaml.safe_load((CONFIGS_DIR / "default.yaml").read_text())
    default = parse_config(base, "default")
    # MPPI's box, except spins drawn from standstill (design.md section 2)
    assert default.command == CommandSpec(ostrich_yaw_gain=default.trial.ostrich_yaw_gain, spin_min=0.0), \
        "default.yaml's command block is MPPI's box, spin_min 0 aside (design.md)"
    edge = next(i for i, e in enumerate(base["mix"]) if e["strategy"] == "edge")

    def mutate(fn: Callable[[dict], None]) -> dict:
        data = copy.deepcopy(base)
        fn(data)
        return data

    raises(mutate(lambda d: d["mix"][0].update(percent=d["mix"][0]["percent"] - 10)), "sum to 90")
    raises(mutate(lambda d: d["mix"][edge].update(map="rough")), "cannot run on map category 'rough'")
    raises(mutate(lambda d: d["mix"][edge].update(strategy="edge_sideways")), "unknown strategy")
    raises(mutate(lambda d: d["mix"][edge]["params"].update(wiggle=1)), f"mix[{edge}].params.wiggle")
    raises(mutate(lambda d: d["mix"][edge]["params"].pop("band")), f"mix[{edge}].params.band")
    raises(mutate(lambda d: d.update(sed=0)), "unknown key(s) sed")
    raises(mutate(lambda d: d["trial"].update(kappa_max=2.0)), "trial.kappa_max")
    raises(mutate(lambda d: d["trial"].update(warmup_s=0.375)), "multiple of 0.1")
    raises(mutate(lambda d: d["trial"].update(warmup_s=0.1)), ">= 0.2")
    raises(mutate(lambda d: d["trial"].update(yaw_gain=0.49)), "trial.yaw_gain")
    raises(mutate(lambda d: d["trial"].update(ostrich_yaw_gain=0.87)), "ostrich_yaw_gain must be >= 1")
    raises(mutate(lambda d: d["trial"].pop("k_turn")), "trial.k_turn")
    raises(mutate(lambda d: d["command"].update(spin_min=3.6)), "leaves no spin band")
    raises(mutate(lambda d: d["entry"].update(jump_frac=1.5)), "jump_frac must be in [0, 1]")
    raises(mutate(lambda d: d["command"].update(wmin=-4.0)), "no reverse")
    raises(mutate(lambda d: d["command"].pop("sigma")), "command.sigma")
    raises(mutate(lambda d: d["command"]["family_mix"].update(pivot=0.1)), "command.family_mix.pivot")
    raises(mutate(lambda d: d["command"]["family_mix"].update(spin=-0.1)), "must be >= 0")
    raises(mutate(lambda d: d["entry"].update(v_max=2.0)), "exceeds the command box")
    raises(mutate(lambda d: d["entry"].update(kappa_max=2.0)), "entry.kappa_max")
    raises(mutate(lambda d: d["run"].pop("chunk")), "run.chunk")
    raises(mutate(lambda d: d["mix"].append(copy.deepcopy(d["mix"][0]))), "duplicate entry")
    print("[validate] 22 broken variants all rejected with the offending key named")

    pool = {c: [pathlib.Path(f"/nonexistent/{c}_i{i:04d}") for i in range(default.maps.n_maps)] for c in default.categories}
    alloc = allocate(default, pool, np.random.default_rng(0))
    assert len(alloc.maps) == default.maps.n_maps
    assert all(sum(m.counts.values()) == default.maps.trials_per_map for m in alloc.maps)
    print(alloc.table())
    print(f"[platform] ramp_down needs {platform_length(default):.2f} m of plateau at this config")

    # sampling end to end on a synthetic box map, CPU settle
    import warp as wp

    wp.init()
    xs = np.arange(-8.0, 8.0, 0.05) + 0.025
    X, Y = np.meshgrid(xs, xs)
    box = HeightMapReader(np.where((np.abs(X) < 1.5) & (np.abs(Y) < 1.5), 0.3, 0.0), origin=(-8.0, -8.0), cell=0.05)
    edge_cfg = parse_config(mutate(lambda d: d.update(mix=[dict(map="curbs_and_walls", strategy="edge", percent=60.0, params=d["mix"][edge]["params"]),
                                                            dict(map="curbs_and_walls", strategy="uniform", percent=40.0, params={})])))
    batch = sample_map_mix(box, {}, {0: 12, 1: 8}, edge_cfg, np.random.default_rng(0), device="cpu")
    assert len(batch.pose) == 20 and sorted(set(batch.strategy)) == ["edge", "uniform"]
    print("[sample] edge + uniform on one map: 20 rows")
    print("all self-checks ok")
