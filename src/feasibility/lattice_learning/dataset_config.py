"""The YAML config `generate_dataset.py` consumes: parsing, validation, the strategy registry, and
the allocation of maps and rows to the config's sampling `mix`.

`generate_dataset.py` takes ONE argument, a config name (a file in `configs/`) or a path, and
nothing else -- every knob that used to be a Hydra `+key=` override lives in the file, so a dataset
is reproducible from its config alone (the h5 embeds the file's text as `config_yaml`).
`configs/default.yaml` documents every key. Every key is REQUIRED: a missing one raises instead of
falling back to a code default, so changing a default in code can never silently change what a
config file generates.

The `mix` is a list of `{map, strategy, percent, params}` entries -- "`percent` of the dataset's
samples are drawn by `strategy` on maps whose sidecar `category` is `map`". Checked here:
  - percents are > 0 and sum to 100;
  - the strategy exists in `STRATEGIES` and `map` is one of its eligible categories;
  - `params` has exactly that strategy's params fields, each within range;
  - no (map, strategy) pair appears twice; no unknown or missing key anywhere.
Against a map directory (`check_maps`): every category the mix names has maps, and `ramp_down`
only runs on ramps whose plateau is a standing platform (`spawn_sampling.required_platform_length`).

Allocation (`allocate`) -- entries on the same category SHARE its maps:
  1. a category's share is the sum of its entries' percents; it gets `m_c` maps, the
     largest-remainder rounding of `n_maps * share / 100` (so the counts sum to `n_maps`), drawn
     without replacement from that category's sorted pool;
  2. its `m_c * trials_per_map` rows are split among its entries by the same rounding;
  3. those rows are dealt round-robin over its maps, so every map gets exactly `trials_per_map`
     rows and each entry floor or ceil of its rows / `m_c` on every map (16 trials, 50/50 -> 8 + 8).
Realized percents therefore differ from the requested ones only by map granularity; `table()`
prints both.

Adding a strategy or a map category: write the sampler in `spawn_sampling.py`, a frozen params
dataclass here (range checks in `__post_init__`), and one `STRATEGIES` entry.

Usage (smoke test -- no assets, no GPU):
    python src/feasibility/lattice_learning/dataset_config.py
"""
from __future__ import annotations

import dataclasses
import pathlib
import typing
from typing import Callable

import numpy as np
import yaml
from helhest.engine import RobotParams

from feasibility.heightmap import HeightMapReader
from feasibility.heightmap.create_maps_for_lattice_learning import largest_remainder
from feasibility.lattice_learning.patch import PatchSpec
from feasibility.lattice_learning.spawn_sampling import concat_batches
from feasibility.lattice_learning.spawn_sampling import map_metadata
from feasibility.lattice_learning.spawn_sampling import ramp_faces
from feasibility.lattice_learning.spawn_sampling import required_platform_length
from feasibility.lattice_learning.spawn_sampling import sample_edge_trials
from feasibility.lattice_learning.spawn_sampling import sample_ramp_down_trials
from feasibility.lattice_learning.spawn_sampling import sample_ramp_up_trials
from feasibility.lattice_learning.spawn_sampling import sample_rotate_in_place_trials
from feasibility.lattice_learning.spawn_sampling import sample_trials
from feasibility.lattice_learning.spawn_sampling import SpawnBatch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
CONFIGS_DIR = pathlib.Path(__file__).resolve().parent / "configs"
MAP_GLOB = "*.png"
PERCENT_TOL = 0.01  # percents must sum to 100 within this, so thirds can be written 33.34/33.33/33.33


class ConfigError(ValueError):
    """A config file that cannot generate a dataset; the message names the file and the key."""


# --- sections ------------------------------------------------------------------------------------


def _check(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _fraction(name: str, value: float | None, allow_none: bool = False) -> None:
    if value is None and allow_none:
        return
    _check(value is not None and 0.0 <= value <= 1.0, f"{name} must be in [0, 1], got {value}")


@dataclasses.dataclass(frozen=True)
class MapsConfig:
    dir: str
    n_maps: int
    trials_per_map: int

    def __post_init__(self) -> None:
        _check(self.n_maps >= 1, f"n_maps must be >= 1, got {self.n_maps}")
        _check(self.trials_per_map >= 1, f"trials_per_map must be >= 1, got {self.trials_per_map}")

    @property
    def path(self) -> pathlib.Path:
        """Repo-root-relative or absolute, like every heightmap/create_*.py generator's output."""
        p = pathlib.Path(self.dir)
        return p if p.is_absolute() else REPO_ROOT / p


@dataclasses.dataclass(frozen=True)
class TrialConfig:
    kappa_max: float
    warmup_s: float
    settle_steps: int
    mu: float
    yaw_gain: float
    interact_relief: float
    router_cell: float
    n_theta: int

    def __post_init__(self) -> None:
        _check(self.kappa_max >= 0.0, f"kappa_max must be >= 0, got {self.kappa_max}")
        _check(self.warmup_s >= 0.0, f"warmup_s must be >= 0, got {self.warmup_s}")
        _check(self.settle_steps >= 0, f"settle_steps must be >= 0, got {self.settle_steps}")
        _check(self.mu > 0.0, f"mu must be > 0, got {self.mu}")
        # 1.0 = command the nominal twist and let ostrich fall short (every dataset before this
        # key existed); above 1.0 would mean ostrich OVER-rotates, which no measurement shows and
        # which would command less than the primitive asks for.
        _check(0.0 < self.yaw_gain <= 1.0, f"yaw_gain must be in (0, 1], got {self.yaw_gain}")
        _check(self.interact_relief > 0.0, f"interact_relief must be > 0, got {self.interact_relief}")
        _check(self.router_cell > 0.0, f"router_cell must be > 0, got {self.router_cell}")
        _check(self.n_theta >= 1, f"n_theta must be >= 1, got {self.n_theta}")

    @property
    def xy_jitter(self) -> float:
        """m, belief xy jitter half-width: half the router's lattice cell (design.md section 1c)."""
        return self.router_cell / 2.0

    @property
    def yaw_jitter(self) -> float:
        """rad, belief yaw jitter half-width: half the router's heading bin."""
        return np.pi / self.n_theta


@dataclasses.dataclass(frozen=True)
class RunConfig:
    chunk: int
    maps_per_build: int
    device: str
    dry_run: bool

    def __post_init__(self) -> None:
        _check(self.chunk >= 1, f"chunk must be >= 1, got {self.chunk}")
        _check(self.maps_per_build >= 1, f"maps_per_build must be >= 1, got {self.maps_per_build}")


# --- strategies ----------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class UniformParams:
    pass


@dataclasses.dataclass(frozen=True)
class TargetedParams:
    interact_frac: float

    def __post_init__(self) -> None:
        _fraction("interact_frac", self.interact_frac)


@dataclasses.dataclass(frozen=True)
class RampParams:
    straight_frac: float
    yaw_jitter_deg: float
    fallback_interact_frac: float | None

    def __post_init__(self) -> None:
        _fraction("straight_frac", self.straight_frac)
        _check(0.0 <= self.yaw_jitter_deg <= 90.0, f"yaw_jitter_deg must be in [0, 90], got {self.yaw_jitter_deg}")
        _fraction("fallback_interact_frac", self.fallback_interact_frac, allow_none=True)


@dataclasses.dataclass(frozen=True)
class EdgeParams:
    interact_frac: float | None
    band: float
    facing_frac: float
    down_frac: float

    def __post_init__(self) -> None:
        _fraction("interact_frac", self.interact_frac, allow_none=True)
        _check(self.band > 0.0, f"band must be > 0, got {self.band}")
        _fraction("facing_frac", self.facing_frac)
        _fraction("down_frac", self.down_frac)


@dataclasses.dataclass(frozen=True)
class RotateParams:
    interact_frac: float | None
    band: float
    min_clearance: float

    def __post_init__(self) -> None:
        _fraction("interact_frac", self.interact_frac, allow_none=True)
        _check(self.band > 0.0, f"band must be > 0, got {self.band}")
        _check(self.min_clearance >= 0.0, f"min_clearance must be >= 0, got {self.min_clearance}")


@dataclasses.dataclass(frozen=True)
class StrategySpec:
    name: str
    categories: tuple[str, ...] | None  # sidecar categories it may run on; None = any
    params: type
    spawn_only: bool  # True: the NOMINAL arc end need not be settle-feasible (spawn_sampling docstring)
    sampler: Callable[..., SpawnBatch]  # spawn_sampling function, called by sample_map_mix
    needs_faces: bool = False  # sampler takes the map's ramp faces (spawn_sampling.ramp_faces)
    fixed: dict = dataclasses.field(default_factory=dict)  # keyword args not exposed as params


RAMP_CATEGORIES = ("ramps",)
EDGE_CATEGORIES = ("curbs_and_walls", "poles_and_walls", "walls", "boxes")  # walls/boxes: retired
# categories of old dirs

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


def _type_ok(value: object, hint: object) -> bool:
    """A YAML scalar against a dataclass field's type hint: int is not bool, float accepts int."""
    if hint is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if hint is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if hint in (str, bool):
        return isinstance(value, hint)
    args = typing.get_args(hint)
    if args and type(None) in args:
        return value is None or any(_type_ok(value, a) for a in args if a is not type(None))
    raise TypeError(f"unsupported field type {hint}")


def _keys(data: object, required: list[str], where: str) -> dict:
    if not isinstance(data, dict):
        raise ValueError(f"{where or 'top level'}: expected a mapping, got {type(data).__name__}")
    unknown = sorted(set(data) - set(required))
    missing = [k for k in required if k not in data]
    prefix = f"{where}." if where else ""
    if unknown:
        raise ValueError(f"unknown key(s) {', '.join(prefix + k for k in unknown)} "
                         f"(expected: {', '.join(required) or 'none'})")
    if missing:
        raise ValueError(f"missing required key(s) {', '.join(prefix + k for k in missing)}")
    return data


def _build(cls: type, data: object, where: str) -> object:
    """`cls(**data)` with exact keys, per-field type checks, and `where` in every error."""
    hints = typing.get_type_hints(cls)
    names = [f.name for f in dataclasses.fields(cls)]
    data = _keys({} if data is None and not names else data, names, where)
    for name in names:
        if not _type_ok(data[name], hints[name]):
            raise ValueError(f"{where}.{name}: expected {hints[name]}, got {data[name]!r}")
    try:
        return cls(**{name: float(data[name]) if hints[name] is float else data[name] for name in names})
    except ValueError as err:
        raise ValueError(f"{where}: {err}") from None


def _parse(data: object, name: str, path: pathlib.Path | None, raw: str) -> DatasetConfig:
    top = _keys(data, ["seed", "maps", "trial", "run", "ostrich_overrides", "mix"], "")
    if not _type_ok(top["seed"], int):
        raise ValueError(f"seed: expected int, got {top['seed']!r}")
    overrides = top["ostrich_overrides"]
    if not isinstance(overrides, list) or not all(isinstance(o, str) for o in overrides):
        raise ValueError(f"ostrich_overrides: expected a list of strings, got {overrides!r}")

    mix = top["mix"]
    if not isinstance(mix, list) or not mix:
        raise ValueError("mix: expected a non-empty list of {map, strategy, percent, params} entries")
    entries, seen = [], set()
    for i, item in enumerate(mix):
        where = f"mix[{i}]"
        item = _keys(item, ["map", "strategy", "percent", "params"], where)
        category, strategy, percent = item["map"], item["strategy"], item["percent"]
        if not isinstance(category, str) or not isinstance(strategy, str):
            raise ValueError(f"{where}: map and strategy must be strings")
        if strategy not in STRATEGIES:
            raise ValueError(f"{where}.strategy: unknown strategy {strategy!r} "
                             f"(known: {', '.join(STRATEGIES)})")
        spec = STRATEGIES[strategy]
        if spec.categories is not None and category not in spec.categories:
            raise ValueError(
                f"{where}: strategy {strategy!r} cannot run on map category {category!r} -- it needs "
                f"one of {', '.join(spec.categories)}; {category!r} allows: "
                f"{', '.join(strategies_for(category))}"
            )
        if not _type_ok(percent, float) or percent <= 0.0:
            raise ValueError(f"{where}.percent: expected a number > 0, got {percent!r}")
        if (category, strategy) in seen:
            raise ValueError(f"{where}: duplicate entry {strategy!r} on {category!r}")
        seen.add((category, strategy))
        params = _build(spec.params, item["params"], f"{where}.params")
        entries.append(MixEntry(category, strategy, float(percent), params))
    total = sum(e.percent for e in entries)
    if abs(total - 100.0) > PERCENT_TOL:
        raise ValueError(f"mix: percents sum to {total:g}, must be 100")

    return DatasetConfig(
        seed=top["seed"],
        maps=_build(MapsConfig, top["maps"], "maps"),
        trial=_build(TrialConfig, top["trial"], "trial"),
        run=_build(RunConfig, top["run"], "run"),
        ostrich_overrides=tuple(overrides),
        mix=tuple(entries),
        name=name,
        path=path,
        raw_yaml=raw,
    )


def resolve_config_path(name_or_path: str) -> pathlib.Path:
    """A path (absolute, cwd- or repo-root-relative) if it exists, else `configs/<name>[.yaml]`."""
    p = pathlib.Path(name_or_path)
    candidates = [p, REPO_ROOT / p, CONFIGS_DIR / p, CONFIGS_DIR / f"{name_or_path}.yaml"]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    known = ", ".join(sorted(c.stem for c in CONFIGS_DIR.glob("*.yaml")))
    raise ConfigError(f"no config {name_or_path!r}: not a file, and not one of configs/ ({known})")


def parse_config(data: object, name: str = "<dict>", path: pathlib.Path | None = None,
                 raw: str = "") -> DatasetConfig:
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


# --- maps ----------------------------------------------------------------------------------------


def check_maps(cfg: DatasetConfig, lead: float) -> dict[str, list[pathlib.Path]]:
    """{category: sorted map stems} for every category the mix names, from `maps.dir`'s sidecars.
    Raises if the directory, or a category, is missing, and if `ramp_down` would run on a ramp
    whose plateau is shorter than `required_platform_length(lead)` (maps generated before
    create_ramps.RampsConfig's standing-platform floor)."""
    where = cfg.path or cfg.name
    maps_dir = cfg.maps.path
    if not maps_dir.is_dir():
        raise ConfigError(f"{where}: maps.dir {maps_dir} does not exist")
    pool: dict[str, list[pathlib.Path]] = {}
    metas: dict[pathlib.Path, dict] = {}
    for png in sorted(maps_dir.glob(MAP_GLOB)):
        stem = png.with_suffix("")
        metas[stem] = map_metadata(stem)
        category = metas[stem].get("category")
        if category is not None:
            pool.setdefault(str(category), []).append(stem)
    for category in cfg.categories:
        if category not in pool:
            found = ", ".join(f"{c} ({len(v)})" for c, v in pool.items()) or "none"
            raise ConfigError(f"{where}: the mix uses map category {category!r}, but "
                              f"{maps_dir} has no such maps (categories found: {found})")

    need = required_platform_length(lead)
    for entry in cfg.mix:
        if entry.strategy != "ramp_down":
            continue
        short = [
            (stem.name, min(plateaus))
            for stem in pool[entry.map]
            if (plateaus := [float(r["plateau"]) for r in metas[stem].get("ramps") or []])
            and min(plateaus) < need
        ]
        if short:
            raise ConfigError(
                f"{where}: ramp_down needs every ramp plateau >= {need:.2f} m (a standing platform "
                f"at warmup_s={cfg.trial.warmup_s}), but {len(short)}/{len(pool[entry.map])} "
                f"{entry.map!r} maps in {maps_dir} have shorter ones (e.g. {short[0][0]}: "
                f"{short[0][1]:.2f} m). Regenerate them with "
                f"heightmap/create_maps_for_lattice_learning.py, whose ramps now keep that floor."
            )
    return pool


@dataclasses.dataclass(frozen=True)
class AllocatedMap:
    path: pathlib.Path
    category: str
    counts: dict[int, int]  # mix entry index -> trials on this map; sums to trials_per_map


@dataclasses.dataclass(frozen=True)
class Allocation:
    cfg: DatasetConfig
    maps: list[AllocatedMap]

    def entry_rows(self) -> list[int]:
        rows = [0] * len(self.cfg.mix)
        for m in self.maps:
            for e, k in m.counts.items():
                rows[e] += k
        return rows

    def table(self) -> str:
        n = self.cfg.n
        rows = self.entry_rows()
        maps_per_cat = {c: sum(m.category == c for m in self.maps) for c in self.cfg.categories}
        lines = [f"{'strategy':<15} {'map category':<16} {'maps':>4} {'rows':>6} {'requested':>9} "
                 f"{'realized':>8}"]
        for e, entry in enumerate(self.cfg.mix):
            lines.append(f"{entry.strategy:<15} {entry.map:<16} {maps_per_cat[entry.map]:>4} {rows[e]:>6} "
                         f"{entry.percent:>8.2f}% {100.0 * rows[e] / n:>7.2f}%")
        lines.append(f"{'total':<32} {len(self.maps):>4} {sum(rows):>6}")
        return "\n".join(lines)


def allocate(cfg: DatasetConfig, pool: dict[str, list[pathlib.Path]], rng: np.random.Generator) -> Allocation:
    """Maps and per-map trial counts for `cfg.mix` -- see the module docstring."""
    where = cfg.path or cfg.name
    categories = cfg.categories
    shares = [sum(e.percent for e in cfg.mix if e.map == c) for c in categories]
    n_cat_maps = largest_remainder(shares, cfg.maps.n_maps)
    T = cfg.maps.trials_per_map

    maps: list[AllocatedMap] = []
    for category, m_c in zip(categories, n_cat_maps):
        if m_c == 0:
            raise ConfigError(f"{where}: maps.n_maps={cfg.maps.n_maps} leaves map category "
                              f"{category!r} no map at its share -- raise n_maps")
        candidates = sorted(pool.get(category, []))
        if len(candidates) < m_c:
            raise ConfigError(f"{where}: the mix needs {m_c} {category!r} maps, {cfg.maps.path} has "
                              f"{len(candidates)} -- generate more or lower maps.n_maps")
        chosen = [candidates[i] for i in rng.choice(len(candidates), size=m_c, replace=False)]

        entry_ids = [e for e, entry in enumerate(cfg.mix) if entry.map == category]
        entry_rows = largest_remainder([cfg.mix[e].percent for e in entry_ids], m_c * T)
        for e, rows in zip(entry_ids, entry_rows):
            if rows == 0:
                raise ConfigError(f"{where}: mix entry {cfg.mix[e].label} ({cfg.mix[e].percent:g}%) "
                                  f"rounds to 0 rows -- raise maps.n_maps or trials_per_map")
        labels = [e for e, rows in zip(entry_ids, entry_rows) for _ in range(rows)]
        counts = [{e: 0 for e in entry_ids} for _ in chosen]
        for i, e in enumerate(labels):  # round-robin deal: every map gets exactly T rows
            counts[i % m_c][e] += 1
        maps.extend(AllocatedMap(p, category, {e: k for e, k in c.items() if k}) for p, c in zip(chosen, counts))
    return Allocation(cfg, maps)


def sample_map_mix(
    terrain: HeightMapReader,
    meta: dict,
    spec: PatchSpec,
    counts: dict[int, int],
    cfg: DatasetConfig,
    rng: np.random.Generator,
    *,
    lead: float,
    device: str,
    robot: RobotParams | None = None,
) -> SpawnBatch:
    """One map's trials: each mix entry's sampler for its share of `counts`, merged by
    `spawn_sampling.concat_batches` into one batch in random row order."""
    batches = []
    for e in sorted(counts):
        entry = cfg.mix[e]
        common = dict(kappa_max=cfg.trial.kappa_max, lead=lead, mu=cfg.trial.mu, device=device,
                      interact_relief=cfg.trial.interact_relief, robot=robot,
                      require_endpoint=not entry.spec.spawn_only)
        strategy = entry.spec
        faces = (ramp_faces(meta),) if strategy.needs_faces else ()
        batches.append(strategy.sampler(
            terrain, *faces, spec, counts[e], rng, **strategy.fixed, **dataclasses.asdict(entry.params),
            **common,
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
    assert {c.stem for c in configs} >= {"default", "up_down"}, configs
    for c in configs:
        cfg = load_config(c.stem)
        assert load_config(str(c)) == cfg, "a name and its path must load the same config"
        print(f"[load] {c.stem}: n={cfg.n}, mix {', '.join(f'{e.label} {e.percent:g}%' for e in cfg.mix)}")

    base = yaml.safe_load((CONFIGS_DIR / "up_down.yaml").read_text())
    ramp_down = next(i for i, e in enumerate(base["mix"]) if e["strategy"] == "ramp_down")
    edge = next(i for i, e in enumerate(base["mix"]) if e["strategy"] == "edge")

    def mutate(fn: Callable[[dict], None]) -> dict:
        data = copy.deepcopy(base)
        fn(data)
        return data

    raises(mutate(lambda d: d["mix"][0].update(percent=15)), "sum to 90")
    raises(mutate(lambda d: d["mix"][ramp_down].update(map="rough")), "cannot run on map category 'rough'")
    raises(mutate(lambda d: d["mix"][ramp_down].update(strategy="ramp_sideways")), "unknown strategy")
    raises(mutate(lambda d: d["mix"][edge]["params"].update(wiggle=1)), "mix[3].params.wiggle")
    raises(mutate(lambda d: d["mix"][edge]["params"].pop("band")), "mix[3].params.band")
    raises(mutate(lambda d: d["mix"][edge]["params"].update(down_frac=1.5)), "down_frac must be in [0, 1]")
    raises(mutate(lambda d: d["mix"][edge]["params"].update(band="wide")), "mix[3].params.band: expected")
    raises(mutate(lambda d: d.update(sed=0)), "unknown key(s) sed")
    raises(mutate(lambda d: d["trial"].update(warmup=0.3)), "trial.warmup")
    raises(mutate(lambda d: d["run"].pop("chunk")), "run.chunk")
    raises(mutate(lambda d: d["maps"].update(n_maps=0)), "n_maps must be >= 1")
    raises(mutate(lambda d: d["run"].update(dry_run="yes")), "run.dry_run: expected")
    raises(mutate(lambda d: d["mix"].append(copy.deepcopy(d["mix"][0]))), "duplicate entry")
    raises(mutate(lambda d: d.update(mix=[])), "non-empty list")
    rotate = dict(map="curbs_and_walls", strategy="rotate_in_place", percent=10.0,
                  params=dict(interact_frac=0.5, band=0.3, min_clearance=0.05))

    def with_rotate(d: dict) -> None:
        d["mix"][edge]["percent"] -= 10.0
        d["mix"].append(copy.deepcopy(rotate))

    rot_cfg = parse_config(mutate(with_rotate), "up_down")
    assert rot_cfg.mix[-1].spec.name == "rotate_in_place" and rot_cfg.mix[-1].spec.spawn_only
    raises(mutate(lambda d: (with_rotate(d), d["mix"][-1].update(map="ramps"))),
           "cannot run on map category 'ramps'")
    raises(mutate(lambda d: (with_rotate(d), d["mix"][-1]["params"].update(min_clearance=-0.1))),
           "min_clearance must be >= 0")
    print("[validate] 16 broken variants all rejected with the offending key named; rotate_in_place parses")

    # allocation on a synthetic pool, no files touched
    cfg = parse_config(base, "up_down")
    pool = {c: [pathlib.Path(f"/nonexistent/{c}_i{i:04d}") for i in range(20)] for c in ("rough", "ramps", "curbs_and_walls")}
    for n_maps, T in ((12, 16), (13, 7), (40, 16), (4, 3)):
        cfg_n = dataclasses.replace(cfg, maps=MapsConfig(cfg.maps.dir, n_maps, T))
        alloc = allocate(cfg_n, pool, np.random.default_rng(0))
        assert len(alloc.maps) == n_maps and len({m.path for m in alloc.maps}) == n_maps
        assert all(sum(m.counts.values()) == T for m in alloc.maps)
        assert sum(alloc.entry_rows()) == n_maps * T
        for e, entry in enumerate(cfg.mix):
            on = [m.counts.get(e, 0) for m in alloc.maps if m.category == entry.map]
            assert max(on) - min(on) <= 1, (n_maps, T, entry.label, on)
            assert all(m.category == entry.map for m in alloc.maps if e in m.counts)
        again = allocate(cfg_n, pool, np.random.default_rng(0))
        assert [m.path for m in again.maps] == [m.path for m in alloc.maps] and again.maps == alloc.maps
    print(alloc.table())
    try:
        allocate(dataclasses.replace(cfg, maps=MapsConfig(cfg.maps.dir, 100, 16)), pool, np.random.default_rng(0))
    except ConfigError as err:
        assert "needs" in str(err)
    else:
        raise AssertionError("a pool too small must raise")
    try:
        allocate(dataclasses.replace(cfg, maps=MapsConfig(cfg.maps.dir, 2, 16)), pool, np.random.default_rng(0))
    except ConfigError as err:
        assert "no map at its share" in str(err)
    else:
        raise AssertionError("a category rounded to 0 maps must raise")
    assert largest_remainder([33.34, 33.33, 33.33], 100) == [34, 33, 33]
    print("[allocate] per-map totals, entry sums, floor/ceil spread, reproducibility, pool/0-map errors ok")
    print("all self-checks ok")
