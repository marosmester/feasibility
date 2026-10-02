"""Shared W&B-artifact helpers for `.pt` checkpoints: upload one, resolve its project.

`outputs/` is gitignored, so checkpoints reach other machines through W&B Artifacts instead. Each
trainer logs its best checkpoint to ITS OWN W&B project (the `--wandb-project` it ran under), as a
`model` artifact named after the file stem -- the stem already encodes dataset, map count, seed and
variant (`_vwz`, `_rpy`, ...), and W&B artifact names accept exactly that character set. Metadata is
read off the checkpoint itself (it is self-describing), so the W&B UI can filter by it.

`log_checkpoint` is what the five `train.py` scripts call; `backfill_checkpoints.py` and
`fetch_checkpoint.py` are the other two ends. Neutral package: imports nothing from the model trees.
"""

import pathlib
import re
from datetime import datetime
from datetime import timezone

import torch
import wandb

ARTIFACT_TYPE = "model"

# Fallback for a checkpoint with no `args` (or none naming a project): by dataset-file prefix. The
# projects are the `--wandb-project` defaults of the corresponding train.py.
PROJECT_BY_PREFIX = {
    "dataset_arc_": "feasibility-arc-divergence",
    "dataset_mppi_": "feasibility-mppi-window-divergence",
    "dataset_grid_": "feasibility-grid-pose-error",
    "dataset_patch_": "feasibility-pose-error-mlp",
    "dataset_": "feasibility-pose-error-mlp",
}

_BAD_NAME_CHARS = re.compile(r"[^A-Za-z0-9_.-]")

# Checkpoint keys worth showing in the W&B UI; the rest (state dicts, tensors) are not metadata.
METADATA_KEYS = (
    "epoch", "val_loss", "base_width", "embed_dim", "head_fusion", "command_mode", "label_mode",
    "blur_terrain", "drop_blocked_endpoints", "drop_endpoint_infeasible", "drop_twin_flagged",
    "target_names", "dataset_path", "dataset_paths", "label_attrs", "v_scale", "wz_scale",
)


def artifact_name(path: pathlib.Path) -> str:
    return _BAD_NAME_CHARS.sub("_", path.stem)


def load_meta(path: pathlib.Path) -> dict:
    """The checkpoint's own description, as plain JSON-able values (no weights are kept)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta: dict = {}
    for key in METADATA_KEYS:
        if key in ckpt:
            value = ckpt[key]
            meta[key] = list(value) if isinstance(value, tuple) else value
    args = ckpt.get("args")
    if isinstance(args, dict):
        meta["args"] = {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in args.items()}
    for key in ("dataset_git", "train_git"):
        if key in ckpt:
            meta[key] = ckpt[key]
    return {k: v for k, v in meta.items() if _json_safe(v)}


def _json_safe(value) -> bool:
    if isinstance(value, (str, int, float, bool, type(None))):
        return True
    if isinstance(value, (list, tuple)):
        return all(_json_safe(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _json_safe(v) for k, v in value.items())
    return False


def project_of(path: pathlib.Path, meta: dict) -> tuple[str | None, str | None]:
    """`(project, entity)` the checkpoint was trained under, from its stored args, else its name."""
    args = meta.get("args", {})
    if args.get("wandb_project"):
        return args["wandb_project"], args.get("wandb_entity")
    for prefix, project in PROJECT_BY_PREFIX.items():
        if path.name.startswith(prefix):
            return project, None
    return None, None


def log_checkpoint(
    run: wandb.sdk.wandb_run.Run | None,
    path: pathlib.Path | None,
    aliases: list[str] | None = None,
    metadata: dict | None = None,
) -> wandb.Artifact | None:
    """Uploads `path` as a model artifact of `run` (so it carries the run's config and lineage).

    A no-op for a disabled/offline-less run or a missing checkpoint (a trainer returns None when no
    epoch ever improved), so callers need no guard."""
    if run is None or path is None or getattr(run, "disabled", False):
        return None
    path = pathlib.Path(path)
    if not path.exists():
        return None
    artifact = wandb.Artifact(
        artifact_name(path), type=ARTIFACT_TYPE, metadata={**load_meta(path), **(metadata or {})}
    )
    artifact.add_file(str(path), name=path.name)
    run.log_artifact(artifact, aliases=aliases)
    print(f"[artifact] logged {path.name} to {run.project}/{artifact.name}")
    return artifact


def mtime_iso(path: pathlib.Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
