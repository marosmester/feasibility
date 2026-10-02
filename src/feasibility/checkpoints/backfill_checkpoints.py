"""One-off: uploads the `.pt` checkpoints that already exist as W&B model artifacts.

Each file goes to the project it was trained under -- read off the checkpoint's own stored
`args["wandb_project"]` / `args["wandb_entity"]`, falling back to the dataset-file prefix
(`artifacts.PROJECT_BY_PREFIX`) -- so a lattice checkpoint lands in `feasibility-arc-divergence`
and an MPPI one in `feasibility-mppi-window-divergence`, exactly where a fresh training run would
put it. The artifact is named after the file stem. W&B deduplicates by content hash, so re-running
this creates no new versions.

A back-filled artifact has no lineage (it is logged by a `backfill` run, not by the training run
that produced it); the file's real modification time is stored as `created` in its metadata so the
training date is not lost.

CLI parameters:
    paths             checkpoint files (default: every outputs/checkpoints/*.pt)
    --entity STR      W&B entity overriding each checkpoint's own (default: from the checkpoint)
    --alias STR       extra alias added to every upload besides `latest` (repeatable)
    --dry-run         print where each file would go, upload nothing

Usage:
    python src/feasibility/checkpoints/backfill_checkpoints.py --dry-run
    python src/feasibility/checkpoints/backfill_checkpoints.py
    python src/feasibility/checkpoints/backfill_checkpoints.py outputs/checkpoints/x.pt --alias prod
"""

import argparse
import pathlib

import wandb

from feasibility.checkpoints.artifacts import ARTIFACT_TYPE
from feasibility.checkpoints.artifacts import artifact_name
from feasibility.checkpoints.artifacts import load_meta
from feasibility.checkpoints.artifacts import mtime_iso
from feasibility.checkpoints.artifacts import project_of
from feasibility.comparator.common import OUT_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("paths", type=pathlib.Path, nargs="*", help="default: outputs/checkpoints/*.pt")
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--alias", type=str, action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = args.paths or sorted((OUT_DIR / "checkpoints").glob("*.pt"))
    if not paths:
        parser.error("no checkpoints found")

    uploads: dict[tuple[str | None, str | None], list[tuple[pathlib.Path, dict]]] = {}
    for path in paths:
        meta = load_meta(path)
        project, entity = project_of(path, meta)
        if project is None:
            print(f"[skip] {path.name}: no wandb_project in its args and no known filename prefix")
            continue
        entity = args.entity or entity
        meta["created"] = mtime_iso(path)
        print(f"[plan] {path.name} -> {entity or '<default entity>'}/{project}/{artifact_name(path)}")
        uploads.setdefault((project, entity), []).append((path, meta))
    if args.dry_run:
        return

    for (project, entity), items in uploads.items():
        run = wandb.init(project=project, entity=entity, job_type="backfill", name="backfill-checkpoints")
        try:
            for path, meta in items:
                artifact = wandb.Artifact(artifact_name(path), type=ARTIFACT_TYPE, metadata=meta)
                artifact.add_file(str(path), name=path.name)
                run.log_artifact(artifact, aliases=["latest", *args.alias])
                print(f"[done] {path.name} -> {project}")
        finally:
            run.finish()


if __name__ == "__main__":
    main()
