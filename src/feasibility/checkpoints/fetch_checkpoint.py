"""Downloads checkpoints from W&B Artifacts into `outputs/checkpoints/`.

`name` is the artifact name -- the checkpoint's file stem -- optionally `:alias` or `:vN`
(default `:latest`). The project is where it was logged: give `--project`, or omit it to search
the known trainer projects (`artifacts.PROJECT_BY_PREFIX`) for the first that holds that name.
Existing files are kept unless `--force`.

`--all` instead fetches the `:latest` of every model artifact in the searched project(s) --
i.e. every checkpoint found in those W&B projects that isn't already in `--out-dir` (existing
files are still kept unless `--force`, so a plain re-run only pulls the new/missing ones).

CLI parameters:
    name              artifact name, `stem`, `stem:v3` or `stem:prod` (omit with --all/--list)
    --all             fetch every artifact's `:latest` from --project (or every known project)
    --project STR     W&B project (default: searched, see above)
    --entity STR      W&B entity (default: your default entity)
    --out-dir PATH    where to put the file (default: outputs/checkpoints)
    --force           overwrite a file that is already there
    --list            list the model artifacts of --project (or every known project) and exit

Usage:
    python src/feasibility/checkpoints/fetch_checkpoint.py --list
    python src/feasibility/checkpoints/fetch_checkpoint.py dataset_arc_my_config_M300_R8_seed0_vwz_rpy
    python src/feasibility/checkpoints/fetch_checkpoint.py dataset_arc_my_config_M300_R8_seed0_vwz_rpy:v1 --project feasibility-arc-divergence
    python src/feasibility/checkpoints/fetch_checkpoint.py --all
"""

import argparse
import pathlib
import shutil
import tempfile

import wandb

from feasibility.checkpoints.artifacts import ARTIFACT_TYPE
from feasibility.checkpoints.artifacts import PROJECT_BY_PREFIX
from feasibility.comparator.common import OUT_DIR


def projects_to_search(project: str | None) -> list[str]:
    return [project] if project else list(dict.fromkeys(PROJECT_BY_PREFIX.values()))


def download_artifact(artifact: wandb.Artifact, out_dir: pathlib.Path, force: bool) -> None:
    """Moves every `.pt` of `artifact` into `out_dir`, keeping an existing file unless `force`."""
    with tempfile.TemporaryDirectory() as tmp:
        artifact.download(root=tmp)
        for file in sorted(pathlib.Path(tmp).glob("*.pt")):
            destination = out_dir / file.name
            if destination.exists() and not force:
                print(f"[keep] {destination} exists (--force to overwrite)")
                continue
            shutil.move(str(file), destination)
            print(f"[done] {artifact.source_name} ({artifact.project}) -> {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("name", nargs="?", help="artifact name, optionally :alias or :vN")
    parser.add_argument("--all", action="store_true", help="fetch every artifact's :latest")
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR / "checkpoints")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    api = wandb.Api()
    entity = args.entity or api.default_entity

    if args.list:
        for project in projects_to_search(args.project):
            try:
                collections = api.artifact_type(ARTIFACT_TYPE, f"{entity}/{project}").collections()
                for collection in collections:
                    print(f"{project}/{collection.name}")
            except wandb.errors.CommError:
                print(f"[skip] {project}: not found")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        if args.name:
            parser.error("name and --all are mutually exclusive")
        for project in projects_to_search(args.project):
            try:
                collections = api.artifact_type(ARTIFACT_TYPE, f"{entity}/{project}").collections()
            except wandb.errors.CommError:
                print(f"[skip] {project}: not found")
                continue
            for collection in collections:
                try:
                    artifact = api.artifact(f"{entity}/{project}/{collection.name}:latest", type=ARTIFACT_TYPE)
                except wandb.errors.CommError:
                    print(f"[skip] {project}/{collection.name}: no :latest")
                    continue
                download_artifact(artifact, args.out_dir, args.force)
        return

    if not args.name:
        parser.error("name is required (or --all / --list)")

    name = args.name if ":" in args.name else f"{args.name}:latest"
    artifact = None
    for project in projects_to_search(args.project):
        try:
            artifact = api.artifact(f"{entity}/{project}/{name}", type=ARTIFACT_TYPE)
            break
        except wandb.errors.CommError:
            continue
    if artifact is None:
        parser.error(f"{name} not found in {projects_to_search(args.project)} under {entity}")

    download_artifact(artifact, args.out_dir, args.force)


if __name__ == "__main__":
    main()
