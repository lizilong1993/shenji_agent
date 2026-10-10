from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from .collection import CollectionDependencies, run_collection_job
from .coverage import build_coverage_report
from .jobs import CollectionJob, expand_jobs, load_collection_manifest
from .runtime import (
    load_miaosuan_assets,
    resolve_agent_class,
    resolve_miaosuan_env_class,
)


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(_json_text(value) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _dataset_root(manifest_path: Path, override: Path | None) -> Path:
    if override is not None:
        return override.resolve()
    if manifest_path.parent.name == "configs":
        return manifest_path.parent.parent.resolve()
    return manifest_path.parent.resolve()


def _load_context(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.resolve()
    manifest = load_collection_manifest(manifest_path)
    jobs = expand_jobs(manifest)
    dataset_root = _dataset_root(manifest_path, getattr(args, "dataset_root", None))
    project_root = (
        args.project_root.resolve()
        if getattr(args, "project_root", None) is not None
        else dataset_root.parent.resolve()
    )
    return {
        "manifest_path": manifest_path,
        "manifest": manifest,
        "jobs": jobs,
        "dataset_root": dataset_root,
        "project_root": project_root,
        "data_root": _resolve(dataset_root, manifest["data_root"]),
        "output_root": _resolve(dataset_root, manifest["output_root"]),
        "canonical_root": _resolve(
            dataset_root,
            manifest.get("canonical_root", "canonical/shenji_schema_v1/pilot_v1"),
        ),
    }


def _resolve_artifact_paths(job: CollectionJob, dataset_root: Path) -> CollectionJob:
    red = (
        str(_resolve(dataset_root, job.red_artifacts_dir))
        if job.red_artifacts_dir is not None
        else None
    )
    blue = (
        str(_resolve(dataset_root, job.blue_artifacts_dir))
        if job.blue_artifacts_dir is not None
        else None
    )
    return replace(job, red_artifacts_dir=red, blue_artifacts_dir=blue)


def dependencies_for_job(
    job: CollectionJob, project_root: Path, data_root: Path
) -> CollectionDependencies:
    env_class = resolve_miaosuan_env_class(project_root)
    red_class = resolve_agent_class(job.red_agent_class, project_root)
    blue_class = resolve_agent_class(job.blue_agent_class, project_root)
    return CollectionDependencies(
        env_factory=env_class,
        red_agent_factory=red_class,
        blue_agent_factory=blue_class,
        asset_loader=lambda current_job: load_miaosuan_assets(current_job, data_root),
    )


def _asset_probe(jobs: list[CollectionJob], data_root: Path) -> dict[str, Any]:
    missing: set[str] = set()
    checked: set[str] = set()
    for job in jobs:
        for relative in (
            job.scenario_path,
            job.basic_path,
            job.cost_path,
            job.see_path,
        ):
            path = (data_root / relative).resolve()
            checked.add(str(path))
            if not path.is_file():
                missing.add(str(path))
    return {
        "ok": not missing,
        "checked_count": len(checked),
        "missing": sorted(missing),
    }


def _probe(args: argparse.Namespace) -> int:
    context = _load_context(args)
    checks: dict[str, Any] = {}
    try:
        env_class = resolve_miaosuan_env_class(context["project_root"])
        checks["environment"] = {
            "ok": True,
            "class": f"{env_class.__module__}:{env_class.__name__}",
        }
    except Exception as exc:
        checks["environment"] = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    agent_results: dict[str, Any] = {}
    for specification in sorted(
        {
            job.red_agent_class
            for job in context["jobs"]
        }
        | {job.blue_agent_class for job in context["jobs"]}
    ):
        try:
            agent_class = resolve_agent_class(specification, context["project_root"])
            agent_results[specification] = {
                "ok": True,
                "class": f"{agent_class.__module__}:{agent_class.__name__}",
            }
        except Exception as exc:
            agent_results[specification] = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    checks["agents"] = agent_results
    checks["assets"] = _asset_probe(context["jobs"], context["data_root"])
    ready = (
        checks["environment"]["ok"]
        and all(item["ok"] for item in agent_results.values())
        and checks["assets"]["ok"]
    )
    result = {
        "dataset": "Shenji",
        "ready": ready,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "project_root": str(context["project_root"]),
        "data_root": str(context["data_root"]),
        "job_count": len(context["jobs"]),
        "checks": checks,
    }
    if args.output is not None:
        _write_json(args.output, result)
    print(_json_text(result))
    return 0 if ready else 1


def _run(args: argparse.Namespace) -> int:
    context = _load_context(args)
    jobs = context["jobs"]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be a positive integer")
        jobs = jobs[: args.limit]
    if args.dry_run:
        print(
            _json_text(
                {
                    "dataset": "Shenji",
                    "job_count": len(jobs),
                    "output_root": str(context["output_root"]),
                    "canonical_root": str(context["canonical_root"]),
                    "jobs": [job.to_dict() for job in jobs],
                }
            )
        )
        return 0

    failed = 0
    for original_job in jobs:
        job = _resolve_artifact_paths(original_job, context["dataset_root"])
        dependencies = dependencies_for_job(
            job, context["project_root"], context["data_root"]
        )
        status = run_collection_job(
            job,
            dependencies,
            context["output_root"],
            canonical_root=context["canonical_root"],
            resume=not args.no_resume,
        )
        print(json.dumps(status, ensure_ascii=False, sort_keys=True))
        if status.get("state") != "completed":
            failed += 1
    return 1 if failed else 0


def _coverage(args: argparse.Namespace) -> int:
    targets = None
    if args.targets is not None:
        with args.targets.open("r", encoding="utf-8") as stream:
            targets = json.load(stream)
    report = build_coverage_report(args.raw_root, targets=targets)
    if args.output is not None:
        _write_json(args.output, report)
    print(_json_text(report))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shenji-collect",
        description="Run and audit reproducible Shenji pilot collection jobs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe = subparsers.add_parser("probe", help="check real MiaoSuan runtime and assets")
    probe.add_argument("--manifest", type=Path, required=True)
    probe.add_argument("--dataset-root", type=Path)
    probe.add_argument("--project-root", type=Path)
    probe.add_argument("--output", type=Path)
    probe.set_defaults(handler=_probe)

    run = subparsers.add_parser("run", help="run or resume pilot collection")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--dataset-root", type=Path)
    run.add_argument("--project-root", type=Path)
    run.add_argument("--limit", type=int)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--no-resume", action="store_true")
    run.set_defaults(handler=_run)

    coverage = subparsers.add_parser("coverage", help="summarize pilot label coverage")
    coverage.add_argument("--raw-root", type=Path, required=True)
    coverage.add_argument("--targets", type=Path)
    coverage.add_argument("--output", type=Path)
    coverage.set_defaults(handler=_coverage)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, ValueError, RuntimeError, ImportError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
