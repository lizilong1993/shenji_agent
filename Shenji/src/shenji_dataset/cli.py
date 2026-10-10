from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .normalizer import normalize_episode
from .ontology import load_ontology
from .release import validate_release, verify_release, write_release
from .validation import validate_ontology, validate_tables


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _print_errors(errors: list[str]) -> None:
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)


def _build(args: argparse.Namespace) -> int:
    raw_episode = _read_json(args.raw)
    intent_plan = _read_json(args.intent_plan) if args.intent_plan else None
    tables = normalize_episode(raw_episode, intent_plan)
    errors = validate_ontology(load_ontology())
    errors.extend(validate_tables(tables, load_ontology()))
    if errors:
        _print_errors(errors)
        return 1
    manifest = write_release(tables, args.output, formats=tuple(args.formats))
    print(
        f"built Shenji release with {len(manifest['artifacts'])} artifacts at {args.output}"
    )
    return 0


def _verify(args: argparse.Namespace) -> int:
    errors = verify_release(args.release)
    if errors:
        _print_errors(errors)
        return 1
    print("release verification passed")
    return 0


def _validate(args: argparse.Namespace) -> int:
    errors = validate_release(args.release)
    if errors:
        _print_errors(errors)
        return 1
    print("dataset validation passed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shenji-dataset",
        description="Build and validate Shenji dataset releases.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="normalize one raw episode into a release")
    build.add_argument("--raw", type=Path, required=True, help="raw_episode.json")
    build.add_argument("--intent-plan", type=Path, help="optional intent_plan.json")
    build.add_argument("--output", type=Path, required=True, help="release directory")
    build.add_argument(
        "--format",
        dest="formats",
        action="append",
        choices=("jsonl", "parquet"),
        help="output format; repeat to request both",
    )
    build.set_defaults(handler=_build)

    verify = subparsers.add_parser("verify", help="verify artifact hashes and row counts")
    verify.add_argument("release", type=Path)
    verify.set_defaults(handler=_verify)

    validate = subparsers.add_parser("validate", help="validate a complete release")
    validate.add_argument("release", type=Path)
    validate.set_defaults(handler=_validate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "build" and args.formats is None:
        args.formats = ["jsonl"]
    try:
        return int(args.handler(args))
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
