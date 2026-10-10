from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .normalizer import CANONICAL_TABLES
from .ontology import DEFAULT_ONTOLOGY_PATH, load_ontology
from .validation import validate_ontology, validate_tables


SUPPORTED_FORMATS = {"jsonl", "parquet"}


# Version constants — single source of truth for each release.
# Bump these before generating a candidate or release.
DATASET_VERSION = "0.4.0-candidate"
SCHEMA_VERSION = "2.0.0"
ONTOLOGY_VERSION = "2.0.0"
LABELER_VERSION = "0.4.0"
SPLIT_VERSION = "0.1.0"


_SUBPROCESS_COMMIT: str | None = None


def _get_code_commit() -> str:
    global _SUBPROCESS_COMMIT
    if _SUBPROCESS_COMMIT is None:
        import subprocess
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
            )
            commit = result.stdout.strip()
            # 40-char hex SHA-1
            if len(commit) == 40 and all(c in "0123456789abcdef" for c in commit):
                _SUBPROCESS_COMMIT = commit
            else:
                _SUBPROCESS_COMMIT = "unknown"
        except Exception:
            _SUBPROCESS_COMMIT = "unknown"
    return _SUBPROCESS_COMMIT


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            stream.write("\n")


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Parquet output requires the optional 'parquet' dependency: pip install -e .[parquet]"
        ) from exc
    table = pa.Table.from_pylist(rows) if rows else pa.table({})
    pq.write_table(table, path, compression="zstd")


def write_release(
    tables: dict[str, list[dict[str, Any]]],
    output_dir: Path | str,
    formats: tuple[str, ...] = ("jsonl",),
) -> dict[str, Any]:
    """Write canonical tables and a deterministic integrity manifest."""

    normalized_formats = tuple(dict.fromkeys(value.lower() for value in formats))
    if not normalized_formats:
        raise ValueError("at least one release format is required")
    unsupported = sorted(set(normalized_formats) - SUPPORTED_FORMATS)
    if unsupported:
        raise ValueError(f"unsupported release format: {unsupported}")
    missing_tables = sorted(set(CANONICAL_TABLES) - set(tables))
    if missing_tables:
        raise ValueError(f"missing canonical tables: {missing_tables}")
    for name in CANONICAL_TABLES:
        if not isinstance(tables[name], list):
            raise TypeError(f"canonical table {name} must be a list")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []
    for table_name in CANONICAL_TABLES:
        rows = tables[table_name]
        for output_format in normalized_formats:
            path = output / f"{table_name}.{output_format}"
            if output_format == "jsonl":
                _write_jsonl(path, rows)
            else:
                _write_parquet(path, rows)
            artifacts.append(
                {
                    "table": table_name,
                    "format": output_format,
                    "path": path.name,
                    "rows": len(rows),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )

    # Compute ontology_sha256 from the ontology file actually loaded
    onto_path = Path(DEFAULT_ONTOLOGY_PATH)
    onto_sha256 = _sha256(onto_path) if onto_path.is_file() else "unknown"

    manifest = {
        "dataset": "Shenji",
        "dataset_version": DATASET_VERSION,
        "schema_version": SCHEMA_VERSION,
        "ontology_version": ONTOLOGY_VERSION,
        "ontology_sha256": onto_sha256,
        "labeler_version": LABELER_VERSION,
        "split_version": SPLIT_VERSION,
        "code_commit": _get_code_commit(),
        "validation_mode": "candidate" if "-candidate" in DATASET_VERSION else "release",
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "table_order": list(CANONICAL_TABLES),
        "formats": list(normalized_formats),
        "artifacts": artifacts,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _jsonl_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def _parquet_row_count(path: Path) -> int | None:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None
    return pq.read_metadata(path).num_rows


def verify_release(output_dir: Path | str) -> list[str]:
    """Verify release paths, byte sizes, row counts, and SHA-256 hashes."""

    output = Path(output_dir)
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        return ["manifest.json is missing"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return [f"manifest.json is invalid: {exc}"]
    errors: list[str] = []
    if manifest.get("dataset") != "Shenji":
        errors.append("manifest dataset must be Shenji")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"manifest schema_version must be {SCHEMA_VERSION}, got {manifest.get('schema_version')}")
    if manifest.get("dataset_version") != DATASET_VERSION:
        errors.append(f"manifest dataset_version must be {DATASET_VERSION}, got {manifest.get('dataset_version')}")
    if manifest.get("ontology_version") != ONTOLOGY_VERSION:
        errors.append(f"manifest ontology_version must be {ONTOLOGY_VERSION}, got {manifest.get('ontology_version')}")
    if manifest.get("code_commit") in (None, "", "unknown"):
        errors.append("manifest code_commit is missing or unknown")
    if manifest.get("ontology_sha256") in (None, "", "unknown"):
        errors.append("manifest ontology_sha256 is missing or unknown")
    else:
        # Recompute ontology hash locally and compare
        onto_path = Path(DEFAULT_ONTOLOGY_PATH)
        if onto_path.is_file():
            expected = _sha256(onto_path)
            if manifest["ontology_sha256"] != expected:
                errors.append(f"manifest ontology_sha256 {manifest['ontology_sha256']} does not match local file {expected}")
    if "-release" in manifest.get("dataset_version", "") and manifest.get("code_commit", "") == "unknown":
        errors.append("release manifest must have a valid code_commit, not 'unknown'")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        return [*errors, "manifest artifacts must be a list"]
    for artifact in artifacts:
        relative = artifact.get("path")
        path = output / str(relative)
        if not path.is_file():
            errors.append(f"artifact is missing: {relative}")
            continue
        actual_bytes = path.stat().st_size
        if actual_bytes != artifact.get("bytes"):
            errors.append(
                f"artifact byte count mismatch for {relative}: expected {artifact.get('bytes')}, got {actual_bytes}"
            )
        actual_sha = _sha256(path)
        if actual_sha != artifact.get("sha256"):
            errors.append(f"artifact SHA-256 mismatch for {relative}")
        output_format = artifact.get("format")
        if output_format == "jsonl":
            actual_rows = _jsonl_row_count(path)
        elif output_format == "parquet":
            actual_rows = _parquet_row_count(path)
        else:
            errors.append(f"unsupported artifact format in manifest: {output_format}")
            continue
        if actual_rows is not None and actual_rows != artifact.get("rows"):
            errors.append(
                f"artifact row count mismatch for {relative}: expected {artifact.get('rows')}, got {actual_rows}"
            )
    return errors


def load_release(output_dir: Path | str) -> dict[str, list[dict[str, Any]]]:
    """Load canonical tables, preferring dependency-free JSONL artifacts."""

    output = Path(output_dir)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts", [])
    tables: dict[str, list[dict[str, Any]]] = {}
    for table_name in CANONICAL_TABLES:
        candidates = [item for item in artifacts if item.get("table") == table_name]
        jsonl = next((item for item in candidates if item.get("format") == "jsonl"), None)
        if jsonl is not None:
            rows: list[dict[str, Any]] = []
            with (output / jsonl["path"]).open("r", encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        rows.append(json.loads(line))
            tables[table_name] = rows
            continue
        parquet = next((item for item in candidates if item.get("format") == "parquet"), None)
        if parquet is None:
            raise ValueError(f"manifest has no readable artifact for table {table_name}")
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("loading a Parquet-only release requires pyarrow") from exc
        tables[table_name] = pq.read_table(output / parquet["path"]).to_pylist()
    return tables


def validate_release(output_dir: Path | str) -> list[str]:
    """Run integrity, ontology, and cross-table validation for one release."""

    errors = verify_release(output_dir)
    if errors:
        return errors
    ontology = load_ontology()
    errors.extend(validate_ontology(ontology))
    errors.extend(validate_tables(load_release(output_dir), ontology))
    return errors
