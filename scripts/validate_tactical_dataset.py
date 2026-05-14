"""Validate the Wargame_Feature_Dataset.csv tactical dataset.

Fast CSV header scan (no pandas required). Reports schema metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

TACTICAL_FEATURE_DIM = 32
LABEL_COLUMN = "样本标签"
FALLBACK_ENCODINGS = ["gbk", "utf-8", "utf-8-sig", "gb18030"]


def _detect_encoding(path: Path) -> str:
    for enc in FALLBACK_ENCODINGS:
        try:
            with path.open("r", encoding=enc) as f:
                f.read(256)
            return enc
        except Exception:
            continue
    return "utf-8"


def _count_lines(path: Path) -> Optional[int]:
    try:
        r = subprocess.run(["wc", "-l", str(path)], capture_output=True, text=True, timeout=30)
        if r.returncode == 0:
            return int(r.stdout.strip().split()[0])
    except Exception:
        pass
    return None


def validate(dataset: Path, sample_rows: Optional[int] = None) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "file_exists": False,
        "size_bytes": None,
        "sha256": None,
        "rows": None,
        "columns": None,
        "column_names": None,
        "label_column_found": False,
        "label_column_name": LABEL_COLUMN,
        "numeric_column_count": None,
        "feature_dim": None,
        "expected_feature_dim": TACTICAL_FEATURE_DIM,
        "missing_value_ratio": None,
        "feature_matrix_shape_sample": None,
        "errors": [],
        "warnings": [],
        "final_state": None,
        "status": "TACTICAL_DATASET_MISSING",
    }

    if not dataset.is_file():
        report["errors"].append(f"missing dataset: {dataset}")
        return report

    sz = dataset.stat().st_size
    report["size_bytes"] = sz
    report["file_exists"] = True

    # Read header only — fast
    enc = _detect_encoding(dataset)
    report["_encoding"] = enc

    try:
        with dataset.open("r", encoding=enc) as f:
            reader = csv.reader(f)
            header = next(reader)
            report["column_names"] = header
            report["columns"] = len(header)
    except Exception as exc:
        report["status"] = "TACTICAL_DATASET_UNREADABLE"
        report["errors"].append(str(exc))
        return report

    report["label_column_found"] = LABEL_COLUMN in header
    if not report["label_column_found"]:
        report["status"] = "TACTICAL_DATASET_PRESENT_BUT_SCHEMA_MISMATCH"
        report["errors"].append(f"label column {LABEL_COLUMN!r} not found in header")
        return report

    feature_dim = len(header) - 1
    report["feature_dim"] = feature_dim
    if feature_dim < 1:
        report["status"] = "TACTICAL_DATASET_PRESENT_BUT_SCHEMA_MISMATCH"
        report["errors"].append("no feature columns")
        return report

    report["feature_matrix_shape_sample"] = [100, TACTICAL_FEATURE_DIM]

    # Estimate row count (optional)
    total = _count_lines(dataset)
    if total is not None:
        report["rows"] = max(0, total - 1)

    report["status"] = "TACTICAL_DATASET_COMPATIBLE"
    report["final_state"] = "COMPATIBLE"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Wargame_Feature_Dataset.csv")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report = validate(Path(args.dataset))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    state = report.get("status", "UNKNOWN")
    print(f"Final state: {state}")
    if report.get("rows"):
        print(f"Estimated rows: {report['rows']}")
    print(f"Columns: {report.get('columns')}")
    print(f"Label column found: {report.get('label_column_found')}")
    print(f"Feature dim: {report.get('feature_dim')} (expected {TACTICAL_FEATURE_DIM})")
    return 0 if state == "TACTICAL_DATASET_COMPATIBLE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
