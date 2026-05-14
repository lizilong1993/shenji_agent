from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def validate(dataset: Path, *, sample_rows: int = 5000, label_column: str = "样本标签") -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "dataset": str(dataset),
        "exists": dataset.exists(),
        "status": "TACTICAL_DATASET_MISSING",
        "errors": [],
        "warnings": [],
    }
    if not dataset.exists():
        report["errors"].append(f"missing dataset: {dataset}")
        return report

    report["size_bytes"] = dataset.stat().st_size
    try:
        from training.data import TacticalPositionDataset, _read_csv_with_fallback
        from ai.features import TACTICAL_FEATURE_DIM

        header = _read_csv_with_fallback(dataset, nrows=0)
        report["column_count"] = int(len(header.columns))
        report["columns"] = [str(column) for column in header.columns]
        report["label_column"] = label_column
        report["label_column_present"] = label_column in header.columns
        if not report["label_column_present"]:
            report["status"] = "TACTICAL_DATASET_PRESENT_BUT_SCHEMA_MISMATCH"
            report["errors"].append(f"missing tactical label column: {label_column}")
            return report

        frame = _read_csv_with_fallback(dataset, nrows=sample_rows)
        report["sample_rows_checked"] = int(len(frame))
        report["missing_values_in_sample"] = int(frame.isna().sum().sum())
        report["label_distribution_sample"] = {
            str(key): int(value) for key, value in frame[label_column].value_counts(dropna=False).to_dict().items()
        }

        materialized = TacticalPositionDataset(dataset, label_column=label_column, max_rows=sample_rows)
        report["row_count_sample"] = int(len(materialized))
        report["feature_columns_sample"] = materialized.feature_columns
        report["feature_column_count_sample"] = int(len(materialized.feature_columns))
        report["feature_matrix_shape_sample"] = list(materialized.features.shape)
        report["expected_feature_dim"] = int(TACTICAL_FEATURE_DIM)
        report["label_distribution_parsed_sample"] = materialized.label_distribution
        if materialized.features.shape[1] != TACTICAL_FEATURE_DIM:
            report["status"] = "TACTICAL_DATASET_PRESENT_BUT_SCHEMA_MISMATCH"
            report["errors"].append(
                f"feature width {materialized.features.shape[1]} does not match {TACTICAL_FEATURE_DIM}"
            )
            return report
    except Exception as exc:
        report["status"] = "TACTICAL_DATASET_UNREADABLE"
        report["errors"].append(str(exc))
        return report

    report["status"] = "TACTICAL_DATASET_COMPATIBLE"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate restored tactical dataset schema.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sample-rows", type=int, default=5000)
    args = parser.parse_args()

    report = validate(
        (ROOT / args.dataset).resolve() if not Path(args.dataset).is_absolute() else Path(args.dataset),
        sample_rows=args.sample_rows,
    )
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(report["status"])
    return 0 if report["status"] == "TACTICAL_DATASET_COMPATIBLE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
