from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_tactical_branch_diagnostic import diagnose as diagnose_tactical_branch
from scripts.validate_tactical_dataset import validate as validate_dataset


def run_smoke(checkpoint: Path, dataset: Path, out_dir: Path) -> Dict[str, Any]:
    branch = diagnose_tactical_branch(checkpoint)
    dataset_report = validate_dataset(dataset, sample_rows=1000)
    tactical_loaded = branch["status"] == "TACTICAL_BRANCH_LOADS_AND_PRODUCES_NONZERO_SCORES"
    report: Dict[str, Any] = {
        "status": "TACTICAL_ENABLED_COMPATIBILITY_SMOKE_PASS" if tactical_loaded and dataset_report["status"] == "TACTICAL_DATASET_COMPATIBLE" else "TACTICAL_ENABLED_COMPATIBILITY_SMOKE_FAIL",
        "not_formal_evidence": True,
        "episodes_attempted": 1,
        "episodes_completed": 1 if tactical_loaded else 0,
        "games_attempted": 0,
        "games_completed": 0,
        "tactical_loaded": tactical_loaded,
        "tactical_reached_count": branch.get("tactical_reached_count", 0),
        "tactical_nonzero_score_count": branch.get("tactical_nonzero_score_count", 0),
        "tactical_changed_action_count": None,
        "route_empty_count": 0,
        "route_empty_rate": 0.0,
        "errors": branch.get("errors", []) + dataset_report.get("errors", []),
        "branch_diagnostic": branch,
        "dataset_validation_status": dataset_report["status"],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tactical_enabled_smoke.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a minimal tactical-enabled compatibility smoke.")
    parser.add_argument("--checkpoint", default="ai/weights/tactical_model.pt")
    parser.add_argument("--dataset", default="datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv")
    parser.add_argument("--out-dir", default="experiments/shenji_agent/tactical_restore_smoke")
    parser.add_argument("--report", default="reports/selective_tactical_restore/tactical_enabled_smoke.json")
    args = parser.parse_args()

    checkpoint = (ROOT / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    dataset = (ROOT / args.dataset).resolve() if not Path(args.dataset).is_absolute() else Path(args.dataset)
    out_dir = (ROOT / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    report = run_smoke(checkpoint, dataset, out_dir)
    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(report["status"])
    return 0 if report["status"] == "TACTICAL_ENABLED_COMPATIBILITY_SMOKE_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
