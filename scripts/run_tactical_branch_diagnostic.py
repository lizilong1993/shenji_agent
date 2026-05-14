from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_tactical_checkpoint import validate as validate_checkpoint


def diagnose(checkpoint: Path) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "tactical_checkpoint_resolved": checkpoint.exists(),
        "tactical_checkpoint_loaded": False,
        "tactical_model_class": None,
        "tactical_feature_builder_available": False,
        "tactical_candidate_count": 0,
        "tactical_reached_count": 0,
        "tactical_nonzero_score_count": 0,
        "tactical_changed_action_count": None,
        "reason_if_zero": None,
        "status": "TACTICAL_BRANCH_MISSING",
        "errors": [],
    }
    validation = validate_checkpoint(checkpoint)
    report["checkpoint_validation_status"] = validation["status"]
    if validation["status"] != "TACTICAL_CHECKPOINT_COMPATIBLE":
        report["status"] = "TACTICAL_BRANCH_BLOCKED_BY_CHECKPOINT_INCOMPATIBILITY"
        report["errors"].extend(validation.get("errors", []))
        return report

    try:
        import torch
        from ai.features import TACTICAL_FEATURE_DIM
        from ai.models import TacticalScorer

        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model_state = payload["model_state"]
        hidden_dim = int(model_state["model.0.weight"].shape[0])
        input_dim = int(payload.get("input_dim", TACTICAL_FEATURE_DIM))
        model = TacticalScorer(input_dim=input_dim, hidden_dim=hidden_dim)
        model.load_state_dict(model_state)
        model.eval()
        report["tactical_checkpoint_loaded"] = True
        report["tactical_model_class"] = "ai.models.TacticalScorer"
        report["tactical_feature_builder_available"] = input_dim == TACTICAL_FEATURE_DIM

        candidates = torch.eye(input_dim, dtype=torch.float32)[: min(input_dim, 16)]
        if len(candidates) == 0:
            candidates = torch.zeros((1, input_dim), dtype=torch.float32)
        with torch.no_grad():
            scores = model(candidates).reshape(-1).tolist()
        report["tactical_candidate_count"] = len(scores)
        report["tactical_reached_count"] = len(scores)
        report["score_sample"] = [float(score) for score in scores[:10]]
        report["tactical_nonzero_score_count"] = sum(1 for score in scores if abs(float(score)) > 1e-9)
        if report["tactical_nonzero_score_count"] == 0:
            report["reason_if_zero"] = "model produced only zero logits for deterministic feature probes"
            report["status"] = "TACTICAL_BRANCH_LOADS_BUT_ZERO_SCORES"
        elif report["tactical_feature_builder_available"]:
            report["status"] = "TACTICAL_BRANCH_LOADS_AND_PRODUCES_NONZERO_SCORES"
        else:
            report["status"] = "TACTICAL_BRANCH_BLOCKED_BY_FEATURE_SCHEMA"
            report["reason_if_zero"] = f"checkpoint input_dim={input_dim} does not match TACTICAL_FEATURE_DIM"
    except Exception as exc:
        report["status"] = "TACTICAL_BRANCH_BLOCKED_BY_CHECKPOINT_INCOMPATIBILITY"
        report["errors"].append(str(exc))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a fail-closed tactical branch diagnostic.")
    parser.add_argument("--checkpoint", default="ai/weights/tactical_model.pt")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    checkpoint = (ROOT / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    report = diagnose(checkpoint)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(report["status"])
    return 0 if report["status"] == "TACTICAL_BRANCH_LOADS_AND_PRODUCES_NONZERO_SCORES" else 2


if __name__ == "__main__":
    raise SystemExit(main())
