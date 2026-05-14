"""Run a fail-closed tactical branch diagnostic.

Checks:
1. Checkpoint exists and validates
2. TacticalScorer loads and produces non-trivial outputs
3. Feature dimension compatibility

Does not require full game engine or selfplay.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_tactical_checkpoint import validate as validate_checkpoint


def _torch_available() -> bool:
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.__version__)"],
            capture_output=True, text=True, timeout=15,
        )
        return result.returncode == 0
    except Exception:
        return False


def _probe_with_torch(model_state, input_dim, hidden_dim, report):
    import torch
    from ai.features import TACTICAL_FEATURE_DIM
    from ai.models import TacticalScorer

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
    nonzero = sum(1 for score in scores if abs(float(score)) > 1e-9)
    report["tactical_nonzero_score_count"] = nonzero
    if nonzero == 0:
        report["reason_if_zero"] = "model produced only zero logits for deterministic feature probes"
        report["status"] = "TACTICAL_BRANCH_LOADS_BUT_ZERO_SCORES"
    elif report["tactical_feature_builder_available"]:
        report["status"] = "TACTICAL_BRANCH_LOADS_AND_PRODUCES_NONZERO_SCORES"
    else:
        report["status"] = "TACTICAL_BRANCH_BLOCKED_BY_FEATURE_SCHEMA"
        report["reason_if_zero"] = f"checkpoint input_dim={input_dim} does not match TACTICAL_FEATURE_DIM"
    return report


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

    if not _torch_available():
        report["status"] = "TACTICAL_BRANCH_BLOCKED_BY_CHECKPOINT_INCOMPATIBILITY"
        report["errors"].append("torch not available for model probing")
        report["partial_state"] = validation
        return report

    try:
        import torch
        model_state_shapes = validation.get("model_state_shapes", {})
        input_dim = int(model_state_shapes.get("model.0.weight", [None, 32])[1] or 32)
        hidden_dim = int(model_state_shapes["model.0.weight"][0])

        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        report = _probe_with_torch(payload["model_state"], input_dim, hidden_dim, report)
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
