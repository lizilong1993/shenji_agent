from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_load(path: Path) -> Tuple[Optional[Any], Optional[str], bool]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        return None, f"torch_unavailable: {exc}", False

    try:
        return torch.load(path, map_location="cpu", weights_only=True), None, True
    except TypeError:
        try:
            return torch.load(path, map_location="cpu"), None, False
        except Exception as exc:
            return None, f"torch_load_failed_without_weights_only: {exc}", False
    except Exception as exc:
        return None, f"torch_load_failed: {exc}", True


def validate(checkpoint: Path) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "exists": checkpoint.exists(),
        "status": "TACTICAL_CHECKPOINT_MISSING",
        "errors": [],
        "warnings": [],
    }
    if not checkpoint.exists():
        report["errors"].append(f"missing checkpoint: {checkpoint}")
        return report

    report["size_bytes"] = checkpoint.stat().st_size
    report["sha256"] = _sha256(checkpoint)

    payload, load_error, used_weights_only = _torch_load(checkpoint)
    report["torch_weights_only_used"] = used_weights_only
    if load_error:
        report["status"] = "TACTICAL_CHECKPOINT_UNREADABLE"
        report["errors"].append(load_error)
        return report

    report["top_level_type"] = type(payload).__name__
    if not isinstance(payload, dict):
        report["status"] = "TACTICAL_CHECKPOINT_PRESENT_BUT_INCOMPATIBLE"
        report["errors"].append("checkpoint payload is not a dict")
        return report

    report["top_level_keys"] = sorted(str(key) for key in payload.keys())
    model_state = payload.get("model_state")
    if not isinstance(model_state, dict):
        report["status"] = "TACTICAL_CHECKPOINT_PRESENT_BUT_INCOMPATIBLE"
        report["errors"].append("checkpoint payload missing dict model_state")
        return report

    shapes = {str(key): list(value.shape) for key, value in model_state.items() if hasattr(value, "shape")}
    report["model_state_shapes"] = shapes
    required = {
        "model.0.weight": [None, int(payload.get("input_dim", 32))],
        "model.0.bias": [None],
        "model.3.weight": [None, None],
        "model.3.bias": [None],
        "model.5.weight": [1, None],
        "model.5.bias": [1],
    }
    missing = [key for key in required if key not in shapes]
    if missing:
        report["status"] = "TACTICAL_CHECKPOINT_PRESENT_BUT_INCOMPATIBLE"
        report["errors"].append(f"missing TacticalScorer state keys: {missing}")
        return report

    try:
        from ai.models import TacticalScorer

        input_dim = int(payload.get("input_dim", shapes["model.0.weight"][1]))
        hidden_dim = int(shapes["model.0.weight"][0])
        model = TacticalScorer(input_dim=input_dim, hidden_dim=hidden_dim)
        model.load_state_dict(model_state)
        model.eval()
        report["tactical_model_class"] = "ai.models.TacticalScorer"
        report["input_dim"] = input_dim
        report["hidden_dim"] = hidden_dim
    except Exception as exc:
        report["status"] = "TACTICAL_CHECKPOINT_PRESENT_BUT_INCOMPATIBLE"
        report["errors"].append(f"TacticalScorer load failed: {exc}")
        return report

    report["status"] = "TACTICAL_CHECKPOINT_COMPATIBLE"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a restored tactical checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report = validate((ROOT / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint))
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(report["status"])
    return 0 if report["status"] == "TACTICAL_CHECKPOINT_COMPATIBLE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
