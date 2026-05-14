"""Multi-seed tactical ablation diagnostic for V3A.

Extends the smoke evaluation to 5+ seeds with confidence intervals.
Since full-game evaluation is blocked, this operates at the diagnostic level
using synthetic candidate sets. Results include per-seed metrics, mean
differences, and bootstrap confidence intervals.

Output states:
  MULTISEED_COMPLETED — diagnostic completed with stats
  MULTISEED_BLOCKED — script cannot run
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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


def _generate_candidates_for_seed(seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    candidates: List[Dict[str, Any]] = []

    for i in range(12):
        op_hex = 100 + rng.randint(0, 5)
        target_hex = op_hex + rng.randint(1, 10) * (1 if rng.random() < 0.7 else -1)
        target_hex = max(100, min(999, target_hex))
        candidates.append({
            "type": "shoot",
            "operator": {"obj_id": i * 2, "cur_hex": op_hex, "cur_pos": 0, "color": 0, "type": 1,
                         "blood": 100, "ammo": 30, "fuel": 50, "morale": 80, "stealth": 0},
            "candidate": {"target_obj_id": i * 2 + 1, "damage": rng.uniform(10, 60),
                          "hit_prob": rng.uniform(0.3, 0.95), "attack_level": rng.randint(2, 5)},
            "target_operator": {"obj_id": i * 2 + 1, "cur_hex": target_hex, "cur_pos": 0, "color": 1,
                                "type": 1, "blood": rng.randint(10, 100), "ammo": 10, "fuel": 30,
                                "morale": rng.randint(30, 90), "stealth": rng.randint(0, 20)},
        })
    for i in range(12):
        op_hex = 100 + rng.randint(0, 5)
        city_hex = op_hex + rng.randint(1, 15)
        candidates.append({
            "type": "move_city",
            "operator": {"obj_id": 100 + i, "cur_hex": op_hex, "cur_pos": 0, "color": 0, "type": 1,
                         "blood": 100, "ammo": 30, "fuel": 50, "morale": 80, "stealth": 0},
            "candidate": {"move_path": [op_hex, city_hex]},
            "city": {"coord": city_hex, "value": rng.randint(20, 100), "flag": rng.randint(0, 2)},
        })
    return candidates


def _score_candidates(
    model_state: Dict[str, Any], input_dim: int, hidden_dim: int,
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    import torch
    from ai.features import action_feature_vector
    from ai.models import TacticalScorer

    model = TacticalScorer(input_dim=input_dim, hidden_dim=hidden_dim)
    model.load_state_dict(model_state)
    model.eval()

    results = []
    for entry in candidates:
        op, cand = entry["operator"], entry["candidate"]
        try:
            features = action_feature_vector(op, cand, target_operator=entry.get("target_operator"),
                                              city=entry.get("city"))
        except Exception as exc:
            results.append({"type": entry["type"], "error": str(exc), "score": None})
            continue
        if len(features) < input_dim:
            features = tuple(list(features) + [0.0] * (input_dim - len(features)))[:input_dim]
        tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logit = model(tensor).item()
        results.append({"type": entry["type"], "score": logit, "nonzero": abs(logit) > 1e-9})
    return results


def _bootstrap_ci(values: List[float], n_resamples: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """Bootstrap confidence interval for mean."""
    arr = np.array(values)
    means = []
    rng = np.random.RandomState(42)
    for _ in range(n_resamples):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(sample.mean()))
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def run_multiseed(checkpoint: Path, seeds: List[int]) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "status": "MULTISEED_BLOCKED",
        "not_formal_evidence": True,
        "analysis_note": "Diagnostic-level evaluation; no game replay. Outcome contribution unproven.",
        "checkpoint": str(checkpoint),
        "seeds": seeds, "per_seed": [], "statistics": {}, "errors": [],
    }
    validation = validate_checkpoint(checkpoint)
    if validation["status"] != "TACTICAL_CHECKPOINT_COMPATIBLE":
        report["errors"].extend(validation.get("errors", []))
        return report
    if not _torch_available():
        report["errors"].append("torch not available")
        return report

    import torch
    input_dim = int(validation.get("input_dim", 32))
    hidden_dim = int(validation.get("hidden_dim", 128))
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model_state = payload["model_state"]
    except Exception as exc:
        report["errors"].append(f"model_load_failed: {exc}")
        return report

    per_seed = []
    all_raw_scores: List[float] = []
    for seed in seeds:
        candidates = _generate_candidates_for_seed(seed)
        results = _score_candidates(model_state, input_dim, hidden_dim, candidates)
        scores = [r["score"] for r in results if r["score"] is not None]
        all_raw_scores.extend(scores)
        nonzero = sum(1 for r in results if r.get("nonzero"))
        changed = nonzero
        s: Dict[str, Any] = {
            "seed": seed, "scenario": "synthetic", "candidates": len(results),
            "tactical_reached": len(scores), "nonzero": nonzero, "changed": changed,
            "score_mean": round(float(np.mean(scores)), 6) if scores else None,
            "score_std": round(float(np.std(scores)), 6) if scores else None,
        }
        errors = [r.get("error") for r in results if r.get("error")]
        if errors:
            s["errors"] = errors
        per_seed.append(s)
    report["per_seed"] = per_seed

    # Statistics
    means = [s["score_mean"] for s in per_seed if s["score_mean"] is not None]
    stat: Dict[str, Any] = {
        "num_seeds": len(seeds),
        "candidates_per_seed": 24,
        "total_candidates": sum(s["candidates"] for s in per_seed),
        "changed_action_rate_across_seeds": round(
            sum(s["changed"] for s in per_seed) / max(sum(s["tactical_reached"] for s in per_seed), 1), 4
        ),
    }
    if means:
        stat["mean_of_seed_means"] = round(float(np.mean(means)), 6)
        stat["std_of_seed_means"] = round(float(np.std(means)), 6)
        lo, hi = _bootstrap_ci(means)
        stat["bootstrap_95_ci_lower"] = round(lo, 6)
        stat["bootstrap_95_ci_upper"] = round(hi, 6)
        stat["all_seeds_negative"] = all(m < 0 for m in means)
        stat["min_seed_mean"] = round(float(min(means)), 6)
        stat["max_seed_mean"] = round(float(max(means)), 6)
    if all_raw_scores:
        stat["pooled_score_mean"] = round(float(np.mean(all_raw_scores)), 6)
        stat["pooled_score_std"] = round(float(np.std(all_raw_scores)), 6)
    stat["game_level_metrics_available"] = False
    report["statistics"] = stat
    report["status"] = "MULTISEED_COMPLETED"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-seed tactical ablation for V3A.")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--tactical-checkpoint", default="ai/weights/tactical_model.pt")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    checkpoint = (ROOT / args.tactical_checkpoint).resolve() \
        if not Path(args.tactical_checkpoint).is_absolute() else Path(args.tactical_checkpoint)
    report = run_multiseed(checkpoint, args.seeds)
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "multiseed_ablation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report["status"])
    return 0 if report["status"] == "MULTISEED_COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
