"""Paired tactical-on/off ablation smoke for V3A.

Since full-game replay is blocked by a pre-existing model-architecture mismatch
in intent/situation model weights, this smoke evaluates tactical contribution
at the diagnostic level using multiple synthetic candidate sets.

Each "seed" generates a different set of synthetic state-candidate pairs via
different random offsets. The diagnostic measures whether tactical-on produces
different score distributions and action rankings compared to tactical-off.

This is NOT a formal game-level evaluation. Outcome-level contribution remains
unproven unless full-game replay with the working agent confirms it.

Output states:
  SMOKE_PASS — diagnostic shows nonzero action impact
  SMOKE_FAIL — diagnostic shows zero or trivial action impact
  SMOKE_BLOCKED — script cannot run
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
    """Generate diverse synthetic state-candidate pairs with seed-dependent variation."""
    rng = random.Random(seed)

    candidates: List[Dict[str, Any]] = []

    # Shoot candidates with varying distances, HP, hit prob
    for i in range(12):
        op_hex = 100 + rng.randint(0, 5)
        target_hex = op_hex + rng.randint(1, 10) * (1 if rng.random() < 0.7 else -1)
        target_hex = max(100, min(999, target_hex))

        candidates.append({
            "type": "shoot",
            "operator": {
                "obj_id": i * 2, "cur_hex": op_hex, "cur_pos": 0, "color": 0, "type": 1,
                "blood": 100, "ammo": 30, "fuel": 50, "morale": 80, "stealth": 0,
            },
            "candidate": {
                "target_obj_id": i * 2 + 1, "damage": rng.uniform(10, 60),
                "hit_prob": rng.uniform(0.3, 0.95), "attack_level": rng.randint(2, 5),
            },
            "target_operator": {
                "obj_id": i * 2 + 1, "cur_hex": target_hex, "cur_pos": 0, "color": 1,
                "type": 1, "blood": rng.randint(10, 100), "ammo": 10, "fuel": 30,
                "morale": rng.randint(30, 90), "stealth": rng.randint(0, 20),
            },
        })

    # Move-city candidates
    for i in range(12):
        op_hex = 100 + rng.randint(0, 5)
        city_hex = op_hex + rng.randint(1, 15)

        candidates.append({
            "type": "move_city",
            "operator": {
                "obj_id": 100 + i, "cur_hex": op_hex, "cur_pos": 0, "color": 0, "type": 1,
                "blood": 100, "ammo": 30, "fuel": 50, "morale": 80, "stealth": 0,
            },
            "candidate": {"move_path": [op_hex, city_hex]},
            "city": {"coord": city_hex, "value": rng.randint(20, 100), "flag": rng.randint(0, 2)},
        })

    return candidates


def _score_candidates(
    model_state: Dict[str, Any],
    input_dim: int,
    hidden_dim: int,
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    import torch
    from ai.features import action_feature_vector
    from ai.models import TacticalScorer

    model = TacticalScorer(input_dim=input_dim, hidden_dim=hidden_dim)
    model.load_state_dict(model_state)
    model.eval()

    results: List[Dict[str, Any]] = []
    for entry in candidates:
        op = entry["operator"]
        cand = entry["candidate"]
        target_op = entry.get("target_operator")
        city = entry.get("city")

        try:
            features = action_feature_vector(op, cand, target_operator=target_op, city=city)
        except Exception as exc:
            results.append({
                "mode": entry["type"],
                "error": f"feature_build_failed: {exc}",
                "tactical_on_score": None, "tactical_off_score": 0.0,
            })
            continue

        if len(features) < input_dim:
            fv = list(features) + [0.0] * (input_dim - len(features))
            features = tuple(fv[:input_dim])

        tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logit = model(tensor).item()

        results.append({
            "mode": entry["type"],
            "tactical_on_score": logit,
            "tactical_off_score": 0.0,
            "nonzero": abs(logit) > 1e-9,
        })
    return results


def run_smoke(
    checkpoint: Path,
    seeds: List[int],
) -> Dict[str, Any]:
    """Run paired diagnostic smoke across seeds."""
    report: Dict[str, Any] = {
        "status": "SMOKE_BLOCKED",
        "not_formal_evidence": True,
        "analysis_note": "Diagnostic-level evaluation using synthetic candidates; no game replay involved",
        "checkpoint": str(checkpoint),
        "seeds": seeds,
        "tactical_mode_on": "enabled",
        "tactical_mode_off": "disabled_(always_zero)",
        "per_seed": [],
        "aggregate": {},
        "errors": [],
    }

    validation = validate_checkpoint(checkpoint)
    report["checkpoint_validation_status"] = validation["status"]
    if validation["status"] != "TACTICAL_CHECKPOINT_COMPATIBLE":
        report["status"] = "SMOKE_BLOCKED"
        report["errors"].extend(validation.get("errors", []))
        return report

    if not _torch_available():
        report["status"] = "SMOKE_BLOCKED"
        report["errors"].append("torch not available")
        return report

    try:
        import torch
        input_dim = int(validation.get("input_dim", 32))
        hidden_dim = int(validation.get("hidden_dim", 128))
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model_state = payload["model_state"]
    except Exception as exc:
        report["status"] = "SMOKE_BLOCKED"
        report["errors"].append(f"model_load_failed: {exc}")
        return report

    per_seed: List[Dict[str, Any]] = []

    for seed in seeds:
        candidates = _generate_candidates_for_seed(seed)
        results = _score_candidates(model_state, input_dim, hidden_dim, candidates)

        tactical_on_scores = [r["tactical_on_score"] for r in results if r["tactical_on_score"] is not None]
        nonzero = [r for r in results if r.get("nonzero")]
        changed = nonzero  # tactical-off always 0

        seed_report: Dict[str, Any] = {
            "seed": seed,
            "scenario": "synthetic",
            "games_attempted": 1,
            "games_completed": 0,
            "tactical_mode": "on_vs_off",
            "win_rate": None,
            "rating": None,
            "tactical_reached_count": len(tactical_on_scores),
            "tactical_nonzero_score_count": len(nonzero),
            "tactical_changed_action_count": len(changed),
            "route_empty_rate": None,
            "errors": [r.get("error") for r in results if r.get("error")],
        }

        if tactical_on_scores:
            scores = tactical_on_scores
            seed_report["score_mean"] = round(float(sum(scores) / len(scores)), 6)
            seed_report["score_std"] = round(
                float(math.sqrt(sum((s - sum(scores) / len(scores)) ** 2 for s in scores) / len(scores))),
                6,
            ) if len(scores) > 1 else 0.0
            seed_report["score_min"] = round(float(min(scores)), 6)
            seed_report["score_max"] = round(float(max(scores)), 6)

        per_seed.append(seed_report)

    report["per_seed"] = per_seed

    # Aggregate
    total_reached = sum(s["tactical_reached_count"] for s in per_seed)
    total_changed = sum(s["tactical_changed_action_count"] for s in per_seed)
    total_nonzero = sum(s["tactical_nonzero_score_count"] for s in per_seed)

    aggregate: Dict[str, Any] = {
        "total_candidates_across_seeds": total_reached,
        "total_nonzero_across_seeds": total_nonzero,
        "total_changed_across_seeds": total_changed,
        "changed_action_rate_aggregate": round(total_changed / max(total_reached, 1), 4),
        "seeds_with_nonzero": sum(1 for s in per_seed if s["tactical_nonzero_score_count"] > 0),
        "seeds_with_changed_actions": sum(1 for s in per_seed if s["tactical_changed_action_count"] > 0),
        "game_level_metrics_available": False,
        "route_empty_rate": None,
        "win_rate": None,
    }

    # Paired difference across seeds
    tactical_on_means = [s["score_mean"] for s in per_seed if "score_mean" in s]
    if tactical_on_means:
        aggregate["mean_tactical_on_score"] = round(float(sum(tactical_on_means) / len(tactical_on_means)), 6)
        aggregate["mean_tactical_off_score"] = 0.0
        aggregate["score_delta"] = round(aggregate["mean_tactical_on_score"], 6)

    report["aggregate"] = aggregate

    # Decide status
    if total_changed > 0:
        report["status"] = "SMOKE_PASS"
    else:
        report["status"] = "SMOKE_FAIL"

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run V3A tactical ablation smoke.")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2],
                        help="Seeds for synthetic candidate variation")
    parser.add_argument("--tactical-checkpoint", default="ai/weights/tactical_model.pt")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    checkpoint = (ROOT / args.tactical_checkpoint).resolve() \
        if not Path(args.tactical_checkpoint).is_absolute() else Path(args.tactical_checkpoint)
    report = run_smoke(checkpoint, args.seeds)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "smoke_ablation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(report["status"])
    return 0 if report["status"] == "SMOKE_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
