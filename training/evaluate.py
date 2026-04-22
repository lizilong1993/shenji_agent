from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .common import repo_root, save_json


def bootstrap_ci(values: List[float], *, rounds: int = 2000, alpha: float = 0.05) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    samples = []
    arr = np.array(values, dtype=np.float32)
    for _ in range(rounds):
        boot = np.random.choice(arr, size=len(arr), replace=True)
        samples.append(float(np.mean(boot)))
    lower = np.quantile(samples, alpha / 2)
    upper = np.quantile(samples, 1 - alpha / 2)
    return float(lower), float(upper)


def simple_elo(win_rate: float) -> float:
    win_rate = min(max(win_rate, 1e-4), 1 - 1e-4)
    return -400.0 * math.log10((1 / win_rate) - 1)


def _merge_policy_stats(matches: Iterable[Dict[str, object]]) -> Dict[str, object]:
    totals = {
        "move_total": 0,
        "move_learned": 0,
        "shoot_total": 0,
        "shoot_learned": 0,
        "score_sums": {
            "move": {"samples": 0, "intent_bias": 0.0, "situation_score": 0.0, "combined_score": 0.0},
            "shoot": {"samples": 0, "intent_bias": 0.0, "situation_score": 0.0, "combined_score": 0.0},
        },
    }
    for match in matches:
        stats = match.get("candidate_policy_stats", {})
        counts = stats.get("counts", {}) if isinstance(stats, dict) else {}
        move_counts = counts.get("move", {}) if isinstance(counts, dict) else {}
        shoot_counts = counts.get("shoot", {}) if isinstance(counts, dict) else {}
        totals["move_total"] += int(move_counts.get("total", 0))
        totals["move_learned"] += int(move_counts.get("learned", 0))
        totals["shoot_total"] += int(shoot_counts.get("total", 0))
        totals["shoot_learned"] += int(shoot_counts.get("learned", 0))
        score_means = stats.get("score_means", {}) if isinstance(stats, dict) else {}
        for action_key in ("move", "shoot"):
            action_scores = score_means.get(action_key, {}) if isinstance(score_means, dict) else {}
            samples = int(action_scores.get("samples", 0))
            totals["score_sums"][action_key]["samples"] += samples
            for metric in ("intent_bias", "situation_score", "combined_score"):
                totals["score_sums"][action_key][metric] += float(action_scores.get(metric, 0.0)) * samples

    def coverage(total: int, learned: int) -> float:
        if total <= 0:
            return 0.0
        return float(learned / total)

    score_means = {}
    for action_key, values in totals["score_sums"].items():
        samples = int(values["samples"])
        score_means[action_key] = {
            "samples": samples,
            "intent_bias": float(values["intent_bias"] / samples) if samples else 0.0,
            "situation_score": float(values["situation_score"] / samples) if samples else 0.0,
            "combined_score": float(values["combined_score"] / samples) if samples else 0.0,
        }

    move_total = int(totals["move_total"])
    move_learned = int(totals["move_learned"])
    shoot_total = int(totals["shoot_total"])
    shoot_learned = int(totals["shoot_learned"])
    total_actions = move_total + shoot_total
    learned_actions = move_learned + shoot_learned
    return {
        "learned_path_coverage": coverage(total_actions, learned_actions),
        "move_path_coverage": coverage(move_total, move_learned),
        "shoot_path_coverage": coverage(shoot_total, shoot_learned),
        "counts": {
            "move": {"total": move_total, "learned": move_learned},
            "shoot": {"total": shoot_total, "learned": shoot_learned},
        },
        "score_means": score_means,
    }


def summarize_matches(matches: Iterable[Dict[str, object]]) -> Dict[str, object]:
    match_list = list(matches)
    overall_scores: List[float] = []
    by_bucket: Dict[str, List[float]] = defaultdict(list)
    for match in match_list:
        score = float(match.get("paired_score", 0.0))
        bucket = str(match.get("bucket", "medium"))
        overall_scores.append(score)
        by_bucket[bucket].append(score)

    overall_mean = float(np.mean(overall_scores)) if overall_scores else 0.0
    overall_ci = bootstrap_ci(overall_scores)
    buckets = {}
    for bucket, values in by_bucket.items():
        buckets[bucket] = {
            "point_estimate": float(np.mean(values)),
            "ci95": bootstrap_ci(values),
            "count": len(values),
        }
    return {
        "overall": {
            "point_estimate": overall_mean,
            "ci95": overall_ci,
            "elo": simple_elo(overall_mean) if overall_scores else 0.0,
            "count": len(overall_scores),
        },
        "buckets": buckets,
        "policy": _merge_policy_stats(match_list),
    }


def promotion_decision(summary: Dict[str, object], config: Dict[str, object]) -> Dict[str, object]:
    overall = summary["overall"]
    buckets = summary["buckets"]
    policy = summary.get("policy", {})
    complex_bucket = buckets.get("complex", {"point_estimate": 0.0, "ci95": (0.0, 0.0)})
    failures = []
    if overall["point_estimate"] < config["overall_point_estimate_min"]:
        failures.append("overall_point_estimate")
    if overall["ci95"][0] <= config["overall_ci_lower_min"]:
        failures.append("overall_ci_lower")
    if complex_bucket["point_estimate"] < config["complex_point_estimate_min"]:
        failures.append("complex_point_estimate")
    if complex_bucket["ci95"][0] <= config["complex_ci_lower_min"]:
        failures.append("complex_ci_lower")
    min_learned_coverage = float(config.get("min_learned_path_coverage", 0.0))
    if min_learned_coverage > 0.0 and float(policy.get("learned_path_coverage", 0.0)) < min_learned_coverage:
        failures.append("learned_path_coverage")
    return {
        "accepted": not failures,
        "failures": failures,
        "elo_delta_lower_bound": overall["elo"],
        "learned_path_coverage": float(policy.get("learned_path_coverage", 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Shenji self-play outputs.")
    parser.add_argument("--results", required=True, help="Path to match_results.json")
    parser.add_argument("--config", default="configs/selfplay_baseline.yaml")
    args = parser.parse_args()

    results_path = Path(args.results)
    config = json.loads(Path(repo_root() / args.config).read_text(encoding="utf-8")) if args.config.endswith(".json") else None
    if config is None:
        import yaml

        config = yaml.safe_load((repo_root() / args.config).read_text(encoding="utf-8"))

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    summary = summarize_matches(payload["matches"])
    decision = promotion_decision(summary, config["promotion_gate"])
    out_dir = results_path.parent
    save_json(summary, out_dir / "rating.json")
    save_json(decision, out_dir / "promotion.json")

    with (out_dir / "payoff_matrix.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["bucket", "point_estimate", "ci95_lower", "ci95_upper", "count"])
        writer.writerow(["overall", summary["overall"]["point_estimate"], summary["overall"]["ci95"][0], summary["overall"]["ci95"][1], summary["overall"]["count"]])
        for bucket, values in summary["buckets"].items():
            writer.writerow([bucket, values["point_estimate"], values["ci95"][0], values["ci95"][1], values["count"]])


if __name__ == "__main__":
    main()
