from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
        "manifest_loaded_match_count": 0,
        "intent_model_loaded_match_count": 0,
        "situation_model_loaded_match_count": 0,
        "tactical_model_loaded_match_count": 0,
        "learned_policy_enabled_match_count": 0,
        "fallback_reasons": set(),
        "profiles": set(),
        "artifact_dirs": set(),
        "manifest_paths": set(),
        "score_sums": {
            "move": {
                "samples": 0,
                "intent_bias": 0.0,
                "situation_score": 0.0,
                "tactical_score": 0.0,
                "objective_score": 0.0,
                "combined_score": 0.0,
            },
            "shoot": {
                "samples": 0,
                "intent_bias": 0.0,
                "situation_score": 0.0,
                "tactical_score": 0.0,
                "objective_score": 0.0,
                "combined_score": 0.0,
            },
        },
        "move_route_agg": {
            "total_attempts": 0,
            "empty_route_count": 0,
            "reachable_city_total": 0,
            "by_move_type": {},
            "route_failure_reasons": {},
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
            for metric in ("intent_bias", "situation_score", "tactical_score", "objective_score", "combined_score"):
                totals["score_sums"][action_key][metric] += float(action_scores.get(metric, 0.0)) * samples
        diagnostics = match.get("candidate_policy_diagnostics", {})
        if isinstance(diagnostics, dict):
            totals["manifest_loaded_match_count"] += int(bool(diagnostics.get("manifest_loaded")))
            totals["intent_model_loaded_match_count"] += int(bool(diagnostics.get("intent_model_loaded")))
            totals["situation_model_loaded_match_count"] += int(bool(diagnostics.get("situation_model_loaded")))
            totals["tactical_model_loaded_match_count"] += int(bool(diagnostics.get("tactical_model_loaded")))
            totals["learned_policy_enabled_match_count"] += int(bool(diagnostics.get("learned_policy_enabled")))
            profile = diagnostics.get("profile")
            if profile:
                totals["profiles"].add(str(profile))
            artifact_dir = diagnostics.get("artifact_dir")
            if artifact_dir:
                totals["artifact_dirs"].add(str(artifact_dir))
            manifest_path = diagnostics.get("manifest_path")
            if manifest_path:
                totals["manifest_paths"].add(str(manifest_path))
            fallback_reason = diagnostics.get("fallback_reason")
            if isinstance(fallback_reason, list):
                totals["fallback_reasons"].update(str(reason) for reason in fallback_reason if reason)
            elif fallback_reason:
                totals["fallback_reasons"].add(str(fallback_reason))
        route_agg = stats.get("move_route_aggregate", {})
        if isinstance(route_agg, dict):
            totals["move_route_agg"]["total_attempts"] += int(route_agg.get("total_attempts", 0))
            totals["move_route_agg"]["empty_route_count"] += int(route_agg.get("empty_route_count", 0))
            totals["move_route_agg"]["reachable_city_total"] += int(route_agg.get("reachable_city_total", 0))
            for mt, mt_data in route_agg.get("by_move_type", {}).items():
                existing = totals["move_route_agg"]["by_move_type"].setdefault(str(mt), {"total": 0, "empty": 0})
                if isinstance(mt_data, dict):
                    existing["total"] += int(mt_data.get("total", 0))
                    existing["empty"] += int(mt_data.get("empty", 0))
            for reason, count in route_agg.get("route_failure_reasons", {}).items():
                r = str(reason)
                totals["move_route_agg"]["route_failure_reasons"][r] = totals["move_route_agg"]["route_failure_reasons"].get(r, 0) + int(count)

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
            "tactical_score": float(values["tactical_score"] / samples) if samples else 0.0,
            "objective_score": float(values["objective_score"] / samples) if samples else 0.0,
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
        "manifest_loaded_match_count": int(totals["manifest_loaded_match_count"]),
        "intent_model_loaded_match_count": int(totals["intent_model_loaded_match_count"]),
        "situation_model_loaded_match_count": int(totals["situation_model_loaded_match_count"]),
        "tactical_model_loaded_match_count": int(totals["tactical_model_loaded_match_count"]),
        "learned_policy_enabled_match_count": int(totals["learned_policy_enabled_match_count"]),
        "profiles": sorted(totals["profiles"]),
        "artifact_dirs": sorted(totals["artifact_dirs"]),
        "manifest_paths": sorted(totals["manifest_paths"]),
        "fallback_reasons": sorted(totals["fallback_reasons"]),
        "move_route_aggregate": totals["move_route_agg"],
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


def promotion_decision(
    summary: Dict[str, object],
    config: Dict[str, object],
    exploiter_results: Optional[Dict[str, Dict[str, object]]] = None,
) -> Dict[str, object]:
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
    min_move_coverage = float(config.get("min_move_path_coverage", 0.0))
    if min_move_coverage > 0.0 and float(policy.get("move_path_coverage", 0.0)) < min_move_coverage:
        failures.append("move_path_coverage")
    min_shoot_coverage = float(config.get("min_shoot_path_coverage", 0.0))
    if min_shoot_coverage > 0.0 and float(policy.get("shoot_path_coverage", 0.0)) < min_shoot_coverage:
        failures.append("shoot_path_coverage")

    overall_ci_lower = float(overall["ci95"][0])
    elo_delta_lower_bound = simple_elo(overall_ci_lower) if int(overall.get("count", 0)) > 0 else 0.0
    elo_delta_lower_min = float(config.get("elo_delta_lower_min", 0.0))
    if elo_delta_lower_min > 0.0 and elo_delta_lower_bound < elo_delta_lower_min:
        failures.append("elo_delta_lower")

    bucket_regressions = {
        bucket: max(0.0, 0.5 - float(values.get("point_estimate", 0.0)))
        for bucket, values in buckets.items()
    }
    max_bucket_regression = max(bucket_regressions.values(), default=0.0)
    allowed_bucket_regression = float(config.get("max_bucket_regression", 1.0))
    if allowed_bucket_regression < 1.0 and max_bucket_regression > allowed_bucket_regression:
        failures.append("bucket_regression")

    max_exploiter_win_rate = 0.0
    if exploiter_results:
        max_exploiter_win_rate = max(
            float(result.get("win_rate_vs_candidate", 0.0))
            for result in exploiter_results.values()
        )
    allowed_exploiter_win_rate = float(config.get("max_exploiter_win_rate", 1.0))
    if exploiter_results and allowed_exploiter_win_rate < 1.0 and max_exploiter_win_rate > allowed_exploiter_win_rate:
        failures.append("exploiter_win_rate")
    return {
        "accepted": not failures,
        "failures": failures,
        "elo_delta_lower_bound": elo_delta_lower_bound,
        "learned_path_coverage": float(policy.get("learned_path_coverage", 0.0)),
        "move_path_coverage": float(policy.get("move_path_coverage", 0.0)),
        "shoot_path_coverage": float(policy.get("shoot_path_coverage", 0.0)),
        "max_bucket_regression": max_bucket_regression,
        "bucket_regressions": bucket_regressions,
        "max_exploiter_win_rate": max_exploiter_win_rate,
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
