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


def summarize_matches(matches: Iterable[Dict[str, object]]) -> Dict[str, object]:
    overall_scores: List[float] = []
    by_bucket: Dict[str, List[float]] = defaultdict(list)
    for match in matches:
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
    }


def promotion_decision(summary: Dict[str, object], config: Dict[str, object]) -> Dict[str, object]:
    overall = summary["overall"]
    buckets = summary["buckets"]
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
    return {
        "accepted": not failures,
        "failures": failures,
        "elo_delta_lower_bound": overall["elo"],
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
