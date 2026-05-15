"""Metrics computation and bootstrap statistics for V3B formal tactical ablation."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def extract_paired_deltas(
    paired_results: List[Dict[str, Any]],
    metric_key: str,
) -> List[float]:
    """Extract paired deltas from a list of paired results."""
    deltas = []
    for pair in paired_results:
        delta = pair.get("paired_delta", {})
        val = delta.get(metric_key)
        if val is not None and isinstance(val, (int, float)):
            deltas.append(float(val))
    return deltas


def extract_metric(
    results: List[Dict[str, Any]],
    metric_key: str,
    tactical_mode: str,
) -> List[float]:
    """Extract a metric from tactical_on or tactical_off results."""
    values = []
    for pair in results:
        sub = pair.get(f"tactical_{tactical_mode}", {})
        val = sub.get(metric_key)
        if val is not None and isinstance(val, (int, float)):
            values.append(float(val))
    return values


def extract_score_deltas(
    paired_results: List[Dict[str, Any]],
) -> List[float]:
    """Extract score_red deltas: tactical_on minus tactical_off."""
    deltas = []
    for pair in paired_results:
        on_score = pair.get("tactical_on", {}).get("score_red")
        off_score = pair.get("tactical_off", {}).get("score_red")
        if on_score is not None and off_score is not None:
            deltas.append(float(on_score) - float(off_score))
    return deltas


def bootstrap_ci(
    values: List[float],
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, List[float]]:
    """Bootstrap confidence interval for the mean.

    Returns (lower, upper, bootstrap_means).
    """
    if len(values) < 2:
        return float("nan"), float("nan"), []
    arr = np.array(values)
    rng = np.random.RandomState(seed)
    means = []
    for _ in range(n_resamples):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(sample.mean()))
    alpha = (1.0 - confidence_level) / 2.0
    lower = float(np.quantile(means, alpha))
    upper = float(np.quantile(means, 1.0 - alpha))
    return lower, upper, means


def cohens_d(
    paired_deltas: List[float],
) -> Optional[float]:
    """Compute Cohen's d for paired deltas (mean / std)."""
    if len(paired_deltas) < 2:
        return None
    arr = np.array(paired_deltas)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    if abs(std) < 1e-12:
        return None
    return round(mean / std, 4)


def aggregate_pairwise(
    paired_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate paired results into summary statistics.

    Computes:
      - mean/std for score deltas
      - bootstrap CI for score delta
      - win rate comparison
      - route-empty rate comparison
      - effect size
    """
    result: Dict[str, Any] = {
        "total_pairs": len(paired_results),
        "completed_pairs": 0,
        "failed_on": 0,
        "failed_off": 0,
        "crashed_on": 0,
        "crashed_off": 0,
        "statistics": {},
        "warnings": [],
    }

    # Gather per-mode results
    score_deltas = extract_paired_deltas(paired_results, "score_delta")
    on_route_empty = [
        pair.get("tactical_on", {}).get("route_metrics", {}).get("route_empty_rate", 0.0)
        for pair in paired_results
    ]
    off_route_empty = [
        pair.get("tactical_off", {}).get("route_metrics", {}).get("route_empty_rate", 0.0)
        for pair in paired_results
    ]
    on_steps = extract_metric(paired_results, "steps", "on")
    off_steps = extract_metric(paired_results, "steps", "off")
    on_duration = extract_metric(paired_results, "duration", "on")
    off_duration = extract_metric(paired_results, "duration", "off")

    for pair in paired_results:
        on_status = pair.get("tactical_on", {}).get("status")
        off_status = pair.get("tactical_off", {}).get("status")
        if on_status == "completed":
            result["completed_pairs"] += 1
        if on_status in ("crashed",):
            result["crashed_on"] += 1
        if off_status in ("crashed",):
            result["crashed_off"] += 1
        if on_status not in ("completed",):
            result["failed_on"] += 1
        if off_status not in ("completed",):
            result["failed_off"] += 1

    stats: Dict[str, Any] = {}

    # Score delta statistics
    if score_deltas:
        arr = np.array(score_deltas)
        ci_lo, ci_hi, bootstrap_means = bootstrap_ci(score_deltas)
        d = cohens_d(score_deltas)
        stats["score_delta"] = {
            "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std(ddof=1)), 4),
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
            "bootstrap_95_ci_lower": round(ci_lo, 4),
            "bootstrap_95_ci_upper": round(ci_hi, 4),
            "n_resamples": 1000,
            "effect_size_cohens_d": d,
            "n": len(score_deltas),
            "positive_count": int(sum(1 for d in score_deltas if d > 0)),
            "negative_count": int(sum(1 for d in score_deltas if d < 0)),
            "zero_count": int(sum(1 for d in score_deltas if d == 0)),
        }

    # Route empty rate comparison
    if on_route_empty and off_route_empty:
        on_arr = np.array(on_route_empty)
        off_arr = np.array(off_route_empty)
        deltas = [o - f for o, f in zip(on_route_empty, off_route_empty)]
        ci_lo_route, ci_hi_route, _ = bootstrap_ci(deltas)
        stats["route_empty_rate"] = {
            "tactical_on_mean": round(float(on_arr.mean()), 4),
            "tactical_off_mean": round(float(off_arr.mean()), 4),
            "mean_delta": round(float(np.array(deltas).mean()), 4),
            "bootstrap_95_ci_lower": round(ci_lo_route, 4),
            "bootstrap_95_ci_upper": round(ci_hi_route, 4),
            "n": len(deltas),
        }

    # Steps comparison
    if on_steps and off_steps:
        on_arr = np.array(on_steps)
        off_arr = np.array(off_steps)
        stats["steps"] = {
            "tactical_on_mean": round(float(on_arr.mean()), 1),
            "tactical_off_mean": round(float(off_arr.mean()), 1),
            "mean_delta": round(float((on_arr - off_arr).mean()), 1),
        }

    # Duration comparison
    if on_duration and off_duration:
        on_arr = np.array(on_duration)
        off_arr = np.array(off_duration)
        stats["duration"] = {
            "tactical_on_mean": round(float(on_arr.mean()), 2),
            "tactical_off_mean": round(float(off_arr.mean()), 2),
            "mean_delta": round(float((on_arr - off_arr).mean()), 2),
        }

    # Win rate
    on_wins = sum(
        1 for pair in paired_results
        if pair.get("tactical_on", {}).get("winner") == 0
    )
    off_wins = sum(
        1 for pair in paired_results
        if pair.get("tactical_off", {}).get("winner") == 0
    )
    completed_on = sum(
        1 for pair in paired_results
        if pair.get("tactical_on", {}).get("status") == "completed"
    )
    completed_off = sum(
        1 for pair in paired_results
        if pair.get("tactical_off", {}).get("status") == "completed"
    )
    stats["win_rate"] = {
        "tactical_on": {
            "wins": on_wins,
            "games": completed_on,
            "win_rate": round(on_wins / max(completed_on, 1), 4),
        },
        "tactical_off": {
            "wins": off_wins,
            "games": completed_off,
            "win_rate": round(off_wins / max(completed_off, 1), 4),
        },
        "win_rate_delta": round(
            on_wins / max(completed_on, 1) - off_wins / max(completed_off, 1), 4
        ),
    }

    result["statistics"] = stats

    # Warnings
    if result["failed_on"] > 0 or result["failed_off"] > 0:
        result["warnings"].append(
            f"{result['failed_on']} tactical-on and {result['failed_off']} "
            f"tactical-off games failed/crashed."
        )
    if result["completed_pairs"] < len(paired_results):
        result["warnings"].append(
            f"Only {result['completed_pairs']}/{len(paired_results)} pairs completed."
        )

    return result


def format_statistics_table(stats: Dict[str, Any]) -> str:
    """Format aggregate statistics as a markdown table."""
    lines = ["| Metric | Tactical On | Tactical Off | Delta | 95% CI |",
             "|--------|-------------|--------------|-------|--------|"]

    wr = stats.get("win_rate", {})
    if wr:
        on_wr = wr.get("tactical_on", {}).get("win_rate", "N/A")
        off_wr = wr.get("tactical_off", {}).get("win_rate", "N/A")
        delta_wr = wr.get("win_rate_delta", "N/A")
        lines.append(f"| Win rate | {on_wr} | {off_wr} | {delta_wr} | — |")

    sd = stats.get("score_delta", {})
    if sd:
        lines.append(f"| Score delta mean | — | — | {sd.get('mean', 'N/A')} | [{sd.get('bootstrap_95_ci_lower', 'N/A')}, {sd.get('bootstrap_95_ci_upper', 'N/A')}] |")
        lines.append(f"| Score delta std | — | — | {sd.get('std', 'N/A')} | — |")
        if sd.get("effect_size_cohens_d") is not None:
            lines.append(f"| Cohen's d | — | — | {sd.get('effect_size_cohens_d')} | — |")


    re = stats.get("route_empty_rate", {})
    if re:
        lines.append(f"| Route empty rate | {re.get('tactical_on_mean', 'N/A')} | {re.get('tactical_off_mean', 'N/A')} | {re.get('mean_delta', 'N/A')} | [{re.get('bootstrap_95_ci_lower', 'N/A')}, {re.get('bootstrap_95_ci_upper', 'N/A')}] |")

    st = stats.get("steps", {})
    if st:
        lines.append(f"| Steps | {st.get('tactical_on_mean', 'N/A')} | {st.get('tactical_off_mean', 'N/A')} | {st.get('mean_delta', 'N/A')} | — |")

    du = stats.get("duration", {})
    if du:
        lines.append(f"| Duration (s) | {du.get('tactical_on_mean', 'N/A')} | {du.get('tactical_off_mean', 'N/A')} | {du.get('mean_delta', 'N/A')} | — |")

    return "\n".join(lines)
