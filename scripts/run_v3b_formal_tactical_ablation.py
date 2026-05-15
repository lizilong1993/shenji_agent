#!/usr/bin/env python3
"""V3B Formal Tactical Outcome Ablation via SDK Replay.

Paired tactical-on vs tactical-off evaluation with:
  - Identical seeds, scenarios, maps
  - Structured JSON output
  - Bootstrap confidence intervals
  - Route reliability audit
  - Reproducible manifests

Usage:
  python scripts/run_v3b_formal_tactical_ablation.py \
      --seeds 0 1 2 3 4 \
      --scenarios 1231 1531 1910631192 \
      --out experiments/shenji_agent/v3b_formal_tactical_ablation
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.v3b_pairing import run_paired_game, resolve_map_name
from scripts.v3b_metrics import aggregate_pairwise, format_statistics_table


def _check_environment() -> List[str]:
    errors = []
    try:
        import torch
    except ImportError:
        errors.append("torch not available")
    try:
        from train_env import TrainEnv
    except ImportError:
        errors.append("SDK train_env not available")
    try:
        from ai import Agent
    except ImportError:
        errors.append("ai module not available")
    # Check checkpoint
    ckpt = ROOT / "ai" / "weights" / "tactical_model.pt"
    if not ckpt.exists():
        errors.append(f"tactical checkpoint not found: {ckpt}")
    return errors


def _run_evaluation(
    scenarios: List[str],
    seeds: List[int],
    artifacts_dir: Optional[str] = None,
    policy_profile: str = "challenger",
    timeout_steps: int = 2000,
    report_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the full paired evaluation across all scenarios and seeds."""
    manifest = {
        "start_time": datetime.now().isoformat(),
        "protocol": "V3B Formal Tactical Outcome Ablation",
        "scenarios": scenarios,
        "seeds": seeds,
        "policy_profile": policy_profile,
        "timeout_steps": timeout_steps,
        "artifacts_dir": artifacts_dir,
        "pairs_attempted": len(scenarios) * len(seeds),
    }

    all_paired_results: List[Dict[str, Any]] = []
    per_scenario: Dict[str, Dict[str, Any]] = {}
    errors: List[Dict[str, Any]] = []

    total_pairs = len(scenarios) * len(seeds)
    completed = 0

    for scenario_id in scenarios:
        scenario_results = {"scenario": scenario_id, "pairs": [], "statistics": {}}
        scenario_pairs: List[Dict[str, Any]] = []

        for seed in seeds:
            print(f"  [{completed+1}/{total_pairs}] Running scenario={scenario_id} seed={seed}")
            sys.stdout.flush()

            map_name = resolve_map_name(scenario_id)
            pair_result = run_paired_game(
                scenario_id=scenario_id,
                map_name=map_name,
                seed=seed,
                artifacts_dir=artifacts_dir,
                policy_profile=policy_profile,
                timeout_steps=timeout_steps,
            )
            scenario_pairs.append(pair_result)
            all_paired_results.append(pair_result)

            on_status = pair_result.get("tactical_on", {}).get("status")
            off_status = pair_result.get("tactical_off", {}).get("status")
            if on_status == "completed" and off_status == "completed":
                completed += 1

            # Check for errors
            on_error = pair_result.get("tactical_on", {}).get("error")
            off_error = pair_result.get("tactical_off", {}).get("error")
            if on_error or off_error:
                errors.append({
                    "scenario": scenario_id,
                    "seed": seed,
                    "tactical_on_error": on_error,
                    "tactical_off_error": off_error,
                })

            # Save per-game metrics incrementally
            if report_dir:
                per_game_dir = report_dir / "per_game_metrics"
                per_game_dir.mkdir(parents=True, exist_ok=True)
                game_file = per_game_dir / f"{scenario_id}_seed{seed}.json"
                game_file.write_text(
                    json.dumps(pair_result, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

        # Scenario-level aggregation
        scenario_stats = aggregate_pairwise(scenario_pairs)
        scenario_results["pairs"] = scenario_pairs
        scenario_results["statistics"] = scenario_stats
        per_scenario[scenario_id] = scenario_results

    # Overall aggregation
    overall_stats = aggregate_pairwise(all_paired_results)

    final_report = {
        "manifest": manifest,
        "end_time": datetime.now().isoformat(),
        "overall": {
            "total_pairs_attempted": total_pairs,
            "total_pairs_completed": completed,
            "total_errors": len(errors),
            "statistics": overall_stats,
            "warnings": overall_stats.get("warnings", []),
        },
        "per_scenario": {
            sid: {
                "statistics": sc["statistics"],
                "pair_count": len(sc["pairs"]),
            }
            for sid, sc in per_scenario.items()
        },
        "errors": errors,
        "raw_pairs": all_paired_results,
    }
    return final_report


def _compute_divergence(
    paired_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute action divergence between tactical on and off runs.

    Note: True action-by-action divergence requires recording per-step actions
    in the game loop. This computes a proxy based on policy diagnostics.
    """
    divergences = []
    for pair in paired_results:
        on_stats = pair.get("tactical_on", {}).get("policy_stats", {})
        off_stats = pair.get("tactical_off", {}).get("policy_stats", {})

        on_learned_move = on_stats.get("action_counts", {}).get("move", {}).get("learned", 0)
        on_learned_shoot = on_stats.get("action_counts", {}).get("shoot", {}).get("learned", 0)
        off_learned_move = off_stats.get("action_counts", {}).get("move", {}).get("learned", 0)
        off_learned_shoot = off_stats.get("action_counts", {}).get("shoot", {}).get("learned", 0)

        on_tactical_stats = pair.get("tactical_on", {}).get("tactical_stats", {})
        off_tactical_stats = pair.get("tactical_off", {}).get("tactical_stats", {})

        divergences.append({
            "seed": pair.get("seed"),
            "scenario": pair.get("scenario"),
            "on_learned_move": on_learned_move,
            "off_learned_move": off_learned_move,
            "on_learned_shoot": on_learned_shoot,
            "off_learned_shoot": off_learned_shoot,
            "on_tactical_invocation": on_tactical_stats.get("tactical_invocation_count", 0),
            "off_tactical_invocation": off_tactical_stats.get("tactical_invocation_count", 0),
        })
    return {"divergence_per_pair": divergences}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="V3B Formal Tactical Outcome Ablation via SDK Replay"
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--scenarios", nargs="*", type=str,
                        default=["1231", "1531", "1910631192"])
    parser.add_argument("--artifacts-dir", type=str, default=None)
    parser.add_argument("--policy-profile", type=str, default="challenger")
    parser.add_argument("--timeout-steps", type=int, default=2000)
    parser.add_argument("--out", type=str,
                        default="experiments/shenji_agent/v3b_formal_tactical_ablation")
    parser.add_argument("--report-dir", type=str,
                        default="reports/v3b_formal_tactical_ablation")
    parser.add_argument("--skip-env-check", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="Run smoke test (1 pair only)")
    args = parser.parse_args()

    out_dir = (ROOT / args.out).resolve()
    report_dir = (ROOT / args.report_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Environment check
    if not args.skip_env_check:
        print("Checking environment...")
        env_errors = _check_environment()
        if env_errors:
            for err in env_errors:
                print(f"  ERROR: {err}")
            print("Environment check FAILED. Use --skip-env-check to override.")
            return 1
        print("Environment OK.")

    # If smoke mode, use only first seed + first scenario
    seeds = args.seeds
    scenarios = args.scenarios
    if args.smoke:
        seeds = seeds[:1]
        scenarios = scenarios[:1]
        print(f"SMOKE MODE: 1 pair ({scenarios[0]}, seed={seeds[0]})")
    else:
        print(f"Running evaluation: {len(scenarios)} scenarios x {len(seeds)} seeds = {len(scenarios)*len(seeds)} pairs")

    # Run evaluation
    print(f"Start time: {datetime.now().isoformat()}")
    start_wall = time.time()

    report = _run_evaluation(
        scenarios=scenarios,
        seeds=seeds,
        artifacts_dir=args.artifacts_dir,
        policy_profile=args.policy_profile,
        timeout_steps=args.timeout_steps,
        report_dir=report_dir,
    )

    elapsed = time.time() - start_wall
    report["elapsed_seconds"] = round(elapsed, 2)
    overall = report.get("overall", {})
    print(f"\nElapsed: {elapsed:.1f}s")
    print(f"Pairs attempted: {overall.get('total_pairs_attempted', 0)}")
    print(f"Pairs completed: {overall.get('total_pairs_completed', 0)}")
    print(f"Errors: {overall.get('total_errors', 0)}")

    # Compute action divergence
    divergence = _compute_divergence(report.get("raw_pairs", []))
    report["action_divergence"] = divergence

    # Save full report
    report_path = out_dir / "paired_results.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Full report saved: {report_path}")

    # Save paired results summary (without raw pairs for readability)
    summary = {k: v for k, v in report.items() if k != "raw_pairs"}
    summary_path = out_dir / "paired_results_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Summary saved: {summary_path}")

    # Save bootstrap statistics
    stats = overall.get("statistics", {})
    stats_path = out_dir / "bootstrap_statistics.json"
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Bootstrap statistics saved: {stats_path}")

    # Print statistics table
    print("\n=== Aggregate Statistics ===")
    inner_stats = stats if "win_rate" in stats else stats.get("statistics", stats)
    print(format_statistics_table(inner_stats))

    # Print warnings
    warnings = overall.get("warnings", [])
    if warnings:
        print("\n=== Warnings ===")
        for w in warnings:
            print(f"  ! {w}")

    if overall.get("total_errors", 0) > 0:
        print(f"\n  {overall['total_errors']} game(s) with errors. See error log for details.")

    print(f"\nDone. All outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
