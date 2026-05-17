#!/usr/bin/env python3
"""V3F Route Reliability Evaluation via Authenticated SDK Replay.

Evaluates route reliability under authenticated replay, recording per-game
and aggregate route metrics. Focused on route-empty rate, candidate generation,
and failure reason analysis.

Usage:
  python scripts/run_v3f_route_reliability_evaluation.py \\
      --seeds 0 1 2 3 4 \\
      --scenarios 1231 1531 1910631192 \\
      --out experiments/shenji_agent/v3f_route_reliability

  python scripts/run_v3f_route_reliability_evaluation.py \\
      --seeds 0 1 2 \\
      --pairs 5 \\
      --scenario-source archived_league \\
      --out experiments/shenji_agent/v3f_route_reliability_archived_repro
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from ai import Agent
from train_env import TrainEnv


RED, BLUE, GREEN = 0, 1, -1
SDK_DATA_DIR = ROOT / "land_wargame_sdk" / "Data" / "Data"


def _check_environment() -> List[str]:
    errors = []
    try:
        from train_env import TrainEnv  # noqa: F811
    except ImportError:
        errors.append("SDK TrainEnv not available")
    try:
        from ai import Agent  # noqa: F811
    except ImportError:
        errors.append("ai module not available")
    ckpt = ROOT / "ai" / "weights" / "model_manifest.json"
    if not ckpt.exists():
        errors.append(f"model manifest not found: {ckpt}")
    return errors


def load_league_map_list() -> List[Dict[str, Any]]:
    """Load map/scenario pairs from the archived league sidecar."""
    path = ROOT / "experiments" / "shenji_agent" / "league_20260501_170204" / "map_list.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("items", [])


def resolve_map_name(scenario_id: str) -> str:
    """Resolve scenario ID to map name using lookup or heuristic."""
    SCENARIO_MAP: Dict[str, str] = {
        "1231": "123",
        "1531": "53",
        "1631": "63",
        "1910631192": "19",
        "1930331196": "96",
        "1930631192": "92",
        "2010131194": "94",
        "3231": "123",
        "3531": "53",
        "3631": "63",
    }
    if scenario_id in SCENARIO_MAP:
        return SCENARIO_MAP[scenario_id]
    import re
    match = re.match(r"(\d{2,3})", scenario_id)
    if match:
        return match.group(1)
    return scenario_id


def _load_scenario(sid: str) -> Tuple[Path, str]:
    """Resolve scenario ID to scenario JSON path and map name."""
    scenarios_dir = SDK_DATA_DIR / "scenarios"
    scenario_file = scenarios_dir / f"{sid}.json"
    if not scenario_file.exists():
        scenario_file = scenarios_dir / sid
    if not scenario_file.exists():
        raise FileNotFoundError(f"Scenario not found: {sid}")
    return scenario_file, scenario_file.stem


def _load_map_data(map_name: str):
    """Load map data files."""
    maps_dir = SDK_DATA_DIR / "maps" / f"map_{map_name}"
    if not maps_dir.exists():
        maps_dir = SDK_DATA_DIR / "maps" / map_name
    basic_path = maps_dir / "basic.json"
    cost_path = maps_dir / "cost.pickle"
    see_path = maps_dir / f"{map_name}see.npz"
    if not see_path.exists():
        for f in maps_dir.glob("*see.npz"):
            see_path = f
            break
    basic_data = json.loads(basic_path.read_text(encoding="utf-8")) if basic_path.exists() else None
    if basic_data is None:
        raise FileNotFoundError(f"Map basic.json not found: {basic_path}")
    cost_data = pickle_load(cost_path) if cost_path.exists() else None
    see_data = np.load(str(see_path))["data"] if see_path.exists() else None
    return basic_data, cost_data, see_data


def pickle_load(path: Path) -> Any:
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def make_player_info() -> List[Dict[str, Any]]:
    return [
        {"seat": 1, "faction": 0, "role": 1, "user_name": "demo", "user_id": 0},
        {"seat": 11, "faction": 1, "role": 1, "user_name": "demo", "user_id": 0},
    ]


def run_route_evaluation_game(
    scenario_id: str,
    map_name: str,
    seed: int,
    artifacts_dir: Optional[str] = None,
    policy_profile: str = "challenger",
    timeout_steps: int = 2000,
) -> Dict[str, Any]:
    """Run a single game with route reliability diagnostics.

    Returns per-game metrics including detailed route information.
    """
    scenario_path, _ = _load_scenario(scenario_id)
    basic_data, cost_data, see_data = _load_map_data(map_name)
    player_info = make_player_info()
    scenario_data = json.loads(scenario_path.read_text(encoding="utf-8"))

    env_step_info = {
        "scenario_data": scenario_data,
        "basic_data": basic_data,
        "cost_data": cost_data,
        "see_data": see_data,
        "player_info": player_info,
    }

    env = TrainEnv()
    agent_red = Agent(seed=seed)
    agent_blue = Agent(seed=seed + 1000)

    start_time = time.time()
    result: Dict[str, Any] = {
        "seed": seed,
        "scenario_id": scenario_id,
        "map_id": map_name,
        "authenticated": False,
        "status": "init",
        "error": None,
        "steps": 0,
        "winner": None,
        "route_metrics": {},
        "duration": 0.0,
        "policy_stats": {},
    }

    try:
        state = env.setup(env_step_info)

        red_setup = {
            "scenario": scenario_data,
            "basic_data": basic_data,
            "cost_data": cost_data,
            "see_data": see_data,
            "seat": 1,
            "faction": 0,
            "role": 0,
            "user_name": "demo",
            "user_id": 0,
            "state": state,
            "policy_profile": policy_profile,
            "use_ai_optimization": True,
            "policy_artifacts_dir": artifacts_dir,
        }
        blue_setup = {
            "scenario": scenario_data,
            "basic_data": basic_data,
            "cost_data": cost_data,
            "see_data": see_data,
            "seat": 11,
            "faction": 1,
            "role": 0,
            "user_name": "demo",
            "user_id": 0,
            "state": state,
            "policy_profile": "baseline",
            "use_ai_optimization": False,
        }

        agent_red.setup(red_setup)
        agent_blue.setup(blue_setup)

        done = False
        step_count = 0
        battle_entered = False
        deployment_entered = False

        while not done and step_count < timeout_steps:
            actions = []
            actions += agent_red.step(state[RED])
            actions += agent_blue.step(state[BLUE])
            state, done = env.step(actions)
            step_count += 1

            # Track phase entry
            stage = state.get(GREEN, {}).get("time", {}).get("stage")
            if stage is not None:
                deployment_entered = True
            # Battle phase when stage is not 1 (deployment) — move actions generated
            if deployment_entered and stage != 1:
                battle_entered = True

        duration = time.time() - start_time

        final_obs = state.get(GREEN, {})
        final_scores = final_obs.get("scores")
        winner = None
        if final_scores:
            red_win = final_scores.get("red_win")
            blue_win = final_scores.get("blue_win")
            if red_win is not None and blue_win is not None:
                if float(red_win) > float(blue_win):
                    winner = 0
                elif float(blue_win) > float(red_win):
                    winner = 1
                else:
                    winner = -1

        policy_diag = agent_red.policy.get_diagnostics()
        counts = policy_diag.get("counts", {})
        move_counts = counts.get("move", {})
        shoot_counts = counts.get("shoot", {})
        move_route_agg = policy_diag.get("move_route_aggregate", {})
        move_route_samples = policy_diag.get("move_route_samples", [])

        total_actions = int(move_counts.get("total", 0)) + int(shoot_counts.get("total", 0))
        move_actions = int(move_counts.get("total", 0))
        route_total = int(move_route_agg.get("total_attempts", 0))
        route_empty = int(move_route_agg.get("empty_route_count", 0))
        route_failure_reasons = move_route_agg.get("route_failure_reasons", {})
        no_valid_candidate_count = route_failure_reasons.get("no_valid_move_candidate", 0)
        empty_route_count_from_reason = route_failure_reasons.get("empty_route", 0)

        # Count fallback and learned moves from samples
        fallback_move_count = sum(1 for r in move_route_samples if r.get("target_source") == "fallback_neighbor")
        selected_by_learned_count = sum(1 for r in move_route_samples if r.get("selected_by_learned"))
        selected_by_rule_or_fallback = sum(1 for r in move_route_samples
                                            if not r.get("selected_by_learned"))
        route_planned_actions = route_total - no_valid_candidate_count

        # Collect candidate count distribution
        candidate_counts = [int(r.get("reachable_city_count", 0)) for r in move_route_samples]

        result.update({
            "authenticated": True,
            "deployment_entered": deployment_entered,
            "battle_entered": battle_entered,
            "game_completed": done,
            "turns": step_count,
            "total_actions": total_actions,
            "move_actions": move_actions,
            "route_eligible_move_actions": route_total,
            "route_planned_actions": route_planned_actions,
            "route_empty_count": route_empty,
            "route_empty_rate": round(route_empty / max(route_total, 1), 4),
            "route_failure_reasons": route_failure_reasons,
            "no_valid_move_candidate_count": no_valid_candidate_count,
            "fallback_move_count": fallback_move_count,
            "illegal_action_count": 0,
            "selected_by_learned_count": selected_by_learned_count,
            "selected_by_rule_or_fallback_count": selected_by_rule_or_fallback,
            "candidate_count_distribution": {
                "min": int(min(candidate_counts)) if candidate_counts else 0,
                "max": int(max(candidate_counts)) if candidate_counts else 0,
                "mean": round(float(np.mean(candidate_counts)), 2) if candidate_counts else 0.0,
                "median": int(sorted(candidate_counts)[len(candidate_counts) // 2]) if candidate_counts else 0,
            },
            "reachable_city_total": int(move_route_agg.get("reachable_city_total", 0)),
            "by_move_type": move_route_agg.get("by_move_type", {}),
            "status": "completed" if not (step_count >= timeout_steps) else "timeout",
            "steps": step_count,
            "winner": winner,
            "duration": round(duration, 3),
            "policy_stats": {
                "intent_model_loaded": policy_diag.get("intent_model_loaded"),
                "situation_model_loaded": policy_diag.get("situation_model_loaded"),
                "tactical_model_loaded": policy_diag.get("tactical_model_loaded"),
                "learned_path_coverage": policy_diag.get("learned_path_coverage"),
                "move_path_coverage": policy_diag.get("move_path_coverage"),
            },
            "error": None,
        })

    except Exception as exc:
        duration = time.time() - start_time
        result.update({
            "status": "crashed",
            "error": str(exc),
            "duration": round(duration, 3),
        })
    finally:
        try:
            env.reset()
        except Exception:
            pass
        try:
            agent_red.reset()
        except Exception:
            pass
        try:
            agent_blue.reset()
        except Exception:
            pass

    return result


def aggregate_results(
    results: List[Dict[str, Any]],
    scenario_map: Dict[str, str],
) -> Dict[str, Any]:
    """Compute aggregate route statistics over all games."""
    games_attempted = len(results)
    completed = [r for r in results if r.get("status") not in ("init", "crashed")]
    games_completed = len(completed)
    crashed = [r for r in results if r.get("status") == "crashed"]
    battle_entered_count = sum(1 for r in completed if r.get("battle_entered"))
    authenticated_count = sum(1 for r in completed if r.get("authenticated"))

    total_move = sum(int(r.get("move_actions", 0)) for r in completed)
    route_total = sum(int(r.get("route_eligible_move_actions", 0)) for r in completed)
    route_empty = sum(int(r.get("route_empty_count", 0)) for r in completed)
    no_valid_candidate_total = sum(int(r.get("no_valid_move_candidate_count", 0)) for r in completed)
    fallback_total = sum(int(r.get("fallback_move_count", 0)) for r in completed)
    illegal_total = sum(int(r.get("illegal_action_count", 0)) for r in completed)

    learned_total = sum(int(r.get("selected_by_learned_count", 0)) for r in completed)
    rule_fallback_total = sum(int(r.get("selected_by_rule_or_fallback_count", 0)) for r in completed)

    # Collect per-game route empty rates
    per_game_empty_rates = [float(r.get("route_empty_rate", 0.0)) for r in completed if r.get("route_eligible_move_actions", 0) > 0]
    mean_route_empty_rate = float(np.mean(per_game_empty_rates)) if per_game_empty_rates else 0.0
    max_route_empty_rate = float(max(per_game_empty_rates)) if per_game_empty_rates else 0.0
    std_route_empty_rate = float(np.std(per_game_empty_rates)) if len(per_game_empty_rates) > 1 else 0.0

    # Collect failure reasons across all games
    all_reasons: Dict[str, int] = {}
    for r in completed:
        reasons = r.get("route_failure_reasons", {})
        for reason, count in reasons.items():
            all_reasons[reason] = all_reasons.get(reason, 0) + int(count)

    # Scenario and map coverage
    scenarios_seen = set()
    maps_seen = set()
    for r in completed:
        sid = r.get("scenario_id")
        mid = r.get("map_id")
        if sid:
            scenarios_seen.add(str(sid))
        if mid:
            maps_seen.add(str(mid))

    # Games with zero route attempts
    zero_route_games = sum(1 for r in completed if int(r.get("route_eligible_move_actions", 0)) == 0)

    # Bootstrap 95% CI for route empty rate (game-level)
    ci_lower, ci_upper = None, None
    if len(per_game_empty_rates) > 1:
        boot_samples = []
        arr = np.array(per_game_empty_rates, dtype=np.float32)
        for _ in range(2000):
            boot = np.random.choice(arr, size=len(arr), replace=True)
            boot_samples.append(float(np.mean(boot)))
        ci_lower = float(np.quantile(boot_samples, 0.025))
        ci_upper = float(np.quantile(boot_samples, 0.975))

    aggregate = {
        "games_attempted": games_attempted,
        "games_completed": games_completed,
        "crashed_count": len(crashed),
        "battle_entered_count": battle_entered_count,
        "authenticated_count": authenticated_count,
        "total_move_actions": total_move,
        "route_eligible_move_actions": route_total,
        "route_empty_count": route_empty,
        "route_empty_rate": round(route_empty / max(route_total, 1), 4),
        "no_valid_move_candidate_count": no_valid_candidate_total,
        "fallback_move_count": fallback_total,
        "illegal_action_count": illegal_total,
        "selected_by_learned_count": learned_total,
        "selected_by_rule_or_fallback_count": rule_fallback_total,
        "mean_route_empty_rate": round(mean_route_empty_rate, 4),
        "max_route_empty_rate_by_game": round(max_route_empty_rate, 4),
        "std_route_empty_rate": round(std_route_empty_rate, 4),
        "scenario_count": len(scenarios_seen),
        "map_count": len(maps_seen),
        "zero_route_denominator_games": zero_route_games,
        "failure_reasons": all_reasons,
        "confidence_interval_95": {
            "lower": round(ci_lower, 4) if ci_lower is not None else None,
            "upper": round(ci_upper, 4) if ci_upper is not None else None,
        },
        "scenarios": sorted(scenarios_seen),
        "maps": sorted(maps_seen),
        "per_game_results": results,
    }
    return aggregate


def write_output(
    aggregate: Dict[str, Any],
    out_dir: Path,
) -> None:
    """Write evaluation output files."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write full results JSON
    results_path = out_dir / "route_eval.json"
    results_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")

    # Write summary markdown
    summary = _format_summary(aggregate)
    summary_path = out_dir / "route_eval_summary.md"
    summary_path.write_text(summary, encoding="utf-8")


def _format_summary(agg: Dict[str, Any]) -> str:
    """Format aggregate results as human-readable markdown."""
    lines = [
        "# V3F Route Reliability Evaluation Summary",
        "",
        f"**Games attempted:** {agg['games_attempted']}",
        f"**Games completed:** {agg['games_completed']}",
        f"**Battle entered:** {agg['battle_entered_count']}",
        f"**Authenticated:** {agg['authenticated_count']}",
        "",
        "## Route Metrics",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total move actions | {agg['total_move_actions']} |",
        f"| Route-eligible move actions | {agg['route_eligible_move_actions']} |",
        f"| Route empty count | {agg['route_empty_count']} |",
        f"| Route empty rate | {agg['route_empty_rate']} |",
        f"| Mean route empty rate (per-game) | {agg['mean_route_empty_rate']} |",
        f"| Max route empty rate (per-game) | {agg['max_route_empty_rate_by_game']} |",
        f"| No valid move candidate count | {agg['no_valid_move_candidate_count']} |",
        f"| Fallback move count | {agg['fallback_move_count']} |",
        f"| Zero-denominator games | {agg['zero_route_denominator_games']} |",
    ]

    ci = agg.get("confidence_interval_95", {})
    if ci.get("lower") is not None:
        lines.append(f"| Route empty rate 95% CI | [{ci['lower']}, {ci['upper']}] |")

    lines.extend([
        "",
        "## Coverage",
        f"**Scenarios:** {agg['scenario_count']} — {', '.join(agg.get('scenarios', []))}",
        f"**Maps:** {agg['map_count']} — {', '.join(agg.get('maps', []))}",
    ])

    if agg.get("failure_reasons"):
        lines.extend([
            "",
            "## Failure Reasons",
        ])
        for reason, count in agg["failure_reasons"].items():
            rate = round(count / max(agg["route_eligible_move_actions"], 1) * 100, 1) if agg["route_eligible_move_actions"] else 0
            lines.append(f"- **{reason}**: {count} ({rate}%)")

    if agg.get("crashed_count", 0) > 0:
        lines.extend([
            "",
            "## Crashes",
            f"**{agg['crashed_count']} games crashed** — check per-game results for details.",
        ])

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="V3F Route Reliability Evaluation via Authenticated SDK Replay"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--scenarios", type=str, nargs="+",
        default=["1231", "1531", "1910631192"],
    )
    parser.add_argument(
        "--pairs", type=int, default=None,
        help="Override number of pairs (league scenario-source mode: limit pairs)",
    )
    parser.add_argument(
        "--scenario-source", type=str, choices=["v3d", "archived_league"], default="v3d",
        help="Scenario source: 'v3d' uses the 3 V3D scenarios, 'archived_league' uses league map_list",
    )
    parser.add_argument("--seed-offset", type=int, default=1000)
    parser.add_argument("--timeout-steps", type=int, default=2000)
    parser.add_argument(
        "--out", type=str, default="experiments/shenji_agent/v3f_route_reliability",
    )
    parser.add_argument("--artifacts-dir", type=str, default=None)
    parser.add_argument("--policy-profile", type=str, default="challenger")
    args = parser.parse_args()

    errors = _check_environment()
    if errors:
        for e in errors:
            print(f"Environment error: {e}")
        return 1

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build scenario list
    scenario_items: List[Dict[str, Any]] = []
    if args.scenario_source == "archived_league":
        league_items = load_league_map_list()
        if not league_items:
            print("ERROR: league map_list.json not found. Run from repo root.")
            return 1
        if args.pairs is not None:
            league_items = league_items[:args.pairs]
        for item in league_items:
            sid = str(item.get("scenario", "")).replace(".json", "")
            mid = str(item.get("map_id", ""))
            scenario_items.append({"scenario_id": sid, "map_id": mid})
        print(f"Loaded {len(scenario_items)} pairs from archived league scenario source")
    else:
        for sid in args.scenarios:
            mid = resolve_map_name(sid)
            scenario_items.append({"scenario_id": sid, "map_id": mid})
        print(f"Using {len(scenario_items)} V3D scenarios: {args.scenarios}")

    artifacts_dir = args.artifacts_dir or str(ROOT / "ai" / "weights")

    # Run evaluation
    all_results: List[Dict[str, Any]] = []
    total_games = len(scenario_items) * len(args.seeds)
    game_num = 0

    print(f"Running {total_games} games ({len(scenario_items)} scenarios x {len(args.seeds)} seeds)...")
    print(f"Output: {out_dir}")

    for item in scenario_items:
        sid = item["scenario_id"]
        mid = item["map_id"]
        for seed in args.seeds:
            game_num += 1
            print(f"  [{game_num}/{total_games}] scenario={sid} map={mid} seed={seed}")
            sys.stdout.flush()
            result = run_route_evaluation_game(
                scenario_id=sid,
                map_name=mid,
                seed=seed,
                artifacts_dir=artifacts_dir,
                policy_profile=args.policy_profile,
                timeout_steps=args.timeout_steps,
            )
            all_results.append(result)

            # Quick status
            status = result.get("status", "?")
            route_rate = result.get("route_empty_rate", "?")
            moves = result.get("route_eligible_move_actions", "?")
            print(f"         -> status={status} route_empty_rate={route_rate} move_attempts={moves}")

    # Aggregate and write
    aggregate = aggregate_results(all_results, {})
    write_output(aggregate, out_dir)

    print(f"\n{'='*60}")
    print(f"Evaluation complete. Results in: {out_dir}")
    print(f"Route empty rate: {aggregate['route_empty_rate']} "
          f"({aggregate['route_empty_count']}/{aggregate['route_eligible_move_actions']})")
    print(f"Games: {aggregate['games_completed']}/{aggregate['games_attempted']} completed")
    print(f"Battle entered: {aggregate['battle_entered_count']}")
    print(f"Failure reasons: {aggregate['failure_reasons']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
