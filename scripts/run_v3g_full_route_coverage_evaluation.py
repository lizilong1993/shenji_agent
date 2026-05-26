#!/usr/bin/env python3
"""V3G Full Route Coverage Evaluation via Authenticated SDK Replay.

Evaluates route reliability across the full archived league scenario set
with enhanced instrumentation for route planner metrics.  Superset of the
V3F evaluator: collects all V3F fields PLUS separate "SDK Move available"
from "route planner failed" from "fallback failed" counts, explicit route
planner invocation/success/empty-path counters, and per-move-type breakdowns.

Usage (full):
  python scripts/run_v3g_full_route_coverage_evaluation.py \
      --scenario-source archived_league \
      --max-scenarios 100 \
      --seeds 0 \
      --out experiments/shenji_agent/v3g_full_route_coverage

Usage (smoke):
  python scripts/run_v3g_full_route_coverage_evaluation.py \
      --scenario-source archived_league \
      --max-scenarios 5 \
      --seeds 0 \
      --out experiments/shenji_agent/v3g_full_route_coverage_smoke

Output files (all written to --out dir):
  results.json       Aggregate metrics (all the V3F fields + enhanced V3G fields)
  per_game.jsonl     One JSON line per game
  per_scenario.csv   CSV with one row per scenario
  summary.md         Markdown summary
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai import Agent
from ai.agent import ActionType
from train_env import TrainEnv


RED, BLUE, GREEN = 0, 1, -1
SDK_DATA_DIR = ROOT / "land_wargame_sdk" / "Data" / "Data"

# Move type name mapping (mirrors agent.py)
MOVE_TYPE_NAMES = {0: "Maneuver", 1: "March", 2: "Walk", 3: "Fly"}


# ---------------------------------------------------------------------------
# Environment helpers (shared with V3F evaluator)
# ---------------------------------------------------------------------------

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
    match = re.match(r"(\d{2,3})", scenario_id)
    if match:
        return match.group(1)
    return scenario_id


def _load_scenario(sid: str) -> Tuple[Path, str]:
    """Resolve scenario ID to scenario JSON path and stem."""
    scenarios_dir = SDK_DATA_DIR / "scenarios"
    scenario_file = scenarios_dir / f"{sid}.json"
    if not scenario_file.exists():
        scenario_file = scenarios_dir / sid
    if not scenario_file.exists():
        raise FileNotFoundError(f"Scenario not found: {sid}")
    return scenario_file, scenario_file.stem


def _load_map_data(map_name: str):
    """Load map data files (basic.json, cost.pickle, see.npz)."""
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


# ---------------------------------------------------------------------------
# Per-game evaluation
# ---------------------------------------------------------------------------

def run_v3g_route_evaluation_game(
    scenario_id: str,
    map_name: str,
    seed: int,
    artifacts_dir: Optional[str] = None,
    policy_profile: str = "challenger",
    timeout_steps: int = 2000,
) -> Dict[str, Any]:
    """Run a single V3G game with enhanced route diagnostics.

    Returns a per-game metrics dict with all fields required for the
    per_game.jsonl output format.
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
        "turns": 0,
        "winner": None,
        "duration": 0.0,
        "policy_stats": {},
    }

    # SDK move key presence counters (tracked externally from valid_actions)
    sdk_move_key_present = 0
    sdk_move_key_absent = 0

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
            # -- Track SDK Move key presence from current RED observation --
            obs_red = state.get(RED, {})
            valid_actions_red = obs_red.get("valid_actions", {})
            move_type = ActionType.Move  # Move == 1
            for obj_id, action_map in valid_actions_red.items():
                if not isinstance(action_map, dict):
                    continue
                if move_type in action_map:
                    sdk_move_key_present += 1
                else:
                    sdk_move_key_absent += 1

            actions = []
            actions += agent_red.step(state[RED])
            actions += agent_blue.step(state[BLUE])
            state, done = env.step(actions)
            step_count += 1

            # Track phase entry
            stage = state.get(GREEN, {}).get("time", {}).get("stage")
            if stage is not None:
                deployment_entered = True
            if deployment_entered and stage != 1:
                battle_entered = True

        duration = time.time() - start_time

        # Determine winner
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

        # -- Extract diagnostics from RED agent's policy --
        policy_diag = agent_red.policy.get_diagnostics()
        counts = policy_diag.get("counts", {})
        move_counts = counts.get("move", {})
        shoot_counts = counts.get("shoot", {})

        move_route_agg = policy_diag.get("move_route_aggregate", {})
        move_route_samples = policy_diag.get("move_route_samples", [])

        # Aggregate-level counters (cover ALL move actions, not just samples)
        total_attempts = int(move_route_agg.get("total_attempts", 0))
        empty_route_count = int(move_route_agg.get("empty_route_count", 0))
        route_failure_reasons = move_route_agg.get("route_failure_reasons", {})
        by_move_type = move_route_agg.get("by_move_type", {})

        total_actions = int(move_counts.get("total", 0)) + int(shoot_counts.get("total", 0))
        move_actions = int(move_counts.get("total", 0))

        no_valid_candidate_count = route_failure_reasons.get("no_valid_move_candidate", 0)

        # -- Sample-level counters (up to 50 samples per game) --
        n_samples = len(move_route_samples)

        reachable_positive = sum(
            1 for r in move_route_samples if int(r.get("reachable_city_count", 0)) > 0
        )
        reachable_zero = sum(
            1 for r in move_route_samples if int(r.get("reachable_city_count", 0)) == 0
        )
        fallback_move_count = sum(
            1 for r in move_route_samples if r.get("target_source") == "fallback_neighbor"
        )
        selected_by_learned_count = sum(
            1 for r in move_route_samples if r.get("selected_by_learned")
        )

        # Candidate count distribution from samples
        candidate_counts = [int(r.get("reachable_city_count", 0)) for r in move_route_samples]

        # Move type breakdowns
        selected_move_type_counts: Dict[str, int] = {}
        candidate_move_type_counts: Dict[str, int] = {}
        for mt_name, mt_data in by_move_type.items():
            total_mt = int(mt_data.get("total", 0))
            empty_mt = int(mt_data.get("empty", 0))
            candidate_move_type_counts[mt_name] = total_mt
            selected_move_type_counts[mt_name] = total_mt - empty_mt

        result.update({
            "authenticated": True,
            "deployment_entered": deployment_entered,
            "battle_entered": battle_entered,
            "game_completed": done,
            "turns": step_count,
            "total_actions": total_actions,
            "move_actions": move_actions,
            "route_eligible_move_actions": total_attempts,
            # Route planner metrics (from aggregate + samples)
            "route_planner_invocation_count": total_attempts,
            "route_planner_success_count": reachable_positive,
            "route_planner_empty_path_count": reachable_zero,
            # Route empty metrics
            "route_empty_count": empty_route_count,
            "route_empty_rate": round(empty_route_count / max(total_attempts, 1), 4),
            "no_valid_move_candidate_count": no_valid_candidate_count,
            # Fallback
            "fallback_move_count": fallback_move_count,
            # Action outcome counts
            "illegal_action_count": 0,
            "no_op_count": 0,
            # SDK Move key presence
            "sdk_move_key_present_count": sdk_move_key_present,
            "sdk_move_key_absent_count": sdk_move_key_absent,
            # Move type breakdowns
            "selected_move_type_counts": selected_move_type_counts,
            "candidate_move_type_counts": candidate_move_type_counts,
            # Failure reason breakdown
            "failure_reasons": dict(route_failure_reasons),
            # Runtime error
            "runtime_errors": None,
            # Candidate count distribution
            "candidate_count_distribution": {
                "min": int(min(candidate_counts)) if candidate_counts else 0,
                "max": int(max(candidate_counts)) if candidate_counts else 0,
                "mean": round(float(np.mean(candidate_counts)), 2) if candidate_counts else 0.0,
                "median": (
                    int(sorted(candidate_counts)[len(candidate_counts) // 2])
                    if candidate_counts else 0
                ),
            },
            # Reachable city total (from aggregate)
            "reachable_city_total": int(move_route_agg.get("reachable_city_total", 0)),
            # Phase entry
            "selected_by_learned_count": selected_by_learned_count,
            # Legacy V3F fields for backward compat
            "selected_by_rule_or_fallback_count": n_samples - selected_by_learned_count,
            "route_failure_reasons": dict(route_failure_reasons),
            "by_move_type": dict(by_move_type),
            # Status
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


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_v3g_results(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute aggregate V3G route statistics over all games."""
    games_attempted = len(results)
    completed = [r for r in results if r.get("status") not in ("init", "crashed")]
    games_completed = len(completed)
    crashed = [r for r in results if r.get("status") == "crashed"]
    battle_entered_count = sum(1 for r in completed if r.get("battle_entered"))
    authenticated_count = sum(1 for r in completed if r.get("authenticated"))

    # -- Aggregate counters (summed across all completed games) --
    total_move = sum(int(r.get("move_actions", 0)) for r in completed)
    route_eligible = sum(int(r.get("route_eligible_move_actions", 0)) for r in completed)
    route_empty = sum(int(r.get("route_empty_count", 0)) for r in completed)
    route_empty_rate = round(route_empty / max(route_eligible, 1), 4)

    route_planner_invocation = sum(
        int(r.get("route_planner_invocation_count", 0)) for r in completed
    )
    route_planner_success = sum(
        int(r.get("route_planner_success_count", 0)) for r in completed
    )
    route_planner_empty = sum(
        int(r.get("route_planner_empty_path_count", 0)) for r in completed
    )

    no_valid_candidate_total = sum(
        int(r.get("no_valid_move_candidate_count", 0)) for r in completed
    )
    fallback_total = sum(int(r.get("fallback_move_count", 0)) for r in completed)
    illegal_total = sum(int(r.get("illegal_action_count", 0)) for r in completed)
    no_op_total = sum(int(r.get("no_op_count", 0)) for r in completed)

    sdk_present_total = sum(
        int(r.get("sdk_move_key_present_count", 0)) for r in completed
    )
    sdk_absent_total = sum(
        int(r.get("sdk_move_key_absent_count", 0)) for r in completed
    )

    learned_total = sum(
        int(r.get("selected_by_learned_count", 0)) for r in completed
    )

    # -- Per-game route empty rates --
    per_game_empty_rates = [
        float(r.get("route_empty_rate", 0.0))
        for r in completed
        if int(r.get("route_eligible_move_actions", 0)) > 0
    ]
    mean_route_empty_rate = float(np.mean(per_game_empty_rates)) if per_game_empty_rates else 0.0
    max_route_empty_rate = float(max(per_game_empty_rates)) if per_game_empty_rates else 0.0
    std_route_empty_rate = float(np.std(per_game_empty_rates)) if len(per_game_empty_rates) > 1 else 0.0

    # -- Failure reasons across all games --
    all_reasons: Dict[str, int] = {}
    for r in completed:
        reasons = r.get("failure_reasons", {})
        for reason, count in reasons.items():
            all_reasons[reason] = all_reasons.get(reason, 0) + int(count)

    top_failure_reasons = sorted(
        [
            {
                "reason": reason,
                "count": count,
                "rate": round(count / max(route_eligible, 1), 4),
            }
            for reason, count in all_reasons.items()
        ],
        key=lambda x: x["count"],
        reverse=True,
    )

    # -- Empty rate by scenario --
    scenario_groups: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"empty": 0, "eligible": 0, "move_attempts": 0}
    )
    map_groups: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"empty": 0, "eligible": 0}
    )
    seed_groups: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"empty": 0, "eligible": 0}
    )
    move_type_groups: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"empty": 0, "eligible": 0}
    )

    for r in completed:
        sid = str(r.get("scenario_id", "?"))
        mid = str(r.get("map_id", "?"))
        seed = str(r.get("seed", "?"))
        empty = int(r.get("route_empty_count", 0))
        eligible = int(r.get("route_eligible_move_actions", 0))
        move_acts = int(r.get("move_actions", 0))

        scenario_groups[sid]["empty"] += empty
        scenario_groups[sid]["eligible"] += eligible
        scenario_groups[sid]["move_attempts"] += move_acts

        map_groups[mid]["empty"] += empty
        map_groups[mid]["eligible"] += eligible

        seed_groups[seed]["empty"] += empty
        seed_groups[seed]["eligible"] += eligible

        # Per move type within each game
        sel_mt = r.get("selected_move_type_counts", {})
        for mt_name, mt_count in sel_mt.items():
            # We approximate per-move-type empty: the selected count includes
            # non-empty moves.  But a better proxy is using by_move_type from
            # per-game, which has "total" and "empty" per move type.
            pass

    # Actually populate move_type_groups from per-game by_move_type
    for r in completed:
        by_mt = r.get("by_move_type", {})
        for mt_name, mt_data in by_mt.items():
            mt_total = int(mt_data.get("total", 0))
            mt_empty = int(mt_data.get("empty", 0))
            move_type_groups[mt_name]["eligible"] += mt_total
            move_type_groups[mt_name]["empty"] += mt_empty

    empty_rate_by_scenario = {
        sid: round(g["empty"] / max(g["eligible"], 1), 4)
        for sid, g in scenario_groups.items()
    }
    empty_rate_by_map = {
        mid: round(g["empty"] / max(g["eligible"], 1), 4)
        for mid, g in map_groups.items()
    }
    empty_rate_by_move_type = {
        mt: round(g["empty"] / max(g["eligible"], 1), 4)
        for mt, g in move_type_groups.items()
    }
    empty_rate_by_seed = {
        seed: round(g["empty"] / max(g["eligible"], 1), 4)
        for seed, g in seed_groups.items()
    }

    # -- Scenario-level empty rate stats --
    scenario_empty_rates = list(empty_rate_by_scenario.values())
    max_scenario_empty_rate = max(scenario_empty_rates) if scenario_empty_rates else 0.0
    mean_scenario_empty_rate = float(np.mean(scenario_empty_rates)) if scenario_empty_rates else 0.0
    sorted_scenario_rates = sorted(scenario_empty_rates)
    median_scenario_empty_rate = (
        sorted_scenario_rates[len(sorted_scenario_rates) // 2]
        if sorted_scenario_rates
        else 0.0
    )

    # -- Worst scenarios --
    worst_scenarios = sorted(
        [
            {
                "scenario_id": sid,
                "map_id": str(
                    next(
                        (
                            r.get("map_id")
                            for r in completed
                            if str(r.get("scenario_id", "")) == sid
                        ),
                        "?",
                    )
                ),
                "empty_rate": round(g["empty"] / max(g["eligible"], 1), 4),
                "move_attempts": g["move_attempts"],
            }
            for sid, g in scenario_groups.items()
        ],
        key=lambda x: x["empty_rate"],
        reverse=True,
    )

    # -- Specific scenario empty rate --
    scenario_1910631192_empty_rate = empty_rate_by_scenario.get("1910631192", 0.0)

    # -- Scenario and map coverage --
    scenarios_seen = sorted(
        {str(r.get("scenario_id", "")) for r in completed if r.get("scenario_id")}
    )
    maps_seen = sorted(
        {str(r.get("map_id", "")) for r in completed if r.get("map_id")}
    )
    zero_route_games = sum(
        1 for r in completed if int(r.get("route_eligible_move_actions", 0)) == 0
    )

    # -- Bootstrap 95% CI for route empty rate (game-level) --
    ci_lower, ci_upper = None, None
    if len(per_game_empty_rates) > 1:
        boot_samples = []
        arr = np.array(per_game_empty_rates, dtype=np.float32)
        for _ in range(2000):
            boot = np.random.choice(arr, size=len(arr), replace=True)
            boot_samples.append(float(np.mean(boot)))
        ci_lower = float(np.quantile(boot_samples, 0.025))
        ci_upper = float(np.quantile(boot_samples, 0.975))

    # -- Aggregate per-move-type counts --
    agg_selected_move_type_counts: Dict[str, int] = {}
    agg_candidate_move_type_counts: Dict[str, int] = {}
    for r in completed:
        sel_mt = r.get("selected_move_type_counts", {})
        cand_mt = r.get("candidate_move_type_counts", {})
        for mt_name, count in sel_mt.items():
            agg_selected_move_type_counts[mt_name] = (
                agg_selected_move_type_counts.get(mt_name, 0) + int(count)
            )
        for mt_name, count in cand_mt.items():
            agg_candidate_move_type_counts[mt_name] = (
                agg_candidate_move_type_counts.get(mt_name, 0) + int(count)
            )

    aggregate: Dict[str, Any] = {
        # -- V3F backward-compatible fields --
        "games_attempted": games_attempted,
        "games_completed": games_completed,
        "crashed_count": len(crashed),
        "battle_entered_count": battle_entered_count,
        "authenticated_count": authenticated_count,
        "total_move_actions": total_move,
        "route_eligible_move_actions": route_eligible,
        "route_empty_count": route_empty,
        "route_empty_rate": route_empty_rate,
        "mean_route_empty_rate": round(mean_route_empty_rate, 4),
        "max_route_empty_rate_by_game": round(max_route_empty_rate, 4),
        "std_route_empty_rate": round(std_route_empty_rate, 4),
        "scenario_count": len(scenarios_seen),
        "map_count": len(maps_seen),
        "zero_route_denominator_games": zero_route_games,
        "failure_reasons": dict(all_reasons),
        "confidence_interval_95": {
            "lower": round(ci_lower, 4) if ci_lower is not None else None,
            "upper": round(ci_upper, 4) if ci_upper is not None else None,
        },
        "scenarios": scenarios_seen,
        "maps": maps_seen,
        "selected_by_learned_count": learned_total,
        # -- V3G enhanced fields --
        "route_planner_invocation_count": route_planner_invocation,
        "route_planner_success_count": route_planner_success,
        "route_planner_empty_path_count": route_planner_empty,
        "no_valid_move_candidate_count": no_valid_candidate_total,
        "fallback_move_count": fallback_total,
        "illegal_action_count": illegal_total,
        "no_op_count": no_op_total,
        "sdk_move_key_present_count": sdk_present_total,
        "sdk_move_key_absent_count": sdk_absent_total,
        # -- Per-group empty rates --
        "empty_rate_by_scenario": empty_rate_by_scenario,
        "empty_rate_by_map": empty_rate_by_map,
        "empty_rate_by_move_type": empty_rate_by_move_type,
        "empty_rate_by_seed": empty_rate_by_seed,
        "max_scenario_empty_rate": round(max_scenario_empty_rate, 4),
        "mean_scenario_empty_rate": round(mean_scenario_empty_rate, 4),
        "median_scenario_empty_rate": round(median_scenario_empty_rate, 4),
        "top_failure_reasons": top_failure_reasons,
        "worst_scenarios": worst_scenarios,
        "scenario_1910631192_empty_rate": round(scenario_1910631192_empty_rate, 4),
        # -- Aggregate move type breakdowns --
        "selected_move_type_counts": agg_selected_move_type_counts,
        "candidate_move_type_counts": agg_candidate_move_type_counts,
    }
    return aggregate


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def write_v3g_output(
    aggregate: Dict[str, Any],
    out_dir: Path,
) -> None:
    """Write V3G evaluation output files (results.json, per_game.jsonl,
    per_scenario.csv, summary.md).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Write per_game.jsonl (one JSON line per game) --
    per_game_results = aggregate.get("per_game_results", [])
    per_game_path = out_dir / "per_game.jsonl"
    with open(per_game_path, "w", encoding="utf-8") as f:
        for game in per_game_results:
            f.write(json.dumps(game, sort_keys=True) + "\n")

    # -- Write per_scenario.csv --
    per_scenario_path = out_dir / "per_scenario.csv"
    _write_per_scenario_csv(per_scenario_path, per_game_results)

    # -- Write results.json (aggregate, WITHOUT embedded per_game_results) --
    results_out = dict(aggregate)
    results_out.pop("per_game_results", None)  # not embedded in results.json
    results_path = out_dir / "results.json"
    results_path.write_text(
        json.dumps(results_out, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # -- Write summary.md --
    summary = _format_summary(results_out)
    summary_path = out_dir / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")


def _write_per_scenario_csv(path: Path, per_game_results: List[Dict[str, Any]]) -> None:
    """Aggregate per-game results per scenario and write CSV."""
    scenario_rows: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "scenario_id": "",
        "map_id": "",
        "games": 0,
        "move_actions": 0,
        "route_eligible": 0,
        "route_empty": 0,
        "no_valid_candidate": 0,
        "route_planner_invoked": 0,
        "route_planner_empty": 0,
        "fallback_count": 0,
        "sdk_move_unavailable": 0,
    })

    for game in per_game_results:
        sid = str(game.get("scenario_id", ""))
        mid = str(game.get("map_id", ""))
        row = scenario_rows[sid]
        row["scenario_id"] = sid
        row["map_id"] = mid
        row["games"] += 1
        row["move_actions"] += int(game.get("move_actions", 0))
        row["route_eligible"] += int(game.get("route_eligible_move_actions", 0))
        row["route_empty"] += int(game.get("route_empty_count", 0))
        row["no_valid_candidate"] += int(game.get("no_valid_move_candidate_count", 0))
        row["route_planner_invoked"] += int(game.get("route_planner_invocation_count", 0))
        row["route_planner_empty"] += int(game.get("route_planner_empty_path_count", 0))
        row["fallback_count"] += int(game.get("fallback_move_count", 0))
        row["sdk_move_unavailable"] += int(game.get("sdk_move_key_absent_count", 0))

    fieldnames = [
        "scenario_id",
        "map_id",
        "games",
        "move_actions",
        "route_eligible",
        "route_empty",
        "route_empty_rate",
        "no_valid_candidate",
        "route_planner_invoked",
        "route_planner_empty",
        "fallback_count",
        "sdk_move_unavailable",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sid in sorted(scenario_rows.keys()):
            row = scenario_rows[sid]
            eligible = max(int(row["route_eligible"]), 1)
            row["route_empty_rate"] = round(int(row["route_empty"]) / eligible, 4)
            writer.writerow(row)


def _format_summary(agg: Dict[str, Any]) -> str:
    """Format aggregate results as human-readable markdown."""
    lines = [
        "# V3G Full Route Coverage Evaluation Summary",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overview",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Games attempted | {agg['games_attempted']} |",
        f"| Games completed | {agg['games_completed']} |",
        f"| Battle entered | {agg['battle_entered_count']} |",
        f"| Authenticated | {agg['authenticated_count']} |",
        f"| Crashed | {agg['crashed_count']} |",
        f"| Scenarios | {agg['scenario_count']} |",
        f"| Maps | {agg['map_count']} |",
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
        f"| Std route empty rate (per-game) | {agg['std_route_empty_rate']} |",
        "",
        "## Route Planner Metrics",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Route planner invocations | {agg['route_planner_invocation_count']} |",
        f"| Route planner success (path found) | {agg['route_planner_success_count']} |",
        f"| Route planner empty path | {agg['route_planner_empty_path_count']} |",
        f"| No valid move candidate | {agg['no_valid_move_candidate_count']} |",
        f"| Fallback move count | {agg['fallback_move_count']} |",
        f"| SDK Move key present | {agg['sdk_move_key_present_count']} |",
        f"| SDK Move key absent | {agg['sdk_move_key_absent_count']} |",
    ]

    ci = agg.get("confidence_interval_95", {})
    if ci.get("lower") is not None:
        lines.append(
            f"| Route empty rate 95% CI | [{ci['lower']}, {ci['upper']}] |"
        )
    lines.append(f"| Zero-denominator games | {agg.get('zero_route_denominator_games', 0)} |")

    # Per-scenario rates
    lines.extend([
        "",
        "## Empty Rate by Scenario",
        "",
    ])
    scenario_rates = agg.get("empty_rate_by_scenario", {})
    if scenario_rates:
        lines.append(f"| Scenario | Empty Rate |")
        lines.append(f"|---|---|")
        for sid in sorted(scenario_rates.keys()):
            lines.append(f"| {sid} | {scenario_rates[sid]} |")
        lines.append("")
        lines.append(
            f"**Max scenario empty rate:** {agg.get('max_scenario_empty_rate', '?')}"
        )
        lines.append(
            f"**Mean scenario empty rate:** {agg.get('mean_scenario_empty_rate', '?')}"
        )
        lines.append(
            f"**Median scenario empty rate:** {agg.get('median_scenario_empty_rate', '?')}"
        )
        lines.append(
            f"**Scenario 1910631192 empty rate:** {agg.get('scenario_1910631192_empty_rate', '?')}"
        )

    # Empty rate by map
    lines.extend([
        "",
        "## Empty Rate by Map",
        "",
    ])
    map_rates = agg.get("empty_rate_by_map", {})
    if map_rates:
        lines.append(f"| Map | Empty Rate |")
        lines.append(f"|---|---|")
        for mid in sorted(map_rates.keys()):
            lines.append(f"| {mid} | {map_rates[mid]} |")

    # Move type breakdown
    lines.extend([
        "",
        "## Move Type Breakdown",
        "",
    ])
    sel_mt = agg.get("selected_move_type_counts", {})
    cand_mt = agg.get("candidate_move_type_counts", {})
    if cand_mt:
        lines.append(f"| Move Type | Candidate | Selected | Empty Rate |")
        lines.append(f"|---|---|---|---|")
        for mt_name in sorted(cand_mt.keys()):
            cand = cand_mt.get(mt_name, 0)
            sel = sel_mt.get(mt_name, 0)
            mt_empty_rate = agg.get("empty_rate_by_move_type", {}).get(mt_name, 0)
            lines.append(f"| {mt_name} | {cand} | {sel} | {mt_empty_rate} |")

    # Failure reasons
    top_reasons = agg.get("top_failure_reasons", [])
    if top_reasons:
        lines.extend([
            "",
            "## Top Failure Reasons",
            "",
            f"| Reason | Count | Rate |",
            f"|---|---|---|",
        ])
        for fr in top_reasons:
            lines.append(
                f"| {fr['reason']} | {fr['count']} | {fr['rate']} |"
            )

    # Worst scenarios
    worst = agg.get("worst_scenarios", [])
    if worst:
        lines.extend([
            "",
            "## Worst Scenarios (by empty rate)",
            "",
            f"| Scenario | Map | Empty Rate | Move Attempts |",
            f"|---|---|---|---|",
        ])
        for ws in worst[:10]:
            lines.append(
                f"| {ws['scenario_id']} | {ws['map_id']} | {ws['empty_rate']} | {ws['move_attempts']} |"
            )

    if agg.get("crashed_count", 0) > 0:
        lines.extend([
            "",
            "## Crashes",
            f"**{agg['crashed_count']} games crashed** — check per_game.jsonl for details.",
        ])

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="V3G Full Route Coverage Evaluation via Authenticated SDK Replay"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0],
        help="Seeds to evaluate (default: [0])",
    )
    parser.add_argument(
        "--scenarios", type=str, nargs="+", default=None,
        help="Specific scenario IDs (ignored when --scenario-source=archived_league)",
    )
    parser.add_argument(
        "--scenario-source", type=str,
        choices=["v3d", "archived_league", "explicit"],
        default="archived_league",
        help="Scenario source: 'v3d' uses default V3D scenarios, "
             "'archived_league' uses league map_list, 'explicit' uses --scenarios",
    )
    parser.add_argument(
        "--max-scenarios", type=int, default=None,
        help="Limit number of scenarios from the league (for smoke tests)",
    )
    parser.add_argument("--seed-offset", type=int, default=1000)
    parser.add_argument("--timeout-steps", type=int, default=2000)
    parser.add_argument(
        "--out", type=str,
        default="experiments/shenji_agent/v3g_full_route_coverage",
    )
    parser.add_argument("--artifacts-dir", type=str, default=None)
    parser.add_argument("--policy-profile", type=str, default="challenger")
    args = parser.parse_args()

    # Environment check
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
        if args.max_scenarios is not None:
            league_items = league_items[:args.max_scenarios]
        for item in league_items:
            sid = str(item.get("scenario", "")).replace(".json", "")
            mid = str(item.get("map_id", ""))
            scenario_items.append({"scenario_id": sid, "map_id": mid})
        print(
            f"Loaded {len(scenario_items)} pairs from archived league scenario source"
        )

    elif args.scenario_source == "v3d":
        default_scenarios = args.scenarios or ["1231", "1531", "1910631192"]
        for sid in default_scenarios:
            mid = resolve_map_name(sid)
            scenario_items.append({"scenario_id": sid, "map_id": mid})
        print(f"Using {len(scenario_items)} V3D scenarios: {default_scenarios}")

    else:  # explicit
        if not args.scenarios:
            print("ERROR: --scenarios required when --scenario-source=explicit")
            return 1
        for sid in args.scenarios:
            mid = resolve_map_name(sid)
            scenario_items.append({"scenario_id": sid, "map_id": mid})
        print(f"Using {len(scenario_items)} explicitly specified scenarios")

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
            result = run_v3g_route_evaluation_game(
                scenario_id=sid,
                map_name=mid,
                seed=seed,
                artifacts_dir=artifacts_dir,
                policy_profile=args.policy_profile,
                timeout_steps=args.timeout_steps,
            )
            all_results.append(result)

            status = result.get("status", "?")
            route_rate = result.get("route_empty_rate", "?")
            moves = result.get("route_eligible_move_actions", "?")
            print(
                f"         -> status={status} "
                f"route_empty_rate={route_rate} "
                f"move_attempts={moves}"
            )

    # Embed per-game results in aggregate for output writing
    aggregate = aggregate_v3g_results(all_results)
    aggregate["per_game_results"] = all_results

    # Write output files
    write_v3g_output(aggregate, out_dir)

    # Print console summary
    print(f"\n{'=' * 60}")
    print(f"Evaluation complete. Results in: {out_dir}")
    print(f"Route empty rate: {aggregate['route_empty_rate']} "
          f"({aggregate['route_empty_count']}/{aggregate['route_eligible_move_actions']})")
    print(f"Route planner: {aggregate['route_planner_success_count']} success / "
          f"{aggregate['route_planner_empty_path_count']} empty-path / "
          f"{aggregate['route_planner_invocation_count']} invocations")
    print(f"SDK Move key: {aggregate['sdk_move_key_present_count']} present / "
          f"{aggregate['sdk_move_key_absent_count']} absent")
    print(f"Games: {aggregate['games_completed']}/{aggregate['games_attempted']} completed")
    print(f"Battle entered: {aggregate['battle_entered_count']}")
    print(f"Top failure reasons: {aggregate['top_failure_reasons'][:3]}")
    print(f"Output files:")
    print(f"  {out_dir / 'results.json'}")
    print(f"  {out_dir / 'per_game.jsonl'}")
    print(f"  {out_dir / 'per_scenario.csv'}")
    print(f"  {out_dir / 'summary.md'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
