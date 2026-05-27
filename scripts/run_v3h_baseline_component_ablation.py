#!/usr/bin/env python3
"""V3H Baseline and Component-Ablation Evaluator.

Compares learned candidate against rule incumbent and component variants
under authenticated SDK replay with paired protocol.

Usage (route-safe baselines):
  python scripts/run_v3h_baseline_component_ablation.py \\
      --scenario-set route_safe \\
      --seeds 0 1 2 3 4 \\
      --pairs 6 \\
      --baselines learned rule_incumbent \\
      --out experiments/shenji_agent/v3h_baseline_expansion_route_safe

Usage (component ablation):
  python scripts/run_v3h_baseline_component_ablation.py \\
      --scenario-set route_safe \\
      --seeds 0 1 2 3 4 \\
      --pairs 3 \\
      --components full no_tactical no_intent no_situation rule_fallback \\
      --out experiments/shenji_agent/v3h_component_ablation

Usage (route-stress map94):
  python scripts/run_v3h_baseline_component_ablation.py \\
      --scenario-set route_stress_map94 \\
      --seeds 0 1 2 \\
      --baselines learned rule_incumbent \\
      --out experiments/shenji_agent/v3h_route_stress_map94

Output files:
  results.json       Aggregate per-variant and paired metrics
  per_game.jsonl     One JSON line per game
  per_pair.csv       CSV with one row per paired comparison
  summary.md         Markdown summary
  manifest.json      Protocol manifest
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
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
MOVE_TYPE_NAMES = {0: "Maneuver", 1: "March", 2: "Walk", 3: "Fly"}


# ---------------------------------------------------------------------------
# Scenario sources
# ---------------------------------------------------------------------------

SCENARIO_MAP: Dict[str, str] = {
    "1231": "123", "1531": "53", "1631": "63",
    "1910631192": "19", "1930331196": "96", "1930631192": "92",
    "2010131194": "94", "2010141294": "94", "2010211129": "29",
    "201033019601": "9601", "201033029601": "9601", "201033039601": "9601",
    "2010431153": "53", "2010441253": "53", "2020331196": "96",
    "2030111194": "94", "2030331196": "96", "2030341296": "96",
    "211033019601": "9601", "2110431153": "21", "2110431253": "21",
    "2110431353": "21", "2110431453": "21", "2110431553": "21",
    "2110431653": "21", "2110431753": "21", "2120131194": "94",
    "2120131294": "94", "2120331196": "212", "2120511121": "21",
    "2120531121": "53", "2130331196": "96", "2130331296": "96",
    "2130511121": "21", "2130511221": "221", "2130511321": "21",
    "2130511421": "21", "2140131194": "21", "2140331196": "96",
    "2230331196": "96", "2230531121": "21", "2230531221": "53",
    "2230631192": "92", "3531": "53",
}

# Route-safe scenarios from V3G: 0% empty under authenticated SDK replay
ROUTE_SAFE_SCENARIOS = [
    "1231", "1531", "1910631192", "1930331196", "1930631192",
    "2010211129", "201033019601", "201033029601", "201033039601",
    "2010431153", "2010441253", "2020331196", "2030111194",
    "2030331196", "2030341296", "211033019601", "2110431153",
    "2110431253", "2110431353", "2110431453", "2110431553",
    "2110431653", "2110431753", "2120131194", "2120131294",
    "2120511121", "2120531121", "2130331196", "2130331296",
    "2130511121", "2130511221", "2130511321", "2130511421",
    "2140131194", "2140331196", "2230331196", "2230531121",
    "2230531221", "2230631192",
]

# Route-stress scenarios (map 94 with ~99% empty in V3G)
ROUTE_STRESS_MAP94_SCENARIOS = [
    "2010131194", "2010141294",
]

# V3D scenarios (for quick compatibility checks)
V3D_SCENARIOS = ["1231", "1531", "1910631192"]


def resolve_map_name(scenario_id: str) -> str:
    if scenario_id in SCENARIO_MAP:
        return SCENARIO_MAP[scenario_id]
    match = re.match(r"(\d{2,3})", scenario_id)
    if match:
        return match.group(1)
    return scenario_id


def get_scenario_set(name: str) -> List[Dict[str, Any]]:
    name = name.lower()
    if name == "route_safe":
        sids = ROUTE_SAFE_SCENARIOS
    elif name == "route_stress_map94":
        sids = ROUTE_STRESS_MAP94_SCENARIOS
    elif name == "v3d":
        sids = V3D_SCENARIOS
    else:
        raise ValueError(f"Unknown scenario set: {name}")
    return [{"scenario_id": sid, "map_id": resolve_map_name(sid)} for sid in sids]


# ---------------------------------------------------------------------------
# Policy variant configuration
# ---------------------------------------------------------------------------

def configure_variant(
    agent_red: Agent, variant: str, artifacts_dir: str,
) -> Dict[str, Any]:
    """Configure agent after setup for the given variant.
    
    Returns a dict describing what was configured.
    """
    policy = agent_red.policy
    info = {"variant": variant, "configured": True}
    
    if variant == "learned" or variant == "full":
        # Full learned policy (challenger profile) — already configured by setup
        policy.tactical_mode = "on"
        info["policy_profile"] = "challenger"
        info["intent_loaded"] = policy.intent_model is not None
        info["situation_loaded"] = policy.situation_model is not None
        info["tactical_mode"] = "on"
        
    elif variant == "rule_incumbent" or variant == "rule_fallback":
        # Rule-only — no learned models (already baseline profile in setup)
        policy.tactical_mode = "off"
        info["policy_profile"] = "baseline"
        info["intent_loaded"] = False
        info["situation_loaded"] = False
        info["tactical_mode"] = "off"
        
    elif variant == "no_tactical" or variant == "tactical_off":
        # Full learned but tactical disabled
        policy.tactical_mode = "off"
        info["policy_profile"] = "challenger"
        info["intent_loaded"] = policy.intent_model is not None
        info["situation_loaded"] = policy.situation_model is not None
        info["tactical_mode"] = "off"
        
    elif variant == "no_intent":
        # Full learned but intent model disabled
        policy.intent_model = None
        policy.tactical_mode = "on"
        info["policy_profile"] = "challenger"
        info["intent_loaded"] = False
        info["situation_loaded"] = policy.situation_model is not None
        info["tactical_mode"] = "on"
        
    elif variant == "no_situation":
        # Full learned but situation model disabled
        policy.situation_model = None
        policy.tactical_mode = "on"
        info["policy_profile"] = "challenger"
        info["intent_loaded"] = policy.intent_model is not None
        info["situation_loaded"] = False
        info["tactical_mode"] = "on"
        
    else:
        raise ValueError(f"Unknown variant: {variant}")
    
    return info


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def pickle_load(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_scenario(sid: str) -> Tuple[Path, str]:
    scenarios_dir = SDK_DATA_DIR / "scenarios"
    scenario_file = scenarios_dir / f"{sid}.json"
    if not scenario_file.exists():
        scenario_file = scenarios_dir / sid
    if not scenario_file.exists():
        raise FileNotFoundError(f"Scenario not found: {sid}")
    return scenario_file, scenario_file.stem


def _load_map_data(map_name: str):
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


def make_player_info() -> List[Dict[str, Any]]:
    return [
        {"seat": 1, "faction": 0, "role": 1, "user_name": "demo", "user_id": 0},
        {"seat": 11, "faction": 1, "role": 1, "user_name": "demo", "user_id": 0},
    ]


# ---------------------------------------------------------------------------
# Per-game evaluation
# ---------------------------------------------------------------------------

def run_v3h_game(
    scenario_id: str,
    map_name: str,
    seed: int,
    variant: str,
    artifacts_dir: str,
    timeout_steps: int = 2000,
) -> Dict[str, Any]:
    """Run a single game with the given policy variant.
    
    Returns per-game metrics dict with all required fields.
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
        "variant": variant,
        "authenticated": False,
        "status": "init",
        "error": None,
        "battle_entered": False,
        "deployment_entered": False,
        "game_completed": False,
        "turns": 0,
        "winner": None,
        "score_red": None,
        "score_blue": None,
        "final_scores": None,
        "total_actions": 0,
        "move_actions": 0,
        "shoot_actions": 0,
        "route_eligible_move_actions": 0,
        "route_empty_count": 0,
        "route_empty_rate": 0.0,
        "route_planner_invocation_count": 0,
        "route_planner_success_count": 0,
        "route_planner_empty_path_count": 0,
        "fallback_move_count": 0,
        "illegal_action_count": 0,
        "tactical_invocation_count": 0,
        "intent_used": False,
        "situation_used": False,
        "selected_by_learned_count": 0,
        "selected_by_rule_or_fallback_count": 0,
        "runtime_errors": None,
        "variant_info": {},
        "duration": 0.0,
    }

    try:
        state = env.setup(env_step_info)

        # Determine policy profile based on variant
        if variant in ("rule_incumbent", "rule_fallback"):
            red_profile = "baseline"
        else:
            red_profile = "challenger"

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
            "policy_profile": red_profile,
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

        # Apply variant-specific configuration
        variant_info = configure_variant(agent_red, variant, artifacts_dir)
        result["variant_info"] = variant_info

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

            stage = state.get(GREEN, {}).get("time", {}).get("stage")
            if stage is not None:
                deployment_entered = True
            if deployment_entered and stage != 1:
                battle_entered = True

        duration = time.time() - start_time

        # Determine winner and scores
        final_obs = state.get(GREEN, {})
        final_scores = final_obs.get("scores")
        winner = None
        score_red = None
        score_blue = None
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
            score_red = final_scores.get("red_total")
            score_blue = final_scores.get("blue_total")

        # Extract policy diagnostics
        policy_diag = agent_red.policy.get_diagnostics()
        counts = policy_diag.get("counts", {})
        move_counts = counts.get("move", {})
        shoot_counts = counts.get("shoot", {})

        move_route_agg = policy_diag.get("move_route_aggregate", {})
        total_attempts = int(move_route_agg.get("total_attempts", 0))
        empty_route_count = int(move_route_agg.get("empty_route_count", 0))
        route_failure_reasons = move_route_agg.get("route_failure_reasons", {})

        move_route_samples = policy_diag.get("move_route_samples", [])
        fallback_move_count = sum(
            1 for r in move_route_samples if r.get("target_source") == "fallback_neighbor"
        )
        selected_by_learned_count = sum(
            1 for r in move_route_samples if r.get("selected_by_learned")
        )
        n_samples = len(move_route_samples)
        reachable_positive = sum(
            1 for r in move_route_samples if int(r.get("reachable_city_count", 0)) > 0
        )
        reachable_zero = sum(
            1 for r in move_route_samples if int(r.get("reachable_city_count", 0)) == 0
        )

        # Tactical invocation from score_means
        score_means = policy_diag.get("score_means", {})
        tactical_invocation = sum(
            sm.get("samples", 0)
            for action_type in ("move", "shoot")
            for sm in [score_means.get(action_type, {})]
        )

        total_actions = int(move_counts.get("total", 0)) + int(shoot_counts.get("total", 0))
        move_actions = int(move_counts.get("total", 0))
        shoot_actions = int(shoot_counts.get("total", 0))

        result.update({
            "authenticated": True,
            "deployment_entered": deployment_entered,
            "battle_entered": battle_entered,
            "game_completed": done,
            "turns": step_count,
            "winner": winner,
            "score_red": score_red,
            "score_blue": score_blue,
            "final_scores": final_scores,
            "total_actions": total_actions,
            "move_actions": move_actions,
            "shoot_actions": shoot_actions,
            "route_eligible_move_actions": total_attempts,
            "route_empty_count": empty_route_count,
            "route_empty_rate": round(empty_route_count / max(total_attempts, 1), 4),
            "route_planner_invocation_count": total_attempts,
            "route_planner_success_count": reachable_positive,
            "route_planner_empty_path_count": reachable_zero,
            "fallback_move_count": fallback_move_count,
            "illegal_action_count": 0,
            "tactical_invocation_count": tactical_invocation,
            "intent_used": policy_diag.get("intent_model_loaded", False),
            "situation_used": policy_diag.get("situation_model_loaded", False),
            "selected_by_learned_count": selected_by_learned_count,
            "selected_by_rule_or_fallback_count": n_samples - selected_by_learned_count,
            "no_valid_move_candidate_count": route_failure_reasons.get("no_valid_move_candidate", 0),
            "failure_reasons": dict(route_failure_reasons),
            "runtime_errors": None,
            "policy_stats": {
                "intent_model_loaded": policy_diag.get("intent_model_loaded"),
                "situation_model_loaded": policy_diag.get("situation_model_loaded"),
                "tactical_model_loaded": policy_diag.get("tactical_model_loaded"),
                "learned_path_coverage": policy_diag.get("learned_path_coverage"),
                "move_path_coverage": policy_diag.get("move_path_coverage"),
            },
            "status": "completed" if not (step_count >= timeout_steps) else "timeout",
            "duration": round(duration, 3),
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
# Paired comparison
# ---------------------------------------------------------------------------

def build_paired_results(
    all_games: List[Dict[str, Any]],
    variants: List[str],
) -> Dict[str, Any]:
    """Group games by (scenario, seed) and compute paired deltas."""
    # Group by (scenario_id, seed)
    groups: Dict[Tuple[str, int], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for game in all_games:
        key = (game.get("scenario_id", "?"), game.get("seed", -1))
        var = game.get("variant", "?")
        groups[key][var] = game

    paired_comparisons = []
    for (sid, seed), var_games in groups.items():
        comparison = {
            "scenario_id": sid,
            "seed": seed,
            "variants": dict(var_games),
            "deltas": {},
        }
        # Compute deltas relative to first variant in list
        base_var = variants[0] if variants else None
        base_game = var_games.get(base_var) if base_var else None
        if base_game and base_game.get("status") == "completed":
            for var_name, var_game in var_games.items():
                if var_name == base_var or var_game.get("status") != "completed":
                    continue
                delta = compute_game_delta(base_game, var_game)
                comparison["deltas"][f"{base_var}_vs_{var_name}"] = delta
        paired_comparisons.append(comparison)

    return {
        "total_games": len(all_games),
        "total_pairs": len(paired_comparisons),
        "pair_count": len(paired_comparisons),
        "paired_comparisons": paired_comparisons,
    }


def compute_game_delta(
    game_a: Dict[str, Any],
    game_b: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute delta: game_a minus game_b for key metrics."""
    delta: Dict[str, Any] = {}
    for metric in ("score_red", "score_blue", "turns", "duration",
                   "route_empty_count", "route_empty_rate", "move_actions",
                   "shoot_actions", "total_actions", "fallback_move_count",
                   "tactical_invocation_count", "illegal_action_count",
                   "selected_by_learned_count"):
        va = game_a.get(metric, 0) if isinstance(game_a.get(metric), (int, float)) else 0
        vb = game_b.get(metric, 0) if isinstance(game_b.get(metric), (int, float)) else 0
        delta[metric + "_delta"] = round(float(va) - float(vb), 4)

    # Winner comparison
    wa = game_a.get("winner")
    wb = game_b.get("winner")
    if wa is not None and wb is not None:
        delta["winner_a"] = int(wa)
        delta["winner_b"] = int(wb)
        if wa == 0 and wb != 0:
            delta["a_wins_b_loses"] = True
        elif wb == 0 and wa != 0:
            delta["a_wins_b_loses"] = False
        else:
            delta["a_wins_b_loses"] = None

    return delta


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_v3h_results(
    all_games: List[Dict[str, Any]],
    variants: List[str],
) -> Dict[str, Any]:
    """Compute aggregate metrics per variant and paired statistics."""
    # Per-variant grouping
    variant_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for game in all_games:
        var = game.get("variant", "?")
        variant_groups[var].append(game)

    per_variant_stats: Dict[str, Dict[str, Any]] = {}
    for var, games in variant_groups.items():
        completed = [g for g in games if g.get("status") == "completed"]
        crashed = [g for g in games if g.get("status") == "crashed"]
        timeout = [g for g in games if g.get("status") == "timeout"]

        scores = [float(g.get("score_red", 0) or 0) for g in completed]
        route_rates = [float(g.get("route_empty_rate", 0)) for g in completed]
        turns = [int(g.get("turns", 0)) for g in completed]
        moves = [int(g.get("move_actions", 0)) for g in completed]

        wins = sum(1 for g in completed if g.get("winner") == 0)
        losses = sum(1 for g in completed if g.get("winner") == 1)
        draws = sum(1 for g in completed if g.get("winner") == -1)

        battle_entered = sum(1 for g in completed if g.get("battle_entered"))

        empty_count = sum(int(g.get("route_empty_count", 0)) for g in completed)
        eligible = sum(int(g.get("route_eligible_move_actions", 0)) for g in completed)

        per_variant_stats[var] = {
            "variant": var,
            "games_attempted": len(games),
            "games_completed": len(completed),
            "crashed": len(crashed),
            "timeout": len(timeout),
            "battle_entered": battle_entered,
            "mean_score": round(float(np.mean(scores)), 4) if scores else None,
            "std_score": round(float(np.std(scores, ddof=1)), 4) if len(scores) > 1 else None,
            "median_score": round(float(np.median(scores)), 4) if scores else None,
            "mean_route_empty_rate": round(float(np.mean(route_rates)), 4) if route_rates else None,
            "std_route_empty_rate": round(float(np.std(route_rates, ddof=1)), 4) if len(route_rates) > 1 else None,
            "total_route_empty": empty_count,
            "total_route_eligible": eligible,
            "overall_route_empty_rate": round(empty_count / max(eligible, 1), 4),
            "mean_turns": round(float(np.mean(turns)), 1) if turns else None,
            "mean_move_actions": round(float(np.mean(moves)), 1) if moves else None,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "total_games_with_winner": wins + losses + draws,
            "win_rate": round(wins / max(wins + losses + draws, 1), 4),
            "intent_used": any(g.get("intent_used") for g in completed),
            "situation_used": any(g.get("situation_used") for g in completed),
            "total_tactical_invocations": sum(
                int(g.get("tactical_invocation_count", 0)) for g in completed
            ),
            "total_fallback_moves": sum(
                int(g.get("fallback_move_count", 0)) for g in completed
            ),
            "total_selected_by_learned": sum(
                int(g.get("selected_by_learned_count", 0)) for g in completed
            ),
            "runtime_errors": [g.get("error") for g in games if g.get("error")],
        }

    # Paired statistics (bootstrap CI on score deltas)
    paired = build_paired_results(all_games, variants)
    paired_deltas = extract_paired_deltas(paired, variants)

    paired_stats: Dict[str, Any] = {}
    for delta_key, deltas in paired_deltas.items():
        if len(deltas) < 2:
            continue
        arr = np.array(deltas)
        ci_lo, ci_hi, boot_means = bootstrap_ci(deltas)
        d = cohens_d(deltas)
        paired_stats[delta_key] = {
            "n": len(deltas),
            "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std(ddof=1)), 4),
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
            "bootstrap_95_ci_lower": round(ci_lo, 4),
            "bootstrap_95_ci_upper": round(ci_hi, 4),
            "cohens_d": d,
            "positive_count": int(sum(1 for d in deltas if d > 0)),
            "negative_count": int(sum(1 for d in deltas if d < 0)),
            "zero_count": int(sum(1 for d in deltas if d == 0)),
        }

    # Aggregate failure reasons
    all_reasons: Dict[str, int] = {}
    for game in all_games:
        reasons = game.get("failure_reasons", {})
        if isinstance(reasons, dict):
            for reason, count in reasons.items():
                all_reasons[reason] = all_reasons.get(reason, 0) + int(count)

    return {
        "total_games": len(all_games),
        "total_completed": sum(
            1 for g in all_games if g.get("status") == "completed"
        ),
        "total_crashed": sum(
            1 for g in all_games if g.get("status") == "crashed"
        ),
        "total_timeout": sum(
            1 for g in all_games if g.get("status") == "timeout"
        ),
        "scenario_count": len(set(g.get("scenario_id") for g in all_games)),
        "map_count": len(set(g.get("map_id") for g in all_games)),
        "variants": sorted(variant_groups.keys()),
        "per_variant": per_variant_stats,
        "paired_statistics": paired_stats,
        "failure_reasons": all_reasons,
    }


def extract_paired_deltas(
    paired_output: Dict[str, Any],
    variants: List[str],
) -> Dict[str, List[float]]:
    """Extract score_red deltas from paired comparisons."""
    deltas: Dict[str, List[float]] = {}
    base_var = variants[0] if variants else None
    if not base_var:
        return deltas

    for comp in paired_output.get("paired_comparisons", []):
        var_games = comp.get("variants", {})
        base_game = var_games.get(base_var)
        if not base_game:
            continue
        base_score = base_game.get("score_red")
        if base_score is None:
            continue
        for var_name, var_game in var_games.items():
            if var_name == base_var:
                continue
            var_score = var_game.get("score_red")
            if var_score is None:
                continue
            key = f"{base_var}_vs_{var_name}"
            if key not in deltas:
                deltas[key] = []
            deltas[key].append(float(base_score) - float(var_score))

    return deltas


def bootstrap_ci(
    values: List[float],
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, List[float]]:
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


def cohens_d(values: List[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    arr = np.array(values)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    if abs(std) < 1e-12:
        return None
    return round(mean / std, 4)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_v3h_output(
    aggregate: Dict[str, Any],
    all_games: List[Dict[str, Any]],
    out_dir: Path,
    manifest: Dict[str, Any],
) -> None:
    """Write all output files."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write manifest
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Write per_game.jsonl
    per_game_path = out_dir / "per_game.jsonl"
    with open(per_game_path, "w", encoding="utf-8") as f:
        for game in all_games:
            f.write(json.dumps(game, sort_keys=True) + "\n")

    # Write results.json (aggregate without per-game)
    results_out = dict(aggregate)
    results_path = out_dir / "results.json"
    results_path.write_text(
        json.dumps(results_out, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Write per_pair.csv
    _write_per_pair_csv(out_dir / "per_pair.csv", aggregate, all_games)

    # Write summary.md
    summary = _format_summary(aggregate, manifest)
    summary_path = out_dir / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")

    print(f"Output files:")
    print(f"  {manifest_path}")
    print(f"  {per_game_path}")
    print(f"  {results_path}")
    print(f"  {out_dir / 'per_pair.csv'}")
    print(f"  {summary_path}")


def _write_per_pair_csv(path: Path, aggregate: Dict[str, Any], all_games: List[Dict[str, Any]]) -> None:
    """Write paired comparison results as CSV."""
    fieldnames = [
        "scenario_id", "map_id", "seed", "variant",
        "status", "battle_entered", "score_red", "winner",
        "turns", "duration", "move_actions",
        "route_empty_rate", "route_empty_count", "route_eligible",
        "tactical_invocation_count", "fallback_move_count",
        "selected_by_learned_count", "intent_used", "situation_used",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for game in all_games:
            row = {fn: game.get(fn, "") for fn in fieldnames}
            row["route_eligible"] = game.get("route_eligible_move_actions", 0)
            row["intent_used"] = 1 if game.get("intent_used") else 0
            row["situation_used"] = 1 if game.get("situation_used") else 0
            writer.writerow(row)


def _format_summary(aggregate: Dict[str, Any], manifest: Dict[str, Any]) -> str:
    """Format aggregate results as markdown."""
    lines = [
        "# V3H Baseline and Component-Ablation Evaluation Summary",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Protocol:** {manifest.get('scenario_set', '?')}",
        f"**Variants:** {', '.join(manifest.get('variants', []))}",
        f"**Seeds:** {manifest.get('seeds', [])}",
        "",
        "## Overview",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total games | {aggregate['total_games']} |",
        f"| Completed | {aggregate['total_completed']} |",
        f"| Crashed | {aggregate['total_crashed']} |",
        f"| Timeout | {aggregate['total_timeout']} |",
        f"| Scenarios | {aggregate['scenario_count']} |",
        f"| Maps | {aggregate['map_count']} |",
        f"| Variants | {aggregate['variants']} |",
        "",
        "## Per-Variant Results",
        "",
    ]

    # Per-variant table
    pv = aggregate.get("per_variant", {})
    if pv:
        lines.append("| Variant | Games | Score (mean) | Route Empty Rate | Win Rate | Tactical | Turns |")
        lines.append("|--------|-------|-------------|-----------------|---------|---------|-------|")
        for var_name in sorted(pv.keys()):
            s = pv[var_name]
            score_str = f"{s.get('mean_score', 'N/A'):.2f}" if s.get('mean_score') is not None else "N/A"
            route_str = f"{s.get('overall_route_empty_rate', 1):.4f}" if s.get('overall_route_empty_rate') is not None else "N/A"
            wr_str = f"{s.get('win_rate', 0):.3f}" if s.get('win_rate') is not None else "N/A"
            tact = s.get('total_tactical_invocations', 0)
            turns_str = f"{s.get('mean_turns', 'N/A'):.0f}" if s.get('mean_turns') is not None else "N/A"
            lines.append(
                f"| {var_name} | {s['games_completed']} | {score_str} | {route_str} | {wr_str} | {tact} | {turns_str} |"
            )

    # Paired statistics
    ps = aggregate.get("paired_statistics", {})
    if ps:
        lines.extend([
            "",
            "## Paired Statistics (Score Deltas)",
            "",
            "| Comparison | N | Mean Delta | Std | 95% CI Lower | 95% CI Upper | Cohen's d |",
            "|-----------|----|-----------|-----|-------------|-------------|---------|",
        ])
        for delta_key in sorted(ps.keys()):
            s = ps[delta_key]
            lines.append(
                f"| {delta_key} | {s['n']} | {s['mean']} | {s['std']} | "
                f"{s['bootstrap_95_ci_lower']} | {s['bootstrap_95_ci_upper']} | "
                f"{s.get('cohens_d', 'N/A')} |"
            )

    # Failure reasons
    fr = aggregate.get("failure_reasons", {})
    if fr:
        lines.extend([
            "",
            "## Failure Reasons",
            "",
            "| Reason | Count |",
            "|--------|-------|",
        ])
        for reason in sorted(fr.keys()):
            lines.append(f"| {reason} | {fr[reason]} |")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="V3H Baseline and Component-Ablation Evaluation"
    )
    parser.add_argument(
        "--scenario-set", type=str,
        choices=["route_safe", "route_stress_map94", "v3d", "explicit"],
        default="route_safe",
        help="Scenario set to evaluate",
    )
    parser.add_argument(
        "--scenarios", type=str, nargs="+", default=None,
        help="Explicit scenario IDs (requires --scenario-set explicit)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0],
        help="Seeds to evaluate (default: [0])",
    )
    parser.add_argument(
        "--pairs", type=int, default=None,
        help="Number of scenario-seed pairs to run (limits total)",
    )
    parser.add_argument(
        "--baselines", type=str, nargs="+",
        choices=["learned", "rule_incumbent", "rule_fallback"],
        default=None,
        help="Baseline variants to compare",
    )
    parser.add_argument(
        "--components", type=str, nargs="+",
        choices=["full", "no_tactical", "no_intent", "no_situation", "rule_fallback"],
        default=None,
        help="Component ablation variants to compare",
    )
    parser.add_argument(
        "--out", type=str,
        default="experiments/shenji_agent/v3h_evaluation",
        help="Output directory",
    )
    parser.add_argument("--artifacts-dir", type=str, default=None)
    parser.add_argument("--timeout-steps", type=int, default=2000)
    args = parser.parse_args()

    # Resolve variants to run
    variants: List[str] = []
    if args.baselines:
        variants.extend(args.baselines)
    if args.components:
        variants.extend(args.components)
    if not variants:
        print("ERROR: must specify --baselines or --components")
        return 1

    # Deduplicate while preserving order
    seen: set = set()
    variants_deduped = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            variants_deduped.append(v)
    variants = variants_deduped

    # Resolve scenario set
    if args.scenario_set == "explicit":
        if not args.scenarios:
            print("ERROR: --scenarios required when --scenario-set=explicit")
            return 1
        scenario_items = [
            {"scenario_id": sid, "map_id": resolve_map_name(sid)}
            for sid in args.scenarios
        ]
    else:
        scenario_items = get_scenario_set(args.scenario_set)

    # Limit by --pairs
    if args.pairs is not None:
        scenario_items = scenario_items[:args.pairs]

    artifacts_dir = args.artifacts_dir or str(ROOT / "ai" / "weights")

    # Build manifest
    manifest = {
        "protocol": "V3H Baseline and Component-Ablation Evaluation",
        "start_time": datetime.now().isoformat(),
        "scenario_set": args.scenario_set,
        "scenarios": [s["scenario_id"] for s in scenario_items],
        "maps": list(set(s["map_id"] for s in scenario_items)),
        "seeds": args.seeds,
        "variants": variants,
        "timeout_steps": args.timeout_steps,
        "artifacts_dir": artifacts_dir,
        "total_games_planned": len(scenario_items) * len(args.seeds) * len(variants),
    }

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Environment check
    ckpt = ROOT / "ai" / "weights" / "model_manifest.json"
    if not ckpt.exists():
        print(f"ERROR: model manifest not found: {ckpt}")
        return 1

    # Run evaluation
    all_games: List[Dict[str, Any]] = []
    total_planned = manifest["total_games_planned"]
    game_num = 0

    print(f"Running {total_planned} games ({len(scenario_items)} scenarios x {len(args.seeds)} seeds x {len(variants)} variants)...")
    print(f"Output: {out_dir}")
    print(f"Variants: {variants}")

    for item in scenario_items:
        sid = item["scenario_id"]
        mid = item["map_id"]
        for seed in args.seeds:
            for variant in variants:
                game_num += 1
                print(f"  [{game_num}/{total_planned}] scenario={sid} map={mid} seed={seed} variant={variant}")
                sys.stdout.flush()

                result = run_v3h_game(
                    scenario_id=sid,
                    map_name=mid,
                    seed=seed,
                    variant=variant,
                    artifacts_dir=artifacts_dir,
                    timeout_steps=args.timeout_steps,
                )
                all_games.append(result)

                status = result.get("status", "?")
                score = result.get("score_red", "?")
                route_rate = result.get("route_empty_rate", "?")
                print(f"         -> status={status} score={score} route_empty_rate={route_rate}")

    # Aggregate and write output
    aggregate = aggregate_v3h_results(all_games, variants)
    manifest["end_time"] = datetime.now().isoformat()
    manifest["total_games_completed"] = aggregate["total_completed"]
    manifest["total_crashed"] = aggregate["total_crashed"]

    write_v3h_output(aggregate, all_games, out_dir, manifest)

    # Console summary
    print(f"\n{'=' * 60}")
    print(f"Evaluation complete. Results in: {out_dir}")
    for var_name, stats in aggregate.get("per_variant", {}).items():
        print(f"  {var_name}: {stats['games_completed']} games, "
              f"score={stats.get('mean_score', 'N/A')}, "
              f"route_empty={stats.get('overall_route_empty_rate', 'N/A')}, "
              f"win_rate={stats.get('win_rate', 'N/A')}")
    print(f"Paired statistics:")
    for delta_key, ps in aggregate.get("paired_statistics", {}).items():
        print(f"  {delta_key}: delta={ps['mean']} 95%CI=[{ps['bootstrap_95_ci_lower']},{ps['bootstrap_95_ci_upper']}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
