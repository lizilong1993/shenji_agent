"""Paired replay runner for V3B formal tactical ablation.

Each pair runs the SAME seed/scenario/map twice:
  - tactical_mode="on"
  - tactical_mode="off"

Only tactical mode differs between the two runs.
"""

from __future__ import annotations

import json
import pickle
import time
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ai import Agent
from train_env import TrainEnv

RED, BLUE, GREEN = 0, 1, -1

SDK_DATA_DIR = Path(__file__).resolve().parents[1] / "land_wargame_sdk" / "Data" / "Data"


def _load_scenario(sid: str) -> Tuple[str, str]:
    """Resolve scenario ID to scenario JSON path and map name."""
    scenarios_dir = SDK_DATA_DIR / "scenarios"
    scenario_file = scenarios_dir / f"{sid}.json"
    if not scenario_file.exists():
        # Try as full filename
        scenario_file = scenarios_dir / sid
    if not scenario_file.exists():
        raise FileNotFoundError(f"Scenario not found: {sid}")
    return str(scenario_file), scenario_file.stem


def _load_map_data(map_name: str):
    """Load map data files for a given map name."""
    maps_dir = SDK_DATA_DIR / "maps" / f"map_{map_name}"
    if not maps_dir.exists():
        maps_dir = SDK_DATA_DIR / "maps" / map_name
    basic_path = maps_dir / "basic.json"
    cost_path = maps_dir / "cost.pickle"
    see_path = maps_dir / f"{map_name}see.npz"
    if not see_path.exists():
        see_path = maps_dir / f"{map_name.split('_')[-1]}see.npz"
    if not basic_path.exists():
        raise FileNotFoundError(f"Map basic.json not found: {basic_path}")
    basic_data = json.loads(basic_path.read_text(encoding="utf-8"))
    cost_data = pickle.loads(cost_path.read_bytes()) if cost_path.exists() else None
    see_data = np.load(str(see_path))["data"] if see_path.exists() else None
    return basic_data, cost_data, see_data


def _make_player_info(map_name: str) -> List[Dict[str, Any]]:
    return [
        {"seat": 1, "faction": 0, "role": 1, "user_name": "demo", "user_id": 0},
        {"seat": 11, "faction": 1, "role": 1, "user_name": "demo", "user_id": 0},
    ]


def run_single_game(
    scenario_id: str,
    map_name: str,
    seed: int,
    tactical_mode: str,
    artifacts_dir: Optional[str] = None,
    policy_profile: str = "challenger",
    timeout_steps: int = 2000,
) -> Dict[str, Any]:
    """Run a single game with given tactical mode and record all metrics.

    Returns a dict with game-level metrics.
    """
    scenario_path, _ = _load_scenario(scenario_id)
    basic_data, cost_data, see_data = _load_map_data(map_name)
    player_info = _make_player_info(map_name)

    scenario_data = json.loads(Path(scenario_path).read_text(encoding="utf-8"))

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
        "scenario": scenario_id,
        "map": map_name,
        "tactical_mode": tactical_mode,
        "policy_profile": policy_profile,
        "status": "init",
        "error": None,
        "steps": 0,
        "winner": None,
        "score": None,
        "reward": None,
        "route_metrics": {"route_empty_count": 0, "total_move_attempts": 0},
        "tactical_stats": {"tactical_invocation_count": 0, "tactical_nonzero_count": 0},
        "action_divergence": None,
        "duration": 0.0,
        "policy_stats": {},
    }

    try:
        state = env.setup(env_step_info)
        all_states = [state[GREEN]]

        agent_setup_info = {
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
        blue_setup_info = {
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

        agent_red.setup(agent_setup_info)
        agent_blue.setup(blue_setup_info)

        # Set tactical mode on the red agent's policy
        agent_red.policy.tactical_mode = tactical_mode

        done = False
        step_count = 0
        final_scores = None

        while not done and step_count < timeout_steps:
            actions = []
            actions += agent_red.step(state[RED])
            actions += agent_blue.step(state[BLUE])
            state, done = env.step(actions)
            all_states.append(state[GREEN])
            step_count += 1

        # Game ended — collect diagnostics ONCE at end
        duration = time.time() - start_time

        # Scores from final observation
        final_obs = state.get(GREEN, {})
        final_scores = final_obs.get("scores")
        # Determine winner: compare red_win vs blue_win
        winner = None
        score_red = None
        score_blue = None
        if final_scores:
            red_win = final_scores.get("red_win")
            blue_win = final_scores.get("blue_win")
            if red_win is not None and blue_win is not None:
                if float(red_win) > float(blue_win):
                    winner = 0  # red
                elif float(blue_win) > float(red_win):
                    winner = 1  # blue
                else:
                    winner = -1  # draw
            score_red = final_scores.get("red_total")
            score_blue = final_scores.get("blue_total")

        # Policy diagnostics (read once at end)
        policy_diag = agent_red.policy.get_diagnostics()
        move_routes = policy_diag.get("move_route_samples", [])
        route_empty_count = sum(1 for r in move_routes if r.get("route_empty"))
        total_move_attempts = len(move_routes)

        score_means = policy_diag.get("score_means", {})
        t_invocation = sum(s.get("samples", 0) for s in score_means.values())
        t_nonzero = 0
        for action_type in ("move", "shoot"):
            sm = score_means.get(action_type, {})
            if sm.get("samples", 0) > 0 and abs(sm.get("tactical_score", 0.0)) > 1e-9:
                t_nonzero += sm["samples"]

        result.update({
            "status": "completed" if not (step_count >= timeout_steps) else "timeout",
            "steps": step_count,
            "winner": winner,
            "score_red": score_red,
            "score_blue": score_blue,
            "final_scores": final_scores,
            "duration": round(duration, 3),
            "route_metrics": {
                "route_empty_count": route_empty_count,
                "total_move_attempts": total_move_attempts,
                "route_empty_rate": round(route_empty_count / max(total_move_attempts, 1), 4),
            },
            "tactical_stats": {
                "tactical_invocation_count": t_invocation,
                "tactical_nonzero_count": t_nonzero,
            },
            "policy_stats": policy_diag,
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


def run_paired_game(
    scenario_id: str,
    map_name: str,
    seed: int,
    artifacts_dir: Optional[str] = None,
    policy_profile: str = "challenger",
    timeout_steps: int = 2000,
) -> Dict[str, Any]:
    """Run a paired game: tactical_on then tactical_off with same seed/scenario/map.

    Returns a dict with both results and the paired delta.
    """
    on_result = run_single_game(
        scenario_id=scenario_id,
        map_name=map_name,
        seed=seed,
        tactical_mode="on",
        artifacts_dir=artifacts_dir,
        policy_profile=policy_profile,
        timeout_steps=timeout_steps,
    )
    off_result = run_single_game(
        scenario_id=scenario_id,
        map_name=map_name,
        seed=seed,
        tactical_mode="off",
        artifacts_dir=artifacts_dir,
        policy_profile=policy_profile,
        timeout_steps=timeout_steps,
    )

    paired = {
        "seed": seed,
        "scenario": scenario_id,
        "map": map_name,
        "tactical_on": on_result,
        "tactical_off": off_result,
        "paired_delta": _compute_paired_delta(on_result, off_result),
    }
    return paired


# Map lookup: scenario_id -> map_name
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


def resolve_map_name(scenario_id: str) -> str:
    """Resolve scenario ID to map name."""
    if scenario_id in SCENARIO_MAP:
        return SCENARIO_MAP[scenario_id]
    # Fallback: try extracting numeric prefix
    import re
    match = re.match(r"(\d{2,3})", scenario_id)
    if match:
        return match.group(1)
    return scenario_id


def _compute_paired_delta(
    on_result: Dict[str, Any],
    off_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute paired deltas: tactical_on minus tactical_off."""
    delta: Dict[str, Any] = {}

    # Score delta (red score)
    on_score = on_result.get("score_red")
    off_score = off_result.get("score_red")
    if on_score is not None and off_score is not None:
        delta["score_delta"] = round(float(on_score) - float(off_score), 4)
    else:
        delta["score_delta"] = None

    # Route empty rate delta
    on_route_empty = on_result.get("route_metrics", {}).get("route_empty_rate", 0.0)
    off_route_empty = off_result.get("route_metrics", {}).get("route_empty_rate", 0.0)
    delta["route_empty_rate_delta"] = round(on_route_empty - off_route_empty, 4)

    # Steps delta
    on_steps = on_result.get("steps", 0)
    off_steps = off_result.get("steps", 0)
    delta["steps_delta"] = on_steps - off_steps

    # Duration delta
    on_dur = on_result.get("duration", 0.0)
    off_dur = off_result.get("duration", 0.0)
    delta["duration_delta"] = round(on_dur - off_dur, 3)

    # Status
    delta["on_status"] = on_result.get("status")
    delta["off_status"] = off_result.get("status")

    return delta
