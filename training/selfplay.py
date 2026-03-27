from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .common import dump_yaml, ensure_dir, load_config, now_tag, repo_root, save_json, set_seed
from .evaluate import promotion_decision, summarize_matches
from .maps import build_map_catalog, infer_map_id_from_scenario_id, map_bucket_lookup

try:
    from ai.agent import Agent
    from land_wargame_train_env import TrainEnv
except ImportError:  # pragma: no cover
    from ai.agent import Agent
    from mock_train_env import MockTrainEnv as TrainEnv


RED, BLUE = 0, 1


def _load_case_data(scenario_path: Path, map_id: int) -> Dict[str, object]:
    data_root = repo_root() / "land_wargame_sdk" / "Data" / "Data"
    with scenario_path.open("r", encoding="utf-8") as fh:
        scenario_data = json.load(fh)
    with (data_root / "maps" / f"map_{map_id}" / "basic.json").open("r", encoding="utf-8") as fh:
        basic_data = json.load(fh)
    with (data_root / "maps" / f"map_{map_id}" / "cost.pickle").open("rb") as fh:
        cost_data = pickle.load(fh)
    see_path = data_root / "maps" / f"map_{map_id}" / f"{map_id}see.npz"
    see_data = np.load(see_path)["data"] if see_path.exists() else None
    return {
        "scenario_data": scenario_data,
        "basic_data": basic_data,
        "cost_data": cost_data,
        "see_data": see_data,
    }


def _run_single(
    scenario_path: Path,
    map_id: int,
    red_profile: Dict[str, str],
    blue_profile: Dict[str, str],
    max_steps: int,
) -> Tuple[float, float]:
    env = TrainEnv()
    shared = _load_case_data(scenario_path, map_id)
    player_info = [
        {"seat": 1, "faction": 0, "role": 1, "user_name": red_profile["user_name"], "user_id": 1},
        {"seat": 11, "faction": 1, "role": 1, "user_name": blue_profile["user_name"], "user_id": 11},
    ]
    env_info = {**shared, "player_info": player_info}
    state = env.setup(env_info)
    red = Agent(log_level=logging.WARNING)
    blue = Agent(log_level=logging.WARNING)
    red.setup({**shared, "seat": 1, "faction": 0, "role": 1, "user_name": red_profile["user_name"], "policy_profile": red_profile["profile"], "policy_artifacts_dir": red_profile["artifacts"], "state": state})
    blue.setup({**shared, "seat": 11, "faction": 1, "role": 1, "user_name": blue_profile["user_name"], "policy_profile": blue_profile["profile"], "policy_artifacts_dir": blue_profile["artifacts"], "state": state})
    done = False
    steps = 0
    while not done and steps < max_steps:
        actions = red.step(state[RED]) + blue.step(state[BLUE])
        state, done = env.step(actions)
        steps += 1
    red_score = float(state[RED].get("reward", 0.0))
    blue_score = float(state[BLUE].get("reward", 0.0))
    return red_score, blue_score


def _paired_score(candidate_a: float, opponent_a: float, candidate_b: float, opponent_b: float) -> float:
    def outcome(left: float, right: float) -> float:
        if left > right:
            return 1.0
        if left < right:
            return 0.0
        return 0.5

    return (outcome(candidate_a, opponent_a) + outcome(candidate_b, opponent_b)) / 2.0


def _bootstrap_league(league_path: Path) -> Dict[str, object]:
    if league_path.exists():
        return json.loads(league_path.read_text(encoding="utf-8"))
    state = {
        "champion": "ai/weights",
        "historical_mains": [],
        "exploiters": ["aggressive_exploiter", "occupy_exploiter"],
        "specialists": [],
        "consecutive_windows": 0,
    }
    save_json(state, league_path)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Shenji self-play league evaluation.")
    parser.add_argument("--config", default="configs/selfplay_baseline.yaml")
    parser.add_argument("--candidate", default="ai/weights")
    parser.add_argument("--incumbent", default="ai/weights")
    args = parser.parse_args()

    config = load_config(repo_root() / args.config)
    set_seed(int(config["seed"]))
    run_dir = ensure_dir(repo_root() / config["output_root"] / f"league_{now_tag()}")
    league_dir = ensure_dir(repo_root() / config["league_dir"])
    league_state_path = league_dir / "league_state.json"
    league_state = _bootstrap_league(league_state_path)

    maps_root = repo_root() / "land_wargame_sdk" / "Data" / "Data" / "maps"
    scenario_root = repo_root() / "land_wargame_sdk" / "Data" / "Data" / "scenarios"
    records = build_map_catalog(maps_root, repo_root() / "training" / "cache" / "map_catalog.json")
    bucket_lookup = map_bucket_lookup(records)
    scenarios_by_bucket: Dict[str, List[Tuple[Path, int]]] = {"simple": [], "medium": [], "complex": []}
    for scenario_path in sorted(scenario_root.glob("*.json")):
        scenario_id = scenario_path.stem
        map_id = infer_map_id_from_scenario_id(scenario_id, bucket_lookup.keys())
        if map_id is None:
            continue
        scenarios_by_bucket[bucket_lookup[map_id]].append((scenario_path, map_id))

    matches: List[Dict[str, object]] = []
    total_pairs = int(config["paired_matches"])
    ratios = config["map_sampling"]
    for bucket, ratio in ratios.items():
        bucket_pairs = max(1, int(round(total_pairs * float(ratio))))
        candidates = scenarios_by_bucket.get(bucket, [])
        if not candidates:
            continue
        for idx in range(bucket_pairs):
            scenario_path, map_id = candidates[idx % len(candidates)]
            challenger = {"artifacts": args.candidate, "profile": "challenger", "user_name": "CurrentAI"}
            incumbent = {"artifacts": args.incumbent, "profile": "incumbent", "user_name": "IncumbentAI"}
            red_score, blue_score = _run_single(scenario_path, map_id, challenger, incumbent, int(config["max_steps"]))
            blue_red_score, blue_blue_score = _run_single(scenario_path, map_id, incumbent, challenger, int(config["max_steps"]))
            matches.append(
                {
                    "scenario": scenario_path.name,
                    "bucket": bucket,
                    "map_id": map_id,
                    "paired_score": _paired_score(red_score, blue_score, blue_blue_score, blue_red_score),
                    "first_leg": {"candidate": red_score, "opponent": blue_score},
                    "second_leg": {"candidate": blue_blue_score, "opponent": blue_red_score},
                }
            )

    summary = summarize_matches(matches)
    decision = promotion_decision(summary, config["promotion_gate"])

    exploiter_results = {}
    for profile in league_state["exploiters"]:
        profile_scores = []
        for scenario_path, map_id in scenarios_by_bucket.get("complex", [])[:4]:
            challenger = {"artifacts": args.candidate, "profile": "challenger", "user_name": "CurrentAI"}
            exploiter = {"artifacts": args.incumbent, "profile": profile, "user_name": "ExploiterAI"}
            score_a, score_b = _run_single(scenario_path, map_id, challenger, exploiter, int(config["max_steps"]))
            profile_scores.append(1.0 if score_b > score_a else 0.0)
        exploiter_results[profile] = {
            "win_rate_vs_candidate": float(np.mean(profile_scores)) if profile_scores else 0.0,
            "matches": len(profile_scores),
        }

    if decision["accepted"]:
        league_state["consecutive_windows"] = int(league_state.get("consecutive_windows", 0)) + 1
    else:
        league_state["consecutive_windows"] = 0
    if league_state["consecutive_windows"] >= int(config["promotion_gate"]["consecutive_windows"]):
        league_state["historical_mains"] = [league_state.get("champion", args.incumbent)] + list(league_state.get("historical_mains", []))[:7]
        league_state["champion"] = args.candidate
        league_state["consecutive_windows"] = 0

    save_json({"matches": matches}, run_dir / "match_results.json")
    save_json(summary, run_dir / "rating.json")
    save_json(decision, run_dir / "promotion.json")
    save_json(exploiter_results, run_dir / "exploitability.json")
    dump_yaml(config, run_dir / "config.yaml")
    save_json(league_state, league_state_path)
    save_json(league_state, run_dir / "league_state.json")
    (run_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Shenji Self-Play Summary",
                "",
                f"- overall_point_estimate: `{summary['overall']['point_estimate']:.4f}`",
                f"- overall_ci95_lower: `{summary['overall']['ci95'][0]:.4f}`",
                f"- complex_point_estimate: `{summary['buckets'].get('complex', {}).get('point_estimate', 0.0):.4f}`",
                f"- promotion_accepted: `{decision['accepted']}`",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
