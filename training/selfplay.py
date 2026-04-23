from __future__ import annotations

import argparse
import importlib
import json
import logging
import pickle
import random
import sys
import types
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from .common import dump_yaml, ensure_dir, git_head_sha, git_show_text, load_config, now_tag, repo_root, save_json, set_seed
from .evaluate import promotion_decision, summarize_matches
from .maps import build_map_catalog, infer_map_id_from_scenario_id, map_bucket_lookup

RED, BLUE = 0, 1
_SNAPSHOT_ROLE_TO_PROFILE = {
    "bootstrap_rule_baseline": "baseline",
    "accepted_learned_mainline": "incumbent",
}
_TRAIN_ENV_CLASS: type[Any] | None = None
_TRAIN_ENV_BACKEND: str | None = None
_AGENT_CLASS_CACHE: Dict[str, type[Any]] = {}


def _resolve_train_env_class(
    import_module: Callable[[str], Any] = importlib.import_module,
) -> Tuple[type[Any], str]:
    for module_name, class_name in (
        ("land_wargame_train_env", "TrainEnv"),
        ("train_env", "TrainEnv"),
    ):
        try:
            module = import_module(module_name)
        except ImportError:
            continue
        env_class = getattr(module, class_name, None)
        if env_class is not None:
            return env_class, f"{module_name}.{class_name}"

    mock_module = import_module("mock_train_env")
    return mock_module.MockTrainEnv, "mock_train_env.MockTrainEnv"


def _train_env_runtime() -> Tuple[type[Any], str]:
    global _TRAIN_ENV_CLASS, _TRAIN_ENV_BACKEND
    if _TRAIN_ENV_CLASS is None or _TRAIN_ENV_BACKEND is None:
        _TRAIN_ENV_CLASS, _TRAIN_ENV_BACKEND = _resolve_train_env_class()
    return _TRAIN_ENV_CLASS, _TRAIN_ENV_BACKEND


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
    red_profile: Dict[str, Any],
    blue_profile: Dict[str, Any],
    max_steps: int,
) -> Tuple[float, float, Dict[str, object], Dict[str, object]]:
    env_class, _ = _train_env_runtime()
    env = env_class()
    shared = _load_case_data(scenario_path, map_id)
    player_info = [
        {"seat": 1, "faction": 0, "role": 1, "user_name": red_profile["user_name"], "user_id": 1},
        {"seat": 11, "faction": 1, "role": 1, "user_name": blue_profile["user_name"], "user_id": 11},
    ]
    env_info = {**shared, "player_info": player_info}
    state = env.setup(env_info)
    red = red_profile["agent_class"](log_level=logging.WARNING)
    blue = blue_profile["agent_class"](log_level=logging.WARNING)
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
    return red_score, blue_score, red.get_policy_stats(), blue.get_policy_stats()


def _paired_score(candidate_a: float, opponent_a: float, candidate_b: float, opponent_b: float) -> float:
    def outcome(left: float, right: float) -> float:
        if left > right:
            return 1.0
        if left < right:
            return 0.0
        return 0.5

    return (outcome(candidate_a, opponent_a) + outcome(candidate_b, opponent_b)) / 2.0


def _collect_scenario_candidates(scenario_root: Path, bucket_lookup: Dict[int, str]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for scenario_path in sorted(scenario_root.glob("*.json")):
        scenario_id = scenario_path.stem
        map_id = infer_map_id_from_scenario_id(scenario_id, bucket_lookup.keys())
        if map_id is None:
            continue
        candidates.append(
            {
                "scenario": scenario_path.name,
                "map_id": int(map_id),
                "bucket": str(bucket_lookup.get(map_id, "medium")),
            }
        )
    return candidates


def _build_paired_map_list(scenario_candidates: List[Dict[str, Any]], paired_matches: int) -> Dict[str, Any]:
    if paired_matches <= 0:
        raise SystemExit("paired_matches must be positive for self-play evaluation.")
    if not scenario_candidates:
        raise SystemExit("No valid scenarios found under land_wargame_sdk/Data/Data/scenarios for self-play evaluation.")

    unique_prefix = random.sample(scenario_candidates, k=min(paired_matches, len(scenario_candidates)))
    ordered_items = list(unique_prefix)
    duplicates_used = paired_matches > len(unique_prefix)
    while len(ordered_items) < paired_matches:
        ordered_items.append(random.choice(scenario_candidates))

    return {
        "sampling_protocol": "unique_first_then_replacement",
        "scenario_source": "land_wargame_sdk/Data/Data/scenarios/*.json",
        "requested_pairs": paired_matches,
        "candidate_pool_size": len(scenario_candidates),
        "unique_item_count_before_replacement": min(paired_matches, len(scenario_candidates)),
        "duplicates_used": duplicates_used,
        "items": [
            {
                "scenario": str(item["scenario"]),
                "map_id": int(item["map_id"]),
                "bucket": str(item["bucket"]),
                "order_index": order_index,
            }
            for order_index, item in enumerate(ordered_items)
        ],
    }


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


def _load_artifact_manifest(artifacts: str) -> Dict[str, Any]:
    manifest_path = Path(artifacts) / "model_manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _load_agent_class_from_commit(commit_sha: str) -> type[Any]:
    cached = _AGENT_CLASS_CACHE.get(commit_sha)
    if cached is not None:
        return cached

    package_name = f"_selfplay_snapshot_{commit_sha[:12]}"
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = []  # type: ignore[attr-defined]
        package.__package__ = package_name
        sys.modules[package_name] = package

    for module_name in ("features", "models", "policy", "agent"):
        qualified_name = f"{package_name}.{module_name}"
        if qualified_name in sys.modules:
            continue
        source = git_show_text(f"{commit_sha}:ai/{module_name}.py", cwd=repo_root())
        module = types.ModuleType(qualified_name)
        module.__file__ = f"{commit_sha}:ai/{module_name}.py"
        module.__package__ = package_name
        sys.modules[qualified_name] = module
        exec(compile(source, module.__file__, "exec"), module.__dict__)

    agent_class = getattr(sys.modules[f"{package_name}.agent"], "Agent")
    _AGENT_CLASS_CACHE[commit_sha] = agent_class
    return agent_class


def _resolve_source_commit_sha(artifacts: str, fallback_commit_sha: str) -> str:
    manifest = _load_artifact_manifest(artifacts)
    commit_sha = str(manifest.get("source_commit_sha", "")).strip()
    return commit_sha or str(fallback_commit_sha)


def _build_participant_spec(artifacts: str, profile: str, fallback_commit_sha: str, user_name: str) -> Dict[str, Any]:
    source_commit_sha = _resolve_source_commit_sha(artifacts, fallback_commit_sha)
    return {
        "artifacts": artifacts,
        "profile": profile,
        "user_name": user_name,
        "source_commit_sha": source_commit_sha,
        "code_origin": f"git:{source_commit_sha}",
        "agent_class": _load_agent_class_from_commit(source_commit_sha),
    }


def _resolve_incumbent_profile(artifacts: str, requested_profile: str | None, fallback_profile: str) -> str:
    if requested_profile:
        return str(requested_profile)
    manifest = _load_artifact_manifest(artifacts)
    snapshot_role = str(manifest.get("snapshot_role", "")).strip()
    return _SNAPSHOT_ROLE_TO_PROFILE.get(snapshot_role, str(fallback_profile))


def _participant_payload(artifacts: str, profile: str, source_commit_sha: str, code_origin: str) -> Dict[str, Any]:
    manifest = _load_artifact_manifest(artifacts)
    payload: Dict[str, Any] = {
        "artifacts": _resolved_path_string(artifacts),
        "profile": profile,
        "source_commit_sha": source_commit_sha,
        "code_origin": code_origin,
    }
    if manifest:
        payload["snapshot_role"] = manifest.get("snapshot_role")
        payload["manifest_generated_at"] = manifest.get("generated_at")
    return payload


def _resolved_path_string(value: str) -> str:
    return str(Path(value).resolve())


def _merge_policy_stats(*payloads: Dict[str, object]) -> Dict[str, object]:
    merged = {
        "move": {"total": 0, "learned": 0},
        "shoot": {"total": 0, "learned": 0},
    }
    score_sums = {
        "move": {"samples": 0, "intent_bias": 0.0, "situation_score": 0.0, "combined_score": 0.0},
        "shoot": {"samples": 0, "intent_bias": 0.0, "situation_score": 0.0, "combined_score": 0.0},
    }
    traces: List[Dict[str, object]] = []
    policy_profiles = set()
    loaded_from_dirs = set()
    intent_model_loaded_count = 0
    situation_model_loaded_count = 0
    for payload in payloads:
        counts = payload.get("counts", {}) if isinstance(payload, dict) else {}
        score_means = payload.get("score_means", {}) if isinstance(payload, dict) else {}
        for action_key in ("move", "shoot"):
            action_counts = counts.get(action_key, {}) if isinstance(counts, dict) else {}
            merged[action_key]["total"] += int(action_counts.get("total", 0))
            merged[action_key]["learned"] += int(action_counts.get("learned", 0))
            action_scores = score_means.get(action_key, {}) if isinstance(score_means, dict) else {}
            samples = int(action_scores.get("samples", 0))
            score_sums[action_key]["samples"] += samples
            for metric in ("intent_bias", "situation_score", "combined_score"):
                score_sums[action_key][metric] += float(action_scores.get(metric, 0.0)) * samples
        if isinstance(payload, dict):
            traces.extend(payload.get("trace_samples", [])[:6])
            profile = payload.get("policy_profile")
            if profile:
                policy_profiles.add(str(profile))
            loaded_from = payload.get("loaded_from")
            if loaded_from:
                loaded_from_dirs.add(str(loaded_from))
            intent_model_loaded_count += int(bool(payload.get("intent_model_loaded")))
            situation_model_loaded_count += int(bool(payload.get("situation_model_loaded")))

    def coverage(total: int, learned: int) -> float:
        if total <= 0:
            return 0.0
        return float(learned / total)

    move_total = merged["move"]["total"]
    move_learned = merged["move"]["learned"]
    shoot_total = merged["shoot"]["total"]
    shoot_learned = merged["shoot"]["learned"]
    score_means = {}
    for action_key, values in score_sums.items():
        samples = int(values["samples"])
        score_means[action_key] = {
            "samples": samples,
            "intent_bias": float(values["intent_bias"] / samples) if samples else 0.0,
            "situation_score": float(values["situation_score"] / samples) if samples else 0.0,
            "combined_score": float(values["combined_score"] / samples) if samples else 0.0,
        }

    return {
        "counts": merged,
        "learned_path_coverage": coverage(move_total + shoot_total, move_learned + shoot_learned),
        "move_path_coverage": coverage(move_total, move_learned),
        "shoot_path_coverage": coverage(shoot_total, shoot_learned),
        "score_means": score_means,
        "policy_profiles": sorted(policy_profiles),
        "loaded_from_dirs": sorted(loaded_from_dirs),
        "intent_model_loaded_count": intent_model_loaded_count,
        "situation_model_loaded_count": situation_model_loaded_count,
        "trace_samples": traces[:12],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Shenji self-play league evaluation.")
    parser.add_argument("--config", default="configs/selfplay_baseline.yaml")
    parser.add_argument("--candidate", default="ai/weights")
    parser.add_argument("--incumbent", default="ai/weights")
    parser.add_argument("--candidate-profile", default=None)
    parser.add_argument("--incumbent-profile", default=None)
    parser.add_argument("--allow-self-match-smoke", action="store_true")
    args = parser.parse_args()

    config = load_config(repo_root() / args.config)
    if not args.allow_self_match_smoke and _resolved_path_string(args.candidate) == _resolved_path_string(args.incumbent):
        raise SystemExit("candidate and incumbent must point to different artifact directories unless --allow-self-match-smoke is set.")

    set_seed(int(config["seed"]))
    run_dir = ensure_dir(repo_root() / config["output_root"] / f"league_{now_tag()}")
    league_dir = ensure_dir(repo_root() / config["league_dir"])
    league_state_path = league_dir / "league_state.json"
    league_state = _bootstrap_league(league_state_path)
    current_head_sha = git_head_sha(repo_root())
    candidate_profile = str(args.candidate_profile or config.get("candidate_profile", "challenger"))
    incumbent_profile = _resolve_incumbent_profile(
        args.incumbent,
        args.incumbent_profile,
        str(config.get("incumbent_profile", "baseline")),
    )
    candidate_participant = _build_participant_spec(args.candidate, candidate_profile, current_head_sha, "CurrentAI")
    incumbent_participant = _build_participant_spec(args.incumbent, incumbent_profile, current_head_sha, "IncumbentAI")
    _, env_backend = _train_env_runtime()
    if env_backend == "mock_train_env.MockTrainEnv":
        raise SystemExit(
            "Formal self-play evaluation requires a real TrainEnv backend; resolved mock_train_env.MockTrainEnv instead."
        )

    maps_root = repo_root() / "land_wargame_sdk" / "Data" / "Data" / "maps"
    scenario_root = repo_root() / "land_wargame_sdk" / "Data" / "Data" / "scenarios"
    records = build_map_catalog(maps_root, repo_root() / "training" / "cache" / "map_catalog.json")
    bucket_lookup = map_bucket_lookup(records)
    scenario_candidates = _collect_scenario_candidates(scenario_root, bucket_lookup)
    paired_map_list = _build_paired_map_list(scenario_candidates, int(config["paired_matches"]))
    save_json(paired_map_list, run_dir / "map_list.json")
    scenarios_by_bucket: Dict[str, List[Tuple[Path, int]]] = {"simple": [], "medium": [], "complex": []}
    for item in scenario_candidates:
        scenario_path = scenario_root / str(item["scenario"])
        scenarios_by_bucket[str(item["bucket"])].append((scenario_path, int(item["map_id"])))

    matches: List[Dict[str, object]] = []
    for map_item in paired_map_list["items"]:
        scenario_path = scenario_root / str(map_item["scenario"])
        map_id = int(map_item["map_id"])
        bucket = str(map_item["bucket"])
        order_index = int(map_item["order_index"])
        red_score, blue_score, red_stats, blue_stats = _run_single(
            scenario_path,
            map_id,
            candidate_participant,
            incumbent_participant,
            int(config["max_steps"]),
        )
        blue_red_score, blue_blue_score, blue_leg_red_stats, blue_leg_blue_stats = _run_single(
            scenario_path,
            map_id,
            incumbent_participant,
            candidate_participant,
            int(config["max_steps"]),
        )
        candidate_stats = _merge_policy_stats(red_stats, blue_leg_blue_stats)
        opponent_stats = _merge_policy_stats(blue_stats, blue_leg_red_stats)
        matches.append(
            {
                "scenario": scenario_path.name,
                "bucket": bucket,
                "map_id": map_id,
                "order_index": order_index,
                "paired_score": _paired_score(red_score, blue_score, blue_blue_score, blue_red_score),
                "first_leg": {"candidate": red_score, "opponent": blue_score},
                "second_leg": {"candidate": blue_blue_score, "opponent": blue_red_score},
                "candidate_artifacts": _resolved_path_string(args.candidate),
                "opponent_artifacts": _resolved_path_string(args.incumbent),
                "candidate_source_commit_sha": candidate_participant["source_commit_sha"],
                "opponent_source_commit_sha": incumbent_participant["source_commit_sha"],
                "candidate_code_origin": candidate_participant["code_origin"],
                "opponent_code_origin": incumbent_participant["code_origin"],
                "candidate_profile": candidate_profile,
                "opponent_profile": incumbent_profile,
                "candidate_policy_stats": candidate_stats,
                "opponent_policy_stats": opponent_stats,
            }
        )

    summary = summarize_matches(matches)

    exploiter_results = {}
    for profile in league_state["exploiters"]:
        exploiter_participant = _build_participant_spec(args.incumbent, profile, current_head_sha, "ExploiterAI")
        profile_scores = []
        profile_coverages = []
        for scenario_path, map_id in scenarios_by_bucket.get("complex", [])[:4]:
            score_a, score_b, candidate_stats, _ = _run_single(
                scenario_path,
                map_id,
                candidate_participant,
                exploiter_participant,
                int(config["max_steps"]),
            )
            profile_scores.append(1.0 if score_b > score_a else 0.0)
            profile_coverages.append(float(candidate_stats.get("learned_path_coverage", 0.0)))
        exploiter_results[profile] = {
            "win_rate_vs_candidate": float(np.mean(profile_scores)) if profile_scores else 0.0,
            "matches": len(profile_scores),
            "candidate_learned_path_coverage": float(np.mean(profile_coverages)) if profile_coverages else 0.0,
        }

    decision = promotion_decision(summary, config["promotion_gate"], exploiter_results)

    if decision["accepted"]:
        league_state["consecutive_windows"] = int(league_state.get("consecutive_windows", 0)) + 1
    else:
        league_state["consecutive_windows"] = 0
    if league_state["consecutive_windows"] >= int(config["promotion_gate"]["consecutive_windows"]):
        league_state["historical_mains"] = [league_state.get("champion", args.incumbent)] + list(league_state.get("historical_mains", []))[:7]
        league_state["champion"] = args.candidate
        league_state["consecutive_windows"] = 0

    save_json({"env_backend": env_backend, "matches": matches}, run_dir / "match_results.json")
    save_json(summary, run_dir / "rating.json")
    save_json(decision, run_dir / "promotion.json")
    save_json(exploiter_results, run_dir / "exploitability.json")
    save_json(
        {
            "env_backend": env_backend,
            "candidate": _participant_payload(
                args.candidate,
                candidate_profile,
                candidate_participant["source_commit_sha"],
                candidate_participant["code_origin"],
            ),
            "incumbent": _participant_payload(
                args.incumbent,
                incumbent_profile,
                incumbent_participant["source_commit_sha"],
                incumbent_participant["code_origin"],
            ),
        },
        run_dir / "participants.json",
    )
    dump_yaml(config, run_dir / "config.yaml")
    save_json(league_state, league_state_path)
    save_json(league_state, run_dir / "league_state.json")
    (run_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Shenji Self-Play Summary",
                "",
                f"- env_backend: `{env_backend}`",
                f"- paired_map_items: `{len(paired_map_list['items'])}`",
                f"- map_list_duplicates_used: `{paired_map_list['duplicates_used']}`",
                f"- candidate_source_commit_sha: `{candidate_participant['source_commit_sha']}`",
                f"- incumbent_source_commit_sha: `{incumbent_participant['source_commit_sha']}`",
                f"- overall_point_estimate: `{summary['overall']['point_estimate']:.4f}`",
                f"- overall_ci95_lower: `{summary['overall']['ci95'][0]:.4f}`",
                f"- complex_point_estimate: `{summary['buckets'].get('complex', {}).get('point_estimate', 0.0):.4f}`",
                f"- complex_ci95_lower: `{summary['buckets'].get('complex', {}).get('ci95', [0.0, 0.0])[0]:.4f}`",
                f"- learned_path_coverage: `{summary['policy'].get('learned_path_coverage', 0.0):.4f}`",
                f"- move_path_coverage: `{summary['policy'].get('move_path_coverage', 0.0):.4f}`",
                f"- shoot_path_coverage: `{summary['policy'].get('shoot_path_coverage', 0.0):.4f}`",
                f"- elo_delta_lower_bound: `{decision.get('elo_delta_lower_bound', 0.0):.4f}`",
                f"- max_exploiter_win_rate: `{decision.get('max_exploiter_win_rate', 0.0):.4f}`",
                f"- promotion_accepted: `{decision['accepted']}`",
                f"- promotion_failures: `{decision.get('failures', [])}`",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
