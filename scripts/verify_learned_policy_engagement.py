#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.common import git_head_sha, repo_root, save_json
from training.selfplay import (
    _build_match_policy_diagnostics,
    _build_participant_spec,
    _configure_torch_cpu_inference_threads,
    _load_artifact_manifest,
    _load_case_data,
    _manifest_path_string,
    _merge_policy_stats,
)


def _evaluate_contract(payload: Dict[str, Any]) -> Dict[str, Any]:
    failures: list[str] = []
    candidate = payload.get("candidate", {})
    candidate_participant = candidate.get("participant", {}) if isinstance(candidate, dict) else {}
    candidate_diagnostics = candidate.get("policy_diagnostics_after_setup", {}) if isinstance(candidate, dict) else {}
    incumbent = payload.get("incumbent", {})
    incumbent_participant = incumbent.get("participant", {}) if isinstance(incumbent, dict) else {}
    incumbent_diagnostics = incumbent.get("policy_diagnostics_after_setup", {}) if isinstance(incumbent, dict) else {}

    if not bool(candidate.get("manifest_loaded")):
        failures.append("candidate_manifest_missing")
    if str(candidate_participant.get("user_name")) != "CurrentAI":
        failures.append("candidate_user_name_not_currentai")
    if str(candidate_diagnostics.get("profile")) != "challenger":
        failures.append("candidate_profile_not_challenger")
    if not bool(candidate_diagnostics.get("manifest_loaded")):
        failures.append("candidate_manifest_not_loaded")
    if not bool(candidate_diagnostics.get("intent_model_loaded")):
        failures.append("candidate_intent_model_not_loaded")
    if not bool(candidate_diagnostics.get("situation_model_loaded")):
        failures.append("candidate_situation_model_not_loaded")
    if not bool(candidate_diagnostics.get("learned_policy_enabled")):
        failures.append("candidate_learned_policy_not_enabled")
    if not bool(candidate_diagnostics.get("use_ai_optimization")):
        failures.append("candidate_ai_optimization_disabled")

    if str(incumbent_participant.get("user_name")) != "IncumbentAI":
        failures.append("incumbent_user_name_not_incumbentai")
    if str(incumbent_diagnostics.get("profile")) != "baseline":
        failures.append("incumbent_profile_not_baseline")

    return {
        "passed": not failures,
        "failures": failures,
    }


def _agent_setup_diagnostics(
    *,
    artifacts: str,
    profile: str,
    user_name: str,
    seat: int,
    faction: int,
    case_data: Dict[str, object],
) -> Dict[str, Any]:
    current_head = git_head_sha(repo_root())
    participant = _build_participant_spec(artifacts, profile, current_head, user_name)
    agent = participant["agent_class"](log_level=logging.CRITICAL)
    agent.setup(
        {
            **case_data,
            "seat": seat,
            "faction": faction,
            "role": 1,
            "user_name": user_name,
            "policy_profile": profile,
            "policy_artifacts_dir": artifacts,
        }
    )
    raw_stats = agent.get_policy_stats()
    merged_stats = _merge_policy_stats(raw_stats)
    diagnostics = _build_match_policy_diagnostics(
        artifacts=artifacts,
        participant=participant,
        profile=profile,
        stats=merged_stats,
    )
    diagnostics["use_ai_optimization"] = bool(raw_stats.get("use_ai_optimization"))
    diagnostics["policy_profiles"] = list(merged_stats.get("policy_profiles", []))
    diagnostics["trace_samples"] = list(merged_stats.get("trace_samples", []))
    return {
        "participant": {
            "user_name": participant["user_name"],
            "profile": participant["profile"],
            "artifacts": str(Path(artifacts).resolve()),
            "source_commit_sha": participant["source_commit_sha"],
            "code_origin": participant["code_origin"],
        },
        "manifest_path": _manifest_path_string(artifacts),
        "manifest_loaded": bool(_load_artifact_manifest(artifacts)),
        "policy_stats_after_setup": raw_stats,
        "policy_diagnostics_after_setup": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify that formal self-play candidate wiring loads and enables learned policy artifacts.")
    parser.add_argument("--candidate", default="ai/weights")
    parser.add_argument("--candidate-profile", default="challenger")
    parser.add_argument("--candidate-user-name", default="CurrentAI")
    parser.add_argument("--incumbent", default="ai/weights_incumbent_rule")
    parser.add_argument("--incumbent-profile", default="baseline")
    parser.add_argument("--incumbent-user-name", default="IncumbentAI")
    parser.add_argument("--scenario", default="2230531221.json")
    parser.add_argument("--map-id", type=int, default=53)
    parser.add_argument(
        "--output",
        default="experiments/shenji_agent/learned_policy_engagement_verify.json",
        help="Path to write the verification JSON artifact.",
    )
    args = parser.parse_args()

    _configure_torch_cpu_inference_threads()
    root = repo_root()
    scenario_path = root / "land_wargame_sdk" / "Data" / "Data" / "scenarios" / args.scenario
    case_data = _load_case_data(scenario_path, int(args.map_id))

    payload = {
        "git_head": git_head_sha(root),
        "candidate": _agent_setup_diagnostics(
            artifacts=args.candidate,
            profile=str(args.candidate_profile),
            user_name=str(args.candidate_user_name),
            seat=1,
            faction=0,
            case_data=case_data,
        ),
        "incumbent": _agent_setup_diagnostics(
            artifacts=args.incumbent,
            profile=str(args.incumbent_profile),
            user_name=str(args.incumbent_user_name),
            seat=11,
            faction=1,
            case_data=case_data,
        ),
        "scenario": str(scenario_path.resolve()),
        "map_id": int(args.map_id),
    }
    payload["contract_verdict"] = _evaluate_contract(payload)
    save_json(payload, root / args.output)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not payload["contract_verdict"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
