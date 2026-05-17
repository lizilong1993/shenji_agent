#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parents[1]
PROBE_PATH = REPO_ROOT / ".agent" / "round18_coverage_probe" / "probe.py"
BASELINE_SUMMARY_PATH = REPO_ROOT / ".agent" / "round18_probe_gpu_20260425_154341" / "probe_output" / "summary.json"
SCENARIO_NAME = "2230531221.json"
MAX_STEPS = 400

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True).strip()


def git_dirty_paths() -> list[str]:
    output = subprocess.check_output(
        [
            "git",
            "status",
            "--short",
            "training/maps.py",
            "training/selfplay.py",
            "ai/agent.py",
            "ai/policy.py",
            ".agent/round18_coverage_probe/probe.py",
        ],
        cwd=str(REPO_ROOT),
        text=True,
    )
    return [line.rstrip() for line in output.splitlines() if line.strip()]


def load_probe_module() -> Any:
    spec = importlib.util.spec_from_file_location("round18_verify_223_probe", PROBE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load probe module from {PROBE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def first_param_device(model: Any) -> str | None:
    if model is None:
        return None
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return None
    try:
        first = next(parameters())
    except StopIteration:
        return None
    return str(getattr(first, "device", None))


def policy_snapshot(label: str, artifact_dir: Path, profile: str) -> Dict[str, Any]:
    from ai.policy import LearnedPolicy

    policy = LearnedPolicy()
    faction = 0 if label == "candidate" else 1
    policy.configure(faction=faction, artifact_dir=str(artifact_dir), policy_profile=profile)
    diagnostics = policy.get_diagnostics()
    return {
        "artifacts": str(artifact_dir),
        "profile": profile,
        "policy_device": str(getattr(policy, "device", "unknown")),
        "requested_device": str(getattr(policy, "requested_device", "unknown")),
        "loaded_from": diagnostics.get("loaded_from"),
        "intent_model_loaded": bool(diagnostics.get("intent_model_loaded")),
        "situation_model_loaded": bool(diagnostics.get("situation_model_loaded")),
        "intent_param_device": first_param_device(policy.intent_model),
        "situation_param_device": first_param_device(policy.situation_model),
    }


def participant(label: str, artifact_dir: Path, profile: str, source_commit_sha: str) -> Dict[str, Any]:
    from ai.agent import Agent

    return {
        "label": label,
        "artifacts": str(artifact_dir),
        "profile": profile,
        "user_name": "CurrentAI" if label == "candidate" else "IncumbentAI",
        "source_commit_sha": source_commit_sha,
        "code_origin": f"worktree:{REPO_ROOT}@{source_commit_sha}",
        "agent_class": Agent,
    }


def compute_candidate_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    windows = payload["move_failures"]["candidate"]
    ordered_windows = ["0-120", "121-280", "281-400"]
    return {
        "map_id": int(payload["map_id"]),
        "candidate_move_path_coverage": float(payload["merged_policy_stats"]["candidate"]["move_path_coverage"]),
        "candidate_empty_route_total": int(sum(windows[label]["empty_route"] for label in ordered_windows)),
        "candidate_empty_route_with_no_reachable_city_total": int(
            sum(windows[label]["empty_route_with_no_reachable_city"] for label in ordered_windows)
        ),
        "candidate_empty_route_with_reachable_alternative_total": int(
            sum(windows[label]["empty_route_with_reachable_alternative"] for label in ordered_windows)
        ),
        "candidate_windows": windows,
    }


def main() -> int:
    from training.common import set_seed
    from training.maps import (
        _middle_token_map_id,
        build_map_catalog,
        infer_map_id_from_scenario_id,
        map_bucket_lookup,
        scenario_coordinate_overflow_counts,
    )
    from training.selfplay import (
        _collect_scenario_candidates,
        _configure_torch_cpu_inference_threads,
        _merge_policy_stats,
        _train_env_runtime,
    )

    scenario_root = REPO_ROOT / "land_wargame_sdk" / "Data" / "Data" / "scenarios"
    maps_root = REPO_ROOT / "land_wargame_sdk" / "Data" / "Data" / "maps"
    scenario_path = scenario_root / SCENARIO_NAME
    scenario_data = json.loads(scenario_path.read_text(encoding="utf-8"))
    records = build_map_catalog(maps_root)
    bucket_lookup = map_bucket_lookup(records)
    candidates = _collect_scenario_candidates(scenario_root, bucket_lookup, records)
    resolved_item = next(item for item in candidates if item["scenario"] == SCENARIO_NAME)

    record_lookup = {record.map_id: record for record in records}
    heuristic_map_id = infer_map_id_from_scenario_id(scenario_path.stem, record_lookup.keys())
    structured_fallback_map_id = _middle_token_map_id(scenario_path.stem, record_lookup.keys())
    city_total = len(scenario_data.get("cities", []))
    operator_total = len(scenario_data.get("operators", []))

    def map_fit(map_id: int | None) -> Dict[str, Any] | None:
        if map_id is None:
            return None
        record = record_lookup.get(map_id)
        if record is None:
            return None
        overflows = scenario_coordinate_overflow_counts(scenario_data, rows=record.rows, cols=record.cols)
        return {
            "map_id": int(map_id),
            "rows": int(record.rows),
            "cols": int(record.cols),
            "city_total": city_total,
            "city_in_bounds": int(city_total - overflows["city_oob_count"]),
            "city_oob_count": int(overflows["city_oob_count"]),
            "operator_total": operator_total,
            "operator_in_bounds": int(operator_total - overflows["operator_oob_count"]),
            "operator_oob_count": int(overflows["operator_oob_count"]),
        }

    resolution = {
        "generated_at": now_utc(),
        "scenario": SCENARIO_NAME,
        "scenario_path": str(scenario_path),
        "resolved_via_collect_scenario_candidates": resolved_item,
        "heuristic_map_id": heuristic_map_id,
        "structured_fallback_map_id": structured_fallback_map_id,
        "fits": {
            "heuristic": map_fit(heuristic_map_id),
            "structured_fallback": map_fit(structured_fallback_map_id),
            "resolved": map_fit(int(resolved_item["map_id"])),
        },
    }
    write_json(RUN_DIR / "resolution.json", resolution)

    import ai.agent as agent_module
    import ai.policy as policy_module

    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]

    _, env_backend = _train_env_runtime()
    current_head = git_head()
    dirty_paths = git_dirty_paths()
    source_commit_sha = f"{current_head}+dirty" if dirty_paths else current_head
    candidate_artifacts = REPO_ROOT / "ai" / "weights"
    incumbent_artifacts = REPO_ROOT / "artifacts" / "best_agent" / "weights"
    preflight = {
        "generated_at": now_utc(),
        "repo_root": str(REPO_ROOT),
        "run_dir": str(RUN_DIR),
        "probe_path": str(PROBE_PATH),
        "scenario": SCENARIO_NAME,
        "env_backend": env_backend,
        "git_head": current_head,
        "git_dirty_paths": dirty_paths,
        "requested_device_env": os.environ.get("SHENJI_POLICY_DEVICE", ""),
        "ai_policy_file": str(Path(policy_module.__file__).resolve()),
        "ai_agent_file": str(Path(agent_module.__file__).resolve()),
        "torch_available": torch is not None,
        "torch_cuda_available": bool(torch is not None and torch.cuda.is_available()),
        "torch_cuda_device_count": int(torch.cuda.device_count()) if torch is not None and torch.cuda.is_available() else 0,
        "torch_cuda_current_device": int(torch.cuda.current_device()) if torch is not None and torch.cuda.is_available() else None,
        "torch_cuda_device_name": (
            str(torch.cuda.get_device_name(torch.cuda.current_device()))
            if torch is not None and torch.cuda.is_available()
            else None
        ),
        "participants": {
            "candidate": policy_snapshot("candidate", candidate_artifacts, "challenger"),
            "incumbent": policy_snapshot("incumbent", incumbent_artifacts, "incumbent"),
        },
    }
    write_json(RUN_DIR / "preflight.json", preflight)
    if env_backend == "mock_train_env.MockTrainEnv":
        raise SystemExit("Real runtime blocked: resolved mock_train_env.MockTrainEnv instead of a real TrainEnv backend.")

    probe = load_probe_module()
    recorder = probe.ProbeRecorder()
    from ai.agent import Agent

    probe.patch_agent_class(Agent, recorder)
    _configure_torch_cpu_inference_threads()
    set_seed(42)

    candidate = participant("candidate", candidate_artifacts, "challenger", source_commit_sha)
    incumbent = participant("opponent", incumbent_artifacts, "incumbent", source_commit_sha)
    resolved_map_id = int(resolved_item["map_id"])

    leg0 = probe.run_leg(
        scenario_path=scenario_path,
        map_id=resolved_map_id,
        red_participant=candidate,
        blue_participant=incumbent,
        red_role="candidate",
        blue_role="opponent",
        leg_index=0,
        max_steps=MAX_STEPS,
        recorder=recorder,
    )
    leg1 = probe.run_leg(
        scenario_path=scenario_path,
        map_id=resolved_map_id,
        red_participant=incumbent,
        blue_participant=candidate,
        red_role="opponent",
        blue_role="candidate",
        leg_index=1,
        max_steps=MAX_STEPS,
        recorder=recorder,
    )

    step_records = list(leg0["step_records"]) + list(leg1["step_records"])
    scenario_events = [event for event in recorder.move_events if event.get("scenario") == SCENARIO_NAME]
    summary = {
        "generated_at": now_utc(),
        "scenario": SCENARIO_NAME,
        "env_backend": env_backend,
        "resolved_item": resolved_item,
        "max_steps": MAX_STEPS,
        "legs": [
            {
                "leg_index": leg0["leg_index"],
                "red_role": leg0["red_role"],
                "blue_role": leg0["blue_role"],
                "steps_executed": leg0["steps_executed"],
                "done": leg0["done"],
                "red_score": leg0["red_score"],
                "blue_score": leg0["blue_score"],
                "red_policy_stats": leg0["red_policy_stats"],
                "blue_policy_stats": leg0["blue_policy_stats"],
            },
            {
                "leg_index": leg1["leg_index"],
                "red_role": leg1["red_role"],
                "blue_role": leg1["blue_role"],
                "steps_executed": leg1["steps_executed"],
                "done": leg1["done"],
                "red_score": leg1["red_score"],
                "blue_score": leg1["blue_score"],
                "red_policy_stats": leg1["red_policy_stats"],
                "blue_policy_stats": leg1["blue_policy_stats"],
            },
        ],
        "merged_policy_stats": {
            "candidate": _merge_policy_stats(leg0["red_policy_stats"], leg1["blue_policy_stats"]),
            "opponent": _merge_policy_stats(leg0["blue_policy_stats"], leg1["red_policy_stats"]),
        },
        "windows": probe.summarize_step_windows(step_records),
        "move_failures": probe.summarize_move_events(scenario_events),
    }
    write_json(RUN_DIR / "summary.json", summary)

    with (RUN_DIR / "move_events.jsonl").open("w", encoding="utf-8") as fh:
        for event in recorder.move_events:
            fh.write(json.dumps(event, ensure_ascii=True, sort_keys=True))
            fh.write("\n")

    with (RUN_DIR / "step_records.jsonl").open("w", encoding="utf-8") as fh:
        for record in step_records:
            fh.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
            fh.write("\n")

    baseline_summary = json.loads(BASELINE_SUMMARY_PATH.read_text(encoding="utf-8"))
    baseline_metrics = compute_candidate_metrics(baseline_summary["scenarios"][SCENARIO_NAME])
    after_metrics = compute_candidate_metrics(summary)
    comparison = {
        "generated_at": now_utc(),
        "scenario": SCENARIO_NAME,
        "baseline_summary_path": str(BASELINE_SUMMARY_PATH),
        "baseline": baseline_metrics,
        "after": after_metrics,
        "delta": {
            "candidate_move_path_coverage": float(
                after_metrics["candidate_move_path_coverage"] - baseline_metrics["candidate_move_path_coverage"]
            ),
            "candidate_empty_route_total": int(
                after_metrics["candidate_empty_route_total"] - baseline_metrics["candidate_empty_route_total"]
            ),
            "candidate_empty_route_with_no_reachable_city_total": int(
                after_metrics["candidate_empty_route_with_no_reachable_city_total"]
                - baseline_metrics["candidate_empty_route_with_no_reachable_city_total"]
            ),
        },
    }
    write_json(RUN_DIR / "comparison.json", comparison)

    memo_lines = [
        "# Round 18 2230531221 Runtime Verification",
        "",
        f"- generated_at: `{comparison['generated_at']}`",
        f"- resolved_map_id: `{resolved_map_id}`",
        f"- heuristic_map_id: `{heuristic_map_id}`",
        f"- structured_fallback_map_id: `{structured_fallback_map_id}`",
        f"- baseline_map_id: `{baseline_metrics['map_id']}`",
        f"- candidate move coverage before -> after: `{baseline_metrics['candidate_move_path_coverage']:.6f}` -> `{after_metrics['candidate_move_path_coverage']:.6f}`",
        f"- candidate empty_route_with_no_reachable_city before -> after: `{baseline_metrics['candidate_empty_route_with_no_reachable_city_total']}` -> `{after_metrics['candidate_empty_route_with_no_reachable_city_total']}`",
        "",
    ]
    (RUN_DIR / "memo.md").write_text("\n".join(memo_lines), encoding="utf-8")

    invocation = {
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "run_dir": str(RUN_DIR),
        "requested_device_env": os.environ.get("SHENJI_POLICY_DEVICE", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    write_json(RUN_DIR / "invocation.json", invocation)

    print(json.dumps({"resolution": resolution, "comparison": comparison}, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
