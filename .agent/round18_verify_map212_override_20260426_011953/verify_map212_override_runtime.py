#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parents[1]
PROBE_PATH = REPO_ROOT / ".agent" / "round18_coverage_probe" / "probe.py"
BASELINE_SUMMARY_PATH = (
    REPO_ROOT / ".agent" / "round18_probe_212013_family_20260426_010048" / "probe_output" / "summary.json"
)
SCENARIOS = ["2120131194.json", "2120131294.json"]
MAX_STEPS = 400
WINDOW_LABELS = ["0-120", "121-280", "281-400"]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.common import save_json, set_seed
from training.maps import build_map_catalog, infer_map_id_from_scenario_id, map_bucket_lookup, scenario_coordinate_overflow_counts
from training.selfplay import _collect_scenario_candidates, _configure_torch_cpu_inference_threads, _merge_policy_stats, _paired_score, _train_env_runtime


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            fh.write("\n")


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True).strip()


def git_dirty_paths() -> List[str]:
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
    spec = importlib.util.spec_from_file_location("round18_verify_map212_override_probe", PROBE_PATH)
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


def load_baseline_summary() -> Dict[str, Any]:
    return json.loads(BASELINE_SUMMARY_PATH.read_text(encoding="utf-8"))


def map_fit(record_lookup: Dict[int, Any], scenario_data: Dict[str, Any], map_id: int) -> Dict[str, Any]:
    record = record_lookup[map_id]
    overflows = scenario_coordinate_overflow_counts(scenario_data, rows=record.rows, cols=record.cols)
    city_total = len(scenario_data.get("cities", []))
    operator_total = len(scenario_data.get("operators", []))
    return {
        "map_id": int(map_id),
        "map_name": str(record.name),
        "bucket": str(record.bucket),
        "rows": int(record.rows),
        "cols": int(record.cols),
        "city_total": int(city_total),
        "city_in_bounds": int(city_total - overflows["city_oob_count"]),
        "city_oob_count": int(overflows["city_oob_count"]),
        "operator_total": int(operator_total),
        "operator_in_bounds": int(operator_total - overflows["operator_oob_count"]),
        "operator_oob_count": int(overflows["operator_oob_count"]),
    }


def sum_windows(payload: Dict[str, Dict[str, int]], field: str) -> int:
    return int(sum(int((payload.get(label) or {}).get(field, 0)) for label in WINDOW_LABELS))


def build_candidate_metrics(
    merged_candidate_stats: Dict[str, Any],
    candidate_failures: Dict[str, Dict[str, int]],
    candidate_windows: Dict[str, Any],
    leg0: Dict[str, Any],
    leg1: Dict[str, Any],
) -> Dict[str, Any]:
    leg0_candidate_score = float(leg0["red_score"])
    leg0_opponent_score = float(leg0["blue_score"])
    leg1_candidate_score = float(leg1["blue_score"])
    leg1_opponent_score = float(leg1["red_score"])
    return {
        "paired_score": float(
            _paired_score(
                leg0_candidate_score,
                leg0_opponent_score,
                leg1_candidate_score,
                leg1_opponent_score,
            )
        ),
        "move_total": int(((merged_candidate_stats.get("counts") or {}).get("move") or {}).get("total", 0)),
        "move_learned": int(((merged_candidate_stats.get("counts") or {}).get("move") or {}).get("learned", 0)),
        "move_path_coverage": float(merged_candidate_stats.get("move_path_coverage", 0.0)),
        "learned_path_coverage": float(merged_candidate_stats.get("learned_path_coverage", 0.0)),
        "shoot_path_coverage": float(merged_candidate_stats.get("shoot_path_coverage", 0.0)),
        "empty_route_total": sum_windows(candidate_failures, "empty_route"),
        "empty_route_with_no_reachable_city_total": sum_windows(
            candidate_failures,
            "empty_route_with_no_reachable_city",
        ),
        "empty_route_with_reachable_alternative_total": sum_windows(
            candidate_failures,
            "empty_route_with_reachable_alternative",
        ),
        "reachable_route_available_total": sum_windows(candidate_failures, "reachable_route_available"),
        "reachable_route_unavailable_total": sum_windows(candidate_failures, "reachable_route_unavailable"),
        "learned_success_total": sum_windows(candidate_failures, "learned_success"),
        "total_move_evals": sum_windows(candidate_failures, "total_move_evals"),
        "window_move_path_coverage": {
            label: float(((candidate_windows.get(label) or {}).get("move_path_coverage")) or 0.0)
            for label in WINDOW_LABELS
        },
    }


def before_after(
    baseline_summary: Dict[str, Any],
    scenario_name: str,
    after_map_id: int,
    after_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    before_payload = baseline_summary["scenarios_by_name"][scenario_name]["maps"]["map_212"]
    before_metrics = before_payload["candidate_metrics"]
    return {
        "before_map_id": 212,
        "after_map_id": int(after_map_id),
        "before_move_path_coverage": float(before_metrics["move_path_coverage"]),
        "after_move_path_coverage": float(after_metrics["move_path_coverage"]),
        "before_learned_path_coverage": float(before_metrics["learned_path_coverage"]),
        "after_learned_path_coverage": float(after_metrics["learned_path_coverage"]),
        "before_empty_route_with_no_reachable_city": int(before_metrics["empty_route_with_no_reachable_city_total"]),
        "after_empty_route_with_no_reachable_city": int(after_metrics["empty_route_with_no_reachable_city_total"]),
        "before_empty_route_with_reachable_alternative": int(
            before_metrics["empty_route_with_reachable_alternative_total"]
        ),
        "after_empty_route_with_reachable_alternative": int(
            after_metrics["empty_route_with_reachable_alternative_total"]
        ),
        "before_operator_fit": before_payload["map_fit"]["operator_in_bounds"],
        "before_operator_total": before_payload["map_fit"]["operator_total"],
    }


def build_memo(summary: Dict[str, Any]) -> str:
    lines = [
        "# Round 18 Runtime Verification: map_212 Explicit Overrides",
        "",
        f"- generated_at: `{summary['generated_at']}`",
        f"- scenarios: `{', '.join(summary['scenarios'])}`",
        f"- env_backend: `{summary['env_backend']}`",
        f"- requested_device_env: `{summary['requested_device_env']}`",
        "",
    ]
    for scenario_name, payload in summary["scenarios_by_name"].items():
        metrics = payload["candidate_metrics"]
        comparison = payload["before_after_vs_prepatch_map212"]
        fit = payload["resolved_map_fit"]
        lines.extend(
            [
                f"## {scenario_name}",
                f"- resolved map: `map_{payload['resolved_map_id']}` bucket `{payload['resolved_bucket']}`",
                f"- fit: `cities {fit['city_in_bounds']}/{fit['city_total']}`, `operators {fit['operator_in_bounds']}/{fit['operator_total']}`",
                f"- candidate coverage: `move {metrics['move_path_coverage']:.4f}`, `learned {metrics['learned_path_coverage']:.4f}`, `shoot {metrics['shoot_path_coverage']:.4f}`",
                f"- candidate empty_route_with_no_reachable_city: `{metrics['empty_route_with_no_reachable_city_total']}`",
                f"- pre-patch map_212 comparison: `move {comparison['before_move_path_coverage']:.4f} -> {comparison['after_move_path_coverage']:.4f}`, `learned {comparison['before_learned_path_coverage']:.4f} -> {comparison['after_learned_path_coverage']:.4f}`, `no_reachable {comparison['before_empty_route_with_no_reachable_city']} -> {comparison['after_empty_route_with_no_reachable_city']}`",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    from training.maps import _middle_token_map_id

    scenario_root = REPO_ROOT / "land_wargame_sdk" / "Data" / "Data" / "scenarios"
    maps_root = REPO_ROOT / "land_wargame_sdk" / "Data" / "Data" / "maps"
    records = build_map_catalog(maps_root)
    bucket_lookup = map_bucket_lookup(records)
    candidates = _collect_scenario_candidates(scenario_root, bucket_lookup, records)
    resolved_lookup = {item["scenario"]: item for item in candidates if item["scenario"] in SCENARIOS}
    record_lookup = {record.map_id: record for record in records}
    baseline_summary = load_baseline_summary()

    resolution_payload: Dict[str, Any] = {}
    for scenario_name in SCENARIOS:
        scenario_data = json.loads((scenario_root / scenario_name).read_text(encoding="utf-8"))
        heuristic_map_id = infer_map_id_from_scenario_id(Path(scenario_name).stem, record_lookup.keys())
        structured_fallback_map_id = _middle_token_map_id(Path(scenario_name).stem, record_lookup.keys())
        resolved_item = resolved_lookup[scenario_name]
        resolution_payload[scenario_name] = {
            "scenario": scenario_name,
            "resolved_via_collect_scenario_candidates": resolved_item,
            "heuristic_map_id": heuristic_map_id,
            "structured_fallback_map_id": structured_fallback_map_id,
            "fits": {
                "heuristic": map_fit(record_lookup, scenario_data, heuristic_map_id) if heuristic_map_id is not None else None,
                "resolved": map_fit(record_lookup, scenario_data, int(resolved_item["map_id"])),
            },
        }
    write_json(RUN_DIR / "resolution.json", {"generated_at": now_utc(), "scenarios": resolution_payload})

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
        "baseline_summary_path": str(BASELINE_SUMMARY_PATH),
        "scenarios": list(SCENARIOS),
        "env_backend": env_backend,
        "git_head": current_head,
        "git_dirty_paths": dirty_paths,
        "requested_device_env": os.environ.get("SHENJI_POLICY_DEVICE", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
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
        "resolved_lookup": resolved_lookup,
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

    status = {
        "phase": "running",
        "scenarios": list(SCENARIOS),
        "completed_scenarios": [],
        "current_scenario": None,
        "current_leg_index": None,
        "max_steps": MAX_STEPS,
        "updated_at": now_utc(),
    }
    write_json(RUN_DIR / "status.json", status)

    all_step_records: List[Dict[str, Any]] = []
    scenario_outputs: Dict[str, Any] = {}

    for scenario_name in SCENARIOS:
        resolved_map_id = int(resolved_lookup[scenario_name]["map_id"])
        status.update(
            {
                "current_scenario": scenario_name,
                "current_leg_index": 0,
                "updated_at": now_utc(),
            }
        )
        write_json(RUN_DIR / "status.json", status)

        leg0 = probe.run_leg(
            scenario_path=scenario_root / scenario_name,
            map_id=resolved_map_id,
            red_participant=candidate,
            blue_participant=incumbent,
            red_role="candidate",
            blue_role="opponent",
            leg_index=0,
            max_steps=MAX_STEPS,
            recorder=recorder,
        )
        status.update(
            {
                "current_scenario": scenario_name,
                "current_leg_index": 1,
                "updated_at": now_utc(),
            }
        )
        write_json(RUN_DIR / "status.json", status)

        leg1 = probe.run_leg(
            scenario_path=scenario_root / scenario_name,
            map_id=resolved_map_id,
            red_participant=incumbent,
            blue_participant=candidate,
            red_role="opponent",
            blue_role="candidate",
            leg_index=1,
            max_steps=MAX_STEPS,
            recorder=recorder,
        )

        all_step_records.extend(leg0["step_records"])
        all_step_records.extend(leg1["step_records"])
        scenario_events = [event for event in recorder.move_events if event.get("scenario") == scenario_name]
        scenario_steps = [record for record in all_step_records if record.get("scenario") == scenario_name]

        merged_candidate_stats = _merge_policy_stats(leg0["red_policy_stats"], leg1["blue_policy_stats"])
        merged_opponent_stats = _merge_policy_stats(leg0["blue_policy_stats"], leg1["red_policy_stats"])
        windows = probe.summarize_step_windows(scenario_steps)
        move_failures = probe.summarize_move_events(scenario_events)
        candidate_failures = move_failures.get("candidate", {})
        candidate_windows = windows.get("candidate", {})
        candidate_metrics = build_candidate_metrics(
            merged_candidate_stats,
            candidate_failures,
            candidate_windows,
            leg0,
            leg1,
        )

        scenario_data = json.loads((scenario_root / scenario_name).read_text(encoding="utf-8"))
        scenario_outputs[scenario_name] = {
            "scenario": scenario_name,
            "resolved_map_id": resolved_map_id,
            "resolved_bucket": str(resolved_lookup[scenario_name]["bucket"]),
            "resolved_map_fit": map_fit(record_lookup, scenario_data, resolved_map_id),
            "legs": [
                {
                    "leg_index": int(leg0["leg_index"]),
                    "red_role": str(leg0["red_role"]),
                    "blue_role": str(leg0["blue_role"]),
                    "steps_executed": int(leg0["steps_executed"]),
                    "done": bool(leg0["done"]),
                    "red_score": float(leg0["red_score"]),
                    "blue_score": float(leg0["blue_score"]),
                },
                {
                    "leg_index": int(leg1["leg_index"]),
                    "red_role": str(leg1["red_role"]),
                    "blue_role": str(leg1["blue_role"]),
                    "steps_executed": int(leg1["steps_executed"]),
                    "done": bool(leg1["done"]),
                    "red_score": float(leg1["red_score"]),
                    "blue_score": float(leg1["blue_score"]),
                },
            ],
            "merged_policy_stats": {
                "candidate": merged_candidate_stats,
                "opponent": merged_opponent_stats,
            },
            "windows": windows,
            "move_failures": move_failures,
            "candidate_metrics": candidate_metrics,
            "before_after_vs_prepatch_map212": before_after(
                baseline_summary,
                scenario_name,
                resolved_map_id,
                candidate_metrics,
            ),
        }
        status["completed_scenarios"].append(scenario_name)
        status["current_scenario"] = None
        status["current_leg_index"] = None
        status["updated_at"] = now_utc()
        write_json(RUN_DIR / "status.json", status)

    summary = {
        "generated_at": now_utc(),
        "env_backend": env_backend,
        "requested_device_env": os.environ.get("SHENJI_POLICY_DEVICE", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "baseline_summary_path": str(BASELINE_SUMMARY_PATH),
        "scenarios": list(SCENARIOS),
        "scenarios_by_name": scenario_outputs,
    }
    output_dir = RUN_DIR / "verify_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(summary, output_dir / "summary.json")
    (output_dir / "memo.md").write_text(build_memo(summary), encoding="utf-8")
    append_jsonl(output_dir / "move_events.jsonl", recorder.move_events)
    append_jsonl(output_dir / "step_records.jsonl", all_step_records)
    status.update(
        {
            "phase": "completed",
            "summary_path": str((output_dir / "summary.json").resolve()),
            "memo_path": str((output_dir / "memo.md").resolve()),
            "move_events_path": str((output_dir / "move_events.jsonl").resolve()),
            "step_records_path": str((output_dir / "step_records.jsonl").resolve()),
            "updated_at": now_utc(),
        }
    )
    write_json(RUN_DIR / "status.json", status)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
