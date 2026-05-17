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
SCENARIO_NAME = "2120511121.json"
EXPECTED_RESOLVED_MAP_ID = 21
BASELINE_MAP_ID = 212
MAX_STEPS = 400
WINDOW_LABELS = ["0-120", "121-280", "281-400"]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
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
            "tests/test_mainline_contract.py",
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
    spec = importlib.util.spec_from_file_location("round18_verify_212051_override_probe", PROBE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load probe module from {PROBE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sum_windows(payload: Dict[str, Dict[str, int]], field: str) -> int:
    return int(sum(int((payload.get(label) or {}).get(field, 0)) for label in WINDOW_LABELS))


def map_fit(
    record_lookup: Dict[int, Any],
    scenario_data: Dict[str, Any],
    map_id: int,
) -> Dict[str, Any]:
    from training.maps import scenario_coordinate_overflow_counts

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


def build_participant(label: str, artifact_dir: Path, profile: str, source_commit_sha: str) -> Dict[str, Any]:
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


def build_metrics(
    merged_candidate_stats: Dict[str, Any],
    candidate_failures: Dict[str, Dict[str, int]],
    candidate_windows: Dict[str, Any],
    leg0: Dict[str, Any],
    leg1: Dict[str, Any],
) -> Dict[str, Any]:
    from training.selfplay import _paired_score

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
        "candidate_leg_scores": [
            {"leg_index": 0, "candidate_score": leg0_candidate_score, "opponent_score": leg0_opponent_score},
            {"leg_index": 1, "candidate_score": leg1_candidate_score, "opponent_score": leg1_opponent_score},
        ],
        "move_total": int(((merged_candidate_stats.get("counts") or {}).get("move") or {}).get("total", 0)),
        "move_learned": int(((merged_candidate_stats.get("counts") or {}).get("move") or {}).get("learned", 0)),
        "shoot_total": int(((merged_candidate_stats.get("counts") or {}).get("shoot") or {}).get("total", 0)),
        "shoot_learned": int(((merged_candidate_stats.get("counts") or {}).get("shoot") or {}).get("learned", 0)),
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
        "window_learned_path_coverage": {
            label: float(((candidate_windows.get(label) or {}).get("learned_path_coverage")) or 0.0)
            for label in WINDOW_LABELS
        },
    }


def build_memo(summary: Dict[str, Any]) -> str:
    before = summary["comparison"]["before_map_212"]
    after = summary["comparison"]["after_resolved"]
    before_map_key = f"map_{BASELINE_MAP_ID}"
    after_map_key = f"map_{after['map_id']}"
    before_fit = summary["maps"][before_map_key]["map_fit"]
    after_fit = summary["maps"][after_map_key]["map_fit"]
    lines = [
        "# Round 18 Verify 212051 Override",
        "",
        f"- generated_at: `{summary['generated_at']}`",
        f"- scenario: `{summary['scenario']}`",
        f"- env_backend: `{summary['env_backend']}`",
        f"- resolver map_id: `{summary['resolution']['resolved_map_id']}`",
        f"- expected resolver map_id: `{summary['resolution']['expected_resolved_map_id']}`",
        f"- baseline unresolved map: `map_{BASELINE_MAP_ID}`",
        "",
        "## Before vs After",
        f"- before `map_{BASELINE_MAP_ID}` move/learned coverage: `{before['move_path_coverage']:.4f}` / `{before['learned_path_coverage']:.4f}`",
        f"- after `map_{after['map_id']}` move/learned coverage: `{after['move_path_coverage']:.4f}` / `{after['learned_path_coverage']:.4f}`",
        f"- before `empty_route_with_no_reachable_city`: `{before['empty_route_with_no_reachable_city_total']}`",
        f"- after `empty_route_with_no_reachable_city`: `{after['empty_route_with_no_reachable_city_total']}`",
        f"- decisive_recovery: `{summary['comparison']['decisive_recovery']}`",
        "",
        "## Fit",
        f"- `map_{BASELINE_MAP_ID}`: cities `{before_fit['city_in_bounds']}/{before_fit['city_total']}`, operators `{before_fit['operator_in_bounds']}/{before_fit['operator_total']}` in bounds",
        f"- `map_{after['map_id']}`: cities `{after_fit['city_in_bounds']}/{after_fit['city_total']}`, operators `{after_fit['operator_in_bounds']}/{after_fit['operator_total']}` in bounds",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    from training.common import set_seed
    from training.maps import build_map_catalog, map_bucket_lookup
    from training.selfplay import (
        _collect_scenario_candidates,
        _configure_torch_cpu_inference_threads,
        _merge_policy_stats,
        _train_env_runtime,
    )

    invocation = {
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "run_dir": str(RUN_DIR),
        "probe_path": str(PROBE_PATH),
        "requested_device_env": os.environ.get("SHENJI_POLICY_DEVICE", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    write_json(RUN_DIR / "invocation.json", invocation)

    scenario_root = REPO_ROOT / "land_wargame_sdk" / "Data" / "Data" / "scenarios"
    maps_root = REPO_ROOT / "land_wargame_sdk" / "Data" / "Data" / "maps"
    scenario_path = scenario_root / SCENARIO_NAME
    scenario_data = json.loads(scenario_path.read_text(encoding="utf-8"))

    records = build_map_catalog(maps_root)
    record_lookup = {record.map_id: record for record in records}
    bucket_lookup = map_bucket_lookup(records)
    resolved_candidates = _collect_scenario_candidates(scenario_root, bucket_lookup, records)
    resolved_item = next(item for item in resolved_candidates if item["scenario"] == SCENARIO_NAME)
    resolved_map_id = int(resolved_item["map_id"])

    resolution = {
        "generated_at": now_utc(),
        "scenario": SCENARIO_NAME,
        "resolved_item": resolved_item,
        "resolved_map_id": resolved_map_id,
        "expected_resolved_map_id": EXPECTED_RESOLVED_MAP_ID,
        "resolved_matches_expected": bool(resolved_map_id == EXPECTED_RESOLVED_MAP_ID),
        "baseline_map_id": BASELINE_MAP_ID,
    }
    write_json(RUN_DIR / "resolution.json", resolution)

    _, env_backend = _train_env_runtime()
    if env_backend == "mock_train_env.MockTrainEnv":
        raise SystemExit("Real verification blocked: resolved mock_train_env.MockTrainEnv instead of a real TrainEnv backend.")

    current_head = git_head()
    dirty_paths = git_dirty_paths()
    source_commit_sha = f"{current_head}+dirty" if dirty_paths else current_head
    candidate = build_participant("candidate", REPO_ROOT / "ai" / "weights", "challenger", source_commit_sha)
    incumbent = build_participant("opponent", REPO_ROOT / "artifacts" / "best_agent" / "weights", "incumbent", source_commit_sha)

    preflight = {
        "generated_at": now_utc(),
        "repo_root": str(REPO_ROOT),
        "run_dir": str(RUN_DIR),
        "scenario": SCENARIO_NAME,
        "env_backend": env_backend,
        "git_head": current_head,
        "git_dirty_paths": dirty_paths,
        "requested_device_env": os.environ.get("SHENJI_POLICY_DEVICE", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "candidate": {key: candidate[key] for key in ("artifacts", "profile", "source_commit_sha", "code_origin")},
        "incumbent": {key: incumbent[key] for key in ("artifacts", "profile", "source_commit_sha", "code_origin")},
    }
    write_json(RUN_DIR / "preflight.json", preflight)

    probe = load_probe_module()
    recorder = probe.ProbeRecorder()

    from ai.agent import Agent

    probe.patch_agent_class(Agent, recorder)
    _configure_torch_cpu_inference_threads()

    maps_to_run = [BASELINE_MAP_ID]
    if resolved_map_id != BASELINE_MAP_ID:
        maps_to_run.append(resolved_map_id)

    all_move_events: List[Dict[str, Any]] = []
    all_step_records: List[Dict[str, Any]] = []
    results: Dict[str, Any] = {}

    for map_id in maps_to_run:
        set_seed(42)
        move_event_start = len(recorder.move_events)
        step_record_start = len(all_step_records)
        leg0 = probe.run_leg(
            scenario_path=scenario_path,
            map_id=int(map_id),
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
            map_id=int(map_id),
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

        map_events = recorder.move_events[move_event_start:]
        map_step_records = all_step_records[step_record_start:]
        all_move_events.extend(map_events)

        move_failures = probe.summarize_move_events(map_events)
        windows = probe.summarize_step_windows(map_step_records)
        merged_candidate_stats = _merge_policy_stats(leg0["red_policy_stats"], leg1["blue_policy_stats"])
        results[f"map_{map_id}"] = {
            "map_fit": map_fit(record_lookup, scenario_data, map_id),
            "merged_policy_stats": {
                "candidate": merged_candidate_stats,
            },
            "move_failures": move_failures,
            "windows": windows,
            "candidate_metrics": build_metrics(
                merged_candidate_stats,
                move_failures.get("candidate", {}),
                windows.get("candidate", {}),
                leg0,
                leg1,
            ),
            "legs": [
                {
                    "leg_index": int(leg0["leg_index"]),
                    "steps_executed": int(leg0["steps_executed"]),
                    "done": bool(leg0["done"]),
                    "red_role": str(leg0["red_role"]),
                    "blue_role": str(leg0["blue_role"]),
                    "red_score": float(leg0["red_score"]),
                    "blue_score": float(leg0["blue_score"]),
                },
                {
                    "leg_index": int(leg1["leg_index"]),
                    "steps_executed": int(leg1["steps_executed"]),
                    "done": bool(leg1["done"]),
                    "red_role": str(leg1["red_role"]),
                    "blue_role": str(leg1["blue_role"]),
                    "red_score": float(leg1["red_score"]),
                    "blue_score": float(leg1["blue_score"]),
                },
            ],
        }

    write_jsonl(RUN_DIR / "move_events.jsonl", all_move_events)
    write_jsonl(RUN_DIR / "step_records.jsonl", all_step_records)

    before_metrics = results[f"map_{BASELINE_MAP_ID}"]["candidate_metrics"]
    after_metrics = results[f"map_{resolved_map_id}"]["candidate_metrics"]
    comparison = {
        "before_map_212": {"map_id": BASELINE_MAP_ID, **before_metrics},
        "after_resolved": {"map_id": resolved_map_id, **after_metrics},
        "move_path_coverage_delta": float(after_metrics["move_path_coverage"] - before_metrics["move_path_coverage"]),
        "learned_path_coverage_delta": float(
            after_metrics["learned_path_coverage"] - before_metrics["learned_path_coverage"]
        ),
        "empty_route_with_no_reachable_city_delta": int(
            after_metrics["empty_route_with_no_reachable_city_total"]
            - before_metrics["empty_route_with_no_reachable_city_total"]
        ),
        "decisive_recovery": bool(
            after_metrics["move_path_coverage"] > before_metrics["move_path_coverage"]
            and after_metrics["learned_path_coverage"] > before_metrics["learned_path_coverage"]
            and after_metrics["empty_route_with_no_reachable_city_total"]
            < before_metrics["empty_route_with_no_reachable_city_total"]
        ),
    }

    summary = {
        "generated_at": now_utc(),
        "scenario": SCENARIO_NAME,
        "env_backend": env_backend,
        "max_steps": MAX_STEPS,
        "resolution": resolution,
        "maps": results,
        "comparison": comparison,
    }
    write_json(RUN_DIR / "summary.json", summary)
    (RUN_DIR / "memo.md").write_text(build_memo(summary) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
