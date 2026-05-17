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
REQUESTED_MAP_IDS = [212, 21, 53, 221]
MAX_STEPS = 400
WINDOW_LABELS = ["0-120", "121-280", "281-400"]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.common import save_json, set_seed
from training.maps import build_map_catalog, map_bucket_lookup, scenario_coordinate_overflow_counts
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
            "ai/policy.py",
            "ai/agent.py",
            "training/selfplay.py",
            ".agent/round18_coverage_probe/probe.py",
        ],
        cwd=str(REPO_ROOT),
        text=True,
    )
    return [line.rstrip() for line in output.splitlines() if line.strip()]


def load_probe_module() -> Any:
    spec = importlib.util.spec_from_file_location("round18_probe_212051_maps", PROBE_PATH)
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


def build_participant(label: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    from ai.agent import Agent

    return {
        "label": label,
        "artifacts": str(payload["artifacts"]),
        "profile": str(payload["profile"]),
        "user_name": "CurrentAI" if label == "candidate" else "IncumbentAI",
        "source_commit_sha": str(payload["source_commit_sha"]),
        "code_origin": str(payload["code_origin"]),
        "agent_class": Agent,
    }


def sum_windows(payload: Dict[str, Dict[str, int]], field: str) -> int:
    return int(sum(int((payload.get(label) or {}).get(field, 0)) for label in WINDOW_LABELS))


def scenario_path() -> Path:
    return REPO_ROOT / "land_wargame_sdk" / "Data" / "Data" / "scenarios" / SCENARIO_NAME


def load_scenario_data() -> Dict[str, Any]:
    return json.loads(scenario_path().read_text(encoding="utf-8"))


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


def update_status(**fields: Any) -> None:
    current = {}
    status_path = RUN_DIR / "status.json"
    if status_path.exists():
        current = json.loads(status_path.read_text(encoding="utf-8"))
    current.update(fields)
    current["updated_at"] = now_utc()
    write_json(status_path, current)


def build_participants_payload() -> Dict[str, Any]:
    head = git_head()
    return {
        "env_backend": _train_env_runtime()[1],
        "provenance_mode": "current_worktree_ai_modules",
        "candidate": {
            "artifacts": str(REPO_ROOT / "ai" / "weights"),
            "profile": "challenger",
            "source_commit_sha": head,
            "code_origin": f"worktree:{REPO_ROOT}@{head}",
        },
        "incumbent": {
            "artifacts": str(REPO_ROOT / "artifacts" / "best_agent" / "weights"),
            "profile": "incumbent",
            "source_commit_sha": head,
            "code_origin": f"worktree:{REPO_ROOT}@{head}",
        },
    }


def preflight(participants_payload: Dict[str, Any], map_fits: Dict[str, Any], resolved_item: Dict[str, Any]) -> Dict[str, Any]:
    import ai.agent as agent_module
    import ai.policy as policy_module

    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]

    _, env_backend = _train_env_runtime()
    cuda_available = bool(torch is not None and torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
    cuda_current_device = int(torch.cuda.current_device()) if cuda_available else None
    cuda_device_name = str(torch.cuda.get_device_name(cuda_current_device)) if cuda_available else None
    payload = {
        "generated_at": now_utc(),
        "repo_root": str(REPO_ROOT),
        "run_dir": str(RUN_DIR),
        "probe_path": str(PROBE_PATH),
        "scenario": SCENARIO_NAME,
        "current_resolved_map_id": int(resolved_item["map_id"]),
        "current_resolved_bucket": str(resolved_item["bucket"]),
        "requested_map_ids": list(REQUESTED_MAP_IDS),
        "map_fits": map_fits,
        "git_head": git_head(),
        "git_dirty_paths": git_dirty_paths(),
        "env_backend": env_backend,
        "requested_device_env": os.environ.get("SHENJI_POLICY_DEVICE", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "ai_policy_file": str(Path(policy_module.__file__).resolve()),
        "ai_agent_file": str(Path(agent_module.__file__).resolve()),
        "torch_available": torch is not None,
        "torch_cuda_available": cuda_available,
        "torch_cuda_device_count": cuda_device_count,
        "torch_cuda_current_device": cuda_current_device,
        "torch_cuda_device_name": cuda_device_name,
        "participants": {
            "candidate": policy_snapshot(
                "candidate",
                Path(participants_payload["candidate"]["artifacts"]),
                str(participants_payload["candidate"]["profile"]),
            ),
            "incumbent": policy_snapshot(
                "incumbent",
                Path(participants_payload["incumbent"]["artifacts"]),
                str(participants_payload["incumbent"]["profile"]),
            ),
        },
    }
    write_json(RUN_DIR / "preflight.json", payload)
    if env_backend == "mock_train_env.MockTrainEnv":
        raise SystemExit("Real probe blocked: resolved mock_train_env.MockTrainEnv instead of a real TrainEnv backend.")
    requested = (os.environ.get("SHENJI_POLICY_DEVICE") or "").strip().lower()
    if requested.startswith("cuda"):
        if not cuda_available:
            raise SystemExit("GPU preflight failed: torch.cuda.is_available() is False under requested CUDA path.")
        for label, snapshot in payload["participants"].items():
            policy_device = str(snapshot["policy_device"])
            if not policy_device.startswith("cuda"):
                raise SystemExit(f"GPU preflight failed: {label} policy configured device={policy_device!r}.")
            for key in ("intent_param_device", "situation_param_device"):
                value = snapshot.get(key)
                if value is not None and not str(value).startswith("cuda"):
                    raise SystemExit(f"GPU preflight failed: {label} {key}={value!r}.")
    return payload


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
    lines = [
        "# Round 18 Targeted Probe: 2120511121 Candidate Maps",
        "",
        f"- generated_at: `{summary['generated_at']}`",
        f"- scenario: `{summary['scenario']}`",
        f"- env_backend: `{summary['env_backend']}`",
        f"- requested_device_env: `{summary['requested_device_env']}`",
        f"- current_resolved_map_id: `{summary['current_resolved_map_id']}`",
        f"- tested_maps: `{', '.join(f'map_{map_id}' for map_id in summary['tested_map_ids'])}`",
        "",
    ]
    for map_key, payload in summary["maps"].items():
        metrics = payload["candidate_metrics"]
        fit = payload["map_fit"]
        lines.extend(
            [
                f"## {map_key}",
                f"- fit: `cities {fit['city_in_bounds']}/{fit['city_total']}` in bounds, `operators {fit['operator_in_bounds']}/{fit['operator_total']}` in bounds, bucket `{fit['bucket']}`",
                f"- candidate coverage: `move {metrics['move_path_coverage']:.4f}`, `learned {metrics['learned_path_coverage']:.4f}`, `shoot {metrics['shoot_path_coverage']:.4f}`",
                f"- candidate empty_route: total `{metrics['empty_route_total']}`, no reachable city `{metrics['empty_route_with_no_reachable_city_total']}`, reachable alternative `{metrics['empty_route_with_reachable_alternative_total']}`",
                f"- reachable-route split: available `{metrics['reachable_route_available_total']}`, unavailable `{metrics['reachable_route_unavailable_total']}`, learned_success `{metrics['learned_success_total']}`",
                f"- candidate windows move coverage: early `{metrics['window_move_path_coverage']['0-120']:.4f}`, mid `{metrics['window_move_path_coverage']['121-280']:.4f}`, late `{metrics['window_move_path_coverage']['281-400']:.4f}`",
                f"- paired_score: `{metrics['paired_score']:.4f}`",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    output_dir = RUN_DIR / "probe_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    invocation = {
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "run_dir": str(RUN_DIR),
        "output_dir": str(output_dir),
        "requested_device_env": os.environ.get("SHENJI_POLICY_DEVICE", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    write_json(RUN_DIR / "invocation.json", invocation)

    scenario_data = load_scenario_data()
    maps_root = REPO_ROOT / "land_wargame_sdk" / "Data" / "Data" / "maps"
    records = build_map_catalog(maps_root)
    bucket_lookup = map_bucket_lookup(records)
    resolved_candidates = _collect_scenario_candidates(scenario_path().parent, bucket_lookup, records)
    resolved_item = next(item for item in resolved_candidates if item["scenario"] == SCENARIO_NAME)
    record_lookup = {record.map_id: record for record in records}
    map_fits = {f"map_{map_id}": map_fit(record_lookup, scenario_data, map_id) for map_id in REQUESTED_MAP_IDS}

    participants_payload = build_participants_payload()
    write_json(RUN_DIR / "participants_worktree.json", participants_payload)
    preflight_payload = preflight(participants_payload, map_fits, resolved_item)

    probe = load_probe_module()
    recorder = probe.ProbeRecorder()

    from ai.agent import Agent

    probe.patch_agent_class(Agent, recorder)
    _configure_torch_cpu_inference_threads()
    set_seed(42)

    candidate = build_participant("candidate", participants_payload["candidate"])
    incumbent = build_participant("opponent", participants_payload["incumbent"])

    update_status(
        phase="running",
        scenario=SCENARIO_NAME,
        tested_map_ids=list(REQUESTED_MAP_IDS),
        completed_map_ids=[],
        current_map_id=None,
        current_leg_index=None,
        max_steps=MAX_STEPS,
    )

    all_step_records: List[Dict[str, Any]] = []
    results: Dict[str, Any] = {}
    completed_map_ids: List[int] = []

    for map_id in REQUESTED_MAP_IDS:
        update_status(current_map_id=int(map_id), current_leg_index=0, completed_map_ids=completed_map_ids)
        leg0 = probe.run_leg(
            scenario_path=scenario_path(),
            map_id=int(map_id),
            red_participant=candidate,
            blue_participant=incumbent,
            red_role="candidate",
            blue_role="opponent",
            leg_index=0,
            max_steps=MAX_STEPS,
            recorder=recorder,
        )
        update_status(current_map_id=int(map_id), current_leg_index=1, completed_map_ids=completed_map_ids)
        leg1 = probe.run_leg(
            scenario_path=scenario_path(),
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

        map_events = [
            event
            for event in recorder.move_events
            if event.get("scenario") == SCENARIO_NAME and int(event.get("map_id", -1)) == int(map_id)
        ]
        map_steps = [
            record
            for record in all_step_records
            if record.get("scenario") == SCENARIO_NAME and int(record.get("map_id", -1)) == int(map_id)
        ]

        merged_candidate_stats = _merge_policy_stats(leg0["red_policy_stats"], leg1["blue_policy_stats"])
        merged_opponent_stats = _merge_policy_stats(leg0["blue_policy_stats"], leg1["red_policy_stats"])
        windows = probe.summarize_step_windows(map_steps)
        move_failures = probe.summarize_move_events(map_events)
        candidate_failures = move_failures.get("candidate", {})
        candidate_windows = windows.get("candidate", {})
        candidate_metrics = build_candidate_metrics(
            merged_candidate_stats,
            candidate_failures,
            candidate_windows,
            leg0,
            leg1,
        )

        results[f"map_{map_id}"] = {
            "scenario": SCENARIO_NAME,
            "map_id": int(map_id),
            "map_fit": map_fits[f"map_{map_id}"],
            "legs": [
                {
                    "leg_index": int(leg0["leg_index"]),
                    "red_role": str(leg0["red_role"]),
                    "blue_role": str(leg0["blue_role"]),
                    "steps_executed": int(leg0["steps_executed"]),
                    "done": bool(leg0["done"]),
                    "red_score": float(leg0["red_score"]),
                    "blue_score": float(leg0["blue_score"]),
                    "red_policy_stats": leg0["red_policy_stats"],
                    "blue_policy_stats": leg0["blue_policy_stats"],
                },
                {
                    "leg_index": int(leg1["leg_index"]),
                    "red_role": str(leg1["red_role"]),
                    "blue_role": str(leg1["blue_role"]),
                    "steps_executed": int(leg1["steps_executed"]),
                    "done": bool(leg1["done"]),
                    "red_score": float(leg1["red_score"]),
                    "blue_score": float(leg1["blue_score"]),
                    "red_policy_stats": leg1["red_policy_stats"],
                    "blue_policy_stats": leg1["blue_policy_stats"],
                },
            ],
            "merged_policy_stats": {
                "candidate": merged_candidate_stats,
                "opponent": merged_opponent_stats,
            },
            "windows": windows,
            "move_failures": move_failures,
            "candidate_metrics": candidate_metrics,
        }
        completed_map_ids.append(int(map_id))
        update_status(current_map_id=int(map_id), current_leg_index=None, completed_map_ids=completed_map_ids)

    summary = {
        "generated_at": now_utc(),
        "scenario": SCENARIO_NAME,
        "env_backend": preflight_payload["env_backend"],
        "requested_device_env": str(preflight_payload["requested_device_env"]),
        "cuda_visible_devices": str(preflight_payload["cuda_visible_devices"]),
        "current_resolved_map_id": int(resolved_item["map_id"]),
        "current_resolved_bucket": str(resolved_item["bucket"]),
        "tested_map_ids": list(REQUESTED_MAP_IDS),
        "maps": results,
    }
    save_json(summary, output_dir / "summary.json")
    (output_dir / "memo.md").write_text(build_memo(summary), encoding="utf-8")
    append_jsonl(output_dir / "move_events.jsonl", recorder.move_events)
    append_jsonl(output_dir / "step_records.jsonl", all_step_records)

    update_status(
        phase="completed",
        current_map_id=None,
        current_leg_index=None,
        completed_map_ids=completed_map_ids,
        summary_path=str((output_dir / "summary.json").resolve()),
        memo_path=str((output_dir / "memo.md").resolve()),
        move_events_path=str((output_dir / "move_events.jsonl").resolve()),
        step_records_path=str((output_dir / "step_records.jsonl").resolve()),
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
