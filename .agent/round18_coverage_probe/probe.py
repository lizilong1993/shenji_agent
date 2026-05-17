#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.common import save_json, set_seed
from training.selfplay import (
    _configure_torch_cpu_inference_threads,
    _load_agent_class_from_commit,
    _load_case_data,
    _merge_policy_stats,
    _train_env_runtime,
)

RED, BLUE = 0, 1
WINDOWS: List[Tuple[str, int, int]] = [
    ("0-120", 1, 120),
    ("121-280", 121, 280),
    ("281-400", 281, 400),
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def window_name(step: int) -> str | None:
    for label, begin, end in WINDOWS:
        if begin <= step <= end:
            return label
    return None


def zero_counts() -> Dict[str, Dict[str, int]]:
    return {
        "move": {"total": 0, "learned": 0},
        "shoot": {"total": 0, "learned": 0},
    }


def copy_counts(payload: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    counts = payload.get("counts", {}) if isinstance(payload, dict) else {}
    return {
        "move": {
            "total": int(((counts.get("move") or {}).get("total")) or 0),
            "learned": int(((counts.get("move") or {}).get("learned")) or 0),
        },
        "shoot": {
            "total": int(((counts.get("shoot") or {}).get("total")) or 0),
            "learned": int(((counts.get("shoot") or {}).get("learned")) or 0),
        },
    }


def diff_counts(before: Dict[str, Dict[str, int]], after: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    return {
        action: {
            metric: int(after[action][metric] - before[action][metric])
            for metric in ("total", "learned")
        }
        for action in ("move", "shoot")
    }


def add_counts(target: Dict[str, Dict[str, int]], delta: Dict[str, Dict[str, int]]) -> None:
    for action in ("move", "shoot"):
        for metric in ("total", "learned"):
            target[action][metric] += int(delta[action][metric])


def coverage(total: int, learned: int) -> float:
    if total <= 0:
        return 0.0
    return float(learned / total)


def summarize_counts(counts: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    move_total = int(counts["move"]["total"])
    move_learned = int(counts["move"]["learned"])
    shoot_total = int(counts["shoot"]["total"])
    shoot_learned = int(counts["shoot"]["learned"])
    return {
        "counts": counts,
        "learned_path_coverage": coverage(move_total + shoot_total, move_learned + shoot_learned),
        "move_path_coverage": coverage(move_total, move_learned),
        "shoot_path_coverage": coverage(shoot_total, shoot_learned),
    }


class ProbeRecorder:
    def __init__(self) -> None:
        self.move_events: List[Dict[str, Any]] = []

    def record_move_event(self, event: Dict[str, Any]) -> None:
        self.move_events.append(event)


def collect_reachable_city_routes(
    map_obj: Any,
    cur_hex: int,
    cities: Iterable[Dict[str, Any]],
    move_type: int,
) -> List[Tuple[Dict[str, Any], List[int]]]:
    reachable_city_routes: List[Tuple[Dict[str, Any], List[int]]] = []
    if map_obj is None:
        return reachable_city_routes

    for city in cities:
        city_hex = city.get("coord")
        if city_hex is None:
            continue
        route = map_obj.gen_move_route(int(cur_hex), int(city_hex), int(move_type))
        if route:
            reachable_city_routes.append((city, route))
    return reachable_city_routes


def run_self_check() -> None:
    class FakeMap:
        def __init__(self) -> None:
            self.calls: List[Tuple[int, int, int]] = []

        def gen_move_route(self, begin: int, end: int, mode: int) -> List[int]:
            self.calls.append((begin, end, mode))
            return {
                101: [11, 101],
                102: [],
                103: [13, 103],
            }.get(end, [])

    class FakePolicy:
        def __init__(self) -> None:
            self.calls: List[List[int]] = []

        def choose_city(
            self,
            bop: Dict[str, Any],
            cities: List[Dict[str, Any]],
            observation: Dict[str, Any],
            map_obj: Any,
        ) -> Tuple[Dict[str, Any] | None, Dict[str, float]]:
            self.calls.append([int(city["coord"]) for city in cities if city.get("coord") is not None])
            if not cities:
                return None, {}
            return cities[-1], {"combined_score": 1.0}

    fake_map = FakeMap()
    fake_policy = FakePolicy()
    bop = {"cur_hex": 1}
    cities = [{"coord": 101}, {"coord": 102}, {"coord": 103}, {"name": "missing-coord"}]
    move_type = 0

    reachable_city_routes = collect_reachable_city_routes(fake_map, int(bop["cur_hex"]), cities, move_type)
    assert [int(city["coord"]) for city, _ in reachable_city_routes] == [101, 103]
    selected_city, _ = fake_policy.choose_city(
        bop,
        [city for city, _ in reachable_city_routes],
        {"cities": cities},
        fake_map,
    )
    assert fake_policy.calls == [[101, 103]]
    assert selected_city is not None
    route_by_city = {
        int(city["coord"]): route for city, route in reachable_city_routes if city.get("coord") is not None
    }
    assert route_by_city[int(selected_city["coord"])] == [13, 103]
    print("self_check_ok", flush=True)


def patch_agent_class(agent_class: type[Any], recorder: ProbeRecorder) -> None:
    if getattr(agent_class, "_round18_coverage_probe_patched", False):
        return

    original_generate_action = agent_class._generate_action
    action_type_enum = original_generate_action.__globals__["ActionType"]

    def wrapped_generate_action(self: Any, obj_id: int, action_type: int, candidate: List[Dict[str, Any]]) -> Any:
        if action_type != action_type_enum.Move or not getattr(self, "use_ai_optimization", False):
            return original_generate_action(self, obj_id, action_type, candidate)

        bop = self.get_bop(obj_id)
        if not bop or bop.get("sub_type") == 3:
            return None

        cities = self.observation.get("cities", [])
        event: Dict[str, Any] = {
            "recorded_at": now_utc(),
            "scenario": getattr(self, "_probe_scenario_name", None),
            "map_id": getattr(self, "_probe_map_id", None),
            "leg_index": getattr(self, "_probe_leg_index", None),
            "step": getattr(self, "_probe_step", None),
            "role": getattr(self, "_probe_role", None),
            "color": getattr(self, "_probe_color", None),
            "seat": getattr(self, "seat", None),
            "policy_profile": getattr(self.policy, "policy_profile", None),
            "source_commit_sha": getattr(self, "_probe_source_commit_sha", None),
            "obj_id": int(obj_id),
            "cur_hex": int(bop.get("cur_hex", -1)),
            "sub_type": bop.get("sub_type"),
            "type": bop.get("type"),
            "move_state": bop.get("move_state"),
            "stage": ((self.observation.get("time") or {}).get("stage")),
            "candidate_payload_type": type(candidate).__name__,
            "candidate_payload_len": len(candidate) if isinstance(candidate, list) else None,
            "city_count": len(cities) if isinstance(cities, list) else 0,
        }

        if not cities:
            event["reason"] = "no_cities"
            recorder.record_move_event(event)
            return None

        move_type = self.get_move_type(bop)
        reachable_city_routes = collect_reachable_city_routes(self.map, int(bop["cur_hex"]), cities, int(move_type))
        reachable_city_coords = [
            int(city["coord"]) for city, _ in reachable_city_routes if city.get("coord") is not None
        ]

        event["move_type"] = int(move_type)
        event["reachable_city_count"] = len(reachable_city_coords)
        event["reachable_city_coords_sample"] = reachable_city_coords[:8]
        event["any_reachable"] = bool(reachable_city_coords)

        if not reachable_city_routes:
            event["candidate_id"] = None
            event["selected_city_reachable"] = False
            event["reason"] = "empty_route"
            recorder.record_move_event(event)
            self.policy.note_action("move", False, {"obj_id": obj_id, "reason": "empty_route"})
            return None

        selected_city, trace = self.policy.choose_city(
            bop,
            [city for city, _ in reachable_city_routes],
            self.observation,
            self.map,
        )
        if trace:
            for field in ("intent_bias", "situation_score", "combined_score"):
                if field in trace:
                    event[field] = float(trace[field])

        route_by_city = {
            int(city["coord"]): route for city, route in reachable_city_routes if city.get("coord") is not None
        }

        if selected_city is not None and self.map is not None:
            best_city = int(selected_city["coord"])
            route = route_by_city.get(best_city, [])
            event["candidate_id"] = best_city
            event["selected_city_reachable"] = bool(route)
            if route:
                event["reason"] = "learned_success"
                recorder.record_move_event(event)
                self.policy.note_action("move", True, trace)
                return {"actor": self.seat, "obj_id": obj_id, "type": action_type_enum.Move, "move_path": route}

            event["reason"] = "empty_route"
            recorder.record_move_event(event)
            self.policy.note_action("move", False, {"obj_id": obj_id, "candidate_id": best_city, "reason": "empty_route"})
            return None

        event["candidate_id"] = int(selected_city["coord"]) if selected_city is not None and selected_city.get("coord") is not None else None
        event["selected_city_reachable"] = False
        event["reason"] = "map_missing" if self.map is None else "no_selection"
        recorder.record_move_event(event)
        return None

    agent_class._generate_action = wrapped_generate_action
    agent_class._round18_coverage_probe_patched = True


def build_participant(label: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    artifacts = str(payload["artifacts"])
    profile = str(payload["profile"])
    source_commit_sha = str(payload["source_commit_sha"])
    return {
        "label": label,
        "artifacts": artifacts,
        "profile": profile,
        "user_name": "CurrentAI" if label == "candidate" else "IncumbentAI",
        "source_commit_sha": source_commit_sha,
        "code_origin": str(payload["code_origin"]),
        "agent_class": _load_agent_class_from_commit(source_commit_sha),
    }


def run_leg(
    *,
    scenario_path: Path,
    map_id: int,
    red_participant: Dict[str, Any],
    blue_participant: Dict[str, Any],
    red_role: str,
    blue_role: str,
    leg_index: int,
    max_steps: int,
    recorder: ProbeRecorder,
) -> Dict[str, Any]:
    env_class, env_backend = _train_env_runtime()
    env = env_class()
    shared = _load_case_data(scenario_path, map_id)
    player_info = [
        {"seat": 1, "faction": 0, "role": 1, "user_name": red_participant["user_name"], "user_id": 1},
        {"seat": 11, "faction": 1, "role": 1, "user_name": blue_participant["user_name"], "user_id": 11},
    ]
    env_info = {**shared, "player_info": player_info}
    state = env.setup(env_info)

    red = red_participant["agent_class"](log_level=logging.CRITICAL)
    blue = blue_participant["agent_class"](log_level=logging.CRITICAL)
    red.setup(
        {
            **shared,
            "seat": 1,
            "faction": 0,
            "role": 1,
            "user_name": red_participant["user_name"],
            "policy_profile": red_participant["profile"],
            "policy_artifacts_dir": red_participant["artifacts"],
            "state": state,
        }
    )
    blue.setup(
        {
            **shared,
            "seat": 11,
            "faction": 1,
            "role": 1,
            "user_name": blue_participant["user_name"],
            "policy_profile": blue_participant["profile"],
            "policy_artifacts_dir": blue_participant["artifacts"],
            "state": state,
        }
    )

    done = False
    steps = 0
    step_records: List[Dict[str, Any]] = []
    previous_counts = {"red": zero_counts(), "blue": zero_counts()}

    while not done and steps < max_steps:
        step_no = steps + 1
        red._probe_scenario_name = scenario_path.name
        red._probe_map_id = int(map_id)
        red._probe_leg_index = int(leg_index)
        red._probe_step = int(step_no)
        red._probe_role = red_role
        red._probe_color = "red"
        red._probe_source_commit_sha = red_participant["source_commit_sha"]

        blue._probe_scenario_name = scenario_path.name
        blue._probe_map_id = int(map_id)
        blue._probe_leg_index = int(leg_index)
        blue._probe_step = int(step_no)
        blue._probe_role = blue_role
        blue._probe_color = "blue"
        blue._probe_source_commit_sha = blue_participant["source_commit_sha"]

        red_actions = red.step(state[RED])
        blue_actions = blue.step(state[BLUE])

        red_counts = copy_counts(red.get_policy_stats())
        blue_counts = copy_counts(blue.get_policy_stats())
        red_delta = diff_counts(previous_counts["red"], red_counts)
        blue_delta = diff_counts(previous_counts["blue"], blue_counts)
        previous_counts["red"] = red_counts
        previous_counts["blue"] = blue_counts

        window = window_name(step_no)
        step_records.append(
            {
                "scenario": scenario_path.name,
                "map_id": int(map_id),
                "leg_index": int(leg_index),
                "step": int(step_no),
                "window": window,
                "env_backend": env_backend,
                "role": red_role,
                "color": "red",
                "counts_delta": red_delta,
                "counts_cumulative": red_counts,
                "actions_returned": len(red_actions),
            }
        )
        step_records.append(
            {
                "scenario": scenario_path.name,
                "map_id": int(map_id),
                "leg_index": int(leg_index),
                "step": int(step_no),
                "window": window,
                "env_backend": env_backend,
                "role": blue_role,
                "color": "blue",
                "counts_delta": blue_delta,
                "counts_cumulative": blue_counts,
                "actions_returned": len(blue_actions),
            }
        )

        state, done = env.step(red_actions + blue_actions)
        steps += 1

    red_score = float(state[RED].get("reward", 0.0))
    blue_score = float(state[BLUE].get("reward", 0.0))
    return {
        "scenario": scenario_path.name,
        "map_id": int(map_id),
        "leg_index": int(leg_index),
        "steps_executed": int(steps),
        "done": bool(done),
        "env_backend": env_backend,
        "red_role": red_role,
        "blue_role": blue_role,
        "red_score": red_score,
        "blue_score": blue_score,
        "red_policy_stats": red.get_policy_stats(),
        "blue_policy_stats": blue.get_policy_stats(),
        "step_records": step_records,
    }


def summarize_step_windows(step_records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    windows_by_role: Dict[str, Dict[str, Dict[str, Dict[str, int]]]] = defaultdict(
        lambda: {label: zero_counts() for label, _, _ in WINDOWS}
    )
    for record in step_records:
        role = str(record["role"])
        window = record.get("window")
        if not window:
            continue
        add_counts(windows_by_role[role][str(window)], record["counts_delta"])

    summary: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for role, window_counts in windows_by_role.items():
        summary[role] = {}
        for label, _, _ in WINDOWS:
            summary[role][label] = summarize_counts(window_counts[label])
    return summary


def summarize_move_events(move_events: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, int]]]:
    template = {
        "total_move_evals": 0,
        "learned_success": 0,
        "empty_route": 0,
        "no_selection": 0,
        "no_cities": 0,
        "map_missing": 0,
        "reachable_route_available": 0,
        "reachable_route_unavailable": 0,
        "empty_route_with_reachable_alternative": 0,
        "empty_route_with_no_reachable_city": 0,
    }
    summary: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: {label: dict(template) for label, _, _ in WINDOWS}
    )
    for event in move_events:
        role = str(event.get("role"))
        step = int(event.get("step") or 0)
        window = window_name(step)
        if not role or not window:
            continue
        bucket = summary[role][window]
        bucket["total_move_evals"] += 1
        reason = str(event.get("reason"))
        if reason in bucket:
            bucket[reason] += 1
        if bool(event.get("any_reachable")):
            bucket["reachable_route_available"] += 1
        else:
            bucket["reachable_route_unavailable"] += 1
        if reason == "empty_route":
            if bool(event.get("any_reachable")):
                bucket["empty_route_with_reachable_alternative"] += 1
            else:
                bucket["empty_route_with_no_reachable_city"] += 1
    return summary


def scenario_bucket(scenario_name: str) -> str:
    if scenario_name == "2230531221.json":
        return "complex"
    if scenario_name == "2120511121.json":
        return "medium"
    return "unknown"


def build_memo(summary: Dict[str, Any]) -> str:
    lines = [
        "# Round 18 Coverage Horizon Probe",
        "",
        f"- generated_at: `{summary['generated_at']}`",
        f"- participants_source: `{summary['participants_source']}`",
        f"- env_backend: `{summary['env_backend']}`",
        "",
    ]
    for scenario_name, payload in summary["scenarios"].items():
        candidate_windows = payload["windows"].get("candidate", {})
        candidate_failures = payload["move_failures"].get("candidate", {})
        early = candidate_windows.get("0-120", {})
        mid = candidate_windows.get("121-280", {})
        late = candidate_windows.get("281-400", {})
        early_fail = candidate_failures.get("0-120", {})
        mid_fail = candidate_failures.get("121-280", {})
        late_fail = candidate_failures.get("281-400", {})

        lines.extend(
            [
                f"## {scenario_name}",
                f"- map_id: `{payload['map_id']}` bucket: `{payload['bucket']}`",
                f"- candidate move coverage: early `{early.get('move_path_coverage', 0.0):.4f}`, mid `{mid.get('move_path_coverage', 0.0):.4f}`, late `{late.get('move_path_coverage', 0.0):.4f}`",
                f"- candidate empty_route split: early `{early_fail.get('empty_route', 0)}` total / `{early_fail.get('empty_route_with_reachable_alternative', 0)}` with reachable alternative / `{early_fail.get('empty_route_with_no_reachable_city', 0)}` with no reachable city",
                f"- candidate empty_route split: mid `{mid_fail.get('empty_route', 0)}` total / `{mid_fail.get('empty_route_with_reachable_alternative', 0)}` with reachable alternative / `{mid_fail.get('empty_route_with_no_reachable_city', 0)}` with no reachable city",
                f"- candidate empty_route split: late `{late_fail.get('empty_route', 0)}` total / `{late_fail.get('empty_route_with_reachable_alternative', 0)}` with reachable alternative / `{late_fail.get('empty_route_with_no_reachable_city', 0)}` with no reachable city",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted formal-compare-like coverage horizon probe.")
    parser.add_argument(
        "--participants",
        default=str(REPO_ROOT / "experiments" / "shenji_agent" / "league_20260424_170516" / "participants.json"),
        help="participants.json from the compare run to mirror",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / ".agent" / "round18_coverage_probe"),
        help="Directory for raw outputs and summary artifacts",
    )
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    if args.self_check:
        run_self_check()
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    participants_payload = json.loads(Path(args.participants).read_text(encoding="utf-8"))
    candidate = build_participant("candidate", participants_payload["candidate"])
    incumbent = build_participant("opponent", participants_payload["incumbent"])

    recorder = ProbeRecorder()
    patch_agent_class(candidate["agent_class"], recorder)
    patch_agent_class(incumbent["agent_class"], recorder)

    _configure_torch_cpu_inference_threads()
    set_seed(42)
    _, env_backend = _train_env_runtime()
    if env_backend == "mock_train_env.MockTrainEnv":
        raise SystemExit("Real probe blocked: resolved mock_train_env.MockTrainEnv instead of a real TrainEnv backend.")

    scenarios = [
        ("2230531221.json", 221),
        ("2120511121.json", 212),
    ]

    all_step_records: List[Dict[str, Any]] = []
    scenario_outputs: Dict[str, Any] = {}

    for scenario_name, map_id in scenarios:
        scenario_path = REPO_ROOT / "land_wargame_sdk" / "Data" / "Data" / "scenarios" / scenario_name
        leg0 = run_leg(
            scenario_path=scenario_path,
            map_id=map_id,
            red_participant=candidate,
            blue_participant=incumbent,
            red_role="candidate",
            blue_role="opponent",
            leg_index=0,
            max_steps=int(args.max_steps),
            recorder=recorder,
        )
        leg1 = run_leg(
            scenario_path=scenario_path,
            map_id=map_id,
            red_participant=incumbent,
            blue_participant=candidate,
            red_role="opponent",
            blue_role="candidate",
            leg_index=1,
            max_steps=int(args.max_steps),
            recorder=recorder,
        )
        all_step_records.extend(leg0["step_records"])
        all_step_records.extend(leg1["step_records"])

        merged_candidate_stats = _merge_policy_stats(leg0["red_policy_stats"], leg1["blue_policy_stats"])
        merged_opponent_stats = _merge_policy_stats(leg0["blue_policy_stats"], leg1["red_policy_stats"])
        scenario_events = [event for event in recorder.move_events if event.get("scenario") == scenario_name]
        scenario_steps = [record for record in all_step_records if record.get("scenario") == scenario_name]

        scenario_outputs[scenario_name] = {
            "scenario": scenario_name,
            "bucket": scenario_bucket(scenario_name),
            "map_id": int(map_id),
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
                "candidate": merged_candidate_stats,
                "opponent": merged_opponent_stats,
            },
            "windows": summarize_step_windows(scenario_steps),
            "move_failures": summarize_move_events(scenario_events),
        }

    summary = {
        "generated_at": now_utc(),
        "participants_source": str(Path(args.participants).resolve()),
        "env_backend": env_backend,
        "max_steps": int(args.max_steps),
        "scenarios": scenario_outputs,
    }
    memo_text = build_memo(summary)

    save_json(
        {
            "generated_at": summary["generated_at"],
            "participants_source": summary["participants_source"],
            "env_backend": env_backend,
            "max_steps": int(args.max_steps),
            "argv": sys.argv,
        },
        output_dir / "invocation.json",
    )
    save_json(summary, output_dir / "summary.json")
    (output_dir / "memo.md").write_text(memo_text, encoding="utf-8")

    with (output_dir / "move_events.jsonl").open("w", encoding="utf-8") as fh:
        for event in recorder.move_events:
            fh.write(json.dumps(event, ensure_ascii=True, sort_keys=True))
            fh.write("\n")

    with (output_dir / "step_records.jsonl").open("w", encoding="utf-8") as fh:
        for record in all_step_records:
            fh.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
            fh.write("\n")

    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
