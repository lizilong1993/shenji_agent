from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


OBJECTIVE_FIELDS = [
    "match_id",
    "turn",
    "side",
    "operator_id",
    "operator_type",
    "current_hex",
    "action_type",
    "target_hex",
    "target_city_id",
    "target_owner_before",
    "target_owner_after",
    "city_value",
    "distance_before",
    "distance_after",
    "distance_delta",
    "occupy_available",
    "occupy_selected",
    "occupy_success",
    "damage_dealt",
    "damage_taken",
    "kill_dealt",
    "kill_taken",
    "score_before",
    "score_after",
    "score_delta",
    "final_score_delta",
    "win_loss_result",
    "reward_proxy",
    "label_kind",
    "label_confidence",
    "source_file",
]


def _hex_distance(left: Optional[int], right: Optional[int]) -> Optional[int]:
    if left is None or right is None:
        return None
    row1, col1 = divmod(int(left), 100)
    row2, col2 = divmod(int(right), 100)
    q1 = col1 - (row1 - (row1 & 1)) // 2
    q2 = col2 - (row2 - (row2 & 1)) // 2
    r1 = row1
    r2 = row2
    s1 = -q1 - r1
    s2 = -q2 - r2
    return int((abs(q1 - q2) + abs(r1 - r2) + abs(s1 - s2)) // 2)


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def _empty_label(*, source_file: str) -> Dict[str, Any]:
    label = {field: None for field in OBJECTIVE_FIELDS}
    label["source_file"] = source_file
    return label


def _city_lookup(scenario_path: Path) -> Dict[int, Dict[str, Any]]:
    if not scenario_path.exists():
        return {}
    data = json.loads(scenario_path.read_text(encoding="utf-8"))
    lookup: Dict[int, Dict[str, Any]] = {}
    for city in data.get("cities", []):
        coord = _as_int(city.get("coord"))
        if coord is not None:
            lookup[coord] = city
    return lookup


def compute_reward_proxy(record: Dict[str, Any]) -> Tuple[float, str, str]:
    score_delta = _as_float(record.get("score_delta"))
    if score_delta is None:
        before = _as_float(record.get("score_before"))
        after = _as_float(record.get("score_after"))
        if before is not None and after is not None:
            score_delta = after - before
    if score_delta is not None:
        return float(score_delta), "score_delta", "high"

    owner_before = record.get("target_owner_before")
    owner_after = record.get("target_owner_after")
    side = record.get("side")
    city_value = _as_float(record.get("city_value")) or 0.0
    if owner_before is not None and owner_after is not None and side is not None:
        if int(owner_before) != int(side) and int(owner_after) == int(side):
            return max(1.0, city_value), "city_capture_delta", "high"
        if int(owner_before) == int(side) and int(owner_after) != int(side):
            return -max(1.0, city_value), "city_capture_delta", "high"

    if record.get("occupy_success") is True:
        return max(1.0, city_value), "occupy_success", "high"

    damage_dealt = _as_float(record.get("damage_dealt")) or 0.0
    damage_taken = _as_float(record.get("damage_taken")) or 0.0
    kill_dealt = _as_float(record.get("kill_dealt")) or 0.0
    kill_taken = _as_float(record.get("kill_taken")) or 0.0
    damage_value = damage_dealt - damage_taken + kill_dealt * 25.0 - kill_taken * 25.0
    if damage_value != 0.0:
        return float(damage_value), "damage_kill_delta", "high"

    distance_delta = _as_float(record.get("distance_delta"))
    if distance_delta is None:
        before = _as_float(record.get("distance_before"))
        after = _as_float(record.get("distance_after"))
        if before is not None and after is not None:
            distance_delta = before - after
    if distance_delta is not None:
        target_value = max(1.0, city_value)
        return float(distance_delta) * min(target_value / 20.0, 5.0), "distance_to_objective_delta", "medium"

    final_delta = _as_float(record.get("final_score_delta"))
    if final_delta is not None:
        return float(final_delta), "final_outcome_credit", "low"

    return 0.0, "heuristic_no_observable_progress", "low"


def normalize_label(record: Dict[str, Any], *, source_file: str) -> Dict[str, Any]:
    label = _empty_label(source_file=source_file)
    label.update({key: record.get(key) for key in OBJECTIVE_FIELDS if key in record})
    if label.get("distance_delta") is None:
        before = _as_float(label.get("distance_before"))
        after = _as_float(label.get("distance_after"))
        if before is not None and after is not None:
            label["distance_delta"] = before - after
    reward, kind, confidence = compute_reward_proxy(label)
    label["reward_proxy"] = reward
    label["label_kind"] = kind
    label["label_confidence"] = confidence
    label["source_file"] = source_file
    return label


def labels_from_action_outcomes(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("actions", payload if isinstance(payload, list) else [])
    if not isinstance(records, list):
        return []
    return [normalize_label(record, source_file=str(path)) for record in records if isinstance(record, dict)]


def labels_from_events(path: Path) -> List[Dict[str, Any]]:
    """从 `events.jsonl` 的**引擎裁决**与**夺控**事件导出高置信 objective 标签。

    ## 为什么加这个函数（I06 步骤 2）

第 50 轮查清：`ADJUDICATION` 的 payload 已携带
    `damage` / `target_obj_id` / `attack_color` / `att_obj_blood` / `cur_step`，
    但本模块**从不读它** —— 于是 `high_confidence` 恒为 0，判据 1 无法达成。
    根因不是"引擎给不出数据"，而是"标签器没读引擎已经给的数据"。

    ## 字段口径（**全部由事件本身推导，不靠外部假设**）

    每个 `ADJUDICATION` 描述"某方攻击某目标"这一事实，故对**两个阵营各产生一条标签**：

      - 攻方：`damage_dealt = -damage`（`damage` 是受击方的血量变化，负数为掉血）
      - 受方：`damage_taken = -damage`
      - 击杀：受方 `att_obj_blood + damage <= 0` 时记 1 次（**由同一 payload 推导**）

    同一步内多条裁决会**按阵营累加**，得到该步每方的净伤害/击杀，
    再交给既有的 `compute_reward_proxy()` —— 其 `damage_kill_delta` 分支返回 `high`。

    `SEIZE` 事件记 `occupy_success = True`（攻方）与 `occupy_loss = True`（另一方），
    交给既有 `occupy_success` 分支，同样返回 `high`。

    ## 不做什么

    - **不新增置信度档位**，不改既有分支 —— 那等于把标准往实现上凑；
    - 字段缺失时**留 null**，不编造（沿用本模块既有约定）。
    """
    labels: List[Dict[str, Any]] = []
    episode_id = path.parent.name

    # step -> side -> 聚合
    agg: Dict[Tuple[int, int], Dict[str, float]] = {}
    captures: Dict[Tuple[int, int], bool] = {}

    def bucket(step: int, side: int) -> Dict[str, float]:
        return agg.setdefault((step, side), {"damage_dealt": 0.0, "damage_taken": 0.0,
                                              "kill_dealt": 0.0, "kill_taken": 0.0})

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except (TypeError, ValueError):
                continue
            kind = event.get("event_type")
            step = _as_int(event.get("step"))
            payload_raw = event.get("payload_json")
            try:
                payload = json.loads(payload_raw) if payload_raw else {}
            except (TypeError, ValueError):
                payload = {}
            if not isinstance(payload, dict):
                payload = {}

            if kind == "ADJUDICATION":
                if step is None:
                    step = _as_int(payload.get("cur_step"))
                if step is None:
                    continue
                damage = _as_float(payload.get("damage"))
                if damage is None:
                    continue
                attacker_side = _as_int(payload.get("attack_color"))
                target_side = _as_int(payload.get("target_color"))
                if attacker_side is None or target_side is None:
                    continue
                target_blood = _as_float(payload.get("att_obj_blood"))
                # damage 是受击方血量变化；掉血为负 → 施加的伤害 = -damage
                applied = -float(damage)
                killed = bool(target_blood is not None and target_blood + damage <= 0)
                atk = bucket(step, attacker_side)
                atk["damage_dealt"] += applied
                atk["kill_dealt"] += 1.0 if killed else 0.0
                tgt = bucket(step, target_side)
                tgt["damage_taken"] += applied
                tgt["kill_taken"] += 1.0 if killed else 0.0

            elif kind == "SEIZE":
                if step is None:
                    step = _as_int(payload.get("cur_step"))
                actor = _as_int(payload.get("actor"))
                if step is None or actor is None:
                    continue
                captures[(step, actor)] = True

    for (step, side), values in sorted(agg.items()):
        record: Dict[str, Any] = {
            "match_id": episode_id,
            "cur_step": step,
            "side": side,
            "damage_dealt": values["damage_dealt"],
            "damage_taken": values["damage_taken"],
            "kill_dealt": values["kill_dealt"],
            "kill_taken": values["kill_taken"],
            "label_kind_hint": "engine_adjudication",
        }
        labels.append(normalize_label(record, source_file=str(path)))

    for (step, side), _ in sorted(captures.items()):
        record = {
            "match_id": episode_id,
            "cur_step": step,
            "side": side,
            "occupy_success": True,
            "label_kind_hint": "engine_seize",
        }
        labels.append(normalize_label(record, source_file=str(path)))

    return labels


def labels_from_objective_progress(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels: List[Dict[str, Any]] = []
    for record in payload.get("turns", payload if isinstance(payload, list) else []):
        if not isinstance(record, dict):
            continue
        for side_key in ("red", "blue", "candidate", "opponent"):
            side_payload = record.get(side_key)
            if not isinstance(side_payload, dict):
                continue
            labels.append(
                normalize_label(
                    {
                        "match_id": record.get("match_id"),
                        "turn": record.get("turn"),
                        "side": side_payload.get("side", side_key),
                        "score_before": side_payload.get("score_before"),
                        "score_after": side_payload.get("score_after"),
                        "score_delta": side_payload.get("score_delta"),
                        "damage_dealt": side_payload.get("damage_dealt"),
                        "damage_taken": side_payload.get("damage_taken"),
                        "kill_dealt": side_payload.get("kill_dealt"),
                        "kill_taken": side_payload.get("kill_taken"),
                    },
                    source_file=str(path),
                )
            )
    return labels


def _final_score_delta(match: Dict[str, Any], side: str) -> Optional[float]:
    first = match.get("first_leg", {})
    second = match.get("second_leg", {})
    if not isinstance(first, dict) or not isinstance(second, dict):
        return None
    cand = (_as_float(first.get("candidate")) or 0.0) + (_as_float(second.get("candidate")) or 0.0)
    opp = (_as_float(first.get("opponent")) or 0.0) + (_as_float(second.get("opponent")) or 0.0)
    return cand - opp if side == "candidate" else opp - cand


def labels_from_match_results(path: Path, *, scenario_root: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels: List[Dict[str, Any]] = []
    for match_index, match in enumerate(payload.get("matches", [])):
        if not isinstance(match, dict):
            continue
        scenario_path = scenario_root / str(match.get("scenario", ""))
        cities = _city_lookup(scenario_path)
        match_id = f"{path.parent.name}:{match_index}:{match.get('scenario')}"
        for side, stats_key in (("candidate", "candidate_policy_stats"), ("opponent", "opponent_policy_stats")):
            stats = match.get(stats_key, {})
            if not isinstance(stats, dict):
                continue
            final_delta = _final_score_delta(match, side)
            for route in stats.get("move_route_samples", []):
                if not isinstance(route, dict):
                    continue
                target_city = _as_int(route.get("target_city_id") or route.get("target_hex"))
                city = cities.get(target_city or -1, {})
                current_hex = _as_int(route.get("current_hex"))
                target_hex = _as_int(route.get("target_hex") or target_city)
                distance_before = _hex_distance(current_hex, target_city)
                route_len = _as_int(route.get("route_len"))
                distance_after = None
                if distance_before is not None and route_len is not None:
                    distance_after = max(0, distance_before - max(1, min(route_len, distance_before)))
                labels.append(
                    normalize_label(
                        {
                            "match_id": match_id,
                            "turn": None,
                            "side": side,
                            "operator_id": route.get("operator_id"),
                            "operator_type": route.get("operator_type"),
                            "current_hex": current_hex,
                            "action_type": "Move",
                            "target_hex": target_hex,
                            "target_city_id": target_city,
                            "target_owner_before": city.get("flag"),
                            "target_owner_after": None,
                            "city_value": city.get("value"),
                            "distance_before": distance_before,
                            "distance_after": distance_after,
                            "occupy_available": None,
                            "occupy_selected": False,
                            "occupy_success": None,
                            "final_score_delta": final_delta,
                        },
                        source_file=str(path),
                    )
                )
            for trace in stats.get("trace_samples", []):
                if not isinstance(trace, dict) or str(trace.get("action", "")).lower() != "shoot":
                    continue
                labels.append(
                    normalize_label(
                        {
                            "match_id": match_id,
                            "side": side,
                            "operator_id": trace.get("obj_id"),
                            "action_type": "Shoot",
                            "target_hex": trace.get("candidate_id"),
                            "final_score_delta": final_delta,
                        },
                        source_file=str(path),
                    )
                )
    return labels


def discover_source_files(input_paths: Iterable[Path]) -> List[Path]:
    files: List[Path] = []
    for path in input_paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("match_results.json")))
            files.extend(sorted(path.rglob("action_outcome_diagnostics.json")))
            files.extend(sorted(path.rglob("objective_progress_diagnostics.json")))
            # I06 步骤 2：引擎裁决事件（ADJUDICATION/SEIZE）是高置信标签的真实来源
            files.extend(sorted(path.rglob("events.jsonl")))
    return sorted(set(files))


def build_labels(input_paths: Iterable[Path], *, scenario_root: Path) -> List[Dict[str, Any]]:
    labels: List[Dict[str, Any]] = []
    for path in discover_source_files(input_paths):
        if path.name == "match_results.json":
            labels.extend(labels_from_match_results(path, scenario_root=scenario_root))
        elif path.name == "action_outcome_diagnostics.json":
            labels.extend(labels_from_action_outcomes(path))
        elif path.name == "objective_progress_diagnostics.json":
            labels.extend(labels_from_objective_progress(path))
        elif path.name == "events.jsonl":
            labels.extend(labels_from_events(path))
    return labels


def write_outputs(labels: List[Dict[str, Any]], output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    label_path = output_dir / "objective_labels.jsonl"
    with label_path.open("w", encoding="utf-8") as fh:
        for label in labels:
            fh.write(json.dumps(label, ensure_ascii=False, sort_keys=True) + "\n")
    confidence_counts = Counter(str(label.get("label_confidence")) for label in labels)
    kind_counts = Counter(str(label.get("label_kind")) for label in labels)
    action_counts = Counter(str(label.get("action_type")) for label in labels)
    reward_values = [float(label.get("reward_proxy", 0.0)) for label in labels]
    summary = {
        "label_count": len(labels),
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "label_kind_counts": dict(sorted(kind_counts.items())),
        "action_type_counts": dict(sorted(action_counts.items())),
        "positive_reward_count": int(sum(1 for value in reward_values if value > 0)),
        "negative_reward_count": int(sum(1 for value in reward_values if value < 0)),
        "zero_reward_count": int(sum(1 for value in reward_values if value == 0)),
        "reward_proxy_mean": float(sum(reward_values) / len(reward_values)) if reward_values else 0.0,
        "outputs": {
            "objective_labels_jsonl": str(label_path),
            "summary_json": str(output_dir / "summary.json"),
            "readme": str(output_dir / "README.md"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Objective Labels",
                "",
                "These labels are derived from available self-play diagnostics and scenario/map metadata.",
                "Missing fields are emitted as null rather than fabricated.",
                "Confidence is high for actual score/capture/damage events, medium for distance progress, and low for heuristic/final-outcome-only labels.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build objective-aware labels from self-play artifacts.")
    parser.add_argument("--input", action="append", required=True, help="Artifact file or directory. Can be repeated.")
    parser.add_argument("--scenario-root", default="land_wargame_sdk/Data/Data/scenarios")
    parser.add_argument("--output-dir", default="experiments/shenji_agent/objective_labels/latest")
    args = parser.parse_args()

    labels = build_labels([Path(item) for item in args.input], scenario_root=Path(args.scenario_root))
    summary = write_outputs(labels, Path(args.output_dir))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
