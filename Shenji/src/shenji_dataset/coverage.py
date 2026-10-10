from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _ordered(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def build_coverage_report(
    raw_root: Path | str, targets: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Summarize pilot job state and intent coverage from episode directories."""

    root = Path(raw_root)
    state_counts: Counter[str] = Counter()
    by_intent: Counter[str] = Counter()
    by_label_kind: Counter[str] = Counter()
    by_evidence_grade: Counter[str] = Counter()
    by_segment_status: Counter[str] = Counter()
    by_outcome: Counter[str] = Counter()
    scenarios: Counter[str] = Counter()
    policies: Counter[str] = Counter()
    total_segments = 0
    completed_without_labels = 0
    errors: list[str] = []
    episode_dirs = (
        sorted(path for path in root.iterdir() if path.is_dir()) if root.is_dir() else []
    )
    for episode_dir in episode_dirs:
        status_path = episode_dir / "status.json"
        if not status_path.is_file():
            state_counts["missing_status"] += 1
            continue
        try:
            status = _read_json(status_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            state_counts["invalid_status"] += 1
            errors.append(f"{episode_dir.name}/status.json: {exc}")
            continue
        state = str(status.get("state", "unknown"))
        state_counts[state] += 1
        job_path = episode_dir / "job.json"
        if job_path.is_file():
            try:
                job = _read_json(job_path)
                scenarios[str(job.get("scenario_id", "unknown"))] += 1
                for key in ("red_policy_code", "blue_policy_code"):
                    if job.get(key) is not None:
                        policies[str(job[key])] += 1
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"{episode_dir.name}/job.json: {exc}")
        intent_path = episode_dir / "intent_plan.json"
        if not intent_path.is_file():
            if state == "completed":
                completed_without_labels += 1
            continue
        try:
            plan = _read_json(intent_path)
            segments = plan.get("segments", [])
            if not isinstance(segments, list):
                raise ValueError("segments must be a list")
            if state == "completed" and not segments:
                completed_without_labels += 1
            for segment in segments:
                if not isinstance(segment, dict):
                    errors.append(
                        f"{episode_dir.name}/intent_plan.json contains a non-object segment"
                    )
                    continue
                total_segments += 1
                by_intent[str(segment.get("primary_intent", "unknown"))] += 1
                by_label_kind[str(segment.get("label_kind", "unknown"))] += 1
                by_evidence_grade[str(segment.get("evidence_grade", "unknown"))] += 1
                by_segment_status[str(segment.get("status", "unknown"))] += 1
                by_outcome[str(segment.get("outcome", "unknown"))] += 1
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{episode_dir.name}/intent_plan.json: {exc}")

    report: dict[str, Any] = {
        "dataset": "Shenji",
        "raw_root": str(root),
        "jobs": {"total": len(episode_dirs), **_ordered(state_counts)},
        "segments": {
            "total": total_segments,
            "completed_episodes_without_labels": completed_without_labels,
            "by_intent": _ordered(by_intent),
            "by_label_kind": _ordered(by_label_kind),
            "by_evidence_grade": _ordered(by_evidence_grade),
            "by_status": _ordered(by_segment_status),
            "by_outcome": _ordered(by_outcome),
        },
        "scenarios": _ordered(scenarios),
        "policies": _ordered(policies),
        "errors": sorted(errors),
    }
    if targets is not None:
        intent_targets = targets.get("intent_targets", {})
        progress: dict[str, Any] = {}
        for intent in sorted(intent_targets):
            target = intent_targets[intent]
            actual = by_intent.get(intent, 0)
            pilot_target = int(target.get("pilot_min_segments", 0))
            progress[intent] = {
                "actual": actual,
                "pilot_target": pilot_target,
                "shortfall": max(0, pilot_target - actual),
                "complete": actual >= pilot_target,
            }
        report["target_progress"] = progress
    return report
