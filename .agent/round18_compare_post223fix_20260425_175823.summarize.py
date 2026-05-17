#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean

REPO_ROOT = Path("/remote-home/lizilong/shenji_agent")
PREFIX = REPO_ROOT / ".agent" / "round18_compare_post223fix_20260425_175823"
STATUS_PATH = PREFIX.with_suffix(".status.json")
OUTPUT_PATH = PREFIX.with_suffix(".summary.json")
BASELINE_RUN = REPO_ROOT / "experiments" / "shenji_agent" / "league_20260425_040750"
SCENARIOS = ("2230531221.json", "2120511121.json")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_run_dir() -> Path:
    candidates = sorted((REPO_ROOT / "experiments" / "shenji_agent").glob("league_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        map_list = path / "map_list.json"
        if not map_list.exists():
            continue
        try:
            items = _load_json(map_list)["items"]
        except Exception:
            continue
        if any(item.get("scenario") == "2230531221.json" and item.get("map_id") == 53 for item in items):
            return path
    raise FileNotFoundError("Unable to locate post-223-fix compare run_dir.")


def _scenario_summary(run_dir: Path) -> dict:
    matches = _load_json(run_dir / "match_results.json")["matches"]
    out: dict[str, dict] = {}
    for scenario in SCENARIOS:
        picked = [m for m in matches if m.get("scenario") == scenario]
        if not picked:
            out[scenario] = {"count": 0}
            continue
        move_coverages = [float(m["candidate_policy_stats"]["move_path_coverage"]) for m in picked]
        learned_coverages = [float(m["candidate_policy_stats"]["learned_path_coverage"]) for m in picked]
        shoot_coverages = [float(m["candidate_policy_stats"]["shoot_path_coverage"]) for m in picked]
        paired_scores = [float(m["paired_score"]) for m in picked]
        map_ids = sorted({int(m["map_id"]) for m in picked})
        out[scenario] = {
            "count": len(picked),
            "map_ids": map_ids,
            "paired_score_mean": mean(paired_scores),
            "move_path_coverage_mean": mean(move_coverages),
            "learned_path_coverage_mean": mean(learned_coverages),
            "shoot_path_coverage_mean": mean(shoot_coverages),
            "first_trace_samples": picked[0]["candidate_policy_stats"].get("trace_samples", []),
        }
    return out


def main() -> int:
    while not STATUS_PATH.exists():
        time.sleep(30)
    status = _load_json(STATUS_PATH)
    run_dir = _find_run_dir()
    result = {
        "status": status,
        "run_dir": str(run_dir),
    }
    if status.get("exit_code") == 0:
        promotion = _load_json(run_dir / "promotion.json")
        rating = _load_json(run_dir / "rating.json")
        map_list = _load_json(run_dir / "map_list.json")
        result.update(
            {
                "promotion": promotion,
                "rating": rating,
                "scenarios": {
                    "baseline": _scenario_summary(BASELINE_RUN),
                    "post223fix": _scenario_summary(run_dir),
                },
                "map_list_targets": [
                    item
                    for item in map_list.get("items", [])
                    if item.get("scenario") in SCENARIOS
                ],
            }
        )
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
