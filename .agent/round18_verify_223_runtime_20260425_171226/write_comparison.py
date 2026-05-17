#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent
BASELINE = RUN_DIR.parents[1] / ".agent" / "round18_probe_gpu_20260425_154341" / "probe_output" / "summary.json"
SUMMARY = RUN_DIR / "summary.json"
RESOLUTION = RUN_DIR / "resolution.json"


def metrics(payload: dict, *, map_id: int) -> dict:
    windows = payload["move_failures"]["candidate"]
    order = ["0-120", "121-280", "281-400"]
    return {
        "map_id": int(map_id),
        "candidate_move_path_coverage": float(payload["merged_policy_stats"]["candidate"]["move_path_coverage"]),
        "candidate_empty_route_total": int(sum(windows[label]["empty_route"] for label in order)),
        "candidate_empty_route_with_no_reachable_city_total": int(
            sum(windows[label]["empty_route_with_no_reachable_city"] for label in order)
        ),
        "candidate_empty_route_with_reachable_alternative_total": int(
            sum(windows[label]["empty_route_with_reachable_alternative"] for label in order)
        ),
        "candidate_windows": windows,
    }


def main() -> int:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))["scenarios"]["2230531221.json"]
    after = json.loads(SUMMARY.read_text(encoding="utf-8"))
    resolution = json.loads(RESOLUTION.read_text(encoding="utf-8"))

    comparison = {
        "generated_at": after["generated_at"],
        "scenario": "2230531221.json",
        "baseline_summary_path": str(BASELINE.resolve()),
        "after_summary_path": str(SUMMARY.resolve()),
        "baseline": metrics(baseline, map_id=baseline["map_id"]),
        "after": metrics(after, map_id=after["resolved_item"]["map_id"]),
    }
    comparison["delta"] = {
        "candidate_move_path_coverage": (
            comparison["after"]["candidate_move_path_coverage"] - comparison["baseline"]["candidate_move_path_coverage"]
        ),
        "candidate_empty_route_total": (
            comparison["after"]["candidate_empty_route_total"] - comparison["baseline"]["candidate_empty_route_total"]
        ),
        "candidate_empty_route_with_no_reachable_city_total": (
            comparison["after"]["candidate_empty_route_with_no_reachable_city_total"]
            - comparison["baseline"]["candidate_empty_route_with_no_reachable_city_total"]
        ),
    }

    (RUN_DIR / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (RUN_DIR / "memo.md").write_text(
        "\n".join(
            [
                "# Round 18 2230531221 Runtime Verification",
                "",
                f"- resolved map via real path: `{after['resolved_item']['map_id']}`",
                f"- heuristic map: `{resolution['heuristic_map_id']}`",
                (
                    f"- candidate move coverage before -> after: "
                    f"`{comparison['baseline']['candidate_move_path_coverage']:.6f}` -> "
                    f"`{comparison['after']['candidate_move_path_coverage']:.6f}`"
                ),
                (
                    f"- candidate empty_route_with_no_reachable_city before -> after: "
                    f"`{comparison['baseline']['candidate_empty_route_with_no_reachable_city_total']}` -> "
                    f"`{comparison['after']['candidate_empty_route_with_no_reachable_city_total']}`"
                ),
                (
                    f"- candidate empty_route total before -> after: "
                    f"`{comparison['baseline']['candidate_empty_route_total']}` -> "
                    f"`{comparison['after']['candidate_empty_route_total']}`"
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(comparison, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
