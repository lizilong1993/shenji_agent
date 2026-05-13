from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Round-005 route diagnostic availability.")
    parser.add_argument("--league-dir", default="experiments/shenji_agent/league_20260501_170204")
    parser.add_argument("--out", default="reports/round005_source_recovery/route_diagnostic.json")
    args = parser.parse_args()
    league_dir = ROOT / args.league_dir
    rating_path = league_dir / "rating.json"
    payload: Dict[str, Any] = {
        "league_dir": str(league_dir),
        "rating_exists": rating_path.exists(),
        "total_move_attempts": 0,
        "empty_route_count": 0,
        "empty_route_rate": None,
        "blockers": [],
    }
    if rating_path.exists():
        rating = json.loads(rating_path.read_text(encoding="utf-8"))
        route = rating.get("policy", {}).get("move_route_aggregate", {})
        total = int(route.get("total_attempts", 0) or 0)
        empty = int(route.get("empty_route_count", 0) or 0)
        payload.update(
            {
                "total_move_attempts": total,
                "empty_route_count": empty,
                "empty_route_rate": float(empty / total) if total else None,
                "route_failure_reasons": route.get("route_failure_reasons", {}),
                "by_move_type": route.get("by_move_type", {}),
            }
        )
    else:
        payload["blockers"].append(f"missing rating sidecar: {rating_path}")
    if payload["total_move_attempts"] <= 0:
        payload["blockers"].append("no executable route-attempt records available in selected sidecar")
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return 2 if payload["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
