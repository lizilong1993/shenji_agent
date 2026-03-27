from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def classify_event(run_dir: Path) -> Tuple[str, str]:
    rating_path = run_dir / "rating.json"
    exploitability_path = run_dir / "exploitability.json"
    promotion_path = run_dir / "promotion.json"
    if promotion_path.exists():
        promotion = _read_json(promotion_path)
        if promotion.get("accepted"):
            return "breakthrough", "候选模型通过正式晋升窗口。"
    if rating_path.exists():
        rating = _read_json(rating_path)
        overall = rating.get("overall", {})
        if float(overall.get("point_estimate", 0.0)) >= 0.60 and float(overall.get("ci95", [0.0])[0]) > 0.55:
            return "breakthrough", "总体 paired score 和置信下界同时达标。"
    if exploitability_path.exists():
        exploitability = _read_json(exploitability_path)
        for profile, values in exploitability.items():
            if float(values.get("win_rate_vs_candidate", 0.0)) > 0.52:
                return "stall", f"{profile} 对候选模型的 exploitability 超阈值。"
    return "none", "无关键突破或明确停滞。"


def main() -> None:
    parser = argparse.ArgumentParser(description="Report whether a Shenji run contains a critical event.")
    parser.add_argument("--experiments-root", default="experiments/shenji_agent")
    args = parser.parse_args()

    root = Path(args.experiments_root)
    run_dirs = sorted([path for path in root.iterdir() if path.is_dir()], reverse=True)
    if not run_dirs:
        raise SystemExit("No experiments found.")
    event, reason = classify_event(run_dirs[0])
    payload = {
        "run_dir": str(run_dirs[0]),
        "event": event,
        "reason": reason,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
