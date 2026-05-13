from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]


def diagnose(artifact_dir: Path) -> Dict[str, Any]:
    manifest_path = artifact_dir / "model_manifest.json"
    payload: Dict[str, Any] = {
        "artifact_dir": str(artifact_dir),
        "manifest_exists": manifest_path.exists(),
        "tactical_model_loaded": False,
        "tactical_nonzero_count": 0,
        "blockers": [],
    }
    if not manifest_path.exists():
        payload["blockers"].append(f"missing manifest: {manifest_path}")
        return payload
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tactical_name = manifest.get("tactical_model") or "tactical_model.pt"
    tactical_path = artifact_dir / str(tactical_name)
    payload["tactical_model_path"] = str(tactical_path)
    payload["tactical_model_exists"] = tactical_path.exists()
    if not tactical_path.exists():
        payload["blockers"].append(f"missing tactical model file: {tactical_path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose tactical model availability for Round-005.")
    parser.add_argument("--artifact-dir", default="ai/weights")
    parser.add_argument("--out", default="reports/round005_source_recovery/tactical_diagnostic.json")
    args = parser.parse_args()
    report = diagnose((ROOT / args.artifact_dir).resolve())
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"tactical_nonzero_count={report['tactical_nonzero_count']}")
    return 2 if report["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
