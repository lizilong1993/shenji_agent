from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.artifacts import missing_required_assets, naming_mismatches, resolution_to_dict, resolve_manifest_assets


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def manifest_missing_files(artifact_dir: Path) -> List[str]:
    manifest_path = artifact_dir / "model_manifest.json"
    if not manifest_path.exists():
        return [str(manifest_path)]
    manifest = read_json(manifest_path)
    if manifest.get("snapshot_role") == "bootstrap_rule_baseline":
        return []
    resolutions = resolve_manifest_assets(artifact_dir, manifest)
    missing = missing_required_assets(resolutions)
    return [str(item.path) for item in missing.values()]


def manifest_asset_report(artifact_dir: Path) -> Dict[str, Any]:
    manifest_path = artifact_dir / "model_manifest.json"
    if not manifest_path.exists():
        return {
            "manifest_path": str(manifest_path),
            "manifest_exists": False,
            "resolutions": {},
            "missing_required": [str(manifest_path)],
            "naming_mismatches": {},
        }
    manifest = read_json(manifest_path)
    models_required = manifest.get("snapshot_role") != "bootstrap_rule_baseline"
    resolutions = resolve_manifest_assets(artifact_dir, manifest)
    missing = missing_required_assets(resolutions) if models_required else {}
    mismatches = naming_mismatches(resolutions)
    return {
        "manifest_path": str(manifest_path),
        "manifest_exists": True,
        "models_required": models_required,
        "snapshot_role": manifest.get("snapshot_role"),
        "resolutions": {key: resolution_to_dict(value) for key, value in resolutions.items()},
        "missing_required": {key: resolution_to_dict(value) for key, value in missing.items()},
        "naming_mismatches": {key: resolution_to_dict(value) for key, value in mismatches.items()},
    }


def scenario_count() -> int:
    return len(list((ROOT / "land_wargame_sdk" / "Data" / "Data" / "scenarios").glob("*.json")))


def map_basic_count() -> int:
    return len(list((ROOT / "land_wargame_sdk" / "Data" / "Data" / "maps").glob("map_*/basic.json")))


def build_report(candidate: Path, incumbent: Path) -> Dict[str, Any]:
    missing_candidate = manifest_missing_files(candidate)
    missing_incumbent = manifest_missing_files(incumbent)
    scenarios = scenario_count()
    maps = map_basic_count()
    blockers = []
    if missing_candidate:
        blockers.append({"component": "candidate_artifact_dir", "missing": missing_candidate})
    if missing_incumbent:
        blockers.append({"component": "incumbent_artifact_dir", "missing": missing_incumbent})
    if scenarios <= 0:
        blockers.append({"component": "scenario_dataset", "missing": ["land_wargame_sdk/Data/Data/scenarios/*.json"]})
    if maps <= 0:
        blockers.append({"component": "map_dataset", "missing": ["land_wargame_sdk/Data/Data/maps/map_*/basic.json"]})
    return {
        "status": "blocked" if blockers else "ready",
        "candidate_artifact_dir": str(candidate),
        "incumbent_artifact_dir": str(incumbent),
        "candidate_assets": manifest_asset_report(candidate),
        "incumbent_assets": manifest_asset_report(incumbent),
        "scenario_count": scenarios,
        "map_basic_count": maps,
        "blockers": blockers,
        "would_run": (
            "python -m training.selfplay --config configs/selfplay_minimal.yaml "
            f"--candidate {candidate} --incumbent {incumbent}"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Round-005 fail-closed smoke pipeline preflight.")
    parser.add_argument("--out", default="reports/round005_smoke")
    parser.add_argument("--candidate", default="ai/weights")
    parser.add_argument("--incumbent", default="ai/weights_incumbent_rule")
    args = parser.parse_args()

    out_dir = ROOT / args.out
    candidate = (ROOT / args.candidate).resolve()
    incumbent = (ROOT / args.incumbent).resolve()
    report = build_report(candidate, incumbent)
    write_json(out_dir / "smoke_preflight.json", report)
    if report["status"] != "ready":
        (out_dir / "BLOCKED.md").write_text(
            "# Round-005 Smoke Pipeline Blocked\n\n"
            "The smoke pipeline failed closed before launching self-play because required "
            "external inputs are missing. See `smoke_preflight.json` for exact paths.\n",
            encoding="utf-8",
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
