#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ENGINE_CONFIG = Path("/usr/local/lib/python3.10/dist-packages/train_env/env/authenticate/.engine_config")
DEFAULT_EXPERIMENTS_ROOT = Path("experiments/shenji_agent")


def _json_load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _current_macs() -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for path in sorted(Path("/sys/class/net").glob("*/address")):
        interface = path.parent.name
        try:
            address = path.read_text(encoding="utf-8").strip().lower()
        except OSError:
            continue
        if not address or address == "00:00:00:00:00:00":
            continue
        results.append({"interface": interface, "mac": address})
    return results


def _decode_engine_config(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "bound_mac": None,
        "timestamp": None,
        "decode_error": None,
    }
    if not path.exists():
        return result
    try:
        raw = path.read_text(encoding="utf-8", errors="replace").strip()
        decoded = base64.b64decode(raw).decode("utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover - exact decoder exception is platform-dependent.
        result["decode_error"] = f"{type(exc).__name__}: {exc}"
        return result
    parts = decoded.split(",", 1)
    if parts and parts[0]:
        result["bound_mac"] = parts[0].strip().lower()
    if len(parts) > 1 and parts[1]:
        result["timestamp"] = parts[1].strip()
    return result


def _latest_run_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    candidates = [
        path
        for path in root.glob("league_*")
        if path.is_dir()
        and (
            (path / "auth_status_diagnostics.json").exists()
            or (path / "phase_action_diagnostics.json").exists()
            or (path / "promotion.json").exists()
        )
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _auth_status_summary(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "latest_status": None, "count": 0}
    file_path = path / "auth_status_diagnostics.json"
    payload = _json_load(file_path)
    records = payload.get("auth_statuses", [])
    if not isinstance(records, list):
        records = []
    latest = records[-1] if records and isinstance(records[-1], dict) else {}
    return {
        "path": str(file_path),
        "exists": file_path.exists(),
        "latest_status": latest.get("status"),
        "latest": latest,
        "count": len(records),
    }


def _phase_action_summary(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "summary": {}}
    file_path = path / "phase_action_diagnostics.json"
    payload = _json_load(file_path)
    summary = payload.get("summary", {})
    return {
        "path": str(file_path),
        "exists": file_path.exists(),
        "summary": summary if isinstance(summary, dict) else {},
    }


def _promotion_summary(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "accepted": None, "failures": []}
    file_path = path / "promotion.json"
    payload = _json_load(file_path)
    failures = payload.get("failures", [])
    return {
        "path": str(file_path),
        "exists": file_path.exists(),
        "accepted": payload.get("accepted"),
        "failures": failures if isinstance(failures, list) else [],
    }


def _git(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def build_report(engine_config: Path, experiments_root: Path) -> dict[str, Any]:
    macs = _current_macs()
    mac_values = {item["mac"] for item in macs}
    config = _decode_engine_config(engine_config)
    bound_mac = config.get("bound_mac")
    mac_match = bool(bound_mac and bound_mac in mac_values)
    latest_run = _latest_run_dir(experiments_root)
    auth = _auth_status_summary(latest_run)
    phase = _phase_action_summary(latest_run)
    promotion = _promotion_summary(latest_run)
    phase_summary = phase.get("summary", {})
    latest_auth = auth.get("latest_status")

    if not config["exists"]:
        blocker = "engine_config_missing"
        next_action = "Install or obtain the official SDK .engine_config for this runtime, then rerun preflight."
    elif not bound_mac:
        blocker = "engine_config_unreadable"
        next_action = "Verify the official SDK package/config; do not edit .engine_config manually."
    elif not mac_match:
        blocker = "mac_mismatch"
        next_action = (
            f"Run in an authorized environment whose non-loopback MAC is {bound_mac}, "
            "or request a regenerated .engine_config from the SDK provider for the actual runtime MAC."
        )
    elif str(latest_auth or "").startswith("failed"):
        blocker = str(latest_auth)
        next_action = "Rerun self-play after fixing SDK auth; if failure persists, contact the SDK/platform owner."
    elif phase_summary.get("battle_phase_entered") is False:
        blocker = "battle_phase_not_entered"
        next_action = "Auth appears locally plausible; rerun battle smoke and inspect phase_action_diagnostics.json."
    else:
        blocker = None
        next_action = "Run scripts/continue_after_auth_fixed.sh to continue objective diagnostics and promotion."

    ready = blocker is None or (mac_match and latest_auth not in {"failed_mac_mismatch", "engine_config_missing"})
    commands = [
        "python3.10 scripts/check_sdk_auth_environment.py --output experiments/shenji_agent/sdk_auth_environment_check.json",
        "python3.10 -m training.selfplay --config configs/selfplay_smoke_battle_phase.yaml --candidate ai/weights --incumbent ai/weights_incumbent_rule",
        "bash scripts/continue_after_auth_fixed.sh",
    ]
    if bound_mac:
        commands.insert(1, "bash scripts/admin_run_authorized_selfplay.sh")

    status_short = _git(["status", "--short"]) or ""
    status_lines = status_short.splitlines()
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cwd": str(Path.cwd()),
        "hostname": socket.gethostname(),
        "git": {
            "head": _git(["rev-parse", "HEAD"]),
            "status_short": "\n".join(status_lines[:80]),
            "status_short_truncated": len(status_lines) > 80,
            "status_short_count": len(status_lines),
        },
        "current_macs": macs,
        "engine_config": config,
        "mac_match": mac_match,
        "latest_run_dir": str(latest_run) if latest_run else None,
        "latest_auth_status_diagnostics": auth,
        "latest_phase_action_diagnostics": phase,
        "latest_promotion": promotion,
        "auth_environment_ready": bool(ready),
        "blocker_reason": blocker,
        "next_action": next_action,
        "run_after_fix_commands": commands,
        "safety": {
            "mutates_system": False,
            "patches_sdk": False,
            "fakes_auth_or_battle_evidence": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check SDK auth/MAC readiness without host privileges.")
    parser.add_argument("--engine-config", default=os.environ.get("SDK_ENGINE_CONFIG", str(DEFAULT_ENGINE_CONFIG)))
    parser.add_argument("--experiments-root", default=str(DEFAULT_EXPERIMENTS_ROOT))
    parser.add_argument("--output", help="Optional JSON output path.")
    args = parser.parse_args()

    report = build_report(Path(args.engine_config), Path(args.experiments_root))
    text = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
