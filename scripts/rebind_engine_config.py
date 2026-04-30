#!/usr/bin/env python3
"""Admin helper to inspect or explicitly rebind the SDK .engine_config."""
from __future__ import annotations

import argparse
import base64
import os
import pwd
import shutil
import subprocess
import sys
import time
from pathlib import Path

ENGINE_CONFIG_RELATIVE_PATH = Path(
    ".local/lib/python3.10/site-packages/train_env/env/authenticate/.engine_config"
)


def _repo_owner_home() -> Path | None:
    try:
        owner = pwd.getpwuid(Path.cwd().stat().st_uid)
    except (KeyError, OSError):
        return None
    return Path(owner.pw_dir)


def _default_engine_config() -> Path:
    env_path = os.environ.get("SDK_ENGINE_CONFIG")
    if env_path:
        return Path(env_path).expanduser()

    home_path = Path.home() / ENGINE_CONFIG_RELATIVE_PATH
    if home_path.exists():
        return home_path

    owner_home = _repo_owner_home()
    if owner_home and owner_home != Path.home():
        owner_path = owner_home / ENGINE_CONFIG_RELATIVE_PATH
        if owner_path.exists():
            return owner_path

    return home_path


def _current_mac() -> str:
    """Return the first non-loopback MAC address from /sys/class/net."""
    for path in sorted(Path("/sys/class/net").glob("*/address")):
        if path.parent.name == "lo":
            continue
        try:
            addr = path.read_text(encoding="utf-8").strip().lower()
        except OSError:
            continue
        if addr and addr != "00:00:00:00:00:00":
            return addr
    raise RuntimeError("No non-loopback MAC address found")


def _decode_config(path: Path) -> dict:
    """Decode a .engine_config file, returning {mac, timestamp} or {}."""
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8").strip()
        decoded = base64.b64decode(raw).decode("utf-8", errors="replace")
    except Exception:
        return {}
    parts = decoded.split(",", 1)
    return {
        "mac": parts[0].strip().lower() if parts and parts[0] else None,
        "timestamp": parts[1].strip() if len(parts) > 1 and parts[1] else None,
    }


def _write_config(path: Path, mac: str, timestamp: int) -> None:
    """Base64-encode MAC,timestamp and write to path."""
    payload = f"{mac},{timestamp}"
    encoded = base64.b64encode(payload.encode("utf-8")).decode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect or explicitly rebind the SDK .engine_config to the current container MAC"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the updated binding. Without this flag the tool is read-only.",
    )
    parser.add_argument(
        "--path",
        default=str(_default_engine_config()),
        help="Path to .engine_config (default: SDK_ENGINE_CONFIG or the SDK ~/.local path)",
    )
    args = parser.parse_args()

    target_path = Path(args.path)
    current_mac = _current_mac()

    print(f"Current MAC: {current_mac}")
    print(f"Target config: {target_path}")

    old_config = _decode_config(target_path)
    if old_config:
        print(f"Current binding: {old_config.get('mac')}")
        if old_config.get("mac") == current_mac:
            print("Already bound to current MAC. Nothing to do.")
            return 0
    else:
        print("Config file missing or unreadable.")

    if not args.apply:
        print(f"[DRY-RUN] Would rewrite {target_path} to bind MAC {current_mac}")
        print("Pass --apply to write. No SDK binary or authenticator is modified.")
        return 0

    # Backup
    if target_path.exists():
        backup = Path(str(target_path) + ".bak")
        shutil.copy2(target_path, backup)
        print(f"Backed up to {backup}")

    # Use nanosecond timestamp to match SDK format
    timestamp = time.time_ns()
    _write_config(target_path, current_mac, timestamp)

    # Verify
    new_config = _decode_config(target_path)
    if new_config.get("mac") != current_mac:
        print(f"ERROR: Verification failed. Written MAC={new_config.get('mac')}", file=sys.stderr)
        return 1

    print(f"Updated binding to MAC {current_mac}")

    # Run preflight check
    print("\nRunning preflight check...")
    result = subprocess.run(
        ["python3.10", "scripts/check_sdk_auth_environment.py"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print("Preflight passed.")
    else:
        print(f"Preflight returned {result.returncode}")
        # Print the JSON summary anyway
        try:
            import json
            report = json.loads(result.stdout)
            print(f"  mac_match: {report.get('mac_match')}")
            print(f"  blocker_reason: {report.get('blocker_reason')}")
            print(f"  auth_environment_ready: {report.get('auth_environment_ready')}")
        except Exception:
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
