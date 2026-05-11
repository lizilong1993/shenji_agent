#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)


def _write_json_atomic(path: Path, payload: dict) -> None:
    _write_text_atomic(path, json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n")


def _normalize_command(command: Sequence[str]) -> list[str]:
    parts = list(command)
    if parts and parts[0] == "--":
        parts = parts[1:]
    if not parts:
        raise SystemExit("compare command is required after '--'")
    return parts


def _resolve(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch training.selfplay in a detached, supervised session.")
    parser.add_argument("--cwd", default=os.getcwd(), help="Working directory for the detached compare process.")
    parser.add_argument("--pid-file", required=True, help="Path to write the actual compare PID.")
    parser.add_argument("--stdout-log", required=True, help="Path for compare stdout.")
    parser.add_argument("--stderr-log", required=True, help="Path for compare stderr.")
    parser.add_argument("--status-file", required=True, help="Path for compare exit status JSON.")
    parser.add_argument("--metadata-file", required=True, help="Path for compare metadata JSON.")
    parser.add_argument(
        "--launcher-log",
        help="Optional path for detached supervisor launcher logs. Defaults to '<stderr-log>.launcher.log'.",
    )
    parser.add_argument(
        "--pid-wait-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for the detached supervisor to write the compare PID.",
    )
    parser.add_argument("--supervise", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def _remove_stale_outputs(paths: Sequence[Path]) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _signal_name(signal_number: int | None) -> str | None:
    if signal_number is None:
        return None
    try:
        return signal.Signals(signal_number).name
    except ValueError:
        return None


def _build_status_payload(pid: int, command: Sequence[str], started_at: str, return_code: int) -> dict:
    signal_number = -return_code if return_code < 0 else None
    exit_code = return_code if return_code >= 0 else None
    return {
        "command": list(command),
        "ended_at": _utc_now(),
        "exit_code": exit_code,
        "pid": pid,
        "return_code": return_code,
        "signal_name": _signal_name(signal_number),
        "signal_number": signal_number,
        "started_at": started_at,
    }


def _supervise(args: argparse.Namespace) -> int:
    cwd = _resolve(args.cwd)
    pid_file = _resolve(args.pid_file)
    stdout_log = _resolve(args.stdout_log)
    stderr_log = _resolve(args.stderr_log)
    status_file = _resolve(args.status_file)
    metadata_file = _resolve(args.metadata_file)
    command = _normalize_command(args.command)

    for path in (pid_file, stdout_log, stderr_log, status_file, metadata_file):
        path.parent.mkdir(parents=True, exist_ok=True)

    started_at = _utc_now()
    child: subprocess.Popen[bytes] | None = None
    try:
        with stdout_log.open("ab", buffering=0) as stdout_handle, stderr_log.open("ab", buffering=0) as stderr_handle:
            child = subprocess.Popen(
                command,
                cwd=str(cwd),
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                start_new_session=True,
                close_fds=True,
            )
            metadata = {
                "command": command,
                "compare_pgid": os.getpgid(child.pid),
                "compare_pid": child.pid,
                "compare_sid": os.getsid(child.pid),
                "cwd": str(cwd),
                "launcher_pid": os.getpid(),
                "launcher_pgid": os.getpgid(0),
                "launcher_sid": os.getsid(0),
                "metadata_file": str(metadata_file),
                "pid_file": str(pid_file),
                "started_at": started_at,
                "status_file": str(status_file),
                "stderr_log": str(stderr_log),
                "stdin_target": "/dev/null",
                "stdout_log": str(stdout_log),
            }
            _write_text_atomic(pid_file, f"{child.pid}\n")
            _write_json_atomic(metadata_file, metadata)
            return_code = child.wait()
    except BaseException as exc:
        payload = {
            "command": command,
            "ended_at": _utc_now(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "pid": child.pid if child is not None else None,
            "return_code": None,
            "signal_name": None,
            "signal_number": None,
            "started_at": started_at,
        }
        _write_json_atomic(status_file, payload)
        raise

    _write_json_atomic(status_file, _build_status_payload(child.pid, command, started_at, return_code))
    return 0


def _launch(args: argparse.Namespace) -> int:
    command = _normalize_command(args.command)
    cwd = _resolve(args.cwd)
    pid_file = _resolve(args.pid_file)
    stdout_log = _resolve(args.stdout_log)
    stderr_log = _resolve(args.stderr_log)
    status_file = _resolve(args.status_file)
    metadata_file = _resolve(args.metadata_file)
    launcher_log = _resolve(args.launcher_log) if args.launcher_log else Path(f"{stderr_log}.launcher.log").resolve()
    for path in (launcher_log,):
        path.parent.mkdir(parents=True, exist_ok=True)

    _remove_stale_outputs((pid_file, status_file, metadata_file))

    supervise_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--supervise",
        "--cwd",
        str(cwd),
        "--pid-file",
        str(pid_file),
        "--stdout-log",
        str(stdout_log),
        "--stderr-log",
        str(stderr_log),
        "--status-file",
        str(status_file),
        "--metadata-file",
        str(metadata_file),
        "--",
        *command,
    ]
    if args.launcher_log:
        supervise_command[2:2] = ["--launcher-log", str(launcher_log)]

    with launcher_log.open("ab", buffering=0) as launcher_handle:
        supervisor = subprocess.Popen(
            supervise_command,
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=launcher_handle,
            stderr=launcher_handle,
            start_new_session=True,
            close_fds=True,
        )

    deadline = time.monotonic() + float(args.pid_wait_timeout)
    compare_pid: int | None = None
    while time.monotonic() < deadline:
        if pid_file.exists():
            compare_pid = int(pid_file.read_text(encoding="utf-8").strip())
            break
        if supervisor.poll() is not None:
            break
        time.sleep(0.1)

    if compare_pid is None:
        raise SystemExit(
            f"detached supervisor {supervisor.pid} did not publish compare pid within {args.pid_wait_timeout:.1f}s"
        )

    payload = {
        "command": command,
        "compare_pid": compare_pid,
        "cwd": str(cwd),
        "launcher_log": str(launcher_log),
        "metadata_file": str(metadata_file),
        "pid_file": str(pid_file),
        "status_file": str(status_file),
        "stderr_log": str(stderr_log),
        "stdout_log": str(stdout_log),
        "supervisor_pid": supervisor.pid,
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


def main() -> int:
    args = _parse_args()
    if args.supervise:
        return _supervise(args)
    return _launch(args)


if __name__ == "__main__":
    raise SystemExit(main())
