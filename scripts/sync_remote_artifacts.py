from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ALLOWED_FILES = {
    "summary.md",
    "metrics.json",
    "slice_report.json",
    "rating.json",
    "exploitability.json",
    "config.yaml",
    "promotion.json",
    "league_state.json",
}


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync selected Shenji artifacts from remote to local.")
    parser.add_argument("--remote", default="lizilong@146.56.220.99")
    parser.add_argument("--port", default="21427")
    parser.add_argument("--remote-root", default="/remote-home/lizilong/experiments/shenji_agent")
    parser.add_argument("--local-root", default="artifacts/remote_sync")
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    if args.run_id:
        run_id = args.run_id
    else:
        latest = run(["ssh", "-p", args.port, args.remote, f"ls -1dt {args.remote_root}/* | head -1"]).stdout.strip()
        if not latest:
            raise SystemExit("No remote experiment run found.")
        run_id = Path(latest).name

    local_dir = Path(args.local_root) / run_id
    local_dir.mkdir(parents=True, exist_ok=True)
    for file_name in sorted(ALLOWED_FILES):
        remote_path = f"{args.remote}:{args.remote_root}/{run_id}/{file_name}"
        subprocess.run(["scp", "-P", args.port, remote_path, str(local_dir / file_name)], check=False)


if __name__ == "__main__":
    main()
