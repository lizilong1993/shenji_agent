from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.common import repo_root


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap Shenji on the remote container.")
    parser.add_argument("--remote", default="lizilong@146.56.220.99")
    parser.add_argument("--port", default="21427")
    parser.add_argument("--repo-url", default="https://github.com/lizilong1993/shenji_agent.git")
    parser.add_argument("--repo-path", default="/remote-home/lizilong/shenji_agent")
    parser.add_argument("--venv-path", default="/remote-home/lizilong/venvs/shenji")
    args = parser.parse_args()

    remote_setup = (
        f"set -e; "
        f"if [ -d {args.repo_path} ]; then git -C {args.repo_path} fetch origin && git -C {args.repo_path} reset --hard origin/main; "
        f"else git clone {args.repo_url} {args.repo_path}; fi; "
        f"python3 -m venv {args.venv_path}; "
        f". {args.venv_path}/bin/activate; "
        f"python -m pip install --upgrade pip; "
        f"python -m pip install -r {args.repo_path}/requirements.txt; "
        f"python -m pip install {args.repo_path}/land_wargame_sdk/land_wargame_train_env-4.1.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl; "
        f"mkdir -p /remote-home/lizilong/experiments/shenji_agent; "
        f"python -c 'import torch; print(torch.cuda.device_count())'"
    )
    run(["ssh", "-p", args.port, args.remote, remote_setup])


if __name__ == "__main__":
    main()
