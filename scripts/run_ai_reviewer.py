"""
Paper-specific AI reviewer has moved to paper/scripts/.

Run from the paper/ directory:
    cd paper && python scripts/run_ai_reviewer.py --config configs/ai_reviewer.yaml [--llm]

This file is kept as a forwarding wrapper for backward compatibility.
"""

import sys
import subprocess
from pathlib import Path


def main() -> int:
    paper_script = Path(__file__).resolve().parent.parent / "paper" / "scripts" / "run_ai_reviewer.py"
    if not paper_script.exists():
        print(f"ERROR: paper reviewer not found at {paper_script}", file=sys.stderr)
        print("Run from paper/ directory: cd paper && python scripts/run_ai_reviewer.py", file=sys.stderr)
        return 1

    cmd = [sys.executable, str(paper_script)] + sys.argv[1:]
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
