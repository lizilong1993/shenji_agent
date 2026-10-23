#!/usr/bin/env python3
"""Report the public 50%-detection calibration for each sensor kind."""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from sim import calibration  # noqa: E402


def main() -> int:
    print(json.dumps(calibration.report(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
