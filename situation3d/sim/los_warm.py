"""Deterministic parallel warm-up of the static cell-pair LOS cache.

Only the pure terrain/buildings line-of-sight result is computed in worker
processes; detection/adjudication stays in the parent, so results are
identical to the serial path.  Used at load time for large land worlds.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

_ENV = None


def _warm(chunk: List[Tuple[Tuple[int, int], Tuple[int, int]]]):
    result: Dict[Tuple[Tuple[int, int], Tuple[int, int]], bool] = {}
    for ca, cb in chunk:
        result[(ca, cb)] = _ENV.cell_line_of_sight(ca, cb)
        result[(cb, ca)] = _ENV.cell_line_of_sight(cb, ca)
    return result
