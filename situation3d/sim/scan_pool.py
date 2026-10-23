"""Deterministic fork-based chunks for large sensor scans.

Workers only compute per-observer ``seen`` sets from the forked world; the
parent merges them in uid order and advances the scan clocks itself, so the
result is identical to the serial loop.
"""

from __future__ import annotations

from typing import Dict, List, Set

_WORLD = None


def _scan_chunk(uids: List[int]) -> Dict[int, Set[int]]:
    from sim.context import WorldRuleContext
    world = _WORLD
    result = world._compute_table_seen(WorldRuleContext(world),
                                       only_uids=set(uids))
    return {uid: set(result.get(uid, ())) for uid in uids}
