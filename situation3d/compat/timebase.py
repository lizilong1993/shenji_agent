"""Time-base conversion between 庙算 SDK scenarios and the self-built engine.

Two clocks exist (plan §5.3):

* The SDK scenario ``time`` block: ``tick`` is **frames per step** and
  ``max_time`` is the maximum frame count.  The SDK advances one frame per
  second, so total wall time is ``max_time * 1 s`` and the number of steps is
  ``max_time / tick``.  ``tick`` is *not* a number of seconds.
* The self-built campaign engine: ``CampaignSpec.tick_seconds`` (30 s) and a
  tick counter.  ``scenario.build_state`` currently ignores the scenario
  ``time.tick`` value entirely.

This module fixes the conversion formulas in one place so P2+ code and the
regression fixture cannot silently reinterpret ``tick`` as seconds.
"""

from __future__ import annotations

import math

# The SDK advances one simulation frame per second: the public rule
# "36 km/h crosses one 200 m hex in 20 s" means 20 frames per hex at 36 km/h.
# NOTE: this 1 s/frame ratio is inferred from the public rules, not yet
# confirmed by a black-box run (see docs/geo_engine/P0_status.md).  The
# formulas here are the single place to change once it is confirmed.
FRAME_SECONDS = 1.0


def sdk_max_step(max_time: int, tick: float) -> int:
    """Number of engine steps for an SDK scenario (floor(max_time / tick))."""
    if tick is None or float(tick) <= 0:
        return int(max_time)
    return int(math.floor(float(max_time) / float(tick)))


def sdk_total_seconds(max_time: int, frame_seconds: float = FRAME_SECONDS) -> float:
    """Total simulated seconds for an SDK scenario."""
    return float(max_time) * float(frame_seconds)


def sdk_step_seconds(tick: float, frame_seconds: float = FRAME_SECONDS) -> float:
    """Wall-clock seconds represented by one SDK engine step."""
    return float(tick) * float(frame_seconds)


def legacy_event_seconds(tick: int, tick_seconds: float) -> float:
    """Event time of a self-built engine tick in seconds."""
    return float(tick) * float(tick_seconds)
