"""Line-of-sight rules for the clean-room land wargame core.

Public rule summary (from the 庙算 rules document):

* LOS is evaluated on the centre-to-centre hex line.
* When observer and target are at different elevations, the higher one is the
  observer and gets one extra elevation grade ("高看低").
* The line elevation is linearly interpolated between the two endpoints.
* An intermediate hex blocks if its elevation is higher than the line.
  Urban and forest hexes add one elevation grade for this test.
* Adjacent hexes are always visible.

The implementation is deliberately small and testable; it is not a bit-exact
copy of the closed engine.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from . import hexmap


class LineOfSight(object):
    def __init__(self, hmap: hexmap.HexMap):
        self.map = hmap
        self._cache: Dict[Tuple[int, int, float, float], bool] = {}

    def visible(self, a: int, b: int,
                observer_elev: Optional[float] = None,
                target_elev: Optional[float] = None) -> bool:
        if a == b:
            return True
        if hexmap.distance(a, b) <= 1:
            return True
        ea = self.map.elev(a) if observer_elev is None else float(observer_elev)
        eb = self.map.elev(b) if target_elev is None else float(target_elev)
        key = (a, b, ea, eb)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        # "High looks low": the higher endpoint gets one extra grade.  The
        # line is always walked from a to b, so the bonus is applied to
        # whichever endpoint is higher.
        if ea >= eb:
            start, end = ea + self.map.ele_grade, eb
        else:
            start, end = ea, eb + self.map.ele_grade

        cells = hexmap.line(a, b)
        n = len(cells) - 1
        if n <= 1:
            self._cache[key] = True
            return True
        blocked = False
        for i in range(1, n):
            t = i / float(n)
            line_elev = start + (end - start) * t
            cell_elev = self.map.elev(cells[i]) + self.map.los_bonus(cells[i])
            if cell_elev > line_elev + 1e-6:
                blocked = True
                break
        self._cache[key] = not blocked
        return not blocked

    def clear_cache(self) -> None:
        self._cache.clear()
