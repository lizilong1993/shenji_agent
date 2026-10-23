"""Behavior intents (P1.6): what to do, not where to drive.

The behaviour layer chooses an objective cell and a pace/risk style; the
planner resolves the actual coordinates (cell centre, formation offset,
route) and the motion layer enforces it.  ``Intent.point`` is the only
bridge from intent to geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

PACE_SCALE = {"dash": 1.0, "normal": 0.9, "cautious": 0.65}
# ``risk`` is the commander's risk appetite: low appetite slows down.
RISK_SCALE = {"low": 0.9, "normal": 1.0, "high": 1.0}


@dataclass(frozen=True)
class Intent(object):
    cell: Optional[Tuple[int, int]] = None
    lon: Optional[float] = None
    lat: Optional[float] = None
    pace: str = "normal"      # dash | normal | cautious
    risk: str = "normal"      # low | normal | high
    objective: str = ""

    def point(self, rule_grid) -> Tuple[float, float]:
        if self.cell is not None:
            return rule_grid.cell_center_lonlat(int(self.cell[0]),
                                                int(self.cell[1]))
        if self.lon is not None and self.lat is not None:
            return float(self.lon), float(self.lat)
        raise ValueError("intent needs a cell or a point")

    def speed_scale(self) -> float:
        return (PACE_SCALE.get(str(self.pace), 1.0)
                * RISK_SCALE.get(str(self.risk), 1.0))
