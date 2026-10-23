"""Public empirical calibration for sensor detection probability.

The physical perception model uses ``p(d) = p0 * exp(-d / scale)``.  Sensor
records quote a nominal detection range, conventionally the range at which
detection probability is 0.5 (Johnson/NIIRS target-acquisition practice).
Calibrating ``scale`` from that convention gives

    p(range) = 0.5  ->  scale = range / ln(p0 / 0.5)

which is what :func:`nominal_scale_frac` returns.  The Johnson cycle
criteria (detection 2, recognition 4, identification 8 cycles across the
target) are kept in ``rules/data/detection_calibration.json`` as the public
reference; this is an empirical convention, not a site-specific field test.
"""

from __future__ import annotations

import json
import math
import os

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "rules", "data", "detection_calibration.json")


def reference() -> dict:
    with open(_DATA, encoding="utf-8") as fh:
        return json.load(fh)


def nominal_scale_frac(p0: float) -> float:
    """Scale fraction so ``p(nominal_range) == 0.5`` for ``p0 * exp(-d/s)``."""
    p0 = float(p0)
    if p0 <= 0.5:
        return 1.0
    return 1.0 / math.log(p0 / 0.5)


def probability_at(distance_m: float, range_m: float, p0: float,
                   scale_frac: float) -> float:
    if distance_m < 0.0 or distance_m > range_m:
        return 0.0
    return p0 * math.exp(-max(1e-3, distance_m)
                         / max(1e-3, range_m * scale_frac))


def cycle_thresholds() -> dict:
    return dict(reference().get("criteria_cycles") or {})


def report() -> dict:
    data = reference()
    rows = {}
    nominal_m = 1000.0
    for kind, p0 in sorted((data.get("kinds") or {}).items()):
        frac = nominal_scale_frac(p0)
        rows[kind] = {
            "p0": p0,
            "scale_frac": round(frac, 4),
            "p_nominal": round(probability_at(nominal_m, nominal_m,
                                              p0, frac), 4),
            "p_half": round(probability_at(nominal_m * 0.5, nominal_m,
                                           p0, frac), 4),
            "p_quarter": round(probability_at(nominal_m * 0.25, nominal_m,
                                              p0, frac), 4),
        }
    return {"model": data.get("model"), "criteria_cycles": cycle_thresholds(),
            "kinds": rows}
