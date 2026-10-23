"""Public-convention calibration checks (Johnson/NIIRS 50% range)."""

from __future__ import annotations

import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from sim import calibration  # noqa: E402
from sim import perception  # noqa: E402


class CalibrationTest(unittest.TestCase):
    def test_reference_has_public_criteria_and_sources(self):
        data = calibration.reference()
        cycles = data["criteria_cycles"]
        self.assertEqual(cycles["detection"], 2.0)
        self.assertEqual(cycles["recognition"], 4.0)
        self.assertEqual(cycles["identification"], 8.0)
        self.assertTrue(data["references"])
        for ref in data["references"]:
            self.assertTrue(ref.get("citation"))

    def test_nominal_range_is_fifty_percent(self):
        for kind, defaults in perception._KINDS.items():
            scale = defaults["scale_frac"]
            p0 = defaults["p0"]
            p = calibration.probability_at(1.0, 1.0, p0, scale)
            self.assertAlmostEqual(
                p, 0.5, places=6,
                msg="%s nominal P=%.3f" % (kind, p))

    def test_probability_falls_with_distance(self):
        for kind, defaults in perception._KINDS.items():
            scale = defaults["scale_frac"]
            p0 = defaults["p0"]
            ps = [calibration.probability_at(d, 1.0, p0, scale)
                  for d in (0.0, 0.25, 0.5, 0.75, 1.0)]
            self.assertEqual(ps, sorted(ps, reverse=True), kind)
            self.assertEqual(ps[-1], 0.5)
            self.assertEqual(calibration.probability_at(1.01, 1.0, p0,
                                                        scale), 0.0)

    def test_report_is_finite(self):
        report = calibration.report()
        self.assertEqual(sorted(report["kinds"]),
                         sorted(perception._KINDS))
        for row in report["kinds"].values():
            for key in ("p0", "scale_frac", "p_nominal", "p_half",
                        "p_quarter"):
                self.assertTrue(math.isfinite(row[key]))
            self.assertAlmostEqual(row["p_nominal"], 0.5, places=3)
            self.assertGreater(row["p_half"], row["p_nominal"])
            self.assertGreater(row["p_quarter"], row["p_half"])


if __name__ == "__main__":
    unittest.main()
