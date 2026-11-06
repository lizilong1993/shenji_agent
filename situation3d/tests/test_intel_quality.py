"""Tests for the data-quality enum."""

from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from intel.quality import QUALITY_LEVELS, age_hours, quality_of  # noqa: E402

NOW = datetime(2026, 11, 6, 12, 0, 0, tzinfo=timezone.utc)


def record(**overrides):
    value = {
        "source_id": "usgs", "domain": "hazard",
        "geo_precision": "exact", "is_centroid": False,
        "event_time": (NOW - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ingest_time": (NOW - timedelta(minutes=50)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    value.update(overrides)
    return value


class QualityTest(unittest.TestCase):
    def test_fresh_exact_is_measured(self):
        result = quality_of(record(), now=NOW)
        self.assertEqual(result["level"], "measured")
        self.assertEqual(result["age_h"], 1.0)
        self.assertIn("usgs", result["provenance"]["source_id"])

    def test_beyond_half_life_is_stale(self):
        # hazard 半衰期 6h
        result = quality_of(record(event_time=(
            NOW - timedelta(hours=8)).strftime("%Y-%m-%dT%H:%M:%SZ")), now=NOW)
        self.assertEqual(result["level"], "stale")
        self.assertTrue(any("半衰期" in r for r in result["reasons"]))

    def test_domain_selects_the_half_life(self):
        # geography 半衰期 168h：8 小时远未过期
        value = quality_of(record(domain="geography", event_time=(
            NOW - timedelta(hours=8)).strftime("%Y-%m-%dT%H:%M:%SZ")), now=NOW)
        self.assertEqual(value["level"], "measured")
        self.assertEqual(value["half_life_h"], 168.0)

    def test_centroid_is_not_measured(self):
        result = quality_of(record(is_centroid=True), now=NOW)
        self.assertEqual(result["level"], "interpolated")

    def test_admin_precision_is_not_measured(self):
        result = quality_of(record(geo_precision="admin0"), now=NOW)
        self.assertEqual(result["level"], "interpolated")

    def test_declared_level_wins(self):
        result = quality_of(record(quality="predicted"), now=NOW)
        self.assertEqual(result["level"], "predicted")

    def test_stale_overrides_declared(self):
        result = quality_of(record(
            quality="predicted",
            event_time=(NOW - timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%SZ")),
            now=NOW)
        self.assertEqual(result["level"], "stale")

    def test_unregistered_source_is_reported(self):
        result = quality_of(record(source_id="mystery"), now=NOW)
        self.assertTrue(any("未登记" in r for r in result["reasons"]))

    def test_missing_event_time_is_not_stale(self):
        result = quality_of(record(event_time=""), now=NOW)
        self.assertIsNone(result["age_h"])
        self.assertEqual(result["level"], "measured")

    def test_levels_are_the_documented_enum(self):
        self.assertEqual(QUALITY_LEVELS,
                         ("measured", "interpolated", "predicted", "stale"))

    def test_age_hours_none_for_bad_input(self):
        self.assertIsNone(age_hours("not-a-date", now=NOW))


if __name__ == "__main__":
    unittest.main()
