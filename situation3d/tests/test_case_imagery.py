"""Case imagery recipes and cached reuse wiring."""

from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from resources import case_imagery  # noqa: E402


class CaseImageryTest(unittest.TestCase):
    def test_recipes(self):
        self.assertEqual(case_imagery.case_keys(), ["india_china", "taiwan"])
        tw = case_imagery.CASES["taiwan"]
        self.assertEqual(tw.bbox, (120.055, 22.915, 120.235, 23.025))
        self.assertTrue(case_imagery.CASES["india_china"].mosaic)
        self.assertEqual(tw.tileset().max_zoom, 13)

    def test_unknown_case(self):
        with self.assertRaises(KeyError):
            case_imagery.ensure_case_imagery(None, "nope")

    def test_ensure_forwards_recipe(self):
        fake = {"sentinel": None, "tiles": None, "fallback": False}
        with mock.patch.object(case_imagery, "acquire_sentinel_tiles",
                               return_value=fake) as call:
            result = case_imagery.ensure_case_imagery(
                object(), "taiwan", "/tmp/cache")
        self.assertIs(result, fake)
        kwargs = call.call_args.kwargs
        self.assertEqual(kwargs["target_date"], "2026-10-26")
        self.assertEqual(kwargs["resolution_m"], 20.0)
        self.assertFalse(kwargs["mosaic"])


if __name__ == "__main__":
    unittest.main()
