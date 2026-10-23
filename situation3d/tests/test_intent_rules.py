"""Tests for the rule-based intent baseline."""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from policy.intent_rules import RuleIntentClassifier  # noqa: E402


def window(label, steps=5):
    frames = []
    events = []
    lon, lat = 119.95, 23.10
    for i in range(steps):
        speed = 0.0
        if label == "assault":
            lon -= 0.0008
            speed = 8.0
        elif label == "retreat":
            lon += 0.0008
            speed = 8.0
        elif label == "recon":
            lat += 0.0008
            speed = 6.0
        objective = None
        if label == "defend":
            objective = {"pos": {"lon": lon, "lat": lat}, "owner": 1}
        frames.append({
            "time_s": i * 10.0,
            "entities": [
                {"id": 1, "affiliation": 0, "pos": {"lon": 119.90, "lat": 23.10},
                 "speed_ms": 0.0, "unit_kind": "vehicle"},
                {"id": 2, "affiliation": 1, "pos": {"lon": lon, "lat": lat},
                 "speed_ms": speed, "unit_kind": "vehicle"},
            ],
            "objectives": [objective] if objective else [],
        })
        if label == "assault" and i >= 3:
            events.append({"type": "direct_fire", "actors": [2]})
    return frames, events


class TestRuleIntent(unittest.TestCase):
    def test_classes(self):
        clf = RuleIntentClassifier(("assault", "defend", "recon", "retreat"))
        for label in ("assault", "defend", "recon", "retreat"):
            frames, events = window(label)
            self.assertEqual(clf.classify(frames, events, 2, 1), label)

    def test_features(self):
        frames, events = window("assault")
        f = RuleIntentClassifier().features(frames, events, 2, 1)
        self.assertGreater(f["closing"], 0.0)
        self.assertEqual(f["fired"], 1.0)


if __name__ == "__main__":
    unittest.main()
