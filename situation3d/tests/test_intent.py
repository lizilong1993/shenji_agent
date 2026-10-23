"""Tests for advisory intent recognition / behavior prediction (mock JEV)."""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from policy.intent import (  # noqa: E402
    BEHAVIOR_LABELS,
    INTENT_LABELS,
    BehaviorPredictor,
    IntentEstimator,
)
from policy.jev import MockJevClient  # noqa: E402
from geoio.observation import observation  # noqa: E402
from geo.rule_grid import RuleGrid  # noqa: E402
from env.service import FlatEnvironment  # noqa: E402
from sim.entity import Entity, Pose  # noqa: E402
from sim.world import World  # noqa: E402


def obs(t=10.0):
    return {
        "time_s": t,
        "entities": [
            {"id": 1, "affiliation": 0, "unit_kind": "vehicle", "name": "红",
             "pos": {"lon": 119.9, "lat": 23.1}, "blood": 3, "speed_ms": 0.0,
             "domain": "land"},
            {"id": 2, "affiliation": 1, "unit_kind": "vehicle", "name": "蓝",
             "pos": {"lon": 119.92, "lat": 23.1}, "blood": 5, "speed_ms": 5.0,
             "domain": "land"},
        ],
        "objectives": [{"name": "OBJ", "owner": -1, "value": 10}],
    }


class TestIntentEstimator(unittest.TestCase):
    def test_estimate_and_history(self):
        est = IntentEstimator(MockJevClient(preference=("assault", "advance", "hold")))
        est.update(obs(10.0), [{"type": "direct_fire", "actors": [2]}])
        est.update(obs(20.0))
        results = est.estimate(obs(30.0), seat=0)
        self.assertIn(2, results)
        self.assertIn(results[2].label, INTENT_LABELS)
        self.assertAlmostEqual(sum(results[2].probabilities.values()), 1.0, places=5)
        self.assertEqual(len(est.history[2]), 2)
        formatted = est.format(results)
        self.assertIn("2", formatted)
        self.assertIn("probabilities", formatted["2"])

    def test_own_entities_excluded(self):
        est = IntentEstimator(MockJevClient())
        results = est.estimate(obs(), seat=0)
        self.assertNotIn(1, results)


class TestWorldIntelIntegration(unittest.TestCase):
    def test_observation_and_commander_state_include_intel(self):
        grid = RuleGrid(119.9, 23.1, cols=20, rows=20)
        world = World(FlatEnvironment(), grid)
        lon, lat = grid.cell_center_lonlat(5, 5)
        world.add_entity(Entity(1, "A", 0, "land", "tracked", Pose(lon, lat)))
        payload = {"intent": {"2": {"label": "assault", "confidence": 0.7}},
                   "behavior": {"2": {"label": "advance", "confidence": 0.6}}}
        world.set_intel(0, payload)
        obs = observation(world, seat=0)
        self.assertEqual(obs["intel"]["intent"]["2"]["label"], "assault")
        state = world._commander_state(0)
        self.assertEqual(state["intel"]["behavior"]["2"]["label"], "advance")


class TestBehaviorPredictor(unittest.TestCase):
    def test_prediction_labels(self):
        predictor = BehaviorPredictor(MockJevClient(preference=("advance", "attack")))
        results = predictor.predict(obs(), seat=0)
        self.assertIn(2, results)
        self.assertIn(results[2].label, BEHAVIOR_LABELS)
        self.assertAlmostEqual(sum(results[2].probabilities.values()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
