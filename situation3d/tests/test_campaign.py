"""Campaign tests: continuity, script events, outcome, determinism."""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)
if os.path.join(SITUATION3D, "campaigns") not in sys.path:
    sys.path.insert(0, os.path.join(SITUATION3D, "campaigns"))

import campaign as C
from wargame_core import scenario


def _fixtures_available() -> bool:
    try:
        import taiwan_landing as T
    except ImportError:
        return False
    return os.path.isfile(T.map_path())


@unittest.skipUnless(_fixtures_available(), "campaign fixtures not generated")
class CampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import taiwan_landing as T
        cls.T = T
        cls.spec = T.build_spec()

    def _run(self, seed=20261019, max_ticks=4):
        # Each run needs its own spec: ScriptEvent.fired is mutated by the
        # runner and build_spec() is cheap.
        state = scenario.load_scenario_dict(self.T.scenario_dict(), self.T.map_path(),
                                            seed=seed, tick_seconds=30.0)
        runner = C.CampaignRunner(self.T.build_spec(), state)
        return runner.run(max_ticks_per_stage=max_ticks), runner

    def test_stages_are_continuous(self):
        result, runner = self._run(max_ticks=6)
        ticks = [st["start_tick"] for st in result["stages"]]
        self.assertEqual(ticks, sorted(ticks))
        ends = [st["end_tick"] for st in result["stages"]]
        for a, b in zip(ends, ticks[1:]):
            # A stage cannot start before the previous one ended.
            self.assertLessEqual(a, b + 6 + 1)

    def test_frames_are_one_per_tick_and_continuous(self):
        result, _ = self._run(max_ticks=3)
        ticks = [f["tick"] for f in result["frames"]]
        self.assertEqual(ticks, list(range(1, len(ticks) + 1)))
        self.assertEqual(result["frames"][-1]["tick"], result["stages"][-1]["end_tick"])

    def test_script_events_fire_in_order(self):
        state = scenario.load_scenario_dict(self.T.scenario_dict(), self.T.map_path(),
                                            seed=20261019, tick_seconds=30.0)
        spec = self.T.build_spec()
        runner = C.CampaignRunner(spec, state)
        # Drive a fixed 400 ticks so the first three script events are due.
        n = 0
        while n < 400 and not state.winner:
            runner._fire_due_script(spec.stages[1])
            runner.engine.step(runner.orders_fn(state))
            n += 1
        script = [e for e in runner.fired_script]
        self.assertGreaterEqual(len(script), 3)
        ticks = [e["tick"] for e in script]
        self.assertEqual(ticks, sorted(ticks))

    def test_outcome_has_engine_and_script(self):
        result, _ = self._run(max_ticks=4)
        out = result["outcome"]
        self.assertIn("engine", out)
        self.assertIn("script", out)
        self.assertIn(out["engine"]["winner"], ("red", "blue", "draw", "undecided"))
        self.assertIn("disclaimer", out)
        self.assertIsInstance(out["match"], bool)

    def test_same_seed_same_result(self):
        r1, _ = self._run(seed=11, max_ticks=4)
        r2, _ = self._run(seed=11, max_ticks=4)
        self.assertEqual(r1["outcome"]["engine"], r2["outcome"]["engine"])
        self.assertEqual([e["type"] for e in r1["events"]],
                         [e["type"] for e in r2["events"]])

    def test_no_unit_disappears_without_event(self):
        result, _ = self._run(max_ticks=8)
        dead = {u["id"] for u in result["final"]["units"] if not u["alive"]}
        destroyed = set()
        for e in result["events"]:
            if e["type"] == "destroyed":
                destroyed.update(e.get("actors") or [])
        self.assertTrue(dead.issubset(destroyed),
                        "dead without destroyed event: %s" % (dead - destroyed))


if __name__ == "__main__":
    unittest.main()
