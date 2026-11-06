"""滑动基线异常检测（Welford）。

覆盖四条刻意的取舍：冷启动不报、先算后更新、方差下限、方向可见。
"""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from intel.baseline import (  # noqa: E402
    DEFAULT_MIN_SAMPLES, Baseline, BaselineTracker)


class WelfordTest(unittest.TestCase):
    def test_matches_naive_mean_and_variance(self):
        values = [3.0, 5.0, 7.0, 9.0, 11.0]
        base = Baseline()
        for value in values:
            base.update(value)
        naive_mean = sum(values) / len(values)
        naive_var = sum((v - naive_mean) ** 2 for v in values) / (len(values) - 1)
        self.assertAlmostEqual(base.mean, naive_mean, places=9)
        self.assertAlmostEqual(base.variance, naive_var, places=9)

    def test_count_and_extremes(self):
        base = Baseline()
        for value in (4, 9, 1, 7):
            base.update(value)
        self.assertEqual(base.count, 4)
        self.assertEqual(base.min_value, 1.0)
        self.assertEqual(base.max_value, 9.0)

    def test_non_numeric_ignored(self):
        base = Baseline()
        self.assertIsNone(base.update("abc"))
        self.assertIsNone(base.update(None))
        self.assertIsNone(base.update(float("nan")))
        self.assertIsNone(base.update(float("inf")))
        self.assertEqual(base.count, 0)

    def test_z_uses_state_before_update(self):
        # 先算后更新：把尖峰自己的值算进均值会系统性压低 z
        base = Baseline()
        for _ in range(20):
            base.update(10.0)
        z = base.z_score(20.0)
        base.update(20.0)
        # 更新后均值被拉高，同样的值 z 应更小
        self.assertIsNotNone(z)
        self.assertLess(base.z_score(20.0), z)

    def test_needs_two_samples_for_z(self):
        base = Baseline()
        self.assertIsNone(base.z_score(1.0))
        base.update(1.0)
        self.assertIsNone(base.z_score(1.0))

    def test_zero_variance_uses_floor(self):
        # 全部相同 → 方差 0，直接除会炸；用下限保证偏离仍可测
        base = Baseline()
        for _ in range(20):
            base.update(5.0)
        z = base.z_score(8.0)
        self.assertIsNotNone(z)
        self.assertGreater(z, 0)
        self.assertTrue(base.floor() > 0)


class TrackerTest(unittest.TestCase):
    def test_cold_start_does_not_alert(self):
        tracker = BaselineTracker()
        alerts = []
        for i in range(DEFAULT_MIN_SAMPLES):
            got = tracker.observe("g1", 100.0 if i % 2 else 90.0)
            if got:
                alerts.append(got)
        self.assertEqual(alerts, [], "样本不足时不应报异常")

    def test_spike_is_detected_after_warmup(self):
        tracker = BaselineTracker()
        for value in (10, 12, 9, 11, 10, 12, 9, 11, 10, 12):
            tracker.observe("g1", value)
        got = tracker.observe("g1", 40.0)
        self.assertIsNotNone(got)
        self.assertEqual(got.direction, "up")
        self.assertGreater(got.z, 3)

    def test_drop_is_detected_with_negative_direction(self):
        tracker = BaselineTracker()
        for value in (100, 102, 99, 101, 100, 102, 99, 101, 100, 102):
            tracker.observe("g1", value)
        got = tracker.observe("g1", 10.0)
        self.assertIsNotNone(got)
        self.assertEqual(got.direction, "down")
        self.assertLess(got.z, -3)

    def test_normal_value_does_not_alert(self):
        tracker = BaselineTracker()
        for value in (10, 12, 9, 11, 10, 12, 9, 11, 10, 12):
            tracker.observe("g1", value)
        self.assertIsNone(tracker.observe("g1", 11.0))

    def test_series_are_independent(self):
        tracker = BaselineTracker()
        for _ in range(10):
            tracker.observe("a", 100.0)
            tracker.observe("b", 1.0)
        got = tracker.observe("a", 500.0)
        self.assertIsNotNone(got)
        self.assertEqual(got.key, "a")
        # b 的基线没被 a 的样本污染
        self.assertEqual(tracker.series["b"].count, 10)

    def test_snapshot_and_reset(self):
        tracker = BaselineTracker()
        for value in (1, 2, 3):
            tracker.observe("k", value)
        snap = tracker.snapshot()
        self.assertEqual(snap["k"]["count"], 3)
        self.assertAlmostEqual(snap["k"]["mean"], 2.0)
        tracker.reset()
        self.assertEqual(tracker.snapshot(), {})

    def test_anomaly_carries_baseline_context(self):
        tracker = BaselineTracker()
        for value in (10, 12, 9, 11, 10, 12, 9, 11, 10, 12):
            tracker.observe("g1", value)
        got = tracker.observe("g1", 40.0)
        payload = got.as_dict()
        self.assertEqual(payload["key"], "g1")
        self.assertIn("baseline_mean", payload)
        self.assertGreater(payload["baseline_stdev"], 0)


if __name__ == "__main__":
    unittest.main()
