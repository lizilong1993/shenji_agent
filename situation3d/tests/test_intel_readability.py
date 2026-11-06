"""情报卡片可读性：关键读数、标题、快照去重。

这些测试锁住的是"读者能不能看懂"这件事，不是内部结构：
之前卡片只有标题和坐标，同一地点重复出现多条，读者无法判断说了什么。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from intel.connectors import weather  # noqa: E402
from intel.management import IntelManager, _facts  # noqa: E402
from intel.weather_text import describe, wind_direction  # noqa: E402


class WeatherTextTest(unittest.TestCase):
    def test_known_codes(self):
        self.assertEqual(describe(0), "晴")
        self.assertEqual(describe(3), "阴")
        self.assertEqual(describe(61), "小雨")
        self.assertEqual(describe(95), "雷暴")

    def test_unknown_code_still_readable(self):
        self.assertEqual(describe(1234), "天气代码 1234")
        self.assertEqual(describe(None), "")

    def test_wind_direction_octants(self):
        self.assertEqual(wind_direction(0), "北风")
        self.assertEqual(wind_direction(90), "东风")
        self.assertEqual(wind_direction(315), "西北风")
        self.assertEqual(wind_direction(22), "北风")     # 边界内
        self.assertEqual(wind_direction(23), "东北风")   # 越过边界
        self.assertEqual(wind_direction(None), "")


class WeatherTitleTest(unittest.TestCase):
    def _payload(self, **current):
        base = {"time": "2026-11-06T11:00", "temperature_2m": 21.5,
                "relative_humidity_2m": 78, "precipitation": 0.0,
                "weather_code": 2, "wind_speed_10m": 14.0,
                "wind_direction_10m": 135, "visibility": 12000,
                "interval": 900}
        base.update(current)
        return {"current": base, "elevation": 8}

    def test_title_carries_readings_not_coordinates(self):
        obs = weather.normalize_payload(self._payload(), "2026-11-06T11:05:00Z",
                                        23.0, 120.2)[0]
        self.assertIn("局部多云", obs.title)
        self.assertIn("22°C", obs.title)
        self.assertIn("东南风", obs.title)
        self.assertNotIn("Weather at", obs.title)

    def test_bare_coordinates_only_when_no_fields(self):
        empty = {"time": "2026-11-06T11:00", "temperature_2m": None,
                 "relative_humidity_2m": None, "precipitation": None,
                 "weather_code": None, "wind_speed_10m": None,
                 "wind_direction_10m": None, "visibility": None,
                 "interval": 900}
        # 全空会被上游判为无效载荷；这里直接测标题兜底
        from intel.connectors.weather import _title
        self.assertIn("气象观测", _title({}, 23.0, 120.2))


class FactsTest(unittest.TestCase):
    def test_weather_facts_are_labeled_and_formatted(self):
        facts = _facts({"source_id": "open-meteo", "properties": {
            "temperature_c": 21.5, "wind_speed_kmh": 14.0,
            "precipitation_mm": 0.0, "weather_code": 2}})
        labels = {row["label"]: row["value"] for row in facts}
        self.assertEqual(labels["气温"], "21.5°C")
        self.assertEqual(labels["风速"], "14.0 km/h")
        self.assertEqual(labels["天气"], "局部多云")

    def test_adsb_facts(self):
        facts = _facts({"source_id": "adsb", "properties": {
            "aircraft_type": "A320", "altitude_baro_ft": 30025,
            "ground_speed_kt": 460.0, "military": False}})
        labels = {row["label"]: row["value"] for row in facts}
        self.assertEqual(labels["机型"], "A320")
        self.assertEqual(labels["气压高度"], "30025 ft")
        self.assertEqual(labels["军用"], "否")

    def test_none_values_are_skipped(self):
        facts = _facts({"source_id": "adsb", "properties": {
            "aircraft_type": "A320", "altitude_baro_ft": None,
            "ground_speed_kt": None}})
        self.assertEqual([row["label"] for row in facts], ["机型"])

    def test_unknown_source_has_no_facts(self):
        self.assertEqual(_facts({"source_id": "mystery", "properties": {
            "anything": 1}}), [])


class SnapshotDedupeTest(unittest.TestCase):
    def _write(self, root, case_id, observations):
        directory = os.path.join(root, case_id)
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "bundle.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"bundle_id": "b", "scenario_id": case_id,
                       "synthetic": False, "hash": "x",
                       "observations": observations}, handle)

    def _obs(self, obs_id, source_id, event_time, lat=23.0, lon=120.2,
             type_="weather"):
        return {"observation_id": obs_id, "source_id": source_id,
                "source_native_id": obs_id, "domain": "weather",
                "type": type_, "event_time": event_time,
                "ingest_time": event_time, "lat": lat, "lon": lon,
                "title": obs_id, "confidence": 0.8, "status": "active"}

    def test_same_site_keeps_only_latest(self):
        with tempfile.TemporaryDirectory() as root:
            self._write(root, "c1", [
                self._obs("open-meteo:a:1", "open-meteo",
                          "2026-11-06T08:00:00Z"),
                self._obs("open-meteo:a:2", "open-meteo",
                          "2026-11-06T09:00:00Z"),
                self._obs("open-meteo:a:3", "open-meteo",
                          "2026-11-06T10:00:00Z"),
            ])
            items = IntelManager(root).overview("c1")["items"]
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["id"], "open-meteo:a:3")

    def test_different_sites_are_not_merged(self):
        with tempfile.TemporaryDirectory() as root:
            self._write(root, "c1", [
                self._obs("open-meteo:a:1", "open-meteo",
                          "2026-11-06T08:00:00Z", lon=120.2),
                self._obs("open-meteo:a:2", "open-meteo",
                          "2026-11-06T09:00:00Z", lon=119.9),
            ])
            items = IntelManager(root).overview("c1")["items"]
            self.assertEqual(len(items), 2)

    def test_event_sources_are_never_merged(self):
        # 同一地点两次地震是两起事件，合并就是丢情报。
        with tempfile.TemporaryDirectory() as root:
            self._write(root, "c1", [
                self._obs("usgs:1", "usgs", "2026-11-06T08:00:00Z",
                          type_="earthquake"),
                self._obs("usgs:2", "usgs", "2026-11-06T09:00:00Z",
                          type_="earthquake"),
            ])
            items = IntelManager(root).overview("c1")["items"]
            self.assertEqual(len(items), 2)


if __name__ == "__main__":
    unittest.main()
