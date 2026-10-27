"""Tests for the broad-collect → clean → analyze pipeline."""

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

from intel.case import CaseIntelProfile  # noqa: E402
from intel.gateway import IntelBundle  # noqa: E402
from intel.model import Observation  # noqa: E402
from intel.pipeline import clean_candidates, run_case  # noqa: E402

START = "2026-10-24T00:00:00Z"
END = "2026-10-24T23:59:59Z"
FETCHED = "2026-10-24T12:00:00Z"
BBOX = (119.5, 22.5, 120.5, 23.5)
RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Local news</title>
<item><title>Port activity</title><link>https://example.invalid/a</link>
<description>Shipping and port workers report activity.</description>
<pubDate>Sat, 24 Oct 2026 12:00:00 GMT</pubDate></item>
</channel></rss>"""


def fake_fetch(url):
    if "earthquake.usgs.gov" in url:
        return json.dumps({"features": [{
            "id": "us1",
            "geometry": {"type": "Point", "coordinates": [120.1, 23.0]},
            "properties": {"time": 1792818000000, "mag": 4.2,
                           "place": "6 km E of Oakland, CA"},
        }]}).encode("utf-8")
    if "api.open-meteo.com" in url:
        return json.dumps({
            "latitude": 23.0, "longitude": 120.0, "elevation": 10,
            "current": {"time": "2026-10-24T12:00", "interval": 900,
                        "temperature_2m": 26.0, "relative_humidity_2m": 70,
                        "precipitation": 0.0, "weather_code": 1,
                        "wind_speed_10m": 8.0, "wind_direction_10m": 90,
                        "visibility": 20000},
        }).encode("utf-8")
    if "example.invalid/feed" in url:
        return RSS
    raise AssertionError("unexpected URL: %s" % (url,))


def profile():
    return CaseIntelProfile(
        case_id="case1", name="case1", bbox=BBOX,
        domains=("hazard", "news", "weather", "society", "other"),
        sources=("usgs", "open-meteo", "local_news"),
        keywords=("taiwan", "port"), lookback_hours=24.0,
    )


class PipelineTest(unittest.TestCase):
    def test_broad_collection_keeps_no_keyword_item(self):
        with tempfile.TemporaryDirectory() as root:
            result = run_case(
                "case1", profile(), START, END,
                sources=("usgs", "open-meteo", "local_news"),
                root=root, custom_sources=[{
                    "source_id": "local_news",
                    "name": "Local news",
                    "kind": "rss",
                    "domains": ["news", "society"],
                    "url": "https://example.invalid/feed",
                }], fetcher=fake_fetch)
            self.assertEqual(len(result.candidates.observations), 3)
            self.assertEqual(len(result.bundle.observations), 3)
            self.assertGreater(result.report["stage2"]["kept_without_keyword"], 0)
            titles = [obs.title for obs in result.bundle.observations]
            self.assertIn("6 km E of Oakland, CA", titles)
            self.assertEqual(result.report["stage1"]["observations"], 3)
            self.assertTrue(os.path.isfile(os.path.join(root, "case1",
                                                        "candidates.json")))
            self.assertTrue(os.path.isfile(os.path.join(root, "case1",
                                                        "bundle.json")))
            self.assertTrue(os.path.isfile(os.path.join(root, "case1",
                                                        "pipeline.json")))
            self.assertEqual(result.bundle.scenario_id, "case1")

    def test_cleaning_drops_duplicates_and_outside(self):
        observations = [
            Observation("usgs:a", "usgs", "a", "hazard", "earthquake",
                        23.0, 120.0, START, FETCHED, 0, 0.5, "", "same"),
            Observation("usgs:b", "usgs", "b", "hazard", "earthquake",
                        23.0, 120.0, START, FETCHED, 1, 0.4, "", "same"),
            Observation("usgs:c", "usgs", "c", "hazard", "earthquake",
                        30.0, 120.0, START, FETCHED, 2, 0.8, "", "outside"),
        ]
        bundle = IntelBundle("b", "case1", observations, START,
                             bbox=BBOX, time_window=(START, END),
                             source_allowlist=("usgs",))
        cleaned, report = clean_candidates(bundle, profile(), START, END,
                                           ("usgs", "open-meteo"))
        self.assertEqual(len(cleaned.observations), 1)
        self.assertEqual(report["dropped_duplicates"], 1)
        self.assertEqual(report["dropped_out_of_scope"], 1)
        self.assertEqual(cleaned.observations[0].properties["relevance"],
                         "medium")


class ConnectorTest(unittest.TestCase):
    def test_source_descriptions_cover_builtins(self):
        from intel.sources import SourceStore
        with tempfile.TemporaryDirectory() as root:
            sources = SourceStore(os.path.join(root, "sources.json")).list()
            by_id = {item["source_id"]: item for item in sources}
            for source_id in ("manual", "usgs", "gdacs", "firms", "eonet",
                              "adsb", "gdelt", "open-meteo"):
                self.assertIn(source_id, by_id)
                self.assertTrue(by_id[source_id]["summary"])
                self.assertTrue(by_id[source_id]["coverage"])
                self.assertTrue(by_id[source_id]["access"])

    def test_custom_source_metadata_and_private_host_guard(self):
        from intel.connectors.http import FetchError, ensure_public_host
        from intel.sources import SourceStore
        with tempfile.TemporaryDirectory() as root:
            store = SourceStore(os.path.join(root, "sources.json"))
            added = store.add({
                "source_id": "local_feed",
                "name": "Local feed",
                "kind": "rss",
                "url": "https://example.invalid/feed",
                "domains": ["news"],
                "summary": "Local reporting feed",
                "features": ["hourly", "public"],
                "coverage": "Tainan",
                "access": "HTTPS RSS",
            })
            self.assertEqual(added["summary"], "Local reporting feed")
            self.assertEqual(added["features"], ["hourly", "public"])
            self.assertEqual(store.get("local_feed")["coverage"], "Tainan")
        with self.assertRaises(FetchError):
            ensure_public_host("https://127.0.0.1/feed")
        with self.assertRaises(FetchError):
            ensure_public_host("http://example.invalid/feed")

    def test_rss_parser_places_items_at_centroid(self):
        from intel.connectors.rss import parse_feed
        items = parse_feed(RSS, {"source_id": "local_news", "kind": "rss",
                                 "domains": ["news"]}, FETCHED, (23.0, 120.0),
                           start=START, end=END)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].title, "Port activity")
        self.assertTrue(items[0].is_centroid)
        self.assertEqual(items[0].geo_precision, "unknown")

    def test_open_meteo_normalizes_current_weather(self):
        from intel.connectors.weather import normalize_payload
        payload = json.loads(fake_fetch("https://api.open-meteo.com/v1/forecast")
                             .decode("utf-8"))
        items = normalize_payload(payload, FETCHED, 23.0, 120.0)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].domain, "weather")
        self.assertEqual(items[0].properties["weather_code"], 1)
        self.assertEqual(items[0].properties["temperature_c"], 26.0)


if __name__ == "__main__":
    unittest.main()
