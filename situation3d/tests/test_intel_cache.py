"""Tests for the rolling intel cache and incremental collection."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlsplit

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from intel.cache import IntelCache, cache_db_path  # noqa: E402
from intel.case import CaseIntelProfile  # noqa: E402
from intel.model import Observation, iso  # noqa: E402
from intel.pipeline import run_case  # noqa: E402
from intel.refresh import fetch_windows  # noqa: E402

BBOX = (119.5, 22.5, 120.5, 23.5)
START = "2026-10-24T00:00:00Z"
END = "2026-10-24T23:59:59Z"
UTC = timezone.utc

USGS_ONE = {
    "features": [{
        "id": "us1",
        "geometry": {"type": "Point", "coordinates": [120.1, 23.0]},
        "properties": {"time": 1792818000000, "mag": 4.2,
                       "place": "6 km E of Oakland, CA"},
    }]
}


def obs(obs_id, event=START, lat=23.0, lon=120.0):
    return Observation(obs_id, "usgs", obs_id.split(":")[-1], "hazard",
                       "earthquake", lat, lon, event,
                       "2026-10-24T12:00:00Z", 0, 0.5, "", "quake")


def profile():
    return CaseIntelProfile(case_id="case1", name="case1", bbox=BBOX,
                            domains=("hazard",), sources=("usgs",),
                            keywords=("taiwan",), lookback_hours=24.0)


class CacheTest(unittest.TestCase):
    def test_roundtrip_and_prune(self):
        with tempfile.TemporaryDirectory() as root:
            cache = IntelCache(os.path.join(root, "cache.db"))
            try:
                cache.upsert([obs("usgs:a"),
                              obs("usgs:b", event="2026-10-10T00:00:00Z")])
                self.assertEqual(len(cache.query(START, END, ("usgs",))), 1)
                self.assertEqual(cache.prune("2026-10-17T00:00:00Z"), 1)
                self.assertEqual(
                    len(cache.query("2026-10-01T00:00:00Z",
                                    "2026-10-31T00:00:00Z")), 1)
            finally:
                cache.close()

    def test_state_is_keyed_by_bbox(self):
        with tempfile.TemporaryDirectory() as root:
            cache = IntelCache(os.path.join(root, "cache.db"))
            try:
                cache.mark("usgs", BBOX, cursor_until=END,
                           fetched_at="2026-10-24T12:00:00Z")
                self.assertEqual(cache.state("usgs", BBOX)["cursor_until"],
                                 END)
                self.assertEqual(cache.state("usgs", (0, 0, 1, 1)), {})
                cache.mark("usgs", BBOX, error="boom")
                state = cache.state("usgs", BBOX)
                self.assertEqual(state["cursor_until"], END)  # kept on error
                self.assertEqual(state["last_error"], "boom")
            finally:
                cache.close()

    def test_env_override(self):
        previous = os.environ.get("SITUATION_INTEL_CACHE_DB")
        os.environ["SITUATION_INTEL_CACHE_DB"] = os.path.join(
            tempfile.gettempdir(), "intel-override.db")
        try:
            self.assertTrue(
                cache_db_path("/repo/var/intel").endswith(
                    "intel-override.db"))
        finally:
            if previous is None:
                os.environ.pop("SITUATION_INTEL_CACHE_DB", None)
            else:
                os.environ["SITUATION_INTEL_CACHE_DB"] = previous


class WindowTest(unittest.TestCase):
    def setUp(self):
        self.start = datetime(2026, 10, 24, tzinfo=UTC)
        self.end = self.start + timedelta(hours=24)
        self.now = datetime(2026, 10, 24, 23, 0, tzinfo=UTC)

    def test_cold_start_is_the_full_window(self):
        windows = fetch_windows("usgs", self.start, self.end, None,
                                self.now, False)
        self.assertEqual(windows, [(self.start, self.end)])

    def test_cursor_requests_only_the_delta(self):
        cursor = iso(self.start + timedelta(hours=22))
        windows = fetch_windows("usgs", self.start, self.end, cursor,
                                self.now, False)
        self.assertEqual(windows,
                         [(self.start + timedelta(hours=21), self.end)])

    def test_cursor_inside_window_re_requests_the_overlap(self):
        windows = fetch_windows("usgs", self.start, self.end, iso(self.end),
                                self.now, False)
        self.assertEqual(windows,
                         [(self.end - timedelta(hours=1), self.end)])

    def test_cursor_beyond_end_skips_upstream(self):
        beyond = iso(self.end + timedelta(hours=2))
        windows = fetch_windows("usgs", self.start, self.end, beyond,
                                self.now, False)
        self.assertEqual(windows, [])

    def test_firms_cold_start_is_chunked(self):
        end = self.start + timedelta(days=7)
        windows = fetch_windows("firms", self.start, end, None, self.now,
                                False)
        self.assertEqual(len(windows), 2)
        for window_start, window_end in windows:
            self.assertLessEqual((window_end - window_start).days, 5)

    def test_live_source_skips_historical_window(self):
        self.assertEqual(fetch_windows("adsb", self.start, self.end, None,
                                       self.now, False), [])
        self.assertTrue(fetch_windows("adsb", self.start, self.end, None,
                                      self.now, True))


class IncrementalRunTest(unittest.TestCase):
    def test_second_run_serves_cache_and_fetches_only_the_delta(self):
        urls = []

        def first(url):
            urls.append(url)
            if "earthquake.usgs.gov" in url:
                return json.dumps(USGS_ONE).encode("utf-8")
            raise AssertionError("unexpected URL: %s" % (url,))

        with tempfile.TemporaryDirectory() as root:
            result = run_case("case1", profile(), START, END,
                              sources=("usgs",), root=root, fetcher=first)
            self.assertEqual(len(result.bundle.observations), 1)
            self.assertEqual(len(urls), 1)
            self.assertEqual(
                parse_qs(urlsplit(urls[0]).query)["starttime"][0], START)

            urls.clear()

            def second(url):
                urls.append(url)
                if "earthquake.usgs.gov" in url:
                    return b"{\"features\": []}"
                raise AssertionError("unexpected URL: %s" % (url,))

            later = "2026-10-25T01:00:00Z"
            again = run_case("case1", profile(), START, later,
                             sources=("usgs",), root=root, fetcher=second)
            # Nothing new upstream, yet the first observation still serves.
            self.assertEqual(len(again.bundle.observations), 1)
            self.assertEqual(len(urls), 1)
            self.assertEqual(
                parse_qs(urlsplit(urls[0]).query)["starttime"][0],
                "2026-10-24T22:59:59Z")
            report = again.report["stage1"]["cache"]
            self.assertTrue(report["enabled"])
            self.assertIn("fetched",
                          [entry["mode"] for entry in report["windows"]])
            self.assertEqual(report["served_observations"], 1)

    def test_outage_keeps_cached_observations(self):
        def good(url):
            if "earthquake.usgs.gov" in url:
                return json.dumps(USGS_ONE).encode("utf-8")
            raise AssertionError("unexpected URL: %s" % (url,))

        def broken(url):
            raise RuntimeError("network down")

        with tempfile.TemporaryDirectory() as root:
            run_case("case1", profile(), START, END,
                     sources=("usgs",), root=root, fetcher=good)
            later = "2026-10-25T01:00:00Z"
            again = run_case("case1", profile(), START, later,
                             sources=("usgs",), root=root, fetcher=broken)
            self.assertEqual(len(again.bundle.observations), 1)
            self.assertFalse(again.statuses[0].ok)
            report = again.report["stage1"]["cache"]
            self.assertTrue(any(entry["mode"] == "fetched"
                                and not entry["ok"]
                                for entry in report["windows"]))


def gdacs_obs(from_time="2026-10-20T00:00:00Z",
              to_time="2026-10-30T00:00:00Z"):
    return Observation("gdacs:e1", "gdacs", "e1", "hazard", "flood",
                       23.0, 120.0, from_time, "2026-10-24T12:00:00Z",
                       0, 0.6, "", "flood",
                       properties={"from_time": from_time,
                                   "to_time": to_time})


def gdelt_obs():
    return Observation("gdelt:a", "gdelt", "a", "news", "report",
                       23.0, 120.0, START, "2026-10-24T12:00:00Z",
                       0, 0.5, "", "news")


class CacheRegressionTest(unittest.TestCase):
    def test_gdacs_active_event_survives_window_prefilter(self):
        with tempfile.TemporaryDirectory() as root:
            cache = IntelCache(os.path.join(root, "cache.db"))
            try:
                cache.upsert([gdacs_obs()])
                got = cache.query("2026-10-25T00:00:00Z",
                                  "2026-10-26T00:00:00Z", ("gdacs",))
                self.assertEqual([o.observation_id for o in got], ["gdacs:e1"])
            finally:
                cache.close()

    def test_gdelt_rows_are_scoped_per_case(self):
        with tempfile.TemporaryDirectory() as root:
            cache = IntelCache(os.path.join(root, "cache.db"))
            try:
                cache.upsert([gdelt_obs()], scope="caseA")
                self.assertEqual(
                    [o.observation_id for o in cache.query(
                        START, END, ("gdelt",), scope="caseA")], ["gdelt:a"])
                self.assertEqual(cache.query(
                    START, END, ("gdelt",), scope="caseB"), [])
            finally:
                cache.close()

    def test_state_variant_separates_buffer_changes(self):
        with tempfile.TemporaryDirectory() as root:
            cache = IntelCache(os.path.join(root, "cache.db"))
            try:
                cache.mark("usgs", BBOX, cursor_until=END, variant="buf=1.000")
                self.assertEqual(cache.state(
                    "usgs", BBOX, variant="buf=1.000")["cursor_until"], END)
                self.assertEqual(cache.state(
                    "usgs", BBOX, variant="buf=2.000"), {})
            finally:
                cache.close()

    def test_firms_chunks_start_on_utc_midnight(self):
        start = datetime(2026, 10, 24, 10, tzinfo=UTC)
        end = start + timedelta(days=12)
        windows = fetch_windows("firms", start, end, None,
                                datetime(2026, 10, 25, tzinfo=UTC), False)
        self.assertEqual(windows[0][0], start)
        self.assertEqual(windows[-1][1], end)
        for window_start, window_end in windows:
            self.assertLessEqual((window_end - window_start).days, 5)
        for (_, prior_end), (next_start, _) in zip(windows, windows[1:]):
            self.assertEqual(prior_end, next_start)
            self.assertEqual((next_start.hour, next_start.minute,
                              next_start.second), (0, 0, 0))

    def test_cursor_past_window_refetches_when_cache_pruned(self):
        urls = []

        def fetcher(url):
            urls.append(url)
            if "earthquake.usgs.gov" in url:
                return json.dumps(USGS_ONE).encode("utf-8")
            raise AssertionError("unexpected URL: %s" % (url,))

        with tempfile.TemporaryDirectory() as root:
            run_case("case1", profile(), START, END,
                     sources=("usgs",), root=root, fetcher=fetcher)
            cache = IntelCache(os.path.join(root, "cache.db"))
            try:
                cache.mark("usgs", BBOX, cursor_until="2026-10-26T00:00:00Z",
                           fetched_at=END, variant="buf=0.000")
                with cache._lock, cache._conn:
                    cache._conn.execute("DELETE FROM observations")
            finally:
                cache.close()
            urls.clear()
            again = run_case("case1", profile(), START, END,
                             sources=("usgs",), root=root, fetcher=fetcher)
            self.assertEqual(len(again.bundle.observations), 1)
            self.assertTrue(any(
                parse_qs(urlsplit(url).query)["starttime"][0] == START
                for url in urls), urls)

    def test_degraded_fetch_does_not_advance_cursor(self):
        def fetcher(url):
            if "VIIRS_SNPP_NRT" in url:
                return b"latitude,longitude,acq_date\n23.0,120.0,2026-10-24\n"
            raise RuntimeError("sensor down")

        firms_profile = CaseIntelProfile(
            case_id="case1", name="case1", bbox=BBOX,
            domains=("hazard",), sources=("firms",),
            keywords=("taiwan",), lookback_hours=24.0)
        with tempfile.TemporaryDirectory() as root:
            result = run_case("case1", firms_profile, START, END,
                              sources=("firms",), root=root, fetcher=fetcher,
                              firms_map_key="KEY")
            self.assertEqual(result.statuses[0].state, "degraded")
            cache = IntelCache(os.path.join(root, "cache.db"))
            try:
                self.assertIsNone(cache.state(
                    "firms", BBOX, variant="buf=0.000").get("cursor_until"))
            finally:
                cache.close()
