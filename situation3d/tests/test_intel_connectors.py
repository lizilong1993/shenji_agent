"""Offline tests for the P0 VGI connectors and contract fields."""

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

from intel import IntelBundle, IntelGateway  # noqa: E402
from intel.connectors import build_bundle, archive_raw, read_archive  # noqa: E402
from intel.connectors import firms, gdacs, usgs  # noqa: E402
from intel.connectors.http import FetchError, validate_url  # noqa: E402
from intel.normalize import normalize_usgs  # noqa: E402

FIXTURES = os.path.join(HERE, "fixtures", "intel")
START = "2026-10-24T00:00:00Z"
END = "2026-10-24T23:59:59Z"
FETCHED = "2026-10-24T12:00:00Z"
BBOX = (119.5, 22.5, 120.5, 23.5)

FIRMS_CSV = (
    "latitude,longitude,brightness,scan,track,acq_date,acq_time,"
    "satellite,instrument,confidence,version,frp,daynight\n"
    "22.9,120.0,320.1,1.0,1.0,2026-10-24,1200,VIIRS,NOAA-20,n,2.0NRT,12.0,D\n"
    "23.4,120.5,330.2,1.0,1.0,2026-10-24,1300,VIIRS,NOAA-20,n,2.0NRT,8.0,D\n"
)

EMPTY_FIRMS = "latitude,longitude,acq_date,acq_time\n"


def load(name):
    with open(os.path.join(FIXTURES, name), "r", encoding="utf-8") as handle:
        return json.load(handle)


def fake_fetch(url):
    if "earthquake.usgs.gov" in url:
        return json.dumps(load("usgs.json")).encode("utf-8")
    if "gdacs.org" in url:
        return json.dumps(load("gdacs.json")).encode("utf-8")
    if "firms.modaps.eosdis.nasa.gov" in url:
        if "VIIRS_SNPP_NRT" in url:
            return FIRMS_CSV.encode("utf-8")
        return EMPTY_FIRMS.encode("utf-8")
    raise AssertionError("unexpected URL: %s" % (url,))


class ConnectorTest(unittest.TestCase):
    def test_url_allow_list(self):
        allowed = ["example.com"]
        self.assertEqual(validate_url("https://example.com/a", allowed),
                         "https://example.com/a")
        for bad in ("http://example.com/a", "https://evil.example/a",
                    "https://example.com:8443/a",
                    "https://user:pass@example.com/a"):
            with self.assertRaises(FetchError):
                validate_url(bad, allowed)

    def test_url_builders(self):
        usgs_url = usgs.build_url(START, END, BBOX)
        self.assertIn("format=geojson", usgs_url)
        self.assertIn("minlatitude=22.500000", usgs_url)
        self.assertIn("maxlongitude=120.500000", usgs_url)
        gdacs_url = gdacs.build_url(START, END, BBOX)
        self.assertIn("fromdate=2026-10-24", gdacs_url)
        self.assertIn("eventlist=EQ,TC,FL,VO,WF,DR", gdacs_url)
        firms_url = firms.build_url("TESTKEY", "VIIRS_SNPP_NRT", BBOX,
                                    START, END)
        self.assertIn("VIIRS_SNPP_NRT", firms_url)
        self.assertIn("119.500000,22.500000,120.500000,23.500000", firms_url)
        self.assertIn("/1/2026-10-24", firms_url)
        self.assertEqual(firms.day_span("2026-10-24T12:00:00Z",
                                        "2026-10-25T12:00:00Z"), 2)

    def test_firms_rows_convert_date_and_time(self):
        rows = firms.prepare_rows(firms.parse_csv(FIRMS_CSV))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["acq_time"], "2026-10-24T12:00:00Z")
        self.assertEqual(rows[1]["acq_time"], "2026-10-24T13:00:00Z")
        self.assertEqual(rows[0]["latitude"], 22.9)

    def test_firms_bad_rows_are_skipped(self):
        text = ("latitude,longitude,acq_date,acq_time\n"
                "bad,120.0,2026-10-24,1200\n"
                "23.0,120.0,not-a-date,1200\n")
        self.assertEqual(firms.prepare_rows(firms.parse_csv(text)), [])

    def test_archive_roundtrip(self):
        with tempfile.TemporaryDirectory() as root:
            payload = b'{"ok": true}'
            record = archive_raw(root, "usgs", payload, FETCHED, ext="json")
            self.assertEqual(record.bytes, len(payload))
            self.assertEqual(read_archive(record.path), payload)
            self.assertTrue(os.path.exists(record.path))
            self.assertEqual(len(record.sha256), 64)

    def test_build_bundle_offline(self):
        with tempfile.TemporaryDirectory() as root:
            result = build_bundle(
                source_ids=("usgs", "gdacs", "firms"),
                bbox=BBOX,
                start=START,
                end=END,
                fetched_at=FETCHED,
                scenario_id="s1",
                archive_root=root,
                fetcher=fake_fetch,
                firms_map_key="TESTKEY",
            )
            bundle = result.bundle
            self.assertEqual(len(bundle.observations), 6)
            self.assertEqual(bundle.domain, "hazard")
            self.assertEqual(bundle.bbox, tuple(float(v) for v in BBOX))
            self.assertEqual(bundle.time_window, (START, END))
            self.assertEqual(bundle.source_allowlist, ("usgs", "gdacs", "firms"))
            self.assertEqual({s.source_id for s in result.statuses},
                             {"usgs", "gdacs", "firms"})
            counts = {s.source_id: s.count for s in result.statuses}
            self.assertEqual(counts, {"usgs": 3, "gdacs": 1, "firms": 2})
            self.assertTrue(all(s.ok for s in result.statuses))
            self.assertEqual(len(result.archives), 6)
            again = build_bundle(
                source_ids=("usgs", "gdacs", "firms"),
                bbox=BBOX,
                start=START,
                end=END,
                fetched_at=FETCHED,
                scenario_id="s1",
                archive_root=root,
                fetcher=fake_fetch,
                firms_map_key="TESTKEY",
            )
            self.assertEqual(bundle.hash, again.bundle.hash)

    def test_bundle_and_observation_contract_fields(self):
        with tempfile.TemporaryDirectory() as root:
            result = build_bundle(
                source_ids=("usgs",), bbox=BBOX, start=START, end=END,
                fetched_at=FETCHED, archive_root=root, fetcher=fake_fetch)
        payload = result.bundle.as_dict()
        for key in ("domain", "bbox", "time_window", "source_allowlist"):
            self.assertIn(key, payload)
        observation = payload["observations"][0]
        self.assertEqual(observation["schema_version"], "situation3d.intel.v1")
        self.assertEqual(observation["license"], "U.S. public domain")
        self.assertIn("provenance", observation)
        self.assertEqual(observation["provenance"]["parser_version"],
                         "vgi-connectors-1")
        self.assertIn("geometry", observation)

    def test_partial_source_failure_is_reported(self):
        def flaky(url):
            if "gdacs.org" in url:
                raise FetchError("upstream unavailable")
            return fake_fetch(url)

        result = build_bundle(
            source_ids=("usgs", "gdacs"), bbox=BBOX, start=START, end=END,
            fetched_at=FETCHED, fetcher=flaky)
        self.assertEqual(len(result.bundle.observations), 3)
        statuses = {s.source_id: s for s in result.statuses}
        self.assertTrue(statuses["usgs"].ok)
        self.assertFalse(statuses["gdacs"].ok)
        self.assertEqual(statuses["gdacs"].state, "unavailable")
        self.assertEqual(result.errors[0]["source_id"], "gdacs")

    def test_missing_firms_key_is_reported(self):
        result = build_bundle(
            source_ids=("firms",), bbox=BBOX, start=START, end=END,
            fetched_at=FETCHED, fetcher=fake_fetch, firms_map_key="")
        self.assertEqual(len(result.bundle.observations), 0)
        self.assertFalse(result.statuses[0].ok)
        self.assertIn("FIRMS_MAP_KEY", result.statuses[0].error)

    def test_fused_cues_use_observation_kind(self):
        observations = normalize_usgs(load("usgs.json"), FETCHED)
        bundle = IntelBundle("b1", "s1", observations, START)
        batch = IntelGateway(bundle, fusion=(10000.0, 3600.0)).consume(
            1, 0.0, -90000.0, 400.0)
        self.assertTrue(batch.cues)
        self.assertTrue(all(cue.cue_kind == "observation" for cue in batch.cues))
        self.assertTrue(all(cue.provenance.get("fused") is True
                            for cue in batch.cues))

    def test_bundle_derives_missing_contract_fields(self):
        observations = normalize_usgs(load("usgs.json"), FETCHED)
        bundle = IntelBundle("b1", "s1", observations, START)
        payload = bundle.as_dict()
        self.assertEqual(payload["domain"], "multi")
        self.assertEqual(payload["source_allowlist"], ["usgs"])
        self.assertEqual(payload["time_window"][0], START)
        self.assertLessEqual(payload["bbox"][0], payload["bbox"][2])

    def test_bundle_roundtrip_from_dict(self):
        observations = normalize_usgs(load("usgs.json"), FETCHED)
        bundle = IntelBundle("b1", "s1", observations, START,
                             bbox=(119.0, 22.0, 121.0, 24.0),
                             time_window=(START, END),
                             source_allowlist=("usgs",))
        restored = IntelBundle.from_dict(bundle.as_dict())
        self.assertEqual(restored.hash, bundle.hash)
        self.assertEqual(restored.envelope_hash, bundle.envelope_hash)
        self.assertEqual(restored.bbox, bundle.bbox)
        self.assertEqual(restored.source_allowlist, bundle.source_allowlist)

    def test_ongoing_gdacs_event_is_kept(self):
        payload = {
            "type": "FeatureCollection",
            "features": [{
                "geometry": {"type": "Point", "coordinates": [120.1, 23.0]},
                "properties": {
                    "eventid": "ongoing",
                    "eventtype": "TC",
                    "eventname": "ONGOING-26",
                    "alertlevel": "Red",
                    "fromdate": "2026-10-20T00:00:00",
                    "todate": "2026-10-26T00:00:00",
                    "url": {"report": "https://example.invalid/ongoing"},
                },
            }],
        }

        def fetch(url):
            return json.dumps(payload).encode("utf-8")

        result = build_bundle(
            source_ids=("gdacs",), bbox=BBOX,
            start="2026-10-24T00:00:00Z", end="2026-10-26T23:59:59Z",
            fetched_at=FETCHED, fetcher=fetch)
        self.assertEqual(len(result.bundle.observations), 1)
        observation = result.bundle.observations[0]
        self.assertEqual(observation.event_time, "2026-10-24T00:00:00Z")
        self.assertEqual(observation.properties["event_start_time"],
                         "2026-10-20T00:00:00Z")
        self.assertEqual(observation.properties["from_time"],
                         "2026-10-20T00:00:00Z")
        self.assertEqual(observation.properties["to_time"],
                         "2026-10-26T00:00:00Z")
        self.assertEqual(observation.source_url,
                         "https://example.invalid/ongoing")

    def test_unknown_source_is_rejected_before_fetch(self):
        calls = []

        def fetch(url):
            calls.append(url)
            return b"{}"

        with self.assertRaises(ValueError):
            build_bundle(
                source_ids=("usgs", "bogus"), bbox=BBOX,
                start=START, end=END, fetcher=fetch)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
