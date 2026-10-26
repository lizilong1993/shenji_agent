"""Offline tests for case profiles, global sources and runtime filtering."""

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

from intel import (CaseIntelProfile, IntelBundle, IntelGateway,  # noqa: E402
                   Observation, load_case_profile)
from intel.fusion import fuse  # noqa: E402
from intel.connectors import adsb, eonet, gdelt, build_bundle  # noqa: E402
from intel.connectors.http import FetchError  # noqa: E402
from intel.normalize import (normalize_adsb, normalize_eonet,  # noqa: E402
                             normalize_gdelt)

START = "2026-10-24T00:00:00Z"
END = "2026-10-24T23:59:59Z"
FETCHED = "2026-10-24T12:00:00Z"
TAIWAN = (118.8, 21.8, 121.0, 24.2)


def observation(obs_id, lat, lon, title="", domain="hazard",
                source_id="usgs", event_time=START, geo_precision="exact",
                is_centroid=False, **properties):
    return Observation(
        observation_id=obs_id,
        source_id=source_id,
        source_native_id=obs_id,
        domain=domain,
        type=properties.pop("type", "event"),
        lat=lat,
        lon=lon,
        event_time=event_time,
        ingest_time=FETCHED,
        ingest_seq=0,
        confidence=0.9,
        source_url="",
        title=title,
        geo_precision=geo_precision,
        is_centroid=is_centroid,
        properties=properties,
    )


def fake_fetch(url):
    if "api.adsb.lol" in url:
        return json.dumps({"ac": [{
            "hex": "abc123", "flight": "TEST1", "lat": 23.5, "lon": 120.5,
            "alt_baro": 10000, "gs": 200, "track": 90, "dbFlags": 1,
            "t": "A320",
        }]}).encode("utf-8")
    if "eonet.gsfc.nasa.gov" in url:
        return json.dumps({"events": [{
            "id": "EONET_1",
            "title": "Typhoon near Taiwan",
            "categories": [{"id": "severeStorms", "title": "Severe Storms"}],
            "sources": [{"url": "https://example.invalid/eonet"}],
            "geometry": [{
                "type": "Point", "coordinates": [120.5, 23.5],
                "date": "2026-10-24T12:00:00Z", "magnitudeValue": 50,
            }],
        }]}).encode("utf-8")
    if "api.gdeltproject.org" in url:
        return json.dumps({"articles": [
            {"title": "Taiwan Strait exercise", "url": "https://example.invalid/1",
             "seendate": "20261024T120000Z", "domain": "example",
             "sourcecountry": "Taiwan", "language": "English"},
            {"title": "Weather in Paris", "url": "https://example.invalid/2",
             "seendate": "20261024T120000Z", "domain": "example",
             "sourcecountry": "France", "language": "English"},
        ]}).encode("utf-8")
    raise AssertionError("unexpected URL: %s" % (url,))


class CaseProfileTest(unittest.TestCase):
    def test_builtin_profiles_load(self):
        taiwan = load_case_profile("taiwan_landing")
        self.assertEqual(taiwan.case_id, "taiwan_landing")
        self.assertEqual(taiwan.bbox, TAIWAN)
        self.assertIn("adsb", taiwan.sources)
        india = load_case_profile("india_china_west")
        self.assertEqual(india.bbox, (77.0, 31.8, 81.2, 35.7))
        self.assertIn("eonet", india.sources)

    def test_profile_matching_bbox_domain_keyword_country(self):
        profile = load_case_profile("taiwan_landing")
        inside = observation("ok", 23.5, 120.5, "Taiwan Strait patrol",
                             sourcecountry="Taiwan")
        outside = observation("out", 30.0, 120.5, "Taiwan Strait patrol")
        wrong_domain = observation("domain", 23.5, 120.5, "Taiwan Strait",
                                   domain="health")
        wrong_source = observation("source", 23.5, 120.5, "Taiwan Strait",
                                   source_id="mystery")
        wrong_keyword = observation("keyword", 23.5, 120.5, "Weather in Paris",
                                    source_id="gdelt", geo_precision="unknown",
                                    is_centroid=True, sourcecountry="France")
        self.assertTrue(profile.matches(inside))
        self.assertFalse(profile.matches(outside))
        self.assertFalse(profile.matches(wrong_domain))
        self.assertFalse(profile.matches(wrong_source))
        self.assertFalse(profile.matches(wrong_keyword))

    def test_profile_overrides_and_hash(self):
        profile = load_case_profile("taiwan_landing")
        changed = profile.with_overrides(bbox=(10.0, 10.0, 11.0, 11.0),
                                         domains=("maritime",),
                                         buffer_km=100.0)
        self.assertEqual(changed.bbox, (10.0, 10.0, 11.0, 11.0))
        self.assertEqual(changed.domains, ("maritime",))
        self.assertNotEqual(profile.filter_hash, changed.filter_hash)

    def test_expanded_bbox_respects_buffer(self):
        profile = load_case_profile("taiwan_landing")
        expanded = profile.expanded_bbox
        self.assertLess(expanded[0], profile.bbox[0])
        self.assertGreater(expanded[2], profile.bbox[2])

    def test_buffer_survives_bundle_prefilter(self):
        profile = load_case_profile("taiwan_landing").with_overrides(
            sources=("usgs",))
        payload = {"features": [{
            "id": "buffered",
            "geometry": {"type": "Point", "coordinates": [121.3, 23.5]},
            "properties": {"time": 1792800060000, "mag": 4.5,
                           "place": "buffered edge"},
        }]}

        def fetch(url):
            return json.dumps(payload).encode("utf-8")

        result = build_bundle(start=START, end=END, fetched_at=FETCHED,
                              fetcher=fetch, case_profile=profile)
        self.assertEqual(len(result.bundle.observations), 1)
        self.assertEqual(result.bundle.observations[0].observation_id,
                         "usgs:buffered")

    def test_eonet_uses_point_inside_aoi_not_latest_global(self):
        payload = {"events": [{
            "id": "EONET_cross",
            "title": "Storm crossing the strait",
            "categories": [{"id": "severeStorms"}],
            "geometry": [
                {"type": "Point", "coordinates": [120.5, 23.5],
                 "date": "2026-10-24T10:00:00Z"},
                {"type": "Point", "coordinates": [124.0, 27.0],
                 "date": "2026-10-24T18:00:00Z"},
            ],
        }]}
        observations = normalize_eonet(payload, FETCHED, bbox=TAIWAN,
                                       start=START, end=END)
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].lon, 120.5)


class GlobalSourceTest(unittest.TestCase):
    def test_adsb_points_and_normalization(self):
        self.assertEqual(len(adsb.sample_points(TAIWAN)), 1)
        self.assertEqual(len(adsb.sample_points(
            (77.0, 31.8, 81.2, 35.7))), 1)
        with self.assertRaises(ValueError):
            adsb.sample_points((-180.0, -90.0, 180.0, 90.0))
        payload = json.loads(fake_fetch("https://api.adsb.lol/x").decode())
        observations = normalize_adsb(payload, FETCHED)
        self.assertEqual(len(observations), 1)
        item = observations[0]
        self.assertEqual(item.domain, "aviation")
        self.assertEqual(item.properties["icao24"], "abc123")
        self.assertTrue(item.properties["military"])

    def test_eonet_url_and_point_selection(self):
        url = eonet.build_url(START, END, TAIWAN)
        self.assertIn("status=open", url)
        self.assertNotIn("bbox=", url)
        payload = json.loads(fake_fetch("https://eonet.gsfc.nasa.gov/x").decode())
        observations = normalize_eonet(payload, FETCHED, bbox=TAIWAN,
                                       start=START, end=END)
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].type, "severestorms")
        self.assertEqual(observations[0].lon, 120.5)

    def test_gdelt_uses_case_centroid(self):
        url = gdelt.build_url(["taiwan", "strait"], START, END)
        self.assertIn("mode=artlist", url)
        self.assertIn("startdatetime=20261024000000", url)
        cjk_url = gdelt.build_url(["台湾", "台南"], START, END)
        self.assertIn("%E5%8F%B0", cjk_url)
        payload = json.loads(fake_fetch(
            "https://api.gdeltproject.org/x").decode())
        observations = normalize_gdelt(payload, FETCHED, (120.0, 23.0))
        self.assertEqual(len(observations), 2)
        self.assertEqual(observations[0].geo_precision, "unknown")
        self.assertTrue(observations[0].is_centroid)
        self.assertEqual(observations[0].lat, 23.0)

    def test_build_bundle_with_case_profile_and_new_sources(self):
        profile = load_case_profile("taiwan_landing").with_overrides(
            sources=("adsb", "eonet", "gdelt"))
        with tempfile.TemporaryDirectory() as root:
            result = build_bundle(
                start=START, end=END, fetched_at=FETCHED,
                archive_root=root, fetcher=fake_fetch,
                case_profile=profile)
        self.assertEqual(len(result.bundle.observations), 3)
        self.assertEqual(result.bundle.domain, "multi")
        self.assertEqual(result.bundle.source_allowlist,
                         ("adsb", "eonet", "gdelt"))
        self.assertEqual({s.source_id: s.count for s in result.statuses},
                         {"adsb": 1, "eonet": 1, "gdelt": 1})
        gdelt_obs = [obs for obs in result.bundle.observations
                     if obs.source_id == "gdelt"]
        self.assertEqual(len(gdelt_obs), 1)  # Paris article filtered by keywords
        self.assertIn("Taiwan", gdelt_obs[0].title)

    def test_explicit_bbox_overrides_profile(self):
        profile = load_case_profile("taiwan_landing").with_overrides(
            sources=("usgs",))
        payload = {"features": [{
            "id": "elsewhere",
            "geometry": {"type": "Point", "coordinates": [10.5, 10.5]},
            "properties": {"time": 1792800060000, "mag": 5.0,
                           "place": "elsewhere"},
        }]}

        def fetch(url):
            return json.dumps(payload).encode("utf-8")

        result = build_bundle(
            source_ids=("usgs",), bbox=(10.0, 10.0, 11.0, 11.0),
            start=START, end=END, fetched_at=FETCHED,
            fetcher=fetch, case_profile=profile)
        self.assertEqual(len(result.bundle.observations), 1)
        self.assertEqual(result.bundle.observations[0].observation_id,
                         "usgs:elsewhere")

    def test_profile_from_dict_null_defaults(self):
        profile = CaseIntelProfile.from_dict({
            "case_id": "nulls", "name": "nulls",
            "bbox": [1.0, 2.0, 3.0, 4.0],
            "buffer_km": None, "lookback_hours": None, "max_items": None,
        })
        self.assertEqual(profile.buffer_km, 0.0)
        self.assertEqual(profile.lookback_hours, 24.0)
        self.assertEqual(profile.max_items, 5000)

    def test_eonet_storm_fuses_with_gdacs_cyclone(self):
        eonet_obs = observation("eonet:storm", 23.5, 120.5,
                                type="severestorms", source_id="eonet")
        gdacs_obs = observation("gdacs:storm", 23.51, 120.51,
                                type="tc", source_id="gdacs")
        events = fuse([eonet_obs, gdacs_obs], radius_m=10000, window_s=3600)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].type, "cyclone")

    def test_max_items_keeps_newest(self):
        profile = load_case_profile("taiwan_landing").with_overrides(
            sources=("usgs",), max_items=2)
        payload = {"features": [
            {"id": "t1", "geometry": {"type": "Point",
                                      "coordinates": [120.1, 23.0]},
             "properties": {"time": 1792803600000, "place": "01:00"}},
            {"id": "t5", "geometry": {"type": "Point",
                                      "coordinates": [120.2, 23.1]},
             "properties": {"time": 1792818000000, "place": "05:00"}},
            {"id": "t9", "geometry": {"type": "Point",
                                      "coordinates": [120.3, 23.2]},
             "properties": {"time": 1792832400000, "place": "09:00"}},
        ]}

        def fetch(url):
            return json.dumps(payload).encode("utf-8")

        result = build_bundle(start=START, end=END, fetched_at=FETCHED,
                              fetcher=fetch, case_profile=profile)
        self.assertEqual([obs.observation_id for obs in result.bundle.observations],
                         ["usgs:t5", "usgs:t9"])
        self.assertTrue(result.truncated)

    def test_optional_gdelt_failure_is_a_warning(self):
        profile = load_case_profile("taiwan_landing").with_overrides(
            sources=("gdelt",))

        def fetch(url):
            raise FetchError("http 429 from api.gdeltproject.org")

        result = build_bundle(start=START, end=END, fetched_at=FETCHED,
                              fetcher=fetch, case_profile=profile)
        self.assertEqual(result.errors, ())
        self.assertEqual(len(result.warnings), 1)
        self.assertEqual(result.statuses[0].state, "unavailable")


class GatewayFilterTest(unittest.TestCase):
    def test_gateway_filters_cues_and_can_change_aoi(self):
        observations = [
            observation("inside", 23.5, 120.5, event_time=START),
            observation("outside", 30.0, 120.5, event_time=START),
        ]
        bundle = IntelBundle("b1", "s1", observations, START)
        profile = CaseIntelProfile(
            case_id="x", name="x", bbox=TAIWAN, domains=("hazard",))
        gateway = IntelGateway(bundle, case_filter=profile)
        batch = gateway.consume(1, 0.0, 0.0, 400.0)
        self.assertEqual([cue.observation_id for cue in batch.cues],
                         ["inside"])

        wider = profile.with_overrides(bbox=(115.0, 20.0, 125.0, 35.0))
        gateway.set_case_filter(wider)
        batch = gateway.consume(1, 0.0, 0.0, 400.0)
        self.assertEqual([cue.observation_id for cue in batch.cues],
                         ["outside"])
        self.assertEqual(gateway.filter_hash, wider.filter_hash)


if __name__ == "__main__":
    unittest.main()
