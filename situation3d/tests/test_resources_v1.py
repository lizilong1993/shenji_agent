"""Resource registry contracts and store tests (P0)."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from resources.catalog import default_criticality, kind_spec, validate_kind  # noqa: E402
from resources.model import (ResourceDecision, ResourceEntry,  # noqa: E402
                             ResourcePlan, hash_path)
from resources.scanners import scan_envpack, scan_repo  # noqa: E402
from resources.store import (ResourceConflict, ResourceNotFound,  # noqa: E402
                             ResourceStore)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False)


def make_envpack(root, name="p1", built="2026-10-23", fallback=False):
    write_json(os.path.join(root, "manifest.json"), {
        "schema": "msgeo.envpack.v2",
        "pack_id": name,
        "theatre": "test",
        "built": built,
        "bbox": [120.0, 22.9, 120.2, 23.0],
        "layers": {
            "terrain": {"format": "grid-json", "path": "terrain/grid.json",
                        "source": "fallback-basic" if fallback else "copernicus-glo30"},
            "roads": {"format": "graph-json", "path": "roads/graph.json",
                      "source": "offline-cache"},
        },
        "provenance": {"mode": "offline", "osm": "offline-cache",
                       "dem": "fallback-basic" if fallback else "copernicus-glo30",
                       "fallback": bool(fallback)},
    })
    write_json(os.path.join(root, "terrain", "grid.json"),
               {"lon0": 120.0, "lat0": 22.9, "values": [[1.0, 2.0]]})
    write_json(os.path.join(root, "roads", "graph.json"),
               {"nodes": {"a": [120.0, 22.9]}, "edges": []})
    return root


class EnvpackScanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="res-scan-")
        self.pack = make_envpack(os.path.join(self.tmp, "pack"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_metadata(self):
        entry = scan_envpack(self.pack)
        self.assertEqual(entry.entry_id, "envpack:p1")
        self.assertEqual(entry.kind, "envpack")
        self.assertEqual(entry.schema, "msgeo.envpack.v2")
        self.assertEqual(entry.as_of, "2026-10-23")
        self.assertEqual(entry.quality, "measured")
        self.assertEqual(entry.bbox, (120.0, 22.9, 120.2, 23.0))
        self.assertIn("msgeo.envpack.v2", entry.source)
        self.assertEqual(len(entry.content_hash), 64)
        self.assertGreater(entry.size, 0)
        self.assertTrue(os.path.isabs(entry.path))

    def test_fallback_is_derived_quality(self):
        pack = make_envpack(os.path.join(self.tmp, "pack2"), name="p2",
                            fallback=True)
        entry = scan_envpack(pack)
        self.assertEqual(entry.quality, "derived")

    def test_hash_changes_with_layer_content(self):
        first = scan_envpack(self.pack)
        write_json(os.path.join(self.pack, "roads", "graph.json"),
                   {"nodes": {"a": [120.0, 22.9], "b": [120.1, 22.95]},
                    "edges": []})
        second = scan_envpack(self.pack)
        self.assertNotEqual(first.content_hash, second.content_hash)

    def test_rejects_non_envpack(self):
        write_json(os.path.join(self.tmp, "bad", "manifest.json"),
                   {"schema": "other.v1"})
        with self.assertRaises(ValueError):
            scan_envpack(os.path.join(self.tmp, "bad"))


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="res-store-")
        self.pack = make_envpack(os.path.join(self.tmp, "pack"))
        self.store = ResourceStore(os.path.join(self.tmp, "index"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_roundtrip_and_persistence(self):
        entry = self.store.register(scan_envpack(self.pack))
        again = ResourceStore(os.path.join(self.tmp, "index")).get(entry.entry_id)
        self.assertEqual(again.content_hash, entry.content_hash)
        self.assertEqual(again.size, entry.size)
        self.assertEqual(len(self.store.entries("envpack")), 1)
        self.assertEqual(self.store.entries("platforms"), [])

    def test_verify_ok_then_mismatch_then_missing(self):
        entry = self.store.register(scan_envpack(self.pack))
        self.assertEqual(self.store.verify(entry.entry_id)["status"], "ok")

        layer = os.path.join(self.pack, "terrain", "grid.json")
        with open(layer, "r+b") as fh:  # same size, different bytes
            fh.seek(0)
            fh.write(b"[")
        result = self.store.verify(entry.entry_id)
        self.assertFalse(result["ok"])
        self.assertIn(result["status"], ("hash_mismatch", "size_mismatch"))

        with open(layer, "ab") as fh:
            fh.write(b" ")
        self.assertEqual(self.store.verify(entry.entry_id)["status"],
                         "size_mismatch")

        shutil.rmtree(self.pack)
        self.assertEqual(self.store.verify(entry.entry_id)["status"], "missing")

    def test_conflict_and_replace(self):
        first = self.store.register(scan_envpack(self.pack))
        write_json(os.path.join(self.pack, "roads", "graph.json"),
                   {"nodes": {}, "edges": [1]})
        changed = scan_envpack(self.pack)
        with self.assertRaises(ResourceConflict):
            self.store.register(changed)
        self.store.set_pin(first.entry_id, True)
        replaced = self.store.register(changed, replace=True)
        self.assertEqual(replaced.content_hash, changed.content_hash)
        self.assertTrue(replaced.pinned)

    def test_duplicate_id_with_same_content_merges_path(self):
        entry = self.store.register(scan_envpack(self.pack))
        copy = make_envpack(os.path.join(self.tmp, "copy"), name="p1")
        merged = self.store.register(scan_envpack(copy))
        self.assertEqual(merged.content_hash, entry.content_hash)
        self.assertEqual(merged.path, os.path.abspath(copy))

    def test_pin_blocks_remove(self):
        entry = self.store.register(scan_envpack(self.pack))
        self.store.set_pin(entry.entry_id, True)
        with self.assertRaises(ResourceConflict):
            self.store.remove(entry.entry_id)
        self.store.remove(entry.entry_id, force=True)
        with self.assertRaises(ResourceNotFound):
            self.store.get(entry.entry_id)

    def test_relative_path_rejected(self):
        entry = scan_envpack(self.pack)
        bad = ResourceEntry(entry_id=entry.entry_id, kind="envpack",
                            path="relative/path", content_hash=entry.content_hash,
                            size=entry.size)
        with self.assertRaises(ValueError):
            self.store.register(bad)

    def test_index_hash_changes_with_membership(self):
        before = self.store.index_hash()
        self.store.register(scan_envpack(self.pack))
        self.assertNotEqual(before, self.store.index_hash())


class CatalogTest(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(default_criticality("envpack"), "critical")
        self.assertEqual(default_criticality("intel_bundle"), "advisory")
        self.assertEqual(default_criticality("tiles3d"), "cosmetic")
        self.assertEqual(kind_spec("envpack").ttl_days, 30.0)

    def test_unknown_kind(self):
        with self.assertRaises(KeyError):
            validate_kind("not-a-kind")


class RepoScanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="res-repo-")
        root = self.tmp
        make_envpack(os.path.join(root, "envpacks", "pack"))
        write_json(os.path.join(root, "platforms", "wheeled.json"), {
            "schema": "msplatform.v1", "key": "wheeled",
            "identity": {"name_zh": "测试平台"}})
        write_json(os.path.join(root, "rules", "data", "manifest.json"), {
            "schema": "msgeo.rules.v1", "source_date": "2026-09-15 (snap)",
            "generated": "2026-10-21", "files": ["wpdata.json"]})
        write_json(os.path.join(root, "rules", "data", "wpdata.json"), {"x": 1})
        write_json(os.path.join(root, "engine", "rolecards", "data",
                                "classes.json"), {
            "schema": "situation3d.rolecard-classes.v1", "classes": []})
        write_json(os.path.join(root, "engine", "data", "roads", "tw.json"), {
            "schema": "situation3d.roads.v1", "theatre": "tw",
            "osm_base_utc": "2026-07-24T11:04:51Z",
            "fetched_utc": "2026-09-26T05:42:20Z",
            "bbox": [118.0, 22.5, 120.5, 26.0], "classes": []})

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_discovers_all_kinds(self):
        entries, errors = scan_repo(self.tmp)
        self.assertEqual(errors, [])
        by_kind = {}
        for entry in entries:
            by_kind.setdefault(entry.kind, []).append(entry)
        self.assertEqual(sorted(by_kind), ["case_roads", "envpack", "platforms",
                                           "rolecards", "weapons"])
        roads = by_kind["case_roads"][0]
        self.assertEqual(roads.entry_id, "case_roads:tw")
        self.assertEqual(roads.as_of, "2026-07-24")
        self.assertEqual(roads.fetched_at, "2026-09-26T05:42:20Z")
        self.assertEqual(roads.license, "ODbL-1.0")

    def test_malformed_artifact_reported_not_fatal(self):
        write_json(os.path.join(self.tmp, "envpacks", "bad", "manifest.json"),
                   {"schema": "wrong"})
        entries, errors = scan_repo(self.tmp)
        self.assertTrue(entries)
        self.assertEqual(len(errors), 1)
        self.assertIn("bad", errors[0]["path"])


class PlanContractTest(unittest.TestCase):
    def test_decision_timestamps_do_not_break_hash(self):
        entry = ResourceEntry(
            entry_id="envpack:p1", kind="envpack", path="/tmp/p1",
            content_hash="a" * 64, size=1)
        d1 = ResourceDecision(action="use_cached_stale", entry_id="envpack:p1")
        d2 = ResourceDecision(action="use_cached_stale", entry_id="envpack:p1")
        p1 = ResourcePlan(plan_id="p", scenario_date="2026-10-26",
                          entries=(entry,), decisions=(d1,))
        p2 = ResourcePlan(plan_id="p", scenario_date="2026-10-26",
                          entries=(entry,), decisions=(d2,))
        self.assertEqual(p1.hash(), p2.hash())

    def test_plan_hash_is_content_sensitive(self):
        entry = ResourceEntry(
            entry_id="envpack:p1", kind="envpack", path="/tmp/p1",
            content_hash="a" * 64, size=1, version="a", as_of="2026-10-23")
        plan = ResourcePlan(plan_id="plan-1", scenario_date="2026-10-26",
                            entries=(entry,))
        other = ResourceEntry(
            entry_id="envpack:p1", kind="envpack", path="/tmp/p1",
            content_hash="b" * 64, size=1, version="b", as_of="2026-10-24")
        changed = ResourcePlan(plan_id="plan-1", scenario_date="2026-10-26",
                               entries=(other,))
        self.assertNotEqual(plan.hash(), changed.hash())
        payload = plan.as_dict()
        self.assertEqual(payload["schema"], "s3d.resources.plan.v1")
        self.assertEqual(len(payload["plan_hash"]), 64)


class HashPathTest(unittest.TestCase):
    def test_directory_hash_ignores_file_creation_order(self):
        root = tempfile.mkdtemp(prefix="res-hash-")
        try:
            a = os.path.join(root, "a")
            b = os.path.join(root, "b")
            for target, order in ((a, ("x", "y")), (b, ("y", "x"))):
                os.makedirs(target)
                for name in order:
                    with open(os.path.join(target, name), "wb") as fh:
                        fh.write(name.encode())
            self.assertEqual(hash_path(a), hash_path(b))
        finally:
            shutil.rmtree(root, ignore_errors=True)



class SqliteStoreCrudTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="res-sqlite-")
        self.store = ResourceStore(os.path.join(self.tmp, "index"))
        self.artifact = os.path.join(self.tmp, "artifact.bin")
        with open(self.artifact, "wb") as fh:
            fh.write(b"first version")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_sqlite_db_and_crud_roundtrip(self):
        entry = self.store.create_entry(
            "satellite_imagery", "demo", self.artifact,
            license="test-license", quality="derived",
            bbox=[120.0, 22.9, 120.2, 23.0], notes={"role": "test"})
        self.assertEqual(entry.entry_id, "satellite_imagery:demo")
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, "index",
                                                    "resources.db")))
        self.assertEqual(self.store.stats()["satellite_imagery"]["count"], 1)

        updated = self.store.update(entry.entry_id, {
            "license": "updated-license",
            "as_of": "2026-10-27",
            "notes": {"role": "edited"},
            "pinned": True,
        })
        self.assertEqual(updated.license, "updated-license")
        self.assertEqual(updated.as_of, "2026-10-27")
        self.assertTrue(updated.pinned)
        self.assertEqual(updated.notes, {"role": "edited"})

        with self.assertRaises(ResourceConflict):
            self.store.remove(entry.entry_id)
        self.store.remove(entry.entry_id, force=True)
        with self.assertRaises(ResourceNotFound):
            self.store.get(entry.entry_id)

    def test_update_path_rehashes(self):
        entry = self.store.create_entry("landcover", "lc", self.artifact)
        other = os.path.join(self.tmp, "other.bin")
        with open(other, "wb") as fh:
            fh.write(b"different content")
        updated = self.store.update(entry.entry_id, {"path": other})
        self.assertEqual(updated.path, os.path.abspath(other))
        self.assertNotEqual(updated.content_hash, entry.content_hash)
        self.assertEqual(updated.size, len(b"different content"))

    def test_migration_skips_bad_rows(self):
        pack = make_envpack(os.path.join(self.tmp, "pack"), name="good")
        entry = scan_envpack(pack)
        write_json(os.path.join(self.tmp, "index", "index.json"), {
            "schema": "s3d.resources.index.v1",
            "entries": {
                entry.entry_id: entry.as_dict(),
                "satellite_imagery:bad": {
                    "entry_id": "satellite_imagery:bad",
                    "kind": "satellite_imagery",
                    "path": "/tmp/nope",
                    "content_hash": "not-a-hash",
                    "size": 1,
                },
            },
        })
        store = ResourceStore(os.path.join(self.tmp, "index"))
        self.assertEqual(store.get(entry.entry_id).content_hash,
                         entry.content_hash)
        with self.assertRaises(ResourceNotFound):
            store.get("satellite_imagery:bad")

    def test_index_json_is_migrated_once(self):
        pack = make_envpack(os.path.join(self.tmp, "pack"), name="legacy")
        entry = scan_envpack(pack)
        write_json(os.path.join(self.tmp, "index", "index.json"), {
            "schema": "s3d.resources.index.v1",
            "entries": {entry.entry_id: entry.as_dict()},
        })
        migrated = ResourceStore(os.path.join(self.tmp, "index")).get(
            entry.entry_id)
        self.assertEqual(migrated.content_hash, entry.content_hash)
        # The JSON is now a backup; changing it must not affect the database.
        with open(os.path.join(self.tmp, "index", "index.json"), "w",
                  encoding="utf-8") as fh:
            fh.write("{}")
        again = ResourceStore(os.path.join(self.tmp, "index")).get(
            entry.entry_id)
        self.assertEqual(again.content_hash, entry.content_hash)

if __name__ == "__main__":
    unittest.main()
