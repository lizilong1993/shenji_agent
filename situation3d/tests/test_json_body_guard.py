"""Request-body structure guard (GP-03).

A body can be valid JSON and still be unsafe to keep.  Three cases, all
reachable from the public Funnel entry:

* ``json.loads`` accepts ``NaN`` / ``Infinity`` by default, and ``json.dumps``
  writes them back verbatim — that is not JSON, so the browser's
  ``JSON.parse`` throws on the response.  ``/api/resources`` copies ``bbox``
  and friends straight into the entry, so the bad value is persisted too.
* Deep nesting survives ``json.loads`` (the C scanner goes past 5000 levels)
  but blows the recursion limit on any later walk.
* ``__proto__`` / ``prototype`` / ``constructor`` keys mean nothing to Python
  and ride along into the frontend untouched.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
SERVER = os.path.join(SITUATION3D, "server")
for p in (SITUATION3D, SERVER):
    if p not in sys.path:
        sys.path.insert(0, p)

import serve  # noqa: E402


def free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class GuardUnitTest(unittest.TestCase):
    """The predicate itself: no server, no sockets."""

    def test_accepts_ordinary_bodies(self):
        for body in (
            {},
            {"case_id": "TW3", "verdict": "confirm"},
            {"bbox": [119.0, 22.0, 121.0, 24.0], "replace": False},
            {"nested": {"list": [1, 2.5, None, True, {"deep": "ok"}]}},
            [],
            "text",
            7,
            None,
        ):
            self.assertTrue(serve._json_body_ok(body), repr(body))

    def test_rejects_non_finite_numbers(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            self.assertFalse(serve._json_body_ok({"bbox": value}))
            self.assertFalse(serve._json_body_ok({"a": [{"b": value}]}))

    def test_rejects_prototype_keys(self):
        for key in ("__proto__", "prototype", "constructor"):
            self.assertFalse(serve._json_body_ok({key: {}}))
            self.assertFalse(serve._json_body_ok({"a": [{key: 1}]}))

    def test_rejects_excessive_depth(self):
        deep = current = {}
        for _ in range(serve._JSON_MAX_DEPTH + 5):
            current["n"] = {}
            current = current["n"]
        self.assertFalse(serve._json_body_ok(deep))

    def test_accepts_depth_at_the_limit(self):
        deep = current = {}
        for _ in range(serve._JSON_MAX_DEPTH - 1):
            current["n"] = {}
            current = current["n"]
        self.assertTrue(serve._json_body_ok(deep))

    def test_lists_count_towards_depth(self):
        deep = current = []
        for _ in range(serve._JSON_MAX_DEPTH + 5):
            nxt = []
            current.append(nxt)
            current = nxt
        self.assertFalse(serve._json_body_ok(deep))

    def test_guard_itself_does_not_recurse_to_death(self):
        # The guard must bail out on its own counter, not on Python's stack.
        deep = current = {}
        for _ in range(4000):
            current["n"] = {}
            current = current["n"]
        self.assertFalse(serve._json_body_ok(deep))


class GuardHttpTest(unittest.TestCase):
    """The guard has to sit on the real request path, not just exist."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.resources = os.path.join(self.tmp.name, "resources")
        os.makedirs(self.resources)
        self.procs = []

    def tearDown(self):
        for proc in self.procs:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        self.tmp.cleanup()

    def _start(self):
        port = free_port()
        env = dict(os.environ,
                   SITUATION_DIR=os.path.join(self.tmp.name, "situations"),
                   SITUATION_RESOURCES_DIR=self.resources,
                   PYTHONUNBUFFERED="1")
        os.makedirs(env["SITUATION_DIR"], exist_ok=True)
        proc = subprocess.Popen(
            [sys.executable, os.path.join(SERVER, "serve.py"),
             "--port", str(port), "--mode", "test",
             "--runs-dir", os.path.join(self.tmp.name, "runs")],
            cwd=SITUATION3D, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.procs.append(proc)
        for _ in range(60):
            try:
                req = urllib.request.Request(
                    "http://127.0.0.1:%d/api/health" % port)
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        return port
            except Exception:
                pass
            time.sleep(0.2)
        raise AssertionError("server did not start")

    def _post_raw(self, port, path, raw):
        req = urllib.request.Request(
            "http://127.0.0.1:%d%s" % (port, path), data=raw,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8")

    def _entry_body(self, extra):
        target = os.path.join(self.resources, "probe.json")
        with open(target, "w", encoding="utf-8") as fh:
            fh.write("{}")
        body = {"kind": "envpack", "name": "probe", "path": target}
        body.update(extra)
        return body

    def test_non_finite_note_is_refused_and_not_stored(self):
        # ``notes`` is Dict[str, Any] with no per-value validation, so it is
        # the field that actually carries a raw float into storage.  (bbox
        # and quality are already checked by ResourceEntry.validate.)
        port = self._start()
        raw = json.dumps(self._entry_body({"notes": {"score": 0.0}}))
        raw = raw.replace("0.0", "NaN")
        status, text = self._post_raw(port, "/api/resources", raw.encode())
        self.assertEqual(status, 400, text)
        # Nothing persisted, and every later read is still parseable JSON.
        with urllib.request.urlopen(
                "http://127.0.0.1:%d/api/resources" % port, timeout=10) as resp:
            listing = json.loads(resp.read().decode("utf-8"))
        self.assertEqual(listing.get("entries") or [], [])

    def test_prototype_key_is_refused(self):
        port = self._start()
        body = self._entry_body({"notes": {"__proto__": {"polluted": True}}})
        status, text = self._post_raw(port, "/api/resources",
                                     json.dumps(body).encode("utf-8"))
        self.assertEqual(status, 400, text)

    def test_deeply_nested_body_is_4xx_not_500(self):
        port = self._start()
        raw = b'{"kind":"envpack","name":"probe","notes":' + \
            b"[" * 40000 + b"1" + b"]" * 40000 + b"}"
        status, text = self._post_raw(port, "/api/resources", raw)
        self.assertIn(status, (400, 413), text[:200])

    def test_ordinary_resource_create_still_works(self):
        port = self._start()
        body = self._entry_body({"bbox": [119.0, 22.0, 121.0, 24.0],
                                 "notes": {"why": "regression"}})
        status, text = self._post_raw(port, "/api/resources",
                                     json.dumps(body).encode("utf-8"))
        self.assertEqual(status, 201, text)
        entry = json.loads(text)["entry"]
        self.assertEqual(entry["kind"], "envpack")
        self.assertEqual(entry["bbox"], [119.0, 22.0, 121.0, 24.0])
        self.assertEqual(entry["notes"], {"why": "regression"})


if __name__ == "__main__":
    unittest.main()