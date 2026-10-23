"""HTTP API integration tests for the test/demo server (TW2 D2)."""

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

from run_store import RunStore  # noqa: E402

CASE = "taiwan_dense_transfer_v1"


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def request(port, path, method="GET", body=None):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path),
                                 data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


class ServerApiTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.runs = os.path.join(self.tmp.name, "runs")
        self.situations = os.path.join(self.tmp.name, "situations")
        os.makedirs(self.situations)
        self.store = RunStore(self.runs)
        self.rid = self.store.create_run(CASE, seed=1)
        self.store.complete_run(CASE, self.rid,
                                {"trajectory.json": b"{\"frames\":[]}"},
                                metrics={"events": 2},
                                validation={"passed": True})
        self.procs = []

    def tearDown(self):
        for proc in self.procs:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        self.tmp.cleanup()

    def _start(self, mode):
        port = free_port()
        env = dict(os.environ, SITUATION_DIR=self.situations,
                   PYTHONUNBUFFERED="1")
        proc = subprocess.Popen(
            [sys.executable, os.path.join(SERVER, "serve.py"),
             "--port", str(port), "--mode", mode, "--runs-dir", self.runs],
            cwd=SITUATION3D, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.procs.append(proc)
        for _ in range(60):
            try:
                status, _body = request(port, "/api/health")
                if status == 200:
                    return port
            except Exception:
                pass
            time.sleep(0.2)
        raise AssertionError("server did not start")

    def test_test_mode_list_select_pin(self):
        port = self._start("test")
        status, body = request(port, "/api/sim/runs?case_id=" + CASE)
        self.assertEqual(status, 200)
        self.assertEqual([r["run_id"] for r in body["runs"]], [self.rid])
        status, body = request(port, "/api/sim/runs/%s/select" % self.rid,
                               "POST", {"case_id": CASE, "confirm": True,
                                        "note": "api"})
        self.assertEqual(status, 200)
        self.assertIn("best", body["run"]["roles"])
        status, body = request(port, "/api/sim/runs/%s/pin" % self.rid,
                               "POST", {"case_id": CASE})
        self.assertEqual(status, 200)
        self.assertIn("pinned", body["run"]["roles"])
        status, body = request(port, "/api/sim/runs/%s?case_id=%s" % (self.rid, CASE))
        self.assertEqual(status, 200)
        self.assertEqual(body["run_id"], self.rid)

    def test_error_branches_are_4xx_not_500(self):
        port = self._start("test")
        status, _body = request(port, "/api/sim/runs/nope?case_id=" + CASE)
        self.assertEqual(status, 404)
        queued = self.store.create_run(CASE, seed=9)
        status, body = request(port, "/api/sim/runs/%s/delete" % queued,
                               "POST", {"case_id": CASE, "confirm": True})
        self.assertEqual(status, 409)
        status, body = request(port, "/api/sim/runs", "POST",
                               {"case_id": CASE, "seed": 3, "ticks": 10})
        self.assertEqual(status, 409)
        self.assertIn("active", body)

    def test_demo_mode_blocks_simulation(self):
        port = self._start("demo")
        status, _body = request(port, "/api/sim/runs?case_id=" + CASE)
        self.assertEqual(status, 404)
        status, _body = request(port, "/api/sim/runs", "POST",
                                {"case_id": CASE, "seed": 1, "ticks": 10})
        self.assertEqual(status, 403)


if __name__ == "__main__":
    unittest.main()
