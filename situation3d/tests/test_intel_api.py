"""HTTP contract tests for the case-driven intel collection API."""

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


def free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
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
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


class IntelApiTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.situations = os.path.join(self.tmp.name, "situations")
        self.runs = os.path.join(self.tmp.name, "runs")
        os.makedirs(self.situations)
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

    def test_collect_validation_and_job_status(self):
        port = self._start("test")
        status, body = request(
            port, "/api/intel/collect", "POST",
            {"case_id": "../escape", "sources": ["usgs"]})
        self.assertEqual(status, 400)
        self.assertIn("case_id", body["error"])

        status, body = request(
            port, "/api/intel/collect", "POST",
            {"case_id": "taiwan_dense_transfer_v1", "sources": ["not_a_source"]})
        self.assertEqual(status, 400)
        self.assertIn("unsupported source", body["error"])

        status, body = request(
            port, "/api/intel/collect", "POST",
            {"case_id": "taiwan_dense_transfer_v1",
             "start": "2026-10-25T12:00:00Z",
             "end": "2026-10-24T12:00:00Z"})
        self.assertEqual(status, 400)
        self.assertIn("start", body["error"])

        status, body = request(
            port, "/api/intel/job?case_id=taiwan_dense_transfer_v1")
        self.assertEqual(status, 200)
        self.assertEqual(body["state"], "idle")

        status, body = request(port, "/api/intel/job?case_id=../bad")
        self.assertEqual(status, 400)

    def test_collect_is_read_only_in_demo_mode(self):
        port = self._start("demo")
        status, body = request(
            port, "/api/intel/collect", "POST",
            {"case_id": "taiwan_dense_transfer_v1", "sources": ["usgs"]})
        self.assertEqual(status, 403)
        self.assertIn("disabled", body["error"])


if __name__ == "__main__":
    unittest.main()
