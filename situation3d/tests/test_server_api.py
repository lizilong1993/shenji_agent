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

    def test_startup_fails_stale_active_run(self):
        stale = self.store.create_run(CASE, seed=11)
        self.store.update_run(CASE, stale, status="running")
        self._start("test")
        self.assertEqual(self.store.get_run(CASE, stale)["status"], "failed")

    def test_cancel_active_run(self):
        port = self._start("test")
        queued = self.store.create_run(CASE, seed=9)
        status, _body = request(port, "/api/sim/runs/%s/cancel" % queued,
                                "POST", {"case_id": CASE})
        self.assertEqual(status, 200)
        self.assertEqual(self.store.get_run(CASE, queued)["status"], "failed")

    def _write_live(self, rid=None):
        """Write the incremental artifacts the live viewer polls."""
        live = os.path.join(self.runs, CASE, rid or self.rid, "live")
        os.makedirs(live, exist_ok=True)
        offs, blob = [], b""
        for tick in range(3):
            line = (json.dumps({"tick": tick, "t_seconds": tick * 30.0,
                                "occupation": {}, "objectives": {},
                                "targets": []}) + "\n").encode("utf-8")
            offs.append(len(blob))
            blob += line
        with open(os.path.join(live, "frames.jsonl"), "wb") as fh:
            fh.write(blob)
        with open(os.path.join(live, "frames.idx"), "wb") as fh:
            fh.write("".join("%d\n" % o for o in offs).encode("ascii"))
        events = [{"id": "D-1", "tick": 1, "type": "destroy",
                   "description": "target destroyed"},
                  {"id": "O-2", "tick": 2, "type": "objective_progress",
                   "description": "area captured"}]
        with open(os.path.join(live, "events.json"), "w", encoding="utf-8") as fh:
            json.dump({"tick": 2, "events": events}, fh, ensure_ascii=False)
        with open(os.path.join(live, "nodes.json"), "w", encoding="utf-8") as fh:
            json.dump({"schema": "sim.nodes.v1",
                       "nodes": [{"tick": 1, "type": "first_destroy",
                                  "title": "first destroy"}]},
                      fh, ensure_ascii=False)
        with open(os.path.join(live, "state.json"), "w", encoding="utf-8") as fh:
            json.dump({"tick": 2, "status": "running", "frame_count": 3}, fh)

    def test_live_incremental_read(self):
        self._write_live()
        port = self._start("test")
        base = "/api/sim/runs/%s/live?case_id=%s" % (self.rid, CASE)
        status, body = request(port, base + "&frames=0&events=0")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["frames"]), 3)
        self.assertEqual(body["frames"][2]["tick"], 2)
        self.assertEqual(body["frame_count"], 3)
        self.assertEqual(len(body["events"]), 2)
        self.assertEqual(body["event_count"], 2)
        self.assertEqual(body["nodes"][0]["type"], "first_destroy")
        # setUp() 已把 run 标记为 completed；live/state.json 里的 running 是
        # 进程被杀前的旧值，运行库终态优先。
        self.assertEqual(body["live_status"], "completed")
        status, body = request(port, base + "&frames=2&events=1")
        self.assertEqual(status, 200)
        self.assertEqual([f["tick"] for f in body["frames"]], [2])
        self.assertEqual([e["id"] for e in body["events"]], ["O-2"])

    def test_live_prefers_terminal_run_status(self):
        rid = self.store.create_run(CASE, seed=12)
        self.store.update_run(CASE, rid, status="running")
        self._write_live(rid)
        self.store.fail_run(CASE, rid, "killed by test")
        port = self._start("test")
        status, body = request(
            port, "/api/sim/runs/%s/live?case_id=%s&frames=0" % (rid, CASE))
        self.assertEqual(status, 200)
        self.assertEqual(body["live_status"], "failed")

    def test_live_is_absent_in_demo(self):
        self._write_live()
        port = self._start("demo")
        status, _body = request(
            port, "/api/sim/runs/%s/live?case_id=%s&frames=0" % (self.rid, CASE))
        self.assertEqual(status, 404)

    def test_demo_mode_blocks_simulation(self):
        port = self._start("demo")
        status, _body = request(port, "/api/sim/runs?case_id=" + CASE)
        self.assertEqual(status, 404)
        status, _body = request(port, "/api/sim/runs", "POST",
                                {"case_id": CASE, "seed": 1, "ticks": 10})
        self.assertEqual(status, 403)


if __name__ == "__main__":
    unittest.main()
