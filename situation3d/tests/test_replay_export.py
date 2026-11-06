"""回放导出：把成品运行打包成单文件 HTML。

锁住的是"带得走、离线能开"这件事：包内自包含、数据完整、阵营信息可还原。
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

sys.path.insert(0, os.path.join(SITUATION3D, "server"))
from replay_export import build_payload, render  # noqa: E402


def _write(root, name, payload):
    path = os.path.join(root, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)


class PayloadTest(unittest.TestCase):
    def _run_dir(self, root):
        _write(root, "run.json", {
            "case_id": "c1", "run_id": "r1", "status": "completed",
            "seed": 7, "metrics": {"events": 2, "fires": 1, "destroyed": 1},
            "config": {"ticks": 1},
        })
        _write(root, "trajectory.json", {
            "case_id": "c1",
            "targets": [{"id": "A", "affiliation": "red"},
                        {"id": "B", "affiliation": "blue"}],
            "frames": [{"tick": 0, "targets": [{"id": "A", "lon": 120.0,
                                                "lat": 23.0}]},
                       {"tick": 1, "targets": [{"id": "A", "lon": 120.1,
                                                "lat": 23.1}]}],
        })
        _write(root, "summary.json", {"events": [{"tick": 1, "type": "fire"}]})
        _write(root, "nodes.json", {"nodes": [{"tick": 1, "type": "first_fire"}]})
        return root

    def test_payload_collects_all_parts(self):
        with tempfile.TemporaryDirectory() as root:
            self._run_dir(root)
            payload = build_payload(root)
            self.assertEqual(payload["case_id"], "c1")
            self.assertEqual(payload["run_id"], "r1")
            self.assertEqual(len(payload["frames"]), 2)
            self.assertEqual(len(payload["targets"]), 2)
            self.assertGreaterEqual(len(payload["events"]), 1)
            self.assertEqual(len(payload["nodes"]), 1)
            self.assertEqual(payload["metrics"]["fires"], 1)

    def test_missing_files_do_not_crash(self):
        with tempfile.TemporaryDirectory() as root:
            payload = build_payload(root)
            self.assertEqual(payload["frames"], [])
            self.assertEqual(payload["targets"], [])


class RenderTest(unittest.TestCase):
    def _payload(self):
        return {
            "schema": "situation3d.replay.v1",
            "case_id": "c1", "run_id": "r1", "status": "completed", "seed": 7,
            "config": {}, "metrics": {},
            "targets": [{"id": "A", "affiliation": "red"}],
            "frames": [{"tick": 0, "targets": []}],
            "events": [{"tick": 0, "type": "fire", "description": "开火"}],
            "nodes": [],
        }

    def test_html_is_self_contained(self):
        html = render(self._payload())
        # 不带任何外部资源引用：离线打开要能工作
        self.assertNotIn("<script src=", html)
        self.assertNotIn("<link rel=", html)
        self.assertNotIn("http://", html.split("<body>")[0])
        self.assertIn("<!doctype html>", html.lower())

    def test_data_is_inlined(self):
        html = render(self._payload())
        self.assertIn("situation3d.replay.v1", html)
        self.assertIn('"case_id":"c1"', html)

    def test_script_tag_in_data_cannot_break_out(self):
        payload = self._payload()
        payload["events"] = [{"tick": 0, "type": "x",
                              "description": "a</script><b>c"}]
        html = render(payload)
        # 内联 JSON 里出现 </script> 会提前闭合脚本块
        self.assertNotIn("</script><b>", html)
        self.assertIn("<\\/script>", html)

    def test_title_is_escaped(self):
        html = render(self._payload(), title="<b>x</b>")
        self.assertIn("&lt;b&gt;x&lt;/b&gt;", html)

    def test_side_index_is_present(self):
        # 帧里没有阵营字段，播放器必须靠 targets 建索引
        html = render(self._payload())
        self.assertIn("function sideOf", html)
        self.assertIn("const SIDE = {}", html)


if __name__ == "__main__":
    unittest.main()
