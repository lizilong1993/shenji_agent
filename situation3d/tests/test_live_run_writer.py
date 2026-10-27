"""LiveRunWriter：推演中的增量产物（frames.jsonl / frames.idx / 快照）。"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
for p in (SITUATION3D, os.path.join(SITUATION3D, "server"),
          os.path.join(SITUATION3D, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_store import RunStore  # noqa: E402
import run_situation_case as runner  # noqa: E402

CASE = "taiwan_dense_transfer_v1"


class LiveRunWriterTest(unittest.TestCase):
    def test_append_frames_offsets_and_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(tmp)
            rid = store.create_run(CASE, seed=1)
            writer = runner.LiveRunWriter(store, CASE, rid)
            writer.meta({"schema": "situation3d.live.v1", "tick_count": 5})
            writer.publish(0, "running")
            for tick in range(5):
                writer.append_frame({"tick": tick, "t_seconds": tick * 30.0,
                                     "targets": []})
            writer.publish(4, "running",
                           events=[{"id": "E1", "tick": 4}],
                           nodes=[{"tick": 4, "type": "first_fire"}])
            writer.close()

            live = os.path.join(store.run_dir(CASE, rid), "live")
            with open(os.path.join(live, "frames.idx"), encoding="ascii") as fh:
                offsets = [int(line) for line in fh.read().splitlines()]
            self.assertEqual(len(offsets), 5)
            self.assertEqual(offsets[0], 0)
            with open(os.path.join(live, "frames.jsonl"), "rb") as fh:
                data = fh.read()
            frames = [json.loads(line) for line in data.splitlines()]
            self.assertEqual([f["tick"] for f in frames], list(range(5)))
            # 偏移必须精确指向对应行；读取方就是靠它 seek。
            with open(os.path.join(live, "frames.jsonl"), "rb") as fh:
                fh.seek(offsets[3])
                self.assertEqual(json.loads(fh.readline())["tick"], 3)
            with open(os.path.join(live, "events.json"), encoding="utf-8") as fh:
                self.assertEqual(json.load(fh)["events"][0]["id"], "E1")
            with open(os.path.join(live, "nodes.json"), encoding="utf-8") as fh:
                self.assertEqual(json.load(fh)["nodes"][0]["type"], "first_fire")
            with open(os.path.join(live, "state.json"), encoding="utf-8") as fh:
                state = json.load(fh)
            self.assertEqual(state["status"], "running")
            self.assertEqual(state["frame_count"], 5)
            self.assertEqual(state["tick"], 4)
            with open(os.path.join(live, "meta.json"), encoding="utf-8") as fh:
                self.assertEqual(json.load(fh)["tick_count"], 5)


if __name__ == "__main__":
    unittest.main()
