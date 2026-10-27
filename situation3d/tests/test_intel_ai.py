"""Tests for the AI suggestion store and analyst review."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from intel.ai import AISuggestionStore  # noqa: E402


def payload(**overrides):
    value = {
        "task": "extract",
        "title": "港口附近出现异常集结",
        "observation_id": "rss:abc",
        "claims": {"place": "安平", "event": "集结"},
        "evidence": ["原文：港口附近出现多辆车辆"],
        "confidence": 0.72,
        "model": "qwen2.5-7b-instruct",
        "prompt_version": "intel-extract-v1",
        "source_text": "港口附近出现多辆车辆",
    }
    value.update(overrides)
    return value


class AISuggestionStoreTest(unittest.TestCase):
    def test_add_list_review_preserves_original(self):
        with tempfile.TemporaryDirectory() as root:
            store = AISuggestionStore(os.path.join(root, "ai.json"))
            item = store.add(payload())
            self.assertTrue(item["id"].startswith("ai:"))
            self.assertEqual(item["status"], "suggested")
            self.assertEqual(store.counts()["suggested"], 1)
            reviewed = store.review(item["id"], "accepted", "分析员确认")
            self.assertEqual(reviewed["status"], "accepted")
            self.assertEqual(reviewed["claims"], item["claims"])
            self.assertEqual(reviewed["evidence"], item["evidence"])
            self.assertEqual(reviewed["review_note"], "分析员确认")
            self.assertEqual(store.counts()["accepted"], 1)
            self.assertEqual(store.list("accepted")[0]["id"], item["id"])

    def test_validation(self):
        with tempfile.TemporaryDirectory() as root:
            store = AISuggestionStore(os.path.join(root, "ai.json"))
            with self.assertRaises(ValueError):
                store.add(payload(confidence=1.2))
            with self.assertRaises(ValueError):
                store.add(payload(evidence=[]))
            with self.assertRaises(ValueError):
                store.add(payload(model=""))
            item = store.add(payload())
            with self.assertRaises(ValueError):
                store.review(item["id"], "verified")

    def test_rejected_is_not_verified(self):
        with tempfile.TemporaryDirectory() as root:
            store = AISuggestionStore(os.path.join(root, "ai.json"))
            item = store.add(payload())
            store.review(item["id"], "rejected", "证据不足")
            self.assertEqual(store.counts()["rejected"], 1)
            self.assertEqual(store.counts()["accepted"], 0)


if __name__ == "__main__":
    unittest.main()
