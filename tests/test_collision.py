import unittest

from situation3d.engine.collision import Box, overlap, frame_collisions, swept_collisions


def target(length=10.0, width=2.0):
    return {"id": "t", "domain": "land",
            "collision_bbox": {"length_m": length, "width_m": width, "height_m": 2.0}}


class CollisionTests(unittest.TestCase):
    def test_same_box_overlaps_by_shorter_half_axis(self):
        a = Box("a", 22.97, 120.16, 0.0, 10.0, 2.0)
        b = Box("b", 22.97, 120.16, 0.0, 10.0, 2.0)
        self.assertAlmostEqual(overlap(a, b), 2.0, places=3)

    def test_side_by_side_boxes_do_not_overlap(self):
        a = Box("a", 22.97, 120.16, 0.0, 10.0, 2.0)
        b = Box("b", 22.97, 120.16003, 0.0, 10.0, 2.0)
        self.assertIsNone(overlap(a, b))

    def test_crossing_boxes_overlap(self):
        a = Box("a", 22.97, 120.16, 0.0, 10.0, 2.0)
        b = Box("b", 22.97, 120.16, 90.0, 10.0, 2.0)
        self.assertIsNotNone(overlap(a, b))

    def test_frame_collision_reports_pair(self):
        targets = {"a": target(), "b": target()}
        a = {"id": "a", "lat": 22.97, "lon": 120.16, "heading": 0.0, "mode": "ground"}
        b = {"id": "b", "lat": 22.97, "lon": 120.16, "heading": 0.0, "mode": "ground"}
        hits = frame_collisions({"tick": 1, "targets": [a, b]}, targets)
        self.assertEqual(len(hits), 1)
        self.assertEqual({hits[0]["a"], hits[0]["b"]}, {"a", "b"})

    def test_embarked_vehicle_is_ignored(self):
        targets = {"a": target(), "b": target()}
        a = {"id": "a", "lat": 22.97, "lon": 120.16, "heading": 0.0, "mode": "ground"}
        b = {"id": "b", "lat": 22.97, "lon": 120.16, "heading": 0.0, "mode": "embarked"}
        self.assertEqual(frame_collisions({"tick": 1, "targets": [a, b]}, targets), [])

    def test_swept_detects_crossing_between_frames(self):
        targets = {"a": target(10.0, 2.0), "b": target(10.0, 2.0)}
        # 0.001 deg latitude is ~110 m; the pair swaps places between frames and
        # only the interpolated sample at the middle overlaps.
        f0 = {"tick": 0, "targets": [
            {"id": "a", "lat": 22.9700, "lon": 120.16, "heading": 0.0, "mode": "ground"},
            {"id": "b", "lat": 22.9709, "lon": 120.16, "heading": 180.0, "mode": "ground"},
        ]}
        f1 = {"tick": 1, "targets": [
            {"id": "a", "lat": 22.9709, "lon": 120.16, "heading": 0.0, "mode": "ground"},
            {"id": "b", "lat": 22.9700, "lon": 120.16, "heading": 180.0, "mode": "ground"},
        ]}
        hits = swept_collisions([f0, f1], targets, substeps=4)
        self.assertTrue(hits)
        self.assertEqual(hits[0]["interval"], [0, 1])


if __name__ == "__main__":
    unittest.main()
