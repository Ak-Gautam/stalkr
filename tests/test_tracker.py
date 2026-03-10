from __future__ import annotations

import unittest

from stalkr import (
    Detection,
    FrameDetections,
    LightweightTracker,
    RFDETRDetectionAdapter,
    StalkrTracker,
    TrackerConfig,
    TrackingPipeline,
)


class LightweightTrackerTests(unittest.TestCase):
    def test_creates_tracked_state_and_updates_path(self) -> None:
        tracker = LightweightTracker(TrackerConfig(min_hits=2))

        frame_0 = FrameDetections(
            frame_index=0,
            items=[Detection(box=(0, 0, 10, 10), score=0.9, class_id=1)],
        )
        tracks = tracker.update(frame_0)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].state, "tracked")
        self.assertEqual(tracks[0].hits, 1)
        self.assertEqual(tracks[0].misses, 0)

        frame_1 = FrameDetections(
            frame_index=1,
            items=[Detection(box=(1, 0, 11, 10), score=0.95, class_id=1)],
        )
        tracks = tracker.update(frame_1)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].track_id, 1)
        self.assertEqual(tracks[0].state, "tracked")
        self.assertEqual(tracks[0].hits, 2)
        self.assertGreaterEqual(len(tracks[0].path), 2)
        self.assertNotEqual(tracks[0].velocity, (0.0, 0.0, 0.0, 0.0))

    def test_second_pass_keeps_track_alive_with_lower_score_detection(self) -> None:
        tracker = LightweightTracker(
            TrackerConfig(
                min_hits=1,
                high_confidence_threshold=0.6,
                low_confidence_threshold=0.2,
            )
        )

        tracker.update(
            FrameDetections(
                frame_index=0,
                items=[Detection(box=(0, 0, 10, 10), score=0.95, class_id=1)],
            )
        )

        tracks = tracker.update(
            FrameDetections(
                frame_index=1,
                items=[Detection(box=(0.5, 0, 10.5, 10), score=0.3, class_id=1)],
            )
        )

        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].track_id, 1)
        self.assertEqual(tracks[0].misses, 0)

    def test_lost_track_can_be_reactivated_by_high_confidence_detection(self) -> None:
        tracker = LightweightTracker(TrackerConfig(min_hits=1, max_age=3))

        tracker.update(
            FrameDetections(
                frame_index=0,
                items=[Detection(box=(0, 0, 10, 10), score=0.95, class_id=1)],
            )
        )
        tracker.update(FrameDetections(frame_index=1, items=[]))
        lost_tracks = tracker.tracks(include_lost=True)
        self.assertEqual(lost_tracks[0].state, "lost")

        tracks = tracker.update(
            FrameDetections(
                frame_index=2,
                items=[Detection(box=(0.8, 0, 10.8, 10), score=0.91, class_id=1)],
            )
        )
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].track_id, 1)
        self.assertEqual(tracks[0].state, "tracked")

    def test_class_aware_matching_prevents_cross_class_reuse(self) -> None:
        tracker = LightweightTracker(TrackerConfig(min_hits=1, class_aware=True))

        tracker.update(
            FrameDetections(
                frame_index=0,
                items=[Detection(box=(0, 0, 10, 10), score=0.9, class_id=1)],
            )
        )

        tracks = tracker.update(
            FrameDetections(
                frame_index=1,
                items=[Detection(box=(0, 0, 10, 10), score=0.9, class_id=2)],
            )
        )

        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].track_id, 2)

    def test_track_is_removed_after_max_age(self) -> None:
        tracker = LightweightTracker(TrackerConfig(min_hits=1, max_age=1))
        tracker.update(
            FrameDetections(
                frame_index=0,
                items=[Detection(box=(0, 0, 10, 10), score=0.95, class_id=1)],
            )
        )

        tracks = tracker.update(FrameDetections(frame_index=1, items=[]))
        self.assertEqual(tracks, [])

        lost_tracks = tracker.tracks(include_lost=True)
        self.assertEqual(len(lost_tracks), 1)
        self.assertEqual(lost_tracks[0].state, "lost")

        tracks = tracker.update(FrameDetections(frame_index=2, items=[]))
        self.assertEqual(tracks, [])

    def test_hungarian_assignment_prefers_global_optimum(self) -> None:
        tracker = LightweightTracker(
            TrackerConfig(
                min_hits=1,
                match_iou_threshold=0.05,
                second_match_iou_threshold=0.05,
            )
        )
        tracker.update(
            FrameDetections(
                frame_index=0,
                items=[
                    Detection(box=(0, 0, 10, 10), score=0.95, class_id=1),
                    Detection(box=(20, 0, 30, 10), score=0.95, class_id=1),
                ],
            )
        )

        tracks = tracker.update(
            FrameDetections(
                frame_index=1,
                items=[
                    Detection(box=(19, 0, 29, 10), score=0.95, class_id=1),
                    Detection(box=(1, 0, 11, 10), score=0.95, class_id=1),
                ],
            )
        )
        self.assertEqual([track.track_id for track in tracks], [1, 2])
        self.assertLess(tracks[0].box[0], tracks[1].box[0])

    def test_distance_gate_blocks_far_detection_match(self) -> None:
        tracker = LightweightTracker(
            TrackerConfig(
                min_hits=1,
                match_iou_threshold=0.0,
                distance_gate_multiplier=1.0,
            )
        )
        tracker.update(
            FrameDetections(
                frame_index=0,
                items=[Detection(box=(0, 0, 10, 10), score=0.95, class_id=1)],
            )
        )

        tracks = tracker.update(
            FrameDetections(
                frame_index=1,
                items=[Detection(box=(100, 100, 110, 110), score=0.95, class_id=1)],
            )
        )
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].track_id, 2)

    def test_boxes_array_is_cached(self) -> None:
        detections = FrameDetections(
            items=[
                Detection(box=(0, 0, 10, 10), score=0.9),
                Detection(box=(10, 10, 20, 20), score=0.8),
            ]
        )
        try:
            boxes = detections.boxes_array()
        except ModuleNotFoundError:
            self.skipTest("numpy is not installed")
        self.assertEqual(boxes.shape, (2, 4))
        self.assertIs(boxes, detections.boxes_array())

    def test_tracking_pipeline_uses_adapter_output(self) -> None:
        class DummyDetector:
            def predict(self, frame: object) -> object:
                return {"frame": frame}

        class DummyAdapter:
            def parse(
                self,
                raw_output: object,
                *,
                frame_index: int | None = None,
                timestamp: float | None = None,
            ) -> FrameDetections:
                self.assertIsInstance(raw_output, dict)
                return FrameDetections(
                    frame_index=frame_index,
                    timestamp=timestamp,
                    items=[Detection(box=(5, 5, 15, 15), score=0.9, class_id=0)],
                )

            def assertIsInstance(self, value: object, expected_type: type[object]) -> None:
                if not isinstance(value, expected_type):
                    raise AssertionError(f"expected {expected_type}, got {type(value)}")

        pipeline = TrackingPipeline(
            detector=DummyDetector(),
            adapter=DummyAdapter(),
            tracker=LightweightTracker(TrackerConfig(min_hits=1)),
        )
        tracks = pipeline.process(frame=object(), frame_index=4, timestamp=1.25)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].frame_index, 4)
        self.assertEqual(tracks[0].timestamp, 1.25)

    def test_rfdetr_adapter_converts_supervision_like_output(self) -> None:
        class DummyDetections:
            xyxy = [[1.0, 2.0, 11.0, 12.0]]
            confidence = [0.9]
            class_id = [3]

        detections = RFDETRDetectionAdapter().parse(
            DummyDetections(),
            frame_index=7,
            timestamp=0.25,
        )

        self.assertEqual(detections.frame_index, 7)
        self.assertEqual(detections.timestamp, 0.25)
        self.assertEqual(len(detections.items), 1)
        self.assertEqual(detections.items[0].box, (1.0, 2.0, 11.0, 12.0))
        self.assertAlmostEqual(detections.items[0].score, 0.9, places=6)
        self.assertEqual(detections.items[0].class_id, 3)

    def test_stalkr_tracker_alias(self) -> None:
        tracker = StalkrTracker()
        self.assertIsInstance(tracker, LightweightTracker)


if __name__ == "__main__":
    unittest.main()
