from __future__ import annotations

from dataclasses import dataclass

from .assignment import solve_assignment
from .kalman import KalmanFilter
from .types import Detection, FrameDetections, Track
from .utils import box_center, clamp_box, cosine_similarity, iou


@dataclass(slots=True)
class TrackerConfig:
    assignment_backend: str = "auto"
    high_confidence_threshold: float = 0.5
    low_confidence_threshold: float = 0.1
    match_iou_threshold: float = 0.3
    second_match_iou_threshold: float = 0.2
    distance_gate_multiplier: float = 2.0
    appearance_weight: float = 0.2
    use_appearance: bool = True
    class_aware: bool = True
    min_hits: int = 2
    max_age: int = 30
    history_size: int = 32


class LightweightTracker:
    """Online tracker with Kalman motion and ByteTrack-style association."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self._tracks: dict[int, Track] = {}
        self._next_track_id = 1
        self._kalman = KalmanFilter()

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1

    def tracks(self, *, include_lost: bool = False) -> list[Track]:
        return self._visible_tracks(include_lost=include_lost)

    def update(self, detections: FrameDetections) -> list[Track]:
        tracked_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if track.state == "tracked"
        ]
        lost_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if track.state == "lost"
        ]
        self._predict(tracked_ids + lost_ids)

        high_confidence = [
            detection
            for detection in detections.items
            if detection.score >= self.config.high_confidence_threshold
        ]
        low_confidence = [
            detection
            for detection in detections.items
            if self.config.low_confidence_threshold <= detection.score < self.config.high_confidence_threshold
        ]

        first_stage_ids = tracked_ids + lost_ids
        high_matches, unmatched_first_stage_ids, unmatched_high = self._associate(
            first_stage_ids,
            high_confidence,
            threshold=self.config.match_iou_threshold,
        )
        self._apply_matches(high_matches, high_confidence, detections)

        second_stage_ids = [
            track_id for track_id in unmatched_first_stage_ids if track_id in tracked_ids
        ]
        low_matches, unmatched_second_stage_ids, unmatched_low = self._associate(
            second_stage_ids,
            low_confidence,
            threshold=self.config.second_match_iou_threshold,
        )
        self._apply_matches(low_matches, low_confidence, detections)

        for track_id in unmatched_second_stage_ids:
            self._mark_missed(track_id, detections)

        for track_id in unmatched_first_stage_ids:
            if track_id in lost_ids:
                self._mark_missed(track_id, detections)

        for detection_index in unmatched_high:
            self._spawn_track(high_confidence[detection_index], detections)

        for detection_index in unmatched_low:
            _ = detection_index

        return self._visible_tracks()

    def _predict(self, track_ids: list[int]) -> None:
        for track_id in track_ids:
            track = self._tracks[track_id]
            if not track.mean or not track.covariance:
                continue
            track.mean, track.covariance = self._kalman.predict(track.mean, track.covariance)
            self._sync_track_from_filter(track)
            track.age += 1

    def _associate(
        self,
        track_ids: list[int],
        detections: list[Detection],
        *,
        threshold: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not track_ids or not detections:
            return [], track_ids[:], list(range(len(detections)))

        similarities = [
            [
                self._similarity(self._tracks[track_id], detection)
                for detection in detections
            ]
            for track_id in track_ids
        ]
        large_cost = 1e6
        cost_matrix = [
            [
                1.0 - similarity if similarity >= threshold else large_cost
                for similarity in row
            ]
            for row in similarities
        ]

        matched_rows = set()
        matched_detections = set()
        matches: list[tuple[int, int]] = []

        for row_index, detection_index in solve_assignment(
            cost_matrix,
            backend=self.config.assignment_backend,
        ):
            if cost_matrix[row_index][detection_index] >= large_cost:
                continue
            matched_rows.add(row_index)
            matched_detections.add(detection_index)
            matches.append((track_ids[row_index], detection_index))

        unmatched_tracks = [
            track_id for row_index, track_id in enumerate(track_ids) if row_index not in matched_rows
        ]
        unmatched_detections = [
            detection_index
            for detection_index in range(len(detections))
            if detection_index not in matched_detections
        ]
        return matches, unmatched_tracks, unmatched_detections

    def _similarity(self, track: Track, detection: Detection) -> float:
        if self.config.class_aware and track.class_id is not None and detection.class_id is not None:
            if track.class_id != detection.class_id:
                return -1.0

        if not self._passes_distance_gate(track, detection):
            return -1.0

        overlap = iou(track.box, detection.box)
        if not self.config.use_appearance:
            return overlap

        appearance = cosine_similarity(track.embedding, detection.embedding)
        if appearance is None:
            return overlap

        appearance_weight = min(max(self.config.appearance_weight, 0.0), 1.0)
        return (1.0 - appearance_weight) * overlap + appearance_weight * appearance

    def _apply_matches(
        self,
        matches: list[tuple[int, int]],
        detections: list[Detection],
        frame: FrameDetections,
    ) -> None:
        for track_id, detection_index in matches:
            track = self._tracks[track_id]
            detection = detections[detection_index]

            track.mean, track.covariance = self._kalman.update(
                track.mean,
                track.covariance,
                detection.box,
            )
            self._sync_track_from_filter(track)
            track.score = detection.score
            track.class_id = detection.class_id
            track.embedding = detection.embedding or track.embedding
            track.metadata = dict(detection.metadata)
            track.frame_index = frame.frame_index
            track.timestamp = frame.timestamp
            track.hits += 1
            track.misses = 0
            track.state = "tracked"
            self._append_path(track)

    def _mark_missed(self, track_id: int, frame: FrameDetections) -> None:
        track = self._tracks[track_id]
        track.misses += 1
        track.frame_index = frame.frame_index
        track.timestamp = frame.timestamp

        if track.hits < self.config.min_hits:
            track.state = "removed"
            return

        if track.misses > self.config.max_age:
            track.state = "removed"
        else:
            track.state = "lost"

    def _spawn_track(self, detection: Detection, frame: FrameDetections) -> None:
        mean, covariance = self._kalman.initiate(detection.box)
        track = Track(
            track_id=self._next_track_id,
            box=clamp_box(detection.box),
            score=detection.score,
            class_id=detection.class_id,
            state="tracked",
            hits=1,
            age=1,
            misses=0,
            path=[],
            mean=mean,
            covariance=covariance,
            embedding=detection.embedding,
            frame_index=frame.frame_index,
            timestamp=frame.timestamp,
            metadata=dict(detection.metadata),
        )
        self._sync_track_from_filter(track)
        self._append_path(track)
        self._tracks[track.track_id] = track
        self._next_track_id += 1

    def _append_path(self, track: Track) -> None:
        center = box_center(track.box)
        if not track.path or track.path[-1] != center:
            track.path.append(center)
        if len(track.path) > self.config.history_size:
            del track.path[:-self.config.history_size]

    def _visible_tracks(self, *, include_lost: bool = False) -> list[Track]:
        states = {"tracked", "lost"} if include_lost else {"tracked"}
        visible = [
            track
            for track in self._tracks.values()
            if track.state in states
        ]
        visible.sort(key=lambda track: track.track_id)
        return visible

    def _sync_track_from_filter(self, track: Track) -> None:
        track.box = self._kalman.box_from_mean(track.mean)
        track.velocity = self._kalman.velocity_from_mean(track.mean)

    def _passes_distance_gate(self, track: Track, detection: Detection) -> bool:
        track_cx, track_cy = box_center(track.box)
        detection_cx, detection_cy = box_center(detection.box)
        distance_x = detection_cx - track_cx
        distance_y = detection_cy - track_cy
        squared_distance = distance_x * distance_x + distance_y * distance_y

        x1, y1, x2, y2 = track.box
        gate_radius = max(abs(x2 - x1), abs(y2 - y1), 1.0) * self.config.distance_gate_multiplier
        return squared_distance <= gate_radius * gate_radius
