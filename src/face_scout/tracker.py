"""Greedy IoU tracker: turns per-frame detections into stable, aging tracks."""

from __future__ import annotations

from .geometry import iou
from .types import FaceObservation, Track


class Tracker:
    """Associates detections to existing tracks by greedy descending IoU.

    A track that is not matched in a frame ages by one; once it exceeds
    ``max_age`` it is dropped. Unmatched detections spawn new tracks. Tracks
    coasting through a brief miss (age > 0) stay in the list so identity and the
    stable id survive short occlusions.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 15) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._next_id = 0
        self.tracks: list[Track] = []

    def update(self, observations: list[FaceObservation]) -> list[Track]:
        pairs: list[tuple[float, int, int]] = []
        for ti, track in enumerate(self.tracks):
            for oi, obs in enumerate(observations):
                score = iou(track.bbox, obs.bbox)
                if score >= self.iou_threshold:
                    pairs.append((score, ti, oi))
        pairs.sort(reverse=True)

        matched_tracks: set[int] = set()
        matched_obs: set[int] = set()
        for _score, ti, oi in pairs:
            if ti in matched_tracks or oi in matched_obs:
                continue
            obs = observations[oi]
            track = self.tracks[ti]
            track.bbox = obs.bbox
            track.landmarks = obs.landmarks
            track.age = 0
            track.hits += 1
            matched_tracks.add(ti)
            matched_obs.add(oi)

        for ti, track in enumerate(self.tracks):
            if ti not in matched_tracks:
                track.age += 1

        for oi, obs in enumerate(observations):
            if oi not in matched_obs:
                self.tracks.append(
                    Track(
                        track_id=self._next_id,
                        bbox=obs.bbox,
                        landmarks=obs.landmarks,
                        age=0,
                        hits=1,
                    )
                )
                self._next_id += 1

        self.tracks = [t for t in self.tracks if t.age <= self.max_age]
        return list(self.tracks)
