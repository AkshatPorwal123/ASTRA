"""
Lightweight centroid-distance tracker. Assigns persistent track IDs to
detections across frames by nearest-centroid matching, with a grace period
before dropping a track — giving basic occlusion tolerance (Section 5)
without pulling in a full ByteTrack/BoT-SORT dependency yet.

This is the Phase-2 tracking baseline; ByteTrack/BoT-SORT are the documented
upgrade path once identity-switch/occlusion robustness is actually measured
and found lacking (Section 40's benchmark-before-upgrading principle).
"""
from __future__ import annotations
import math


class CentroidTracker:
    def __init__(self, max_missed_frames: int = 15, max_match_distance: float = 120.0):
        self.max_missed_frames = max_missed_frames
        self.max_match_distance = max_match_distance
        self._next_id = 1
        self._tracks: dict[int, dict] = {}   # id -> {centroid, bbox, missed}

    @staticmethod
    def _centroid(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def update(self, detections: list[dict]) -> list[dict]:
        """
        detections: list of {"bbox": [x1,y1,x2,y2], "confidence": float, ...}
        Returns the same list with a "track_id" key added to each detection.
        """
        det_centroids = [self._centroid(d["bbox"]) for d in detections]
        unmatched_dets = set(range(len(detections)))
        matched_track_ids = set()

        # Greedy nearest-centroid matching
        for track_id, track in list(self._tracks.items()):
            best_idx, best_dist = None, self.max_match_distance
            for i in unmatched_dets:
                dist = math.dist(track["centroid"], det_centroids[i])
                if dist < best_dist:
                    best_idx, best_dist = i, dist
            if best_idx is not None:
                detections[best_idx]["track_id"] = track_id
                self._tracks[track_id] = {
                    "centroid": det_centroids[best_idx],
                    "bbox": detections[best_idx]["bbox"],
                    "missed": 0,
                }
                unmatched_dets.discard(best_idx)
                matched_track_ids.add(track_id)

        # Age out tracks that weren't matched this frame
        for track_id in list(self._tracks.keys()):
            if track_id not in matched_track_ids:
                self._tracks[track_id]["missed"] += 1
                if self._tracks[track_id]["missed"] > self.max_missed_frames:
                    del self._tracks[track_id]

        # New tracks for anything still unmatched
        for i in unmatched_dets:
            track_id = self._next_id
            self._next_id += 1
            detections[i]["track_id"] = track_id
            self._tracks[track_id] = {"centroid": det_centroids[i], "bbox": detections[i]["bbox"], "missed": 0}

        return detections
