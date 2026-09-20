"""
Temporal action segmentation (Section 12): raw per-frame activity labels
are noisy, so this smooths them over a sliding window and only emits a
segment boundary once a label has been dominant for enough consecutive
frames — the same "don't trust a single frame" principle used elsewhere
(evidence accumulation in the procedure engine).

This is genuinely reusable machinery, independent of what produces the
raw per-frame labels — it doesn't care whether those came from a rule-based
baseline (current) or a trained model (future upgrade).
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass


@dataclass
class ActionSegment:
    label: str
    start_ts: float
    end_ts: float
    mean_confidence: float
    frame_count: int


class TemporalActionSegmenter:
    def __init__(self, smoothing_window: int = 5, min_segment_frames: int = 4):
        self.smoothing_window = smoothing_window
        self.min_segment_frames = min_segment_frames
        self._recent = deque(maxlen=smoothing_window)
        self._current_segment: ActionSegment | None = None
        self._pending_label: str | None = None
        self._pending_count = 0
        self.completed_segments: list[ActionSegment] = []

    def _smoothed_label(self) -> str | None:
        if not self._recent:
            return None
        counts: dict[str, int] = {}
        for label, _conf in self._recent:
            counts[label] = counts.get(label, 0) + 1
        return max(counts, key=counts.get)

    def observe(self, label: str, confidence: float, ts: float) -> ActionSegment | None:
        """
        Feed one raw per-frame observation. Returns a newly-completed
        ActionSegment if this observation caused a segment boundary,
        else None. Call current_segment() to see the in-progress one.
        """
        self._recent.append((label, confidence))
        smoothed = self._smoothed_label()

        if self._current_segment is None:
            self._current_segment = ActionSegment(smoothed, ts, ts, confidence, 1)
            self._pending_label = None
            self._pending_count = 0
            return None

        if smoothed == self._current_segment.label:
            # Continuing the same segment
            self._current_segment.end_ts = ts
            n = self._current_segment.frame_count
            self._current_segment.mean_confidence = (
                self._current_segment.mean_confidence * n + confidence
            ) / (n + 1)
            self._current_segment.frame_count += 1
            self._pending_label = None
            self._pending_count = 0
            return None

        # Smoothed label differs from current segment — require it to
        # persist for min_segment_frames before actually cutting a new
        # segment (avoids single-frame flicker cutting many tiny segments).
        if smoothed == self._pending_label:
            self._pending_count += 1
        else:
            self._pending_label = smoothed
            self._pending_count = 1

        if self._pending_count >= self.min_segment_frames:
            completed = self._current_segment
            self.completed_segments.append(completed)
            self._current_segment = ActionSegment(smoothed, ts, ts, confidence, 1)
            self._pending_label = None
            self._pending_count = 0
            return completed

        return None

    def current_segment(self) -> ActionSegment | None:
        return self._current_segment
