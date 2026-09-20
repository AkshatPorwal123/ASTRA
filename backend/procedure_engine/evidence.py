"""
Evidence accumulation (Section 17 / 12 of the design doc): a deviation or
step-match is never triggered off a single noisy frame. It moves through
TENTATIVE -> CONFIRMED -> (optionally) RECOVERED based on sustained,
consistent evidence.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque


@dataclass
class EvidenceTrack:
    key: str                       # e.g. step_id or alert type
    window: deque = field(default_factory=lambda: deque(maxlen=60))
    status: str = "PENDING"        # PENDING | TENTATIVE | CONFIRMED | RECOVERED

    def push(self, matched: bool, ts: float, confidence: float):
        self.window.append({"ts": ts, "matched": matched, "confidence": confidence})

    def consecutive_matches(self) -> int:
        count = 0
        for entry in reversed(self.window):
            if entry["matched"]:
                count += 1
            else:
                break
        return count

    def trace(self) -> list[dict]:
        return list(self.window)


class EvidenceAccumulator:
    """Tracks one EvidenceTrack per key (step_id or alert type) and applies
    the tentative/confirmed frame thresholds from the experiment config."""

    def __init__(self, tentative_frames: int, confirmed_frames: int):
        self.tentative_frames = tentative_frames
        self.confirmed_frames = confirmed_frames
        self._tracks: dict[str, EvidenceTrack] = {}

    def _track(self, key: str) -> EvidenceTrack:
        if key not in self._tracks:
            self._tracks[key] = EvidenceTrack(key=key)
        return self._tracks[key]

    def observe(self, key: str, matched: bool, ts: float, confidence: float) -> str:
        """Push one observation and return the (possibly updated) status."""
        track = self._track(key)
        track.push(matched, ts, confidence)
        streak = track.consecutive_matches()

        if not matched:
            if track.status == "CONFIRMED":
                track.status = "RECOVERED"
            elif track.status == "TENTATIVE":
                track.status = "PENDING"
            return track.status

        if streak >= self.confirmed_frames:
            track.status = "CONFIRMED"
        elif streak >= self.tentative_frames:
            track.status = "TENTATIVE"
        return track.status

    def evidence_for(self, key: str) -> list[dict]:
        return self._track(key).trace()

    def status_of(self, key: str) -> str:
        """Returns the current status of a key without pushing a new
        observation — used to detect status *transitions* (edge-trigger)
        rather than re-firing on every frame a condition remains true."""
        if key not in self._tracks:
            return "PENDING"
        return self._tracks[key].status

    def reset(self, key: str):
        self._tracks.pop(key, None)

    def confirmed_keys_with_prefix(self, prefix: str) -> list[str]:
        """Returns keys under `prefix` whose track ever reached CONFIRMED and
        is still sitting at CONFIRMED — used by the procedure engine to
        detect RECOVERED alerts once the step they were blocking finally
        completes correctly (Section 17's TENTATIVE/CONFIRMED/RECOVERED
        lifecycle)."""
        return [k for k, t in self._tracks.items() if k.startswith(prefix) and t.status == "CONFIRMED"]
