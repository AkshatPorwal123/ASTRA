"""
Model interface contracts (Section 3/32 of the design doc).

Every real model (YOLO, RTMPose, ST-GCN, etc.) implemented in later
milestones must satisfy these Protocols. The orchestrator only ever talks
to these interfaces, never to a concrete model — that's what lets models
be swapped without touching backend/API code.

Phase 1 ships STUB implementations that return structurally-correct but
synthetic output, clearly labeled, so the rest of the system (procedure
engine, dashboard, logging) can be built and tested against a stable
contract before any real model exists.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Protocol
import random
import time


# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------

@dataclass
class ModelMetadata:
    name: str
    version: str
    status: str          # IMPLEMENTED | PARTIALLY_IMPLEMENTED | PLACEHOLDER | RESEARCH_EXTENSION
    device: str = "cpu"


@dataclass
class FrameBundle:
    frame_ts: float
    run_id: str
    frame: Any = None    # numpy array (BGR) when real capture is wired in
    # Populated by the orchestrator AFTER upstream stages run, so downstream
    # models (e.g. hand-object interaction) can consume already-computed
    # pose/object output for this same frame without changing this Protocol's
    # infer(frame) signature — the "frame" carries richer context as the
    # pipeline progresses (Section 3: "Pose + Objects -> Interaction").
    pose: dict | None = None
    objects: list | None = None
    interactions: list | None = None


@dataclass
class StructuredOutput:
    stage: str
    frame_ts: float
    confidence: float
    payload: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Protocol every perception model must implement
# ---------------------------------------------------------------------------

class PerceptionModel(Protocol):
    def infer(self, frame: FrameBundle) -> StructuredOutput: ...
    def warmup(self) -> None: ...
    def get_metadata(self) -> ModelMetadata: ...


# ---------------------------------------------------------------------------
# Phase 1 stubs — PLACEHOLDER status, clearly labeled per Section 39
# ---------------------------------------------------------------------------

class StubPersonDetector:
    def warmup(self) -> None:
        pass

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata("StubPersonDetector", "0.1", "PLACEHOLDER")

    def infer(self, frame: FrameBundle) -> StructuredOutput:
        return StructuredOutput(
            stage="person_detection",
            frame_ts=frame.frame_ts,
            confidence=0.99,
            payload={"persons": [{"track_id": "P1", "bbox": [100, 80, 300, 480]}]},
        )


class StubPoseEstimator:
    def warmup(self) -> None:
        pass

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata("StubPoseEstimator", "0.1", "PLACEHOLDER")

    def infer(self, frame: FrameBundle) -> StructuredOutput:
        return StructuredOutput(
            stage="pose_2d",
            frame_ts=frame.frame_ts,
            confidence=0.9,
            payload={"track_id": "P1", "joints": {}, "frame_type": "2d"},
        )


class StubObjectDetector:
    def warmup(self) -> None:
        pass

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata("StubObjectDetector", "0.1", "PLACEHOLDER")

    def infer(self, frame: FrameBundle) -> StructuredOutput:
        return StructuredOutput(
            stage="object_detection",
            frame_ts=frame.frame_ts,
            confidence=0.85,
            payload={"objects": []},
        )


class StubObjectTracker:
    def warmup(self) -> None:
        pass

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata("StubObjectTracker", "0.1", "PLACEHOLDER")

    def infer(self, frame: FrameBundle) -> StructuredOutput:
        return StructuredOutput(
            stage="object_tracking", frame_ts=frame.frame_ts, confidence=1.0, payload={"tracks": []}
        )


class StubHandObjectInteraction:
    def warmup(self) -> None:
        pass

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata("StubHandObjectInteraction", "0.1", "PLACEHOLDER")

    def infer(self, frame: FrameBundle) -> StructuredOutput:
        return StructuredOutput(
            stage="interaction", frame_ts=frame.frame_ts, confidence=0.0, payload={"interactions": []}
        )


class StubActivityRecognizer:
    """
    Cycles through the demo experiment's expected activities on a timer so
    the procedure engine, dashboard, and alerts have something realistic to
    react to in Phase 1 — without a trained model. This is synthetic data
    and must never be presented as real recognition.
    """
    _DEMO_SEQUENCE = ["PICK", "OPEN", "POUR", "MIX", "CLOSE", "PLACE"]

    def __init__(self, seconds_per_activity: float = 8.0):
        self._start = time.time()
        self._seconds_per_activity = seconds_per_activity

    def warmup(self) -> None:
        pass

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata("StubActivityRecognizer (synthetic demo sequence)", "0.1", "PLACEHOLDER")

    def infer(self, frame: FrameBundle) -> StructuredOutput:
        elapsed = time.time() - self._start
        idx = int(elapsed // self._seconds_per_activity) % len(self._DEMO_SEQUENCE)
        label = self._DEMO_SEQUENCE[idx]
        confidence = round(random.uniform(0.7, 0.95), 2)
        return StructuredOutput(
            stage="activity",
            frame_ts=frame.frame_ts,
            confidence=confidence,
            payload={"label": label, "track_id": "P1"},
        )
