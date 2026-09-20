"""
Runs the perception stage graph over each incoming frame and emits
structured events onto the run's event bus.

Milestone status:
- PersonDetector, ObjectDetector: REAL (Milestone 2)
- PoseEstimator: REAL, requires scripts/download_models.py (Milestone 3)
- ObjectTracker, HandObjectInteraction: REAL, rule-based baseline (Milestone 4)
- ActivityRecognizer: REAL rule-based baseline (Milestone 5) OR a genuinely
  trained model (Milestone 8), whichever is available — see
  _select_activity_recognizer() below. Never silently pretends the
  rule-based fallback is the trained model or vice versa.
"""
from __future__ import annotations
import time
from typing import Callable, Awaitable

from .interfaces import FrameBundle
from .real_models import (
    RealPersonDetector, RealPoseEstimator, CombinedObjectDetector,
    RealObjectTracker, RealHandObjectInteraction, RealActivityRecognizer,
    TrainedActivityRecognizer, ObjectAppearanceClassifier,
)
from backend.vision.rack_reference import RackReferenceTracker

EventCallback = Callable[[dict], Awaitable[None]]


class AIPipelineOrchestrator:
    def __init__(self, use_rack_reference: bool = False, rack_marker_id: int = 0):
        self.person_detector = RealPersonDetector()
        self.pose_estimator = RealPoseEstimator()
        self.object_detector = CombinedObjectDetector()
        self.object_tracker = RealObjectTracker()
        self.interaction_model = RealHandObjectInteraction()

        # Optional second opinion (Milestone 12) — annotates detected boxes
        # with a learned verified_class/verified_confidence if a trained
        # appearance classifier exists; a no-op passthrough otherwise. Never
        # overrides the geometric detector's own class label.
        self.object_classifier = ObjectAppearanceClassifier()

        # Optional orientation-agnostic tracking (Milestone 13) — addresses
        # the official spec's "no fixed up/down, track relative to the
        # rack" challenge via ArUco marker detection, NOT full 3D Human
        # Mesh Recovery (see rack_reference.py's scope note for why: SMPL
        # body-model weights are license-gated and unobtainable here).
        # Off by default — most experiments don't have a rack marker
        # mounted, and a "marker not found" state on every frame would
        # just be noise for those. Opt in via ActiveRun(use_rack_reference=True).
        self.use_rack_reference = use_rack_reference
        self.rack_tracker = RackReferenceTracker(marker_id=rack_marker_id) if use_rack_reference else None

        # Prefer a genuinely trained model (Milestone 8) if one exists and
        # its feature schema still matches; otherwise fall back to the
        # rule-based baseline (Milestone 5). Decided once at warmup, not
        # per-frame — metadata() reports which one actually ended up active
        # so this is never ambiguous from the outside.
        self._trained_recognizer = TrainedActivityRecognizer()
        self._rule_based_recognizer = RealActivityRecognizer()
        self.activity_recognizer = self._rule_based_recognizer  # default until warmup() decides
        self._warmed_up = False

    def warmup(self):
        for m in (
            self.person_detector, self.pose_estimator, self.object_detector,
            self.object_tracker, self.interaction_model, self.object_classifier,
            self._trained_recognizer, self._rule_based_recognizer,
        ):
            m.warmup()
        self.activity_recognizer = (
            self._trained_recognizer if self._trained_recognizer.is_available()
            else self._rule_based_recognizer
        )
        self._warmed_up = True

    def metadata(self) -> list[dict]:
        models = list((
            self.person_detector, self.pose_estimator, self.object_detector,
            self.object_tracker, self.interaction_model, self.object_classifier,
            self.activity_recognizer,
        ))
        result = [m.get_metadata().__dict__ for m in models]
        if self.use_rack_reference:
            result.append({
                "name": "RackReferenceTracker (ArUco, orientation-agnostic — NOT full 3D HMR, see scope note)",
                "version": "1.0", "status": "IMPLEMENTED", "device": "cpu",
            })
        return result

    async def process_frame(self, run_id: str, on_event: EventCallback, frame=None) -> dict:
        """
        Runs one frame through the perception graph and emits a
        'perception.update' event. `frame` is a BGR numpy array from the
        video source, or None if the camera is unavailable that tick —
        real models degrade gracefully to empty output in that case
        (Section 27: failure handling).

        Stage order matters here (Section 3's pipeline graph): pose and
        object detection must run before tracking/interaction, since those
        stages consume pose+objects together via FrameBundle's context
        fields rather than raw pixels.
        """
        frame_ts = time.time()
        bundle = FrameBundle(frame_ts=frame_ts, run_id=run_id, frame=frame)

        person = self.person_detector.infer(bundle)
        pose = self.pose_estimator.infer(bundle)
        objects = self.object_detector.infer(bundle)

        # Optional appearance second-opinion (no-op if no trained model) —
        # annotates in place on the payload's object list, doesn't touch
        # detection/localization itself.
        annotated_objects = self.object_classifier.annotate(frame, objects.payload.get("objects", []))
        objects.payload["objects"] = annotated_objects

        # Attach this frame's pose/object output so downstream stages can
        # consume them without re-running perception on raw pixels.
        context_bundle = FrameBundle(
            frame_ts=frame_ts, run_id=run_id, frame=frame,
            pose=pose.payload, objects=objects.payload.get("objects", []),
        )
        tracks = self.object_tracker.infer(context_bundle)

        # Interaction needs tracked (identity-stable) objects, not the raw
        # per-frame detections — update the bundle with tracked objects.
        interaction_bundle = FrameBundle(
            frame_ts=frame_ts, run_id=run_id, frame=frame,
            pose=pose.payload, objects=tracks.payload.get("tracks", []),
        )
        interaction = self.interaction_model.infer(interaction_bundle)

        activity_bundle = FrameBundle(
            frame_ts=frame_ts, run_id=run_id, frame=frame,
            pose=pose.payload, objects=tracks.payload.get("tracks", []),
            interactions=interaction.payload.get("interactions", []),
        )
        activity = self.activity_recognizer.infer(activity_bundle)

        rack_reference = None
        if self.use_rack_reference:
            marker_result = self.rack_tracker.detect(frame)
            joints = (pose.payload or {}).get("joints", {})
            rack_relative_joints = self.rack_tracker.transform_joints(joints, marker_result)
            rack_reference = {
                "marker": marker_result,
                "joints_rack_relative": rack_relative_joints,   # None if marker not visible this frame
            }

        result = {
            "run_id": run_id,
            "frame_ts": frame_ts,
            "person": person.payload,
            "pose": pose.payload,
            "objects": objects.payload,
            "tracks": tracks.payload,
            "interaction": interaction.payload,
            "activity": activity.payload,
            "activity_confidence": activity.confidence,
            "rack_reference": rack_reference,
        }

        await on_event({"type": "perception.update", "payload": result})
        return result
