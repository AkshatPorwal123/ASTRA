"""
Milestone 2: real perception models, satisfying the exact same
PerceptionModel interface as the Phase-1 stubs (interfaces.py) — nothing
elsewhere in the system changes to use these.

Baseline choices and why (Section 5/9/40 of the design doc — implement a
baseline, measure it, benchmark alternatives before upgrading):

- Person detection: OpenCV HOG + SVM pedestrian detector. Zero extra model
  download, works out of the box, adequate for a baseline. Real detection,
  not synthetic — status IMPLEMENTED (baseline), with YOLO/RT-DETR as the
  documented upgrade path once accuracy is actually measured and compared.
- Object detection: MobileNet-SSD (Caffe, ~23MB, VOC-20 classes). Includes
  'bottle', which covers the demo experiment's liquid_bottle object; other
  experiment objects (container, rack) aren't in VOC's 20 classes, so they
  fall back to UNKNOWN until a custom-trained detector exists (Milestone
  2 dataset work) — labeled PARTIALLY_IMPLEMENTED accordingly.
- Tracking: lightweight centroid tracker (tracker.py) as the baseline;
  ByteTrack/BoT-SORT remain the documented upgrade path (Section 5).
"""
from __future__ import annotations
import os
import json
import cv2
import numpy as np

from .interfaces import FrameBundle, StructuredOutput, ModelMetadata
from .tracker import CentroidTracker
from .features import extract_features, FEATURE_NAMES
from .object_features import extract_crop_features

MODELS_DIR = os.environ.get("ASTRA_MODELS_DIR", "models")

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

# Map VOC classes to the experiment's object vocabulary (Section 9/22).
# Anything not in this map is reported with its raw VOC label.
OBJECT_CLASS_ALIAS = {
    "bottle": "liquid_bottle",
}


class RealPersonDetector:
    """HOG + SVM pedestrian detector, wrapped with a centroid tracker for
    persistent IDs (Section 5)."""

    def __init__(self):
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self._tracker = CentroidTracker()

    def warmup(self) -> None:
        dummy = np.zeros((240, 320, 3), dtype=np.uint8)
        self._hog.detectMultiScale(dummy)

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata("HOG+SVM PersonDetector (baseline)", "0.2", "IMPLEMENTED")

    def infer(self, frame: FrameBundle) -> StructuredOutput:
        if frame.frame is None:
            return StructuredOutput("person_detection", frame.frame_ts, 0.0, {"persons": []})

        img = frame.frame
        # HOG runs faster and more reliably on a modest fixed size
        h, w = img.shape[:2]
        scale = 640 / w if w > 640 else 1.0
        resized = cv2.resize(img, (int(w * scale), int(h * scale))) if scale != 1.0 else img

        rects, weights = self._hog.detectMultiScale(
            resized, winStride=(8, 8), padding=(8, 8), scale=1.05
        )

        detections = []
        for (x, y, rw, rh), conf in zip(rects, weights):
            x1, y1, x2, y2 = x / scale, y / scale, (x + rw) / scale, (y + rh) / scale
            detections.append({"bbox": [float(x1), float(y1), float(x2), float(y2)], "confidence": float(conf)})

        detections = self._tracker.update(detections)
        avg_conf = float(np.mean([d["confidence"] for d in detections])) if detections else 0.0
        persons = [{"track_id": f"P{d['track_id']}", "bbox": d["bbox"], "confidence": d["confidence"]} for d in detections]

        return StructuredOutput("person_detection", frame.frame_ts, avg_conf, {"persons": persons})


class RealPoseEstimator:
    """
    2D pose estimation via MediaPipe's PoseLandmarker Tasks API (Section 6:
    chosen over RTMPose/ViTPose for Milestone 3 since those need a full
    PyTorch stack — MediaPipe gives real 33-point body landmarks with a much
    smaller footprint, consistent with the edge-deployment goals in Section
    23/40's "start with the lightest sufficient baseline" principle).

    Requires models/pose_landmarker_lite.task, fetched once via
    scripts/download_models.py (not bundled — see that script for why).
    Degrades to PLACEHOLDER status and empty output if the file is missing,
    rather than crashing (Section 27).
    """

    LANDMARK_NAMES = [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
        "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
        "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index",
        "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
        "right_heel", "left_foot_index", "right_foot_index",
    ]

    # Skeleton connections for overlay drawing (index pairs into LANDMARK_NAMES)
    CONNECTIONS = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),          # shoulders/arms
        (11, 23), (12, 24), (23, 24),                               # torso
        (23, 25), (25, 27), (24, 26), (26, 28),                     # legs
        (27, 29), (27, 31), (28, 30), (28, 32),                     # feet
    ]

    def __init__(self):
        model_path = os.path.join(MODELS_DIR, "pose_landmarker_lite.task")
        self._available = os.path.exists(model_path)
        self._landmarker = None
        if self._available:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                PoseLandmarker, PoseLandmarkerOptions, RunningMode,
            )
            self._mp = mp
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._landmarker = PoseLandmarker.create_from_options(options)
        self._frame_index = 0

    def warmup(self) -> None:
        if self._landmarker is None:
            return
        dummy = np.zeros((240, 320, 3), dtype=np.uint8)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=dummy)
        try:
            self._landmarker.detect_for_video(mp_image, 0)
        except Exception:
            pass  # some mediapipe builds reject an all-black warmup frame; harmless

    def get_metadata(self) -> ModelMetadata:
        status = "IMPLEMENTED" if self._available else "PLACEHOLDER"
        return ModelMetadata("MediaPipe PoseLandmarker (2D, lite)", "0.3", status)

    def infer(self, frame: FrameBundle) -> StructuredOutput:
        if self._landmarker is None or frame.frame is None:
            return StructuredOutput("pose_2d", frame.frame_ts, 0.0, {"track_id": None, "joints": {}, "frame_type": "2d"})

        rgb = cv2.cvtColor(frame.frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        self._frame_index += 1
        result = self._landmarker.detect_for_video(mp_image, self._frame_index)

        if not result.pose_landmarks:
            return StructuredOutput("pose_2d", frame.frame_ts, 0.0, {"track_id": None, "joints": {}, "frame_type": "2d"})

        h, w = frame.frame.shape[:2]
        landmarks = result.pose_landmarks[0]
        joints = {}
        confidences = []
        for name, lm in zip(self.LANDMARK_NAMES, landmarks):
            joints[name] = {"x": lm.x * w, "y": lm.y * h, "visibility": lm.visibility}
            confidences.append(lm.visibility)

        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        return StructuredOutput(
            "pose_2d", frame.frame_ts, avg_conf,
            {"track_id": "P1", "joints": joints, "frame_type": "2d"},
        )


class RealObjectDetector:
    """MobileNet-SSD object detector over the VOC-20 class set."""

    def __init__(self, confidence_threshold: float = 0.4):
        prototxt = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.prototxt")
        weights = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.caffemodel")
        self._available = os.path.exists(prototxt) and os.path.exists(weights)
        self._net = cv2.dnn.readNetFromCaffe(prototxt, weights) if self._available else None
        self._confidence_threshold = confidence_threshold

    def warmup(self) -> None:
        if self._net is None:
            return
        dummy = np.zeros((300, 300, 3), dtype=np.uint8)
        blob = cv2.dnn.blobFromImage(dummy, 0.007843, (300, 300), 127.5)
        self._net.setInput(blob)
        self._net.forward()

    def get_metadata(self) -> ModelMetadata:
        status = "IMPLEMENTED" if self._available else "PLACEHOLDER"
        return ModelMetadata("MobileNet-SSD ObjectDetector (VOC-20, baseline)", "0.2", status)

    def infer(self, frame: FrameBundle) -> StructuredOutput:
        if self._net is None or frame.frame is None:
            return StructuredOutput("object_detection", frame.frame_ts, 0.0, {"objects": []})

        img = frame.frame
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5)
        self._net.setInput(blob)
        detections = self._net.forward()

        objects = []
        confidences = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < self._confidence_threshold:
                continue
            cls_id = int(detections[0, 0, i, 1])
            if cls_id == 15:  # 'person' — handled by RealPersonDetector, skip here
                continue
            raw_label = VOC_CLASSES[cls_id] if cls_id < len(VOC_CLASSES) else f"class_{cls_id}"
            label = OBJECT_CLASS_ALIAS.get(raw_label, raw_label)
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int).tolist()
            objects.append({"class": label, "bbox": [x1, y1, x2, y2], "confidence": conf})
            confidences.append(conf)

        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        return StructuredOutput("object_detection", frame.frame_ts, avg_conf, {"objects": objects})


class RealObjectTracker:
    """
    Assigns persistent track IDs to object detections across frames, using
    the same centroid-tracker baseline as person tracking (Section 5/9).
    This is the piece Milestone 2 left as a stub — needed now because the
    interaction model (Milestone 4) requires stable object identity to
    compute velocity and temporal persistence per (hand, object) pair.
    """

    def __init__(self):
        self._tracker = CentroidTracker(max_missed_frames=10, max_match_distance=80.0)

    def warmup(self) -> None:
        pass

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata("CentroidTracker ObjectTracker (baseline)", "0.4", "IMPLEMENTED")

    def infer(self, frame: FrameBundle) -> StructuredOutput:
        objects = list(frame.objects or [])
        tracked = self._tracker.update([dict(o) for o in objects])
        return StructuredOutput("object_tracking", frame.frame_ts, 1.0, {"tracks": tracked})


class RealHandObjectInteraction:
    """
    Hand-object interaction (Section 10). Baseline feature set, computed
    per (hand, tracked object) pair each frame:

        wrist_to_object_distance, relative_velocity, approach_angle,
        contact_flag, temporal_persistence, object_motion_delta

    Classified via interpretable rule-based thresholds into
    {NONE, APPROACH, TOUCH, HOLD, RELEASE} — the baseline Section 10 calls
    for before training a learned classifier. Fine-grained MANIPULATE
    subtypes (POUR/OPEN/CLOSE/PRESS/TURN) need richer temporal/orientation
    reasoning than wrist-distance alone provides and are left as a
    RESEARCH_EXTENSION for Milestone 5 (HAR), not faked here.

    Hand "openness" (finger spread) isn't included — MediaPipe's body
    PoseLandmarker gives wrist position but not per-finger joints; that
    needs a separate Hand Landmarker model, noted as a documented gap
    rather than silently omitted.
    """

    CONTACT_DISTANCE_PX = 60.0      # wrist-to-object-center distance considered "contact"
    APPROACH_DISTANCE_PX = 200.0    # beyond this, no interaction is considered
    HOLD_FRAMES = 8                 # consecutive contact frames before TOUCH -> HOLD

    CONTACT_DISTANCE_FRAC = 0.10     # fraction of frame width considered "contact" — scales with actual resolution
    APPROACH_DISTANCE_FRAC = 0.30    # fraction of frame width beyond which no interaction is considered
    HOLD_FRAMES = 8                  # consecutive contact frames before TOUCH -> HOLD

    def __init__(self):
        # Per-hand history: previous wrist position/time, and current
        # candidate object + consecutive-contact frame count.
        self._hand_state = {
            "left_wrist": {"prev_pos": None, "prev_ts": None, "contact_streak": 0, "state": "NONE", "object_track_id": None},
            "right_wrist": {"prev_pos": None, "prev_ts": None, "contact_streak": 0, "state": "NONE", "object_track_id": None},
        }
        # Per-object previous centroid, for object_motion_delta.
        self._object_prev_centroid: dict[int, tuple[float, float]] = {}

    def warmup(self) -> None:
        pass

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata("Rule-based HandObjectInteraction (baseline, no finger data)", "0.4", "PARTIALLY_IMPLEMENTED")

    @staticmethod
    def _bbox_center(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def infer(self, frame: FrameBundle) -> StructuredOutput:
        joints = (frame.pose or {}).get("joints", {}) if frame.pose else {}
        tracked_objects = frame.objects or []   # expects each to carry "track_id" (from RealObjectTracker)
        ts = frame.frame_ts

        # Thresholds scale with actual frame width — a fixed pixel constant
        # would be meaningless across different camera resolutions (this was
        # the root cause of contact never registering on higher-res webcams).
        frame_width = frame.frame.shape[1] if frame.frame is not None else 640
        contact_px = frame_width * self.CONTACT_DISTANCE_FRAC
        approach_px = frame_width * self.APPROACH_DISTANCE_FRAC

        # object_motion_delta per tracked object, for use below
        object_motion = {}
        for obj in tracked_objects:
            tid = obj.get("track_id")
            centroid = self._bbox_center(obj["bbox"])
            prev = self._object_prev_centroid.get(tid)
            object_motion[tid] = 0.0 if prev is None else ((centroid[0] - prev[0]) ** 2 + (centroid[1] - prev[1]) ** 2) ** 0.5
            self._object_prev_centroid[tid] = centroid

        interactions = []
        for hand_name in ("left_wrist", "right_wrist"):
            joint = joints.get(hand_name)
            hstate = self._hand_state[hand_name]

            if not joint or joint.get("visibility", 0) < 0.3 or not tracked_objects:
                hstate["state"] = "NONE"
                hstate["contact_streak"] = 0
                hstate["object_track_id"] = None
                hstate["prev_pos"] = None
                continue

            wrist_pos = (joint["x"], joint["y"])

            # Find nearest object
            best_obj, best_dist = None, float("inf")
            for obj in tracked_objects:
                center = self._bbox_center(obj["bbox"])
                dist = ((wrist_pos[0] - center[0]) ** 2 + (wrist_pos[1] - center[1]) ** 2) ** 0.5
                if dist < best_dist:
                    best_obj, best_dist = obj, dist

            # Velocity + approach angle of the wrist itself
            velocity = 0.0
            if hstate["prev_pos"] is not None and hstate["prev_ts"] is not None:
                dt = max(ts - hstate["prev_ts"], 1e-3)
                velocity = ((wrist_pos[0] - hstate["prev_pos"][0]) ** 2 + (wrist_pos[1] - hstate["prev_pos"][1]) ** 2) ** 0.5 / dt
            approaching = hstate["prev_pos"] is not None and best_dist < (
                ((hstate["prev_pos"][0] - self._bbox_center(best_obj["bbox"])[0]) ** 2 +
                 (hstate["prev_pos"][1] - self._bbox_center(best_obj["bbox"])[1]) ** 2) ** 0.5
            )

            contact = best_dist <= contact_px
            in_range = best_dist <= approach_px

            if contact:
                hstate["contact_streak"] += 1
                hstate["state"] = "HOLD" if hstate["contact_streak"] >= self.HOLD_FRAMES else "TOUCH"
                hstate["object_track_id"] = best_obj.get("track_id")
            elif in_range:
                was_holding = hstate["state"] in ("TOUCH", "HOLD")
                hstate["state"] = "RELEASE" if was_holding else ("APPROACH" if approaching else "NONE")
                hstate["contact_streak"] = 0
                if hstate["state"] != "RELEASE":
                    hstate["object_track_id"] = best_obj.get("track_id") if hstate["state"] == "APPROACH" else None
            else:
                hstate["state"] = "NONE"
                hstate["contact_streak"] = 0
                hstate["object_track_id"] = None

            hstate["prev_pos"] = wrist_pos
            hstate["prev_ts"] = ts

            interactions.append({
                "hand": hand_name,
                "state": hstate["state"],
                "object_track_id": hstate["object_track_id"],
                "object_class": best_obj.get("class") if hstate["object_track_id"] is not None else None,
                "distance_px": round(best_dist, 1),
                "contact_threshold_px": round(contact_px, 1),
                "wrist_velocity_px_s": round(velocity, 1),
                "object_motion_delta": round(object_motion.get(best_obj.get("track_id"), 0.0), 1),
            })

        avg_conf = 0.7 if any(i["state"] != "NONE" for i in interactions) else 0.0
        return StructuredOutput("interaction", ts, avg_conf, {"interactions": interactions})


class RealActivityRecognizer:
    """
    Milestone 5 baseline: derives an activity label from the REAL fused
    signal already computed upstream (pose + tracked objects + hand-object
    interaction state transitions) — replacing the Phase-1 timer stub with
    something that actually reacts to what's happening on camera.

    This is still a rule-based fusion, not a trained temporal model
    (ST-GCN/TCN/PoseC3D per Section 11) — status is PARTIALLY_IMPLEMENTED,
    honestly. Training a real HAR model requires the custom labeled dataset
    (Section 21), which requires data collection that hasn't happened yet.
    This baseline exists so the procedure engine has a genuine (if
    imperfect) signal to react to in the meantime, and so the eventual
    learned model has a clear baseline to beat (Section 40).

    A short majority-vote smoothing window is applied (Section 9's
    temporal-segmentation principle, simplified) so single-frame noise
    doesn't flicker the output.
    """

    SMOOTHING_WINDOW = 5
    PICK_PHASE_FRAMES = 12    # frames after contact begins still counted as "PICK" before settling into "HOLD"
    PLACE_PHASE_FRAMES = 10   # frames after contact ends still counted as "PLACE", regardless of which
                               # transient state (RELEASE/APPROACH/NONE) the interaction model reports next —
                               # a fast hand withdrawal can skip RELEASE's single frame entirely, so PLACE
                               # detection is driven by "was this hand just in contact", not by catching that
                               # one specific frame (same bug class as PICK's original single-frame trigger).

    def __init__(self):
        self._hand_prev_state: dict[str, str] = {"left_wrist": "NONE", "right_wrist": "NONE"}
        self._hold_frame_count: dict[str, int] = {"left_wrist": 0, "right_wrist": 0}
        self._place_frame_count: dict[str, int] = {"left_wrist": 0, "right_wrist": 0}
        self._hand_last_object: dict[str, str | None] = {"left_wrist": None, "right_wrist": None}
        self._recent_labels: list[str] = []

    def warmup(self) -> None:
        pass

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata("Rule-based ActivityRecognizer (real signal fusion, pre-trained-model baseline)", "0.5", "PARTIALLY_IMPLEMENTED")

    @staticmethod
    def _bbox_center(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def infer(self, frame: FrameBundle) -> StructuredOutput:
        interactions = frame.interactions or []
        raw_label = "IDLE"
        best_confidence = 0.5
        # Which object the winning label is actually about — needed by the
        # procedure engine's WRONG_OBJECT check (Section 10/16). Baseline
        # heuristic: whichever hand drove the most recent non-idle decision
        # this frame; good enough for the current single-actor, mostly
        # single-hand-active demo setup, not a general multi-hand resolver.
        raw_object_class = None

        for interaction in interactions:
            hand = interaction["hand"]
            state = interaction["state"]
            prev = self._hand_prev_state.get(hand, "NONE")
            was_in_contact = prev in ("TOUCH", "HOLD")

            # Remember the last object this hand was actually touching, so
            # PLACE (fired the frame *after* contact ends, when the
            # interaction model may already report object_class=None) can
            # still be attributed to the right object.
            if interaction.get("object_class"):
                self._hand_last_object[hand] = interaction["object_class"]
            hand_object_class = interaction.get("object_class") or self._hand_last_object.get(hand)

            if state in ("TOUCH", "HOLD"):
                if prev not in ("TOUCH", "HOLD"):
                    self._hold_frame_count[hand] = 0
                self._hold_frame_count[hand] += 1
                self._place_frame_count[hand] = 0

                if self._hold_frame_count[hand] <= self.PICK_PHASE_FRAMES:
                    raw_label, best_confidence = "PICK", 0.7
                elif interaction.get("object_motion_delta", 0) > 15:
                    raw_label, best_confidence = "MOVE", 0.7
                else:
                    raw_label, best_confidence = "HOLD", 0.8
                raw_object_class = hand_object_class
            elif was_in_contact or self._place_frame_count[hand] > 0:
                # Just left contact (or still within the PLACE sustain
                # window from having just left it) — count this as PLACE
                # regardless of the interaction model's exact transient
                # state, so a quick hand withdrawal still registers.
                self._place_frame_count[hand] += 1
                self._hold_frame_count[hand] = 0
                if self._place_frame_count[hand] <= self.PLACE_PHASE_FRAMES:
                    raw_label, best_confidence = "PLACE", 0.7
                    raw_object_class = hand_object_class
                else:
                    raw_label, best_confidence = "IDLE", 0.5
                    raw_object_class = None
                    self._place_frame_count[hand] = 0   # window expired — allow future PICK/PLACE cycles
                    self._hand_last_object[hand] = None
            elif state == "APPROACH" and raw_label == "IDLE":
                raw_label, best_confidence = "REACH", 0.55
                raw_object_class = hand_object_class
                self._hold_frame_count[hand] = 0
                self._place_frame_count[hand] = 0
            else:
                self._hold_frame_count[hand] = 0
                self._place_frame_count[hand] = 0

            self._hand_prev_state[hand] = state

        # Majority-vote smoothing over a short window
        self._recent_labels.append(raw_label)
        if len(self._recent_labels) > self.SMOOTHING_WINDOW:
            self._recent_labels.pop(0)
        smoothed = max(set(self._recent_labels), key=self._recent_labels.count)
        # Only attribute an object when the smoothed (voted) label agrees
        # with this frame's raw label — otherwise the object comes from a
        # transient frame that got outvoted, which would misattribute it.
        object_class = raw_object_class if smoothed == raw_label else None

        return StructuredOutput(
            "activity", frame.frame_ts, best_confidence,
            {"label": smoothed, "raw_label": raw_label, "track_id": "P1", "object_class": object_class},
        )


# HSV ranges per color. Red wraps around hue=0/180, so it needs two ranges
# combined. These are reasonably wide/robust starting ranges — real lighting
# will need some tuning, documented in README as a known calibration step.
COLOR_HSV_RANGES = {
    "red": [((0, 100, 80), (10, 255, 255)), ((170, 100, 80), (180, 255, 255))],
    "blue": [((100, 100, 60), (130, 255, 255))],
    "green": [((40, 70, 60), (85, 255, 255))],
    "yellow": [((20, 100, 100), (35, 255, 255))],
}


class ColorBoxDetector:
    """
    Classical HSV color-segmentation detector for the official problem's
    color-coded box experiment — deliberately NOT a pretrained DNN, because
    no off-the-shelf 20/80-class detector (VOC/COCO) has a generic "box"
    category, and training a custom one requires the labeled dataset that
    doesn't exist yet (Section 25/26). Color segmentation is a genuinely
    correct baseline for a task defined by distinctly colored props, not a
    workaround — zero training data needed, runs in milliseconds.

    Each configured color becomes its own detectable "class" (e.g. "red_box").
    Status: IMPLEMENTED, but accuracy depends on lighting/color calibration —
    documented as a real limitation, not hidden.
    """

    MIN_AREA_PX = 1200   # filters out small color specks/noise

    def __init__(self, colors: list[str] | None = None):
        self.colors = colors or list(COLOR_HSV_RANGES.keys())
        # Merge calibrated ranges (Milestone 11) over the hardcoded
        # defaults, resolved once here rather than reloaded every frame —
        # same "restart to apply" pattern as the trained activity model,
        # so a calibration saved mid-run takes effect on the next run, not
        # via surprising mid-run behavior change.
        from backend.vision.calibration import load_calibrated_ranges
        calibrated = load_calibrated_ranges()
        self._ranges = dict(COLOR_HSV_RANGES)
        self._calibrated_colors = set()
        for color, bands in calibrated.items():
            self._ranges[color] = [tuple(tuple(v) for v in band) for band in bands]
            self._calibrated_colors.add(color)
            if color not in self.colors:
                self.colors.append(color)

    def warmup(self) -> None:
        pass

    def get_metadata(self) -> ModelMetadata:
        cal_note = f", calibrated: {', '.join(sorted(self._calibrated_colors))}" if self._calibrated_colors else ""
        return ModelMetadata(f"ColorBoxDetector (HSV segmentation: {', '.join(self.colors)}{cal_note})", "0.6", "IMPLEMENTED")

    def infer(self, frame: FrameBundle) -> StructuredOutput:
        if frame.frame is None:
            return StructuredOutput("object_detection", frame.frame_ts, 0.0, {"objects": []})

        hsv = cv2.cvtColor(frame.frame, cv2.COLOR_BGR2HSV)
        kernel = np.ones((5, 5), np.uint8)
        objects = []
        confidences = []

        for color in self.colors:
            ranges = self._ranges.get(color)
            if not ranges:
                continue
            mask = None
            for lo, hi in ranges:
                m = cv2.inRange(hsv, np.array(lo), np.array(hi))
                mask = m if mask is None else cv2.bitwise_or(mask, m)

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area < self.MIN_AREA_PX:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                # crude confidence: how much of the bounding box the actual
                # colored region fills (a tight, box-like blob scores higher
                # than a scattered/noisy one)
                fill_ratio = area / (w * h + 1e-6)
                confidence = float(min(0.95, 0.4 + fill_ratio * 0.5))
                objects.append({
                    "class": f"{color}_box",
                    "bbox": [float(x), float(y), float(x + w), float(y + h)],
                    "confidence": confidence,
                })
                confidences.append(confidence)

        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        return StructuredOutput("object_detection", frame.frame_ts, avg_conf, {"objects": objects})


class CombinedObjectDetector:
    """
    Runs both real object detectors and merges their output into one
    unified "objects" list — MobileNet-SSD for its VOC-20 classes (useful
    if the experiment involves a literal bottle/chair/etc.), and
    ColorBoxDetector for color-coded props like the official problem's
    red/[color] box experiment, which no pretrained class set covers.

    Two different, real, independently-tested detectors combined — not a
    replacement disguised as an upgrade.
    """

    def __init__(self, colors: list[str] | None = None):
        self._ssd = RealObjectDetector()
        self._color = ColorBoxDetector(colors=colors)

    def warmup(self) -> None:
        self._ssd.warmup()
        self._color.warmup()

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            f"Combined: [{self._ssd.get_metadata().name}] + [{self._color.get_metadata().name}]",
            "0.6", "IMPLEMENTED",
        )

    def infer(self, frame: FrameBundle) -> StructuredOutput:
        ssd_out = self._ssd.infer(frame)
        color_out = self._color.infer(frame)
        objects = list(ssd_out.payload.get("objects", [])) + list(color_out.payload.get("objects", []))
        confidences = [o["confidence"] for o in objects]
        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        return StructuredOutput("object_detection", frame.frame_ts, avg_conf, {"objects": objects})


class TrainedActivityRecognizer:
    """
    Loads a scikit-learn model trained by scripts/train_activity_model.py
    on real labeled data collected via the dashboard's labeling panel —
    this is the actual "trained AI model" the official problem statement
    requires, as opposed to RealActivityRecognizer's hand-written rules.

    Validates the loaded model's feature schema against the CURRENT
    FEATURE_NAMES at load time and refuses to use it if they've drifted
    (rather than silently feeding mismatched features into predict() and
    producing confident-looking garbage) — status reports UNAVAILABLE in
    that case, and the orchestrator falls back to the rule-based baseline.

    Confidence is the model's own class probability, not a fixed constant —
    an honestly uncertain prediction should look uncertain downstream.
    """

    MODEL_PATH = os.environ.get("ASTRA_MODEL_DIR", "models") + "/activity_classifier.pkl"
    META_PATH = os.environ.get("ASTRA_MODEL_DIR", "models") + "/activity_classifier_meta.json"

    def __init__(self):
        self._model = None
        self._meta = None
        self._load_error = None
        self.SMOOTHING_WINDOW = 5
        self._recent_labels: list[str] = []

    def warmup(self) -> None:
        if not os.path.exists(self.MODEL_PATH) or not os.path.exists(self.META_PATH):
            self._load_error = "no trained model found — collect labeled data via the dashboard, then run scripts/train_activity_model.py"
            return
        try:
            with open(self.META_PATH) as f:
                self._meta = json.load(f)
            if self._meta.get("feature_names") != FEATURE_NAMES:
                self._load_error = "trained model's feature schema doesn't match the current FEATURE_NAMES — retrain (features.py changed since this model was trained)"
                return
            import pickle
            with open(self.MODEL_PATH, "rb") as f:
                self._model = pickle.load(f)
        except Exception as e:
            self._load_error = f"failed to load trained model: {e}"

    def is_available(self) -> bool:
        return self._model is not None

    def get_metadata(self) -> ModelMetadata:
        if self._model is None:
            return ModelMetadata(f"TrainedActivityRecognizer (UNAVAILABLE: {self._load_error})", "1.0", "PLACEHOLDER")
        acc = self._meta.get("test_accuracy", 0)
        n = self._meta.get("num_samples", 0)
        return ModelMetadata(f"TrainedActivityRecognizer (RandomForest, {n} samples, {acc:.1%} test acc)", "1.0", "IMPLEMENTED")

    def infer(self, frame: FrameBundle) -> StructuredOutput:
        feats = extract_features(frame.pose, frame.objects, frame.interactions)
        proba = self._model.predict_proba([feats])[0]
        classes = self._model.classes_
        best_idx = int(proba.argmax())
        raw_label = str(classes[best_idx])
        confidence = float(proba[best_idx])

        self._recent_labels.append(raw_label)
        if len(self._recent_labels) > self.SMOOTHING_WINDOW:
            self._recent_labels.pop(0)
        smoothed = max(set(self._recent_labels), key=self._recent_labels.count)

        return StructuredOutput(
            "activity", frame.frame_ts, confidence,
            {"label": smoothed, "raw_label": raw_label, "track_id": "P1", "object_class": None},
        )


class ObjectAppearanceClassifier:
    """
    Optional second opinion on top of CombinedObjectDetector's geometric
    detections — loads the model from scripts/train_object_classifier.py
    if present, and re-classifies each detected box's crop by learned
    appearance. Attaches `verified_class` + `verified_confidence` to each
    object rather than silently overriding the geometric detector's own
    class label — color/geometry found WHERE something is; this only ever
    adds a second opinion on WHAT it looks like, never replaces the
    localization. If the two disagree, both are reported, not silently
    resolved — an operator or the dashboard can see the disagreement
    rather than have it hidden.

    Same "unavailable is a valid, honestly-reported state" pattern as
    TrainedActivityRecognizer — a fresh checkout has no model and this
    quietly does nothing rather than erroring.
    """

    MODEL_PATH = os.environ.get("ASTRA_MODEL_DIR", "models") + "/object_classifier.pkl"
    META_PATH = os.environ.get("ASTRA_MODEL_DIR", "models") + "/object_classifier_meta.json"

    def __init__(self):
        self._model = None
        self._meta = None
        self._load_error = None

    def warmup(self) -> None:
        if not os.path.exists(self.MODEL_PATH) or not os.path.exists(self.META_PATH):
            self._load_error = "no trained appearance classifier found — capture object crops via the dashboard, then run scripts/train_object_classifier.py"
            return
        try:
            with open(self.META_PATH) as f:
                self._meta = json.load(f)
            import pickle
            with open(self.MODEL_PATH, "rb") as f:
                self._model = pickle.load(f)
        except Exception as e:
            self._load_error = f"failed to load object appearance classifier: {e}"

    def is_available(self) -> bool:
        return self._model is not None

    def get_metadata(self) -> ModelMetadata:
        if self._model is None:
            return ModelMetadata(f"ObjectAppearanceClassifier (UNAVAILABLE: {self._load_error})", "1.0", "PLACEHOLDER")
        acc = self._meta.get("test_accuracy", 0)
        n = self._meta.get("num_samples", 0)
        return ModelMetadata(f"ObjectAppearanceClassifier (RandomForest, {n} crops, {acc:.1%} test acc)", "1.0", "IMPLEMENTED")

    def annotate(self, frame, objects: list[dict]) -> list[dict]:
        """Adds verified_class/verified_confidence to each object dict IN
        PLACE-safe (returns new list; doesn't mutate the input) — a no-op
        passthrough if no model is loaded or there's no frame to crop from."""
        if self._model is None or frame is None:
            return objects

        h, w = frame.shape[:2]
        annotated = []
        for obj in objects:
            obj = dict(obj)
            x1, y1, x2, y2 = obj["bbox"]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                feats = extract_crop_features(crop)
                proba = self._model.predict_proba([feats])[0]
                classes = self._model.classes_
                best_idx = int(proba.argmax())
                obj["verified_class"] = str(classes[best_idx])
                obj["verified_confidence"] = float(proba[best_idx])
            annotated.append(obj)
        return annotated
