"""
Converts one frame's already-computed perception output (pose + tracked
objects + hand-object interaction) into a fixed-length numeric feature
vector — the fused representation Section 8 describes as the input to a
trained activity classifier.

Used in two places that MUST stay in lockstep:
  1. backend/dataset/recorder.py — records (features, label) pairs during
     a live run for later training.
  2. backend/ai_pipeline/real_models.py's TrainedActivityRecognizer —
     extracts the same features at inference time.

If you change FEATURE_NAMES here, any previously-trained model becomes
incompatible (its input shape no longer matches) — scripts/train_activity_model.py
stores FEATURE_NAMES in the model's metadata file specifically so this
mismatch is caught loudly at load time instead of silently producing
garbage predictions.
"""
from __future__ import annotations

FEATURE_NAMES = [
    "right_wrist_visible",
    "left_wrist_visible",
    "right_wrist_x", "right_wrist_y",
    "left_wrist_x", "left_wrist_y",
    "right_hand_state_code",   # NONE=0 APPROACH=1 TOUCH=2 HOLD=3 RELEASE=4
    "left_hand_state_code",
    "right_distance_px",
    "left_distance_px",
    "right_wrist_velocity",
    "left_wrist_velocity",
    "right_object_motion_delta",
    "left_object_motion_delta",
    "num_tracked_objects",
]

_STATE_CODES = {"NONE": 0, "APPROACH": 1, "TOUCH": 2, "HOLD": 3, "RELEASE": 4}


def extract_features(pose: dict | None, tracked_objects: list | None, interactions: list | None) -> list[float]:
    """
    Pure function: same inputs always produce the same feature vector.
    Missing data (e.g. pose not available) degrades to zeros rather than
    raising — a frame with no camera should be a valid (if uninformative)
    training/inference sample, not a crash.
    """
    joints = (pose or {}).get("joints", {}) if pose else {}
    tracked_objects = tracked_objects or []
    interactions = {i["hand"]: i for i in (interactions or [])}

    def joint_xy(name):
        j = joints.get(name)
        if not j or j.get("visibility", 0) < 0.3:
            return 0.0, 0.0, 0.0
        return 1.0, float(j["x"]), float(j["y"])

    r_vis, r_x, r_y = joint_xy("right_wrist")
    l_vis, l_x, l_y = joint_xy("left_wrist")

    def hand_features(hand_name):
        i = interactions.get(hand_name)
        if not i:
            return 0.0, 0.0, 0.0, 0.0
        return (
            float(_STATE_CODES.get(i["state"], 0)),
            float(i.get("distance_px", 0.0)),
            float(i.get("wrist_velocity_px_s", 0.0)),
            float(i.get("object_motion_delta", 0.0)),
        )

    r_state, r_dist, r_vel, r_motion = hand_features("right_wrist")
    l_state, l_dist, l_vel, l_motion = hand_features("left_wrist")

    return [
        r_vis, l_vis,
        r_x, r_y, l_x, l_y,
        r_state, l_state,
        r_dist, l_dist,
        r_vel, l_vel,
        r_motion, l_motion,
        float(len(tracked_objects)),
    ]
