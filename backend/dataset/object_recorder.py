"""
Captures labeled object-detection samples during a live run — the second
half of the official problem's "Dataset generation to train model for
object detection, pose estimation and hand-object interaction" that
Milestone 8 didn't cover (Milestone 8 only handled activity labels).

Unlike the activity dataset (a numeric feature vector per sample), an
object detector needs the actual image plus box coordinates. Saves in a
simple, standard-adjacent format:

    datasets/object_detection/{run_id}/
        images/{sample_id}.jpg
        annotations.jsonl   — one line per sample:
            {"image": "images/0001.jpg", "boxes": [{"class": "red_box", "bbox": [x1,y1,x2,y2]}], "ts": ...}

This intentionally captures the CURRENT detector's own output as the
starting annotation (operator corrects/confirms via the dashboard, same
"suggest then confirm" pattern as the activity labeling panel) rather than
requiring the operator to draw boxes from scratch — much faster to
generate real volume this way, and a from-scratch annotation tool is a
much bigger UI undertaking than this project's scope justifies right now.
"""
from __future__ import annotations
import os
import json
import cv2

DATASET_ROOT = os.environ.get("ASTRA_OBJECT_DATASET_DIR", "datasets/object_detection")


class ObjectDatasetRecorder:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.run_dir = os.path.join(DATASET_ROOT, run_id)
        self.images_dir = os.path.join(self.run_dir, "images")
        self.annotations_path = os.path.join(self.run_dir, "annotations.jsonl")
        self._sample_count = 0
        self._last_frame = None
        self._last_boxes: list[dict] = []

    def update_current_frame(self, frame, boxes: list[dict]):
        """Called every loop iteration so capture_sample() has the current
        frame + the detector's current suggested boxes ready, without the
        API layer needing to reach into orchestrator internals."""
        self._last_frame = frame
        self._last_boxes = boxes

    def capture_sample(self, confirmed_boxes: list[dict] | None = None, ts: float | None = None) -> dict:
        """Saves the current frame + boxes. `confirmed_boxes` lets the
        operator override the detector's own suggestion (e.g. it missed a
        box, or mislabeled the color); if omitted, saves the detector's
        current suggestion as-is — still useful as a starting point for
        review later, just not implicitly treated as ground truth the way
        DatasetRecorder's "confirmed" activity samples are."""
        if self._last_frame is None:
            raise ValueError("No frame available yet for this run — nothing to capture.")

        os.makedirs(self.images_dir, exist_ok=True)
        self._sample_count += 1
        image_name = f"{self._sample_count:04d}.jpg"
        image_path = os.path.join(self.images_dir, image_name)
        cv2.imwrite(image_path, self._last_frame)

        boxes = confirmed_boxes if confirmed_boxes is not None else self._last_boxes
        sample = {
            "image": os.path.join("images", image_name),
            "boxes": boxes,
            "ts": ts,
            "confirmed": confirmed_boxes is not None,
        }
        with open(self.annotations_path, "a") as f:
            f.write(json.dumps(sample) + "\n")
        return sample
