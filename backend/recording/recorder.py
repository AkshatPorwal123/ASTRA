"""
Local recording with a sidecar events.jsonl correlating video timestamps
to AI/procedure events, enabling synchronized replay (Section 21/28).
"""
from __future__ import annotations
import json
import os
import time
import cv2


class RunRecorder:
    def __init__(self, run_id: str, output_dir: str = "recordings", fps: int = 15, frame_size=(640, 480)):
        self.run_id = run_id
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.video_path = os.path.join(output_dir, f"{run_id}.mp4")
        self.events_path = os.path.join(output_dir, f"{run_id}.events.jsonl")
        self._writer = cv2.VideoWriter(
            self.video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size
        )
        self._events_file = open(self.events_path, "a")

    def write_frame(self, frame):
        if frame is not None:
            self._writer.write(frame)

    def log_event(self, event: dict):
        event = {**event, "recorded_at": time.time()}
        self._events_file.write(json.dumps(event) + "\n")
        self._events_file.flush()

    def close(self):
        self._writer.release()
        self._events_file.close()
