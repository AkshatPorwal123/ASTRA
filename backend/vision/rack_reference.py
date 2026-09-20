"""
Addresses the official problem statement's OPTIONAL challenge:

    "Standard 2D or ground-based 3D posture models fail because
    astronauts do not have a fixed 'up' or 'down' orientation. The AI
    model should use orientation-agnostic 3D Human Mesh Recovery (HMR) to
    track the astronaut's body relative to the payload rack, not the
    floor."

SCOPE NOTE, stated plainly: this is NOT full 3D Human Mesh Recovery.
HMR (SMPL-based pipelines like HMR/SPIN/PIXIE) needs body-model weights
that are gated behind a manual license-agreement registration
(https://smpl.is.tue.mpg.de/) — there is no way to obtain them in this
environment, and shipping a "HMR module" that can't actually load its
required model files would be exactly the kind of unverified, faked
capability this whole project has deliberately avoided at every step.

What this DOES genuinely solve, and fully tests: the actual underlying
problem stated above — "no fixed up/down, track relative to the rack, not
the floor" — using ArUco fiducial marker detection (already in
opencv-contrib-python, no new dependencies, no license gate) as a
physical reference frame. A marker fixed to the payload rack gives a
real, stable "this way is up relative to the rack" reference that doesn't
depend on gravity or camera orientation — which is the actual physical
problem an astronaut's lack of fixed orientation creates. Pose keypoints
get expressed relative to that marker's in-frame rotation instead of raw
image coordinates, so "the astronaut is reaching up" means "up relative
to the rack", not "toward the top of the camera frame" — solving the
real problem with 2D geometry, not the heavier 3D mesh pipeline the
spec's OPTIONAL text suggests as one possible (not mandatory) approach.

Requires a printable ArUco marker (DICT_4X4_50, ID 0 by default) affixed
to the payload rack in view of the camera. Generate one with:
    python scripts/generate_rack_marker.py
"""
from __future__ import annotations
import math
import cv2
import numpy as np

ARUCO_DICT = cv2.aruco.DICT_4X4_50
DEFAULT_MARKER_ID = 0


class RackReferenceTracker:
    def __init__(self, marker_id: int = DEFAULT_MARKER_ID):
        self.marker_id = marker_id
        self._dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self._detector = cv2.aruco.ArucoDetector(self._dictionary, cv2.aruco.DetectorParameters())
        self.last_marker_center: tuple[float, float] | None = None
        self.last_marker_angle_deg: float | None = None

    def detect(self, frame) -> dict:
        """Returns the rack marker's in-frame position and rotation angle
        this frame, or a clear 'not visible' result — never a stale/guessed
        value silently reused, since a wrong reference frame would corrupt
        every downstream joint transform."""
        if frame is None:
            return {"visible": False, "reason": "no frame"}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)

        if ids is None or self.marker_id not in ids.flatten():
            self.last_marker_center = None
            self.last_marker_angle_deg = None
            return {"visible": False, "reason": "marker not in view"}

        idx = list(ids.flatten()).index(self.marker_id)
        pts = corners[idx][0]   # 4 corner points, in order: top-left, top-right, bottom-right, bottom-left

        center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
        # Angle of the marker's own "up" edge (top-left -> top-right) relative
        # to the image's horizontal — this defines what "up relative to the
        # rack" means in this frame, regardless of how the camera itself is
        # mounted or rotated.
        dx = pts[1][0] - pts[0][0]
        dy = pts[1][1] - pts[0][1]
        angle_deg = math.degrees(math.atan2(dy, dx))

        self.last_marker_center = center
        self.last_marker_angle_deg = angle_deg
        return {"visible": True, "center": center, "angle_deg": angle_deg, "corners": pts.tolist()}

    @staticmethod
    def to_rack_relative(joint_xy: tuple[float, float], marker_center: tuple[float, float],
                          marker_angle_deg: float) -> tuple[float, float]:
        """Rotates+translates an image-coordinate point into the rack
        marker's own frame: origin at the marker center, and the marker's
        own top edge defining the rack's 'up' direction, instead of the
        image's top edge (which means nothing in zero-g)."""
        x, y = joint_xy[0] - marker_center[0], joint_xy[1] - marker_center[1]
        theta = -math.radians(marker_angle_deg)
        rx = x * math.cos(theta) - y * math.sin(theta)
        ry = x * math.sin(theta) + y * math.cos(theta)
        return (rx, ry)

    def transform_joints(self, joints: dict, marker_result: dict) -> dict | None:
        """Applies to_rack_relative() to every visible joint. Returns None
        (not a guess, not the last-known frame) if the marker isn't
        visible this frame — a silently-stale reference frame would be
        actively misleading for exactly the orientation-tracking problem
        this exists to solve."""
        if not marker_result.get("visible"):
            return None
        center, angle = marker_result["center"], marker_result["angle_deg"]
        out = {}
        for name, j in joints.items():
            if j.get("visibility", 0) < 0.3:
                continue
            rx, ry = self.to_rack_relative((j["x"], j["y"]), center, angle)
            out[name] = {"x": rx, "y": ry, "visibility": j["visibility"]}
        return out
