"""
Fixes the "not tuned to your lighting" limitation flagged since Milestone
5 — instead of relying solely on the hardcoded COLOR_HSV_RANGES defaults
in real_models.py, an operator can click on their actual box in the live
video and have the system derive an HSV range from what the camera is
actually seeing, under actual lighting.

Calibrated ranges are saved to configs/calibrated_colors.json and merged
over the hardcoded defaults at ColorBoxDetector construction time — same
"takes effect on next run start" pattern as the trained activity model in
Milestone 8, not hot-swapped mid-run, to avoid concurrency complications.
"""
from __future__ import annotations
import os
import json
import cv2
import numpy as np

CALIBRATION_PATH = os.environ.get("ASTRA_CALIBRATION_PATH", "configs/calibrated_colors.json")


def sample_hsv_range(frame, x: int, y: int, radius: int = 15, std_multiplier: float = 2.5) -> tuple:
    """
    Samples an NxN neighborhood around (x, y) in the given BGR frame and
    derives an HSV range wide enough to cover normal lighting variation
    but tight enough to stay specific to the sampled color.

    Returns ((h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi)). Raises ValueError if
    (x, y) is out of bounds — a clear error beats a silent wraparound or a
    crash deep in numpy indexing.
    """
    h, w = frame.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError(f"Sample point ({x}, {y}) is outside the frame ({w}x{h}).")

    x0, x1 = max(0, x - radius), min(w, x + radius)
    y0, y1 = max(0, y - radius), min(h, y + radius)
    patch = frame[y0:y1, x0:x1]
    hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

    h_vals = hsv_patch[:, :, 0].astype(np.float32)
    s_vals = hsv_patch[:, :, 1].astype(np.float32)
    v_vals = hsv_patch[:, :, 2].astype(np.float32)

    # Hue wraps around 0/180 (red straddles this), so compute its mean via
    # circular statistics rather than a naive average, which would badly
    # misplace the center for a color like red (e.g. averaging 2 and 178
    # naively gives 90 — pure cyan, the opposite of red).
    angles = h_vals * (2 * np.pi / 180.0)
    mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    h_mean = (mean_angle * 180.0 / (2 * np.pi)) % 180
    # circular std approximation
    r = np.sqrt(np.mean(np.sin(angles)) ** 2 + np.mean(np.cos(angles)) ** 2)
    h_std = np.sqrt(max(0.0, -2 * np.log(max(r, 1e-6)))) * (180.0 / (2 * np.pi))

    h_spread = min(30, max(5, h_std * std_multiplier))
    s_mean, s_std = float(np.mean(s_vals)), float(np.std(s_vals))
    v_mean, v_std = float(np.mean(v_vals)), float(np.std(v_vals))

    lo = (
        int(round(h_mean - h_spread)) % 180,
        max(30, int(round(s_mean - s_std * std_multiplier))),
        max(20, int(round(v_mean - v_std * std_multiplier))),
    )
    hi = (
        int(round(h_mean + h_spread)) % 180,
        255,
        255,
    )
    return lo, hi, {"h_mean": round(float(h_mean), 1), "s_mean": round(s_mean, 1), "v_mean": round(v_mean, 1)}


def load_calibrated_ranges() -> dict:
    if not os.path.exists(CALIBRATION_PATH):
        return {}
    with open(CALIBRATION_PATH) as f:
        return json.load(f)


def save_calibrated_range(color_name: str, lo: tuple, hi: tuple):
    ranges = load_calibrated_ranges()
    if lo[0] > hi[0]:
        # Hue wraps around the 0/180 boundary (e.g. red) — cv2.inRange has
        # no concept of circular ranges, so a single (lo_h=175, hi_h=5)
        # pair would silently match nothing. Split into two bands at the
        # boundary instead, same pattern as the hardcoded "red" entry in
        # COLOR_HSV_RANGES.
        bands = [
            [[lo[0], lo[1], lo[2]], [179, hi[1], hi[2]]],
            [[0, lo[1], lo[2]], [hi[0], hi[1], hi[2]]],
        ]
    else:
        bands = [[list(lo), list(hi)]]
    ranges[color_name] = bands
    os.makedirs(os.path.dirname(CALIBRATION_PATH) or ".", exist_ok=True)
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(ranges, f, indent=2)
