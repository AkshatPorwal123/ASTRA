"""
Shared crop-appearance feature extraction, used by both
scripts/train_object_classifier.py (training) and
backend/ai_pipeline/real_models.py's ObjectAppearanceClassifier
(inference) — kept here, not in scripts/, so backend code never has to
import from the scripts/ directory (fragile: depends on how the process
was launched and whether cwd ends up on sys.path). scripts/ imports FROM
backend instead, same direction as Milestone 8's features.py.
"""
from __future__ import annotations
import cv2
import numpy as np

CROP_SIZE = (64, 64)
FEATURE_DIM = 16 + 8 + 8 + 9   # recorded in training metadata so inference can validate compatibility


def extract_crop_features(crop_bgr) -> list[float]:
    """Color histogram (HSV, coarse bins) + a simple HOG-style gradient
    histogram — cheap, standard baseline features for a small
    appearance-classification task like this one."""
    resized = cv2.resize(crop_bgr, CROP_SIZE)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
    color_features = np.concatenate([hist_h, hist_s, hist_v])
    color_features = color_features / (color_features.sum() + 1e-6)   # normalize — invariant to crop size

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ang = (np.arctan2(gy, gx) * 180 / np.pi) % 180
    hist_grad, _ = np.histogram(ang, bins=9, range=(0, 180), weights=mag)
    hist_grad = hist_grad / (hist_grad.sum() + 1e-6)

    return list(color_features) + list(hist_grad)
