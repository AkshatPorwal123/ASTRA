"""
Trains an appearance classifier from the object-detection dataset captured
via the dashboard's "Capture frame for object-detection training" button
(backend/dataset/object_recorder.py) — closing the loop Milestone 11 left
half-finished (capture existed, nothing consumed it).

IMPORTANT SCOPE NOTE, stated plainly rather than overclaimed: every
captured sample is a region some detector already proposed as a box —
there are no negative/background examples. That makes training a full
from-scratch object DETECTOR (localization + classification) unsound with
this data; a detector needs to learn what ISN'T a box too. What this
script trains instead is an APPEARANCE CLASSIFIER: given a region some
detector (color-based or MobileNet-SSD) already proposed, learn to
recognize which of YOUR specific labeled objects it is, from real crops.
This is a genuine, correctly-scoped use of the captured data — a second
opinion layered on top of geometric detection, not a replacement for it.

Usage:
    python scripts/train_object_classifier.py [--datasets-dir datasets/object_detection] [--min-samples 20]

Output:
    models/object_classifier.pkl
    models/object_classifier_meta.json
"""
import argparse
import glob
import json
import os
import pickle
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from backend.ai_pipeline.object_features import extract_crop_features, CROP_SIZE, FEATURE_DIM


def load_dataset(datasets_dir: str):
    X, y = [], []
    run_dirs = sorted(glob.glob(os.path.join(datasets_dir, "*")))
    samples_seen, samples_used = 0, 0
    for run_dir in run_dirs:
        ann_path = os.path.join(run_dir, "annotations.jsonl")
        if not os.path.exists(ann_path):
            continue
        with open(ann_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                image_path = os.path.join(run_dir, row["image"])
                img = cv2.imread(image_path)
                if img is None:
                    continue
                h, w = img.shape[:2]
                for box in row.get("boxes", []):
                    samples_seen += 1
                    x1, y1, x2, y2 = box["bbox"]
                    x1, y1 = max(0, int(x1)), max(0, int(y1))
                    x2, y2 = min(w, int(x2)), min(h, int(y2))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop = img[y1:y2, x1:x2]
                    X.append(extract_crop_features(crop))
                    y.append(box["class"])
                    samples_used += 1
    return X, y, samples_seen, samples_used, len(run_dirs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-dir", default=os.environ.get("ASTRA_OBJECT_DATASET_DIR", "datasets/object_detection"))
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--min-samples", type=int, default=20,
                         help="Refuse to train below this many usable crops — same principle as "
                              "train_activity_model.py: a model 'trained' on too little data is "
                              "worse than no model at all.")
    args = parser.parse_args()

    X, y, seen, used, n_runs = load_dataset(args.datasets_dir)
    print(f"Scanned {n_runs} run(s), {seen} annotated boxes, {used} usable crops (after bounds clipping).")

    if len(X) < args.min_samples:
        print(f"\nNot enough data yet: {used} usable crops, need at least {args.min_samples}.")
        print("Use the dashboard's 'Capture frame for object-detection training' button during more "
              "runs to build this up. This is expected early on, not an error.")
        sys.exit(1)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    classes = sorted(set(y))
    if len(classes) < 2:
        print(f"\nOnly one class present ({classes}) — need at least 2 distinct classes to train a "
              f"classifier. Capture crops of more than one labeled object/color and try again.")
        sys.exit(1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if min(y.count(c) for c in classes) >= 2 else None,
    )

    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {accuracy:.3f} (on {len(X_test)} held-out crops)")
    print(classification_report(y_test, y_pred, zero_division=0))

    os.makedirs(args.models_dir, exist_ok=True)
    model_path = os.path.join(args.models_dir, "object_classifier.pkl")
    meta_path = os.path.join(args.models_dir, "object_classifier_meta.json")

    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_dim": FEATURE_DIM,
        "crop_size": list(CROP_SIZE),
        "classes": classes,
        "num_samples": len(X),
        "test_accuracy": accuracy,
        "scope_note": "Appearance classifier for regions another detector already proposed. "
                       "Not a standalone detector — trained on positive crops only, no negative examples.",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved model to {model_path}")
    print(f"Saved metadata to {meta_path}")
    print("Restart the backend to pick it up.")


if __name__ == "__main__":
    main()
