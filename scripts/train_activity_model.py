"""
Trains a lightweight activity classifier from the JSONL datasets recorded
via the dashboard's labeling panel (backend/dataset/recorder.py) — the
actual "trained AI model" deliverable the official problem statement asks
for, built from real labeled examples of the real experiment.

Usage:
    python scripts/train_activity_model.py [--datasets-dir datasets] [--min-samples 30]

Only "confirmed" samples are used (see recorder.py) — auto-suggested
labels the person never reviewed don't count as ground truth.

Output:
    models/activity_classifier.pkl        — the trained sklearn model
    models/activity_classifier_meta.json  — classes, feature schema,
                                             accuracy, sample count, so
                                             TrainedActivityRecognizer can
                                             validate compatibility at
                                             load time instead of silently
                                             feeding it mismatched features.

Deliberately a small RandomForest, not a deep model — Section 40's
incremental principle: this is the first real trained baseline, not the
final word. It needs FAR less data than a temporal deep net (ST-GCN/TCN)
to produce something legitimately better than the hand-tuned rule-based
recognizer, which is the actual goal at this dataset size.
"""
import argparse
import glob
import json
import os
import pickle
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.ai_pipeline.features import FEATURE_NAMES


def load_dataset(datasets_dir: str):
    X, y = [], []
    files = sorted(glob.glob(os.path.join(datasets_dir, "*.jsonl")))
    for path in files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("source") != "confirmed":
                    continue
                if len(row["features"]) != len(FEATURE_NAMES):
                    print(f"  WARNING: skipping a sample in {path} — feature count "
                          f"{len(row['features'])} != current schema {len(FEATURE_NAMES)} "
                          f"(FEATURE_NAMES changed since this was recorded)")
                    continue
                X.append(row["features"])
                y.append(row["label"])
    return X, y, files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-dir", default=os.environ.get("ASTRA_DATASETS_DIR", "datasets"))
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--min-samples", type=int, default=30,
                         help="Refuse to train below this many labeled samples — a model "
                              "'trained' on a handful of examples is worse than the honest "
                              "rule-based baseline it would replace.")
    args = parser.parse_args()

    X, y, files = load_dataset(args.datasets_dir)
    print(f"Loaded {len(X)} confirmed samples from {len(files)} run(s) in {args.datasets_dir}/")

    if len(X) < args.min_samples:
        print(f"\nNot enough data yet: {len(X)} samples, need at least {args.min_samples}.")
        print("Run more experiments and use the dashboard's Dataset Labeling panel to confirm "
              "labels as you go — each confirmed tap is one training sample. This is expected "
              "at the start of data collection, not an error.")
        sys.exit(1)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    classes = sorted(set(y))
    if len(classes) < 2:
        print(f"\nOnly one class present ({classes}) — need at least 2 distinct labels to train "
              f"a classifier. Label a wider variety of activities and try again.")
        sys.exit(1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if min(y.count(c) for c in classes) >= 2 else None,
    )

    clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"\nTest accuracy: {accuracy:.3f} (on {len(X_test)} held-out samples)")
    print(report)

    os.makedirs(args.models_dir, exist_ok=True)
    model_path = os.path.join(args.models_dir, "activity_classifier.pkl")
    meta_path = os.path.join(args.models_dir, "activity_classifier_meta.json")

    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_names": FEATURE_NAMES,
        "classes": classes,
        "num_samples": len(X),
        "num_train": len(X_train),
        "num_test": len(X_test),
        "test_accuracy": accuracy,
        "source_files": [os.path.basename(f) for f in files],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved model to {model_path}")
    print(f"Saved metadata to {meta_path}")
    print("\nRestart the backend to pick up the new model — TrainedActivityRecognizer "
          "loads it automatically if present and falls back to the rule-based baseline if not.")


if __name__ == "__main__":
    main()
