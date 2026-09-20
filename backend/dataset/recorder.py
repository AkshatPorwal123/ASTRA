"""
Records (features, label) pairs during a live run, driven by the person
running the experiment confirming or correcting the rule-based recognizer's
suggested label — this is the actual "dataset generation" the official
problem statement calls for (not synthetic — real labeled examples from
real runs against the real camera/objects).

One JSONL file per run under datasets/, one line per labeled sample:
    {"ts": ..., "features": [...], "label": "PICK", "source": "confirmed"}

`source` distinguishes a label the person explicitly confirmed/corrected
("confirmed") from one auto-suggested but never reviewed — training only
uses "confirmed" samples (see scripts/train_activity_model.py), so a
person who never touches the labeling UI doesn't silently pollute the
dataset with unverified auto-labels.
"""
from __future__ import annotations
import os
import json
import time

DATASETS_DIR = os.environ.get("ASTRA_DATASETS_DIR", "datasets")


class DatasetRecorder:
    def __init__(self, run_id: str):
        self.run_id = run_id
        os.makedirs(DATASETS_DIR, exist_ok=True)
        self.path = os.path.join(DATASETS_DIR, f"{run_id}.jsonl")
        self._last_features: list[float] | None = None
        self._last_suggested_label: str | None = None

    def update_current_frame(self, features: list[float], suggested_label: str):
        """Called every frame so record_label() can attach the right
        feature vector to whatever the person confirms next, without the
        API layer needing to re-derive features itself."""
        self._last_features = features
        self._last_suggested_label = suggested_label

    def record_label(self, label: str, ts: float | None = None) -> dict:
        if self._last_features is None:
            raise ValueError("No frame processed yet for this run — nothing to label.")
        sample = {
            "ts": ts or time.time(),
            "features": self._last_features,
            "label": label,
            "suggested_label": self._last_suggested_label,
            "source": "confirmed",
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(sample) + "\n")
        return sample

    def current_suggestion(self) -> str | None:
        return self._last_suggested_label
