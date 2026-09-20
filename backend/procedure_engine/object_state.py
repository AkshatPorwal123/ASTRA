"""
Object State Tracker (Section 9 / 14 of the design doc).

Maintains the authoritative per-object *attributes* that the procedure
engine's preconditions/postconditions are evaluated against. The primary
attribute is `state` (AVAILABLE, HELD, OPEN, CLOSED, FILLED, EMPTY, MOVED,
PLACED, REMOVED, UNKNOWN, ... per Section 9), but the schema's example
configs also reference secondary boolean/positional attributes (e.g.
`container.mixed == true`, `container.location == rack` in
liquid_mixing_v1.yaml) — so this tracks arbitrary `<object>.<attribute>`
pairs, not just `.state`, rather than forcing every experiment author to
awkwardly encode everything into one `state` enum.

Design choice (documented, not hidden): state is updated ONLY by validated
step postconditions — i.e. once the procedure engine has actually CONFIRMED
a step (sustained evidence, per Section 17), not by raw per-frame
hand-object interaction signal. This keeps a single, deterministic source
of truth for "what state is this object in" rather than letting noisy
per-frame perception and validated procedure state disagree with each
other. A richer version could reconcile both sources (Section 9's
multimodal fusion) — that's a research extension, not implemented here.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import time


@dataclass
class ObjectStateTracker:
    # key: "<object_id>.<attribute>" -> value string
    attributes: dict[str, str] = field(default_factory=dict)
    history: dict[str, list[dict]] = field(default_factory=dict)

    @staticmethod
    def _key(object_id: str, attribute: str = "state") -> str:
        return f"{object_id}.{attribute}"

    def seed(self, object_id: str, initial_state: str = "AVAILABLE", attribute: str = "state"):
        self.attributes.setdefault(self._key(object_id, attribute), initial_state)

    def get(self, object_id: str, attribute: str = "state") -> str:
        return self.attributes.get(self._key(object_id, attribute), "UNKNOWN")

    def set(self, object_id: str, value: str, attribute: str = "state",
            ts: float | None = None, source: str = "postcondition"):
        key = self._key(object_id, attribute)
        self.attributes[key] = value
        self.history.setdefault(key, []).append({
            "value": value, "ts": ts if ts is not None else time.time(), "source": source,
        })

    def snapshot(self) -> dict:
        """Returns {object_id: {attribute: value}} for reporting/dashboard use."""
        out: dict[str, dict[str, str]] = {}
        for key, value in self.attributes.items():
            obj_id, _, attr = key.partition(".")
            out.setdefault(obj_id, {})[attr or "state"] = value
        return out
