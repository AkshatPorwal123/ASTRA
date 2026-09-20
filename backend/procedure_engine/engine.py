"""
The Procedure Engine: hierarchical state machine + sequence validator +
error detector + next-action reasoner, all driven purely by the loaded
ExperimentConfig graph (Section 11/12/13/16 of the design doc).

Deliberately deterministic and rule-based on top of ML confidence inputs,
so its decisions stay explainable and auditable.

Milestone 6 (Procedure intelligence) additions over the Phase-1 baseline:
  - Preconditions / postconditions are actually evaluated, via an
    ObjectStateTracker + a small condition-expression grammar
    (conditions.py). A step can no longer "complete" just because the
    right activity+object were observed if its preconditions aren't met.
  - Simple graph branching: `branches` conditions are evaluated against
    object state after a step completes, so the next step isn't always
    a single fixed successor.
  - `repeatable` steps: a step marked repeatable can be observed again
    later without raising REPEATED.
  - `recovery_step`: after repeated timeouts on a step, the engine can
    redirect into a defined recovery path instead of just failing.

Milestone 7 (Error intelligence) additions:
  - The full Section 16 taxonomy is now actually distinguished, not just
    a subset: CORRECT (implicit — step advances), SKIPPED, WRONG_ORDER,
    REPEATED, WRONG_OBJECT, INVALID_ACTION, FAILED, TIMEOUT, UNCERTAIN.
  - TIMEOUT previously computed evidence but never turned it into an
    event — it silently vanished. Fixed: timeouts now emit real alerts,
    with retry/recovery/failure escalation.
  - Precondition violations (activity+object correct, but the object
    isn't in the state the step requires) now surface as FAILED rather
    than either silently succeeding or being lumped into UNCERTAIN.
  - RECOVERED lifecycle: once a step that had an outstanding CONFIRMED
    deviation finally completes correctly, a RECOVERED event is emitted
    for each such deviation (Section 17).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import time

from backend.config.schema import ExperimentConfig, StepDef
from .evidence import EvidenceAccumulator
from .object_state import ObjectStateTracker
from . import conditions

TERMINAL = "TERMINAL"

# Labels that mean "nothing procedurally relevant is happening yet" —
# never treated as a deviation from the expected step, just ignored for
# sequence-validation purposes (they still count toward the step's own
# timeout clock).
NON_ACTION_LABELS = {"IDLE", "REACH", "UNKNOWN"}


@dataclass
class NextActionAdvice:
    next_step: str | None
    reason: str
    required_object: str | None
    target: str | None
    voice_instruction: str | None


@dataclass
class ProcedureEngineState:
    current_step_id: str
    completed_steps: list[str] = field(default_factory=list)
    skipped_steps: list[str] = field(default_factory=list)
    status: str = "IN_PROGRESS"     # IN_PROGRESS | COMPLETED | FAILED
    step_started_at: float = field(default_factory=time.time)


class ProcedureEngine:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.state = ProcedureEngineState(current_step_id=config.first_step().id)
        self.evidence = EvidenceAccumulator(
            tentative_frames=config.thresholds.tentative_frames,
            confirmed_frames=config.thresholds.confirmed_frames,
        )
        self.objects = ObjectStateTracker()
        for obj in config.objects:
            initial = obj.states[0] if obj.states else "AVAILABLE"
            self.objects.seed(obj.id, initial)

        self._known_activities = {s.activity for s in config.steps}
        self._step_index = {s.id: i for i, s in enumerate(config.steps)}
        self._timeout_retries: dict[str, int] = {}
        self._skip_alerted: set[str] = set()

    # ------------------------------------------------------------------
    # Core update loop: called once per recognized activity observation
    # ------------------------------------------------------------------
    def observe_activity(
        self,
        activity_label: str,
        confidence: float,
        ts: float | None = None,
        object_class: str | None = None,
    ) -> dict:
        ts = ts or time.time()
        events: list[dict] = []
        current_step = self.config.step_by_id(self.state.current_step_id)
        if current_step is None or self.state.status != "IN_PROGRESS":
            return self._snapshot(events=events)

        threshold = current_step.confidence_threshold or self.config.thresholds.default_confidence
        activity_ok = (activity_label == current_step.activity) and (confidence >= threshold)
        object_ok = (
            current_step.required_object is None
            or object_class is None
            or object_class == current_step.required_object
        )
        precond_ok, failing_preconditions = (
            conditions.evaluate_all(current_step.preconditions, self.objects)
            if current_step.preconditions else (True, [])
        )
        matched = activity_ok and object_ok and precond_ok

        match_status = self.evidence.observe(current_step.id, matched, ts, confidence)

        if match_status == "CONFIRMED" and current_step.id not in self.state.completed_steps:
            events.extend(self._advance(current_step, ts))

        elif activity_ok and object_ok and not precond_ok:
            # Right action, right object, but the experiment state doesn't
            # actually allow it yet (e.g. trying to POUR before the
            # container is OPEN). Section 17: accumulate evidence before
            # calling it a real failure.
            dev_key = f"{current_step.id}::precond_fail"
            prev = self.evidence.status_of(dev_key)
            dev_status = self.evidence.observe(dev_key, True, ts, confidence)
            if dev_status == "CONFIRMED" and prev != "CONFIRMED":
                events.append({
                    "type": "alert", "alert_type": "FAILED", "step_id": current_step.id,
                    "ts": ts, "severity": "ERROR",
                    "evidence": {"reason": "preconditions_not_met", "failing": failing_preconditions},
                })

        elif activity_ok and not object_ok:
            dev_key = f"{current_step.id}::wrong_object::{object_class}"
            prev = self.evidence.status_of(dev_key)
            dev_status = self.evidence.observe(dev_key, True, ts, confidence)
            if dev_status == "CONFIRMED" and prev != "CONFIRMED":
                events.append({
                    "type": "alert", "alert_type": "WRONG_OBJECT", "step_id": current_step.id,
                    "ts": ts,
                    "evidence": {"expected_object": current_step.required_object, "observed_object": object_class},
                })

        elif not activity_ok and activity_label not in NON_ACTION_LABELS:
            dev_key = f"{current_step.id}::deviation::{activity_label}"
            prev = self.evidence.status_of(dev_key)
            dev_status = self.evidence.observe(dev_key, True, ts, confidence)
            if dev_status == "CONFIRMED" and prev != "CONFIRMED":
                events.extend(self._classify_deviation(current_step, activity_label, ts, confidence))

        events.extend(self._check_timeout(current_step, ts))
        return self._snapshot(events=events)

    # ------------------------------------------------------------------
    def _advance(self, step: StepDef, ts: float) -> list[dict]:
        self.state.completed_steps.append(step.id)
        if step.postconditions:
            conditions.apply_postconditions(step.postconditions, self.objects, ts=ts)

        events: list[dict] = []

        # RECOVERED: any deviation track that this step accumulated while
        # blocked (wrong object, wrong action, precondition failure) and
        # that reached CONFIRMED is now resolved, since the step just
        # completed correctly.
        for dev_key in self.evidence.confirmed_keys_with_prefix(f"{step.id}::"):
            if dev_key == f"{step.id}::timeout":
                alert_type = "TIMEOUT"
            elif dev_key == f"{step.id}::precond_fail":
                alert_type = "FAILED"
            elif "::wrong_object::" in dev_key:
                alert_type = "WRONG_OBJECT"
            elif "::deviation::" in dev_key:
                alert_type = "UNCERTAIN"
            else:
                alert_type = "UNKNOWN"
            events.append({
                "type": "alert_recovered", "original_alert_type": alert_type,
                "step_id": step.id, "ts": ts,
            })
            self.evidence.reset(dev_key)

        next_id = step.next or step.on_success
        for cond_expr, target_id in step.branches.items():
            try:
                if conditions.evaluate_condition(cond_expr, self.objects):
                    next_id = target_id
                    break
            except conditions.ConditionError:
                continue  # malformed branch condition — fall back to default `next`

        events.append({
            "type": "step_completed", "step_id": step.id,
            "next_step_id": next_id or TERMINAL, "ts": ts,
        })

        if next_id is None:
            self.state.status = "COMPLETED"
            self.state.current_step_id = TERMINAL
        else:
            self.state.current_step_id = next_id
            self.state.step_started_at = ts

        self.evidence.reset(step.id)
        self._timeout_retries.pop(step.id, None)
        return events

    def _advance_skip(self, step: StepDef, ts: float) -> list[dict]:
        """Like _advance, but for an optional step auto-skipped after
        timeout: no postconditions applied (nothing actually happened),
        recorded as SKIPPED rather than COMPLETED."""
        self.state.skipped_steps.append(step.id)
        next_id = step.next or step.on_success
        events = [{"type": "step_skipped", "step_id": step.id, "next_step_id": next_id or TERMINAL, "ts": ts}]
        if next_id is None:
            self.state.status = "COMPLETED"
            self.state.current_step_id = TERMINAL
        else:
            self.state.current_step_id = next_id
            self.state.step_started_at = ts
        self.evidence.reset(step.id)
        self._timeout_retries.pop(step.id, None)
        return events

    def _classify_deviation(self, step: StepDef, label: str, ts: float, confidence: float) -> list[dict]:
        """Classifies an unmatched, non-filler observation against the
        sequence-validation taxonomy (Section 16): INVALID_ACTION,
        REPEATED, WRONG_ORDER (+ any SKIPPED steps it implies), UNCERTAIN.
        """
        if label not in self._known_activities:
            return [{
                "type": "alert", "alert_type": "INVALID_ACTION", "step_id": step.id, "ts": ts,
                "evidence": {"observed_label": label, "confidence": confidence},
            }]

        matching_completed = next(
            (s for s in self.config.steps if s.activity == label and s.id in self.state.completed_steps),
            None,
        )
        if matching_completed:
            if matching_completed.repeatable:
                return []  # expected repeat of a repeatable action — not an error
            return [{"type": "alert", "alert_type": "REPEATED", "step_id": matching_completed.id, "ts": ts}]

        matching_future = next(
            (s for s in self.config.steps
             if s.activity == label and s.id not in self.state.completed_steps and s.id != step.id),
            None,
        )
        if matching_future:
            events: list[dict] = []
            cur_idx = self._step_index.get(step.id, 0)
            fut_idx = self._step_index.get(matching_future.id, cur_idx)
            for skipped in self.config.steps[cur_idx:fut_idx]:
                if skipped.id in self.state.completed_steps or skipped.optional or skipped.id == matching_future.id:
                    continue
                if skipped.id not in self._skip_alerted:
                    self._skip_alerted.add(skipped.id)
                    events.append({
                        "type": "alert", "alert_type": "SKIPPED", "step_id": skipped.id, "ts": ts,
                        "evidence": {"observed_activity": label, "jumped_to": matching_future.id},
                    })
            events.append({
                "type": "alert", "alert_type": "WRONG_ORDER", "step_id": matching_future.id, "ts": ts,
                "evidence": {"current_step": step.id},
            })
            return events

        return [{
            "type": "alert", "alert_type": "UNCERTAIN", "step_id": step.id, "ts": ts,
            "evidence": {"observed_label": label, "confidence": confidence},
        }]

    def _check_timeout(self, step: StepDef, ts: float) -> list[dict]:
        timeout = step.timeout_s or self.config.thresholds.default_timeout_s
        if ts - self.state.step_started_at <= timeout:
            return []

        dev_key = f"{step.id}::timeout"
        prev = self.evidence.status_of(dev_key)
        status = self.evidence.observe(dev_key, True, ts, 1.0)
        if not (status == "CONFIRMED" and prev != "CONFIRMED"):
            return []

        events: list[dict] = []
        if step.optional:
            events.append({
                "type": "alert", "alert_type": "TIMEOUT", "step_id": step.id, "ts": ts,
                "evidence": {"action": "auto_skip_optional_step"},
            })
            events.append({
                "type": "alert", "alert_type": "SKIPPED", "step_id": step.id, "ts": ts,
                "evidence": {"reason": "timeout_on_optional_step"},
            })
            events.extend(self._advance_skip(step, ts))
            return events

        retries = self._timeout_retries.get(step.id, 0) + 1
        self._timeout_retries[step.id] = retries

        if retries > self.config.thresholds.max_timeout_retries:
            if step.recovery_step:
                events.append({
                    "type": "alert", "alert_type": "TIMEOUT", "step_id": step.id, "ts": ts,
                    "severity": "ERROR",
                    "evidence": {"action": "recovery_redirect", "recovery_step": step.recovery_step},
                })
                events.append({
                    "type": "recovery_redirect", "step_id": step.id,
                    "recovery_step": step.recovery_step, "ts": ts,
                })
                self.state.current_step_id = step.recovery_step
                self.state.step_started_at = ts
                self.evidence.reset(dev_key)
                self.evidence.reset(step.id)
            else:
                events.append({
                    "type": "alert", "alert_type": "FAILED", "step_id": step.id, "ts": ts,
                    "severity": "ERROR",
                    "evidence": {"reason": "timeout_retries_exhausted", "retries": retries},
                })
                self.state.status = "FAILED"
        else:
            events.append({
                "type": "alert", "alert_type": "TIMEOUT", "step_id": step.id, "ts": ts,
                "evidence": {"retry": retries, "max_retries": self.config.thresholds.max_timeout_retries},
            })
            # Give the step a fresh window to retry rather than firing
            # the same TIMEOUT alert every frame from here on.
            self.evidence.reset(dev_key)
            self.state.step_started_at = ts

        return events

    # ------------------------------------------------------------------
    def next_action(self) -> NextActionAdvice:
        if self.state.status == "COMPLETED":
            return NextActionAdvice(None, "Experiment complete.", None, None, "Experiment complete.")
        if self.state.status == "FAILED":
            return NextActionAdvice(
                None, "Experiment failed.", None, None,
                "The experiment could not continue. Please check the alerts.",
            )
        step = self.config.step_by_id(self.state.current_step_id)
        if step is None:
            return NextActionAdvice(None, "Unknown state.", None, None, None)
        return NextActionAdvice(
            next_step=step.id,
            reason=f"Awaiting '{step.activity}' on '{step.required_object}'",
            required_object=step.required_object,
            target=step.target_object,
            voice_instruction=step.voice_instruction,
        )

    def _snapshot(self, events: list[dict]) -> dict:
        return {
            "current_step_id": self.state.current_step_id,
            "completed_steps": list(self.state.completed_steps),
            "skipped_steps": list(self.state.skipped_steps),
            "status": self.state.status,
            "object_states": self.objects.snapshot(),
            "events": events,
            "event": events[-1] if events else None,   # kept for dashboard backward-compat
            "next_action": self.next_action().__dict__,
        }
