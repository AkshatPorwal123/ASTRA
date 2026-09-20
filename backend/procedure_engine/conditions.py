"""
Minimal condition-expression evaluator for the experiment schema's
`preconditions` / `postconditions` string lists (Section 14).

Deliberately a tiny, fixed grammar rather than a general expression
language — the procedure engine must stay deterministic and explainable
(Section 15/31), and a full expression evaluator (arbitrary Python `eval`)
would be both a security risk and harder to audit than a config author
needs for "object X's attribute Y must equal Z".

Supported grammar (one comparison per string):
    "<object_id>.<attribute> == <VALUE>"
    "<object_id>.<attribute> != <VALUE>"

`<attribute>` is usually `state` (AVAILABLE/HELD/OPEN/... per Section 9)
but can be any identifier a config author defines (e.g. `mixed`,
`location`) — the tracker doesn't enforce a fixed attribute set.

Preconditions are CHECKED against the current ObjectStateTracker.
Postconditions are APPLIED (the `==` form assigns that value once the
step is confirmed) — see engine.py's `_advance`.
"""
from __future__ import annotations
import re

from .object_state import ObjectStateTracker

_COND_RE = re.compile(
    r"^\s*([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)\s*(==|!=)\s*([A-Za-z0-9_]+)\s*$"
)


class ConditionError(ValueError):
    pass


def parse_condition(expr: str) -> tuple[str, str, str, str]:
    """Returns (object_id, attribute, operator, expected_value). Raises ConditionError on bad syntax."""
    m = _COND_RE.match(expr)
    if not m:
        raise ConditionError(
            f"Unsupported condition syntax: {expr!r}. "
            f"Expected '<object_id>.<attribute> == <VALUE>' or '... != ...'."
        )
    return m.group(1), m.group(2), m.group(3), m.group(4)


def evaluate_condition(expr: str, tracker: ObjectStateTracker) -> bool:
    obj_id, attribute, op, expected = parse_condition(expr)
    actual = tracker.get(obj_id, attribute)
    return (actual == expected) if op == "==" else (actual != expected)


def evaluate_all(exprs: list[str], tracker: ObjectStateTracker) -> tuple[bool, list[str]]:
    """Checks every condition. Returns (all_satisfied, list of failing expressions)."""
    failing = [e for e in exprs if not evaluate_condition(e, tracker)]
    return (len(failing) == 0, failing)


def apply_postconditions(exprs: list[str], tracker: ObjectStateTracker, ts: float | None = None):
    """
    Applies postconditions as attribute assignments once a step is
    confirmed. Only the '==' form makes sense as an assignment; a '!='
    postcondition is a malformed config (can't assign "not equal to X")
    and raises ConditionError so the config author sees it immediately
    rather than the run silently doing nothing.
    """
    for expr in exprs:
        obj_id, attribute, op, expected = parse_condition(expr)
        if op != "==":
            raise ConditionError(
                f"Postcondition {expr!r} uses '!=' — postconditions must assign a "
                f"value with '==' (they describe what becomes true, not what stays false)."
            )
        tracker.set(obj_id, expected, attribute=attribute, ts=ts, source="postcondition")
