"""
Standalone smoke test for the Milestone 6/7 procedure engine logic.
Stubs out backend.config.schema with plain dataclasses (same field shape
as the real pydantic model) so this can run without pydantic installed in
this sandbox -- the real backend still uses the pydantic schema.py as-is.
"""
import sys, os, types, dataclasses, time
from typing import Optional

# ---- stub backend.config.schema -------------------------------------
schema_mod = types.ModuleType("backend.config.schema")

@dataclasses.dataclass
class ObjectDef:
    id: str
    class_name: str = ""
    states: list = dataclasses.field(default_factory=list)

@dataclasses.dataclass
class StepDef:
    id: str
    activity: str
    name: Optional[str] = None
    required_object: Optional[str] = None
    target_object: Optional[str] = None
    preconditions: list = dataclasses.field(default_factory=list)
    postconditions: list = dataclasses.field(default_factory=list)
    confidence_threshold: Optional[float] = None
    timeout_s: Optional[float] = None
    next: Optional[str] = None
    on_success: Optional[str] = None
    branches: dict = dataclasses.field(default_factory=dict)
    voice_instruction: Optional[str] = None
    optional: bool = False
    repeatable: bool = False
    recovery_step: Optional[str] = None

@dataclasses.dataclass
class Thresholds:
    default_confidence: float = 0.6
    default_timeout_s: float = 60.0
    tentative_frames: int = 2
    confirmed_frames: int = 3
    max_timeout_retries: int = 1

@dataclasses.dataclass
class ExperimentConfig:
    experiment: str
    steps: list
    objects: list = dataclasses.field(default_factory=list)
    thresholds: Thresholds = dataclasses.field(default_factory=Thresholds)

    def step_by_id(self, step_id):
        return next((s for s in self.steps if s.id == step_id), None)

    def first_step(self):
        return self.steps[0]

schema_mod.ObjectDef = ObjectDef
schema_mod.StepDef = StepDef
schema_mod.Thresholds = Thresholds
schema_mod.ExperimentConfig = ExperimentConfig
sys.modules["backend.config.schema"] = schema_mod
sys.modules.setdefault("backend", types.ModuleType("backend"))
sys.modules.setdefault("backend.config", types.ModuleType("backend.config"))

# Resolve relative to this script's own location, not a hardcoded sandbox
# path — the previous version only ran on the machine it was written on.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.modules["backend"].__path__ = [os.path.join(_PROJECT_ROOT, "backend")]
sys.modules["backend.config"].__path__ = [os.path.join(_PROJECT_ROOT, "backend", "config")]

sys.path.insert(0, _PROJECT_ROOT)
from backend.procedure_engine.engine import ProcedureEngine  # noqa: E402

PASS, FAIL = 0, 0
def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  OK  {label}")
    else:
        FAIL += 1
        print(f" FAIL {label}")


def make_config():
    steps = [
        StepDef(id="S1_PICK", activity="PICK", required_object="red_box",
                postconditions=["red_box.state == HELD"], next="S2_HOLD", timeout_s=5),
        StepDef(id="S2_HOLD", activity="HOLD", required_object="red_box",
                preconditions=["red_box.state == HELD"], next="S3_MOVE", timeout_s=5),
        StepDef(id="S3_MOVE", activity="MOVE", required_object="red_box", next="S4_PLACE", timeout_s=5),
        StepDef(id="S4_PLACE", activity="PLACE", required_object="red_box",
                postconditions=["red_box.state == PLACED"], next=None, timeout_s=5),
    ]
    objs = [ObjectDef(id="red_box", class_name="red_box", states=["AVAILABLE"])]
    return ExperimentConfig(experiment="test", steps=steps, objects=objs, thresholds=Thresholds())


def confirm(engine, label, object_class=None, n=4, ts=None, conf=0.9):
    """Push `n` matching frames so evidence crosses CONFIRMED (confirmed_frames=3)."""
    events = []
    for i in range(n):
        snap = engine.observe_activity(label, conf, ts=(ts or time.time()) + i * 0.1, object_class=object_class)
        events.extend(snap["events"])
    return events


print("== Test 1: happy path (CORRECT sequence + postconditions) ==")
cfg = make_config()
eng = ProcedureEngine(cfg)
check("initial object state AVAILABLE", eng.objects.get("red_box") == "AVAILABLE")
evs = confirm(eng, "PICK", object_class="red_box")
check("S1 completed", any(e["type"] == "step_completed" and e["step_id"] == "S1_PICK" for e in evs))
check("postcondition applied -> HELD", eng.objects.get("red_box") == "HELD")
evs = confirm(eng, "HOLD", object_class="red_box")
check("S2 completed (precondition satisfied)", any(e.get("step_id") == "S2_HOLD" for e in evs))
evs = confirm(eng, "MOVE", object_class="red_box")
evs = confirm(eng, "PLACE", object_class="red_box")
check("experiment COMPLETED", eng.state.status == "COMPLETED")
check("final state PLACED", eng.objects.get("red_box") == "PLACED")

print("\n== Test 2: WRONG_OBJECT ==")
cfg = make_config()
eng = ProcedureEngine(cfg)
evs = confirm(eng, "PICK", object_class="blue_box")
check("WRONG_OBJECT alert fired", any(e.get("alert_type") == "WRONG_OBJECT" for e in evs))
check("step NOT completed on wrong object", "S1_PICK" not in eng.state.completed_steps)

print("\n== Test 3: precondition failure -> FAILED ==")
cfg = make_config()
eng = ProcedureEngine(cfg)
eng.state.current_step_id = "S2_HOLD"          # jump straight to HOLD without red_box HELD yet
eng.state.step_started_at = time.time()
evs = confirm(eng, "HOLD", object_class="red_box")
check("FAILED alert on unmet precondition", any(e.get("alert_type") == "FAILED" for e in evs))
check("step NOT completed", "S2_HOLD" not in eng.state.completed_steps)

print("\n== Test 4: SKIPPED + WRONG_ORDER ==")
cfg = make_config()
eng = ProcedureEngine(cfg)
confirm(eng, "PICK", object_class="red_box")     # completes S1 -> HELD
evs = confirm(eng, "PLACE", object_class="red_box")  # jump ahead, skipping HOLD/MOVE
check("SKIPPED alert for S2_HOLD", any(e.get("alert_type") == "SKIPPED" and e.get("step_id") == "S2_HOLD" for e in evs))
check("SKIPPED alert for S3_MOVE", any(e.get("alert_type") == "SKIPPED" and e.get("step_id") == "S3_MOVE" for e in evs))
check("WRONG_ORDER alert for S4_PLACE", any(e.get("alert_type") == "WRONG_ORDER" and e.get("step_id") == "S4_PLACE" for e in evs))
check("engine stayed blocked on S2_HOLD (didn't silently advance)", eng.state.current_step_id == "S2_HOLD")

print("\n== Test 5: REPEATED ==")
cfg = make_config()
eng = ProcedureEngine(cfg)
confirm(eng, "PICK", object_class="red_box")   # S1 complete
confirm(eng, "HOLD", object_class="red_box")   # S2 complete
evs = confirm(eng, "PICK", object_class="red_box")  # repeat S1's activity while on S3
check("REPEATED alert fired", any(e.get("alert_type") == "REPEATED" and e.get("step_id") == "S1_PICK" for e in evs))

def force_timeout_cycle(eng, gap=100):
    """Forces step_started_at far enough in the past, then pushes enough
    IDLE frames (confirmed_frames=3) for the timeout evidence to CONFIRM."""
    eng.state.step_started_at = time.time() - gap
    evs = []
    for i in range(3):
        evs.extend(eng.observe_activity("IDLE", 0.5, ts=time.time() + i * 0.1)["events"])
    return evs

print("\n== Test 6: TIMEOUT -> retry -> FAILED (no recovery_step) ==")
cfg = make_config()
eng = ProcedureEngine(cfg)
evs = force_timeout_cycle(eng)
check("first TIMEOUT alert (retry)", any(e.get("alert_type") == "TIMEOUT" for e in evs))
check("run still IN_PROGRESS after first (allowed) retry", eng.state.status == "IN_PROGRESS")
evs3 = force_timeout_cycle(eng)
check("second TIMEOUT escalates to FAILED (no recovery_step configured)",
      any(e.get("alert_type") == "FAILED" for e in evs3))
check("run status FAILED", eng.state.status == "FAILED")

print("\n== Test 7: TIMEOUT with recovery_step redirects instead of failing ==")
cfg = make_config()
cfg.steps[0].recovery_step = "S4_PLACE"
cfg.thresholds.max_timeout_retries = 0
eng = ProcedureEngine(cfg)
evs = force_timeout_cycle(eng)
check("recovery_redirect event fired", any(e.get("type") == "recovery_redirect" for e in evs))
check("engine jumped to recovery_step", eng.state.current_step_id == "S4_PLACE")

print("\n== Test 8: optional step auto-skips on timeout ==")
cfg = make_config()
cfg.steps[1].optional = True   # S2_HOLD becomes optional
eng = ProcedureEngine(cfg)
confirm(eng, "PICK", object_class="red_box")
evs = force_timeout_cycle(eng)
check("SKIPPED alert for optional timeout", any(e.get("alert_type") == "SKIPPED" for e in evs))
check("auto-advanced past optional step", eng.state.current_step_id == "S3_MOVE")
check("S2_HOLD recorded in skipped_steps", "S2_HOLD" in eng.state.skipped_steps)

print("\n== Test 9: RECOVERED lifecycle ==")
cfg = make_config()
eng = ProcedureEngine(cfg)
evs = confirm(eng, "MOVE", object_class="red_box", n=3)  # wrong activity while on S1 -> UNCERTAIN/deviation
check("deviation alert raised while stuck on S1", any(e.get("type") == "alert" for e in evs))
evs = confirm(eng, "PICK", object_class="red_box")        # now do it right
check("RECOVERED event emitted once corrected", any(e.get("type") == "alert_recovered" for e in evs))

print(f"\n{'='*50}\n{PASS} passed, {FAIL} failed\n{'='*50}")
sys.exit(1 if FAIL else 0)
