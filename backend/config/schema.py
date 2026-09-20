"""
Pydantic schema for data-driven experiment configuration.
No experiment step logic is ever hardcoded in Python — everything here
is loaded from YAML/JSON at run start (see /configs/experiments/*.yaml).
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class ObjectDef(BaseModel):
    id: str
    class_name: str = Field(alias="class")
    states: list[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class StepDef(BaseModel):
    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    activity: str                      # expected activity/action label
    required_object: Optional[str] = None
    target_object: Optional[str] = None
    preconditions: list[str] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)
    expected_duration_s: Optional[list[float]] = None   # [min, max]
    confidence_threshold: Optional[float] = None
    timeout_s: Optional[float] = None
    next: Optional[str] = None             # default success transition
    on_success: Optional[str] = None       # alias, supports branching later
    branches: dict[str, str] = Field(default_factory=dict)  # condition -> step_id
    voice_instruction: Optional[str] = None
    error_conditions: dict[str, str] = Field(default_factory=dict)
    optional: bool = False
    # Milestone 6 additions (Section 15/16 — recovery states, repeated steps)
    repeatable: bool = False           # if True, re-observing this step's activity is CORRECT, not REPEATED
    recovery_step: Optional[str] = None  # step to jump to after repeated timeout/failure on this step


class Thresholds(BaseModel):
    default_confidence: float = 0.6
    default_timeout_s: float = 60.0
    tentative_frames: int = 5          # frames of evidence before TENTATIVE
    confirmed_frames: int = 12         # frames of evidence before CONFIRMED
    # Milestone 7 addition: how many confirmed timeouts on the same step
    # before escalating to recovery_step (if set) or FAILED (if not)
    max_timeout_retries: int = 1


class ExperimentConfig(BaseModel):
    experiment: str
    version: str = "1.0"
    objects: list[ObjectDef] = Field(default_factory=list)
    steps: list[StepDef]
    thresholds: Thresholds = Field(default_factory=Thresholds)

    def step_by_id(self, step_id: str) -> Optional[StepDef]:
        return next((s for s in self.steps if s.id == step_id), None)

    def first_step(self) -> StepDef:
        return self.steps[0]
