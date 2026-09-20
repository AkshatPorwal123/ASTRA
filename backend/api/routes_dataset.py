from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.run_manager import registry

router = APIRouter(prefix="/api/dataset", tags=["dataset"])


class LabelRequest(BaseModel):
    label: str


@router.get("/{run_id}/suggestion")
def get_suggestion(run_id: str):
    """The rule-based recognizer's current best guess, for the labeling UI
    to pre-fill (person just confirms or corrects it — faster than typing
    every label from scratch)."""
    active = registry.get(run_id)
    if not active:
        raise HTTPException(404, "Run not active")
    return {"suggested_label": active.dataset_recorder.current_suggestion()}


class ObjectBox(BaseModel):
    class_: str = Field(alias="class")
    bbox: list[float]


class CaptureObjectSampleRequest(BaseModel):
    boxes: list[ObjectBox] | None = None   # omit to accept the detector's current suggestion as-is


@router.post("/{run_id}/capture_object_sample")
def capture_object_sample(run_id: str, req: CaptureObjectSampleRequest):
    """Saves the current frame + box annotations for object-detector
    training — the object-detection half of the official 'dataset
    generation' requirement (Milestone 8's dataset panel only covered
    activity labels)."""
    active = registry.get(run_id)
    if not active:
        raise HTTPException(404, "Run not active")
    confirmed = [b.model_dump(by_alias=True) for b in req.boxes] if req.boxes is not None else None
    try:
        sample = active.object_dataset_recorder.capture_sample(confirmed_boxes=confirmed)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"captured": True, "sample": sample}


@router.post("/{run_id}/label")
def label_current_frame(run_id: str, req: LabelRequest):
    """Records a ground-truth label for whatever the run's most recent
    frame was — the actual 'dataset generation' deliverable: real labeled
    examples from a real run, not synthetic data."""
    active = registry.get(run_id)
    if not active:
        raise HTTPException(404, "Run not active")
    try:
        sample = active.dataset_recorder.record_label(req.label)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"recorded": True, "sample": sample}
