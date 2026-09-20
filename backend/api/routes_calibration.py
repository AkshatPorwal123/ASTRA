from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.run_manager import registry
from backend.vision.calibration import sample_hsv_range, save_calibrated_range, load_calibrated_ranges

router = APIRouter(prefix="/api/calibration", tags=["calibration"])


class CalibrateRequest(BaseModel):
    color_name: str
    # Fractions of the frame's width/height (0.0-1.0), not raw pixels —
    # the dashboard doesn't need to know the camera's actual resolution
    # this way, only where the operator clicked on the displayed image.
    fx: float = Field(ge=0.0, le=1.0)
    fy: float = Field(ge=0.0, le=1.0)


@router.post("/{run_id}/sample")
def calibrate_color(run_id: str, req: CalibrateRequest):
    active = registry.get(run_id)
    if not active:
        raise HTTPException(404, "Run not active")
    if active.latest_raw_frame is None:
        raise HTTPException(400, "No camera frame available yet for this run — wait for the feed to start.")

    h, w = active.latest_raw_frame.shape[:2]
    x, y = int(req.fx * w), int(req.fy * h)

    try:
        lo, hi, stats = sample_hsv_range(active.latest_raw_frame, x, y)
    except ValueError as e:
        raise HTTPException(400, str(e))

    save_calibrated_range(req.color_name, lo, hi)
    return {
        "color_name": req.color_name, "sampled_at_px": [x, y],
        "hsv_stats": stats,
        "note": "Saved. Takes effect on the NEXT run start, not this one — restart to apply.",
    }


@router.get("")
def get_calibrated_ranges():
    return load_calibrated_ranges()
