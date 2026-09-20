import os
import psutil
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.db import models
from backend.run_manager import registry
from backend.streaming.mjpeg import mjpeg_multipart_frame

router = APIRouter(tags=["misc"])


@router.get("/api/procedure/{run_id}")
def get_procedure_state(run_id: str, db: Session = Depends(get_db)):
    active = registry.get(run_id)
    if active:
        return active.engine._snapshot(events=[])
    # Fallback: reconstruct last known state from DB for stopped runs
    last = (
        db.query(models.ProcedureState)
        .filter_by(run_id=run_id)
        .order_by(models.ProcedureState.ts.desc())
        .first()
    )
    if not last:
        raise HTTPException(404, "No procedure state found for this run")
    return last.snapshot_json


@router.get("/api/alerts/{run_id}")
def get_alerts(run_id: str, db: Session = Depends(get_db)):
    alerts = db.query(models.Alert).filter_by(run_id=run_id).order_by(models.Alert.ts).all()
    return [{
        "id": a.id, "ts": a.ts, "type": a.type, "step_id": a.step_id,
        "severity": a.severity, "status": a.status, "evidence": a.evidence_json,
    } for a in alerts]


@router.get("/api/logs/{run_id}")
def get_logs(run_id: str, format: str = "json"):
    path = os.path.join("recordings", f"{run_id}.events.jsonl")
    if not os.path.exists(path):
        raise HTTPException(404, "No logs found for this run")
    if format == "raw":
        return FileResponse(path, media_type="application/jsonl")
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    return {"run_id": run_id, "event_count": len(lines), "events_file": path}


@router.get("/api/system")
def system_status():
    return {
        "cpu_percent": psutil.cpu_percent(),
        "ram_percent": psutil.virtual_memory().percent,
        "active_runs": [],  # populated below
    }


@router.get("/api/inference/status")
def inference_status(run_id: str | None = None):
    if run_id:
        active = registry.get(run_id)
        if not active:
            raise HTTPException(404, "Run not active")
        return {"run_id": run_id, "models": active.pipeline.metadata()}
    return {"note": "Pass ?run_id= to get per-model status for an active run."}


@router.get("/api/video/{run_id}/stream")
def video_stream(run_id: str):
    active = registry.get(run_id)
    if not active:
        raise HTTPException(404, "Run not active — no live stream available")

    def generate():
        import time
        while True:
            if active.latest_jpeg:
                yield mjpeg_multipart_frame(active.latest_jpeg)
            time.sleep(0.1)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@router.get("/api/video/{run_id}/file")
def video_file(run_id: str):
    path = os.path.join("recordings", f"{run_id}.mp4")
    if not os.path.exists(path):
        raise HTTPException(404, "Recorded video not found")
    return FileResponse(path, media_type="video/mp4")
