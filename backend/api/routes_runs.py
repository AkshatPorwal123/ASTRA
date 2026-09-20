import os
import uuid
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.db import models
from backend.run_manager import ActiveRun, registry

router = APIRouter(prefix="/api/runs", tags=["runs"])
CONFIG_DIR = os.environ.get("ASTRA_CONFIG_DIR", "configs/experiments")


class StartRunRequest(BaseModel):
    experiment_name: str
    source: str | int = 0          # 0 = default webcam, or a file path
    participant_id: str | None = None
    record: bool = True
    stream_to: str | None = None   # "ip:port" — pushes the video feed there (official "stream to specific IP" requirement)
    use_rack_reference: bool = False   # ArUco marker-based orientation-agnostic tracking (Milestone 13) — needs a printed marker, see scripts/generate_rack_marker.py
    rack_marker_id: int = 0


@router.post("")
async def start_run(req: StartRunRequest, db: Session = Depends(get_db)):
    config_path = os.path.join(CONFIG_DIR, f"{req.experiment_name}.yaml")
    if not os.path.exists(config_path):
        raise HTTPException(404, f"Unknown experiment: {req.experiment_name}")

    experiment = db.query(models.Experiment).filter_by(name=req.experiment_name).first()
    if experiment is None:
        experiment = models.Experiment(name=req.experiment_name, config_path=config_path)
        db.add(experiment)
        db.commit()
        db.refresh(experiment)

    run_id = str(uuid.uuid4())
    db_run = models.ExperimentRun(
        id=run_id, experiment_id=experiment.id, participant_id=req.participant_id,
        status="RUNNING", source_type="webcam" if req.source == 0 else "upload",
    )
    db.add(db_run)
    db.commit()

    try:
        active = ActiveRun(
            run_id=run_id, experiment_config_path=config_path, source=req.source,
            record=req.record, stream_to=req.stream_to,
            use_rack_reference=req.use_rack_reference, rack_marker_id=req.rack_marker_id,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    registry.add(active)
    await active.start()

    return {"run_id": run_id, "status": "RUNNING", "experiment": req.experiment_name, "streaming_to": req.stream_to}


@router.post("/{run_id}/stop")
async def stop_run(run_id: str):
    active = registry.get(run_id)
    if active is None:
        raise HTTPException(404, "Run not found or already stopped")
    await active.stop()
    registry.remove(run_id)
    return {"run_id": run_id, "status": "STOPPED"}


@router.get("")
def list_runs(db: Session = Depends(get_db)):
    runs = db.query(models.ExperimentRun).order_by(models.ExperimentRun.started_at.desc()).all()
    return [{
        "run_id": r.id, "experiment_id": r.experiment_id, "status": r.status,
        "started_at": r.started_at, "ended_at": r.ended_at,
    } for r in runs]


@router.get("/{run_id}")
def get_run(run_id: str, db: Session = Depends(get_db)):
    r = db.query(models.ExperimentRun).get(run_id)
    if not r:
        raise HTTPException(404, "Run not found")
    return {
        "run_id": r.id, "experiment_id": r.experiment_id, "status": r.status,
        "started_at": r.started_at, "ended_at": r.ended_at, "video_file_ref": r.video_file_ref,
    }
