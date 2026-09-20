import os
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.db import models
from backend.reporting.generator import generate_report, REPORTS_DIR
from backend.procedure_engine.graph_loader import load_graph

router = APIRouter(prefix="/api/reports", tags=["reports"])


@router.get("/{run_id}")
def get_report(run_id: str, db: Session = Depends(get_db)):
    """Returns the most recent report for a run, generating one on the fly
    if none exists yet (e.g. the run was stopped before this feature existed,
    or a report is requested mid-run for a partial summary)."""
    existing = (
        db.query(models.Report)
        .filter_by(run_id=run_id)
        .order_by(models.Report.generated_at.desc())
        .first()
    )
    if existing:
        return existing.summary_json

    run = db.query(models.ExperimentRun).get(run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    experiment = db.query(models.Experiment).get(run.experiment_id)
    config = load_graph(experiment.config_path)
    return generate_report(db, run_id, config)


@router.get("/{run_id}/text")
def get_report_text(run_id: str):
    path = os.path.join(REPORTS_DIR, f"{run_id}.report.txt")
    if not os.path.exists(path):
        raise HTTPException(404, "No text report found — try GET /api/reports/{run_id} first to generate one")
    return FileResponse(path, media_type="text/plain", filename=f"{run_id}.report.txt")
