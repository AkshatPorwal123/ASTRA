import glob
import os
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.db import models
from backend.config.loader import load_experiment_config, ConfigLoadError

router = APIRouter(prefix="/api/experiments", tags=["experiments"])

CONFIG_DIR = os.environ.get("ASTRA_CONFIG_DIR", "configs/experiments")


@router.get("")
def list_experiments(db: Session = Depends(get_db)):
    """Lists both DB-registered experiments and any config files on disk
    not yet registered, so new YAML files are discoverable immediately."""
    registered = {e.name: e for e in db.query(models.Experiment).all()}
    discovered = []
    for path in glob.glob(os.path.join(CONFIG_DIR, "*.yaml")):
        try:
            cfg = load_experiment_config(path)
        except ConfigLoadError as e:
            discovered.append({"file": path, "error": str(e)})
            continue
        discovered.append({
            "name": cfg.experiment,
            "version": cfg.version,
            "num_steps": len(cfg.steps),
            "config_path": path,
            "registered": cfg.experiment in registered,
        })
    return {"experiments": discovered}


@router.get("/{name}")
def get_experiment(name: str):
    path = os.path.join(CONFIG_DIR, f"{name}.yaml")
    try:
        cfg = load_experiment_config(path)
    except ConfigLoadError as e:
        raise HTTPException(404, str(e))
    return cfg.model_dump()
