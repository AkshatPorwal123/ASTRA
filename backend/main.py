"""
ASTRA backend entrypoint.
Run with:  uvicorn backend.main:app --reload --port 8000
(run from the astra/ project root so `backend` and `configs` resolve)
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.db.database import init_db
from backend.api import routes_experiments, routes_runs, routes_misc, routes_reports, routes_dataset, routes_calibration
from backend.ws import routes_ws
from backend.auth import ApiKeyMiddleware, is_auth_enabled, log_auth_status

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="ASTRA Backend",
    description="Autonomous Space Task Recognition & Assistance System",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(ApiKeyMiddleware)

app.include_router(routes_experiments.router)
app.include_router(routes_runs.router)
app.include_router(routes_misc.router)
app.include_router(routes_reports.router)
app.include_router(routes_dataset.router)
app.include_router(routes_calibration.router)
app.include_router(routes_ws.router)


@app.on_event("startup")
def on_startup():
    init_db()
    log_auth_status()


@app.get("/")
def root():
    return {
        "system": "ASTRA",
        "status": "All milestones (1-9) active — see README.md for the full "
                  "real-vs-placeholder breakdown.",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def get_config():
    """Unauthenticated by design — the frontend needs this before it has
    a key to send, and it reveals nothing sensitive (not the key itself,
    just whether one is required)."""
    return {"auth_required": is_auth_enabled()}
