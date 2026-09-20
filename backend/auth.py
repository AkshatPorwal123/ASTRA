"""
Lightweight API key auth (Section 26 — flagged as pending since Milestone
1's design doc, closed here in Milestone 9).

Deliberately simple: a single shared key via ASTRA_API_KEY, checked
against an `X-API-Key` header. This is a standalone lab-bench / space-
station-payload system operated by one team on an isolated local network,
not a multi-tenant public service — per-user accounts and OAuth would be
solving a problem this deployment doesn't have. A shared operator key
that actually gates access beats an elaborate auth system nobody sets up.

If ASTRA_API_KEY is unset, auth is OFF and a loud startup warning is
logged once — silent insecure-by-default would be the actual footgun;
an explicit "you are running unauthenticated" notice is not.

WebSocket connections can't set custom headers from a browser, so the
key is also accepted as a `?api_key=` query parameter for the /ws/ route
specifically (see backend/ws/routes_ws.py) — everywhere else uses the
header.
"""
from __future__ import annotations
import os
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger("astra.auth")

API_KEY = os.environ.get("ASTRA_API_KEY", "").strip()

# Paths that stay open even when auth is enabled — health checks and the
# one endpoint the frontend needs to hit before it has a key to send.
OPEN_PATHS = {"/health", "/", "/api/config", "/docs", "/openapi.json", "/redoc", "/docs/oauth2-redirect"}


def is_auth_enabled() -> bool:
    return bool(API_KEY)


class ApiKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not API_KEY:
            return await call_next(request)

        path = request.url.path
        if path in OPEN_PATHS or path.startswith("/ws/") or not path.startswith("/api/"):
            # WS auth is handled inside routes_ws.py itself (query param),
            # not here — Starlette's WebSocket lifecycle doesn't go through
            # this HTTP middleware the same way.
            return await call_next(request)

        provided = request.headers.get("x-api-key", "") or request.query_params.get("api_key", "")
        if provided != API_KEY:
            return JSONResponse({"detail": "Invalid or missing X-API-Key header (or ?api_key= for endpoints an <img>/<video> tag loads, like the MJPEG stream, since those can't set custom headers)"}, status_code=401)

        return await call_next(request)


def log_auth_status():
    if API_KEY:
        logger.info("[AUTH] API key required for all /api/* routes.")
    else:
        logger.warning(
            "[AUTH] ASTRA_API_KEY is not set — the API is running WITHOUT authentication. "
            "Anyone on this network can start/stop runs and read data. Set ASTRA_API_KEY "
            "before deploying on a shared network."
        )
