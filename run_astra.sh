#!/usr/bin/env bash
# ASTRA standalone startup (Linux/Mac).
# Starts backend + dashboard together. Both run entirely on this machine —
# no internet or ground-control dependency, matching the official
# problem's "standalone operation" requirement.
#
# Usage: ./run_astra.sh   (from the astra/ root folder)

set -e
echo "============================================"
echo "  ASTRA — starting standalone system"
echo "============================================"

if [ ! -f ".venv/bin/activate" ]; then
    echo "ERROR: .venv not found. Run this first:"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -r backend/requirements.txt"
    exit 1
fi

if [ ! -f "models/pose_landmarker_lite.task" ]; then
    echo "WARNING: pose model not found — pose estimation will stay inactive."
    echo "Run: python scripts/download_models.py"
fi

if [ -z "$ASTRA_API_KEY" ]; then
    echo "NOTE: ASTRA_API_KEY is not set — API auth is DISABLED for this run."
    echo "Set it before deploying on any shared network:  export ASTRA_API_KEY=your-secret-here"
fi

TLS_ARGS=""
if [ -f "certs/cert.pem" ] && [ -f "certs/key.pem" ]; then
    echo "NOTE: certs/cert.pem + certs/key.pem found — serving over HTTPS."
    TLS_ARGS="--ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem"
else
    echo "NOTE: no certs/cert.pem + certs/key.pem — serving over plain HTTP."
    echo "For HTTPS: python scripts/generate_self_signed_cert.py"
fi

echo ""
echo "Starting backend on http://localhost:8000 ..."
source .venv/bin/activate
uvicorn backend.main:app --port 8000 $TLS_ARGS &
BACKEND_PID=$!

sleep 2

echo "Starting dashboard on http://localhost:3000 ..."
(cd dashboard && npm run dev) &
DASHBOARD_PID=$!

echo ""
echo "Both processes started (backend PID $BACKEND_PID, dashboard PID $DASHBOARD_PID)."
echo "Open http://localhost:3000 once the dashboard finishes starting."
echo "Press Ctrl+C to stop both."

trap "kill $BACKEND_PID $DASHBOARD_PID 2>/dev/null" EXIT
wait
