@echo off
REM ASTRA standalone startup (Windows).
REM Starts backend + dashboard together — the two-process "offline
REM standalone system" the official problem statement calls for: nothing
REM here talks to the internet or ground control, both processes run
REM entirely on this machine.
REM
REM Usage: double-click, or run from a terminal in the astra\ root folder.

echo ============================================
echo   ASTRA — starting standalone system
echo ============================================

if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: .venv not found. Run this first:
    echo   py -m venv .venv
    echo   .venv\Scripts\activate
    echo   pip install -r backend\requirements.txt
    pause
    exit /b 1
)

if not exist "models\pose_landmarker_lite.task" (
    echo WARNING: pose model not found — pose estimation will stay inactive.
    echo Run: python scripts\download_models.py
)

if "%ASTRA_API_KEY%"=="" (
    echo NOTE: ASTRA_API_KEY is not set — API auth is DISABLED for this run.
    echo Set it before deploying on any shared network:  set ASTRA_API_KEY=your-secret-here
)

set TLS_ARGS=
if exist "certs\cert.pem" if exist "certs\key.pem" (
    echo NOTE: certs\cert.pem + certs\key.pem found — serving over HTTPS.
    set TLS_ARGS=--ssl-keyfile certs\key.pem --ssl-certfile certs\cert.pem
) else (
    echo NOTE: no certs\cert.pem + certs\key.pem — serving over plain HTTP.
    echo For HTTPS: python scripts\generate_self_signed_cert.py
)

echo.
echo Starting backend on http://localhost:8000 ...
start "ASTRA Backend" cmd /k ".venv\Scripts\activate.bat && uvicorn backend.main:app --port 8000 %TLS_ARGS%"

timeout /t 3 /nobreak > nul

echo Starting dashboard on http://localhost:3000 ...
start "ASTRA Dashboard" cmd /k "cd dashboard && npm run dev"

echo.
echo Both processes started in separate windows.
echo Open http://localhost:3000 in a browser once the dashboard finishes starting.
echo Close both windows to stop ASTRA.
pause
