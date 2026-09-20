# ASTRA : Autonomous Space Task Recognition & Assistance System

## Overview

ASTRA (Autonomous Space Task Recognition & Assistance System) is an intelligent, edge-capable monitoring and assistance architecture designed for complex operational environments. **Phase 1** establishes the core foundation: a robust backend infrastructure, a deterministic procedure-reasoning engine, and a real-time reactive dashboard. 

Per the architectural design mandates, AI perception models (person, pose, object, interaction, and activity recognition) are implemented as clearly labeled, modular placeholders to maintain strict behavioral transparency during evaluation.

---

## Architecture & Experiment Configs

The system is driven by YAML experiment configurations that define step sequences, vocabularies, preconditions, postconditions, and error-handling policies:

*   **`red_box_demo.yaml` (Dashboard Default):** Utilizes the color detector to track the primary test object (red box). Features a `PICK/HOLD/MOVE/PLACE` vocabulary. Exercises advanced procedure logic including preconditions, postconditions, optional steps, repeatable steps, and `recovery_step` definitions.
*   **`pick_place_demo.yaml`:** Utilizes the same vocabulary and advanced feature set (Milestone 6/7) configured for bottle-shaped objects via MobileNet-SSD.
*   **`liquid_mixing_v1.yaml`:** Legacy reference configuration. Features strict multi-attribute preconditions and postconditions (e.g., `container.mixed == true`). *Note: Designed to test engine enforcement, this config expects an `OPEN/POUR/MIX/CLOSE` vocabulary. Because current rule-based recognizers cannot distinguish these actions without a trained Human Activity Recognition (HAR) model (Milestone 11+), running this with real perception active will intentionally pause mid-experiment as per architectural safety rules.*
*   **Official Mission Config:** Reserved for the full problem-statement step sequence upon release.

---

## System Requirements

*   **Python:** `3.10` or higher
*   **Node.js:** `18` or higher
*   **Perception Input:** Standard webcam (`index 0`) or an accessible video file path.

---

## Quick Start Guide

### 1. Backend Setup

Navigate to the project root and initialize the Python virtual environment:

```bash
cd astra
python3 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt

```

> **Important:** Run all commands from the `astra/` project root (not inside the `backend/` directory) to ensure proper package resolution for `backend` and `configs/`.

Launch the FastAPI backend server:

```bash
uvicorn backend.main:app --reload --port 8000

```

* **API Documentation:** Access interactive Swagger documentation at [http://localhost:8000/docs](http://localhost:8000/docs?utm_source=gemini) for manual payload testing and endpoint invocation via `/api/runs`.

### 2. Dashboard Setup

In a separate terminal, initialize and start the frontend interface:

```bash
cd dashboard
npm install
npm run dev

```

* **User Interface:** Access the dashboard at [http://localhost:3000](http://localhost:3000?utm_source=gemini).

---

## Operational Guide

### Running a Demonstration Experiment

1. Ensure both the backend and frontend servers are running, then open the dashboard.
2. Select **`liquid_mixing_v1`** from the experiment dropdown (loaded automatically from `configs/experiments/liquid_mixing_v1.yaml`) and click **Start Run**.
3. If a webcam is detected at index 0, the live video feed will initialize; otherwise, the pipeline executes headless (perception stubs remain active regardless).
4. Monitor the **Procedure Status** panel as it tracks progression through states automatically. The stub activity recognizer cycles through `PICK` $\rightarrow$ `OPEN` $\rightarrow$ `POUR` $\rightarrow$ `MIX` $\rightarrow$ `CLOSE` $\rightarrow$ `PLACE` at regular intervals, demonstrating end-to-end pipeline integration (Perception $\rightarrow$ Procedure Engine $\rightarrow$ Alerts $\rightarrow$ Voice Synthesizer $\rightarrow$ Recording $\rightarrow$ Dashboard) without requiring a trained model.
5. Click **Stop Run** to terminate early, or allow the sequence to complete naturally.
6. Execution artifacts, including recorded video files and event logs, are saved to `backend/recordings/{run_id}.*`.

### Running Against a Video File

To execute a run using pre-recorded media instead of a live webcam, pass the absolute path in the request body:

```json
{
  "source": "/path/to/video.mp4"
}

```

*Tip: You can execute this call directly using the Swagger UI at `/docs`.*

---

## Validation & Testing

Run the standalone procedure-engine test suite directly from the project root:

```bash
python3 test_engine.py

```

This suite executes **9 core scenarios spanning 26 assertions**, validating:

* Happy-path execution flow
* `WRONG_OBJECT` error handling
* Precondition failure leading to `FAILED` states
* `SKIPPED` and `WRONG_ORDER` sequence management
* `REPEATED` step detection
* `TIMEOUT` retry policies, recovery routines, and failure escalations
* Optional-step automatic skipping
* The complete `RECOVERED` lifecycle

*Note: The test suite stubs out Pydantic-based configuration schemas using lightweight dataclasses, allowing rapid verification of engine logic in isolation without requiring the full backend dependency stack.*

```

```
