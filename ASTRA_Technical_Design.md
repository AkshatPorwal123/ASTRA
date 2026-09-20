# ASTRA — Autonomous Space Task Recognition & Assistance System
## Complete Technical Design Document
### AI Human Activity Recognition for On-board BAS Experiments (SIH)

**Status:** Design phase — no code yet. This document is the approved-pending blueprint for Phase 1 implementation.

---

## 1. System Architecture (High Level)

ASTRA is organized into eight loosely-coupled subsystems that communicate through well-defined APIs/events, not direct function calls. This is what lets AI modules be built, swapped, or stubbed independently of the dashboard.

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                              │
│   Dashboard (React/Next.js) · Voice Output Device · LAN Viewer   │
└───────────────────────────▲────────────────────────────────────-┘
                             │ WebSocket + REST
┌───────────────────────────┴────────────────────────────────────-┐
│                         API GATEWAY                               │
│         FastAPI (REST + WS) — auth, routing, validation           │
└───┬─────────┬──────────┬──────────┬──────────┬──────────┬────────┘
    │         │          │          │          │          │
┌───▼───┐ ┌───▼────┐ ┌───▼─────┐ ┌──▼──────┐ ┌─▼───────┐ ┌▼────────┐
│ VIDEO │ │   AI   │ │PROCEDURE│ │ VOICE   │ │RECORDING│ │ STREAM  │
│ INGEST│ │ PIPELINE│ │ ENGINE  │ │ ASSIST  │ │ SERVICE │ │ SERVICE │
└───┬───┘ └───┬────┘ └───┬─────┘ └─────────┘ └────┬────┘ └────┬────┘
    │         │          │                        │           │
    └─────────┴──────────┴────────────┬───────────┴───────────┘
                                       │
                              ┌────────▼────────┐
                              │   DATABASE       │
                              │ (PostgreSQL)     │
                              └──────────────────┘
```

**Design principle:** the AI Pipeline is runnable headless from the command line against a webcam or a video file, producing the same event stream it would emit to the dashboard. The dashboard is a *consumer* of that event stream, never a dependency of it.

---

## 2. Component Architecture

| Component | Responsibility | Can run without dashboard? |
|---|---|---|
| Video Ingestion | Camera/file capture, frame sync, buffering | Yes |
| AI Inference Pipeline | Person/pose/object/interaction/activity models | Yes |
| Procedure Engine | State machine, sequence validation, next-action reasoning | Yes |
| Voice Assistant | Offline TTS, event-triggered phrases | Yes |
| Recording Service | Local video + timestamp-correlated event log | Yes |
| Streaming Service | LAN/IP MJPEG/RTSP relay | Optional |
| API Gateway | REST + WebSocket surface | Bridges everything to clients |
| Dashboard | Visualization, control, replay | No (thin client) |
| Database | Persistent state, logs, config | Shared dependency |

Each AI component (`PersonDetector`, `PoseEstimator`, `ObjectDetector`, `HandObjectInteraction`, `ActivityRecognizer`, `ProcedureEngine`) is defined as an **abstract interface** with a stub/mock implementation in Phase 1, so later milestones swap implementations without touching the orchestrator or API layer.

---

## 3. AI Architecture

The AI pipeline is a **directed acyclic graph of stateless-per-frame stages**, with an explicit temporal buffer stage between per-frame perception and sequence-level reasoning:

```
Frame → PersonDetector → Tracker → PoseEstimator(2D→3D) → OrientationEstimator
Frame → ObjectDetector → ObjectTracker → ObjectStateEstimator
(Pose + Objects) → HandObjectInteractionModel
(Pose + Objects + Interaction) → Spatio-Temporal Feature Buffer (sliding window)
Feature Buffer → ActivityRecognizer → Temporal Action Segmenter
Segmented Actions → ExperimentStepRecognizer
Step Recognition → ProcedureEngine (state machine + sequence validator)
ProcedureEngine → { ErrorDetector, NextActionReasoner }
→ Voice Assistant / Dashboard / Logger
```

Every stage emits a structured, versioned JSON payload (`schema_version`, `frame_ts`, `run_id`, `confidence`) so any stage can be logged, replayed, or evaluated independently — this is what makes per-layer evaluation (Section 37) possible.

**Model interface contract** (all perception models implement this):
```python
class PerceptionModel(Protocol):
    def infer(self, frame: FrameBundle) -> StructuredOutput: ...
    def warmup(self) -> None: ...
    def get_metadata(self) -> ModelMetadata: ...  # name, version, input_shape, device
```

---

## 4. Data Flow (End-to-End)

1. Camera driver pushes frames into a ring buffer (video ingestion).
2. Frame Synchronizer timestamps and, if multi-camera, aligns frames across sources.
3. Frames are simultaneously (a) recorded to disk, (b) sent to the AI pipeline, (c) optionally re-encoded for LAN streaming.
4. AI pipeline runs asynchronously (frame skipping allowed) and emits per-frame perception events onto an internal event bus (in-process pub/sub in Phase 1; can graduate to Redis Streams later).
5. The Spatio-Temporal buffer aggregates N seconds of perception events into windows for the Activity Recognizer.
6. Recognized activities feed the Experiment Step Recognizer, which checks multi-condition step definitions (Section 13/14).
7. The Procedure Engine consumes step events, updates the state machine, runs sequence validation, and emits: `step_completed`, `deviation_detected`, `next_action`.
8. Voice Assistant, Dashboard (via WebSocket), and Logger all subscribe to Procedure Engine events — fan-out, not sequential chaining.
9. Logger writes to DB + JSON/CSV; Recording Service tags the video file with the same `run_id`/timestamps for later replay correlation.

---

## 5. Model Pipeline (Stage-by-Stage Candidates)

| Stage | Baseline (Phase 1–2) | Advanced candidates to benchmark |
|---|---|---|
| Person detection | YOLOv8n/YOLO11n | RT-DETR, YOLO11 variants |
| Tracking | ByteTrack | BoT-SORT, DeepSORT w/ ReID |
| 2D pose | MediaPipe Pose (fast, easy) | RTMPose, ViTPose, YOLO-Pose |
| 3D pose / mesh | Lifting from 2D (VideoPose3D-style) | 4D-Humans, PARE, ROMP (research extension) |
| Object detection | YOLOv8n fine-tuned on custom classes | RT-DETR fine-tuned |
| Hand keypoints | MediaPipe Hands | RTMPose-hand, WiLoR |
| Interaction model | Rule-based (distance/velocity/contact heuristics) | Learned classifier on hand-object relational features |
| Activity recognition | TCN over pose+object features | ST-GCN / CTR-GCN (skeleton), PoseC3D, temporal transformer |
| Procedure reasoning | Deterministic hierarchical FSM + graph | (kept deterministic by design — see Section 15) |

All models load through ONNX Runtime where possible so the same weights run across dev machine / edge device without re-implementation.

---

## 6. 3D Pose / HMR Strategy

Given microgravity has no reliable "up," 3D representation matters more than usual, but full HMR (SMPL mesh) is the highest-risk, most compute-heavy component. Staged approach:

1. **Baseline:** 2D pose only, joints reported in image space. Sufficient to bootstrap interaction/activity models.
2. **Stage 2:** Monocular 3D pose lifting (2D→3D regression) — cheap, works from a single camera, gives joint depth-relative geometry.
3. **Stage 3 (research extension):** Full mesh recovery (4D-Humans / PARE-class models) *only if* Stage 2 accuracy is insufficient for orientation-agnostic reasoning, given the added latency/VRAM cost.
4. All 3D outputs are expressed in the **payload/rack coordinate frame**, not a "floor" frame: a calibration step establishes the camera-to-rack transform once per setup, and all joint coordinates are reported relative to it.

Labeled status: 2D pose = IMPLEMENTED in Phase 3 target; 3D lifting = PARTIALLY IMPLEMENTED / research milestone; full HMR = RESEARCH EXTENSION, not committed for the SIH demo unless time allows.

---

## 7. Human-Object Interaction Strategy

Interaction is computed as a **feature vector per (hand, object) candidate pair**, not a single distance threshold:

```
features = [
  wrist_to_object_center_distance,
  relative_velocity (wrist vs object),
  approach_angle,
  hand_openness (finger spread, if available),
  contact_flag (bbox overlap or keypoint-in-bbox),
  temporal_persistence (frames in near-contact),
  object_motion_delta (did object move while hand nearby),
]
```

Phase 1–2: rule-based thresholding on this vector (interpretable, no training data needed) classifies into {NONE, APPROACH, TOUCH, HOLD, MANIPULATE(POUR/OPEN/CLOSE/PRESS/TURN), RELEASE}.
Phase 3+: replace the rule thresholds with a small learned classifier (gradient-boosted trees or shallow MLP) trained on the custom dataset, keeping the same feature vector so the swap is drop-in.

---

## 8. HAR (Human Activity Recognition) Strategy

Two parallel tracks evaluated and benchmarked against each other rather than assumed:

- **Skeleton-based:** pose sequence → graph → ST-GCN/CTR-GCN. Lightweight, orientation can be partially normalized, doesn't need RGB.
- **RGB+pose fusion:** short clip features (PoseC3D-style or a lightweight video transformer) + skeleton stream, late-fused. More robust to occlusion/ambiguous poses, heavier compute.

Baseline for Phase 1–3 demo: a TCN over a fused per-frame vector `[pose_features, object_features, interaction_features]` — cheap, real-time capable, and directly interpretable (important for explainability requirement). Upgrade path to ST-GCN benchmarked once labeled clips exist.

---

## 9. Temporal Action Segmentation Strategy

Sliding-window classification with a smoothing/segmentation post-process (not raw per-frame class output, which is noisy):

1. Per-frame activity logits from the Activity Recognizer.
2. Median filter / HMM-style smoothing over a short window to suppress flicker.
3. Boundary detection: an action "starts" when smoothed confidence crosses threshold for `T_min` consecutive frames, "ends" symmetrically.
4. Emit `(action_label, start_ts, end_ts, mean_confidence)` segments — these are what feed Experiment Step Recognition.

This directly satisfies Section 12's frame-range example format.

---

## 10. Procedure Reasoning Architecture

Three-layer reasoning, each independently testable:

1. **Step Recognizer** — evaluates whether the *current* action-segment stream satisfies a step's multi-condition definition (actor, object, target, action, required object-state, minimum duration). Outputs `step_candidate(step_id, confidence)`.
2. **State Machine** — hierarchical FSM (see Section 11) consumes confirmed step candidates and transitions procedure state.
3. **Sequence Validator** — compares the *observed* step order against the *expected* graph (Section 16) and classifies each transition (CORRECT / SKIPPED / WRONG_ORDER / REPEATED / WRONG_OBJECT / INVALID_ACTION / TIMEOUT / UNCERTAIN).

Kept deterministic and rule-driven on top of the ML step-confidence inputs, so the reasoning is explainable/auditable even though perception underneath is learned.

---

## 11. State-Machine / Procedure-Graph Design

Represented as a **Hierarchical Procedure Graph**:

```json
{
  "step_id": "S3_ADD_LIQUID",
  "name": "Add liquid",
  "preconditions": ["S2_OPEN_CONTAINER.postcondition_met"],
  "required_activity": "POUR",
  "required_object": "liquid_bottle",
  "target_object": "container",
  "expected_duration_s": [3, 12],
  "confidence_threshold": 0.65,
  "timeout_s": 45,
  "postconditions": ["container.state == FILLED"],
  "on_success": "S4_MIX",
  "on_timeout": "ERROR_S3_TIMEOUT",
  "voice_instruction": "Next step: add liquid to the open container.",
  "error_conditions": {
    "wrong_object": "WARN_WRONG_OBJECT",
    "container_not_open": "WARN_PRECONDITION_FAILED"
  }
}
```

Nodes: `NORMAL`, `OPTIONAL`, `BRANCH`, `RECOVERY`, `TIMEOUT`, `FAILURE`, `UNCERTAIN`. Edges carry conditions (precondition satisfied, activity confirmed, timeout elapsed). This graph is data (YAML/JSON), never hardcoded Python — satisfying Section 33.

Evidence accumulation (Section 17) sits between raw ML confidence and state transitions: a candidate deviation moves `TENTATIVE → CONFIRMED` only after N consecutive frames/segments of consistent evidence, and can move to `RECOVERED` if the astronaut corrects course — this is what prevents single noisy frames from firing false alarms.

---

## 12. Error Detection Architecture

```
Step confidence stream ──▶ Evidence Accumulator (sliding window, EMA of confidence)
                                   │
                     threshold crossed for N frames?
                          │ yes              │ no
                          ▼                  ▼
                    TENTATIVE            (no state change)
                          │
              evidence persists for M more frames?
                          │ yes
                          ▼
                     CONFIRMED ──▶ emit alert (type, step, evidence trace)
                          │
              corrective action observed?
                          │ yes
                          ▼
                      RECOVERED ──▶ emit recovery event
```

Every alert stores its full evidence trace (which frames/segments contributed) for explainability and later replay review.

---

## 13. Next-Action Reasoning Architecture

Pure function of current procedure-graph node + last N events, no separate ML model required:

```python
def next_action(state: ProcedureState) -> NextActionAdvice:
    node = graph[state.current_step]
    if state.status == "COMPLETED":
        candidates = node.on_success_options()  # supports branches
    elif state.status == "DEVIATION_CONFIRMED":
        candidates = node.recovery_options()
    return NextActionAdvice(
        next_step=candidates[0],
        reason=explain(state, node),
        required_object=node.required_object,
        target=node.target_object,
        voice_instruction=node.voice_instruction,
    )
```

Output feeds both the Voice Assistant and the dashboard "Current State" panel identically — one source of truth.

---

## 14. Offline Edge Architecture

- All models exported to **ONNX**; runtime is **ONNX Runtime** with execution providers selected at startup: CUDA/TensorRT EP if NVIDIA GPU present, OpenVINO EP on Intel CPU/iGPU, CPU EP as universal fallback.
- Quantization: FP16 by default on GPU; INT8 (post-training static quantization, calibrated on a held-out clip set) evaluated for CPU-only edge targets.
- Pipeline is asynchronous: capture, inference, and I/O run in separate threads/processes connected by bounded queues, with **frame skipping** (process every Kth frame under load) rather than blocking capture.
- Zero dependency on cloud inference or cloud TTS at runtime — internet only used at build/train time (Section 24).
- Metrics collected continuously: per-stage latency, end-to-end FPS, CPU/GPU/RAM/VRAM (via `psutil` + `pynvml`/`py-spy`), surfaced on the System Monitoring dashboard panel.

---

## 15. Video Streaming Architecture

Two independent streams from one capture source:

- **AI-resolution stream**: full-res/full-rate frames fed only to the inference pipeline (never leaves the device).
- **Monitoring stream**: downscaled, re-encoded (MJPEG over HTTP for simplicity in Phase 1; RTSP/WebRTC as a later upgrade) for LAN dashboard/authorized viewers, bandwidth-capped and resolution-configurable.

Local recording writes the *AI-resolution* stream (or a configurable resolution) to disk with a sidecar `events.jsonl` correlating video PTS to AI/procedure events, enabling synchronized replay (Section 21).

---

## 16. Database Schema (Core Tables)

```sql
experiments(id, name, description, config_yaml_ref, created_at)
experiment_runs(id, experiment_id FK, participant_id FK, started_at, ended_at, status, video_file_ref)
participants(id, name, metadata_json)
objects(id, run_id FK, class_name, first_seen_ts, last_seen_ts)
object_states(id, object_id FK, state, ts, confidence)
detections(id, run_id FK, frame_ts, entity_type, entity_id, bbox_json, confidence)
poses(id, run_id FK, frame_ts, person_track_id, joints_json, frame_type[2d|3d])
activities(id, run_id FK, start_ts, end_ts, label, confidence, source_model)
interactions(id, run_id FK, ts, hand_id, object_id, interaction_type, confidence)
procedure_states(id, run_id FK, ts, step_id, status, state_machine_snapshot_json)
alerts(id, run_id FK, ts, type, step_id, severity, evidence_json, status[tentative|confirmed|recovered])
logs(id, run_id FK, ts, level, message, context_json)
reports(id, run_id FK, generated_at, summary_json, file_ref)
```

PostgreSQL chosen over SQLite: multiple concurrent writers (ingestion, inference, procedure engine, API) plus JSONB columns for flexible per-stage payloads and future analytical querying outweigh SQLite's simplicity advantage; SQLite remains a fallback for isolated single-process dev/test runs. Managed with Alembic migrations.

---

## 17. API Architecture

```
/api/experiments        CRUD experiment definitions (config-driven)
/api/runs                start/stop/list runs
/api/video/{run_id}      stream/download recorded video
/api/inference/status    model load state, active EPs, warmup status
/api/detections/{run_id} paginated detection/pose/object records
/api/activities/{run_id} activity segments
/api/procedure/{run_id}  current + historical procedure state
/api/alerts/{run_id}     alerts with evidence traces
/api/logs/{run_id}       raw event log (JSON/CSV export)
/api/reports/{run_id}    generated report (PDF/HTML)
/api/system              live resource metrics
```

REST for CRUD/history/export; everything *real-time* goes over WebSocket.

---

## 18. WebSocket Architecture

Single multiplexed WS channel per run (`/ws/runs/{run_id}`), message-typed:

```json
{"type": "perception.update", "payload": {...}}
{"type": "activity.segment", "payload": {...}}
{"type": "procedure.state", "payload": {...}}
{"type": "alert", "payload": {...}}
{"type": "system.metrics", "payload": {...}}
```

Dashboard subscribes once and routes by `type`. Reconnection is stateless — on reconnect the client calls REST `/api/procedure/{run_id}` to resync current state, then resumes streaming deltas, so dropped connections never desync the UI.

---

## 19. Dashboard Architecture

Next.js + React, WebSocket client with a typed event reducer feeding independent panels (Section 28/29): Live Video w/ AI overlay canvas, Procedure Status checklist, Current State card, Alerts feed, Timeline, System Monitoring, Video Replay. Panels are independent React components subscribing to slices of the same event store (Zustand/Redux) — no panel blocks another. Progressive disclosure: skeleton/3D/graph visualizations are opt-in expandable sections, not shown by default.

---

## 20. Dataset Strategy

Public candidates to evaluate for pretraining/transfer (not assumed sufficient): general human activity sets (NTU RGB+D) for skeleton-model pretraining, hand-object interaction sets (Assembly101) for interaction-model pretraining, and any available astronaut/microgravity pose research data (MicroG-4M, Astro-Pose, PhysAstro-Pose) if accessible and licensed — treated as *research references to investigate*, not guaranteed usable, since availability/licensing must be verified before relying on them. None of these solve procedure-recognition directly; a custom dataset is required regardless (Section 21).

---

## 21. Custom Dataset Design

Recording protocol producing labeled runs across: multiple participants, clothing, lighting, backgrounds, object/camera placements, execution speed, and — critically — both correct and *deliberately flawed* runs (skipped/reordered/repeated/wrong-object/failed/interrupted/uncertain), plus orientation variants (rotated camera, non-upright body, simulated-microgravity posture using a tilted rig or floor mat improvisation). Each run stored with: raw video, per-frame manual/semi-automated pose+object annotations (bootstrapped from baseline model predictions, human-corrected), and a step-segment label file. Split by **participant ID** (subject-independent) for train/val/test to avoid identity leakage into accuracy numbers.

---

## 22. Experiment Configuration Schema

```yaml
experiment: liquid_mixing_v1
objects:
  - id: container
    class: container
    states: [AVAILABLE, OPEN, CLOSED, FILLED, EMPTY]
  - id: liquid_bottle
    class: bottle
    states: [AVAILABLE, HELD]
steps:
  - id: S1_PICK_CONTAINER
    activity: PICK
    required_object: container
    postconditions: ["container.state == HELD"]
    next: S2_OPEN_CONTAINER
    voice_instruction: "Pick up the container."
  - id: S2_OPEN_CONTAINER
    activity: OPEN
    required_object: container
    preconditions: ["container.state == HELD"]
    postconditions: ["container.state == OPEN"]
    next: S3_ADD_LIQUID
    voice_instruction: "Open the container."
  # ... S3–S6 as in Section 38
thresholds:
  default_confidence: 0.65
  default_timeout_s: 45
```

Loaded and validated (JSON-schema) at run start; the Procedure Engine builds its graph purely from this file — zero hardcoded step logic in Python.

---

## 23. Model Training Strategy

1. Bootstrap with pretrained public weights for person/pose/object detectors (transfer learning), fine-tune object detector on custom object classes only (small, cheap).
2. Interaction and activity models trained on the custom dataset once a minimum viable set of labeled runs exists (target: ≥20 runs across ≥3 participants before first training pass).
3. Iterate baseline → benchmark alternatives → integrate strongest, per Section 40's rule — never skip the baseline step.
4. Track experiments (config, metrics, weights) with a lightweight experiment tracker (e.g., a local MLflow instance) for reproducibility.

---

## 24. Model Evaluation Strategy

Per-layer metrics exactly as enumerated in Section 37, computed automatically after each run against ground-truth labels where available, and surfaced in a dedicated `/api/reports` evaluation report. **Procedure Compliance Accuracy** (correct classification of CORRECT/SKIPPED/WRONG_ORDER/etc. across the full test set) is treated as the headline metric since it's the actual research contribution, not raw detection/pose accuracy.

---

## 25. Edge Optimization Strategy

Benchmark matrix run on target hardware before committing to a configuration: {FP32, FP16, INT8} × {CPU EP, CUDA/TensorRT EP, OpenVINO EP} × {full pipeline, frame-skipped pipeline}, measuring latency/FPS/resource use per Section 23. Chosen configuration is whichever meets a real-time target (defined as sustained ≥10 FPS end-to-end) at the lowest resource cost, documented with the actual numbers rather than assumed.

---

## 26. Security Considerations

- LAN streaming and dashboard access behind authenticated sessions (token-based); streaming endpoint not exposed beyond the local network by default.
- No experiment video/data leaves the device unless explicitly exported.
- Config files and procedure graphs validated against a strict schema before load (prevents malformed/injected experiment definitions from corrupting the state machine).
- Structured logs avoid storing raw biometric imagery outside the recording service's access-controlled storage.

---

## 27. Failure Handling & Fallback Mechanisms

| Failure | Fallback |
|---|---|
| Camera disconnect | Freeze last frame, flag `camera_status: DOWN` on dashboard, pause AI pipeline, auto-resume on reconnect |
| Model load failure | Component reports `UNAVAILABLE`; downstream stages degrade gracefully (e.g., no 3D pose → fall back to 2D-only reasoning) rather than crash |
| GPU unavailable | Auto-fallback to CPU EP with reduced frame rate, logged clearly |
| DB write failure | Buffer events in memory/local file, retry with backoff, alert on dashboard |
| WebSocket drop | Client resyncs via REST on reconnect (Section 18) |
| Low-confidence perception | Feed into evidence accumulator as `UNCERTAIN` rather than forcing a hard classification |

---

## 28. Complete Folder Structure

```
astra/
├── backend/
│   ├── api/                # FastAPI routers (Section 17)
│   ├── ws/                 # WebSocket handlers
│   ├── ingestion/          # camera/file capture, frame sync
│   ├── ai_pipeline/
│   │   ├── interfaces/     # PerceptionModel protocol + all model interfaces
│   │   ├── person/
│   │   ├── pose/
│   │   ├── objects/
│   │   ├── interaction/
│   │   ├── activity/
│   │   └── orchestrator.py
│   ├── procedure_engine/
│   │   ├── graph_loader.py
│   │   ├── state_machine.py
│   │   ├── sequence_validator.py
│   │   ├── error_detector.py
│   │   └── next_action.py
│   ├── voice/
│   ├── recording/
│   ├── streaming/
│   ├── db/                 # models, migrations
│   └── config/             # experiment YAML/JSON schemas
├── dashboard/               # Next.js app
├── datasets/
│   ├── raw_runs/
│   ├── annotations/
│   └── splits/
├── models/                  # ONNX weights, versioned
├── training/                 # training/eval scripts, experiment tracking
├── configs/experiments/      # per-experiment YAML definitions
├── scripts/                  # benchmarking, edge-profile scripts
└── docs/
```

---

## 29. Development Roadmap

Adopting the 13 milestones exactly as specified in the brief (Section 36), unchanged — foundation → perception → human understanding → interaction → HAR → procedure intelligence → error intelligence → mission assistance → experiment record → edge optimization → orientation/microgravity research → evaluation → final demonstration.

---

## 30. Team Role Distribution (suggested, 5–6 person team)

| Role | Owns |
|---|---|
| AI/ML Lead (CV) | Person/object detection, tracking, pose |
| CV Researcher (temporal) | HAR, temporal segmentation, interaction model |
| Backend Engineer | API, WebSocket, DB, procedure engine plumbing |
| Edge-AI Engineer | ONNX export, quantization, latency profiling |
| Full-Stack Engineer | Dashboard, replay, streaming integration |
| (Shared) | Dataset collection/annotation, procedure graph authoring, voice assistant |

---

## 31. Expected Technical Difficulties

- 3D pose/orientation-agnostic reasoning without ground-truth microgravity data — highest research risk.
- Interaction model false positives/negatives under occlusion (hands frequently occluded by objects).
- Custom dataset size will likely be small relative to what skeleton/temporal models typically want — mitigated via transfer learning + rule-based baselines as fallback.
- Real-time budget is tight once pose + object + interaction + activity all run per frame — frame skipping and async pipelines are load-bearing, not optional.
- False-alarm suppression (evidence accumulation) needs careful threshold tuning to avoid feeling either "trigger-happy" or "unresponsive" during the live demo.

---

## 32. Research Novelty

Per Section 34: individual perception components (detection, pose, HAR, mesh recovery) are **not** claimed as novel. The contribution is the **integration** — combining multimodal visual perception with a formal, data-driven procedure model to autonomously supervise experiment execution and detect procedural deviations onboard, orientation-agnostically. Research question stated as-is from the brief.

---

## 33. Existing Work Comparison (to be researched and filled in with verified sources)

A comparison table will be built during Milestone 6 research against: NASA ISAAC (autonomous robotic/perception systems for ISS), CIMON (astronaut-assistant robot, conversational/monitoring focus), published astronaut/microgravity pose-estimation papers, and general procedural/human-object-interaction-recognition literature — each entry filled in only from verifiable official/peer-reviewed sources, with explicit "what we reuse / limitation / gap" fields, never fabricated citations.

---

## 34. Final SIH Demonstration Architecture

Live run of the Liquid Mixing Experiment (Section 38) end-to-end on the actual dashboard: camera feed with real-time overlay, procedure checklist advancing live, at least one deliberately-triggered deviation (e.g., wrong order) shown being detected and voiced, followed by a recovery, ending in an auto-generated report — demonstrating perception → reasoning → guidance → logging as one continuous, explainable loop.

---

## 35. What Can Be Built Now vs. What Needs More

| Category | Items |
|---|---|
| **Buildable now** | Backend skeleton, DB schema, API/WS scaffolding, video ingestion (webcam + file), recording, LAN streaming foundation, dashboard shell + panels wired to mock data, experiment config schema + loader, procedure engine (graph/state machine/sequence validator) running on synthetic step events |
| **Requires data collection** | Custom dataset (Section 21), any interaction/activity model training |
| **Requires model training** | Fine-tuned object detector, learned interaction classifier, activity recognizer beyond rule-based baseline |
| **Requires research/experimentation** | 3D pose/HMR strategy validation, orientation-agnostic feature design, edge optimization benchmark matrix |
| **High risk** | Real-time full pipeline on modest hardware, interaction-model accuracy under occlusion, orientation-agnostic reasoning |
| **Optional but desirable** | Full SMPL mesh recovery, RTSP/WebRTC streaming upgrade, INT8 quantization, MLflow experiment tracking |

---

## Next Step

This document is the design baseline. **Phase 1** (per Section 42) will implement: backend + frontend scaffolding, video ingestion (webcam + upload), local recording, LAN streaming foundation, DB + migrations, WebSocket plumbing, dashboard shell with system monitoring, and the configurable experiment framework — with clearly stubbed interfaces (`PersonDetector`, `PoseEstimator`, `ObjectDetector`, `ObjectTracker`, `HandObjectInteraction`, `ActivityRecognizer`, `ProcedureEngine`) that return placeholder/mock structured outputs so the rest of the system can be built and tested against a stable contract.

I'll wait for your go-ahead before starting Phase 1.
