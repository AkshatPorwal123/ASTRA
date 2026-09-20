"""
Orchestrates a single experiment run's background loop:
  capture frame -> AI pipeline -> procedure engine -> {voice, recording,
  logging, DB, WebSocket broadcast}

This is the "glue" that Section 31 requires to be decoupled: capture,
inference, and I/O run as an asyncio task per run, communicating only
through explicit function calls / events — no component reaches into
another's internals.
"""
from __future__ import annotations
import asyncio
import time
import cv2
import psutil

from backend.ingestion.capture import VideoSource
from backend.ai_pipeline.orchestrator import AIPipelineOrchestrator
from backend.procedure_engine.engine import ProcedureEngine
from backend.procedure_engine.graph_loader import load_graph
from backend.recording.recorder import RunRecorder
from backend.voice.tts import VoiceAssistant
from backend.streaming.mjpeg import encode_jpeg
from backend.ws.manager import manager
from backend.db.database import SessionLocal
from backend.db import models
from backend.ai_pipeline.features import extract_features
from backend.dataset.recorder import DatasetRecorder
from backend.dataset.object_recorder import ObjectDatasetRecorder
from backend.streaming.push import StreamPusher, parse_destination


class ActiveRun:
    def __init__(self, run_id: str, experiment_config_path: str, source: str | int = 0,
                 record: bool = True, stream_to: str | None = None,
                 use_rack_reference: bool = False, rack_marker_id: int = 0):
        self.run_id = run_id
        self.source = VideoSource(source)
        self.pipeline = AIPipelineOrchestrator(use_rack_reference=use_rack_reference, rack_marker_id=rack_marker_id)
        self.config = load_graph(experiment_config_path)
        self.engine = ProcedureEngine(self.config)
        self.voice = VoiceAssistant()
        self.recorder = RunRecorder(run_id) if record else None
        self.dataset_recorder = DatasetRecorder(run_id)
        self.object_dataset_recorder = ObjectDatasetRecorder(run_id)
        self.latest_jpeg: bytes | None = None
        self.latest_raw_frame = None   # for calibration sampling (backend/vision/calibration.py) — needs true camera pixels, not the annotated overlay
        self._task: asyncio.Task | None = None
        self._running = False
        self._frame_count = 0
        self._last_metrics_ts = 0.0

        self.stream_pusher: StreamPusher | None = None
        if stream_to:
            ip, port = parse_destination(stream_to)   # raises ValueError with a clear message on bad input
            self.stream_pusher = StreamPusher(ip, port, run_id=run_id)

    async def _emit(self, event: dict):
        await manager.broadcast(self.run_id, event)
        if self.recorder:
            self.recorder.log_event(event)

    async def start(self):
        opened = self.source.open()
        self._running = True
        if self.stream_pusher:
            self.stream_pusher.start()
        self.voice.say("Experiment started.")
        await self._emit({"type": "procedure.state", "payload": self.engine._snapshot(events=[{
            "type": "run_started", "camera_available": opened
        }])})
        self._task = asyncio.create_task(self._loop(opened))

    async def _loop(self, camera_available: bool):
        self.pipeline.warmup()
        while self._running:
            frame = None
            if camera_available and self.source.is_open():
                ok, frame, _ts = self.source.read()
                if not ok:
                    frame = None

            result = await self.pipeline.process_frame(self.run_id, self._emit, frame=frame)
            self._frame_count += 1

            if frame is not None:
                self.latest_raw_frame = frame
                self.object_dataset_recorder.update_current_frame(frame, result["objects"].get("objects", []))
                overlay = self._draw_overlay(frame, result)
                self.latest_jpeg = encode_jpeg(overlay)
                if self.recorder:
                    self.recorder.write_frame(overlay)
                if self.stream_pusher and self.latest_jpeg:
                    self.stream_pusher.push(self.latest_jpeg)

            activity = result["activity"]

            feats = extract_features(result["pose"], result["tracks"].get("tracks", []), result["interaction"].get("interactions", []))
            self.dataset_recorder.update_current_frame(feats, activity.get("label", "UNKNOWN"))

            snapshot = self.engine.observe_activity(
                activity_label=activity.get("label", "UNKNOWN"),
                confidence=result["activity_confidence"],
                ts=result["frame_ts"],
                object_class=activity.get("object_class"),
            )
            await self._emit({"type": "procedure.state", "payload": snapshot})

            for event in snapshot.get("events", []):
                await self._handle_engine_event(event)

            await self._emit_system_metrics()

            if self.engine.state.status in ("COMPLETED", "FAILED"):
                await self.stop()
                break

            await asyncio.sleep(1 / 10)   # ~10 Hz loop; frame skipping in spirit of Section 23

    @staticmethod
    def _draw_overlay(frame, result: dict):
        """Draws real detection boxes onto the frame for the dashboard's
        Live Video panel (Section 28/29: AI overlay). Non-destructive —
        operates on a copy so the recorded/streamed frame reflects what the
        pipeline actually saw, which matters for the replay/explainability
        requirement in Section 21."""
        overlay = frame.copy()
        for p in result.get("person", {}).get("persons", []):
            x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(overlay, str(p.get("track_id", "")), (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        for o in result.get("objects", {}).get("objects", []):
            x1, y1, x2, y2 = [int(v) for v in o["bbox"]]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(overlay, o.get("class", ""), (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        joints = result.get("pose", {}).get("joints", {})
        if joints:
            from backend.ai_pipeline.real_models import RealPoseEstimator
            for a_idx, b_idx in RealPoseEstimator.CONNECTIONS:
                a_name = RealPoseEstimator.LANDMARK_NAMES[a_idx]
                b_name = RealPoseEstimator.LANDMARK_NAMES[b_idx]
                a, b = joints.get(a_name), joints.get(b_name)
                if a and b and a["visibility"] > 0.3 and b["visibility"] > 0.3:
                    cv2.line(overlay, (int(a["x"]), int(a["y"])), (int(b["x"]), int(b["y"])), (255, 200, 0), 2)
            for name, j in joints.items():
                if j["visibility"] > 0.3:
                    cv2.circle(overlay, (int(j["x"]), int(j["y"])), 3, (255, 200, 0), -1)

        # Hand-object interaction: draw a line from the wrist to whichever
        # object it's interacting with, colored by state, labeled with the
        # state name (Section 10 — makes the interaction model's output
        # visible/explainable, not just logged).
        state_colors = {
            "APPROACH": (0, 220, 220), "TOUCH": (0, 165, 255),
            "HOLD": (0, 0, 255), "RELEASE": (180, 0, 255),
        }
        tracks_by_id = {t.get("track_id"): t for t in result.get("tracks", {}).get("tracks", [])}
        for interaction in result.get("interaction", {}).get("interactions", []):
            state = interaction["state"]
            if state == "NONE" or interaction.get("object_track_id") is None:
                continue
            wrist = joints.get(interaction["hand"])
            obj = tracks_by_id.get(interaction["object_track_id"])
            if not wrist or not obj:
                continue
            color = state_colors.get(state, (255, 255, 255))
            ox1, oy1, ox2, oy2 = obj["bbox"]
            obj_center = (int((ox1 + ox2) / 2), int((oy1 + oy2) / 2))
            wrist_pt = (int(wrist["x"]), int(wrist["y"]))
            cv2.line(overlay, wrist_pt, obj_center, color, 2)
            cv2.putText(overlay, state, (wrist_pt[0] + 6, wrist_pt[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return overlay

    # Alert types that represent a real procedure violation vs. ones that
    # are informational/recoverable — drives voice tone and DB severity
    # (Section 17/19).
    _ERROR_ALERT_TYPES = {"FAILED", "INVALID_ACTION"}

    def _step_label(self, step_id: str | None) -> str:
        """Human-readable step name for voice/alert messages, e.g. 'Hold
        the box steady' instead of the raw config ID 'S2_HOLD' — voice
        output should read like an instruction, not a log line. Falls back
        to the raw ID for anything the config doesn't recognize (e.g.
        'TERMINAL', or a stale ID from before a config change)."""
        if not step_id:
            return "this step"
        step = self.config.step_by_id(step_id)
        return (step.name if step and step.name else step_id)

    async def _handle_engine_event(self, event: dict):
        db = SessionLocal()
        try:
            etype = event["type"]

            if etype == "step_completed":
                step = self.config.step_by_id(event["step_id"])
                self.voice.step_completed(step.name or step.id)
                nxt = self.engine.next_action()
                if nxt.voice_instruction:
                    self.voice.next_step(nxt.voice_instruction)
                db.add(models.ProcedureState(
                    run_id=self.run_id, ts=event["ts"], step_id=event["step_id"],
                    status="COMPLETED", snapshot_json=event,
                ))

            elif etype == "step_skipped":
                step = self.config.step_by_id(event["step_id"])
                self.voice.warning(f"Skipping optional step: {step.name or step.id}.")
                db.add(models.ProcedureState(
                    run_id=self.run_id, ts=event["ts"], step_id=event["step_id"],
                    status="SKIPPED", snapshot_json=event,
                ))

            elif etype == "alert":
                alert_type = event["alert_type"]
                severity = event.get("severity", "ERROR" if alert_type in self._ERROR_ALERT_TYPES else "WARNING")
                step_label = self._step_label(event.get("step_id"))
                if alert_type == "FAILED":
                    failing = (event.get("evidence") or {}).get("failing") or []
                    if failing:
                        message = f"{step_label} cannot proceed: {'; '.join(failing)} not satisfied."
                    else:
                        message = f"{step_label} could not be validated."
                else:
                    message = {
                        "SKIPPED": f"{step_label} appears to have been skipped.",
                        "WRONG_ORDER": f"Please complete '{step_label}' before continuing.",
                        "REPEATED": f"{step_label} was already completed.",
                        "WRONG_OBJECT": f"Wrong object for {step_label}.",
                        "INVALID_ACTION": "Unrecognized action observed.",
                        "TIMEOUT": f"{step_label} is taking longer than expected.",
                        "UNCERTAIN": "Uncertain activity detected.",
                    }.get(alert_type, f"{alert_type} detected near {step_label}.")
                if severity == "ERROR":
                    self.voice.error(message)
                else:
                    self.voice.warning(message)
                db.add(models.Alert(
                    run_id=self.run_id, ts=event["ts"], type=alert_type,
                    step_id=event.get("step_id"), severity=severity,
                    evidence_json=event, status="TENTATIVE",
                ))
                await self._emit({"type": "alert", "payload": event})

            elif etype == "alert_recovered":
                stale = (
                    db.query(models.Alert)
                    .filter_by(run_id=self.run_id, step_id=event["step_id"], type=event["original_alert_type"])
                    .filter(models.Alert.status != "RECOVERED")
                    .order_by(models.Alert.ts.desc())
                    .first()
                )
                if stale:
                    stale.status = "RECOVERED"
                step_label = self._step_label(event.get("step_id"))
                original = event["original_alert_type"].replace("_", " ").lower()
                self.voice.recovery(f"Resolved: {original} on {step_label} — back on track.")
                await self._emit({"type": "alert", "payload": {
                    "type": "alert", "alert_type": "RECOVERED", "step_id": event["step_id"],
                    "ts": event["ts"], "evidence": {"original_alert_type": event["original_alert_type"]},
                }})

            elif etype == "recovery_redirect":
                step_label = self._step_label(event.get("recovery_step"))
                self.voice.recovery(f"Redirecting to recovery step: {step_label}.")

            db.commit()
        finally:
            db.close()

    async def _emit_system_metrics(self):
        now = time.time()
        if now - self._last_metrics_ts < 1.0:
            return
        self._last_metrics_ts = now
        metrics = {
            "cpu_percent": psutil.cpu_percent(),
            "ram_percent": psutil.virtual_memory().percent,
            "frame_count": self._frame_count,
            "camera_open": self.source.is_open(),
        }
        await self._emit({"type": "system.metrics", "payload": metrics})

    async def stop(self):
        self._running = False
        self.source.release()
        if self.stream_pusher:
            self.stream_pusher.stop()
        if self.recorder:
            self.recorder.close()
        if self.engine.state.status == "COMPLETED":
            self.voice.experiment_complete()
        elif self.engine.state.status == "FAILED":
            self.voice.error("Experiment failed. Please review the alert log.")
        else:
            self.voice.say("Run stopped.")
        db = SessionLocal()
        try:
            run = db.query(models.ExperimentRun).get(self.run_id)
            if run:
                run.status = self.engine.state.status if self.engine.state.status in ("COMPLETED", "FAILED") else "ABORTED"
                run.ended_at = __import__("datetime").datetime.utcnow()
                db.commit()
            from backend.reporting.generator import generate_report
            generate_report(db, self.run_id, self.config)
        finally:
            db.close()


class RunRegistry:
    def __init__(self):
        self._runs: dict[str, ActiveRun] = {}

    def get(self, run_id: str) -> ActiveRun | None:
        return self._runs.get(run_id)

    def add(self, run: ActiveRun):
        self._runs[run.run_id] = run

    def remove(self, run_id: str):
        self._runs.pop(run_id, None)


registry = RunRegistry()
