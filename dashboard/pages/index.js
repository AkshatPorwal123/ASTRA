import { useEffect, useRef, useState } from "react";
import { useRunEvents } from "../components/useRunEvents";
import { authHeaders, withApiKeyParam, getStoredApiKey, setStoredApiKey } from "../components/apiKey";

const API = process.env.NEXT_PUBLIC_API_BASE;

export default function Dashboard() {
  const [experiments, setExperiments] = useState([]);
  const [selectedExperiment, setSelectedExperiment] = useState("red_box_demo");
  const [experimentSteps, setExperimentSteps] = useState([]);
  const [runId, setRunId] = useState(null);
  const [starting, setStarting] = useState(false);
  const [authRequired, setAuthRequired] = useState(false);
  const [apiKeyInput, setApiKeyInput] = useState("");
  const [hasKey, setHasKey] = useState(true);   // assume fine until /api/config says otherwise
  const [streamTo, setStreamTo] = useState("");
  const [useRackReference, setUseRackReference] = useState(false);
  const videoRef = useRef(null);

  const { perception, procedure, alerts, metrics, connected } = useRunEvents(runId);

  useEffect(() => {
    fetch(`${API}/api/config`)
      .then((r) => r.json())
      .then((d) => {
        setAuthRequired(!!d.auth_required);
        if (d.auth_required) setHasKey(!!getStoredApiKey());
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    fetch(`${API}/api/experiments`, { headers: authHeaders() })
      .then((r) => r.json())
      .then((d) => setExperiments(d.experiments || []))
      .catch(() => {});
  }, [hasKey]);

  // Fetch the full step list for whichever experiment is selected, so the
  // Procedure Status panel can show every step upfront (not just ones
  // discovered so far as the run progresses).
  useEffect(() => {
    if (!selectedExperiment) return;
    fetch(`${API}/api/experiments/${selectedExperiment}`, { headers: authHeaders() })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => setExperimentSteps((d && d.steps) || []))
      .catch(() => setExperimentSteps([]));
  }, [selectedExperiment, hasKey]);

  function submitApiKey() {
    setStoredApiKey(apiKeyInput.trim());
    setHasKey(!!apiKeyInput.trim());
  }

  async function startRun() {
    setStarting(true);
    try {
      const res = await fetch(`${API}/api/runs`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify({ experiment_name: selectedExperiment, source: 0, record: true, stream_to: streamTo.trim() || null, use_rack_reference: useRackReference }),
      });
      const data = await res.json();
      setRunId(data.run_id);
    } finally {
      setStarting(false);
    }
  }

  async function stopRun() {
    if (!runId) return;
    await fetch(`${API}/api/runs/${runId}/stop`, { method: "POST", headers: authHeaders() });
    setRunId(null);
  }

  return (
    <div>
      {authRequired && !hasKey && (
        <div style={{ background: "#3a1f1f", padding: 10, display: "flex", gap: 8, alignItems: "center" }}>
          <span style={{ fontSize: 13, color: "#f87171" }}>This ASTRA instance requires an API key.</span>
          <input
            type="password"
            value={apiKeyInput}
            onChange={(e) => setApiKeyInput(e.target.value)}
            placeholder="Enter API key"
            style={{ background: "#131722", color: "#e6e6e6", border: "1px solid #232838", borderRadius: 6, padding: "4px 8px" }}
          />
          <button onClick={submitApiKey} style={btnStyle}>Save</button>
        </div>
      )}
      <header style={{ padding: "16px 16px 0", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 20 }}>ASTRA</h1>
          <div style={{ color: "#8a93a8", fontSize: 13 }}>Autonomous Space Task Recognition &amp; Assistance — Phase 1</div>
        </div>
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <span className={`dot ${connected ? "green" : "red"}`}></span>
          <span style={{ fontSize: 13, color: "#8a93a8" }}>{connected ? "Live" : "Not connected"}</span>
          {!runId ? (
            <>
              <select
                value={selectedExperiment}
                onChange={(e) => setSelectedExperiment(e.target.value)}
                style={{ background: "#131722", color: "#e6e6e6", border: "1px solid #232838", borderRadius: 6, padding: "6px 10px" }}
              >
                {(experiments.length ? experiments : [{ name: "liquid_mixing_v1" }]).map((e) => (
                  <option key={e.name} value={e.name}>{e.name}</option>
                ))}
              </select>
              <input
                type="text"
                value={streamTo}
                onChange={(e) => setStreamTo(e.target.value)}
                placeholder="Push to IP:port (optional)"
                style={{ background: "#131722", color: "#e6e6e6", border: "1px solid #232838", borderRadius: 6, padding: "6px 10px", width: 170 }}
              />
              <label style={{ fontSize: 12, display: "flex", alignItems: "center", gap: 4, color: "#8a93a8" }}>
                <input type="checkbox" checked={useRackReference} onChange={(e) => setUseRackReference(e.target.checked)} />
                Rack-relative tracking
              </label>
              <button onClick={startRun} disabled={starting} style={btnStyle}>
                {starting ? "Starting…" : "Start Run"}
              </button>
            </>
          ) : (
            <button onClick={stopRun} style={{ ...btnStyle, background: "#3a1f1f", color: "#f87171" }}>
              Stop Run
            </button>
          )}
        </div>
      </header>

      <div className="grid">
        <div>
          <div className="panel">
            <h2>Live Video</h2>
            {runId ? (
              <div>
                <img
                  ref={videoRef}
                  className="stream"
                  src={withApiKeyParam(`${API}/api/video/${runId}/stream`)}
                  alt="live feed"
                />
                <ColorCalibrationControl runId={runId} videoRef={videoRef} />
                <ObjectCaptureButton runId={runId} />
              </div>
            ) : (
              <div style={{ color: "#8a93a8", fontSize: 13 }}>Start a run to view the live feed.</div>
            )}
          </div>

          <div className="panel">
            <h2>Timeline / Alerts</h2>
            {alerts.length === 0 && <div style={{ color: "#8a93a8", fontSize: 13 }}>No alerts yet.</div>}
            {alerts.map((a, i) => (
              <div className="alert-item" key={i}>
                <strong>{a.type || a.alert_type}</strong> — step {a.step_id} · t={Number(a.ts).toFixed(1)}s
              </div>
            ))}
          </div>

          <div className="panel">
            <h2>Object States</h2>
            <ObjectStatesPanel procedure={procedure} />
          </div>
        </div>

        <div>
          <div className="panel">
            <h2>Procedure Status</h2>
            <StepChecklist procedure={procedure} allSteps={experimentSteps} />
          </div>

          <div className="panel">
            <h2>Current State</h2>
            <CurrentState procedure={procedure} perception={perception} />
          </div>

          {runId && (
            <div className="panel">
              <h2>Dataset Labeling</h2>
              <LabelingPanel runId={runId} perception={perception} />
            </div>
          )}

          <div className="panel">
            <h2>System Monitoring</h2>
            <div className="metric-row"><span>CPU</span><span>{metrics ? `${metrics.cpu_percent}%` : "—"}</span></div>
            <div className="metric-row"><span>RAM</span><span>{metrics ? `${metrics.ram_percent}%` : "—"}</span></div>
            <div className="metric-row"><span>Frames processed</span><span>{metrics ? metrics.frame_count : "—"}</span></div>
            <div className="metric-row"><span>Camera</span><span>{metrics ? (metrics.camera_open ? "OK" : "Unavailable") : "—"}</span></div>
            <div className="metric-row"><span>Recording</span><span>{runId ? "Active" : "Idle"}</span></div>
          </div>
        </div>
      </div>
    </div>
  );
}

function ObjectCaptureButton({ runId }) {
  const [status, setStatus] = useState(null);

  async function capture() {
    try {
      const res = await fetch(`${API}/api/dataset/${runId}/capture_object_sample`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify({}),
      });
      const data = await res.json();
      setStatus(res.ok ? `Captured ${data.sample.image} (${data.sample.boxes.length} box(es))` : `Error: ${data.detail}`);
    } catch {
      setStatus("Capture failed.");
    }
  }

  return (
    <div style={{ marginTop: 8, fontSize: 13 }}>
      <button onClick={capture} style={btnStyle}>Capture frame for object-detection training</button>
      {status && <span style={{ marginLeft: 8, color: "#8a93a8" }}>{status}</span>}
    </div>
  );
}

function ColorCalibrationControl({ runId, videoRef }) {
  const [active, setActive] = useState(false);
  const [colorName, setColorName] = useState("");
  const [status, setStatus] = useState(null);

  useEffect(() => {
    const img = videoRef.current;
    if (!img || !active) return;

    async function handleImageClick(e) {
      const rect = img.getBoundingClientRect();
      const fx = (e.clientX - rect.left) / rect.width;
      const fy = (e.clientY - rect.top) / rect.height;
      try {
        const res = await fetch(`${API}/api/calibration/${runId}/sample`, {
          method: "POST",
          headers: { "Content-Type": "application/json", ...authHeaders() },
          body: JSON.stringify({ color_name: colorName.trim(), fx, fy }),
        });
        const data = await res.json();
        setStatus(res.ok
          ? `Calibrated "${colorName}" — takes effect on next run start.`
          : `Error: ${data.detail || "calibration failed"}`);
      } catch {
        setStatus("Calibration request failed.");
      }
      setActive(false);
    }

    img.style.cursor = "crosshair";
    img.addEventListener("click", handleImageClick);
    return () => {
      img.style.cursor = "default";
      img.removeEventListener("click", handleImageClick);
    };
  }, [active, colorName, runId, videoRef]);

  return (
    <div style={{ marginTop: 8, fontSize: 13 }}>
      <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
        <input
          type="text"
          value={colorName}
          onChange={(e) => setColorName(e.target.value)}
          placeholder="Color name (e.g. red)"
          style={{ background: "#131722", color: "#e6e6e6", border: "1px solid #232838", borderRadius: 6, padding: "4px 8px", width: 140 }}
        />
        <button
          onClick={() => setActive((a) => !a)}
          disabled={!colorName.trim()}
          style={{ ...btnStyle, background: active ? "#3a2f10" : undefined, color: active ? "#facc15" : undefined }}
        >
          {active ? "Click the box in the video… (or click here to cancel)" : "Calibrate color"}
        </button>
      </div>
      {status && <div style={{ marginTop: 6, color: "#8a93a8" }}>{status}</div>}
    </div>
  );
}

function ObjectStatesPanel({ procedure }) {
  const states = procedure?.object_states || {};
  const entries = Object.entries(states);
  if (entries.length === 0) {
    return <div style={{ color: "#8a93a8", fontSize: 13 }}>No tracked objects yet.</div>;
  }
  return (
    <div style={{ fontSize: 13 }}>
      {entries.map(([objId, info]) => (
        <div className="metric-row" key={objId}>
          <span>{objId}</span>
          <span className={`badge ${info.state === "HELD" ? "current" : "pending"}`}>{info.state}</span>
        </div>
      ))}
    </div>
  );
}

function StepChecklist({ procedure, allSteps }) {
  const completed = new Set(procedure?.completed_steps || []);
  const skipped = new Set(procedure?.skipped_steps || []);
  const current = procedure?.current_step_id;

  // Full expected sequence when we have it (fetched from
  // /api/experiments/{name}); falls back to only-discovered-so-far if the
  // experiment definition hasn't loaded yet, rather than showing nothing.
  const steps = allSteps && allSteps.length > 0
    ? allSteps
    : [...completed, ...skipped, current].filter((s) => s && s !== "TERMINAL").map((id) => ({ id, name: id }));

  if (steps.length === 0) {
    return <div style={{ color: "#8a93a8", fontSize: 13 }}>No experiment selected.</div>;
  }

  return (
    <div>
      {steps.map((step) => {
        const stepId = step.id;
        const isDone = completed.has(stepId);
        const isSkipped = skipped.has(stepId);
        const isCurrent = stepId === current;
        const badge = isDone ? "ok" : isSkipped ? "skipped" : isCurrent ? "current" : "pending";
        const label = isDone ? "✓" : isSkipped ? "⤳" : isCurrent ? "…" : "○";
        return (
          <div className="step-row" key={stepId}>
            <span className={`badge ${badge}`}>{label}</span>
            <span>{step.name || stepId}{isSkipped ? " (skipped — optional)" : ""}</span>
          </div>
        );
      })}
      {procedure && procedure.status === "COMPLETED" && (
        <div style={{ marginTop: 10 }}><span className="badge ok">✓ Experiment complete</span></div>
      )}
    </div>
  );
}

const LABEL_OPTIONS = ["PICK", "HOLD", "MOVE", "PLACE", "REACH", "IDLE"];

function LabelingPanel({ runId, perception }) {
  const [lastRecorded, setLastRecorded] = useState(null);
  const suggested = perception?.activity?.label;

  async function submitLabel(label) {
    try {
      await fetch(`${API}/api/dataset/${runId}/label`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label }),
      });
      setLastRecorded(label);
    } catch {
      // best-effort — dataset labeling failing shouldn't disrupt the run
    }
  }

  return (
    <div style={{ fontSize: 13 }}>
      <div style={{ color: "#8a93a8", marginBottom: 8 }}>
        System suggests: <strong>{suggested ?? "—"}</strong>. Tap what's actually happening to
        record a training example.
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
        {LABEL_OPTIONS.map((label) => (
          <button
            key={label}
            onClick={() => submitLabel(label)}
            style={{
              padding: "6px 12px", borderRadius: 6, fontSize: 12, cursor: "pointer",
              border: label === suggested ? "1px solid #4ade80" : "1px solid #2a2f3f",
              background: lastRecorded === label ? "#1f2b1f" : "#151823",
              color: "#e2e6ee",
            }}
          >
            {label}
          </button>
        ))}
      </div>
      {lastRecorded && (
        <div style={{ marginTop: 8, color: "#4ade80" }}>Recorded "{lastRecorded}" as a training example.</div>
      )}
    </div>
  );
}

function CurrentState({ procedure, perception }) {
  if (!procedure) return <div style={{ color: "#8a93a8", fontSize: 13 }}>No active run.</div>;
  const next = procedure.next_action || {};
  const interactions = perception?.interaction?.interactions || [];
  return (
    <div style={{ fontSize: 13 }}>
      <div className="metric-row"><span>Status</span><span>{procedure.status}</span></div>
      <div className="metric-row"><span>Current step</span><span>{procedure.current_step_id}</span></div>
      <div className="metric-row"><span>Observed activity</span><span>{perception?.activity?.label ?? "—"}</span></div>
      <div className="metric-row"><span>Confidence</span><span>{perception ? Number(perception.activity_confidence).toFixed(2) : "—"}</span></div>
      <div className="metric-row"><span>Next expected</span><span>{next.next_step ?? "—"}</span></div>
      {perception?.rack_reference && (
        <div className="metric-row">
          <span>Rack reference</span>
          <span className={`badge ${perception.rack_reference.marker?.visible ? "ok" : "warn"}`}>
            {perception.rack_reference.marker?.visible ? "Locked" : "Marker not visible"}
          </span>
        </div>
      )}
      {next.voice_instruction && (
        <div style={{ marginTop: 8, color: "#facc15" }}>"{next.voice_instruction}"</div>
      )}
      {interactions.length > 0 && (
        <div style={{ marginTop: 10, borderTop: "1px solid #232838", paddingTop: 8 }}>
          {interactions.map((i, idx) => (
            <div key={idx} style={{ marginBottom: 4 }}>
              <div className="metric-row">
                <span>{i.hand.replace("_", " ")}</span>
                <span><span className="badge current">{i.state}</span> {i.object_class ?? ""}</span>
              </div>
              <div className="metric-row" style={{ color: "#6b7386", fontSize: 11 }}>
                <span>distance / contact threshold</span>
                <span>{i.distance_px}px / {i.contact_threshold_px}px</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

const btnStyle = {
  background: "#1d2636",
  color: "#e6e6e6",
  border: "1px solid #2c3547",
  borderRadius: 6,
  padding: "6px 14px",
  cursor: "pointer",
  fontSize: 13,
};
