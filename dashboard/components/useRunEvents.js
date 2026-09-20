import { useEffect, useRef, useState } from "react";
import { authHeaders, withApiKeyParam } from "./apiKey";

/**
 * Subscribes to /ws/runs/{runId} and reduces incoming typed events into a
 * single state object each panel reads a slice of. On reconnect, resyncs
 * current procedure state via REST rather than trusting the socket alone
 * (Section 18 of the design doc: dropped connections never desync the UI).
 */
export function useRunEvents(runId) {
  const [perception, setPerception] = useState(null);
  const [procedure, setProcedure] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef(null);

  useEffect(() => {
    if (!runId) return;

    const apiBase = process.env.NEXT_PUBLIC_API_BASE;
    const wsBase = process.env.NEXT_PUBLIC_WS_BASE;

    // Resync current state via REST first
    fetch(`${apiBase}/api/procedure/${runId}`, { headers: authHeaders() })
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => data && setProcedure(data))
      .catch(() => {});

    fetch(`${apiBase}/api/alerts/${runId}`, { headers: authHeaders() })
      .then((r) => (r.ok ? r.json() : []))
      .then((data) => setAlerts(data || []))
      .catch(() => {});

    // WebSocket can't set custom headers from the browser, so the key
    // (if any) goes as a query param instead — see backend/ws/routes_ws.py.
    const ws = new WebSocket(withApiKeyParam(`${wsBase}/ws/runs/${runId}`));
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);

    ws.onmessage = (msg) => {
      let event;
      try {
        event = JSON.parse(msg.data);
      } catch {
        return;
      }
      switch (event.type) {
        case "perception.update":
          setPerception(event.payload);
          break;
        case "procedure.state":
          setProcedure(event.payload);
          break;
        case "alert":
          setAlerts((prev) => [event.payload, ...prev].slice(0, 50));
          break;
        case "system.metrics":
          setMetrics(event.payload);
          break;
        default:
          break;
      }
    };

    return () => ws.close();
  }, [runId]);

  return { perception, procedure, alerts, metrics, connected };
}
