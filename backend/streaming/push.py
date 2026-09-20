"""
Pushes the MJPEG stream to a configured destination IP:port via HTTP POST
— the literal "Stream the video of the experiment to specific IP"
requirement in the official problem statement. Distinct from the existing
pull-based /api/video/{run_id}/stream (which the dashboard reads from) —
this is for pushing out to an external monitoring station's own address,
which is what "to specific IP" actually asks for.

Runs its own background thread with a maxsize=1, drop-oldest queue, so a
slow/unreachable destination degrades to dropped frames rather than
blocking or crashing the main capture/inference loop (Section 27's
failure-handling principle, applied here too). A connection failure logs
once, not once per frame, and retries silently after that — an
unreachable monitoring station is a real, expected condition on a
station's local network, not something that should spam the log or take
down the run.
"""
from __future__ import annotations
import http.client
import logging
import queue
import threading

logger = logging.getLogger("astra.stream_push")


class StreamPusher:
    def __init__(self, dest_ip: str, dest_port: int, run_id: str = "", path: str = "/astra/frame"):
        self.dest_ip = dest_ip
        self.dest_port = dest_port
        self.run_id = run_id
        self.path = path
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._conn: http.client.HTTPConnection | None = None
        self._error_logged = False
        self.frames_pushed = 0
        self.last_error: str | None = None

    def start(self):
        self._thread.start()

    def push(self, jpeg_bytes: bytes):
        if jpeg_bytes is None:
            return
        try:
            self._queue.get_nowait()   # drop the previous unpushed frame — streaming wants latest, not a backlog
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(jpeg_bytes)
        except queue.Full:
            pass

    def _worker(self):
        while not self._stop.is_set():
            try:
                jpeg_bytes = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if self._conn is None:
                    self._conn = http.client.HTTPConnection(self.dest_ip, self.dest_port, timeout=2)
                self._conn.request(
                    "POST", self.path, body=jpeg_bytes,
                    headers={"Content-Type": "image/jpeg", "X-ASTRA-Run-Id": self.run_id},
                )
                resp = self._conn.getresponse()
                resp.read()
                self.frames_pushed += 1
                self._error_logged = False
                self.last_error = None
            except Exception as e:
                self._conn = None
                self.last_error = str(e)
                if not self._error_logged:
                    logger.warning(
                        f"[STREAM_PUSH] Could not reach {self.dest_ip}:{self.dest_port} — {e}. "
                        f"Will keep retrying quietly (this message won't repeat every frame)."
                    )
                    self._error_logged = True

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass


def parse_destination(stream_to: str) -> tuple[str, int]:
    """'192.168.1.50:9000' -> ('192.168.1.50', 9000). Raises ValueError with
    a clear message on malformed input, rather than a confusing traceback
    deep in a background thread."""
    if ":" not in stream_to:
        raise ValueError(f"stream_to must be 'ip:port', got: {stream_to!r}")
    ip, _, port_str = stream_to.rpartition(":")
    try:
        port = int(port_str)
    except ValueError:
        raise ValueError(f"stream_to port must be an integer, got: {port_str!r}")
    return ip, port
