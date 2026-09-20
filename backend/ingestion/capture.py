"""
Video ingestion: webcam or uploaded-file capture, wrapped so the rest of
the system doesn't care which source is active. Frame Synchronization for
multi-camera setups is a documented extension point (SOURCES dict below)
but Phase 1 targets a single source, per the incremental-build principle.
"""
from __future__ import annotations
import platform
import cv2
import time


class VideoSource:
    def __init__(self, source: str | int):
        """
        source: 0 (default webcam index), another int for a different camera,
        or a filesystem path to a video file for offline/demo playback.
        """
        self.source = source
        self._cap: cv2.VideoCapture | None = None
        self._consecutive_failures = 0

    def open(self) -> bool:
        # Integer sources are camera indices; string sources are file paths.
        # On Windows, the default backend (MSMF) frequently stalls after a
        # few frames — DirectShow (CAP_DSHOW) is far more reliable there.
        if isinstance(self.source, int) and platform.system() == "Windows":
            self._cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(self.source)
        return self._cap.isOpened()

    def read(self):
        if self._cap is None:
            raise RuntimeError("VideoSource not opened — call open() first")
        ok, frame = self._cap.read()
        if ok:
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1
            # A camera that stalls on Windows sometimes recovers if reopened
            # rather than retried in place — reopen after a short run of
            # failed reads instead of freezing on the last good frame forever.
            if isinstance(self.source, int) and self._consecutive_failures >= 20:
                self.release()
                self.open()
                self._consecutive_failures = 0
        return ok, frame, time.time()

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()
