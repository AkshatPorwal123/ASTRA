"""
LAN/IP monitoring stream. Kept independent from the AI-resolution frames
used for inference (Section 15) — this encodes a possibly-downscaled copy
for authorized LAN viewers only, never leaves the local network by default.
"""
from __future__ import annotations
import cv2


def encode_jpeg(frame, max_width: int = 640, quality: int = 70) -> bytes | None:
    if frame is None:
        return None
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


def mjpeg_multipart_frame(jpeg_bytes: bytes) -> bytes:
    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
    )
