"""
Standalone webcam diagnostic — run this directly to check whether OpenCV
can maintain a stable camera connection on this machine, independent of
the ASTRA backend. Prints a line every second showing success/failure
counts for the last second's worth of read attempts.

Run:  python webcam_test.py
Stop: press Ctrl+C, or 'q' in the preview window if it opens.
"""
import cv2
import time
import platform

print(f"OS: {platform.system()} | OpenCV: {cv2.__version__}")

backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
cap = cv2.VideoCapture(0, backend)

if not cap.isOpened():
    print("FAILED to open camera at index 0. Try index 1 (edit this script) "
          "or check if another app has the camera locked.")
    raise SystemExit(1)

print("Camera opened OK. Reading frames for 15 seconds...")
print(f"Reported resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print(f"Reported FPS: {cap.get(cv2.CAP_PROP_FPS)}")

start = time.time()
success = 0
failure = 0
last_report = start

try:
    while time.time() - start < 15:
        ok, frame = cap.read()
        if ok:
            success += 1
        else:
            failure += 1

        now = time.time()
        if now - last_report >= 1.0:
            print(f"[t={now - start:4.1f}s] success={success} failure={failure}")
            success = 0
            failure = 0
            last_report = now
finally:
    cap.release()
    print("Done. Camera released.")
