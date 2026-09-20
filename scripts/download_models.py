"""
Downloads model files that aren't bundled in the repo (either too large,
or hosted somewhere only reachable from your own machine's network).

Run this once, from the astra/ project root, with your venv active:

    python scripts/download_models.py

Safe to re-run — skips files that already exist.
"""
import os
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

FILES = [
    {
        "name": "pose_landmarker_lite.task",
        "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        "used_by": "Milestone 3 — RealPoseEstimator (2D body pose)",
        "approx_size": "~5 MB",
    },
]


def download(name: str, url: str) -> bool:
    dest = os.path.join(MODELS_DIR, name)
    if os.path.exists(dest):
        print(f"  [skip] {name} already present")
        return True
    print(f"  [downloading] {name} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        size_kb = os.path.getsize(dest) / 1024
        print(f"  [done] {name} ({size_kb:.0f} KB)")
        return True
    except Exception as e:
        print(f"  [FAILED] {name}: {e}")
        if os.path.exists(dest):
            os.remove(dest)  # don't leave a partial/corrupt file behind
        return False


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"Model directory: {MODELS_DIR}\n")
    all_ok = True
    for f in FILES:
        print(f"{f['name']} — {f['used_by']} ({f['approx_size']})")
        ok = download(f["name"], f["url"])
        all_ok = all_ok and ok
        print()

    if all_ok:
        print("All model files present. Real pose estimation is now active —")
        print("restart the backend (uvicorn) if it's already running.")
    else:
        print("Some downloads failed. Check your internet connection and rerun this script.")
        print("Affected features will report status=PLACEHOLDER until the file is present —")
        print("the system still runs fine without them, just without that capability.")


if __name__ == "__main__":
    main()
