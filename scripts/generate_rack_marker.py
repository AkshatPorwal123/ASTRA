"""
Generates a printable ArUco marker to affix to the payload rack, for
orientation-agnostic tracking (backend/vision/rack_reference.py) —
addresses the official problem statement's optional "track relative to
the rack, not the floor" challenge.

Usage:
    python scripts/generate_rack_marker.py [--marker-id 0] [--size-px 600] [--output models/rack_marker.png]

Print the output at a KNOWN physical size (e.g. exactly 10cm x 10cm) and
mount it flat on the rack, in view of the camera, oriented however makes
sense for the rack itself — its own top edge becomes "up relative to the
rack" for every joint transform, regardless of how the camera or
astronaut is oriented.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from backend.vision.rack_reference import ARUCO_DICT, DEFAULT_MARKER_ID


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker-id", type=int, default=DEFAULT_MARKER_ID)
    parser.add_argument("--size-px", type=int, default=600)
    parser.add_argument("--output", default="models/rack_marker.png")
    args = parser.parse_args()

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    marker_img = cv2.aruco.generateImageMarker(dictionary, args.marker_id, args.size_px)

    # Add a white border — ArUco detection needs quiet space around the
    # marker to reliably find its edges, printing it edge-to-edge on paper
    # is a common reason detection fails.
    border = args.size_px // 8
    bordered = cv2.copyMakeBorder(marker_img, border, border, border, border,
                                   cv2.BORDER_CONSTANT, value=255)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    cv2.imwrite(args.output, bordered)
    print(f"Saved marker (ID {args.marker_id}) to {args.output}")
    print("Print it at a known physical size and mount it flat on the payload rack, in camera view.")
    print(f"If you use a different --marker-id, also pass marker_id={args.marker_id} when constructing "
          f"RackReferenceTracker in your run — they must match.")


if __name__ == "__main__":
    main()
