#!/usr/bin/env python3
"""
lane_drawer.py

Interactive tool to create lane polygons for SanchalanAI.
 - Click points on the image to define a polygon.
 - Press 'n' to start a new lane.
 - Press 's' to save all lanes into a JSON file.
 - Press 'q' to quit without saving.
"""

import cv2
import json
import argparse
import numpy as np

lanes = {}
current_points = []
image = None
clone = None

# ----------------------------
# Mouse click callback
# ----------------------------
def click_event(event, x, y, flags, param):
    global current_points, clone
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append([x, y])
        # Draw point
        cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
        # Draw line from previous point
        if len(current_points) > 1:
            cv2.line(clone, tuple(current_points[-2]), tuple(current_points[-1]), (0, 0, 255), 2)

# ----------------------------
# Draw saved lanes
# ----------------------------
def redraw_lanes():
    global clone
    clone = image.copy()
    for name, pts in lanes.items():
        poly = cv2.polylines(clone, [np.array(pts, dtype=np.int32)], True, (255, 0, 0), 2)
        cx, cy = np.mean(pts, axis=0).astype(int)
        cv2.putText(clone, name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    # Draw current polygon
    for i in range(1, len(current_points)):
        cv2.line(clone, tuple(current_points[i-1]), tuple(current_points[i]), (0, 0, 255), 2)
    for pt in current_points:
        cv2.circle(clone, tuple(pt), 5, (0, 0, 255), -1)
    # Instructions
    cv2.putText(clone, "'n' to save lane, 's' to save JSON, 'q' to quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Lane Drawer for SanchalanAI")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="lanes.json", help="Output JSON file for lanes")
    args = parser.parse_args()

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not load {args.image}")
    clone = image.copy()

    cv2.namedWindow("Lane Drawer")
    cv2.setMouseCallback("Lane Drawer", click_event)

    print("Instructions:")
    print(" - Click to add points for current lane polygon.")
    print(" - Press 'n' to name and save current polygon as a lane.")
    print(" - Press 's' to save all lanes into JSON.")
    print(" - Press 'q' to quit without saving.")

    while True:
        redraw_lanes()
        cv2.imshow("Lane Drawer", clone)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("n"):  # finish current lane
            if current_points:
                lane_name = input("Enter lane name (e.g., north, south, east, west): ")
                lanes[lane_name] = current_points.copy()
                current_points = []
                print(f"Saved lane '{lane_name}' with {len(lanes[lane_name])} points.")

        elif key == ord("s"):  # save lanes to JSON
            with open(args.output, "w") as f:
                json.dump(lanes, f, indent=2)
            print(f"Lanes saved to {args.output}")
            break

        elif key == ord("q"):  # quit
            print("Quitting without saving.")
            break

    cv2.destroyAllWindows()
