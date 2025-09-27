#!/usr/bin/env python3
"""
sanchalan_ai_ml.py

Unified script for SanchalanAI:
 - Lane drawing and loading (lanes.json)
 - YOLOv8 detection (ambulance, vehicles, pedestrians)
 - State building for RL agent
 - Inference pipeline (image/video)
 - RL training (placeholder hooks)
"""

import os
import cv2
import json
import argparse
import numpy as np
from ultralytics import YOLO

# ==========================
# Config
# ==========================
DEFAULT_YOLO_WEIGHTS = r"D:\25050\backend\runs\detect\train3\weights\best.pt"
LANES_JSON = "lanes.json"

# ==========================
# Lane Drawing / Loading
# ==========================
def draw_lanes(image_path, output_json=LANES_JSON):
    """Interactive tool to draw lanes and save to JSON."""
    lanes, current_points = {}, []
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load {image_path}")
    clone = img.copy()

    def click_event(event, x, y, flags, param):
        nonlocal current_points, clone
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append([x, y])
            cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
            if len(current_points) > 1:
                cv2.line(clone, tuple(current_points[-2]), tuple(current_points[-1]), (0, 0, 255), 2)

    cv2.namedWindow("Lane Drawer")
    cv2.setMouseCallback("Lane Drawer", click_event)

    print("Instructions:")
    print(" - Click to add points for current lane polygon.")
    print(" - Press 'n' to name and save current polygon as a lane.")
    print(" - Press 's' to save all lanes into lanes.json.")
    print(" - Press 'q' to quit without saving.")

    while True:
        cv2.imshow("Lane Drawer", clone)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("n"):
            if current_points:
                lane_name = input("Enter lane name (e.g. north, south, east, west, center): ")
                lanes[lane_name] = current_points
                current_points = []
                clone = img.copy()

                # redraw saved lanes
                for name, pts in lanes.items():
                    poly = cv2.polylines(clone, [np.array(pts, dtype=np.int32)], True, (255, 0, 0), 2)
                    cx, cy = np.mean(pts, axis=0).astype(int)
                    cv2.putText(clone, name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        elif key == ord("s"):
            with open(output_json, "w") as f:
                json.dump(lanes, f, indent=2)
            print(f"Lanes saved to {output_json}")
            break

        elif key == ord("q"):
            print("Quitting without saving.")
            break

    cv2.destroyAllWindows()


def load_lanes(json_path=LANES_JSON):
    """Load lane polygons from JSON file."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"{json_path} not found. Run draw_lanes first.")
    with open(json_path, "r") as f:
        return json.load(f)

# ==========================
# YOLO Detector
# ==========================
class YOLODetector:
    def __init__(self, model_path=DEFAULT_YOLO_WEIGHTS, conf=0.35):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, image):
        results = self.model(image, conf=self.conf)[0]
        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            label = self.model.names[cls]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append((label, conf, (x1, y1, x2, y2)))
        return detections

# ==========================
# Lane Assignment + State Builder
# ==========================
def assign_lane(detection, lanes):
    """Assigns detected object to a lane polygon (if inside)."""
    x1, y1, x2, y2 = detection[2]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    point = (cx, cy)
    for name, pts in lanes.items():
        if cv2.pointPolygonTest(np.array(pts, dtype=np.int32), point, False) >= 0:
            return name
    return None


def build_state(detections, lanes):
    """Builds state as count of detections per lane per class."""
    state = {lane: {} for lane in lanes.keys()}
    for det in detections:
        label, _, _ = det
        lane = assign_lane(det, lanes)
        if lane:
            state[lane][label] = state[lane].get(label, 0) + 1
    return state

# ==========================
# RL Agent (placeholder)
# ==========================
class RlAgent:
    def __init__(self, num_lanes):
        self.num_lanes = num_lanes

    def choose_action(self, state):
        lane_counts = {lane: sum(counts.values()) for lane, counts in state.items()}
        if not lane_counts:  # ✅ avoid empty error
            return "no_action"
        return max(lane_counts, key=lane_counts.get)

# ==========================
# Pipelines
# ==========================
def detect_demo(image_path, model_path=DEFAULT_YOLO_WEIGHTS):
    """Quick YOLO detection demo on single image."""
    detector = YOLODetector(model_path)
    image = cv2.imread(image_path)
    detections = detector.detect(image)

    for label, conf, (x1, y1, x2, y2) in detections:
        if "ambulance" in label.lower():
            color = (0, 255, 0)  # Green
        elif "person" in label.lower():
            color = (255, 0, 0)  # Blue
        else:
            color = (0, 255, 255)  # Yellow
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Detections", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def pipeline_video(video_path, model_path=DEFAULT_YOLO_WEIGHTS, lanes_path=LANES_JSON):
    """Run detection + RL pipeline on video."""
    lanes = load_lanes(lanes_path)
    detector = YOLODetector(model_path)
    agent = RlAgent(len(lanes))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        state = build_state(detections, lanes)
        action = agent.choose_action(state)

        # Draw detections
        for label, conf, (x1, y1, x2, y2) in detections:
            if "ambulance" in label.lower():
                color = (0, 255, 0)
            elif "person" in label.lower():
                color = (255, 0, 0)
            else:
                color = (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw lane polygons
        for name, pts in lanes.items():
            cv2.polylines(frame, [np.array(pts, dtype=np.int32)], True, (0, 255, 255), 2)
            cx, cy = np.mean(pts, axis=0).astype(int)
            cv2.putText(frame, name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(frame, f"Action: Open {action}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Video Pipeline", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ==========================
# Main CLI
# ==========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["draw_lanes", "detect_image", "pipeline_video"], required=True)
    parser.add_argument("--input", help="Image or video path")
    parser.add_argument("--weights", default=DEFAULT_YOLO_WEIGHTS)
    parser.add_argument("--lanes", default=LANES_JSON)
    args = parser.parse_args()

    if args.mode == "draw_lanes":
        draw_lanes(args.input, args.lanes)
    elif args.mode == "detect_image":
        detect_demo(args.input, args.weights)
    elif args.mode == "pipeline_video":
        pipeline_video(args.input, args.weights, args.lanes)
