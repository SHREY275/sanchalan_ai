import cv2
import numpy as np
import argparse
import logging
import json
from stable_baselines3 import DQN
from ultralytics import YOLO

# ------------------------------------------
# Logging
# ------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pipeline_video")

# ------------------------------------------
# Constants
# ------------------------------------------
DEFAULT_YOLO_WEIGHTS = "yolov8n.pt"
RL_MODEL_PATH = "traffic_dqn"
LANE_FILE = "lanes.json"
MAX_AMBULANCE_FRAMES = 10  # persist ambulance override for N frames

# ------------------------------------------
# Lane Config Loader
# ------------------------------------------
def load_lane_config(filepath=LANE_FILE):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Failed loading lane config: %s", e)
        return {}

def assign_lane_polygon(cx: int, cy: int, lanes_config: dict) -> str:
    point = (cx, cy)
    for lane_name, polygon_coords in lanes_config.items():
        polygon = np.array(polygon_coords, dtype=np.int32)
        if cv2.pointPolygonTest(polygon, point, False) >= 0:
            return lane_name
    return "unknown"

# ------------------------------------------
# YOLO Detector Wrapper
# ------------------------------------------
class YOLODetector:
    def __init__(self, weights=DEFAULT_YOLO_WEIGHTS, device="cpu"):
        logger.info("Loading YOLO model: %s on %s", weights, device)
        self.model = YOLO(weights)

    def detect(self, frame, conf: float = 0.1, imgsz: int = 1024):
        results = self.model(frame, imgsz=imgsz, conf=conf)
        dets = []
        for r in results:
            for box in r.boxes:
                dets.append({
                    "xmin": int(box.xyxy[0][0]),
                    "ymin": int(box.xyxy[0][1]),
                    "xmax": int(box.xyxy[0][2]),
                    "ymax": int(box.xyxy[0][3]),
                    "conf": float(box.conf[0]),
                    "cls_id": int(box.cls[0]),
                    "label": self.model.names[int(box.cls[0])]
                })
        return dets

# ------------------------------------------
# Pipeline Video with Persistent Ambulance Detection
# ------------------------------------------
def pipeline_video(source=0,
                   yolo_weights=DEFAULT_YOLO_WEIGHTS,
                   rl_model_path=RL_MODEL_PATH,
                   save_path="pipeline_out.avi"):

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(save_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    detector = YOLODetector(yolo_weights)
    lanes_config = load_lane_config(LANE_FILE)

    try:
        agent = DQN.load(rl_model_path)
    except Exception as e:
        logger.warning("Could not load RL model (%s), using dummy action", e)
        agent = None

    ambulance_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect vehicles
        dets = detector.detect(frame, conf=0.1, imgsz=1024)
        lane_features = {lane: {"car":0,"bus":0,"bike":0,"ambulance":0} for lane in lanes_config}

        ambulance_detected = False
        for d in dets:
            cx, cy = (d["xmin"]+d["xmax"])//2, (d["ymin"]+d["ymax"])//2
            lane = assign_lane_polygon(cx, cy, lanes_config)
            if lane in lane_features and d["label"] in lane_features[lane]:
                lane_features[lane][d["label"]] += 1

            if d["label"] == "ambulance":
                ambulance_detected = True

            # Draw boxes & labels
            cv2.rectangle(frame, (d["xmin"], d["ymin"]), (d["xmax"], d["ymax"]), (0,255,0), 2)
            cv2.putText(frame, f"{d['label']} ({lane})", (d["xmin"], max(d["ymin"]-6,0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # Draw lane polygons
        for lane, polygon in lanes_config.items():
            poly = np.array(polygon, dtype=np.int32)
            cv2.polylines(frame, [poly], True, (255,0,0), 2)
            cx, cy = np.mean(poly, axis=0).astype(int)
            cv2.putText(frame, lane, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        # Update ambulance frame counter
        if ambulance_detected:
            ambulance_counter = MAX_AMBULANCE_FRAMES
        elif ambulance_counter > 0:
            ambulance_counter -= 1

        # Build RL state
        state = np.array([
            lane_features["north"]["car"] + lane_features["north"]["bus"] + lane_features["north"]["bike"],
            lane_features["south"]["car"] + lane_features["south"]["bus"] + lane_features["south"]["bike"],
            lane_features["east"]["car"] + lane_features["east"]["bus"] + lane_features["east"]["bike"],
            lane_features["west"]["car"] + lane_features["west"]["bus"] + lane_features["west"]["bike"],
            1.0 if ambulance_counter > 0 else 0.0
        ], dtype=np.float32)

        # RL agent decision
        if agent:
            action, _ = agent.predict(state, deterministic=True)
        else:
            action = 0

        # Decision overlay
        if state[-1] == 1.0:
            decision_text = "OVERRIDE: GREEN for AMBULANCE"
        else:
            decision_text = f"AUTO: Green -> {['North','South','East','West'][action]}"

        cv2.putText(frame, decision_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow("SanchalanAI - Live", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    logger.info("✅ Video pipeline saved to %s", save_path)


# ------------------------------------------
# CLI
# ------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Video Pipeline for SanchalanAI")
    parser.add_argument("--video", type=str, help="Path to video file (default: webcam)")
    args = parser.parse_args()

    source = 0 if not args.video else args.video
    pipeline_video(source)
