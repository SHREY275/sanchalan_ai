import cv2
from ultralytics import YOLO
import os
import argparse

def run_inference(weights, input_video, output_dir):
    # Load YOLO model
    model = YOLO(weights)

    # Prepare output path
    os.makedirs(output_dir, exist_ok=True)
    output_video = os.path.join(output_dir, "output_inference.mp4")

    # Open input video
    cap = cv2.VideoCapture(input_video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    print("🔄 Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = model(frame, conf=0.35)

        # Draw bounding boxes & labels
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}",
                            (x1, max(y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ Output saved at: {output_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to YOLO weights (.pt)")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, default="runs/detect/output", help="Directory to save output")
    args = parser.parse_args()

    run_inference(args.weights, args.video, args.output)
