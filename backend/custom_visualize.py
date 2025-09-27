import cv2
import os
from ultralytics import YOLO

# Load model
model = YOLO("D:/25050/backend/runs/detect/train3/weights/last.pt")
results = model.predict(source="D:/25050/final_dataset/test_images", save=False, conf=0.25)
out_dir = "D:/25050/backend/custom_outputs"

# Define class names (match your dataset yaml order!)
class_names = ["car", "bike", "bus", "person", "ambulance"]

# Colors for bounding boxes
colors = {
    "ambulance": (0, 255, 0),  # Green
    "person": (255, 0, 0),     # Blue
    "default": (0, 0, 255)     # Red for others
}

# Run predictions
results = model.predict(source="D:/25050/final_dataset/test_images", save=False, conf=0.25)

# Output folder
out_dir = "D:/25050/backend/custom_outputs"
os.makedirs(out_dir, exist_ok=True)

for i, result in enumerate(results):
    img = result.orig_img.copy()
    counts = {}

    for box in result.boxes:
        cls_id = int(box.cls[0])
        label = class_names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Choose color
        if label == "ambulance":
            color = colors["ambulance"]
        elif label == "person":
            color = colors["person"]
        else:
            color = colors["default"]

        # Draw box + label
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Count objects
        counts[label] = counts.get(label, 0) + 1

    # Overlay counts
    y_offset = 30
    for label, cnt in counts.items():
        cv2.putText(img, f"{label}: {cnt}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y_offset += 30

    # Save image
    cv2.imwrite(f"{out_dir}/frame_{i}.jpg", img)

print("Custom results saved to:", out_dir)
