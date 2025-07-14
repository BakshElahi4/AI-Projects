from ultralytics import YOLO
import cv2
import numpy as np

MODEL_PATH      = 'yolov8n.pt'
VIDEO_IN        = 'input_video2.mp4'
VIDEO_OUT       = 'output_counted.mp4'
CONF_THRES      = 0.3
TARGET_CLASS_ID = 2  # Class ID for car in COCO

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    raise FileNotFoundError(f"❌ Cannot open {VIDEO_IN}")

w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps= cap.get(cv2.CAP_PROP_FPS) or 30

out = cv2.VideoWriter(VIDEO_OUT,
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps,
                      (w, h))

# ✅ Horizontal line (center of the frame)
line_y = h // 2 + 400

# Counts
up_count = 0
down_count = 0

# Store previous y-centroids
centroids_prev = {}
counted_up = set()
counted_down = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model.track(frame, persist=True, verbose=False, conf=CONF_THRES)[0]

    # ✅ Draw horizontal line
    cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 2)

    for box in result.boxes:
        cls_id = int(box.cls[0])
        tid = int(box.id[0]) if box.id is not None else None
        conf = float(box.conf[0])

        if cls_id != TARGET_CLASS_ID or tid is None:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        # ✅ Draw box, center, and ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 100, 255), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
        cv2.putText(frame, f"ID {tid}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)

        # ✅ Get previous y and compare to detect crossing
        prev_y = centroids_prev.get(tid)
        if prev_y is not None:
            if prev_y < line_y <= cy and tid not in counted_down:
                down_count += 1
                counted_down.add(tid)
            elif prev_y > line_y >= cy and tid not in counted_up:
                up_count += 1
                counted_up.add(tid)

        centroids_prev[tid] = cy

    # ✅ Show count on screen
    cv2.putText(frame, f"↓ Downward: {down_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
    cv2.putText(frame, f"↑ Upward: {up_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)

    out.write(frame)

    # ✅ Resize only for display (fixes zoom issue)
    display_frame = cv2.resize(frame, (1280, 720))
    cv2.imshow("Top-View Car Counter (q to quit)", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Done! Cars ↓ Downward: {down_count}, ↑ Upward: {up_count}")
print(f"🎥 Saved to: {VIDEO_OUT}")
