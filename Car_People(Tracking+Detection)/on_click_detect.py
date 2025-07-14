from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Use yolov8s.pt for better accuracy if needed

# Target classes: person = 0, car = 2
TARGET_CLASSES = [0, 2]

# Input/output video paths
video_path = 'input_video0.mp4'
output_path = 'output_detected_click.mp4'

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"❌ Cannot open video: {video_path}")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define output video writer
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# Variables to manage selection and tracking
clicked_ids = []
last_boxes = []
use_tracking = False  # Start with detection

# Mouse click handler
def select_object(event, x, y, flags, param):
    global clicked_ids, last_boxes, use_tracking
    if event == cv2.EVENT_LBUTTONDOWN:
        for (tid, class_id, x1, y1, x2, y2) in last_boxes:
            if x1 <= x <= x2 and y1 <= y <= y2:
                if tid not in clicked_ids:
                    clicked_ids.append(tid)
                    print(f"✅ Selected ID: {tid}, class: {model.names[class_id]}")
                    use_tracking = True  # Start tracking from next frame
                break

# Setup window and click callback
cv2.namedWindow("Click-to-Track Detection")
cv2.setMouseCallback("Click-to-Track Detection", select_object)

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    last_boxes.clear()

    if not use_tracking:
        # Stage 1: Detection-only, show all targets for selection
        results = model.predict(frame, conf=0.3, verbose=False)[0]

        # Assign fake IDs to boxes just for selection
        for idx, box in enumerate(results.boxes):
            class_id = int(box.cls[0])
            if class_id in TARGET_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                fake_id = idx + 1  # Unique fake ID

                # Save to clickable boxes list
                last_boxes.append((fake_id, class_id, x1, y1, x2, y2))

                # Draw box (before click)
                color = (0, 255, 0) if class_id == 0 else (0, 0, 255)  # ✅ Red for cars
                label = f"{model.names[class_id]}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # If clicked, skip this detection frame and switch to tracking
        if len(clicked_ids) > 0:
            out.write(frame)  # Save this frame before skip
            display_frame = cv2.resize(frame, (1280, 720))
            cv2.imshow("Click-to-Track Detection", display_frame)
            continue  # Skip drawing boxes this frame (avoid saving all boxes)

    else:
        # Stage 2: Tracking mode - only track selected IDs
        result = model.track(frame, persist=True, verbose=False)[0]

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                tid = int(box.id[0]) if box.id is not None else None
                if tid is None or class_id not in TARGET_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                last_boxes.append((tid, class_id, x1, y1, x2, y2))

                # Draw only selected ID(s)
                if tid in clicked_ids:
                    color = (0, 255, 0) if class_id == 0 else (0, 0, 255)  # ✅ Red for cars
                    label = f"{model.names[class_id]} ID:{tid}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Save and show the frame
    out.write(frame)
    display_frame = cv2.resize(frame, (1280, 720))
    cv2.imshow("Click-to-Track Detection", display_frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Done! Tracked output saved to: {output_path}")
