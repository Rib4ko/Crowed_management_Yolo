import cv2
import numpy as np  # <-- ADDED: Needed for handling polygon coordinates
from ultralytics import YOLO
import random

# --- Configuration ---
model = YOLO('yolov8n.pt')
VIDEO_PATH = "cctv.mp4"
PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.5

# --- DAY 2, STEP 1: Define the Risk Zone (ROI) ---
#
# !! IMPORTANT !!
# You MUST adjust these (x, y) coordinates to fit your 'test.mp4' video
# I've put in some example coordinates for a zone on the right side.
#
roi_vertices = np.array([
    [500, 0],  # Top-left corner of the zone
    [800, 0],  # Top-right corner
    [900, 1000],  # Bottom-right corner
    [400, 1000]  # Bottom-left corner
], np.int32)
# --------------------------------------------------

track_colors = {}

# --- Video Processing ---
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: Could not open video file '{VIDEO_PATH}'.")
    exit()

print("Day 2, Step 1: Drawing the ROI. Press 'ESC' to exit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video stream.")
        break

    # --- DAY 2, STEP 1: Draw the Risk Zone on the Frame ---
    # We draw the polygon on every frame
    cv2.polylines(
        frame,
        [roi_vertices],  # The coordinates
        isClosed=True,  # Connect the last point to the first
        color=(255, 0, 0),  # Blue color (BGR)
        thickness=2
    )


    # ------------------------------------------------------

    # 1. Run YOLOv8 Tracking (Same as Day 1)
    results = model.track(
        source=frame,
        persist=True,
        classes=[PERSON_CLASS_ID],
        conf=CONFIDENCE_THRESHOLD,
        tracker="bytetrack.yaml",
        verbose=False
    )

    person_count_total = 0

    # 2. Process Tracking Results (Same as Day 1)
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        person_count_total = len(track_ids)

        # 3. Draw Bounding Boxes and Track IDs (Same as Day 1)
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)

            if track_id not in track_colors:
                track_colors[track_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            color = track_colors[track_id]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"ID: {track_id}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1 - 10), color, -1)
            cv2.putText(frame, label, (x1, y1 - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 4. Display the total count (Same as Day 1)
    count_text = f"Total People: {person_count_total}"
    cv2.putText(frame, count_text, (20, 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Crowd Management MVP - Step 1: ROI", frame)

    # Break loop on 'ESC' key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("Step 1 complete. The ROI is now being drawn.")