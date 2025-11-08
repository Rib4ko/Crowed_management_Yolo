import cv2
import numpy as np
from ultralytics import YOLO
import random

# --- Configuration ---
model = YOLO('yolov8n.pt')
VIDEO_PATH = "cctv.mp4"
PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.5

# --- DAY 2, STEP 1: Define the Risk Zone (ROI) ---
# Make sure these coordinates are correct for your video!
roi_vertices = np.array([
    [500, 0],  # Top-left corner of the zone
    [800, 0],  # Top-right corner
    [900, 1000],  # Bottom-right corner
    [400, 1000]  # Bottom-left corner
], np.int32)
# --------------------------------------------------

# --- DAY 2, STEP 3: Define Alert Threshold ---
# This is the "max capacity" for your zone.
# Change this number based on your video and zone size.
maxperzone = 10
# ---------------------------------------------

track_colors = {}

# --- Video Processing ---
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: Could not open video file '{VIDEO_PATH}'.")
    exit()

print("Day 2, Step 3: Running full MVP with Alert System. Press 'ESC' to exit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video stream.")
        break

    # --- DAY 2, STEP 1: Draw the Risk Zone ---
    cv2.polylines(frame, [roi_vertices], isClosed=True, color=(255, 0, 0), thickness=2)

    # --- DAY 2, STEP 2 & 3: Initialize counters and flags ---
    person_count_total = 0
    zone_count = 0
    is_alert = False  # --- DAY 2, STEP 3: Alert flag, reset every frame
    # ----------------------------------------------------

    # 1. Run YOLOv8 Tracking
    results = model.track(
        source=frame,
        persist=True,
        classes=[PERSON_CLASS_ID],
        conf=CONFIDENCE_THRESHOLD,
        tracker="bytetrack.yaml",
        verbose=False
    )

    # 2. Process Tracking Results
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        person_count_total = len(track_ids)

        # 3. Draw Bounding Boxes and Track IDs
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

            # --- DAY 2, STEP 2: Check if person is in the zone ---
            center_x = (x1 + x2) // 2
            center_y = y2

            if cv2.pointPolygonTest(roi_vertices, (center_x, center_y), False) > 0:
                zone_count += 1
            # ----------------------------------------------------

    # 4. Display Counts and Handle Alerts
    # Display total count
    cv2.putText(frame, f"Total People: {person_count_total}", (20, 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)  # White text

    # --- DAY 2, STEP 3: Check alert condition and display metrics ---
    zone_color = (0, 255, 0)  # Default: Green (Safe)

    # Check if the count exceeds the threshold

    if zone_count > maxperzone:
        zone_color = (0, 0, 255)  # Alert: Red
        is_alert = True

    # Display the zone count (e.g., "Zone Count: 12 / 10")
    cv2.putText(frame, f"Zone Count: {zone_count} / {maxperzone}", (20, 90),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, zone_color, 2)

    # If in alert state, draw the big warning message
    if is_alert:
        # Get frame dimensions to position text in the center
        (text_w, text_h), _ = cv2.getTextSize("!! it's Crowded !!", cv2.FONT_HERSHEY_TRIPLEX, 1.5, 3)
        center_x = (frame.shape[1] - text_w) // 2
        center_y = (frame.shape[0] + text_h) // 2

        cv2.putText(frame, "!! it's Crowded !!",

                    (center_x, center_y),
                    cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 255), 3)
    # ---------------------------------------------------

    # Show the frame
    cv2.imshow("Crowd Management MVP - FINAL ALERT SYSTEM", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("MVP logic complete. Ready to present!")
