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

track_colors = {}

# --- Video Processing ---
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: Could not open video file '{VIDEO_PATH}'.")
    exit()

print("Day 2, Step 2: Counting people in ROI. Press 'ESC' to exit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video stream.")
        break

    # --- DAY 2, STEP 1: Draw the Risk Zone (No change here) ---
    cv2.polylines(frame, [roi_vertices], isClosed=True, color=(255, 0, 0), thickness=2)
    # ------------------------------------------------------

    # --- DAY 2, STEP 2: Initialize counters ---
    person_count_total = 0
    zone_count = 0  # New counter for people inside the zone
    # ------------------------------------------

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

        person_count_total = len(track_ids)  # This is the total count

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
            # We calculate the bottom-center point of the bounding box.
            # This point is a good proxy for the person's "location" on the ground.
            center_x = (x1 + x2) // 2
            center_y = y2  # Use the bottom 'y' coordinate

            # Check if this (x, y) point is inside our polygon
            # cv2.pointPolygonTest returns:
            #   > 0 if inside
            #   = 0 if on the line
            #   < 0 if outside
            if cv2.pointPolygonTest(roi_vertices, (center_x, center_y), False) > 0:
                # This person is INSIDE the zone
                zone_count += 1
            # ----------------------------------------------------

    # 4. Display Counts
    # Display total count (Same as before)
    cv2.putText(frame, f"Total People: {person_count_total}", (20, 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)

    # --- DAY 2, STEP 2: Display the new zone count ---
    # We'll display this count in green, just below the total count
    cv2.putText(frame, f"Zone Count: {zone_count}", (20, 90),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)  # Green text
    # ---------------------------------------------------

    # Show the frame
    cv2.imshow("Crowd Management MVP - Step 2: Counting", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("Step 2 complete. The system is now counting people inside the zone.")