import cv2
from ultralytics import YOLO
import random

# --- Configuration ---
# Load the YOLOv8 model (yolov8n.pt is fast, yolov8m.pt is more accurate)
model = YOLO('yolov8n.pt')

# Define the path to your video file
VIDEO_PATH = "cctv.mp4"

# COCO Dataset Class ID for 'person'
PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.5

# Store for track history (to draw paths, etc. - good for demos)
# We'll also store a random color for each track ID
track_colors = {}

# --- Video Processing ---
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: Could not open video file '{VIDEO_PATH}'.")
    exit()

print("Starting video processing for Day 1 MVP... Press 'ESC' to exit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video stream.")
        break

    # 1. Run YOLOv8 Tracking
    # 'persist=True' tells the tracker to remember IDs between frames
    # 'tracker="bytetrack.yaml"' specifies the tracker algorithm
    results = model.track(
        source=frame,
        persist=True,
        classes=[PERSON_CLASS_ID],
        conf=CONFIDENCE_THRESHOLD,
        tracker="bytetrack.yaml",
        verbose=False  # Suppress console logs
    )

    person_count = 0

    # 2. Process Tracking Results
    # Check if any tracks were found
    if results[0].boxes.id is not None:

        # Get all boxes, track IDs, and confidences
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()

        person_count = len(track_ids)

        # 3. Draw Bounding Boxes and Track IDs
        for box, track_id, conf in zip(boxes, track_ids, confs):
            x1, y1, x2, y2 = map(int, box)

            # --- Get or create a unique color for this track_id ---
            if track_id not in track_colors:
                # Generate a random color (BGR format)
                track_colors[track_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            color = track_colors[track_id]

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Prepare text: "ID: [id]"
            label = f"ID: {track_id}"

            # Calculate text size for background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            # Draw a filled rectangle as a background for the label
            cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1 - 10), color, -1)
            # Draw the text label
            cv2.putText(frame, label, (x1, y1 - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # White text

    # 4. Display the total count
    count_text = f"People Count: {person_count}"
    cv2.putText(frame, count_text, (20, 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)  # Red Count

    # Show the frame
    cv2.imshow("Crowd Management MVP - Day 1: Tracking", frame)

    # Break loop on 'ESC' key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("Day 1 MVP processing complete.")