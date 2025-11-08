from flask import Flask, Response
from flask_socketio import SocketIO
import cv2
import numpy as np
from ultralytics import YOLO
import random
import time

# --- 1. FLASK AND SOCKETIO SETUP ---
app = Flask(__name__)
# Set a secret key (required for SocketIO)
app.config['SECRET_KEY'] = 'a_secure_secret_key_for_vision_app'
# Initialize SocketIO, allowing cross-origin requests for React (on a different port)
socketio = SocketIO(app, cors_allowed_origins="*")

# --- 2. GLOBAL CV CONFIGURATION (From your Day 2 code) ---
# Initialize YOLO model once globally
model = YOLO('yolov8n.pt')
VIDEO_PATH = "cctv.mp4"  # Make sure this file exists in the same directory!
# If you want to use the webcam instead, set: VIDEO_PATH = 0
PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.5
maxperzone = 10  # Alert threshold

# ROI Vertices - Define the risk zone polygon (coordinates must match your video)
roi_vertices = np.array([
    [500, 0],
    [800, 0],
    [900, 1000],
    [400, 1000]
], np.int32)

# Tracking data
track_colors = {}
camera = None  # Initialize camera globally, opened in generate_frames

# Real-Time Metrics Dictionary (Pushed via SocketIO to React)
current_metrics = {
    "total_people": 0,
    "zone_count": 0,
    "alert": False,
    "max_capacity": maxperzone,
    "status_message": "Awaiting video stream..."
}


# --- 3. CORE PROCESSING FUNCTION (Your Day 2 Logic) ---
def process_frame(frame):
    """
    Runs YOLO tracking, calculates metrics, draws overlays, and updates global state.
    """
    global current_metrics, track_colors, model, roi_vertices, maxperzone

    person_count_total = 0
    zone_count = 0
    is_alert = False

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

        # 3. Draw Bounding Boxes and Check Zone Entry
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)

            if track_id not in track_colors:
                track_colors[track_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            color = track_colors[track_id]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw ID label
            label = f"ID: {track_id}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1 - 10), color, -1)
            cv2.putText(frame, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Check if person is in the zone (using the bottom center point)
            center_x = (x1 + x2) // 2
            center_y = y2

            if cv2.pointPolygonTest(roi_vertices, (center_x, center_y), False) > 0:
                zone_count += 1

    # 4. Display Metrics and Alerts on the Frame
    # Draw Risk Zone
    zone_color_alert = (0, 255, 0)  # Green (Safe)
    if zone_count > maxperzone:
        zone_color_alert = (0, 0, 255)  # Red (Alert)
        is_alert = True

        # Draw the big warning message
        (text_w, text_h), _ = cv2.getTextSize("!! CROWD ALERT !!", cv2.FONT_HERSHEY_TRIPLEX, 1.5, 3)
        center_x_frame = (frame.shape[1] - text_w) // 2
        center_y_frame = (frame.shape[0] + text_h) // 2
        cv2.putText(frame, "!! CROWD ALERT !!",
                    (center_x_frame, center_y_frame),
                    cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 255), 3)  # Red text

    # Draw ROI polygon with dynamic color
    cv2.polylines(frame, [roi_vertices], isClosed=True, color=zone_color_alert, thickness=5)

    # Display counts on frame
    cv2.putText(frame, f"TOTAL: {person_count_total}", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, f"ZONE: {zone_count} / {maxperzone}", (20, 90), cv2.FONT_HERSHEY_DUPLEX, 1.0, zone_color_alert,
                2)

    # 5. UPDATE GLOBAL METRICS (for SocketIO)
    current_metrics["total_people"] = person_count_total
    current_metrics["zone_count"] = zone_count
    current_metrics["alert"] = is_alert
    current_metrics["status_message"] = "CROWD ALERT: Capacity Exceeded" if is_alert else "Monitoring Safely"

    return frame


# --- 4. MJPEG STREAM GENERATOR ---
def generate_frames():
    """
    Generator that handles video capture, processing, encoding, and metric pushing.
    """
    global camera

    # Initialize camera/video stream on the first call
    if camera is None:
        camera = cv2.VideoCapture(VIDEO_PATH)
        print(f"INFO: Attempting to open video source: {VIDEO_PATH}")

    if not camera.isOpened():
        # IMPORTANT DIAGNOSTIC MESSAGE
        error_msg = f"ERROR: Could not open video source {VIDEO_PATH}. Check file path or webcam availability."
        print(error_msg)
        current_metrics["status_message"] = error_msg

        # FIX: Removed 'broadcast=True'
        socketio.emit('metrics_update', current_metrics)
        return

    while True:
        success, frame = camera.read()

        if not success:
            # End of video file or stream error
            error_msg = "Video stream ended or failed. Restarting source..."
            print(error_msg)
            current_metrics["status_message"] = error_msg

            # FIX: Removed 'broadcast=True'
            socketio.emit('metrics_update', current_metrics)
            camera.release()

            # If using a video file, restart the loop; if using webcam (0), try re-opening
            if VIDEO_PATH != 0:
                camera = cv2.VideoCapture(VIDEO_PATH)
                time.sleep(1)  # Wait briefly before trying again
                continue
            else:
                break  # Exit loop if webcam fails

        # Process the frame (updates global metrics)
        processed_frame = process_frame(frame)

        # PUSH METRICS VIA SOCKETIO: Send the latest status to React
        # FIX: Removed 'broadcast=True'
        socketio.emit('metrics_update', current_metrics)

        # Encode the processed frame to JPEG for the web stream
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()

        # YIELD the frame data in the MJPEG stream format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# --- 5. FLASK ROUTES ---

@app.route('/video_feed')
def video_feed():
    """
    The main endpoint for the real-time MJPEG video stream.
    """
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """
    A simple health check endpoint.
    """
    return "Python Backend Running on Port 5000"


# --- 6. RUN THE SERVER ---
if __name__ == '__main__':
    print(f"Starting backend server on http://0.0.0.0:5000")
    print("Video source set to:", VIDEO_PATH)
    # Use socketio.run to start the server, enabling both HTTP and WebSockets
    socketio.run(app, host='0.0.0.0', port='5000', debug=True, allow_unsafe_werkzeug=True)

# Release the camera if the server stops gracefully
if camera:
    camera.release()