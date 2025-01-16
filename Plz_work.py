import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2, Preview
from tracker import *
import cvzone

# Initialize YOLO model (using lightweight YOLOv8 Nano)
model = YOLO('yolov8n.pt')

# Set up PiCamera2
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(main={"size": (640, 360)})
picam2.configure(camera_config)
picam2.start()

# Initialize output video writer
output = cv2.VideoWriter('output_final.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (640, 360))

# Load COCO class names
with open('coco.names', 'r') as file:
    class_list = file.read().splitlines()

count = 0
persondown = {}
tracker = Tracker()
counter1 = []

personup = {}
counter2 = []
cy1 = 200
cy2 = 300
offset = 6

try:
    while True:
        # Capture frame from PiCamera
        frame = picam2.capture_array()

        count += 1
        if count % 5 != 0:  # Skip more frames for better performance
            continue

        # Resize frame to reduce computation
        frame = cv2.resize(frame, (640, 360))

        # Run YOLO model
        results = model.predict(frame, verbose=False)  # Suppress YOLO logs

        # Extract bounding box data
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")

        detections = []
        for _, row in px.iterrows():
            x1, y1, x2, y2, _, cls = map(int, row)
            if class_list[cls] == 'person':  # Filter for 'person' class
                detections.append([x1, y1, x2, y2])

        # Update tracker
        bbox_id = tracker.update(detections)
        for bbox in bbox_id:
            x3, y3, x4, y4, obj_id = bbox
            cx = (x3 + x4) // 2
            cy = (y3 + y4) // 2

            cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)

            # Downward motion detection
            if cy1 - offset < cy < cy1 + offset:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                cvzone.putTextRect(frame, f'{obj_id}', (x3, y3), 1, 2)
                persondown[obj_id] = (cx, cy)

            if obj_id in persondown:
                if cy2 - offset < cy < cy2 + offset:
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 255), 2)
                    cvzone.putTextRect(frame, f'{obj_id}', (x3, y3), 1, 2)
                    if obj_id not in counter1:
                        counter1.append(obj_id)

            # Upward motion detection
            if cy2 - offset < cy < cy2 + offset:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f'{obj_id}', (x3, y3), 1, 2)
                personup[obj_id] = (cx, cy)

            if obj_id in personup:
                if cy1 - offset < cy < cy1 + offset:
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 255), 2)
                    cvzone.putTextRect(frame, f'{obj_id}', (x3, y3), 1, 2)
                    if obj_id not in counter2:
                        counter2.append(obj_id)

        # Draw lines
        cv2.line(frame, (3, cy1), (637, cy1), (0, 255, 0), 2)
        cv2.line(frame, (5, cy2), (639, cy2), (0, 255, 255), 2)

        # Display counts
        downcount = len(counter1)
        upcount = len(counter2)

        cvzone.putTextRect(frame, f'Down: {downcount}', (50, 60), 2, 2)
        cvzone.putTextRect(frame, f'Up: {upcount}', (50, 160), 2, 2)

        # Write frame to output file
        output.write(frame)

        # Display frame (optional, disable for headless mode)
        cv2.imshow('RGB', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Exit on 'Esc' key
            break
finally:
    # Clean up resources
    picam2.stop()
    output.release()
    cv2.destroyAllWindows()
