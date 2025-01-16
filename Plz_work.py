import pygame
import numpy as np
from ultralytics import YOLO
from tracker import *
import pandas as pd

# Initialize YOLO model
model = YOLO('yolov8s.pt')

# Initialize PyGame
pygame.init()
screen_width, screen_height = 1020, 500
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Person Tracker")

# Set up webcam capture
webcam = pygame.camera.Camera(pygame.camera.list_cameras()[0], (screen_width, screen_height))
webcam.start()

# Read class list
with open('coco.names', 'r') as f:
    class_list = f.read().splitlines()

tracker = Tracker()

cy1, cy2 = 300, 400
offset = 6
counter1, counter2 = [], []
persondown, personup = {}, {}

clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Capture frame from webcam
    frame_surface = webcam.get_image()
    frame = pygame.surfarray.array3d(frame_surface).transpose((1, 0, 2))

    # YOLO prediction
    results = model.predict(frame)
    detections = results[0].boxes.data
    px = pd.DataFrame(detections).astype("float")

    detected_objects = []
    for index, row in px.iterrows():
        x1, y1, x2, y2, conf, cls = map(int, row[:6])
        label = class_list[cls]
        if label == 'person':
            detected_objects.append([x1, y1, x2, y2])

    bbox_id = tracker.update(detected_objects)
    for bbox in bbox_id:
        x1, y1, x2, y2, obj_id = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        pygame.draw.circle(screen, (255, 0, 255), (cx, cy), 4)

        # Logic for counting people
        if cy1 - offset < cy < cy1 + offset:
            pygame.draw.rect(screen, (0, 0, 255), (x1, y1, x2 - x1, y2 - y1), 2)
            persondown[obj_id] = (cx, cy)

        if obj_id in persondown and cy2 - offset < cy < cy2 + offset:
            if obj_id not in counter1:
                counter1.append(obj_id)

        if cy2 - offset < cy < cy2 + offset:
            pygame.draw.rect(screen, (0, 255, 0), (x1, y1, x2 - x1, y2 - y1), 2)
            personup[obj_id] = (cx, cy)

        if obj_id in personup and cy1 - offset < cy < cy1 + offset:
            if obj_id not in counter2:
                counter2.append(obj_id)

    # Draw counting lines
    pygame.draw.line(screen, (0, 255, 0), (0, cy1), (screen_width, cy1), 2)
    pygame.draw.line(screen, (255, 255, 0), (0, cy2), (screen_width, cy2), 2)

    # Display counts
    font = pygame.font.SysFont("Arial", 30)
    down_text = font.render(f"Down: {len(counter1)}", True, (255, 255, 255))
    up_text = font.render(f"Up: {len(counter2)}", True, (255, 255, 255))
    screen.blit(down_text, (50, 50))
    screen.blit(up_text, (50, 100))

    # Update display
    pygame.display.flip()
    clock.tick(30)

# Clean up
webcam.stop()
pygame.quit()
