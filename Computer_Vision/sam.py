import cv2
import torch
from ultralytics.models.sam import SAM2VideoPredictor
import numpy as np

# Create SAM2VideoPredictor
overrides = dict(conf=0.25, task="segment", mode="video", imgsz=1024, model="sam2_b.pt")
predictor = SAM2VideoPredictor(overrides=overrides)

# IP camera stream URL
IP_CAMERA_URL = "http://172.20.10.2:81/stream"

# Initialize video capture
cap = cv2.VideoCapture(IP_CAMERA_URL)
fps = cap.get(cv2.CAP_PROP_FPS)

if not cap.isOpened():
    print("Error: Unable to access the IP camera stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to fetch frame from IP camera.")
        break

    # Convert frame to appropriate format (for SAM model)
    # SAM model expects a BGR image
    # Here we assume that `frame` is already a valid BGR image (from OpenCV)
    
    # You can modify this part to add your points of interest
    input_point = [[920, 470]]  # Example point for segmentation
    input_label = [1]  # Label for the point (for segmentation)

    # Run inference with SAM2VideoPredictor for segmentation
    results = predictor(source=IP_CAMERA_URL, points=input_point, labels=input_label)
    
    # Extract masks from results
    masks = results.masks  # This assumes that `results.masks` is a valid mask array

    # Visualize the mask on the frame
    if masks is not None:
        for mask in masks:
            frame[mask] = [0, 255, 0]  # Apply green overlay for the mask

    # Display the video frame with segmentation results
    cv2.imshow("SAM2 Live Segmentation", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
