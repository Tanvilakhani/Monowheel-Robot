import cv2
import torch
from ultralytics import YOLO
import numpy as np

MODEL_PATH = "models/yolo11m.pt"  
model = YOLO(MODEL_PATH)  

IP_CAMERA_URL = "http://172.20.10.2:81/stream"

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

    results = model(frame)  

    boxes = results[0].boxes.xyxy.cpu().numpy()  
    scores = results[0].boxes.conf.cpu().numpy()  
    labels = results[0].boxes.cls.cpu().numpy().astype(int) 
    class_names = results[0].names

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        label_name = class_names[label] 
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'{label_name} {score:.2f}', (int(x1), int(y1)-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLO Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
