import sys
import cv2
import numpy as np
from ultralytics import YOLO
from collections import OrderedDict
from scipy.spatial import distance as dist
import time

model = YOLO("yolov8n.pt")  

class ObjectTracker:
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, bbox, class_name):
        self.objects[self.next_object_id] = {"bbox": bbox, "class_name": class_name}
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections):
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.array([((x + x2) // 2, (y + y2) // 2) for x, y, x2, y2, _ in detections])
        input_bboxes = [(x, y, x2, y2) for x, y, x2, y2, _ in detections]
        class_names = [detection[4] for detection in detections]

        if len(self.objects) == 0:
            for bbox, class_name in zip(input_bboxes, class_names):
                self.register(bbox, class_name)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = [((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                                for bbox in [data["bbox"] for data in self.objects.values()]]
            D = dist.cdist(np.array(object_centroids), input_centroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > 50:
                    continue

                object_id = object_ids[row]
                self.objects[object_id]["bbox"] = input_bboxes[col]
                self.objects[object_id]["class_name"] = class_names[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            for col in unused_cols:
                self.register(input_bboxes[col], class_names[col])

        return self.objects


tracker = ObjectTracker(max_disappeared=30)

stream_url = "http://172.20.10.4:81/stream"
# stream_url = 0
cap = cv2.VideoCapture(stream_url)  

if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit()

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from webcam.")
        break

   
    resized_frame = cv2.resize(frame, (640, 480))

    if frame_count % 5 == 0:
        results = model(resized_frame, conf=0.5)
        detections = []
        for box in results[0].boxes:
            x, y, x2, y2 = box.xyxy[0].numpy()
            conf = box.conf[0].item()
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id]
            if conf > 0.7:  
                detections.append((int(x), int(y), int(x2), int(y2), class_name))

        
        tracked_objects = tracker.update(detections)

  
    for object_id, data in tracker.objects.items():
        bbox = data["bbox"]
        class_name = data["class_name"]
        x, y, x2, y2 = bbox
        
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
       
        cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

   
    cv2.imshow("YOLO Object Tracking with Bounding Boxes", frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
end_time = time.time()
print(f"Processed {frame_count} frames in {end_time - start_time:.2f} seconds.")