import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry
from collections import OrderedDict
from scipy.spatial import distance as dist

# Initialize YOLOv8
model = YOLO("yolov8s.pt")  # Replace with your YOLO model path

# Initialize SAM
sam_checkpoint = r"C:\Users\Sdhavamani\Documents\Monowheel-Robot\sam_vit_h_4b8939.pth"  # Replace with your SAM checkpoint path
sam_model = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
sam_predictor = SamPredictor(sam_model)

class ObjectTracker:
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, centroid, mask):
        self.objects[self.next_object_id] = {"centroid": centroid, "mask": mask}
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections, masks):
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.array([((x + x2) // 2, (y + y2) // 2) for x, y, x2, y2 in detections])

        if len(self.objects) == 0:
            for centroid, mask in zip(input_centroids, masks):
                self.register(centroid, mask)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = [data["centroid"] for data in self.objects.values()]
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
                self.objects[object_id] = {"centroid": input_centroids[col], "mask": masks[col]}
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
                self.register(input_centroids[col], masks[col])

        return self.objects

# Initialize tracker
tracker = ObjectTracker(max_disappeared=30)

# Process video
input_video_path = r"C:\Users\Sdhavamani\Documents\Monowheel-Robot\Videos\Cars.mp4"  # Replace with your video path
output_video_path = r"C:\Users\Sdhavamani\Documents\Monowheel-Robot\Videos\Cars_output.mp4"

cap = cv2.VideoCapture(input_video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Validate the frame
    if frame is None or frame.size == 0:
        print("Error: Empty frame detected!")
        break

    # Run YOLO detection
    results = model(frame)
    detections = []
    for box in results[0].boxes:
        x, y, x2, y2 = box.xyxy[0].numpy()

        # Validate bounding box
        if x < 0 or y < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
            print(f"Warning: Invalid bounding box ({x}, {y}, {x2}, {y2}) skipped!")
            continue

        detections.append((int(x), int(y), int(x2), int(y2)))

        # Generate masks using SAM
        sam_predictor.set_image(frame)
        masks = []
        for detection in detections:
            x, y, x2, y2 = detection
            bbox = np.array([x, y, x2 - x, y2 - y])  # SAM expects bbox as [x, y, width, height]
            print(f"Processing bounding box: {bbox}")

            try:
                mask, _, _ = sam_predictor.predict(box=bbox, point_coords=None, point_labels=None, multimask_output=False)

                if mask is None or mask.size == 0:
                    print(f"Warning: Empty or invalid mask for bbox {bbox}")
                    continue

                # Validate mask dimensions
                if len(mask.shape) != 2:
                    print(f"Warning: Unexpected mask dimensions {mask.shape} for bbox {bbox}")
                    continue

                # Resize mask to match frame dimensions
                mask_resized = cv2.resize(mask.astype("uint8"), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                masks.append(mask_resized)

            except Exception as e:
                print(f"Error during mask generation or resizing for bbox {bbox}: {e}")
            continue

    # Update tracker with detections and resized masks
    tracked_objects = tracker.update(detections, masks)

    # Draw tracked objects
    for object_id, data in tracked_objects.items():
        centroid = data["centroid"]
        mask = data["mask"]
        color = np.random.randint(0, 255, (3,), dtype=int).tolist()

        # Apply mask to frame
        frame[mask > 0] = frame[mask > 0] * 0.5 + np.array(color) * 0.5
        cv2.putText(frame, f"ID {object_id}", (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, tuple(centroid), 4, (0, 255, 0), -1)
        cv2.imshow("YOLO + SAM Tracking", frame)

    # Write frame to output video
    out.write(frame)

    # Display frame (optional)
    # cv2.imshow("YOLO + SAM Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
