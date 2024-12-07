import cv2
import numpy as np
import requests
import threading
from flask import Flask, request, jsonify

# Flask App for Distance API
app = Flask(__name__)

# Global variable to store the ultrasonic distance
ultrasonic_distance_data = {"distance": None}

@app.route('/data', methods=['POST'])
def handle_post():
    global ultrasonic_distance_data
    # Extract JSON data from the POST request
    data = request.get_json()
    if data and "distance" in data:
        ultrasonic_distance_data["distance"] = float(data["distance"])
        return jsonify({"status": "success", "received_distance": data["distance"]})
    return jsonify({"status": "error", "message": "Invalid data"}), 400

def run_flask_app():
    app.run(host='0.0.0.0', port=5001, debug=False)

# Object Detection and Distance Fusion Code
IMAGE_FEED_URL = "http://10.136.45.13:81/stream" 
FLASK_DISTANCE_URL = "http://127.0.0.1:5001/data"
ULTRASONIC_WEIGHT = 0.7
IMAGE_WEIGHT = 0.3

# Pre-trained model for object detection (replace with custom model if needed)
YOLO_CONFIG = "models/yolov3.cfg"  # Path to YOLO config file
YOLO_WEIGHTS = "models/yolov3.weights"  # Path to YOLO weights file
YOLO_CLASSES = "models/coco.names"  # Path to YOLO class names

# Load YOLO model
net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CONFIG)
with open(YOLO_CLASSES, "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

def get_image_feed():
    try:
        response = requests.get(IMAGE_FEED_URL, stream=True)
        response.raise_for_status()
        image_bytes = np.asarray(bytearray(response.content), dtype=np.uint8)
        return cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error getting image feed: {e}")
        return None

def process_image(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indexes) > 0:
        i = indexes[0][0]
        box = boxes[i]
        x, y, w, h = box
        distance = height / h  # Approximate distance based on object size
        return classes[class_ids[i]], distance, (x, y, w, h)
    return None, None, None

def main():
    global ultrasonic_distance_data
    while True:
        # Get ultrasonic distance from Flask app
        # ultrasonic_distance = ultrasonic_distance_data.get("distance")
        ultrasonic_distance = ultrasonic_distance_data.get("15")

        # Get image frame
        frame = get_image_feed()
        if frame is None:
            continue

        # Process the image
        obj_class, image_distance, box = process_image(frame)

        if obj_class is not None and ultrasonic_distance is not None:
            # Sensor fusion: weighted average
            fused_distance = (ULTRASONIC_WEIGHT * ultrasonic_distance +
                              IMAGE_WEIGHT * image_distance)

            # Draw bounding box and information
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{obj_class}: {fused_distance:.2f} cm"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            print(f"Object: {obj_class}, Ultrasonic: {ultrasonic_distance} cm, Image: {image_distance} cm, Fused: {fused_distance} cm")

        cv2.imshow("Object Tracking", frame)

        # Break loop on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
#     flask_thread = threading.Thread(target=run_flask_app)
#     flask_thread.daemon = True
#     flask_thread.start()

    main()
