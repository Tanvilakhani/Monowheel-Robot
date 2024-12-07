import cv2
import numpy as np
import requests
import json

IMAGE_FEED_URL = "http://172.20.10.2:81/stream" 
DISTANCE_URL = "http://your-server-address/ultrasonic_distance"  
ULTRASONIC_WEIGHT = 0.7
IMAGE_WEIGHT = 0.3

# Pre-trained model for object detection (you can replace it with a custom model)
YOLO_CONFIG = "models/yolov3.cfg"  # Path to YOLO config file
YOLO_WEIGHTS = "models/yolov3.weights"  # Path to YOLO weights file
YOLO_CLASSES = "models/coco.names"  # Path to YOLO class names

# Load YOLO model
net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CONFIG)
with open(YOLO_CLASSES, "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Function to get the ultrasonic distance from the server
def get_ultrasonic_distance():
    try:
        response = requests.get(DISTANCE_URL)
        response.raise_for_status()
        return float(response.json()["distance"])
    except Exception as e:
        print(f"Error getting ultrasonic distance: {e}")
        return None

# Function to get the image feed from the server
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

# Main function
def main():
    while True:
        # Get ultrasonic distance
        # ultrasonic_distance = get_ultrasonic_distance()
        print('getting ultrasonic distance')
        ultrasonic_distance = 12

        # Get image frame
        print('Getting image feed')
        frame = get_image_feed()
        if frame is None:
            continue


        # Process the image
        print('Processing the image')
        obj_class, image_distance, box = process_image(frame)

        if obj_class is not None and ultrasonic_distance is not None:
            # Sensor fusion: weighted average
            fused_distance = (ULTRASONIC_WEIGHT * ultrasonic_distance +
                              IMAGE_WEIGHT * image_distance)

            # Draw bounding box and information
            print('Drawing bounding box')
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{obj_class}: {fused_distance:.2f} cm"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            print(f"Object: {obj_class}, Ultrasonic: {ultrasonic_distance} cm, Image: {image_distance} cm, Fused: {fused_distance} cm")

        # Display the frame
        cv2.imshow("Object Tracking", frame)

        # Break loop on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    

main()
