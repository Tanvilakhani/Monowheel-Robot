import cv2
from flask import Flask, request, jsonify, Response
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from scipy.interpolate import interp1d

distances = np.array([18.72,22.22,24.41,26.42,28.1,31.72,35.99,40.38, 42.98, 43.71, 49.52, 52.5, 55.62, 56.63, 58.5, 59.31])  # Example calibration distances in cm
pixel_differences = np.array([228.19,226.91,191.83,185.97,186.54,185.82,184.73,83.47, 82.64, 83.13, 88.69, 61.58,62.11, 61.64, 72.98, 72.89])  # Example pixel height differences
object_height = 10  # Known actual height of a object in cm

pixel_to_distance = interp1d(pixel_differences, distances, fill_value="extrapolate")

# Step 2: Kalman Filter Setup
kf = KalmanFilter(dim_x=2, dim_z=1)
kf.x = np.array([50, 0])  # Initial state [distance, velocity]
kf.F = np.array([[1, 1], [0, 1]])  # State transition matrix
kf.H = np.array([[1, 0]])  # Measurement function
kf.P *= 10  # Covariance matrix
kf.R = 5  # Measurement noise
kf.Q = 0.1 * np.eye(2)

app = Flask(__name__)

YOLO_MODEL = "models/yolo11m.pt" 

model = YOLO(YOLO_MODEL)

current_distance = None

def calculate_height(distance, pixel_diff):
    interpolated_distance = pixel_to_distance(pixel_diff)
    scale_factor = object_height / interpolated_distance
    print(f'test: {scale_factor * distance}')
    return scale_factor * distance

def generate_frames(url):
    """Generator function for streaming frames over HTTP"""
    cap = cv2.VideoCapture(url)
    
    if not cap.isOpened():
        print(f"Error: Could not open video stream from {url}")
        return None
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Cannot receive frame")
            break
        
        f_frame = sensor_fusion(frame)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', f_frame)
        if not ret:
            continue
        
        # Convert to bytes
        frame_bytes = buffer.tobytes()
        
        # Yield the frame in a format suitable for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    # Release the video capture when done
    cap.release()

def process_image(frame):
    """
    Process an image using YOLOv8 for object detection and distance estimation
    
    Args:
        frame (numpy.ndarray): Input image frame
    
    Returns:
        tuple: (detected class, estimated distance, bounding box)
               or (None, None, None) if no object detected
    """
    # Get image dimensions
    height, width, _ = frame.shape
    
    # Perform object detection
    results = model(frame, conf=0.5)[0]  # 0.5 confidence threshold
    
    if len(results) > 0:
        # Get the detection with highest confidence
        best_detection = max(results.boxes, key=lambda box: box.conf[0])
        
        # Extract bounding box details
        box = best_detection.xyxy[0].cpu().numpy()
        x, y, x2, y2 = box[:4]
        
        # Calculate bounding box dimensions
        w = x2 - x
        h = y2 - y
        
        # Estimate distance (simplified method based on object height)
        distance = h
        
        # Get class name
        class_id = int(best_detection.cls[0])
        class_name = model.names[class_id]
        
        return class_name, distance, (int(x), int(y), int(w), int(h))
    
    return None, None, None

def sensor_fusion(frame):
    
    while True:
        ultrasonic_distance = current_distance  
        ultrasonic_distance = float(ultrasonic_distance.get("distance", 0.0))
        if frame is None:
            continue

        frame_height, frame_width = frame.shape[:2]

        print(f"Distance: {ultrasonic_distance} cm")
        obj_class, image_distance, box = process_image(frame)

        if box is not None:
            x, y, w, h = box

            if obj_class is not None and ultrasonic_distance:
                print(f"Frame width: {frame_width}")
                print(f"us: {ultrasonic_distance}")
                print(f"image_distance: {image_distance}")

                kf.update(ultrasonic_distance)
                filtered_distance = kf.x[0]

                fused_height = calculate_height(filtered_distance, h)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = f"{obj_class}, Height: {fused_height:.2f} cm"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                print(f"Object: {obj_class}, Ultrasonic: {ultrasonic_distance} cm, "
                    f"Image: {image_distance} cm, Fused: {fused_height:.2f} cm")
            else:
                print("No object class detected")
        else:
            print("No bounding box detected")

        return frame


@app.route('/video_feed')
def video_feed():
    """Endpoint to stream video"""
    return Response(generate_frames('http://10.136.45.13:81/stream'), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data', methods=['POST'])
def get_distance():
    global current_distance
    
    distance = request.get_json()
    
    if distance:
        current_distance = distance
        print("Received ULTRASONIC_DISTANCE_DATA:", distance)
        return jsonify({"received_data": distance})
    
    return distance


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
   