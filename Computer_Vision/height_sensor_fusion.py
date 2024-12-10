import cv2
from flask import Flask, request, jsonify, Response
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from scipy.interpolate import interp1d
import requests

# distances = np.array([49.66,50.01,49.95,43.26,42.89,42.91,42.43,38.33,37.43,35.67,36.43,35.55,34.71,33.93,32.4,32.38,30.72,30.31,31.69,27.91,27.88,14.58,54.37,60.78,51.22,50.58,50.01,43.26,24.65,28.34,27.45,22.81,23.29,25.09,26.2,23.83,23.36,24.24,25.13,28.24,21.76,20.15,19.24,17.59,18.43])  # Example calibration distances in cm
distances = np.array([18.43,25.13,30.72,35.67, 42.43,49.95 ])  # Example calibration distances in cm
# pixel_differences = np.array([65.25,64.11,65.30,69.00,65.56,65.61,77.55,90.90,89.84,90.48,90.14,101.78,103.82,102.06,103.46,101.87,100.58,116.52,115.48,114.84,115.79,157.42,239.51,116.80,53.16,53.88,64.50,65.46,67.05,108.31,122.78,122.00,136.14,133.44,132.72,133.72,139.53,134.45,140.07,139.94,149.20,142.34,143.75,143.68,144.14]
pixel_differences = np.array([144.14,140.07,100.58,90.48,77.55, 65.30]
)  # Example pixel height differences
object_height = 10  # Known actual height of a object in cm

pixel_to_distance = interp1d(pixel_differences, distances, fill_value="extrapolate")

# Step 2: Kalman Filter Setup
kf = KalmanFilter(dim_x=2, dim_z=1)
kf.x = np.array([50, 0])  # Initial state [distance, velocity]
kf.F = np.array([[1, 1], [0, 1]])  # State transition matrix
kf.H = np.array([[1, 0]])  # Measurement function
kf.P *= 12  # Covariance matrix
kf.R = 5 # Measurement noise
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

        obj_class, image_distance, box = process_image(frame)

        if box is not None:
            x, y, w, h = box

            if obj_class is not None and ultrasonic_distance:
                print(f"ulatrasonic: {ultrasonic_distance}")
                print(f"pixel height: {image_distance}")

                kf.update(ultrasonic_distance)
                filtered_distance = kf.x[0]

                fused_height = calculate_height(filtered_distance, h)
                error = 1.3
                fused_height = calculate_height(filtered_distance, h) - error
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = f"{obj_class}, Height: {fused_height:.2f} cm"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                print(f"Object: {obj_class}, Ultrasonic: {ultrasonic_distance} cm, "
                    f"Pixel Diff: {image_distance} cm, Fused Height: {fused_height:.2f} cm")
                
                if ultrasonic_distance < 30:
                    if fused_height < 3:
                        try:
                            response = requests.get('http://172.20.10.4:90/climb')
                            print(f"Alert sent. Server response: {response.status_code}")
                        except requests.RequestException as e:
                            print(f"Error sending alert: {e}")
                    else:
                        try:
                            response = requests.get('http://172.20.10.4:90/reverse')
                            print(f"Alert sent. Server response: {response.status_code}")
                        except requests.RequestException as e:
                            print(f"Error sending alert: {e}")

            else:
                print("No object class detected")
        else:
            print("No bounding box detected")

        return frame


@app.route('/video_feed')
def video_feed():
    """Endpoint to stream video"""
    return Response(generate_frames('http://172.20.10.4:81/stream'), 
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
   