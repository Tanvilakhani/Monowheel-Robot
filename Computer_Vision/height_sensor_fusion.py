import cv2
from flask import Flask, request, jsonify, Response
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from scipy.interpolate import interp1d
import requests


#initializing the actual test object height, ultrasonic distances and their corresponding bounding box heights
object_height = 9  
distances = np.array([25,30,35,40,45,50])  
pixel_differences = np.array([243,187,174,138,132,123])  


#interpolation and kalman filter initialization
pixel_to_distance = interp1d(pixel_differences, distances, fill_value="extrapolate")
kf = KalmanFilter(dim_x=2, dim_z=1)
kf.x = np.array([25, 0])  # Initial state [distance, velocity]
kf.F = np.array([[1, 1], [0, 1]])  # State transition matrix
kf.H = np.array([[1, 0]])  # Measurement function
kf.P *= 5  # Covariance matrix
kf.R = 5 # Measurement noise
kf.Q = 0.3 * np.eye(2)

app = Flask(__name__)

YOLO_MODEL = "models/yolo11m.pt" 

model = YOLO(YOLO_MODEL)

current_distance = None


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

        ret, buffer = cv2.imencode('.jpg', f_frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
    
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

def process_image(frame):
    """ Process frames using YOLOv11 for object detection and distance estimation"""

    height, width, _ = frame.shape

    results = model(frame, conf=0.5)[0] 
    
    if len(results) > 0:
  
        best_detection = max(results.boxes, key=lambda box: box.conf[0])
        box = best_detection.xyxy[0].cpu().numpy()
        x, y, x2, y2 = box[:4]
        

        w = x2 - x
        h = y2 - y
        
        distance = h
        
        class_id = int(best_detection.cls[0])
        class_name = model.names[class_id]
        
        return class_name, distance, (int(x), int(y), int(w), int(h))
    
    return None, None, None

def calculate_height(distance, pixel_diff):
    """Final object height calculation function"""
    interpolated_distance = pixel_to_distance(pixel_diff)
    scale_factor = object_height / interpolated_distance
    print(f'test: {scale_factor * distance}')
    return scale_factor * distance

def sensor_fusion(frame):
    """combines the ultrasonicvalues and bounding box heights to detect actual height"""
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

                kf.update(ultrasonic_distance)
                filtered_distance = kf.x[0]

                fused_height = calculate_height(filtered_distance, h)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = f"{obj_class}, Height: {fused_height:.2f} cm"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                print(f"Object: {obj_class}, Ultrasonic Distance: {ultrasonic_distance} cm, "
                    f"Pixel Height: {image_distance} cm, Fused Height: {fused_height:.2f} cm")
                
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
    """gets the ultrasonic distance from the sensor distance via server"""
    global current_distance
    
    distance = request.get_json()
    
    if distance:
        current_distance = distance
        print("Received ULTRASONIC_DISTANCE_DATA:", distance)
        return jsonify({"received_data": distance})
    
    return distance


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
