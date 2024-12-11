import sys
import cv2
import numpy as np
from PIL import Image  # Import for image processing
import torch
from torchvision import transforms
import pandas as pd
import torch.nn as nn
import time

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the custom CNN model
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 41 * 41, 512),  # Adjusted based on image dimensions
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Load the trained model
model = CustomCNN(num_classes=17).to(device)
model.load_state_dict(torch.load(r"custom_cnn_model.pth", map_location=device))
model.eval()
print("Custom CNN model loaded successfully!")

# Define image dimensions
IMG_HEIGHT, IMG_WIDTH = 331, 331

# Define a mapping of set numbers to class names
class_label_mapping = {
    0: "Rock and Gravel",
    1: "Brown Leaves",
    2: "Wet Sand Path with Debris",
    3: "Wood",
    4: "Young Grass Growing in the Dog Park",
    5: "Wavy Wet Beach Sand",
    6: "Dry Dirt Road and Debris from Trees",
    7: "Sandy Beach Path with Grass Clumps",
    8: "Pine Needles",
    9: "Dry Grass with Pine Needles",
    10: "Chipped Stones, Broken Leaves and Twiglets",
    11: "Grass Clumps and Cracked Dirt",
    12: "Dirt, Stones, Rock, Twigs...",
    13: "Plants with Flowers on Dry Leaves",
    14: "Footpath with Snow",
    15: "Pine Needles Forest Floor",
    16: "Snow on Grass and Dried Leaves",
}

# Define preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((331, 331)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Open the live video stream (webcam)
stream_url = "http://192.168.213.153:81/stream"
cap = cv2.VideoCapture(stream_url)  
  # Use 0 for default webcam, or replace with the camera index

# Check if the webcam is accessible
if not cap.isOpened():
    print("Unable to access the camera. Exiting...")
    exit()

# Initialize the table
predictions_table = []

print("Processing live video stream...")
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Unable to capture video frame. Exiting...")
        break

    frame_count += 1

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Preprocess the image
    pil_image = Image.fromarray(rgb_frame)
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_index = torch.argmax(outputs, dim=1).item()
        confidence = torch.softmax(outputs, dim=1)[0, predicted_index].item()

    # Get class name
    predicted_class_name = class_label_mapping[predicted_index]

    # Append to the table
    predictions_table.append({
        "Frame": frame_count,
        "Predicted Class": predicted_class_name,
        "Confidence": f"{confidence:.2f}"
    })

    # Overlay prediction on the video frame
    label = f"{predicted_class_name} ({confidence:.2f})"
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the video frame
    cv2.imshow("Live Stream Prediction", frame)

    # Press 'q' to quit the live stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting live stream...")
        break

end_time = time.time()

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()

# Convert the predictions table to a DataFrame
df_predictions = pd.DataFrame(predictions_table)

# Display the predictions table
print("Prediction Table:")
print(df_predictions)

# Save the table as a CSV file
df_predictions.to_csv("live_stream_predictions.csv", index=False)
print(f"Processed {frame_count} frames in {end_time - start_time:.2f} seconds.")
print("Prediction table saved as 'live_stream_predictions.csv'.")
