import torch
from torchvision import transforms
import cv2
import pandas as pd
import torch.nn as nn

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
            nn.Linear(128 * 41 * 41, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomCNN(num_classes=17).to(device)
model.load_state_dict(torch.load("custom_cnn_model.pth", map_location=device))
model.eval()
print("Custom CNN model loaded successfully!")

IMG_HEIGHT, IMG_WIDTH = 331, 331

class_label_mapping = {
    0: "Rock and Gravel",
    1: "Brown Leaves on Wet Ground",
    2: "Wet Sand Path with Debris",
    3: "Wood",
    4: "Young Grass Growing in the Dog Park",
    5: "Wavy Wet Beach Sand",
    6: "Dry Dirt Road and Debris from Trees",
    7: "Sandy Beach Path with Grass Clumps",
    8: "Pine Needles and Brown Leaves on Park Floor",
    9: "Dry Grass with Pine Needles",
    10: "Chipped Stones, Broken Leaves and Twiglets",
    11: "Grass Clumps and Cracked Dirt",
    12: "Dirt, Stones, Rock, Twigs...",
    13: "Plants with Flowers on Dry Leaves",
    14: "Footpath with Snow",
    15: "Pine Needles Forest Floor",
    16: "Snow on Grass and Dried Leaves",
}

transform = transforms.Compose([
    transforms.Resize((331, 331)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

stream_url = "http://192.168.213.153:81/stream"
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Unable to access the camera. Exiting...")
    exit()

predictions_table = []

print("Processing live video stream...")
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Unable to capture video frame. Exiting...")
        break

    frame_count += 1

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pil_image = Image.fromarray(rgb_frame)
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_index = torch.argmax(outputs, dim=1).item()
        confidence = torch.softmax(outputs, dim=1)[0, predicted_index].item()

    predicted_class_name = class_label_mapping[predicted_index]

    predictions_table.append({
        "Frame": frame_count,
        "Predicted Class": predicted_class_name,
        "Confidence": f"{confidence:.2f}"
    })

    label = f"{predicted_class_name} ({confidence:.2f})"
    cv2.putText(frame, label, (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 4)

    cv2.imshow("Live Stream Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting live stream...")
        break

cap.release()
cv2.destroyAllWindows()
print("Live video processing complete.")

df_predictions = pd.DataFrame(predictions_table)

print("Prediction Table:")
print(df_predictions)

df_predictions.to_csv("live_stream_predictions.csv", index=False)
print("Prediction table saved as 'live_stream_predictions.csv'.")
