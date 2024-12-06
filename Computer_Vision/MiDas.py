import torch
import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# Load the MiDaS model
model_type = "DPT_Large"  # Can be 'DPT_Hybrid' or 'MiDaS_small' for smaller models
# midas = torch.hub.load("intel-isl/MiDaS", model_type)

# # Download the model if not already downloaded
# model_file = f"{model_type}.pt"
# urllib.request.urlretrieve(model_path, model_file)

# Load the model
model = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Prepare the image
def preprocess_image(image_path):
    transform = Compose([
        Resize(384),  # Resize to 384 px height while keeping aspect ratio
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform(img).unsqueeze(0)
    return img.to(device)

# Predict depth
def predict_depth(image_path):
    img = preprocess_image(image_path)
    with torch.no_grad():
        depth = model(img)
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1), size=img.shape[2:], mode="bicubic", align_corners=False
    ).squeeze().cpu().numpy()
    return depth

# Visualize depth
def display_depth(image_path, depth):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("Depth Map")
    plt.imshow(depth, cmap="inferno")
    plt.axis("off")
    plt.show()

# Test with an image
image_path = "Computer_Vision/IMG_2951.jpg"  # Replace with your image path
depth_map = predict_depth(image_path)
display_depth(image_path, depth_map)
