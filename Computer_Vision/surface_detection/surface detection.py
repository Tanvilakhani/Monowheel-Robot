!pip install torch torchvision torchaudio
!pip install datasets
!pip install pillow
!pip install matplotlib

"""Loading and Preprocessing dataset"""

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import os
import numpy as np

def save_images(dataset_split, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, entry in enumerate(dataset_split):
        image = entry['image']
        label = entry['set']
        label_dir = os.path.join(output_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)
        image.save(os.path.join(label_dir, f"{i}.png"))

print("Loading dataset...")
dataset = load_dataset('texturedesign/td01_natural-ground-textures', 'PNG@1K', num_proc=4, trust_remote_code=True)

print("Saving dataset to Colab environment...")
save_images(dataset['train'], "train")
save_images(dataset['test'], "test")

IMG_HEIGHT, IMG_WIDTH = 331, 331
transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CustomDataset(Dataset):
    def __init__(self, dataset_split, transform=None):
        self.images = [entry['image'] for entry in dataset_split]
        self.labels = [entry['set'] for entry in dataset_split]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

print("Preprocessing data...")
train_dataset = CustomDataset(dataset['train'], transform=transform)
test_dataset = CustomDataset(dataset['test'], transform=transform)

label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform([entry['set'] for entry in dataset['train']])
test_labels = label_encoder.transform([entry['set'] for entry in dataset['test']])

class_names = label_encoder.classes_
print(f"Classes: {class_names}")

train_size = int(0.8 * len(train_dataset))
valid_size = len(train_dataset) - train_size
train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])

BATCH_SIZE = 32

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Training Set: {len(train_loader)} batches")
print(f"Validation Set: {len(valid_loader)} batches")
print(f"Test Set: {len(test_loader)} batches")

"""Training the custom model"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

IMG_HEIGHT, IMG_WIDTH = 331, 331
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

train_dir = "/content/train"
test_dir = "/content/test"

train_transforms = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_transforms = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

num_classes = len(train_dataset.classes)
print(f"Number of classes: {num_classes}")

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
            nn.Linear(128 * (IMG_HEIGHT // 8) * (IMG_WIDTH // 8), 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomCNN(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

train_losses, val_losses = [], []
train_accs, val_accs = [], []

print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_losses.append(running_loss / total)
    train_accs.append(correct / total)

    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_losses.append(val_loss / val_total)
    val_accs.append(val_correct / val_total)

    print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_losses[-1]:.4f} | Train Acc: {train_accs[-1]:.4f} | Val Loss: {val_losses[-1]:.4f} | Val Acc: {val_accs[-1]:.4f}")

    scheduler.step()

torch.save(model.state_dict(), "custom_cnn_model.pth")
print("Model saved as 'custom_cnn_model.pth'")

plt.figure(figsize=(12, 6))
plt.plot(range(EPOCHS), train_accs, label='Training Accuracy')
plt.plot(range(EPOCHS), val_accs, label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(range(EPOCHS), train_losses, label='Training Loss')
plt.plot(range(EPOCHS), val_losses, label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import torch
import numpy as np
import matplotlib.pyplot as plt

print("Evaluating the model...")
model.eval()
all_preds, all_labels = [], []
test_loss, test_correct, test_total = 0.0, 0, 0

criterion = torch.nn.CrossEntropyLoss()

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)

test_loss /= test_total
test_accuracy = test_correct / test_total
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

print("Generating classification report...")
class_labels = test_dataset.classes
report = classification_report(all_labels, all_preds, target_names=class_labels)
print(report)

print("Generating confusion matrix...")
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap='viridis', xticks_rotation=45)
plt.show()

"""Testing code for images"""

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import random
import torch.nn as nn

IMG_HEIGHT, IMG_WIDTH = 331, 331

test_dir = "/content/test"

idx_to_class = {
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
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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
            nn.Linear(128 * (IMG_HEIGHT // 8) * (IMG_WIDTH // 8), 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomCNN(num_classes=len(idx_to_class)).to(device)
model.load_state_dict(torch.load("custom_cnn_model.pth", map_location=device))
model.eval()
print("Custom CNN model loaded and ready for inference!")

test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

random_indices = random.sample(range(len(test_dataset)), 5)
selected_images = [test_dataset[i] for i in random_indices]
selected_filepaths = [test_dataset.samples[i][0] for i in random_indices]
selected_labels = [test_dataset.samples[i][1] for i in random_indices]

print("Displaying randomly selected test images...")
for image, true_label, filepath in zip(selected_images, selected_labels, selected_filepaths):

    img, _ = image
    img_tensor = img.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        predicted_index = torch.argmax(outputs, dim=1).item()
        confidence = torch.softmax(outputs, dim=1)[0, predicted_index].item()

    true_class_name = idx_to_class[true_label]
    predicted_class_name = idx_to_class[predicted_index]

    img_original = Image.open(filepath)
    img_resized = img_original.resize((150, 150))

    plt.figure(figsize=(4, 4))
    plt.imshow(img_resized)
    plt.axis("off")
    plt.title(f"True: {true_class_name}\nPred: {predicted_class_name} ({confidence:.2f})")
    plt.show()

"""Testing code for video input"""

import torch
from torchvision import transforms
import cv2
import os
from PIL import Image
import pandas as pd
from google.colab.patches import cv2_imshow
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

video_path = input("Enter the path to the video file: ").strip()
if not os.path.isfile(video_path):
    print("Invalid video path. Exiting...")
    exit()

cap = cv2.VideoCapture(video_path)

predictions_table = []
first_frame_displayed = False

print("Processing video...")
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
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

    if not first_frame_displayed:
        label = f"{predicted_class_name} ({confidence:.2f})"
        cv2.putText(frame, label, (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 4)
        cv2_imshow(frame)
        first_frame_displayed = True

cap.release()
print("Video processing complete.")

df_predictions = pd.DataFrame(predictions_table)

print("Prediction Table:")
print(df_predictions)

df_predictions.to_csv("video_predictions.csv", index=False)
print("Prediction table saved as 'video_predictions.csv'.")
