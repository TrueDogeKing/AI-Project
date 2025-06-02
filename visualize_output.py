# Load required libraries
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
from PIL import Image, ImageOps
from torchvision.datasets import ImageFolder
import string
import math
import os
label_map = list(string.ascii_uppercase + string.ascii_lowercase)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reload the model class
class EMNIST_Classifier(nn.Module):
    def __init__(self):
        super(EMNIST_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.act = nn.LeakyReLU(0.1)

        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 52)

    def forward(self, x):
        x = self.pool(self.act(self.bn1(self.conv1(x))))  # 28x28 → 14x14
        x = self.pool(self.act(self.bn2(self.conv2(x))))  # 14x14 → 7x7
        x = self.pool(self.act(self.bn3(self.conv3(x))))  # 7x7 → 3x3
        x = self.dropout1(x)
        x = x.view(-1, 128 * 3 * 3)
        x = self.dropout2(self.act(self.fc1(x)))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x


def label_to_char(index):
    return label_map[index]  # index is from 1–26, so subtract 1

# Load model
all_images = []
all_labels = []
all_preds = []


model = EMNIST_Classifier().to(device)
model.load_state_dict(torch.load('saved_models/emnist_letters_improved.pth', map_location=device))
model.eval()

# Load test dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


letter_indices = list(range(10, 62))
original_to_letter_idx = {orig: i for i, orig in enumerate(letter_indices)}
letter_indices_set = set(letter_indices)

# Download and filter dataset
emnist_test = datasets.EMNIST(root='./dataset', split='byclass', train=False, download=True, transform=transform)
letter_only_indices = [i for i, (_, label) in enumerate(emnist_test) if label in letter_indices_set]
letter_test_dataset = Subset(emnist_test, letter_only_indices)

# Remap labels to 0–51
class LetterOnlyDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        label = original_to_letter_idx[label]
        if self.transform:
            image = self.transform(image)
        return image, label

test_dataset = LetterOnlyDataset(letter_test_dataset, transform=None)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)

        all_images.append(images.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())


        for i in range(len(labels)):
            true_label = label_to_char(labels[i].item())
            pred_label = label_to_char(preds[i].item())
            correct_case_insensitive = true_label.lower() == pred_label.lower()

            if correct_case_insensitive:
                correct += 1



accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')

# Show more predictions (e.g., 32 examples)
display_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
images, labels = next(iter(display_loader))
images, labels = images.to(device), labels.to(device)

with torch.no_grad():
    outputs = model(images)
    _, preds = torch.max(outputs, 1)


def show_all_batches(images, labels, preds, batch_size=32):
    total_images = len(labels)
    num_batches = math.ceil(total_images / batch_size)

    print(f"Total batches (plots) to display: {num_batches}\n")
    if num_batches>100:
        num_batches=10

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total_images)

        fig, axes = plt.subplots(4, 8, figsize=(16, 8))
        axes = axes.flatten()

        for i in range(start, end):
            img = images[i].squeeze()
            img = np.rot90(img, k=-1)  # Rotate 90° right
            img = img[:, ::-1]  # Mirror horizontally (no channel dim)

            true_label = label_to_char(labels[i].item())
            pred_label = label_to_char(preds[i].item())
            correct_case_insensitive = true_label.lower() == pred_label.lower()
            color = 'green' if correct_case_insensitive else 'red'

            axes[i - start].imshow(img, cmap='gray')
            axes[i - start].set_title(f"T: {true_label} | P: {pred_label}", fontsize=8, color=color)
            axes[i - start].axis('off')

        # Hide unused subplots
        for j in range(end - start, batch_size):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show(block=True)  # Block execution until window is closed

# Usage example (inside your code after predictions):

all_images = np.concatenate(all_images, axis=0)
all_labels = np.concatenate(all_labels, axis=0)
all_preds = np.concatenate(all_preds, axis=0)

show_all_batches(all_images, all_labels, all_preds, batch_size=32)



custom_root = "my_dataset"

# Fix RotateAndMirror
class RotateAndMirror:
    def __call__(self, img):
        # img jest PIL Image
        img = np.array(img)          # Konwersja do numpy
        img = np.rot90(img, k=-1)   # Obrót o 90° w prawo
        img = img[:, ::-1]           # Odbicie lustrzane w poziomie
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
        img = ImageOps.invert(img)
        return img

# Fix transform for custom data
custom_transform = transforms.Compose([
    RotateAndMirror(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Reload the dataset with the updated transform
custom_dataset = ImageFolder(
    root=custom_root,
    transform=custom_transform
)

# Run inference with corrected tensors
custom_loader = DataLoader(custom_dataset, batch_size=32, shuffle=True)
custom_all_images = []
custom_all_labels = []
custom_all_preds = []
images, labels = next(iter(custom_loader))
images, labels = images.to(device), labels.to(device)
total = 0
correct = 0
with torch.no_grad():
    for images, labels in custom_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)

        custom_all_images.append(images.cpu().numpy())
        custom_all_labels.append(labels.cpu().numpy())
        custom_all_preds.append(preds.cpu().numpy())

        for i in range(len(labels)):
            true_label = label_to_char(labels[i].item())
            pred_label = label_to_char(preds[i].item())
            correct_case_insensitive = true_label.lower() == pred_label.lower()

            if correct_case_insensitive:
                correct += 1

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

custom_all_images = np.concatenate(custom_all_images, axis=0)
custom_all_labels = np.concatenate(custom_all_labels, axis=0)
custom_all_preds = np.concatenate(custom_all_preds, axis=0)

def label_to_char(index):
    return label_map[index]

show_all_batches(custom_all_images, custom_all_labels, custom_all_preds, batch_size=32)
