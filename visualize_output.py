# Load required libraries
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
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
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 52)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def label_to_char(index):
    return label_map[index]  # index is from 1–26, so subtract 1

# Load model
all_images = []
all_labels = []
all_preds = []


model = EMNIST_Classifier().to(device)
model.load_state_dict(torch.load('saved_models/emnist_onlyLetters_classifier_final.pth'))
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