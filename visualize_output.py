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
import seaborn as sns
from collections import Counter, defaultdict
from sklearn.metrics import confusion_matrix

label_map = list(string.ascii_uppercase + string.ascii_lowercase)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        x = self.pool(self.act(self.bn1(self.conv1(x))))
        x = self.pool(self.act(self.bn2(self.conv2(x))))
        x = self.pool(self.act(self.bn3(self.conv3(x))))
        x = self.dropout1(x)
        x = x.view(-1, 128 * 3 * 3)
        x = self.dropout2(self.act(self.fc1(x)))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x

def label_to_char(index):
    return label_map[index]

model = EMNIST_Classifier().to(device)
model.load_state_dict(torch.load('saved_models/emnist_letters_improved.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

letter_indices = list(range(10, 62))
original_to_letter_idx = {orig: i for i, orig in enumerate(letter_indices)}
letter_indices_set = set(letter_indices)
emnist_test = datasets.EMNIST(root='./dataset', split='byclass', train=False, download=True, transform=transform)
letter_only_indices = [i for i, (_, label) in enumerate(emnist_test) if label in letter_indices_set]
letter_test_dataset = Subset(emnist_test, letter_only_indices)

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

custom_root = "my_dataset"

class RotateAndMirror:
    def __call__(self, img):
        img = np.array(img)
        img = np.rot90(img, k=-1)
        img = img[:, ::-1]
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
        img = ImageOps.invert(img)
        return img

custom_transform = transforms.Compose([
    RotateAndMirror(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

custom_dataset = ImageFolder(root=custom_root, transform=custom_transform)
custom_loader = DataLoader(custom_dataset, batch_size=64, shuffle=True)

custom_all_images = []
custom_all_labels = []
custom_all_preds = []
total = 0
correct_case_insensitive = 0
correct_case_sensitive = 0

with torch.no_grad():
    for images, labels in custom_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        custom_all_images.append(images.cpu().numpy())
        custom_all_labels.append(labels.cpu().numpy())
        custom_all_preds.append(preds.cpu().numpy())

        for i in range(len(labels)):
            true_label = label_to_char(labels[i].item())
            pred_label = label_to_char(preds[i].item())
            if true_label.lower() == pred_label.lower():
                correct_case_insensitive += 1
                if true_label == pred_label:
                    correct_case_sensitive += 1

accuracy_case_insensitive = 100 * correct_case_insensitive / total
accuracy_case_sensitive = 100 * correct_case_sensitive / total
print(f"Case-Insensitive Accuracy: {accuracy_case_insensitive:.2f}%")
print(f"Case-Sensitive Accuracy: {accuracy_case_sensitive:.2f}%")

custom_all_images = np.concatenate(custom_all_images, axis=0)
custom_all_labels = np.concatenate(custom_all_labels, axis=0)
custom_all_preds = np.concatenate(custom_all_preds, axis=0)

# Case-sensitive confusion matrix
conf_mat_sensitive = confusion_matrix(custom_all_labels, custom_all_preds, labels=list(range(52)))
plt.figure(figsize=(12, 10))
sns.heatmap(conf_mat_sensitive, xticklabels=label_map, yticklabels=label_map, annot=False, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Case-Sensitive)")
plt.tight_layout()
plt.show()

# Case-insensitive confusion matrix
label_map_lower = [c.lower() for c in label_map]
true_lower = [label_map[l].lower() for l in custom_all_labels]
pred_lower = [label_map[p].lower() for p in custom_all_preds]
unique_labels_lower = sorted(set(label_map_lower))
label_to_index_lower = {c: i for i, c in enumerate(unique_labels_lower)}
true_lower_indices = [label_to_index_lower[c] for c in true_lower]
pred_lower_indices = [label_to_index_lower[c] for c in pred_lower]

conf_mat_insensitive = confusion_matrix(true_lower_indices, pred_lower_indices, labels=list(range(len(unique_labels_lower))))
plt.figure(figsize=(12, 10))
sns.heatmap(conf_mat_insensitive, xticklabels=unique_labels_lower, yticklabels=unique_labels_lower, annot=False, fmt="d", cmap="Purples")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Case-Insensitive)")
plt.tight_layout()
plt.show()

predicted_wrong_images = defaultdict(list)
predicted_wrong_summary = defaultdict(lambda: [0, 0])

for img, true, pred in zip(custom_all_images, custom_all_labels, custom_all_preds):
    pred_char = label_to_char(pred)
    true_char = label_to_char(true)
    if pred_char.lower() != true_char.lower():
        predicted_wrong_summary[pred_char][0] += 1
        predicted_wrong_summary[pred_char][1] += 1
        predicted_wrong_images[pred_char].append((img.squeeze(), true_char))

print("\nPrediction Summary (model predictions and incorrect cases — Case-Insensitive Only):\n")
sorted_summary = sorted(predicted_wrong_summary.items(), key=lambda x: x[1][1], reverse=True)
for pred_char, (total, wrong) in sorted_summary:
    print(f"Predicted '{pred_char}' {total} times — {wrong} were incorrect")

def show_wrong_predictions(predicted_wrong_images, max_per_pred=12):
    for pred_char, entries in predicted_wrong_images.items():
        num_images = min(len(entries), max_per_pred)
        rows = (num_images + 3) // 4
        fig, axes = plt.subplots(rows, 4, figsize=(10, 2.5 * rows))
        axes = axes.flatten()

        for i in range(num_images):
            img, true_char = entries[i]
            img = np.rot90(img, k=-1)
            img = img[:, ::-1]
            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f"T: {true_char}", fontsize=10)

        for j in range(num_images, len(axes)):
            axes[j].axis('off')

        plt.suptitle(f"Predicted: '{pred_char}' — Showing {num_images} incorrect cases", fontsize=14, color='red')
        plt.tight_layout()
        plt.show()

show_wrong_predictions(predicted_wrong_images, max_per_pred=12)
