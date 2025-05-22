# Load required libraries
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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
        self.fc2 = nn.Linear(128, 62)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# Load model
model = EMNIST_Classifier().to(device)
model.load_state_dict(torch.load('saved_models/emnist_classifier_final.pth'))
model.eval()

# Load test dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.EMNIST(root='./data', split='digits', train=False, download=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')

# Show more predictions (e.g., 32 examples)
display_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
images, labels = next(iter(display_loader))
images, labels = images.to(device), labels.to(device)

with torch.no_grad():
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

# Mapping labels to characters
label_map = test_dataset.classes
def label_to_char(index):
    return label_map[index]

# Plotting 32 predictions
images = images.cpu().numpy()
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
axes = axes.flatten()
for i in range(32):
    img = images[i].squeeze()
    true_label = label_to_char(labels[i].item())
    pred_label = label_to_char(preds[i].item())
    color = 'green' if true_label == pred_label else 'red'
    
    axes[i].imshow((img), cmap='gray')
    axes[i].set_title(f"T: {true_label} | P: {pred_label}", fontsize=8, color=color)
    axes[i].axis('off')

plt.tight_layout()
plt.show()