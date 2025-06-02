import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Reproducibility
torch.manual_seed(42)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Save directory
os.makedirs('saved_models', exist_ok=True)

## --- Filter letters only from EMNIST ---
def filter_letters(dataset):
    letter_labels = list(range(10, 62))
    indices = [i for i, (_, label) in enumerate(dataset) if label in letter_labels]
    filtered_dataset = Subset(dataset, indices)

    def remap_labels(idx):
        img, label = dataset[idx]
        return img, label - 10  # [10..61] → [0..51]

    class RemappedDataset(torch.utils.data.Dataset):
        def __init__(self, subset):
            self.subset = subset
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, idx):
            original_idx = self.subset.indices[idx]
            return remap_labels(original_idx)

    return RemappedDataset(filtered_dataset)

## --- Load EMNIST letters-only training data ---
def load_emnist_letters_only():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_train_dataset = datasets.EMNIST(
        root='./dataset',
        split='byclass',
        train=True,
        download=True,
        transform=transform
    )

    train_dataset = filter_letters(full_train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    return train_loader

## --- Improved CNN Model ---
class EMNIST_Classifier_LettersOnly(nn.Module):
    def __init__(self):
        super().__init__()
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
        self.fc2 = nn.Linear(256, 128)  # << extra layer
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


## --- Training Function ---
def train(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    train_losses = []
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    for epoch in range(epochs):
        running_loss = 0.0
        print(f"Epoch {epoch+1}/{epochs}")

        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        scheduler.step()

        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), 'saved_models/emnist_letters_only_trained.pth')
    print("✅ Model saved to 'saved_models/emnist_letters_only_trained.pth'")
    return train_losses

## --- Main Execution ---
if __name__ == '__main__':
    train_loader = load_emnist_letters_only()
    model = EMNIST_Classifier_LettersOnly().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("🚀 Starting training on EMNIST letters only...")
    train_losses = train(model, train_loader, criterion, optimizer, epochs=10)

    # Optional: Plot training loss
    plt.plot(train_losses)
    plt.title('Training Loss (Letters Only)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
