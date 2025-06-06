import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Subset  # Added Subset import
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

# Set random seed for reproducibility
torch.manual_seed(42)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create directory for saving models if it doesn't exist
os.makedirs('saved_models', exist_ok=True)

## Data Loading and Preprocessing
def filter_letters(dataset):
    # byclass split has 62 classes:
    # digits 0-9: labels 0-9
    # uppercase letters: 10-35
    # lowercase letters: 36-61
    # So keep only labels in [10..61]
    letter_labels = list(range(10, 62))
    
    # Find indices where label is in letter_labels
    indices = [i for i, (_, label) in enumerate(dataset) if label in letter_labels]

    # Create a subset with only letters
    filtered_dataset = Subset(dataset, indices)
    
    # Remap labels from [10..61] to [0..51]
    def remap_labels(idx):
        img, label = dataset[idx]
        # Shift label by -10 to get 0-51
        return img, label - 10
    
    # We'll create a wrapper dataset that applies remapping
    class RemappedDataset(torch.utils.data.Dataset):
        def __init__(self, subset):
            self.subset = subset
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, idx):
            original_idx = self.subset.indices[idx]
            return remap_labels(original_idx)
    
    return RemappedDataset(filtered_dataset)

## Data Loading and Preprocessing (modified)
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
    
    full_test_dataset = datasets.EMNIST(
        root='./dataset',
        split='byclass',
        train=False,
        download=True,
        transform=transform
    )
    
    # Filter datasets to letters only and remap labels
    train_dataset = filter_letters(full_train_dataset)
    test_dataset = filter_letters(full_test_dataset)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1000,
        shuffle=False
    )
    
    return train_loader, test_loader



## Define the Neural Network Architecture
class EMNIST_Classifier_LettersOnly(nn.Module):
    def __init__(self):
        super(EMNIST_Classifier_LettersOnly, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 52)  # 52 outputs for letters only
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

## Training Function with Checkpointing
def train(model, train_loader, criterion, optimizer, epochs=10, resume=False):
    model.train()
    train_losses = []
    start_epoch = 0
    
    if resume:
        try:
            checkpoint = torch.load('saved_models/checkpoint.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            train_losses = checkpoint['train_losses']
            print(f"Resuming training from epoch {start_epoch}")
        except FileNotFoundError:
            print("No checkpoint found. Starting fresh training.")
    
    for epoch in range(start_epoch, epochs):
        running_loss = 0.0
        print(f"Epoch {epoch+1}/{epochs}")
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'loss': epoch_loss,
        }, 'saved_models/checkpoint.pth')

        print(f"Loss: {epoch_loss:.4f}")

    
    # Save final model - !! change name for letter only !!
    torch.save(model.state_dict(), 'saved_models/emnist_onlyLetters_classifier_final.pth')
    return train_losses

## Evaluation Function
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

## Main Execution
if __name__ == '__main__':
    # Load filtered data
    train_loader, test_loader = load_emnist_letters_only()
    
    model = EMNIST_Classifier_LettersOnly().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting training on letters only...")
    train_losses = train(model, train_loader, criterion, optimizer, epochs=10, resume=False)
    
    plt.plot(train_losses)
    plt.title('Training Loss (Letters Only)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    
    print("Evaluating model on letters only...")
    accuracy = evaluate(model, test_loader)
    
    torch.save(model.state_dict(), 'saved_models/emnist_onlyLetters_classifier_final.pth')
    print("Model saved to saved_models/emnist_onlyLetters_classifier_final.pth")