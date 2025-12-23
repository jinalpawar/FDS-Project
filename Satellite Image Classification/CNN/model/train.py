"""
train.py
--------
Training script for the 5-channel Satellite Image Classification CNN.
- Loads image paths and labels
- Splits into train/test sets
- Builds DataLoaders
- Trains the CNN using your original training loop
- Tracks accuracy and loss
- Prints progress each epoch
- Saves the trained model
"""

import os
import glob
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from model import ConvolutionalNetwork
from dataset import SatelliteDataset

# Load dataset file paths
from data_utils import load_dataset_paths
image_paths, labels, class_to_label, label_to_class = load_dataset_paths("data")

# Split into Train & Test
X_train, X_test, y_train, y_test = train_test_split(
    image_paths,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# Create Dataset objects
train_dataset = SatelliteDataset(X_train, y_train, image_size=(224,224))
test_dataset = SatelliteDataset(X_test, y_test, image_size=(224,224))

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(41)

model = ConvolutionalNetwork().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

# Training Loop
start_time = time.time()

epochs = 20
train_losses = []
test_losses = []
train_correct = []
test_correct = []

for epoch in range(epochs):
    model.train()

    trn_corr = 0
    total_train = 0

    for batch_idx, (X_train, y_train) in enumerate(train_loader, start=1):

        X_train, y_train = X_train.to(device), y_train.to(device)

        # Forward pass
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        # Compute training correctness
        predicted = torch.max(y_pred, 1)[1]
        batch_corr = (predicted == y_train).sum().item()

        trn_corr += batch_corr
        total_train += y_train.size(0)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # show first 10 batch loss only
        if batch_idx == 10:
            print(f"Epoch {epoch+1}/{epochs} | Batch: {batch_idx} | Loss: {loss.item():.4f}")

    train_losses.append(loss.item())
    train_acc = trn_corr / total_train * 100
    train_correct.append(train_acc)

    # TEST
    model.eval()
    tst_corr = 0
    test_loss = 0
    total_test = 0

    with torch.no_grad():
        for X_test, y_test in test_loader:

            X_test, y_test = X_test.to(device), y_test.to(device)
            y_val = model(X_test)

            loss = criterion(y_val, y_test)
            test_loss += loss.item()

            predicted = torch.max(y_val, 1)[1]
            tst_corr += (predicted == y_test).sum().item()
            total_test += y_test.size(0)

    test_losses.append(test_loss / len(test_loader))
    test_acc = tst_corr / total_test * 100
    test_correct.append(test_acc)

    print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

# Save the trained model 
os.makedirs("model", exist_ok=True)
torch.save(
    model.state_dict(),
    "model/final_satellite_cnn_5channel.pth"
)
print("Final model saved to model/final_satellite_cnn_5channel.pth")
