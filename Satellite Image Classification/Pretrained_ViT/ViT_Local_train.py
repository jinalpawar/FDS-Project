from ViT_Local import VisionTransformer
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset


import numpy as np
from sklearn.model_selection import train_test_split

path = "satellite_data/"
batch_size = 32
height = 224
width = 224
val_split_ratio = 0.2

# Load full dataset
full_dataset = datasets.ImageFolder(root=path)
print(f"Full dataset: {len(full_dataset)} samples")
print(f"Classes: {full_dataset.classes}")

# STRATIFIED SPLIT - Guarantees all classes in train AND val
train_indices, val_indices = [], []
for label in range(len(full_dataset.classes)):
    # Get all indices for this class
    class_indices = [i for i, (_, lbl) in enumerate(full_dataset.samples) if lbl == label]
    
    # Split 80/20 within each class
    n_val = int(len(class_indices) * val_split_ratio)
    train_idx, val_idx = train_test_split(class_indices, test_size=n_val, random_state=42)
    
    train_indices.extend(train_idx)
    val_indices.extend(val_idx)

# Shuffle indices for proper DataLoader behavior
np.random.shuffle(train_indices)
np.random.shuffle(val_indices)

print(f"Train: {len(train_indices)} samples")
print("Train class dist:", np.bincount([full_dataset.samples[i][1] for i in train_indices]))
print(f"Val:   {len(val_indices)} samples") 
print("Val class dist:  ", np.bincount([full_dataset.samples[i][1] for i in val_indices]))

# Your existing transforms and AugmentedDataset (unchanged)
train_transform = transforms.Compose([
    transforms.Resize((height, width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize((height, width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_subset = Subset(full_dataset, train_indices)
val_subset = Subset(full_dataset, val_indices)

class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        x = self.transform(x)
        return x, y
    def __len__(self):
        return len(self.dataset)

train_dataset = AugmentedDataset(train_subset, train_transform)
val_dataset = AugmentedDataset(val_subset, val_transform)

# DataLoaders - WINDOWS SAFE (num_workers=0 prevents 95% of hangs)
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=0,           # ← FIXED: 0 prevents multiprocessing deadlock
    pin_memory=False,        # ← FIXED: False prevents GPU pinning issues
    persistent_workers=False # ← FIXED: False prevents worker hang
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    shuffle=False,           # ← CORRECT: False for validation
    num_workers=0,           # ← FIXED: 0 prevents validation hangs
    pin_memory=False,        # ← FIXED: False for stability
    drop_last=False,         # ← ADDED: Keep last incomplete batch
    persistent_workers=False
)

print("All 4 classes guaranteed in both train and val!")

# Make sure augmentation is working properly by showing random images from augmented dataset

import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

def show_augmented_images(dataset, n=5):
    fig, axes = plt.subplots(1, n, figsize=(15, 5))
    for i in range(n):
        img, label = dataset[i]
        # Convert tensor to numpy image with proper normalization reversal for display
        img_show = img.permute(1, 2, 0).numpy()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img_show = std * img_show + mean
        img_show = img_show.clip(0, 1)
        axes[i].imshow(img_show)
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    plt.show()

show_augmented_images(train_dataset, n=5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
##### This code is the original locally trained and should be run at standard resolution (not 224x224)
# for replication of the results make sure the pipiline is using standard resolution (written in a comment)

import time
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import precision_score, recall_score, f1_score

##### Parameters:
##### original: lr 0.001 / epoch 5 / batch_size 32
lr = 1e-3  # ← INCREASED: From 3e-4 (too low for from-scratch)
epoch_count = 20
batch_size = 32
weight_decay = 0.05

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = VisionTransformer(num_classes=4)  # ← FIXED: num_classes=4 (was 10)
model.to(device)

# ← NEW: Class weights for imbalance (Desert has fewer samples)
import numpy as np
train_labels = [y for _, y in train_dataset]
class_counts = np.bincount(train_labels)
class_weights = 1.0 / class_counts
class_weights = torch.FloatTensor(class_weights).to(device)
print("Class weights:", class_weights.cpu().numpy())

criterion = nn.CrossEntropyLoss(weight=class_weights)  # ← FIXED: Weighted loss
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


# Start total timer
total_start_time = time.time()

# To be used later for graphs
class TrainingHistory:
    def __init__(self):
        # Initialize with epoch 0 values all set to zero
        self.losses = [0.0]
        self.accuracies = [0.0]
        self.precisions = [0.0]
        self.recalls = [0.0]
        self.f1s = [0.0]
        self.durations = [0.0]

# Create history object before training
history = TrainingHistory()

# ensure consistency in epoch count, use the same digits count so that i have formatted outputs
# i didn't bother running it with 100 epoch count, but it does work with 1 to 20 epochs
width = len(str(epoch_count))

print(f"Epoch [{0:0{width}d}/{epoch_count}] "
      f"Loss: {0.0:.4f} Accuracy: {0.0:.4f} Precision: {0.0:.4f} "
      f"Recall: {0.0:.4f} F1-score: {0.0:.4f} "
      f"Epoch Time: {0.00:.2f} seconds")



for epoch in range(epoch_count):  # Train for epoch_count epochs
    epoch_start_time = time.time()  # Start timer for this epoch

    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(train_loader)
    accuracy = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    epoch_end_time = time.time()  # End timer for this epoch
    epoch_duration = epoch_end_time - epoch_start_time

    # Store metrics and duration in history object
    history.losses.append(epoch_loss)
    history.accuracies.append(accuracy)
    history.precisions.append(precision)
    history.recalls.append(recall)
    history.f1s.append(f1)
    history.durations.append(epoch_duration)


    print(f"Epoch [{epoch+1:0{width}d}/{epoch_count}] "
        f"Loss: {epoch_loss:.4f} Accuracy: {accuracy:.4f} Precision: {precision:.4f} "
        f"Recall: {recall:.4f} F1-score: {f1:.4f} "
        f"Epoch Time: {epoch_duration:.2f} seconds")


# End total timer
total_end_time = time.time()
total_duration = total_end_time - total_start_time

print(f"Total Training Time: {total_duration:.2f} seconds")

##### HYBRID VERSION, using many parts of the final pretrained model so that they can be both tested in the same conditions
# make sure this one is using 224x244 images to test 1:1 with the pretrained, check transforms in the above cells


# To better understand the metrics used:
# 
# Loss: Measures the model's error in prediction by quantifying how far the predictions are from the true labels using the loss function (in this case cross-entropy). Lower loss indicates better fit to training data. Loss is crucial during training as it is the value optimized by the model's algorithm to improve performance
# (The lower the better)
# 
# Accuracy: Represents the proportion of correct predictions out of all predictions made. It's a simple and intuitive measure of overall correctness but can be misleading when classes training data is imbalanced in the ratio (in this dataset it's fairly even, except for desert which has roughly 80% of the data count)
# Cloudy 1500 / Desert 1131 / Green_Area 1500 / Water 1500
# 
# Precision: Indicates how many predicted positive cases were actually positive. This metric is especially important when false positives are costly, such as in medical diagnoses or fraud detection, ensuring that positive predictions are reliable
# 
# Recall: Measures how many actual positive cases were correctly identified. It's critical in scenarios where missing positive cases has a high cost (false negatives), like disease screening, emphasizing sensitivity
# 
# F1-score: The harmonic mean of precision and recall, providing a balanced metric especially valuable when you need to account for both false positives and false negatives, and when class distribution is uneven. It summarizes model performance in one number
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# use history (from the earlier model's training results) to compute graphs
epochs = range(0, len(history.losses))



# Plot Loss (Loss may not cap at 1.0 but for consistency, set max to 1.0)
plt.figure(figsize=(10, 6))
plt.plot(epochs, history.losses, 'r-', label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid(True)
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
#ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))  
plt.xlim(0, epoch_count)
#plt.ylim(0, 1.0)  # y-axis max 1.0
plt.show()

# Plot Accuracy
plt.figure(figsize=(10, 6))
plt.plot(epochs, history.accuracies, 'b-', label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Over Epochs')
plt.legend()
plt.grid(True)
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
plt.xlim(0, epoch_count)
plt.ylim(0, 1.0)
plt.show()

# Plot Precision, Recall, and F1-score together
plt.figure(figsize=(10, 6))
plt.plot(epochs, history.precisions, 'g-', label='Precision')
plt.plot(epochs, history.recalls, 'm-', label='Recall')
plt.plot(epochs, history.f1s, 'c-', label='F1-score')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Precision, Recall, F1-score Over Epochs')
plt.legend()
plt.grid(True)
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
plt.xlim(0, epoch_count)
plt.ylim(0, 1.0)
plt.show()


# Plot Epoch Duration
plt.figure(figsize=(10, 6))
plt.plot(epochs, history.durations, 'k-', label='Epoch Duration (seconds)')
plt.xlabel('Epoch')
plt.ylabel('Duration (s)')
plt.title('Epoch Duration Over Time')
plt.legend()
plt.grid(True)
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))  # set tick interval to 1
plt.xlim(0, epoch_count)
plt.ylim(bottom=0) # start y-axis at 0
plt.show()

from sklearn.metrics import confusion_matrix
import numpy as np

# After the epoch evaluation (after collecting all_preds and all_labels)

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)

print("Confusion Matrix:")
print(cm)


epoch_loss = running_loss / len(train_loader)
accuracy = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

# New code for confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)

print(f"Epoch [{epoch+1}/{epoch+1}] Loss: {epoch_loss:.4f} Accuracy: {accuracy:.4f} Precision: {precision:.4f} Recall: {recall:.4f} F1-score: {f1:.4f}")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# Assuming 'all_labels' and 'all_preds' contain true and predicted labels respectively
labels_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()

# Labeling the axes with class names
num_classes = cm.shape[0]
plt.xticks(np.arange(num_classes), labels=labels_names, rotation=45)
plt.yticks(np.arange(num_classes), labels=labels_names)

# Axis labels
plt.xlabel('Predicted label')
plt.ylabel('True label')

# Adding counts on the plot
thresh = cm.max() / 2
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.show()


# From this confusion matrix we can confidently say there's a bit of error when predicting labels 2 and 3 (Green_Area and Water), labels 0 and 1 show a similar problem (Cloudy and Desert), but not in a very accentuated way
##### for testing purposes only

from PIL import Image
from collections import defaultdict
import os

# full_dataset is ImageFolder with transform=transform (might convert images, so ignore it here)

# Map from class index to class name
class_names = full_dataset.classes

# Store counts of resolutions per class
resolutions_per_class = {cls: defaultdict(int) for cls in class_names}

# Iterate over dataset samples (image paths are in dataset.imgs or dataset.samples)
for img_path, label in full_dataset.samples:
    with Image.open(img_path) as img:
        # Get image size (width, height)
        size = img.size
    class_name = class_names[label]
    # Increment count for this resolution for this class
    resolutions_per_class[class_name][size] += 1

# Print summary for each class
for cls, res_counts in resolutions_per_class.items():
    print(f"Class '{cls}':")
    for res, count in sorted(res_counts.items()):
        print(f"  Resolution {res[0]}x{res[1]} : {count} images")
    print()


# Results in locally trained baseline (before data augmentation):
# 
# From the results from being run on a 4060 Mobile GPU we can confidently say that the model shows marginal changes after the 10th epoch, but usually goes from 90 to 92.5% accuracy at around 13th epoch, it might actually be counterproductive to run the model for longer since it seems to degenerate into slightly worse performance
# 
# Results in locally trained with data augmentation:
# 
# Precision down from 90/92% to 88%, loss is consistently the same throughout the whole train sequence (0.35 average on augmented, while goes down to 0.2 in baseline), this is suggesting that something is wrong with augmentation, will require further testing
# Tried disabling roation (having a 64x64 images there isn't much to rotate at all) and it had no effect, but maybe testing for longer (50 epochs would? we are training at twice the speed, it's only fair to run the model for twice as long to give it the same time to shine)
# Apparently not, doesn't even touch the 90% accuracy, and something odd happened at epoch 37, train time jumped from 20 to 28.8s for no reason
# Epoch [37/50] Loss: 0.3286 Accuracy: 0.8744 Precision: 0.8667 Recall: 0.8744 F1-score: 0.8648 Epoch Time: 28.82 seconds
# Dataloades have workers set at 4
# 
# Resizing to 244x244 doesn't help, it only slows down the model and needs 70s on average for each epoch, turning off flipping and general augmentation
# Kept jitter on since it should be only a minimal touch with a deep effect, it's supposed to scramble things off a bit, but it might be a problem on purely color based datasets? as usual tests indicate that accuracy goes up to 89% and then down to 87%, turning off the data augmentation for another test
# Noticed that i forgot to add " transforms.CenterCrop(224), " to my transform, which means that this might have been compromising my results, testing normal usage without it to confirm it aligns with baseline
# 
# ODD behavior i didn't foresee:
# Turning off that data transform actually helped, so augmentation actually hurt the performance enough to push it much further down than expected. Results from baseline without center crop
# Epoch [20/20] Loss: 0.1612 Accuracy: 0.9505 Precision: 0.9499 Recall: 0.9505 F1-score: 0.9484 Epoch Time: 14.48 seconds
# Perhaps a 20 epoch was too little, 50 epochs trial:
# Epoch [42/50] Loss: 0.1161 Accuracy: 0.9607 Precision: 0.9599 Recall: 0.9607 F1-score: 0.9598 Epoch Time: 14.32 seconds
# Epoch [50/50] Loss: 0.1808 Accuracy: 0.9347 Precision: 0.9325 Recall: 0.9347 F1-score: 0.9326 Epoch Time: 14.02 seconds
# 96% is quite the result given the dataset, but the model tends to go back and forth and show a bit of instability on the long run, dipping into the 88% territory:
# Epoch [27/50] Loss: 0.3760 Accuracy: 0.8883 Precision: 0.8843 Recall: 0.8883 F1-score: 0.8851 Epoch Time: 14.60 seconds
# 
# Swapped my RELU for GELU:
#         self.mlp = nn.Sequential(
#             nn.Linear(embed_dim, mlp_dim),
#             nn.ReLU(),
#             nn.Linear(mlp_dim, embed_dim)
#         )
# 
#         self.mlp = nn.Sequential(
#             nn.Linear(embed_dim, mlp_dim),
#             nn.GELU(),
#             nn.Dropout(0.1),
#             nn.Linear(mlp_dim, embed_dim),
#             nn.Dropout(0.1)
# )
# Apparently it yields slightly worse results, so RELU stays
# Epoch [16/20] Loss: 0.1726 Accuracy: 0.9458 Precision: 0.9445 Recall: 0.9458 F1-score: 0.9438 Epoch Time: 14.40 seconds
# 
# Labels and quantities:
# Cloudy 1500
# Desert 1131
# Green_Area 1500
# Water 1500
# 
# Confirmed through dataset looping:
# Class 'cloudy':  Resolution 256x256 : 1500 images
# Class 'desert':  Resolution 256x256 : 1131 images
# Class 'green_area':  Resolution 64x64 : 1500 images
# Class 'water':  Resolution 64x64 : 1500 images
# 
# A Visual Transformer tends to be data hungry, so perhaps there's an inherent flaw in attempting this with a small dataset with small images
# VALIDATION EVALUATION (after training completes)
print("FINAL VALIDATION RESULTS")

model.eval()
val_running_loss = 0.0
val_all_preds = []
val_all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:  # Uses your existing val_loader
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_running_loss += loss.item()
        
        _, preds = torch.max(outputs, 1)
        val_all_preds.extend(preds.cpu().numpy())
        val_all_labels.extend(labels.cpu().numpy())

# Same metrics format
from sklearn.metrics import precision_score, recall_score, f1_score
val_loss = val_running_loss / len(val_loader)
val_accuracy = (torch.tensor(val_all_preds) == torch.tensor(val_all_labels)).float().mean().item()
val_precision = precision_score(val_all_labels, val_all_preds, average='weighted', zero_division=0)
val_recall = recall_score(val_all_labels, val_all_preds, average='weighted', zero_division=0)
val_f1 = f1_score(val_all_labels, val_all_preds, average='weighted', zero_division=0)

print(f"Loss: {val_loss:.4f} Accuracy: {val_accuracy:.4f} Precision: {val_precision:.4f}")
print(f"Recall: {val_recall:.4f} F1-score: {val_f1:.4f}")
print(f"Validation samples: {len(val_all_labels)}")
print("="*80)

# DEBUG: Check what's actually happening
print("Unique true labels:", np.unique(val_all_labels))
print("Unique predictions:", np.unique(val_all_preds))
print("Prediction counts:", np.bincount(val_all_preds))
print("Label counts:", np.bincount(val_all_labels))




# VALIDATION EVALUATION (after training completes)
print("FINAL VALIDATION RESULTS")

model.eval()
val_running_loss = 0.0
val_all_preds = []
val_all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_running_loss += loss.item()
        
        _, preds = torch.max(outputs, 1)
        val_all_preds.extend(preds.cpu().numpy())
        val_all_labels.extend(labels.cpu().numpy())

# Metrics
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
val_loss = val_running_loss / len(val_loader)
val_accuracy = (torch.tensor(val_all_preds) == torch.tensor(val_all_labels)).float().mean().item()
val_precision = precision_score(val_all_labels, val_all_preds, average='weighted', zero_division=0)
val_recall = recall_score(val_all_labels, val_all_preds, average='weighted', zero_division=0)
val_f1 = f1_score(val_all_labels, val_all_preds, average='weighted', zero_division=0)

print(f"Loss: {val_loss:.4f} Accuracy: {val_accuracy:.4f} Precision: {val_precision:.4f}")
print(f"Recall: {val_recall:.4f} F1-score: {val_f1:.4f}")
print(f"Validation samples: {len(val_all_labels)}")
print("="*80)

# FIXED CONFUSION MATRIX PLOT
labels_names = full_dataset.classes
print(f"Classes detected: {labels_names}")

cm = confusion_matrix(val_all_labels, val_all_preds)
print(f"Confusion matrix shape: {cm.shape}")

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Validation Confusion Matrix')
plt.colorbar()

num_classes = cm.shape[0]
plt.xticks(np.arange(num_classes), labels_names[:num_classes], rotation=45)
plt.yticks(np.arange(num_classes), labels_names[:num_classes])

thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.show()

print("=== FULL DATASET CHECK ===")
print(f"Total samples: {len(full_dataset)}")
print(f"Classes: {full_dataset.classes}")

print("Class counts:")
for i, cls in enumerate(full_dataset.classes):
    count = sum(1 for _, label in full_dataset.samples if label == i)
    print(f"  {cls}: {count}")

print("\n=== SPLIT INDICES CHECK ===")
print(f"Train size: {len(train_indices)}, Val size: {len(val_indices)}")

train_labels = [full_dataset.samples[i][1] for i in train_indices]
val_labels = [full_dataset.samples[i][1] for i in val_indices]

print("Train unique labels:", sorted(set(train_labels)))
print("Train counts:", np.bincount(train_labels))
print("Val unique labels: ", sorted(set(val_labels)))
print("Val counts:     ", np.bincount(val_labels))
