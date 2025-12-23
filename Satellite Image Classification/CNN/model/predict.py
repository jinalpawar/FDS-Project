"""
predict.py
----------
Loads a trained 5-channel CNN and predicts the class of a random satellite image.
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvolutionalNetwork(num_classes=len(label_to_class))  # adjust if needed

model.load_state_dict(
    torch.load("model/final_satellite_cnn_5channel.pth", map_location=device)
)
model = model.to(device)
model.eval()

# Pick random image
random_index = random.randint(0, len(image_paths) - 1)
random_image_path = image_paths[random_index]
true_label = labels[random_index]

# Load image
img = Image.open(random_image_path).convert("RGB")
img_np = np.array(img)

# Compute 5 channels
r = img_np[:, :, 0]
g = img_np[:, :, 1]
b = img_np[:, :, 2]
rg = r - g
gb = g - b

# Stack into (H, W, 5)
img_5ch = np.stack([r, g, b, rg, gb], axis=2)

# Convert to tensor → (1, 5, H, W)
tensor_img = (
    torch.tensor(img_5ch, dtype=torch.float32)
    .permute(2, 0, 1)
    .unsqueeze(0)
    / 255.0
)

# Resize
tensor_img = F.interpolate(
    tensor_img,
    size=(224, 224),
    mode="bilinear",
    align_corners=False
)

# Normalize (MATCH TRAINING)
mean = tensor_img.mean(dim=(2, 3), keepdim=True)
std = tensor_img.std(dim=(2, 3), keepdim=True) + 1e-7
tensor_img = (tensor_img - mean) / std

# Move to device
tensor_img = tensor_img.to(device)
with torch.no_grad():
    outputs = model(tensor_img)
    predicted_class = torch.argmax(outputs, dim=1).item()
print("Predicted:", label_to_class[predicted_class])
print("Actual:", label_to_class[true_label])

plt.imshow(img)
plt.title(
    f"Predicted: {label_to_class[predicted_class]} | "
    f"Actual: {label_to_class[true_label]}"
)
plt.axis("off")
plt.show()
