import streamlit as st
from PIL import Image
import torch
import timm
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
from Pretrained_ViT.ViT_Local_model import VisionTransformer
from CNN.model.model import ConvolutionalNetwork
import numpy as np

# ViT Block
vit = VisionTransformer(num_classes=4)
vit_data = torch.load("C:/Users/Jinal/Nextcloud/My Stuff/ImpDocs/Masters/Sapienza/Academic/Y1S1/FDS/FDS Project/Satellite Image Classification/Pretrained_ViT/Visual_Transformer_Local_LightDump.pth", map_location="cpu")
vit.load_state_dict(vit_data)
vit.eval()

# Pre Trained ResNet18
pretrained_resnet_data = torch.load("C:/Users/Jinal/Nextcloud/My Stuff/ImpDocs/Masters/Sapienza/Academic/Y1S1/FDS/FDS Project/Satellite Image Classification/Pretrained resnet18/sat_img_classifier_resnet18.pth")
pretrained_resnet = models.resnet18()
pretrained_resnet.fc = torch.nn.Linear(pretrained_resnet.fc.in_features, 4)
pretrained_resnet.load_state_dict(pretrained_resnet_data["model_state_dict"])
pretrained_resnet.eval()

# Pretrained ViT Block
pretrained_vit_data = torch.load("C:/Users/Jinal/Nextcloud/My Stuff/ImpDocs/Masters/Sapienza/Academic/Y1S1/FDS/FDS Project/Satellite Image Classification/Pretrained_ViT/Visual_Transformer_Pretrained_LightDump.pth", map_location="cpu")
pretrained_vit = timm.create_model('vit_base_patch16_224', pretrained=True)
pretrained_vit.head = torch.nn.Linear(pretrained_vit.head.in_features, 4)
pretrained_vit.load_state_dict(pretrained_vit_data)

# CNN Block
cnn = ConvolutionalNetwork(num_classes=4)
cnn_data = torch.load("C:/Users/Jinal/Nextcloud/My Stuff/ImpDocs/Masters/Sapienza/Academic/Y1S1/FDS/FDS Project/Satellite Image Classification/CNN/model/final_satellite_cnn_5channel.pth", map_location="cpu")
cnn.load_state_dict(cnn_data)
cnn.eval()

def image_preprocessing_cnn(img):
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

    return tensor_img

def image_preprocessing(image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    input_tensor = transform(image).unsqueeze(0)
    
    return input_tensor

def predict(image_tensor, model, idx_to_classes):
    with torch.no_grad():
        out = model(image_tensor)
        _, pred = torch.max(out, 1)
        predicted_class = idx_to_classes[pred.item()]
    
    return predicted_class


idx_to_classes = {v:k for k, v in pretrained_resnet_data["class_to_idx"].items()}

st.title("Sat Img Classifier")
upload = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])


if upload:
    col1, col2, col3, col4= st.columns(4)
        
    predicted_classes = []

    for model in [cnn, pretrained_resnet, vit, pretrained_vit]:
        if model == cnn:
            image = image_preprocessing_cnn(Image.open(upload).convert("RGB")) 
        else:
            image = image_preprocessing(Image.open(upload).convert("RGB"))
        predicted_classes.append(predict(image, model, idx_to_classes))

    display = Image.open(upload).convert("RGB")
    st.image(display, width="stretch")

    with col1:
        st.write(f"CNN:")
        st.write(f"**{predicted_classes[0]}**")
    
    with col2:
        st.write(f"Pretrained ResNet18:")
        st.write(f"**{predicted_classes[1]}**")
    
    with col3:
        st.write(f"ViT:")
        st.write(f"**{predicted_classes[2]}**")

    with col4:
        st.write(f"Pretrained ViT:")
        st.write(f"**{predicted_classes[3]}**")


