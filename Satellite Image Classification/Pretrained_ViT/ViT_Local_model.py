
# USING THIS AS A REFERENCE
# 
# https://www.geeksforgeeks.org/deep-learning/building-a-vision-transformer-from-scratch-in-pytorch

# Dataset used:
# https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification/data
# 
# Satellite image Classification Dataset-RSI-CB256 , This dataset has 4 different classes mixed from Sensors and google map snapshot
# Labels and quantities:
# Cloudy 1500
# Desert 1131
# Green_Area 1500
# Water 1500
import platform
import psutil
import torch
import sys

# Import other key packages safely
try:
    import timm
except ImportError:
    timm = None

try:
    import safetensors
except ImportError:
    safetensors = None

try:
    import sklearn
except ImportError:
    sklearn = None

try:
    import numpy
except ImportError:
    numpy = None


def get_windows_version():
    if platform.system() == "Windows":
        build = sys.getwindowsversion().build
        if build >= 22000:
            return "Windows 11"
        else:
            return "Windows 10 or earlier"
    else:
        return platform.system()


def get_detailed_cpu_info():
    info = {}

    # CPU brand and architecture
    info['Processor'] = platform.processor()
    info['Machine'] = platform.machine()

    # Physical cores
    info['Physical cores'] = psutil.cpu_count(logical=False)

    # Logical processors (threads)
    info['Logical processors'] = psutil.cpu_count(logical=True)

    # CPU frequency info per core
    freq = psutil.cpu_freq(percpu=True)
    if freq:
        info['CPU frequency per core (MHz)'] = [f.current for f in freq]
    else:
        info['CPU frequency per core (MHz)'] = None

    # Average CPU frequency
    avg_freq = psutil.cpu_freq()
    info['Average CPU frequency (MHz)'] = avg_freq.current if avg_freq else None

    return info


def get_system_info():
    info = get_detailed_cpu_info()

    # RAM information (GB)
    mem = psutil.virtual_memory()
    info['RAM_GB'] = round(mem.total / (1024 ** 3), 2)

    # Operating System details with accurate Windows version detection
    info['OS_System'] = platform.system()
    info['OS_Version'] = get_windows_version() if info['OS_System'] == "Windows" else platform.version()
    info['OS_Node'] = platform.node()
    info['OS_Release'] = platform.release()
    info['OS_Machine'] = platform.machine()
    info['OS_Processor'] = platform.processor()

    # Python version
    info['Python_Version'] = platform.python_version()

    # Package versions where available
    info['PyTorch_Version'] = torch.__version__
    info['Timm_Version'] = timm.__version__ if timm else "Not installed"
    info['Safetensors_Version'] = safetensors.__version__ if safetensors else "Not installed"
    info['Scikit-learn_Version'] = sklearn.__version__ if sklearn else "Not installed"
    info['NumPy_Version'] = numpy.__version__ if numpy else "Not installed"

    # CUDA info
    info['CUDA_Available'] = torch.cuda.is_available()
    info['CUDA_Version'] = torch.version.cuda if torch.cuda.is_available() else None
    info['GPU_Name'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

    return info


if __name__ == "__main__":
    system_info = get_system_info()
    for k, v in system_info.items():
        print(f"{k}: {v}")

import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # Batch size, Channels, Height, Width of the input tensor
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
# 2. Adding Positional Embeddings
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, seq_len):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len + 1, embed_dim))  # Adjusted for [CLS] token

    def forward(self, x):
        return x + self.pos_embed
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        # x: (B, S, E) where
        # B = batch size,
        # S = sequence length (number of patches or tokens),
        # E = embedding dimension (feature size per token)
        out, _ = self.attn(x, x, x)
        return out

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, embed_dim)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)        


    def forward(self, x):

        # Apply LayerNorm then Multi-Head Self-Attention to capture contextual relationships across tokens
        x = x + self.attn(self.norm1(x))
        # Residual connection adds input back after attention

        # Apply LayerNorm then position-wise Feed-Forward Network (MLP) for non-linear token-wise transformation
        x = x + self.mlp(self.norm2(x))
        # Residual connection again
        
        return x
    
    
'''
GELU attempt that didn't yield good results

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(0.1)
)

'''
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=10, embed_dim=768, num_heads=8, depth=6, mlp_dim=1024):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, (img_size // patch_size) ** 2)
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim) for _ in range(depth)
        ])
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.mlp_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embedding(x)  # shape: (B, num_patches, embed_dim)
        
        ##### Added by Archit
        if x.size(1) != self.pos_encoding.pos_embed.size(1): pos_embed = self.pos_encoding.pos_embed[:, :x.size(1), :]

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)

        # Adjust positional encoding to input sequence length
        if x.size(1) != self.pos_encoding.pos_embed.size(1):
            pos_embed = self.pos_encoding.pos_embed[:, :x.size(1), :]
        else:
            pos_embed = self.pos_encoding.pos_embed

        x = x + pos_embed

        for block in self.transformer_blocks:
            x = block(x)

        return self.mlp_head(x[:, 0])
