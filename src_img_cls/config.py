# config.py

# Hugging Face Configuration
# Replace "YOUR_HUGGINGFACE_TOKEN" with your actual Hugging Face token
# You can find your token in your Hugging Face account settings
# HUGGINGFACE_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx" 

# Basic Model Parameters
IMG_SIZE = 224
PATCH_SIZE = 16
IN_CHANNELS = 3
NUM_CLASSES = 1000

# ViT Model Variants Configurations
VIT_TINY_CONFIG = {
    "embed_dim": 192,
    "depth": 12,
    "num_heads": 3,
    "mlp_ratio": 4.0,
    "qkv_bias": True,
}

VIT_SMALL_CONFIG = {
    "embed_dim": 384,
    "depth": 12,
    "num_heads": 6,
    "mlp_ratio": 4.0,
    "qkv_bias": True,
}

VIT_BASE_CONFIG = {
    "embed_dim": 768,
    "depth": 12,
    "num_heads": 12,
    "mlp_ratio": 4.0,
    "qkv_bias": True,
}

VIT_LARGE_CONFIG = {
    "embed_dim": 1024,
    "depth": 24,
    "num_heads": 16,
    "mlp_ratio": 4.0,
    "qkv_bias": True,
}

# DeiT Model Variants Configurations
DEIT_TINY_CONFIG = {
    "embed_dim": 192,
    "depth": 12,
    "num_heads": 3,
    "mlp_ratio": 4.0,
    "qkv_bias": True,
}

DEIT_SMALL_CONFIG = {
    "embed_dim": 384,
    "depth": 12,
    "num_heads": 6,
    "mlp_ratio": 4.0,
    "qkv_bias": True,
}

DEIT_BASE_CONFIG = {
    "embed_dim": 768,
    "depth": 12,
    "num_heads": 12,
    "mlp_ratio": 4.0,
    "qkv_bias": True,
}

# Swin Transformer Variants Configurations
SWIN_TINY_CONFIG = {
    "embed_dim": 96,
    "depths": [2, 2, 6, 2],
    "num_heads": [3, 6, 12, 24],
    "window_size": 7,
}

SWIN_SMALL_CONFIG = {
    "embed_dim": 96,
    "depths": [2, 2, 18, 2],
    "num_heads": [3, 6, 12, 24],
    "window_size": 7,
}

SWIN_BASE_CONFIG = {
    "embed_dim": 128,
    "depths": [2, 2, 18, 2],

    "num_heads": [4, 8, 16, 32],
    "window_size": 7,
}

SWIN_LARGE_CONFIG = {
    "embed_dim": 192,
    "depths": [2, 2, 18, 2],
    "num_heads": [6, 12, 24, 48],
    "window_size": 7,
}


# Testing Configuration
# SAMPLE_NUM = 50000 # Number of sample images to test on MAX 50000
SAMPLE_NUM = 4096 # Number of sample images to test on MAX 50000
BATCH_SIZE = 512 # Batch size for data loader
