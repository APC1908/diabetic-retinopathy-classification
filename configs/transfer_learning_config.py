"""
Configuration for the EfficientNetV2-S transfer learning model.
"""

# Input/output configuration
IMAGE_SIZE = 380  # Input image size
NUM_CLASSES = 5  # 0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative DR

# Training configuration
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5

# Model configuration
FREEZE_LAYERS = -30  # Number of layers to unfreeze from the end (-30 means unfreeze last 30 layers)
DENSE_UNITS = 512  # Number of units in the dense layer
DROPOUT_RATE_1 = 0.3  # Dropout rate after global average pooling
DROPOUT_RATE_2 = 0.4  # Dropout rate after dense layer

# Class weights to handle imbalance (approximate distribution)
CLASS_WEIGHTS = {
    0: 0.2,  # No DR (most common)
    1: 1.8,  # Mild
    2: 0.8,  # Moderate
    3: 2.0,  # Severe
    4: 2.2   # Proliferative (least common)
}

# Loss function configuration
FOCAL_LOSS_ALPHA = 0.25
FOCAL_LOSS_GAMMA = 2.0
HYBRID_LOSS_ALPHA = 0.5  # Weight for Focal Loss
HYBRID_LOSS_BETA = 0.5   # Weight for Dice Loss

# Test-time augmentation configuration
USE_TTA = True
TTA_NUM_AUGMENTATIONS = 5