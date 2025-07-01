"""
Configuration for the optimal baseline model.
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
USE_SE_BLOCKS = True  # Whether to use Squeeze-Excitation blocks
SE_RATIO = 16  # Reduction ratio for SE blocks

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