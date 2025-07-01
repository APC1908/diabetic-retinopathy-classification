"""
baseline_model.py - Optimal Baseline Model for Diabetic Retinopathy Classification
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

from src.loss_functions import HybridLoss

# Configuration
IMAGE_SIZE = 380  # Input image size
NUM_CLASSES = 5  # 0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative DR
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5

# Class weights to handle imbalance (approximate distribution)
CLASS_WEIGHTS = {
    0: 0.2,  # No DR (most common)
    1: 1.8,  # Mild
    2: 0.8,  # Moderate
    3: 2.0,  # Severe
    4: 2.2   # Proliferative (least common)
}

def squeeze_excite_block(input_tensor, ratio=16):
    """
    Squeeze and Excitation block for attention mechanism.
    
    Args:
        input_tensor: Input tensor
        ratio: Reduction ratio for the squeeze operation
        
    Returns:
        tensorflow.Tensor: Output tensor after applying SE block
    """
    filters = input_tensor.shape[-1]
    se_shape = (1, 1, filters)
    
    # Squeeze (Global Average Pooling)
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Reshape(se_shape)(se)
    
    # Excitation (Two Dense layers with bottleneck)
    se = layers.Dense(filters // ratio, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    
    # Scale the input
    x = layers.Multiply()([input_tensor, se])
    return x

def residual_block(x, filters, kernel_size=3, stride=1, use_se=True, se_ratio=16):
    """
    Residual block with optional Squeeze-Excitation attention.
    
    Args:
        x: Input tensor
        filters: Number of filters in the convolutional layers
        kernel_size: Size of the convolutional kernels
        stride: Stride for the first convolution
        use_se: Whether to use Squeeze-Excitation block
        se_ratio: Reduction ratio for SE block
        
    Returns:
        tensorflow.Tensor: Output tensor after applying residual block
    """
    # Store input
    shortcut = x
    
    # First convolution
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Second convolution
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Apply SE block if specified
    if use_se:
        x = squeeze_excite_block(x, ratio=se_ratio)
    
    # Handle shortcut connection (identity or projection)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    # Add shortcut connection
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    
    return x

def build_optimal_baseline_model():
    """Build and compile the optimal baseline model."""
    # Input layer
    inputs = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    
    # Initial block
    x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    
    # Residual stages with increasing filter counts
    # Stage 1: 64 filters
    for i in range(2):
        x = residual_block(x, 64, stride=1 if i > 0 else 1)
    
    # Stage 2: 128 filters
    for i in range(2):
        x = residual_block(x, 128, stride=2 if i == 0 else 1)
    
    # Stage 3: 256 filters
    for i in range(2):
        x = residual_block(x, 256, stride=2 if i == 0 else 1)
    
    # Stage 4: 512 filters
    for i in range(2):
        x = residual_block(x, 512, stride=2 if i == 0 else 1)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compile model with hybrid loss
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=HybridLoss(),
        metrics=['accuracy']
    )
    
    return model

def get_callbacks(model_checkpoint_path):
    """Define callbacks for training."""
    callbacks_list = [
        callbacks.ModelCheckpoint(
            filepath=model_checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-6,
            verbose=1
        )
    ]
    return callbacks_list

def train_model(train_data_dir, val_data_dir, model_save_path):
    """
    Train the optimal baseline model on diabetic retinopathy dataset.
    
    Args:
        train_data_dir: Directory containing training data
        val_data_dir: Directory containing validation data
        model_save_path: Path to save the trained model
        
    Returns:
        tuple: Trained model and training history
    """
    from src.data_preprocessing import get_train_data_generator, get_validation_data_generator
    
    # Create data generators
    train_datagen = get_train_data_generator(batch_size=BATCH_SIZE)
    val_datagen = get_validation_data_generator()
    
    # Load data
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    # Build model
    model = build_optimal_baseline_model()
    model.summary()
    
    # Define callbacks
    callbacks_list = get_callbacks(model_save_path)
    
    # Train model
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks_list,
        class_weight=CLASS_WEIGHTS
    )
    
    return model, history

def evaluate_model(model, test_data_dir):
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained model
        test_data_dir: Directory containing test data
        
    Returns:
        tuple: Test accuracy and confusion matrix
    """
    from src.data_preprocessing import get_validation_data_generator
    
    # Create test data generator
    test_datagen = get_validation_data_generator()
    
    # Load test data
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    # Evaluate model
    results = model.evaluate(test_generator, verbose=1)
    accuracy = results[1]  # Accuracy is the second metric
    
    # Get predictions for confusion matrix
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    pred_classes = np.argmax(predictions, axis=1)
    true_labels = test_generator.classes[:len(pred_classes)]
    
    # Create confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, pred_classes)
    
    return accuracy, cm

def save_model(model, save_path):
    """Save the trained model."""
    model.save(save_path)
    print(f"Model saved to {save_path}")