"""
transfer_learning.py - EfficientNetV2-S transfer learning model for diabetic retinopathy classification
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetV2S

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

def build_efficientnetv2s_model():
    """
    Build and compile EfficientNetV2-S transfer learning model.
    
    Returns:
        tensorflow.keras.Model: Compiled model
    """
    # Load pre-trained EfficientNetV2-S
    base_model = EfficientNetV2S(
        weights='imagenet',
        include_top=False,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )
    
    # Freeze early layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # Build model with custom top
    inputs = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compile model with hybrid loss
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=HybridLoss(),
        metrics=['accuracy']
    )
    
    return model

def get_callbacks(model_checkpoint_path):
    """
    Define callbacks for training.
    
    Args:
        model_checkpoint_path: Path to save the best model
        
    Returns:
        list: List of callbacks
    """
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

def test_time_augmentation(model, image, num_augmentations=5):
    """
    Apply test-time augmentation for more robust predictions.
    
    Args:
        model: Trained model
        image: Input image
        num_augmentations: Number of augmentations to apply
        
    Returns:
        numpy.ndarray: Averaged prediction
    """
    # Original image prediction
    orig_pred = model.predict(np.expand_dims(image, axis=0))[0]
    
    predictions = [orig_pred]
    
    # Horizontal flip
    flipped = np.fliplr(image)
    flip_pred = model.predict(np.expand_dims(flipped, axis=0))[0]
    predictions.append(flip_pred)
    
    # Slight rotations
    for angle in [-5, 5, -10, 10]:
        if len(predictions) >= num_augmentations:
            break
            
        rotated = tf.keras.preprocessing.image.apply_affine_transform(
            image, theta=angle, fill_mode='constant', cval=0.0
        )
        rot_pred = model.predict(np.expand_dims(rotated, axis=0))[0]
        predictions.append(rot_pred)
    
    # Average predictions
    avg_pred = np.mean(predictions, axis=0)
    
    return avg_pred

def train_model(train_data_dir, val_data_dir, model_save_path):
    """
    Train the EfficientNetV2-S model on diabetic retinopathy dataset.
    
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
    model = build_efficientnetv2s_model()
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

def evaluate_model(model, test_data_dir, use_tta=True):
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained model
        test_data_dir: Directory containing test data
        use_tta: Whether to use test-time augmentation
        
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
    
    if not use_tta:
        # Standard evaluation
        results = model.evaluate(test_generator, verbose=1)
        accuracy = results[1]  # Accuracy is the second metric
        
        # Get predictions for confusion matrix
        test_generator.reset()
        predictions = model.predict(test_generator, verbose=1)
        pred_classes = np.argmax(predictions, axis=1)
        true_labels = test_generator.classes[:len(pred_classes)]
    else:
        # Evaluation with test-time augmentation
        test_generator.reset()
        
        # Get all images and labels
        images = []
        labels = []
        
        for i in range(len(test_generator)):
            batch_images, batch_labels = test_generator[i]
            images.extend(batch_images)
            labels.extend(batch_labels)
            
            if len(images) >= len(test_generator.classes):
                break
        
        # Apply TTA to each image
        predictions = []
        for image in images:
            pred = test_time_augmentation(model, image)
            predictions.append(pred)
        
        # Convert to numpy arrays
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        # Calculate accuracy
        pred_classes = np.argmax(predictions, axis=1)
        true_labels = np.argmax(labels, axis=1)
        accuracy = np.mean(pred_classes == true_labels)
    
    # Create confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, pred_classes)
    
    return accuracy, cm

def save_model(model, save_path):
    """Save the trained model."""
    model.save(save_path)
    print(f"Model saved to {save_path}")