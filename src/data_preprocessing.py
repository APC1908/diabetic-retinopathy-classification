"""
data_preprocessing.py - Preprocessing utilities for diabetic retinopathy images
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import albumentations as A

def create_circular_mask(height, width, center=None, radius=None):
    """
    Create a circular mask for retinal images.
    
    Args:
        height: Height of the image
        width: Width of the image
        center: Center of the circle (default: center of the image)
        radius: Radius of the circle (default: 80% of the smaller dimension)
        
    Returns:
        numpy.ndarray: Binary mask with circular region as 1 and outside as 0
    """
    if center is None:
        center = (width // 2, height // 2)
    if radius is None:
        radius = min(width, height) // 2 * 0.8
        
    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    
    mask = dist_from_center <= radius
    return mask

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance contrast.
    
    Args:
        image: Input image
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        
    Returns:
        numpy.ndarray: CLAHE enhanced image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[..., 0] = clahe.apply(lab[..., 0])
    
    # Convert back to RGB
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return enhanced

def preprocess_image(image, target_size=(380, 380), apply_mask=True, enhance_contrast=True):
    """
    Preprocess a retinal image for model input.
    
    Args:
        image: Input image
        target_size: Target size for resizing
        apply_mask: Whether to apply circular masking
        enhance_contrast: Whether to apply CLAHE enhancement
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Resize image
    if image.shape[0] != target_size[0] or image.shape[1] != target_size[1]:
        image = cv2.resize(image, target_size)
    
    # Apply circular mask if requested
    if apply_mask:
        mask = create_circular_mask(image.shape[0], image.shape[1])
        image = image.copy()  # Create a copy to avoid modifying the original
        image[~mask] = 0  # Set pixels outside the mask to black
    
    # Apply CLAHE if requested
    if enhance_contrast:
        image = apply_clahe(image)
    
    # Normalize pixel values to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    return image

def get_train_data_generator(batch_size=32, rotation_range=20, width_shift_range=0.1,
                           height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True,
                           brightness_range=(0.8, 1.2)):
    """
    Create a data generator with augmentation for training.
    
    Args:
        batch_size: Batch size
        rotation_range: Degree range for random rotations
        width_shift_range: Fraction of width for random horizontal shifts
        height_shift_range: Fraction of height for random vertical shifts
        zoom_range: Range for random zoom
        horizontal_flip: Whether to randomly flip images horizontally
        brightness_range: Range for random brightness adjustment
        
    Returns:
        ImageDataGenerator: Data generator for training
    """
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_image,
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        brightness_range=brightness_range,
        fill_mode='constant',
        cval=0
    )
    
    return train_datagen

def get_validation_data_generator():
    """
    Create a data generator without augmentation for validation/testing.
    
    Returns:
        ImageDataGenerator: Data generator for validation/testing
    """
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_image
    )
    
    return val_datagen

def get_albumentation_transforms(p=0.5):
    """
    Create a pipeline of image transformations using Albumentations.
    
    Args:
        p: Probability of applying each transform
        
    Returns:
        A.Compose: Composed transformations
    """
    transforms = A.Compose([
        A.RandomRotate90(p=p),
        A.HorizontalFlip(p=p),
        A.VerticalFlip(p=p),
        A.RandomBrightnessContrast(p=p),
        A.GaussNoise(p=p),
        A.GaussianBlur(p=p),
        A.ElasticTransform(p=p, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        A.GridDistortion(p=p),
        A.OpticalDistortion(p=p)
    ])
    
    return transforms