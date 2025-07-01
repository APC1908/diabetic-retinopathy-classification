"""
loss_functions.py - Custom loss functions for diabetic retinopathy classification
"""

import tensorflow as tf
from tensorflow.keras import backend as K

class FocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss for addressing class imbalance.
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    where p_t is the model's estimated probability for the class with true label y.
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for the rare class
            gamma: Focusing parameter that reduces the loss contribution from easy examples
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        
    def __call__(self, y_true, y_pred, sample_weight=None):
        """
        Calculate Focal Loss.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            sample_weight: Optional sample weights
            
        Returns:
            Focal Loss value
        """
        # Clip predictions to prevent numerical instability
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate cross entropy
        cross_entropy = -y_true * K.log(y_pred)
        
        # Calculate focal loss
        loss = self.alpha * K.pow(1.0 - y_pred, self.gamma) * cross_entropy
        
        # Sum over classes
        return K.mean(K.sum(loss, axis=-1))

class DiceLoss(tf.keras.losses.Loss):
    """
    Dice Loss for addressing class imbalance.
    
    DL = 1 - (2 * intersection) / (union + epsilon)
    
    where intersection is the element-wise product of y_true and y_pred,
    and union is their sum.
    """
    
    def __init__(self, smooth=1e-6, **kwargs):
        """
        Initialize Dice Loss.
        
        Args:
            smooth: Smoothing factor to prevent division by zero
        """
        super().__init__(**kwargs)
        self.smooth = smooth
        
    def __call__(self, y_true, y_pred, sample_weight=None):
        """
        Calculate Dice Loss.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            sample_weight: Optional sample weights
            
        Returns:
            Dice Loss value
        """
        # Flatten the predictions and true values
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        
        # Calculate intersection and union
        intersection = K.sum(y_true_f * y_pred_f)
        union = K.sum(y_true_f) + K.sum(y_pred_f)
        
        # Calculate Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return Dice loss
        return 1.0 - dice

class HybridLoss(tf.keras.losses.Loss):
    """
    Hybrid Loss combining Focal Loss and Dice Loss for improved performance.
    
    Hybrid Loss = alpha * Focal Loss + beta * Dice Loss
    """
    
    def __init__(self, alpha=0.5, beta=0.5, focal_gamma=2.0, focal_alpha=0.25, **kwargs):
        """
        Initialize Hybrid Loss.
        
        Args:
            alpha: Weight for Focal Loss
            beta: Weight for Dice Loss
            focal_gamma: Focusing parameter for Focal Loss
            focal_alpha: Weighting factor for Focal Loss
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss()
        
    def __call__(self, y_true, y_pred, sample_weight=None):
        """
        Calculate Hybrid Loss.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            sample_weight: Optional sample weights
            
        Returns:
            Hybrid Loss value
        """
        focal = self.focal_loss(y_true, y_pred, sample_weight)
        dice = self.dice_loss(y_true, y_pred, sample_weight)
        
        return self.alpha * focal + self.beta * dice