"""
evaluation.py - Utilities for evaluating diabetic retinopathy classification models
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, cohen_kappa_score
import tensorflow as tf

def plot_training_history(history, save_path=None):
    """
    Plot training and validation accuracy/loss curves.
    
    Args:
        history: Training history object from model.fit()
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', color='royalblue')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', color='darkorange')
    axes[0].set_title('Model Accuracy', fontsize=14)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Training Loss', color='royalblue')
    axes[1].plot(history.history['val_loss'], label='Validation Loss', color='darkorange')
    axes[1].set_title('Model Loss', fontsize=14)
    axes[1].set_ylabel('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()

def plot_confusion_matrix(cm, class_names, normalize=False, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        normalize: Whether to normalize the confusion matrix
        save_path: Path to save the plot (optional)
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized Confusion Matrix (Test Set)'
    else:
        title = 'Confusion Matrix (Test Set)'
    
    plt.figure(figsize=(10, 8), facecolor='white')
    
    # Use a colormap that's easy to read
    cmap = sns.color_palette("Blues", as_cmap=True)
    
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
                cmap=cmap, cbar=True, square=True,
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title(title, fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to {save_path}")
    
    plt.show()

def plot_classification_report(y_true, y_pred, class_names, save_path=None):
    """
    Create a visual plot of the classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot (optional)
    """
    # Get classification report as dictionary
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Extract class metrics
    class_metrics = {}
    for i, class_name in enumerate(class_names):
        class_metrics[class_name] = {
            'Precision': report[class_name]['precision'],
            'Recall': report[class_name]['recall'],
            'F1-Score': report[class_name]['f1-score'],
            'Support': report[class_name]['support']
        }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
    
    # Set up positions for bars
    class_positions = np.arange(len(class_names))
    bar_width = 0.2
    
    # Plot bars for each metric
    ax.bar(class_positions - bar_width*1.5, [class_metrics[c]['Precision'] for c in class_names], 
           width=bar_width, label='Precision', color='#3498db')
    ax.bar(class_positions - bar_width/2, [class_metrics[c]['Recall'] for c in class_names], 
           width=bar_width, label='Recall', color='#2ecc71')
    ax.bar(class_positions + bar_width/2, [class_metrics[c]['F1-Score'] for c in class_names], 
           width=bar_width, label='F1-Score', color='#e74c3c')
    
    # Add text for support
    for i, class_name in enumerate(class_names):
        ax.text(i, 0.05, f"n={int(class_metrics[class_name]['Support'])}", 
                ha='center', va='bottom', fontweight='bold')
    
    # Add accuracy and macro avg as text
    ax.text(0.5, 0.95, f"Accuracy: {report['accuracy']:.4f}", 
            transform=ax.transAxes, ha='center', fontsize=14, fontweight='bold')
    
    # Customize plot
    ax.set_title('Classification Report', fontsize=16)
    ax.set_xticks(class_positions)
    ax.set_xticklabels(class_names)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Classification report plot saved to {save_path}")
    
    plt.show()

def plot_roc_curves(y_true_onehot, y_pred_proba, class_names, save_path=None):
    """
    Plot ROC curves for each class.
    
    Args:
        y_true_onehot: One-hot encoded true labels
        y_pred_proba: Predicted probabilities
        class_names: List of class names
        save_path: Path to save the plot (optional)
    """
    n_classes = len(class_names)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot all ROC curves
    plt.figure(figsize=(12, 8), facecolor='white')
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=16)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves plot saved to {save_path}")
    
    plt.show()

def calculate_kappa_score(y_true, y_pred):
    """
    Calculate Cohen's Kappa score for multi-class classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        float: Cohen's Kappa score
    """
    kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    return kappa

def evaluate_model_comprehensive(model, test_generator, class_names, use_tta=False, save_dir=None):
    """
    Perform comprehensive evaluation of a model.
    
    Args:
        model: Trained model
        test_generator: Data generator for test data
        class_names: List of class names
        use_tta: Whether to use test-time augmentation
        save_dir: Directory to save evaluation plots (optional)
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    from src.models.transfer_learning import test_time_augmentation
    
    # Reset generator
    test_generator.reset()
    
    # Get true labels
    y_true = test_generator.classes
    
    # Get predictions
    if use_tta:
        # Manual prediction with TTA
        predictions = []
        
        for i in range(len(test_generator)):
            batch_images, _ = test_generator[i]
            for image in batch_images:
                # Apply TTA
                pred = test_time_augmentation(model, image)
                predictions.append(pred)
            
        y_pred_proba = np.array(predictions)
    else:
        # Standard prediction
        y_pred_proba = model.predict(test_generator)
    
    # Convert to class indices
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Convert true labels to one-hot encoding for ROC curve
    y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=len(class_names))
    
    # Calculate metrics
    accuracy = np.mean(y_pred == y_true)
    kappa = calculate_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    # Create plots if save_dir is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Plot confusion matrices
        plot_confusion_matrix(cm, class_names, normalize=False, 
                             save_path=os.path.join(save_dir, 'confusion_matrix.png'))
        plot_confusion_matrix(cm, class_names, normalize=True, 
                             save_path=os.path.join(save_dir, 'normalized_confusion_matrix.png'))
        
        # Plot classification report
        plot_classification_report(y_true, y_pred, class_names, 
                                  save_path=os.path.join(save_dir, 'classification_report.png'))
        
        # Plot ROC curves
        plot_roc_curves(y_true_onehot, y_pred_proba, class_names, 
                       save_path=os.path.join(save_dir, 'roc_curves.png'))
    
    # Compile results
    results = {
        'accuracy': accuracy,
        'kappa': kappa,
        'confusion_matrix': cm,
        'classification_report': classification_report(y_true, y_pred, target_names=class_names),
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    return results