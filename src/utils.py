"""
utils.py - Utility functions for diabetic retinopathy classification
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

def create_directory_structure(base_dir):
    """
    Create directory structure for the project.
    
    Args:
        base_dir: Base directory path
    """
    # Create main directories
    directories = [
        os.path.join(base_dir, 'models'),
        os.path.join(base_dir, 'data', 'train'),
        os.path.join(base_dir, 'data', 'validation'),
        os.path.join(base_dir, 'data', 'test'),
        os.path.join(base_dir, 'results')
    ]
    
    # Create class subdirectories in data directories
    class_names = ['0_No_DR', '1_Mild', '2_Moderate', '3_Severe', '4_Proliferative']
    for data_dir in [os.path.join(base_dir, 'data', split) for split in ['train', 'validation', 'test']]:
        for class_name in class_names:
            directories.append(os.path.join(data_dir, class_name))
    
    # Create all directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def compare_models_performance(models_results, save_path=None):
    """
    Create a bar chart comparing performance metrics across models.
    
    Args:
        models_results: Dictionary with model names as keys and performance metrics as values
        save_path: Path to save the comparison plot (optional)
    """
    model_names = list(models_results.keys())
    accuracies = [models_results[model]['accuracy'] for model in model_names]
    kappas = [models_results[model]['kappa'] for model in model_names]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
    
    # Set up positions for bars
    x = np.arange(len(model_names))
    bar_width = 0.35
    
    # Plot bars
    ax.bar(x - bar_width/2, accuracies, bar_width, label='Accuracy', color='#3498db')
    ax.bar(x + bar_width/2, kappas, bar_width, label='Quadratic Kappa', color='#2ecc71')
    
    # Add values on top of bars
    for i, v in enumerate(accuracies):
        ax.text(i - bar_width/2, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    
    for i, v in enumerate(kappas):
        ax.text(i + bar_width/2, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    
    # Customize plot
    ax.set_title('Model Performance Comparison', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to {save_path}")
    
    plt.show()

def visualize_gradcam(model, img_path, layer_name, class_idx=None, save_path=None):
    """
    Visualize Grad-CAM heatmap for model interpretability.
    
    Args:
        model: Trained model
        img_path: Path to input image
        layer_name: Name of the layer to use for Grad-CAM
        class_idx: Index of the class to explain (default: predicted class)
        save_path: Path to save the visualization (optional)
    """
    # Load and preprocess image
    from src.data_preprocessing import preprocess_image
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    processed_img = preprocess_image(img.copy())
    
    # Create model for GradCAM
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    # Get prediction if class_idx is not specified
    if class_idx is None:
        prediction = model.predict(np.expand_dims(processed_img, axis=0))
        class_idx = np.argmax(prediction[0])
    
    # Compute GradCAM
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.expand_dims(processed_img, axis=0))
        loss = predictions[:, class_idx]
    
    # Extract gradients
    grads = tape.gradient(loss, conv_outputs)
    
    # Compute importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight feature maps with gradients
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    
    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    # Resize heatmap to original image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose heatmap on original image
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='white')
    
    # Plot original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot heatmap
    axes[1].imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    # Plot superimposed image
    axes[2].imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Superimposed')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grad-CAM visualization saved to {save_path}")
    
    plt.show()
    
    return heatmap, superimposed_img