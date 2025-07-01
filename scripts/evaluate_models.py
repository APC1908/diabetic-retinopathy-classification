"""
evaluate_models.py - Script to evaluate and compare diabetic retinopathy classification models
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from src.loss_functions import HybridLoss, FocalLoss, DiceLoss
from src.evaluation import evaluate_model_comprehensive
from src.utils import compare_models_performance

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate diabetic retinopathy classification models')
    parser.add_argument('--test_dir', type=str, required=True, help='Path to test data directory')
    parser.add_argument('--models_dir', type=str, default='models', help='Directory containing model files')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory to save evaluation results')
    parser.add_argument('--use_tta', action='store_true', help='Use test-time augmentation for evaluation')
    
    return parser.parse_args()

def load_model(model_path):
    """Load a model with custom objects."""
    custom_objects = {
        'HybridLoss': HybridLoss,
        'FocalLoss': FocalLoss,
        'DiceLoss': DiceLoss
    }
    
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    return model

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Define model paths
    model_paths = {
        'Optimal Baseline': os.path.join(args.models_dir, 'optimal_baseline_dr_model.keras'),
        'EfficientNetV2-S': os.path.join(args.models_dir, 'efficientnetv2s_dr_model.keras')
    }
    
    # Define class names
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    
    # Create data generator for test data
    from src.data_preprocessing import get_validation_data_generator
    test_datagen = get_validation_data_generator()
    test_generator = test_datagen.flow_from_directory(
        args.test_dir,
        target_size=(380, 380),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    # Evaluate models
    models_results = {}
    
    for model_name, model_path in model_paths.items():
        print(f"Evaluating {model_name} model...")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            continue
        
        # Load model
        model = load_model(model_path)
        
        # Create save directory for this model
        model_results_dir = os.path.join(args.results_dir, model_name.lower().replace(' ', '_'))
        os.makedirs(model_results_dir, exist_ok=True)
        
        # Evaluate model
        results = evaluate_model_comprehensive(
            model, 
            test_generator, 
            class_names, 
            use_tta=args.use_tta and model_name == 'EfficientNetV2-S',  # Only use TTA for EfficientNetV2-S
            save_dir=model_results_dir
        )
        
        # Store results
        models_results[model_name] = {
            'accuracy': results['accuracy'],
            'kappa': results['kappa'],
            'confusion_matrix': results['confusion_matrix']
        }
        
        # Print results
        print(f"{model_name} Test Accuracy: {results['accuracy']:.4f}")
        print(f"{model_name} Quadratic Kappa: {results['kappa']:.4f}")
        print(f"{model_name} Classification Report:\n{results['classification_report']}")
        print("-" * 50)
    
    # Compare models
    compare_models_performance(
        models_results, 
        save_path=os.path.join(args.results_dir, 'model_comparison.png')
    )
    
    # Compare per-class accuracy (recall)
    per_class_accuracy = {'Class': class_names}
    
    for model_name, results in models_results.items():
        cm = results['confusion_matrix']
        recall_per_class = np.diag(cm) / np.sum(cm, axis=1)
        per_class_accuracy[model_name] = recall_per_class
    
    # Print per-class accuracy
    print("Per-Class Accuracy (Recall):")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: ", end="")
        for model_name in models_results.keys():
            print(f"{model_name}: {per_class_accuracy[model_name][i]:.4f}  ", end="")
        print()
    
    print("Evaluation completed.")

if __name__ == '__main__':
    main()