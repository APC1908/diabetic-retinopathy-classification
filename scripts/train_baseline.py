"""
train_baseline.py - Script to train the optimal baseline model
"""

import os
import argparse
import matplotlib.pyplot as plt
from src.models.baseline_model import train_model, evaluate_model
from src.evaluation import plot_training_history, plot_confusion_matrix

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train optimal baseline model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save model')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Set paths
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'validation')
    test_dir = os.path.join(args.data_dir, 'test')
    model_path = os.path.join(args.model_dir, 'optimal_baseline_dr_model.keras')
    
    # Train model
    print("Training optimal baseline model...")
    model, history = train_model(train_dir, val_dir, model_path)
    
    # Plot training history
    history_plot_path = os.path.join(args.results_dir, 'baseline_training_history.png')
    plot_training_history(history, save_path=history_plot_path)
    
    # Evaluate model
    print("Evaluating model on test data...")
    accuracy, cm = evaluate_model(model, test_dir)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Plot confusion matrix
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    cm_plot_path = os.path.join(args.results_dir, 'baseline_confusion_matrix.png')
    plot_confusion_matrix(cm, class_names, normalize=False, save_path=cm_plot_path)
    
    # Plot normalized confusion matrix
    norm_cm_plot_path = os.path.join(args.results_dir, 'baseline_normalized_confusion_matrix.png')
    plot_confusion_matrix(cm, class_names, normalize=True, save_path=norm_cm_plot_path)
    
    print("Training and evaluation completed.")

if __name__ == '__main__':
    main()