# Diabetic Retinopathy Classification

This repository contains implementations of deep learning models for automated diabetic retinopathy (DR) classification from retinal fundus images.

## Overview

Diabetic retinopathy is a diabetes complication that affects the eyes and can lead to blindness if left undiagnosed and untreated. This project implements and compares two approaches for DR classification:

1. **Transfer Learning Model**: An EfficientNetV2-S model pretrained on ImageNet and fine-tuned for DR classification.
2. **Optimal Baseline Model**: A custom CNN architecture with residual connections and squeeze-excitation blocks.

Both models classify DR into five severity levels:
- Class 0: No DR
- Class 1: Mild DR
- Class 2: Moderate DR
- Class 3: Severe DR
- Class 4: Proliferative DR

## Features

- Advanced image preprocessing for retinal fundus images
- Custom hybrid loss function (Focal Loss + Dice Loss)
- Test-time augmentation for robust predictions
- Class weighting to handle imbalanced data
- Comprehensive evaluation metrics

## Installation

```bash
# Clone the repository
git clone https://github.com/APC1908/diabetic-retinopathy-classification.git
cd diabetic-retinopathy-classification

# Create a virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training the Transfer Learning Model
```bash
python scripts/train_transfer_learning.py --data_dir /path/to/data --model_dir models --results_dir results
```

### Training the Baseline Model
```bash
python scripts/train_baseline.py --data_dir /path/to/data --model_dir models --results_dir results
```

### Evaluating Models
```bash
python scripts/evaluate_models.py --test_dir /path/to/test/data --models_dir models --results_dir results --use_tta
```

## Project Structure
```
diabetic-retinopathy-classification/
├── src/                      # Source code
│   ├── __init__.py           # Package initialization
│   ├── data_preprocessing.py # Image preprocessing and augmentation
│   ├── loss_functions.py     # Custom loss functions
│   ├── evaluation.py         # Evaluation metrics and visualization
│   ├── utils.py              # Utility functions
│   └── models/               # Model implementations
│       ├── __init__.py       # Models package initialization
│       ├── baseline_model.py # Optimal baseline model
│       └── transfer_learning.py # EfficientNetV2-S model
├── scripts/                  # Executable scripts
│   ├── train_baseline.py     # Script to train baseline model
│   ├── train_transfer_learning.py # Script to train transfer learning model
│   └── evaluate_models.py    # Script to evaluate and compare models
├── configs/                  # Configuration files
│   ├── baseline_config.py    # Configuration for baseline model
│   └── transfer_learning_config.py # Configuration for transfer learning model
├── .gitignore                # Git ignore rules
├── requirements.txt          # Dependencies
├── setup.py                  # Package installation
├── README.md                 # Project documentation
└── LICENSE                   # Project license
```

## Results
The EfficientNetV2-S transfer learning model achieves superior performance compared to the baseline model:

| Model              | Test Accuracy | Quadratic Kappa |
|--------------------|---------------|-----------------|
| Optimal Baseline   | 87.23%        | 0.856           |
| EfficientNetV2-S   | 94.00%        | 0.927           |

### Per-Class Accuracy
| Class         | Optimal Baseline | EfficientNetV2-S |
|---------------|------------------|------------------|
| No DR         | 90.0%            | 96.2%            |
| Mild          | 76.0%            | 86.1%            |
| Moderate      | 85.0%            | 93.6%            |
| Severe        | 74.0%            | 89.1%            |
| Proliferative | 73.0%            | 91.6%            |

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.