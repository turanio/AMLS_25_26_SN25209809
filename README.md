# AMLS Final Project - BreastMNIST Classification

**Student Number:** SN25209809  
**Course:** Applied Machine Learning Systems (AMLS) 25/26

## Overview

This project implements two machine learning models for binary classification of breast ultrasound images using the BreastMNIST dataset. The task is to classify images as benign or malignant tumors.

### Models Implemented

1. **Model A: PCA + Linear SVM**
   - Feature extraction using Principal Component Analysis (PCA)
   - Classification using Support Vector Machine with linear kernel
   - Data augmentation with random transformations

2. **Model B: Convolutional Neural Network (CNN)**
   - Custom CNN architecture for 28x28 grayscale images
   - Two convolutional layers with max pooling
   - Fully connected layers for classification
   - PyTorch-based implementation

## Project Structure

```
AMLS_25_26_SN25209809/
├── A/                          # Model A (PCA + SVM)
│   ├── __init__.py
│   ├── config.py              # Configuration parameters
│   ├── features.py            # Data augmentation and feature extraction
│   ├── model.py               # Model building pipeline
│   └── train.py               # Training and evaluation logic
├── B/                          # Model B (CNN)
│   ├── __init__.py
│   ├── config.py              # Configuration parameters
│   ├── model.py               # CNN architecture
│   ├── train.py               # Training and evaluation logic
│   └── transforms.py          # Image transformations
├── data/                       # Data loading utilities
│   ├── __init__.py
│   └── breastmnist.py         # BreastMNIST dataset loader
├── utils/                      # Shared utilities
│   ├── __init__.py
│   ├── base_trainer.py        # Abstract base trainer class
│   ├── logger.py              # Custom logging system
│   └── seed.py                # Reproducibility utilities
├── logs/                       # Training logs (auto-generated)
├── Datasets/                   # Dataset directory
├── main.py                     # Main entry point
├── requirements.txt            # Python dependencies
├── environment.yml             # Conda environment specification
└── README.md                   # This file
```

## Installation

### Using Conda (Recommended)

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate amls-final-assignment
```

### Using pip

```bash
# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

### Runtime Dependencies
- Python 3.12
- medmnist - Medical MNIST datasets
- numpy - Numerical computing
- torch - PyTorch deep learning framework
- torchvision - Computer vision utilities
- scikit-learn - Machine learning algorithms
- scipy - Scientific computing
- Pillow - Image processing

### Development Dependencies
- black - Automatic code formatter
- flake8 - Code style and linting
- isort - Import sorting and grouping

## Usage

### Training Both Models

Run the main script to train both models sequentially:

```bash
python main.py
```

This will:
1. Check 'Datasets\' folder and look for the BreastMNIST dataset (has to be already present)
2. Split data into train (70%), validation (10%), and test (20%) sets
3. Train Model A (PCA + SVM) with data augmentation
4. Evaluate Model A on the test set
5. Train Model B (CNN) with data augmentation
6. Evaluate Model B on the test set
7. Log all results to `logs/` directory


### Logging System

The project includes a comprehensive logging system with:
- Separate log files for each model
- Console output for real-time monitoring
- Rotating file handlers (5MB max, 3 backups)
- Separate logs for warnings and errors
- Training progress and evaluation metrics logging

### Reproducibility

All random operations are seeded for reproducibility:
- Python random module
- NumPy random number generator
- PyTorch random number generator
- CUDA random number generator (if available)

## Results

Results are logged to the following locations:

- **Model A logs:** `logs/modelA/`
  - `train.log` - Training progress
  - `outputs.log` - Final evaluation metrics
  
- **Model B logs:** `logs/modelB/`
  - `train.log` - Training progress
  - `outputs.log` - Final evaluation metrics

- **Global logs:**
  - `logs/warnings.log` - All warnings
  - `logs/errors.log` - All errors
  - `logs/data_loader.log` - Dataset loading information

### Evaluation Metrics

Both models report the following metrics on the test set:
- **Accuracy** - Overall classification accuracy
- **Precision** - Positive predictive value
- **Recall** - True positive rate (sensitivity)
- **F1 Score** - Harmonic mean of precision and recall

## Architecture Details

### Model A: PCA + SVM

```
Input (28×28 grayscale) 
    ↓
Flatten (784 features)
    ↓
StandardScaler (normalization)
    ↓
PCA (64 components)
    ↓
Linear SVM (C=1.0)
    ↓
Binary Classification
```

### Model B: CNN

```
Input (1×28×28)
    ↓
Conv2d(1→32, 3×3, padding=1) + ReLU + MaxPool2d(2×2) → 32×14×14
    ↓
Conv2d(32→64, 3×3, padding=1) + ReLU + MaxPool2d(2×2) → 64×7×7
    ↓
Flatten (3136 features)
    ↓
Linear(3136→128) + ReLU
    ↓
Linear(128→1)
    ↓
Binary Classification (with BCEWithLogitsLoss)
```

## Code Quality

The codebase follows best practices:
- **Google-style docstrings** for all modules, classes, and functions
- **Type hints** for all function arguments and return values
- **Abstract base classes** for extensibility
- **Modular design** with clear separation of concerns
- **PEP 8 compliant** (enforced with black and flake8)

## Development

### Code Formatting

```bash
# Format code with black
black .

# Sort and group imports with isort
isort .

# Check code style with flake8
flake8 .
```

## Dataset

**BreastMNIST** is a subset of the larger MedMNIST dataset collection:
- **Size:** 28×28 grayscale images
- **Classes:** 2 (benign/malignant)
- **Total samples:** 780 images
- **Source:** Breast ultrasound images


## License
This project is submitted as part of the AMLS course requirements at UCL.

## Acknowledgments

- MedMNIST dataset: https://medmnist.com/
- Dataset paper: Yang et al., "MedMNIST Classification Decathlon"

## Contact

For questions or issues, please contact via UCL email using student number SN25209809.
