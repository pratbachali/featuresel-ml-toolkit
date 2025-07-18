# Scripts: Feature Selection and ML Pipeline

This directory contains the core executable scripts for running the genomic feature selection and machine learning pipeline. These scripts leverage the modular code structure from the `src` directory and can be executed as standalone tools.

## 🚀 Available Scripts

### 1. Feature Selection: `run_feature_selection.py`

An interactive feature selection script for genomic data that implements multiple selection methods, evaluates them, and produces ML-ready datasets.

**Key Features:**
- Supports both random and predefined train/test data splitting
- Implements 6 different feature selection methods:
  - Univariate F-test
  - Mutual Information
  - Random Forest Importance
  - Recursive Feature Elimination (RFE)
  - Lasso Regularization
  - Gradient Boosting Importance
- Includes variance-based pre-filtering of genes
- Generates comprehensive visualizations
- Outputs both best-method features and combined features
- Cross-validates each method for rigorous comparison

**Usage:**
```bash
python run_feature_selection.py [options]
```

**Example:**
```bash
python run_feature_selection.py --data_file path/to/data.csv --target target_column --n_features 50 --output_dir feature_results
```

**Interactive Mode:**
When run without parameters, the script will guide you through the process with interactive prompts.

### 2. Machine Learning Models: `run_ml_models.py`

A comprehensive machine learning script for training and evaluating multiple classification models on genomic data, with support for top feature selection.

**Key Features:**
- Supports multiple ML algorithms:
  - Random Forest
  - XGBoost/Gradient Boosting
  - Support Vector Machines (SVM)
  - Logistic Regression
  - K-Nearest Neighbors
  - AdaBoost
- Generates ROC curves and performance metrics for all models
- Creates separate visualizations for models trained on all features vs. top features
- Extracts top features from the best model
- Provides comprehensive evaluation metrics:
  - Accuracy
  - Area Under ROC Curve (AUC)
  - Sensitivity/Recall
  - Specificity
  - F1 Score
- Saves trained models and ML-ready datasets

**Usage:**
```bash
python run_ml_models.py [options]
```

**Example:**
```bash
python run_ml_models.py --train_file path/to/train.csv --test_file path/to/test.csv --target target_column --output_dir ml_results
```

### 3. Full Pipeline: `run_full_pipeline.py`

A complete end-to-end pipeline that combines feature selection and machine learning in a single workflow.

**Key Features:**
- Performs all steps from data loading to final model evaluation
- Integrates both feature selection and classification pipeline steps
- Provides a simplified interface for running the complete analysis
- Generates comprehensive reports and visualizations
- Creates ML-ready datasets

**Usage:**
```bash
python run_full_pipeline.py data_file --target target_column [options]
```

**Example:**
```bash
python run_full_pipeline.py path/to/data.csv --target target_column --n_features 50 --output_dir pipeline_results
```

## 📋 Common Parameters

The scripts share several common parameters:

- `--data_file`: Path to input data file (CSV or Excel)
- `--train_file`/`--test_file`: For pre-split train/test data
- `--target`: Column name of the target variable
- `--n_features`: Number of top features to select
- `--output_dir`: Directory to save results
- `--log_level`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `--cv_folds`: Number of cross-validation folds

## 🔄 Pipeline Workflow

The typical workflow follows these steps:

1. **Feature Selection** (`run_feature_selection.py`):
   - Load and preprocess data
   - Apply variance-based pre-filtering
   - Run multiple feature selection methods
   - Evaluate and select best method
   - Generate ML-ready datasets with selected features

2. **Machine Learning** (`run_ml_models.py`):
   - Train multiple classification models
   - Evaluate models on test data
   - Generate ROC curves and performance metrics
   - Train and evaluate models on top features
   - Create comparative visualizations
   - Save best models and results

3. **Full Pipeline** (`run_full_pipeline.py`):
   - Combines steps 1 and 2 in a single workflow
   - Generates comprehensive final results

## 📊 Output Structure

Each script generates outputs in the specified output directory with the following structure:

```
output_dir/
├── feature_selection_results.csv      # Selected features
├── feature_selection_performance.png  # Performance comparison
├── feature_selection.log              # Execution log
├── ml_analysis_summary.json           # ML results summary
├── ml_analysis.log                    # ML execution log
├── best_model_all_features.pkl        # Saved model (all features)
├── best_model_top_features.pkl        # Saved model (top features)
├── ml_ready_train_top_features.csv    # ML-ready train data
├── ml_ready_test_top_features.csv     # ML-ready test data
└── visualizations/                    # Various plots and charts
```

## 🧪 Testing

Each script includes error handling and validation steps. For testing specific components:

```bash
python -m pytest test_feature_selection.py
python -m pytest test_run_ml.py
python -m pytest test_with_combined_dataset.py
```
