# Machine Learning Models Script

## Overview
This script provides a comprehensive supervised machine learning pipeline that:
1. Builds multiple ML models (RandomForest, XGBoost, LightGBM, CatBoost, SVM, LogisticRegression)
2. Evaluates all models with multiple metrics (Accuracy, AUC, Sensitivity, Specificity, F1)
3. Extracts top 20 features from the best performing model
4. Compares performance between using all features vs. top 20 features
5. Generates visualizations (ROC curves, feature importance, metrics table)
6. Saves trained models and comprehensive summary
7. Handles string class labels properly for binary classification (e.g., 'IR', 'IS')

## Requirements
- Python 3.6+
- Required packages: numpy, pandas, matplotlib, seaborn, scikit-learn
- Optional packages (for more models): xgboost, lightgbm, catboost, joblib

To install required packages:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
# Optional packages
pip install xgboost lightgbm catboost joblib
```

## Usage

### Interactive Mode
```bash
python run_ml_models.py --interactive
```

### Command Line Arguments
```bash
python run_ml_models.py \
  --train_file path/to/train_data.csv \
  --test_file path/to/test_data.csv \
  --target target_column_name \
  --use_predefined_split \  # Use separate train/test files
  --output_dir ml_results
```

Or to combine train/test and randomly split:
```bash
python run_ml_models.py \
  --train_file path/to/train_data.csv \
  --test_file path/to/test_data.csv \
  --target target_column_name \
  --test_size 0.3 \  # 30% test size for random split
  --output_dir ml_results
```

## Output Files
- **ml_analysis_summary.json**: Complete analysis summary with all metrics
- **roc_curves_all_features.png**: ROC curves for all models
- **roc_comparison_top_features.png**: Comparison of all features vs. top features
- **top_feature_importances.png**: Bar chart of top feature importance
- **performance_metrics_all_features.png**: Heatmap of model performance metrics
- **top_features.csv**: List of top features with importance scores
- **ml_ready_train_top_features.csv**: Training data with only top features
- **ml_ready_test_top_features.csv**: Test data with only top features
- **best_model_all_features.pkl**: Saved best model using all features
- **best_model_top_features.pkl**: Saved best model using only top features

## Model Notes

### SVM Configuration
For SVM models, the script automatically configures `probability=True` to enable ROC curve generation and proper scoring. This may make the SVM training slightly slower but ensures compatibility with all metrics.

If the model doesn't support `predict_proba()` for any reason, the script will gracefully fall back to using:
1. `decision_function()` if available (normalized to [0,1] range)
2. The predicted class labels as a last resort

This ensures that all models can generate ROC curves and probability-based metrics.
