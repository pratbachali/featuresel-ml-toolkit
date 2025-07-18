# Machine Learning Analysis Script

This script performs comprehensive supervised machine learning analysis on your data, including model comparison, feature selection, and visualization.

## Installation

Make sure you have all required dependencies installed:

```bash
# Install core requirements
pip install -r requirements.txt

# For full functionality (including XGBoost and LightGBM)
pip install xgboost lightgbm
```

The script will automatically detect which models are available in your environment.

## Features

- Supports multiple ML algorithms: Random Forest, Logistic Regression, SVM, XGBoost, LightGBM, etc.
- Extracts top 20 features from the best model
- Compares performance using all features vs top 20 features
- Generates ROC curves, performance metrics tables, and feature importance visualizations
- Supports predefined or random train/test split

## Usage

### Interactive Mode

```bash
python scripts/run_machine_learning.py --interactive
```

This will prompt you for:
- Input data file path
- Target variable name
- Train/test split preference (predefined column or random split)
- Output directory

### Command Line Arguments

```bash
python scripts/run_machine_learning.py \
  --data_file path/to/your/data.csv \
  --target target_column_name \
  --use_predefined_split \
  --category_column Category \
  --output_dir ml_results
```

### Arguments

- `--data_file`: Path to input CSV data file
- `--target`: Name of the target variable column
- `--use_predefined_split`: Flag to use a predefined train/test split from a category column
- `--category_column`: Column name containing train/test split labels (default: 'Category')
- `--test_size`: Test set size for random split (default: 0.2)
- `--output_dir`: Directory to save results (default: 'ml_results')
- `--random_state`: Random seed for reproducibility (default: 42)

## Output

The script generates the following outputs in the specified output directory:

1. **ROC Curves** (`roc_curves_comparison.png`): ROC curves for all models with all features vs top 20 features
2. **Model Metrics** (`model_performance_metrics.csv`): CSV file with all model performance metrics
3. **Metrics Table** (`model_metrics_table.png`): Visual table of model performance metrics
4. **Top Features** (`top_20_features.csv`): CSV file with the top 20 features and their importance scores
5. **Feature Importance** (`top_features_importance.png`): Bar chart visualization of top 20 features
6. **Performance Comparison** (`performance_comparison.png`): Bar charts comparing all features vs top 20 features performance
7. **Log File** (`ml_analysis.log`): Detailed log of the analysis process
8. **Results JSON** (`ml_analysis_results.json`): Complete results in JSON format for further analysis
