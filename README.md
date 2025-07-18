# Genomic Feature Selection Pipeline

A comprehensive, robust feature selection pipeline specifically designed for genomic data analysis with multiple methods, cross-validation evaluation, and ML-ready outputs.

## 🧬 Overview

This pipeline provides a complete solution for feature selection in genomic datasets, supporting both random and predefined train/test splits. It includes variance-based pre-filtering, multiple feature selection methods, comprehensive evaluation, and generates both best-method and combined-features ML-ready datasets.

## 🚀 Key Features

- **Variance-based Pre-filtering**: Selects top N most variable genes before feature selection
- **Multiple Feature Selection Methods**: 6 different methods including univariate, tree-based, and regularization approaches
- **Cross-validation Evaluation**: Rigorous method comparison using stratified k-fold CV
- **Dual Output Generation**: Creates datasets with both best-method features and combined features from all methods
- **Comprehensive Visualizations**: Exploratory analysis, method performance, and best method analysis plots
- **Flexible Data Splitting**: Supports both random and predefined (category-based) train/test splits
- **Interactive User Interface**: Guided prompts for all pipeline parameters
- **Detailed Logging**: Complete execution logs and analysis summaries

## 📁 Project Structure

```
new_modular_pipeline/
├── README.md                          # This documentation
├── requirements.txt                   # Python dependencies
├── pipeline_demo.py                  # Interactive demo script
├── config/
│   ├── __init__.py
│   └── config.py                     # Configuration settings
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                 # Data loading utilities
│   │   └── validator.py              # Data validation
│   ├── features/
│   │   ├── __init__.py
│   │   ├── selector.py               # Feature selection methods
│   │   └── variance_filter.py        # Variance-based filtering
│   ├── models/
│   │   ├── __init__.py
│   │   └── classifier.py             # ML model implementations
│   └── utils/
│       ├── __init__.py
│       ├── helpers.py                # Utility functions
│       └── logger.py                 # Logging configuration
├── scripts/
│   ├── run_feature_selection.py     # Main feature selection pipeline
│   └── run_full_pipeline.py         # Complete analysis pipeline
└── demo_results/                     # Example outputs (generated)
```

## 🔧 Installation

1. **Clone or download the repository**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Feature Selection Methods

The pipeline implements 6 comprehensive feature selection methods:

1. **Univariate F-test**: Statistical significance testing for feature-target relationships
2. **Mutual Information**: Non-linear dependency detection between features and target
3. **Random Forest Importance**: Tree-based feature importance scoring
4. **RFE Random Forest**: Recursive feature elimination with Random Forest
5. **Lasso Regularization**: L1 regularization for automatic feature selection
6. **Gradient Boosting Importance**: Boosting-based feature importance

Each method is evaluated using cross-validation, and the best performing method is automatically selected.

## 🎯 Usage

### Quick Start

Run the main pipeline with guided prompts:

```bash
python scripts/run_feature_selection.py
```

The pipeline will interactively ask for:
- **Data file path**: Path to your CSV file
- **Target variable**: Column name for the target variable
- **Number of variance genes**: Top N most variable genes to select for pre-filtering
- **Split type**: Choose between 'random' or 'predefined' train/test split
- **Top features per method**: Number of top features to select per method
- **Output directory**: Directory to save results

### Interactive Demo

Run the demo to see the pipeline in action with synthetic data:

```bash
python pipeline_demo.py
```

This will:
- Create synthetic genomic data
- Run the pipeline with both random and predefined splits
- Generate example outputs
- Explain all output files

## 📈 Input Data Format

Your CSV file should have:
- **Gene columns**: Numeric expression values (one column per gene)
- **Target column**: Binary target variable (e.g., 'IR', 'IS')
- **Category column** (optional): For predefined splits (e.g., 'Dataset_A', 'Dataset_B')

Example:
```csv
GENE_001,GENE_002,GENE_003,...,IRIS,Category
2.34,1.45,3.67,...,IR,Dataset_A
1.23,2.89,1.45,...,IS,Dataset_B
...
```

## 📁 Output Files

The pipeline generates comprehensive outputs:

### 🎯 Feature Selection Results
- `all_methods_top_N_features.csv` - Top N features from each method
- `best_method_<method>_features.csv` - Features from the best performing method
- `combined_all_methods_features.csv` - Deduplicated features from all methods
- `feature_method_mapping.csv` - Which method selected each feature

### 🤖 ML-Ready Datasets
- `ml_ready_train_data.csv` - Training data with **best method** features
- `ml_ready_test_data.csv` - Test data with **best method** features  
- `ml_ready_combined_train.csv` - Training data with **all methods** features
- `ml_ready_combined_test.csv` - Test data with **all methods** features

### 📊 Visualizations
- `exploratory_data_analysis.png` - Target distribution and PCA plots
- `feature_selection_performance.png` - Method performance comparison
- `best_method_analysis.png` - Best method detailed analysis

### 📋 Summary & Logs
- `comprehensive_analysis_summary.json` - Complete analysis summary
- `feature_selection.log` - Detailed execution log
- `method_evaluation_scores.csv` - Cross-validation scores

## 🔬 Pipeline Steps

1. **Data Loading & Validation**: Load CSV, validate format, check for missing values
2. **Variance-based Pre-filtering**: Select top N most variable genes
3. **Exploratory Analysis**: Generate distribution plots and PCA visualization
4. **Train/Test Split**: Random or predefined (category-based) splitting
5. **Feature Selection**: Apply 6 different methods with cross-validation
6. **Method Evaluation**: Compare methods using AUC and accuracy scores
7. **Best Method Selection**: Automatically select the best performing method
8. **Output Generation**: Create ML-ready datasets and visualizations
9. **Summary Reports**: Generate comprehensive analysis summary

## 🎯 Method Selection Strategy

The pipeline automatically selects the best feature selection method based on:
- **Primary metric**: Cross-validation AUC score
- **Secondary metric**: Cross-validation accuracy
- **Validation**: Stratified k-fold cross-validation (default: 5 folds)

## 📊 Understanding the Results

### Best Method vs Combined Features

- **Best Method datasets**: Use features from the single best performing method
- **Combined datasets**: Use deduplicated features from all methods combined
- **Recommendation**: Start with best method datasets; use combined if you need more features

### Performance Metrics

- **AUC Score**: Area under ROC curve (primary selection metric)
- **Accuracy**: Classification accuracy (secondary metric)
- **Cross-validation**: Robust performance estimation using stratified k-fold

## 🛠️ Advanced Usage

### Custom Method Selection

You can modify the methods used by editing `src/features/selector.py`:

```python
# Default methods
methods = [
    'univariate_f_test', 
    'mutual_information',
    'random_forest_importance',
    'rfe_random_forest',
    'lasso_regularization',
    'gradient_boosting_importance'
]
```

### Configuration

Modify `config/config.py` to change default parameters:

```python
# Default parameters
DEFAULT_CV_FOLDS = 5
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
```

## 🚨 Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed via `pip install -r requirements.txt`
2. **Memory issues**: Reduce the number of variance genes for large datasets
3. **Slow performance**: Consider reducing CV folds or using fewer features per method
4. **Missing values**: The pipeline automatically handles missing values by dropping them

### Performance Tips

- **Large datasets**: Start with fewer variance genes (e.g., 1000-5000)
- **Many samples**: Consider using fewer CV folds (e.g., 3 instead of 5)
- **Quick testing**: Use smaller numbers for top features per method (e.g., 10-20)

## 📖 Examples

### Example 1: Basic Usage
```bash
# Run with guided prompts
python scripts/run_feature_selection.py

# Input prompts:
# Data file: my_genomic_data.csv
# Target: IRIS
# Variance genes: 1000
# Split type: random
# Top features: 50
# Output: results_2024
```

### Example 2: Predefined Split
```bash
# For datasets with multiple cohorts/batches
python scripts/run_feature_selection.py

# Input prompts:
# Data file: multi_cohort_data.csv
# Target: Response
# Variance genes: 2000
# Split type: predefined
# Top features: 100
# Output: cohort_analysis
```

## 📚 Dependencies

See `requirements.txt` for full list. Key dependencies:
- `pandas >= 1.3.0`
- `numpy >= 1.21.0`
- `scikit-learn >= 1.0.0`
- `matplotlib >= 3.5.0`
- `seaborn >= 0.11.0`

## 🤝 Contributing

To contribute to this pipeline:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔬 Citation

If you use this pipeline in your research, please cite:

```
Genomic Feature Selection Pipeline
[Your Institution/Organization]
2024
```

---

**Questions or Issues?** Please create an issue in the repository or contact the development team.
└── requirements.txt                # Dependencies
```

## 🔧 Installation

1. Navigate to the new_modular_pipeline directory
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Quick Start

### Feature Selection Only
```bash
python scripts/run_feature_selection.py data/your_data.csv --target class --n_features 50
```

### Classification Only
```bash
python scripts/run_classification.py data/your_data.csv --target class --features selected_features.csv
```

### Full Pipeline
```bash
python scripts/run_full_pipeline.py data/your_data.csv --target class --output_dir results/
```

## 📊 Usage Examples

### Python API
```python
from src.data.loader import DataLoader
from src.features.selector import FeatureSelector
from src.models.classifier import MLClassifier

# Load data
loader = DataLoader()
X, y = loader.load_csv('data.csv', target_column='class')

# Select features
selector = FeatureSelector()
X_selected = selector.fit_transform(X, y, method='random_forest', n_features=50)

# Train models
classifier = MLClassifier()
classifier.fit(X_selected, y)
results = classifier.evaluate()
```

## 🎛️ Configuration

Edit `config/config.py` to customize:
- Feature selection methods
- ML algorithms and hyperparameters
- Cross-validation settings
- Visualization preferences
- Output formats

## 📈 Outputs

The pipeline generates:
- Feature selection results and rankings
- Model performance metrics
- Visualization plots (PCA, feature importance, ROC curves)
- Comprehensive reports in HTML and PDF
- Serialized models for deployment
