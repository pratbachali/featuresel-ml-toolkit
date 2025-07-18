# Core ML Pipeline Configuration

# Data Processing
DATA_CONFIG = {
    'random_seed': 42,
    'test_size': 0.2,
    'validation_size': 0.2,
    'missing_value_threshold': 0.1,  # Max proportion of missing values allowed
    'variance_threshold': 0.01,      # Minimum variance for feature filtering
}

# Feature Selection
FEATURE_SELECTION_CONFIG = {
    'methods': [
        'variance_threshold',
        'univariate_f_test',
        'mutual_information',
        'random_forest_importance',
        'rfe_random_forest',
        'lasso_regularization',
        'gradient_boosting_importance'
    ],
    'default_n_features': 100,
    'top_variance_genes': 1000,
    'cv_folds': 5,
    'scoring_metric': 'roc_auc',
}

# Machine Learning Models
ML_MODELS_CONFIG = {
    'algorithms': {
        'random_forest': {
            'class': 'RandomForestClassifier',
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
        },
        'logistic_regression': {
            'class': 'LogisticRegression',
            'params': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000, 2000]
            }
        },
        'svm': {
            'class': 'SVC',
            'params': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'probability': [True]
            }
        },
        'gradient_boosting': {
            'class': 'GradientBoostingClassifier',
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7, 10],
                'subsample': [0.8, 0.9, 1.0]
            }
        },
        'extra_trees': {
            'class': 'ExtraTreesClassifier',
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'max_features': ['sqrt', 'log2']
            }
        }
    },
    'cv_folds': 5,
    'scoring_metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
    'n_jobs': -1,  # Use all available cores
}

# Visualization Settings
VISUALIZATION_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8',
    'color_palette': 'husl',
    'save_formats': ['png', 'pdf'],
    'plots': {
        'pca': True,
        'feature_importance': True,
        'confusion_matrix': True,
        'roc_curves': True,
        'learning_curves': True,
        'feature_selection_comparison': True
    }
}

# Output Settings
OUTPUT_CONFIG = {
    'base_dir': 'results',
    'subdirs': {
        'data': 'processed_data',
        'features': 'feature_selection',
        'models': 'trained_models',
        'plots': 'visualizations',
        'reports': 'reports'
    },
    'save_models': True,
    'generate_report': True,
    'log_level': 'INFO'
}

# Advanced Settings
ADVANCED_CONFIG = {
    'enable_early_stopping': True,
    'enable_feature_engineering': False,
    'enable_ensemble_methods': True,
    'enable_dimensionality_reduction': True,
    'parallel_processing': True,
    'memory_optimization': True
}
