#!/usr/bin/env python3
"""
Supervised Machine Learning Model Analysis Script

This script builds multiple ML models with the following features:
1. Support for separate train and test files or combined dataset with random split
2. Multiple ML algorithms (RandomForest, XGBoost, SVM, etc.)
3. ROC curve visualization for all models
4. Extraction of top 20 features from the best model (selected based on test accuracy)
5. Performance comparison between using all features vs. top 20 features
6. Comprehensive evaluation metrics (accuracy, AUC, sensitivity, specificity, F1)
"""

import sys
import os
import time
import argparse
import logging
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, roc_curve, auc, confusion_matrix, 
    precision_recall_curve, f1_score, classification_report,
    precision_score, recall_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Suppress warnings
warnings.filterwarnings('ignore')

# Add the parent directory to sys.path to enable package imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

try:
    from src.utils.logger import setup_logging, get_logger
    from src.utils.helpers import ensure_directory, save_json, format_duration
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Trying alternative import method...")
    
    # Fallback: Direct execution without package structure
    print("Running in standalone mode...")
    
    # Create simple implementations for required functions
    def setup_logging(level='INFO', log_file=None):
        logging.basicConfig(level=getattr(logging, level), format='%(asctime)s - %(levelname)s - %(message)s')
        if log_file:
            handler = logging.FileHandler(log_file)
            handler.setLevel(getattr(logging, level))
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logging.getLogger().addHandler(handler)
    
    def get_logger(name):
        return logging.getLogger(name)
    
    def ensure_directory(path):
        Path(path).mkdir(parents=True, exist_ok=True)
    
    def save_json(data, file_path):
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def format_duration(seconds):
        return f"{seconds:.2f}s"

# Try to import optional packages
HAVE_XGBOOST = False
HAVE_LIGHTGBM = False
HAVE_CATBOOST = False

try:
    import xgboost as xgb
    HAVE_XGBOOST = True
except ImportError:
    pass

try:
    import lightgbm as lgb
    HAVE_LIGHTGBM = True
except ImportError:
    pass

try:
    import catboost as cb
    HAVE_CATBOOST = True
except ImportError:
    pass

def get_user_input():
    """Get user input for all ML pipeline parameters"""
    print("\n" + "="*60)
    print("MACHINE LEARNING MODEL ANALYSIS PIPELINE")
    print("="*60)
    
    # Get train file path
    train_file = input("\nEnter the path to your training CSV file: ").strip()
    while not Path(train_file).exists():
        print(f"Error: File '{train_file}' not found.")
        train_file = input("Please enter a valid file path: ").strip()
    
    # Get test file path
    test_file = input("\nEnter the path to your testing CSV file: ").strip()
    while not Path(test_file).exists():
        print(f"Error: File '{test_file}' not found.")
        test_file = input("Please enter a valid file path: ").strip()
    
    # Load training data to show available columns
    try:
        train_df = pd.read_csv(train_file)
        print(f"\nTraining dataset loaded successfully!")
        print(f"Shape: {train_df.shape}")
        print(f"\nAvailable columns:")
        for i, col in enumerate(train_df.columns, 1):
            print(f"  {i:2d}. {col}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None, None, None, None
    
    # Get target variable
    target = input(f"\nEnter the target column name: ").strip()
    while target not in train_df.columns:
        print(f"Error: Column '{target}' not found in dataset.")
        target = input("Please enter a valid column name: ").strip()
    
    # Ask about train/test split
    use_predefined_split = input(f"\nUse separate train/test files? (y/n, default y): ").strip().lower()
    use_predefined_split = use_predefined_split in ['', 'y', 'yes']
    
    # Get test size for random split if not using predefined
    test_size = 0.2
    if not use_predefined_split:
        while True:
            try:
                test_size = float(input(f"\nEnter test size (0.1-0.5, default 0.2): ") or "0.2")
                if 0.1 <= test_size <= 0.5:
                    break
                else:
                    print("Please enter a value between 0.1 and 0.5.")
            except ValueError:
                print("Please enter a valid number.")
    
    # Ask about hyperparameter tuning
    use_hyperparameter_tuning = input(f"\nUse hyperparameter tuning? This may significantly increase runtime but can improve model performance (y/n, default n): ").strip().lower()
    use_hyperparameter_tuning = use_hyperparameter_tuning in ['y', 'yes']
    
    # Ask about top_n_features
    top_n_features = 20
    try:
        top_n_input = input(f"\nEnter number of top features to select (default 20): ").strip()
        if top_n_input:
            top_n_features = int(top_n_input)
            if top_n_features <= 0:
                print("Number of features must be positive, using default (20).")
                top_n_features = 20
    except ValueError:
        print("Invalid input, using default (20).")
    
    # Get output directory
    output_dir = input(f"\nEnter output directory (default 'ml_model_results'): ").strip()
    if not output_dir:
        output_dir = 'ml_model_results'
    
    return train_file, test_file, target, use_predefined_split, test_size, output_dir, use_hyperparameter_tuning, top_n_features

def load_data(train_file, test_file, target_col, use_predefined_split=True, test_size=0.2, random_state=42):
    """
    Load data from CSV files and prepare train/test splits
    
    Args:
        train_file: Path to training data CSV
        test_file: Path to testing data CSV
        target_col: Target column name
        use_predefined_split: If True, use separate train/test files, otherwise combine and split randomly
        test_size: Test set proportion for random split
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test: Train/test splits
    """
    logger = get_logger(__name__)
    logger.info("Loading data...")
    
    # Load training data
    train_df = pd.read_csv(train_file)
    logger.info(f"Training data shape: {train_df.shape}")
    
    # Load test data
    test_df = pd.read_csv(test_file)
    logger.info(f"Test data shape: {test_df.shape}")
    
    if use_predefined_split:
        logger.info("Using predefined train/test split from separate files")
        
        # Check if target column exists in both files
        if target_col not in train_df.columns:
            logger.error(f"Target column '{target_col}' not found in training data")
            raise ValueError(f"Target column '{target_col}' not found in training data")
            
        if target_col not in test_df.columns:
            logger.error(f"Target column '{target_col}' not found in test data")
            raise ValueError(f"Target column '{target_col}' not found in test data")
        
        # Exclude target variable from features
        train_features = [col for col in train_df.columns if col != target_col]
        test_features = [col for col in test_df.columns if col != target_col]
        
        # Check if feature columns match between train and test
        if set(train_features) != set(test_features):
            logger.warning("Feature columns in train and test data don't match exactly!")
            common_features = list(set(train_features) & set(test_features))
            logger.info(f"Using {len(common_features)} common features")
            
            X_train = train_df[common_features]
            y_train = train_df[target_col]
            X_test = test_df[common_features]
            y_test = test_df[target_col]
        else:
            X_train = train_df[train_features]
            y_train = train_df[target_col]
            X_test = test_df[test_features]
            y_test = test_df[target_col]
        
    else:
        logger.info(f"Combining train and test data for random split (test_size={test_size})")
        
        # Combine datasets
        combined_df = pd.concat([train_df, test_df], axis=0)
        logger.info(f"Combined data shape: {combined_df.shape}")
        
        # Exclude target variable from features
        feature_cols = [col for col in combined_df.columns if col != target_col]
        X = combined_df[feature_cols]
        y = combined_df[target_col]
        
        # Perform random split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    
    # Log information about the data split
    logger.info(f"Training set: {X_train.shape}")
    logger.info(f"Test set: {X_test.shape}")
    logger.info(f"Training target distribution: {pd.Series(y_train).value_counts().to_dict()}")
    logger.info(f"Test target distribution: {pd.Series(y_test).value_counts().to_dict()}")
    
    # Check if target is string/categorical and log
    if pd.api.types.is_object_dtype(y_train) or pd.api.types.is_categorical_dtype(y_train):
        unique_values = sorted(set(pd.concat([y_train, y_test])))
        logger.info(f"Target variable is categorical/string with values: {unique_values}")
        if len(unique_values) == 2:
            logger.info(f"Binary classification detected with class labels: {unique_values}")
            logger.info(f"Treating '{unique_values[1]}' as the positive class for metrics")
    
    # Check and handle non-numeric columns
    numeric_columns = X_train.select_dtypes(include=[np.number]).columns
    non_numeric_columns = X_train.select_dtypes(exclude=[np.number]).columns
    
    if len(non_numeric_columns) > 0:
        logger.info(f"Found {len(non_numeric_columns)} non-numeric columns that will be excluded:")
        for col in non_numeric_columns:
            logger.info(f"  - {col} (dtype: {X_train[col].dtype})")
        
        # Use only numeric columns
        X_train = X_train[numeric_columns]
        X_test = X_test[numeric_columns]
        logger.info(f"After excluding non-numeric columns: {X_train.shape[1]} features")
    else:
        logger.info("All feature columns are numeric")
    
    # Standardize features for better model performance
    logger.info("Standardizing features...")
    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    logger.info("Features standardized")
    
    return X_train, X_test, y_train, y_test

def create_ml_models(X_train, y_train, random_state=42, use_hyperparameter_tuning=True):
    """
    Create a dictionary of machine learning models with optional hyperparameter tuning
    
    Args:
        X_train: Training features
        y_train: Training target variable
        random_state: Random seed for reproducibility
        use_hyperparameter_tuning: Whether to perform hyperparameter tuning
        
    Returns:
        Dictionary of initialized ML models
    """
    # Ensure all models are available in this function's scope
    from sklearn.svm import SVC  # Add this import to avoid UnboundLocalError
    
    logger = get_logger(__name__)
    models = {}
    
    # Features are already standardized in the load_data function
    X_train_scaled = X_train
    
    # Determine if we should use hyperparameter tuning based on dataset size
    # For very large datasets, tune with a subset of hyperparameters or skip tuning
    dataset_size_large = X_train.shape[0] > 10000 or X_train.shape[1] > 1000
    if dataset_size_large:
        logger.info("Large dataset detected. Using limited hyperparameter tuning.")
    
    # Random Forest with hyperparameter tuning
    if use_hyperparameter_tuning and not dataset_size_large:
        logger.info("Creating RandomForest with hyperparameter tuning")
        rf_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        rf_base = RandomForestClassifier(random_state=random_state)
        
        # Use GridSearchCV with improved parameters matching run_feature_selection.py
        models['RandomForest'] = optimize_model_hyperparameters(
            rf_base, X_train_scaled, y_train, rf_params, cv=5, scoring='accuracy'
        )
        logger.info("RandomForest hyperparameter optimization completed")
    else:
        models['RandomForest'] = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=random_state
        )
    
    # Gradient Boosting with hyperparameter tuning
    if use_hyperparameter_tuning and not dataset_size_large:
        logger.info("Creating GradientBoosting with hyperparameter tuning")
        gb_params = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'subsample': [0.8, 0.9, 1.0]
        }
        gb_base = GradientBoostingClassifier(random_state=random_state)
        
        # Use GridSearchCV with improved parameters
        models['GradientBoosting'] = optimize_model_hyperparameters(
            gb_base, X_train_scaled, y_train, gb_params, cv=5, scoring='accuracy'
        )
        logger.info("GradientBoosting hyperparameter optimization completed")
    else:
        models['GradientBoosting'] = GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=random_state
        )
    
    # AdaBoost Classifier with hyperparameter tuning
    if use_hyperparameter_tuning and not dataset_size_large:
        logger.info("Creating AdaBoost with hyperparameter tuning")
        ada_params = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1.0],
            'algorithm': ['SAMME', 'SAMME.R']
        }
        ada_base = AdaBoostClassifier(random_state=random_state)
        models['AdaBoost'] = optimize_model_hyperparameters(
            ada_base, X_train_scaled, y_train, ada_params, cv=5, scoring='accuracy'
        )
        logger.info("AdaBoost hyperparameter optimization completed")
    else:
        models['AdaBoost'] = AdaBoostClassifier(
            n_estimators=100, random_state=random_state
        )
    
    # Add models from optional packages if available
    if HAVE_XGBOOST:
        if use_hyperparameter_tuning and not dataset_size_large:
            logger.info("Creating XGBoost with hyperparameter tuning")
            xgb_params = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'min_child_weight': [1, 3, 5]
            }
            xgb_base = xgb.XGBClassifier(random_state=random_state)
            # Use GridSearchCV with improved parameters
            models['XGBoost'] = optimize_model_hyperparameters(
                xgb_base, X_train_scaled, y_train, xgb_params, cv=5, scoring='accuracy'
            )
            logger.info("XGBoost hyperparameter optimization completed")
        else:
            models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=random_state
            )
    
    if HAVE_LIGHTGBM:
        if use_hyperparameter_tuning and not dataset_size_large:
            logger.info("Creating LightGBM with hyperparameter tuning")
            lgb_params = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, -1],  # -1 means no limit
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 100],
                'subsample': [0.8, 0.9, 1.0]
            }
            lgb_base = lgb.LGBMClassifier(random_state=random_state)
            # Use GridSearchCV with improved parameters
            models['LightGBM'] = optimize_model_hyperparameters(
                lgb_base, X_train_scaled, y_train, lgb_params, cv=5, scoring='accuracy'
            )
            logger.info("LightGBM hyperparameter optimization completed")
        else:
            models['LightGBM'] = lgb.LGBMClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=random_state
            )
    
    if HAVE_CATBOOST:
        if use_hyperparameter_tuning and not dataset_size_large:
            logger.info("Creating CatBoost with hyperparameter tuning")
            cb_params = {
                'iterations': [100, 200, 300],
                'depth': [4, 6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'l2_leaf_reg': [1, 3, 5, 7, 9]
            }
            cb_base = cb.CatBoostClassifier(random_state=random_state, verbose=0)
            # Use GridSearchCV with improved parameters
            models['CatBoost'] = optimize_model_hyperparameters(
                cb_base, X_train_scaled, y_train, cb_params, cv=5, scoring='accuracy'
            )
            logger.info("CatBoost hyperparameter optimization completed")
        else:
            models['CatBoost'] = cb.CatBoostClassifier(
                iterations=100, depth=5, learning_rate=0.1, random_state=random_state, verbose=0
            )
    
    # Only initialize SVM and LogisticRegression for small to medium sized datasets
    if X_train.shape[0] < 10000 and X_train.shape[1] < 1000:
        # SVM with hyperparameter tuning
        if use_hyperparameter_tuning:
            logger.info("Creating SVM with hyperparameter tuning")
            svm_params = {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto', 0.1, 0.01]
            }
            svm_base = SVC(probability=True, random_state=random_state)
            # Use GridSearchCV with improved parameters
            try:
                models['SVM'] = optimize_model_hyperparameters(
                    svm_base, X_train_scaled, y_train, svm_params, cv=5, scoring='accuracy'
                )
                logger.info("SVM hyperparameter optimization completed")
            except Exception as e:
                logger.warning(f"SVM hyperparameter tuning failed: {str(e)}. Using default SVM.")
                models['SVM'] = SVC(probability=True, kernel='rbf', C=1.0, random_state=random_state)
        else:
            models['SVM'] = SVC(probability=True, kernel='rbf', random_state=random_state)
        
        logger.info("Initialized SVM with probability=True for ROC curve support")
        
        # Logistic Regression with hyperparameter tuning
        if use_hyperparameter_tuning:
            logger.info("Creating LogisticRegression with hyperparameter tuning")
            lr_params = {
                'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'solver': ['liblinear', 'lbfgs', 'saga'],
                'penalty': ['l1', 'l2', 'elasticnet', None],
                'max_iter': [1000, 2000, 3000]
            }
            lr_base = LogisticRegression(random_state=random_state)
            # Use GridSearchCV with improved parameters
            try:
                models['LogisticRegression'] = optimize_model_hyperparameters(
                    lr_base, X_train_scaled, y_train, lr_params, cv=5, scoring='accuracy'
                )
                logger.info("LogisticRegression hyperparameter optimization completed")
            except Exception as e:
                logger.warning(f"LogisticRegression hyperparameter tuning failed: {str(e)}. Using default LogisticRegression.")
                models['LogisticRegression'] = LogisticRegression(max_iter=1000, C=1.0, random_state=random_state)
        else:
            models['LogisticRegression'] = LogisticRegression(max_iter=1000, C=1.0, random_state=random_state)
        
        # KNN classifier
        if use_hyperparameter_tuning:
            logger.info("Creating KNN with hyperparameter tuning")
            knn_params = {
                'n_neighbors': [3, 5, 7, 9, 11, 15],
                'weights': ['uniform', 'distance'],
                'p': [1, 2],  # 1: Manhattan, 2: Euclidean
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            }
            knn_base = KNeighborsClassifier()
            # Use GridSearchCV with improved parameters
            models['KNN'] = optimize_model_hyperparameters(
                knn_base, X_train_scaled, y_train, knn_params, cv=5, scoring='accuracy'
            )
            logger.info("KNN hyperparameter optimization completed")
        else:
            models['KNN'] = KNeighborsClassifier(n_neighbors=5)
    
    # Verify SVM is properly configured
    if 'SVM' in models:
        # Check if SVM has probability enabled
        has_prob = getattr(models['SVM'], 'probability', False)
        if not has_prob:
            logger.warning("SVM model doesn't have probability=True, setting it now")
            # Re-initialize with probability=True
            # SVC is already imported at the top of the function
            models['SVM'] = SVC(probability=True, kernel='rbf', random_state=random_state)
    
    return {k: v for k, v in models.items() if v is not None}

def train_and_evaluate_models(models, X_train, X_test, y_train, y_test, output_dir):
    """
    Train and evaluate all models
    
    Args:
        models: Dictionary of ML models
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        output_dir: Directory to save results
        
    Returns:
        Dictionary of model results and best model (selected based on test accuracy)
    """
    logger = get_logger(__name__)
    logger.info("Training and evaluating models...")
    
    results = {}
    best_accuracy = 0
    best_model_name = None
    best_model = None
    
    # Determine if we have string labels and need to handle pos_label
    has_string_labels = pd.api.types.is_object_dtype(y_test) or pd.api.types.is_categorical_dtype(y_test)
    pos_label = None
    
    if has_string_labels:
        unique_classes = sorted(set(pd.concat([y_train, y_test])))
        if len(unique_classes) == 2:
            # For binary classification with 'IR' and 'IS', we'll use 'IR' as positive class
            # If other classes, fall back to using the second class as positive
            if 'IR' in unique_classes and 'IS' in unique_classes:
                pos_label = 'IR'
                logger.info("Explicitly using 'IR' as the positive class (1) for metrics calculation")
                label_map = {'IS': 0, 'IR': 1}
            else:
                pos_label = unique_classes[1]
                logger.info(f"Using '{pos_label}' as the positive class for metrics calculation")
                # For internal model use, we need to convert string labels to numbers
                label_map = {unique_classes[0]: 0, unique_classes[1]: 1}
            
            y_train_numeric = y_train.map(label_map)
            y_test_numeric = y_test.map(label_map)
            
            # Log the mapping
            logger.info(f"Label mapping for internal calculations: {label_map}")
            logger.info(f"Checking for label consistency: train classes={set(y_train)}, test classes={set(y_test)}")
        else:
            logger.info(f"Multi-class classification with {len(unique_classes)} classes detected")
            # For multi-class, no need to specify pos_label
            y_train_numeric = y_train
            y_test_numeric = y_test
    else:
        # Use numeric labels as-is
        y_train_numeric = y_train
        y_test_numeric = y_test
    
    # Create ROC curve plot
    plt.figure(figsize=(12, 10))
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        start_time = time.time()
        
        # Train the model - use numeric labels if needed for internal calculations
        model.fit(X_train, y_train_numeric if has_string_labels and len(unique_classes) == 2 else y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Convert predictions back to original labels if needed
        if isinstance(y_pred, np.ndarray):
            y_pred = pd.Series(y_pred, index=y_test.index)
            
        # If we have string labels and we're dealing with a binary classification problem
        if has_string_labels and len(unique_classes) == 2:
            # If the model was trained on numeric labels, map predictions back to string labels
            if hasattr(y_train, 'map') and 'label_map' in locals():
                # Create reverse mapping from numeric back to string labels
                reverse_map = {v: k for k, v in label_map.items()}
                y_pred = y_pred.map(reverse_map)
        
        # Get probability estimates - check if model supports predict_proba
        if hasattr(model, 'predict_proba') and callable(model.predict_proba):
            try:
                # Always use column 1 for binary classification
                y_prob = model.predict_proba(X_test)[:, 1]
            except:
                # If predict_proba fails, use decision_function if available
                logger.warning(f"{name} model failed with predict_proba, trying decision_function")
                if hasattr(model, 'decision_function') and callable(model.decision_function):
                    y_prob = model.decision_function(X_test)
                    # Normalize scores if needed
                    y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-8)
                else:
                    # Use 0/1 as a last resort
                    logger.warning(f"{name} model has no probability method, using predicted labels")
                    y_prob = np.array(y_pred).astype(float)
        elif hasattr(model, 'decision_function') and callable(model.decision_function):
            # Use decision_function for SVC without probability=True
            logger.warning(f"{name} model has no predict_proba, using decision_function")
            y_prob = model.decision_function(X_test)
            # Normalize scores to [0, 1] range for ROC curve
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-8)
        else:
            # No probability estimation available, use predicted labels
            logger.warning(f"{name} model has no probability method, using predicted labels")
            y_prob = np.array(y_pred).astype(float)
        
        # Calculate metrics with appropriate pos_label
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calculate ROC curve and AUC
        if has_string_labels and len(unique_classes) == 2:
            fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=pos_label)
        else:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate other metrics with appropriate pos_label
        if has_string_labels and pos_label:
            sensitivity = recall_score(y_test, y_pred, pos_label=pos_label)
            precision = precision_score(y_test, y_pred, pos_label=pos_label)
            f1 = f1_score(y_test, y_pred, pos_label=pos_label)
            specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # TN / (TN + FP)
        else:
            sensitivity = recall_score(y_test, y_pred)  # Same as recall
            precision = precision_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # TN / (TN + FP)
        
        # Store results
        model_results = {
            'accuracy': float(accuracy),
            'auc': float(roc_auc),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'precision': float(precision),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'training_time': format_duration(time.time() - start_time)
        }
        
        results[name] = model_results
        
        # Plot ROC curve
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
        
        # Check if this is the best model - selecting based on accuracy instead of AUC
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            best_model = model
        
        logger.info(f"  {name} - Accuracy: {accuracy:.4f}, AUC: {roc_auc:.4f}, F1: {f1:.4f}")
    
    # Finalize ROC curve plot
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Save ROC curve
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves_all_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create metrics comparison table plot
    create_metrics_table_plot(results, output_dir, 'all_features')
    
    logger.info(f"Best model: {best_model_name} (Accuracy = {best_accuracy:.4f})")
    
    return results, best_model_name, best_model

def train_and_evaluate_top_features_models(models, X_train, X_test, y_train, y_test, top_features, output_dir):
    """
    Train and evaluate all models using only the top features
    
    Args:
        models: Dictionary of ML models
        X_train: Training features (full feature set)
        X_test: Test features (full feature set)
        y_train: Training target
        y_test: Test target
        top_features: List of top feature names to use
        output_dir: Directory to save results
        
    Returns:
        Dictionary of model results and best model (selected based on test accuracy)
    """
    logger = get_logger(__name__)
    logger.info(f"Training and evaluating models using only top {len(top_features)} features...")
    
    # Filter X_train and X_test to include only top features
    X_train_top = X_train[top_features]
    X_test_top = X_test[top_features]
    
    results = {}
    best_accuracy = 0
    best_model_name = None
    best_model = None
    
    # Determine if we have string labels and need to handle pos_label
    has_string_labels = pd.api.types.is_object_dtype(y_test) or pd.api.types.is_categorical_dtype(y_test)
    pos_label = None
    
    if has_string_labels:
        unique_classes = sorted(set(pd.concat([y_train, y_test])))
        if len(unique_classes) == 2:
            # For binary classification with 'IR' and 'IS', we'll use 'IR' as positive class
            # If other classes, fall back to using the second class as positive
            if 'IR' in unique_classes and 'IS' in unique_classes:
                pos_label = 'IR'
                logger.info("Explicitly using 'IR' as the positive class (1) for metrics calculation")
                label_map = {'IS': 0, 'IR': 1}
            else:
                pos_label = unique_classes[1]
                logger.info(f"Using '{pos_label}' as the positive class for metrics calculation")
                # For internal model use, we need to convert string labels to numbers
                label_map = {unique_classes[0]: 0, unique_classes[1]: 1}
            
            y_train_numeric = y_train.map(label_map)
            y_test_numeric = y_test.map(label_map)
            
            # Log the mapping
            logger.info(f"Label mapping for internal calculations: {label_map}")
            logger.info(f"Checking for label consistency: train classes={set(y_train)}, test classes={set(y_test)}")
        else:
            logger.info(f"Multi-class classification with {len(unique_classes)} classes detected")
            # For multi-class, no need to specify pos_label
            y_train_numeric = y_train
            y_test_numeric = y_test
    else:
        # Use numeric labels as-is
        y_train_numeric = y_train
        y_test_numeric = y_test
    
    # Create ROC curve plot for top features
    plt.figure(figsize=(12, 10))
    
    for name, model in models.items():
        logger.info(f"Training {name} with top features...")
        start_time = time.time()
        
        # Create a fresh model instance to avoid any data leakage
        if hasattr(model, 'get_params'):
            params = model.get_params()
            # Create a new model with the same parameters
            if hasattr(model, '__class__'):
                model_for_top = model.__class__(**params)
            else:
                # If we can't create a new instance, use the original model
                model_for_top = model
        else:
            # If we can't get parameters, use the original model
            model_for_top = model
        
        # Train the model - use numeric labels if needed for internal calculations
        model_for_top.fit(X_train_top, y_train_numeric if has_string_labels and len(unique_classes) == 2 else y_train)
        
        # Make predictions
        y_pred = model_for_top.predict(X_test_top)
        
        # Convert predictions back to original labels if needed
        if isinstance(y_pred, np.ndarray):
            y_pred = pd.Series(y_pred, index=y_test.index)
            
        # If we have string labels and we're dealing with a binary classification problem
        if has_string_labels and len(unique_classes) == 2:
            # If the model was trained on numeric labels, map predictions back to string labels
            if hasattr(y_train, 'map') and 'label_map' in locals():
                # Create reverse mapping from numeric back to string labels
                reverse_map = {v: k for k, v in label_map.items()}
                y_pred = y_pred.map(reverse_map)
        
        # Get probability estimates - check if model supports predict_proba
        if hasattr(model_for_top, 'predict_proba') and callable(model_for_top.predict_proba):
            try:
                # Always use column 1 for binary classification
                y_prob = model_for_top.predict_proba(X_test_top)[:, 1]
            except:
                # If predict_proba fails, use decision_function if available
                logger.warning(f"{name} model failed with predict_proba, trying decision_function")
                if hasattr(model_for_top, 'decision_function') and callable(model_for_top.decision_function):
                    y_prob = model_for_top.decision_function(X_test_top)
                    # Normalize scores if needed
                    y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-8)
                else:
                    # Use 0/1 as a last resort
                    logger.warning(f"{name} model has no probability method, using predicted labels")
                    y_prob = np.array(y_pred).astype(float)
        elif hasattr(model_for_top, 'decision_function') and callable(model_for_top.decision_function):
            # Use decision_function for SVC without probability=True
            logger.warning(f"{name} model has no predict_proba, using decision_function")
            y_prob = model_for_top.decision_function(X_test_top)
            # Normalize scores to [0, 1] range for ROC curve
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-8)
        else:
            # No probability estimation available, use predicted labels
            logger.warning(f"{name} model has no probability method, using predicted labels")
            y_prob = np.array(y_pred).astype(float)
        
        # Calculate metrics with appropriate pos_label
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calculate ROC curve and AUC
        if has_string_labels and len(unique_classes) == 2:
            fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=pos_label)
        else:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate other metrics with appropriate pos_label
        if has_string_labels and pos_label:
            sensitivity = recall_score(y_test, y_pred, pos_label=pos_label)
            precision = precision_score(y_test, y_pred, pos_label=pos_label)
            f1 = f1_score(y_test, y_pred, pos_label=pos_label)
            specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # TN / (TN + FP)
        else:
            sensitivity = recall_score(y_test, y_pred)  # Same as recall
            precision = precision_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # TN / (TN + FP)
        
        # Store results
        model_results = {
            'accuracy': float(accuracy),
            'auc': float(roc_auc),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'precision': float(precision),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'training_time': format_duration(time.time() - start_time)
        }
        
        results[name] = model_results
        
        # Plot ROC curve
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
        
        # Check if this is the best model - selecting based on accuracy instead of AUC
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            best_model = model_for_top
        
        logger.info(f"  {name} - Accuracy: {accuracy:.4f}, AUC: {roc_auc:.4f}, F1: {f1:.4f}")
    
    # Finalize ROC curve plot
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - Top {len(top_features)} Features')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Save ROC curve for top features
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves_top_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create metrics comparison table plot for top features
    create_metrics_table_plot(results, output_dir, 'top_features')
    
    logger.info(f"Best model with top features: {best_model_name} (Accuracy = {best_accuracy:.4f})")
    
    return results, best_model_name, best_model

def extract_top_features(model, feature_names, top_n=20, X_train=None, y_train=None):
    """
    Extract the top N most important features from a model
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to extract
        X_train: Training data (used only if model doesn't have feature importances)
        y_train: Training labels (used only if model doesn't have feature importances)
        
    Returns:
        List of top feature names
    """
    logger = get_logger(__name__)
    
    # Check if model has feature_importances_ attribute
    if hasattr(model, 'feature_importances_'):
        # Random Forest, XGBoost, etc.
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Linear models like Logistic Regression
        importances = np.abs(model.coef_[0])
    else:
        logger.warning("Model doesn't have feature_importances_ or coef_ attribute")
        # Select features based on mutual information if training data is provided
        if X_train is not None and y_train is not None:
            from sklearn.feature_selection import mutual_info_classif
            importances = mutual_info_classif(X_train, y_train)
        else:
            # Fallback: assign equal importance to all features
            logger.warning("No feature importance method available, assigning equal weights")
            importances = np.ones(len(feature_names))
    
    # Create feature importance DataFrame
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # Get top features
    top_features = feature_importance.head(top_n)['feature'].tolist()
    
    return top_features, feature_importance

def create_metrics_table_plot(results, output_dir, suffix):
    """
    Create a visual table of model performance metrics
    
    Args:
        results: Dictionary of model results
        output_dir: Directory to save results
        suffix: Suffix for the output file name
    """
    # Prepare data for the table
    metrics = ['accuracy', 'auc', 'sensitivity', 'specificity', 'precision', 'f1_score']
    models = list(results.keys())
    
    # Create a DataFrame for the table
    table_data = []
    for metric in metrics:
        row = [metric.title()]
        for model in models:
            row.append(f"{results[model][metric]:.4f}")
        table_data.append(row)
    
    df = pd.DataFrame(table_data, columns=['Metric'] + models)
    
    # Convert to numeric for heatmap (skip first column)
    heatmap_data = df.iloc[:, 1:].astype(float)
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, len(metrics) * 0.8 + 2))
    
    # Create heatmap
    cmap = plt.cm.YlGnBu
    im = ax.imshow(heatmap_data.values, cmap=cmap, aspect='auto')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Score", rotation=-90, va="bottom")
    
    # Add labels
    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels(models)
    ax.set_yticklabels(metrics)
    
    # Rotate the tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(metrics)):
        for j in range(len(models)):
            text = ax.text(j, i, f"{float(heatmap_data.iloc[i, j]):.4f}",
                          ha="center", va="center", color="black" if float(heatmap_data.iloc[i, j]) < 0.7 else "white",
                          fontweight="bold")
    
    # Add title and adjust layout
    ax.set_title(f"Model Performance Metrics Comparison")
    fig.tight_layout()
    
    # Save figure
    plt.savefig(output_dir / f'performance_metrics_{suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()

def compare_feature_sets(X_train, X_test, y_train, y_test, top_features, best_model_class, output_dir, random_state=42):
    """
    Compare model performance using all features vs. top features
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        top_features: List of top feature names
        best_model_class: Best model class for retraining
        output_dir: Directory to save results
        random_state: Random seed for reproducibility
    """
    logger = get_logger(__name__)
    logger.info(f"Comparing performance with all features vs top {len(top_features)} features")
    
    # Get top features subset
    X_train_top = X_train[top_features]
    X_test_top = X_test[top_features]
    
    # Determine if we have string labels and need to handle pos_label
    has_string_labels = pd.api.types.is_object_dtype(y_test) or pd.api.types.is_categorical_dtype(y_test)
    pos_label = None
    unique_classes = None
    
    if has_string_labels:
        unique_classes = sorted(set(pd.concat([y_train, y_test])))
        if len(unique_classes) == 2:
            # For binary classification with 'IR' and 'IS', we'll use 'IR' as positive class
            # If other classes, fall back to using the second class as positive
            if 'IR' in unique_classes and 'IS' in unique_classes:
                pos_label = 'IR'
                logger.info("Explicitly using 'IR' as the positive class (1) for feature comparison metrics")
                label_map = {'IS': 0, 'IR': 1}
            else:
                pos_label = unique_classes[1]
                logger.info(f"Using '{pos_label}' as the positive class for feature comparison metrics")
                # For internal model use, we need to convert string labels to numbers
                label_map = {unique_classes[0]: 0, unique_classes[1]: 1}
            
            y_train_numeric = y_train.map(label_map)
            y_test_numeric = y_test.map(label_map)
        else:
            logger.info(f"Multi-class feature comparison with {len(unique_classes)} classes detected")
            y_train_numeric = y_train
            y_test_numeric = y_test
    else:
        # Use numeric labels as-is
        y_train_numeric = y_train
        y_test_numeric = y_test
    
    # Initialize a new instance of the best model with top features
    if isinstance(best_model_class, RandomForestClassifier):
        model_top = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=random_state
        )
    elif HAVE_XGBOOST and isinstance(best_model_class, xgb.XGBClassifier):
        model_top = xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=random_state
        )
    elif HAVE_LIGHTGBM and isinstance(best_model_class, lgb.LGBMClassifier):
        model_top = lgb.LGBMClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=random_state
        )
    elif HAVE_CATBOOST and isinstance(best_model_class, cb.CatBoostClassifier):
        model_top = cb.CatBoostClassifier(
            iterations=100, depth=5, learning_rate=0.1, random_state=random_state, verbose=0
        )
    elif hasattr(best_model_class, '__class__') and best_model_class.__class__.__name__ == 'SVC':
        # For SVC models, make sure probability=True is set
        # SVC is already imported at the top of the function
        logger.info("Creating SVC model for top features with probability=True")
        model_top = SVC(probability=True, kernel='rbf', random_state=random_state)
    else:
        try:
            # Try to create a new instance of the same class
            model_top = best_model_class.__class__()
            # If it's an SVC, ensure probability=True
            if hasattr(model_top, '__class__') and model_top.__class__.__name__ == 'SVC':
                model_top.probability = True
                logger.info("Set probability=True for new SVC instance")
        except:
            logger.warning("Could not create a new instance of the best model class")
            model_top = RandomForestClassifier(n_estimators=100, random_state=random_state)
    
    # Train model with top features - use numeric labels if needed
    if has_string_labels and len(unique_classes) == 2:
        model_top.fit(X_train_top, y_train_numeric)
    else:
        model_top.fit(X_train_top, y_train)
    
    # Make predictions
    y_pred_top = model_top.predict(X_test_top)
    
    # Convert predictions back if needed
    if has_string_labels and len(unique_classes) == 2 and label_map:
        # Map numeric predictions back to string labels for evaluation
        reverse_map = {v: k for k, v in label_map.items()}
        y_pred_top = pd.Series(y_pred_top).map(reverse_map)
    
    # Get probability estimates - check if model supports predict_proba
    if hasattr(model_top, 'predict_proba') and callable(model_top.predict_proba):
        try:
            # Always use column 1 for binary classification
            y_prob_top = model_top.predict_proba(X_test_top)[:, 1]
            logger.info("Successfully used predict_proba for top features model")
        except Exception as e:
            # If predict_proba fails, use decision_function if available
            logger.warning(f"Error with predict_proba in top features model: {str(e)}")
            if hasattr(model_top, 'decision_function') and callable(model_top.decision_function):
                logger.info("Falling back to decision_function for top features model")
                y_prob_top = model_top.decision_function(X_test_top)
                # Normalize scores if needed
                y_prob_top = (y_prob_top - y_prob_top.min()) / (y_prob_top.max() - y_prob_top.min() + 1e-8)
            else:
                # Use 0/1 as a last resort
                logger.warning("No probability method available, using predicted labels")
                y_prob_top = np.array(y_pred_top).astype(float)
    elif hasattr(model_top, 'decision_function') and callable(model_top.decision_function):
        # Use decision_function for SVC without probability=True
        logger.warning("Model has no predict_proba, using decision_function")
        y_prob_top = model_top.decision_function(X_test_top)
        # Normalize scores to [0, 1] range for ROC curve
        y_prob_top = (y_prob_top - y_prob_top.min()) / (y_prob_top.max() - y_prob_top.min() + 1e-8)
    else:
        # No probability estimation available, use predicted labels
        logger.warning("Model has no probability method, using predicted labels")
        y_prob_top = np.array(y_pred_top).astype(float)
    
    # Calculate accuracy with original labels
    accuracy_top = accuracy_score(y_test, y_pred_top)
    
    # Calculate ROC curve using numeric values if needed
    if has_string_labels and len(unique_classes) == 2:
        fpr_top, tpr_top, _ = roc_curve(y_test, y_prob_top, pos_label=pos_label)
    else:
        fpr_top, tpr_top, _ = roc_curve(y_test, y_prob_top)
    roc_auc_top = auc(fpr_top, tpr_top)
    
    # Create comparison plot
    plt.figure(figsize=(10, 8))
    
    # Get original metrics for the best model - with error handling
    if hasattr(best_model_class, 'predict_proba') and callable(best_model_class.predict_proba):
        try:
            y_prob_all = best_model_class.predict_proba(X_test)[:, 1]
        except Exception as e:
            logger.warning(f"Error using predict_proba with best model: {str(e)}")
            # Fall back to decision_function if available
            if hasattr(best_model_class, 'decision_function') and callable(best_model_class.decision_function):
                logger.info("Using decision_function for best model")
                y_prob_all = best_model_class.decision_function(X_test)
                # Normalize scores
                y_prob_all = (y_prob_all - y_prob_all.min()) / (y_prob_all.max() - y_prob_all.min() + 1e-8)
            else:
                # Last resort: use predicted labels
                logger.warning("No probability method available for best model, using labels")
                y_prob_all = np.array(best_model_class.predict(X_test)).astype(float)
    elif hasattr(best_model_class, 'decision_function') and callable(best_model_class.decision_function):
        logger.info("Best model doesn't have predict_proba, using decision_function")
        y_prob_all = best_model_class.decision_function(X_test)
        # Normalize scores
        y_prob_all = (y_prob_all - y_prob_all.min()) / (y_prob_all.max() - y_prob_all.min() + 1e-8)
    else:
        # Last resort: use predicted labels
        logger.warning("No probability method available for best model, using labels")
        y_prob_all = np.array(best_model_class.predict(X_test)).astype(float)
    
    # Calculate ROC curve with numeric values if needed
    if has_string_labels and len(unique_classes) == 2:
        fpr_all, tpr_all, _ = roc_curve(y_test, y_prob_all, pos_label=pos_label)
    else:
        fpr_all, tpr_all, _ = roc_curve(y_test, y_prob_all)
    
    roc_auc_all = auc(fpr_all, tpr_all)
    
    # Plot ROC curves
    plt.plot(fpr_all, tpr_all, color='blue', lw=2,
             label=f'All Features (AUC = {roc_auc_all:.3f})')
    plt.plot(fpr_top, tpr_top, color='red', lw=2,
             label=f'Top {len(top_features)} Features (AUC = {roc_auc_top:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: All Features vs. Top Features')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Save comparison plot
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_comparison_top_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create feature importance plot
    if hasattr(model_top, 'feature_importances_'):
        plt.figure(figsize=(12, len(top_features) * 0.3 + 2))
        
        # Get feature importances
        importances = model_top.feature_importances_
        feature_imp = pd.DataFrame({
            'feature': top_features,
            'importance': importances
        }).sort_values('importance', ascending=True)
        
        # Create horizontal bar plot
        plt.barh(range(len(feature_imp)), feature_imp['importance'], color='skyblue')
        plt.yticks(range(len(feature_imp)), feature_imp['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {len(top_features)} Feature Importances')
        plt.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'top_feature_importances.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Return comparison results
    y_pred_all = best_model_class.predict(X_test)
    if has_string_labels and len(unique_classes) == 2 and label_map:
        # Convert predictions back if they are numeric
        if hasattr(y_pred_all, 'dtype') and np.issubdtype(y_pred_all.dtype, np.number):
            reverse_map = {v: k for k, v in label_map.items()}
            y_pred_all = pd.Series(y_pred_all).map(reverse_map)
    
    # Calculate metrics for comparison
    # For top features model
    if has_string_labels and len(unique_classes) == 2:
        accuracy_top = accuracy_score(y_test, y_pred_top)
        cm_top = confusion_matrix(y_test, y_pred_top)
        sensitivity_top = recall_score(y_test, y_pred_top, pos_label=pos_label)
        precision_top = precision_score(y_test, y_pred_top, pos_label=pos_label)
        f1_top = f1_score(y_test, y_pred_top, pos_label=pos_label)
        specificity_top = cm_top[0, 0] / (cm_top[0, 0] + cm_top[0, 1])
    else:
        accuracy_top = accuracy_score(y_test, y_pred_top)
        cm_top = confusion_matrix(y_test, y_pred_top)
        sensitivity_top = recall_score(y_test, y_pred_top)
        precision_top = precision_score(y_test, y_pred_top)
        f1_top = f1_score(y_test, y_pred_top)
        specificity_top = cm_top[0, 0] / (cm_top[0, 0] + cm_top[0, 1])
        
    # For all features model
    if has_string_labels and len(unique_classes) == 2:
        accuracy_all = accuracy_score(y_test, y_pred_all)
        cm_all = confusion_matrix(y_test, y_pred_all)
        sensitivity_all = recall_score(y_test, y_pred_all, pos_label=pos_label)
        precision_all = precision_score(y_test, y_pred_all, pos_label=pos_label)
        f1_all = f1_score(y_test, y_pred_all, pos_label=pos_label)
        specificity_all = cm_all[0, 0] / (cm_all[0, 0] + cm_all[0, 1])
    else:
        accuracy_all = accuracy_score(y_test, y_pred_all)
        cm_all = confusion_matrix(y_test, y_pred_all)
        sensitivity_all = recall_score(y_test, y_pred_all)
        precision_all = precision_score(y_test, y_pred_all)
        f1_all = f1_score(y_test, y_pred_all)
        specificity_all = cm_all[0, 0] / (cm_all[0, 0] + cm_all[0, 1])
    
    # Create comprehensive comparison dictionary
    comparison = {
        'all_features': {
            'accuracy': accuracy_all,
            'auc': roc_auc_all,
            'sensitivity': sensitivity_all,
            'specificity': specificity_all,
            'precision': precision_all,
            'f1_score': f1_all,
            'feature_count': X_train.shape[1],
            'confusion_matrix': cm_all.tolist()
        },
        'top_features': {
            'accuracy': accuracy_top,
            'auc': roc_auc_top,
            'sensitivity': sensitivity_top,
            'specificity': specificity_top,
            'precision': precision_top,
            'f1_score': f1_top,
            'feature_count': len(top_features),
            'confusion_matrix': cm_top.tolist()
        }
    }
    
    # Plot confusion matrices for both models
    plot_confusion_matrix(y_test, y_pred_all, output_dir, 'confusion_matrix_all_features.png', 
                          f'Confusion Matrix - All Features ({X_train.shape[1]})')
    plot_confusion_matrix(y_test, y_pred_top, output_dir, 'confusion_matrix_top_features.png', 
                          f'Confusion Matrix - Top {len(top_features)} Features')
    
    logger.info("Performance comparison:")
    logger.info(f"  All features ({X_train.shape[1]}): AUC = {roc_auc_all:.4f}, Accuracy = {accuracy_all:.4f}")
    logger.info(f"  Top features ({len(top_features)}): AUC = {roc_auc_top:.4f}, Accuracy = {accuracy_top:.4f}")
    
    return comparison, model_top

def train_top_features(X_train, X_test, y_train, y_test, best_model_name, best_model_class, 
                     top_features, output_dir, random_state=42):
    """Train the best model using only the top features"""
    logger = get_logger(__name__)
    
    # Ensure all required model classes are imported
    from sklearn.svm import SVC
    
    # Filter top features
    if top_features and len(top_features) > 0:
        logger.info(f"Training best model ({best_model_name}) with top {len(top_features)} features...")
        
        # Select only top features
        X_train_top = X_train[top_features]
        X_test_top = X_test[top_features]
        
        # Save top feature datasets
        logger.info(f"Saving ML-ready train/test datasets with top features...")
        X_train_top.to_csv(output_dir / 'ml_ready_train_top_features.csv', index=False)
        X_test_top.to_csv(output_dir / 'ml_ready_test_top_features.csv', index=False)
    
def optimize_model_hyperparameters(model, X, y, param_grid, cv=5, scoring='roc_auc'):
    """
    Optimize model hyperparameters using GridSearchCV
    
    Args:
        model: Base model to optimize
        X: Features dataframe
        y: Target series
        param_grid: Dictionary of hyperparameter grids
        cv: Number of cross-validation folds
        scoring: Scoring metric for evaluation
        
    Returns:
        Best model with optimized hyperparameters
    """
    logger = get_logger(__name__)
    logger.info(f"Optimizing hyperparameters for {type(model).__name__}")
    
    grid_search = GridSearchCV(
        model, 
        param_grid, 
        cv=cv,
        scoring=scoring,
        n_jobs=-1,  # Use all available processors
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best {scoring} score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def plot_confusion_matrix(y_true, y_pred, output_dir, file_name='confusion_matrix.png', title='Confusion Matrix'):
    """
    Plot confusion matrix for model predictions
    
    Args:
        y_true: True target labels
        y_pred: Predicted target labels
        output_dir: Directory to save the plot
        file_name: Name of the output file
        title: Title for the plot
        
    Returns:
        Path to the saved confusion matrix image
    """
    logger = get_logger(__name__)
    logger.info(f"Plotting confusion matrix...")
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot using seaborn for better styling
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sorted(set(y_true)), 
                yticklabels=sorted(set(y_true)))
    
    # Add labels and title
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title, fontsize=15, fontweight='bold', pad=20)
    
    # Add correct/incorrect labels
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    correct = tn + tp
    accuracy = correct / total
    
    plt.figtext(0.5, 0.01, f'Accuracy: {accuracy:.4f} ({correct}/{total} correct predictions)',
                ha='center', fontsize=12, bbox=dict(facecolor='lightgray', alpha=0.5))
    
    # Save figure
    output_path = output_dir / file_name
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix saved to {output_path}")
    return output_path

def create_comparison_table(all_features_results, top_features_results, best_model_name, output_dir):
    """
    Create a comparison table of all metrics for all features vs top features
    
    Args:
        all_features_results: Dictionary with results for all features
        top_features_results: Dictionary with results for top features
        best_model_name: Name of the best model
        output_dir: Directory to save the plot
        
    Returns:
        Path to the saved comparison table image
    """
    logger = get_logger(__name__)
    logger.info("Creating comprehensive performance comparison table...")
    
    # Metrics to include
    metrics = ['Accuracy', 'AUC', 'Sensitivity', 'Specificity', 'Precision', 'F1 Score']
    
    # Data for the table
    all_features_metrics = [
        all_features_results[best_model_name]['accuracy'],
        all_features_results[best_model_name]['auc'],
        all_features_results[best_model_name]['sensitivity'],
        all_features_results[best_model_name]['specificity'],
        all_features_results[best_model_name]['precision'],
        all_features_results[best_model_name]['f1_score']
    ]
    
    # Extract top features results
    try:
        top_features_metrics = [
            top_features_results['accuracy'],
            top_features_results['auc'],
            top_features_results['sensitivity'], 
            top_features_results['specificity'],
            top_features_results['precision'],
            top_features_results['f1_score']
        ]
    except KeyError as e:
        logger.error(f"Missing key in top_features_results: {e}")
        logger.error(f"Available keys: {list(top_features_results.keys())}")
        # Provide fallback values
        top_features_metrics = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Create figure and axis
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
    
    # Hide axes
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table_data = []
    for i, metric in enumerate(metrics):
        row = [metric, f"{all_features_metrics[i]:.4f}", f"{top_features_metrics[i]:.4f}"]
        table_data.append(row)
    
    # Create table with coloring based on which is better
    cell_colors = []
    for i in range(len(metrics)):
        if all_features_metrics[i] > top_features_metrics[i]:
            cell_colors.append(['white', 'lightgreen', 'white'])
        elif top_features_metrics[i] > all_features_metrics[i]:
            cell_colors.append(['white', 'white', 'lightgreen'])
        else:
            cell_colors.append(['white', 'lightblue', 'lightblue'])
    
    # Get feature counts from X_train and top_features
    # For the table headers, we don't need exact feature counts
    # Let's use a simple description instead
    the_table = ax.table(
        cellText=table_data,
        colLabels=['Metric', 'All Features', 'Top Features'],
        cellColours=cell_colors,
        colWidths=[0.3, 0.35, 0.35],
        loc='center',
        cellLoc='center'
    )
    
    # Style the table
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)
    the_table.scale(1.2, 1.5)
    
    # Add title
    plt.title(f'Performance Comparison for {best_model_name}', fontsize=15, fontweight='bold', pad=20)
    
    # Save the figure
    output_path = output_dir / 'performance_comparison_table.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Performance comparison table saved to {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Supervised Machine Learning Analysis')
    parser.add_argument('--interactive', action='store_true', 
                       help='Run in interactive mode with user prompts')
    parser.add_argument('--train_file', help='Path to training data file (CSV)')
    parser.add_argument('--test_file', help='Path to test data file (CSV)')
    parser.add_argument('--target', help='Target column name')
    parser.add_argument('--use_predefined_split', action='store_true',
                       help='Use predefined train/test split from separate files')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size for random split (0.1-0.5, default: 0.2). Ignored if --use_predefined_split is enabled.')
    parser.add_argument('--output_dir', default='ml_model_results',
                       help='Output directory (default: ml_model_results)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility (default: 42)')
    parser.add_argument('--top_n_features', type=int, default=20,
                       help='Number of top features to extract from best model (default: 20)')
    parser.add_argument('--use_hyperparameter_tuning', action='store_true',
                       help='Enable hyperparameter tuning for models (can significantly increase runtime)')
    
    args = parser.parse_args()
    
    # Get input parameters
    if args.interactive or not args.train_file or not args.test_file or not args.target:
        try:
            train_file, test_file, target, use_predefined_split, test_size, output_dir, use_hyperparameter_tuning, top_n_features = get_user_input()
            if not train_file or not test_file:
                return
            random_state = args.random_state
        except ValueError as e:
            print(f"Error with user input: {str(e)}")
            return
    else:
        train_file = args.train_file
        test_file = args.test_file
        target = args.target
        use_predefined_split = args.use_predefined_split
        test_size = args.test_size
        output_dir = args.output_dir
        top_n_features = args.top_n_features
        random_state = args.random_state
        use_hyperparameter_tuning = args.use_hyperparameter_tuning
    
    # Setup logging
    output_dir = Path(output_dir)
    ensure_directory(output_dir)
    setup_logging(level='INFO', log_file=str(output_dir / "ml_analysis.log"))
    logger = get_logger(__name__)
    
    logger.info("=" * 60)
    logger.info("SUPERVISED MACHINE LEARNING MODEL ANALYSIS")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    try:
        # Step 1: Load and prepare data
        logger.info("Step 1: Loading and preparing data...")
        X_train, X_test, y_train, y_test = load_data(
            train_file, test_file, target, use_predefined_split, test_size, random_state
        )
        
        # Step 2: Create ML models
        logger.info("Step 2: Creating ML models...")
        logger.info(f"Hyperparameter tuning: {'Enabled' if use_hyperparameter_tuning else 'Disabled'}")
        models = create_ml_models(X_train, y_train, random_state, use_hyperparameter_tuning)
        logger.info(f"Created {len(models)} models: {', '.join(models.keys())}")
        
        # Step 3: Train and evaluate models
        logger.info("Step 3: Training and evaluating models...")
        results, best_model_name, best_model = train_and_evaluate_models(
            models, X_train, X_test, y_train, y_test, output_dir
        )
        
        # Step 4: Extract top features from best model (selected based on accuracy)
        logger.info(f"Step 4: Extracting top {top_n_features} features from best model ({best_model_name}) selected based on accuracy...")
        logger.info(f"Using {top_n_features} features for enhanced performance")
        top_features, feature_importance = extract_top_features(
            best_model, X_train.columns, top_n_features, X_train, y_train
        )
        
        # Save top features
        top_features_df = feature_importance.head(top_n_features).reset_index(drop=True)
        top_features_output = output_dir / "top_features.csv"
        top_features_df.to_csv(top_features_output, index=False)
        logger.info(f"Top features saved to: {top_features_output}")
        
        # Step 4.5: Train and evaluate models using only top features
        logger.info(f"Step 4.5: Training and evaluating models using only top {len(top_features)} features...")
        top_features_results, best_top_model_name, best_top_model = train_and_evaluate_top_features_models(
            models, X_train, X_test, y_train, y_test, top_features, output_dir
        )
        logger.info(f"Best model with top features: {best_top_model_name}")
        
        # Step 5: Compare performance with all features vs. top features
        logger.info("Step 5: Comparing performance with all features vs. top features...")
        comparison_results, top_model = compare_feature_sets(
            X_train, X_test, y_train, y_test, top_features, best_model, output_dir, random_state
        )
        
        # Create comprehensive comparison table
        logger.info("Creating comprehensive performance comparison table...")
        try:
            # Let's debug the comparison_results structure
            logger.info(f"Comparison results keys: {list(comparison_results.keys())}")
            
            comparison_table_path = create_comparison_table(
                results, comparison_results['top_features'], best_model_name, output_dir
            )
        except Exception as e:
            logger.error(f"Error creating comparison table: {str(e)}")
            # We'll continue without the comparison table
        
        # Step 6: Save results and summary
        logger.info("Step 6: Saving results and summary...")
        
        # Save models
        import joblib
        joblib.dump(best_model, output_dir / f"best_model_all_features.pkl")
        joblib.dump(top_model, output_dir / f"best_model_top_features.pkl")
        logger.info("Models saved to output directory")
        
        # Save comprehensive summary
        summary = {
            'runtime': format_duration(time.time() - start_time),
            'data': {
                'train_file': str(train_file),
                'test_file': str(test_file),
                'target_column': target,
                'feature_count': X_train.shape[1],
                'train_samples': X_train.shape[0],
                'test_samples': X_test.shape[0],
            },
            'configuration': {
                'hyperparameter_tuning': use_hyperparameter_tuning,
                'top_features_used': top_n_features,
                'random_state': random_state
            },
            'models': {
                model_name: {
                    'accuracy': results[model_name]['accuracy'],
                    'auc': results[model_name]['auc'],
                    'sensitivity': results[model_name]['sensitivity'],
                    'specificity': results[model_name]['specificity'],
                    'f1_score': results[model_name]['f1_score'],
                    'training_time': results[model_name]['training_time']
                } for model_name in results
            },
            'best_model': {
                'name': best_model_name,
                'selection_criterion': 'accuracy',
                'all_features': {
                    'accuracy': results[best_model_name]['accuracy'],
                    'auc': results[best_model_name]['auc'],
                },
                'top_features': {
                    'feature_count': len(top_features),
                    'accuracy': comparison_results['top_features']['accuracy'],
                    'auc': comparison_results['top_features']['auc'],
                    'features': top_features
                }
            },
            'output_files': {
                'top_features': str(top_features_output),
                'roc_curves_all': str(output_dir / 'roc_curves_all_features.png'),
                'roc_curves_top': str(output_dir / 'roc_curves_top_features.png'),
                'roc_comparison': str(output_dir / 'roc_comparison_top_features.png'),
                'feature_importance': str(output_dir / 'top_feature_importances.png'),
                'performance_metrics_all': str(output_dir / 'performance_metrics_all_features.png'),
                'performance_metrics_top': str(output_dir / 'performance_metrics_top_features.png'),
                'performance_comparison_table': str(output_dir / 'performance_comparison_table.png'),
                'confusion_matrix_all': str(output_dir / 'confusion_matrix_all_features.png'),
                'confusion_matrix_top': str(output_dir / 'confusion_matrix_top_features.png'),
                'best_model_all': str(output_dir / 'best_model_all_features.pkl'),
                'best_model_top': str(output_dir / 'best_model_top_features.pkl')
            }
        }
        
        # Save summary as JSON
        summary_output = output_dir / "ml_analysis_summary.json"
        save_json(summary, summary_output)
        logger.info(f"Summary saved to: {summary_output}")
        
        # Create datasets with only top features
        X_train_top = X_train[top_features]
        X_test_top = X_test[top_features]
        
        # Save top features datasets
        train_top_df = pd.concat([X_train_top, pd.Series(y_train, name=target, index=X_train_top.index)], axis=1)
        test_top_df = pd.concat([X_test_top, pd.Series(y_test, name=target, index=X_test_top.index)], axis=1)
        
        train_top_output = output_dir / "ml_ready_train_top_features.csv"
        test_top_output = output_dir / "ml_ready_test_top_features.csv"
        
        train_top_df.to_csv(train_top_output, index=False)
        test_top_df.to_csv(test_top_output, index=False)
        
        logger.info(f"ML-ready training data with top features saved to: {train_top_output}")
        logger.info(f"ML-ready test data with top features saved to: {test_top_output}")
        
        # Print completion message
        logger.info("=" * 60)
        logger.info(f"ANALYSIS COMPLETED IN {format_duration(time.time() - start_time)}")
        logger.info("=" * 60)
        logger.info(f"Best model: {best_model_name}")
        logger.info(f"All features ({X_train.shape[1]}) - AUC: {results[best_model_name]['auc']:.4f}, Accuracy: {results[best_model_name]['accuracy']:.4f}")
        logger.info(f"Top features ({len(top_features)}) - AUC: {comparison_results['top_features']['auc']:.4f}, Accuracy: {comparison_results['top_features']['accuracy']:.4f}")
        logger.info(f"Results saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
