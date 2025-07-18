#!/usr/bin/env python3
"""
Supervised Machine Learning Script with Interactive Prompts

This script performs supervised machine learning analysis on genomic data,
including proper train/test separation, visualizations, and comprehensive model evaluation.
It selects the best model based on accuracy and compares performance between using all features 
versus top features from this best model.
"""

import sys
import os
import time
import json
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)

# ML models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC  # Ensure this is imported at the top level
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
# Import XGBoost and LightGBM if available
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Skipping XGBoost classifier.")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Skipping LightGBM classifier.")

# Add the parent directory to sys.path to enable package imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

try:
    from src.data.loader import DataLoader
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
    
    # Simple implementations for classes
    class DataLoader:
        @staticmethod
        def load_csv(file_path):
            return pd.read_csv(file_path)
    
    print("Using simplified implementations...")


def get_user_input():
    """Get user input for all pipeline parameters"""
    print("\n" + "="*60)
    print("SUPERVISED MACHINE LEARNING PIPELINE")
    print("="*60)
    
    # Get data file path
    data_file = input("\nEnter the path to your CSV data file: ").strip()
    while not Path(data_file).exists():
        print(f"Error: File '{data_file}' not found.")
        data_file = input("Please enter a valid file path: ").strip()
    
    # Load data to show available columns
    try:
        df = pd.read_csv(data_file)
        print(f"\nDataset loaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"\nAvailable columns:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None, None
    
    # Get target variable
    target = input(f"\nEnter the target column name: ").strip()
    while target not in df.columns:
        print(f"Error: Column '{target}' not found in dataset.")
        target = input("Please enter a valid column name: ").strip()
    
    # Check if Category column exists for predefined train/test split
    has_category = 'Category' in df.columns
    use_predefined_split = False
    category_column = 'Category'
    test_size = 0.2
    
    if has_category:
        print(f"\n📋 Found 'Category' column with predefined train/test split:")
        print(f"   {df['Category'].value_counts()}")
        use_predefined = input("\nUse predefined train/test split from 'Category' column? (y/n, default y): ").strip().lower()
        if use_predefined in ['', 'y', 'yes']:
            use_predefined_split = True
            category_column = 'Category'
            print("✅ Will use predefined train/test split from 'Category' column")
        else:
            print("✅ Will use random train/test split")
    else:
        custom_split = input(f"\nNo 'Category' column found. Do you have another column for train/test split? (y/n, default n): ").strip().lower()
        if custom_split in ['y', 'yes']:
            category_column = input("Enter the column name for train/test split: ").strip()
            while category_column not in df.columns:
                print(f"Error: Column '{category_column}' not found in dataset.")
                category_column = input("Please enter a valid column name (or 'skip' to use random split): ").strip()
                if category_column.lower() == 'skip':
                    break
            
            if category_column.lower() != 'skip':
                use_predefined_split = True
                print(f"✅ Will use predefined train/test split from '{category_column}' column")
            else:
                use_predefined_split = False
                print("✅ Will use random train/test split")
    
    # Get test size for random split if not using predefined
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
    
    # Get output directory
    output_dir = input(f"\nEnter output directory (default 'ml_results'): ").strip()
    if not output_dir:
        output_dir = 'ml_results'
    
    return data_file, target, use_predefined_split, category_column, test_size, output_dir


def create_ml_models():
    """Create a dictionary of ML models to be evaluated"""
    models = {
        'logistic_regression': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'params': {'C': 1.0, 'solver': 'lbfgs'},
            'name': 'Logistic Regression',
            'color': 'blue',
            'feature_importance': True  # Has coef_ attribute for feature importance
        },
        'random_forest': {
            'model': RandomForestClassifier(n_estimators=100, random_state=42),
            'params': {'n_estimators': 100, 'max_depth': None},
            'name': 'Random Forest',
            'color': 'green',
            'feature_importance': True  # Has feature_importances_ attribute
        },
        'gradient_boosting': {
            'model': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'params': {'n_estimators': 100, 'max_depth': 3},
            'name': 'Gradient Boosting',
            'color': 'red',
            'feature_importance': True  # Has feature_importances_ attribute
        },
        'svm': {
            # Ensure we're using the imported SVC from sklearn.svm
            'model': SVC(probability=True, random_state=42),
            'params': {'C': 1.0, 'kernel': 'rbf'},
            'name': 'Support Vector Machine',
            'color': 'purple',
            'feature_importance': False  # No direct feature importance
        },
        'knn': {
            'model': KNeighborsClassifier(n_neighbors=5),
            'params': {'n_neighbors': 5, 'weights': 'uniform'},
            'name': 'K-Nearest Neighbors',
            'color': 'orange',
            'feature_importance': False  # No direct feature importance
        },
        'mlp': {
            'model': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42),
            'params': {'hidden_layer_sizes': (100,), 'activation': 'relu'},
            'name': 'Neural Network (MLP)',
            'color': 'brown',
            'feature_importance': False  # No direct feature importance
        }
    }
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models['xgboost'] = {
            'model': XGBClassifier(n_estimators=100, random_state=42),
            'params': {'n_estimators': 100, 'max_depth': 3},
            'name': 'XGBoost',
            'color': 'teal',
            'feature_importance': True  # Has feature_importances_ attribute
        }
    
    # Add LightGBM if available
    if LIGHTGBM_AVAILABLE:
        models['lightgbm'] = {
            'model': LGBMClassifier(n_estimators=100, random_state=42),
            'params': {'n_estimators': 100, 'max_depth': -1},
            'name': 'LightGBM',
            'color': 'cyan',
            'feature_importance': True  # Has feature_importances_ attribute
        }
    
    return models


def extract_feature_importance(model_info, model, X_train):
    """Extract feature importance from a trained model if available"""
    feature_importances = None
    
    if not model_info['feature_importance']:
        return None
    
    try:
        if hasattr(model, 'feature_importances_'):
            feature_importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            feature_importances = np.abs(model.coef_[0])  # For binary classification
        else:
            return None
        
        # Create a DataFrame with feature names and importance scores
        feature_importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': feature_importances
        })
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
        return feature_importance_df
    except:
        return None


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Train and evaluate a single model, return performance metrics"""
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Handle prediction probabilities carefully for models like SVC
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    except (AttributeError, IndexError) as e:
        logger = get_logger(__name__)
        logger.warning(f"Error getting predict_proba: {e}. Using decision_function if available.")
        if hasattr(model, 'decision_function'):
            decision_scores = model.decision_function(X_test)
            y_pred_proba = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min())
        else:
            # Fallback to binary predictions if no probability method available
            y_pred_proba = y_pred.astype(float)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)  # Sensitivity
    specificity = recall_score(y_test, y_pred, pos_label=0)
    f1 = f1_score(y_test, y_pred)
    
    # Get ROC curve data
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Store the trained model
    trained_model = model
    
    # Return metrics and ROC data
    metrics = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,  # same as sensitivity
        'sensitivity': recall,
        'specificity': specificity,
        'f1_score': f1,
        'confusion_matrix': cm,
        'trained_model': trained_model,
        'y_pred': y_pred,
        'y_test': y_test,
        'roc_data': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
    }
    
    return metrics


def run_machine_learning_analysis(X_train, X_test, y_train, y_test, output_dir):
    """Run the complete ML analysis pipeline, selecting the best model based on accuracy"""
    logger = get_logger(__name__)
    logger.info("Running machine learning analysis...")
    
    # Create models dictionary
    models = create_ml_models()
    
    # Dictionary to store results
    full_results = {
        'all_features': {},
        'top_features': {},
        'best_model': None,
        'top_features_list': []
    }
    
    # PART 1: Train and evaluate models using all features
    logger.info(f"Training models with all features ({X_train.shape[1]} features)...")
    
    for model_name, model_info in models.items():
        logger.info(f"Training {model_info['name']}...")
        
        # Get model instance
        model = model_info['model']
        
        # Evaluate model
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
        
        # Extract feature importance if available
        feature_importance_df = extract_feature_importance(model_info, model, X_train)
        
        # Store results
        full_results['all_features'][model_name] = {
            'metrics': metrics,
            'feature_importance': feature_importance_df.to_dict('records') if feature_importance_df is not None else None,
            'model_name': model_info['name']
        }
        
        logger.info(f"  - Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  - ROC AUC: {metrics['roc_auc']:.4f}")
    
    # Determine best model based on accuracy
    best_model_name = max(
        full_results['all_features'],
        key=lambda m: full_results['all_features'][m]['metrics']['accuracy']
    )
    
    best_model_info = models[best_model_name]
    best_model_metrics = full_results['all_features'][best_model_name]['metrics']
    
    full_results['best_model'] = {
        'name': best_model_name,
        'display_name': best_model_info['name'],
        'metrics': best_model_metrics
    }
    
    logger.info(f"Best model: {best_model_info['name']} (Accuracy: {best_model_metrics['accuracy']:.4f})")        # Extract top 20 features from best model if feature importance is available
    if full_results['all_features'][best_model_name]['feature_importance'] is not None:
        top_features = [
            item['feature'] for item in 
            full_results['all_features'][best_model_name]['feature_importance'][:20]
        ]
        full_results['top_features_list'] = top_features
        
        # Save top features to CSV
        top_features_df = pd.DataFrame({
            'feature': top_features,
            'importance': [
                item['importance'] for item in 
                full_results['all_features'][best_model_name]['feature_importance'][:20]
            ]
        })
        top_features_df.to_csv(output_dir / 'top_20_features.csv', index=False)
        logger.info(f"Saved top 20 features to: {output_dir / 'top_20_features.csv'}")
    else:
        logger.warning(f"No feature importance available for {best_model_info['name']}. Using a different model for top features.")
        
        # Find another model with feature importance
        for model_name, model_data in full_results['all_features'].items():
            if model_data['feature_importance'] is not None:
                top_features = [item['feature'] for item in model_data['feature_importance'][:20]]
                full_results['top_features_list'] = top_features
                
                # Save top features to CSV
                top_features_df = pd.DataFrame({
                    'feature': top_features,
                    'importance': [item['importance'] for item in model_data['feature_importance'][:20]]
                })
                top_features_df.to_csv(output_dir / 'top_20_features.csv', index=False)
                logger.info(f"Saved top 20 features to: {output_dir / 'top_20_features.csv'}")
                break
    
    # PART 2: Train and evaluate models using only top 20 features
    if full_results['top_features_list']:
        logger.info(f"Training models with top 20 features...")
        
        # Select only top features
        X_train_top = X_train[full_results['top_features_list']]
        X_test_top = X_test[full_results['top_features_list']]
        
        for model_name, model_info in models.items():
            logger.info(f"Training {model_info['name']} with top 20 features...")
            
            # Get model instance (fresh)
            model = model_info['model'].__class__(**model_info['params'])
            
            # Evaluate model
            metrics = evaluate_model(model, X_train_top, y_train, X_test_top, y_test)
            
            # Store results
            full_results['top_features'][model_name] = {
                'metrics': metrics,
                'model_name': model_info['name']
            }
            
            logger.info(f"  - Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  - ROC AUC: {metrics['roc_auc']:.4f}")
    else:
        logger.warning("Could not extract top features. Skipping top features analysis.")
    
    # Save full results to JSON
    save_json(full_results, output_dir / 'ml_analysis_results.json')
    logger.info(f"Saved full ML analysis results to: {output_dir / 'ml_analysis_results.json'}")
    
    return full_results


def create_roc_curve_visualization(results, output_dir):
    """Create ROC curve visualization for all models"""
    logger = get_logger(__name__)
    logger.info("Creating ROC curve visualizations...")
    
    # Create figure with two subplots: All Features vs Top 20 Features
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot ROC curves for all features
    ax1 = axes[0]
    for model_name, model_data in results['all_features'].items():
        model_info = create_ml_models()[model_name]
        color = model_info['color']
        
        fpr = model_data['metrics']['roc_data']['fpr']
        tpr = model_data['metrics']['roc_data']['tpr']
        roc_auc = model_data['metrics']['roc_auc']
        
        ax1.plot(fpr, tpr, color=color, lw=2,
                 label=f'{model_info["name"]} (AUC = {roc_auc:.3f})')
    
    ax1.plot([0, 1], [0, 1], 'k--', lw=2)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curves - All Features', fontsize=14, fontweight='bold')
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot ROC curves for top 20 features
    ax2 = axes[1]
    if results['top_features']:
        for model_name, model_data in results['top_features'].items():
            model_info = create_ml_models()[model_name]
            color = model_info['color']
            
            fpr = model_data['metrics']['roc_data']['fpr']
            tpr = model_data['metrics']['roc_data']['tpr']
            roc_auc = model_data['metrics']['roc_auc']
            
            ax2.plot(fpr, tpr, color=color, lw=2,
                     label=f'{model_info["name"]} (AUC = {roc_auc:.3f})')
        
        ax2.plot([0, 1], [0, 1], 'k--', lw=2)
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate', fontsize=12)
        ax2.set_ylabel('True Positive Rate', fontsize=12)
        ax2.set_title(f'ROC Curves - Top 20 Features', fontsize=14, fontweight='bold')
        ax2.legend(loc="lower right", fontsize=10)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No top features analysis available',
                ha='center', va='center', fontsize=14)
        ax2.set_title('ROC Curves - Top 20 Features (N/A)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved ROC curves comparison to: {output_dir / 'roc_curves_comparison.png'}")


def create_metrics_table_visualization(results, output_dir):
    """Create a table visualization of all metrics for both all features and top 20 features"""
    logger = get_logger(__name__)
    logger.info("Creating metrics table visualization...")
    
    # Create DataFrames for all features and top features
    all_metrics = []
    for model_name, model_data in results['all_features'].items():
        metrics = model_data['metrics']
        row = {
            'Model': model_data['model_name'],
            'Features': 'All',
            'Accuracy': metrics['accuracy'],
            'AUC': metrics['roc_auc'],
            'Sensitivity': metrics['sensitivity'],
            'Specificity': metrics['specificity'],
            'F1 Score': metrics['f1_score']
        }
        all_metrics.append(row)
    
    if results['top_features']:
        for model_name, model_data in results['top_features'].items():
            metrics = model_data['metrics']
            row = {
                'Model': model_data['model_name'],
                'Features': 'Top 20',
                'Accuracy': metrics['accuracy'],
                'AUC': metrics['roc_auc'],
                'Sensitivity': metrics['sensitivity'],
                'Specificity': metrics['specificity'],
                'F1 Score': metrics['f1_score']
            }
            all_metrics.append(row)
    
    metrics_df = pd.DataFrame(all_metrics)
    
    # Save metrics as CSV
    metrics_df.to_csv(output_dir / 'model_performance_metrics.csv', index=False)
    logger.info(f"Saved model performance metrics to: {output_dir / 'model_performance_metrics.csv'}")
    
    # Create a styled table visualization
    plt.figure(figsize=(15, len(metrics_df) * 0.5 + 2))
    ax = plt.subplot(111)
    ax.axis('off')
    
    # Create table
    cell_text = []
    for _, row in metrics_df.iterrows():
        cell_text.append([
            row['Model'],
            row['Features'],
            f"{row['Accuracy']:.4f}",
            f"{row['AUC']:.4f}",
            f"{row['Sensitivity']:.4f}",
            f"{row['Specificity']:.4f}",
            f"{row['F1 Score']:.4f}"
        ])
    
    columns = ['Model', 'Features', 'Accuracy', 'AUC', 'Sensitivity', 'Specificity', 'F1 Score']
    
    # Add a table at the bottom of the axes
    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        loc='center',
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Highlight the best model for each metric type
    best_acc_all = metrics_df[metrics_df['Features'] == 'All']['Accuracy'].max()
    best_auc_all = metrics_df[metrics_df['Features'] == 'All']['AUC'].max()
    
    if 'Top 20' in metrics_df['Features'].values:
        best_acc_top = metrics_df[metrics_df['Features'] == 'Top 20']['Accuracy'].max()
        best_auc_top = metrics_df[metrics_df['Features'] == 'Top 20']['AUC'].max()
    
    # Add title
    plt.title('Model Performance Metrics Comparison', fontsize=16, pad=20)
    plt.savefig(output_dir / 'model_metrics_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved model metrics table to: {output_dir / 'model_metrics_table.png'}")


def create_confusion_matrix_visualization(results, output_dir):
    """Create a visualization of the confusion matrix for the best model (selected based on accuracy)"""
    logger = get_logger(__name__)
    logger.info("Creating confusion matrix visualization for the best classifier (selected based on accuracy)...")
    
    # Get the best model from all features
    best_model_name = results['best_model']['name']
    
    # Get metrics for both all features and top 20 features
    all_features_metrics = results['all_features'][best_model_name]['metrics']
    
    # Create a figure with two subplots (all features and top 20 features if available)
    if results['top_features'] and best_model_name in results['top_features']:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        top_features_metrics = results['top_features'][best_model_name]['metrics']
        
        # Confusion matrix for all features
        cm_all = all_features_metrics['confusion_matrix']
        sns.heatmap(cm_all, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Predicted 0', 'Predicted 1'],
                    yticklabels=['Actual 0', 'Actual 1'], ax=axes[0])
        
        # Calculate and display correct/incorrect predictions
        total_all = cm_all.sum()
        correct_all = cm_all[0, 0] + cm_all[1, 1]
        incorrect_all = total_all - correct_all
        
        axes[0].set_title(f"{results['best_model']['display_name']} Confusion Matrix - All Features\n"
                          f"(Best Model by Accuracy: {all_features_metrics['accuracy']:.4f})\n"
                          f"Correct: {correct_all} | Incorrect: {incorrect_all}", 
                          fontsize=12)
        
        # Confusion matrix for top features
        cm_top = top_features_metrics['confusion_matrix']
        sns.heatmap(cm_top, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Predicted 0', 'Predicted 1'],
                    yticklabels=['Actual 0', 'Actual 1'], ax=axes[1])
        
        # Calculate and display correct/incorrect predictions
        total_top = cm_top.sum()
        correct_top = cm_top[0, 0] + cm_top[1, 1]
        incorrect_top = total_top - correct_top;
        
        axes[1].set_title(f"{results['best_model']['display_name']} Confusion Matrix - Top 20 Features\n"
                          f"(Best Model by Accuracy: {top_features_metrics['accuracy']:.4f})\n"
                          f"Correct: {correct_top} | Incorrect: {incorrect_top}", 
                          fontsize=12)
    else:
        # Only one confusion matrix for all features
        plt.figure(figsize=(8, 7))
        cm_all = all_features_metrics['confusion_matrix']
        sns.heatmap(cm_all, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Predicted 0', 'Predicted 1'],
                    yticklabels=['Actual 0', 'Actual 1'])
        
        # Calculate and display correct/incorrect predictions
        total_all = cm_all.sum()
        correct_all = cm_all[0, 0] + cm_all[1, 1]
        incorrect_all = total_all - correct_all
        
        plt.title(f"{results['best_model']['display_name']} Confusion Matrix - All Features\n"
                  f"(Best Model by Accuracy: {all_features_metrics['accuracy']:.4f})\n"
                  f"Correct: {correct_all} | Incorrect: {incorrect_all}", 
                  fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved confusion matrix visualization to: {output_dir / 'confusion_matrix_visualization.png'}")


def create_feature_importance_visualization(results, output_dir):
    """Create a visualization of the top 20 features from the best model (selected based on accuracy)"""
    logger = get_logger(__name__)
    
    # Check if top features are available
    if not results['top_features_list']:
        logger.warning("No top features available. Skipping feature importance visualization.")
        return
    
    # Get feature importance data from the best model
    best_model_name = results['all_features'][results['best_model']['name']]['model_name']
    
    # If the best model doesn't have feature importance, find one that does
    if results['all_features'][best_model_name]['feature_importance'] is None:
        for model_name, model_data in results['all_features'].items():
            if model_data['feature_importance'] is not None:
                best_model_name = model_name
                break
    
    if results['all_features'][best_model_name]['feature_importance'] is None:
        logger.warning("No feature importance data available. Skipping feature importance visualization.")
        return
    
    # Get feature importance data
    feature_importance = results['all_features'][best_model_name]['feature_importance'][:20]
    features = [item['feature'] for item in feature_importance]
    importance = [item['importance'] for item in feature_importance]
    
    # Create bar plot
    plt.figure(figsize=(14, 10))  # Adjusted height for 20 features
    plt.barh(range(len(features)), importance, align='center', color='skyblue')
    plt.yticks(range(len(features)), features, fontsize=10)  # Slightly larger font for fewer features
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title(f'Top 20 Features from {results["all_features"][best_model_name]["model_name"]} (Best Model by Accuracy)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'top_20_features_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved top features importance visualization to: {output_dir / 'top_20_features_importance.png'}")


def create_performance_comparison_visualization(results, output_dir):
    """Create visualization comparing model performance with all features vs top 20 features"""
    logger = get_logger(__name__)
    logger.info("Creating performance comparison visualization...")
    
    if not results['top_features']:
        logger.warning("No top features analysis available. Skipping performance comparison visualization.")
        return
    
    # Prepare data for comparison
    model_names = []
    acc_all = []
    acc_top = []
    auc_all = []
    auc_top = []
    
    for model_name in results['all_features'].keys():
        if model_name in results['top_features']:
            model_names.append(create_ml_models()[model_name]['name'])
            acc_all.append(results['all_features'][model_name]['metrics']['accuracy'])
            acc_top.append(results['top_features'][model_name]['metrics']['accuracy'])
            auc_all.append(results['all_features'][model_name]['metrics']['roc_auc'])
            auc_top.append(results['top_features'][model_name]['metrics']['roc_auc'])
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Set position of bars on X axis
    x = np.arange(len(model_names))
    width = 0.35
    
    # Create accuracy comparison subplot
    bars1 = axes[0].bar(x - width/2, acc_all, width, label='All Features', color='steelblue')
    bars2 = axes[0].bar(x + width/2, acc_top, width, label='Top 20 Features', color='lightcoral')
    
    # Add labels, title and legend
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Accuracy Comparison: All vs Top 20 Features', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar1, bar2 in zip(bars1, bars2):
        axes[0].text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.01,
                    f'{bar1.get_height():.3f}', ha='center', va='bottom', fontsize=9)
        axes[0].text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.01,
                    f'{bar2.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    
    # Create AUC comparison subplot
    bars3 = axes[1].bar(x - width/2, auc_all, width, label='All Features', color='steelblue')
    bars4 = axes[1].bar(x + width/2, auc_top, width, label='Top 20 Features', color='lightcoral')
    
    # Add labels, title and legend
    axes[1].set_ylabel('AUC', fontsize=12)
    axes[1].set_title('AUC Comparison: All vs Top 20 Features', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar3, bar4 in zip(bars3, bars4):
        axes[1].text(bar3.get_x() + bar3.get_width()/2, bar3.get_height() + 0.01,
                    f'{bar3.get_height():.3f}', ha='center', va='bottom', fontsize=9)
        axes[1].text(bar4.get_x() + bar4.get_width()/2, bar4.get_height() + 0.01,
                    f'{bar4.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved performance comparison visualization to: {output_dir / 'performance_comparison.png'}")


def create_detailed_performance_table(results, output_dir):
    """Create a detailed performance table comparing all metrics for all vs top 20 features"""
    logger = get_logger(__name__)
    logger.info("Creating detailed performance metrics table...")
    
    # Create a table for each classifier with all metrics
    all_models = list(results['all_features'].keys())
    
    # Only include models that have both all features and top features metrics
    if results['top_features']:
        models_to_include = [model for model in all_models if model in results['top_features']]
    else:
        models_to_include = all_models
        
    # Highlight the best model
    best_model_name = results['best_model']['name']
    
    # Create a comparison table that shows both feature sets side by side for each metric
    metric_names = ['Accuracy', 'AUC', 'Sensitivity', 'Specificity', 'F1 Score']
    metric_keys = ['accuracy', 'roc_auc', 'sensitivity', 'specificity', 'f1_score']
    
    # Create the table using Matplotlib
    fig, ax = plt.subplots(figsize=(12, len(models_to_include) * 1.2 + 2))
    ax.axis('off')
    
    # Prepare table data - for each model, we'll show "All | Top 20" for each metric
    table_data = []
    header = ['Model'] + metric_names
    
    for model_name in models_to_include:
        model_display_name = results['all_features'][model_name]['model_name']
        row = [model_display_name]
        
        for metric_key in metric_keys:
            all_value = results['all_features'][model_name]['metrics'][metric_key]
            
            if results['top_features'] and model_name in results['top_features']:
                top_value = results['top_features'][model_name]['metrics'][metric_key]
                # Format as "All | Top 20"
                cell_text = f"{all_value:.4f} | {top_value:.4f}"
            else:
                cell_text = f"{all_value:.4f} | N/A"
                
            row.append(cell_text)
        
        table_data.append(row)
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=header,
        loc='center',
        cellLoc='center'
    )
    
    # Formatting
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    # Highlight the best model row
    best_model_index = models_to_include.index(best_model_name) if best_model_name in models_to_include else -1
    if best_model_index >= 0:
        for j in range(len(header)):
            cell = table[best_model_index + 1, j]  # +1 for header offset
            cell.set_facecolor('#e6f5ff')  # Light blue highlight
    
    plt.title('Performance Metrics Comparison: All Features | Top 20 Features', fontsize=16, pad=20)
    plt.savefig(output_dir / 'detailed_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save as CSV
    csv_data = []
    for i, model_name in enumerate(models_to_include):
        model_display_name = results['all_features'][model_name]['model_name']
        # All features row
        all_row = {
            'Model': model_display_name,
            'Feature Set': 'All'
        }
        # Top features row
        top_row = {
            'Model': model_display_name,
            'Feature Set': 'Top 20'
        }
        
        for j, metric_key in enumerate(metric_keys):
            all_row[metric_names[j]] = results['all_features'][model_name]['metrics'][metric_key]
            if results['top_features'] and model_name in results['top_features']:
                top_row[metric_names[j]] = results['top_features'][model_name]['metrics'][metric_key]
            else:
                top_row[metric_names[j]] = float('nan')  # N/A as NaN for CSV
        
        csv_data.append(all_row)
        csv_data.append(top_row)
    
    # Convert to DataFrame and save as CSV
    metrics_df = pd.DataFrame(csv_data)
    metrics_df.to_csv(output_dir / 'detailed_performance_metrics.csv', index=False)
    
    logger.info(f"Saved detailed performance metrics to: {output_dir / 'detailed_metrics_comparison.png'}")
    logger.info(f"Saved detailed performance metrics to CSV: {output_dir / 'detailed_performance_metrics.csv'}")


def main():
    parser = argparse.ArgumentParser(description='Supervised Machine Learning Analysis')
    parser.add_argument('--interactive', action='store_true', 
                       help='Run in interactive mode with user prompts')
    parser.add_argument('--data_file', help='Path to data file (CSV)')
    parser.add_argument('--target', help='Target column name')
    parser.add_argument('--use_predefined_split', action='store_true',
                       help='Use predefined train/test split from category column')
    parser.add_argument('--category_column', default='Category',
                       help='Column name containing train/test split labels (default: Category)')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size for random split (0.1-0.5, default: 0.2)')
    parser.add_argument('--output_dir', default='ml_results',
                       help='Output directory (default: ml_results)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Get input parameters
    if args.interactive or not args.data_file or not args.target:
        data_file, target, use_predefined_split, category_column, test_size, output_dir = get_user_input()
        if not data_file:
            return
    else:
        data_file = args.data_file
        target = args.target
        use_predefined_split = args.use_predefined_split
        category_column = args.category_column
        test_size = args.test_size
        output_dir = args.output_dir
    
    # Setup logging
    output_dir = Path(output_dir)
    ensure_directory(output_dir)
    setup_logging(level='INFO', log_file=str(output_dir / "ml_analysis.log"))
    logger = get_logger(__name__)
    
    logger.info("=" * 60)
    logger.info("SUPERVISED MACHINE LEARNING ANALYSIS")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    try:
        # Load and examine data
        logger.info("Loading data...")
        df = pd.read_csv(data_file)
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Target variable: {target}")
        logger.info(f"Target distribution:\n{df[target].value_counts()}")
        
        # Prepare features and target
        # Exclude common non-feature columns
        exclude_columns = {target, 'Category', 'Dataset', 'Patient_ID', 'X', 'Unnamed: 0'}
        if category_column:
            exclude_columns.add(category_column)
        
        # Remove target and other non-feature columns
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        X = df[feature_columns]
        y = df[target]
        
        logger.info(f"Features: {len(X.columns)}")
        
        # Identify and exclude non-numeric columns
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns
        
        if len(non_numeric_columns) > 0:
            logger.info(f"Found {len(non_numeric_columns)} non-numeric columns that will be excluded:")
            for col in non_numeric_columns:
                logger.info(f"  - {col} (dtype: {X[col].dtype})")
            
            # Use only numeric columns for ML modeling
            X = X[numeric_columns]
            logger.info(f"After excluding non-numeric columns: {len(X.columns)} features")
        else:
            logger.info("All feature columns are numeric")
        
        # Split into train/test sets
        if use_predefined_split:
            logger.info(f"Using predefined train/test split from '{category_column}' column...")
            
            if category_column not in df.columns:
                logger.error(f"Category column '{category_column}' not found in dataset.")
                raise ValueError(f"Category column '{category_column}' not found in dataset")
            
            logger.info(f"Category distribution: {df[category_column].value_counts().to_dict()}")
            
            # Split based on Category column - look for train/test labels
            train_mask = df[category_column].str.lower().isin(['train', 'training'])
            test_mask = df[category_column].str.lower().isin(['test', 'testing'])
            
            # If no explicit train/test labels, use first category as train, rest as test
            if not train_mask.any() or not test_mask.any():
                logger.info(f"No explicit 'train'/'test' labels found. Using first category as train, others as test.")
                unique_categories = df[category_column].unique()
                train_category = unique_categories[0]
                train_mask = df[category_column] == train_category
                test_mask = ~train_mask
                logger.info(f"Train category: {train_category}")
                logger.info(f"Test categories: {unique_categories[1:]}")
            
            # Convert masks to numpy arrays for indexing
            train_indices = train_mask.values
            test_indices = test_mask.values
            
            X_train = X[train_indices]
            X_test = X[test_indices]
            
            # Handle y indexing properly whether it's pandas Series or numpy array
            if isinstance(y, pd.Series):
                y_train = y[train_mask].values
                y_test = y[test_mask].values
            else:
                y_train = y[train_indices]
                y_test = y[test_indices]
            
        else:
            logger.info(f"Using random train/test split with test_size={test_size}...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, 
                random_state=args.random_state, stratify=y
            )
        
        logger.info(f"Training set: {X_train.shape}")
        logger.info(f"Test set: {X_test.shape}")
        logger.info(f"Training target distribution:\n{pd.Series(y_train).value_counts()}")
        logger.info(f"Test target distribution:\n{pd.Series(y_test).value_counts()}")
        
        # Standardize features
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
        
        # Run ML analysis
        results = run_machine_learning_analysis(X_train, X_test, y_train, y_test, output_dir)
        
        # Create visualizations
        create_roc_curve_visualization(results, output_dir)
        create_metrics_table_visualization(results, output_dir)
        create_confusion_matrix_visualization(results, output_dir)
        create_feature_importance_visualization(results, output_dir)
        create_performance_comparison_visualization(results, output_dir)
        # Create a dedicated detailed performance table
        create_detailed_performance_table(results, output_dir)
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("MACHINE LEARNING ANALYSIS SUMMARY")
        logger.info("=" * 60)
        
        logger.info(f"Best model (based on accuracy): {results['best_model']['display_name']}")
        logger.info(f"Best model accuracy: {results['best_model']['metrics']['accuracy']:.4f}")
        logger.info(f"Best model AUC: {results['best_model']['metrics']['roc_auc']:.4f}")
        
        if results['top_features_list']:
            logger.info(f"\nTop 20 features extracted from the best model:")
            for i, feature in enumerate(results['top_features_list'][:20], 1):
                logger.info(f"  {i:2d}. {feature}")
        
        if results['top_features']:
            # Compare best model with all features vs top 20 features
            best_model = results['best_model']['name']
            all_acc = results['all_features'][best_model]['metrics']['accuracy']
            top_acc = results['top_features'][best_model]['metrics']['accuracy']
            all_auc = results['all_features'][best_model]['metrics']['roc_auc']
            top_auc = results['top_features'][best_model]['metrics']['roc_auc']
            
            logger.info(f"\nBest model performance comparison:")
            logger.info(f"  All features:   Accuracy = {all_acc:.4f}, AUC = {all_auc:.4f}")
            logger.info(f"  Top 20 features: Accuracy = {top_acc:.4f}, AUC = {top_auc:.4f}")
            
            if top_acc >= all_acc:
                logger.info(f"  ✅ Top 20 features achieved better or equal accuracy!")
            else:
                logger.info(f"  ℹ️ Using all features achieved better accuracy by {all_acc - top_acc:.4f}")
            
            if top_auc >= all_auc:
                logger.info(f"  ✅ Top 20 features achieved better or equal AUC!")
            else:
                logger.info(f"  ℹ️ Using all features achieved better AUC by {all_auc - top_auc:.4f}")
        
        # Print execution time
        end_time = time.time()
        logger.info(f"\nExecution time: {format_duration(end_time - start_time)}")
        logger.info("\nAll output files saved to: " + str(output_dir))
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
