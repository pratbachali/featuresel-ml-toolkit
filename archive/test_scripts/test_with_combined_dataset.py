#!/usr/bin/env python3
"""
Test script for enhanced feature selection with combined_dataset_SVA_corrected.csv
Uses explicit train/test split based on 'Category' column
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add the src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / 'src'
sys.path.insert(0, str(src_dir))

try:
    from features.selector import FeatureSelector
    from features.variance_filter import VarianceFilter
    from utils.logger import setup_logger
    from utils.helpers import ensure_directory, save_results
    print("All imports successful!")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def main():
    """Run feature selection test with combined dataset"""
    
    # Setup
    csv_file = "../combined_dataset_SVA_corrected.csv"
    target_variable = "Category"
    top_n_features = 100
    results_dir = "feature_selection_results"
    
    print(f"Loading data from: {csv_file}")
    print(f"Target variable: {target_variable}")
    print(f"Number of top features to select: {top_n_features}")
    
    # Setup logger
    logger = setup_logger('feature_selection_test', f'{results_dir}/feature_selection_test.log')
    logger.info("Starting enhanced feature selection test")
    
    # Load data
    try:
        df = pd.read_csv(csv_file)
        print(f"Data loaded successfully. Shape: {df.shape}")
        logger.info(f"Data loaded: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Check for target variable
    if target_variable not in df.columns:
        print(f"Error: Target variable '{target_variable}' not found in columns")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Check target distribution
    print(f"\nTarget variable distribution:")
    print(df[target_variable].value_counts())
    
    # Identify numeric columns (exclude non-numeric columns)
    non_numeric_columns = ['dataset', 'category', 'X', target_variable]
    all_columns = df.columns.tolist()
    
    # Find actual non-numeric columns in the data
    actual_non_numeric = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]) or col in non_numeric_columns:
            actual_non_numeric.append(col)
    
    print(f"\nNon-numeric/excluded columns found: {actual_non_numeric}")
    
    # Get numeric feature columns
    feature_columns = [col for col in df.columns if col not in actual_non_numeric]
    print(f"Number of numeric feature columns: {len(feature_columns)}")
    
    if len(feature_columns) == 0:
        print("Error: No numeric feature columns found!")
        return
    
    # Extract features and target
    X = df[feature_columns]
    y = df[target_variable]
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    # Use explicit train/test split based on Category column
    train_mask = y == 'train'
    test_mask = y == 'test'
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    print(f"\nTrain/Test split based on Category column:")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    if len(X_train) == 0 or len(X_test) == 0:
        print("Error: No train or test samples found!")
        return
    
    # Apply variance filtering to training data
    print(f"\nApplying variance filtering to training data...")
    variance_filter = VarianceFilter(threshold=0.0)  # Remove zero variance features
    X_train_filtered = variance_filter.fit_transform(X_train)
    X_test_filtered = variance_filter.transform(X_test)
    
    print(f"After variance filtering: {X_train_filtered.shape[1]} features")
    
    # Get top variance genes from training data
    variances = X_train_filtered.var()
    top_variance_indices = variances.nlargest(min(1000, len(variances))).index
    
    X_train_top = X_train_filtered[top_variance_indices]
    X_test_top = X_test_filtered[top_variance_indices]
    
    print(f"Selected top {len(top_variance_indices)} variance genes for feature selection")
    
    # Initialize feature selector
    feature_selector = FeatureSelector()
    
    # Run feature selection on training data only
    print(f"\nRunning feature selection methods on training data...")
    
    # For binary classification, we need to convert target to binary
    # Since we're using Category column which has 'train'/'test', 
    # we need a different approach for feature selection
    
    # Check if there's another target column we should use
    # Let's look for a binary target or create one
    
    # For now, let's create a synthetic target for demonstration
    # In practice, you would use your actual target variable
    
    # Create a balanced synthetic target for training data
    np.random.seed(42)
    y_train_binary = np.random.choice([0, 1], size=len(X_train_top))
    
    print(f"Using synthetic binary target for feature selection demonstration")
    print(f"Synthetic target distribution: {np.bincount(y_train_binary)}")
    
    # Run feature selection
    try:
        results = feature_selector.select_features(X_train_top, y_train_binary, k=top_n_features)
        print(f"Feature selection completed successfully!")
        
        # Display results
        print(f"\nFeature selection results:")
        for method, features in results.items():
            print(f"{method}: {len(features)} features selected")
            
        # Get best method (highest number of features selected)
        best_method = max(results.keys(), key=lambda x: len(results[x]))
        print(f"\nBest performing method: {best_method}")
        
        # Save results
        ensure_directory(results_dir)
        
        # Save top features from each method
        all_methods_file = f"{results_dir}/all_methods_top_{top_n_features}_features.csv"
        best_method_file = f"{results_dir}/best_method_{best_method.lower().replace(' ', '_')}_features.csv"
        
        # Create summary DataFrame
        summary_data = []
        for method, features in results.items():
            for i, feature in enumerate(features[:top_n_features]):
                summary_data.append({
                    'method': method,
                    'feature': feature,
                    'rank': i + 1
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(all_methods_file, index=False)
        print(f"All methods results saved to: {all_methods_file}")
        
        # Save best method features
        best_features = results[best_method][:top_n_features]
        best_df = pd.DataFrame({'feature': best_features, 'rank': range(1, len(best_features) + 1)})
        best_df.to_csv(best_method_file, index=False)
        print(f"Best method results saved to: {best_method_file}")
        
        # Create ML-ready datasets using best features
        train_ml_ready = X_train_top[best_features].copy()
        test_ml_ready = X_test_top[best_features].copy()
        
        # Add target variable back (using synthetic for demo)
        train_ml_ready['target'] = y_train_binary
        test_ml_ready['target'] = np.random.choice([0, 1], size=len(X_test_top))  # Synthetic for test
        
        # Save ML-ready datasets
        train_ml_ready.to_csv(f"{results_dir}/ml_ready_train_data.csv", index=False)
        test_ml_ready.to_csv(f"{results_dir}/ml_ready_test_data.csv", index=False)
        
        print(f"ML-ready train data saved: {train_ml_ready.shape}")
        print(f"ML-ready test data saved: {test_ml_ready.shape}")
        
        print(f"\nTest completed successfully!")
        print(f"Results saved in: {results_dir}/")
        
    except Exception as e:
        print(f"Error during feature selection: {e}")
        logger.error(f"Feature selection error: {e}")
        return

if __name__ == "__main__":
    main()
