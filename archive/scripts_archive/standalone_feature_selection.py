#!/usr/bin/env python3
"""
Enhanced Feature Selection for Genomic Data Analysis
Self-contained script to avoid import issues
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import json
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def get_user_input():
    """Get user input for data file and target variable"""
    print("\n" + "="*60)
    print("ENHANCED FEATURE SELECTION PIPELINE")
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
        return None, None, None, None
    
    # Get target variable
    target = input(f"\nEnter the target column name: ").strip()
    while target not in df.columns:
        print(f"Error: Column '{target}' not found in dataset.")
        target = input("Please enter a valid column name: ").strip()
    
    # Get number of top features to output
    while True:
        try:
            n_features = int(input(f"\nEnter number of top features to select (default 100): ") or "100")
            if n_features > 0:
                break
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Check if Category column exists for predefined train/test split
    has_category = 'Category' in df.columns
    if has_category:
        print(f"\n📋 Found 'Category' column with predefined train/test split:")
        print(f"   {df['Category'].value_counts()}")
        use_predefined = input("\nUse predefined train/test split from 'Category' column? (y/n, default y): ").strip().lower()
        if use_predefined in ['', 'y', 'yes']:
            return data_file, target, n_features, None, True
    
    # Get test size for random split
    while True:
        try:
            test_size = float(input(f"\nEnter test size (0.1-0.5, default 0.2): ") or "0.2")
            if 0.1 <= test_size <= 0.5:
                break
            else:
                print("Please enter a value between 0.1 and 0.5.")
        except ValueError:
            print("Please enter a valid number.")
    
    return data_file, target, n_features, test_size, False

def filter_top_variance_genes(X, n_genes=1000):
    """Filter top variance genes"""
    variances = X.var()
    top_features = variances.nlargest(n_genes).index.tolist()
    return top_features

def perform_feature_selection(X_train, y_train, n_features=100):
    """Perform feature selection using multiple methods"""
    print(f"\n🔍 Step 3: Performing feature selection ({n_features} features)...")
    
    methods = {}
    scores = {}
    
    print("  - Univariate feature selection (F-test)...")
    selector_f = SelectKBest(score_func=f_classif, k=n_features)
    selector_f.fit(X_train, y_train)
    methods['univariate_f'] = X_train.columns[selector_f.get_support(indices=True)].tolist()
    
    print("  - Mutual information feature selection...")
    selector_mi = SelectKBest(score_func=mutual_info_classif, k=n_features)
    selector_mi.fit(X_train, y_train)
    methods['mutual_info'] = X_train.columns[selector_mi.get_support(indices=True)].tolist()
    
    print("  - Random Forest feature importance...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    feature_importance = pd.Series(rf.feature_importances_, index=X_train.columns)
    methods['random_forest'] = feature_importance.nlargest(n_features).index.tolist()
    
    print("  - RFE with Logistic Regression...")
    lr = LogisticRegression(random_state=42, max_iter=1000)
    rfe = RFE(lr, n_features_to_select=n_features)
    rfe.fit(X_train, y_train)
    methods['rfe_logistic'] = X_train.columns[rfe.get_support(indices=True)].tolist()
    
    # Evaluate each method with cross-validation
    print("  - Evaluating methods with cross-validation...")
    for method_name, features in methods.items():
        X_selected = X_train[features]
        rf_eval = RandomForestClassifier(n_estimators=50, random_state=42)
        cv_scores = cross_val_score(rf_eval, X_selected, y_train, cv=5, scoring='accuracy')
        scores[method_name] = cv_scores.mean()
        print(f"    {method_name}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return methods, scores

def create_visualizations(methods, scores, X_train, y_train, output_dir, feature_columns=None):
    """Create comprehensive visualizations"""
    print("\n📊 Creating visualizations...")
    
    def select_features_safely(X, features, feature_cols=None):
        """Safely select features from X whether it's pandas DataFrame or numpy array"""
        if hasattr(X, 'iloc'):  # pandas DataFrame
            return X[features]
        else:  # numpy array
            if feature_cols is not None and isinstance(features[0], str):
                # Convert feature names to indices
                feature_indices = [i for i, col in enumerate(feature_cols) if col in features]
                return X[:, feature_indices]
            else:
                # Assume features are already indices or use all features
                return X
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Figure 1: Method Performance Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Feature Selection Methods Performance Analysis', fontsize=16, fontweight='bold')
    
    # Method scores bar plot
    method_names = list(scores.keys())
    method_scores = list(scores.values())
    
    axes[0, 0].bar(method_names, method_scores, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    axes[0, 0].set_title('Cross-Validation Accuracy by Method')
    axes[0, 0].set_ylabel('CV Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)
     # Feature overlap heatmap with gene counts
    all_features = set()
    for features in methods.values():
        all_features.update(features)

    overlap_matrix = np.zeros((len(methods), len(methods)))
    overlap_counts = np.zeros((len(methods), len(methods)), dtype=int)
    method_list = list(methods.keys())

    for i, method1 in enumerate(method_list):
        for j, method2 in enumerate(method_list):
            features1 = set(methods[method1])
            features2 = set(methods[method2])
            intersection_count = len(features1.intersection(features2))
            union_count = len(features1.union(features2))
            overlap = intersection_count / union_count if union_count > 0 else 0
            overlap_matrix[i, j] = overlap
            overlap_counts[i, j] = intersection_count

    im = axes[0, 1].imshow(overlap_matrix, cmap='Blues', aspect='auto')
    axes[0, 1].set_title('Feature Selection Methods Overlap\n(Jaccard Index with Gene Counts)')
    axes[0, 1].set_xticks(range(len(method_list)))
    axes[0, 1].set_yticks(range(len(method_list)))
    axes[0, 1].set_xticklabels(method_list, rotation=45)
    axes[0, 1].set_yticklabels(method_list)
    
    # Add text annotations with overlap counts and percentages
    for i in range(len(method_list)):
        for j in range(len(method_list)):
            if i == j:
                # Diagonal: show total features selected
                text = f'{len(methods[method_list[i]])}'
                axes[0, 1].text(j, i, text, ha="center", va="center", 
                               color="white" if overlap_matrix[i, j] > 0.5 else "black",
                               fontsize=9, fontweight='bold')
            else:
                # Off-diagonal: show intersection count and Jaccard index
                text = f'{overlap_counts[i, j]}\n({overlap_matrix[i, j]:.2f})'
                axes[0, 1].text(j, i, text, ha="center", va="center", 
                               color="white" if overlap_matrix[i, j] > 0.5 else "black",
                               fontsize=8)
    
    plt.colorbar(im, ax=axes[0, 1], label='Jaccard Index')
    
    # Number of selected features by method
    feature_counts = [len(features) for features in methods.values()]
    axes[1, 0].bar(method_names, feature_counts, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    axes[1, 0].set_title('Number of Features Selected by Method')
    axes[1, 0].set_ylabel('Number of Features')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Best method features (top 20)
    best_method = max(scores.keys(), key=lambda k: scores[k])
    best_features = methods[best_method][:20]
    
    # Get feature importance for visualization
    rf_viz = RandomForestClassifier(n_estimators=100, random_state=42)
    X_train_selected = select_features_safely(X_train, best_features, feature_columns)
    rf_viz.fit(X_train_selected, y_train)
    feature_importance = pd.Series(rf_viz.feature_importances_, index=best_features)
    feature_importance.sort_values(ascending=True).plot(kind='barh', ax=axes[1, 1])
    axes[1, 1].set_title(f'Top 20 Features from Best Method ({best_method})')
    axes[1, 1].set_xlabel('Feature Importance')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_selection_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create and save detailed overlap summary table
    overlap_summary = []
    for i, method1 in enumerate(method_list):
        for j, method2 in enumerate(method_list):
            if i < j:  # Only upper triangle to avoid duplicates
                features1 = set(methods[method1])
                features2 = set(methods[method2])
                intersection = features1.intersection(features2)
                union = features1.union(features2)
                jaccard = len(intersection) / len(union) if len(union) > 0 else 0
                
                overlap_summary.append({
                    'Method 1': method1.replace('_', ' ').title(),
                    'Method 2': method2.replace('_', ' ').title(),
                    'Method 1 Features': len(features1),
                    'Method 2 Features': len(features2),
                    'Overlapping Genes': len(intersection),
                    'Union Total': len(union),
                    'Jaccard Index': round(jaccard, 3),
                    'Overlap Percentage': round(jaccard * 100, 1)
                })
    
    overlap_df = pd.DataFrame(overlap_summary)
    overlap_summary_file = output_dir / "feature_overlap_summary.csv"
    overlap_df.to_csv(overlap_summary_file, index=False)
    
    print(f"  📊 Feature overlap summary table saved to: {overlap_summary_file.name}")
    
    # Figure 2: Best Method Analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Best Method Analysis: {best_method.replace("_", " ").title()}', fontsize=16, fontweight='bold')
    
    # Feature importance
    top_features = feature_importance.nlargest(15)
    top_features.plot(kind='bar', ax=axes[0, 0], color='steelblue')
    axes[0, 0].set_title('Top 15 Feature Importances')
    axes[0, 0].set_ylabel('Importance')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Cross-validation scores
    rf_cv = RandomForestClassifier(n_estimators=50, random_state=42)
    cv_scores = cross_val_score(rf_cv, select_features_safely(X_train, best_features, feature_columns), y_train, cv=10, scoring='accuracy')
    axes[0, 1].hist(cv_scores, bins=8, alpha=0.7, color='lightblue', edgecolor='black')
    axes[0, 1].axvline(cv_scores.mean(), color='red', linestyle='--', label=f'Mean: {cv_scores.mean():.4f}')
    axes[0, 1].set_title('Cross-Validation Score Distribution')
    axes[0, 1].set_xlabel('Accuracy')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # Method comparison scores
    method_performance = pd.Series(scores)
    method_performance.plot(kind='bar', ax=axes[1, 0], color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    axes[1, 0].set_title('Method Performance Comparison')
    axes[1, 0].set_ylabel('CV Accuracy')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Learning curve for best method
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores_list = []
    val_scores_list = []
    
    for train_size in train_sizes:
        n_samples = int(train_size * len(X_train))
        if n_samples < 10:
            continue
        
        # For learning curve, use all features in X_train (we don't need to subset by best_features here)
        X_subset = X_train[:n_samples]
        y_subset = y_train[:n_samples]
        
        rf_temp = RandomForestClassifier(n_estimators=50, random_state=42)
        cv_scores_temp = cross_val_score(rf_temp, X_subset, y_subset, cv=3, scoring='accuracy')
        
        train_scores_list.append(cv_scores_temp.mean())
        val_scores_list.append(cv_scores_temp.mean())
    
    axes[1, 1].plot(train_sizes[:len(train_scores_list)], train_scores_list, 'o-', label='Training Score', color='blue')
    axes[1, 1].plot(train_sizes[:len(val_scores_list)], val_scores_list, 'o-', label='Validation Score', color='red')
    axes[1, 1].set_title('Learning Curve')
    axes[1, 1].set_xlabel('Training Set Size')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'best_method_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Visualizations saved to {output_dir}")

def main():
    """Main function"""
    print("Enhanced Feature Selection Analysis")
    print("=" * 50)
    
    # Get user input
    data_file, target, n_features, test_size, use_predefined_split = get_user_input()
    if not data_file:
        return
    
    # Setup output directory
    output_dir = Path("feature_selection_results")
    output_dir.mkdir(exist_ok=True)
    
    start_time = time.time()
    
    try:
        # Load and examine data
        print(f"\n📁 Step 1: Loading data from {data_file}...")
        df = pd.read_csv(data_file)
        print(f"  Dataset shape: {df.shape}")
        print(f"  Target variable: {target}")
        
        # Identify numeric columns (gene features)
        # Exclude common non-gene columns
        exclude_columns = {target, 'Category', 'Dataset', 'Patient_ID', 'X', 'Unnamed: 0'}
        if hasattr(df, 'index'):
            exclude_columns.add('index')
        
        # Get numeric columns only
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        print(f"  Total columns: {len(df.columns)}")
        print(f"  Numeric feature columns: {len(feature_columns)}")
        print(f"  Excluded columns: {[col for col in df.columns if col not in feature_columns]}")
        
        # Prepare features and target
        X = df[feature_columns]
        y = df[target]
        
        print(f"  Target distribution:\n{y.value_counts()}")
        
        # Encode target if necessary
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            print(f"  Target encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        print(f"  Original features: {len(X.columns)}")
        
        # Step 1: Apply variance filtering FIRST (on entire dataset)
        print(f"\n🔬 Step 2: Filtering top 1000 variance genes...")
        top_variance_features = filter_top_variance_genes(X, 1000)
        X_filtered = X[top_variance_features]
        print(f"  After variance filtering: {len(X_filtered.columns)} features")
        
        # Step 2: Handle train/test split
        if use_predefined_split and 'Category' in df.columns:
            print(f"\n✂️  Step 3: Using predefined train/test split from 'Category' column...")
            
            # Split based on Category column
            train_mask = df['Category'].str.lower() == 'train'
            test_mask = df['Category'].str.lower() == 'test'
            
            # Convert masks to numpy arrays for indexing
            train_indices = train_mask.values
            test_indices = test_mask.values
            
            X_train = X_filtered[train_indices]
            X_test = X_filtered[test_indices]
            
            # Handle y indexing properly whether it's pandas Series or numpy array
            if isinstance(y, pd.Series):
                y_train = y[train_mask].values
                y_test = y[test_mask].values
            else:
                y_train = y[train_indices]
                y_test = y[test_indices]
            
            print(f"  Training set: {X_train.shape}")
            print(f"  Test set: {X_test.shape}")
            print(f"  Training target distribution:\n{pd.Series(y_train).value_counts()}")
            print(f"  Test target distribution:\n{pd.Series(y_test).value_counts()}")
            
        else:
            print(f"\n✂️  Step 3: Splitting into train/test sets (test_size={test_size})...")
            X_train, X_test, y_train, y_test = train_test_split(
                X_filtered, y, test_size=test_size, 
                random_state=42, stratify=y
            )
            
            print(f"  Training set: {X_train.shape}")
            print(f"  Test set: {X_test.shape}")
        
        # Step 3: Feature selection on training data only
        methods, scores = perform_feature_selection(X_train, y_train, n_features)
        
        # Find best method
        best_method = max(scores.keys(), key=lambda k: scores[k])
        best_features = methods[best_method]
        
        print(f"\n🏆 Best performing method: {best_method.replace('_', ' ').title()}")
        print(f"   CV Score: {scores[best_method]:.4f}")
        
        # Train final model and test
        print(f"\n🧪 Step 4: Training final model and testing...")
        
        # Helper function to select features safely
        def select_features_safely_main(X, features, feature_cols):
            """Safely select features from X whether it's pandas DataFrame or numpy array"""
            if hasattr(X, 'iloc'):  # pandas DataFrame
                return X[features]
            else:  # numpy array
                if isinstance(features[0], str):
                    # Convert feature names to indices
                    feature_indices = [i for i, col in enumerate(feature_cols) if col in features]
                    return X[:, feature_indices]
                else:
                    # Assume features are already indices
                    return X[:, features] if hasattr(features, '__iter__') else X
        
        X_train_best = select_features_safely_main(X_train, best_features, feature_columns)
        X_test_best = select_features_safely_main(X_test, best_features, feature_columns)
        
        final_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        final_rf.fit(X_train_best, y_train)
        
        y_pred = final_rf.predict(X_test_best)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        
        # Create visualizations
        create_visualizations(methods, scores, X_train, y_train, output_dir, feature_columns)
        
        # Save results
        print(f"\n💾 Step 5: Saving results...")
        
        # Save all method features
        all_features_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in methods.items()]))
        all_features_output = output_dir / f"top_{n_features}_features_all_methods.csv"
        all_features_df.to_csv(all_features_output, index=False)
        
        # Save best method features
        best_features_df = pd.DataFrame({'feature': best_features})
        best_features_output = output_dir / f"top_{n_features}_features_best_method_{best_method}.csv"
        best_features_df.to_csv(best_features_output, index=False)
        
        # Save ML-ready datasets
        train_output = output_dir / f"ml_ready_train_data_{len(best_features)}_features.csv"
        test_output = output_dir / f"ml_ready_test_data_{len(best_features)}_features.csv"
        
        train_data = X_train_best.copy()
        train_data[target] = y_train
        train_data.to_csv(train_output, index=False)
        
        test_data = X_test_best.copy()
        test_data[target] = y_test
        test_data.to_csv(test_output, index=False)
        
        # Save comprehensive summary
        summary = {
            'dataset_info': {
                'file': str(data_file),
                'original_shape': df.shape,
                'filtered_features': len(X_filtered.columns),
                'target_variable': target,
                'test_size': test_size
            },
            'feature_selection': {
                'n_features_selected': n_features,
                'methods_evaluated': list(methods.keys()),
                'method_scores': scores,
                'best_method': best_method,
                'best_method_score': scores[best_method]
            },
            'model_performance': {
                'test_accuracy': test_accuracy,
                'best_features_count': len(best_features)
            },
            'runtime_seconds': time.time() - start_time
        }
        
        summary_output = output_dir / "comprehensive_analysis_summary.json"
        with open(summary_output, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Final summary
        print("")
        print("=" * 60)
        print("🎉 FEATURE SELECTION ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"✅ Best method: {best_method.replace('_', ' ').title()}")
        print(f"✅ CV Score: {scores[best_method]:.4f}")
        print(f"✅ Test Accuracy: {test_accuracy:.4f}")
        print(f"✅ Features selected: {len(best_features)}")
        print("")
        print("📁 OUTPUT FILES:")
        print(f"  📋 All methods features: {all_features_output.name}")
        print(f"  🏆 Best method features: {best_features_output.name}")
        print(f"  🤖 ML-ready train data: {train_output.name}")
        print(f"  🧪 ML-ready test data: {test_output.name}")
        print(f"  📊 Performance plots: feature_selection_performance.png")
        print(f"  🔍 Best method analysis: best_method_analysis.png")
        print(f"  � Feature overlap summary: feature_overlap_summary.csv")
        print(f"  �📈 Complete summary: comprehensive_analysis_summary.json")
        print("")
        print(f"⏱️  Total runtime: {time.time() - start_time:.2f} seconds")
        print("=" * 60)
        
        print(f"\n🔥 TOP 10 FEATURES FROM BEST METHOD ({best_method.replace('_', ' ').title()}):")
        print("-" * 30)
        for i, feature in enumerate(best_features[:10], 1):
            print(f"  {i:2d}. {feature}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
