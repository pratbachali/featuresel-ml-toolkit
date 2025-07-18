#!/usr/bin/env python3
"""
Test Enhanced Feature Selection with combined_dataset_SVA_corrected.csv
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

def filter_top_variance_genes(X, n_features):
    """Filter top N variance genes"""
    variances = X.var()
    top_features = variances.nlargest(n_features).index.tolist()
    return top_features

def enhanced_feature_selection(X_train, y_train, n_features=100):
    """Perform enhanced feature selection using multiple methods"""
    
    methods = {}
    scores = {}
    
    print(f"    Running univariate feature selection...")
    # Univariate feature selection
    selector_f = SelectKBest(score_func=f_classif, k=n_features)
    selector_f.fit(X_train, y_train)
    methods['univariate_f'] = selector_f.get_support(indices=True)
    
    print(f"    Running mutual information...")
    # Mutual information
    selector_mi = SelectKBest(score_func=mutual_info_classif, k=n_features)
    selector_mi.fit(X_train, y_train)
    methods['mutual_info'] = selector_mi.get_support(indices=True)
    
    print(f"    Running Random Forest feature importance...")
    # Random Forest feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    feature_importance = pd.Series(rf.feature_importances_, index=X_train.columns)
    methods['random_forest'] = feature_importance.nlargest(n_features).index
    
    print(f"    Running RFE with Logistic Regression...")
    # RFE with Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    rfe = RFE(lr, n_features_to_select=n_features)
    rfe.fit(X_train, y_train)
    methods['rfe_logistic'] = rfe.get_support(indices=True)
    
    print(f"    Evaluating methods with cross-validation...")
    # Evaluate each method with cross-validation
    for method_name, features in methods.items():
        if isinstance(features, np.ndarray):
            feature_names = X_train.columns[features]
        else:
            feature_names = features
        
        X_selected = X_train[feature_names]
        rf_eval = RandomForestClassifier(n_estimators=50, random_state=42)
        cv_scores = cross_val_score(rf_eval, X_selected, y_train, cv=5, scoring='accuracy')
        scores[method_name] = cv_scores.mean()
        print(f"      {method_name}: {cv_scores.mean():.4f}")
    
    # Find best method
    best_method = max(scores, key=scores.get)
    best_features = methods[best_method]
    if isinstance(best_features, np.ndarray):
        best_features = X_train.columns[best_features]
    
    return methods, scores, best_method, list(best_features)

def create_visualizations(methods, scores, output_dir):
    """Create visualizations of feature selection results"""
    
    print(f"    Creating visualizations...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Feature Selection Methods Performance Analysis', fontsize=16, fontweight='bold')
    
    # Method scores bar plot
    method_names = list(scores.keys())
    score_values = list(scores.values())
    
    ax1 = axes[0]
    bars = ax1.bar(range(len(method_names)), score_values, color=plt.cm.Set3(np.linspace(0, 1, len(method_names))))
    ax1.set_title('Cross-Validation Accuracy by Method', fontweight='bold')
    ax1.set_xlabel('Feature Selection Method')
    ax1.set_ylabel('CV Accuracy Score')
    ax1.set_xticks(range(len(method_names)))
    ax1.set_xticklabels([m.replace('_', ' ').title() for m in method_names], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, score_values)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Method ranking
    ax2 = axes[1]
    sorted_methods = sorted(zip(method_names, score_values), key=lambda x: x[1], reverse=True)
    ranked_methods, ranked_scores = zip(*sorted_methods)
    
    colors = ['gold' if i == 0 else 'silver' if i == 1 else 'chocolate' if i == 2 else 'lightblue' 
              for i in range(len(ranked_methods))]
    bars2 = ax2.barh(range(len(ranked_methods)), ranked_scores, color=colors)
    ax2.set_title('Method Performance Ranking', fontweight='bold')
    ax2.set_xlabel('CV Accuracy Score')
    ax2.set_yticks(range(len(ranked_methods)))
    ax2.set_yticklabels([m.replace('_', ' ').title() for m in ranked_methods])
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars2, ranked_scores)):
        ax2.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "feature_selection_results.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    Visualization saved: {output_dir / 'feature_selection_results.png'}")

def main():
    """Main function to run feature selection"""
    
    # Configuration
    data_file = "/Users/prathyushabachali/Library/CloudStorage/GoogleDrive-prathyusha.bachali@ampelbiosolutions.com/Shared drives/CONFIDENTIAL - WellGENE/Prat/ML_analysis/FinalMLAnalysis/combined_dataset_SVA_corrected.csv"
    target = "Category"
    n_features = 100
    test_size = 0.2
    output_dir = Path("feature_selection_results")
    
    print("\n" + "="*60)
    print("ENHANCED FEATURE SELECTION PIPELINE TEST")
    print("="*60)
    
    start_time = time.time()
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Load data
        print(f"\n📁 Step 1: Loading data...")
        df = pd.read_csv(data_file)
        print(f"  Dataset shape: {df.shape}")
        print(f"  Target variable: {target}")
        print(f"  Target distribution:\n{df[target].value_counts()}")
        
        # Prepare features and target
        X = df.drop(columns=[target])
        y = df[target]
        
        # Encode target if necessary
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            print(f"  Target encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        print(f"  Original features: {len(X.columns)}")
        
        # Identify and exclude non-numeric columns
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns
        
        if len(non_numeric_columns) > 0:
            print(f"  Found {len(non_numeric_columns)} non-numeric columns that will be excluded:")
            for col in non_numeric_columns:
                print(f"    - {col} (dtype: {X[col].dtype})")
            
            # Use only numeric columns for feature selection
            X = X[numeric_columns]
            print(f"  After excluding non-numeric columns: {len(X.columns)} features")
        else:
            print("  All feature columns are numeric")
        
        # Step 1: Apply variance filtering FIRST (on entire dataset)
        print(f"\n🔬 Step 2: Filtering top 1000 variance genes...")
        top_variance_features = filter_top_variance_genes(X, 1000)
        X_filtered = X[top_variance_features]
        print(f"  After variance filtering: {len(X_filtered.columns)} features")
        
        # Step 2: Handle train/test split based on Category column
        print(f"\n✂️  Step 3: Using predefined train/test split from 'Category' column...")
        
        # Split based on Category column
        train_mask = df['Category'].str.lower() == 'train'
        test_mask = df['Category'].str.lower() == 'test'
        
        # Convert masks to numpy arrays for proper indexing
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
        print(f"  Training target distribution: {np.bincount(y_train)}")
        print(f"  Test target distribution: {np.bincount(y_test)}")
        
        # Step 3: Feature selection on TRAINING DATA ONLY
        print(f"\n🎯 Step 4: Running feature selection on training data only...")
        methods, scores, best_method, best_features = enhanced_feature_selection(
            X_train, y_train, n_features
        )
        
        print(f"\n  Best method: {best_method} (score: {scores[best_method]:.4f})")
        
        # Step 4: Create outputs
        print(f"\n💾 Step 5: Creating outputs...")
        
        # Save all features from all methods to CSV
        all_features_df = pd.DataFrame()
        for method in methods:
            if isinstance(methods[method], np.ndarray):
                method_features = X_train.columns[methods[method]].tolist()
            else:
                method_features = list(methods[method])
            
            method_df = pd.DataFrame({
                'method': method,
                'feature': method_features,
                'rank': range(1, len(method_features) + 1)
            })
            all_features_df = pd.concat([all_features_df, method_df], ignore_index=True)
        
        all_features_output = output_dir / f"all_methods_top_{n_features}_features.csv"
        all_features_df.to_csv(all_features_output, index=False)
        print(f"  All methods features saved to: {all_features_output}")
        
        # Save best method features as a separate CSV
        best_features_df = pd.DataFrame({
            'feature': best_features,
            'rank': range(1, len(best_features) + 1),
            'method': best_method
        })
        
        best_features_output = output_dir / f"best_method_{best_method}_features.csv"
        best_features_df.to_csv(best_features_output, index=False)
        print(f"  Best method features saved to: {best_features_output}")
        
        # Create ML-ready datasets
        X_train_best = X_train[best_features]
        X_test_best = X_test[best_features]
        
        # Save train and test sets with selected features
        train_df = pd.concat([X_train_best, pd.Series(y_train, name=target, index=X_train_best.index)], axis=1)
        test_df = pd.concat([X_test_best, pd.Series(y_test, name=target, index=X_test_best.index)], axis=1)
        
        train_output = output_dir / "ml_ready_train_data.csv"
        test_output = output_dir / "ml_ready_test_data.csv"
        
        train_df.to_csv(train_output, index=False)
        test_df.to_csv(test_output, index=False)
        
        print(f"  ML-ready training data saved to: {train_output}")
        print(f"  ML-ready test data saved to: {test_output}")
        
        # Step 5: Create visualizations
        print(f"\n📊 Step 6: Creating visualizations...")
        create_visualizations(methods, scores, output_dir)
        
        # Step 6: Evaluate best method on test set
        print(f"\n🎯 Step 7: Evaluating best method on test set...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train_best, y_train)
        y_pred = rf_model.predict(X_test_best)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"  Test set accuracy with {best_method}: {test_accuracy:.4f}")
        
        # Step 7: Save comprehensive summary
        print(f"\n📋 Step 8: Saving comprehensive summary...")
        
        summary = {
            'analysis_info': {
                'input_file': str(data_file),
                'target_column': target,
                'test_size': test_size,
                'runtime_seconds': time.time() - start_time,
                'runtime_formatted': f"{time.time() - start_time:.2f}s"
            },
            'data_info': {
                'original_features': len(df.columns) - 1,  # Exclude target
                'non_numeric_excluded': len(non_numeric_columns),
                'after_variance_filtering': len(X_filtered.columns),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'train_target_distribution': np.bincount(y_train).tolist(),
                'test_target_distribution': np.bincount(y_test).tolist()
            },
            'feature_selection': {
                'methods_evaluated': list(methods.keys()),
                'method_scores': scores,
                'best_method': best_method,
                'best_method_score': scores[best_method],
                'selected_features_count': len(best_features),
                'test_accuracy': test_accuracy
            },
            'output_files': {
                'all_features': str(all_features_output),
                'best_features': str(best_features_output),
                'train_data': str(train_output),
                'test_data': str(test_output),
                'visualization': str(output_dir / "feature_selection_results.png")
            }
        }
        
        summary_output = output_dir / "analysis_summary.json"
        with open(summary_output, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"  Summary saved to: {summary_output}")
        
        # Final summary
        runtime = time.time() - start_time
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print(f"📈 Best performing method: {best_method} (CV score: {scores[best_method]:.4f})")
        print(f"🎯 Test accuracy: {test_accuracy:.4f}")
        print(f"⏱️  Total runtime: {runtime:.2f} seconds")
        print(f"📁 Results saved in: {output_dir}")
        
        print(f"\n📋 SUMMARY:")
        print(f"  • Original features: {len(df.columns) - 1}")
        print(f"  • Non-numeric excluded: {len(non_numeric_columns)}")
        print(f"  • After variance filtering: {len(X_filtered.columns)}")
        print(f"  • Selected features: {len(best_features)}")
        print(f"  • Best method: {best_method}")
        print(f"  • CV accuracy: {scores[best_method]:.4f}")
        print(f"  • Test accuracy: {test_accuracy:.4f}")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
