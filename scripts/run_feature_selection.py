#!/usr/bin/env python3
"""
Enhanced Feature Selection Script with Interactive Prompts

This script provides an interactive feature selection analysis for genomic data,
including proper train/test separation, visualizations, and comprehensive outputs.
"""

import sys
import argparse
from pathlib import Path
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to sys.path to enable package imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

try:
    from src.data.loader import DataLoader
    from src.features.selector import FeatureSelector
    from src.features.variance_filter import VarianceFilter
    from src.utils.logger import setup_logging, get_logger
    from src.utils.helpers import ensure_directory, save_json, format_duration
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Trying alternative import method...")
    
    # Fallback: Direct execution without package structure
    print("Running in standalone mode...")
    import logging
    
    # Create simple implementations for required functions
    def setup_logging(level='INFO', log_file=None):
        import logging
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
        import json
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def format_duration(seconds):
        return f"{seconds:.2f}s"
    
    # Simple implementations for classes
    class DataLoader:
        @staticmethod
        def load_csv(file_path):
            return pd.read_csv(file_path)
    
    class VarianceFilter:
        def select_top_variance_features(self, X, n_features):
            variances = X.var()
            top_features = variances.nlargest(n_features).index.tolist()
            return top_features
    
    class FeatureSelector:
        def __init__(self):
            self.method_scores = {}
            self.method_features = {}
        
        def select_features(self, X_train, y_train, n_features=100):
            from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score
            
            methods = {}
            scores = {}
            
            # Univariate feature selection
            selector_f = SelectKBest(score_func=f_classif, k=n_features)
            selector_f.fit(X_train, y_train)
            methods['univariate_f'] = selector_f.get_support(indices=True)
            
            # Mutual information
            selector_mi = SelectKBest(score_func=mutual_info_classif, k=n_features)
            selector_mi.fit(X_train, y_train)
            methods['mutual_info'] = selector_mi.get_support(indices=True)
            
            # Random Forest feature importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            feature_importance = pd.Series(rf.feature_importances_, index=X_train.columns)
            methods['random_forest'] = feature_importance.nlargest(n_features).index
            
            # RFE with Logistic Regression
            lr = LogisticRegression(random_state=42, max_iter=1000)
            rfe = RFE(lr, n_features_to_select=n_features)
            rfe.fit(X_train, y_train)
            methods['rfe_logistic'] = rfe.get_support(indices=True)
            
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
                
                self.method_features[method_name] = list(feature_names)
            
            self.method_scores = scores
            return methods
    
    print("Using simplified implementations...")

def get_user_input():
    """Get user input for all pipeline parameters"""
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
        return None, None, None, None, None, None, None, None  # Updated to return 8 values including hyperparameter_tuning
    
    # Get target variable
    target = input(f"\nEnter the target column name: ").strip()
    while target not in df.columns:
        print(f"Error: Column '{target}' not found in dataset.")
        target = input("Please enter a valid column name: ").strip()
    
    # Get number of top variance genes for pre-filtering
    while True:
        try:
            n_variance_genes = int(input(f"\nEnter number of top variance genes for pre-filtering (default 1000): ") or "1000")
            if n_variance_genes > 0:
                break
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Check if Category column exists for predefined train/test split
    has_category = 'Category' in df.columns
    use_predefined_split = False
    test_size = 0.2
    
    if has_category:
        print(f"\n📋 Found 'Category' column with predefined train/test split:")
        print(f"   {df['Category'].value_counts()}")
        use_predefined = input("\nUse predefined train/test split from 'Category' column? (y/n, default y): ").strip().lower()
        if use_predefined in ['', 'y', 'yes']:
            use_predefined_split = True
            print("✅ Will use predefined train/test split from 'Category' column")
        else:
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
    
    # Get number of top features to select from each method
    while True:
        try:
            n_features = int(input(f"\nEnter number of top features to select from each method (default 100): ") or "100")
            if n_features > 0:
                break
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Ask about hyperparameter tuning
    use_hyperparameter = input(f"\nUse hyperparameter tuning for better accuracy? (slower but more accurate) (y/n, default y): ").strip().lower()
    use_hyperparameter_tuning = use_hyperparameter in ['', 'y', 'yes']
    
    if use_hyperparameter_tuning:
        print("✅ Will use hyperparameter tuning (may take longer but provides better results)")
    else:
        print("✅ Will use default parameters (faster execution)")
    
    # Get output directory
    output_dir = input(f"\nEnter output directory (default 'feature_selection_results'): ").strip()
    if not output_dir:
        output_dir = 'feature_selection_results'
    
    return data_file, target, n_variance_genes, use_predefined_split, test_size, n_features, output_dir, use_hyperparameter_tuning

def create_exploratory_analysis(df, target_col, output_dir, feature_columns):
    """Create exploratory data analysis visualizations"""
    print("\n📊 Creating exploratory data analysis...")
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Exploratory Data Analysis', fontsize=16, fontweight='bold')
    
    # 1. Target distribution overall
    target_counts = df[target_col].value_counts()
    axes[0, 0].bar(target_counts.index, target_counts.values, 
                   color=['skyblue', 'lightcoral'], alpha=0.7)
    axes[0, 0].set_title(f'Overall {target_col} Distribution')
    axes[0, 0].set_xlabel(target_col)
    axes[0, 0].set_ylabel('Count')
    for i, v in enumerate(target_counts.values):
        axes[0, 0].text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
    
    # 2. Target distribution by Dataset (if Dataset column exists)
    if 'Dataset' in df.columns:
        dataset_target_counts = df.groupby(['Dataset', target_col]).size().unstack(fill_value=0)
        dataset_target_counts.plot(kind='bar', ax=axes[0, 1], 
                                  color=['skyblue', 'lightcoral'], alpha=0.7)
        axes[0, 1].set_title(f'{target_col} Distribution by Dataset')
        axes[0, 1].set_xlabel('Dataset')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].legend(title=target_col)
        
        # Add value labels on bars
        for container in axes[0, 1].containers:
            axes[0, 1].bar_label(container, label_type='edge')
    else:
        axes[0, 1].text(0.5, 0.5, 'No Dataset column found', 
                       transform=axes[0, 1].transAxes, ha='center', va='center',
                       fontsize=12, style='italic')
        axes[0, 1].set_title('Dataset Distribution (N/A)')
    
    # 3. PCA plot by target variable
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Prepare data for PCA (use a subset of features for speed)
    X_pca = df[feature_columns].fillna(0)  # Fill NaN values
    
    # Use top 1000 variance features for PCA to speed up computation
    if len(feature_columns) > 1000:
        variances = X_pca.var()
        top_variance_features = variances.nlargest(1000).index.tolist()
        X_pca = X_pca[top_variance_features]
    
    # Standardize features
    scaler = StandardScaler()
    X_pca_scaled = scaler.fit_transform(X_pca)
    
    # Apply PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca_transformed = pca.fit_transform(X_pca_scaled)
    
    # Plot PCA by target variable
    target_unique = df[target_col].unique()
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold'][:len(target_unique)]
    
    for i, target_val in enumerate(target_unique):
        mask = df[target_col] == target_val
        axes[1, 0].scatter(X_pca_transformed[mask, 0], X_pca_transformed[mask, 1], 
                          c=colors[i], label=f'{target_col}={target_val}', alpha=0.7, s=50)
    
    axes[1, 0].set_title(f'PCA Plot by {target_col}')
    axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. PCA plot by Dataset (if Dataset column exists)
    if 'Dataset' in df.columns:
        dataset_unique = df['Dataset'].unique()
        colors_dataset = ['red', 'blue', 'green', 'purple'][:len(dataset_unique)]
        
        for i, dataset_val in enumerate(dataset_unique):
            mask = df['Dataset'] == dataset_val
            axes[1, 1].scatter(X_pca_transformed[mask, 0], X_pca_transformed[mask, 1], 
                              c=colors_dataset[i], label=f'Dataset={dataset_val}', alpha=0.7, s=50)
        
        axes[1, 1].set_title('PCA Plot by Dataset')
        axes[1, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[1, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # If no Dataset column, create a combined plot showing both target and any other categorical variable
        axes[1, 1].text(0.5, 0.5, 'No Dataset column found', 
                       transform=axes[1, 1].transAxes, ha='center', va='center',
                       fontsize=12, style='italic')
        axes[1, 1].set_title('PCA by Dataset (N/A)')
    
    plt.tight_layout()
    
    # Save the plot
    eda_output = output_dir / 'exploratory_data_analysis.png'
    plt.savefig(eda_output, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Exploratory analysis saved to: {eda_output.name}")
    
    # Print summary statistics
    print(f"\n📈 Data Summary:")
    print(f"  • Total samples: {len(df)}")
    print(f"  • {target_col} distribution: {dict(df[target_col].value_counts())}")
    if 'Dataset' in df.columns:
        print(f"  • Dataset distribution: {dict(df['Dataset'].value_counts())}")
    print(f"  • PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}")
    print(f"  • Total explained variance: {pca.explained_variance_ratio_.sum():.2%}")

def create_visualizations(selector, output_dir, X_train, y_train):
    """Create comprehensive visualizations"""
    logger = get_logger(__name__)
    logger.info("Creating visualizations...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Figure 1: Method Performance Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Feature Selection Methods Performance Analysis', fontsize=16, fontweight='bold')
    
    # Method scores bar plot
    methods = list(selector.method_scores.keys())
    scores = list(selector.method_scores.values())
    
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(methods)), scores, color=plt.cm.Set3(np.linspace(0, 1, len(methods))))
    ax1.set_title('Cross-Validation Accuracy by Method', fontweight='bold')
    ax1.set_xlabel('Feature Selection Method')
    ax1.set_ylabel('CV Accuracy Score')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Feature counts by method
    ax2 = axes[0, 1]
    feature_counts = [len(selector.method_features[method]) for method in methods]
    bars2 = ax2.bar(range(len(methods)), feature_counts, color=plt.cm.Set2(np.linspace(0, 1, len(methods))))
    ax2.set_title('Number of Features Selected by Method', fontweight='bold')
    ax2.set_xlabel('Feature Selection Method')
    ax2.set_ylabel('Number of Features')
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, count in zip(bars2, feature_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    # Method ranking
    ax3 = axes[1, 0]
    sorted_methods = sorted(zip(methods, scores), key=lambda x: x[1], reverse=True)
    ranked_methods, ranked_scores = zip(*sorted_methods)
    
    colors = ['gold' if i == 0 else 'silver' if i == 1 else 'chocolate' if i == 2 else 'lightblue' 
              for i in range(len(ranked_methods))]
    bars3 = ax3.barh(range(len(ranked_methods)), ranked_scores, color=colors)
    ax3.set_title('Method Performance Ranking', fontweight='bold')
    ax3.set_xlabel('CV Accuracy Score')
    ax3.set_yticks(range(len(ranked_methods)))
    ax3.set_yticklabels([m.replace('_', ' ').title() for m in ranked_methods])
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars3, ranked_scores)):
        ax3.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', ha='left', va='center', fontweight='bold')
    
    # Feature overlap analysis
    ax4 = axes[1, 1]
    if len(methods) >= 2:
        # Calculate pairwise overlaps
        overlap_matrix = np.zeros((len(methods), len(methods)))
        count_matrix = np.zeros((len(methods), len(methods)))
        
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i != j:
                    set1 = set(selector.method_features[method1])
                    set2 = set(selector.method_features[method2])
                    intersection_count = len(set1.intersection(set2))
                    union_count = len(set1.union(set2))
                    
                    overlap = intersection_count / union_count if union_count > 0 else 0
                    overlap_matrix[i][j] = overlap
                    count_matrix[i][j] = intersection_count
                else:
                    overlap_matrix[i][j] = 1.0
                    count_matrix[i][j] = len(selector.method_features[method1])
        
        im = ax4.imshow(overlap_matrix, cmap='Blues', aspect='auto')
        ax4.set_title('Feature Selection Overlap (Jaccard Index & Gene Count)', fontweight='bold')
        ax4.set_xticks(range(len(methods)))
        ax4.set_yticks(range(len(methods)))
        ax4.set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=45, ha='right')
        ax4.set_yticklabels([m.replace('_', ' ').title() for m in methods])
        
        # Add text annotations with both Jaccard index and gene count
        for i in range(len(methods)):
            for j in range(len(methods)):
                if i == j:
                    text = ax4.text(j, i, f'{overlap_matrix[i, j]:.2f}\n({int(count_matrix[i, j])})',
                                   ha="center", va="center", color="black" if overlap_matrix[i, j] < 0.5 else "white",
                                   fontsize=8, fontweight='bold')
                else:
                    text = ax4.text(j, i, f'{overlap_matrix[i, j]:.2f}\n({int(count_matrix[i, j])})',
                                   ha="center", va="center", color="black" if overlap_matrix[i, j] < 0.5 else "white",
                                   fontsize=8)
        
        plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    else:
        ax4.text(0.5, 0.5, 'Need at least 2 methods\nfor overlap analysis', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Feature Selection Overlap Analysis', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_selection_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Best Method Feature Analysis
    if selector.best_method and selector.best_method in selector.method_features:
        fig, axes = plt.subplots(2, 2, figsize=(20, 18))  # Increased height for better vertical spacing
        fig.suptitle(f'Best Method Analysis: {selector.best_method.replace("_", " ").title()}', 
                    fontsize=16, fontweight='bold', y=0.98)  # Adjusted position of title
        
        # Add more spacing between subplots
        plt.subplots_adjust(hspace=0.4, wspace=0.35)  # Increased spacing between subplots
        
        best_features = selector.method_features[selector.best_method]
        
        # 1. Cross-validation score comparison (why this method was chosen)
        ax1 = axes[0, 0]
        methods = list(selector.method_scores.keys())
        scores = list(selector.method_scores.values())
        
        # Color the best method differently
        colors = ['gold' if method == selector.best_method else 'lightblue' for method in methods]
        bars = ax1.bar(range(len(methods)), scores, color=colors, width=0.65)  # Slightly narrower bars for better spacing
        
        # Adjust ylim to add more space at the top for labels
        y_max = max(scores) + 0.05  # Add 5% headroom for labels
        ax1.set_ylim(0, min(1.0, y_max))
        
        ax1.set_title('Method Selection: CV Performance Comparison', fontweight='bold', pad=15)  # Added padding to title
        ax1.set_xlabel('Feature Selection Method', labelpad=10)  # Added padding to xlabel
        ax1.set_ylabel('Cross-Validation AUC Score', labelpad=10)  # Added padding to ylabel
        ax1.set_xticks(range(len(methods)))
        ax1.set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=45, ha='right', fontsize=9)  # Reduced font size
        ax1.grid(True, alpha=0.3)
        
        # Add score labels on bars with improved positioning
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Add annotation for best method - positioned to avoid blocking numbers
        best_idx = methods.index(selector.best_method)
        y_max = max(scores)
        annotation_y = max(scores[best_idx] + 0.03, y_max * 0.85)  # Adjusted positioning
        ax1.annotate('BEST', xy=(best_idx, scores[best_idx]), 
                    xytext=(best_idx, annotation_y),
                    ha='center', va='bottom', fontweight='bold', color='red', fontsize=10,
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # 2. Cross-validation performance comparison (AUC vs Accuracy)
        ax2 = axes[0, 1]
        methods = list(selector.method_scores.keys())
        auc_scores = list(selector.method_scores.values())
        accuracy_scores = list(selector.method_accuracy_scores.values())
        
        x = np.arange(len(methods))
        width = 0.25  # Narrower bars for better spacing
        
        # Create bars for AUC and Accuracy
        bars1 = ax2.bar(x - width/2, auc_scores, width, label='AUC', alpha=0.8, color='skyblue')
        bars2 = ax2.bar(x + width/2, accuracy_scores, width, label='Accuracy', alpha=0.8, color='lightcoral')
        
        ax2.set_title('Cross-Validation Performance: AUC vs Accuracy', fontweight='bold', pad=15)  # Added padding
        ax2.set_xlabel('Feature Selection Method', labelpad=10)
        ax2.set_ylabel('Score', labelpad=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=45, ha='right', fontsize=9)  # Reduced font size
        ax2.legend(loc='upper right', framealpha=0.7)
        ax2.grid(True, alpha=0.3)
        
        # Set ylim to add headroom for labels
        max_val = max(max(auc_scores), max(accuracy_scores))
        ax2.set_ylim(0, min(1.0, max_val + 0.1))  # More space at top for labels
        
        # Add value labels on bars with improved spacing and positioning
        for i, (bar, score) in enumerate(zip(bars1, auc_scores)):
            height = bar.get_height()
            # Alternating heights for labels to avoid overlap in narrow bars
            offset = 0.02 if i % 2 == 0 else 0.04
            ax2.text(bar.get_x() + bar.get_width()/2., height + offset,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        for i, (bar, score) in enumerate(zip(bars2, accuracy_scores)):
            height = bar.get_height()
            # Alternating heights for labels to avoid overlap in narrow bars
            offset = 0.04 if i % 2 == 0 else 0.02
            ax2.text(bar.get_x() + bar.get_width()/2., height + offset,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Highlight best method with better positioning
        best_idx = methods.index(selector.best_method)
        ax2.axvline(x=best_idx, color='red', linestyle='--', alpha=0.7, linewidth=2)
        
        # Position BEST text more carefully to avoid overlap
        max_height = max(max(auc_scores), max(accuracy_scores))
        best_text_y = min(0.70, max_height - 0.25)  # Adjusted positioning
        
        ax2.text(best_idx, best_text_y, 'BEST', ha='center', va='center', fontweight='bold', 
                color='red', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="yellow", alpha=0.8, edgecolor='red'))
        
        # 3. Class separation analysis
        ax3 = axes[1, 0]
        if len(best_features) >= 2:
            # Use first two features for scatter plot
            feature1, feature2 = best_features[0], best_features[1]
            for class_label in np.unique(y_train):
                mask = y_train == class_label
                ax3.scatter(X_train.loc[mask, feature1], X_train.loc[mask, feature2], 
                           label=f'Class {class_label}', alpha=0.6)
            
            ax3.set_xlabel(f'{feature1[:20]}...' if len(feature1) > 20 else feature1)
            ax3.set_ylabel(f'{feature2[:20]}...' if len(feature2) > 20 else feature2)
            ax3.set_title('Class Separation (Top 2 Features)', fontweight='bold', pad=15)
            ax3.legend(loc='best', framealpha=0.7)
            ax3.grid(True, alpha=0.3)
    
            # 4. Training Performance Details
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            # Performance summary text with improved formatting
            performance_text = f"""
        🏆 BEST METHOD SELECTED: {selector.best_method.replace('_', ' ').title()}
        
        📊 SELECTION CRITERIA:
        ✓ Cross-validation on TRAINING data only
        ✓ 5-fold stratified cross-validation
        ✓ AUC scoring metric for selection
        ✓ No test data used for selection
        
        📈 PERFORMANCE METRICS:
        • Best CV AUC Score: {selector.method_scores[selector.best_method]:.4f}
        • Best CV Accuracy: {selector.method_accuracy_scores[selector.best_method]:.4f}
        • Features Selected: {len(best_features)}
        • Training Set Size: {len(X_train)} samples
        
        🔬 METHODOLOGY:
        1. Variance filtering (top 1000 features)
        2. Train/test split
        3. Feature selection on training only
        4. Cross-validation evaluation (AUC & Accuracy)
        5. Best method selection (based on AUC)
        
        📋 TOP 10 FEATURES:
        """
        
        for i, feature in enumerate(best_features[:10], 1):
            performance_text += f"\n  {i:2d}. {feature[:25]}{'...' if len(feature) > 25 else ''}"
        
        ax4.text(0.05, 0.95, performance_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        # Better layout adjustment with more space between subplots and around the figure
        plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=1.0, w_pad=0.8)  # Increased padding
        plt.savefig(output_dir / 'best_method_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Visualizations saved to {output_dir}")

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
    
    # Log results
    logger = get_logger(__name__)
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def main():
    parser = argparse.ArgumentParser(description='Enhanced Feature Selection Analysis')
    parser.add_argument('--interactive', action='store_true', 
                       help='Run in interactive mode with user prompts')
    parser.add_argument('--data_file', help='Path to data file (CSV)')
    parser.add_argument('--target', help='Target column name')
    parser.add_argument('--top_variance_genes', type=int, default=1000,
                       help='Number of top variance genes to pre-filter (default: 1000)')
    parser.add_argument('--use_predefined_split', action='store_true',
                       help='Use predefined train/test split from category column. When enabled, the category column will be used to split data into train/test sets instead of random splitting.')
    parser.add_argument('--category_column', default='Category',
                       help='Column name containing train/test split labels (e.g., "train", "test", "Dataset_A", "Dataset_B"). Only used when --use_predefined_split is enabled. (default: Category)')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size for random split (0.1-0.5, default: 0.2). Ignored if --use_predefined_split is enabled.')
    parser.add_argument('--n_features', type=int, default=100, 
                       help='Number of features to select per method (default: 100)')
    parser.add_argument('--output_dir', default='feature_selection_results',
                       help='Output directory (default: feature_selection_results)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility (default: 42)')
    parser.add_argument('--hyperparameter_tuning', action='store_true',
                       help='Enable hyperparameter tuning for better accuracy (slower)')
    parser.add_argument('--no_hyperparameter_tuning', dest='hyperparameter_tuning', action='store_false',
                       help='Disable hyperparameter tuning for faster execution')
    parser.set_defaults(hyperparameter_tuning=True)
    
    args = parser.parse_args()
    
    # Get input parameters
    if args.interactive or not args.data_file or not args.target:
        data_file, target, n_variance_genes, use_predefined_split, test_size, n_features, output_dir, use_hyperparameter_tuning = get_user_input()
        if not data_file:
            return
    else:
        data_file = args.data_file
        target = args.target
        n_variance_genes = args.top_variance_genes
        n_features = args.n_features
        use_predefined_split = args.use_predefined_split
        output_dir = args.output_dir
        use_hyperparameter_tuning = args.hyperparameter_tuning
        
        # Only use test_size if not using predefined split
        if use_predefined_split:
            test_size = None  # Not used for predefined splits
            print(f"Using predefined train/test split from '{args.category_column}' column")
        else:
            test_size = args.test_size
            print(f"Using random train/test split with test_size={test_size}")
        
        # Print hyperparameter tuning status
        if use_hyperparameter_tuning:
            print(f"Hyperparameter tuning enabled (may take longer but provides better results)")
        else:
            print(f"Hyperparameter tuning disabled (faster execution with default parameters)")
    
    # Setup logging
    output_dir = Path(output_dir)  # Use the output_dir from parameters, not args
    ensure_directory(output_dir)
    setup_logging(level='INFO', log_file=str(output_dir / "feature_selection.log"))
    logger = get_logger(__name__)
    
    logger.info("=" * 60)
    logger.info("ENHANCED FEATURE SELECTION ANALYSIS")
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
        exclude_columns = {target, args.category_column, 'Dataset', 'Patient_ID', 'X', 'Unnamed: 0'}
        if hasattr(df, 'index'):
            exclude_columns.add('index')
        
        # Remove target and other non-feature columns
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        X = df[feature_columns]
        y = df[target]
        
        logger.info(f"Original features: {len(X.columns)}")
        logger.info(f"Excluded columns: {[col for col in df.columns if col not in feature_columns]}")
        
        # Identify and exclude non-numeric columns
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns
        
        if len(non_numeric_columns) > 0:
            logger.info(f"Found {len(non_numeric_columns)} non-numeric columns that will be excluded:")
            for col in non_numeric_columns:
                logger.info(f"  - {col} (dtype: {X[col].dtype})")
            
            # Use only numeric columns for feature selection
            X = X[numeric_columns]
            logger.info(f"After excluding non-numeric columns: {len(X.columns)} features")
        else:
            logger.info("All feature columns are numeric")
        
        # Step 1: Apply variance filtering FIRST (on entire dataset)
        logger.info(f"Step 1: Filtering top {args.top_variance_genes} variance genes...")
        variance_filter = VarianceFilter()
        top_variance_features = variance_filter.select_top_variance_features(
            X, args.top_variance_genes
        )
        X_filtered = X[top_variance_features]
        logger.info(f"After variance filtering: {len(X_filtered.columns)} features")
        
        # Step 1.5: Create exploratory data analysis
        create_exploratory_analysis(df, target, output_dir, X_filtered.columns.tolist())
        
        # Step 2: Split into train/test AFTER variance filtering
        if use_predefined_split:
            category_column = args.category_column
            if category_column not in df.columns:
                logger.error(f"Category column '{category_column}' not found in dataset. Available columns: {list(df.columns)}")
                raise ValueError(f"Category column '{category_column}' not found in dataset")
            
            logger.info(f"Step 2: Using predefined train/test split from '{category_column}' column...")
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
            
            X_train = X_filtered[train_indices]
            X_test = X_filtered[test_indices]
            
            # Handle y indexing properly whether it's pandas Series or numpy array
            if isinstance(y, pd.Series):
                y_train = y[train_mask].values
                y_test = y[test_mask].values
            else:
                y_train = y[train_indices]
                y_test = y[test_indices]
            
            logger.info(f"Training set: {X_train.shape}")
            logger.info(f"Test set: {X_test.shape}")
            logger.info(f"Training target distribution:\n{pd.Series(y_train).value_counts()}")
            logger.info(f"Test target distribution:\n{pd.Series(y_test).value_counts()}")
            
        else:
            logger.info(f"Step 2: Splitting into train/test sets (test_size={test_size})...")
            X_train, X_test, y_train, y_test = train_test_split(
                X_filtered, y, test_size=test_size, 
                random_state=args.random_state, stratify=y
            )
            
            logger.info(f"Training set: {X_train.shape}")
            logger.info(f"Test set: {X_test.shape}")
            logger.info(f"Training target distribution:\n{pd.Series(y_train).value_counts()}")
            logger.info(f"Test target distribution:\n{pd.Series(y_test).value_counts()}")
        
        # Step 3: Feature selection on TRAINING DATA ONLY
        logger.info(f"Step 3: Running feature selection on training data only...")
        
        # Add command line argument for hyperparameter tuning
        use_hyperparameter_tuning = args.hyperparameter_tuning if hasattr(args, 'hyperparameter_tuning') else True
        
        if use_hyperparameter_tuning:
            logger.info("Hyperparameter tuning enabled - this may take longer but will provide better results")
        else:
            logger.info("Hyperparameter tuning disabled - faster execution with default parameters")
        
        selector = FeatureSelector(random_state=args.random_state, use_hyperparameter_tuning=use_hyperparameter_tuning)
        
        # Use all available methods (excluding variance threshold)
        all_methods = ['univariate_f_test', 'mutual_information',
                      'random_forest_importance', 'rfe_random_forest', 'lasso_regularization',
                      'gradient_boosting_importance']
        
        X_selected = selector.fit_transform(
            X_train,
            y_train,
            methods=all_methods,
            n_features=n_features,
            cv_folds=5,
            use_hyperparameter_tuning=use_hyperparameter_tuning
        )
        
        # Step 4: Create comprehensive outputs
        logger.info("Step 4: Creating outputs...")
        
        # Save all features from all methods to CSV
        all_features_df = pd.DataFrame()
        for method in selector.method_features:
            method_features = selector.method_features[method]
            method_df = pd.DataFrame({
                'method': method,
                'feature': method_features,
                'rank': range(1, len(method_features) + 1)
            })
            all_features_df = pd.concat([all_features_df, method_df], ignore_index=True)
        
        all_features_output = output_dir / f"all_methods_top_{n_features}_features.csv"
        all_features_df.to_csv(all_features_output, index=False)
        logger.info(f"All methods features saved to: {all_features_output}")
        
        # Save best method features as a separate CSV for ML modeling
        best_method = selector.best_method
        best_features = selector.method_features[best_method]
        
        best_features_df = pd.DataFrame({
            'feature': best_features,
            'rank': range(1, len(best_features) + 1),
            'method': best_method
        })
        
        best_features_output = output_dir / f"best_method_{best_method}_features.csv"
        best_features_df.to_csv(best_features_output, index=False)
        logger.info(f"Best method features saved to: {best_features_output}")
        
        # Create combined dataset with top features from all methods (deduplicated)
        logger.info("Creating combined dataset with top features from all methods...")
        all_combined_features = set()
        
        # Collect top features from all methods and deduplicate
        for method, features in selector.method_features.items():
            # Take all features from each method (already filtered to requested n_features)
            all_combined_features.update(features)
        
        # Convert to sorted list for consistency
        all_combined_features = sorted(list(all_combined_features))
        
        logger.info(f"Combined features: {len(all_combined_features)} unique features from all methods (deduplicated)")
        logger.info(f"Feature breakdown by method:")
        for method, features in selector.method_features.items():
            logger.info(f"  - {method}: {len(features)} features")
        
        # Create combined train and test datasets with all top features from all methods
        logger.info("Creating ML-ready combined train and test datasets...")
        X_train_combined = X_train[all_combined_features]
        X_test_combined = X_test[all_combined_features]
        
        # Save combined features list
        combined_features_df = pd.DataFrame({
            'feature': all_combined_features,
            'rank': range(1, len(all_combined_features) + 1)
        })
        
        combined_features_output = output_dir / f"combined_all_methods_features.csv"
        combined_features_df.to_csv(combined_features_output, index=False)
        logger.info(f"Combined features list saved to: {combined_features_output}")
        
        # Save combined ML-ready datasets with top features from all methods
        combined_train_df = pd.concat([X_train_combined, pd.Series(y_train, name=target, index=X_train_combined.index)], axis=1)
        combined_test_df = pd.concat([X_test_combined, pd.Series(y_test, name=target, index=X_test_combined.index)], axis=1)
        
        # Use clearer naming for combined datasets
        combined_train_output = output_dir / f"ml_ready_combined_train.csv"
        combined_test_output = output_dir / f"ml_ready_combined_test.csv"
        
        combined_train_df.to_csv(combined_train_output, index=False)
        combined_test_df.to_csv(combined_test_output, index=False)
        
        logger.info(f"ML-ready combined training data (top features from all methods) saved to: {combined_train_output}")
        logger.info(f"ML-ready combined test data (top features from all methods) saved to: {combined_test_output}")
        
        # Create feature method mapping for reference
        feature_method_mapping = []
        for method, features in selector.method_features.items():
            for feature in features:
                feature_method_mapping.append({
                    'feature': feature,
                    'method': method,
                    'method_rank': features.index(feature) + 1,
                    'method_score': selector.method_scores[method]
                })
        
        feature_mapping_df = pd.DataFrame(feature_method_mapping)
        feature_mapping_output = output_dir / "feature_method_mapping.csv"
        feature_mapping_df.to_csv(feature_mapping_output, index=False)
        logger.info(f"Feature-method mapping saved to: {feature_mapping_output}")
        
        # Create ML-ready dataset with best features (original functionality)
        X_train_best = X_train[best_features]
        X_test_best = X_test[best_features]
        
        # Save train and test sets with selected features
        train_df = pd.concat([X_train_best, pd.Series(y_train, name=target, index=X_train_best.index)], axis=1)
        test_df = pd.concat([X_test_best, pd.Series(y_test, name=target, index=X_test_best.index)], axis=1)
        
        train_output = output_dir / "ml_ready_train_data.csv"
        test_output = output_dir / "ml_ready_test_data.csv"
        
        train_df.to_csv(train_output, index=False)
        test_df.to_csv(test_output, index=False)
        
        logger.info(f"ML-ready training data saved to: {train_output}")
        logger.info(f"ML-ready test data saved to: {test_output}")
        
        # Step 5: Create visualizations
        logger.info("Step 5: Creating visualizations...")
        create_visualizations(selector, output_dir, X_train, y_train)
        
        # Step 6: Evaluate best method on test set
        logger.info("Step 6: Evaluating best method on test set...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=args.random_state)
        rf_model.fit(X_train_best, y_train)
        y_pred = rf_model.predict(X_test_best)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Test set accuracy with {best_method}: {test_accuracy:.4f}")
        
        # Hyperparameter tuning for the best method
        logger.info("Hyperparameter tuning for the best method...")
        if best_method == 'random_forest_importance':
            # Define hyperparameter grid for Random Forest
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            # Optimize hyperparameters
            best_rf_model = optimize_model_hyperparameters(rf_model, X_train_best, y_train, param_grid, cv=5, scoring='accuracy')
            
            # Evaluate on test set
            y_pred_rf = best_rf_model.predict(X_test_best)
            test_accuracy_rf = accuracy_score(y_test, y_pred_rf)
            logger.info(f"Optimized Random Forest test set accuracy: {test_accuracy_rf:.4f}")
        
        # Step 7: Save comprehensive summary
        logger.info("Step 7: Saving comprehensive summary...")
        
        summary = {
            'analysis_info': {
                'input_file': str(data_file),
                'target_column': target,
                'test_size': test_size if not use_predefined_split else 'N/A (predefined split)',
                'use_predefined_split': use_predefined_split,
                'category_column': args.category_column,
                'random_state': args.random_state,
                'runtime_seconds': time.time() - start_time,
                'runtime_formatted': format_duration(time.time() - start_time)
            },
            'data_info': {
                'original_features': len(X.columns),
                'after_variance_filtering': len(X_filtered.columns),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'train_target_distribution': pd.Series(y_train).value_counts().to_dict(),
                'test_target_distribution': pd.Series(y_test).value_counts().to_dict()
            },
            'feature_selection_results': {
                'methods_used': list(selector.method_features.keys()),
                'method_scores': selector.method_scores,
                'method_accuracy_scores': selector.method_accuracy_scores,
                'best_method': best_method,
                'best_score': selector.method_scores[best_method],
                'best_accuracy': selector.method_accuracy_scores[best_method],
                'features_selected_per_method': {method: len(features) 
                                               for method, features in selector.method_features.items()},
                'combined_features_count': len(all_combined_features),
                'test_accuracy_best_method': test_accuracy
            },
            'output_files': {
                'all_methods_features': str(all_features_output),
                'best_method_features': str(best_features_output),
                'combined_features_list': str(combined_features_output),
                'feature_method_mapping': str(feature_mapping_output),
                'ml_ready_train_data': str(train_output),
                'ml_ready_test_data': str(test_output),
                'combined_train_data': str(combined_train_output),
                'combined_test_data': str(combined_test_output),
                'exploratory_analysis': str(output_dir / 'exploratory_analysis.png'),
                'performance_visualization': str(output_dir / 'feature_selection_performance.png'),
                'best_method_analysis': str(output_dir / 'best_method_analysis.png')
            }
        }
        
        save_json(summary, output_dir / "comprehensive_analysis_summary.json")
        
        # Save detailed results from selector
        selector.save_results(str(output_dir))
        
        # Step 8: Print final summary
        logger.info("=" * 60)
        logger.info("FEATURE SELECTION ANALYSIS COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Dataset: {Path(data_file).name}")
        logger.info(f"Target: {target}")
        logger.info(f"Original features: {len(X.columns)}")
        logger.info(f"After variance filtering: {len(X_filtered.columns)}")
        logger.info(f"Train/Test split: {len(X_train)}/{len(X_test)} samples")
        logger.info(f"Split method: {'Predefined (Category column)' if use_predefined_split else 'Random'}")
        logger.info("")
        logger.info("FEATURE SELECTION METHODS PERFORMANCE:")
        logger.info("-" * 50)
        
        # Sort methods by performance
        sorted_methods = sorted(selector.method_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (method, score) in enumerate(sorted_methods, 1):
            accuracy = selector.method_accuracy_scores[method]
            star = " ⭐" if method == best_method else ""
            logger.info(f"  {i}. {method.replace('_', ' ').title()}: AUC {score:.4f}, Accuracy {accuracy:.4f}{star}")
        
        logger.info("")
        logger.info(f"🏆 BEST METHOD: {best_method.replace('_', ' ').title()}")
        logger.info(f"🎯 BEST CV AUC SCORE: {selector.method_scores[best_method]:.4f}")
        logger.info(f"🎯 BEST CV ACCURACY: {selector.method_accuracy_scores[best_method]:.4f}")
        logger.info(f"🧪 TEST ACCURACY: {test_accuracy:.4f}")
        logger.info(f"📊 FEATURES SELECTED: {len(best_features)}")
        logger.info("")
        logger.info("TOP 10 SELECTED FEATURES:")
        logger.info("-" * 30)
        for i, feature in enumerate(best_features[:10], 1):
            logger.info(f"  {i:2d}. {feature}")
        
        logger.info("")
        logger.info("OUTPUT FILES:")
        logger.info("-" * 20)
        logger.info(f"📁 Results directory: {output_dir}")
        logger.info(f"📋 All methods features: {all_features_output.name}")
        logger.info(f"🏆 Best method features: {best_features_output.name}")
        logger.info(f"🔗 Combined features list: {combined_features_output.name}")
        logger.info(f"🗺️  Feature-method mapping: {feature_mapping_output.name}")
        logger.info(f"🤖 ML-ready train data (best): {train_output.name}")
        logger.info(f"🧪 ML-ready test data (best): {test_output.name}")
        logger.info(f"� Combined train data (all methods): {combined_train_output.name}")
        logger.info(f"🔄 Combined test data (all methods): {combined_test_output.name}")
        logger.info(f"🔍 Exploratory analysis: exploratory_analysis.png")
        logger.info(f"�📊 Performance plots: feature_selection_performance.png")
        logger.info(f"🏆 Best method analysis: best_method_analysis.png")
        logger.info(f"📈 Complete summary: comprehensive_analysis_summary.json")
        
        logger.info("")
        logger.info(f"📈 COMBINED DATASET SUMMARY:")
        logger.info(f"  • Total unique features: {len(all_combined_features)}")
        logger.info(f"  • Best method features: {len(best_features)}")
        logger.info(f"  • Feature reduction: {len(all_combined_features)} → {len(best_features)}")
        logger.info("")
        logger.info(f"⏱️  Total runtime: {format_duration(time.time() - start_time)}")
        logger.info("=" * 60)
        
        # Display final message to user
        print(f"\n{'='*60}")
        print("🎉 FEATURE SELECTION ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"✅ Best method: {best_method.replace('_', ' ').title()}")
        print(f"✅ CV AUC Score: {selector.method_scores[best_method]:.4f}")
        print(f"✅ CV Accuracy: {selector.method_accuracy_scores[best_method]:.4f}")
        print(f"✅ Test Accuracy: {test_accuracy:.4f}")
        print(f"✅ Features selected (best): {len(best_features)}")
        print(f"✅ Combined features (all methods): {len(all_combined_features)}")
        print(f"✅ Results saved to: {output_dir}")
        print(f"{'='*60}")
        
        return {
            'best_method': best_method,
            'best_features': best_features,
            'combined_features': all_combined_features,
            'train_data': X_train_best,
            'test_data': X_test_best,
            'combined_train_data': X_train_combined,
            'combined_test_data': X_test_combined,
            'train_target': y_train,
            'test_target': y_test,
            'test_accuracy': test_accuracy
        }
        
    except Exception as e:
        logger.error(f"Error in feature selection analysis: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
