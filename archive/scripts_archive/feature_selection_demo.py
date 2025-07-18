#!/usr/bin/env python3
"""
Demo Script for Enhanced Feature Selection Pipeline

This script demonstrates how to use the enhanced feature selection pipeline
with proper train/test separation and comprehensive outputs.
"""

import os
import sys
from pathlib import Path

# Add the new_modular_pipeline to path
pipeline_dir = Path(__file__).parent
sys.path.append(str(pipeline_dir))

def demo_interactive_feature_selection():
    """Demonstrate interactive feature selection"""
    print("=" * 60)
    print("ENHANCED FEATURE SELECTION PIPELINE DEMO")
    print("=" * 60)
    
    print("\n📋 This enhanced pipeline provides:")
    print("  ✅ Interactive prompts for input data and parameters")
    print("  ✅ Proper train/test separation (variance filtering → split → feature selection)")
    print("  ✅ Multiple feature selection methods with performance comparison")
    print("  ✅ Comprehensive visualizations and analysis")
    print("  ✅ CSV outputs with top N features from all methods")
    print("  ✅ ML-ready datasets for immediate modeling")
    print("  ✅ Best method identification and detailed analysis")
    
    print("\n📊 Available Feature Selection Methods:")
    methods = [
        'variance_threshold', 'univariate_f_test', 'mutual_information',
        'random_forest_importance', 'rfe_random_forest', 'lasso_regularization',
        'gradient_boosting_importance'
    ]
    for i, method in enumerate(methods, 1):
        print(f"  {i}. {method.replace('_', ' ').title()}")
    
    print("\n📁 Pipeline Outputs:")
    outputs = [
        "all_methods_top_N_features.csv - Features from all methods",
        "best_method_features.csv - Features from best performing method",
        "ml_ready_train_data.csv - Training data with selected features",
        "ml_ready_test_data.csv - Test data with selected features", 
        "feature_selection_performance.png - Method comparison visualization",
        "best_method_analysis.png - Detailed analysis of best method",
        "comprehensive_analysis_summary.json - Complete analysis summary",
        "feature_selection.log - Detailed execution log"
    ]
    for output in outputs:
        print(f"  📄 {output}")
    
    print("\n🚀 Usage Examples:")
    print("\n1. Interactive Mode (Recommended):")
    print("   python scripts/run_feature_selection.py --interactive")
    
    print("\n2. Command Line Mode:")
    print("   python scripts/run_feature_selection.py \\")
    print("     --data_file your_data.csv \\")
    print("     --target target_column \\")
    print("     --n_features 100 \\")
    print("     --test_size 0.2 \\")
    print("     --output_dir results/")
    
    print("\n3. For Your Specific Dataset:")
    print("   # Your workflow: 85 samples, 18,646 genes, 34 test patients")
    print("   python scripts/run_feature_selection.py \\")
    print("     --data_file your_genomic_data.csv \\")
    print("     --target insulin_resistance \\")
    print("     --n_features 100 \\")
    print("     --test_size 0.4 \\")  # 34/85 ≈ 0.4
    print("     --top_variance_genes 1000 \\")
    print("     --output_dir ml_results/")
    
    print("\n🔬 Methodology:")
    print("  1. Load genomic dataset")
    print("  2. Filter top 1000 variance genes (entire dataset)")
    print("  3. Split into train/test sets (51 train, 34 test)")
    print("  4. Run feature selection ONLY on training data")
    print("  5. Compare methods using cross-validation on training data")
    print("  6. Identify best performing method")
    print("  7. Create visualizations and comprehensive outputs")
    print("  8. Prepare ML-ready datasets with selected features")
    
    print("\n💡 Key Advantages:")
    print("  🎯 No data leakage - test data never used for feature selection")
    print("  📊 Visual comparison of all methods")
    print("  🏆 Automatic best method identification")
    print("  📈 Ready-to-use datasets for ML modeling")
    print("  🔍 Comprehensive logging and analysis")
    
    print("\n" + "=" * 60)

def run_sample_analysis():
    """Run a sample analysis if data is available"""
    print("\n🔍 Checking for sample data...")
    
    # Look for any CSV files in the workspace
    workspace_dir = Path(__file__).parent.parent
    csv_files = list(workspace_dir.glob("*.csv"))
    
    if csv_files:
        print(f"Found {len(csv_files)} CSV files in workspace:")
        for i, csv_file in enumerate(csv_files[:5], 1):  # Show first 5
            print(f"  {i}. {csv_file.name}")
        
        print("\n💡 To run analysis on any of these files:")
        print("   cd new_modular_pipeline")
        print("   python scripts/run_feature_selection.py --interactive")
        
    else:
        print("No CSV files found in workspace.")
        print("To test the pipeline, you can:")
        print("1. Add a CSV file to the workspace")
        print("2. Run: python scripts/run_feature_selection.py --interactive")
    
def main():
    """Main demo function"""
    demo_interactive_feature_selection()
    run_sample_analysis()
    
    print(f"\n📁 Current directory: {os.getcwd()}")
    print(f"📁 Pipeline directory: {Path(__file__).parent}")
    print("\nReady to run feature selection! 🚀")

if __name__ == "__main__":
    main()
