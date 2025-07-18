#!/usr/bin/env python3
"""
Feature Selection Pipeline Demo

This script demonstrates how to use the robust feature selection pipeline
for genomic data analysis. It shows all the key features and outputs.

Author: Pipeline Development Team
Date: 2024
"""

import sys
import os
import subprocess
from pathlib import Path

# Add the src directory to the path so we can import modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from scripts.run_feature_selection import main as run_feature_selection
import pandas as pd
import numpy as np


def create_demo_dataset():
    """Create a synthetic genomic dataset for demonstration."""
    print("🧬 Creating synthetic genomic dataset for demonstration...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create synthetic gene expression data
    n_samples = 100
    n_genes = 50
    
    # Generate gene names (similar to real genomic data)
    gene_names = [
        f"GENE_{i:03d}" for i in range(1, n_genes + 1)
    ]
    
    # Create different expression patterns for IR vs IS
    # Some genes will be differentially expressed
    X = np.random.randn(n_samples, n_genes)
    
    # Make some genes more discriminative
    discriminative_genes = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    
    # Create IR/IS target (50-50 split)
    y = np.array(['IR'] * 50 + ['IS'] * 50)
    
    # Make discriminative genes actually discriminative
    for i, gene_idx in enumerate(discriminative_genes):
        # IR samples have higher expression for even indices, lower for odd
        if i % 2 == 0:
            X[:50, gene_idx] += 2  # IR higher
        else:
            X[:50, gene_idx] -= 2  # IR lower
    
    # Create dataset categories for predefined split demo
    categories = np.array(['Dataset_A'] * 30 + ['Dataset_B'] * 30 + ['Dataset_C'] * 40)
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=gene_names)
    df['IRIS'] = y
    df['Category'] = categories
    
    # Save to CSV
    output_file = "demo_genomic_data.csv"
    df.to_csv(output_file, index=False)
    
    print(f"✅ Created {output_file} with:")
    print(f"   - {n_samples} samples")
    print(f"   - {n_genes} genes")
    print(f"   - {len(discriminative_genes)} discriminative genes")
    print(f"   - Target: IRIS (IR/IS)")
    print(f"   - Categories: {np.unique(categories)}")
    print(f"   - IR samples: {np.sum(y == 'IR')}")
    print(f"   - IS samples: {np.sum(y == 'IS')}")
    
    return output_file


def demo_random_split():
    """Demonstrate the pipeline with random train/test split."""
    print("\n" + "="*70)
    print("🎲 DEMO 1: Random Train/Test Split")
    print("="*70)
    
    # Create demo dataset
    data_file = create_demo_dataset()
    
    # Simulate user inputs (in real usage, these would be prompted)
    print("\n📋 Pipeline Configuration:")
    print("   - Data file: demo_genomic_data.csv")
    print("   - Target variable: IRIS")
    print("   - Variance genes: 30")
    print("   - Split type: Random (80/20)")
    print("   - Top features per method: 10")
    print("   - Output directory: demo_results_random")
    
    # Mock user inputs by setting them in the environment or using the script directly
    import subprocess
    
    # Create a temporary input file to simulate user responses
    user_inputs = [
        data_file,                          # Data file
        "IRIS",                            # Target variable
        "30",                              # Number of variance genes
        "random",                          # Split type
        "10",                              # Top features per method
        "demo_results_random"              # Output directory
    ]
    
    # Write inputs to a temporary file
    with open("temp_inputs.txt", "w") as f:
        for inp in user_inputs:
            f.write(inp + "\n")
    
    print("\n🚀 Running feature selection pipeline...")
    try:
        # Run the pipeline with inputs
        result = subprocess.run([
            sys.executable, 
            "scripts/run_feature_selection.py"
        ], 
        input="\n".join(user_inputs), 
        text=True, 
        capture_output=True, 
        timeout=300
        )
        
        if result.returncode == 0:
            print("✅ Pipeline completed successfully!")
            print("\n📊 Check the demo_results_random/ directory for outputs")
        else:
            print(f"❌ Pipeline failed with error: {result.stderr}")
    
    except subprocess.TimeoutExpired:
        print("⏰ Pipeline timed out after 5 minutes")
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
    
    # Clean up
    if os.path.exists("temp_inputs.txt"):
        os.remove("temp_inputs.txt")


def demo_predefined_split():
    """Demonstrate the pipeline with predefined train/test split."""
    print("\n" + "="*70)
    print("📊 DEMO 2: Predefined Train/Test Split")
    print("="*70)
    
    # Use the same demo dataset
    data_file = "demo_genomic_data.csv"
    
    print("\n📋 Pipeline Configuration:")
    print("   - Data file: demo_genomic_data.csv")
    print("   - Target variable: IRIS")
    print("   - Variance genes: 40")
    print("   - Split type: Predefined (Category column)")
    print("   - Top features per method: 15")
    print("   - Output directory: demo_results_predefined")
    
    # User inputs for predefined split
    user_inputs = [
        data_file,                          # Data file
        "IRIS",                            # Target variable
        "40",                              # Number of variance genes
        "predefined",                      # Split type
        "15",                              # Top features per method
        "demo_results_predefined"          # Output directory
    ]
    
    print("\n🚀 Running feature selection pipeline...")
    try:
        # Run the pipeline with inputs
        result = subprocess.run([
            sys.executable, 
            "scripts/run_feature_selection.py"
        ], 
        input="\n".join(user_inputs), 
        text=True, 
        capture_output=True, 
        timeout=300
        )
        
        if result.returncode == 0:
            print("✅ Pipeline completed successfully!")
            print("\n📊 Check the demo_results_predefined/ directory for outputs")
        else:
            print(f"❌ Pipeline failed with error: {result.stderr}")
    
    except subprocess.TimeoutExpired:
        print("⏰ Pipeline timed out after 5 minutes")
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")


def explain_outputs():
    """Explain the outputs generated by the pipeline."""
    print("\n" + "="*70)
    print("📁 UNDERSTANDING PIPELINE OUTPUTS")
    print("="*70)
    
    outputs_explanation = """
🎯 KEY OUTPUT FILES:

1. 📊 FEATURE SELECTION RESULTS:
   • all_methods_top_N_features.csv - Top N features from each method
   • best_method_<method>_features.csv - Features from the best performing method
   • combined_all_methods_features.csv - Deduplicated features from all methods
   • feature_method_mapping.csv - Which method selected each feature

2. 🤖 ML-READY DATASETS:
   • ml_ready_train_data.csv - Training data with BEST method features
   • ml_ready_test_data.csv - Test data with BEST method features
   • ml_ready_combined_train.csv - Training data with ALL methods features
   • ml_ready_combined_test.csv - Test data with ALL methods features

3. 📈 VISUALIZATIONS:
   • exploratory_data_analysis.png - Bar plots (IR/IS, by dataset) and PCA plots
   • feature_selection_performance.png - Method performance comparison
   • best_method_analysis.png - Analysis of the best performing method

4. 📋 SUMMARY & LOGS:
   • comprehensive_analysis_summary.json - Complete analysis summary
   • feature_selection.log - Detailed execution log
   • method_evaluation_scores.csv - Cross-validation scores for each method

🔍 FEATURE SELECTION METHODS USED:
   • Univariate F-test: Statistical significance testing
   • Mutual Information: Non-linear dependency detection
   • Random Forest Importance: Tree-based feature importance
   • RFE Random Forest: Recursive feature elimination
   • Lasso Regularization: L1 regularization for feature selection
   • Gradient Boosting Importance: Boosting-based feature importance

🎯 BEST METHOD SELECTION:
   • Methods are ranked by cross-validation AUC score
   • The best method is used to create the primary ML-ready datasets
   • Combined datasets include features from all methods (deduplicated)

📊 EXPLORATORY ANALYSIS:
   • Bar plots show target distribution overall and by dataset
   • PCA plots show data separation and potential batch effects
   • Helps identify data quality issues before feature selection
    """
    
    print(outputs_explanation)


def main():
    """Main demo function."""
    print("🧬 GENOMIC FEATURE SELECTION PIPELINE DEMO")
    print("="*60)
    print("""
This demo shows how to use the robust feature selection pipeline for genomic data.
The pipeline includes:
- Variance-based pre-filtering
- Multiple feature selection methods
- Cross-validation evaluation
- Comprehensive outputs and visualizations
- Support for both random and predefined train/test splits
    """)
    
    try:
        # Demo 1: Random split
        demo_random_split()
        
        # Demo 2: Predefined split
        demo_predefined_split()
        
        # Explain outputs
        explain_outputs()
        
        print("\n" + "="*70)
        print("🎉 DEMO COMPLETED!")
        print("="*70)
        print("""
Next steps:
1. Check the demo_results_random/ and demo_results_predefined/ directories
2. Examine the ML-ready datasets for your downstream analysis
3. Review the comprehensive_analysis_summary.json for detailed results
4. Use the best method features or combined features for your ML models

To run with your own data:
python scripts/run_feature_selection.py

The pipeline will prompt you for all necessary inputs.
        """)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
