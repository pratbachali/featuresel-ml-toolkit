#!/usr/bin/env python3
"""
Run feature selection on combined_dataset_SVA_corrected.csv
This script provides all inputs automatically (non-interactive)
"""

import sys
import os

# Add the current directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_feature_selection_automated():
    """Run feature selection with predefined parameters"""
    
    # Import the function from standalone_feature_selection
    try:
        from standalone_feature_selection import run_enhanced_feature_selection
        
        # Parameters for the run
        input_file = "../combined_dataset_SVA_corrected.csv"
        target_column = "Category"  # This will be used for train/test split
        output_dir = "feature_selection_results"
        n_features = 100
        test_size = 0.3  # Won't be used since we're using Category column
        
        print("🚀 Running Enhanced Feature Selection Pipeline")
        print("=" * 60)
        print(f"Input file: {input_file}")
        print(f"Target column: {target_column}")
        print(f"Number of features to select: {n_features}")
        print(f"Output directory: {output_dir}")
        print()
        
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"❌ Error: Input file '{input_file}' not found!")
            print("Available CSV files:")
            for file in os.listdir(".."):
                if file.endswith('.csv'):
                    print(f"  - {file}")
            return
        
        # Run the feature selection
        run_enhanced_feature_selection(
            input_file=input_file,
            target_column=target_column,
            output_dir=output_dir,
            n_features=n_features,
            test_size=test_size
        )
        
        print("\n✅ Feature selection completed successfully!")
        print(f"Results saved in: {output_dir}/")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required packages are installed:")
        print("  pip install pandas numpy matplotlib seaborn scikit-learn")
    except Exception as e:
        print(f"❌ Error during feature selection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_feature_selection_automated()
