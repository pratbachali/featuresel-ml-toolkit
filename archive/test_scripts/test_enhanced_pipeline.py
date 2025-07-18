#!/usr/bin/env python3
"""
Test Enhanced Feature Selection Pipeline

Quick test to verify the enhanced pipeline functionality.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def create_test_data():
    """Create a synthetic genomic-like dataset for testing"""
    print("Creating synthetic test data...")
    
    # Create synthetic data similar to genomic data
    # 85 samples, 1000 features (simulating after variance filtering)
    X, y = make_classification(
        n_samples=85,
        n_features=1000,
        n_informative=50,
        n_redundant=10,
        n_classes=2,
        random_state=42,
        flip_y=0.05  # 5% label noise
    )
    
    # Create feature names like gene names
    feature_names = [f"GENE_{i:04d}" for i in range(1000)]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['insulin_resistance'] = y
    
    # Save test data
    test_file = current_dir / "test_genomic_data.csv"
    df.to_csv(test_file, index=False)
    
    print(f"✅ Test data created: {test_file}")
    print(f"   Shape: {df.shape}")
    print(f"   Target distribution: {df['insulin_resistance'].value_counts().to_dict()}")
    
    return test_file

def test_pipeline_components():
    """Test individual pipeline components"""
    print("\n🧪 Testing pipeline components...")
    
    try:
        # Test imports
        from src.data.loader import DataLoader
        from src.features.selector import FeatureSelector
        from src.features.variance_filter import VarianceFilter
        from src.utils.logger import setup_logging, get_logger
        from src.utils.helpers import ensure_directory
        
        print("✅ All imports successful")
        
        # Test basic functionality
        test_file = create_test_data()
        
        # Test data loading
        df = pd.read_csv(test_file)
        X = df.drop(columns=['insulin_resistance'])
        y = df['insulin_resistance']
        
        print(f"✅ Data loading successful: {X.shape}")
        
        # Test variance filter
        variance_filter = VarianceFilter()
        top_features = variance_filter.select_top_variance_features(X, 100)
        print(f"✅ Variance filtering successful: {len(top_features)} features selected")
        
        # Test feature selector initialization
        selector = FeatureSelector()
        print("✅ Feature selector initialization successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False

def show_usage_instructions():
    """Show usage instructions"""
    print("\n" + "="*60)
    print("📖 HOW TO USE THE ENHANCED FEATURE SELECTION PIPELINE")
    print("="*60)
    
    print("\n🚀 Quick Start:")
    print("1. Navigate to the pipeline directory:")
    print("   cd new_modular_pipeline")
    
    print("\n2. Run interactive mode:")
    print("   python scripts/run_feature_selection.py --interactive")
    
    print("\n3. Or test with synthetic data:")
    print("   python feature_selection_demo.py")
    
    print("\n📊 For your specific genomic data:")
    print("   python scripts/run_feature_selection.py \\")
    print("     --data_file /path/to/your/genomic_data.csv \\")
    print("     --target insulin_resistance \\")
    print("     --n_features 100 \\")
    print("     --test_size 0.4 \\")
    print("     --output_dir results/")
    
    print("\n📁 Expected outputs in results/ directory:")
    outputs = [
        "all_methods_top_100_features.csv",
        "best_method_[method_name]_features.csv", 
        "ml_ready_train_data.csv",
        "ml_ready_test_data.csv",
        "feature_selection_performance.png",
        "best_method_analysis.png",
        "comprehensive_analysis_summary.json"
    ]
    for output in outputs:
        print(f"  📄 {output}")

def main():
    """Main test function"""
    print("="*60)
    print("🧪 TESTING ENHANCED FEATURE SELECTION PIPELINE")
    print("="*60)
    
    # Test components
    success = test_pipeline_components()
    
    if success:
        print("\n✅ All tests passed!")
        show_usage_instructions()
    else:
        print("\n❌ Some tests failed. Please check the setup.")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
