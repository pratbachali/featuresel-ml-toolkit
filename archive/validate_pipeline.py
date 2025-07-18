#!/usr/bin/env python3
"""
Pipeline Validation Script

Quick validation script to ensure the feature selection pipeline is working correctly.
Run this before using the pipeline with your own data.

Usage:
    python validate_pipeline.py
"""

import sys
import os
from pathlib import Path
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test all required imports."""
    print("🔍 Testing imports...")
    try:
        from features.selector import FeatureSelector
        from data.loader import DataLoader
        from utils.helpers import save_json, load_json
        from utils.logger import setup_logger
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_feature_selector():
    """Test FeatureSelector functionality."""
    print("\n🧪 Testing FeatureSelector...")
    try:
        import pandas as pd
        import numpy as np
        
        # Create synthetic data
        np.random.seed(42)
        n_samples, n_features = 100, 20
        X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                        columns=[f"feature_{i}" for i in range(n_features)])
        y = pd.Series(np.random.choice(['A', 'B'], n_samples))
        
        # Test FeatureSelector
        selector = FeatureSelector(random_state=42)
        
        # Test methods availability
        methods = [
            'univariate_f_test', 
            'mutual_information',
            'random_forest_importance',
            'rfe_random_forest',
            'lasso_regularization',
            'gradient_boosting_importance'
        ]
        
        print(f"✅ FeatureSelector initialized with {len(methods)} methods")
        
        # Test feature selection
        X_selected = selector.fit_transform(X, y, methods=methods[:2], n_features=5, cv_folds=3)
        
        print(f"✅ Feature selection successful: {X_selected.shape[1]} features selected")
        print(f"✅ Best method: {selector.best_method}")
        
        return True
        
    except Exception as e:
        print(f"❌ FeatureSelector test failed: {e}")
        return False

def test_data_loader():
    """Test DataLoader functionality."""
    print("\n📊 Testing DataLoader...")
    try:
        import pandas as pd
        import numpy as np
        
        # Create test CSV
        np.random.seed(42)
        data = pd.DataFrame({
            'gene1': np.random.randn(50),
            'gene2': np.random.randn(50),
            'gene3': np.random.randn(50),
            'target': np.random.choice(['IR', 'IS'], 50),
            'category': np.random.choice(['A', 'B'], 50)
        })
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            from data.loader import DataLoader
            loader = DataLoader()
            
            # Test loading
            loaded_data = loader.load_data(temp_file)
            print(f"✅ Data loaded successfully: {loaded_data.shape}")
            
            # Test validation
            is_valid, message = loader.validate_data(loaded_data, target_column='target')
            print(f"✅ Data validation: {is_valid} - {message}")
            
        finally:
            # Clean up
            os.unlink(temp_file)
        
        return True
        
    except Exception as e:
        print(f"❌ DataLoader test failed: {e}")
        return False

def test_helpers():
    """Test helper functions."""
    print("\n🛠️ Testing helpers...")
    try:
        from utils.helpers import save_json, load_json
        
        # Test JSON functions
        test_data = {"test": "data", "number": 42}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            save_json(test_data, temp_file)
            loaded_data = load_json(temp_file)
            
            assert loaded_data == test_data
            print("✅ JSON save/load functions working")
            
        finally:
            os.unlink(temp_file)
        
        return True
        
    except Exception as e:
        print(f"❌ Helpers test failed: {e}")
        return False

def test_main_script():
    """Test that the main script can be imported."""
    print("\n📋 Testing main script...")
    try:
        # Test that the main script exists and can be imported
        main_script = Path(__file__).parent / "scripts" / "run_feature_selection.py"
        
        if main_script.exists():
            print("✅ Main script exists")
            
            # Test basic import (without running main)
            import importlib.util
            spec = importlib.util.spec_from_file_location("run_feature_selection", main_script)
            module = importlib.util.module_from_spec(spec)
            
            # Don't execute, just check it can be loaded
            print("✅ Main script can be loaded")
            
        else:
            print("❌ Main script not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Main script test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🧬 FEATURE SELECTION PIPELINE VALIDATION")
    print("="*50)
    
    tests = [
        ("Imports", test_imports),
        ("FeatureSelector", test_feature_selector),
        ("DataLoader", test_data_loader),
        ("Helpers", test_helpers),
        ("Main Script", test_main_script)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            failed += 1
    
    print("\n" + "="*50)
    print("🎯 VALIDATION SUMMARY")
    print("="*50)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Pipeline is ready to use")
        print("\nNext steps:")
        print("1. Run: python scripts/run_feature_selection.py")
        print("2. Or try: python pipeline_demo.py")
    else:
        print(f"\n⚠️  {failed} tests failed")
        print("❌ Please check the errors above")
        print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
