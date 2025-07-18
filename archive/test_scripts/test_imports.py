#!/usr/bin/env python3
"""
Test script to check if imports work correctly
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

print("Basic imports successful")

# Test module imports
script_dir = Path(__file__).parent
src_dir = script_dir / "src"
print(f"Script dir: {script_dir}")
print(f"Src dir: {src_dir}")
print(f"Src exists: {src_dir.exists()}")

if src_dir.exists():
    print(f"Src contents: {list(src_dir.iterdir())}")
    
    # Try to import using sys.path
    sys.path.insert(0, str(src_dir))
    
    try:
        print("Attempting imports...")
        
        # Check if we can read the files
        loader_file = src_dir / "data" / "loader.py"
        print(f"Loader file exists: {loader_file.exists()}")
        
        if loader_file.exists():
            with open(loader_file, 'r') as f:
                content = f.read()
                print(f"Loader file size: {len(content)} characters")
        
        # Try simple import
        from data.loader import DataLoader
        print("✓ DataLoader imported successfully")
        
        from features.selector import FeatureSelector
        print("✓ FeatureSelector imported successfully")
        
        from features.variance_filter import VarianceFilter
        print("✓ VarianceFilter imported successfully")
        
        from utils.logger import setup_logging, get_logger
        print("✓ Logger modules imported successfully")
        
        from utils.helpers import ensure_directory, save_json, format_duration
        print("✓ Helper modules imported successfully")
        
        print("All imports successful!")
        
    except Exception as e:
        print(f"Import error: {e}")
        import traceback
        traceback.print_exc()
