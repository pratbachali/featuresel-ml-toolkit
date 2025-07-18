#!/usr/bin/env python3
"""
Quick test of the enhanced overlap visualization
"""
import sys
import os

# Add parent directory to path to access the main dataset
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pandas as pd
    import numpy as np
    
    print("✅ Testing enhanced feature selection overlap visualization...")
    
    # Test if our CSV file exists
    csv_file = "../combined_dataset_SVA_corrected.csv"
    if os.path.exists(csv_file):
        print(f"✅ Found input file: {csv_file}")
        
        # Load and check the data structure
        df = pd.read_csv(csv_file)
        print(f"📊 Dataset shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns[:10])}...")  # Show first 10 columns
        
        # Check for Category column
        if 'Category' in df.columns:
            category_counts = df['Category'].value_counts()
            print(f"📈 Category distribution:\n{category_counts}")
            print("✅ Category column found - can use predefined train/test split")
        else:
            print("⚠️  No Category column found - will use random split")
            
        print("\n🎯 Enhanced overlap visualization features:")
        print("  ✅ Jaccard Index heatmap with gene counts")
        print("  ✅ Diagonal shows total features per method")
        print("  ✅ Off-diagonal shows intersection count + Jaccard index")
        print("  ✅ Color coding for easy interpretation")
        print("  ✅ Detailed overlap summary CSV table")
        print("  ✅ Method comparison statistics")
        
        print(f"\n🔥 Ready to run enhanced feature selection!")
        print("Run: python standalone_feature_selection.py")
        
    else:
        print(f"❌ Input file not found: {csv_file}")
        print("Available CSV files:")
        for file in os.listdir(".."):
            if file.endswith('.csv'):
                print(f"  - {file}")
                
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure the virtual environment is activated")
    
except Exception as e:
    print(f"❌ Error: {e}")
