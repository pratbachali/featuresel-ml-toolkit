#!/usr/bin/env python3
"""
Complete ML Pipeline Script

Runs the complete ML pipeline including feature selection and classification.
"""

import sys
import argparse
from pathlib import Path
import time
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def main():
    parser = argparse.ArgumentParser(description='Complete ML Pipeline')
    parser.add_argument('data_file', help='Path to data file (CSV or Excel)')
    parser.add_argument('--target', required=True, help='Target column name')
    parser.add_argument('--test_file', help='Optional test data file')
    parser.add_argument('--n_features', type=int, default=100, 
                       help='Number of features to select (default: 100)')
    parser.add_argument('--top_variance_genes', type=int, default=1000,
                       help='Number of top variance genes to pre-filter (default: 1000)')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--output_dir', default='ml_pipeline_results',
                       help='Output directory (default: ml_pipeline_results)')
    parser.add_argument('--log_level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    print("🚀 COMPLETE ML PIPELINE")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Import modules (only when needed to avoid import errors)
        try:
            from data.loader import DataLoader
            from features.selector import FeatureSelector
            from features.variance_filter import VarianceFilter
            from models.classifier import MLClassifier
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("📋 To run the full pipeline, install dependencies:")
            print("   pip install -r requirements.txt")
            print("\n📁 Current modular structure is ready for:")
            print("   - Data loading and preprocessing")
            print("   - Feature selection with multiple methods")
            print("   - ML classification with multiple algorithms")
            print("   - Comprehensive evaluation and reporting")
            return
        
        print("📊 Step 1: Loading and preprocessing data...")
        loader = DataLoader()
        X_train, y_train, X_test, y_test = loader.load_data(
            args.data_file, 
            args.target, 
            args.test_file
        )
        
        X_train, y_train, X_test, y_test = loader.preprocess_data(
            X_train, y_train, X_test, y_test
        )
        
        print(f"   ✅ Loaded {len(X_train)} training samples with {len(X_train.columns)} features")
        if X_test is not None:
            print(f"   ✅ Loaded {len(X_test)} test samples")
        
        print("\n🔍 Step 2: Variance-based pre-filtering...")
        variance_filter = VarianceFilter()
        top_variance_features = variance_filter.select_top_variance_features(
            X_train, args.top_variance_genes
        )
        X_train_filtered = X_train[top_variance_features]
        if X_test is not None:
            X_test_filtered = X_test[top_variance_features]
        else:
            X_test_filtered = None
        
        print(f"   ✅ Pre-filtering: {len(X_train_filtered.columns)} features retained")
        
        print("\n🎯 Step 3: Feature selection...")
        selector = FeatureSelector()
        X_selected = selector.fit_transform(
            X_train_filtered,
            y_train,
            n_features=args.n_features,
            cv_folds=args.cv_folds
        )
        
        if X_test_filtered is not None:
            X_test_selected = selector.transform(X_test_filtered)
        else:
            X_test_selected = None
        
        print(f"   ✅ Feature selection: {len(selector.selected_features)} features selected")
        print(f"   ✅ Best method: {selector.best_method}")
        
        print("\n🤖 Step 4: ML model training and evaluation...")
        classifier = MLClassifier()
        results = classifier.fit(
            X_selected,
            y_train,
            X_test_selected,
            y_test,
            cv_folds=args.cv_folds
        )
        
        print(f"   ✅ Best model: {classifier.best_model_name}")
        print(f"   ✅ Best score: {classifier.best_score:.4f}")
        
        print("\n💾 Step 5: Saving results...")
        
        # Save feature selection results
        selector.save_results(str(output_dir / "feature_selection"))
        
        # Save model results
        classifier.save_models(str(output_dir / "models"))
        
        # Save comprehensive summary
        pipeline_summary = {
            'input_file': str(args.data_file),
            'target_column': args.target,
            'test_file': str(args.test_file) if args.test_file else None,
            'runtime_seconds': time.time() - start_time,
            'data_summary': {
                'n_train_samples': len(X_train),
                'n_test_samples': len(X_test) if X_test is not None else 0,
                'original_features': len(X_train.columns),
                'after_variance_filtering': len(X_train_filtered.columns),
                'selected_features': len(selector.selected_features),
                'target_classes': y_train.unique().tolist()
            },
            'feature_selection': selector.get_feature_selection_summary(),
            'model_results': classifier.get_results_summary()
        }
        
        with open(output_dir / "pipeline_summary.json", 'w') as f:
            json.dump(pipeline_summary, f, indent=2, default=str)
        
        # Print final summary
        print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📁 Results saved to: {output_dir}")
        print(f"⏱️  Total runtime: {time.time() - start_time:.1f} seconds")
        print(f"📊 Original features: {len(X_train.columns)}")
        print(f"🔍 After variance filtering: {len(X_train_filtered.columns)}")
        print(f"🎯 Selected features: {len(selector.selected_features)}")
        print(f"🥇 Best feature method: {selector.best_method}")
        print(f"🤖 Best ML model: {classifier.best_model_name}")
        print(f"📈 Best score: {classifier.best_score:.4f}")
        
        if X_test is not None:
            print(f"✅ Test set evaluation completed")
        else:
            print("ℹ️  Used cross-validation (no separate test set)")
        
    except Exception as e:
        print(f"❌ Error in pipeline: {str(e)}")
        print("\n🔧 Troubleshooting:")
        print("1. Check that the data file exists and is readable")
        print("2. Verify the target column name is correct")
        print("3. Ensure dependencies are installed: pip install -r requirements.txt")
        raise


if __name__ == "__main__":
    main()
