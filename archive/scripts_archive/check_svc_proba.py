#!/usr/bin/env python3
"""
Check if SVC models have predict_proba() method available.
This script helps diagnose the 'SVC has no attribute predict_proba' error.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve, auc

print("Testing SVC predict_proba support")

# Generate synthetic data
X, y = make_classification(n_samples=100, n_features=20, random_state=42)

# Test SVC with probability=True (should work)
print("\nTest 1: SVC with probability=True")
model1 = SVC(probability=True, random_state=42)
model1.fit(X, y)
if hasattr(model1, 'predict_proba') and callable(model1.predict_proba):
    print("✅ predict_proba is available")
    y_prob = model1.predict_proba(X)[:, 1]
    print(f"Shape of probability output: {y_prob.shape}")
    
    # Try to calculate ROC curve
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc:.4f}")
else:
    print("❌ predict_proba is NOT available (this shouldn't happen)")

# Test SVC without probability=True (should fail but gracefully)
print("\nTest 2: SVC without probability=True")
model2 = SVC(random_state=42)
model2.fit(X, y)
if hasattr(model2, 'predict_proba') and callable(model2.predict_proba):
    print("predict_proba is available (unexpected!)")
else:
    print("❌ predict_proba is NOT available (expected)")
    print("Using decision_function instead...")
    
    # Use decision_function as an alternative
    if hasattr(model2, 'decision_function') and callable(model2.decision_function):
        y_score = model2.decision_function(X)
        print(f"Shape of decision_function output: {y_score.shape}")
        
        # Normalize scores to [0, 1] range
        y_score_norm = (y_score - y_score.min()) / (y_score.max() - y_score.min())
        
        # Try to calculate ROC curve with decision_function
        fpr, tpr, _ = roc_curve(y, y_score_norm)
        roc_auc = auc(fpr, tpr)
        print(f"ROC AUC with decision_function: {roc_auc:.4f}")

print("\nConclusion:")
print("If you want to use predict_proba() with SVC, always initialize with:")
print("    SVC(probability=True, ...)")
print("\nThis will enable ROC curve calculation and probability-based metrics.")
