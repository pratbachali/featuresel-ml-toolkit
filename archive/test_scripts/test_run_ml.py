#!/usr/bin/env python3
"""
Test script to run the ML pipeline with a small synthetic dataset.
This will verify that the SVC predict_proba issue is properly handled.
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Generate synthetic data
X, y = make_classification(
    n_samples=200, 
    n_features=20, 
    n_informative=10,
    n_redundant=5, 
    n_classes=2, 
    random_state=42
)

# Create string labels to test that functionality
# Map 0 -> "IS", 1 -> "IR" to match what might be in real data
label_map = {0: "IS", 1: "IR"}
y_str = np.array([label_map[label] for label in y])

# Create a DataFrame
feature_names = [f"feature_{i+1}" for i in range(X.shape[1])]
df_train = pd.DataFrame(X[:150], columns=feature_names)
df_train["target"] = y_str[:150]

df_test = pd.DataFrame(X[150:], columns=feature_names)
df_test["target"] = y_str[150:]

# Save to CSV files
train_file = "test_train_data.csv"
test_file = "test_test_data.csv"

df_train.to_csv(train_file, index=False)
df_test.to_csv(test_file, index=False)

print(f"Created train file: {train_file} with {df_train.shape[0]} samples")
print(f"Created test file: {test_file} with {df_test.shape[0]} samples")
print(f"Target distribution in train: {df_train['target'].value_counts().to_dict()}")
print(f"Target distribution in test: {df_test['target'].value_counts().to_dict()}")

# Run the ML pipeline script
print("\nRunning ML pipeline script...")
cmd = f"python scripts/run_ml_models.py --train_file {train_file} --test_file {test_file} --target target --use_predefined_split --output_dir test_ml_output"
print(f"Command: {cmd}")

import subprocess
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

if result.returncode == 0:
    print("\nML pipeline executed successfully!")
else:
    print("\nML pipeline execution failed!")
    print(f"Return code: {result.returncode}")
    print("\nSTDOUT:")
    print(result.stdout)
    print("\nSTDERR:")
    print(result.stderr)
