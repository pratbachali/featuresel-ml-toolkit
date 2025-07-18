# Fixing "SVC has no attribute 'predict_proba'" Error

## Problem

The error message "AttributeError: This 'SVC' has no attribute 'predict_proba'" occurs when trying to use the `predict_proba()` method on an SVC model that was initialized without setting `probability=True`.

By default, SVC models in scikit-learn don't calculate probability estimates because:
1. It requires an additional computational overhead
2. It uses cross-validation internally, making the training slower
3. The default SVC focuses on classification boundaries, not probability estimation

## Solution

Always initialize SVC models with `probability=True` when you need to:
- Generate ROC curves
- Get probability scores
- Calculate AUC or other probability-based metrics

```python
from sklearn.svm import SVC
model = SVC(probability=True, random_state=42)  # Enables predict_proba()
```

## Status of Files in This Project

We've reviewed all files in the project and confirmed that all instances of SVC initialization already include `probability=True`:

1. `scripts/run_ml_models.py` - SVC is correctly initialized with `probability=True`
2. `scripts/run_machine_learning.py` - SVC is correctly initialized with `probability=True`
3. `src/models/classifier.py` - SVC is correctly initialized with `probability=True`

## Possible Reasons You're Still Seeing the Error

1. **Running an old version of the script**: Make sure you're running the latest version with the fixes.

2. **Environment issues**: Check if your environment has the correct scikit-learn version (0.20+ recommended).

3. **Code not being executed**: The code that initializes SVC with `probability=True` might not be getting executed.

4. **Pickle loading old models**: If you're loading saved models (via pickle or joblib), those might be old SVC models without `probability=True`.

## Additional Improvements

We've added a diagnostic script `check_svc_proba.py` that:
- Tests SVC with and without `probability=True`
- Demonstrates alternatives when `predict_proba()` isn't available
- Shows how to use `decision_function()` as a fallback

## Fallback Solution

If you can't modify the model initialization, you can:

1. Use `decision_function()` instead:
```python
if hasattr(model, 'predict_proba'):
    y_proba = model.predict_proba(X)[:, 1]
else:
    # Normalize decision_function scores to [0,1] range
    y_scores = model.decision_function(X)
    y_proba = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
```

2. Wrap the model with `CalibratedClassifierCV`:
```python
from sklearn.calibration import CalibratedClassifierCV
base_model = SVC(kernel='rbf')
calibrated_model = CalibratedClassifierCV(base_model, cv=5)
calibrated_model.fit(X_train, y_train)
# Now you can use predict_proba()
probabilities = calibrated_model.predict_proba(X_test)
```
