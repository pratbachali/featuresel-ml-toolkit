"""
Data Validation Module

Provides comprehensive data validation functionality for ML pipelines.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates data quality and compatibility for ML pipelines.
    """
    
    def __init__(self, missing_threshold: float = 0.1):
        """
        Initialize DataValidator.
        
        Args:
            missing_threshold: Maximum proportion of missing values allowed
        """
        self.missing_threshold = missing_threshold
    
    def validate_dataframe(self, df: pd.DataFrame, target_column: str) -> bool:
        """
        Validate a dataframe for ML readiness.
        
        Args:
            df: DataFrame to validate
            target_column: Name of target column
            
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If validation fails
        """
        # Check if dataframe is empty
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        # Check if target column exists
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        # Check for excessive missing values
        self._check_missing_values(df)
        
        # Check for constant features
        self._check_constant_features(df, target_column)
        
        # Check target variable
        self._validate_target(df[target_column])
        
        logger.info("Data validation passed")
        return True
    
    def validate_compatibility(
        self, 
        train_df: pd.DataFrame, 
        test_df: pd.DataFrame,
        target_column: str
    ) -> bool:
        """
        Validate compatibility between training and test data.
        
        Args:
            train_df: Training dataframe
            test_df: Test dataframe
            target_column: Name of target column
            
        Returns:
            True if compatible
            
        Raises:
            ValueError: If incompatible
        """
        # Get numeric features (excluding target)
        train_numeric = train_df.select_dtypes(include=[np.number]).columns
        train_features = [col for col in train_numeric if col != target_column]
        
        test_numeric = test_df.select_dtypes(include=[np.number]).columns
        test_features = [col for col in test_numeric if col != target_column]
        
        # Check for common features
        common_features = set(train_features) & set(test_features)
        
        if len(common_features) == 0:
            raise ValueError("No common numeric features between train and test data")
        
        if len(common_features) < len(train_features) * 0.8:
            logger.warning(f"Only {len(common_features)}/{len(train_features)} features are common")
        
        # Check target compatibility
        train_classes = set(train_df[target_column].unique())
        test_classes = set(test_df[target_column].unique())
        
        if not test_classes.issubset(train_classes):
            unknown_classes = test_classes - train_classes
            logger.warning(f"Test data contains unknown classes: {unknown_classes}")
        
        logger.info("Data compatibility validation passed")
        return True
    
    def _check_missing_values(self, df: pd.DataFrame) -> None:
        """Check for excessive missing values."""
        missing_props = df.isnull().mean()
        excessive_missing = missing_props > self.missing_threshold
        
        if excessive_missing.any():
            problematic_cols = missing_props[excessive_missing].index.tolist()
            raise ValueError(
                f"Columns with >={self.missing_threshold*100}% missing values: {problematic_cols}"
            )
    
    def _check_constant_features(self, df: pd.DataFrame, target_column: str) -> None:
        """Check for constant features."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != target_column]
        
        constant_features = []
        for col in feature_cols:
            if df[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            logger.warning(f"Constant features detected: {constant_features}")
    
    def _validate_target(self, target: pd.Series) -> None:
        """Validate target variable."""
        if target.isnull().any():
            raise ValueError("Target variable contains missing values")
        
        n_classes = target.nunique()
        if n_classes < 2:
            raise ValueError("Target variable must have at least 2 classes")
        
        if n_classes > 10:
            logger.warning(f"Target has {n_classes} classes - consider if this is intended")
        
        # Check class balance
        class_counts = target.value_counts()
        min_class_size = class_counts.min()
        max_class_size = class_counts.max()
        
        if min_class_size < 5:
            logger.warning(f"Smallest class has only {min_class_size} samples")
        
        if max_class_size / min_class_size > 10:
            logger.warning("Severe class imbalance detected")
    
    def get_data_quality_report(self, df: pd.DataFrame, target_column: str) -> dict:
        """Generate comprehensive data quality report."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != target_column]
        
        return {
            'total_samples': len(df),
            'total_features': len(feature_cols),
            'missing_values': {
                'total': df.isnull().sum().sum(),
                'by_column': df.isnull().sum().to_dict()
            },
            'data_types': df.dtypes.to_dict(),
            'target_info': {
                'name': target_column,
                'classes': df[target_column].unique().tolist(),
                'class_distribution': df[target_column].value_counts().to_dict()
            },
            'feature_summary': {
                'numeric_features': len(feature_cols),
                'constant_features': [col for col in feature_cols if df[col].nunique() <= 1],
                'low_variance_features': [col for col in feature_cols if df[col].var() < 0.01]
            }
        }
