"""
Data Loading and Preprocessing Module

This module provides comprehensive data loading, validation, and preprocessing
functionality for machine learning pipelines, specifically designed for genomic data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union, List
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from ..utils.logger import get_logger
from .validator import DataValidator

logger = get_logger(__name__)


class DataLoader:
    """
    Comprehensive data loader with preprocessing capabilities.
    
    Handles CSV, Excel files and provides automatic data cleaning,
    validation, and preprocessing for ML pipelines.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize DataLoader.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.validator = DataValidator()
        self.feature_names = None
        self.target_name = None
        
    def load_data(
        self, 
        file_path: Union[str, Path], 
        target_column: str,
        test_file_path: Optional[Union[str, Path]] = None,
        index_col: Optional[int] = 0
    ) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        Load training and optional test data from file.
        
        Args:
            file_path: Path to training data file (CSV or Excel)
            target_column: Name of target column
            test_file_path: Optional path to test data file
            index_col: Column to use as index (default: first column)
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
            X_test and y_test are None if no test file provided
        """
        logger.info(f"Loading training data from: {file_path}")
        
        # Load training data
        train_df = self._load_file(file_path, index_col)
        logger.info(f"Training data shape: {train_df.shape}")
        
        # Load test data if provided
        test_df = None
        if test_file_path:
            logger.info(f"Loading test data from: {test_file_path}")
            test_df = self._load_file(test_file_path, index_col)
            logger.info(f"Test data shape: {test_df.shape}")
        
        # Validate data
        self.validator.validate_dataframe(train_df, target_column)
        if test_df is not None:
            self.validator.validate_dataframe(test_df, target_column)
            self.validator.validate_compatibility(train_df, test_df, target_column)
        
        # Separate features and target
        X_train, y_train = self._separate_features_target(train_df, target_column)
        
        X_test, y_test = None, None
        if test_df is not None:
            X_test, y_test = self._separate_features_target(test_df, target_column)
        
        # Store metadata
        self.feature_names = X_train.columns.tolist()
        self.target_name = target_column
        
        logger.info(f"Features extracted: {len(self.feature_names)}")
        logger.info(f"Target variable: {target_column}")
        logger.info(f"Target classes: {y_train.unique()}")
        
        return X_train, y_train, X_test, y_test
    
    def _load_file(self, file_path: Union[str, Path], index_col: Optional[int]) -> pd.DataFrame:
        """Load data from CSV or Excel file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            if file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, index_col=index_col)
            elif file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path, index_col=index_col)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
                
            logger.info(f"Successfully loaded {file_path.name}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            raise
    
    def _separate_features_target(
        self, 
        df: pd.DataFrame, 
        target_column: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate features and target variable."""
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Get target
        y = df[target_column].copy()
        
        # Get features (only numeric columns, excluding target)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != target_column]
        
        if len(feature_cols) == 0:
            raise ValueError("No numeric feature columns found")
        
        X = df[feature_cols].copy()
        
        logger.info(f"Selected {len(feature_cols)} numeric features")
        
        return X, y
    
    def preprocess_data(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        handle_missing: str = 'mean',
        scale_features: bool = True,
        encode_target: bool = True
    ) -> Tuple:
        """
        Preprocess the data with cleaning, scaling, and encoding.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features (optional)
            y_test: Test target (optional)
            handle_missing: Method to handle missing values ('mean', 'median', 'drop')
            scale_features: Whether to scale features
            encode_target: Whether to encode target labels
            
        Returns:
            Preprocessed data tuple
        """
        logger.info("Starting data preprocessing...")
        
        # Handle missing values
        if handle_missing != 'drop':
            X_train = self._handle_missing_values(X_train, method=handle_missing)
            if X_test is not None:
                X_test = self._handle_missing_values(X_test, method=handle_missing)
        
        # Scale features
        if scale_features:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_train = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
            
            if X_test is not None:
                X_test_scaled = self.scaler.transform(X_test)
                X_test = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
            
            logger.info("Features scaled using StandardScaler")
        
        # Encode target
        if encode_target:
            y_train_encoded = self.label_encoder.fit_transform(y_train)
            y_train = pd.Series(y_train_encoded, index=y_train.index)
            
            if y_test is not None:
                y_test_encoded = self.label_encoder.transform(y_test)
                y_test = pd.Series(y_test_encoded, index=y_test.index)
            
            logger.info(f"Target encoded: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
        
        logger.info("Data preprocessing completed")
        
        return X_train, y_train, X_test, y_test
    
    def _handle_missing_values(self, df: pd.DataFrame, method: str = 'mean') -> pd.DataFrame:
        """Handle missing values in the dataframe."""
        if method == 'mean':
            return df.fillna(df.mean())
        elif method == 'median':
            return df.fillna(df.median())
        elif method == 'drop':
            return df.dropna()
        else:
            raise ValueError(f"Unknown missing value handling method: {method}")
    
    def create_train_validation_split(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        validation_size: float = 0.2,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Create train/validation split.
        
        Args:
            X: Features
            y: Target
            validation_size: Proportion for validation set
            stratify: Whether to stratify split
            
        Returns:
            X_train, X_val, y_train, y_val
        """
        stratify_param = y if stratify else None
        
        return train_test_split(
            X, y, 
            test_size=validation_size, 
            random_state=self.random_state,
            stratify=stratify_param
        )
    
    def get_data_summary(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Get comprehensive data summary."""
        return {
            'n_samples': len(X),
            'n_features': len(X.columns),
            'feature_names': X.columns.tolist(),
            'target_name': self.target_name,
            'target_classes': y.unique().tolist(),
            'class_distribution': y.value_counts().to_dict(),
            'missing_values': X.isnull().sum().sum(),
            'feature_types': X.dtypes.to_dict()
        }
