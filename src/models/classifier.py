"""
Machine Learning Classifier Module

Provides comprehensive ML classification with multiple algorithms and evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class MLClassifier:
    """
    Comprehensive ML classifier with multiple algorithms and evaluation.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize MLClassifier.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0.0
        
    def define_models(self, config: Optional[Dict] = None) -> None:
        """
        Define ML models to train.
        
        Args:
            config: Optional configuration dictionary for custom models
        """
        if config is None:
            # Default model configuration
            self.models = {
                'Random Forest': {
                    'model': RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [None, 10, 20],
                        'min_samples_split': [2, 5, 10]
                    }
                },
                'Logistic Regression': {
                    'model': LogisticRegression(random_state=self.random_state, max_iter=1000),
                    'params': {
                        'C': [0.1, 1, 10, 100],
                        'penalty': ['l1', 'l2'],
                        'solver': ['liblinear']
                    }
                },
                'SVM': {
                    'model': SVC(probability=True, random_state=self.random_state),
                    'params': {
                        'C': [0.1, 1, 10],
                        'kernel': ['rbf', 'linear'],
                        'gamma': ['scale', 'auto']
                    }
                },
                'Gradient Boosting': {
                    'model': GradientBoostingClassifier(random_state=self.random_state),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 5, 7]
                    }
                }
            }
        else:
            self.models = config
        
        logger.info(f"Defined {len(self.models)} models for training")
    
    def fit(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        cv_folds: int = 5,
        scoring: str = 'accuracy'
    ) -> Dict[str, Any]:
        """
        Train and evaluate all models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features (optional)
            y_test: Test targets (optional)
            cv_folds: Number of CV folds
            scoring: Scoring metric for optimization
            
        Returns:
            Dictionary with training results
        """
        logger.info("Starting model training and evaluation...")
        
        if not self.models:
            self.define_models()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for model_name, model_config in self.models.items():
            logger.info(f"Training {model_name}...")
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                model_config['model'],
                model_config['params'],
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train_scaled, y_train)
            
            # Store results
            result = {
                'best_params': grid_search.best_params_,
                'cv_score': grid_search.best_score_,
                'model': grid_search.best_estimator_
            }
            
            # Evaluate on test set if available
            if X_test is not None and y_test is not None:
                y_pred = grid_search.predict(X_test_scaled)
                y_prob = grid_search.predict_proba(X_test_scaled)[:, 1] if len(np.unique(y_test)) == 2 else None
                
                result.update({
                    'test_accuracy': accuracy_score(y_test, y_pred),
                    'test_precision': precision_score(y_test, y_pred, average='weighted'),
                    'test_recall': recall_score(y_test, y_pred, average='weighted'),
                    'test_f1': f1_score(y_test, y_pred, average='weighted'),
                    'predictions': y_pred,
                    'probabilities': y_prob
                })
                
                if y_prob is not None:
                    result['test_auc'] = roc_auc_score(y_test, y_prob)
            
            self.results[model_name] = result
            
            logger.info(f"  CV Score: {grid_search.best_score_:.4f}")
            if X_test is not None:
                logger.info(f"  Test Accuracy: {result['test_accuracy']:.4f}")
            
            # Track best model
            current_score = result.get('test_accuracy', result['cv_score'])
            if current_score > self.best_score:
                self.best_score = current_score
                self.best_model = grid_search.best_estimator_
                self.best_model_name = model_name
        
        logger.info(f"Best model: {self.best_model_name} (Score: {self.best_score:.4f})")
        
        return self.results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the best model.
        
        Args:
            X: Features to predict
            
        Returns:
            Predictions
        """
        if self.best_model is None:
            raise ValueError("No model trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.best_model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities using the best model.
        
        Args:
            X: Features to predict
            
        Returns:
            Prediction probabilities
        """
        if self.best_model is None:
            raise ValueError("No model trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.best_model.predict_proba(X_scaled)
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Get feature importance from the best model.
        
        Returns:
            Feature importance series or None if not available
        """
        if self.best_model is None:
            return None
        
        if hasattr(self.best_model, 'feature_importances_'):
            # Tree-based models
            return pd.Series(
                self.best_model.feature_importances_,
                index=self.scaler.feature_names_in_
            )
        elif hasattr(self.best_model, 'coef_'):
            # Linear models
            return pd.Series(
                np.abs(self.best_model.coef_[0]),
                index=self.scaler.feature_names_in_
            )
        else:
            logger.warning(f"Feature importance not available for {self.best_model_name}")
            return None
    
    def get_results_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive results summary.
        
        Returns:
            Dictionary with results summary
        """
        if not self.results:
            return {}
        
        summary = {
            'models_trained': list(self.results.keys()),
            'best_model': self.best_model_name,
            'best_score': self.best_score,
            'model_comparison': {}
        }
        
        for model_name, result in self.results.items():
            model_summary = {
                'cv_score': result['cv_score'],
                'best_params': result['best_params']
            }
            
            if 'test_accuracy' in result:
                model_summary.update({
                    'test_accuracy': result['test_accuracy'],
                    'test_precision': result['test_precision'],
                    'test_recall': result['test_recall'],
                    'test_f1': result['test_f1']
                })
                
                if 'test_auc' in result:
                    model_summary['test_auc'] = result['test_auc']
            
            summary['model_comparison'][model_name] = model_summary
        
        return summary
    
    def save_models(self, output_dir: str) -> None:
        """
        Save trained models to disk.
        
        Args:
            output_dir: Directory to save models
        """
        import os
        import joblib
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save best model
        if self.best_model is not None:
            joblib.dump(self.best_model, f"{output_dir}/best_model.pkl")
            joblib.dump(self.scaler, f"{output_dir}/scaler.pkl")
            logger.info(f"Best model saved to {output_dir}/best_model.pkl")
        
        # Save all models
        for model_name, result in self.results.items():
            safe_name = model_name.replace(' ', '_').lower()
            joblib.dump(result['model'], f"{output_dir}/{safe_name}_model.pkl")
        
        logger.info(f"All models saved to {output_dir}")
    
    def load_model(self, model_path: str, scaler_path: str) -> None:
        """
        Load a pre-trained model.
        
        Args:
            model_path: Path to model file
            scaler_path: Path to scaler file
        """
        import joblib
        
        self.best_model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.best_model_name = "Loaded Model"
        
        logger.info(f"Model loaded from {model_path}")
    
    def create_classification_report(self, y_true: pd.Series, y_pred: np.ndarray) -> str:
        """
        Create detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Classification report string
        """
        return classification_report(y_true, y_pred)
