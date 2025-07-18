"""
Feature Selection Module

Comprehensive feature selection methods for ML pipelines with hyperparameter optimization.

Key features:
- Multiple feature selection methods (univariate, importance-based, regularization)
- Automatic hyperparameter tuning for each method
- Cross-validation based evaluation
- Comprehensive visualization and reporting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, 
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Comprehensive feature selection with multiple methods and evaluation.
    """
    
    def __init__(self, random_state: int = 42, use_hyperparameter_tuning: bool = True):
        """
        Initialize FeatureSelector.
        
        Args:
            random_state: Random seed for reproducibility
            use_hyperparameter_tuning: Whether to use hyperparameter optimization (slower but more accurate)
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.selection_results = {}
        self.method_features = {}  # Alias for backward compatibility
        self.method_scores = {}
        self.method_accuracy_scores = {}  # Store accuracy scores
        self.best_method = None
        self.selected_features = None
        self.use_hyperparameter_tuning = use_hyperparameter_tuning
        
    def fit_transform(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        methods: Optional[List[str]] = None,
        n_features: int = 100,
        cv_folds: int = 5,
        use_hyperparameter_tuning: bool = True
    ) -> pd.DataFrame:
        """
        Apply feature selection methods and return best features.
        
        Args:
            X: Feature matrix
            y: Target vector
            methods: List of methods to use (None for all)
            n_features: Number of features to select
            cv_folds: Number of CV folds for evaluation
            use_hyperparameter_tuning: Whether to optimize hyperparameters (slower but more accurate)
            
        Returns:
            DataFrame with selected features
        """
        # Store the hyperparameter tuning preference
        self.use_hyperparameter_tuning = use_hyperparameter_tuning
        logger.info(f"Starting feature selection with {len(X.columns)} features")
        
        # Default methods if none specified
        if methods is None:
            methods = [
                'univariate_f_test', 
                'mutual_information',
                'random_forest_importance',
                'rfe_random_forest',
                'lasso_regularization',
                'gradient_boosting_importance'
            ]
        
        # Apply each feature selection method
        for method in methods:
            logger.info(f"Applying {method}")
            selected_features = self._apply_method(X, y, method, n_features)
            self.selection_results[method] = selected_features
            self.method_features[method] = selected_features  # For backward compatibility
        
        # Evaluate methods using cross-validation
        self._evaluate_methods(X, y, cv_folds)
        
        # Select best method and features
        self.best_method = max(self.method_scores, key=self.method_scores.get)
        self.selected_features = self.selection_results[self.best_method]
        
        logger.info(f"Best method: {self.best_method}")
        logger.info(f"Selected {len(self.selected_features)} features")
        
        return X[self.selected_features]
    
    def _apply_method(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        method: str, 
        n_features: int
    ) -> List[str]:
        """Apply specific feature selection method."""
        
        if method == 'univariate_f_test':
            return self._univariate_f_test(X, y, n_features)
            
        elif method == 'mutual_information':
            return self._mutual_information(X, y, n_features)
            
        elif method == 'random_forest_importance':
            return self._random_forest_importance(X, y, n_features)
            
        elif method == 'rfe_random_forest':
            return self._rfe_random_forest(X, y, n_features)
            
        elif method == 'lasso_regularization':
            return self._lasso_regularization(X, y, n_features)
            
        elif method == 'gradient_boosting_importance':
            return self._gradient_boosting_importance(X, y, n_features)
            
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
    
    def _univariate_f_test(self, X: pd.DataFrame, y: pd.Series, k: int) -> List[str]:
        """Univariate F-test feature selection."""
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X, y)
        return X.columns[selector.get_support()].tolist()
    
    def _mutual_information(self, X: pd.DataFrame, y: pd.Series, k: int) -> List[str]:
        """Mutual information feature selection."""
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        selector.fit(X, y)
        return X.columns[selector.get_support()].tolist()
    
    def _random_forest_importance(self, X: pd.DataFrame, y: pd.Series, k: int) -> List[str]:
        """Random Forest importance-based feature selection with hyperparameter tuning."""
        # Scale features for consistent results
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        base_rf = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        
        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Optimize hyperparameters
        rf = self._optimize_model(base_rf, X_scaled_df, y, param_grid)
        
        # Get feature importances
        importance_df = pd.Series(rf.feature_importances_, index=X.columns)
        return importance_df.nlargest(k).index.tolist()
    
    def _rfe_random_forest(self, X: pd.DataFrame, y: pd.Series, k: int) -> List[str]:
        """Recursive Feature Elimination with optimized Random Forest."""
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        base_rf = RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Define a smaller hyperparameter grid for RFE as it's computationally intensive
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10]
        }
        
        # Optimize hyperparameters for the base estimator
        rf = self._optimize_model(base_rf, X_scaled_df, y, param_grid, cv=3)
        
        selector = RFE(estimator=rf, n_features_to_select=k)
        selector.fit(X_scaled_df, y)
        
        return X.columns[selector.support_].tolist()
    
    def _lasso_regularization(self, X: pd.DataFrame, y: pd.Series, k: int) -> List[str]:
        """L1 regularization (Lasso) feature selection with hyperparameter tuning."""
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        base_lasso = LogisticRegression(
            penalty='l1', 
            solver='liblinear',
            random_state=self.random_state,
            max_iter=2000
        )
        
        # Define hyperparameter grid
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0],  # Inverse of regularization strength
            'class_weight': [None, 'balanced']
        }
        
        # Optimize hyperparameters
        lasso = self._optimize_model(base_lasso, X_scaled_df, y, param_grid)
        
        selector = SelectFromModel(lasso, max_features=k)
        selector.fit(X_scaled_df, y)
        
        selected = X.columns[selector.get_support()]
        
        # If no features selected, use coefficient magnitude
        if len(selected) == 0:
            lasso.fit(X_scaled_df, y)
            coef_importance = pd.Series(np.abs(lasso.coef_[0]), index=X.columns)
            return coef_importance.nlargest(k).index.tolist()
        
        return selected.tolist()
    
    def _gradient_boosting_importance(self, X: pd.DataFrame, y: pd.Series, k: int) -> List[str]:
        """Gradient Boosting importance-based feature selection with hyperparameter tuning."""
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        base_gb = GradientBoostingClassifier(random_state=self.random_state)
        
        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5]
        }
        
        # Optimize hyperparameters
        gb = self._optimize_model(base_gb, X_scaled_df, y, param_grid)
        
        importance_df = pd.Series(gb.feature_importances_, index=X.columns)
        return importance_df.nlargest(k).index.tolist()
    
    def _evaluate_methods(self, X: pd.DataFrame, y: pd.Series, cv_folds: int) -> None:
        """Evaluate feature selection methods using cross-validation with hyperparameter tuning."""
        logger.info("Evaluating feature selection methods with optimized models...")
        
        # Use an optimized model for evaluation
        base_evaluator = RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Define evaluation hyperparameter grid
        eval_param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'class_weight': [None, 'balanced']
        }
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for method, features in self.selection_results.items():
            if len(features) > 0:
                X_subset = X[features]
                
                # Use hyperparameter optimization for more accurate evaluation
                try:
                    logger.info(f"Optimizing evaluation model for {method}...")
                    # Use fewer hyperparameters for faster evaluation
                    eval_param_grid_reduced = {
                        'n_estimators': [50, 100],
                        'max_depth': [None, 10]
                    }
                    evaluator = self._optimize_model(base_evaluator, X_subset, y, eval_param_grid_reduced, cv=3)
                    
                    # Calculate both AUC and accuracy scores with optimized model
                    auc_scores = cross_val_score(evaluator, X_subset, y, cv=cv, scoring='roc_auc')
                    accuracy_scores = cross_val_score(evaluator, X_subset, y, cv=cv, scoring='accuracy')
                    
                    self.method_scores[method] = auc_scores.mean()
                    self.method_accuracy_scores[method] = accuracy_scores.mean()
                    
                    logger.info(f"{method}: AUC {auc_scores.mean():.4f} ± {auc_scores.std():.4f}, "
                               f"Accuracy {accuracy_scores.mean():.4f} ± {accuracy_scores.std():.4f} "
                               f"(with optimized evaluation model)")
                except Exception as e:
                    # Fallback to standard evaluation if optimization fails
                    logger.warning(f"Optimization failed for {method}, using default model: {str(e)}")
                    default_evaluator = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
                    
                    auc_scores = cross_val_score(default_evaluator, X_subset, y, cv=cv, scoring='roc_auc')
                    accuracy_scores = cross_val_score(default_evaluator, X_subset, y, cv=cv, scoring='accuracy')
                    
                    self.method_scores[method] = auc_scores.mean()
                    self.method_accuracy_scores[method] = accuracy_scores.mean()
                    
                    logger.info(f"{method}: AUC {auc_scores.mean():.4f} ± {auc_scores.std():.4f}, "
                               f"Accuracy {accuracy_scores.mean():.4f} ± {accuracy_scores.std():.4f} "
                               f"(with default model)")
            else:
                self.method_scores[method] = 0.0
                self.method_accuracy_scores[method] = 0.0
                logger.warning(f"{method}: No features selected")
    
    def get_feature_selection_summary(self) -> Dict:
        """Get comprehensive summary of feature selection results."""
        return {
            'methods_applied': list(self.selection_results.keys()),
            'features_per_method': {
                method: len(features) for method, features in self.selection_results.items()
            },
            'method_scores': self.method_scores,
            'method_accuracy_scores': self.method_accuracy_scores,
            'best_method': self.best_method,
            'best_score': self.method_scores.get(self.best_method, 0.0),
            'best_accuracy': self.method_accuracy_scores.get(self.best_method, 0.0),
            'selected_features': self.selected_features,
            'n_selected_features': len(self.selected_features) if self.selected_features else 0
        }
    
    def get_feature_rankings(self) -> Dict[str, List[str]]:
        """Get feature rankings for each method."""
        # Update method_features for backward compatibility
        self.method_features = self.selection_results.copy()
        return self.selection_results.copy()
    
    def save_results(self, output_dir: str) -> None:
        """Save feature selection results to files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save feature selections
        results_df = pd.DataFrame.from_dict(
            {method: pd.Series(features) for method, features in self.selection_results.items()},
            orient='index'
        ).T
        results_df.to_csv(f"{output_dir}/feature_selection_results.csv")
        
        # Save method scores
        scores_df = pd.DataFrame({
            'auc_score': self.method_scores,
            'accuracy_score': self.method_accuracy_scores
        })
        scores_df.to_csv(f"{output_dir}/method_evaluation_scores.csv")
        
        # Save selected features
        if self.selected_features:
            pd.Series(self.selected_features).to_csv(
                f"{output_dir}/selected_features.csv", 
                header=['feature_name']
            )
        
        logger.info(f"Feature selection results saved to {output_dir}")
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using selected features."""
        if self.selected_features is None:
            raise ValueError("No features selected. Run fit_transform first.")
        
        return X[self.selected_features]
    
    def _optimize_model(self, model, X: pd.DataFrame, y: pd.Series, param_grid: dict, cv: int = 5) -> object:
        """
        Optimize model hyperparameters using grid search cross-validation
        
        Args:
            model: Base model to optimize
            X: Features dataframe
            y: Target series
            param_grid: Dictionary of hyperparameter grids
            cv: Number of cross-validation folds
            
        Returns:
            Best model with optimized hyperparameters
        """
        # If hyperparameter tuning is disabled, return the base model with default parameters
        if not self.use_hyperparameter_tuning:
            logger.info(f"Hyperparameter tuning disabled. Using default parameters for {type(model).__name__}")
            model.fit(X, y)
            return model
        
        logger.info(f"Optimizing hyperparameters for {type(model).__name__}")
        
        grid_search = GridSearchCV(
            model, 
            param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
