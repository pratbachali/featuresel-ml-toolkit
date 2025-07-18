"""
Variance-based Feature Filtering Module

Handles variance-based feature filtering for dimensionality reduction.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class VarianceFilter:
    """
    Filter features based on variance thresholds.
    
    Useful for removing low-variance features and selecting
    top variance features for initial dimensionality reduction.
    """
    
    def __init__(self, variance_threshold: float = 0.01):
        """
        Initialize VarianceFilter.
        
        Args:
            variance_threshold: Minimum variance threshold for features
        """
        self.variance_threshold = variance_threshold
        self.feature_variances = None
        self.selected_features = None
        
    def fit(self, X: pd.DataFrame) -> 'VarianceFilter':
        """
        Fit the variance filter to the data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Self for method chaining
        """
        # Calculate variances
        self.feature_variances = X.var()
        
        # Identify features above threshold
        self.selected_features = self.feature_variances[
            self.feature_variances >= self.variance_threshold
        ].index.tolist()
        
        logger.info(f"Variance filtering: {len(self.selected_features)}/{len(X.columns)} features retained")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by selecting high-variance features.
        
        Args:
            X: Feature matrix to transform
            
        Returns:
            Transformed feature matrix
        """
        if self.selected_features is None:
            raise ValueError("VarianceFilter not fitted. Call fit() first.")
        
        return X[self.selected_features]
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the filter and transform the data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Transformed feature matrix
        """
        return self.fit(X).transform(X)
    
    def select_top_variance_features(
        self, 
        X: pd.DataFrame, 
        n_features: int = 1000
    ) -> List[str]:
        """
        Select top N features by variance.
        
        Args:
            X: Feature matrix
            n_features: Number of top variance features to select
            
        Returns:
            List of selected feature names
        """
        variances = X.var()
        top_variance_features = variances.nlargest(n_features).index.tolist()
        
        logger.info(f"Selected top {len(top_variance_features)} variance features")
        
        return top_variance_features
    
    def get_variance_summary(self, X: pd.DataFrame) -> dict:
        """
        Get summary statistics about feature variances.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary with variance statistics
        """
        variances = X.var()
        
        return {
            'total_features': len(X.columns),
            'mean_variance': variances.mean(),
            'median_variance': variances.median(),
            'min_variance': variances.min(),
            'max_variance': variances.max(),
            'features_below_threshold': (variances < self.variance_threshold).sum(),
            'zero_variance_features': (variances == 0).sum(),
            'low_variance_features': (variances < 0.01).sum()
        }
    
    def plot_variance_distribution(self, X: pd.DataFrame, save_path: Optional[str] = None):
        """
        Plot the distribution of feature variances.
        
        Args:
            X: Feature matrix
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            variances = X.var()
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Histogram of variances
            ax1.hist(variances, bins=50, alpha=0.7, edgecolor='black')
            ax1.axvline(self.variance_threshold, color='red', linestyle='--', 
                       label=f'Threshold: {self.variance_threshold}')
            ax1.set_xlabel('Variance')
            ax1.set_ylabel('Number of Features')
            ax1.set_title('Distribution of Feature Variances')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Box plot of variances (log scale)
            log_variances = np.log10(variances + 1e-10)  # Add small constant to avoid log(0)
            ax2.boxplot(log_variances)
            ax2.set_ylabel('Log10(Variance)')
            ax2.set_title('Feature Variances (Log Scale)')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Variance distribution plot saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available. Cannot create variance plots.")
    
    def identify_problematic_features(self, X: pd.DataFrame) -> dict:
        """
        Identify potentially problematic features.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary with lists of problematic features
        """
        variances = X.var()
        
        return {
            'zero_variance': variances[variances == 0].index.tolist(),
            'low_variance': variances[
                (variances > 0) & (variances < self.variance_threshold)
            ].index.tolist(),
            'very_high_variance': variances[
                variances > variances.quantile(0.99)
            ].index.tolist()
        }
