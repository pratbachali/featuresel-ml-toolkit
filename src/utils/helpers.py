"""
Utility Helper Functions

Common utility functions used across the ML pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import json
import joblib
import logging

logger = logging.getLogger(__name__)


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Output file path
    """
    path_obj = Path(file_path)
    ensure_directory(path_obj.parent)
    
    with open(path_obj, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    logger.info(f"JSON data saved to {file_path}")


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load dictionary from JSON file.
    
    Args:
        file_path: Input file path
        
    Returns:
        Loaded dictionary
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"JSON data loaded from {file_path}")
    return data


def save_model(model: Any, file_path: Union[str, Path]) -> None:
    """
    Save model using joblib.
    
    Args:
        model: Model to save
        file_path: Output file path
    """
    path_obj = Path(file_path)
    ensure_directory(path_obj.parent)
    
    joblib.dump(model, path_obj)
    logger.info(f"Model saved to {file_path}")


def load_model(file_path: Union[str, Path]) -> Any:
    """
    Load model using joblib.
    
    Args:
        file_path: Input file path
        
    Returns:
        Loaded model
    """
    model = joblib.load(file_path)
    logger.info(f"Model loaded from {file_path}")
    return model


def get_memory_usage(df: pd.DataFrame) -> Dict[str, str]:
    """
    Get memory usage information for a DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with memory usage info
    """
    memory_usage = df.memory_usage(deep=True)
    total_mb = memory_usage.sum() / 1024**2
    
    return {
        'total_memory_mb': f"{total_mb:.2f}",
        'shape': str(df.shape),
        'dtypes': df.dtypes.value_counts().to_dict()
    }


def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce memory usage of DataFrame by optimizing dtypes.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with optimized dtypes
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                    
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
    
    end_mem = df.memory_usage().sum() / 1024**2
    logger.info(f"Memory usage reduced from {start_mem:.2f}MB to {end_mem:.2f}MB "
                f"({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")
    
    return df


def validate_input_data(
    X: pd.DataFrame, 
    y: pd.Series, 
    min_samples: int = 10
) -> None:
    """
    Validate input data for ML pipeline.
    
    Args:
        X: Feature matrix
        y: Target vector
        min_samples: Minimum number of samples required
        
    Raises:
        ValueError: If validation fails
    """
    if len(X) != len(y):
        raise ValueError("X and y must have same number of samples")
    
    if len(X) < min_samples:
        raise ValueError(f"Dataset too small: {len(X)} samples, minimum {min_samples} required")
    
    if X.empty:
        raise ValueError("Feature matrix is empty")
    
    if y.empty:
        raise ValueError("Target vector is empty")
    
    if y.nunique() < 2:
        raise ValueError("Target must have at least 2 classes")


def create_summary_report(
    results: Dict[str, Any], 
    output_path: Union[str, Path]
) -> None:
    """
    Create a summary report from results dictionary.
    
    Args:
        results: Results dictionary
        output_path: Output file path
    """
    ensure_directory(Path(output_path).parent)
    
    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ML Pipeline Results Summary</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .metric {{ background-color: #e8f4fd; }}
        </style>
    </head>
    <body>
        <h1>ML Pipeline Results Summary</h1>
        <h2>Dataset Information</h2>
        <p><strong>Samples:</strong> {results.get('n_samples', 'N/A')}</p>
        <p><strong>Features:</strong> {results.get('n_features', 'N/A')}</p>
        <p><strong>Classes:</strong> {results.get('n_classes', 'N/A')}</p>
        
        <h2>Best Model Performance</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
    """
    
    if 'best_model_results' in results:
        for metric, value in results['best_model_results'].items():
            if isinstance(value, float):
                value = f"{value:.4f}"
            html_content += f"<tr><td>{metric}</td><td class='metric'>{value}</td></tr>"
    
    html_content += """
        </table>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Summary report saved to {output_path}")


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"


def check_package_versions() -> Dict[str, str]:
    """
    Check versions of key packages.
    
    Returns:
        Dictionary of package versions
    """
    import pkg_resources
    
    packages = [
        'pandas', 'numpy', 'scikit-learn', 
        'matplotlib', 'seaborn', 'joblib'
    ]
    
    versions = {}
    for package in packages:
        try:
            version = pkg_resources.get_distribution(package).version
            versions[package] = version
        except pkg_resources.DistributionNotFound:
            versions[package] = 'Not installed'
    
    return versions
