"""Utility functions for tabml package.

This module provides common utility functions used throughout the TabML
package, including configuration management, memory optimization, and
helper functions for data manipulation.

Functions:
    set_random_seed: Set seeds for reproducibility
    load_config: Load configuration from YAML/JSON
    save_config: Save configuration to YAML/JSON
    reduce_memory_usage: Optimize DataFrame memory usage
    create_folds: Create cross-validation folds
    get_categorical_columns: Identify categorical columns
    get_numeric_columns: Identify numeric columns
    timer: Decorator for timing functions
    log_dataframe_info: Log DataFrame statistics
    save_predictions: Save predictions to CSV
    
Example:
    Common utility usage::
    
        from tabml.utils import set_random_seed, reduce_memory_usage, timer
        
        # Ensure reproducibility
        set_random_seed(42)
        
        # Optimize memory
        df = reduce_memory_usage(df)
        
        # Time a function
        @timer
        def train_model():
            # Training code
            pass
"""

import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import random
import os
from loguru import logger


def set_random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility.
    
    Sets random seeds for Python's random module, NumPy, and PyTorch
    (if available) to ensure reproducible results across runs.
    
    Args:
        seed: Random seed value. Common choices:
            - 42: The answer to everything
            - 0: Simple and clean
            - Your favorite number
            
    Example:
        >>> # At the beginning of your script
        >>> set_random_seed(42)
        >>> 
        >>> # Now all random operations are reproducible
        >>> np.random.rand(3)  # Will be same across runs
        
    Note:
        - Also sets PYTHONHASHSEED environment variable
        - For PyTorch, disables cuDNN benchmarking for determinism
        - Call this before any random operations
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file.
    
    Automatically detects file format based on extension and loads
    configuration data into a dictionary.
    
    Args:
        config_path: Path to configuration file. Supports:
            - .yaml, .yml: YAML format
            - .json: JSON format
            
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If file format is not supported
        
    Example:
        >>> # Load model configuration
        >>> config = load_config('configs/model_config.yaml')
        >>> 
        >>> # Access configuration
        >>> learning_rate = config['training']['learning_rate']
        >>> model_params = config['model']['params']
        
    Note:
        YAML is recommended for human-readable configs with comments
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")


def save_config(config: Dict[str, Any], save_path: Union[str, Path]) -> None:
    """Save configuration to YAML or JSON file.
    
    Saves configuration dictionary to file with automatic format
    detection based on file extension.
    
    Args:
        config: Configuration dictionary to save
        save_path: Output path with extension (.yaml/.yml or .json)
        
    Raises:
        ValueError: If file format is not supported
        
    Example:
        >>> # Save experiment configuration
        >>> config = {
        ...     'model': {'type': 'xgboost', 'n_estimators': 1000},
        ...     'training': {'learning_rate': 0.01, 'early_stopping': 50}
        ... }
        >>> save_config(config, 'experiment_001.yaml')
        
    Note:
        - Creates parent directories if they don't exist
        - YAML format preserves order and is more readable
        - JSON format is more portable
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if save_path.suffix == '.yaml' or save_path.suffix == '.yml':
        with open(save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    elif save_path.suffix == '.json':
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported config format: {save_path.suffix}")


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Reduce memory usage of DataFrame by optimizing dtypes.
    
    Automatically downcasts numeric types to smallest possible dtype
    that can hold the data without loss. Converts object columns to
    categorical for memory efficiency.
    
    Args:
        df: DataFrame to optimize
        verbose: Whether to log memory reduction statistics
        
    Returns:
        Optimized DataFrame with reduced memory footprint
        
    Example:
        >>> # Load large dataset
        >>> df = pd.read_csv('large_dataset.csv')
        >>> print(f"Original size: {df.memory_usage().sum() / 1024**2:.1f} MB")
        >>> 
        >>> # Optimize memory
        >>> df = reduce_memory_usage(df)
        >>> # Memory usage decreased from 1250.5 MB to 312.3 MB (75.0% reduction)
        
    Optimization strategy:
        - int64 → int8/16/32 based on value range
        - float64 → float16/32 based on value range
        - object → category for string columns
        
    Warning:
        - float16 may lose precision for some calculations
        - Consider keeping high-precision columns as float32/64
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != 'object':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
            
    end_mem = df.memory_usage().sum() / 1024**2
    
    if verbose:
        logger.info(f'Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB '
                   f'({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
        
    return df


def create_folds(df: pd.DataFrame, 
                n_folds: int = 5,
                stratify_column: Optional[str] = None,
                random_state: int = 42) -> pd.DataFrame:
    """Create cross-validation folds.
    
    Adds a 'fold' column to the DataFrame with fold assignments for
    cross-validation. Supports stratified splitting for classification.
    
    Args:
        df: DataFrame to split
        n_folds: Number of folds to create
        stratify_column: Column name for stratified splitting.
            Use for classification to maintain class balance.
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with added 'fold' column (values 0 to n_folds-1)
        
    Example:
        >>> # Create 5-fold CV with stratification
        >>> df_with_folds = create_folds(
        ...     df, 
        ...     n_folds=5,
        ...     stratify_column='target'
        ... )
        >>> 
        >>> # Use folds for training
        >>> for fold in range(5):
        ...     train_df = df_with_folds[df_with_folds['fold'] != fold]
        ...     val_df = df_with_folds[df_with_folds['fold'] == fold]
        ...     # Train model on fold
        
    Note:
        - Preserves class distribution in each fold if stratified
        - Fold column is added to a copy, original df unchanged
    """
    from sklearn.model_selection import StratifiedKFold, KFold
    
    df = df.copy()
    df['fold'] = -1
    
    if stratify_column and stratify_column in df.columns:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        for fold, (_, val_idx) in enumerate(skf.split(df, df[stratify_column])):
            df.loc[val_idx, 'fold'] = fold
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        for fold, (_, val_idx) in enumerate(kf.split(df)):
            df.loc[val_idx, 'fold'] = fold
            
    return df


def get_categorical_columns(df: pd.DataFrame, 
                          max_cardinality: int = 50) -> List[str]:
    """Get categorical columns based on dtype and cardinality.
    
    Identifies columns that should be treated as categorical based on
    data type and number of unique values.
    
    Args:
        df: DataFrame to analyze
        max_cardinality: Maximum unique values for numeric columns
            to be considered categorical
            
    Returns:
        List of categorical column names
        
    Detection logic:
        - Object and category dtypes are always categorical
        - Numeric columns with < max_cardinality unique values
        
    Example:
        >>> cat_cols = get_categorical_columns(df, max_cardinality=20)
        >>> print(f"Categorical columns: {cat_cols}")
        >>> 
        >>> # Use for preprocessing
        >>> encoder = OneHotEncoder()
        >>> encoded = encoder.fit_transform(df[cat_cols])
        
    Note:
        Useful for automatic feature type detection in pipelines
    """
    categorical_cols = []
    
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            categorical_cols.append(col)
        elif df[col].nunique() < max_cardinality:
            categorical_cols.append(col)
            
    return categorical_cols


def get_numeric_columns(df: pd.DataFrame, 
                       exclude_binary: bool = False) -> List[str]:
    """Get numeric columns.
    
    Identifies columns with numeric data types, optionally excluding
    binary columns.
    
    Args:
        df: DataFrame to analyze
        exclude_binary: If True, excludes columns with only 2 unique
            values (often binary indicators)
            
    Returns:
        List of numeric column names
        
    Example:
        >>> # Get all numeric columns
        >>> num_cols = get_numeric_columns(df)
        >>> 
        >>> # Exclude binary columns for scaling
        >>> num_cols_to_scale = get_numeric_columns(df, exclude_binary=True)
        >>> scaler = StandardScaler()
        >>> df[num_cols_to_scale] = scaler.fit_transform(df[num_cols_to_scale])
        
    Note:
        Binary columns often don't need scaling and can be excluded
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if exclude_binary:
        numeric_cols = [col for col in numeric_cols if df[col].nunique() > 2]
        
    return numeric_cols


def timer(func):
    """Decorator to time function execution.
    
    Wraps a function to log its execution time. Useful for profiling
    and optimization.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function that logs execution time
        
    Example:
        >>> @timer
        ... def train_model(X, y):
        ...     # Time-consuming training code
        ...     model = XGBClassifier()
        ...     model.fit(X, y)
        ...     return model
        >>> 
        >>> model = train_model(X_train, y_train)
        >>> # Logs: "train_model took 45.23 seconds"
        
    Note:
        - Uses loguru for logging
        - Preserves function metadata with @wraps
        - Times wall-clock time, not CPU time
    """
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper


def log_dataframe_info(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Log detailed DataFrame information.
    
    Logs comprehensive information about a DataFrame including shape,
    memory usage, column types, and missing values.
    
    Args:
        df: DataFrame to analyze
        name: Name to use in log output for identification
        
    Example:
        >>> log_dataframe_info(train_df, "Training Data")
        >>> # Logs:
        >>> # Training Data Info:
        >>> # Shape: (10000, 50)
        >>> # Memory usage: 45.23 MB
        >>> # Columns: ['id', 'feature1', 'feature2', ...]
        >>> # Dtypes:
        >>> # float64    35
        >>> # int64      10
        >>> # object      5
        >>> # Missing values:
        >>> # feature1    150
        >>> # feature2     89
        
    Note:
        Useful for debugging and understanding data at various
        pipeline stages
    """
    logger.info(f"\n{name} Info:")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Dtypes:\n{df.dtypes.value_counts()}")
    
    missing = df.isnull().sum()
    if missing.sum() > 0:
        logger.info(f"Missing values:\n{missing[missing > 0]}")


def save_predictions(predictions: np.ndarray,
                    test_ids: pd.Series,
                    output_path: Union[str, Path],
                    target_name: str = 'target') -> pd.DataFrame:
    """Save predictions to CSV file.
    
    Creates a submission-ready CSV file with ID and prediction columns,
    commonly used for competition submissions.
    
    Args:
        predictions: Array of predicted values
        test_ids: Series of test IDs corresponding to predictions
        output_path: Path where CSV file will be saved
        target_name: Name for the prediction column
        
    Returns:
        DataFrame that was saved (for verification)
        
    Example:
        >>> # After making predictions
        >>> predictions = model.predict(X_test)
        >>> test_ids = test_df['id']
        >>> 
        >>> # Save submission
        >>> submission = save_predictions(
        ...     predictions,
        ...     test_ids,
        ...     'submissions/submission_v1.csv',
        ...     target_name='price'
        ... )
        >>> print(submission.head())
        
    Note:
        - Creates parent directories if they don't exist
        - Uses ID column name from test_ids.name if available
        - Logs save location for tracking
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    submission = pd.DataFrame({
        test_ids.name or 'id': test_ids,
        target_name: predictions
    })
    
    submission.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")
    
    return submission