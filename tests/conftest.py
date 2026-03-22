"""Pytest configuration and fixtures."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def sample_classification_data():
    """Create sample classification dataset."""
    np.random.seed(42)
    n_samples = 1000
    
    # Create features
    X = pd.DataFrame({
        'numeric_1': np.random.randn(n_samples),
        'numeric_2': np.random.randn(n_samples) * 2 + 1,
        'categorical_1': np.random.choice(['A', 'B', 'C'], n_samples),
        'categorical_2': np.random.choice(['X', 'Y'], n_samples),
        'id': range(n_samples)
    })
    
    # Create target based on features (with some noise)
    y = ((X['numeric_1'] + X['numeric_2'] > 0) & 
         (X['categorical_1'] != 'C')).astype(int)
    # Add noise more carefully to avoid index mismatches
    noise_mask = np.random.rand(n_samples) < 0.1
    y[noise_mask] = 1 - y[noise_mask]
    
    X['target'] = y
    
    # Split into train and test
    train_idx = np.random.rand(n_samples) < 0.8
    train_df = X[train_idx].reset_index(drop=True)
    test_df = X[~train_idx].drop(columns=['target']).reset_index(drop=True)
    
    return train_df, test_df


@pytest.fixture
def sample_regression_data():
    """Create sample regression dataset."""
    np.random.seed(42)
    n_samples = 1000
    
    # Create features
    X = pd.DataFrame({
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples) * 2,
        'feature_3': np.random.exponential(2, n_samples),
        'categorical': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'id': range(n_samples)
    })
    
    # Create target as combination of features
    y = (2 * X['feature_1'] - 
         1.5 * X['feature_2'] + 
         0.5 * X['feature_3'] + 
         X['categorical'].map({'Low': -1, 'Medium': 0, 'High': 1}) +
         np.random.randn(n_samples) * 0.5)
    
    X['target'] = y
    
    # Split into train and test
    train_idx = np.random.rand(n_samples) < 0.8
    train_df = X[train_idx].reset_index(drop=True)
    test_df = X[~train_idx].drop(columns=['target']).reset_index(drop=True)
    
    return train_df, test_df


@pytest.fixture
def sample_timeseries_data():
    """Create sample time series dataset."""
    np.random.seed(42)
    n_days = 365
    
    # Create date range
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    
    # Create features with temporal patterns
    trend = np.linspace(100, 150, n_days)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n_days) / 7)  # Weekly pattern
    noise = np.random.randn(n_days) * 5
    
    df = pd.DataFrame({
        'date': dates,
        'value': trend + seasonal + noise,
        'day_of_week': dates.dayofweek,
        'month': dates.month,
        'is_weekend': dates.dayofweek.isin([5, 6]).astype(int)
    })
    
    # Add lagged features
    df['value_lag1'] = df['value'].shift(1)
    df['value_lag7'] = df['value'].shift(7)
    df['value_rolling_mean_7'] = df['value'].rolling(7).mean()
    
    # Create target (next day value)
    df['target'] = df['value'].shift(-1)
    
    # Remove rows with NaN
    df = df.dropna()
    
    # Split by time
    split_date = df['date'].quantile(0.8)
    train_df = df[df['date'] < split_date].reset_index(drop=True)
    test_df = df[df['date'] >= split_date].drop(columns=['target']).reset_index(drop=True)
    
    return train_df, test_df


@pytest.fixture
def temp_data_dir():
    """Create temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def create_test_files(temp_data_dir, sample_classification_data):
    """Create test CSV files in temporary directory."""
    train_df, test_df = sample_classification_data
    
    # Save as CSV
    train_path = temp_data_dir / "train.csv"
    test_path = temp_data_dir / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # Also create parquet versions if pyarrow is available
    try:
        import pyarrow
        train_df.to_parquet(temp_data_dir / "train.parquet", index=False)
        test_df.to_parquet(temp_data_dir / "test.parquet", index=False)
    except ImportError:
        pass
    
    # And Excel versions if openpyxl is available
    try:
        import openpyxl
        train_df.to_excel(temp_data_dir / "train.xlsx", index=False)
        test_df.to_excel(temp_data_dir / "test.xlsx", index=False)
    except ImportError:
        pass
    
    return temp_data_dir