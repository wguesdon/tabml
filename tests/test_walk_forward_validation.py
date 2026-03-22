"""Tests for walk-forward validation and time series cross-validation."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from tabml.evaluate import CrossValidator


class TestWalkForwardValidation:
    """Test walk-forward validation functionality."""
    
    @pytest.fixture
    def time_series_data(self):
        """Create sample time series data."""
        np.random.seed(42)
        n_samples = 100
        
        # Create time index
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
        
        # Create features with some time dependency
        X = pd.DataFrame({
            'feature1': np.sin(np.arange(n_samples) * 0.1) + np.random.normal(0, 0.1, n_samples),
            'feature2': np.cos(np.arange(n_samples) * 0.1) + np.random.normal(0, 0.1, n_samples),
            'feature3': np.random.normal(0, 1, n_samples),
            'time_idx': np.arange(n_samples)
        }, index=dates)
        
        # Create target with trend and seasonality
        trend = np.arange(n_samples) * 0.1
        seasonality = 5 * np.sin(np.arange(n_samples) * 2 * np.pi / 7)  # Weekly pattern
        noise = np.random.normal(0, 0.5, n_samples)
        y = pd.Series(trend + seasonality + noise, index=dates, name='target')
        
        return X, y
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return LinearRegression()
    
    def test_walk_forward_expanding_window(self, time_series_data, simple_model):
        """Test walk-forward validation with expanding window."""
        X, y = time_series_data
        cv = CrossValidator()
        
        results = cv.walk_forward_validation(
            model=simple_model,
            X=X,
            y=y,
            initial_train_size=0.5,  # 50% initial training
            step_size=5,
            forecast_horizon=1,
            expanding_window=True,
            metric='rmse'
        )
        
        # Check results structure
        assert results['method'] == 'walk_forward'
        assert results['expanding_window'] is True
        assert results['metric'] == 'rmse'
        assert results['n_folds'] > 0
        assert 'overall_score' in results
        assert 'mean_score' in results
        assert 'std_score' in results
        assert len(results['fold_scores']) == results['n_folds']
        assert len(results['predictions']) == len(results['actuals'])
        
        # Check that train size increases in expanding window
        train_sizes = [f['train_size'] for f in results['fold_details']]
        assert all(train_sizes[i] <= train_sizes[i+1] for i in range(len(train_sizes)-1))
    
    def test_walk_forward_sliding_window(self, time_series_data, simple_model):
        """Test walk-forward validation with sliding window."""
        X, y = time_series_data
        cv = CrossValidator()
        
        results = cv.walk_forward_validation(
            model=simple_model,
            X=X,
            y=y,
            initial_train_size=30,  # Fixed 30 samples
            step_size=1,
            forecast_horizon=1,
            expanding_window=False,
            metric='mae'
        )
        
        # Check results
        assert results['method'] == 'walk_forward'
        assert results['expanding_window'] is False
        assert results['metric'] == 'mae'
        
        # Check that train size remains constant in sliding window
        train_sizes = [f['train_size'] for f in results['fold_details']]
        assert len(set(train_sizes)) == 1  # All sizes should be the same
        assert train_sizes[0] == 30
    
    def test_walk_forward_multi_step_forecast(self, time_series_data, simple_model):
        """Test walk-forward validation with multi-step forecast."""
        X, y = time_series_data
        cv = CrossValidator()
        
        results = cv.walk_forward_validation(
            model=simple_model,
            X=X,
            y=y,
            initial_train_size=50,
            step_size=5,
            forecast_horizon=5,  # 5-step ahead forecast
            expanding_window=True,
            metric='rmse'
        )
        
        # Check that each fold predicts 5 steps ahead
        test_sizes = [f['test_size'] for f in results['fold_details']]
        assert all(size == 5 for size in test_sizes)
    
    def test_time_series_cv(self, time_series_data, simple_model):
        """Test time series cross-validation."""
        X, y = time_series_data
        cv = CrossValidator()
        
        results = cv.time_series_cv(
            model=simple_model,
            X=X,
            y=y,
            n_splits=5,
            test_size=10,
            gap=0,
            metric='rmse'
        )
        
        # Check results structure
        assert results['method'] == 'time_series_split'
        assert results['n_splits'] == 5
        assert results['test_size'] == 10
        assert len(results['fold_scores']) == 5
        assert 'mean_score' in results
        assert 'std_score' in results
    
    def test_time_series_cv_with_gap(self, time_series_data, simple_model):
        """Test time series CV with gap between train and test."""
        X, y = time_series_data
        cv = CrossValidator()
        
        results = cv.time_series_cv(
            model=simple_model,
            X=X,
            y=y,
            n_splits=3,
            test_size=10,
            gap=5,  # 5 period gap
            metric='mae'
        )
        
        assert results['gap'] == 5
        assert len(results['fold_scores']) == 3
    
    def test_metric_calculation(self):
        """Test metric calculation function."""
        cv = CrossValidator()
        
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.2])
        
        # Test RMSE
        rmse = cv._calculate_metric(y_true, y_pred, 'rmse')
        assert isinstance(rmse, float)
        assert rmse > 0
        
        # Test MAE
        mae = cv._calculate_metric(y_true, y_pred, 'mae')
        assert isinstance(mae, float)
        assert mae > 0
        
        # Test R2
        r2 = cv._calculate_metric(y_true, y_pred, 'r2')
        assert isinstance(r2, float)
        assert -1 <= r2 <= 1
    
    def test_classification_metrics(self):
        """Test classification metrics in walk-forward validation."""
        cv = CrossValidator()
        
        # Create classification data
        np.random.seed(42)
        n_samples = 100
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples)
        })
        y = pd.Series(np.random.binomial(1, 0.5, n_samples))
        
        # Simple logistic regression model
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42)
        
        results = cv.walk_forward_validation(
            model=model,
            X=X,
            y=y,
            initial_train_size=50,
            step_size=5,
            forecast_horizon=5,
            metric='accuracy'
        )
        
        assert results['metric'] == 'accuracy'
        assert 0 <= results['overall_score'] <= 1
    
    def test_invalid_parameters(self, time_series_data, simple_model):
        """Test validation with invalid parameters."""
        X, y = time_series_data
        cv = CrossValidator()
        
        # Test with initial_train_size too large
        with pytest.raises(ValueError):
            cv.walk_forward_validation(
                model=simple_model,
                X=X,
                y=y,
                initial_train_size=0.99,  # Too large
                forecast_horizon=5
            )
    
    def test_model_compatibility(self, time_series_data):
        """Test with different model types."""
        X, y = time_series_data
        cv = CrossValidator()
        
        # Test with RandomForest
        rf_model = RandomForestRegressor(n_estimators=10, random_state=42)
        
        results = cv.walk_forward_validation(
            model=rf_model,
            X=X,
            y=y,
            initial_train_size=30,
            step_size=10,
            forecast_horizon=1,
            metric='rmse'
        )
        
        assert results['n_folds'] > 0
        assert 'overall_score' in results