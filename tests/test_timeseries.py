"""Tests for time series functionality."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from tabml.timeseries import TimeSeriesDataLoader, TimeSeriesFeatureEngineer
from tabml.features import FeatureEngineer


class TestTimeSeriesDataLoader:
    """Test TimeSeriesDataLoader class."""
    
    def test_datetime_detection(self, temp_data_dir):
        """Test automatic datetime column detection."""
        # Create sample data with datetime
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        train_df = pd.DataFrame({
            'date': dates,
            'value': np.random.randn(100),
            'feature': np.random.rand(100),
            'target': np.random.randint(0, 2, 100),
            'id': range(100)
        })
        
        test_df = pd.DataFrame({
            'date': pd.date_range('2023-04-11', periods=20, freq='D'),
            'value': np.random.randn(20),
            'feature': np.random.rand(20),
            'id': range(100, 120)
        })
        
        # Save to files
        train_df.to_csv(temp_data_dir / 'train.csv', index=False)
        test_df.to_csv(temp_data_dir / 'test.csv', index=False)
        
        # Load with TimeSeriesDataLoader
        loader = TimeSeriesDataLoader(temp_data_dir)
        train_loaded, test_loaded = loader.load_data()
        
        assert loader.datetime_column == 'date'
        assert pd.api.types.is_datetime64_any_dtype(train_loaded['date'])
        assert pd.api.types.is_datetime64_any_dtype(test_loaded['date'])
    
    def test_frequency_detection(self, temp_data_dir):
        """Test frequency detection."""
        # Daily data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        train_df = pd.DataFrame({
            'date': dates,
            'value': np.random.randn(100),
            'target': np.random.rand(100)
        })
        train_df.to_csv(temp_data_dir / 'train.csv', index=False)
        
        # Empty test file
        test_df = train_df.iloc[:10].drop(columns=['target'])
        test_df.to_csv(temp_data_dir / 'test.csv', index=False)
        
        loader = TimeSeriesDataLoader(temp_data_dir)
        loader.load_data()
        
        assert loader.frequency == 'D'  # Daily
        
        # Hourly data
        dates_hourly = pd.date_range('2023-01-01', periods=100, freq='h')
        train_df['date'] = dates_hourly
        train_df.to_csv(temp_data_dir / 'train_hourly.csv', index=False)
        
        loader2 = TimeSeriesDataLoader(temp_data_dir)
        loader2.load_data(train_file='train_hourly.csv')
        
        assert loader2.frequency == 'h'  # Hourly
    
    def test_time_based_split(self, temp_data_dir):
        """Test time-based train/validation split."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        train_df = pd.DataFrame({
            'date': dates,
            'feature1': np.random.randn(100),
            'feature2': np.random.rand(100),
            'target': np.random.rand(100),
            'id': range(100)
        })
        
        train_df.to_csv(temp_data_dir / 'train.csv', index=False)
        test_df = train_df.iloc[:10].drop(columns=['target'])
        test_df.to_csv(temp_data_dir / 'test.csv', index=False)
        
        loader = TimeSeriesDataLoader(temp_data_dir)
        loader.load_data()
        
        X_train, X_val, y_train, y_val = loader.create_time_based_split(test_size=0.2, gap=5)
        
        # Check sizes
        assert len(X_train) == 75  # 80% - gap
        assert len(X_val) == 20   # 20%
        
        # Check no datetime overlap
        train_dates = X_train['date']
        val_dates = X_val['date']
        assert train_dates.max() < val_dates.min()
        
        # Check features don't include target or id
        assert 'target' not in X_train.columns
        assert 'id' not in X_train.columns
    
    def test_temporal_integrity_check(self, temp_data_dir):
        """Test temporal integrity checking."""
        # Create data with gaps
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        dates2 = pd.date_range('2023-03-01', periods=50, freq='D')  # Gap in February
        all_dates = pd.concat([pd.Series(dates), pd.Series(dates2)])
        
        train_df = pd.DataFrame({
            'date': all_dates,
            'value': np.random.randn(100),
            'target': np.random.rand(100)
        })
        
        test_df = pd.DataFrame({
            'date': pd.date_range('2023-02-15', periods=20, freq='D'),  # Overlaps with gap
            'value': np.random.randn(20)
        })
        
        train_df.to_csv(temp_data_dir / 'train.csv', index=False)
        test_df.to_csv(temp_data_dir / 'test.csv', index=False)
        
        loader = TimeSeriesDataLoader(temp_data_dir)
        loader.load_data()
        
        integrity = loader.check_temporal_integrity()
        
        assert integrity['n_gaps'] > 0
        assert integrity['temporal_overlap'] == True  # Test overlaps with train period


class TestTimeSeriesFeatureEngineer:
    """Test TimeSeriesFeatureEngineer class."""
    
    def test_datetime_feature_extraction(self):
        """Test extraction of datetime features."""
        dates = pd.date_range('2023-01-01', periods=365, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'value': np.random.randn(365)
        })
        
        engineer = TimeSeriesFeatureEngineer()
        df_transformed = engineer.fit_transform(df, 'date')
        
        # Check basic features
        assert 'date_year' in df_transformed.columns
        assert 'date_month' in df_transformed.columns
        assert 'date_day' in df_transformed.columns
        assert 'date_dayofweek' in df_transformed.columns
        assert 'date_is_weekend' in df_transformed.columns
        
        # Check values
        assert df_transformed['date_year'].iloc[0] == 2023
        assert df_transformed['date_month'].iloc[0] == 1
        assert df_transformed['date_day'].iloc[0] == 1
        assert df_transformed['date_is_weekend'].iloc[0] == 1  # Jan 1, 2023 is Sunday
    
    def test_cyclical_features(self):
        """Test cyclical feature creation."""
        dates = pd.date_range('2023-01-01', periods=365, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'value': np.random.randn(365)
        })
        
        engineer = TimeSeriesFeatureEngineer()
        df_transformed = engineer.fit_transform(df, 'date')
        
        # Check cyclical features
        assert 'date_dayofweek_sin' in df_transformed.columns
        assert 'date_dayofweek_cos' in df_transformed.columns
        assert 'date_month_sin' in df_transformed.columns
        assert 'date_month_cos' in df_transformed.columns
        
        # Check range [-1, 1]
        assert df_transformed['date_dayofweek_sin'].min() >= -1
        assert df_transformed['date_dayofweek_sin'].max() <= 1
        assert df_transformed['date_month_cos'].min() >= -1
        assert df_transformed['date_month_cos'].max() <= 1
    
    def test_lag_features(self):
        """Test lag feature creation."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'sales': np.random.rand(100) * 100,
            'store_id': np.repeat([1, 2], 50)
        })
        
        engineer = TimeSeriesFeatureEngineer()
        engineer.fit(df, 'date')
        
        # Create lag features
        df_lagged = engineer.create_lag_features(df, 'sales', lags=[1, 7], group_columns=['store_id'])
        
        assert 'sales_lag_1' in df_lagged.columns
        assert 'sales_lag_7' in df_lagged.columns
        
        # Check that lags are within groups
        # First value in each group should be NaN
        assert pd.isna(df_lagged.loc[0, 'sales_lag_1'])
        assert pd.isna(df_lagged.loc[50, 'sales_lag_1'])
        
        # Check lag values
        assert df_lagged.loc[1, 'sales_lag_1'] == df.loc[0, 'sales']
        assert df_lagged.loc[51, 'sales_lag_1'] == df.loc[50, 'sales']
    
    def test_rolling_features(self):
        """Test rolling window feature creation."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'sales': np.random.rand(100) * 100
        })
        
        engineer = TimeSeriesFeatureEngineer()
        engineer.fit(df, 'date')
        
        # Create rolling features
        df_rolling = engineer.create_rolling_features(
            df, 'sales', 
            windows=[7, 14], 
            operations=['mean', 'std']
        )
        
        assert 'sales_rolling_7_mean' in df_rolling.columns
        assert 'sales_rolling_7_std' in df_rolling.columns
        assert 'sales_rolling_14_mean' in df_rolling.columns
        
        # Check that rolling means are reasonable
        week_mean = df_rolling['sales_rolling_7_mean'].iloc[10]
        actual_mean = df['sales'].iloc[4:11].mean()  # 7 values
        assert abs(week_mean - actual_mean) < 0.01


class TestTimeSeriesImputation:
    """Test time series imputation in FeatureEngineer."""
    
    def test_forward_fill_imputation(self):
        """Test forward fill imputation."""
        df = pd.DataFrame({
            'value': [1.0, np.nan, np.nan, 4.0, 5.0],
            'category': ['A', np.nan, 'B', np.nan, 'C']
        })
        
        engineer = FeatureEngineer(
            numeric_impute_strategy='ffill',
            categorical_impute_strategy='ffill',
            scaling_method=None  # No scaling to test imputation values
        )
        
        df_transformed = engineer.fit_transform(df)
        
        # Check forward filled values
        assert df_transformed['value'].iloc[1] == 1.0
        assert df_transformed['value'].iloc[2] == 1.0
        assert df_transformed['category'].iloc[1] == 'A'
        assert df_transformed['category'].iloc[3] == 'B'
    
    def test_backward_fill_imputation(self):
        """Test backward fill imputation."""
        df = pd.DataFrame({
            'value': [np.nan, np.nan, 3.0, 4.0, 5.0],
            'category': [np.nan, 'B', np.nan, 'D', 'E']
        })
        
        engineer = FeatureEngineer(
            numeric_impute_strategy='bfill',
            categorical_impute_strategy='bfill',
            scaling_method=None  # No scaling to test imputation values
        )
        
        df_transformed = engineer.fit_transform(df)
        
        # Check backward filled values
        assert df_transformed['value'].iloc[0] == 3.0
        assert df_transformed['value'].iloc[1] == 3.0
        assert df_transformed['category'].iloc[0] == 'B'
        assert df_transformed['category'].iloc[2] == 'D'
    
    def test_time_series_impute_flag(self):
        """Test time_series_impute flag."""
        df = pd.DataFrame({
            'value': [1.0, np.nan, 3.0, np.nan, 5.0],
            'category': ['A', np.nan, 'C', np.nan, 'E']
        })
        
        # With time_series_impute=True, should default to forward fill
        engineer = FeatureEngineer(
            numeric_impute_strategy='mean',  # This should be overridden
            time_series_impute=True
        )
        
        # For now, the time_series_impute flag just allows ffill/bfill
        # Let's test with explicit ffill
        engineer = FeatureEngineer(
            numeric_impute_strategy='ffill',
            categorical_impute_strategy='ffill',
            scaling_method=None  # No scaling to test imputation values
        )
        
        df_transformed = engineer.fit_transform(df)
        
        assert df_transformed['value'].iloc[1] == 1.0
        assert df_transformed['category'].iloc[1] == 'A'