"""Tests for DataProcessor class."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from tabml.preprocessing import DataProcessor


class TestDataProcessor:
    """Test DataProcessor functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data with various column types."""
        np.random.seed(42)
        n_samples = 100
        
        # Create diverse dataset
        data = pd.DataFrame({
            # Numeric columns
            'numeric1': np.random.normal(0, 1, n_samples),
            'numeric2': np.random.uniform(0, 100, n_samples),
            'numeric_with_missing': np.where(
                np.random.random(n_samples) > 0.8, 
                np.nan, 
                np.random.normal(10, 2, n_samples)
            ),
            
            # Categorical columns
            'category_low': np.random.choice(['A', 'B', 'C'], n_samples),
            'category_medium': np.random.choice([f'Cat_{i}' for i in range(10)], n_samples),
            'category_missing': pd.Series(
                np.where(
                    np.random.random(n_samples) > 0.9,
                    None,
                    np.random.choice(['X', 'Y', 'Z'], n_samples)
                )
            ),
            
            # High cardinality column (create unique values to exceed threshold)
            'high_cardinality': [f'ID_{i}' for i in range(n_samples)],
            
            # Text column
            'text_feature': [
                f"This is text sample {i} with some words" if i % 5 != 0 else None
                for i in range(n_samples)
            ],
            
            # Datetime column
            'datetime': pd.date_range(start='2023-01-01', periods=n_samples, freq='D'),
            
            # Target
            'target': np.random.randint(0, 2, n_samples)
        })
        
        return data
    
    @pytest.fixture
    def simple_config(self):
        """Simple configuration for testing."""
        return {
            'categorical_encoding': {
                'method': 'onehot',
                'handle_unknown': 'ignore',
                'high_cardinality_threshold': 50,
                'high_cardinality_method': 'hashing'
            },
            'text_processing': {
                'method': 'tfidf',
                'max_features': 20,
                'ngram_range': (1, 2),
                'min_df': 1,  # Lower for test data
                'max_df': 1.0
            },
            'scaling': {
                'method': 'standard'
            },
            'imputation': {
                'numeric_strategy': 'median',
                'categorical_strategy': 'most_frequent',
                'text_strategy': 'constant',
                'constant_fill_value': 'missing'
            }
        }
    
    def test_initialization(self, simple_config):
        """Test DataProcessor initialization."""
        # With config - check that provided values are set
        processor = DataProcessor(config=simple_config)
        assert processor.config['categorical_encoding']['method'] == simple_config['categorical_encoding']['method']
        assert processor.config['text_processing']['max_features'] == simple_config['text_processing']['max_features']
        assert not processor.is_fitted
        
        # Without config (default)
        processor_default = DataProcessor()
        assert processor_default.config is not None
        assert 'categorical_encoding' in processor_default.config
    
    def test_column_type_detection(self, sample_data):
        """Test automatic column type detection."""
        processor = DataProcessor()
        
        # Remove target column for testing
        X = sample_data.drop(columns=['target'])
        column_types = processor.detect_column_types(X)
        
        # Check numeric columns
        assert 'numeric1' in column_types['numeric']
        assert 'numeric2' in column_types['numeric']
        assert 'numeric_with_missing' in column_types['numeric']
        
        # Check categorical columns
        assert 'category_low' in column_types['categorical']
        assert 'category_medium' in column_types['categorical']
        
        # Check high cardinality
        # With 100 unique values (ID_0 to ID_99), it should be detected as high cardinality
        assert 'high_cardinality' in column_types['high_cardinality']
        
        # Check text
        assert 'text_feature' in column_types['text']
        
        # Check datetime
        assert 'datetime' in column_types['datetime']
    
    def test_fit_transform(self, sample_data, simple_config):
        """Test fit and transform process."""
        processor = DataProcessor(config=simple_config)
        
        X = sample_data.drop(columns=['target'])
        y = sample_data['target']
        
        # Fit
        processor.fit(X, y)
        assert processor.is_fitted
        
        # Transform
        X_transformed = processor.transform(X)
        
        # Check no missing values after imputation
        assert X_transformed.isnull().sum().sum() == 0
        
        # Check shape increased due to encoding
        assert X_transformed.shape[1] > X.shape[1]
        
        # Check all values are numeric
        assert all(pd.api.types.is_numeric_dtype(X_transformed[col]) 
                  for col in X_transformed.columns)
    
    def test_categorical_encoding_methods(self, sample_data):
        """Test different categorical encoding methods."""
        X = sample_data[['category_low', 'category_medium', 'numeric1']]
        
        # Test OneHot encoding
        config_onehot = {
            'categorical_encoding': {'method': 'onehot'},
            'scaling': {'method': 'none'},
            'text_processing': {'method': 'none'}
        }
        processor_oh = DataProcessor(config=config_onehot)
        X_oh = processor_oh.fit_transform(X)
        
        # Should have more columns due to one-hot encoding
        assert X_oh.shape[1] > X.shape[1]
        
        # Test Label encoding
        config_label = {
            'categorical_encoding': {'method': 'label'},
            'scaling': {'method': 'none'},
            'text_processing': {'method': 'none'}
        }
        processor_label = DataProcessor(config=config_label)
        X_label = processor_label.fit_transform(X)
        
        # Should have same number of columns
        assert X_label.shape[1] == X.shape[1]
        
        # Check categorical columns are now numeric
        assert pd.api.types.is_numeric_dtype(X_label['category_low'])
        assert pd.api.types.is_numeric_dtype(X_label['category_medium'])
    
    def test_text_processing(self, sample_data):
        """Test text vectorization."""
        X = sample_data[['text_feature']]
        
        # Test TF-IDF
        config_tfidf = {
            'text_processing': {
                'method': 'tfidf',
                'max_features': 10,
                'ngram_range': (1, 1)
            },
            'categorical_encoding': {'method': 'none'},
            'scaling': {'method': 'none'}
        }
        processor = DataProcessor(config=config_tfidf)
        X_transformed = processor.fit_transform(X)
        
        # Should have max_features columns
        assert X_transformed.shape[1] <= 10
        
        # All columns should start with original column name
        assert all(col.startswith('text_feature_') for col in X_transformed.columns)
    
    def test_scaling_methods(self, sample_data):
        """Test different scaling methods."""
        X = sample_data[['numeric1', 'numeric2']]
        
        # Test standard scaling
        config_standard = {
            'scaling': {'method': 'standard'},
            'categorical_encoding': {'method': 'none'},
            'text_processing': {'method': 'none'}
        }
        processor_std = DataProcessor(config=config_standard)
        X_std = processor_std.fit_transform(X)
        
        # Check mean ~ 0 and std ~ 1
        assert abs(X_std['numeric1'].mean()) < 0.1
        assert abs(X_std['numeric1'].std() - 1) < 0.1
        
        # Test minmax scaling
        config_minmax = {
            'scaling': {
                'method': 'minmax',
                'feature_range': (0, 1)
            },
            'categorical_encoding': {'method': 'none'},
            'text_processing': {'method': 'none'}
        }
        processor_mm = DataProcessor(config=config_minmax)
        X_mm = processor_mm.fit_transform(X)
        
        # Check values in [0, 1]
        assert X_mm.min().min() >= -0.001  # Small tolerance
        assert X_mm.max().max() <= 1.001
    
    def test_missing_value_imputation(self, sample_data):
        """Test missing value imputation strategies."""
        X = sample_data[['numeric_with_missing', 'category_missing']]
        
        # Test with indicators
        config = {
            'imputation': {
                'numeric_strategy': 'median',
                'categorical_strategy': 'most_frequent',
                'add_indicator': True
            },
            'categorical_encoding': {'method': 'label'},
            'scaling': {'method': 'none'},
            'text_processing': {'method': 'none'}
        }
        processor = DataProcessor(config=config)
        X_transformed = processor.fit_transform(X)
        
        # Check no missing values
        assert X_transformed.isnull().sum().sum() == 0
        
        # Check indicator columns were added
        assert 'numeric_with_missing_was_missing' in X_transformed.columns
        assert 'category_missing_was_missing' in X_transformed.columns
    
    def test_high_cardinality_handling(self, sample_data):
        """Test high cardinality column handling."""
        X = sample_data[['high_cardinality']]
        
        # Test hashing encoder
        config_hash = {
            'categorical_encoding': {
                'method': 'onehot',
                'high_cardinality_threshold': 50,
                'high_cardinality_method': 'hashing'
            },
            'scaling': {'method': 'none'},
            'text_processing': {'method': 'none'}
        }
        processor = DataProcessor(config=config_hash)
        X_transformed = processor.fit_transform(X)
        
        # Should have multiple columns but less than unique values
        assert X_transformed.shape[1] > 1
        assert X_transformed.shape[1] < X['high_cardinality'].nunique()
    
    def test_datetime_feature_extraction(self, sample_data):
        """Test datetime feature extraction."""
        X = sample_data[['datetime']]
        
        config = {
            'datetime_features': True,
            'categorical_encoding': {'method': 'none'},
            'scaling': {'method': 'none'},
            'text_processing': {'method': 'none'}
        }
        processor = DataProcessor(config=config)
        X_transformed = processor.fit_transform(X)
        
        # Check datetime features were extracted
        expected_features = ['year', 'month', 'day', 'dayofweek', 'quarter', 
                           'is_weekend', 'month_sin', 'month_cos', 'day_sin', 'day_cos']
        
        for feature in expected_features:
            assert any(f'datetime_{feature}' in col for col in X_transformed.columns)
        
        # Original datetime column should be dropped
        assert 'datetime' not in X_transformed.columns
    
    def test_data_leakage_prevention(self, sample_data):
        """Test that no data leakage occurs between train and test."""
        # Split data
        train_idx = sample_data.index[:80]
        test_idx = sample_data.index[80:]
        
        X_train = sample_data.loc[train_idx].drop(columns=['target'])
        X_test = sample_data.loc[test_idx].drop(columns=['target'])
        y_train = sample_data.loc[train_idx, 'target']
        
        # Create processor with target encoding
        config = {
            'categorical_encoding': {
                'method': 'target',
                'high_cardinality_method': 'target'
            }
        }
        processor = DataProcessor(config=config)
        
        # Fit on train only
        processor.fit(X_train, y_train)
        
        # Transform both
        X_train_transformed = processor.transform(X_train)
        X_test_transformed = processor.transform(X_test)
        
        # Both should be transformed successfully
        assert X_train_transformed.shape[0] == len(X_train)
        assert X_test_transformed.shape[0] == len(X_test)
        assert X_train_transformed.shape[1] == X_test_transformed.shape[1]
    
    def test_transform_with_missing_columns(self, sample_data):
        """Test transform when some columns are missing."""
        processor = DataProcessor()
        
        # Fit on full data
        X = sample_data.drop(columns=['target'])
        processor.fit(X)
        
        # Transform with missing column
        X_subset = X.drop(columns=['category_medium'])
        X_transformed = processor.transform(X_subset)
        
        # Should still work
        assert X_transformed.shape[0] == len(X_subset)
    
    def test_config_save_load(self, tmp_path, simple_config):
        """Test saving and loading configuration."""
        # Create processor with config
        processor = DataProcessor(config=simple_config)
        
        # Save config
        config_path = tmp_path / "config.json"
        processor.save_config(str(config_path))
        
        # Load config
        processor_loaded = DataProcessor.load_config(str(config_path))
        
        # Check config is the same
        assert processor_loaded.config == processor.config
    
    def test_get_feature_names_out(self, sample_data):
        """Test getting output feature names."""
        processor = DataProcessor()
        X = sample_data.drop(columns=['target'])
        
        # Should raise before fitting
        with pytest.raises(ValueError):
            processor.get_feature_names_out()
        
        # Fit and transform
        X_transformed = processor.fit_transform(X)
        
        # Get feature names
        feature_names = processor.get_feature_names_out()
        assert len(feature_names) == X_transformed.shape[1]
        assert feature_names == X_transformed.columns.tolist()
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        processor = DataProcessor()
        X = pd.DataFrame()
        
        # Should handle empty data gracefully
        processor.fit(X)
        X_transformed = processor.transform(X)
        assert X_transformed.shape == (0, 0)