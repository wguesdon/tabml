"""Tests for data loading and validation modules."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from tabml.data import DataLoader
from tabml.validation import DataValidator


class TestDataLoader:
    """Test DataLoader class."""
    
    def test_load_csv_data(self, create_test_files):
        """Test loading CSV files."""
        data_dir = create_test_files
        loader = DataLoader(data_dir)
        
        train_df, test_df = loader.load_data()
        
        assert train_df is not None
        assert test_df is not None
        assert loader.target_column == 'target'
        assert loader.id_column == 'id'
        assert 'target' in train_df.columns
        assert 'target' not in test_df.columns
    
    def test_load_parquet_data(self, create_test_files):
        """Test loading Parquet files."""
        pytest.importorskip("pyarrow")
        
        data_dir = create_test_files
        
        # Check if parquet files were created
        if not (data_dir / "train.parquet").exists():
            pytest.skip("Parquet files not created (pyarrow not available during fixture)")
        
        loader = DataLoader(data_dir)
        
        train_df, test_df = loader.load_data(
            train_file="train.parquet",
            test_file="test.parquet"
        )
        
        assert train_df is not None
        assert test_df is not None
        assert train_df.shape[0] > 0
        assert test_df.shape[0] > 0
    
    def test_load_excel_data(self, create_test_files):
        """Test loading Excel files."""
        pytest.importorskip("openpyxl")
        
        data_dir = create_test_files
        
        # Check if excel files were created
        if not (data_dir / "train.xlsx").exists():
            pytest.skip("Excel files not created (openpyxl not available during fixture)")
        
        loader = DataLoader(data_dir)
        
        train_df, test_df = loader.load_data(
            train_file="train.xlsx",
            test_file="test.xlsx"
        )
        
        assert train_df is not None
        assert test_df is not None
        assert train_df.shape[0] > 0
        assert test_df.shape[0] > 0
    
    def test_sample_fraction(self, create_test_files):
        """Test data sampling."""
        data_dir = create_test_files
        loader = DataLoader(data_dir)
        
        # Load full data
        train_full, test_full = loader.load_data()
        
        # Load sampled data
        loader2 = DataLoader(data_dir)
        train_sample, test_sample = loader2.load_data(sample_frac=0.5)
        
        assert train_sample.shape[0] < train_full.shape[0]
        assert test_sample.shape[0] < test_full.shape[0]
        assert abs(train_sample.shape[0] - train_full.shape[0] * 0.5) < train_full.shape[0] * 0.1
    
    def test_nrows_parameter(self, create_test_files):
        """Test limiting rows with nrows."""
        data_dir = create_test_files
        loader = DataLoader(data_dir)
        
        train_df, test_df = loader.load_data(nrows=50)
        
        assert train_df.shape[0] <= 50
        assert test_df.shape[0] <= 50
    
    def test_auto_detect_columns(self, create_test_files):
        """Test automatic detection of target and ID columns."""
        data_dir = create_test_files
        loader = DataLoader(data_dir)
        
        loader.load_data()
        
        assert loader.target_column == 'target'
        assert loader.id_column == 'id'
    
    def test_manual_column_specification(self, create_test_files):
        """Test manual specification of target and ID columns."""
        data_dir = create_test_files
        loader = DataLoader(data_dir)
        
        loader.load_data(target_column='target', id_column='id')
        
        assert loader.target_column == 'target'
        assert loader.id_column == 'id'
    
    def test_train_test_split(self, create_test_files):
        """Test train/validation split."""
        data_dir = create_test_files
        loader = DataLoader(data_dir)
        
        loader.load_data()
        X_train, X_val, y_train, y_val = loader.get_train_test_split()
        
        assert X_train.shape[0] > X_val.shape[0]
        assert len(y_train) == X_train.shape[0]
        assert len(y_val) == X_val.shape[0]
        assert loader.id_column not in X_train.columns
        assert loader.target_column not in X_train.columns
    
    def test_get_test_features(self, create_test_files):
        """Test getting test features."""
        data_dir = create_test_files
        loader = DataLoader(data_dir)
        
        loader.load_data()
        X_test = loader.get_test_features()
        
        assert loader.id_column not in X_test.columns
        assert loader.target_column not in X_test.columns
    
    def test_create_submission(self, create_test_files):
        """Test submission creation."""
        data_dir = create_test_files
        loader = DataLoader(data_dir)
        
        loader.load_data()
        test_size = len(loader.test_data)
        predictions = np.random.rand(test_size)
        
        submission = loader.create_submission(predictions)
        
        assert len(submission) == test_size
        assert loader.id_column in submission.columns
        assert loader.target_column in submission.columns
        assert (data_dir / "submission.csv").exists()
    
    def test_get_data_info(self, create_test_files):
        """Test data info retrieval."""
        data_dir = create_test_files
        loader = DataLoader(data_dir)
        
        loader.load_data()
        info = loader.get_data_info()
        
        assert 'train_shape' in info
        assert 'test_shape' in info
        assert 'target_column' in info
        assert 'id_column' in info
        assert 'numeric_features' in info
        assert 'categorical_features' in info
        assert 'missing_values' in info
        assert 'target_info' in info
    
    def test_file_not_found_error(self, temp_data_dir):
        """Test error handling for missing files."""
        loader = DataLoader(temp_data_dir)
        
        with pytest.raises(FileNotFoundError):
            loader.load_data()
    
    def test_unsupported_format_error(self, temp_data_dir):
        """Test error handling for unsupported formats."""
        # Create a dummy file with unsupported extension
        dummy_file = temp_data_dir / "train.txt"
        dummy_file.write_text("dummy content")
        
        loader = DataLoader(temp_data_dir)
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            loader._load_file(dummy_file)


class TestDataValidator:
    """Test DataValidator class."""
    
    def test_basic_validation(self, sample_classification_data):
        """Test basic data validation."""
        train_df, test_df = sample_classification_data
        validator = DataValidator()
        
        results = validator.validate_data(train_df, test_df, target_column='target')
        
        assert 'basic' in results
        assert 'constant_features' in results
        assert 'duplicate_columns' in results
        assert 'leakage_warnings' in results
    
    def test_detect_duplicate_columns(self):
        """Test detection of duplicate columns."""
        # Create data with duplicate columns
        train_df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [1, 2, 3, 4, 5],  # Duplicate of col1
            'col3': [2, 4, 6, 8, 10],
            'target': [0, 1, 0, 1, 0]
        })
        test_df = train_df.drop(columns=['target'])
        
        validator = DataValidator()
        results = validator.validate_data(train_df, test_df)
        
        assert len(results['duplicate_columns']) > 0
        assert ('col1', 'col2') in results['duplicate_columns'] or ('col2', 'col1') in results['duplicate_columns']
    
    def test_detect_constant_features(self):
        """Test detection of constant features."""
        # Create data with constant feature
        train_df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'constant': [1, 1, 1, 1, 1],  # Constant feature
            'target': [0, 1, 0, 1, 0]
        })
        test_df = train_df.drop(columns=['target'])
        
        validator = DataValidator()
        results = validator.validate_data(train_df, test_df)
        
        assert 'constant' in results['constant_features']
    
    def test_detect_target_leakage(self):
        """Test detection of target leakage."""
        # Create data with leakage
        train_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'target': [0, 1, 0, 1, 0],
            'target_copy': [0, 1, 0, 1, 0],  # Direct copy of target
            'target_negative': [0, -1, 0, -1, 0]  # Negative of target
        })
        test_df = train_df.drop(columns=['target'])
        
        validator = DataValidator()
        results = validator.validate_data(train_df, test_df, target_column='target')
        
        assert len(results['leakage_warnings']) > 0
        assert any('identical to target' in warning for warning in results['leakage_warnings'])
    
    def test_detect_perfect_correlation(self):
        """Test detection of perfect correlation with target."""
        # Create data with perfect correlation
        train_df = pd.DataFrame({
            'feature1': np.arange(100),
            'target': np.arange(100) * 2 + 1,  # Perfect linear relationship
            'feature2': np.random.randn(100)
        })
        test_df = train_df.drop(columns=['target'])
        
        validator = DataValidator()
        results = validator.validate_data(train_df, test_df, target_column='target')
        
        assert len(results['leakage_warnings']) > 0
        assert any('perfect correlation' in warning for warning in results['leakage_warnings'])
    
    def test_temporal_leakage_detection(self):
        """Test detection of temporal leakage."""
        # Create data with temporal leakage
        train_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100),
            'feature': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })
        
        # Test data has dates overlapping with training
        test_df = pd.DataFrame({
            'date': pd.date_range('2023-03-01', periods=50),  # Overlaps with train
            'feature': np.random.randn(50)
        })
        
        validator = DataValidator()
        results = validator.validate_data(
            train_df, test_df, 
            target_column='target',
            datetime_columns=['date']
        )
        
        assert len(results['leakage_warnings']) > 0
        assert any('temporal leakage' in warning.lower() for warning in results['leakage_warnings'])
    
    def test_suspicious_column_names(self):
        """Test detection of suspicious column names."""
        # Create data with suspicious names
        train_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'actual_value': [10, 20, 30, 40, 50],  # Suspicious name
            'true_label': [0, 1, 0, 1, 0],  # Suspicious name
            'target': [0, 1, 0, 1, 0]
        })
        test_df = train_df.drop(columns=['target'])
        
        validator = DataValidator()
        results = validator.validate_data(train_df, test_df, target_column='target')
        
        assert any('suspicious name' in warning for warning in results['leakage_warnings'])
    
    def test_validation_report(self, sample_classification_data):
        """Test validation report generation."""
        train_df, test_df = sample_classification_data
        validator = DataValidator()
        
        validator.validate_data(train_df, test_df, target_column='target')
        report = validator.get_validation_report()
        
        assert isinstance(report, str)
        assert 'DATA VALIDATION REPORT' in report
        assert 'Basic Information' in report
        assert 'Train shape' in report
        assert 'Test shape' in report


class TestDataLoaderValidation:
    """Test integration of DataLoader with validation."""
    
    def test_validate_data_method(self, create_test_files):
        """Test DataLoader's validate_data method."""
        data_dir = create_test_files
        loader = DataLoader(data_dir)
        
        loader.load_data()
        results = loader.validate_data()
        
        assert results is not None
        assert 'basic' in results
        assert 'leakage_warnings' in results
        assert loader.validation_results is not None
    
    def test_validate_with_datetime_columns(self, temp_data_dir):
        """Test validation with datetime columns."""
        # Create data with dates
        train_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100),
            'feature': np.random.randn(100),
            'target': np.random.randint(0, 2, 100),
            'id': range(100)
        })
        
        test_df = pd.DataFrame({
            'date': pd.date_range('2023-04-01', periods=50),
            'feature': np.random.randn(50),
            'id': range(100, 150)
        })
        
        train_df.to_csv(temp_data_dir / 'train.csv', index=False)
        test_df.to_csv(temp_data_dir / 'test.csv', index=False)
        
        loader = DataLoader(temp_data_dir)
        loader.load_data()
        results = loader.validate_data(datetime_columns=['date'])
        
        assert 'temporal_issues' in results