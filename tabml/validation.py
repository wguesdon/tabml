"""Data validation and leakage detection utilities.

This module provides comprehensive validation tools to ensure data quality
and detect potential data leakage issues that could compromise model validity.

Classes:
    DataValidator: Main validation class for detecting data issues
    
Example:
    Basic validation workflow::
    
        from tabml.validation import DataValidator
        
        # Initialize validator
        validator = DataValidator()
        
        # Run comprehensive validation
        results = validator.validate_data(
            train_df, test_df,
            target_column='price',
            datetime_columns=['date', 'created_at']
        )
        
        # Check for issues
        if results['leakage_warnings']:
            print("WARNING: Potential data leakage detected!")
            print(validator.get_validation_report())
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from datetime import datetime


class DataValidator:
    """Handles data validation and leakage detection.
    
    Performs comprehensive checks to identify data quality issues and
    potential data leakage that could lead to overly optimistic model
    performance. Detects various forms of leakage including target
    leakage, temporal leakage, and suspicious features.
    
    Attributes:
        validation_results: Dictionary containing all validation findings
        leakage_warnings: List of potential data leakage warnings
        
    Example:
        >>> validator = DataValidator()
        >>> 
        >>> # Basic validation
        >>> results = validator.validate_data(train_df, test_df)
        >>> 
        >>> # With target and temporal validation
        >>> results = validator.validate_data(
        ...     train_df, test_df,
        ...     target_column='sales',
        ...     datetime_columns=['transaction_date']
        ... )
        >>> 
        >>> # Get human-readable report
        >>> print(validator.get_validation_report())
        >>> 
        >>> # Check specific issues
        >>> if results['leakage_features']:
        ...     print(f"Found {len(results['leakage_features'])} leaky features!")
    """
    
    def __init__(self):
        """Initialize data validator.
        
        Example:
            >>> validator = DataValidator()
        """
        self.validation_results = {}
        self.leakage_warnings = []
    
    def validate_data(self, 
                     train_df: pd.DataFrame, 
                     test_df: pd.DataFrame,
                     target_column: Optional[str] = None,
                     datetime_columns: Optional[List[str]] = None) -> Dict[str, any]:
        """Perform comprehensive data validation.
        
        Runs multiple validation checks to identify data quality issues
        and potential sources of data leakage.
        
        Args:
            train_df: Training DataFrame to validate
            test_df: Test DataFrame to validate
            target_column: Name of target column. If provided, checks
                for target leakage and suspicious correlations.
            datetime_columns: List of datetime column names for temporal
                validation. Checks for temporal leakage in time series.
            
        Returns:
            Dictionary containing:
                - 'basic': Basic dataset properties
                - 'duplicate_columns': Pairs of duplicate columns
                - 'constant_features': Features with single unique value
                - 'leakage_features': Features with potential target leakage
                - 'temporal_issues': Columns with temporal leakage
                - 'suspicious_features': Features with suspicious patterns
                - 'leakage_warnings': List of all warning messages
                
        Example:
            >>> # Full validation
            >>> results = validator.validate_data(
            ...     train_df, test_df,
            ...     target_column='price',
            ...     datetime_columns=['listing_date', 'sold_date']
            ... )
            >>> 
            >>> # Check results
            >>> if results['leakage_warnings']:
            ...     for warning in results['leakage_warnings']:
            ...         print(f"⚠️ {warning}")
            
        Note:
            - Perfect correlations (|r| > 0.99) trigger warnings
            - Temporal overlap between train/test triggers warnings
            - Suspicious column names are flagged
            - Results are logged for debugging
        """
        self.validation_results = {}
        self.leakage_warnings = []
        
        # Basic validation
        self._validate_basic_properties(train_df, test_df)
        
        # Check for duplicate columns
        self._check_duplicate_columns(train_df, test_df)
        
        # Check for constant features
        self._check_constant_features(train_df, test_df)
        
        # Check for data leakage
        self._check_data_leakage(train_df, test_df, target_column)
        
        # Check for temporal leakage if datetime columns exist
        if datetime_columns:
            self._check_temporal_leakage(train_df, test_df, datetime_columns)
        
        # Check for suspicious features
        self._check_suspicious_features(train_df, test_df, target_column)
        
        self.validation_results['leakage_warnings'] = self.leakage_warnings
        
        # Log summary
        if self.leakage_warnings:
            logger.warning(f"Found {len(self.leakage_warnings)} potential data leakage issues!")
            for warning in self.leakage_warnings:
                logger.warning(f"  - {warning}")
        else:
            logger.info("No data leakage detected.")
            
        return self.validation_results
    
    def _validate_basic_properties(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Validate basic properties of the datasets.
        
        Checks dataset shapes, column alignment, and memory usage.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            
        Note:
            Results stored in validation_results['basic']
        """
        self.validation_results['basic'] = {
            'train_shape': train_df.shape,
            'test_shape': test_df.shape,
            'train_columns': list(train_df.columns),
            'test_columns': list(test_df.columns),
            'missing_in_test': list(set(train_df.columns) - set(test_df.columns)),
            'missing_in_train': list(set(test_df.columns) - set(train_df.columns)),
            'train_memory_usage': train_df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'test_memory_usage': test_df.memory_usage(deep=True).sum() / 1024**2,  # MB
        }
    
    def _check_duplicate_columns(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Check for duplicate columns.
        
        Identifies columns that contain identical values, which may indicate
        data processing errors or redundant features.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            
        Note:
            Duplicate columns are added to leakage warnings as they may
            indicate data processing issues.
        """
        duplicate_cols = []
        
        # Check within training data
        for i, col1 in enumerate(train_df.columns):
            for col2 in train_df.columns[i+1:]:
                if train_df[col1].equals(train_df[col2]):
                    duplicate_cols.append((col1, col2))
                    self.leakage_warnings.append(f"Duplicate columns found: {col1} and {col2}")
        
        self.validation_results['duplicate_columns'] = duplicate_cols
    
    def _check_constant_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Check for constant features.
        
        Identifies features with only one unique value, which provide no
        information for modeling. Also checks for features constant in test
        but not train, which may indicate leakage.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            
        Warning:
            Features that are constant in test but variable in train may
            indicate that test data was used in feature engineering.
        """
        constant_features = []
        
        # Check training data
        for col in train_df.columns:
            if train_df[col].nunique() == 1:
                constant_features.append(col)
                logger.warning(f"Constant feature in training data: {col}")
        
        # Check if features are constant in test but not in train
        for col in test_df.columns:
            if col in train_df.columns:
                if test_df[col].nunique() == 1 and train_df[col].nunique() > 1:
                    self.leakage_warnings.append(
                        f"Feature '{col}' is constant in test but not in train - possible leakage"
                    )
        
        self.validation_results['constant_features'] = constant_features
    
    def _check_data_leakage(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                           target_column: Optional[str] = None) -> None:
        """Check for various types of data leakage.
        
        Performs multiple checks to identify features that may contain
        information about the target that wouldn't be available at
        prediction time.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame  
            target_column: Name of target column to check against
            
        Checks performed:
            1. Perfect correlation with target (|r| > 0.99)
            2. Features identical to target
            3. Linear transformations of target
            4. Suspicious column names containing target-related keywords
            
        Note:
            This is critical for preventing overly optimistic performance
            estimates that won't generalize to production.
        """
        leakage_features = []
        
        # Check for perfect correlation with target
        if target_column and target_column in train_df.columns:
            target = train_df[target_column]
            
            for col in train_df.columns:
                if col != target_column:
                    # Check numeric columns for perfect correlation
                    if train_df[col].dtype in ['int64', 'float64']:
                        correlation = train_df[col].corr(target)
                        if abs(correlation) > 0.99:
                            leakage_features.append(col)
                            self.leakage_warnings.append(
                                f"Feature '{col}' has perfect correlation ({correlation:.3f}) with target"
                            )
                    
                    # Check if any feature is a transformation of the target
                    if train_df[col].dtype == target.dtype:
                        # Simple transformations
                        if (train_df[col] == target).all():
                            leakage_features.append(col)
                            self.leakage_warnings.append(
                                f"Feature '{col}' is identical to target"
                            )
                        elif (train_df[col] == -target).all():
                            leakage_features.append(col)
                            self.leakage_warnings.append(
                                f"Feature '{col}' is negative of target"
                            )
                        elif target.dtype in ['int64', 'float64']:
                            # Check for linear transformations
                            if target.std() > 0 and train_df[col].std() > 0:
                                normalized_target = (target - target.mean()) / target.std()
                                normalized_col = (train_df[col] - train_df[col].mean()) / train_df[col].std()
                                if np.allclose(normalized_col, normalized_target, rtol=1e-5):
                                    leakage_features.append(col)
                                    self.leakage_warnings.append(
                                        f"Feature '{col}' is a linear transformation of target"
                                    )
        
        # Check for features that shouldn't exist at prediction time
        suspicious_patterns = ['actual', 'true', 'label', 'target', 'y_', 'ground_truth']
        for col in train_df.columns:
            col_lower = col.lower()
            for pattern in suspicious_patterns:
                if pattern in col_lower and col != target_column:
                    self.leakage_warnings.append(
                        f"Feature '{col}' has suspicious name suggesting target leakage"
                    )
        
        self.validation_results['leakage_features'] = leakage_features
    
    def _check_temporal_leakage(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                               datetime_columns: List[str]) -> None:
        """Check for temporal leakage in time series data.
        
        Ensures test data comes after training data chronologically,
        which is critical for valid time series model evaluation.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            datetime_columns: List of datetime column names to check
            
        Checks:
            - Test dates before training dates (future data in training)
            - Overlap between train and test time periods
            - Proper temporal ordering
            
        Example:
            If training data goes up to 2023-12-31 but test data
            contains dates from 2023-11-01, this indicates temporal
            leakage that would invalidate the model evaluation.
        """
        temporal_issues = []
        
        for dt_col in datetime_columns:
            if dt_col in train_df.columns and dt_col in test_df.columns:
                # Convert to datetime if not already
                train_dates = pd.to_datetime(train_df[dt_col], errors='coerce')
                test_dates = pd.to_datetime(test_df[dt_col], errors='coerce')
                
                # Remove NaT values
                train_dates = train_dates.dropna()
                test_dates = test_dates.dropna()
                
                if len(train_dates) > 0 and len(test_dates) > 0:
                    # Check if test data contains dates before training data
                    train_min, train_max = train_dates.min(), train_dates.max()
                    test_min, test_max = test_dates.min(), test_dates.max()
                    
                    if test_min < train_max:
                        temporal_issues.append(dt_col)
                        self.leakage_warnings.append(
                            f"Temporal leakage in '{dt_col}': test data contains dates "
                            f"({test_min}) before latest training date ({train_max})"
                        )
                    
                    # Check for overlap
                    overlap = (test_min <= train_max) and (train_min <= test_max)
                    if overlap:
                        self.leakage_warnings.append(
                            f"Date overlap detected in '{dt_col}' between train and test sets"
                        )
        
        self.validation_results['temporal_issues'] = temporal_issues
    
    def _check_suspicious_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                                  target_column: Optional[str] = None) -> None:
        """Check for features that might indicate data leakage.
        
        Identifies features with suspicious patterns that may indicate
        data leakage or improper feature engineering.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            target_column: Target column name
            
        Checks:
            - ID columns with low uniqueness (may encode target info)
            - Sequential IDs that correlate with target
            - Features with suspicious naming patterns
            
        Note:
            Sequential IDs that correlate with target often indicate
            that data was sorted by target before ID assignment.
        """
        suspicious_features = []
        
        # Check for ID-like columns with information
        id_patterns = ['id', 'index', 'key', 'code', 'number']
        
        for col in train_df.columns:
            if col == target_column:
                continue
                
            col_lower = col.lower()
            
            # Check if column looks like an ID
            is_id_like = any(pattern in col_lower for pattern in id_patterns)
            
            if is_id_like and train_df[col].dtype in ['int64', 'float64']:
                # Check if ID column has unexpected patterns
                unique_ratio = train_df[col].nunique() / len(train_df)
                
                # IDs should be mostly unique
                if unique_ratio < 0.95:
                    suspicious_features.append(col)
                    logger.warning(f"ID-like column '{col}' has low uniqueness ratio: {unique_ratio:.2f}")
                
                # Check if IDs are sequential or have patterns
                if train_df[col].dtype == 'int64':
                    sorted_vals = sorted(train_df[col].unique())
                    if len(sorted_vals) > 1:
                        diffs = np.diff(sorted_vals)
                        if np.all(diffs == diffs[0]):  # Perfect sequence
                            # Check if sequence correlates with target
                            if target_column and target_column in train_df.columns:
                                correlation = train_df[col].corr(train_df[target_column])
                                if abs(correlation) > 0.3:
                                    self.leakage_warnings.append(
                                        f"Sequential ID '{col}' correlates with target (r={correlation:.3f})"
                                    )
        
        self.validation_results['suspicious_features'] = suspicious_features
    
    def get_validation_report(self) -> str:
        """Generate a text report of validation results.
        
        Creates a human-readable summary of all validation findings,
        organized by issue type with clear warnings for critical issues.
        
        Returns:
            Formatted text report suitable for console output or logging
            
        Example:
            >>> validator.validate_data(train_df, test_df, target_column='price')
            >>> report = validator.get_validation_report()
            >>> print(report)
            ==================================================
            DATA VALIDATION REPORT
            ==================================================
            
            ## Basic Information
            Train shape: (10000, 50)
            Test shape: (5000, 49)
            ...
            
            ## Data Leakage Warnings (3)
              ⚠️  Feature 'price_normalized' has perfect correlation (0.998) with target
              ⚠️  Temporal leakage in 'date': test data contains dates before training
              ⚠️  Feature 'actual_price' has suspicious name suggesting target leakage
            
        Note:
            Report is truncated to show most important issues first.
            Full results available in validation_results attribute.
        """
        report = ["=" * 50]
        report.append("DATA VALIDATION REPORT")
        report.append("=" * 50)
        
        # Basic info
        report.append("\n## Basic Information")
        basic = self.validation_results.get('basic', {})
        report.append(f"Train shape: {basic.get('train_shape', 'N/A')}")
        report.append(f"Test shape: {basic.get('test_shape', 'N/A')}")
        report.append(f"Train memory: {basic.get('train_memory_usage', 0):.2f} MB")
        report.append(f"Test memory: {basic.get('test_memory_usage', 0):.2f} MB")
        
        # Issues found
        report.append("\n## Issues Found")
        
        # Constant features
        const_features = self.validation_results.get('constant_features', [])
        if const_features:
            report.append(f"\nConstant features ({len(const_features)}): {', '.join(const_features[:5])}")
            if len(const_features) > 5:
                report.append(f"  ... and {len(const_features) - 5} more")
        
        # Duplicate columns
        dup_cols = self.validation_results.get('duplicate_columns', [])
        if dup_cols:
            report.append(f"\nDuplicate columns ({len(dup_cols)} pairs):")
            for col1, col2 in dup_cols[:3]:
                report.append(f"  - {col1} == {col2}")
            if len(dup_cols) > 3:
                report.append(f"  ... and {len(dup_cols) - 3} more pairs")
        
        # Leakage warnings
        if self.leakage_warnings:
            report.append(f"\n## Data Leakage Warnings ({len(self.leakage_warnings)})")
            for warning in self.leakage_warnings[:10]:
                report.append(f"  ⚠️  {warning}")
            if len(self.leakage_warnings) > 10:
                report.append(f"  ... and {len(self.leakage_warnings) - 10} more warnings")
        else:
            report.append("\n✅ No data leakage detected")
        
        return "\n".join(report)