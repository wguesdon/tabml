"""Data loading and preprocessing utilities.

This module provides classes and functions for loading various data formats
and performing initial preprocessing steps for tabular machine learning tasks.

Example:
    Basic usage::

        from tabml.data import DataLoader
        loader = DataLoader(data_dir="./data")
        train_df, test_df = loader.load_data(
            train_file="train.csv",
            test_file="test.csv",
            target_column="target"
        )
"""

from typing import Dict, Optional, Tuple, Union, List
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from loguru import logger
from .validation import DataValidator


class DataLoader:
    """Handles data loading and basic preprocessing.

    This class provides methods for loading data from various file formats
    (CSV, Parquet, Feather) and performs basic preprocessing including
    type inference and memory optimization.

    Attributes:
        data_dir (Union[str, Path]): Base directory path for data files.
        train_data (Optional[pd.DataFrame]): Loaded training data.
        test_data (Optional[pd.DataFrame]): Loaded test data.
        target_column (Optional[str]): Name of the target column.
        id_column (Optional[str]): Name of the ID column for submissions.
        validator (DataValidator): DataValidator instance for data validation.
        validation_results (Optional[Dict]): Results from the last validation run.
    """
    
    def __init__(self, data_dir: Union[str, Path] = "data"):
        """Initializes the DataLoader.
        
        Args:
            data_dir (Union[str, Path]): Base directory for data files.
                Defaults to "data" in the current directory.
        """
        self.data_dir = Path(data_dir)
        self.train_data = None
        self.test_data = None
        self.target_column = None
        self.id_column = None
        self.validator = DataValidator()
        self.validation_results = None
        
    def _load_file(self, file_path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
        """Loads data from a file based on its extension.

        Automatically detects the file format from the extension and uses the
        appropriate pandas loading function. Supports CSV, Parquet, and Feather
        formats.

        Args:
            file_path (Path): The path to the data file.
            nrows (Optional[int]): The number of rows to read. This is only
                supported for CSV files. For other formats, the full data is
                loaded and then truncated.

        Returns:
            A pandas DataFrame with the loaded data.

        Raises:
            ValueError: If the file extension is not supported.
            FileNotFoundError: If the file does not exist.
        """
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.csv':
            return pd.read_csv(file_path, nrows=nrows)
        elif file_extension in ['.parquet', '.pq']:
            # Parquet doesn't support nrows directly, so we load and then limit
            df = pd.read_parquet(file_path)
            if nrows is not None:
                df = df.head(nrows)
            return df
        elif file_extension in ['.xlsx', '.xls']:
            # Excel files - use openpyxl engine for .xlsx, xlrd for .xls
            engine = 'openpyxl' if file_extension == '.xlsx' else 'xlrd'
            df = pd.read_excel(file_path, engine=engine)
            if nrows is not None:
                df = df.head(nrows)
            return df
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
    def load_data(self, 
                  train_file: str = "train.csv",
                  test_file: str = "test.csv",
                  target_column: Optional[str] = None,
                  id_column: Optional[str] = None,
                  sample_frac: Optional[float] = None,
                  nrows: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads train and test data.

        Loads training and test datasets from specified files, with support for
        multiple file formats and automatic detection of special columns.

        Args:
            train_file (str): The training data filename relative to data_dir.
                Defaults to "train.csv".
            test_file (str): The test data filename relative to data_dir. Defaults
                to "test.csv".
            target_column (Optional[str]): The name of the target column. If
                None, attempts auto-detection.
            id_column (Optional[str]): The name of the ID column. If None,
                attempts auto-detection.
            sample_frac (Optional[float]): The fraction of data to sample (0.0 to
                1.0). Useful for quick testing. If None, uses the full dataset.
            nrows (Optional[int]): The number of rows to read from files. Useful
                for quick testing. If None, reads all rows.

        Returns:
            A tuple containing:
                - train_df: The loaded training DataFrame.
                - test_df: The loaded test DataFrame.

        Raises:
            FileNotFoundError: If train_file or test_file doesn't exist.
            ValueError: If the file format is not supported.
        """
        # Load training data
        train_path = self.data_dir / train_file
        test_path = self.data_dir / test_file
        
        if not train_path.exists():
            raise FileNotFoundError(f"Training file not found: {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Test file not found: {test_path}")
            
        logger.info(f"Loading training data from {train_path}")
        self.train_data = self._load_file(train_path, nrows=nrows)
        
        logger.info(f"Loading test data from {test_path}")
        self.test_data = self._load_file(test_path, nrows=nrows)
        
        # Sample data if requested
        if sample_frac is not None and sample_frac < 1.0:
            logger.info(f"Sampling {sample_frac*100}% of data")
            self.train_data = self.train_data.sample(frac=sample_frac, random_state=42)
            self.test_data = self.test_data.sample(frac=sample_frac, random_state=42)
            
        # Auto-detect target and ID columns
        self._detect_special_columns(target_column, id_column)
        
        logger.info(f"Train shape: {self.train_data.shape}")
        logger.info(f"Test shape: {self.test_data.shape}")
        logger.info(f"Target column: {self.target_column}")
        logger.info(f"ID column: {self.id_column}")
        
        return self.train_data, self.test_data
    
    def _detect_special_columns(self, target_column: Optional[str], id_column: Optional[str]) -> None:
        """Auto-detects target and ID columns.

        Intelligently identifies the target column (present in train but not test)
        and ID column (unique identifier present in both datasets).

        Args:
            target_column (Optional[str]): The explicitly provided target column
                name. If None, auto-detection is performed.
            id_column (Optional[str]): The explicitly provided ID column name. If
                None, auto-detection is performed.
        """
        train_cols = set(self.train_data.columns)
        test_cols = set(self.test_data.columns)
        
        # Detect target column (present in train but not in test)
        if target_column:
            self.target_column = target_column
        else:
            target_candidates = train_cols - test_cols
            if len(target_candidates) == 1:
                self.target_column = target_candidates.pop()
            elif 'target' in train_cols and 'target' not in test_cols:
                self.target_column = 'target'
            elif 'label' in train_cols and 'label' not in test_cols:
                self.target_column = 'label'
            else:
                # Use last column as target
                self.target_column = self.train_data.columns[-1]
                logger.warning(f"Could not auto-detect target column. Using: {self.target_column}")
                
        # Detect ID column
        if id_column:
            self.id_column = id_column
        else:
            common_id_names = ['id', 'ID', 'Id', 'index', 'key']
            for col in common_id_names:
                if col in train_cols and col in test_cols:
                    # Check if it looks like an ID (unique values)
                    if self.train_data[col].nunique() == len(self.train_data):
                        self.id_column = col
                        break
                        
            if self.id_column is None:
                # Check first column
                first_col = self.train_data.columns[0]
                if self.train_data[first_col].nunique() == len(self.train_data):
                    self.id_column = first_col
                    
    def get_train_test_split(self, 
                            test_size: float = 0.2,
                            stratify: bool = True,
                            random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Splits the training data into train and validation sets.

        Creates a train/validation split from the loaded training data, with
        automatic stratification for classification tasks.

        Args:
            test_size (float): The proportion of data to use for validation (0.0
                to 1.0). Defaults to 0.2 (20% validation).
            stratify (bool): Whether to stratify the split based on the target
                distribution. Only applied for classification tasks (< 100 unique
                target values). Defaults to True.
            random_state (int): The random seed for reproducibility. Defaults to 42.

        Returns:
            A tuple containing:
                - X_train: The training features.
                - X_val: The validation features.
                - y_train: The training target values.
                - y_val: The validation target values.

        Raises:
            ValueError: If no training data is loaded.
        """
        if self.train_data is None:
            raise ValueError("No training data loaded. Call load_data() first.")
            
        # Separate features and target
        X = self.train_data.drop(columns=[self.target_column])
        y = self.train_data[self.target_column]
        
        # Remove ID column if present
        if self.id_column and self.id_column in X.columns:
            X = X.drop(columns=[self.id_column])
            
        # Determine if stratification is appropriate
        if stratify and y.nunique() < 100:
            stratify_col = y
        else:
            stratify_col = None
            
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=test_size,
            stratify=stratify_col,
            random_state=random_state
        )
        
        logger.info(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
        
        return X_train, X_val, y_train, y_val
    
    def get_test_features(self) -> pd.DataFrame:
        """Gets the test features, excluding the ID column.

        Returns a copy of the test data with the ID column removed, ready for
        model prediction.

        Returns:
            A DataFrame of test features without the ID column.

        Raises:
            ValueError: If no test data is loaded.
        """
        if self.test_data is None:
            raise ValueError("No test data loaded. Call load_data() first.")
            
        X_test = self.test_data.copy()
        
        # Remove ID column if present
        if self.id_column and self.id_column in X_test.columns:
            X_test = X_test.drop(columns=[self.id_column])
            
        return X_test
    
    def get_test_ids(self) -> pd.Series:
        """Gets the test IDs.

        Extracts the ID column from the test data for creating submissions. If no
        ID column is detected, returns the DataFrame index.

        Returns:
            A Series containing test IDs or indices.

        Raises:
            ValueError: If no test data is loaded.
        """
        if self.test_data is None:
            raise ValueError("No test data loaded. Call load_data() first.")
            
        if self.id_column and self.id_column in self.test_data.columns:
            return self.test_data[self.id_column]
        else:
            # Return index if no ID column
            return pd.Series(self.test_data.index, name='index')
            
    def create_submission(self, predictions: np.ndarray, 
                         submission_file: str = "submission.csv") -> pd.DataFrame:
        """Creates a submission file.

        Creates a properly formatted submission file with test IDs and predictions,
        ready for competition submission.

        Args:
            predictions (np.ndarray): An array of model predictions with the same
                length as the test data.
            submission_file (str): The output filename relative to data_dir.
                Defaults to "submission.csv".

        Returns:
            A submission DataFrame with ID and prediction columns.

        Raises:
            ValueError: If no test data is loaded or if the predictions have a
                length mismatch.
        """
        if self.test_data is None:
            raise ValueError("No test data loaded. Call load_data() first.")
            
        # Get test IDs
        test_ids = self.get_test_ids()
        
        # Create submission DataFrame
        submission = pd.DataFrame({
            self.id_column or 'id': test_ids,
            self.target_column or 'target': predictions
        })
        
        # Save to file
        submission_path = self.data_dir / submission_file
        submission.to_csv(submission_path, index=False)
        logger.info(f"Submission saved to {submission_path}")
        
        return submission
    
    def get_data_info(self) -> Dict[str, any]:
        """Gets comprehensive information about the loaded data.

        Provides detailed statistics including shapes, column types, missing
        values, and target distribution.

        Returns:
            A dictionary containing:
                - train_shape: The training data dimensions.
                - test_shape: The test data dimensions.
                - target_column: The name of the target column.
                - id_column: The name of the ID column.
                - numeric_features: A list of numeric column names.
                - categorical_features: A list of categorical column names.
                - missing_values: Missing value counts per column.
                - target_info: Target variable statistics (if applicable).

        Raises:
            ValueError: If no data is loaded.
        """
        if self.train_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        info = {
            'train_shape': self.train_data.shape,
            'test_shape': self.test_data.shape,
            'target_column': self.target_column,
            'id_column': self.id_column,
            'numeric_features': self.train_data.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_features': self.train_data.select_dtypes(include=['object', 'category']).columns.tolist(),
            'missing_values': {
                'train': self.train_data.isnull().sum().to_dict(),
                'test': self.test_data.isnull().sum().to_dict()
            }
        }
        
        # Target information
        if self.target_column in self.train_data.columns:
            target = self.train_data[self.target_column]
            info['target_info'] = {
                'unique_values': target.nunique(),
                'type': 'classification' if target.nunique() < 100 else 'regression',
                'distribution': target.value_counts().to_dict() if target.nunique() < 20 else None
            }
            
        return info
    
    def validate_data(self, datetime_columns: Optional[List[str]] = None) -> Dict[str, any]:
        """Validates the loaded data for potential issues and leakage.

        Performs comprehensive validation checks including data leakage detection,
        distribution shifts, missing patterns, and temporal consistency.

        Args:
            datetime_columns (Optional[List[str]]): A list of column names that
                should be parsed as datetime. Used for temporal leakage
                detection. If None, no temporal validation is performed.

        Returns:
            A dictionary containing validation results with keys:
                - data_leakage: Potential leakage issues found.
                - distribution_shift: Significant distribution differences.
                - missing_patterns: Missing value pattern analysis.
                - high_cardinality: High cardinality categorical features.
                - constant_features: Features with a single unique value.
                - duplicates: Duplicate row information.
                - temporal_issues: Time-based validation results (if applicable).

        Raises:
            ValueError: If no data is loaded.
        """
        if self.train_data is None or self.test_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Run validation
        self.validation_results = self.validator.validate_data(
            self.train_data,
            self.test_data,
            self.target_column,
            datetime_columns
        )
        
        # Print summary report
        print(self.validator.get_validation_report())
        
        return self.validation_results