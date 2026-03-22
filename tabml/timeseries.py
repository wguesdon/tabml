"""Time series data handling and feature engineering utilities.

This module provides specialized tools for working with time series data,
including temporal validation, date-based splitting, and time series
specific feature engineering.

Classes:
    TimeSeriesDataLoader: Extended data loader with time series capabilities
    TimeSeriesFeatureEngineer: Feature engineering for temporal data
    
Example:
    Basic time series workflow::
    
        from tabml.timeseries import TimeSeriesDataLoader, TimeSeriesFeatureEngineer
        
        # Load time series data
        loader = TimeSeriesDataLoader()
        train_df, test_df = loader.load_data(
            datetime_column='date',
            sort_by_date=True
        )
        
        # Create time-based features
        engineer = TimeSeriesFeatureEngineer()
        train_featured = engineer.fit_transform(train_df, 'date')
        
        # Add lag and rolling features
        train_featured = engineer.create_lag_features(
            train_featured, 'sales', lags=[1, 7, 30]
        )
        train_featured = engineer.create_rolling_features(
            train_featured, 'sales', windows=[7, 14, 30]
        )
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from .data import DataLoader


class TimeSeriesDataLoader(DataLoader):
    """Specialized data loader for time series data.
    
    Extends the base DataLoader with time series specific functionality
    including datetime detection, frequency inference, temporal validation,
    and time-based splitting.
    
    Attributes:
        All attributes from DataLoader plus:
        datetime_column: Name of the datetime column
        datetime_format: Format string for parsing dates
        frequency: Detected frequency of the time series (D, W, M, etc.)
        
    Example:
        >>> # Load time series data
        >>> loader = TimeSeriesDataLoader(data_dir="./timeseries_data")
        >>> train_df, test_df = loader.load_data(
        ...     train_file="sales_train.csv",
        ...     test_file="sales_test.csv",
        ...     datetime_column="date",
        ...     sort_by_date=True
        ... )
        >>> 
        >>> # Check temporal integrity
        >>> integrity = loader.check_temporal_integrity()
        >>> if integrity['temporal_overlap']:
        ...     print("Warning: Train and test periods overlap!")
        >>> 
        >>> # Create time-based validation split
        >>> X_train, X_val, y_train, y_val = loader.create_time_based_split(
        ...     test_size=0.2,
        ...     gap=7  # 7-day gap to prevent leakage
        ... )
    """
    
    def __init__(self, data_dir: Union[str, Path] = "data"):
        """Initialize time series data loader.
        
        Args:
            data_dir: Base directory for data files. Can be string or Path object.
                Defaults to "data" in current directory.
                
        Example:
            >>> loader = TimeSeriesDataLoader(data_dir="./kaggle/store-sales")
        """
        super().__init__(data_dir)
        self.datetime_column = None
        self.datetime_format = None
        self.frequency = None
        
    def load_data(self, 
                  train_file: str = "train.csv",
                  test_file: str = "test.csv",
                  datetime_column: Optional[str] = None,
                  datetime_format: Optional[str] = None,
                  target_column: Optional[str] = None,
                  id_column: Optional[str] = None,
                  sort_by_date: bool = True,
                  sample_frac: Optional[float] = None,
                  nrows: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load time series data with datetime handling.
        
        Extends base load_data with datetime parsing, frequency detection,
        and temporal sorting capabilities.
        
        Args:
            train_file: Training data filename in data_dir
            test_file: Test data filename in data_dir
            datetime_column: Name of datetime column. If None, attempts to
                auto-detect based on column names and content.
            datetime_format: strftime format for parsing dates. If None,
                pandas infers the format. Examples:
                - '%Y-%m-%d': 2023-12-31
                - '%d/%m/%Y': 31/12/2023
                - '%Y-%m-%d %H:%M:%S': 2023-12-31 23:59:59
            target_column: Name of target column (auto-detected if None)
            id_column: Name of ID column (auto-detected if None)
            sort_by_date: Whether to sort data chronologically. Important
                for time series modeling and validation.
            sample_frac: Fraction of data to sample (maintains temporal order)
            nrows: Number of rows to read from beginning of file
            
        Returns:
            Tuple of (train_df, test_df) with datetime columns parsed
            
        Example:
            >>> # Load with auto-detection
            >>> train_df, test_df = loader.load_data(sort_by_date=True)
            >>> 
            >>> # Load with specific datetime format
            >>> train_df, test_df = loader.load_data(
            ...     datetime_column='transaction_date',
            ...     datetime_format='%Y-%m-%d %H:%M:%S',
            ...     target_column='amount'
            ... )
            >>> 
            >>> # Quick testing with subset
            >>> train_df, test_df = loader.load_data(
            ...     sample_frac=0.1  # Use 10% for prototyping
            ... )
            
        Note:
            - Frequency is automatically detected (daily, weekly, monthly, etc.)
            - Common datetime column names are checked if not specified
            - Data is sorted chronologically if sort_by_date=True
        """
        # Load data using parent method
        train_df, test_df = super().load_data(
            train_file, test_file, target_column, id_column, sample_frac, nrows
        )
        
        # Auto-detect datetime column
        self._detect_datetime_column(train_df, datetime_column)
        
        # Convert to datetime
        if self.datetime_column:
            train_df = self._convert_to_datetime(train_df, self.datetime_column, datetime_format)
            test_df = self._convert_to_datetime(test_df, self.datetime_column, datetime_format)
            
            # Sort by date if requested
            if sort_by_date:
                train_df = train_df.sort_values(self.datetime_column)
                test_df = test_df.sort_values(self.datetime_column)
                
            # Detect frequency
            self._detect_frequency(train_df)
            
            logger.info(f"Datetime column: {self.datetime_column}")
            logger.info(f"Detected frequency: {self.frequency}")
        
        self.train_data = train_df
        self.test_data = test_df
        
        return train_df, test_df
    
    def _detect_datetime_column(self, df: pd.DataFrame, datetime_column: Optional[str] = None) -> None:
        """Auto-detect datetime column.
        
        Attempts to identify datetime column using name patterns and content analysis.
        
        Args:
            df: DataFrame to analyze
            datetime_column: Explicitly provided column name (overrides detection)
            
        Note:
            Looks for columns with names containing 'date', 'time', 'timestamp',
            etc., and verifies they can be parsed as datetime.
        """
        if datetime_column:
            self.datetime_column = datetime_column
            return
            
        # Common datetime column names
        datetime_patterns = ['date', 'datetime', 'time', 'timestamp', 'period', 'dt']
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Check if column name matches patterns
            if any(pattern in col_lower for pattern in datetime_patterns):
                # Try to parse as datetime
                try:
                    pd.to_datetime(df[col].iloc[:100])  # Test first 100 values
                    self.datetime_column = col
                    logger.info(f"Auto-detected datetime column: {col}")
                    return
                except:
                    continue
                    
            # Check if column contains date-like strings
            if df[col].dtype == 'object':
                sample = df[col].dropna().iloc[:10].astype(str)
                if all('-' in s or '/' in s for s in sample):
                    try:
                        pd.to_datetime(df[col].iloc[:100])
                        self.datetime_column = col
                        logger.info(f"Auto-detected datetime column: {col}")
                        return
                    except:
                        continue
    
    def _convert_to_datetime(self, df: pd.DataFrame, datetime_column: str, 
                           datetime_format: Optional[str] = None) -> pd.DataFrame:
        """Convert column to datetime.
        
        Safely converts specified column to datetime type.
        
        Args:
            df: DataFrame containing the column
            datetime_column: Name of column to convert
            datetime_format: Optional format string for parsing
            
        Returns:
            DataFrame with converted datetime column
            
        Note:
            Logs warning if conversion fails rather than raising exception.
        """
        if datetime_column not in df.columns:
            return df
            
        try:
            if datetime_format:
                df[datetime_column] = pd.to_datetime(df[datetime_column], format=datetime_format)
            else:
                df[datetime_column] = pd.to_datetime(df[datetime_column])
        except Exception as e:
            logger.warning(f"Could not convert {datetime_column} to datetime: {e}")
            
        return df
    
    def _detect_frequency(self, df: pd.DataFrame) -> None:
        """Detect time series frequency.
        
        Analyzes time differences to infer the frequency of the time series.
        
        Note:
            Sets self.frequency to pandas frequency strings:
            - 'D': Daily
            - 'h': Hourly  
            - 'W': Weekly
            - 'M': Monthly
            - Custom: '{seconds}S' for other frequencies
        """
        if self.datetime_column and self.datetime_column in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[self.datetime_column]):
                # Get time differences
                time_diffs = df[self.datetime_column].diff().dropna()
                
                if len(time_diffs) > 0:
                    # Most common difference
                    mode_diff = time_diffs.mode()[0] if len(time_diffs.mode()) > 0 else time_diffs.iloc[0]
                    
                    # Map to frequency string
                    if mode_diff == pd.Timedelta(days=1):
                        self.frequency = 'D'  # Daily
                    elif mode_diff == pd.Timedelta(hours=1):
                        self.frequency = 'h'  # Hourly (lowercase for pandas 2.0+)
                    elif mode_diff == pd.Timedelta(days=7):
                        self.frequency = 'W'  # Weekly
                    elif mode_diff >= pd.Timedelta(days=28) and mode_diff <= pd.Timedelta(days=31):
                        self.frequency = 'M'  # Monthly
                    else:
                        self.frequency = f"{mode_diff.total_seconds()}S"  # Seconds
    
    def create_time_based_split(self, 
                               test_size: float = 0.2,
                               gap: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Create time-based train/validation split.
        
        Splits data chronologically, ensuring validation set comes after
        training set with optional gap to prevent temporal leakage.
        
        Args:
            test_size: Proportion of data for validation (0.0 to 1.0).
                The most recent test_size fraction becomes validation.
            gap: Number of time periods to skip between train and validation.
                Prevents information leakage in time series. For example,
                gap=7 creates a 7-day buffer between sets.
            
        Returns:
            Tuple containing:
                - X_train: Training features
                - X_val: Validation features
                - y_train: Training target
                - y_val: Validation target
                
        Raises:
            ValueError: If no training data loaded or no datetime column
            
        Example:
            >>> # 80/20 split with 7-day gap
            >>> X_train, X_val, y_train, y_val = loader.create_time_based_split(
            ...     test_size=0.2,
            ...     gap=7
            ... )
            >>> 
            >>> # Check temporal ordering
            >>> print(f"Train ends: {X_train.index.max()}")
            >>> print(f"Val starts: {X_val.index.min()}")
            
        Note:
            - Data is sorted by datetime before splitting
            - ID columns are automatically removed from features
            - Gap is applied as number of rows, not time units
        """
        if self.train_data is None:
            raise ValueError("No training data loaded. Call load_data() first.")
            
        if not self.datetime_column:
            raise ValueError("No datetime column detected. Cannot create time-based split.")
            
        # Sort by datetime
        data = self.train_data.sort_values(self.datetime_column)
        
        # Calculate split point
        n_samples = len(data)
        split_idx = int(n_samples * (1 - test_size))
        
        # Apply gap if specified
        if gap:
            split_idx = max(0, split_idx - gap)
            
        # Split data
        train_data = data.iloc[:split_idx]
        val_data = data.iloc[split_idx + (gap or 0):]
        
        # Separate features and target
        X_train = train_data.drop(columns=[self.target_column])
        y_train = train_data[self.target_column]
        X_val = val_data.drop(columns=[self.target_column])
        y_val = val_data[self.target_column]
        
        # Remove ID column if present
        if self.id_column and self.id_column in X_train.columns:
            X_train = X_train.drop(columns=[self.id_column])
            X_val = X_val.drop(columns=[self.id_column])
            
        logger.info(f"Time-based split: Train {len(X_train)} samples, Val {len(X_val)} samples")
        logger.info(f"Train period: {train_data[self.datetime_column].min()} to {train_data[self.datetime_column].max()}")
        logger.info(f"Val period: {val_data[self.datetime_column].min()} to {val_data[self.datetime_column].max()}")
        
        return X_train, X_val, y_train, y_val
    
    def check_temporal_integrity(self) -> Dict[str, any]:
        """Check temporal integrity of the data.
        
        Performs various checks to ensure time series data is properly
        structured and identify potential issues.
        
        Returns:
            Dictionary containing:
                - 'train_duplicates': Number of duplicate timestamps in train
                - 'train_date_range': (min_date, max_date) tuple for train
                - 'test_duplicates': Number of duplicate timestamps in test
                - 'test_date_range': (min_date, max_date) tuple for test
                - 'temporal_overlap': Boolean, True if train/test overlap
                - 'n_gaps': Number of gaps larger than expected frequency
                - 'largest_gap': Largest time gap found
                - 'error': Error message if datetime column not found
                
        Example:
            >>> integrity = loader.check_temporal_integrity()
            >>> 
            >>> if integrity['temporal_overlap']:
            ...     print("WARNING: Test period overlaps with training!")
            >>> 
            >>> if integrity['n_gaps'] > 0:
            ...     print(f"Found {integrity['n_gaps']} gaps in time series")
            ...     print(f"Largest gap: {integrity['largest_gap']}")
            
        Note:
            This check is crucial for preventing data leakage and
            identifying data quality issues in time series.
        """
        if not self.datetime_column:
            return {"error": "No datetime column detected"}
            
        results = {}
        
        # Check for duplicates
        train_dates = self.train_data[self.datetime_column]
        test_dates = self.test_data[self.datetime_column] if self.datetime_column in self.test_data.columns else None
        
        results['train_duplicates'] = train_dates.duplicated().sum()
        results['train_date_range'] = (train_dates.min(), train_dates.max())
        
        if test_dates is not None:
            results['test_duplicates'] = test_dates.duplicated().sum()
            results['test_date_range'] = (test_dates.min(), test_dates.max())
            
            # Check for overlap
            results['temporal_overlap'] = (
                test_dates.min() < train_dates.max() and 
                train_dates.min() < test_dates.max()
            )
        
        # Check for gaps
        if pd.api.types.is_datetime64_any_dtype(train_dates):
            time_diffs = train_dates.diff().dropna()
            expected_freq = time_diffs.mode()[0] if len(time_diffs.mode()) > 0 else time_diffs.iloc[0]
            gaps = time_diffs[time_diffs > expected_freq * 1.5]
            results['n_gaps'] = len(gaps)
            results['largest_gap'] = gaps.max() if len(gaps) > 0 else pd.Timedelta(0)
        
        return results


class TimeSeriesFeatureEngineer:
    """Feature engineering for time series data.
    
    Provides specialized feature engineering methods for temporal data
    including datetime feature extraction, cyclical encoding, lag features,
    and rolling statistics.
    
    Attributes:
        datetime_column: Name of the datetime column
        created_features: List of feature names created by this engineer
        
    Example:
        >>> # Initialize and extract datetime features
        >>> engineer = TimeSeriesFeatureEngineer()
        >>> df_featured = engineer.fit_transform(df, datetime_column='date')
        >>> 
        >>> # Add lag features for target
        >>> df_featured = engineer.create_lag_features(
        ...     df_featured, 
        ...     target_column='sales',
        ...     lags=[1, 7, 14, 30]  # Yesterday, week ago, etc.
        ... )
        >>> 
        >>> # Add rolling statistics
        >>> df_featured = engineer.create_rolling_features(
        ...     df_featured,
        ...     target_column='sales',
        ...     windows=[7, 30],  # Weekly and monthly
        ...     operations=['mean', 'std', 'max']
        ... )
        >>> 
        >>> # For multiple time series (e.g., per store)
        >>> df_featured = engineer.create_lag_features(
        ...     df_featured,
        ...     target_column='sales',
        ...     lags=[1, 7],
        ...     group_columns=['store_id']  # Separate lags per store
        ... )
    """
    
    def __init__(self):
        """Initialize time series feature engineer.
        
        Example:
            >>> engineer = TimeSeriesFeatureEngineer()
        """
        self.datetime_column = None
        self.created_features = []
        
    def fit(self, X: pd.DataFrame, datetime_column: str) -> 'TimeSeriesFeatureEngineer':
        """Fit the feature engineer.
        
        Stores the datetime column name for use in transform.
        
        Args:
            X: Input DataFrame containing the datetime column
            datetime_column: Name of the datetime column to use
            
        Returns:
            Self for method chaining
            
        Example:
            >>> engineer = TimeSeriesFeatureEngineer()
            >>> engineer.fit(train_df, datetime_column='transaction_date')
        """
        self.datetime_column = datetime_column
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data by adding time series features.
        
        Extracts datetime components and creates cyclical encodings.
        Must call fit() first or use fit_transform().
        
        Args:
            X: Input DataFrame to transform
            
        Returns:
            DataFrame with added time series features:
                - Basic: year, month, day, dayofweek, quarter, etc.
                - Binary: is_weekend, is_month_start, is_month_end, etc.
                - Cyclical: sin/cos encodings for periodic features
                - Time-based: hour, minute (if applicable)
                
        Example:
            >>> engineer.fit(train_df, 'date')
            >>> train_featured = engineer.transform(train_df)
            >>> test_featured = engineer.transform(test_df)
            >>> 
            >>> # Check created features
            >>> print(engineer.created_features)
            
        Note:
            - Datetime column is preserved in the output
            - All created features are prefixed with datetime column name
            - Cyclical features help models understand periodicity
        """
        X = X.copy()
        
        if self.datetime_column not in X.columns:
            logger.warning(f"Datetime column '{self.datetime_column}' not found in DataFrame")
            return X
            
        # Ensure datetime type
        if not pd.api.types.is_datetime64_any_dtype(X[self.datetime_column]):
            X[self.datetime_column] = pd.to_datetime(X[self.datetime_column])
            
        # Extract datetime features
        X = self._extract_datetime_features(X)
        
        # Create cyclical features
        X = self._create_cyclical_features(X)
        
        return X
    
    def fit_transform(self, X: pd.DataFrame, datetime_column: str) -> pd.DataFrame:
        """Fit and transform in one step.
        
        Convenience method combining fit and transform.
        
        Args:
            X: Input DataFrame
            datetime_column: Name of datetime column
            
        Returns:
            Transformed DataFrame with time series features
            
        Example:
            >>> df_featured = engineer.fit_transform(df, 'date')
        """
        return self.fit(X, datetime_column).transform(X)
    
    def _extract_datetime_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract basic datetime features.
        
        Creates interpretable datetime component features.
        
        Args:
            X: DataFrame with datetime column
            
        Returns:
            DataFrame with added datetime features
            
        Note:
            Hour features only added if time component varies.
        """
        dt_col = X[self.datetime_column]
        
        # Basic features
        features = {
            'year': dt_col.dt.year,
            'month': dt_col.dt.month,
            'day': dt_col.dt.day,
            'dayofweek': dt_col.dt.dayofweek,
            'dayofyear': dt_col.dt.dayofyear,
            'quarter': dt_col.dt.quarter,
            'is_weekend': dt_col.dt.dayofweek.isin([5, 6]).astype(int),
            'is_month_start': dt_col.dt.is_month_start.astype(int),
            'is_month_end': dt_col.dt.is_month_end.astype(int),
            'is_quarter_start': dt_col.dt.is_quarter_start.astype(int),
            'is_quarter_end': dt_col.dt.is_quarter_end.astype(int),
        }
        
        # Add hour features if timestamp includes time
        if dt_col.dt.hour.std() > 0:  # Check if there's time variation
            features.update({
                'hour': dt_col.dt.hour,
                'minute': dt_col.dt.minute,
                'is_business_hour': dt_col.dt.hour.between(9, 17).astype(int)
            })
        
        # Add features to DataFrame
        for name, feature in features.items():
            X[f'{self.datetime_column}_{name}'] = feature
            self.created_features.append(f'{self.datetime_column}_{name}')
            
        return X
    
    def _create_cyclical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create cyclical features using sin/cos transformation.
        
        Encodes periodic features to preserve circular relationships
        (e.g., day 31 is close to day 1, hour 23 is close to hour 0).
        
        Args:
            X: DataFrame with datetime features
            
        Returns:
            DataFrame with added cyclical features
            
        Note:
            Creates both sin and cos for each periodic feature to
            uniquely represent each point on the circle.
        """
        dt_col = X[self.datetime_column]
        
        # Day of week cyclical
        X[f'{self.datetime_column}_dayofweek_sin'] = np.sin(2 * np.pi * dt_col.dt.dayofweek / 7)
        X[f'{self.datetime_column}_dayofweek_cos'] = np.cos(2 * np.pi * dt_col.dt.dayofweek / 7)
        
        # Day of month cyclical
        X[f'{self.datetime_column}_day_sin'] = np.sin(2 * np.pi * dt_col.dt.day / 31)
        X[f'{self.datetime_column}_day_cos'] = np.cos(2 * np.pi * dt_col.dt.day / 31)
        
        # Month cyclical
        X[f'{self.datetime_column}_month_sin'] = np.sin(2 * np.pi * dt_col.dt.month / 12)
        X[f'{self.datetime_column}_month_cos'] = np.cos(2 * np.pi * dt_col.dt.month / 12)
        
        # Hour cyclical (if applicable)
        if dt_col.dt.hour.std() > 0:
            X[f'{self.datetime_column}_hour_sin'] = np.sin(2 * np.pi * dt_col.dt.hour / 24)
            X[f'{self.datetime_column}_hour_cos'] = np.cos(2 * np.pi * dt_col.dt.hour / 24)
            
        # Update created features list
        for col in X.columns:
            if col.startswith(f'{self.datetime_column}_') and col.endswith(('_sin', '_cos')):
                if col not in self.created_features:
                    self.created_features.append(col)
                    
        return X
    
    def create_lag_features(self, X: pd.DataFrame, target_column: str, 
                          lags: List[int] = [1, 7, 14, 30],
                          group_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Create lag features.
        
        Adds lagged values of the target variable as features, essential
        for time series modeling.
        
        Args:
            X: Input DataFrame with target column (must be sorted by time)
            target_column: Name of column to create lags for
            lags: List of lag periods. Each creates a feature with
                target value from that many periods ago.
                Common choices:
                - [1]: Previous period only
                - [1, 7]: Daily data with weekly seasonality  
                - [1, 7, 14, 30]: Multiple timescales
            group_columns: For panel data with multiple time series.
                Creates lags within each group separately.
                Example: ['store_id', 'product_id']
            
        Returns:
            DataFrame with added lag features named '{target}_lag_{n}'
            
        Example:
            >>> # Single time series
            >>> df = engineer.create_lag_features(
            ...     df, target_column='sales', lags=[1, 7, 30]
            ... )
            >>> 
            >>> # Multiple time series (e.g., sales per store)
            >>> df = engineer.create_lag_features(
            ...     df, 
            ...     target_column='sales',
            ...     lags=[1, 7],
            ...     group_columns=['store_id']
            ... )
            
        Warning:
            - Creates NaN values for initial periods (handled by models)
            - Ensure data is sorted chronologically before calling
            - Be careful not to create leakage with future information
        """
        X = X.copy()
        
        if group_columns:
            # Create lags within groups
            for lag in lags:
                X[f'{target_column}_lag_{lag}'] = X.groupby(group_columns)[target_column].shift(lag)
                self.created_features.append(f'{target_column}_lag_{lag}')
        else:
            # Create simple lags
            for lag in lags:
                X[f'{target_column}_lag_{lag}'] = X[target_column].shift(lag)
                self.created_features.append(f'{target_column}_lag_{lag}')
                
        return X
    
    def create_rolling_features(self, X: pd.DataFrame, target_column: str,
                              windows: List[int] = [7, 14, 30],
                              operations: List[str] = ['mean', 'std', 'min', 'max'],
                              group_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Create rolling window features.
        
        Computes rolling statistics over specified windows, capturing
        trends and patterns at different timescales.
        
        Args:
            X: Input DataFrame with target column (must be sorted by time)
            target_column: Name of column to compute rolling stats for
            windows: List of window sizes (number of periods).
                Example: [7, 30] for weekly and monthly windows
            operations: Statistical operations to apply. Options:
                - 'mean': Rolling average
                - 'std': Rolling standard deviation (volatility)
                - 'min': Rolling minimum
                - 'max': Rolling maximum
                - 'sum': Rolling sum
                - 'median': Rolling median
            group_columns: For panel data, compute rolling stats
                within each group separately
            
        Returns:
            DataFrame with rolling features named 
            '{target}_rolling_{window}_{operation}'
            
        Example:
            >>> # Add rolling features at multiple scales
            >>> df = engineer.create_rolling_features(
            ...     df,
            ...     target_column='temperature',
            ...     windows=[7, 30],  # Weekly and monthly
            ...     operations=['mean', 'std', 'max']
            ... )
            >>> 
            >>> # For multiple series
            >>> df = engineer.create_rolling_features(
            ...     df,
            ...     target_column='sales',
            ...     windows=[7, 14],
            ...     group_columns=['store_id']
            ... )
            
        Note:
            - Uses min_periods=1 to handle initial periods
            - Captures both level (mean) and volatility (std)
            - Larger windows smooth out noise but increase lag
        """
        X = X.copy()
        
        for window in windows:
            for op in operations:
                feature_name = f'{target_column}_rolling_{window}_{op}'
                
                if group_columns:
                    # Rolling within groups
                    X[feature_name] = X.groupby(group_columns)[target_column].transform(
                        lambda x: x.rolling(window, min_periods=1).agg(op)
                    )
                else:
                    # Simple rolling
                    X[feature_name] = X[target_column].rolling(window, min_periods=1).agg(op)
                    
                self.created_features.append(feature_name)
                
        return X
    
    def get_created_features(self) -> List[str]:
        """Get list of created feature names.
        
        Returns:
            List of all feature names created by this engineer
            
        Example:
            >>> features = engineer.get_created_features()
            >>> print(f"Created {len(features)} time series features:")
            >>> print(features[:10])  # Show first 10
        """
        return self.created_features.copy()