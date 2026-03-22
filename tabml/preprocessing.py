"""Data preprocessing module with comprehensive transformation capabilities.

This module provides a complete preprocessing pipeline for tabular data that
prevents data leakage and handles various data types intelligently.

Classes:
    DataProcessor: Main preprocessing pipeline with configurable transformations
    
Example:
    Basic usage::
    
        from tabml.preprocessing import DataProcessor
        
        # Create processor with custom config
        processor = DataProcessor(config={
            'categorical_encoding': {'method': 'target'},
            'scaling': {'method': 'robust'}
        })
        
        # Fit and transform
        X_train_processed = processor.fit_transform(X_train, y_train)
        X_test_processed = processor.transform(X_test)
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, 
    LabelEncoder, OneHotEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from category_encoders import TargetEncoder, BinaryEncoder, HashingEncoder
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    """Comprehensive data preprocessing pipeline that prevents data leakage.
    
    This class provides a unified interface for all preprocessing steps needed
    for tabular machine learning, with automatic column type detection and
    configurable transformation methods.
    
    Features:
        - Automatic column type detection (numeric, categorical, text, datetime)
        - Multiple categorical encoding methods (one-hot, label, target, etc.)
        - Text vectorization with TF-IDF and count vectorization
        - Flexible scaling options (standard, minmax, robust)
        - Intelligent missing data imputation
        - High cardinality feature handling
        - Datetime feature extraction
        - Prevention of data leakage through proper fit/transform separation
        
    Attributes:
        config: Configuration dictionary for all preprocessing options
        imputers: Dictionary of fitted imputers by column type
        encoders: Dictionary of fitted encoders by column type
        scalers: Dictionary of fitted scalers
        text_vectorizers: Dictionary of fitted text vectorizers
        numeric_columns: List of detected numeric column names
        categorical_columns: List of detected categorical column names
        text_columns: List of detected text column names
        datetime_columns: List of detected datetime column names
        high_cardinality_columns: List of high cardinality categorical columns
        is_fitted: Whether the processor has been fitted
        feature_names_out_: Feature names after transformation
        
    Example:
        >>> # Create processor with custom configuration
        >>> config = {
        ...     'categorical_encoding': {'method': 'target'},
        ...     'scaling': {'method': 'robust'},
        ...     'imputation': {'numeric_strategy': 'median'}
        ... }
        >>> processor = DataProcessor(config=config)
        >>> 
        >>> # Fit on training data
        >>> processor.fit(X_train, y_train)
        >>> 
        >>> # Transform both train and test
        >>> X_train_processed = processor.transform(X_train)
        >>> X_test_processed = processor.transform(X_test)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize DataProcessor with configuration.
        
        Args:
            config: Configuration dictionary with preprocessing options.
                If None, uses default configuration. Custom config is deep-merged
                with defaults, so you only need to specify options you want to change.
                
        Note:
            See _get_default_config() for all available configuration options.
            
        Example:
            >>> # Use all defaults
            >>> processor = DataProcessor()
            >>> 
            >>> # Override specific options
            >>> processor = DataProcessor(config={
            ...     'categorical_encoding': {'method': 'target'},
            ...     'scaling': {'method': 'none'}
            ... })
        """
        # Start with default config
        default_config = self._get_default_config()
        
        # If custom config provided, deep merge with defaults
        if config:
            self.config = self._deep_merge_configs(default_config, config)
        else:
            self.config = default_config
        
        # Initialize component dictionaries
        self.imputers = {}
        self.encoders = {}
        self.scalers = {}
        self.text_vectorizers = {}
        
        # Column type tracking
        self.numeric_columns = []
        self.categorical_columns = []
        self.text_columns = []
        self.datetime_columns = []
        self.high_cardinality_columns = []
        
        # Fitted flag
        self.is_fitted = False
        
        # Feature names after transformation
        self.feature_names_out_ = []
        
        logger.info("DataProcessor initialized with config: %s", self.config)
    
    def _deep_merge_configs(self, default: Dict, custom: Dict) -> Dict:
        """Deep merge custom config with default config.
        
        Recursively merges custom configuration into default configuration,
        preserving nested structure and only overriding specified values.
        
        Args:
            default: Default configuration dictionary
            custom: Custom configuration to merge
            
        Returns:
            Merged configuration dictionary
            
        Example:
            >>> default = {'a': {'b': 1, 'c': 2}, 'd': 3}
            >>> custom = {'a': {'b': 10}}
            >>> merged = _deep_merge_configs(default, custom)
            >>> # Result: {'a': {'b': 10, 'c': 2}, 'd': 3}
        """
        result = default.copy()
        
        for key, value in custom.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration.
        
        Returns:
            Dictionary with all default preprocessing options:
                - categorical_encoding: Options for encoding categorical variables
                - text_processing: Options for text vectorization  
                - scaling: Options for numeric feature scaling
                - imputation: Options for handling missing values
                - drop_columns: List of columns to drop
                - datetime_features: Whether to extract datetime features
                - handle_zeros_as_missing: Treat zeros as missing values
                
        Note:
            This method defines all available configuration options and their
            default values. Users can override any subset of these options.
        """
        return {
            # Categorical encoding
            'categorical_encoding': {
                'method': 'onehot',  # onehot, label, ordinal, target, binary, hashing
                'handle_unknown': 'ignore',
                'high_cardinality_threshold': 50,
                'high_cardinality_method': 'hashing'  # hashing, target, frequency
            },
            
            # Text processing
            'text_processing': {
                'method': 'tfidf',  # tfidf, count, none
                'max_features': 100,
                'ngram_range': (1, 2),
                'min_df': 2,
                'max_df': 0.95
            },
            
            # Scaling
            'scaling': {
                'method': 'standard',  # standard, minmax, robust, none
                'feature_range': (0, 1),  # for minmax
                'with_centering': True  # for robust
            },
            
            # Imputation
            'imputation': {
                'numeric_strategy': 'median',  # mean, median, most_frequent, constant
                'categorical_strategy': 'most_frequent',  # most_frequent, constant
                'text_strategy': 'constant',
                'constant_fill_value': 'missing',
                'add_indicator': False  # Add binary indicators for missing values
            },
            
            # General options
            'drop_columns': [],  # Columns to drop
            'datetime_features': True,  # Extract features from datetime columns
            'handle_zeros_as_missing': False
        }
    
    def detect_column_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Automatically detect column types.
        
        Analyzes DataFrame columns to categorize them into appropriate types
        for preprocessing. Uses heuristics like unique value ratios, average
        string length, and content analysis.
        
        Args:
            df: Input DataFrame to analyze
            
        Returns:
            Dictionary mapping column types to lists of column names:
                - 'numeric': Numeric columns (int, float)
                - 'categorical': Low-cardinality categorical columns
                - 'text': Text columns (long strings, multiple words)
                - 'datetime': Datetime columns
                - 'high_cardinality': High-cardinality categorical columns
                
        Example:
            >>> column_types = processor.detect_column_types(df)
            >>> print(f"Numeric columns: {column_types['numeric']}")
            >>> print(f"Text columns: {column_types['text']}")
            
        Note:
            - Text detection looks for long strings or presence of spaces
            - High cardinality threshold is configurable
            - Columns in drop_columns config are excluded
        """
        column_types = {
            'numeric': [],
            'categorical': [],
            'text': [],
            'datetime': [],
            'high_cardinality': []
        }
        
        for col in df.columns:
            if col in self.config.get('drop_columns', []):
                continue
                
            # Check for datetime
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                column_types['datetime'].append(col)
            
            # Check for numeric
            elif pd.api.types.is_numeric_dtype(df[col]):
                column_types['numeric'].append(col)
            
            # Check for text vs categorical
            elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                # Estimate if it's text based on unique values and length
                unique_ratio = df[col].nunique() / len(df)
                # Handle categorical dtype columns by converting to string first
                if pd.api.types.is_categorical_dtype(df[col]):
                    avg_length = df[col].astype(str).str.len().mean()
                else:
                    avg_length = df[col].fillna('').astype(str).str.len().mean()
                
                # Consider as text if average length is long or contains multiple words
                sample_values = df[col].dropna().astype(str).head(10)
                contains_spaces = any(' ' in val for val in sample_values)
                
                if avg_length > 50 or (contains_spaces and avg_length > 20):
                    column_types['text'].append(col)
                else:
                    n_unique = df[col].nunique()
                    threshold = self.config['categorical_encoding']['high_cardinality_threshold']
                    
                    if n_unique > threshold:
                        column_types['high_cardinality'].append(col)
                    else:
                        column_types['categorical'].append(col)
        
        return column_types
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DataProcessor':
        """Fit the data processor on training data.
        
        Learns all preprocessing parameters from the training data including
        imputation values, encoding mappings, scaling parameters, and text
        vocabulary.
        
        Args:
            X: Training features as pandas DataFrame
            y: Target variable as pandas Series. Required for target encoding,
                optional for other encoding methods.
            
        Returns:
            Self instance for method chaining
            
        Raises:
            ValueError: If X is not a pandas DataFrame
            
        Example:
            >>> processor = DataProcessor()
            >>> processor.fit(X_train, y_train)
            >>> # Now processor can transform any compatible data
            >>> X_transformed = processor.transform(X_test)
            
        Note:
            - Column types are automatically detected during fit
            - All transformation parameters are learned from X
            - The processor must be fitted before calling transform
        """
        logger.info(f"Fitting DataProcessor on data with shape {X.shape}")
        
        # Detect column types
        column_types = self.detect_column_types(X)
        self.numeric_columns = column_types['numeric']
        self.categorical_columns = column_types['categorical']
        self.text_columns = column_types['text']
        self.datetime_columns = column_types['datetime']
        self.high_cardinality_columns = column_types['high_cardinality']
        
        logger.info(f"Detected column types: {len(self.numeric_columns)} numeric, "
                   f"{len(self.categorical_columns)} categorical, "
                   f"{len(self.text_columns)} text, "
                   f"{len(self.datetime_columns)} datetime, "
                   f"{len(self.high_cardinality_columns)} high cardinality")
        
        # Fit imputers
        self._fit_imputers(X)
        
        # Apply imputation to get complete data for fitting other transformers
        X_imputed = self._apply_imputation(X.copy())
        
        # Fit encoders
        self._fit_encoders(X_imputed, y)
        
        # Fit text vectorizers
        self._fit_text_vectorizers(X_imputed)
        
        # Apply encoding and vectorization to get numeric data for scaling
        X_encoded = self._apply_encoding(X_imputed.copy())
        X_vectorized = self._apply_text_vectorization(X_encoded)
        
        # Add missing indicators if configured (need to do this before fitting scalers)
        if self.config['imputation'].get('add_indicator', False):
            missing_masks = {}
            for col in X.columns:
                if X[col].isnull().any():
                    missing_masks[col] = X[col].isnull()
            
            for col, mask in missing_masks.items():
                X_vectorized[f'{col}_was_missing'] = mask.astype(int)
        
        # Fit scalers
        self._fit_scalers(X_vectorized)
        
        self.is_fitted = True
        logger.info("DataProcessor fitting completed")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted preprocessing pipeline.
        
        Applies all preprocessing transformations learned during fit to new data.
        Handles missing columns gracefully and maintains consistency with training.
        
        Args:
            X: Data to transform as pandas DataFrame. Should have same column
                structure as data used in fit (missing columns are handled).
            
        Returns:
            Transformed DataFrame with all preprocessing applied
            
        Raises:
            ValueError: If DataProcessor hasn't been fitted yet
            
        Example:
            >>> # Must fit first
            >>> processor.fit(X_train)
            >>> 
            >>> # Transform test data
            >>> X_test_processed = processor.transform(X_test)
            >>> 
            >>> # Transform new data for prediction  
            >>> X_new_processed = processor.transform(X_new)
            
        Note:
            - Missing columns are handled with appropriate defaults
            - New categories in categorical columns are handled based on config
            - Feature names are stored in feature_names_out_ attribute
        """
        if not self.is_fitted:
            raise ValueError("DataProcessor must be fitted before transform")
        
        logger.info(f"Transforming data with shape {X.shape}")
        
        X_transformed = X.copy()
        
        # Drop specified columns
        drop_cols = [col for col in self.config.get('drop_columns', []) if col in X_transformed.columns]
        if drop_cols:
            X_transformed = X_transformed.drop(columns=drop_cols)
        
        # Extract datetime features
        if self.config.get('datetime_features', True) and self.datetime_columns:
            X_transformed = self._extract_datetime_features(X_transformed)
        
        # Apply imputation
        X_transformed = self._apply_imputation(X_transformed)
        
        # Apply encoding
        X_transformed = self._apply_encoding(X_transformed)
        
        # Apply text vectorization
        X_transformed = self._apply_text_vectorization(X_transformed)
        
        # Apply scaling
        X_transformed = self._apply_scaling(X_transformed)
        
        # Store feature names
        self.feature_names_out_ = X_transformed.columns.tolist()
        
        logger.info(f"Transformation completed. Output shape: {X_transformed.shape}")
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step.
        
        Convenience method that combines fit and transform operations.
        Equivalent to calling fit(X, y) followed by transform(X).
        
        Args:
            X: Training features to fit and transform
            y: Target variable (optional, needed for target encoding)
            
        Returns:
            Transformed training data
            
        Example:
            >>> X_train_processed = processor.fit_transform(X_train, y_train)
            >>> # Equivalent to:
            >>> # processor.fit(X_train, y_train)
            >>> # X_train_processed = processor.transform(X_train)
        """
        return self.fit(X, y).transform(X)
    
    def _fit_imputers(self, X: pd.DataFrame):
        """Fit imputers for different column types.
        
        Creates and fits separate imputers for numeric, categorical, and text
        columns based on configured strategies.
        
        Args:
            X: DataFrame to fit imputers on
            
        Note:
            Imputation strategies are configured via the 'imputation' config key.
            Different strategies can be used for different column types.
        """
        # Numeric imputer
        if self.numeric_columns:
            strategy = self.config['imputation']['numeric_strategy']
            self.imputers['numeric'] = SimpleImputer(strategy=strategy)
            self.imputers['numeric'].fit(X[self.numeric_columns])
        
        # Categorical imputer
        categorical_cols = self.categorical_columns + self.high_cardinality_columns
        if categorical_cols:
            strategy = self.config['imputation']['categorical_strategy']
            fill_value = self.config['imputation']['constant_fill_value']
            self.imputers['categorical'] = SimpleImputer(
                strategy=strategy,
                fill_value=fill_value if strategy == 'constant' else None
            )
            self.imputers['categorical'].fit(X[categorical_cols])
        
        # Text imputer
        if self.text_columns:
            strategy = self.config['imputation']['text_strategy']
            fill_value = self.config['imputation']['constant_fill_value']
            self.imputers['text'] = SimpleImputer(
                strategy=strategy,
                fill_value=fill_value
            )
            self.imputers['text'].fit(X[self.text_columns])
    
    def _fit_encoders(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit encoders for categorical columns.
        
        Creates and fits appropriate encoders based on configuration for both
        regular and high-cardinality categorical columns.
        
        Args:
            X: DataFrame with categorical columns to encode
            y: Target variable (required for target encoding)
            
        Note:
            - Regular and high-cardinality columns use different strategies
            - Encoding method is configured via 'categorical_encoding' config
            - Supports: onehot, label, ordinal, target, binary, hashing
        """
        method = self.config['categorical_encoding']['method']
        
        # Regular categorical columns
        if self.categorical_columns:
            if method == 'onehot':
                self.encoders['categorical'] = OneHotEncoder(
                    sparse_output=False,
                    handle_unknown=self.config['categorical_encoding']['handle_unknown'],
                    drop='first'
                )
                self.encoders['categorical'].fit(X[self.categorical_columns])
            
            elif method == 'label':
                self.encoders['categorical'] = {}
                for col in self.categorical_columns:
                    le = LabelEncoder()
                    le.fit(X[col].fillna('missing'))
                    self.encoders['categorical'][col] = le
            
            elif method == 'ordinal':
                self.encoders['categorical'] = OrdinalEncoder(
                    handle_unknown='use_encoded_value',
                    unknown_value=-1
                )
                self.encoders['categorical'].fit(X[self.categorical_columns])
            
            elif method == 'target' and y is not None:
                self.encoders['categorical'] = TargetEncoder()
                self.encoders['categorical'].fit(X[self.categorical_columns], y)
            
            elif method == 'binary':
                self.encoders['categorical'] = BinaryEncoder()
                self.encoders['categorical'].fit(X[self.categorical_columns])
        
        # High cardinality columns
        if self.high_cardinality_columns:
            hc_method = self.config['categorical_encoding']['high_cardinality_method']
            
            if hc_method == 'hashing':
                # Use fewer components for high cardinality
                n_components = min(32, len(self.high_cardinality_columns) * 8)
                self.encoders['high_cardinality'] = HashingEncoder(
                    n_components=n_components,
                    return_df=True
                )
                self.encoders['high_cardinality'].fit(X[self.high_cardinality_columns])
            
            elif hc_method == 'target' and y is not None:
                self.encoders['high_cardinality'] = TargetEncoder()
                self.encoders['high_cardinality'].fit(X[self.high_cardinality_columns], y)
            
            elif hc_method == 'frequency':
                # Implement frequency encoding
                self.encoders['high_cardinality'] = {}
                for col in self.high_cardinality_columns:
                    freq_map = X[col].value_counts().to_dict()
                    self.encoders['high_cardinality'][col] = freq_map
    
    def _fit_text_vectorizers(self, X: pd.DataFrame):
        """Fit text vectorizers.
        
        Creates and fits text vectorizers (TF-IDF or count) for each text column
        based on configuration.
        
        Args:
            X: DataFrame with text columns to vectorize
            
        Note:
            - Each text column gets its own vectorizer
            - Vectorization parameters configured via 'text_processing' config
            - Supports TF-IDF and count vectorization
        """
        if not self.text_columns:
            return
        
        method = self.config['text_processing']['method']
        
        if method == 'none':
            return
        
        for col in self.text_columns:
            if method == 'tfidf':
                vectorizer = TfidfVectorizer(
                    max_features=self.config['text_processing']['max_features'],
                    ngram_range=self.config['text_processing']['ngram_range'],
                    min_df=self.config['text_processing']['min_df'],
                    max_df=self.config['text_processing']['max_df']
                )
            elif method == 'count':
                vectorizer = CountVectorizer(
                    max_features=self.config['text_processing']['max_features'],
                    ngram_range=self.config['text_processing']['ngram_range'],
                    min_df=self.config['text_processing']['min_df'],
                    max_df=self.config['text_processing']['max_df']
                )
            
            vectorizer.fit(X[col].fillna(''))
            self.text_vectorizers[col] = vectorizer
    
    def _fit_scalers(self, X: pd.DataFrame):
        """Fit scalers on numeric columns.
        
        Creates and fits scalers for numeric features based on configuration.
        Missing indicator columns are excluded from scaling.
        
        Args:
            X: DataFrame with numeric columns to scale
            
        Note:
            - Scaling method configured via 'scaling' config  
            - Supports: standard, minmax, robust, none
            - Missing indicators (_was_missing columns) are not scaled
        """
        method = self.config['scaling']['method']
        
        if method == 'none':
            return
        
        # Get numeric columns after all transformations
        # Exclude missing indicator columns from scaling
        numeric_cols = []
        for col in X.select_dtypes(include=[np.number]).columns:
            if not col.endswith('_was_missing'):
                numeric_cols.append(col)
        
        if not numeric_cols:
            return
        
        if method == 'standard':
            self.scalers['main'] = StandardScaler()
        elif method == 'minmax':
            self.scalers['main'] = MinMaxScaler(
                feature_range=self.config['scaling']['feature_range']
            )
        elif method == 'robust':
            self.scalers['main'] = RobustScaler(
                with_centering=self.config['scaling']['with_centering']
            )
        
        self.scalers['main'].fit(X[numeric_cols])
        self.scalers['numeric_cols'] = numeric_cols
    
    def _apply_imputation(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply imputation to data.
        
        Applies fitted imputers to handle missing values in the data.
        Optionally adds binary indicators for missingness.
        
        Args:
            X: DataFrame with potential missing values
            
        Returns:
            DataFrame with missing values imputed
            
        Note:
            - Different imputation strategies are used for different column types
            - Missing indicators are added if configured
            - Handles missing columns gracefully with default values
        """
        # Track missing values before imputation if indicators requested
        missing_masks = {}
        if self.config['imputation'].get('add_indicator', False):
            for col in X.columns:
                if X[col].isnull().any():
                    missing_masks[col] = X[col].isnull()
        
        # Numeric imputation
        if self.numeric_columns and 'numeric' in self.imputers:
            num_cols = [col for col in self.numeric_columns if col in X.columns]
            if num_cols:
                X[num_cols] = self.imputers['numeric'].transform(X[num_cols])
        
        # Categorical imputation
        if 'categorical' in self.imputers:
            # Only transform columns that exist in current data and were seen during fit
            fitted_cat_cols = self.categorical_columns + self.high_cardinality_columns
            cat_cols_to_transform = [col for col in fitted_cat_cols if col in X.columns]
            
            if cat_cols_to_transform:
                # Create a temporary DataFrame with only the columns that exist
                temp_df = pd.DataFrame(index=X.index)
                
                # Add columns in the order they were fitted
                for col in fitted_cat_cols:
                    if col in X.columns:
                        temp_df[col] = X[col]
                    else:
                        # Add a dummy column with the fill value
                        temp_df[col] = self.config['imputation']['constant_fill_value']
                
                # Transform all columns
                transformed = self.imputers['categorical'].transform(temp_df)
                
                # Only update columns that exist in the input
                for i, col in enumerate(fitted_cat_cols):
                    if col in X.columns:
                        X[col] = transformed[:, i]
        
        # Text imputation
        text_cols = [col for col in self.text_columns if col in X.columns]
        if text_cols and 'text' in self.imputers:
            X[text_cols] = self.imputers['text'].transform(X[text_cols])
        
        # Add missing indicators if requested
        if self.config['imputation'].get('add_indicator', False):
            for col, mask in missing_masks.items():
                X[f'{col}_was_missing'] = mask.astype(int)
        
        return X
    
    def _apply_encoding(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply categorical encoding.
        
        Transforms categorical columns using fitted encoders. Handles both
        regular and high-cardinality categorical features.
        
        Args:
            X: DataFrame with categorical columns to encode
            
        Returns:
            DataFrame with categorical columns encoded
            
        Note:
            - Missing columns are handled with appropriate defaults
            - Unknown categories are handled based on encoder configuration
            - Original categorical columns are replaced with encoded versions
        """
        # Regular categorical columns
        if self.categorical_columns and 'categorical' in self.encoders:
            method = self.config['categorical_encoding']['method']
            
            if method == 'onehot':
                # Handle missing columns by creating dummy data
                temp_df = pd.DataFrame(index=X.index)
                for col in self.categorical_columns:
                    if col in X.columns:
                        temp_df[col] = X[col]
                    else:
                        # Add dummy column with constant value
                        temp_df[col] = self.config['imputation']['constant_fill_value']
                
                encoded = self.encoders['categorical'].transform(temp_df)
                
                # Get all feature names from encoder
                all_feature_names = self.encoders['categorical'].get_feature_names_out(self.categorical_columns)
                
                # Create DataFrame with all encoded features
                encoded_df = pd.DataFrame(encoded, columns=all_feature_names, index=X.index)
                
                # Drop original categorical columns that exist in X
                cat_cols_to_drop = [col for col in self.categorical_columns if col in X.columns]
                X = X.drop(columns=cat_cols_to_drop)
                X = pd.concat([X, encoded_df], axis=1)
                
            elif method == 'label':
                for col in self.categorical_columns:
                    if col in X.columns and col in self.encoders['categorical']:
                        le = self.encoders['categorical'][col]
                        # Handle unknown values
                        col_data = X[col].fillna('missing')
                        col_data = col_data.apply(lambda x: x if x in le.classes_ else 'missing')
                        X[col] = le.transform(col_data)
                
            elif method == 'ordinal':
                # Handle missing columns  
                temp_df = pd.DataFrame(index=X.index)
                for col in self.categorical_columns:
                    if col in X.columns:
                        temp_df[col] = X[col]
                    else:
                        temp_df[col] = self.config['imputation']['constant_fill_value']
                        
                transformed = self.encoders['categorical'].transform(temp_df)
                
                # Update only columns that exist in X
                for i, col in enumerate(self.categorical_columns):
                    if col in X.columns:
                        X[col] = transformed[:, i]
                
            elif method in ['target', 'binary']:
                cat_cols = [col for col in self.categorical_columns if col in X.columns]
                if cat_cols:
                    encoded = self.encoders['categorical'].transform(X[cat_cols])
                    if isinstance(encoded, pd.DataFrame):
                        X = X.drop(columns=cat_cols)
                        X = pd.concat([X, encoded], axis=1)
                    else:
                        X[cat_cols] = encoded
        
        # High cardinality columns
        if self.high_cardinality_columns and 'high_cardinality' in self.encoders:
            hc_cols = [col for col in self.high_cardinality_columns if col in X.columns]
            
            if hc_cols:
                hc_method = self.config['categorical_encoding']['high_cardinality_method']
                
                if hc_method in ['hashing', 'target']:
                    encoded = self.encoders['high_cardinality'].transform(X[hc_cols])
                    X = X.drop(columns=hc_cols)
                    X = pd.concat([X, encoded], axis=1)
                
                elif hc_method == 'frequency':
                    for col in hc_cols:
                        if col in self.encoders['high_cardinality']:
                            freq_map = self.encoders['high_cardinality'][col]
                            X[col] = X[col].map(freq_map).fillna(0)
        
        return X
    
    def _apply_text_vectorization(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply text vectorization.
        
        Transforms text columns into numeric features using fitted vectorizers.
        
        Args:
            X: DataFrame with text columns to vectorize
            
        Returns:
            DataFrame with text columns replaced by vectorized features
            
        Note:
            - Each text column is expanded into multiple features
            - Original text columns are dropped after vectorization
            - Feature names include the original column name as prefix
        """
        if not self.text_columns or self.config['text_processing']['method'] == 'none':
            return X
        
        for col in self.text_columns:
            if col in X.columns and col in self.text_vectorizers:
                vectorizer = self.text_vectorizers[col]
                
                # Transform text
                text_data = X[col].fillna('')
                vectorized = vectorizer.transform(text_data)
                
                # Create feature names
                feature_names = [f'{col}_{feat}' for feat in vectorizer.get_feature_names_out()]
                
                # Convert to DataFrame
                if hasattr(vectorized, 'toarray'):
                    vectorized = vectorized.toarray()
                vectorized_df = pd.DataFrame(vectorized, columns=feature_names, index=X.index)
                
                # Replace original column with vectorized features
                X = X.drop(columns=[col])
                X = pd.concat([X, vectorized_df], axis=1)
        
        return X
    
    def _apply_scaling(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply scaling to numeric features.
        
        Scales numeric features using fitted scalers while preserving
        non-numeric and indicator columns.
        
        Args:
            X: DataFrame with numeric features to scale
            
        Returns:
            DataFrame with numeric features scaled
            
        Note:
            - Only scales columns that were present during fit
            - Missing indicator columns are not scaled
            - Non-numeric columns are preserved unchanged
        """
        if self.config['scaling']['method'] == 'none' or 'main' not in self.scalers:
            return X
        
        # Get numeric columns that exist in current data
        numeric_cols = [col for col in self.scalers['numeric_cols'] if col in X.columns]
        
        if numeric_cols:
            # Only transform columns that exist in both the fitted scaler and current data
            X[numeric_cols] = self.scalers['main'].transform(X[numeric_cols])
        
        return X
    
    def _extract_datetime_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract features from datetime columns.
        
        Converts datetime columns into multiple numeric features including
        year, month, day, day of week, and cyclical encodings.
        
        Args:
            X: DataFrame with datetime columns
            
        Returns:
            DataFrame with datetime columns replaced by numeric features
            
        Note:
            - Original datetime columns are dropped
            - Includes cyclical encoding for periodic features
            - Adds is_weekend binary indicator
            - Handles datetime parsing errors gracefully
        """
        for col in self.datetime_columns:
            if col not in X.columns:
                continue
            
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(X[col]):
                X[col] = pd.to_datetime(X[col], errors='coerce')
            
            # Extract various datetime features
            X[f'{col}_year'] = X[col].dt.year
            X[f'{col}_month'] = X[col].dt.month
            X[f'{col}_day'] = X[col].dt.day
            X[f'{col}_dayofweek'] = X[col].dt.dayofweek
            X[f'{col}_quarter'] = X[col].dt.quarter
            X[f'{col}_is_weekend'] = (X[col].dt.dayofweek >= 5).astype(int)
            
            # Cyclical encoding for month and day
            X[f'{col}_month_sin'] = np.sin(2 * np.pi * X[col].dt.month / 12)
            X[f'{col}_month_cos'] = np.cos(2 * np.pi * X[col].dt.month / 12)
            X[f'{col}_day_sin'] = np.sin(2 * np.pi * X[col].dt.day / 31)
            X[f'{col}_day_cos'] = np.cos(2 * np.pi * X[col].dt.day / 31)
            
            # Drop original datetime column
            X = X.drop(columns=[col])
        
        return X
    
    def get_feature_names_out(self) -> List[str]:
        """Get feature names after transformation.
        
        Returns the names of all features in the transformed output,
        including encoded categorical features and vectorized text features.
        
        Returns:
            List of feature names in the order they appear in transformed data
            
        Raises:
            ValueError: If DataProcessor hasn't been fitted yet
            
        Example:
            >>> processor.fit_transform(X_train)
            >>> feature_names = processor.get_feature_names_out()
            >>> print(f"Number of features: {len(feature_names)}")
            >>> print(f"First 5 features: {feature_names[:5]}")
        """
        if not self.is_fitted:
            raise ValueError("DataProcessor must be fitted first")
        return self.feature_names_out_
    
    def save_config(self, path: str):
        """Save configuration to file.
        
        Saves the current configuration to a JSON file for reproducibility.
        
        Args:
            path: File path to save configuration to
            
        Example:
            >>> processor.save_config('preprocessing_config.json')
            >>> # Later, load the same config:
            >>> new_processor = DataProcessor.load_config('preprocessing_config.json')
            
        Note:
            Only saves configuration, not fitted parameters. To save a fitted
            processor, use pickle or joblib.
        """
        import json
        
        # Convert tuples to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, tuple):
                return list(obj)
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        with open(path, 'w') as f:
            json.dump(convert_for_json(self.config), f, indent=2)
    
    @classmethod
    def load_config(cls, path: str) -> 'DataProcessor':
        """Load DataProcessor with configuration from file.
        
        Creates a new DataProcessor instance with configuration loaded from
        a JSON file.
        
        Args:
            path: File path to load configuration from
            
        Returns:
            New DataProcessor instance with loaded configuration
            
        Example:
            >>> # Load previously saved config
            >>> processor = DataProcessor.load_config('preprocessing_config.json')
            >>> processor.fit(X_train, y_train)
            
        Note:
            This only loads configuration, not fitted parameters. The returned
            processor still needs to be fitted before use.
        """
        import json
        
        # Convert lists back to tuples where appropriate
        def convert_from_json(obj):
            if isinstance(obj, dict):
                result = {}
                for k, v in obj.items():
                    # Convert ngram_range back to tuple
                    if k == 'ngram_range' and isinstance(v, list) and len(v) == 2:
                        result[k] = tuple(v)
                    # Convert feature_range back to tuple
                    elif k == 'feature_range' and isinstance(v, list) and len(v) == 2:
                        result[k] = tuple(v)
                    else:
                        result[k] = convert_from_json(v)
                return result
            elif isinstance(obj, list):
                return [convert_from_json(item) for item in obj]
            else:
                return obj
        
        with open(path, 'r') as f:
            config = json.load(f)
            config = convert_from_json(config)
        
        return cls(config=config)