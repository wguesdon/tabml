"""Feature engineering and selection utilities.

This module provides comprehensive feature engineering and selection tools
for tabular data, including imputation, scaling, encoding, and feature creation.

Classes:
    FeatureEngineer: Complete feature engineering pipeline
    FeatureSelector: Feature selection using various methods
    
Example:
    Basic feature engineering::
    
        from tabml.features import FeatureEngineer, FeatureSelector
        
        # Engineer features
        engineer = FeatureEngineer(
            scaling_method='robust',
            categorical_encoding='target',
            create_interactions=True
        )
        X_engineered = engineer.fit_transform(X_train, y_train)
        
        # Select best features
        selector = FeatureSelector(method='mutual_info', n_features=20)
        X_selected = selector.fit_transform(X_engineered, y_train)
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import (
    SelectKBest, RFE, mutual_info_classif, mutual_info_regression,
    f_classif, f_regression
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from category_encoders import TargetEncoder as CatTargetEncoder
from loguru import logger


class FeatureEngineer:
    """A feature engineering pipeline for tabular data.

    Provides a complete pipeline for feature preprocessing, including imputation,
    scaling, encoding, and feature creation. Handles both numeric and
    categorical features with various strategies.

    Attributes:
        numeric_features (List[str]): A list of numeric column names
            (auto-detected).
        categorical_features (List[str]): A list of categorical column names
            (auto-detected).
        feature_names_out_ (List[str]): The feature names after transformation.
        is_fitted (bool): Whether the engineer has been fitted.
        numeric_imputer: The fitted imputer for numeric features.
        categorical_imputer: The fitted imputer for categorical features.
        scaler: The fitted scaler for numeric features.
        categorical_encoders (Dict): A dictionary of fitted encoders.
        target_encoder: The fitted target encoder (if applicable).
    """
    
    def __init__(self, 
                 numeric_impute_strategy: str = 'median',
                 categorical_impute_strategy: str = 'constant',
                 scaling_method: Optional[str] = 'standard',
                 categorical_encoding: str = 'onehot',
                 create_interactions: bool = False,
                 create_polynomial: bool = False,
                 target_encoding: bool = False,
                 time_series_impute: bool = False,
                 max_cardinality: int = 50,
                 min_frequency: float = 0.01,
                 rare_label: str = 'Other'):
        """Initializes the feature engineer.

        Args:
            numeric_impute_strategy (str): The strategy for numeric imputation.
                Options are 'mean', 'median', 'most_frequent', 'constant',
                'ffill', and 'bfill'.
            categorical_impute_strategy (str): The strategy for categorical
                imputation. Options are 'constant', 'most_frequent', 'ffill',
                and 'bfill'.
            scaling_method (Optional[str]): The method for scaling numeric
                features. Options are 'standard', 'minmax', 'robust', or None.
            categorical_encoding (str): The method for encoding categorical
                features. Options are 'onehot', 'label', 'ordinal', 'target',
                or None.
            create_interactions (bool): Whether to create interaction features
                between numeric columns.
            create_polynomial (bool): Whether to create polynomial features for
                numeric columns.
            target_encoding (bool): Deprecated. Use
                categorical_encoding='target' instead.
            time_series_impute (bool): Whether to use time series-specific
                imputation.
            max_cardinality (int): The maximum number of unique values for
                categorical features before grouping rare categories.
            min_frequency (float): The minimum frequency for a category to avoid
                being grouped as 'Other'.
            rare_label (str): The label to use for grouped rare categories.
        """
        self.numeric_impute_strategy = numeric_impute_strategy
        self.categorical_impute_strategy = categorical_impute_strategy
        self.scaling_method = scaling_method
        self.categorical_encoding = categorical_encoding
        self.create_interactions = create_interactions
        self.create_polynomial = create_polynomial
        # Handle deprecated target_encoding parameter
        if target_encoding and categorical_encoding != 'target':
            logger.warning("target_encoding parameter is deprecated. Use categorical_encoding='target' instead.")
            self.categorical_encoding = 'target' if target_encoding else categorical_encoding
        self.time_series_impute = time_series_impute
        self.max_cardinality = max_cardinality
        self.min_frequency = min_frequency
        self.rare_label = rare_label
        
        self.numeric_imputer = None
        self.categorical_imputer = None
        self.scaler = None
        self.categorical_encoders = {}
        self.target_encoder = None
        self.feature_names_out_ = None
        self.is_fitted = False
        self.cardinality_map_ = {}  # Store mapping for each column
        
    def _reduce_cardinality(self, series: pd.Series, col_name: str) -> pd.Series:
        """Reduces cardinality by grouping rare categories.

        Args:
            series (pd.Series): The categorical column to process.
            col_name (str): The name of the column for tracking mappings.

        Returns:
            A Series with rare categories grouped as `self.rare_label`.
        """
        # Calculate value counts and frequencies
        value_counts = series.value_counts()
        n_unique = len(value_counts)
        
        # Check if reduction is needed
        if n_unique <= self.max_cardinality:
            return series
            
        # Calculate frequencies
        frequencies = value_counts / len(series)
        
        # Two strategies for grouping:
        # 1. If too many categories, keep top max_cardinality-1 and group rest
        # 2. Also group any category below min_frequency threshold
        
        # Strategy 1: Keep top categories
        top_categories = value_counts.head(self.max_cardinality - 1).index.tolist()
        
        # Strategy 2: Filter by frequency
        frequent_categories = frequencies[frequencies >= self.min_frequency].index.tolist()
        
        # Combine both strategies - keep categories that meet either criteria
        categories_to_keep = list(set(top_categories) | set(frequent_categories))
        
        # Ensure we don't exceed max_cardinality
        if len(categories_to_keep) >= self.max_cardinality:
            categories_to_keep = value_counts.head(self.max_cardinality - 1).index.tolist()
        
        # Store mapping for transform
        self.cardinality_map_[col_name] = categories_to_keep
        
        # Log the reduction
        n_grouped = n_unique - len(categories_to_keep)
        logger.info(f"Column '{col_name}': Grouping {n_grouped} rare categories "
                   f"(out of {n_unique} total) into '{self.rare_label}'")
        
        # Apply grouping
        return series.where(series.isin(categories_to_keep), self.rare_label)
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureEngineer':
        """Fits the feature engineering pipeline.

        Learns imputation values, scaling parameters, and encoding mappings from
        the training data.

        Args:
            X (pd.DataFrame): The training features.
            y (Optional[pd.Series]): The target variable (optional, required for
                target encoding).

        Returns:
            The self instance for method chaining.

        Raises:
            ValueError: If target encoding is requested but y is not provided.
        """
        # Identify numeric and categorical columns
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Setup imputers
        if self.numeric_features:
            # Handle time series specific imputation strategies
            if self.numeric_impute_strategy in ['ffill', 'bfill'] or self.time_series_impute:
                # For forward/backward fill, we'll handle in transform
                self.numeric_imputer = self.numeric_impute_strategy
            else:
                self.numeric_imputer = SimpleImputer(strategy=self.numeric_impute_strategy)
                self.numeric_imputer.fit(X[self.numeric_features])
            
        if self.categorical_features:
            if self.categorical_impute_strategy in ['ffill', 'bfill'] or self.time_series_impute:
                self.categorical_imputer = self.categorical_impute_strategy
            else:
                self.categorical_imputer = SimpleImputer(
                    strategy=self.categorical_impute_strategy,
                    fill_value='missing'
                )
                self.categorical_imputer.fit(X[self.categorical_features])
        
        # Setup scaler
        if self.scaling_method and self.numeric_features:
            if self.scaling_method == 'standard':
                self.scaler = StandardScaler()
            elif self.scaling_method == 'minmax':
                self.scaler = MinMaxScaler()
            elif self.scaling_method == 'robust':
                self.scaler = RobustScaler()
                
            # Fit on imputed data
            if isinstance(self.numeric_imputer, str):  # ffill or bfill
                if self.numeric_imputer == 'ffill':
                    X_numeric_imputed = X[self.numeric_features].ffill()
                elif self.numeric_imputer == 'bfill':
                    X_numeric_imputed = X[self.numeric_features].bfill()
                else:
                    X_numeric_imputed = X[self.numeric_features]
            else:
                X_numeric_imputed = self.numeric_imputer.transform(X[self.numeric_features])
            self.scaler.fit(X_numeric_imputed)
        
        # Apply cardinality reduction BEFORE setting up encoders
        if self.categorical_features:
            # First get imputed data
            if isinstance(self.categorical_imputer, str):  # ffill or bfill
                if self.categorical_imputer == 'ffill':
                    X_cat_for_cardinality = X[self.categorical_features].ffill()
                elif self.categorical_imputer == 'bfill':
                    X_cat_for_cardinality = X[self.categorical_features].bfill()
                else:
                    X_cat_for_cardinality = X[self.categorical_features]
            else:
                X_cat_for_cardinality = pd.DataFrame(
                    self.categorical_imputer.transform(X[self.categorical_features]),
                    columns=self.categorical_features,
                    index=X.index
                )
            
            # Check cardinality and warn/reduce if needed
            X_reduced = X.copy()
            for col in self.categorical_features:
                n_unique = X_cat_for_cardinality[col].nunique()
                if n_unique > self.max_cardinality:
                    logger.warning(f"Column '{col}' has {n_unique} unique values, "
                                 f"exceeding max_cardinality={self.max_cardinality}")
                    # Apply cardinality reduction
                    X_reduced[col] = self._reduce_cardinality(X[col], col)
                elif n_unique > 20:
                    # Just warn for moderately high cardinality
                    logger.info(f"Column '{col}' has {n_unique} unique values")
            
            # Update X with reduced cardinality data
            X = X_reduced
        
        # Setup categorical encoders
        if self.categorical_encoding and self.categorical_features:
            # Get imputed categorical data for fitting encoders
            if isinstance(self.categorical_imputer, str):  # ffill or bfill
                if self.categorical_imputer == 'ffill':
                    X_cat_imputed = X[self.categorical_features].ffill()
                elif self.categorical_imputer == 'bfill':
                    X_cat_imputed = X[self.categorical_features].bfill()
                else:
                    X_cat_imputed = X[self.categorical_features]
            else:
                X_cat_imputed = pd.DataFrame(
                    self.categorical_imputer.transform(X[self.categorical_features]),
                    columns=self.categorical_features,
                    index=X.index
                )
            
            if self.categorical_encoding == 'onehot':
                self.categorical_encoders['onehot'] = OneHotEncoder(
                    sparse_output=False, 
                    handle_unknown='ignore',
                    drop='first'  # Drop first to avoid multicollinearity
                )
                self.categorical_encoders['onehot'].fit(X_cat_imputed)
                
            elif self.categorical_encoding == 'label':
                # Create a label encoder for each categorical column
                for col in self.categorical_features:
                    le = LabelEncoder()
                    # Handle unknown categories
                    unique_vals = X_cat_imputed[col].fillna('missing').unique()
                    le.fit(unique_vals)
                    self.categorical_encoders[col] = le
                    
            elif self.categorical_encoding == 'ordinal':
                self.categorical_encoders['ordinal'] = OrdinalEncoder(
                    handle_unknown='use_encoded_value',
                    unknown_value=-1
                )
                self.categorical_encoders['ordinal'].fit(X_cat_imputed)
                
            elif self.categorical_encoding == 'target' and y is not None:
                self.target_encoder = CatTargetEncoder()
                self.target_encoder.fit(X_cat_imputed, y)
            
        self.is_fitted = True
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms the features.

        Applies the fitted preprocessing pipeline to transform the features.

        Args:
            X (pd.DataFrame): The features to transform.

        Returns:
            A transformed DataFrame with engineered features.

        Raises:
            ValueError: If the FeatureEngineer has not been fitted yet.
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform.")
            
        X_transformed = X.copy()
        
        # Impute numeric features
        if self.numeric_features and self.numeric_imputer:
            if isinstance(self.numeric_imputer, str):  # ffill or bfill
                if self.numeric_imputer == 'ffill':
                    X_transformed[self.numeric_features] = X[self.numeric_features].ffill()
                elif self.numeric_imputer == 'bfill':
                    X_transformed[self.numeric_features] = X[self.numeric_features].bfill()
            else:
                X_transformed[self.numeric_features] = self.numeric_imputer.transform(
                    X[self.numeric_features]
                )
        
        # Impute categorical features
        if self.categorical_features and self.categorical_imputer:
            if isinstance(self.categorical_imputer, str):  # ffill or bfill
                if self.categorical_imputer == 'ffill':
                    X_transformed[self.categorical_features] = X[self.categorical_features].ffill()
                elif self.categorical_imputer == 'bfill':
                    X_transformed[self.categorical_features] = X[self.categorical_features].bfill()
            else:
                X_transformed[self.categorical_features] = self.categorical_imputer.transform(
                    X[self.categorical_features]
                )
        
        # Apply cardinality reduction based on fitted mappings
        for col, categories_to_keep in self.cardinality_map_.items():
            if col in X_transformed.columns:
                # Replace categories not in the keep list with rare_label
                X_transformed[col] = X_transformed[col].where(
                    X_transformed[col].isin(categories_to_keep), 
                    self.rare_label
                )
        
        # Scale numeric features
        if self.scaler and self.numeric_features:
            X_transformed[self.numeric_features] = self.scaler.transform(
                X_transformed[self.numeric_features]
            )
        
        # Encode categorical features
        if self.categorical_encoding and self.categorical_features:
            if self.categorical_encoding == 'onehot':
                # One-hot encode
                encoded = self.categorical_encoders['onehot'].transform(X_transformed[self.categorical_features])
                # Create column names
                feature_names = self.categorical_encoders['onehot'].get_feature_names_out(self.categorical_features)
                # Create DataFrame with encoded features
                encoded_df = pd.DataFrame(encoded, columns=feature_names, index=X_transformed.index)
                # Drop original categorical columns and add encoded ones
                X_transformed = X_transformed.drop(columns=self.categorical_features)
                X_transformed = pd.concat([X_transformed, encoded_df], axis=1)
                
            elif self.categorical_encoding == 'label':
                # Label encode each column
                for col in self.categorical_features:
                    le = self.categorical_encoders[col]
                    # Handle unknown categories
                    col_data = X_transformed[col].fillna('missing')
                    # Map unknown values to -1
                    col_data = col_data.apply(lambda x: x if x in le.classes_ else 'missing')
                    X_transformed[col] = le.transform(col_data)
                    
            elif self.categorical_encoding == 'ordinal':
                # Ordinal encode
                X_transformed[self.categorical_features] = self.categorical_encoders['ordinal'].transform(
                    X_transformed[self.categorical_features]
                )
                
            elif self.categorical_encoding == 'target' and self.target_encoder:
                # Target encode
                encoded = self.target_encoder.transform(X_transformed[self.categorical_features])
                X_transformed[self.categorical_features] = encoded
            
        # Create interaction features
        if self.create_interactions and len(self.numeric_features) >= 2:
            for i, col1 in enumerate(self.numeric_features[:-1]):
                for col2 in self.numeric_features[i+1:]:
                    X_transformed[f'{col1}_x_{col2}'] = X_transformed[col1] * X_transformed[col2]
                    
        # Create polynomial features
        if self.create_polynomial and self.numeric_features:
            for col in self.numeric_features[:5]:  # Limit to first 5 to avoid explosion
                X_transformed[f'{col}_squared'] = X_transformed[col] ** 2
                
        self.feature_names_out_ = X_transformed.columns.tolist()
        return X_transformed
        
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fits and transforms in one step.

        A convenience method that combines the fit and transform operations.

        Args:
            X (pd.DataFrame): The training features to fit and transform.
            y (Optional[pd.Series]): The target variable (optional, required for
                target encoding).

        Returns:
            The transformed training features.
        """
        return self.fit(X, y).transform(X)


class FeatureSelector:
    """Selects features for tabular data.

    Provides multiple methods for selecting the most informative features from a
    dataset, helping to reduce dimensionality and improve model performance.

    Attributes:
        method (str): The feature selection method to use.
        n_features (int): The number of features to select.
        task_type (str): The type of machine learning task, either
            'classification' or 'regression'.
        selected_features_ (List[str]): A list of selected feature names after
            fitting.
        feature_scores_ (Dict[str, float]): A dictionary of feature scores
            (method-dependent).
        selector_: The underlying selector object.
        is_fitted (bool): Whether the selector has been fitted.
    """
    
    def __init__(self,
                 method: str = 'mutual_info',
                 n_features: Optional[Union[int, float]] = None,
                 task_type: Optional[str] = None):
        """Initializes the feature selector.

        Args:
            method (str): The feature selection method. Options are
                'mutual_info', 'univariate', 'tree_based', and 'rfe'.
            n_features (Optional[Union[int, float]]): The number of features to
                select. Can be an int, a float (percentage), or None to
                auto-infer.
            task_type (Optional[str]): The type of ML task. Options are
                'classification', 'regression', or None to auto-detect.
        """
        self.method = method
        self.n_features = n_features
        self.task_type = task_type
        self.selected_features_ = None
        self.feature_scores_ = None
        self.selector_ = None
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureSelector':
        """Fits the feature selector.

        Learns which features to select based on the specified method.

        Args:
            X (pd.DataFrame): The feature matrix.
            y (pd.Series): The target variable.

        Returns:
            The self instance for method chaining.

        Raises:
            ValueError: If the method is unknown.
        """
        # Auto-detect task type
        if self.task_type is None:
            self.task_type = 'classification' if y.nunique() < 100 else 'regression'
            
        # Determine number of features to select
        n_features_total = X.shape[1]
        if self.n_features is None:
            k = max(10, n_features_total // 2)
        elif isinstance(self.n_features, float) and self.n_features <= 1.0:
            k = int(n_features_total * self.n_features)
        else:
            k = min(int(self.n_features), n_features_total)
            
        # Select scoring function
        if self.method == 'mutual_info':
            score_func = mutual_info_classif if self.task_type == 'classification' else mutual_info_regression
        elif self.method == 'univariate':
            score_func = f_classif if self.task_type == 'classification' else f_regression
        elif self.method == 'tree_based':
            # Use tree-based feature importance
            if self.task_type == 'classification':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Get feature importances
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:k]
            self.selected_features_ = X.columns[indices].tolist()
            self.feature_scores_ = dict(zip(X.columns, importances))
            self.is_fitted = True
            return self
        elif self.method == 'rfe':
            # Use RFE with a simple model
            if self.task_type == 'classification':
                estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            self.selector_ = RFE(estimator, n_features_to_select=k)
            self.selector_.fit(X, y)
            self.selected_features_ = X.columns[self.selector_.support_].tolist()
            self.is_fitted = True
            return self
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
        # For mutual_info and univariate methods
        if self.method in ['mutual_info', 'univariate']:
            self.selector_ = SelectKBest(score_func=score_func, k=k)
            self.selector_.fit(X, y)
            self.selected_features_ = X.columns[self.selector_.get_support()].tolist()
            self.feature_scores_ = dict(zip(X.columns, self.selector_.scores_))
            
        self.is_fitted = True
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms by selecting features.

        Returns only the selected features from the input DataFrame.

        Args:
            X (pd.DataFrame): The features to transform.

        Returns:
            A DataFrame containing only the selected features.

        Raises:
            ValueError: If the selector has not been fitted yet.
        """
        if not self.is_fitted:
            raise ValueError("FeatureSelector must be fitted before transform.")
            
        return X[self.selected_features_]
        
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fits and transforms in one step.

        A convenience method to fit the selector and transform the features.

        Args:
            X (pd.DataFrame): The feature matrix to fit and transform.
            y (pd.Series): The target variable.

        Returns:
            A DataFrame with selected features only.
        """
        return self.fit(X, y).transform(X)
        
    def get_feature_importance(self) -> pd.DataFrame:
        """Gets feature importance scores.

        Returns a DataFrame with features and their importance scores, sorted by
        importance.

        Returns:
            A DataFrame with columns ['feature', 'score'] sorted by score.

        Raises:
            ValueError: If no scores are available (e.g., the method does not
                provide scores or the selector has not been fitted).
        """
        if not self.is_fitted or self.feature_scores_ is None:
            raise ValueError("No feature scores available.")
            
        importance_df = pd.DataFrame({
            'feature': list(self.feature_scores_.keys()),
            'score': list(self.feature_scores_.values())
        })
        return importance_df.sort_values('score', ascending=False)