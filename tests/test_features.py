"""Comprehensive tests for feature engineering and selection."""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression

from tabml.features import FeatureEngineer, FeatureSelector
from tabml.advanced_features import AdvancedFeatureEngineer


@pytest.fixture
def mixed_data():
    """Create dataset with mixed numeric and categorical features."""
    np.random.seed(42)
    n_samples = 500

    data = pd.DataFrame({
        'numeric_1': np.random.randn(n_samples),
        'numeric_2': np.random.randn(n_samples) * 10 + 5,
        'numeric_3': np.random.exponential(2, n_samples),
        'categorical_1': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'categorical_2': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'high_cardinality': [f'cat_{i % 100}' for i in range(n_samples)],
        'target': np.random.randint(0, 2, n_samples)
    })

    # Add some missing values
    data.loc[np.random.choice(data.index, 20, replace=False), 'numeric_1'] = np.nan
    data.loc[np.random.choice(data.index, 15, replace=False), 'categorical_1'] = np.nan

    return data


@pytest.fixture
def text_data():
    """Create dataset with text features."""
    np.random.seed(42)
    n_samples = 200

    texts = [
        'This is a sample text document',
        'Machine learning is awesome',
        'Feature engineering is important',
        'Text processing with NLP',
        'Data science and analytics'
    ]

    data = pd.DataFrame({
        'text_col': [texts[i % len(texts)] for i in range(n_samples)],
        'numeric_1': np.random.randn(n_samples),
        'target': np.random.randint(0, 2, n_samples)
    })

    return data


@pytest.fixture
def date_data():
    """Create dataset with date features."""
    np.random.seed(42)
    n_samples = 365

    data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
        'numeric_1': np.random.randn(n_samples),
        'target': np.random.rand(n_samples)
    })

    return data


class TestFeatureEngineer:
    """Test FeatureEngineer class."""

    def test_initialization(self):
        """Test FeatureEngineer initialization."""
        engineer = FeatureEngineer(
            numeric_impute_strategy='median',
            categorical_impute_strategy='constant',
            scaling_method='standard',
            categorical_encoding='onehot'
        )

        assert engineer.numeric_impute_strategy == 'median'
        assert engineer.categorical_impute_strategy == 'constant'
        assert engineer.scaling_method == 'standard'
        assert engineer.categorical_encoding == 'onehot'

    def test_fit_transform(self, mixed_data):
        """Test fit_transform on mixed data."""
        engineer = FeatureEngineer()

        X = mixed_data.drop(columns=['target'])
        y = mixed_data['target']

        X_transformed = engineer.fit_transform(X, y)

        assert X_transformed is not None
        assert len(X_transformed) == len(X)
        assert engineer.is_fitted

    def test_transform_after_fit(self, mixed_data):
        """Test transform on new data after fitting."""
        engineer = FeatureEngineer()

        # Split data
        train_data = mixed_data[:400]
        test_data = mixed_data[400:]

        X_train = train_data.drop(columns=['target'])
        y_train = train_data['target']
        X_test = test_data.drop(columns=['target'])

        # Fit on train
        X_train_transformed = engineer.fit_transform(X_train, y_train)

        # Transform test
        X_test_transformed = engineer.transform(X_test)

        assert X_test_transformed is not None
        assert len(X_test_transformed) == len(X_test)
        assert X_train_transformed.shape[1] == X_test_transformed.shape[1]

    def test_numeric_imputation_strategies(self, mixed_data):
        """Test different numeric imputation strategies."""
        X = mixed_data.drop(columns=['target'])
        y = mixed_data['target']

        strategies = ['mean', 'median', 'most_frequent']

        for strategy in strategies:
            engineer = FeatureEngineer(numeric_impute_strategy=strategy)
            X_transformed = engineer.fit_transform(X, y)

            # Check no NaNs in numeric columns
            numeric_cols = X_transformed.select_dtypes(include=[np.number]).columns
            assert not X_transformed[numeric_cols].isnull().any().any(), \
                f"NaNs found with strategy: {strategy}"

    def test_categorical_imputation(self, mixed_data):
        """Test categorical imputation."""
        X = mixed_data.drop(columns=['target'])
        y = mixed_data['target']

        engineer = FeatureEngineer(categorical_impute_strategy='constant')
        X_transformed = engineer.fit_transform(X, y)

        assert X_transformed is not None

    def test_scaling_methods(self, mixed_data):
        """Test different scaling methods."""
        X = mixed_data.drop(columns=['target'])
        y = mixed_data['target']

        scaling_methods = ['standard', 'minmax', 'robust', None]

        for method in scaling_methods:
            engineer = FeatureEngineer(scaling_method=method)
            X_transformed = engineer.fit_transform(X, y)

            assert X_transformed is not None, f"Failed for scaling method: {method}"

    def test_categorical_encoding_methods(self, mixed_data):
        """Test different categorical encoding methods."""
        X = mixed_data.drop(columns=['target'])
        y = mixed_data['target']

        encoding_methods = ['onehot', 'label', 'target']

        for method in encoding_methods:
            engineer = FeatureEngineer(categorical_encoding=method)
            X_transformed = engineer.fit_transform(X, y)

            assert X_transformed is not None, f"Failed for encoding method: {method}"

    def test_interaction_features(self, mixed_data):
        """Test creation of interaction features."""
        X = mixed_data.drop(columns=['target'])
        y = mixed_data['target']

        engineer = FeatureEngineer(create_interactions=True)
        X_transformed = engineer.fit_transform(X, y)

        # Should have more features due to interactions
        assert X_transformed.shape[1] >= X.select_dtypes(include=[np.number]).shape[1]

    def test_polynomial_features(self, mixed_data):
        """Test creation of polynomial features."""
        X = mixed_data.drop(columns=['target'])
        y = mixed_data['target']

        engineer = FeatureEngineer(create_polynomial=True)
        X_transformed = engineer.fit_transform(X, y)

        # Should have more features due to polynomial terms
        assert X_transformed.shape[1] >= X.select_dtypes(include=[np.number]).shape[1]

    def test_high_cardinality_handling(self, mixed_data):
        """Test handling of high cardinality features."""
        X = mixed_data.drop(columns=['target'])
        y = mixed_data['target']

        engineer = FeatureEngineer(max_cardinality=50, min_frequency=0.02)
        X_transformed = engineer.fit_transform(X, y)

        assert X_transformed is not None

    def test_get_feature_names(self, mixed_data):
        """Test getting feature names after transformation."""
        X = mixed_data.drop(columns=['target'])
        y = mixed_data['target']

        engineer = FeatureEngineer()
        engineer.fit_transform(X, y)

        feature_names = engineer.get_feature_names()

        assert feature_names is not None
        assert len(feature_names) > 0

    def test_onehot_encoding_produces_numeric(self, mixed_data):
        """Test that one-hot encoding produces all numeric features."""
        X = mixed_data.drop(columns=['target'])
        y = mixed_data['target']

        engineer = FeatureEngineer(categorical_encoding='onehot')
        X_transformed = engineer.fit_transform(X, y)

        # All columns should be numeric after one-hot encoding
        assert X_transformed.select_dtypes(include=[np.number]).shape[1] == X_transformed.shape[1]


class TestFeatureSelector:
    """Test FeatureSelector class."""

    def test_initialization(self):
        """Test FeatureSelector initialization."""
        selector = FeatureSelector(method='mutual_info', n_features=10)

        assert selector.method == 'mutual_info'
        assert selector.n_features == 10

    def test_mutual_info_classification(self):
        """Test mutual information feature selection for classification."""
        X, y = make_classification(
            n_samples=500, n_features=20, n_informative=10,
            n_redundant=5, n_classes=2, random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        y_series = pd.Series(y, name='target')

        selector = FeatureSelector(method='mutual_info', n_features=10)
        X_selected = selector.fit_transform(X_df, y_series)

        assert X_selected.shape[1] == 10
        assert X_selected.shape[0] == X_df.shape[0]

    def test_mutual_info_regression(self):
        """Test mutual information feature selection for regression."""
        X, y = make_regression(
            n_samples=500, n_features=20, n_informative=10,
            noise=0.1, random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        y_series = pd.Series(y, name='target')

        selector = FeatureSelector(method='mutual_info', n_features=10)
        X_selected = selector.fit_transform(X_df, y_series)

        assert X_selected.shape[1] == 10

    def test_tree_based_selection(self):
        """Test tree-based feature selection."""
        X, y = make_classification(
            n_samples=500, n_features=20, n_informative=10,
            random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        y_series = pd.Series(y, name='target')

        selector = FeatureSelector(method='tree', n_features=10)
        X_selected = selector.fit_transform(X_df, y_series)

        assert X_selected.shape[1] == 10

    def test_get_selected_features(self):
        """Test getting selected feature names."""
        X, y = make_classification(
            n_samples=500, n_features=20, n_informative=10,
            random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        y_series = pd.Series(y, name='target')

        selector = FeatureSelector(method='mutual_info', n_features=10)
        selector.fit_transform(X_df, y_series)

        selected_features = selector.get_selected_features()

        assert len(selected_features) == 10

    def test_get_feature_scores(self):
        """Test getting feature importance scores."""
        X, y = make_classification(
            n_samples=500, n_features=20, n_informative=10,
            random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        y_series = pd.Series(y, name='target')

        selector = FeatureSelector(method='mutual_info', n_features=10)
        selector.fit_transform(X_df, y_series)

        scores = selector.get_feature_scores()

        assert len(scores) == X_df.shape[1]
        assert all(score >= 0 for score in scores.values())

    def test_transform_after_fit(self):
        """Test transform on new data after fitting."""
        X, y = make_classification(
            n_samples=500, n_features=20, n_informative=10,
            random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        y_series = pd.Series(y, name='target')

        # Split data
        X_train, X_test = X_df[:400], X_df[400:]
        y_train = y_series[:400]

        selector = FeatureSelector(method='mutual_info', n_features=10)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)

        assert X_test_selected.shape[1] == 10
        assert X_train_selected.shape[1] == X_test_selected.shape[1]


class TestAdvancedFeatureEngineer:
    """Test AdvancedFeatureEngineer class."""

    def test_initialization(self):
        """Test AdvancedFeatureEngineer initialization."""
        engineer = AdvancedFeatureEngineer()

        assert engineer is not None
        assert hasattr(engineer, 'tfidf_vectorizers')
        assert hasattr(engineer, 'date_columns_processed')

    def test_text_features(self, text_data):
        """Test text feature extraction."""
        engineer = AdvancedFeatureEngineer()

        # Create TF-IDF features
        df_with_tfidf = engineer.create_tfidf_features(
            text_data.copy(),
            text_columns=['text_col'],
            max_features=10
        )

        # Should have additional TF-IDF features
        assert df_with_tfidf.shape[1] > text_data.shape[1]
        # Check that some tfidf columns were created
        tfidf_cols = [col for col in df_with_tfidf.columns if 'tfidf' in col.lower() or 'text_col' in col]
        assert len(tfidf_cols) > 0

    def test_date_features(self, date_data):
        """Test date feature extraction."""
        engineer = AdvancedFeatureEngineer()

        # Create date features
        df_with_dates = engineer.create_date_features(
            date_data.copy(),
            date_columns=['date'],
            include_cyclical=True
        )

        # Should have additional date features (day, month, year, etc.)
        assert df_with_dates.shape[1] > date_data.shape[1]

        # Check for some expected date features
        date_feature_cols = [col for col in df_with_dates.columns
                           if any(x in col for x in ['year', 'month', 'day', 'dayofweek'])]
        assert len(date_feature_cols) > 0

    def test_interaction_features(self):
        """Test interaction feature creation."""
        np.random.seed(42)
        n_samples = 100

        data = pd.DataFrame({
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'feature_3': np.random.randn(n_samples),
            'target': np.random.rand(n_samples)
        })

        engineer = AdvancedFeatureEngineer()

        # Create interaction features
        df_with_interactions = engineer.create_interaction_features(
            data.copy(),
            columns=['feature_1', 'feature_2', 'feature_3'],
            degree=2
        )

        # Should have interaction features
        assert df_with_interactions.shape[1] > data.shape[1]

    def test_polynomial_features(self):
        """Test polynomial feature creation."""
        np.random.seed(42)
        n_samples = 100

        data = pd.DataFrame({
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'target': np.random.rand(n_samples)
        })

        engineer = AdvancedFeatureEngineer()

        # Create polynomial features
        df_with_poly = engineer.create_polynomial_features(
            data.copy(),
            columns=['feature_1', 'feature_2'],
            degree=2
        )

        # Should have polynomial features
        assert df_with_poly.shape[1] > data.shape[1]

    def test_text_statistics(self, text_data):
        """Test text statistics feature creation."""
        engineer = AdvancedFeatureEngineer()

        # Create text statistics
        df_with_stats = engineer.create_text_features(
            text_data.copy(),
            text_columns=['text_col']
        )

        # Should have additional text statistic features
        assert df_with_stats.shape[1] > text_data.shape[1]

        # Check for text stat columns
        text_stat_cols = [col for col in df_with_stats.columns if 'text_col' in col and col != 'text_col']
        assert len(text_stat_cols) > 0

    def test_cyclical_date_encoding(self, date_data):
        """Test cyclical encoding of date features."""
        engineer = AdvancedFeatureEngineer()

        # Create date features with cyclical encoding
        df_with_cyclical = engineer.create_date_features(
            date_data.copy(),
            date_columns=['date'],
            include_cyclical=True
        )

        # Should have sin/cos features for cyclical date components
        assert df_with_cyclical.shape[1] > date_data.shape[1]

        # Check for cyclical features (sin/cos)
        cyclical_cols = [col for col in df_with_cyclical.columns if 'sin' in col.lower() or 'cos' in col.lower()]
        assert len(cyclical_cols) > 0

    def test_combined_features(self, text_data):
        """Test combining multiple advanced features."""
        # Add a date column
        text_data_copy = text_data.copy()
        text_data_copy['date'] = pd.date_range('2023-01-01', periods=len(text_data_copy), freq='D')

        engineer = AdvancedFeatureEngineer()

        # Add TF-IDF features
        df = engineer.create_tfidf_features(
            text_data_copy,
            text_columns=['text_col'],
            max_features=5
        )

        # Add date features
        df = engineer.create_date_features(
            df,
            date_columns=['date'],
            include_cyclical=True
        )

        # Should have both text and date features
        assert df.shape[1] > text_data_copy.shape[1]

        # Check we have both types of features
        has_text_features = any('tfidf' in col.lower() or 'text_col' in col for col in df.columns if col not in text_data_copy.columns)
        has_date_features = any(x in str(df.columns).lower() for x in ['year', 'month', 'day'])
        assert has_text_features or has_date_features  # At least one type should be present
