"""Comprehensive tests for pipeline and evaluation modules."""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from pathlib import Path
import tempfile
import shutil

from tabml.pipeline import TabularPipeline
from tabml.evaluate import CrossValidator


@pytest.fixture
def classification_dataset():
    """Create classification dataset and save to temp files."""
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=7,
        n_redundant=2, n_classes=2, random_state=42
    )

    train_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    train_df['target'] = y
    train_df['id'] = range(len(train_df))

    # Create test set (without target)
    X_test, _ = make_classification(
        n_samples=100, n_features=10, n_informative=7,
        n_redundant=2, n_classes=2, random_state=123
    )
    test_df = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
    test_df['id'] = range(len(train_df), len(train_df) + len(test_df))

    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)

    train_df.to_csv(temp_path / 'train.csv', index=False)
    test_df.to_csv(temp_path / 'test.csv', index=False)

    yield temp_path, train_df, test_df

    shutil.rmtree(temp_dir)


@pytest.fixture
def regression_dataset():
    """Create regression dataset and save to temp files."""
    X, y = make_regression(
        n_samples=500, n_features=10, n_informative=8,
        noise=0.1, random_state=42
    )

    train_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    train_df['target'] = y
    train_df['id'] = range(len(train_df))

    X_test, _ = make_regression(
        n_samples=100, n_features=10, n_informative=8,
        noise=0.1, random_state=123
    )
    test_df = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
    test_df['id'] = range(len(train_df), len(train_df) + len(test_df))

    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)

    train_df.to_csv(temp_path / 'train.csv', index=False)
    test_df.to_csv(temp_path / 'test.csv', index=False)

    yield temp_path, train_df, test_df

    shutil.rmtree(temp_dir)


@pytest.fixture
def simple_classification_data():
    """Create simple in-memory classification data."""
    X, y = make_classification(
        n_samples=300, n_features=8, n_informative=5,
        n_redundant=2, n_classes=2, random_state=42
    )
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y_series = pd.Series(y, name='target')

    return X_df, y_series


@pytest.fixture
def simple_regression_data():
    """Create simple in-memory regression data."""
    X, y = make_regression(
        n_samples=300, n_features=8, n_informative=6,
        noise=0.1, random_state=42
    )
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y_series = pd.Series(y, name='target')

    return X_df, y_series


class TestTabularPipeline:
    """Test TabularPipeline class."""

    def test_initialization(self, classification_dataset):
        """Test pipeline initialization."""
        data_dir, _, _ = classification_dataset

        pipeline = TabularPipeline(data_dir=data_dir)

        assert pipeline.data_dir == data_dir
        assert pipeline.data_loader is not None  # Correct attribute name

    def test_load_data(self, classification_dataset):
        """Test data loading in pipeline."""
        data_dir, train_df, test_df = classification_dataset

        pipeline = TabularPipeline(data_dir=data_dir)
        pipeline.load_data()

        assert pipeline.train_data is not None
        assert pipeline.test_data is not None

    def test_engineer_features(self, classification_dataset):
        """Test feature engineering in pipeline."""
        data_dir, _, _ = classification_dataset

        pipeline = TabularPipeline(data_dir=data_dir)
        pipeline.load_data()

        pipeline.engineer_features(
            scaling='standard',  # Correct parameter name
            create_interactions=False
        )

        assert pipeline.feature_engineer is not None

    def test_train_models(self, classification_dataset):
        """Test model training in pipeline."""
        data_dir, _, _ = classification_dataset

        pipeline = TabularPipeline(data_dir=data_dir)
        pipeline.load_data()
        pipeline.engineer_features()

        # Method is train_models (plural)
        pipeline.train_models(
            model_types=['xgboost'],
            model_params={'xgboost': {'n_estimators': 10}},
            cv_folds=3
        )

        assert pipeline.best_model is not None

    def test_make_predictions(self, classification_dataset):
        """Test making predictions in pipeline."""
        data_dir, _, test_df = classification_dataset

        pipeline = TabularPipeline(data_dir=data_dir)
        pipeline.load_data()
        pipeline.engineer_features()
        pipeline.train_models(
            model_types=['xgboost'],
            model_params={'xgboost': {'n_estimators': 10}}
        )

        predictions = pipeline.predict()

        assert predictions is not None
        assert len(predictions) == len(test_df)

    def test_create_submission(self, classification_dataset):
        """Test submission creation in pipeline."""
        data_dir, _, _ = classification_dataset

        pipeline = TabularPipeline(data_dir=data_dir)
        pipeline.load_data()
        pipeline.engineer_features()
        pipeline.train_models(
            model_types=['xgboost'],
            model_params={'xgboost': {'n_estimators': 10}}
        )

        # create_submission doesn't take predictions as arg - it generates them
        submission = pipeline.create_submission()

        assert submission is not None
        assert 'id' in submission.columns
        assert 'target' in submission.columns

    def test_get_feature_importance(self, classification_dataset):
        """Test getting feature importance."""
        data_dir, _, _ = classification_dataset

        pipeline = TabularPipeline(data_dir=data_dir)
        pipeline.load_data()
        pipeline.engineer_features()
        pipeline.train_models(
            model_types=['xgboost'],
            model_params={'xgboost': {'n_estimators': 10}}
        )

        importance = pipeline.get_feature_importance()

        assert importance is not None

    def test_regression_pipeline(self, regression_dataset):
        """Test full pipeline on regression data."""
        data_dir, _, _ = regression_dataset

        pipeline = TabularPipeline(data_dir=data_dir, task_type='regression')
        pipeline.load_data()
        pipeline.engineer_features()

        pipeline.train_models(
            model_types=['xgboost'],
            model_params={'xgboost': {'n_estimators': 10}}
        )

        assert pipeline.best_model is not None


class TestCrossValidator:
    """Test CrossValidator class."""

    def test_initialization(self):
        """Test CrossValidator initialization."""
        validator = CrossValidator(random_state=42)  # Only random_state in __init__

        assert validator.random_state == 42

    def test_evaluate_model_classification(self, simple_classification_data):
        """Test model evaluation for classification."""
        from tabml.models import XGBoostModel

        X, y = simple_classification_data

        validator = CrossValidator(random_state=42)
        model = XGBoostModel(params={'n_estimators': 10})

        results = validator.evaluate_model(
            model, X, y,
            cv_folds=3,  # Pass cv_folds per call
            scoring='accuracy'  # 'scoring' not 'metric'
        )

        assert 'mean' in results
        assert 'std' in results
        assert 'scores' in results
        assert len(results['scores']) == 3

    def test_evaluate_model_regression(self, simple_regression_data):
        """Test model evaluation for regression."""
        from tabml.models import XGBoostModel

        X, y = simple_regression_data

        validator = CrossValidator(random_state=42)
        model = XGBoostModel(params={'n_estimators': 10})

        results = validator.evaluate_model(
            model, X, y,
            cv_folds=3,
            scoring='neg_mean_squared_error'
        )

        assert 'mean' in results
        assert 'std' in results

    def test_different_cv_folds(self, simple_classification_data):
        """Test different numbers of CV folds."""
        from tabml.models import XGBoostModel

        X, y = simple_classification_data

        validator = CrossValidator(random_state=42)
        model = XGBoostModel(params={'n_estimators': 10})

        for n_folds in [3, 5]:
            results = validator.evaluate_model(
                model, X, y,
                cv_folds=n_folds,
                scoring='accuracy'
            )

            assert len(results['scores']) == n_folds

    def test_walk_forward_validation(self, simple_regression_data):
        """Test walk-forward validation for time series."""
        from tabml.models import XGBoostModel

        X, y = simple_regression_data

        validator = CrossValidator(random_state=42)
        model = XGBoostModel(params={'n_estimators': 10})

        # Use walk_forward_validation method
        results = validator.walk_forward_validation(
            model, X, y,
            initial_train_size=0.6,
            step_size=0.1
        )

        assert results is not None
        assert 'mean' in results or 'scores' in results

    def test_compare_models(self, simple_classification_data):
        """Test comparing multiple models."""
        from tabml.models import XGBoostModel, LightGBMModel

        X, y = simple_classification_data

        validator = CrossValidator(random_state=42)

        models = {
            'xgboost': XGBoostModel(params={'n_estimators': 10}),
            'lightgbm': LightGBMModel(params={'n_estimators': 10, 'verbose': -1})
        }

        comparison = validator.compare_models(models, X, y, cv_folds=3)

        assert comparison is not None
        assert 'xgboost' in comparison
        assert 'lightgbm' in comparison
