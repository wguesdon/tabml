"""Tests for benchmark system."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from tabml.benchmarks import (
    BenchmarkRunner,
    SklearnLoader,
    BenchmarkMetrics,
    RegressionTest
)


class TestSklearnLoader:
    """Test sklearn dataset loader."""

    def test_load_breast_cancer(self):
        """Test loading breast cancer dataset."""
        loader = SklearnLoader()
        dataset = loader.load_dataset('breast_cancer')

        assert dataset['name'] == 'breast_cancer'
        assert dataset['task_type'] == 'classification'
        assert isinstance(dataset['X'], pd.DataFrame)
        assert isinstance(dataset['y'], pd.Series)
        assert len(dataset['X']) == len(dataset['y'])
        assert dataset['X'].shape[0] > 0

    def test_load_diabetes(self):
        """Test loading diabetes dataset."""
        loader = SklearnLoader()
        dataset = loader.load_dataset('diabetes')

        assert dataset['name'] == 'diabetes'
        assert dataset['task_type'] == 'regression'
        assert isinstance(dataset['X'], pd.DataFrame)
        assert isinstance(dataset['y'], pd.Series)

    def test_load_suite(self):
        """Test loading sklearn suite."""
        loader = SklearnLoader()
        datasets = loader.load_suite(max_datasets=2)

        assert len(datasets) == 2
        assert all('X' in d for d in datasets)
        assert all('y' in d for d in datasets)
        assert all('task_type' in d for d in datasets)


class TestBenchmarkMetrics:
    """Test metrics computation."""

    def test_classification_metrics(self):
        """Test classification metrics."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 0])
        y_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2],
                           [0.3, 0.7], [0.7, 0.3], [0.6, 0.4]])

        metrics = BenchmarkMetrics.compute_classification_metrics(
            y_true, y_pred, y_proba
        )

        assert 'accuracy' in metrics
        assert 'roc_auc' in metrics
        assert 'f1_macro' in metrics
        assert metrics['accuracy'] == pytest.approx(0.833, abs=0.01)

    def test_regression_metrics(self):
        """Test regression metrics."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])

        metrics = BenchmarkMetrics.compute_regression_metrics(y_true, y_pred)

        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert metrics['rmse'] > 0
        assert metrics['mae'] > 0
        assert metrics['r2'] > 0.9

    def test_primary_metric(self):
        """Test primary metric selection."""
        assert BenchmarkMetrics.get_primary_metric('classification', 2) == 'roc_auc'
        assert BenchmarkMetrics.get_primary_metric('classification', 3) == 'accuracy'
        assert BenchmarkMetrics.get_primary_metric('regression') == 'rmse'


class TestBenchmarkRunner:
    """Test benchmark runner."""

    def test_sklearn_small_benchmark(self):
        """Test running sklearn-small benchmark suite."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(
                output_dir=Path(tmpdir),
                random_state=42
            )

            # Run on subset for speed
            results = runner.run_suite(
                suite_name='sklearn-small',
                models=['xgboost'],
                max_datasets=1,
                n_folds=2,
                verbose=False
            )

            assert 'suite_name' in results
            assert results['suite_name'] == 'sklearn-small'
            assert 'datasets' in results
            assert len(results['datasets']) > 0

            # Check first dataset has results
            first_dataset = list(results['datasets'].values())[0]
            assert 'models' in first_dataset
            assert 'xgboost' in first_dataset['models']

            # Check metrics exist
            xgb_results = first_dataset['models']['xgboost']
            assert 'metrics' in xgb_results
            assert len(xgb_results['metrics']) > 0

    def test_save_load_results(self):
        """Test saving and loading results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(
                output_dir=Path(tmpdir),
                random_state=42
            )

            # Run small benchmark
            results = runner.run_suite(
                suite_name='sklearn-small',
                models=['xgboost'],
                max_datasets=1,
                n_folds=2,
                verbose=False
            )

            # Save
            runner.save_results(results, 'sklearn-small')

            # Load
            loaded = runner.load_results('sklearn-small', version='latest')

            assert loaded['suite_name'] == results['suite_name']
            assert loaded['version'] == results['version']


class TestRegressionTest:
    """Test regression testing."""

    def test_compare_with_baseline(self):
        """Test comparing results with baseline."""
        # Create mock results
        baseline = {
            'suite_name': 'test-suite',
            'version': '1.0.0',
            'datasets': {
                'test_dataset': {
                    'task_type': 'classification',
                    'models': {
                        'xgboost': {
                            'metrics': {
                                'roc_auc': {'mean': 0.90, 'std': 0.02}
                            }
                        }
                    }
                }
            }
        }

        # Current results with slight degradation
        current = {
            'suite_name': 'test-suite',
            'version': '1.1.0',
            'datasets': {
                'test_dataset': {
                    'task_type': 'classification',
                    'models': {
                        'xgboost': {
                            'metrics': {
                                'roc_auc': {'mean': 0.89, 'std': 0.02}
                            }
                        }
                    }
                }
            }
        }

        tester = RegressionTest(tolerance=0.02)
        passed, comparison = tester.compare_with_baseline(
            current, baseline
        )

        assert 'datasets' in comparison
        assert 'test_dataset' in comparison['datasets']
        assert 'summary' in comparison

    def test_performance_degradation_detection(self):
        """Test detection of significant performance degradation."""
        baseline = {
            'suite_name': 'test-suite',
            'version': '1.0.0',
            'datasets': {
                'test_dataset': {
                    'task_type': 'classification',
                    'models': {
                        'xgboost': {
                            'metrics': {
                                'roc_auc': {'mean': 0.90, 'std': 0.02}
                            }
                        }
                    }
                }
            }
        }

        # Current with significant degradation (5%)
        current = {
            'suite_name': 'test-suite',
            'version': '1.1.0',
            'datasets': {
                'test_dataset': {
                    'task_type': 'classification',
                    'models': {
                        'xgboost': {
                            'metrics': {
                                'roc_auc': {'mean': 0.855, 'std': 0.02}
                            }
                        }
                    }
                }
            }
        }

        tester = RegressionTest(tolerance=0.02)  # 2% tolerance
        passed, comparison = tester.compare_with_baseline(
            current, baseline
        )

        # Should fail due to >2% degradation
        assert not passed
        assert comparison['summary']['failed'] > 0


@pytest.mark.slow
@pytest.mark.skipif(True, reason="Requires openml package and network access")
class TestOpenMLLoader:
    """Test OpenML loader (requires network)."""

    def test_load_openml_task(self):
        """Test loading single OpenML task."""
        from tabml.benchmarks import OpenMLLoader

        loader = OpenMLLoader()
        if not loader.available:
            pytest.skip("openml not installed")

        # Load a small task
        dataset = loader.load_dataset(task_id=3)

        assert 'X' in dataset
        assert 'y' in dataset
        assert 'task_type' in dataset


@pytest.mark.slow
@pytest.mark.skipif(True, reason="Requires pmlb package and network access")
class TestPMLBLoader:
    """Test PMLB loader (requires network)."""

    def test_load_pmlb_dataset(self):
        """Test loading PMLB dataset."""
        from tabml.benchmarks import PMLBLoader

        loader = PMLBLoader()
        if not loader.available:
            pytest.skip("pmlb not installed")

        # Load a small dataset
        dataset = loader.load_dataset('iris')

        assert 'X' in dataset
        assert 'y' in dataset
        assert dataset['task_type'] == 'classification'


@pytest.mark.integration
def test_full_benchmark_workflow():
    """Integration test for full benchmark workflow."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize runner
        runner = BenchmarkRunner(
            output_dir=Path(tmpdir),
            random_state=42
        )

        # Run benchmark
        results = runner.run_suite(
            suite_name='sklearn-small',
            models=['xgboost'],
            max_datasets=1,
            n_folds=2,
            verbose=False
        )

        # Save results
        runner.save_results(results, 'sklearn-small')

        # Set as baseline
        tester = RegressionTest(
            results_dir=Path(tmpdir),
            tolerance=0.02
        )
        tester.save_baseline(results, 'sklearn-small')

        # Compare with itself (should pass)
        passed, comparison = tester.compare_with_baseline(
            results,
            suite_name='sklearn-small'
        )

        assert passed
        assert comparison['summary']['failed'] == 0
