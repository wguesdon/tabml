"""Quick test to verify benchmark system works end-to-end.

Run this script to validate the benchmark installation:
    python examples/test_benchmark_system.py
"""

import sys
from pathlib import Path
from loguru import logger

def test_imports():
    """Test that all benchmark modules can be imported."""
    logger.info("Testing imports...")
    try:
        from tabml.benchmarks import (
            BenchmarkRunner,
            RegressionTest,
            BaselineManager,
            SklearnLoader,
            BenchmarkMetrics,
            BENCHMARK_SUITES
        )
        logger.info("✓ All imports successful")
        return True
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_sklearn_loader():
    """Test sklearn dataset loader."""
    logger.info("\nTesting sklearn loader...")
    try:
        from tabml.benchmarks import SklearnLoader

        loader = SklearnLoader()
        dataset = loader.load_dataset('breast_cancer')

        assert 'X' in dataset
        assert 'y' in dataset
        assert len(dataset['X']) == len(dataset['y'])

        logger.info(f"✓ Loaded {dataset['name']}: {len(dataset['X'])} samples, "
                   f"{dataset['X'].shape[1]} features")
        return True
    except Exception as e:
        logger.error(f"✗ Sklearn loader failed: {e}")
        return False


def test_quick_benchmark():
    """Test running a quick benchmark."""
    logger.info("\nTesting quick benchmark...")
    try:
        from tabml.benchmarks import BenchmarkRunner
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(
                output_dir=Path(tmpdir),
                random_state=42
            )

            # Run tiny benchmark
            results = runner.run_suite(
                suite_name='sklearn-small',
                models=['xgboost'],
                max_datasets=1,
                n_folds=2,
                verbose=False
            )

            assert 'suite_name' in results
            assert 'datasets' in results
            assert len(results['datasets']) > 0

            logger.info("✓ Quick benchmark completed")
            return True
    except Exception as e:
        logger.error(f"✗ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    """Test metrics computation."""
    logger.info("\nTesting metrics...")
    try:
        from tabml.benchmarks import BenchmarkMetrics
        import numpy as np

        # Test classification metrics
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 0])
        y_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2],
                           [0.3, 0.7], [0.7, 0.3], [0.6, 0.4]])

        metrics = BenchmarkMetrics.compute_classification_metrics(
            y_true, y_pred, y_proba
        )

        assert 'accuracy' in metrics
        assert 'roc_auc' in metrics
        logger.info(f"✓ Classification metrics: {list(metrics.keys())}")

        # Test regression metrics
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])

        metrics = BenchmarkMetrics.compute_regression_metrics(y_true, y_pred)

        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        logger.info(f"✓ Regression metrics: {list(metrics.keys())}")

        return True
    except Exception as e:
        logger.error(f"✗ Metrics test failed: {e}")
        return False


def test_regression_tester():
    """Test regression testing."""
    logger.info("\nTesting regression tester...")
    try:
        from tabml.benchmarks import RegressionTest
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tester = RegressionTest(
                tolerance=0.02,
                results_dir=Path(tmpdir)
            )

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

            passed, comparison = tester.compare_with_baseline(
                current, baseline
            )

            assert 'datasets' in comparison
            assert 'summary' in comparison

            logger.info(f"✓ Regression test completed: "
                       f"passed={passed}, "
                       f"comparisons={comparison['summary']['total_comparisons']}")
            return True
    except Exception as e:
        logger.error(f"✗ Regression tester failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_suite_definitions():
    """Test benchmark suite definitions."""
    logger.info("\nTesting suite definitions...")
    try:
        from tabml.benchmarks import BENCHMARK_SUITES

        assert 'sklearn-small' in BENCHMARK_SUITES
        assert 'openml-cc18' in BENCHMARK_SUITES
        assert 'openml-ctr23' in BENCHMARK_SUITES
        assert 'pmlb-mini' in BENCHMARK_SUITES

        logger.info(f"✓ Found {len(BENCHMARK_SUITES)} benchmark suites:")
        for suite_name in BENCHMARK_SUITES:
            logger.info(f"  - {suite_name}")

        return True
    except Exception as e:
        logger.error(f"✗ Suite definitions test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("=" * 70)
    logger.info("TabML Benchmark System Test")
    logger.info("=" * 70)

    tests = [
        ("Imports", test_imports),
        ("Suite Definitions", test_suite_definitions),
        ("Sklearn Loader", test_sklearn_loader),
        ("Metrics", test_metrics),
        ("Regression Tester", test_regression_tester),
        ("Quick Benchmark", test_quick_benchmark),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results.append((test_name, False))

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Test Summary")
    logger.info("=" * 70)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status:8s} {test_name}")

    logger.info("=" * 70)
    logger.info(f"Results: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        logger.info("✓ All tests passed! Benchmark system is working correctly.")
        return 0
    else:
        logger.error(f"✗ {total_count - passed_count} test(s) failed.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
