"""Benchmark validation system for TabML.

This module provides comprehensive benchmark testing against standard datasets
to validate model performance and detect regressions across versions.

Example:
    Quick benchmark test::

        from tabml.benchmarks import BenchmarkRunner

        runner = BenchmarkRunner()
        results = runner.run_suite('sklearn-small')
        runner.compare_with_baseline(results)

    Full OpenML benchmark::

        runner = BenchmarkRunner()
        results = runner.run_suite('openml-cc18', max_datasets=10)
        runner.save_results(results, 'openml-cc18')
"""

from .runner import BenchmarkRunner
from .loaders import (
    OpenMLLoader,
    PMLBLoader,
    SklearnLoader,
    BENCHMARK_SUITES
)
from .metrics import BenchmarkMetrics
from .comparison import RegressionTest, BaselineManager

__all__ = [
    'BenchmarkRunner',
    'OpenMLLoader',
    'PMLBLoader',
    'SklearnLoader',
    'BenchmarkMetrics',
    'RegressionTest',
    'BaselineManager',
    'BENCHMARK_SUITES'
]
