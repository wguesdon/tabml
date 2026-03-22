"""Example: Running TabML benchmarks for validation and regression testing.

This example demonstrates how to:
1. Run benchmark suites (sklearn, OpenML, PMLB)
2. Compare performance against baselines
3. Detect performance regressions
4. Use benchmarks in CI/CD pipelines
"""

from pathlib import Path
from tabml.benchmarks import (
    BenchmarkRunner,
    RegressionTest,
    BaselineManager,
    BENCHMARK_SUITES
)
from loguru import logger


def example_1_quick_benchmark():
    """Example 1: Run quick sklearn benchmarks."""
    logger.info("=" * 70)
    logger.info("Example 1: Quick Sklearn Benchmarks")
    logger.info("=" * 70)

    # Initialize runner
    runner = BenchmarkRunner(
        output_dir=Path("benchmarks/results"),
        random_state=42
    )

    # Run sklearn-small suite (fast, ~1 minute)
    logger.info("Running sklearn-small benchmark suite...")
    results = runner.run_suite(
        suite_name='sklearn-small',
        models=['xgboost', 'lightgbm', 'catboost'],
        n_folds=5,
        verbose=True
    )

    # Print summary
    runner.print_summary(results)

    # Save results
    runner.save_results(results, 'sklearn-small')

    logger.info("\n✓ Benchmark completed and saved!")


def example_2_openml_benchmark():
    """Example 2: Run OpenML-CC18 benchmark (subset)."""
    logger.info("\n" + "=" * 70)
    logger.info("Example 2: OpenML-CC18 Benchmark (10 datasets)")
    logger.info("=" * 70)

    try:
        runner = BenchmarkRunner(random_state=42)

        # Run OpenML-CC18 on first 10 datasets
        # Full suite takes 1-2 hours, subset takes ~10 minutes
        logger.info("Running OpenML-CC18 benchmark (first 10 datasets)...")
        results = runner.run_suite(
            suite_name='openml-cc18',
            models=['xgboost', 'lightgbm'],
            max_datasets=10,  # Limit for faster execution
            n_folds=5,
            verbose=True
        )

        runner.print_summary(results)
        runner.save_results(results, 'openml-cc18')

        logger.info("\n✓ OpenML benchmark completed!")

    except ImportError as e:
        logger.warning(f"OpenML not available: {e}")
        logger.info("Install with: pip install openml")


def example_3_set_baseline():
    """Example 3: Set baseline for regression testing."""
    logger.info("\n" + "=" * 70)
    logger.info("Example 3: Setting Baseline")
    logger.info("=" * 70)

    runner = BenchmarkRunner(random_state=42)

    # Run benchmark
    logger.info("Running benchmark to create baseline...")
    results = runner.run_suite(
        suite_name='sklearn-small',
        models=['xgboost', 'lightgbm', 'catboost'],
        max_datasets=3,
        n_folds=3,
        verbose=False
    )

    # Save as baseline
    tester = RegressionTest()
    tester.save_baseline(results, 'sklearn-small')

    logger.info("\n✓ Baseline set for sklearn-small!")


def example_4_regression_test():
    """Example 4: Run regression test against baseline."""
    logger.info("\n" + "=" * 70)
    logger.info("Example 4: Regression Testing")
    logger.info("=" * 70)

    # First, ensure we have a baseline
    runner = BenchmarkRunner(random_state=42)

    logger.info("Creating baseline...")
    baseline_results = runner.run_suite(
        suite_name='sklearn-small',
        models=['xgboost'],
        max_datasets=2,
        n_folds=3,
        verbose=False
    )

    tester = RegressionTest(tolerance=0.02)  # 2% tolerance
    tester.save_baseline(baseline_results, 'sklearn-small')

    # Run new benchmark
    logger.info("\nRunning new benchmark...")
    current_results = runner.run_suite(
        suite_name='sklearn-small',
        models=['xgboost'],
        max_datasets=2,
        n_folds=3,
        verbose=False
    )

    # Compare with baseline
    logger.info("\nComparing with baseline...")
    passed, comparison = tester.compare_with_baseline(
        current_results,
        suite_name='sklearn-small'
    )

    # Print report
    report = tester.generate_report(comparison)
    print("\n" + report)

    # Print detailed table
    table = tester.create_comparison_table(comparison)
    if not table.empty:
        print("\nDetailed Comparison:")
        print(table.to_string(index=False))

    if passed:
        logger.info("\n✓ No performance regression detected!")
    else:
        logger.warning("\n⚠ Performance regression detected!")


def example_5_list_suites():
    """Example 5: List available benchmark suites."""
    logger.info("\n" + "=" * 70)
    logger.info("Example 5: Available Benchmark Suites")
    logger.info("=" * 70)

    print("\nAvailable Benchmark Suites:")
    print("=" * 70)

    for suite_name, suite_info in BENCHMARK_SUITES.items():
        print(f"\n{suite_name}")
        print(f"  Type:             {suite_info['type']}")
        print(f"  Datasets:         {suite_info['n_datasets']}")
        print(f"  Description:      {suite_info['description']}")
        print(f"  Expected time:    {suite_info['expected_time']}")

        if 'expected_baselines' in suite_info:
            print(f"  Expected baseline:")
            for metric, value in suite_info['expected_baselines'].items():
                print(f"    - {metric}: {value}")

    print("\n" + "=" * 70)


def example_6_manage_baselines():
    """Example 6: Manage baseline results."""
    logger.info("\n" + "=" * 70)
    logger.info("Example 6: Managing Baselines")
    logger.info("=" * 70)

    manager = BaselineManager()

    # List all baselines
    baselines = manager.list_baselines()

    if not baselines:
        logger.info("No baselines found yet. Run example 3 to create one.")
    else:
        print("\nAvailable Baselines:")
        print("=" * 70)
        for baseline in baselines:
            print(f"\nSuite:     {baseline['suite']}")
            print(f"Version:   {baseline['version']}")
            print(f"Timestamp: {baseline['timestamp']}")
            print(f"File:      {baseline['file']}")

    # Get expected performance for a suite
    logger.info("\nExpected performance for sklearn-small:")
    expected = manager.get_expected_performance('sklearn-small')
    for metric, value in expected.items():
        print(f"  {metric}: {value}")


def example_7_ci_cd_workflow():
    """Example 7: CI/CD workflow simulation."""
    logger.info("\n" + "=" * 70)
    logger.info("Example 7: CI/CD Workflow")
    logger.info("=" * 70)

    logger.info("""
This demonstrates how to use benchmarks in CI/CD:

1. Pre-commit: Run quick benchmark
   $ python -m tabml.benchmarks.cli run sklearn-small --max-datasets 2

2. Pull Request: Compare with baseline
   $ python -m tabml.benchmarks.cli compare sklearn-small --fail-on-regression

3. Main branch: Update baseline
   $ python -m tabml.benchmarks.cli baseline set sklearn-small

4. GitHub Actions example:
   See .github/workflows/benchmark.yml
    """)

    # Simulate PR check
    logger.info("\nSimulating PR benchmark check...")

    runner = BenchmarkRunner(random_state=42)

    # Create baseline
    baseline_results = runner.run_suite(
        suite_name='sklearn-small',
        models=['xgboost'],
        max_datasets=1,
        n_folds=2,
        verbose=False
    )

    tester = RegressionTest(tolerance=0.02)
    tester.save_baseline(baseline_results, 'sklearn-small')

    # Run PR benchmark
    pr_results = runner.run_suite(
        suite_name='sklearn-small',
        models=['xgboost'],
        max_datasets=1,
        n_folds=2,
        verbose=False
    )

    # Check for regressions
    passed, comparison = tester.compare_with_baseline(
        pr_results,
        suite_name='sklearn-small'
    )

    if passed:
        logger.info("\n✓ CI Check PASSED: No performance regression")
        logger.info("  → PR can be merged")
    else:
        logger.error("\n✗ CI Check FAILED: Performance regression detected")
        logger.error("  → PR should be reviewed or rejected")


def main():
    """Run all examples."""
    logger.info("TabML Benchmark Validation Examples")
    logger.info("====================================\n")

    # Run examples
    example_1_quick_benchmark()
    # example_2_openml_benchmark()  # Commented out - requires openml and takes longer
    example_3_set_baseline()
    example_4_regression_test()
    example_5_list_suites()
    example_6_manage_baselines()
    example_7_ci_cd_workflow()

    logger.info("\n" + "=" * 70)
    logger.info("All examples completed!")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
