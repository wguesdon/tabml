"""Command-line interface for running benchmarks."""

import argparse
import sys
from pathlib import Path
from loguru import logger

from .runner import BenchmarkRunner
from .comparison import RegressionTest, BaselineManager
from .loaders import BENCHMARK_SUITES


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Run TabML benchmarks and regression tests',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run quick sklearn benchmarks
  tabml-benchmark run sklearn-small

  # Run OpenML-CC18 (first 10 datasets)
  tabml-benchmark run openml-cc18 --max-datasets 10

  # Compare with baseline
  tabml-benchmark compare sklearn-small

  # Set new baseline
  tabml-benchmark baseline sklearn-small

  # List available suites
  tabml-benchmark list
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # List command
    list_parser = subparsers.add_parser('list', help='List available benchmark suites')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run benchmark suite')
    run_parser.add_argument('suite', choices=list(BENCHMARK_SUITES.keys()),
                           help='Benchmark suite to run')
    run_parser.add_argument('--models', nargs='+',
                           choices=['xgboost', 'lightgbm', 'catboost'],
                           default=['xgboost', 'lightgbm', 'catboost'],
                           help='Models to benchmark')
    run_parser.add_argument('--max-datasets', type=int,
                           help='Limit number of datasets (for testing)')
    run_parser.add_argument('--n-folds', type=int, default=5,
                           help='Number of CV folds')
    run_parser.add_argument('--output-dir', type=Path,
                           help='Output directory for results')
    run_parser.add_argument('--save-baseline', action='store_true',
                           help='Save results as baseline')

    # Compare command
    compare_parser = subparsers.add_parser('compare',
                                          help='Compare results with baseline')
    compare_parser.add_argument('suite', help='Benchmark suite name')
    compare_parser.add_argument('--tolerance', type=float, default=0.02,
                               help='Allowed degradation tolerance (default: 0.02 = 2%%)')
    compare_parser.add_argument('--results-file', type=Path,
                               help='Results file to compare (default: latest)')
    compare_parser.add_argument('--fail-on-regression', action='store_true',
                               help='Exit with error code if regression detected')

    # Baseline command
    baseline_parser = subparsers.add_parser('baseline',
                                           help='Manage baselines')
    baseline_subparsers = baseline_parser.add_subparsers(dest='baseline_cmd')

    baseline_list = baseline_subparsers.add_parser('list',
                                                   help='List all baselines')

    baseline_set = baseline_subparsers.add_parser('set',
                                                  help='Set baseline from results')
    baseline_set.add_argument('suite', help='Benchmark suite name')
    baseline_set.add_argument('--results-file', type=Path,
                             help='Results file (default: latest)')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    # Execute command
    if args.command == 'list':
        return cmd_list()
    elif args.command == 'run':
        return cmd_run(args)
    elif args.command == 'compare':
        return cmd_compare(args)
    elif args.command == 'baseline':
        return cmd_baseline(args)
    else:
        parser.print_help()
        return 1


def cmd_list():
    """List available benchmark suites."""
    print("\nAvailable Benchmark Suites:")
    print("=" * 80)

    for suite_name, suite_info in BENCHMARK_SUITES.items():
        print(f"\n{suite_name}")
        print(f"  Type:        {suite_info['type']}")
        print(f"  Datasets:    {suite_info['n_datasets']}")
        print(f"  Description: {suite_info['description']}")
        print(f"  Time:        {suite_info['expected_time']}")

    print("\n" + "=" * 80)
    return 0


def cmd_run(args):
    """Run benchmark suite."""
    logger.info(f"Running benchmark suite: {args.suite}")

    runner = BenchmarkRunner(
        output_dir=args.output_dir,
        random_state=42
    )

    try:
        results = runner.run_suite(
            suite_name=args.suite,
            models=args.models,
            max_datasets=args.max_datasets,
            n_folds=args.n_folds,
            verbose=True
        )

        # Print summary
        runner.print_summary(results)

        # Save results
        runner.save_results(results, args.suite)

        # Save as baseline if requested
        if args.save_baseline:
            tester = RegressionTest(results_dir=runner.output_dir)
            tester.save_baseline(results, args.suite)
            logger.info(f"Saved as baseline for {args.suite}")

        return 0

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1


def cmd_compare(args):
    """Compare results with baseline."""
    logger.info(f"Comparing {args.suite} with baseline...")

    tester = RegressionTest(tolerance=args.tolerance)

    try:
        # Load current results
        if args.results_file:
            import json
            with open(args.results_file) as f:
                current_results = json.load(f)
        else:
            # Load latest
            runner = BenchmarkRunner()
            current_results = runner.load_results(args.suite, version='latest')

        # Compare with baseline
        passed, comparison = tester.compare_with_baseline(
            current_results,
            suite_name=args.suite
        )

        # Print report
        report = tester.generate_report(comparison)
        print(report)

        # Create table
        table = tester.create_comparison_table(comparison)
        if not table.empty:
            print("\nDetailed Comparison:")
            print(table.to_string(index=False))

        # Exit with error if failed and requested
        if not passed and args.fail_on_regression:
            logger.error("Performance regression detected!")
            return 1

        return 0

    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        return 1


def cmd_baseline(args):
    """Manage baselines."""
    manager = BaselineManager()

    if args.baseline_cmd == 'list':
        baselines = manager.list_baselines()

        if not baselines:
            print("No baselines found.")
            return 0

        print("\nAvailable Baselines:")
        print("=" * 80)
        for baseline in baselines:
            print(f"\nSuite:     {baseline['suite']}")
            print(f"Version:   {baseline['version']}")
            print(f"Timestamp: {baseline['timestamp']}")
            print(f"File:      {baseline['file']}")
        print("\n" + "=" * 80)

    elif args.baseline_cmd == 'set':
        tester = RegressionTest()

        # Load results
        if args.results_file:
            import json
            with open(args.results_file) as f:
                results = json.load(f)
        else:
            runner = BenchmarkRunner()
            results = runner.load_results(args.suite, version='latest')

        # Save as baseline
        tester.save_baseline(results, args.suite)
        logger.info(f"Baseline set for {args.suite}")

    else:
        print("Use 'list' or 'set' subcommand")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
