"""Baseline comparison and regression testing for benchmarks."""

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import numpy as np
import pandas as pd
from loguru import logger

from .loaders import BENCHMARK_SUITES


class RegressionTest:
    """Test that performance doesn't degrade across versions.

    This class compares current benchmark results against baseline
    results to detect performance regressions.

    Example:
        >>> tester = RegressionTest(tolerance=0.02)
        >>> current = runner.run_suite('sklearn-small')
        >>> passed, report = tester.compare_with_baseline(current, baseline)
        >>> if not passed:
        ...     print(report)
    """

    def __init__(self,
                 tolerance: float = 0.02,
                 results_dir: Optional[Path] = None):
        """Initialize regression tester.

        Args:
            tolerance: Allowed performance degradation (e.g., 0.02 = 2%)
            results_dir: Directory containing baseline results
        """
        self.tolerance = tolerance
        if results_dir is None:
            results_dir = Path.cwd() / 'benchmarks' / 'results'
        self.results_dir = Path(results_dir)

    def compare_with_baseline(self,
                              current_results: Dict,
                              baseline_results: Optional[Dict] = None,
                              suite_name: Optional[str] = None) -> Tuple[bool, Dict]:
        """Compare current results with baseline.

        Args:
            current_results: Current benchmark results
            baseline_results: Baseline results (if None, load from file)
            suite_name: Suite name (required if baseline_results is None)

        Returns:
            Tuple of (passed: bool, comparison_report: dict)
        """
        # Load baseline if not provided
        if baseline_results is None:
            if suite_name is None:
                suite_name = current_results.get('suite_name')
            baseline_results = self._load_baseline(suite_name)

        # Compare datasets
        comparison = {
            'suite_name': current_results.get('suite_name'),
            'baseline_version': baseline_results.get('version'),
            'current_version': current_results.get('version'),
            'tolerance': self.tolerance,
            'datasets': {},
            'summary': {
                'total_comparisons': 0,
                'passed': 0,
                'failed': 0,
                'warnings': 0
            }
        }

        # Compare each dataset
        for dataset_name in current_results.get('datasets', {}):
            if dataset_name not in baseline_results.get('datasets', {}):
                logger.warning(f"Dataset {dataset_name} not in baseline, skipping")
                continue

            current_data = current_results['datasets'][dataset_name]
            baseline_data = baseline_results['datasets'][dataset_name]

            dataset_comparison = self._compare_dataset(
                current_data, baseline_data, dataset_name
            )
            comparison['datasets'][dataset_name] = dataset_comparison

            # Update summary
            for model_result in dataset_comparison.get('models', {}).values():
                comparison['summary']['total_comparisons'] += 1
                if model_result['status'] == 'passed':
                    comparison['summary']['passed'] += 1
                elif model_result['status'] == 'failed':
                    comparison['summary']['failed'] += 1
                elif model_result['status'] == 'warning':
                    comparison['summary']['warnings'] += 1

        # Overall pass/fail
        passed = comparison['summary']['failed'] == 0
        comparison['overall_status'] = 'passed' if passed else 'failed'

        return passed, comparison

    def _compare_dataset(self,
                        current_data: Dict,
                        baseline_data: Dict,
                        dataset_name: str) -> Dict:
        """Compare results for a single dataset.

        Args:
            current_data: Current dataset results
            baseline_data: Baseline dataset results
            dataset_name: Name of dataset

        Returns:
            Comparison dictionary
        """
        comparison = {
            'dataset_name': dataset_name,
            'task_type': current_data.get('task_type'),
            'models': {}
        }

        # Compare each model
        for model_name in current_data.get('models', {}):
            if model_name not in baseline_data.get('models', {}):
                continue

            current_model = current_data['models'][model_name]
            baseline_model = baseline_data['models'][model_name]

            model_comparison = self._compare_model(
                current_model, baseline_model, model_name,
                current_data.get('task_type')
            )
            comparison['models'][model_name] = model_comparison

        return comparison

    def _compare_model(self,
                      current_model: Dict,
                      baseline_model: Dict,
                      model_name: str,
                      task_type: str) -> Dict:
        """Compare results for a single model.

        Args:
            current_model: Current model results
            baseline_model: Baseline model results
            model_name: Name of model
            task_type: Type of task

        Returns:
            Comparison dictionary
        """
        # Get primary metric
        primary_metric = self._get_primary_metric(task_type)

        comparison = {
            'model_name': model_name,
            'primary_metric': primary_metric,
            'metrics': {}
        }

        # Compare metrics
        if 'metrics' not in current_model or 'metrics' not in baseline_model:
            comparison['status'] = 'error'
            comparison['message'] = 'Missing metrics data'
            return comparison

        for metric_name in current_model['metrics']:
            if metric_name not in baseline_model['metrics']:
                continue

            current_value = current_model['metrics'][metric_name]['mean']
            baseline_value = baseline_model['metrics'][metric_name]['mean']

            # Compute difference
            # For metrics where higher is better (accuracy, AUC, R²)
            # degradation is negative if current < baseline
            # For metrics where lower is better (RMSE, MAE, log_loss)
            # degradation is positive if current > baseline

            is_higher_better = self._is_higher_better(metric_name)

            if is_higher_better:
                degradation = baseline_value - current_value
            else:
                degradation = current_value - baseline_value

            # Normalize by baseline value
            if baseline_value != 0:
                relative_degradation = degradation / abs(baseline_value)
            else:
                relative_degradation = 0

            # Determine status
            if relative_degradation > self.tolerance:
                status = 'failed'
            elif relative_degradation > self.tolerance / 2:
                status = 'warning'
            else:
                status = 'passed'

            comparison['metrics'][metric_name] = {
                'current': current_value,
                'baseline': baseline_value,
                'degradation': degradation,
                'relative_degradation': relative_degradation,
                'status': status
            }

        # Overall model status based on primary metric
        if primary_metric in comparison['metrics']:
            comparison['status'] = comparison['metrics'][primary_metric]['status']
            comparison['primary_metric_degradation'] = \
                comparison['metrics'][primary_metric]['relative_degradation']
        else:
            comparison['status'] = 'unknown'

        return comparison

    def _load_baseline(self, suite_name: str) -> Dict:
        """Load baseline results from file.

        Args:
            suite_name: Name of benchmark suite

        Returns:
            Baseline results dictionary
        """
        baseline_file = self.results_dir / f"{suite_name}_baseline.json"

        if not baseline_file.exists():
            # Try loading latest as baseline
            latest_file = self.results_dir / f"{suite_name}_latest.json"
            if latest_file.exists():
                logger.warning(f"No baseline found, using latest: {latest_file}")
                baseline_file = latest_file
            else:
                raise FileNotFoundError(
                    f"No baseline found for {suite_name}. "
                    f"Create one with: runner.save_baseline(results, '{suite_name}')"
                )

        with open(baseline_file) as f:
            return json.load(f)

    def save_baseline(self, results: Dict, suite_name: str):
        """Save current results as baseline.

        Args:
            results: Results to save as baseline
            suite_name: Name of benchmark suite
        """
        baseline_file = self.results_dir / f"{suite_name}_baseline.json"

        with open(baseline_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Baseline saved: {baseline_file}")

    def _get_primary_metric(self, task_type: str) -> str:
        """Get primary metric for task type.

        Args:
            task_type: 'classification' or 'regression'

        Returns:
            Primary metric name
        """
        if task_type == 'classification':
            return 'roc_auc'
        else:
            return 'rmse'

    def _is_higher_better(self, metric_name: str) -> bool:
        """Check if higher metric values are better.

        Args:
            metric_name: Name of metric

        Returns:
            True if higher is better
        """
        lower_better = ['rmse', 'mae', 'log_loss', 'mape']
        return metric_name not in lower_better

    def generate_report(self, comparison: Dict) -> str:
        """Generate human-readable comparison report.

        Args:
            comparison: Comparison dictionary from compare_with_baseline()

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append(f"Benchmark Regression Test Report: {comparison['suite_name']}")
        report.append("=" * 80)
        report.append(f"Baseline version: {comparison['baseline_version']}")
        report.append(f"Current version:  {comparison['current_version']}")
        report.append(f"Tolerance:        {comparison['tolerance']:.2%}")
        report.append("")

        summary = comparison['summary']
        report.append("Summary:")
        report.append(f"  Total comparisons: {summary['total_comparisons']}")
        report.append(f"  Passed:            {summary['passed']} ✓")
        report.append(f"  Failed:            {summary['failed']} ✗")
        report.append(f"  Warnings:          {summary['warnings']} ⚠")
        report.append("")

        if summary['failed'] > 0 or summary['warnings'] > 0:
            report.append("Details:")
            report.append("-" * 80)

            for dataset_name, dataset_comp in comparison['datasets'].items():
                dataset_has_issues = False

                for model_name, model_comp in dataset_comp.get('models', {}).items():
                    if model_comp.get('status') in ['failed', 'warning']:
                        if not dataset_has_issues:
                            report.append(f"\nDataset: {dataset_name}")
                            dataset_has_issues = True

                        status_icon = '✗' if model_comp['status'] == 'failed' else '⚠'
                        report.append(f"  {status_icon} {model_name}:")

                        primary_metric = model_comp.get('primary_metric')
                        if primary_metric and primary_metric in model_comp['metrics']:
                            metric_data = model_comp['metrics'][primary_metric]
                            report.append(
                                f"      {primary_metric}: "
                                f"{metric_data['current']:.4f} "
                                f"(baseline: {metric_data['baseline']:.4f}, "
                                f"degradation: {metric_data['relative_degradation']:.2%})"
                            )

        report.append("")
        report.append("=" * 80)
        report.append(f"Overall Status: {comparison['overall_status'].upper()}")
        report.append("=" * 80)

        return "\n".join(report)

    def create_comparison_table(self, comparison: Dict) -> pd.DataFrame:
        """Create DataFrame table from comparison results.

        Args:
            comparison: Comparison dictionary

        Returns:
            DataFrame with comparison results
        """
        rows = []

        for dataset_name, dataset_comp in comparison['datasets'].items():
            for model_name, model_comp in dataset_comp.get('models', {}).items():
                primary_metric = model_comp.get('primary_metric')

                if primary_metric and primary_metric in model_comp['metrics']:
                    metric_data = model_comp['metrics'][primary_metric]

                    rows.append({
                        'Dataset': dataset_name,
                        'Model': model_name,
                        'Metric': primary_metric,
                        'Current': f"{metric_data['current']:.4f}",
                        'Baseline': f"{metric_data['baseline']:.4f}",
                        'Degradation': f"{metric_data['relative_degradation']:.2%}",
                        'Status': model_comp['status']
                    })

        return pd.DataFrame(rows)


class BaselineManager:
    """Manage baseline results for multiple suites."""

    def __init__(self, results_dir: Optional[Path] = None):
        """Initialize baseline manager.

        Args:
            results_dir: Directory containing results
        """
        if results_dir is None:
            results_dir = Path.cwd() / 'benchmarks' / 'results'
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def list_baselines(self) -> List[Dict[str, str]]:
        """List all available baselines.

        Returns:
            List of baseline info dictionaries
        """
        baselines = []

        for baseline_file in self.results_dir.glob("*_baseline.json"):
            with open(baseline_file) as f:
                data = json.load(f)

            suite_name = baseline_file.stem.replace('_baseline', '')
            baselines.append({
                'suite': suite_name,
                'version': data.get('version', 'unknown'),
                'timestamp': data.get('timestamp', 'unknown'),
                'file': str(baseline_file)
            })

        return baselines

    def get_expected_performance(self, suite_name: str) -> Dict[str, Any]:
        """Get expected performance targets for a suite.

        Args:
            suite_name: Name of benchmark suite

        Returns:
            Dictionary of expected performance metrics
        """
        if suite_name not in BENCHMARK_SUITES:
            raise ValueError(f"Unknown suite: {suite_name}")

        suite_info = BENCHMARK_SUITES[suite_name]
        return suite_info.get('expected_baselines', {})
