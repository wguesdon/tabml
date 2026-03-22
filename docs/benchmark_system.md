# TabML Benchmark System

A comprehensive benchmarking system for validating TabML models against standard datasets and detecting performance regressions across versions.

## Overview

The benchmark system provides:
- **Standard benchmark suites**: sklearn, OpenML-CC18, OpenML-CTR23, PMLB-mini
- **Automated regression testing**: Detect performance degradations
- **Baseline management**: Track performance over versions
- **CI/CD integration**: Automated quality gates

## Quick Start

### 1. Run Quick Benchmark

```python
from tabml.benchmarks import BenchmarkRunner

runner = BenchmarkRunner()
results = runner.run_suite('sklearn-small')
runner.print_summary(results)
runner.save_results(results, 'sklearn-small')
```

### 2. Set Baseline

```python
from tabml.benchmarks import RegressionTest

tester = RegressionTest()
tester.save_baseline(results, 'sklearn-small')
```

### 3. Check for Regressions

```python
# Run new benchmark
current_results = runner.run_suite('sklearn-small')

# Compare with baseline
passed, comparison = tester.compare_with_baseline(
    current_results,
    suite_name='sklearn-small'
)

# Print report
report = tester.generate_report(comparison)
print(report)
```

## Benchmark Suites

### sklearn-small (Recommended for CI)
- **Datasets**: 5 (breast_cancer, wine, iris, digits, diabetes, california_housing)
- **Time**: < 1 minute
- **Use case**: Fast validation in CI/CD pipelines
- **Expected baselines**:
  - XGBoost breast_cancer AUC: ≥ 0.97
  - XGBoost diabetes R²: ≥ 0.40

### OpenML-CC18 (Classification)
- **Suite ID**: 99
- **Datasets**: 72 curated classification datasets
- **Samples**: 500-100K per dataset
- **Features**: < 5K per dataset
- **Time**: 1-2 hours (full suite), ~10 min (10 datasets)
- **Use case**: Comprehensive classification validation
- **Expected baseline**: XGBoost avg AUC ~0.80

### OpenML-CTR23 (Regression)
- **Suite ID**: 271
- **Datasets**: 30 curated regression datasets
- **Time**: 30-60 minutes (full suite)
- **Use case**: Comprehensive regression validation
- **Expected baseline**: XGBoost rank 1.31

### PMLB-mini (Small Data)
- **Datasets**: 44 binary classification (n ≤ 500)
- **Time**: 5-10 minutes
- **Use case**: Small data regime validation
- **Expected baseline**: XGBoost accuracy ~0.75

## Command-Line Usage

### List Available Suites
```bash
python -m tabml.benchmarks.cli list
```

### Run Benchmark
```bash
# Quick sklearn benchmark
python -m tabml.benchmarks.cli run sklearn-small

# OpenML-CC18 (first 10 datasets)
python -m tabml.benchmarks.cli run openml-cc18 --max-datasets 10

# Specific models only
python -m tabml.benchmarks.cli run sklearn-small --models xgboost lightgbm

# Save as baseline
python -m tabml.benchmarks.cli run sklearn-small --save-baseline
```

### Compare with Baseline
```bash
# Compare latest results
python -m tabml.benchmarks.cli compare sklearn-small

# Fail on regression (for CI)
python -m tabml.benchmarks.cli compare sklearn-small --fail-on-regression

# Custom tolerance (default 2%)
python -m tabml.benchmarks.cli compare sklearn-small --tolerance 0.05
```

### Manage Baselines
```bash
# List all baselines
python -m tabml.benchmarks.cli baseline list

# Set baseline from latest results
python -m tabml.benchmarks.cli baseline set sklearn-small
```

## Python API

### Basic Usage

```python
from tabml.benchmarks import BenchmarkRunner, RegressionTest

# Initialize runner
runner = BenchmarkRunner(
    output_dir='benchmarks/results',
    random_state=42
)

# Run benchmark suite
results = runner.run_suite(
    suite_name='sklearn-small',
    models=['xgboost', 'lightgbm', 'catboost'],
    max_datasets=None,  # All datasets
    n_folds=5,
    verbose=True
)

# Save results
runner.save_results(results, 'sklearn-small')
```

### Regression Testing

```python
from tabml.benchmarks import RegressionTest

# Initialize tester
tester = RegressionTest(
    tolerance=0.02,  # 2% allowed degradation
    results_dir='benchmarks/results'
)

# Set baseline
tester.save_baseline(results, 'sklearn-small')

# Run new benchmark and compare
current_results = runner.run_suite('sklearn-small')
passed, comparison = tester.compare_with_baseline(
    current_results,
    suite_name='sklearn-small'
)

# Generate report
if not passed:
    report = tester.generate_report(comparison)
    print(report)

    # Get detailed table
    table = tester.create_comparison_table(comparison)
    print(table)
```

### Custom Dataset Loaders

```python
from tabml.benchmarks import SklearnLoader, OpenMLLoader, PMLBLoader

# Load sklearn datasets
sklearn_loader = SklearnLoader()
datasets = sklearn_loader.load_suite(max_datasets=5)

# Load OpenML suite
openml_loader = OpenMLLoader()
if openml_loader.available:
    datasets = openml_loader.load_suite(suite_id=99, max_datasets=10)

# Load PMLB datasets
pmlb_loader = PMLBLoader()
if pmlb_loader.available:
    datasets = pmlb_loader.load_suite('small', max_datasets=10)
```

## CI/CD Integration

### GitHub Actions

Create `.github/workflows/benchmark.yml`:

```yaml
name: Benchmark Tests

on:
  pull_request:
    branches: [ main ]
  push:
    branches: [ main ]

jobs:
  benchmark:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install -e ".[all]"

    - name: Run quick benchmarks
      run: |
        python -m tabml.benchmarks.cli run sklearn-small --max-datasets 2

    - name: Compare with baseline
      run: |
        python -m tabml.benchmarks.cli compare sklearn-small --fail-on-regression

    - name: Upload results
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results
        path: benchmarks/results/
```

### Pre-commit Hook

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
echo "Running quick benchmarks..."
python -m tabml.benchmarks.cli run sklearn-small --max-datasets 1 --n-folds 2

if [ $? -ne 0 ]; then
    echo "Benchmark failed!"
    exit 1
fi

echo "Benchmark passed ✓"
```

## Performance Baselines

### Expected Performance Thresholds

Based on research and empirical testing:

| Dataset | Model | Metric | Expected | Notes |
|---------|-------|--------|----------|-------|
| breast_cancer | XGBoost | AUC | ≥ 0.97 | Binary classification |
| wine | XGBoost | Accuracy | ≥ 0.95 | Multi-class |
| diabetes | XGBoost | R² | ≥ 0.40 | Regression |
| OpenML-CC18 | XGBoost | AUC (avg) | ≥ 0.75 | Avg across 72 datasets |
| OpenML-CTR23 | XGBoost | Rank | ≤ 2.0 | Rank among models |

### Degradation Tolerance

Default tolerance: **2%** relative degradation

Example:
- Baseline: 0.90 AUC
- Tolerance: 0.02
- Minimum acceptable: 0.882 AUC (0.90 - 0.90*0.02)

Adjust tolerance based on:
- **Stricter** (0.01): Production critical models
- **Moderate** (0.02): Development/testing (default)
- **Relaxed** (0.05): Early development

## Result Storage

Benchmark results are stored in JSON format:

```
benchmarks/results/
├── sklearn-small_baseline.json    # Current baseline
├── sklearn-small_latest.json      # Most recent run
├── sklearn-small_20250101_120000.json  # Timestamped results
├── openml-cc18_baseline.json
└── openml-cc18_latest.json
```

### Result Structure

```json
{
  "suite_name": "sklearn-small",
  "version": "0.5.1",
  "timestamp": "2025-01-01T12:00:00",
  "n_folds": 5,
  "models": ["xgboost", "lightgbm", "catboost"],
  "datasets": {
    "breast_cancer": {
      "task_type": "classification",
      "n_samples": 569,
      "n_features": 30,
      "models": {
        "xgboost": {
          "metrics": {
            "roc_auc": {
              "mean": 0.9850,
              "std": 0.0123,
              "values": [0.98, 0.99, ...]
            },
            "accuracy": {...}
          },
          "time_seconds": 1.23,
          "status": "success"
        }
      }
    }
  },
  "aggregate": {
    "xgboost": {
      "roc_auc": {
        "mean": 0.9200,
        "std": 0.0543,
        "median": 0.9350,
        "min": 0.8100,
        "max": 0.9950
      }
    }
  }
}
```

## Advanced Usage

### Custom Metrics

```python
from tabml.benchmarks import BenchmarkMetrics

# Compute custom metrics
y_true = [0, 1, 1, 0]
y_pred = [0, 1, 0, 0]
y_proba = [[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.8, 0.2]]

metrics = BenchmarkMetrics.compute_classification_metrics(
    y_true, y_pred, y_proba
)
# Returns: {'accuracy': 0.75, 'roc_auc': 0.833, 'f1_macro': 0.67, ...}
```

### Custom Benchmark Suite

```python
from tabml.benchmarks import BaseLoader

class CustomLoader(BaseLoader):
    def load_dataset(self, dataset_id):
        # Load your custom dataset
        return {
            'name': 'my_dataset',
            'X': X_df,
            'y': y_series,
            'task_type': 'classification',
            'metadata': {}
        }

    def load_suite(self, max_datasets=None):
        # Load multiple datasets
        return [self.load_dataset(i) for i in range(10)]
```

## Troubleshooting

### OpenML Connection Issues

```bash
# Set OpenML API key
export OPENML_API_KEY="your_key_here"

# Or in Python
import openml
openml.config.apikey = 'your_key_here'
```

### Memory Issues with Large Datasets

```python
# Limit datasets
runner.run_suite('openml-cc18', max_datasets=5)

# Reduce folds
runner.run_suite('sklearn-small', n_folds=3)
```

### Baseline Not Found

```python
# Create initial baseline
results = runner.run_suite('sklearn-small')
tester.save_baseline(results, 'sklearn-small')
```

## References

1. **OpenML-CC18**: Bischl et al. (2021) "OpenML Benchmarking Suites"
2. **OpenML-CTR23**: Grinsztajn et al. (2022) "Why do tree-based models still outperform deep learning on tabular data?"
3. **PMLB**: Romano et al. (2021) "PMLB v1.0: an open source dataset collection for benchmarking machine learning methods"

## See Also

- [Benchmark Examples](../examples/benchmark_validation_example.py)
- [Test Suite](../tests/test_benchmarks.py)
- [OpenML Documentation](https://docs.openml.org/)
- [PMLB Repository](https://github.com/EpistasisLab/pmlb)
