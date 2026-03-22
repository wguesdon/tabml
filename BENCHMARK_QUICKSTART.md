# TabML Benchmark System - Quick Start Guide

## Installation

```bash
# Install TabML with benchmark dependencies
pip install -e ".[benchmarks]"

# Or install separately
pip install openml pmlb
```

## 5-Minute Quick Start

### 1. Run Your First Benchmark (30 seconds)

```bash
tabml-benchmark run sklearn-small
```

This runs TabML models on 5 sklearn datasets and saves results.

### 2. Set as Baseline (instant)

```bash
tabml-benchmark baseline set sklearn-small
```

This saves the results as your performance baseline.

### 3. Make Code Changes

Edit TabML code, then validate performance:

```bash
# Run benchmark again
tabml-benchmark run sklearn-small

# Compare with baseline
tabml-benchmark compare sklearn-small
```

If performance degrades >2%, you'll see a warning!

### 4. Check Results

```bash
# View all baselines
tabml-benchmark baseline list

# View available suites
tabml-benchmark list
```

## Python API Quick Start

```python
from tabml.benchmarks import BenchmarkRunner, RegressionTest

# Run benchmark
runner = BenchmarkRunner()
results = runner.run_suite('sklearn-small')
runner.print_summary(results)

# Set baseline
tester = RegressionTest()
tester.save_baseline(results, 'sklearn-small')

# Later: check for regressions
new_results = runner.run_suite('sklearn-small')
passed, comparison = tester.compare_with_baseline(new_results, suite_name='sklearn-small')

if not passed:
    print("⚠️ Performance regression detected!")
    print(tester.generate_report(comparison))
```

## Available Benchmark Suites

| Suite | Time | Datasets | Best For |
|-------|------|----------|----------|
| **sklearn-small** | < 1 min | 5 | Quick validation, CI/CD |
| **openml-cc18** | 1-2 hrs | 72 | Classification validation |
| **openml-ctr23** | 30-60 min | 30 | Regression validation |
| **pmlb-mini** | 5-10 min | 44 | Small data testing |

## Common Use Cases

### Use Case 1: Pre-commit Validation
```bash
# Add to .git/hooks/pre-commit
tabml-benchmark run sklearn-small --max-datasets 1 --n-folds 2
```

### Use Case 2: Pull Request Check
```bash
# In GitHub Actions
tabml-benchmark run sklearn-small
tabml-benchmark compare sklearn-small --fail-on-regression
```

### Use Case 3: Nightly Full Validation
```bash
# Run comprehensive benchmark
tabml-benchmark run openml-cc18 --save-baseline
```

### Use Case 4: Development Testing
```python
from tabml.benchmarks import BenchmarkRunner

runner = BenchmarkRunner()

# Test on subset for speed
results = runner.run_suite(
    'sklearn-small',
    models=['xgboost'],  # Test just one model
    max_datasets=2,      # Just 2 datasets
    n_folds=2,           # Fewer folds
    verbose=True
)

runner.print_summary(results)
```

## Expected Performance

TabML should achieve these minimum scores:

| Dataset | Model | Metric | Expected |
|---------|-------|--------|----------|
| breast_cancer | XGBoost | AUC | ≥ 0.97 |
| wine | XGBoost | Accuracy | ≥ 0.95 |
| diabetes | XGBoost | R² | ≥ 0.40 |

Degradation tolerance: **2%** (adjustable)

## CLI Commands Cheat Sheet

```bash
# List suites
tabml-benchmark list

# Run benchmarks
tabml-benchmark run <suite-name>
tabml-benchmark run sklearn-small --max-datasets 5
tabml-benchmark run openml-cc18 --models xgboost lightgbm

# Baselines
tabml-benchmark baseline list
tabml-benchmark baseline set <suite-name>

# Compare
tabml-benchmark compare <suite-name>
tabml-benchmark compare sklearn-small --tolerance 0.05
tabml-benchmark compare sklearn-small --fail-on-regression
```

## Troubleshooting

**"No baseline found"**
```bash
tabml-benchmark run sklearn-small --save-baseline
```

**"openml not installed"**
```bash
pip install openml
```

**Takes too long**
```bash
# Reduce datasets
tabml-benchmark run sklearn-small --max-datasets 2

# Reduce folds
tabml-benchmark run sklearn-small --n-folds 3
```

## Next Steps

- 📖 [Full Documentation](docs/benchmark_system.md)
- 💻 [Python Examples](examples/benchmark_validation_example.py)
- 🧪 [Test Suite](tests/test_benchmarks.py)
- 📊 [Results Directory](benchmarks/README.md)

## Support

- Report issues: [GitHub Issues](https://github.com/wguesdon/tabml/issues)
- View benchmarks: `benchmarks/results/`
- Logs: Check console output for detailed progress
