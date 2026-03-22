# TabML Benchmark Results

This directory stores benchmark results for regression testing and performance validation.

## Directory Structure

```
benchmarks/
├── results/                      # Benchmark results storage
│   ├── sklearn-small_baseline.json
│   ├── sklearn-small_latest.json
│   ├── sklearn-small_20250101_120000.json
│   ├── openml-cc18_baseline.json
│   └── ...
└── README.md                     # This file
```

## Quick Start

### Run Benchmarks

```bash
# Quick sklearn benchmark (< 1 minute)
tabml-benchmark run sklearn-small

# OpenML-CC18 (first 10 datasets, ~10 minutes)
tabml-benchmark run openml-cc18 --max-datasets 10

# Full OpenML-CC18 (1-2 hours)
tabml-benchmark run openml-cc18
```

### Set Baseline

```bash
# Run and save as baseline
tabml-benchmark run sklearn-small --save-baseline

# Or set from existing results
tabml-benchmark baseline set sklearn-small
```

### Check for Regressions

```bash
# Compare latest results with baseline
tabml-benchmark compare sklearn-small

# Fail CI/CD if regression detected
tabml-benchmark compare sklearn-small --fail-on-regression
```

## Available Suites

| Suite | Type | Datasets | Time | Use Case |
|-------|------|----------|------|----------|
| `sklearn-small` | Mixed | 5 | < 1 min | Quick CI/CD validation |
| `openml-cc18` | Classification | 72 | 1-2 hours | Comprehensive classification |
| `openml-ctr23` | Regression | 30 | 30-60 min | Comprehensive regression |
| `pmlb-mini` | Classification | 44 | 5-10 min | Small data regime |

List all suites: `tabml-benchmark list`

## Result Format

Results are stored in JSON format with the following structure:

```json
{
  "suite_name": "sklearn-small",
  "version": "0.5.1",
  "timestamp": "2025-01-01T12:00:00",
  "datasets": {
    "breast_cancer": {
      "models": {
        "xgboost": {
          "metrics": {
            "roc_auc": {"mean": 0.9850, "std": 0.0123}
          }
        }
      }
    }
  },
  "aggregate": {...}
}
```

## Baseline Management

### Create Baseline
Run a benchmark and save as baseline for future comparisons:
```bash
tabml-benchmark run sklearn-small --save-baseline
```

### Update Baseline
When performance improvements are made:
```bash
tabml-benchmark baseline set sklearn-small
```

### List Baselines
View all current baselines:
```bash
tabml-benchmark baseline list
```

## Performance Thresholds

Default expected performance (used for validation):

**Classification**
- breast_cancer: XGBoost AUC ≥ 0.97
- wine: XGBoost Accuracy ≥ 0.95

**Regression**
- diabetes: XGBoost R² ≥ 0.40

**OpenML Averages**
- OpenML-CC18: XGBoost AUC ≥ 0.75 (average)
- OpenML-CTR23: XGBoost Rank ≤ 2.0

## CI/CD Integration

### GitHub Actions Example

See `.github/workflows/benchmark.yml` for complete workflow.

Quick example:
```yaml
- name: Run benchmarks
  run: tabml-benchmark run sklearn-small --max-datasets 2

- name: Check for regressions
  run: tabml-benchmark compare sklearn-small --fail-on-regression
```

### Pre-commit Hook

```bash
#!/bin/bash
tabml-benchmark run sklearn-small --max-datasets 1 --n-folds 2
```

## Troubleshooting

### No baseline found
Create a baseline first:
```bash
tabml-benchmark run sklearn-small --save-baseline
```

### OpenML/PMLB not available
Install benchmark dependencies:
```bash
pip install -e ".[benchmarks]"
# or
pip install openml pmlb
```

### Memory issues
Reduce dataset count or folds:
```bash
tabml-benchmark run openml-cc18 --max-datasets 5 --n-folds 3
```

## Documentation

- [Full Documentation](../docs/benchmark_system.md)
- [Python API Examples](../examples/benchmark_validation_example.py)
- [Test Suite](../tests/test_benchmarks.py)

## References

- **OpenML-CC18**: [OpenML Suite 99](https://www.openml.org/s/99)
- **OpenML-CTR23**: Grinsztajn et al. (2022)
- **PMLB**: [Penn ML Benchmarks](https://epistasislab.github.io/pmlb/)
