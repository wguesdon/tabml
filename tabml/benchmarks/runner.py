"""Benchmark runner for evaluating TabML models."""

from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import json
import time
import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from loguru import logger
from tqdm import tqdm

from .loaders import (
    SklearnLoader, OpenMLLoader, PMLBLoader,
    BENCHMARK_SUITES
)
from .metrics import BenchmarkMetrics
from ..models import XGBoostModel, LightGBMModel, CatBoostModel


class BenchmarkRunner:
    """Run benchmarks and save results for regression testing.

    Example:
        >>> runner = BenchmarkRunner()
        >>> results = runner.run_suite('sklearn-small')
        >>> runner.save_results(results, 'sklearn-small')
        >>> runner.compare_with_baseline('sklearn-small')
    """

    def __init__(self,
                 output_dir: Optional[Path] = None,
                 cache_dir: Optional[Path] = None,
                 random_state: int = 42):
        """Initialize benchmark runner.

        Args:
            output_dir: Directory to save results
            cache_dir: Directory to cache datasets
            random_state: Random seed for reproducibility
        """
        if output_dir is None:
            output_dir = Path.cwd() / 'benchmarks' / 'results'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cache_dir = cache_dir
        self.random_state = random_state

        # Initialize loaders
        self.sklearn_loader = SklearnLoader(cache_dir=cache_dir)
        self.openml_loader = OpenMLLoader(cache_dir=cache_dir)
        self.pmlb_loader = PMLBLoader(cache_dir=cache_dir)

        # Metrics calculator
        self.metrics = BenchmarkMetrics()

    def run_suite(self,
                  suite_name: str,
                  models: Optional[List[str]] = None,
                  max_datasets: Optional[int] = None,
                  n_folds: int = 5,
                  verbose: bool = True) -> Dict[str, Any]:
        """Run benchmark suite on specified models.

        Args:
            suite_name: Name of benchmark suite (see BENCHMARK_SUITES)
            models: List of model names ('xgboost', 'lightgbm', 'catboost')
            max_datasets: Limit number of datasets (for testing)
            n_folds: Number of cross-validation folds
            verbose: Show progress bars

        Returns:
            Dictionary containing results for each dataset
        """
        if models is None:
            models = ['xgboost', 'lightgbm', 'catboost']

        if suite_name not in BENCHMARK_SUITES:
            raise ValueError(f"Unknown suite: {suite_name}. "
                           f"Available: {list(BENCHMARK_SUITES.keys())}")

        suite_info = BENCHMARK_SUITES[suite_name]
        logger.info(f"Running benchmark suite: {suite_name}")
        logger.info(f"Description: {suite_info['description']}")
        logger.info(f"Expected time: {suite_info['expected_time']}")

        # Load datasets
        datasets = self._load_suite_datasets(suite_name, max_datasets)

        if not datasets:
            raise RuntimeError(f"No datasets loaded for suite: {suite_name}")

        logger.info(f"Loaded {len(datasets)} datasets")

        # Run benchmarks
        results = {
            'suite_name': suite_name,
            'suite_info': suite_info,
            'timestamp': datetime.datetime.now().isoformat(),
            'version': self._get_version(),
            'n_folds': n_folds,
            'models': models,
            'datasets': {}
        }

        for dataset in tqdm(datasets, desc="Benchmarking datasets", disable=not verbose):
            dataset_name = dataset['name']
            logger.info(f"\nBenchmarking: {dataset_name}")

            try:
                dataset_results = self._benchmark_dataset(
                    dataset=dataset,
                    models=models,
                    n_folds=n_folds,
                    verbose=verbose
                )
                results['datasets'][dataset_name] = dataset_results

            except Exception as e:
                logger.error(f"Failed to benchmark {dataset_name}: {e}")
                results['datasets'][dataset_name] = {
                    'error': str(e),
                    'status': 'failed'
                }

        # Compute aggregate statistics
        results['aggregate'] = self._compute_aggregates(results['datasets'])

        return results

    def _load_suite_datasets(self, suite_name: str,
                            max_datasets: Optional[int]) -> List[Dict]:
        """Load datasets for a benchmark suite."""
        if suite_name == 'sklearn-small':
            return self.sklearn_loader.load_suite(max_datasets)

        elif suite_name == 'openml-cc18':
            if not self.openml_loader.available:
                raise ImportError("openml required for OpenML-CC18. "
                                "Install with: pip install openml")
            suite_id = BENCHMARK_SUITES[suite_name]['suite_id']
            return self.openml_loader.load_suite(suite_id, max_datasets)

        elif suite_name == 'openml-ctr23':
            if not self.openml_loader.available:
                raise ImportError("openml required for OpenML-CTR23. "
                                "Install with: pip install openml")
            suite_id = BENCHMARK_SUITES[suite_name]['suite_id']
            return self.openml_loader.load_suite(suite_id, max_datasets)

        elif suite_name == 'pmlb-mini':
            if not self.pmlb_loader.available:
                raise ImportError("pmlb required for PMLB-mini. "
                                "Install with: pip install pmlb")
            return self.pmlb_loader.load_suite('small', max_datasets)

        else:
            raise ValueError(f"Unknown suite: {suite_name}")

    def _benchmark_dataset(self,
                          dataset: Dict,
                          models: List[str],
                          n_folds: int,
                          verbose: bool = True) -> Dict[str, Any]:
        """Benchmark a single dataset with multiple models.

        Args:
            dataset: Dataset dictionary from loader
            models: List of model names
            n_folds: Number of CV folds
            verbose: Show progress

        Returns:
            Results dictionary
        """
        X = dataset['X']
        y = dataset['y']
        task_type = dataset['task_type']

        # Handle missing values and encode categoricals (simple imputation)
        X, y = self._simple_preprocessing(X, y)

        # Setup cross-validation
        if task_type == 'classification':
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                random_state=self.random_state)
        else:
            cv = KFold(n_splits=n_folds, shuffle=True,
                      random_state=self.random_state)

        results = {
            'metadata': dataset['metadata'],
            'task_type': task_type,
            'n_samples': len(y),
            'n_features': X.shape[1],
            'models': {}
        }

        # Benchmark each model
        for model_name in models:
            if verbose:
                logger.info(f"  Testing {model_name}...")

            try:
                model_results = self._cross_validate_model(
                    model_name=model_name,
                    X=X,
                    y=y,
                    cv=cv,
                    task_type=task_type
                )
                results['models'][model_name] = model_results

            except Exception as e:
                logger.warning(f"  {model_name} failed: {e}")
                results['models'][model_name] = {
                    'error': str(e),
                    'status': 'failed'
                }

        return results

    def _cross_validate_model(self,
                             model_name: str,
                             X: pd.DataFrame,
                             y: pd.Series,
                             cv: Any,
                             task_type: str) -> Dict[str, Any]:
        """Cross-validate a single model.

        Args:
            model_name: Name of model
            X: Features
            y: Target
            cv: Cross-validation splitter
            task_type: 'classification' or 'regression'

        Returns:
            Model results dictionary
        """
        fold_results = []
        start_time = time.time()

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Create model
            model = self._create_model(model_name, task_type)

            # Train
            model.fit(X_train, y_train)

            # Predict
            if task_type == 'classification':
                # Get probabilities first
                try:
                    y_proba = model.predict_proba(X_val)
                    # Get class predictions (not probabilities)
                    y_pred = np.argmax(y_proba, axis=1) if y_proba.ndim > 1 else (y_proba > 0.5).astype(int)
                except:
                    y_pred = model.predict(X_val)
                    y_proba = None

                # Ensure predictions are integers
                y_pred = y_pred.astype(int)
            else:
                # Regression
                y_pred = model.predict(X_val)
                y_proba = None

            # Compute metrics
            if task_type == 'classification':
                metrics = self.metrics.compute_classification_metrics(
                    y_val.values.astype(int), y_pred, y_proba
                )
            else:
                metrics = self.metrics.compute_regression_metrics(
                    y_val.values, y_pred
                )

            fold_results.append(metrics)

        elapsed_time = time.time() - start_time

        # Aggregate fold results
        aggregated = {}
        for metric_name in fold_results[0].keys():
            values = [fold[metric_name] for fold in fold_results]
            values = np.array(values)
            values = values[~np.isnan(values)]  # Remove NaN

            aggregated[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': values.tolist()
            }

        return {
            'metrics': aggregated,
            'time_seconds': elapsed_time,
            'status': 'success'
        }

    def _create_model(self, model_name: str, task_type: str) -> Any:
        """Create a model instance.

        Args:
            model_name: Model name
            task_type: Task type

        Returns:
            Model instance
        """
        # Basic parameters for fair comparison
        if model_name == 'xgboost':
            return XGBoostModel(params={
                'random_state': self.random_state,
                'n_estimators': 100,
                'verbose': False
            })
        elif model_name == 'lightgbm':
            return LightGBMModel(params={
                'random_state': self.random_state,
                'n_estimators': 100,
                'verbose': False
            })
        elif model_name == 'catboost':
            # CatBoost uses random_seed, not random_state
            return CatBoostModel(params={
                'random_seed': self.random_state,
                'iterations': 100,
                'verbose': False
            })
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _simple_preprocessing(self, X: pd.DataFrame, y: pd.Series = None) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        """Simple preprocessing for benchmarking.

        Args:
            X: Input features
            y: Target (optional, will be encoded if provided)

        Returns:
            Preprocessed features (and target if provided)
        """
        X = X.copy()

        if y is not None:
            y = y.copy()
            # Encode target if categorical
            if y.dtype == 'object' or y.dtype.name == 'category':
                y = pd.Series(pd.Categorical(y).codes, index=y.index, name=y.name)

        # Handle categorical columns
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                X[col] = pd.Categorical(X[col]).codes

        # Fill missing values
        X = X.fillna(X.median(numeric_only=True))
        X = X.fillna(-999)  # For any remaining NaNs

        if y is not None:
            return X, y
        return X

    def _compute_aggregates(self, datasets_results: Dict) -> Dict[str, Any]:
        """Compute aggregate statistics across datasets.

        Args:
            datasets_results: Results for all datasets

        Returns:
            Aggregate statistics
        """
        # Collect metrics per model
        model_metrics = {}

        for dataset_name, dataset_result in datasets_results.items():
            if 'models' not in dataset_result:
                continue

            for model_name, model_result in dataset_result['models'].items():
                if 'metrics' not in model_result:
                    continue

                if model_name not in model_metrics:
                    model_metrics[model_name] = {}

                for metric_name, metric_data in model_result['metrics'].items():
                    if metric_name not in model_metrics[model_name]:
                        model_metrics[model_name][metric_name] = []
                    model_metrics[model_name][metric_name].append(
                        metric_data['mean']
                    )

        # Compute statistics
        aggregated = {}
        for model_name, metrics in model_metrics.items():
            aggregated[model_name] = {}
            for metric_name, values in metrics.items():
                values = np.array(values)
                values = values[~np.isnan(values)]

                if len(values) > 0:
                    aggregated[model_name][metric_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'median': float(np.median(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values))
                    }

        return aggregated

    def save_results(self, results: Dict, suite_name: str):
        """Save benchmark results to file.

        Args:
            results: Results dictionary
            suite_name: Name of suite
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{suite_name}_{timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to: {filepath}")

        # Also save as "latest"
        latest_path = self.output_dir / f"{suite_name}_latest.json"
        with open(latest_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Latest results: {latest_path}")

    def load_results(self, suite_name: str, version: str = 'latest') -> Dict:
        """Load saved benchmark results.

        Args:
            suite_name: Name of suite
            version: 'latest' or specific timestamp

        Returns:
            Results dictionary
        """
        if version == 'latest':
            filepath = self.output_dir / f"{suite_name}_latest.json"
        else:
            filepath = self.output_dir / f"{suite_name}_{version}.json"

        if not filepath.exists():
            raise FileNotFoundError(f"Results not found: {filepath}")

        with open(filepath) as f:
            return json.load(f)

    def _get_version(self) -> str:
        """Get TabML version."""
        try:
            import tabml
            return tabml.__version__
        except:
            return "unknown"

    def print_summary(self, results: Dict):
        """Print summary of benchmark results.

        Args:
            results: Results dictionary
        """
        print("\n" + "="*70)
        print(f"Benchmark Summary: {results['suite_name']}")
        print("="*70)

        if 'aggregate' in results:
            print("\nAggregate Results:")
            for model_name, metrics in results['aggregate'].items():
                print(f"\n{model_name}:")
                for metric_name, stats in metrics.items():
                    print(f"  {metric_name:15s}: {stats['mean']:.4f} "
                          f"± {stats['std']:.4f} "
                          f"(median: {stats['median']:.4f})")

        print("\n" + "="*70)
