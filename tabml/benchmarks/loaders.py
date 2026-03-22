"""Benchmark dataset loaders for OpenML, PMLB, and sklearn datasets."""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Benchmark suite definitions
BENCHMARK_SUITES = {
    'sklearn-small': {
        'type': 'mixed',
        'n_datasets': 5,
        'description': 'Fast sklearn built-in datasets for quick validation',
        'expected_time': '< 1 minute',
        'expected_baselines': {
            'breast_cancer': {'xgboost_auc': 0.97, 'lightgbm_auc': 0.96, 'catboost_auc': 0.97},
            'wine': {'xgboost_accuracy': 0.95, 'lightgbm_accuracy': 0.94, 'catboost_accuracy': 0.95},
            'diabetes': {'xgboost_r2': 0.40, 'lightgbm_r2': 0.38, 'catboost_r2': 0.40},
        }
    },
    'openml-cc18': {
        'suite_id': 99,
        'type': 'classification',
        'n_datasets': 72,
        'description': 'OpenML Curated Classification benchmark (72 datasets)',
        'expected_time': '1-2 hours (full suite)',
        'expected_baselines': {
            'xgboost_auc_avg': 0.80,
            'lightgbm_auc_avg': 0.78,
            'catboost_auc_avg': 0.79,
        }
    },
    'openml-ctr23': {
        'suite_id': 353,  # OpenML-CTR23 curated regression suite (Fischer et al. 2023)
        'type': 'regression',
        'n_datasets': 35,
        'description': 'OpenML Curated Tabular Regression benchmark (35 datasets)',
        'expected_time': '30-60 minutes (full suite)',
        'expected_baselines': {
            'xgboost_rank': 1.31,
            'lightgbm_rank': 2.0,
            'catboost_rank': 1.8,
        }
    },
    'pmlb-mini': {
        'type': 'classification',
        'n_datasets': 44,
        'description': 'Penn ML Benchmark mini suite (n <= 500)',
        'expected_time': '5-10 minutes',
        'expected_baselines': {
            'xgboost_accuracy_avg': 0.75,
            'lightgbm_accuracy_avg': 0.73,
            'catboost_accuracy_avg': 0.74,
        }
    }
}


class BaseLoader:
    """Base class for dataset loaders."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize loader with cache directory.

        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        if cache_dir is None:
            cache_dir = Path.home() / '.tabml' / 'benchmark_cache'
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_dataset(self, dataset_id: Any) -> Dict[str, Any]:
        """Load a single dataset.

        Returns:
            Dictionary containing:
                - name: Dataset name
                - X: Feature DataFrame
                - y: Target Series
                - task_type: 'classification' or 'regression'
                - metadata: Additional dataset information
        """
        raise NotImplementedError

    def load_suite(self, max_datasets: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load multiple datasets from a suite.

        Args:
            max_datasets: Maximum number of datasets to load (for testing)

        Returns:
            List of dataset dictionaries
        """
        raise NotImplementedError


class SklearnLoader(BaseLoader):
    """Load sklearn built-in datasets for quick validation."""

    DATASETS = {
        'classification': [
            'breast_cancer', 'wine', 'iris', 'digits'
        ],
        'regression': [
            'diabetes', 'california_housing'
        ]
    }

    def load_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Load sklearn dataset.

        Args:
            dataset_name: Name of sklearn dataset

        Returns:
            Dataset dictionary
        """
        from sklearn import datasets

        # Classification datasets
        if dataset_name == 'breast_cancer':
            data = datasets.load_breast_cancer()
            task_type = 'classification'
        elif dataset_name == 'wine':
            data = datasets.load_wine()
            task_type = 'classification'
        elif dataset_name == 'iris':
            data = datasets.load_iris()
            task_type = 'classification'
        elif dataset_name == 'digits':
            data = datasets.load_digits()
            task_type = 'classification'
        # Regression datasets
        elif dataset_name == 'diabetes':
            data = datasets.load_diabetes()
            task_type = 'regression'
        elif dataset_name == 'california_housing':
            data = datasets.fetch_california_housing()
            task_type = 'regression'
        else:
            raise ValueError(f"Unknown sklearn dataset: {dataset_name}")

        return {
            'name': dataset_name,
            'X': pd.DataFrame(data.data, columns=data.feature_names),
            'y': pd.Series(data.target, name='target'),
            'task_type': task_type,
            'metadata': {
                'n_samples': data.data.shape[0],
                'n_features': data.data.shape[1],
                'source': 'sklearn',
                'description': data.DESCR[:200] if hasattr(data, 'DESCR') else ''
            }
        }

    def load_suite(self, max_datasets: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load all sklearn benchmark datasets.

        Args:
            max_datasets: Limit number of datasets

        Returns:
            List of dataset dictionaries
        """
        datasets = []
        all_datasets = (self.DATASETS['classification'] +
                       self.DATASETS['regression'])

        if max_datasets:
            all_datasets = all_datasets[:max_datasets]

        for dataset_name in all_datasets:
            try:
                dataset = self.load_dataset(dataset_name)
                datasets.append(dataset)
                logger.info(f"Loaded {dataset_name}: {dataset['metadata']['n_samples']} samples, "
                          f"{dataset['metadata']['n_features']} features")
            except Exception as e:
                logger.warning(f"Failed to load {dataset_name}: {e}")

        return datasets


class OpenMLLoader(BaseLoader):
    """Load OpenML benchmark suites (CC18, CTR23)."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize OpenML loader."""
        super().__init__(cache_dir)
        try:
            import openml
            self.openml = openml
            self.available = True
        except ImportError:
            logger.warning("openml not installed. Install with: pip install openml")
            self.available = False

    def load_dataset(self, task_id: int) -> Dict[str, Any]:
        """Load single OpenML task.

        Args:
            task_id: OpenML task ID

        Returns:
            Dataset dictionary
        """
        if not self.available:
            raise ImportError("openml package required. Install with: pip install openml")

        task = self.openml.tasks.get_task(task_id)
        dataset = task.get_dataset()
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=dataset.default_target_attribute
        )

        # Determine task type from OpenML task type
        # task_type_id: 1 = Supervised Classification
        # task_type_id: 2 = Supervised Regression
        # task_type_id: 3 = Learning Curve
        # task_type_id: 4 = Supervised Data Stream Classification
        # task_type_id: 5 = Clustering
        # task_type_id: 6 = Machine Learning Challenge
        # task_type_id: 7 = Survival Analysis
        # task_type_id: 8 = Subgroup Discovery

        # Convert TaskType enum to int for comparison
        # OpenML returns a TaskType enum, extract the numeric value
        if hasattr(task.task_type_id, 'value'):
            task_type_value = task.task_type_id.value
        else:
            task_type_value = task.task_type_id

        if task_type_value in [1, 4]:  # Classification tasks
            task_type = 'classification'
        elif task_type_value == 2:  # Regression tasks
            task_type = 'regression'
        else:
            # Fallback: infer from target
            n_unique = len(np.unique(y))
            task_type = 'classification' if n_unique < 100 else 'regression'
            logger.warning(f"Unknown task_type_id {task.task_type_id} for task {task_id}, "
                         f"inferred as {task_type} (n_unique={n_unique})")

        logger.debug(f"Task {task_id} ({dataset.name}): task_type_id={task.task_type_id}, "
                    f"detected as {task_type}, n_classes={len(np.unique(y))}")

        return {
            'name': dataset.name,
            'X': pd.DataFrame(X, columns=attribute_names),
            'y': pd.Series(y, name='target'),
            'task_type': task_type,
            'metadata': {
                'task_id': task_id,
                'task_type_id': str(task.task_type_id),  # Convert enum to string for JSON
                'dataset_id': dataset.dataset_id,
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'n_classes': len(np.unique(y)) if task_type == 'classification' else None,
                'source': 'openml',
                'url': f"https://www.openml.org/t/{task_id}"
            }
        }

    def load_suite(self, suite_id: int = 99,
                   max_datasets: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load OpenML benchmark suite.

        Args:
            suite_id: OpenML suite ID (99=CC18, 271=CTR23)
            max_datasets: Limit number of datasets for testing

        Returns:
            List of dataset dictionaries
        """
        if not self.available:
            raise ImportError("openml package required. Install with: pip install openml")

        logger.info(f"Loading OpenML suite {suite_id}...")
        suite = self.openml.study.get_suite(suite_id)
        tasks = suite.tasks

        if max_datasets:
            tasks = tasks[:max_datasets]
            logger.info(f"Limited to first {max_datasets} datasets")

        datasets = []
        for i, task_id in enumerate(tasks, 1):
            try:
                logger.info(f"Loading task {i}/{len(tasks)}: {task_id}")
                dataset = self.load_dataset(task_id)
                datasets.append(dataset)
            except Exception as e:
                logger.warning(f"Failed to load task {task_id}: {e}")
                continue

        logger.info(f"Successfully loaded {len(datasets)}/{len(tasks)} datasets")
        return datasets


class PMLBLoader(BaseLoader):
    """Load Penn Machine Learning Benchmarks (PMLB) datasets."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize PMLB loader."""
        super().__init__(cache_dir)
        try:
            import pmlb
            self.pmlb = pmlb
            self.available = True
        except ImportError:
            logger.warning("pmlb not installed. Install with: pip install pmlb")
            self.available = False

    def load_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Load single PMLB dataset.

        Args:
            dataset_name: PMLB dataset name

        Returns:
            Dataset dictionary
        """
        if not self.available:
            raise ImportError("pmlb package required. Install with: pip install pmlb")

        # Fetch dataset
        X, y = self.pmlb.fetch_data(dataset_name, return_X_y=True, local_cache_dir=str(self.cache_dir))

        # Get dataset info
        dataset_info = self.pmlb.dataset_names
        task_type = 'classification'  # PMLB-mini is all classification

        return {
            'name': dataset_name,
            'X': pd.DataFrame(X),
            'y': pd.Series(y, name='target'),
            'task_type': task_type,
            'metadata': {
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'source': 'pmlb',
                'url': f"https://epistasislab.github.io/pmlb/"
            }
        }

    def load_suite(self, suite_type: str = 'small',
                   max_datasets: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load PMLB benchmark suite.

        Args:
            suite_type: 'small' (n<=500) or 'all'
            max_datasets: Limit number of datasets

        Returns:
            List of dataset dictionaries
        """
        if not self.available:
            raise ImportError("pmlb package required. Install with: pip install pmlb")

        # Get dataset names
        if suite_type == 'small':
            # Get small datasets (n <= 500)
            dataset_names = [name for name in self.pmlb.classification_dataset_names
                           if self._is_small_dataset(name)]
        else:
            dataset_names = self.pmlb.classification_dataset_names

        if max_datasets:
            dataset_names = dataset_names[:max_datasets]

        datasets = []
        for i, dataset_name in enumerate(dataset_names, 1):
            try:
                logger.info(f"Loading PMLB dataset {i}/{len(dataset_names)}: {dataset_name}")
                dataset = self.load_dataset(dataset_name)
                datasets.append(dataset)
            except Exception as e:
                logger.warning(f"Failed to load {dataset_name}: {e}")
                continue

        logger.info(f"Successfully loaded {len(datasets)}/{len(dataset_names)} datasets")
        return datasets

    def _is_small_dataset(self, dataset_name: str) -> bool:
        """Check if dataset has <= 500 samples."""
        try:
            X, y = self.pmlb.fetch_data(dataset_name, return_X_y=True,
                                       local_cache_dir=str(self.cache_dir))
            return X.shape[0] <= 500
        except:
            return False
