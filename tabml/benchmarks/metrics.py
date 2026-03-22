"""Benchmark metrics computation and aggregation."""

from typing import Dict, List, Optional, Callable
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, log_loss,
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
from sklearn.preprocessing import label_binarize
from loguru import logger


class BenchmarkMetrics:
    """Compute and aggregate benchmark metrics."""

    # Metric definitions
    CLASSIFICATION_METRICS = {
        'accuracy': accuracy_score,
        'roc_auc': 'custom',  # Handled separately for binary/multiclass
        'f1_macro': lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'),
        'log_loss': 'custom',  # Requires probabilities
    }

    REGRESSION_METRICS = {
        'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error,
        'r2': r2_score,
        'mape': lambda y_true, y_pred: mean_absolute_percentage_error(y_true, y_pred) * 100,
    }

    @staticmethod
    def compute_classification_metrics(y_true: np.ndarray,
                                       y_pred: np.ndarray,
                                       y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)

        Returns:
            Dictionary of metric scores
        """
        metrics = {}

        # Accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)

        # F1 scores
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # ROC AUC (requires probabilities)
        if y_proba is not None:
            n_classes = len(np.unique(y_true))

            try:
                if n_classes == 2:
                    # Binary classification
                    if y_proba.ndim == 2:
                        metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                    else:
                        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
                else:
                    # Multiclass classification
                    metrics['roc_auc'] = roc_auc_score(
                        y_true, y_proba,
                        multi_class='ovr',
                        average='macro'
                    )
            except Exception as e:
                logger.warning(f"Could not compute ROC AUC: {e}")
                metrics['roc_auc'] = np.nan

            # Log loss
            try:
                metrics['log_loss'] = log_loss(y_true, y_proba)
            except Exception as e:
                logger.warning(f"Could not compute log loss: {e}")
                metrics['log_loss'] = np.nan

        return metrics

    @staticmethod
    def compute_regression_metrics(y_true: np.ndarray,
                                   y_pred: np.ndarray) -> Dict[str, float]:
        """Compute regression metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of metric scores
        """
        metrics = {}

        # RMSE
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))

        # MAE
        metrics['mae'] = mean_absolute_error(y_true, y_pred)

        # R²
        metrics['r2'] = r2_score(y_true, y_pred)

        # MAPE (only if no zeros in y_true)
        try:
            if not np.any(y_true == 0):
                metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred) * 100
            else:
                metrics['mape'] = np.nan
        except:
            metrics['mape'] = np.nan

        return metrics

    @staticmethod
    def aggregate_results(results: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Aggregate results across multiple datasets.

        Args:
            results: List of result dictionaries

        Returns:
            Aggregated statistics (mean, std, median, min, max)
        """
        # Collect all metrics
        all_metrics = {}
        for result in results:
            for model_name, model_results in result.items():
                if model_name not in all_metrics:
                    all_metrics[model_name] = {}

                for metric_name, value in model_results.items():
                    if metric_name not in all_metrics[model_name]:
                        all_metrics[model_name][metric_name] = []
                    all_metrics[model_name][metric_name].append(value)

        # Compute statistics
        aggregated = {}
        for model_name, metrics in all_metrics.items():
            aggregated[model_name] = {}
            for metric_name, values in metrics.items():
                values = np.array(values)
                # Remove NaN values
                values = values[~np.isnan(values)]

                if len(values) > 0:
                    aggregated[model_name][metric_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'median': float(np.median(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'n_datasets': len(values)
                    }

        return aggregated

    @staticmethod
    def compute_rankings(results: List[Dict], metric: str = 'accuracy') -> Dict[str, float]:
        """Compute average model rankings across datasets.

        Args:
            results: List of result dictionaries
            metric: Metric to rank by

        Returns:
            Dictionary of average ranks per model
        """
        ranks = {}

        for result in results:
            # Get scores for this dataset
            scores = {}
            for model_name, model_results in result.items():
                if metric in model_results:
                    scores[model_name] = model_results[metric]

            if not scores:
                continue

            # Rank models (higher is better for most metrics)
            sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            # Assign ranks
            for rank, (model_name, score) in enumerate(sorted_models, 1):
                if model_name not in ranks:
                    ranks[model_name] = []
                ranks[model_name].append(rank)

        # Compute average ranks
        avg_ranks = {model: np.mean(rank_list) for model, rank_list in ranks.items()}

        return avg_ranks

    @staticmethod
    def format_results_table(aggregated: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Format aggregated results as a DataFrame table.

        Args:
            aggregated: Aggregated results from aggregate_results()

        Returns:
            Formatted DataFrame
        """
        rows = []
        for model_name, metrics in aggregated.items():
            for metric_name, stats in metrics.items():
                rows.append({
                    'Model': model_name,
                    'Metric': metric_name,
                    'Mean': f"{stats['mean']:.4f}",
                    'Std': f"{stats['std']:.4f}",
                    'Median': f"{stats['median']:.4f}",
                    'Min': f"{stats['min']:.4f}",
                    'Max': f"{stats['max']:.4f}",
                    'N': stats['n_datasets']
                })

        return pd.DataFrame(rows)

    @staticmethod
    def get_primary_metric(task_type: str, n_classes: Optional[int] = None) -> str:
        """Get primary metric for task type.

        Args:
            task_type: 'classification' or 'regression'
            n_classes: Number of classes (for classification)

        Returns:
            Primary metric name
        """
        if task_type == 'classification':
            if n_classes == 2:
                return 'roc_auc'
            else:
                return 'accuracy'
        else:  # regression
            return 'rmse'
