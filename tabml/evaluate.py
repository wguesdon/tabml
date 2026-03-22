"""Model evaluation and cross-validation utilities.

This module provides comprehensive tools for evaluating machine learning models
including cross-validation, time series validation, and metric calculation.

Classes:
    CrossValidator: Main class for model evaluation and comparison
    
Example:
    Basic cross-validation::
    
        from tabml.evaluate import CrossValidator
        
        # Initialize validator
        validator = CrossValidator(random_state=42)
        
        # Evaluate single model
        results = validator.evaluate_model(model, X, y, cv_folds=5)
        print(f"Mean score: {results['mean']:.4f} (+/- {results['std']:.4f})")
        
        # Compare multiple models
        models = {'xgboost': xgb_model, 'lightgbm': lgb_model}
        comparison = validator.compare_models(models, X, y)
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
)
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class CrossValidator:
    """Cross-validation and model evaluation utilities.
    
    Provides methods for evaluating models using various cross-validation
    strategies including standard k-fold, stratified k-fold, time series
    split, and walk-forward validation.
    
    Attributes:
        random_state: Random seed for reproducibility
        cv_results: Dictionary storing cross-validation results
        
    Example:
        >>> # Standard cross-validation
        >>> validator = CrossValidator(random_state=42)
        >>> results = validator.evaluate_model(model, X, y, cv_folds=5)
        >>> 
        >>> # Time series validation
        >>> ts_results = validator.walk_forward_validation(
        ...     model, X, y, 
        ...     initial_train_size=0.7,
        ...     expanding_window=True
        ... )
        >>> 
        >>> # Compare multiple models
        >>> comparison_df = validator.compare_models(models_dict, X, y)
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize cross validator.
        
        Args:
            random_state: Random seed for reproducibility in CV splits.
                Ensures consistent results across runs.
        """
        self.random_state = random_state
        self.cv_results = {}
        
    def evaluate_model(self,
                      model: Any,
                      X: pd.DataFrame,
                      y: pd.Series,
                      cv_folds: int = 5,
                      scoring: Optional[str] = None) -> Dict[str, float]:
        """Evaluate model using cross-validation.
        
        Performs k-fold cross-validation with automatic task type detection
        and appropriate scoring metrics.
        
        Args:
            model: Model object with fit and predict methods. Must have
                a .model attribute containing the sklearn-compatible model.
            X: Feature matrix as pandas DataFrame
            y: Target variable as pandas Series
            cv_folds: Number of cross-validation folds. Uses StratifiedKFold
                for classification and KFold for regression.
            scoring: Scoring metric name. If None, auto-selects:
                - Binary classification: 'roc_auc'
                - Multiclass: 'accuracy'
                - Regression: 'neg_mean_squared_error'
            
        Returns:
            Dictionary containing:
                - 'mean': Mean score across folds
                - 'std': Standard deviation of scores
                - 'scores': List of individual fold scores
                - 'metric': Name of metric used
                
        Example:
            >>> validator = CrossValidator()
            >>> results = validator.evaluate_model(
            ...     xgb_model, X_train, y_train, 
            ...     cv_folds=10, scoring='f1'
            ... )
            >>> print(f"F1 Score: {results['mean']:.3f} ± {results['std']:.3f}")
            
        Note:
            Task type (classification/regression) is determined by number
            of unique values in y (< 100 considered classification).
        """
        # Determine task type and scoring metric
        is_classification = y.nunique() < 100
        
        if scoring is None:
            if is_classification:
                if y.nunique() == 2:
                    scoring = 'roc_auc'
                else:
                    scoring = 'accuracy'
            else:
                scoring = 'neg_mean_squared_error'
                
        # Setup cross-validation
        if is_classification:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            
        # Perform cross-validation
        scores = cross_val_score(model.model, X, y, cv=cv, scoring=scoring)
        
        # Convert negative scores for regression metrics
        if scoring.startswith('neg_'):
            scores = -scores
            
        results = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores.tolist(),
            'metric': scoring
        }
        
        return results
        
    def evaluate_predictions(self,
                           y_true: pd.Series,
                           y_pred: np.ndarray,
                           y_proba: Optional[np.ndarray] = None,
                           task_type: Optional[str] = None) -> Dict[str, float]:
        """Evaluate predictions with multiple metrics.
        
        Calculates comprehensive set of metrics appropriate for the task type.
        
        Args:
            y_true: Ground truth target values
            y_pred: Model predictions (class labels for classification)
            y_proba: Prediction probabilities for positive class (binary)
                or all classes (multiclass). Required for ROC-AUC.
            task_type: Either 'classification' or 'regression'. If None,
                auto-detected based on unique values in y_true.
            
        Returns:
            Dictionary of metrics:
                - Classification: accuracy, precision, recall, f1, roc_auc
                - Regression: mse, rmse, mae, r2, mape (if no zeros)
                
        Example:
            >>> # Classification with probabilities
            >>> y_proba = model.predict_proba(X_test)[:, 1]
            >>> y_pred = model.predict(X_test)
            >>> metrics = validator.evaluate_predictions(
            ...     y_true=y_test, 
            ...     y_pred=y_pred,
            ...     y_proba=y_proba
            ... )
            >>> 
            >>> # Regression
            >>> y_pred = model.predict(X_test)
            >>> metrics = validator.evaluate_predictions(y_test, y_pred)
            >>> print(f"RMSE: {metrics['rmse']:.4f}")
            
        Note:
            - Multiclass metrics use weighted averaging
            - MAPE is only calculated if y_true contains no zeros
        """
        # Auto-detect task type
        if task_type is None:
            task_type = 'classification' if y_true.nunique() < 100 else 'regression'
            
        metrics = {}
        
        if task_type == 'classification':
            # Classification metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            
            if y_true.nunique() == 2:
                # Binary classification
                metrics['precision'] = precision_score(y_true, y_pred)
                metrics['recall'] = recall_score(y_true, y_pred)
                metrics['f1'] = f1_score(y_true, y_pred)
                
                if y_proba is not None:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            else:
                # Multiclass
                metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
                metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
                metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
                
        else:
            # Regression metrics
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
            
            # MAPE only if no zeros in y_true
            if not (y_true == 0).any():
                metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
                
        return metrics
        
    def compare_models(self,
                      models: Dict[str, Any],
                      X: pd.DataFrame,
                      y: pd.Series,
                      cv_folds: int = 5) -> pd.DataFrame:
        """Compare multiple models using cross-validation.
        
        Evaluates all provided models using the same CV splits and returns
        a sorted comparison DataFrame.
        
        Args:
            models: Dictionary mapping model names to model objects.
                Each model must have fit/predict methods.
            X: Feature matrix for evaluation
            y: Target variable
            cv_folds: Number of cross-validation folds
            
        Returns:
            DataFrame with columns:
                - 'model': Model name
                - 'mean_score': Mean CV score
                - 'std_score': Standard deviation of CV scores
                - 'metric': Metric used for evaluation
            Sorted by mean_score in descending order.
            
        Example:
            >>> models = {
            ...     'XGBoost': xgb_model,
            ...     'LightGBM': lgb_model,
            ...     'CatBoost': cb_model,
            ...     'Random Forest': rf_model
            ... }
            >>> comparison = validator.compare_models(models, X, y, cv_folds=5)
            >>> print(comparison)
            >>> # Shows models ranked by performance
            
        Note:
            All models are evaluated using the same metric (auto-detected
            based on task type) for fair comparison.
        """
        results = []
        
        for name, model in models.items():
            logger.info(f"Evaluating {name}...")
            scores = self.evaluate_model(model, X, y, cv_folds)
            
            results.append({
                'model': name,
                'mean_score': scores['mean'],
                'std_score': scores['std'],
                'metric': scores['metric']
            })
            
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values('mean_score', ascending=False)
        
        return comparison_df
        
    def plot_cv_scores(self,
                      models: Dict[str, Any],
                      X: pd.DataFrame,
                      y: pd.Series,
                      cv_folds: int = 5) -> None:
        """Plot cross-validation scores for multiple models.
        
        Creates a box plot showing the distribution of CV scores for each model,
        allowing visual comparison of performance and variance.
        
        Args:
            models: Dictionary mapping model names to model objects
            X: Feature matrix
            y: Target variable
            cv_folds: Number of cross-validation folds
            
        Example:
            >>> models = {'XGBoost': xgb, 'LightGBM': lgb, 'CatBoost': cb}
            >>> validator.plot_cv_scores(models, X_train, y_train)
            >>> # Displays box plot of CV scores
            
        Note:
            Requires matplotlib and seaborn. Logs warning if not available.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Collect CV scores for each model
            all_scores = []
            for name, model in models.items():
                scores = self.evaluate_model(model, X, y, cv_folds)
                for score in scores['scores']:
                    all_scores.append({
                        'model': name,
                        'score': score
                    })
                    
            scores_df = pd.DataFrame(all_scores)
            
            # Create plot
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=scores_df, x='model', y='score')
            plt.title('Cross-Validation Scores by Model')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available for plotting")
            
    def get_cv_predictions(self,
                          model: Any,
                          X: pd.DataFrame,
                          y: pd.Series,
                          cv_folds: int = 5) -> np.ndarray:
        """Get out-of-fold predictions using cross-validation.
        
        Generates predictions for each sample using a model trained on folds
        that don't contain that sample. Useful for stacking and model evaluation.
        
        Args:
            model: Model object with fit and predict methods
            X: Feature matrix
            y: Target variable
            cv_folds: Number of cross-validation folds
            
        Returns:
            Array of out-of-fold predictions with same length as y.
            For binary classification, returns probabilities for positive class.
            
        Example:
            >>> # Get OOF predictions for stacking
            >>> oof_preds = validator.get_cv_predictions(
            ...     xgb_model, X_train, y_train, cv_folds=5
            ... )
            >>> 
            >>> # Use as meta-features
            >>> meta_features = pd.DataFrame({
            ...     'xgb_oof': oof_preds,
            ...     'actual': y_train
            ... })
            
        Note:
            - Each prediction is made by a model that wasn't trained on that sample
            - For binary classification, returns probabilities not class labels
        """
        is_classification = y.nunique() < 100
        
        if is_classification:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            
        predictions = np.zeros(len(y))
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Clone and train model
            model_clone = model.__class__(model.params)
            model_clone.fit(X_train, y_train)
            
            # Make predictions
            if is_classification and y.nunique() == 2:
                predictions[val_idx] = model_clone.predict_proba(X_val)[:, 1]
            else:
                predictions[val_idx] = model_clone.predict(X_val)
                
        return predictions
    
    def walk_forward_validation(self,
                               model: Any,
                               X: pd.DataFrame,
                               y: pd.Series,
                               initial_train_size: Union[int, float] = 0.7,
                               step_size: int = 1,
                               forecast_horizon: int = 1,
                               expanding_window: bool = True,
                               metric: Optional[str] = None) -> Dict[str, Any]:
        """Perform walk-forward validation for time series data.
        
        Simulates real-world time series forecasting by training on historical
        data and testing on future periods, walking forward through time.
        
        Args:
            model: Model with sklearn-compatible fit/predict interface
            X: Feature matrix (must be sorted chronologically)
            y: Target values (must be sorted chronologically)
            initial_train_size: Size of initial training set.
                - If int: exact number of samples
                - If float (0-1): percentage of total samples
            step_size: Number of periods to move forward between iterations.
                Typically 1 for daily data.
            forecast_horizon: Number of periods to predict ahead.
                E.g., 7 for week-ahead forecasting.
            expanding_window: Training window strategy:
                - True: Use all historical data (expanding window)
                - False: Use fixed-size window (sliding window)
            metric: Evaluation metric. Auto-detects if None:
                - Classification: 'accuracy'
                - Regression: 'rmse'
                
        Returns:
            Dictionary containing:
                - 'method': 'walk_forward'
                - 'overall_score': Score on all predictions
                - 'fold_scores': List of scores for each fold
                - 'mean_score': Mean of fold scores
                - 'std_score': Standard deviation of fold scores
                - 'predictions': All predictions made
                - 'actuals': Corresponding actual values
                - 'fold_details': Detailed info for each fold
                
        Example:
            >>> # Week-ahead forecasting with expanding window
            >>> results = validator.walk_forward_validation(
            ...     model, X_sorted, y_sorted,
            ...     initial_train_size=365,  # Start with 1 year
            ...     step_size=7,  # Weekly steps
            ...     forecast_horizon=7,  # Predict week ahead
            ...     expanding_window=True
            ... )
            >>> 
            >>> # Plot results
            >>> plt.plot(results['actuals'], label='Actual')
            >>> plt.plot(results['predictions'], label='Predicted')
            
        Note:
            - Data must be sorted by time before calling
            - Expanding window uses all history, sliding maintains fixed size
            - Useful for evaluating models in production-like scenarios
        """
        n_samples = len(X)
        
        # Determine initial train size
        if isinstance(initial_train_size, float):
            initial_train_size = int(n_samples * initial_train_size)
            
        if initial_train_size + forecast_horizon >= n_samples:
            raise ValueError("Initial train size + forecast horizon must be less than total samples")
            
        # Auto-detect metric if not provided
        if metric is None:
            is_classification = y.nunique() < 100
            if is_classification:
                metric = 'accuracy'
            else:
                metric = 'rmse'
                
        # Store results
        fold_results = []
        predictions = []
        actuals = []
        train_sizes = []
        
        # Walk-forward validation loop
        current_train_end = initial_train_size
        fold = 0
        
        while current_train_end + forecast_horizon <= n_samples:
            # Define train and test indices
            if expanding_window:
                train_start = 0
            else:
                # Sliding window: maintain fixed window size
                train_start = max(0, current_train_end - initial_train_size)
                
            train_end = current_train_end
            test_start = train_end
            test_end = test_start + forecast_horizon
            
            # Split data
            X_train = X.iloc[train_start:train_end]
            y_train = y.iloc[train_start:train_end]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]
            
            # Clone and train model
            try:
                model_clone = model.__class__(**model.get_params())
                model_clone.fit(X_train, y_train)
                
                # Make predictions
                if hasattr(model_clone, 'predict_proba') and metric in ['roc_auc', 'log_loss']:
                    y_pred = model_clone.predict_proba(X_test)[:, 1]
                else:
                    y_pred = model_clone.predict(X_test)
                    
                # Calculate metric for this fold
                fold_score = self._calculate_metric(y_test, y_pred, metric)
                
                fold_results.append({
                    'fold': fold,
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'score': fold_score
                })
                
                predictions.extend(y_pred)
                actuals.extend(y_test)
                train_sizes.append(len(X_train))
                
                logger.info(f"Fold {fold}: Train size={len(X_train)}, Test size={len(X_test)}, {metric}={fold_score:.4f}")
                
            except Exception as e:
                logger.error(f"Error in fold {fold}: {e}")
                
            # Move to next fold
            current_train_end += step_size
            fold += 1
            
        # Calculate overall metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        overall_score = self._calculate_metric(actuals, predictions, metric)
        
        results = {
            'method': 'walk_forward',
            'expanding_window': expanding_window,
            'initial_train_size': initial_train_size,
            'step_size': step_size,
            'forecast_horizon': forecast_horizon,
            'n_folds': fold,
            'metric': metric,
            'overall_score': overall_score,
            'fold_scores': [f['score'] for f in fold_results],
            'mean_score': np.mean([f['score'] for f in fold_results]),
            'std_score': np.std([f['score'] for f in fold_results]),
            'fold_details': fold_results,
            'predictions': predictions,
            'actuals': actuals
        }
        
        return results
    
    def time_series_cv(self,
                      model: Any,
                      X: pd.DataFrame,
                      y: pd.Series,
                      n_splits: int = 5,
                      test_size: Optional[int] = None,
                      gap: int = 0,
                      metric: Optional[str] = None) -> Dict[str, Any]:
        """Perform time series cross-validation using sklearn's TimeSeriesSplit.
        
        Uses scikit-learn's TimeSeriesSplit for efficient time series
        cross-validation with customizable test size and gap.
        
        Args:
            model: Model with sklearn-compatible interface
            X: Feature matrix
            y: Target variable
            n_splits: Number of CV splits
            test_size: Fixed size for test set in each split.
                If None, test size increases with each split.
            gap: Number of samples to exclude between train and test.
                Useful to avoid leakage in time series.
            metric: Evaluation metric. Auto-detects if None.
            
        Returns:
            Dictionary containing:
                - 'method': 'time_series_split'
                - 'fold_scores': List of scores for each fold
                - 'mean_score': Mean score across folds
                - 'std_score': Standard deviation of scores
                - 'predictions': Averaged predictions where available
                - 'prediction_indices': Indices of predictions
                
        Example:
            >>> # 5-fold time series CV with 30-day test sets
            >>> results = validator.time_series_cv(
            ...     model, X, y,
            ...     n_splits=5,
            ...     test_size=30,  # Fixed 30-day test window
            ...     gap=7  # 1-week gap to avoid leakage
            ... )
            
        Note:
            Unlike walk-forward validation, this uses sklearn's
            efficient TimeSeriesSplit implementation.
        """
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
        
        # Auto-detect metric
        if metric is None:
            is_classification = y.nunique() < 100
            metric = 'accuracy' if is_classification else 'rmse'
            
        fold_scores = []
        predictions = np.zeros(len(y))
        prediction_counts = np.zeros(len(y))
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Clone and train model
            model_clone = model.__class__(**model.get_params())
            model_clone.fit(X_train, y_train)
            
            # Make predictions
            if hasattr(model_clone, 'predict_proba') and metric in ['roc_auc', 'log_loss']:
                y_pred = model_clone.predict_proba(X_test)[:, 1]
            else:
                y_pred = model_clone.predict(X_test)
                
            # Calculate metric
            fold_score = self._calculate_metric(y_test, y_pred, metric)
            fold_scores.append(fold_score)
            
            # Store predictions
            predictions[test_idx] += y_pred
            prediction_counts[test_idx] += 1
            
            logger.info(f"Fold {fold}: Train size={len(train_idx)}, Test size={len(test_idx)}, {metric}={fold_score:.4f}")
            
        # Average predictions where we have multiple
        mask = prediction_counts > 0
        predictions[mask] = predictions[mask] / prediction_counts[mask]
        
        results = {
            'method': 'time_series_split',
            'n_splits': n_splits,
            'test_size': test_size,
            'gap': gap,
            'metric': metric,
            'fold_scores': fold_scores,
            'mean_score': np.mean(fold_scores),
            'std_score': np.std(fold_scores),
            'predictions': predictions[mask],
            'prediction_indices': np.where(mask)[0]
        }
        
        return results
    
    def _calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
        """Calculate specified metric.
        
        Internal method to calculate various evaluation metrics.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            metric: Name of metric to calculate
            
        Returns:
            Calculated metric value
            
        Raises:
            ValueError: If metric name is unknown
            
        Supported metrics:
            - Regression: 'rmse', 'mse', 'mae', 'r2', 'mape'
            - Classification: 'accuracy', 'roc_auc', 'f1'
        """
        if metric == 'rmse':
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif metric == 'mse':
            return mean_squared_error(y_true, y_pred)
        elif metric == 'mae':
            return mean_absolute_error(y_true, y_pred)
        elif metric == 'r2':
            return r2_score(y_true, y_pred)
        elif metric == 'mape':
            return mean_absolute_percentage_error(y_true, y_pred)
        elif metric == 'accuracy':
            return accuracy_score(y_true, y_pred.round())
        elif metric == 'roc_auc':
            return roc_auc_score(y_true, y_pred)
        elif metric == 'f1':
            return f1_score(y_true, y_pred.round())
        else:
            raise ValueError(f"Unknown metric: {metric}")