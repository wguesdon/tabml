"""Advanced ensemble methods for combining model predictions.

This module provides sophisticated ensemble techniques including:
- OOF (Out-of-Fold) prediction generation
- Stacking with meta-learners
- Blending with validation sets
- Weighted averaging optimization
- Rank averaging for robust ensembles
"""

from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.metrics import roc_auc_score, mean_squared_error
from scipy.optimize import minimize
from scipy.stats import rankdata
import optuna
from loguru import logger
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class OOFEnsemble:
    """Advanced ensemble methods using Out-of-Fold predictions.
    
    This class provides various techniques for combining multiple models
    using their out-of-fold predictions, which helps prevent overfitting
    in the ensemble.
    
    Example:
        >>> ensemble = OOFEnsemble(task_type='classification')
        >>> 
        >>> # Generate OOF predictions for multiple models
        >>> oof_preds = ensemble.get_oof_predictions(
        ...     models=[xgb_model, lgb_model, cat_model],
        ...     X_train, y_train,
        ...     n_folds=5
        ... )
        >>> 
        >>> # Create stacked ensemble
        >>> ensemble.fit_stacking(oof_preds, y_train)
        >>> final_preds = ensemble.predict_stacking(test_predictions)
    """
    
    def __init__(self, task_type: str = 'classification', 
                 metric: Optional[Union[str, Callable]] = None,
                 random_state: int = 42):
        """Initialize OOF Ensemble.
        
        Args:
            task_type: 'classification' or 'regression'
            metric: Evaluation metric (string or callable)
            random_state: Random seed for reproducibility
        """
        self.task_type = task_type
        self.metric = metric
        self.random_state = random_state
        self.meta_model = None
        self.ensemble_weights = None
        self.models = []
        self.oof_predictions = None
        self.test_predictions = None
        
    def get_oof_predictions(self, models: List[Any], 
                           X: pd.DataFrame, y: pd.Series,
                           n_folds: int = 5,
                           stratified: bool = True,
                           groups: Optional[pd.Series] = None,
                           verbose: bool = True) -> pd.DataFrame:
        """Generate out-of-fold predictions for multiple models.
        
        Args:
            models: List of model objects with fit/predict methods
            X: Training features
            y: Training target
            n_folds: Number of CV folds
            stratified: Use stratified folds for classification
            groups: Group labels for GroupKFold
            verbose: Show progress bar
            
        Returns:
            DataFrame with OOF predictions for each model
            
        Example:
            >>> models = [xgb_model, lgb_model, cat_model]
            >>> oof_df = ensemble.get_oof_predictions(
            ...     models, X_train, y_train, n_folds=5
            ... )
            >>> # oof_df has columns: ['model_0', 'model_1', 'model_2']
        """
        self.models = models
        n_samples = len(y)
        n_models = len(models)
        
        # Initialize OOF array
        if self.task_type == 'classification' and y.nunique() == 2:
            oof_preds = np.zeros((n_samples, n_models))
        else:
            oof_preds = np.zeros((n_samples, n_models))
        
        # Setup CV
        if groups is not None:
            from sklearn.model_selection import GroupKFold
            cv = GroupKFold(n_splits=n_folds)
            split_args = (X, y, groups)
        elif stratified and self.task_type == 'classification':
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
            split_args = (X, y)
        else:
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
            split_args = (X, y)
        
        # Store fold scores for each model
        fold_scores = {i: [] for i in range(n_models)}
        
        # Generate OOF predictions for each model
        for model_idx, model in enumerate(tqdm(models, desc="Generating OOF predictions", disable=not verbose)):
            model_name = model.__class__.__name__ if hasattr(model, '__class__') else f"model_{model_idx}"
            
            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(*split_args)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Clone model or create new instance
                if hasattr(model, 'get_params'):
                    model_clone = model.__class__(**model.get_params())
                else:
                    model_clone = model.__class__()
                
                # Fit model
                if hasattr(model_clone, 'fit'):
                    # Try to fit with validation data if the model supports it
                    try:
                        # Most TabML models accept X_val and y_val parameters
                        model_clone.fit(X_train, y_train, X_val=X_val, y_val=y_val)
                    except TypeError:
                        # Fallback to simple fit without validation
                        model_clone.fit(X_train, y_train)
                
                # Generate predictions
                if self.task_type == 'classification':
                    if hasattr(model_clone, 'predict_proba'):
                        if y.nunique() == 2:
                            val_preds = model_clone.predict_proba(X_val)[:, 1]
                            oof_preds[val_idx, model_idx] = val_preds
                        else:
                            # For multiclass, store class with highest probability
                            val_preds = model_clone.predict_proba(X_val).argmax(axis=1)
                            oof_preds[val_idx, model_idx] = val_preds
                    else:
                        val_preds = model_clone.predict(X_val)
                        oof_preds[val_idx, model_idx] = val_preds
                else:
                    val_preds = model_clone.predict(X_val)
                    oof_preds[val_idx, model_idx] = val_preds
                
                # Calculate fold score
                fold_score = self._calculate_score(y_val, val_preds)
                fold_scores[model_idx].append(fold_score)
                
                if verbose:
                    logger.info(f"  {model_name} - Fold {fold_idx + 1}/{n_folds}: {fold_score:.6f}")
        
        # Create DataFrame
        columns = [f"model_{i}" for i in range(n_models)]
        self.oof_predictions = pd.DataFrame(oof_preds, columns=columns, index=X.index)
        
        # Calculate individual model scores
        if verbose:
            logger.info("\n" + "="*60)
            logger.info("Individual Model OOF Summary:")
            logger.info("="*60)
            for i, col in enumerate(columns):
                score = self._calculate_score(y, self.oof_predictions[col])
                model_name = models[i].__class__.__name__ if hasattr(models[i], '__class__') else col
                
                # Display fold scores
                logger.info(f"\n{model_name}:")
                fold_score_str = " | ".join([f"Fold {j+1}: {s:.6f}" for j, s in enumerate(fold_scores[i])])
                logger.info(f"  {fold_score_str}")
                logger.info(f"  Mean: {np.mean(fold_scores[i]):.6f} ± {np.std(fold_scores[i]):.6f}")
                logger.info(f"  Overall OOF: {score:.6f}")
        
        return self.oof_predictions
    
    def fit_stacking(self, oof_predictions: pd.DataFrame, y: pd.Series,
                     meta_model: Optional[Any] = None,
                     use_probas: bool = True,
                     add_original_features: Optional[pd.DataFrame] = None) -> 'OOFEnsemble':
        """Fit a stacking ensemble using OOF predictions.
        
        Args:
            oof_predictions: DataFrame with OOF predictions from base models
            y: True target values
            meta_model: Meta-learner model (if None, uses LogisticRegression or Ridge)
            use_probas: Use probability predictions (classification only)
            add_original_features: Optional original features to include
            
        Returns:
            Self for method chaining
            
        Example:
            >>> # Simple stacking
            >>> ensemble.fit_stacking(oof_preds, y_train)
            >>> 
            >>> # Stacking with original features
            >>> ensemble.fit_stacking(
            ...     oof_preds, y_train,
            ...     add_original_features=X_train[['important_feature']]
            ... )
        """
        # Prepare features for meta-model
        meta_features = oof_predictions.copy()
        
        if add_original_features is not None:
            meta_features = pd.concat([meta_features, add_original_features], axis=1)
        
        # Create default meta-model if not provided
        if meta_model is None:
            if self.task_type == 'classification':
                self.meta_model = LogisticRegression(random_state=self.random_state)
            else:
                self.meta_model = Ridge(random_state=self.random_state)
        else:
            self.meta_model = meta_model
        
        # Fit meta-model
        self.meta_model.fit(meta_features, y)
        
        # Calculate stacking score
        meta_predictions = self.meta_model.predict_proba(meta_features)[:, 1] if \
                          self.task_type == 'classification' and hasattr(self.meta_model, 'predict_proba') else \
                          self.meta_model.predict(meta_features)
        
        score = self._calculate_score(y, meta_predictions)
        logger.info(f"Stacking Ensemble OOF Score: {score:.4f}")
        
        return self
    
    def predict_stacking(self, test_predictions: pd.DataFrame,
                        add_original_features: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Generate predictions using the stacking ensemble.
        
        Args:
            test_predictions: DataFrame with test predictions from base models
            add_original_features: Optional original test features
            
        Returns:
            Array of stacked predictions
        """
        if self.meta_model is None:
            raise ValueError("Must call fit_stacking first")
        
        meta_features = test_predictions.copy()
        
        if add_original_features is not None:
            meta_features = pd.concat([meta_features, add_original_features], axis=1)
        
        if self.task_type == 'classification' and hasattr(self.meta_model, 'predict_proba'):
            return self.meta_model.predict_proba(meta_features)[:, 1]
        else:
            return self.meta_model.predict(meta_features)
    
    def fit_blending(self, blend_predictions: List[pd.DataFrame], 
                     y: pd.Series,
                     blend_size: float = 0.2) -> 'OOFEnsemble':
        """Fit a blending ensemble using a holdout validation set.
        
        Args:
            blend_predictions: List of DataFrames with predictions on blend set
            y: True target values for blend set
            blend_size: Fraction of data to use for blending
            
        Returns:
            Self for method chaining
        """
        # Split data for blending
        blend_idx = int(len(y) * (1 - blend_size))
        y_blend = y.iloc[blend_idx:]
        
        # Combine predictions
        blend_features = pd.concat(blend_predictions, axis=1)
        blend_features = blend_features.iloc[blend_idx:]
        
        # Fit meta-model
        if self.task_type == 'classification':
            self.meta_model = LogisticRegression(random_state=self.random_state)
        else:
            self.meta_model = Ridge(random_state=self.random_state)
        
        self.meta_model.fit(blend_features, y_blend)
        
        return self
    
    def optimize_weights(self, oof_predictions: pd.DataFrame, y: pd.Series,
                        method: str = 'scipy', n_trials: int = 100,
                        n_iterations: int = 1000, patience: int = 100) -> np.ndarray:
        """Optimize ensemble weights using various methods.
        
        Args:
            oof_predictions: DataFrame with OOF predictions
            y: True target values
            method: Optimization method ('scipy', 'optuna', 'grid', 'hill_climbing', 'greedy_forward')
            n_trials: Number of trials for Optuna
            n_iterations: Number of iterations for hill climbing
            patience: Early stopping patience for hill climbing
            
        Returns:
            Array of optimized weights
            
        Example:
            >>> # Hill climbing optimization
            >>> weights = ensemble.optimize_weights(
            ...     oof_preds, y_train, 
            ...     method='hill_climbing',
            ...     n_iterations=2000,
            ...     patience=200
            ... )
            >>> # Use weights for weighted averaging
            >>> final_pred = np.average(test_preds, weights=weights, axis=1)
        """
        n_models = oof_predictions.shape[1]
        
        if method == 'scipy':
            weights = self._optimize_weights_scipy(oof_predictions, y)
        elif method == 'optuna':
            weights = self._optimize_weights_optuna(oof_predictions, y, n_trials)
        elif method == 'grid':
            weights = self._optimize_weights_grid(oof_predictions, y)
        elif method == 'hill_climbing':
            weights = self._optimize_weights_hill_climbing(oof_predictions, y, n_iterations, patience)
        elif method == 'greedy_forward':
            weights = self._optimize_weights_greedy_forward(oof_predictions, y)
        else:
            # Equal weights
            weights = np.ones(n_models) / n_models
        
        self.ensemble_weights = weights
        
        # Calculate weighted ensemble score
        weighted_pred = np.average(oof_predictions.values, weights=weights, axis=1)
        score = self._calculate_score(y, weighted_pred)
        logger.info(f"Weighted Ensemble OOF Score: {score:.4f}")
        logger.info(f"Optimized Weights: {weights}")
        
        return weights
    
    def _optimize_weights_scipy(self, predictions: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Optimize weights using scipy.optimize."""
        n_models = predictions.shape[1]
        
        def objective(weights):
            weighted_pred = np.average(predictions.values, weights=weights, axis=1)
            return -self._calculate_score(y, weighted_pred)
        
        # Constraints: weights sum to 1, all weights >= 0
        constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
        bounds = [(0, 1)] * n_models
        
        # Initial guess: equal weights
        x0 = np.ones(n_models) / n_models
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        return result.x
    
    def _optimize_weights_optuna(self, predictions: pd.DataFrame, y: pd.Series, 
                                 n_trials: int) -> np.ndarray:
        """Optimize weights using Optuna."""
        n_models = predictions.shape[1]
        
        def objective(trial):
            # Sample weights
            weights = []
            for i in range(n_models - 1):
                weights.append(trial.suggest_float(f'weight_{i}', 0, 1))
            
            # Last weight to ensure sum = 1
            last_weight = 1 - sum(weights)
            if last_weight < 0 or last_weight > 1:
                return -1e10  # Invalid weights
            weights.append(last_weight)
            
            # Calculate score
            weighted_pred = np.average(predictions.values, weights=weights, axis=1)
            return self._calculate_score(y, weighted_pred)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        # Extract best weights
        best_weights = []
        for i in range(n_models - 1):
            best_weights.append(study.best_params[f'weight_{i}'])
        best_weights.append(1 - sum(best_weights))
        
        return np.array(best_weights)
    
    def _optimize_weights_grid(self, predictions: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Optimize weights using grid search."""
        n_models = predictions.shape[1]
        
        # Create grid (simplified for efficiency)
        if n_models == 2:
            grid = [(w, 1-w) for w in np.arange(0, 1.1, 0.1)]
        elif n_models == 3:
            grid = [(w1, w2, 1-w1-w2) 
                   for w1 in np.arange(0, 1.1, 0.2)
                   for w2 in np.arange(0, 1.1-w1, 0.2)
                   if 1-w1-w2 >= 0]
        else:
            # Too many models for grid search, use equal weights
            return np.ones(n_models) / n_models
        
        best_score = -np.inf
        best_weights = None
        
        for weights in grid:
            weighted_pred = np.average(predictions.values, weights=weights, axis=1)
            score = self._calculate_score(y, weighted_pred)
            if score > best_score:
                best_score = score
                best_weights = weights
        
        return np.array(best_weights)
    
    def _optimize_weights_hill_climbing(self, predictions: pd.DataFrame, y: pd.Series,
                                       n_iterations: int = 1000, patience: int = 100) -> np.ndarray:
        """Optimize weights using hill climbing algorithm.
        
        Hill climbing iteratively improves weights by making small random changes
        and keeping improvements. Particularly effective for finding good local optima.
        """
        n_models = predictions.shape[1]
        
        # Initialize with equal weights
        current_weights = np.ones(n_models) / n_models
        current_score = self._calculate_score(y, np.average(predictions.values, weights=current_weights, axis=1))
        
        best_weights = current_weights.copy()
        best_score = current_score
        
        # Learning rate schedule
        initial_lr = 0.1
        min_lr = 0.001
        
        no_improvement_count = 0
        
        logger.info("Starting hill climbing optimization...")
        
        for iteration in range(n_iterations):
            # Adaptive learning rate
            lr = max(min_lr, initial_lr * (1 - iteration / n_iterations))
            
            # Generate neighbor solution
            perturbation = np.random.randn(n_models) * lr
            new_weights = current_weights + perturbation
            
            # Ensure weights are valid (non-negative and sum to 1)
            new_weights = np.maximum(new_weights, 0)
            new_weights = new_weights / new_weights.sum()
            
            # Evaluate new weights
            new_pred = np.average(predictions.values, weights=new_weights, axis=1)
            new_score = self._calculate_score(y, new_pred)
            
            # Accept if better (hill climbing)
            if new_score > current_score:
                current_weights = new_weights
                current_score = new_score
                no_improvement_count = 0
                
                if new_score > best_score:
                    best_weights = new_weights.copy()
                    best_score = new_score
                    logger.debug(f"Iteration {iteration}: New best score: {best_score:.6f}")
            else:
                no_improvement_count += 1
                
                # Occasionally accept worse solution to escape local optima (simulated annealing)
                temperature = 0.1 * (1 - iteration / n_iterations)
                if temperature > 0 and np.random.random() < np.exp((new_score - current_score) / temperature):
                    current_weights = new_weights
                    current_score = new_score
            
            # Early stopping
            if no_improvement_count >= patience:
                logger.info(f"Early stopping at iteration {iteration}")
                break
            
            # Progress logging
            if iteration % 100 == 0:
                logger.debug(f"Iteration {iteration}/{n_iterations}, Best Score: {best_score:.6f}")
        
        logger.info(f"Hill climbing completed. Best score: {best_score:.6f}")
        return best_weights
    
    def _optimize_weights_greedy_forward(self, predictions: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Optimize weights using greedy forward selection.
        
        Starts with the best single model and iteratively adds models that improve
        the ensemble score. This can identify the most complementary models.
        """
        n_models = predictions.shape[1]
        
        # Find best single model
        best_single_score = -np.inf
        best_single_idx = 0
        
        for i in range(n_models):
            score = self._calculate_score(y, predictions.iloc[:, i])
            if score > best_single_score:
                best_single_score = score
                best_single_idx = i
        
        # Initialize with best single model
        selected_models = [best_single_idx]
        weights = np.zeros(n_models)
        weights[best_single_idx] = 1.0
        
        logger.info(f"Starting greedy forward selection with model {best_single_idx} (score: {best_single_score:.6f})")
        
        # Iteratively add models
        for step in range(n_models - 1):
            best_addition_score = -np.inf
            best_addition_idx = None
            best_addition_weights = None
            
            # Try adding each remaining model
            for i in range(n_models):
                if i in selected_models:
                    continue
                
                # Create ensemble with current models + candidate
                candidate_models = selected_models + [i]
                candidate_preds = predictions.iloc[:, candidate_models]
                
                # Optimize weights for this subset using scipy
                n_subset = len(candidate_models)
                
                def objective(w):
                    weighted_pred = np.average(candidate_preds.values, weights=w, axis=1)
                    return -self._calculate_score(y, weighted_pred)
                
                from scipy.optimize import minimize
                constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
                bounds = [(0, 1)] * n_subset
                x0 = np.ones(n_subset) / n_subset
                
                result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
                
                if result.success:
                    score = -result.fun
                    if score > best_addition_score:
                        best_addition_score = score
                        best_addition_idx = i
                        # Map subset weights back to full weight vector
                        temp_weights = np.zeros(n_models)
                        for j, model_idx in enumerate(candidate_models):
                            temp_weights[model_idx] = result.x[j]
                        best_addition_weights = temp_weights
            
            # Check if adding a model improves the ensemble
            current_score = self._calculate_score(y, np.average(predictions.values, weights=weights, axis=1))
            
            if best_addition_idx is not None and best_addition_score > current_score:
                selected_models.append(best_addition_idx)
                weights = best_addition_weights
                logger.info(f"Step {step + 1}: Added model {best_addition_idx}, "
                          f"Score improved from {current_score:.6f} to {best_addition_score:.6f}")
            else:
                logger.info(f"No improvement found at step {step + 1}. Stopping.")
                break
        
        logger.info(f"Selected {len(selected_models)} models: {selected_models}")
        logger.info(f"Final weights: {weights[weights > 0]}")
        
        return weights
    
    def rank_average(self, predictions: List[np.ndarray]) -> np.ndarray:
        """Create ensemble using rank averaging.
        
        Rank averaging is robust to different scales and distributions
        of predictions from different models.
        
        Args:
            predictions: List of prediction arrays
            
        Returns:
            Rank-averaged predictions
            
        Example:
            >>> # Combine predictions with different scales
            >>> rank_avg = ensemble.rank_average([
            ...     xgb_preds,  # Range [0, 1]
            ...     nn_preds,   # Range [-5, 5]
            ...     rf_preds    # Range [0, 100]
            ... ])
        """
        ranked_preds = []
        
        for pred in predictions:
            # Rank transform each prediction
            ranked = rankdata(pred) / len(pred)
            ranked_preds.append(ranked)
        
        # Average ranks
        avg_ranks = np.mean(ranked_preds, axis=0)
        
        return avg_ranks
    
    def geometric_mean(self, predictions: List[np.ndarray]) -> np.ndarray:
        """Combine predictions using geometric mean.
        
        Geometric mean is useful when predictions are probabilities
        and you want a more conservative ensemble.
        
        Args:
            predictions: List of prediction arrays (should be positive)
            
        Returns:
            Geometric mean of predictions
        """
        # Ensure all predictions are positive
        predictions = [np.maximum(pred, 1e-15) for pred in predictions]
        
        # Calculate geometric mean
        log_preds = [np.log(pred) for pred in predictions]
        geo_mean = np.exp(np.mean(log_preds, axis=0))
        
        return geo_mean
    
    def _calculate_score(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate evaluation score based on metric."""
        if callable(self.metric):
            return self.metric(y_true, y_pred)
        
        if self.task_type == 'classification':
            if self.metric == 'auc' or self.metric is None:
                if y_true.nunique() == 2:
                    return roc_auc_score(y_true, y_pred)
                else:
                    from sklearn.metrics import accuracy_score
                    return accuracy_score(y_true, y_pred.round())
        else:
            if self.metric == 'rmse' or self.metric is None:
                return -np.sqrt(mean_squared_error(y_true, y_pred))
            elif self.metric == 'mse':
                return -mean_squared_error(y_true, y_pred)
        
        return 0.0
    
    def get_test_predictions(self, models: List[Any], X_test: pd.DataFrame,
                            use_trained_models: bool = True) -> pd.DataFrame:
        """Generate test predictions from all models.
        
        Args:
            models: List of trained models
            X_test: Test features
            use_trained_models: If True, use already trained models
            
        Returns:
            DataFrame with test predictions from each model
        """
        n_models = len(models)
        n_samples = len(X_test)
        
        test_preds = np.zeros((n_samples, n_models))
        
        for i, model in enumerate(models):
            if self.task_type == 'classification' and hasattr(model, 'predict_proba'):
                test_preds[:, i] = model.predict_proba(X_test)[:, 1]
            else:
                test_preds[:, i] = model.predict(X_test)
        
        columns = [f"model_{i}" for i in range(n_models)]
        self.test_predictions = pd.DataFrame(test_preds, columns=columns, index=X_test.index)
        
        return self.test_predictions
    
    def create_submission(self, test_predictions: np.ndarray, 
                         sample_submission: pd.DataFrame,
                         target_column: str = 'target') -> pd.DataFrame:
        """Create submission file for Kaggle competitions.
        
        Args:
            test_predictions: Final test predictions
            sample_submission: Sample submission DataFrame
            target_column: Name of target column in submission
            
        Returns:
            Submission DataFrame ready to save
        """
        submission = sample_submission.copy()
        submission[target_column] = test_predictions
        
        return submission


class AutoEnsemble:
    """Automated ensemble creation with multiple strategies.
    
    Automatically tries different ensemble methods and selects the best.
    
    Example:
        >>> auto = AutoEnsemble(task_type='classification')
        >>> best_ensemble = auto.fit(
        ...     models=[xgb, lgb, cat],
        ...     X_train, y_train,
        ...     strategies=['stacking', 'weighted', 'rank']
        ... )
        >>> predictions = auto.predict(X_test)
    """
    
    def __init__(self, task_type: str = 'classification',
                 metric: Optional[Union[str, Callable]] = None,
                 random_state: int = 42):
        """Initialize auto ensemble."""
        self.task_type = task_type
        self.metric = metric
        self.random_state = random_state
        self.best_strategy = None
        self.best_ensemble = None
        self.ensemble_scores = {}
        
    def fit(self, models: List[Any], X: pd.DataFrame, y: pd.Series,
            strategies: List[str] = ['weighted', 'stacking', 'rank'],
            cv_folds: int = 5) -> 'AutoEnsemble':
        """Automatically find best ensemble strategy.
        
        Args:
            models: List of base models
            X: Training features
            y: Training target
            strategies: List of strategies to try
            cv_folds: Number of CV folds
            
        Returns:
            Self with best ensemble fitted
        """
        logger.info("Testing ensemble strategies...")
        
        # Generate OOF predictions once
        ensemble = OOFEnsemble(self.task_type, self.metric, self.random_state)
        oof_preds = ensemble.get_oof_predictions(models, X, y, cv_folds)
        
        # Try each strategy
        for strategy in strategies:
            logger.info(f"\nTrying {strategy} ensemble...")
            
            if strategy == 'weighted':
                weights = ensemble.optimize_weights(oof_preds, y, method='optuna')
                weighted_pred = np.average(oof_preds.values, weights=weights, axis=1)
                score = ensemble._calculate_score(y, weighted_pred)
                self.ensemble_scores[strategy] = score
                
            elif strategy == 'stacking':
                ensemble_stack = OOFEnsemble(self.task_type, self.metric, self.random_state)
                ensemble_stack.fit_stacking(oof_preds, y)
                meta_pred = ensemble_stack.meta_model.predict_proba(oof_preds)[:, 1] if \
                           self.task_type == 'classification' else \
                           ensemble_stack.meta_model.predict(oof_preds)
                score = ensemble._calculate_score(y, meta_pred)
                self.ensemble_scores[strategy] = score
                
            elif strategy == 'rank':
                rank_pred = ensemble.rank_average([oof_preds[col].values for col in oof_preds.columns])
                score = ensemble._calculate_score(y, rank_pred)
                self.ensemble_scores[strategy] = score
        
        # Select best strategy
        self.best_strategy = max(self.ensemble_scores, key=self.ensemble_scores.get)
        logger.info(f"\nBest strategy: {self.best_strategy} (score: {self.ensemble_scores[self.best_strategy]:.4f})")
        
        # Refit best ensemble
        if self.best_strategy == 'weighted':
            self.best_ensemble = ensemble
            self.best_ensemble.optimize_weights(oof_preds, y, method='optuna')
        elif self.best_strategy == 'stacking':
            self.best_ensemble = OOFEnsemble(self.task_type, self.metric, self.random_state)
            self.best_ensemble.fit_stacking(oof_preds, y)
        elif self.best_strategy == 'rank':
            self.best_ensemble = ensemble
        
        return self
    
    def predict(self, models: List[Any], X_test: pd.DataFrame) -> np.ndarray:
        """Generate predictions using best ensemble strategy.
        
        Args:
            models: List of trained models
            X_test: Test features
            
        Returns:
            Final ensemble predictions
        """
        # Get test predictions from all models
        test_preds = self.best_ensemble.get_test_predictions(models, X_test)
        
        if self.best_strategy == 'weighted':
            return np.average(test_preds.values, weights=self.best_ensemble.ensemble_weights, axis=1)
        elif self.best_strategy == 'stacking':
            return self.best_ensemble.predict_stacking(test_preds)
        elif self.best_strategy == 'rank':
            return self.best_ensemble.rank_average([test_preds[col].values for col in test_preds.columns])
        else:
            return np.mean(test_preds.values, axis=1)