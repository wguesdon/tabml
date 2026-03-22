"""AutoGluon model wrapper for TabML.

This module provides a unified interface for AutoGluon's TabularPredictor,
enabling seamless integration with TabML's ensemble and OOF management system.

Classes:
    AutoGluonModel: AutoGluon TabularPredictor wrapper

Example:
    Basic usage::
    
        from tabml.autogluon_model import AutoGluonModel
        from tabml.ensemble import OOFEnsemble
        
        # Initialize model
        model = AutoGluonModel(
            params={
                'time_limit': 600,
                'presets': 'best_quality',
                'eval_metric': 'roc_auc'
            }
        )
        
        # Get OOF predictions
        ensemble = OOFEnsemble(task_type='classification')
        oof_preds = ensemble.get_oof_predictions(
            models=[model],
            X=X_train,
            y=y_train,
            n_folds=5
        )
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

try:
    from autogluon.tabular import TabularPredictor
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False
    logger.warning("AutoGluon not installed. Install with: pip install autogluon")

from .models import BaseModel


class AutoGluonModel(BaseModel):
    """An AutoGluon TabularPredictor wrapper.
    
    Provides a unified interface to AutoGluon's TabularPredictor that is
    compatible with TabML's ensemble and cross-validation systems. AutoGluon
    automatically trains multiple models and ensembles them internally.
    
    Attributes:
        All attributes from BaseModel plus AutoGluon-specific parameters.
        predictor_path: Path where AutoGluon saves its models
        predictor: The AutoGluon TabularPredictor instance
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initializes the AutoGluonModel.
        
        Args:
            params (Optional[Dict[str, Any]]): Configuration for AutoGluon.
                Common parameters:
                - time_limit: Maximum time in seconds for training
                - presets: Quality preset ('best_quality', 'high_quality', 
                           'good_quality', 'medium_quality', 'optimize_for_deployment')
                - eval_metric: Metric to optimize ('roc_auc', 'accuracy', 'rmse', etc.)
                - num_bag_folds: Number of folds for bagging (default: 0)
                - num_bag_sets: Number of bagging repeats (default: 1)
                - num_stack_levels: Number of stacking levels (default: 0)
                - hyperparameters: Custom model hyperparameters
                - excluded_model_types: List of model types to exclude
                - included_model_types: List of model types to include
                - auto_stack: Whether to use automatic stacking
                - holdout_frac: Fraction of data for holdout validation
                - save_space: Whether to reduce model file size
                - verbosity: Logging level (0-4)
        """
        if not AUTOGLUON_AVAILABLE:
            raise ImportError(
                "AutoGluon is required for AutoGluonModel. "
                "Install with: pip install autogluon"
            )
        
        default_params = {
            'time_limit': 600,  # 10 minutes default
            'presets': 'best_quality',
            'eval_metric': None,  # Will be set based on task type
            'num_bag_folds': 0,  # No bagging by default for speed
            'num_bag_sets': 1,
            'num_stack_levels': 0,  # No stacking by default
            'auto_stack': False,
            'holdout_frac': 0.2,  # 20% holdout for validation
            'save_space': True,  # Reduce model size
            'verbosity': 2,  # Moderate logging
            'keep_only_best': False,  # Keep all models for ensemble
            'save_bag_folds': True,  # Save OOF predictions from bagging
        }
        
        if params:
            default_params.update(params)
            
        super().__init__("AutoGluon", default_params)
        self.predictor_path = None
        self.predictor = None
        self._temp_dir = None
        
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for cloning the model.
        
        Args:
            deep: If True, return deep copy of parameters
            
        Returns:
            Dictionary of parameters wrapped in 'params' key for __init__
        """
        # Return params wrapped in a dict since __init__ expects params as a single argument
        return {'params': self.params.copy() if deep else self.params}
        
    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None,
            **kwargs) -> 'AutoGluonModel':
        """Fits the AutoGluon model.
        
        AutoGluon handles its own internal cross-validation and ensembling.
        If validation data is provided, it will be used for early stopping.
        
        Args:
            X (pd.DataFrame): The training features.
            y (pd.Series): The training target.
            X_val (Optional[pd.DataFrame]): Validation features (optional).
            y_val (Optional[pd.Series]): Validation target (optional).
            **kwargs: Additional arguments passed to TabularPredictor.fit()
        
        Returns:
            The self instance for method chaining.
        """
        self._determine_task_type(y)
        self.feature_names = X.columns.tolist()
        
        # Set eval_metric based on task type if not specified
        if self.params.get('eval_metric') is None:
            if self.is_classification:
                if y.nunique() == 2:
                    self.params['eval_metric'] = 'roc_auc'
                else:
                    self.params['eval_metric'] = 'accuracy'
            else:
                self.params['eval_metric'] = 'rmse'
        
        # Prepare training data
        train_data = X.copy()
        train_data['target'] = y.values
        
        # Prepare validation data if provided
        tuning_data = None
        if X_val is not None and y_val is not None:
            tuning_data = X_val.copy()
            tuning_data['target'] = y_val.values
            # Don't use holdout if we have explicit validation data
            self.params['holdout_frac'] = 0
        
        # Create temporary directory for AutoGluon models
        # Always create a new temp directory for each fit to avoid conflicts in CV
        if self.predictor_path is None or not Path(self.predictor_path).exists():
            self._temp_dir = tempfile.mkdtemp(prefix='autogluon_')
            self.predictor_path = self._temp_dir
        
        # Extract parameters for TabularPredictor initialization vs fit
        init_params = {
            'label': 'target',
            'eval_metric': self.params.get('eval_metric'),
            'path': self.predictor_path,
            'verbosity': self.params.get('verbosity', 2),
            'problem_type': 'binary' if self.is_classification and y.nunique() == 2 else None
        }
        
        # Parameters for fit method
        fit_params = {
            'time_limit': self.params.get('time_limit', 600),
            'presets': self.params.get('presets', 'best_quality'),
            'num_bag_folds': self.params.get('num_bag_folds', 0),
            'num_bag_sets': self.params.get('num_bag_sets', 1),
            'num_stack_levels': self.params.get('num_stack_levels', 0),
            'auto_stack': self.params.get('auto_stack', False),
            'holdout_frac': self.params.get('holdout_frac', 0.2),
            'save_space': self.params.get('save_space', True),
            'keep_only_best': self.params.get('keep_only_best', False),
            'save_bag_folds': self.params.get('save_bag_folds', True),
        }
        
        # Add optional parameters if specified
        if 'hyperparameters' in self.params:
            fit_params['hyperparameters'] = self.params['hyperparameters']
        if 'excluded_model_types' in self.params:
            fit_params['excluded_model_types'] = self.params['excluded_model_types']
        if 'included_model_types' in self.params:
            fit_params['included_model_types'] = self.params['included_model_types']
        
        # Update with any additional kwargs
        fit_params.update(kwargs)
        
        # Initialize and fit predictor
        self.predictor = TabularPredictor(**init_params)
        
        self.predictor.fit(
            train_data=train_data,
            tuning_data=tuning_data,
            **fit_params
        )
        
        # Get feature importance if available
        try:
            importance_df = self.predictor.feature_importance(train_data)
            self.feature_importances_ = importance_df['importance'].values
        except:
            self.feature_importances_ = np.zeros(len(self.feature_names))
            
        self.is_fitted = True
        
        # Log model performance
        if self.params.get('verbosity', 2) > 0:
            logger.info("\nAutoGluon Model Summary:")
            try:
                # Try to get best model name from leaderboard
                leaderboard = self.predictor.leaderboard(silent=True)
                if not leaderboard.empty:
                    best_model = leaderboard.iloc[0]['model']
                    logger.info(f"Best model: {best_model}")
                    logger.info(f"Best validation score: {leaderboard.iloc[0]['score_val']:.6f}")
                    logger.info(f"\nModel Leaderboard:\n{leaderboard.head(10)}")
            except Exception as e:
                logger.debug(f"Could not get leaderboard: {e}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Makes predictions.
        
        Args:
            X (pd.DataFrame): The features to predict on.
        
        Returns:
            An array of predictions.
        
        Raises:
            ValueError: If the model has not been fitted yet.
        """
        if not self.is_fitted or self.predictor is None:
            raise ValueError("AutoGluon model must be fitted before prediction.")
        
        return self.predictor.predict(X).values
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts probabilities (for classification only).
        
        Args:
            X (pd.DataFrame): The features to predict on.
        
        Returns:
            An array of shape (n_samples, n_classes) with class probabilities.
            For binary classification, returns full probability matrix.
        
        Raises:
            ValueError: If the model has not been fitted yet or if the task is
                not classification.
        """
        if not self.is_classification:
            raise ValueError("predict_proba is only available for classification.")
        if not self.is_fitted or self.predictor is None:
            raise ValueError("AutoGluon model must be fitted before prediction.")
        
        proba = self.predictor.predict_proba(X)
        
        # Always return 2D array for compatibility with sklearn and ensemble
        if isinstance(proba, pd.DataFrame):
            return proba.values
        elif isinstance(proba, pd.Series):
            # If AutoGluon returns 1D array for binary, reconstruct 2D array
            pos_proba = proba.values
            neg_proba = 1 - pos_proba
            return np.column_stack([neg_proba, pos_proba])
        else:
            # Check if it's 1D numpy array
            if proba.ndim == 1:
                pos_proba = proba
                neg_proba = 1 - pos_proba
                return np.column_stack([neg_proba, pos_proba])
            else:
                return proba
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about trained models.
        
        Returns:
            Dictionary containing model information including:
            - best_model: Name of the best model
            - best_score: Best validation score
            - model_types: List of model types trained
            - num_models: Total number of models
            - training_time: Total training time
        """
        if not self.is_fitted or self.predictor is None:
            raise ValueError("Model must be fitted first.")
        
        info = self.predictor.info()
        leaderboard = self.predictor.leaderboard(silent=True)
        
        # Get best model from leaderboard
        best_model = None
        best_score = None
        if not leaderboard.empty:
            best_model = leaderboard.iloc[0]['model']
            best_score = leaderboard.iloc[0]['score_val']
        
        return {
            'best_model': best_model,
            'best_score': best_score,
            'model_types': leaderboard['model'].tolist() if not leaderboard.empty else [],
            'num_models': len(leaderboard),
            'training_time': info.get('time_fit', None),
            'leaderboard': leaderboard.to_dict() if not leaderboard.empty else {}
        }
    
    def get_oof_predictions(self) -> Optional[pd.Series]:
        """Get out-of-fold predictions from AutoGluon's internal CV.
        
        Returns:
            OOF predictions if available (requires save_bag_folds=True and bagging).
            Returns None if OOF predictions are not available.
        """
        if not self.is_fitted or self.predictor is None:
            raise ValueError("Model must be fitted first.")
        
        try:
            # Try to get OOF predictions from AutoGluon
            # This requires bagging to be enabled (num_bag_folds > 0)
            oof_pred = self.predictor.predict_oof()
            if oof_pred is not None:
                if self.is_classification and hasattr(self.predictor, 'predict_proba_oof'):
                    # For classification, try to get probability predictions
                    oof_proba = self.predictor.predict_proba_oof()
                    if isinstance(oof_proba, pd.DataFrame) and oof_proba.shape[1] == 2:
                        return oof_proba.iloc[:, 1]
                    return oof_proba
                return oof_pred
        except:
            pass
        
        return None
    
    def cleanup(self):
        """Clean up temporary files created by AutoGluon."""
        if hasattr(self, '_temp_dir') and self._temp_dir and Path(self._temp_dir).exists():
            try:
                shutil.rmtree(self._temp_dir)
                logger.debug(f"Cleaned up AutoGluon temp directory: {self._temp_dir}")
            except:
                logger.warning(f"Could not clean up temp directory: {self._temp_dir}")
        self._temp_dir = None
        self.predictor_path = None
        self.predictor = None
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass  # Silently ignore cleanup errors during deletion