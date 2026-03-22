"""Machine learning models for tabular data.

This module provides unified interfaces for popular gradient boosting and
ensemble models, with built-in hyperparameter optimization support.

Classes:
    BaseModel: Abstract base class for all models
    XGBoostModel: XGBoost implementation
    LightGBMModel: LightGBM implementation
    CatBoostModel: CatBoost implementation
    RandomForestModel: Random Forest implementation
    ModelTrainer: High-level interface for training and optimization
    
Example:
    Basic usage::
    
        from tabml.models import ModelTrainer
        
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Train a single model
        model = trainer.train_model(
            'xgboost', X_train, y_train, X_val, y_val,
            optimize_hyperparams=True
        )
        
        # Make predictions
        predictions = model.predict(X_test)
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, KFold, GroupKFold, 
    RepeatedKFold, RepeatedStratifiedKFold, TimeSeriesSplit
)
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error,
    mean_absolute_percentage_error, make_scorer
)
import optuna
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

try:
    from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    logger.warning("TabNet not installed. Install with: pip install pytorch-tabnet")


class BaseModel:
    """Base class for all models.

    Provides a common interface and functionality for all model implementations,
    including automatic task type detection, prediction methods, and feature
    importance extraction.

    Attributes:
        name (str): The model name identifier.
        params (Dict[str, Any]): A dictionary of model hyperparameters.
        model: The underlying model object (set after fitting).
        is_fitted (bool): Whether the model has been fitted.
        is_classification (bool): Whether this is a classification task.
        feature_names (List[str]): A list of feature names used in training.
        feature_importances_ (np.ndarray): Feature importance scores after fitting.
    """
    
    def __init__(self, name: str, params: Dict[str, Any]):
        """Initializes the BaseModel.

        Args:
            name (str): The name of the model.
            params (Dict[str, Any]): The model's hyperparameters.
        """
        self.name = name
        self.params = params
        self.model = None
        self.is_fitted = False
        self.is_classification = None
        self.feature_names = None
        self.feature_importances_ = None
        
    def _determine_task_type(self, y: pd.Series) -> None:
        """Determines if the task is classification or regression.

        Uses an improved heuristic prioritizing dtype over unique count:
        - Float dtype (float32/float64) → regression (continuous by nature)
        - Integer with < 100 unique values → classification (discrete categories)
        - Otherwise → regression

        Args:
            y (pd.Series): The target variable to analyze.
        """
        unique_count = y.nunique()

        # Float dtype indicates continuous target → always regression
        if y.dtype in ['float32', 'float64']:
            self.is_classification = False
        # Integer with few unique values indicates discrete categories → classification
        elif y.dtype in ['int32', 'int64'] and unique_count < 100:
            self.is_classification = True
        # Default: regression for all other cases
        else:
            self.is_classification = False
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Makes predictions.

        Args:
            X (pd.DataFrame): The features to predict on.

        Returns:
            An array of predictions.

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name} model must be fitted before prediction.")
        return self.model.predict(X)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts probabilities (for classification only).

        Args:
            X (pd.DataFrame): The features to predict on.

        Returns:
            An array of shape (n_samples, n_classes) with class probabilities.

        Raises:
            ValueError: If the model has not been fitted yet or if the task is
                not classification.
        """
        if not self.is_classification:
            raise ValueError("predict_proba is only available for classification.")
        if not self.is_fitted:
            raise ValueError(f"{self.name} model must be fitted before prediction.")
        return self.model.predict_proba(X)


class XGBoostModel(BaseModel):
    """An XGBoost model implementation.

    A wrapper around XGBoost with automatic objective selection and early
    stopping support. Handles both classification and regression tasks.

    Attributes:
        All attributes from BaseModel plus XGBoost-specific parameters.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initializes the XGBoostModel.

        Args:
            params (Optional[Dict[str, Any]]): The hyperparameters for the
                XGBoost model.
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.3,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'min_child_weight': 1,
            'n_jobs': -1,
            'random_state': 42,
            'verbosity': 0
        }
        if params:
            default_params.update(params)
        super().__init__("XGBoost", default_params)
        self._use_booster = False  # True when training via xgboost.train
        self._objective = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None,
            early_stopping_rounds: int = 50) -> 'XGBoostModel':
        """Fits the XGBoost model.

        Automatically detects the task type and sets the appropriate objective.
        Supports early stopping with a validation set.

        Args:
            X (pd.DataFrame): The training features.
            y (pd.Series): The training target.
            X_val (Optional[pd.DataFrame]): The validation features for early
                stopping (optional).
            y_val (Optional[pd.Series]): The validation target for early stopping
                (optional).
            early_stopping_rounds (int): The number of rounds to stop if the
                validation score doesn't improve. Only used if a validation set
                is provided.

        Returns:
            The self instance for method chaining.
        """
        self._determine_task_type(y)
        self.feature_names = X.columns.tolist()
        
        # Set objective based on task type
        if self.is_classification:
            if y.nunique() == 2:
                self.params['objective'] = 'binary:logistic'
            else:
                self.params['objective'] = 'multi:softprob'
                self.params['num_class'] = y.nunique()
        else:
            self.params['objective'] = 'reg:squarederror'
            
        # Decide training path: booster path if we have a validation set and want early stopping
        use_booster = (X_val is not None and y_val is not None and early_stopping_rounds and early_stopping_rounds > 0)

        if use_booster:
            # Train via xgboost.train for robust early stopping support
            self._use_booster = True
            import xgboost as _xgb

            self._objective = self.params.get('objective')
            feature_names_str = [str(fn) for fn in self.feature_names]
            dtrain = _xgb.DMatrix(X, label=y, feature_names=feature_names_str)
            dvalid = _xgb.DMatrix(X_val, label=y_val, feature_names=feature_names_str)

            # Build parameter dictionary for xgb.train
            train_params = dict(self.params)
            # Map common sklearn-style params to core params
            if 'learning_rate' in train_params:
                train_params['eta'] = train_params.pop('learning_rate')
            if 'n_jobs' in train_params:
                train_params['nthread'] = train_params.pop('n_jobs')
            if 'random_state' in train_params:
                train_params['seed'] = train_params.pop('random_state')
            num_boost_round = train_params.pop('n_estimators', 100)

            evals = [(dtrain, 'train'), (dvalid, 'valid')]

            try:
                booster = _xgb.train(
                    train_params,
                    dtrain,
                    num_boost_round=num_boost_round,
                    evals=evals,
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=False
                )
            except TypeError:
                # Newer versions prefer callback API
                callbacks = []
                try:
                    callbacks.append(_xgb.callback.EarlyStopping(rounds=early_stopping_rounds, save_best=True))
                except Exception:
                    callbacks.append(_xgb.callback.EarlyStopping(rounds=early_stopping_rounds))
                booster = _xgb.train(
                    train_params,
                    dtrain,
                    num_boost_round=num_boost_round,
                    evals=evals,
                    callbacks=callbacks,
                    verbose_eval=False
                )

            self.model = booster
        else:
            # Use sklearn API (no early stopping or no validation set)
            if self.is_classification:
                self.model = xgb.XGBClassifier(**self.params)
            else:
                self.model = xgb.XGBRegressor(**self.params)

            self.model.fit(X, y, verbose=False)
        
        # Feature importances
        if self._use_booster:
            # Use gain importance aligned to feature names
            try:
                score = self.model.get_score(importance_type='gain')
                feature_names_str = [str(fn) for fn in self.feature_names]
                importances = np.array([score.get(fn, 0.0) for fn in feature_names_str])
                self.feature_importances_ = importances
            except Exception:
                self.feature_importances_ = None
        else:
            self.feature_importances_ = self.model.feature_importances_
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError(f"{self.name} model must be fitted before prediction.")
        if not self._use_booster:
            return self.model.predict(X)
        import xgboost as _xgb
        feature_names_str = [str(fn) for fn in self.feature_names]
        dtest = _xgb.DMatrix(X, feature_names=feature_names_str)
        preds = self.model.predict(dtest, iteration_range=(0, getattr(self.model, 'best_iteration', 0) or 0))
        if self.is_classification:
            if self.params.get('objective') == 'multi:softprob' or (self._objective == 'multi:softprob'):
                # Return class labels
                return preds.reshape(X.shape[0], -1).argmax(axis=1)
            else:
                # Binary: threshold at 0.5
                return (preds > 0.5).astype(int)
        return preds

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_classification:
            raise ValueError("predict_proba is only available for classification.")
        if not self.is_fitted:
            raise ValueError(f"{self.name} model must be fitted before prediction.")
        if not self._use_booster:
            return self.model.predict_proba(X)
        import xgboost as _xgb
        feature_names_str = [str(fn) for fn in self.feature_names]
        dtest = _xgb.DMatrix(X, feature_names=feature_names_str)
        preds = self.model.predict(dtest, iteration_range=(0, getattr(self.model, 'best_iteration', 0) or 0))
        # Binary returns shape (n,), convert to (n, 2); multiclass returns (n, k)
        if preds.ndim == 1:
            return np.vstack([1 - preds, preds]).T
        else:
            return preds


class LightGBMModel(BaseModel):
    """A LightGBM model implementation.

    A wrapper around LightGBM with automatic objective selection and early
    stopping. Known for fast training speed and good performance on large
    datasets.

    Attributes:
        All attributes from BaseModel plus LightGBM-specific parameters.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initializes the LightGBMModel.

        Args:
            params (Optional[Dict[str, Any]]): The hyperparameters for the
                LightGBM model.
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': -1,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'reg_alpha': 0,
            'reg_lambda': 0,
            'min_child_samples': 20,
            'n_jobs': -1,
            'random_state': 42,
            'verbosity': -1
        }
        if params:
            default_params.update(params)
        super().__init__("LightGBM", default_params)
        
    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None,
            early_stopping_rounds: int = 50) -> 'LightGBMModel':
        """Fits the LightGBM model.

        Trains the model with automatic objective selection and optional early
        stopping based on validation performance.

        Args:
            X (pd.DataFrame): The training features.
            y (pd.Series): The training target.
            X_val (Optional[pd.DataFrame]): The validation features for early
                stopping (optional).
            y_val (Optional[pd.Series]): The validation target for early stopping
                (optional).
            early_stopping_rounds (int): The number of rounds to stop if the
                validation metric doesn't improve.

        Returns:
            The self instance for method chaining.
        """
        self._determine_task_type(y)
        self.feature_names = X.columns.tolist()
        
        # Set objective based on task type
        if self.is_classification:
            if y.nunique() == 2:
                self.params['objective'] = 'binary'
            else:
                self.params['objective'] = 'multiclass'
                self.params['num_class'] = y.nunique()
        else:
            self.params['objective'] = 'regression'
            
        # Create model
        if self.is_classification:
            self.model = lgb.LGBMClassifier(**self.params)
        else:
            self.model = lgb.LGBMRegressor(**self.params)
            
        # Prepare eval set
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            
        # Fit model
        self.model.fit(
            X, y,
            eval_set=eval_set,
            callbacks=[lgb.early_stopping(early_stopping_rounds)] if eval_set else None,
            eval_metric='auc' if self.is_classification and y.nunique() == 2 else None
        )
        
        self.feature_importances_ = self.model.feature_importances_
        self.is_fitted = True
        return self


class CatBoostModel(BaseModel):
    """A CatBoost model implementation.

    A wrapper around CatBoost with automatic categorical feature handling.
    Excellent for datasets with many categorical features.

    Attributes:
        All attributes from BaseModel plus CatBoost-specific parameters.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initializes the CatBoostModel.

        Args:
            params (Optional[Dict[str, Any]]): The hyperparameters for the
                CatBoost model.
        """
        default_params = {
            'iterations': 100,
            'depth': 6,
            'learning_rate': 0.03,
            'l2_leaf_reg': 3,
            'random_seed': 42,
            'verbose': False
        }
        if params:
            default_params.update(params)
        super().__init__("CatBoost", default_params)
        
    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None,
            early_stopping_rounds: int = 50,
            cat_features: Optional[List[str]] = None) -> 'CatBoostModel':
        """Fits the CatBoost model.

        Trains the model with automatic handling of categorical features and
        optional early stopping.

        Args:
            X (pd.DataFrame): The training features.
            y (pd.Series): The training target.
            X_val (Optional[pd.DataFrame]): The validation features for early
                stopping (optional).
            y_val (Optional[pd.Series]): The validation target for early stopping
                (optional).
            early_stopping_rounds (int): The number of rounds to stop training if
                the validation metric doesn't improve.
            cat_features (Optional[List[str]]): A list of categorical feature
                names. If None, automatically detects object and category dtype
                columns.

        Returns:
            The self instance for method chaining.
        """
        self._determine_task_type(y)
        self.feature_names = X.columns.tolist()
        
        # Auto-detect categorical features if not provided
        if cat_features is None:
            cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
        # Create model
        if self.is_classification:
            self.model = cb.CatBoostClassifier(**self.params)
        else:
            self.model = cb.CatBoostRegressor(**self.params)
            
        # Prepare eval set
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = cb.Pool(X_val, y_val, cat_features=cat_features)
            
        # Fit model
        self.model.fit(
            X, y,
            cat_features=cat_features,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds if eval_set else None,
            verbose=False
        )
        
        self.feature_importances_ = self.model.feature_importances_
        self.is_fitted = True
        return self


class TabNetModel(BaseModel):
    """A TabNet model implementation.

    A wrapper around the TabNet neural network architecture specifically
    designed for tabular data. Uses attention mechanisms for interpretability.

    Attributes:
        All attributes from BaseModel plus TabNet-specific parameters.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None, device: str = 'auto'):
        """Initializes the TabNetModel.

        Args:
            params (Optional[Dict[str, Any]]): The hyperparameters for the
                TabNet model.
            device (str): The device to use for training, e.g., 'auto', 'cuda',
                or 'cpu'.
        """
        if not TABNET_AVAILABLE:
            raise ImportError("TabNet requires pytorch-tabnet. Install with: pip install pytorch-tabnet")
            
        default_params = {
            'n_d': 8,
            'n_a': 8,
            'n_steps': 3,
            'gamma': 1.3,
            'n_independent': 2,
            'n_shared': 2,
            'lambda_sparse': 1e-3,
            'momentum': 0.02,
            'clip_value': None,
            'optimizer_fn': 'adam',
            'optimizer_params': {'lr': 0.02},
            'scheduler_fn': None,
            'scheduler_params': {},
            'mask_type': 'sparsemax',
            'seed': 42
        }
        if params:
            default_params.update(params)
            
        # Set device
        if device == 'auto':
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        default_params['device_name'] = device
            
        super().__init__("TabNet", default_params)
        
    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None,
            max_epochs: int = 200,
            patience: int = 15,
            batch_size: int = 1024,
            virtual_batch_size: int = 128,
            drop_last: bool = False) -> 'TabNetModel':
        """Fits the TabNet model.

        Args:
            X (pd.DataFrame): The training features.
            y (pd.Series): The training target.
            X_val (Optional[pd.DataFrame]): The validation features for early
                stopping.
            y_val (Optional[pd.Series]): The validation target for early stopping.
            max_epochs (int): The maximum number of training epochs.
            patience (int): The early stopping patience.
            batch_size (int): The batch size for training.
            virtual_batch_size (int): The virtual batch size for Ghost Batch
                Normalization.
            drop_last (bool): Whether to drop the last incomplete batch.

        Returns:
            The self instance for method chaining.
        """
        self._determine_task_type(y)
        self.feature_names = X.columns.tolist()
        
        # Convert to numpy
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, pd.Series) else y
        
        if X_val is not None:
            X_val_np = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
            y_val_np = y_val.values if isinstance(y_val, pd.Series) else y_val
        
        # Create model
        if self.is_classification:
            self.model = TabNetClassifier(**self.params)
        else:
            self.model = TabNetRegressor(**self.params)
            
        # Prepare eval set
        eval_set = None
        eval_name = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val_np, y_val_np)]
            eval_name = ['valid']
            
        # Fit model
        self.model.fit(
            X_np, y_np,
            eval_set=eval_set,
            eval_name=eval_name,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
            drop_last=drop_last
        )
        
        self.feature_importances_ = self.model.feature_importances_
        self.is_fitted = True
        return self


class RandomForestModel(BaseModel):
    """A Random Forest model implementation.

    A wrapper around scikit-learn's Random Forest with a consistent interface.
    A good baseline model with built-in feature importance.

    Attributes:
        All attributes from BaseModel plus Random Forest parameters.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initializes the RandomForestModel.

        Args:
            params (Optional[Dict[str, Any]]): The hyperparameters for the
                Random Forest model.
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'n_jobs': -1,
            'random_state': 42
        }
        if params:
            default_params.update(params)
        super().__init__("RandomForest", default_params)
        
    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None) -> 'RandomForestModel':
        """Fits the Random Forest model.

        Trains a Random Forest model. The validation set is ignored as Random
        Forest doesn't support early stopping.

        Args:
            X (pd.DataFrame): The training features.
            y (pd.Series): The training target.
            X_val (Optional[pd.DataFrame]): Ignored (kept for API consistency).
            y_val (Optional[pd.Series]): Ignored (kept for API consistency).

        Returns:
            The self instance for method chaining.
        """
        self._determine_task_type(y)
        self.feature_names = X.columns.tolist()
        
        # Create model
        if self.is_classification:
            self.model = RandomForestClassifier(**self.params)
        else:
            self.model = RandomForestRegressor(**self.params)
            
        # Fit model
        self.model.fit(X, y)
        
        self.feature_importances_ = self.model.feature_importances_
        self.is_fitted = True
        return self


class VotingEnsemble(BaseModel):
    """A voting ensemble model implementation.

    Combines predictions from multiple models using a voting strategy. Supports
    both hard and soft voting for classification.
    """
    
    def __init__(self, models: List[BaseModel], voting: str = 'soft', 
                 weights: Optional[List[float]] = None):
        """Initializes the VotingEnsemble.

        Args:
            models (List[BaseModel]): A list of model instances to ensemble.
            voting (str): The voting strategy, either 'hard' or 'soft'
                (classification only).
            weights (Optional[List[float]]): The model weights for weighted
                voting.
        """
        self.models = models
        self.voting = voting
        self.weights = weights
        super().__init__("VotingEnsemble", {})
        
    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None) -> 'VotingEnsemble':
        """Fits all models in the ensemble.

        Args:
            X (pd.DataFrame): The training features.
            y (pd.Series): The training target.
            X_val (Optional[pd.DataFrame]): The validation features (optional).
            y_val (Optional[pd.Series]): The validation target (optional).

        Returns:
            The self instance for method chaining.
        """
        self._determine_task_type(y)
        self.feature_names = X.columns.tolist()
        
        # Create sklearn voting model
        estimators = [(f"model_{i}", model.model if hasattr(model, 'model') else model) 
                      for i, model in enumerate(self.models)]
        
        if self.is_classification:
            self.model = VotingClassifier(
                estimators=estimators,
                voting=self.voting,
                weights=self.weights
            )
        else:
            self.model = VotingRegressor(
                estimators=estimators,
                weights=self.weights
            )
            
        # Fit each model first if not fitted
        for model in self.models:
            if not hasattr(model, 'is_fitted') or not model.is_fitted:
                model.fit(X, y, X_val, y_val)
                
        # Fit voting model
        self.model.fit(X, y)
        self.is_fitted = True
        
        # Average feature importances if available
        importances = []
        for model in self.models:
            if hasattr(model, 'feature_importances_'):
                importances.append(model.feature_importances_)
        if importances:
            self.feature_importances_ = np.mean(importances, axis=0)
            
        return self


class RidgeModel(BaseModel):
    """A Ridge regression model implementation.

    Linear regression with L2 regularization. Good for high-dimensional data and
    when multicollinearity is present.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initializes the RidgeModel.

        Args:
            params (Optional[Dict[str, Any]]): The hyperparameters for the
                Ridge model.
        """
        default_params = {
            'alpha': 1.0,
            'fit_intercept': True,
            'normalize': False,
            'max_iter': None,
            'tol': 0.001,
            'solver': 'auto',
            'random_state': 42
        }
        if params:
            default_params.update(params)
        super().__init__("Ridge", default_params)
        
    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None) -> 'RidgeModel':
        """Fits the Ridge model.

        Args:
            X (pd.DataFrame): The training features.
            y (pd.Series): The training target.
            X_val (Optional[pd.DataFrame]): Ignored (kept for API consistency).
            y_val (Optional[pd.Series]): Ignored (kept for API consistency).

        Returns:
            The self instance for method chaining.
        """
        self._determine_task_type(y)
        self.feature_names = X.columns.tolist()
        
        # Ridge is regression only
        self.model = Ridge(**self.params)
        self.model.fit(X, y)
        
        # Set feature importances as absolute coefficients
        self.feature_importances_ = np.abs(self.model.coef_)
        self.is_fitted = True
        return self


class LinearModel(BaseModel):
    """A linear regression model implementation.

    A simple linear regression model without regularization. Good baseline for
    regression tasks.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initializes the LinearModel.

        Args:
            params (Optional[Dict[str, Any]]): The hyperparameters for the
                linear model.
        """
        default_params = {
            'fit_intercept': True,
            'normalize': False,
            'n_jobs': -1
        }
        if params:
            default_params.update(params)
        super().__init__("LinearRegression", default_params)
        
    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None) -> 'LinearModel':
        """Fits the linear model.

        Args:
            X (pd.DataFrame): The training features.
            y (pd.Series): The training target.
            X_val (Optional[pd.DataFrame]): Ignored (kept for API consistency).
            y_val (Optional[pd.Series]): Ignored (kept for API consistency).

        Returns:
            The self instance for method chaining.
        """
        self._determine_task_type(y)
        self.feature_names = X.columns.tolist()
        
        self.model = LinearRegression(**self.params)
        self.model.fit(X, y)
        
        # Set feature importances as absolute coefficients
        self.feature_importances_ = np.abs(self.model.coef_)
        self.is_fitted = True
        return self


class ModelTrainer:
    """Handles model training and hyperparameter optimization.

    A high-level interface for training multiple models, comparing performance,
    and optimizing hyperparameters using Optuna.

    Attributes:
        task_type (str): The type of machine learning task, either
            'classification' or 'regression' (auto-detected if None).
        models (Dict[str, BaseModel]): A dictionary of trained models by name.
        best_model (BaseModel): A reference to the best performing model.
        optimization_history (Dict[str, optuna.study.Study]): A dictionary of
            Optuna study objects for each optimized model.
    """
    
    def __init__(self, task_type: Optional[str] = None, 
                 metric: Optional[Union[str, Callable]] = None,
                 cv_strategy: str = 'stratified',
                 n_folds: int = 5,
                 random_state: int = 42,
                 gpu: bool = False):
        """Initializes the model trainer.

        Args:
            task_type (Optional[str]): Either 'classification' or 'regression'.
                If None, automatically detected based on the target variable.
            metric (Optional[Union[str, Callable]]): The evaluation metric. Can
                be a string or a custom function. Defaults to 'roc_auc' for
                classification and 'rmse' for regression.
            cv_strategy (str): The cross-validation strategy. Options are
                'stratified', 'kfold', 'group', 'repeated',
                'repeated_stratified', and 'timeseries'.
            n_folds (int): The number of CV folds.
            random_state (int): The random seed for reproducibility.
            gpu (bool): Whether to use GPU acceleration when available.
        """
        self.task_type = task_type
        self.metric = metric
        self.cv_strategy = cv_strategy
        self.n_folds = n_folds
        self.random_state = random_state
        self.gpu = gpu
        self.models = {}
        self.best_model = None
        self.optimization_history = {}
        self.cv_scores = {}  # Store CV scores for each model
        
    def train_model(self, 
                   model_type: str,
                   X_train: pd.DataFrame,
                   y_train: pd.Series,
                   X_val: Optional[pd.DataFrame] = None,
                   y_val: Optional[pd.Series] = None,
                   params: Optional[Dict[str, Any]] = None,
                   optimize_hyperparams: bool = False,
                   n_trials: int = 100) -> BaseModel:
        """Trains a single model.

        Trains a specified model type with optional hyperparameter optimization
        using Optuna.

        Args:
            model_type (str): The type of model to train. Options are 'xgboost',
                'lightgbm', 'catboost', and 'random_forest'.
            X_train (pd.DataFrame): The training features.
            y_train (pd.Series): The training target.
            X_val (Optional[pd.DataFrame]): The validation features (optional but
                recommended).
            y_val (Optional[pd.Series]): The validation target (optional but
                recommended).
            params (Optional[Dict[str, Any]]): The custom model parameters. If
                None, uses defaults.
            optimize_hyperparams (bool): Whether to run Optuna hyperparameter
                search.
            n_trials (int): The number of Optuna optimization trials.

        Returns:
            A trained model instance.

        Raises:
            ValueError: If model_type is not recognized.
        """
        # Auto-detect task type
        if self.task_type is None:
            self.task_type = 'classification' if y_train.nunique() < 100 else 'regression'
            
        # Get model class
        model_classes = {
            'xgboost': XGBoostModel,
            'lightgbm': LightGBMModel,
            'catboost': CatBoostModel,
            'random_forest': RandomForestModel,
            'tabnet': TabNetModel,
            'ridge': RidgeModel,
            'linear': LinearModel
        }
        
        if model_type not in model_classes:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(model_classes.keys())}")
            
        # Optimize hyperparameters if requested
        if optimize_hyperparams:
            logger.info(f"Optimizing hyperparameters for {model_type}")
            params = self._optimize_hyperparams(
                model_type, X_train, y_train, X_val, y_val, n_trials
            )
            
        # Create and train model
        model_class = model_classes[model_type]
        model = model_class(params)
        
        if model_type == 'catboost':
            model.fit(X_train, y_train, X_val, y_val, cat_features=None)
        elif model_type == 'random_forest':
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train, X_val, y_val)
            
        self.models[model_type] = model
        logger.info(f"Trained {model_type} model")
        
        return model
        
    def train_all_models(self,
                        X_train: pd.DataFrame,
                        y_train: pd.Series,
                        X_val: Optional[pd.DataFrame] = None,
                        y_val: Optional[pd.Series] = None,
                        model_types: Optional[List[str]] = None,
                        optimize_hyperparams: bool = False) -> Dict[str, BaseModel]:
        """Trains multiple models.

        Trains all specified model types and automatically selects the best one
        based on validation performance.

        Args:
            X_train (pd.DataFrame): The training features.
            y_train (pd.Series): The training target.
            X_val (Optional[pd.DataFrame]): The validation features (optional).
            y_val (Optional[pd.Series]): The validation target (optional).
            model_types (Optional[List[str]]): A list of model types to train.
                If None, trains all: ['xgboost', 'lightgbm', 'catboost',
                'random_forest'].
            optimize_hyperparams (bool): Whether to optimize each model's
                hyperparameters.

        Returns:
            A dictionary mapping model type to the trained model.
        """
        if model_types is None:
            model_types = ['xgboost', 'lightgbm', 'catboost', 'random_forest']
            
        for model_type in model_types:
            self.train_model(
                model_type, X_train, y_train, X_val, y_val,
                optimize_hyperparams=optimize_hyperparams
            )
            
        # Evaluate models to find best
        if X_val is not None and y_val is not None:
            best_score = -float('inf')
            for name, model in self.models.items():
                score = self._evaluate_model(model, X_val, y_val)
                if score > best_score:
                    best_score = score
                    self.best_model = model
                    
        return self.models
        
    def _evaluate_model(self, model: BaseModel, X: pd.DataFrame, y: pd.Series, 
                       weights: Optional[np.ndarray] = None) -> float:
        """Evaluates model performance.

        Computes the appropriate metric based on the task type and metric setting.

        Args:
            model (BaseModel): The trained model to evaluate.
            X (pd.DataFrame): The features to evaluate on.
            y (pd.Series): The true target values.
            weights (Optional[np.ndarray]): Sample weights for weighted metrics.

        Returns:
            The performance score (higher is better for all metrics).
        """
        predictions = model.predict(X)
        
        # Handle custom metric
        if callable(self.metric):
            return self.metric(y, predictions)
        
        # Default metrics if not specified
        if self.metric is None:
            if self.task_type == 'classification':
                if y.nunique() == 2:
                    proba = model.predict_proba(X)[:, 1]
                    return roc_auc_score(y, proba)
                else:
                    return accuracy_score(y, predictions)
            else:
                return -mean_squared_error(y, predictions)
        
        # Handle string metrics
        if self.task_type == 'classification':
            if self.metric == 'roc_auc':
                if y.nunique() == 2:
                    proba = model.predict_proba(X)[:, 1]
                    return roc_auc_score(y, proba)
                else:
                    # Multiclass ROC-AUC
                    proba = model.predict_proba(X)
                    return roc_auc_score(y, proba, multi_class='ovr')
            elif self.metric == 'accuracy':
                return accuracy_score(y, predictions)
            elif self.metric == 'f1':
                return f1_score(y, predictions, average='weighted' if y.nunique() > 2 else 'binary')
            elif self.metric == 'precision':
                return precision_score(y, predictions, average='weighted' if y.nunique() > 2 else 'binary')
            elif self.metric == 'recall':
                return recall_score(y, predictions, average='weighted' if y.nunique() > 2 else 'binary')
        else:
            # Regression metrics
            if self.metric == 'rmse':
                return -np.sqrt(mean_squared_error(y, predictions, sample_weight=weights))
            elif self.metric == 'mse':
                return -mean_squared_error(y, predictions, sample_weight=weights)
            elif self.metric == 'mae':
                return -mean_absolute_error(y, predictions, sample_weight=weights)
            elif self.metric == 'r2':
                return r2_score(y, predictions, sample_weight=weights)
            elif self.metric == 'mape':
                return -mean_absolute_percentage_error(y, predictions)
            elif self.metric == 'rmsle':
                return -np.sqrt(mean_squared_log_error(np.maximum(y, 0), np.maximum(predictions, 0)))
            
        raise ValueError(f"Unknown metric: {self.metric}")
            
    def _optimize_hyperparams(self,
                             model_type: str,
                             X_train: pd.DataFrame,
                             y_train: pd.Series,
                             X_val: Optional[pd.DataFrame],
                             y_val: Optional[pd.Series],
                             n_trials: int) -> Dict[str, Any]:
        """Optimizes hyperparameters using Optuna.

        Runs Bayesian optimization to find the best hyperparameters for the
        specified model type.

        Args:
            model_type (str): The type of model to optimize.
            X_train (pd.DataFrame): The training features.
            y_train (pd.Series): The training target.
            X_val (Optional[pd.DataFrame]): The validation features (if None,
                uses cross-validation).
            y_val (Optional[pd.Series]): The validation target.
            n_trials (int): The number of optimization trials.

        Returns:
            A dictionary of optimized hyperparameters.
        """
        def objective(trial):
            # Define search space based on model type
            if model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                }
            elif model_type == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                }
            elif model_type == 'catboost':
                params = {
                    'iterations': trial.suggest_int('iterations', 100, 1000),
                    'depth': trial.suggest_int('depth', 4, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                }
            else:  # random_forest
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 50),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                }
                
            # Train model with suggested params
            model = self.train_model(
                model_type, X_train, y_train, X_val, y_val, params
            )
            
            # Evaluate
            if X_val is not None and y_val is not None:
                return self._evaluate_model(model, X_val, y_val)
            else:
                # Use cross-validation
                cv = StratifiedKFold(5) if self.task_type == 'classification' else KFold(5)
                scores = cross_val_score(
                    model.model, X_train, y_train, cv=cv,
                    scoring='roc_auc' if self.task_type == 'classification' else 'neg_mean_squared_error'
                )
                return scores.mean()
                
        # Create study and optimize
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.optimization_history[model_type] = study
        return study.best_params
