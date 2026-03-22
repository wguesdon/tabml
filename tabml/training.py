"""Enhanced training utilities with callbacks and monitoring.

This module provides advanced training capabilities with callbacks,
early stopping, and comprehensive monitoring.
"""

from typing import Any, Dict, List, Optional, Union, Callable
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    StratifiedKFold, KFold, GroupKFold, RepeatedKFold, 
    RepeatedStratifiedKFold, TimeSeriesSplit
)
from loguru import logger
from tqdm import tqdm
import time
from dataclasses import dataclass
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    epoch: int
    train_score: float
    val_score: float
    duration: float
    timestamp: datetime
    
    def to_dict(self):
        return {
            'epoch': self.epoch,
            'train_score': self.train_score,
            'val_score': self.val_score,
            'duration': self.duration,
            'timestamp': self.timestamp.isoformat()
        }


class Callback:
    """Base class for training callbacks."""
    
    def on_train_begin(self, trainer, **kwargs):
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, trainer, **kwargs):
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, trainer, epoch, **kwargs):
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        """Called at the end of each epoch."""
        pass
    
    def on_fold_begin(self, trainer, fold, **kwargs):
        """Called at the beginning of each fold."""
        pass
    
    def on_fold_end(self, trainer, fold, metrics, **kwargs):
        """Called at the end of each fold."""
        pass


class EarlyStoppingCallback(Callback):
    """Early stopping callback to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0001, 
                 mode: str = 'max', restore_best_weights: bool = True):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' for metric optimization direction
            restore_best_weights: Whether to restore best model weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.stopped_epoch = 0
        
    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        """Check if training should stop."""
        current_score = metrics.val_score
        
        if self.best_score is None:
            self.best_score = current_score
            if self.restore_best_weights:
                self.best_weights = trainer.get_model_weights()
        else:
            if self.mode == 'max':
                improved = current_score > self.best_score + self.min_delta
            else:
                improved = current_score < self.best_score - self.min_delta
                
            if improved:
                self.best_score = current_score
                self.counter = 0
                if self.restore_best_weights:
                    self.best_weights = trainer.get_model_weights()
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.stopped_epoch = epoch
                    trainer.stop_training = True
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    
                    if self.restore_best_weights and self.best_weights:
                        trainer.set_model_weights(self.best_weights)
                        logger.info("Restored best model weights")


class ModelCheckpointCallback(Callback):
    """Save model checkpoints during training."""
    
    def __init__(self, filepath: str, monitor: str = 'val_score', 
                 mode: str = 'max', save_best_only: bool = True):
        """Initialize model checkpoint.
        
        Args:
            filepath: Path to save model checkpoints
            monitor: Metric to monitor
            mode: 'min' or 'max' for metric optimization
            save_best_only: Whether to save only best model
        """
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score = None
        
    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        """Save model checkpoint if needed."""
        current_score = getattr(metrics, self.monitor, metrics.val_score)
        
        save_model = False
        if not self.save_best_only:
            save_model = True
        elif self.best_score is None:
            save_model = True
            self.best_score = current_score
        else:
            if self.mode == 'max' and current_score > self.best_score:
                save_model = True
                self.best_score = current_score
            elif self.mode == 'min' and current_score < self.best_score:
                save_model = True
                self.best_score = current_score
                
        if save_model:
            filepath = self.filepath.format(epoch=epoch, **metrics.to_dict())
            trainer.save_model(filepath)
            logger.info(f"Model checkpoint saved to {filepath}")


class ProgressBarCallback(Callback):
    """Display progress bar during training."""
    
    def __init__(self):
        self.pbar = None
        
    def on_train_begin(self, trainer, **kwargs):
        """Initialize progress bar."""
        total_steps = kwargs.get('total_epochs', 100)
        self.pbar = tqdm(total=total_steps, desc="Training")
        
    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        """Update progress bar."""
        if self.pbar:
            self.pbar.update(1)
            self.pbar.set_postfix({
                'train': f"{metrics.train_score:.4f}",
                'val': f"{metrics.val_score:.4f}"
            })
            
    def on_train_end(self, trainer, **kwargs):
        """Close progress bar."""
        if self.pbar:
            self.pbar.close()


class TensorBoardCallback(Callback):
    """Log metrics to TensorBoard."""
    
    def __init__(self, log_dir: str = './logs'):
        """Initialize TensorBoard logging.
        
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = log_dir
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            self.enabled = True
        except ImportError:
            logger.warning("TensorBoard not available. Install with: pip install tensorboard")
            self.enabled = False
            
    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        """Log metrics to TensorBoard."""
        if self.enabled and self.writer:
            self.writer.add_scalar('Loss/train', metrics.train_score, epoch)
            self.writer.add_scalar('Loss/val', metrics.val_score, epoch)
            self.writer.add_scalar('Time/epoch', metrics.duration, epoch)
            
    def on_train_end(self, trainer, **kwargs):
        """Close TensorBoard writer."""
        if self.enabled and self.writer:
            self.writer.close()


class WandbCallback(Callback):
    """Log metrics to Weights & Biases."""
    
    def __init__(self, project: str, name: Optional[str] = None, config: Optional[Dict] = None):
        """Initialize W&B logging.
        
        Args:
            project: W&B project name
            name: Run name
            config: Configuration dictionary
        """
        try:
            import wandb
            self.wandb = wandb
            self.run = wandb.init(project=project, name=name, config=config)
            self.enabled = True
        except ImportError:
            logger.warning("W&B not available. Install with: pip install wandb")
            self.enabled = False
            
    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        """Log metrics to W&B."""
        if self.enabled:
            self.wandb.log({
                'epoch': epoch,
                'train_score': metrics.train_score,
                'val_score': metrics.val_score,
                'duration': metrics.duration
            })
            
    def on_train_end(self, trainer, **kwargs):
        """Finish W&B run."""
        if self.enabled:
            self.wandb.finish()


class MLflowCallback(Callback):
    """Log metrics, models, and datasets to MLflow."""
    
    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        log_models: bool = True,
        log_datasets: bool = True,
        registered_model_name: Optional[str] = None,
        **kwargs
    ):
        """Initialize MLflow logging.
        
        Args:
            experiment_name: Name of MLflow experiment
            run_name: Name for this run
            tracking_uri: MLflow tracking server URI
            tags: Dictionary of tags for the run
            log_models: Whether to log models
            log_datasets: Whether to log dataset info
            registered_model_name: Name to register model under
            **kwargs: Additional parameters for MLflow
        """
        try:
            import mlflow
            import mlflow.sklearn
            self.mlflow = mlflow
            
            # Set tracking URI if provided
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            
            # Create or get experiment
            self.experiment = mlflow.set_experiment(experiment_name)
            
            # Start run
            self.run = mlflow.start_run(
                run_name=run_name,
                experiment_id=self.experiment.experiment_id,
                tags=tags or {}
            )
            
            self.log_models = log_models
            self.log_datasets = log_datasets
            self.registered_model_name = registered_model_name
            self.enabled = True
            
            logger.info(f"MLflow run started: {self.run.info.run_id}")
            
        except ImportError:
            logger.warning("MLflow not available. Install with: pip install mlflow")
            self.enabled = False
        except Exception as e:
            logger.warning(f"Failed to initialize MLflow: {e}")
            self.enabled = False
            
    def on_train_begin(self, trainer, **kwargs):
        """Log training configuration."""
        if self.enabled and hasattr(trainer, 'current_model'):
            # Log model parameters
            if hasattr(trainer.current_model, 'get_params'):
                params = trainer.current_model.get_params()
                for key, value in params.items():
                    # MLflow has limits on param value length
                    if isinstance(value, (list, dict)):
                        value = str(value)[:250]
                    self.mlflow.log_param(f"model_{key}", value)
            
            # Log model type
            self.mlflow.log_param("model_type", type(trainer.current_model).__name__)
            
    def on_epoch_end(self, trainer, epoch, metrics, **kwargs):
        """Log metrics to MLflow."""
        if self.enabled:
            self.mlflow.log_metric("train_score", metrics.train_score, step=epoch)
            self.mlflow.log_metric("val_score", metrics.val_score, step=epoch)
            self.mlflow.log_metric("epoch_duration", metrics.duration, step=epoch)
            
    def on_train_end(self, trainer, **kwargs):
        """Log final model and close MLflow run."""
        if self.enabled:
            # Log final model if available
            if self.log_models and hasattr(trainer, 'current_model'):
                try:
                    # Log model
                    self.mlflow.sklearn.log_model(
                        trainer.current_model,
                        "model",
                        registered_model_name=self.registered_model_name
                    )
                    logger.info("Model logged to MLflow")
                    
                except Exception as e:
                    logger.warning(f"Failed to log model to MLflow: {e}")
            
            # End the run
            self.mlflow.end_run()
            
    def log_dataset(self, X_train, y_train, X_val=None, y_val=None, X_test=None):
        """Log dataset information to MLflow.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            X_test: Test features (optional)
        """
        if self.enabled and self.log_datasets:
            import pandas as pd
            
            # Log dataset shapes
            self.mlflow.log_param("dataset_train_shape", str(X_train.shape))
            self.mlflow.log_param("dataset_train_samples", len(X_train))
            self.mlflow.log_param("dataset_train_features", X_train.shape[1])
            
            if X_val is not None:
                self.mlflow.log_param("dataset_val_shape", str(X_val.shape))
                self.mlflow.log_param("dataset_val_samples", len(X_val))
                
            if X_test is not None:
                self.mlflow.log_param("dataset_test_shape", str(X_test.shape))
                self.mlflow.log_param("dataset_test_samples", len(X_test))
            
            # Log feature names
            if isinstance(X_train, pd.DataFrame):
                # MLflow param limit is 500 chars
                feature_names = str(list(X_train.columns))[:500]
                self.mlflow.log_param("dataset_features", feature_names)
                
                # Log basic statistics as metrics
                stats = X_train.describe()
                for col in stats.columns[:10]:  # Limit to first 10 features
                    for stat in ['mean', 'std', 'min', 'max']:
                        if stat in stats.index:
                            self.mlflow.log_metric(f"data_{col}_{stat}", stats.loc[stat, col])
                
            # Log target distribution
            if y_train is not None:
                if isinstance(y_train, pd.Series):
                    value_counts = y_train.value_counts()
                    for val, count in value_counts.head(10).items():  # Top 10 classes
                        self.mlflow.log_metric(f"target_class_{val}_count", count)
                else:
                    import numpy as np
                    unique, counts = np.unique(y_train, return_counts=True)
                    for val, count in zip(unique[:10], counts[:10]):  # First 10 classes
                        self.mlflow.log_metric(f"target_class_{val}_count", count)


class EnhancedTrainer:
    """Enhanced trainer with callbacks and monitoring.
    
    Example:
        >>> trainer = EnhancedTrainer(
        ...     callbacks=[
        ...         EarlyStoppingCallback(patience=10),
        ...         ModelCheckpointCallback('./checkpoints/model_{epoch}.pkl'),
        ...         ProgressBarCallback()
        ...     ]
        ... )
        >>> trainer.fit(model, X_train, y_train, X_val, y_val)
    """
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        """Initialize enhanced trainer.
        
        Args:
            callbacks: List of callback instances
        """
        self.callbacks = callbacks or []
        self.stop_training = False
        self.history = []
        self.current_model = None
        
    def add_callback(self, callback: Callback):
        """Add a callback to the trainer."""
        self.callbacks.append(callback)
        
    def _call_callbacks(self, method: str, **kwargs):
        """Call a specific method on all callbacks."""
        for callback in self.callbacks:
            if hasattr(callback, method):
                getattr(callback, method)(self, **kwargs)
                
    def fit(self, model, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
            epochs: int = 100, batch_size: Optional[int] = None):
        """Train model with callbacks.
        
        Args:
            model: Model instance to train
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Trained model
        """
        self.current_model = model
        self.stop_training = False
        
        # Call on_train_begin
        self._call_callbacks('on_train_begin', total_epochs=epochs)
        
        for epoch in range(epochs):
            if self.stop_training:
                break
                
            # Call on_epoch_begin
            self._call_callbacks('on_epoch_begin', epoch=epoch)
            
            # Training step
            start_time = time.time()
            
            # Train model (simplified - actual implementation depends on model type)
            model.partial_fit(X_train, y_train) if hasattr(model, 'partial_fit') else None
            
            # Calculate metrics
            train_score = self._evaluate(model, X_train, y_train)
            val_score = self._evaluate(model, X_val, y_val) if X_val is not None else train_score
            
            duration = time.time() - start_time
            
            metrics = TrainingMetrics(
                epoch=epoch,
                train_score=train_score,
                val_score=val_score,
                duration=duration,
                timestamp=datetime.now()
            )
            
            self.history.append(metrics)
            
            # Call on_epoch_end
            self._call_callbacks('on_epoch_end', epoch=epoch, metrics=metrics)
            
        # Call on_train_end
        self._call_callbacks('on_train_end')
        
        return model
    
    def fit_cv(self, model_class, X: pd.DataFrame, y: pd.Series,
               cv_strategy: str = 'stratified', n_folds: int = 5,
               groups: Optional[pd.Series] = None, **model_kwargs):
        """Train with cross-validation.
        
        Args:
            model_class: Model class to instantiate
            X: Features
            y: Target
            cv_strategy: CV strategy name
            n_folds: Number of folds
            groups: Group labels for GroupKFold
            **model_kwargs: Arguments for model initialization
            
        Returns:
            List of trained models
        """
        # Get CV splitter
        cv = self._get_cv_splitter(cv_strategy, n_folds, y)
        
        models = []
        oof_predictions = np.zeros(len(y))
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
            # Call on_fold_begin
            self._call_callbacks('on_fold_begin', fold=fold)
            
            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Create and train model
            model = model_class(**model_kwargs)
            model = self.fit(model, X_train, y_train, X_val, y_val)
            
            models.append(model)
            
            # Store OOF predictions
            oof_predictions[val_idx] = model.predict(X_val)
            
            # Calculate fold metrics
            fold_score = self._evaluate(model, X_val, y_val)
            
            metrics = TrainingMetrics(
                epoch=fold,
                train_score=self._evaluate(model, X_train, y_train),
                val_score=fold_score,
                duration=0,
                timestamp=datetime.now()
            )
            
            # Call on_fold_end
            self._call_callbacks('on_fold_end', fold=fold, metrics=metrics)
            
        return models, oof_predictions
    
    def _get_cv_splitter(self, strategy: str, n_folds: int, y: pd.Series):
        """Get appropriate CV splitter based on strategy."""
        if strategy == 'stratified':
            return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        elif strategy == 'kfold':
            return KFold(n_splits=n_folds, shuffle=True, random_state=42)
        elif strategy == 'group':
            return GroupKFold(n_splits=n_folds)
        elif strategy == 'repeated':
            return RepeatedKFold(n_splits=n_folds, n_repeats=2, random_state=42)
        elif strategy == 'repeated_stratified':
            return RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=2, random_state=42)
        elif strategy == 'timeseries':
            return TimeSeriesSplit(n_splits=n_folds)
        else:
            raise ValueError(f"Unknown CV strategy: {strategy}")
    
    def _evaluate(self, model, X: pd.DataFrame, y: pd.Series) -> float:
        """Evaluate model performance."""
        if X is None or y is None:
            return 0.0
        
        predictions = model.predict(X)
        # Simplified - actual metric calculation depends on task type
        from sklearn.metrics import accuracy_score, mean_squared_error
        
        if len(np.unique(y)) < 100:  # Classification
            return accuracy_score(y, predictions.round())
        else:  # Regression
            return -mean_squared_error(y, predictions)
    
    def get_model_weights(self):
        """Get current model weights/state."""
        if hasattr(self.current_model, 'get_params'):
            return self.current_model.get_params()
        return None
    
    def set_model_weights(self, weights):
        """Set model weights/state."""
        if hasattr(self.current_model, 'set_params'):
            self.current_model.set_params(**weights)
    
    def save_model(self, filepath: str):
        """Save model to file."""
        import joblib
        joblib.dump(self.current_model, filepath)
    
    def save_history(self, filepath: str):
        """Save training history to JSON."""
        history_dict = [m.to_dict() for m in self.history]
        with open(filepath, 'w') as f:
            json.dump(history_dict, f, indent=2)