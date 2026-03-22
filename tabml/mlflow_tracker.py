"""MLflow integration for TabML model tracking and experiment management."""

import os
import json
import hashlib
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature


class MLflowTracker:
    """Track experiments, models, and datasets with MLflow."""
    
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: URI of tracking server (e.g., "http://localhost:5000")
            artifact_location: Storage location for artifacts
            tags: Default tags to apply to all runs
        """
        self.experiment_name = experiment_name
        self.tags = tags or {}
        
        # Set tracking URI if provided
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        elif os.getenv("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
            
        # Create or get experiment
        # Note: artifact_location parameter was deprecated in MLflow 2.9+
        # It's now set at the server level or via tracking URI
        try:
            # Try with artifact_location for older versions
            self.experiment = mlflow.set_experiment(
                experiment_name,
                artifact_location=artifact_location
            )
        except TypeError:
            # Fallback for newer MLflow versions that don't support artifact_location
            self.experiment = mlflow.set_experiment(experiment_name)
        self.experiment_id = self.experiment.experiment_id
        
        # Initialize client
        self.client = MlflowClient()
        
        # Store current run
        self.current_run = None
        
    def start_run(
        self,
        run_name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False,
    ) -> str:
        """Start a new MLflow run.
        
        Args:
            run_name: Name for the run
            description: Description of the run
            tags: Additional tags for this run
            nested: Whether this is a nested run
            
        Returns:
            Run ID
        """
        # Combine default and run-specific tags
        all_tags = {**self.tags, **(tags or {})}
        
        # Start run
        self.current_run = mlflow.start_run(
            run_name=run_name,
            experiment_id=self.experiment_id,
            description=description,
            tags=all_tags,
            nested=nested,
        )
        
        return self.current_run.info.run_id
        
    def end_run(self, status: str = "FINISHED"):
        """End the current run.
        
        Args:
            status: Status of the run (FINISHED, FAILED, KILLED)
        """
        if self.current_run:
            mlflow.end_run(status=status)
            self.current_run = None
            
    def log_params(self, params: Dict[str, Any]):
        """Log parameters.
        
        Args:
            params: Dictionary of parameters to log
        """
        for key, value in params.items():
            # MLflow has limits on param value length
            if isinstance(value, (list, dict)):
                value = json.dumps(value)[:250]
            mlflow.log_param(key, value)
            
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ):
        """Log metrics.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Step number (for iterative training)
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
            
    def log_dataset(
        self,
        df: pd.DataFrame,
        name: str,
        version: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """Log dataset information.
        
        Args:
            df: DataFrame to log
            name: Name of the dataset
            version: Version of the dataset
            description: Description of the dataset
        """
        # Calculate dataset hash for versioning
        if version is None:
            df_bytes = pd.util.hash_pandas_object(df).values.tobytes()
            version = hashlib.sha256(df_bytes).hexdigest()[:8]
            
        # Log dataset metadata
        dataset_info = {
            f"dataset_{name}_shape": str(df.shape),
            f"dataset_{name}_columns": len(df.columns),
            f"dataset_{name}_rows": len(df),
            f"dataset_{name}_version": version,
        }
        
        for key, value in dataset_info.items():
            mlflow.log_param(key, value)
            
        # Log dataset statistics
        if df.select_dtypes(include=[np.number]).columns.tolist():
            stats = df.describe().to_dict()
            for col, col_stats in stats.items():
                for stat_name, stat_value in col_stats.items():
                    mlflow.log_metric(f"{name}_{col}_{stat_name}", stat_value)
                    
        # Save dataset sample as artifact
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.head(1000).to_csv(f, index=False)
            mlflow.log_artifact(f.name, f"datasets/{name}")
            os.unlink(f.name)
            
        # Log description if provided
        if description:
            mlflow.log_param(f"dataset_{name}_description", description[:250])
            
    def log_model(
        self,
        model: Any,
        model_name: str,
        input_example: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        signature: Optional[Any] = None,
        registered_model_name: Optional[str] = None,
        await_registration: bool = False,
    ):
        """Log a model to MLflow.
        
        Args:
            model: Model object to log
            model_name: Name for the model artifact
            input_example: Example input for model signature
            signature: Model signature (inferred if not provided)
            registered_model_name: Name to register model under
            await_registration: Whether to wait for registration to complete
        """
        # Infer signature if not provided
        if signature is None and input_example is not None:
            if hasattr(model, 'predict'):
                predictions = model.predict(input_example)
                signature = infer_signature(input_example, predictions)
                
        # Determine model flavor
        model_class = model.__class__.__name__
        
        # Log model with appropriate flavor
        if 'XGBoost' in model_class:
            mlflow.xgboost.log_model(
                model,
                model_name,
                input_example=input_example,
                signature=signature,
                registered_model_name=registered_model_name,
                await_registration_for=await_registration,
            )
        elif 'LightGBM' in model_class or 'LGBMClassifier' in model_class or 'LGBMRegressor' in model_class:
            mlflow.lightgbm.log_model(
                model,
                model_name,
                input_example=input_example,
                signature=signature,
                registered_model_name=registered_model_name,
                await_registration_for=await_registration,
            )
        elif 'CatBoost' in model_class:
            # CatBoost uses sklearn flavor
            mlflow.sklearn.log_model(
                model,
                model_name,
                input_example=input_example,
                signature=signature,
                registered_model_name=registered_model_name,
                await_registration_for=await_registration,
            )
        else:
            # Default to sklearn flavor
            mlflow.sklearn.log_model(
                model,
                model_name,
                input_example=input_example,
                signature=signature,
                registered_model_name=registered_model_name,
                await_registration_for=await_registration,
            )
            
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact file or directory.
        
        Args:
            local_path: Path to local file or directory
            artifact_path: Destination path within run's artifact directory
        """
        if os.path.isdir(local_path):
            mlflow.log_artifacts(local_path, artifact_path)
        else:
            mlflow.log_artifact(local_path, artifact_path)
            
    def log_figure(
        self,
        figure: Any,
        artifact_file: str,
        artifact_path: Optional[str] = None,
    ):
        """Log a matplotlib figure.
        
        Args:
            figure: Matplotlib figure object
            artifact_file: Name for the artifact file
            artifact_path: Destination path within run's artifact directory
        """
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            figure.savefig(f.name, dpi=100, bbox_inches='tight')
            mlflow.log_artifact(f.name, artifact_path)
            os.unlink(f.name)
            
    def set_tags(self, tags: Dict[str, str]):
        """Set tags for the current run.
        
        Args:
            tags: Dictionary of tags to set
        """
        for key, value in tags.items():
            mlflow.set_tag(key, value)
            
    def search_runs(
        self,
        filter_string: str = "",
        max_results: int = 100,
        order_by: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Search for runs in the experiment.
        
        Args:
            filter_string: Filter query string
            max_results: Maximum number of results
            order_by: List of fields to order by
            
        Returns:
            DataFrame of run information
        """
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=filter_string,
            max_results=max_results,
            order_by=order_by or ["start_time DESC"],
        )
        return runs
        
    def get_best_run(
        self,
        metric: str,
        mode: str = "min",
    ) -> Dict[str, Any]:
        """Get the best run based on a metric.
        
        Args:
            metric: Metric to optimize
            mode: "min" or "max"
            
        Returns:
            Dictionary with best run information
        """
        order = "ASC" if mode == "min" else "DESC"
        runs = self.search_runs(
            order_by=[f"metrics.{metric} {order}"],
            max_results=1,
        )
        
        if runs.empty:
            return None
            
        return runs.iloc[0].to_dict()
        
    def load_model(self, run_id: str, model_name: str = "model") -> Any:
        """Load a model from a run.
        
        Args:
            run_id: ID of the run
            model_name: Name of the model artifact
            
        Returns:
            Loaded model object
        """
        model_uri = f"runs:/{run_id}/{model_name}"
        return mlflow.sklearn.load_model(model_uri)


class MLflowModelRegistry:
    """Manage models in MLflow Model Registry."""
    
    def __init__(self, tracking_uri: Optional[str] = None):
        """Initialize model registry client.
        
        Args:
            tracking_uri: URI of tracking server
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        elif os.getenv("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
            
        self.client = MlflowClient()
        
    def register_model(
        self,
        run_id: str,
        model_name: str,
        artifact_path: str = "model",
        await_registration: bool = True,
    ) -> str:
        """Register a model from a run.
        
        Args:
            run_id: ID of the run containing the model
            model_name: Name to register the model under
            artifact_path: Path to model artifact in run
            await_registration: Whether to wait for registration
            
        Returns:
            Model version
        """
        model_uri = f"runs:/{run_id}/{artifact_path}"
        
        model_details = mlflow.register_model(
            model_uri,
            model_name,
            await_registration_for=await_registration,
        )
        
        return model_details.version
        
    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_existing: bool = True,
    ):
        """Transition a model version to a new stage.
        
        Args:
            model_name: Name of the registered model
            version: Version to transition
            stage: New stage (Staging, Production, Archived, None)
            archive_existing: Whether to archive existing models in stage
        """
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing,
        )
        
    def get_latest_model_version(
        self,
        model_name: str,
        stages: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Get the latest version of a model.
        
        Args:
            model_name: Name of the registered model
            stages: Filter by stages (e.g., ["Production"])
            
        Returns:
            Model version information
        """
        versions = self.client.get_latest_versions(
            name=model_name,
            stages=stages,
        )
        
        if versions:
            return versions[0].__dict__
        return None
        
    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> Any:
        """Load a model from the registry.
        
        Args:
            model_name: Name of the registered model
            version: Specific version to load
            stage: Stage to load from (e.g., "Production")
            
        Returns:
            Loaded model object
        """
        if version:
            model_uri = f"models:/{model_name}/{version}"
        elif stage:
            model_uri = f"models:/{model_name}/{stage}"
        else:
            model_uri = f"models:/{model_name}/latest"
            
        return mlflow.sklearn.load_model(model_uri)
        
    def delete_model_version(self, model_name: str, version: str):
        """Delete a model version.
        
        Args:
            model_name: Name of the registered model
            version: Version to delete
        """
        self.client.delete_model_version(
            name=model_name,
            version=version,
        )
        
    def search_models(
        self,
        filter_string: str = "",
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        """Search for registered models.
        
        Args:
            filter_string: Filter query string
            max_results: Maximum number of results
            
        Returns:
            List of model information dictionaries
        """
        models = self.client.search_registered_models(
            filter_string=filter_string,
            max_results=max_results,
        )
        
        return [model.__dict__ for model in models]