"""Out-of-Fold (OOF) predictions manager for saving and loading.

This module provides utilities for managing OOF predictions across multiple
models and experiments, particularly useful for Kaggle competitions where
you need to combine many models.
"""

import os
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger


class OOFManager:
    """Manage saving and loading of out-of-fold predictions.
    
    This class helps organize OOF predictions from multiple models,
    save them to disk, and load them later for ensemble creation.
    
    Example:
        >>> # Save OOF predictions from multiple models
        >>> manager = OOFManager(output_dir="output/competition_name")
        >>> 
        >>> # Train and save OOF from each model
        >>> for model_name, model in models.items():
        ...     oof_preds = ensemble.get_oof_predictions([model], X_train, y_train)
        ...     manager.save_oof(
        ...         predictions=oof_preds,
        ...         model_name=model_name,
        ...         model_params=model.get_params(),
        ...         cv_score=0.85
        ...     )
        >>> 
        >>> # Later, load all OOF predictions for ensemble
        >>> all_oofs = manager.load_all_oofs()
        >>> combined_df = manager.combine_oofs(all_oofs)
    """
    
    def __init__(self, output_dir: str = "output/oof_predictions"):
        """Initialize OOF Manager.
        
        Args:
            output_dir: Directory to save OOF predictions
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.output_dir / "oof_metadata.json"
        self.metadata = self._load_metadata()
        
    def save_oof(self, 
                 predictions: Union[np.ndarray, pd.DataFrame, pd.Series],
                 model_name: str,
                 model_params: Optional[Dict] = None,
                 cv_score: Optional[float] = None,
                 cv_scores_per_fold: Optional[List[float]] = None,
                 feature_names: Optional[List[str]] = None,
                 experiment_name: Optional[str] = None,
                 tags: Optional[Dict] = None,
                 test_predictions: Optional[Union[np.ndarray, pd.DataFrame]] = None) -> str:
        """Save OOF predictions with metadata.
        
        Args:
            predictions: OOF predictions (can be array, DataFrame, or Series)
            model_name: Name identifier for the model
            model_params: Model hyperparameters
            cv_score: Overall cross-validation score
            cv_scores_per_fold: Scores for each fold
            feature_names: List of feature names used
            experiment_name: Optional experiment identifier
            tags: Additional tags for filtering/searching
            test_predictions: Optional test set predictions to save alongside
            
        Returns:
            Path to saved OOF file
            
        Example:
            >>> # Save OOF with full metadata
            >>> manager.save_oof(
            ...     predictions=oof_preds,
            ...     model_name="xgboost_v3",
            ...     model_params={'n_estimators': 1000, 'max_depth': 6},
            ...     cv_score=0.8523,
            ...     cv_scores_per_fold=[0.85, 0.86, 0.84, 0.85, 0.86],
            ...     tags={'feature_set': 'v2', 'validation': 'stratified'}
            ... )
        """
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}.pkl"
        filepath = self.output_dir / filename
        
        # Convert predictions to DataFrame if needed
        if isinstance(predictions, np.ndarray):
            predictions = pd.DataFrame(predictions, columns=[f"pred_{i}" for i in range(predictions.shape[1])] 
                                     if predictions.ndim > 1 else ["prediction"])
        elif isinstance(predictions, pd.Series):
            # Convert Series to DataFrame preserving the values
            predictions = predictions.to_frame(name="prediction")
        
        # Prepare data to save
        save_data = {
            'predictions': predictions,
            'model_name': model_name,
            'model_params': model_params or {},
            'cv_score': cv_score,
            'cv_scores_per_fold': cv_scores_per_fold,
            'feature_names': feature_names,
            'experiment_name': experiment_name,
            'tags': tags or {},
            'timestamp': timestamp,
            'shape': predictions.shape,
            'test_predictions': test_predictions
        }
        
        # Save to pickle file
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        # Update metadata
        self.metadata[filename] = {
            'model_name': model_name,
            'cv_score': cv_score,
            'timestamp': timestamp,
            'experiment_name': experiment_name,
            'tags': tags or {},
            'shape': predictions.shape
        }
        self._save_metadata()
        
        logger.info(f"Saved OOF predictions to {filepath}")
        if cv_score:
            logger.info(f"  CV Score: {cv_score:.4f}")
        
        return str(filepath)
    
    def load_oof(self, filename: str) -> Dict:
        """Load specific OOF predictions.
        
        Args:
            filename: Name of the OOF file to load
            
        Returns:
            Dictionary containing predictions and metadata
        """
        filepath = self.output_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"OOF file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        logger.info(f"Loaded OOF predictions from {filename}")
        if 'cv_score' in data and data['cv_score']:
            logger.info(f"  Model: {data['model_name']}, CV Score: {data['cv_score']:.4f}")
        
        return data
    
    def load_all_oofs(self, 
                     experiment_name: Optional[str] = None,
                     min_cv_score: Optional[float] = None,
                     tags_filter: Optional[Dict] = None,
                     top_k: Optional[int] = None) -> Dict[str, Dict]:
        """Load all OOF predictions matching criteria.
        
        Args:
            experiment_name: Filter by experiment name
            min_cv_score: Minimum CV score threshold
            tags_filter: Filter by tags
            top_k: Load only top K models by CV score
            
        Returns:
            Dictionary mapping filenames to OOF data
            
        Example:
            >>> # Load top 10 models with CV score > 0.85
            >>> best_oofs = manager.load_all_oofs(
            ...     min_cv_score=0.85,
            ...     top_k=10
            ... )
        """
        filtered_files = []
        
        for filename, meta in self.metadata.items():
            # Apply filters
            if experiment_name and meta.get('experiment_name') != experiment_name:
                continue
            
            if min_cv_score and meta.get('cv_score', 0) < min_cv_score:
                continue
            
            if tags_filter:
                file_tags = meta.get('tags', {})
                if not all(file_tags.get(k) == v for k, v in tags_filter.items()):
                    continue
            
            filtered_files.append((filename, meta.get('cv_score', 0)))
        
        # Sort by CV score and take top K
        filtered_files.sort(key=lambda x: x[1], reverse=True)
        
        if top_k:
            filtered_files = filtered_files[:top_k]
        
        # Load the filtered files
        result = {}
        for filename, _ in filtered_files:
            try:
                result[filename] = self.load_oof(filename)
            except Exception as e:
                logger.warning(f"Failed to load {filename}: {e}")
        
        logger.info(f"Loaded {len(result)} OOF prediction files")
        
        return result
    
    def combine_oofs(self, 
                    oof_dict: Dict[str, Dict],
                    method: str = 'horizontal',
                    use_test: bool = False) -> pd.DataFrame:
        """Combine multiple OOF predictions into single DataFrame.
        
        Args:
            oof_dict: Dictionary of OOF data from load_all_oofs
            method: 'horizontal' to concatenate as columns, 'average' to average
            use_test: If True, combine test predictions instead of OOF
            
        Returns:
            Combined DataFrame with all predictions
            
        Example:
            >>> # Combine OOFs for stacking
            >>> all_oofs = manager.load_all_oofs(top_k=10)
            >>> combined = manager.combine_oofs(all_oofs)
            >>> # combined has columns: model1_pred, model2_pred, ...
        """
        dfs_to_combine = []
        
        for filename, data in oof_dict.items():
            if use_test and 'test_predictions' in data and data['test_predictions'] is not None:
                pred_data = data['test_predictions']
            else:
                pred_data = data['predictions']
            
            # Convert to DataFrame if needed
            if isinstance(pred_data, np.ndarray):
                pred_data = pd.DataFrame(pred_data)
            elif isinstance(pred_data, pd.Series):
                pred_data = pred_data.to_frame()
            
            # Ensure it's a DataFrame with proper column name
            model_name = data['model_name']
            if isinstance(pred_data, pd.DataFrame):
                if pred_data.shape[1] == 1:
                    # Single column - rename it
                    pred_data.columns = [model_name]
                else:
                    # Multiple columns - add prefix
                    pred_data = pred_data.add_prefix(f"{model_name}_")
            
            dfs_to_combine.append(pred_data)
        
        if method == 'horizontal':
            # Concatenate as columns, ignoring index to avoid alignment issues
            # Reset index for all DataFrames to ensure proper alignment
            dfs_reset = [df.reset_index(drop=True) for df in dfs_to_combine]
            combined = pd.concat(dfs_reset, axis=1)
        elif method == 'average':
            # Average all predictions
            dfs_reset = [df.reset_index(drop=True) for df in dfs_to_combine]
            stacked = pd.concat(dfs_reset, axis=1)
            combined = pd.DataFrame(stacked.mean(axis=1), columns=['averaged_prediction'])
        else:
            raise ValueError(f"Unknown method: {method}")
        
        logger.info(f"Combined {len(dfs_to_combine)} OOF predictions")
        logger.info(f"Combined shape: {combined.shape}")
        
        return combined
    
    def list_oofs(self, 
                 sort_by: str = 'cv_score',
                 ascending: bool = False) -> pd.DataFrame:
        """List all saved OOF predictions with metadata.
        
        Args:
            sort_by: Column to sort by ('cv_score', 'timestamp', 'model_name')
            ascending: Sort order
            
        Returns:
            DataFrame with OOF metadata
        """
        if not self.metadata:
            return pd.DataFrame()
        
        df = pd.DataFrame.from_dict(self.metadata, orient='index')
        df.index.name = 'filename'
        df = df.reset_index()
        
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=ascending)
        
        return df
    
    def delete_oof(self, filename: str) -> None:
        """Delete specific OOF file.
        
        Args:
            filename: Name of file to delete
        """
        filepath = self.output_dir / filename
        
        if filepath.exists():
            filepath.unlink()
            
        if filename in self.metadata:
            del self.metadata[filename]
            self._save_metadata()
        
        logger.info(f"Deleted OOF file: {filename}")
    
    def cleanup_old_oofs(self, keep_top_k: int = 20, keep_days: int = 30) -> None:
        """Clean up old or poor-performing OOF files.
        
        Args:
            keep_top_k: Number of top models to keep
            keep_days: Keep files newer than this many days
        """
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        
        # Get files to potentially delete
        files_to_check = []
        for filename, meta in self.metadata.items():
            timestamp_str = meta.get('timestamp', '')
            if timestamp_str:
                file_date = datetime.strptime(timestamp_str[:8], "%Y%m%d")
                if file_date < cutoff_date:
                    files_to_check.append((filename, meta.get('cv_score', 0)))
        
        # Sort by score and mark for deletion
        files_to_check.sort(key=lambda x: x[1], reverse=True)
        files_to_delete = files_to_check[keep_top_k:]
        
        for filename, _ in files_to_delete:
            self.delete_oof(filename)
        
        if files_to_delete:
            logger.info(f"Cleaned up {len(files_to_delete)} old OOF files")
    
    def _load_metadata(self) -> Dict:
        """Load metadata from JSON file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self) -> None:
        """Save metadata to JSON file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def export_summary(self, output_file: Optional[str] = None) -> pd.DataFrame:
        """Export summary of all OOF predictions.
        
        Args:
            output_file: Optional CSV file to save summary
            
        Returns:
            Summary DataFrame
        """
        summary = self.list_oofs(sort_by='cv_score', ascending=False)
        
        if output_file:
            summary.to_csv(output_file, index=False)
            logger.info(f"Exported OOF summary to {output_file}")
        
        return summary


def load_and_ensemble_oofs(output_dir: str,
                          top_k: int = 10,
                          min_cv_score: float = None,
                          ensemble_method: str = 'weighted') -> pd.DataFrame:
    """Quick function to load and ensemble OOF predictions.
    
    Args:
        output_dir: Directory containing OOF files
        top_k: Number of top models to use
        min_cv_score: Minimum CV score threshold
        ensemble_method: Method for ensembling ('weighted', 'average', 'stacking')
        
    Returns:
        Ensembled predictions
        
    Example:
        >>> # Quick ensemble of top 10 models
        >>> final_preds = load_and_ensemble_oofs(
        ...     "output/competition",
        ...     top_k=10,
        ...     min_cv_score=0.85
        ... )
    """
    from .ensemble import OOFEnsemble
    
    manager = OOFManager(output_dir)
    
    # Load top models
    oofs = manager.load_all_oofs(top_k=top_k, min_cv_score=min_cv_score)
    
    if not oofs:
        raise ValueError("No OOF files found matching criteria")
    
    # Combine OOFs
    combined = manager.combine_oofs(oofs, method='horizontal')
    
    # Create ensemble
    if ensemble_method == 'average':
        return combined.mean(axis=1)
    elif ensemble_method == 'weighted':
        # Get CV scores for weighting
        scores = [data['cv_score'] for data in oofs.values() if data.get('cv_score')]
        if scores:
            weights = np.array(scores) / sum(scores)
            return np.average(combined.values, weights=weights, axis=1)
        else:
            return combined.mean(axis=1)
    else:
        return combined