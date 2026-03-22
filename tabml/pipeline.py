"""End-to-end pipeline for tabular ML tasks.

This module provides a high-level interface that combines all TabML components
into a unified pipeline for easy model development and deployment.

Classes:
    TabularPipeline: Complete ML pipeline from data loading to predictions
    
Example:
    Quick start::
    
        from tabml.pipeline import TabularPipeline
        
        # Initialize and load data
        pipeline = TabularPipeline(data_dir="./data")
        pipeline.load_data(train_file="train.csv", test_file="test.csv")
        
        # Run full pipeline with defaults
        submission = pipeline.run_full_pipeline()
        
        # Or customize each step
        pipeline.engineer_features(scaling='robust', create_interactions=True)
        pipeline.select_features(method='tree_based', n_features=50)
        pipeline.train_models(optimize_hyperparams=True)
        predictions = pipeline.predict()
"""

from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

from .data import DataLoader
from .features import FeatureEngineer, FeatureSelector
from .models import ModelTrainer
from .evaluate import CrossValidator


class TabularPipeline:
    """Complete pipeline for tabular ML tasks.
    
    Provides a high-level interface that orchestrates data loading,
    feature engineering, feature selection, model training, and prediction.
    Designed for both quick prototyping and production use.
    
    Attributes:
        data_dir: Directory containing data files
        task_type: 'classification' or 'regression'
        random_state: Random seed for reproducibility
        data_loader: DataLoader instance
        feature_engineer: FeatureEngineer instance
        feature_selector: FeatureSelector instance
        model_trainer: ModelTrainer instance
        cross_validator: CrossValidator instance
        best_model: Best performing model after training
        is_fitted: Whether pipeline has been fitted
        predictions: Predictions on test set
        
    Example:
        >>> # Basic usage
        >>> pipeline = TabularPipeline(data_dir="./competition")
        >>> pipeline.load_data()
        >>> submission = pipeline.run_full_pipeline()
        >>> 
        >>> # Advanced usage with custom configuration
        >>> pipeline = TabularPipeline(task_type='classification')
        >>> pipeline.load_data(target_column='label', sample_frac=0.1)
        >>> 
        >>> # Custom feature engineering
        >>> pipeline.engineer_features(
        ...     numeric_impute='mean',
        ...     scaling='minmax',
        ...     create_interactions=True
        ... )
        >>> 
        >>> # Feature selection
        >>> pipeline.select_features(method='mutual_info', n_features=100)
        >>> 
        >>> # Model training with optimization
        >>> pipeline.train_models(
        ...     model_types=['xgboost', 'lightgbm'],
        ...     optimize_hyperparams=True,
        ...     cv_folds=10
        ... )
        >>> 
        >>> # Get feature importance
        >>> importance = pipeline.get_feature_importance()
    """
    
    def __init__(self,
                 data_dir: Union[str, Path] = "data",
                 task_type: Optional[str] = None,
                 random_state: int = 42):
        """Initialize tabular pipeline.
        
        Args:
            data_dir: Directory containing data files. Can be string or Path.
                Defaults to "data" in current directory.
            task_type: Type of ML task. Options:
                - 'classification': For classification problems
                - 'regression': For regression problems
                - None: Auto-detected based on target variable
            random_state: Random seed for reproducibility across all components.
                Ensures consistent results across runs.
                
        Example:
            >>> # Auto-detect everything
            >>> pipeline = TabularPipeline()
            >>> 
            >>> # Specify task type
            >>> pipeline = TabularPipeline(
            ...     data_dir="./kaggle/competition",
            ...     task_type='regression'
            ... )
        """
        self.data_dir = Path(data_dir)
        self.task_type = task_type
        self.random_state = random_state
        
        # Initialize components
        self.data_loader = DataLoader(data_dir)
        self.feature_engineer = None
        self.feature_selector = None
        self.model_trainer = ModelTrainer(task_type)
        self.cross_validator = CrossValidator()
        
        # Pipeline state
        self.is_fitted = False
        self.best_model = None
        self.predictions = None
        
    def load_data(self,
                  train_file: str = "train.csv",
                  test_file: str = "test.csv",
                  target_column: Optional[str] = None,
                  id_column: Optional[str] = None,
                  sample_frac: Optional[float] = None) -> 'TabularPipeline':
        """Load train and test data.
        
        Loads data files and automatically detects target and ID columns
        if not specified. Also infers task type from target variable.
        
        Args:
            train_file: Training data filename in data_dir.
                Supports CSV, Parquet, and Excel formats.
            test_file: Test data filename in data_dir.
            target_column: Name of target column. If None, detects column
                present in train but not in test.
            id_column: Name of ID column for submissions. If None,
                looks for common ID column names.
            sample_frac: Fraction of data to use (0-1). Useful for
                quick prototyping. None uses all data.
                
        Returns:
            Self for method chaining
            
        Example:
            >>> # Load with auto-detection
            >>> pipeline.load_data()
            >>> 
            >>> # Load with specific columns
            >>> pipeline.load_data(
            ...     train_file="training_data.parquet",
            ...     test_file="testing_data.parquet",
            ...     target_column="SalePrice",
            ...     id_column="Id"
            ... )
            >>> 
            >>> # Quick prototype with 10% of data
            >>> pipeline.load_data(sample_frac=0.1)
            
        Note:
            Task type is automatically detected based on number of unique
            values in target column (< 100 = classification).
        """
        logger.info("Loading data...")
        self.train_data, self.test_data = self.data_loader.load_data(
            train_file=train_file,
            test_file=test_file,
            target_column=target_column,
            id_column=id_column,
            sample_frac=sample_frac
        )
        
        # Get data info
        data_info = self.data_loader.get_data_info()
        logger.info(f"Data loaded successfully. Target: {data_info['target_column']}")
        logger.info(f"Task type: {data_info['target_info']['type']}")
        
        # Set task type if not specified
        if self.task_type is None:
            self.task_type = data_info['target_info']['type']
            
        return self
        
    def engineer_features(self,
                         numeric_impute: str = 'median',
                         categorical_impute: str = 'constant',
                         scaling: Optional[str] = 'standard',
                         create_interactions: bool = False,
                         target_encoding: bool = True) -> 'TabularPipeline':
        """Apply feature engineering.
        
        Configures and applies feature engineering including imputation,
        scaling, encoding, and feature creation.
        
        Args:
            numeric_impute: Strategy for numeric missing values:
                'mean', 'median', 'most_frequent', 'constant'.
            categorical_impute: Strategy for categorical missing values:
                'constant', 'most_frequent'.
            scaling: Method for scaling numeric features:
                'standard', 'minmax', 'robust', None.
            create_interactions: Whether to create interaction features
                between numeric columns (products).
            target_encoding: Whether to use target encoding for
                categorical variables (requires target in training).
                
        Returns:
            Self for method chaining
            
        Example:
            >>> # Standard preprocessing
            >>> pipeline.engineer_features()
            >>> 
            >>> # Custom configuration
            >>> pipeline.engineer_features(
            ...     numeric_impute='mean',
            ...     scaling='robust',  # Better for outliers
            ...     create_interactions=True,
            ...     target_encoding=True  # Good for high cardinality
            ... )
            
        Note:
            - ID columns are automatically excluded from features
            - Feature names are stored in feature_engineer.feature_names_out_
        """
        logger.info("Engineering features...")
        
        self.feature_engineer = FeatureEngineer(
            numeric_impute_strategy=numeric_impute,
            categorical_impute_strategy=categorical_impute,
            scaling_method=scaling,
            create_interactions=create_interactions,
            target_encoding=target_encoding
        )
        
        # Fit on training data
        X_train = self.train_data.drop(columns=[self.data_loader.target_column])
        y_train = self.train_data[self.data_loader.target_column]
        
        # Remove ID column if present
        if self.data_loader.id_column and self.data_loader.id_column in X_train.columns:
            X_train = X_train.drop(columns=[self.data_loader.id_column])
            
        self.feature_engineer.fit(X_train, y_train)
        
        # Transform train and test data
        self.X_train_transformed = self.feature_engineer.transform(X_train)
        self.X_test_transformed = self.feature_engineer.transform(self.data_loader.get_test_features())
        self.y_train = y_train
        
        logger.info(f"Features engineered. Shape: {self.X_train_transformed.shape}")
        return self
        
    def select_features(self,
                       method: str = 'mutual_info',
                       n_features: Optional[Union[int, float]] = None) -> 'TabularPipeline':
        """Select important features.
        
        Applies feature selection to reduce dimensionality and improve
        model performance by keeping only the most informative features.
        
        Args:
            method: Feature selection method:
                - 'mutual_info': Mutual information (captures non-linear)
                - 'univariate': Statistical tests (f_classif/f_regression)
                - 'tree_based': Random Forest importances
                - 'rfe': Recursive Feature Elimination
            n_features: Number of features to select:
                - int: Exact number of features
                - float (0-1): Percentage of features
                - None: Selects max(10, n_features // 2)
                
        Returns:
            Self for method chaining
            
        Example:
            >>> # Select top 50 features using mutual information
            >>> pipeline.select_features(method='mutual_info', n_features=50)
            >>> 
            >>> # Select top 30% of features using tree importance
            >>> pipeline.select_features(method='tree_based', n_features=0.3)
            >>> 
            >>> # Auto-select number of features
            >>> pipeline.select_features()
            
        Note:
            Selected feature names are available in
            feature_selector.selected_features_
        """
        logger.info("Selecting features...")
        
        self.feature_selector = FeatureSelector(
            method=method,
            n_features=n_features,
            task_type=self.task_type
        )
        
        # Fit on transformed training data
        self.feature_selector.fit(self.X_train_transformed, self.y_train)
        
        # Transform data
        self.X_train_selected = self.feature_selector.transform(self.X_train_transformed)
        self.X_test_selected = self.feature_selector.transform(self.X_test_transformed)
        
        logger.info(f"Features selected. Shape: {self.X_train_selected.shape}")
        logger.info(f"Selected features: {self.feature_selector.selected_features_[:10]}...")
        return self
        
    def train_models(self,
                    model_types: Optional[List[str]] = None,
                    optimize_hyperparams: bool = False,
                    cv_folds: int = 5,
                    use_validation: bool = True,
                    val_size: float = 0.2) -> 'TabularPipeline':
        """Train and evaluate models.
        
        Trains multiple models, evaluates them using cross-validation,
        and selects the best performer. Optionally optimizes hyperparameters.
        
        Args:
            model_types: List of models to train. Options:
                - 'xgboost': XGBoost
                - 'lightgbm': LightGBM
                - 'catboost': CatBoost
                - 'random_forest': Random Forest
                - None: Trains all available models
            optimize_hyperparams: Whether to optimize hyperparameters
                using Optuna. Takes longer but improves performance.
            cv_folds: Number of cross-validation folds for evaluation.
            use_validation: Whether to use a validation set for training.
                Helps prevent overfitting.
            val_size: Size of validation set if use_validation=True.
                
        Returns:
            Self for method chaining
            
        Example:
            >>> # Train all models with defaults
            >>> pipeline.train_models()
            >>> 
            >>> # Train specific models with optimization
            >>> pipeline.train_models(
            ...     model_types=['xgboost', 'lightgbm'],
            ...     optimize_hyperparams=True,
            ...     cv_folds=10
            ... )
            >>> 
            >>> # Train without validation set
            >>> pipeline.train_models(
            ...     use_validation=False,  # Use all data
            ...     cv_folds=5  # Rely on CV for evaluation
            ... )
            
        Note:
            - Best model is selected based on CV performance
            - Best model is retrained on full training data
            - Access best model via pipeline.best_model
        """
        logger.info("Training models...")
        
        # Use selected features if available, otherwise use transformed features
        X_train = self.X_train_selected if hasattr(self, 'X_train_selected') else self.X_train_transformed
        
        # Split validation set if requested
        if use_validation:
            X_train_split, X_val, y_train_split, y_val = self.data_loader.get_train_test_split(
                test_size=val_size,
                random_state=self.random_state
            )
            
            # Apply feature engineering and selection to splits
            if self.feature_engineer:
                X_train_split = self.feature_engineer.transform(X_train_split)
                X_val = self.feature_engineer.transform(X_val)
            if self.feature_selector:
                X_train_split = self.feature_selector.transform(X_train_split)
                X_val = self.feature_selector.transform(X_val)
        else:
            X_train_split, y_train_split = X_train, self.y_train
            X_val, y_val = None, None
            
        # Train models
        models = self.model_trainer.train_all_models(
            X_train_split, y_train_split,
            X_val, y_val,
            model_types=model_types,
            optimize_hyperparams=optimize_hyperparams
        )
        
        # Cross-validate models
        logger.info(f"Cross-validating models with {cv_folds} folds...")
        cv_results = {}
        for name, model in models.items():
            scores = self.cross_validator.evaluate_model(
                model, X_train, self.y_train, cv_folds=cv_folds
            )
            cv_results[name] = scores
            logger.info(f"{name} - Mean CV Score: {scores['mean']:.4f} (+/- {scores['std']:.4f})")
            
        # Select best model
        best_score = -float('inf')
        for name, scores in cv_results.items():
            if scores['mean'] > best_score:
                best_score = scores['mean']
                self.best_model = models[name]
                
        logger.info(f"Best model: {self.best_model.name}")
        
        # Retrain best model on full training data
        logger.info("Retraining best model on full training data...")
        self.best_model.fit(X_train, self.y_train)
        self.is_fitted = True
        
        return self
        
    def predict(self) -> np.ndarray:
        """Make predictions on test data.
        
        Uses the best model to generate predictions on the test set.
        For binary classification, returns probabilities.
        
        Returns:
            Array of predictions:
                - Regression: Predicted values
                - Binary classification: Probabilities for positive class
                - Multiclass: Predicted class labels
                
        Raises:
            ValueError: If pipeline hasn't been fitted
            
        Example:
            >>> # Make predictions
            >>> predictions = pipeline.predict()
            >>> 
            >>> # For binary classification, get probabilities
            >>> proba = pipeline.predict()  # Returns P(y=1)
            >>> 
            >>> # Convert to class labels if needed
            >>> labels = (proba > 0.5).astype(int)
            
        Note:
            Predictions are stored in pipeline.predictions
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before prediction.")
            
        logger.info("Making predictions...")
        
        # Use selected features if available
        X_test = self.X_test_selected if hasattr(self, 'X_test_selected') else self.X_test_transformed
        
        # Make predictions
        if self.task_type == 'classification' and self.y_train.nunique() == 2:
            # Binary classification - use probabilities
            self.predictions = self.best_model.predict_proba(X_test)[:, 1]
        else:
            self.predictions = self.best_model.predict(X_test)
            
        return self.predictions
        
    def create_submission(self, filename: str = "submission.csv") -> pd.DataFrame:
        """Create submission file.
        
        Creates a properly formatted submission file with test IDs
        and predictions, ready for competition submission.
        
        Args:
            filename: Name of submission file to create in data_dir.
                
        Returns:
            DataFrame with submission data (ID and prediction columns)
            
        Example:
            >>> # Create submission with default name
            >>> submission_df = pipeline.create_submission()
            >>> 
            >>> # Custom filename
            >>> submission_df = pipeline.create_submission(
            ...     "final_submission_v2.csv"
            ... )
            
        Note:
            - Automatically makes predictions if not already done
            - File is saved to data_dir/filename
            - Returns DataFrame for inspection
        """
        if self.predictions is None:
            self.predict()
            
        return self.data_loader.create_submission(self.predictions, filename)
        
    def run_full_pipeline(self,
                         feature_engineering: Dict[str, Any] = None,
                         feature_selection: Dict[str, Any] = None,
                         model_training: Dict[str, Any] = None) -> pd.DataFrame:
        """Run complete pipeline from data loading to submission.
        
        Convenience method that runs all pipeline steps with configurable
        parameters. Assumes data has already been loaded.
        
        Args:
            feature_engineering: Keyword arguments for engineer_features().
                If None, uses defaults.
            feature_selection: Keyword arguments for select_features().
                If None, uses {'method': 'mutual_info'}.
                Set to empty dict {} to skip feature selection.
            model_training: Keyword arguments for train_models().
                If None, uses {'optimize_hyperparams': False}.
                
        Returns:
            Submission DataFrame
            
        Example:
            >>> # Run with all defaults
            >>> pipeline.load_data()
            >>> submission = pipeline.run_full_pipeline()
            >>> 
            >>> # Run with custom configuration
            >>> submission = pipeline.run_full_pipeline(
            ...     feature_engineering={
            ...         'scaling': 'robust',
            ...         'create_interactions': True
            ...     },
            ...     feature_selection={
            ...         'method': 'tree_based',
            ...         'n_features': 100
            ...     },
            ...     model_training={
            ...         'model_types': ['xgboost', 'lightgbm'],
            ...         'optimize_hyperparams': True,
            ...         'cv_folds': 10
            ...     }
            ... )
            >>> 
            >>> # Skip feature selection
            >>> submission = pipeline.run_full_pipeline(
            ...     feature_selection={}  # Empty dict skips this step
            ... )
            
        Note:
            This method assumes load_data() has been called first.
        """
        # Default configurations
        fe_config = feature_engineering or {}
        fs_config = feature_selection or {'method': 'mutual_info'}
        mt_config = model_training or {'optimize_hyperparams': False}
        
        # Run pipeline steps
        self.engineer_features(**fe_config)
        
        if fs_config:
            self.select_features(**fs_config)
            
        self.train_models(**mt_config)
        self.predict()
        
        return self.create_submission()
        
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from best model.
        
        Extracts feature importances from the best model and returns
        them as a sorted DataFrame.
        
        Returns:
            DataFrame with columns:
                - 'feature': Feature name
                - 'importance': Importance score
            Sorted by importance in descending order.
            
        Raises:
            ValueError: If pipeline hasn't been fitted
            
        Example:
            >>> # Get feature importances
            >>> importance_df = pipeline.get_feature_importance()
            >>> print(importance_df.head(20))  # Top 20 features
            >>> 
            >>> # Plot importances
            >>> import matplotlib.pyplot as plt
            >>> top_features = importance_df.head(30)
            >>> plt.barh(top_features['feature'], top_features['importance'])
            >>> plt.xlabel('Importance')
            >>> plt.title('Top 30 Feature Importances')
            
        Note:
            Uses selected features if feature selection was applied,
            otherwise uses all engineered features.
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted first.")
            
        feature_names = (self.feature_selector.selected_features_ 
                        if hasattr(self, 'feature_selector') and self.feature_selector
                        else self.feature_engineer.feature_names_out_)
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.best_model.feature_importances_
        })
        
        return importance_df.sort_values('importance', ascending=False)