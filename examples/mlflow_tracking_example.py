"""
MLflow Tracking Example for TabML

This example demonstrates how to use MLflow for experiment tracking,
model versioning, and dataset management with TabML.

Prerequisites:
1. Install MLflow: pip install mlflow
2. Start MLflow server: mlflow server --host 0.0.0.0 --port 5000
3. Set environment variables or use .env file
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_wine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import TabML components
from tabml import (
    MLflowTracker,
    MLflowCallback,
    MLflowModelRegistry,
    XGBoostModel,
    LightGBMModel,
    FeatureEngineer,
    OOFEnsemble
)
from tabml.training import EnhancedTrainer, EarlyStoppingCallback


def main():
    """Main function demonstrating MLflow integration with TabML."""
    
    # =========================================================================
    # 1. Setup MLflow Tracker
    # =========================================================================
    
    print("Setting up MLflow tracker...")
    
    # Initialize tracker (uses MLFLOW_TRACKING_URI from environment if not specified)
    tracker = MLflowTracker(
        experiment_name="tabml-demo",
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        tags={
            "framework": "tabml",
            "environment": "development",
            "dataset": "iris"
        }
    )
    
    # =========================================================================
    # 2. Load and Prepare Data
    # =========================================================================
    
    print("Loading dataset...")
    
    # Load iris dataset
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # =========================================================================
    # 3. Start MLflow Run
    # =========================================================================
    
    print("Starting MLflow run...")
    
    # Start a new run
    run_id = tracker.start_run(
        run_name="iris-classification-baseline",
        description="Baseline model for iris classification using XGBoost and LightGBM"
    )
    
    print(f"MLflow run started with ID: {run_id}")
    
    # =========================================================================
    # 4. Log Dataset Information
    # =========================================================================
    
    print("Logging dataset information...")
    
    # Log dataset
    tracker.log_dataset(
        X_train,
        name="iris_train",
        description="Iris training dataset with 80% of samples"
    )
    
    # Log dataset parameters
    tracker.log_params({
        "dataset_name": "iris",
        "n_samples_train": len(X_train),
        "n_samples_test": len(X_test),
        "n_features": X.shape[1],
        "n_classes": len(np.unique(y)),
        "test_size": 0.2,
        "random_state": 42
    })
    
    # =========================================================================
    # 5. Feature Engineering
    # =========================================================================
    
    print("Performing feature engineering...")
    
    # Initialize feature engineer
    engineer = FeatureEngineer(
        numeric_impute_strategy='median',
        scaling_method='standard'
    )
    
    # Transform features
    X_train_transformed = engineer.fit_transform(X_train, y_train)
    X_test_transformed = engineer.transform(X_test)
    
    # Log feature engineering parameters
    tracker.log_params({
        "feature_scaling": "standard",
        "numeric_impute": "median"
    })
    
    # =========================================================================
    # 6. Train Models with MLflow Callback
    # =========================================================================
    
    print("Training models with MLflow tracking...")
    
    # Define models
    models = [
        {
            "name": "XGBoost",
            "model": XGBoostModel(params={
                'n_estimators': 100,
                'max_depth': 3,
                'learning_rate': 0.1,
                'random_state': 42
            })
        },
        {
            "name": "LightGBM",
            "model": LightGBMModel(params={
                'n_estimators': 100,
                'max_depth': 3,
                'learning_rate': 0.1,
                'random_state': 42,
                'verbose': -1
            })
        }
    ]
    
    # Train each model
    trained_models = []
    for model_info in models:
        print(f"Training {model_info['name']}...")
        
        # Log model type
        tracker.log_params({
            f"{model_info['name']}_type": model_info['name'],
            f"{model_info['name']}_n_estimators": 100
        })
        
        # Train model
        model = model_info['model']
        model.fit(X_train_transformed, y_train)
        
        # Evaluate
        train_score = model.score(X_train_transformed, y_train)
        test_score = model.score(X_test_transformed, y_test)
        
        # Log metrics
        tracker.log_metrics({
            f"{model_info['name']}_train_accuracy": train_score,
            f"{model_info['name']}_test_accuracy": test_score
        })
        
        print(f"  Train accuracy: {train_score:.4f}")
        print(f"  Test accuracy: {test_score:.4f}")
        
        # Log model
        tracker.log_model(
            model.model,  # Access the underlying model
            model_name=f"{model_info['name']}_model",
            input_example=X_train_transformed[:5],
            registered_model_name=f"iris-{model_info['name'].lower()}"
        )
        
        trained_models.append(model)
    
    # =========================================================================
    # 7. Ensemble Models with OOF
    # =========================================================================
    
    print("\nCreating ensemble with out-of-fold predictions...")
    
    # Create ensemble
    ensemble = OOFEnsemble(task_type='multiclass')
    
    # Get OOF predictions
    oof_preds = ensemble.get_oof_predictions(
        trained_models,
        X_train_transformed,
        y_train,
        n_folds=5
    )
    
    # Optimize weights
    weights = ensemble.optimize_weights(oof_preds, y_train, method='scipy')
    
    # Log ensemble information
    tracker.log_params({
        "ensemble_method": "weighted_average",
        "n_models": len(trained_models),
        "n_folds": 5
    })
    
    tracker.log_metrics({
        "ensemble_weights": str(weights.tolist()),
        "ensemble_oof_score": ensemble.score_predictions(
            ensemble.weighted_average(oof_preds, weights),
            y_train
        )
    })
    
    # =========================================================================
    # 8. Advanced Training with Callbacks
    # =========================================================================
    
    print("\nTraining with MLflow callback integration...")
    
    # Create enhanced trainer with callbacks
    trainer = EnhancedTrainer(
        callbacks=[
            MLflowCallback(
                experiment_name="tabml-demo",
                run_name="enhanced-training",
                tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
                log_models=True,
                log_datasets=True
            ),
            EarlyStoppingCallback(patience=10, mode='max')
        ]
    )
    
    # Note: This would typically be used in a cross-validation loop
    # For demonstration, we're showing the callback setup
    
    # =========================================================================
    # 9. Model Registry Operations
    # =========================================================================
    
    print("\nWorking with model registry...")
    
    # Initialize model registry
    registry = MLflowModelRegistry(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    
    # Register the best model
    model_version = registry.register_model(
        run_id=run_id,
        model_name="iris-best-model",
        artifact_path="XGBoost_model"
    )
    
    print(f"Model registered with version: {model_version}")
    
    # Transition model to staging
    registry.transition_model_stage(
        model_name="iris-best-model",
        version=model_version,
        stage="Staging"
    )
    
    print(f"Model version {model_version} transitioned to Staging")
    
    # =========================================================================
    # 10. Search and Compare Runs
    # =========================================================================
    
    print("\nSearching for best runs...")
    
    # Search for best runs in the experiment
    best_runs = tracker.search_runs(
        filter_string="metrics.XGBoost_test_accuracy > 0.9",
        max_results=5
    )
    
    if not best_runs.empty:
        print(f"Found {len(best_runs)} runs with accuracy > 0.9")
        print(best_runs[['run_id', 'metrics.XGBoost_test_accuracy']].head())
    
    # Get the best run based on test accuracy
    best_run = tracker.get_best_run(
        metric="XGBoost_test_accuracy",
        mode="max"
    )
    
    if best_run:
        print(f"\nBest run ID: {best_run.get('run_id')}")
        print(f"Best test accuracy: {best_run.get('metrics.XGBoost_test_accuracy')}")
    
    # =========================================================================
    # 11. Load Model from Registry
    # =========================================================================
    
    print("\nLoading model from registry...")
    
    # Load the latest model version
    loaded_model = registry.load_model(
        model_name="iris-best-model",
        stage="Staging"
    )
    
    # Make predictions with loaded model
    predictions = loaded_model.predict(X_test_transformed)
    print(f"Predictions shape: {predictions.shape}")
    
    # =========================================================================
    # 12. End MLflow Run
    # =========================================================================
    
    print("\nEnding MLflow run...")
    
    # Log final summary
    tracker.log_metrics({
        "final_model_count": len(trained_models),
        "final_ensemble_score": ensemble.score_predictions(
            ensemble.weighted_average(oof_preds, weights),
            y_train
        )
    })
    
    # End the run
    tracker.end_run(status="FINISHED")
    
    print("MLflow tracking example completed successfully!")
    print(f"\nView your experiments at: {os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')}")
    
    return trained_models, ensemble, tracker


if __name__ == "__main__":
    # Run the example
    models, ensemble, tracker = main()
    
    print("\n" + "="*80)
    print("MLflow Integration Example Complete!")
    print("="*80)
    print("\nNext steps:")
    print("1. Open MLflow UI: http://localhost:5000")
    print("2. Explore experiments and runs")
    print("3. Compare model metrics")
    print("4. Download artifacts and models")
    print("5. Deploy models from the registry")