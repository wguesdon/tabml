"""
PS5E8 Competition - Advanced TabNet Models
Train TabNet neural network models with advanced feature engineering
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import tabml
sys.path.append(str(Path(__file__).parent.parent.parent))

from tabml import (
    FeatureEngineer, TabNetModel, 
    OOFEnsemble, OOFManager, MLflowTracker
)

# Load environment variables
load_dotenv()

# Setup paths
DATA_DIR = Path("../../data/raw/PS5E8")
ORIGINAL_DATA_PATH = Path("../../data/raw/PS5E8/original_data.csv")
OUTPUT_DIR = Path("output")
OOF_DIR = OUTPUT_DIR / "oof_predictions_advanced"
SUBMISSION_DIR = OUTPUT_DIR / "submissions_advanced"

# Create directories
OOF_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

# Competition settings
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
N_FOLDS = 5
TARGET_COL = 'y'


def load_and_engineer_features():
    """Load data and apply THE SAME feature engineering as XGBoost for neural networks.
    
    Key insight from NN_by_GPT5: Neural networks perform best when using the same
    feature engineering as tree-based models (XGBoost), not simplified features.
    """
    logger.info("Loading competition data...")
    
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    
    logger.info(f"Train shape: {train_df.shape}")
    logger.info(f"Test shape: {test_df.shape}")
    
    # Separate features and target
    X_train = train_df.drop([TARGET_COL, 'id'], axis=1)
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop('id', axis=1)
    
    # IMPORTANT: Use the SAME feature engineering as XGBoost!
    # This is the key insight from the NN_by_GPT5 notebook
    logger.info("Applying XGBoost-style feature engineering for Neural Network...")
    engineer = FeatureEngineer(
        categorical_impute_strategy='constant',
        numeric_impute_strategy='median',
        categorical_encoding='target',  # Same as XGBoost - target encoding
        scaling_method='standard',  # Scaling is crucial for neural networks
        create_interactions=True,  # Same as XGBoost - create interactions
        create_polynomial=True,  # Same as XGBoost - create polynomial features
        max_cardinality=20,
        min_frequency=0.01
    )
    
    X_train_fe = engineer.fit_transform(X_train, y_train)
    X_test_fe = engineer.transform(X_test)
    
    # Remove duplicate columns
    X_train_fe = X_train_fe.loc[:, ~X_train_fe.columns.duplicated()]
    X_test_fe = X_test_fe.loc[:, ~X_test_fe.columns.duplicated()]
    
    # Additional normalization for neural network (critical for convergence)
    # Neural networks need well-scaled inputs
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_fe),
        columns=X_train_fe.columns,
        index=X_train_fe.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_fe),
        columns=X_test_fe.columns,
        index=X_test_fe.index
    )
    
    logger.info(f"Final train shape: {X_train_scaled.shape} (same features as XGBoost)")
    logger.info(f"Final test shape: {X_test_scaled.shape}")
    
    return X_train_scaled, y_train, X_test_scaled, sample_sub


def train_tabnet_models(X_train, y_train, X_test, sample_sub):
    """Train multiple TabNet models with different configurations."""
    logger.info("\n" + "="*60)
    logger.info("TRAINING TABNET MODELS")
    logger.info("="*60)
    
    # Check if TabNet is available
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        logger.warning("TabNet requires PyTorch. Install with:")
        logger.warning("pip install torch")
        logger.warning("Skipping TabNet models...")
        return {}, 0.0
    
    try:
        import pytorch_tabnet
    except ImportError:
        logger.warning("TabNet requires pytorch-tabnet. Install with:")
        logger.warning("pip install pytorch-tabnet")
        logger.warning("Skipping TabNet models...")
        return {}, 0.0
    
    # Initialize OOF manager
    oof_manager = OOFManager(output_dir=str(OOF_DIR))
    
    # Initialize MLflow tracker
    mlflow_tracker = None
    if os.getenv("MLFLOW_TRACKING_URI"):
        mlflow_tracker = MLflowTracker(
            experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "PS5E8-TabNet"),
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
            tags={'stage': 'tabnet_training', 'competition': 'PS5E8'}
        )
        mlflow_tracker.start_run(run_name="PS5E8_TabNet_Advanced")
    
    all_oof_preds = []
    all_test_preds = []
    model_scores = {}
    ensemble = OOFEnsemble(task_type='classification')
    
    # Configuration 1: Standard TabNet
    logger.info("\n--- Training TabNet Standard ---")
    tabnet1_params = {
        'n_d': 24,
        'n_a': 24,
        'n_steps': 4,
        'gamma': 1.3,
        'n_independent': 2,
        'n_shared': 2,
        'lambda_sparse': 1e-3,
        'optimizer_fn': 'adam',
        'optimizer_params': {'lr': 0.02, 'weight_decay': 1e-5},
        'mask_type': 'entmax',
        'scheduler_fn': 'StepLR',
        'scheduler_params': {'step_size': 10, 'gamma': 0.9},
        'seed': RANDOM_SEED
    }
    
    # Training parameters for fit method
    fit_params = {
        'max_epochs': 100,
        'patience': 15,
        'batch_size': 1024,
        'virtual_batch_size': 128
    }
    
    try:
        tabnet1 = TabNetModel(params=tabnet1_params)
        
        oof1 = ensemble.get_oof_predictions(
            models=[tabnet1],
            X=X_train,
            y=y_train,
            n_folds=N_FOLDS,
            stratified=True,
            verbose=True
        ).iloc[:, 0]
        
        tabnet1.fit(X_train, y_train, **fit_params)
        test1 = tabnet1.predict_proba(X_test)[:, 1]
        
        score1 = roc_auc_score(y_train, oof1)
        logger.info(f"TabNet Standard CV Score: {score1:.6f}")
        
        oof_manager.save_oof(
            predictions=oof1,
            model_name="TabNet_Standard",
            model_params=tabnet1_params,
            cv_score=score1,
            test_predictions=test1,
            experiment_name='PS5E8_tabnet',
            tags={'competition': 'PS5E8', 'model_type': 'TabNet', 'version': 'standard'}
        )
        
        all_oof_preds.append(oof1)
        all_test_preds.append(test1)
        model_scores["TabNet_Standard"] = score1
        
    except Exception as e:
        logger.warning(f"TabNet Standard training failed: {e}")
    
    # Configuration 2: Deep TabNet
    logger.info("\n--- Training TabNet Deep ---")
    tabnet2_params = {
        'n_d': 32,
        'n_a': 32,
        'n_steps': 5,
        'gamma': 1.5,
        'n_independent': 3,
        'n_shared': 3,
        'lambda_sparse': 1e-4,
        'optimizer_fn': 'adam',
        'optimizer_params': {'lr': 0.015, 'weight_decay': 1e-4},
        'mask_type': 'sparsemax',
        'scheduler_fn': 'StepLR',
        'scheduler_params': {'step_size': 15, 'gamma': 0.85},
        'seed': RANDOM_SEED + 42
    }
    
    # Training parameters for fit method
    fit_params2 = {
        'max_epochs': 100,
        'patience': 20,
        'batch_size': 512,
        'virtual_batch_size': 64
    }
    
    try:
        tabnet2 = TabNetModel(params=tabnet2_params)
        
        oof2 = ensemble.get_oof_predictions(
            models=[tabnet2],
            X=X_train,
            y=y_train,
            n_folds=N_FOLDS,
            stratified=True,
            verbose=True
        ).iloc[:, 0]
        
        tabnet2.fit(X_train, y_train, **fit_params2)
        test2 = tabnet2.predict_proba(X_test)[:, 1]
        
        score2 = roc_auc_score(y_train, oof2)
        logger.info(f"TabNet Deep CV Score: {score2:.6f}")
        
        oof_manager.save_oof(
            predictions=oof2,
            model_name="TabNet_Deep",
            model_params=tabnet2_params,
            cv_score=score2,
            test_predictions=test2,
            experiment_name='PS5E8_tabnet',
            tags={'competition': 'PS5E8', 'model_type': 'TabNet', 'version': 'deep'}
        )
        
        all_oof_preds.append(oof2)
        all_test_preds.append(test2)
        model_scores["TabNet_Deep"] = score2
        
    except Exception as e:
        logger.warning(f"TabNet Deep training failed: {e}")
    
    # Configuration 3: Light TabNet (faster training)
    logger.info("\n--- Training TabNet Light ---")
    tabnet3_params = {
        'n_d': 16,
        'n_a': 16,
        'n_steps': 3,
        'gamma': 1.2,
        'n_independent': 1,
        'n_shared': 1,
        'lambda_sparse': 5e-3,
        'optimizer_fn': 'adam',
        'optimizer_params': {'lr': 0.025, 'weight_decay': 1e-6},
        'mask_type': 'entmax',
        'scheduler_fn': 'StepLR',
        'scheduler_params': {'step_size': 20, 'gamma': 0.95},
        'seed': RANDOM_SEED + 123
    }
    
    # Training parameters for fit method
    fit_params3 = {
        'max_epochs': 80,
        'patience': 10,
        'batch_size': 2048,
        'virtual_batch_size': 256
    }
    
    try:
        tabnet3 = TabNetModel(params=tabnet3_params)
        
        oof3 = ensemble.get_oof_predictions(
            models=[tabnet3],
            X=X_train,
            y=y_train,
            n_folds=N_FOLDS,
            stratified=True,
            verbose=True
        ).iloc[:, 0]
        
        tabnet3.fit(X_train, y_train, **fit_params3)
        test3 = tabnet3.predict_proba(X_test)[:, 1]
        
        score3 = roc_auc_score(y_train, oof3)
        logger.info(f"TabNet Light CV Score: {score3:.6f}")
        
        oof_manager.save_oof(
            predictions=oof3,
            model_name="TabNet_Light",
            model_params=tabnet3_params,
            cv_score=score3,
            test_predictions=test3,
            experiment_name='PS5E8_tabnet',
            tags={'competition': 'PS5E8', 'model_type': 'TabNet', 'version': 'light'}
        )
        
        all_oof_preds.append(oof3)
        all_test_preds.append(test3)
        model_scores["TabNet_Light"] = score3
        
    except Exception as e:
        logger.warning(f"TabNet Light training failed: {e}")
    
    if all_oof_preds:
        # Create simple ensemble of TabNet models
        logger.info("\n--- Creating TabNet Ensemble ---")
        tabnet_ensemble = np.mean(all_oof_preds, axis=0)
        tabnet_ensemble_test = np.mean(all_test_preds, axis=0)
        ensemble_score = roc_auc_score(y_train, tabnet_ensemble)
        logger.info(f"TabNet Ensemble CV Score: {ensemble_score:.6f}")
        
        # Save ensemble submission
        submission = sample_sub.copy()
        submission['y'] = tabnet_ensemble_test
        submission.to_csv(SUBMISSION_DIR / "submission_tabnet_ensemble.csv", index=False)
    else:
        logger.error("No TabNet models were successfully trained")
        ensemble_score = 0.0
    
    # Log to MLflow
    if mlflow_tracker:
        for model_name, score in model_scores.items():
            mlflow_tracker.log_metrics({f'{model_name}_cv_auc': score})
        if ensemble_score > 0:
            mlflow_tracker.log_metrics({'tabnet_ensemble_cv_auc': ensemble_score})
        mlflow_tracker.end_run()
    
    # Summary
    if model_scores:
        logger.info("\n" + "="*60)
        logger.info("TABNET TRAINING SUMMARY")
        logger.info("="*60)
        
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (model_name, score) in enumerate(sorted_models, 1):
            logger.info(f"{rank}. {model_name:25s}: {score:.6f}")
        
        if ensemble_score > 0:
            logger.info(f"\nTabNet Ensemble: {ensemble_score:.6f}")
            improvement = (ensemble_score - sorted_models[0][1]) * 100
            logger.info(f"Ensemble improvement: {improvement:.3f}%")
    
    return model_scores, ensemble_score


def main():
    """Main training pipeline."""
    logger.info("="*60)
    logger.info("PS5E8 COMPETITION - TABNET ADVANCED MODELS")
    logger.info("="*60)
    
    # Check for PyTorch and TabNet
    try:
        import torch
        import pytorch_tabnet
        logger.info("TabNet dependencies found. Proceeding with training...")
    except ImportError:
        logger.error("TabNet requires PyTorch and pytorch-tabnet. Install with:")
        logger.error("pip install torch pytorch-tabnet")
        logger.error("\nFor GPU support:")
        logger.error("pip install torch --index-url https://download.pytorch.org/whl/cu118")
        logger.error("pip install pytorch-tabnet")
        return
    
    # Load and engineer features
    X_train, y_train, X_test, sample_sub = load_and_engineer_features()
    
    # Train TabNet models
    model_scores, ensemble_score = train_tabnet_models(X_train, y_train, X_test, sample_sub)
    
    if model_scores:
        logger.info("\n" + "="*60)
        logger.info("TABNET TRAINING COMPLETE!")
        logger.info("="*60)
        logger.info(f"Best submission: {SUBMISSION_DIR}/submission_tabnet_ensemble.csv")
        logger.info(f"Best CV Score: {ensemble_score:.6f}")
    else:
        logger.warning("No TabNet models were trained successfully")
        logger.warning("Please ensure PyTorch and pytorch-tabnet are installed")


if __name__ == "__main__":
    main()