"""
PS5E8 Competition - AutoGluon Models
Train AutoGluon with automatic model selection and ensembling
Key insight: AutoGluon handles its own internal ensembling and stacking
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import tabml
sys.path.append(str(Path(__file__).parent.parent.parent))

from tabml import (
    FeatureEngineer, OOFManager, OOFEnsemble
)

try:
    from tabml import AutoGluonModel
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False
    logger.warning("AutoGluon not available in TabML")

# Load environment variables
load_dotenv()

# Setup paths
DATA_DIR = Path("../../data/raw/PS5E8")
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
    """Load data and apply feature engineering similar to other models."""
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
    
    # Apply feature engineering (similar to XGBoost/LightGBM)
    logger.info("Applying feature engineering...")
    engineer = FeatureEngineer(
        categorical_impute_strategy='constant',
        numeric_impute_strategy='median',
        categorical_encoding='target',
        scaling_method='standard',
        create_interactions=True,
        create_polynomial=True,
        max_cardinality=20,
        min_frequency=0.01
    )
    
    X_train_fe = engineer.fit_transform(X_train, y_train)
    X_test_fe = engineer.transform(X_test)
    
    # Remove duplicate columns if any
    X_train_fe = X_train_fe.loc[:, ~X_train_fe.columns.duplicated()]
    X_test_fe = X_test_fe.loc[:, ~X_test_fe.columns.duplicated()]
    
    logger.info(f"Final train shape: {X_train_fe.shape}")
    logger.info(f"Final test shape: {X_test_fe.shape}")
    
    return X_train_fe, y_train, X_test_fe, sample_sub


def train_autogluon_models(X_train, y_train, X_test, sample_sub):
    """Train AutoGluon models with different configurations."""
    logger.info("\n" + "="*60)
    logger.info("TRAINING AUTOGLUON MODELS")
    logger.info("="*60)
    
    if not AUTOGLUON_AVAILABLE:
        logger.error("AutoGluon is not available. Install with:")
        logger.error("pip install autogluon")
        return {}, 0.0
    
    # Initialize OOF manager
    oof_manager = OOFManager(output_dir=str(OOF_DIR))
    
    # Initialize MLflow tracker if available
    mlflow_tracker = None
    try:
        if os.getenv("MLFLOW_TRACKING_URI"):
            from tabml import MLflowTracker
            mlflow_tracker = MLflowTracker(
                experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "PS5E8-AutoGluon"),
                tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
                tags={'stage': 'autogluon_training', 'competition': 'PS5E8'}
            )
            mlflow_tracker.start_run(run_name="PS5E8_AutoGluon")
    except ImportError:
        pass
    
    all_oof_preds = []
    all_test_preds = []
    model_scores = {}
    
    # Configuration 1: Fast mode with limited time
    logger.info("\n--- Training AutoGluon Fast Mode ---")
    ag1_params = {
        'time_limit': 300,  # 5 minutes
        'presets': 'good_quality',
        'eval_metric': 'roc_auc',
        'num_bag_folds': 0,  # No bagging for speed
        'num_stack_levels': 0,  # No stacking for speed
        'holdout_frac': 0.2,
        'save_space': True,
        'verbosity': 2,
        'excluded_model_types': ['KNN', 'NN_TORCH'],  # Exclude slow models
    }
    
    ag1 = AutoGluonModel(params=ag1_params)
    ensemble = OOFEnsemble(task_type='classification', random_state=RANDOM_SEED)
    
    # Get OOF predictions using cross-validation
    oof1 = ensemble.get_oof_predictions(
        models=[ag1],
        X=X_train,
        y=y_train,
        n_folds=N_FOLDS,
        stratified=True
    )
    
    # Train on full data for test predictions
    ag1.fit(X_train, y_train)
    test1 = ag1.predict_proba(X_test)[:, 1] if ag1.is_classification else ag1.predict(X_test)
    score1 = roc_auc_score(y_train, oof1)
    logger.info(f"AutoGluon Fast Mode CV Score: {score1:.6f}")
    
    # Save predictions
    oof_manager.save_oof(
        predictions=oof1,
        model_name="AutoGluon_Fast",
        model_params=ag1_params,
        cv_score=score1,
        test_predictions=test1,
        experiment_name='PS5E8_autogluon',
        tags={'competition': 'PS5E8', 'model_type': 'AutoGluon', 'preset': 'fast'}
    )
    
    all_oof_preds.append(oof1)
    all_test_preds.append(test1)
    model_scores["AutoGluon_Fast"] = score1
    
    # Configuration 2: Best quality with bagging (if time permits)
    logger.info("\n--- Training AutoGluon Best Quality ---")
    ag2_params = {
        'time_limit': 1200,  # 20 minutes
        'presets': 'best_quality',
        'eval_metric': 'roc_auc',
        'num_bag_folds': 5,  # Enable bagging for better performance
        'num_bag_sets': 1,
        'num_stack_levels': 1,  # One level of stacking
        'auto_stack': True,  # Automatic stacking
        'holdout_frac': 0,  # Use all data with bagging
        'save_space': False,  # Keep all models
        'verbosity': 2,
        'keep_only_best': False,  # Keep all models for ensemble
        'save_bag_folds': True,  # Save OOF predictions
    }
    
    ag2 = AutoGluonModel(params=ag2_params)
    
    # Get OOF predictions
    oof2 = ensemble.get_oof_predictions(
        models=[ag2],
        X=X_train,
        y=y_train,
        n_folds=N_FOLDS,
        stratified=True
    )
    
    # Train on full data for test predictions
    ag2.fit(X_train, y_train)
    test2 = ag2.predict_proba(X_test)[:, 1] if ag2.is_classification else ag2.predict(X_test)
    score2 = roc_auc_score(y_train, oof2)
    logger.info(f"AutoGluon Best Quality CV Score: {score2:.6f}")
    
    # Try to get AutoGluon's internal OOF predictions if available
    try:
        # Note: This requires the last fold's model to have bagging enabled
        ag2_internal_oof = ag2.get_oof_predictions()
        if ag2_internal_oof is not None:
            logger.info("AutoGluon internal OOF predictions available")
            internal_score = roc_auc_score(y_train, ag2_internal_oof)
            logger.info(f"AutoGluon internal OOF score: {internal_score:.6f}")
    except:
        pass
    
    # Save predictions
    oof_manager.save_oof(
        predictions=oof2,
        model_name="AutoGluon_BestQuality",
        model_params=ag2_params,
        cv_score=score2,
        test_predictions=test2,
        experiment_name='PS5E8_autogluon',
        tags={'competition': 'PS5E8', 'model_type': 'AutoGluon', 'preset': 'best_quality'}
    )
    
    all_oof_preds.append(oof2)
    all_test_preds.append(test2)
    model_scores["AutoGluon_BestQuality"] = score2
    
    # Configuration 3: Custom hyperparameters focusing on tree models
    logger.info("\n--- Training AutoGluon Custom Trees ---")
    ag3_params = {
        'time_limit': 600,  # 10 minutes
        'presets': 'medium_quality',
        'eval_metric': 'roc_auc',
        'num_bag_folds': 3,  # Some bagging
        'num_stack_levels': 0,  # No stacking
        'holdout_frac': 0.15,
        'save_space': True,
        'verbosity': 2,
        # Focus on tree-based models which typically perform well on tabular data
        'hyperparameters': {
            'GBM': [
                {'num_boost_round': 100, 'learning_rate': 0.1},
                {'num_boost_round': 500, 'learning_rate': 0.03},
            ],
            'XGB': {
                'n_estimators': 300,
                'learning_rate': 0.05,
                'max_depth': 6,
            },
            'CAT': {
                'iterations': 500,
                'learning_rate': 0.05,
                'depth': 6,
            },
            'RF': {
                'n_estimators': 300,
                'max_depth': None,
            },
            'XT': {
                'n_estimators': 300,
                'max_depth': None,
            }
        },
        'excluded_model_types': ['KNN', 'NN_TORCH', 'FASTAI'],  # Exclude non-tree models
    }
    
    ag3 = AutoGluonModel(params=ag3_params)
    
    # Get OOF predictions
    oof3 = ensemble.get_oof_predictions(
        models=[ag3],
        X=X_train,
        y=y_train,
        n_folds=N_FOLDS,
        stratified=True
    )
    
    # Train on full data for test predictions
    ag3.fit(X_train, y_train)
    test3 = ag3.predict_proba(X_test)[:, 1] if ag3.is_classification else ag3.predict(X_test)
    score3 = roc_auc_score(y_train, oof3)
    logger.info(f"AutoGluon Custom Trees CV Score: {score3:.6f}")
    
    # Get model information
    try:
        model_info = ag3.get_model_info()
        logger.info(f"Best model: {model_info.get('best_model', 'N/A')}")
        logger.info(f"Number of models trained: {model_info.get('num_models', 'N/A')}")
    except:
        pass
    
    # Save predictions
    oof_manager.save_oof(
        predictions=oof3,
        model_name="AutoGluon_CustomTrees",
        model_params=ag3_params,
        cv_score=score3,
        test_predictions=test3,
        experiment_name='PS5E8_autogluon',
        tags={'competition': 'PS5E8', 'model_type': 'AutoGluon', 'preset': 'custom_trees'}
    )
    
    all_oof_preds.append(oof3)
    all_test_preds.append(test3)
    model_scores["AutoGluon_CustomTrees"] = score3
    
    # Create ensemble of AutoGluon models
    logger.info("\n--- Creating AutoGluon Ensemble ---")
    ag_ensemble = np.mean(all_oof_preds, axis=0)
    ag_ensemble_test = np.mean(all_test_preds, axis=0)
    ensemble_score = roc_auc_score(y_train, ag_ensemble)
    logger.info(f"AutoGluon Ensemble CV Score: {ensemble_score:.6f}")
    
    # Save ensemble predictions
    oof_manager.save_oof(
        predictions=ag_ensemble,
        model_name="AutoGluon_Ensemble",
        model_params={'models': list(model_scores.keys())},
        cv_score=ensemble_score,
        test_predictions=ag_ensemble_test,
        experiment_name='PS5E8_autogluon',
        tags={'competition': 'PS5E8', 'model_type': 'AutoGluon', 'ensemble': 'mean'}
    )
    
    # Save submission
    submission = sample_sub.copy()
    submission['y'] = ag_ensemble_test
    submission.to_csv(SUBMISSION_DIR / "submission_autogluon_ensemble.csv", index=False)
    logger.info(f"Saved submission to {SUBMISSION_DIR}/submission_autogluon_ensemble.csv")
    
    # Log to MLflow if available
    if mlflow_tracker:
        mlflow_tracker.log_metrics({
            'cv_auc_fast': model_scores.get("AutoGluon_Fast", 0),
            'cv_auc_best': model_scores.get("AutoGluon_BestQuality", 0),
            'cv_auc_custom': model_scores.get("AutoGluon_CustomTrees", 0),
            'cv_auc_ensemble': ensemble_score
        })
        mlflow_tracker.end_run()
    
    # Clean up AutoGluon temp files
    for model in [ag1, ag2, ag3]:
        try:
            model.cleanup()
        except:
            pass
    
    return model_scores, ensemble_score


def main():
    """Main training pipeline."""
    logger.info("="*60)
    logger.info("PS5E8 COMPETITION - AUTOGLUON MODELS")
    logger.info("Automatic model selection and ensembling")
    logger.info("="*60)
    
    # Check for AutoGluon
    if not AUTOGLUON_AVAILABLE:
        logger.error("AutoGluon is required. Install with:")
        logger.error("pip install autogluon")
        logger.error("\nFor GPU support:")
        logger.error("pip install autogluon[torch]")
        return
    
    # Load and engineer features
    X_train, y_train, X_test, sample_sub = load_and_engineer_features()
    
    # Train AutoGluon models
    model_scores, ensemble_score = train_autogluon_models(X_train, y_train, X_test, sample_sub)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("AUTOGLUON TRAINING COMPLETE!")
    logger.info("="*60)
    logger.info("\nModel Scores:")
    for model_name, score in model_scores.items():
        logger.info(f"  {model_name}: {score:.6f}")
    logger.info(f"\nEnsemble Score: {ensemble_score:.6f}")
    logger.info(f"\nBest submission: {SUBMISSION_DIR}/submission_autogluon_ensemble.csv")
    logger.info("\nKey advantages of AutoGluon:")
    logger.info("  - Automatic model selection")
    logger.info("  - Built-in ensembling and stacking")
    logger.info("  - Automatic hyperparameter tuning")
    logger.info("  - OOF predictions compatible with TabML ensemble")


if __name__ == "__main__":
    main()