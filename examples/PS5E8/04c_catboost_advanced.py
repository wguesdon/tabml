"""
PS5E8 Competition - Advanced CatBoost Models
Train multiple CatBoost configurations with advanced feature engineering
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import tabml
sys.path.append(str(Path(__file__).parent.parent.parent))

from tabml import (
    FeatureEngineer, CatBoostModel, 
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
    """Load data and apply feature engineering."""
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
    
    # Apply feature engineering (same as XGBoost/LightGBM)
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
    
    # Remove duplicate columns
    X_train_fe = X_train_fe.loc[:, ~X_train_fe.columns.duplicated()]
    X_test_fe = X_test_fe.loc[:, ~X_test_fe.columns.duplicated()]
    
    logger.info(f"Final train shape: {X_train_fe.shape}")
    logger.info(f"Final test shape: {X_test_fe.shape}")
    
    # CatBoost doesn't need categorical indices when all features are numeric
    return X_train_fe, y_train, X_test_fe, sample_sub


def train_catboost_models(X_train, y_train, X_test, sample_sub):
    """Train multiple CatBoost models with different configurations."""
    logger.info("\n" + "="*60)
    logger.info("TRAINING CATBOOST MODELS")
    logger.info("="*60)
    
    # Initialize OOF manager
    oof_manager = OOFManager(output_dir=str(OOF_DIR))
    
    # Initialize MLflow tracker
    mlflow_tracker = None
    if os.getenv("MLFLOW_TRACKING_URI"):
        mlflow_tracker = MLflowTracker(
            experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "PS5E8-CatBoost"),
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
            tags={'stage': 'catboost_training', 'competition': 'PS5E8'}
        )
        mlflow_tracker.start_run(run_name="PS5E8_CatBoost_Advanced")
    
    all_oof_preds = []
    all_test_preds = []
    model_scores = {}
    ensemble = OOFEnsemble(task_type='classification')
    
    # Configuration 1: Standard CatBoost
    logger.info("\n--- Training CatBoost Standard ---")
    cat1_params = {
        'iterations': 800,
        'depth': 6,
        'learning_rate': 0.015,
        'l2_leaf_reg': 3,
        'border_count': 128,
        'bagging_temperature': 0.5,
        'random_strength': 0.5,
        'od_type': 'Iter',
        'od_wait': 50,
        'random_seed': RANDOM_SEED,
        'verbose': False
    }
    
    cat1 = CatBoostModel(params=cat1_params)
    
    oof1 = ensemble.get_oof_predictions(
        models=[cat1],
        X=X_train,
        y=y_train,
        n_folds=N_FOLDS,
        stratified=True,
        verbose=True
    ).iloc[:, 0]
    
    cat1.fit(X_train, y_train)
    test1 = cat1.predict_proba(X_test)[:, 1]
    
    score1 = roc_auc_score(y_train, oof1)
    logger.info(f"CatBoost Standard CV Score: {score1:.6f}")
    
    oof_manager.save_oof(
        predictions=oof1,
        model_name="CatBoost_Standard",
        model_params=cat1_params,
        cv_score=score1,
        test_predictions=test1,
        experiment_name='PS5E8_catboost',
        tags={'competition': 'PS5E8', 'model_type': 'CatBoost', 'version': 'standard'}
    )
    
    all_oof_preds.append(oof1)
    all_test_preds.append(test1)
    model_scores["CatBoost_Standard"] = score1
    
    # Configuration 2: Deep CatBoost
    logger.info("\n--- Training CatBoost Deep ---")
    cat2_params = {
        'iterations': 600,
        'depth': 8,
        'learning_rate': 0.01,
        'l2_leaf_reg': 5,
        'border_count': 254,
        'bagging_temperature': 1.0,
        'random_strength': 1.0,
        'od_type': 'Iter',
        'od_wait': 30,
        'random_seed': RANDOM_SEED + 42,
        'verbose': False
    }
    
    cat2 = CatBoostModel(params=cat2_params)
    
    oof2 = ensemble.get_oof_predictions(
        models=[cat2],
        X=X_train,
        y=y_train,
        n_folds=N_FOLDS,
        stratified=True,
        verbose=True
    ).iloc[:, 0]
    
    cat2.fit(X_train, y_train)
    test2 = cat2.predict_proba(X_test)[:, 1]
    
    score2 = roc_auc_score(y_train, oof2)
    logger.info(f"CatBoost Deep CV Score: {score2:.6f}")
    
    oof_manager.save_oof(
        predictions=oof2,
        model_name="CatBoost_Deep",
        model_params=cat2_params,
        cv_score=score2,
        test_predictions=test2,
        experiment_name='PS5E8_catboost',
        tags={'competition': 'PS5E8', 'model_type': 'CatBoost', 'version': 'deep'}
    )
    
    all_oof_preds.append(oof2)
    all_test_preds.append(test2)
    model_scores["CatBoost_Deep"] = score2
    
    # Configuration 3: Regularized CatBoost
    logger.info("\n--- Training CatBoost Regularized ---")
    cat3_params = {
        'iterations': 1000,
        'depth': 5,
        'learning_rate': 0.012,
        'l2_leaf_reg': 10,
        'border_count': 64,
        'bagging_temperature': 0.2,
        'random_strength': 0.2,
        'subsample': 0.8,
        'od_type': 'Iter',
        'od_wait': 40,
        'random_seed': RANDOM_SEED + 123,
        'verbose': False
    }
    
    cat3 = CatBoostModel(params=cat3_params)
    
    oof3 = ensemble.get_oof_predictions(
        models=[cat3],
        X=X_train,
        y=y_train,
        n_folds=N_FOLDS,
        stratified=True,
        verbose=True
    ).iloc[:, 0]
    
    cat3.fit(X_train, y_train)
    test3 = cat3.predict_proba(X_test)[:, 1]
    
    score3 = roc_auc_score(y_train, oof3)
    logger.info(f"CatBoost Regularized CV Score: {score3:.6f}")
    
    oof_manager.save_oof(
        predictions=oof3,
        model_name="CatBoost_Regularized",
        model_params=cat3_params,
        cv_score=score3,
        test_predictions=test3,
        experiment_name='PS5E8_catboost',
        tags={'competition': 'PS5E8', 'model_type': 'CatBoost', 'version': 'regularized'}
    )
    
    all_oof_preds.append(oof3)
    all_test_preds.append(test3)
    model_scores["CatBoost_Regularized"] = score3
    
    # Create simple ensemble of CatBoost models
    logger.info("\n--- Creating CatBoost Ensemble ---")
    cat_ensemble = np.mean(all_oof_preds, axis=0)
    cat_ensemble_test = np.mean(all_test_preds, axis=0)
    ensemble_score = roc_auc_score(y_train, cat_ensemble)
    logger.info(f"CatBoost Ensemble CV Score: {ensemble_score:.6f}")
    
    # Save ensemble submission
    submission = sample_sub.copy()
    submission['y'] = cat_ensemble_test
    submission.to_csv(SUBMISSION_DIR / "submission_catboost_ensemble.csv", index=False)
    
    # Log to MLflow
    if mlflow_tracker:
        for model_name, score in model_scores.items():
            mlflow_tracker.log_metrics({f'{model_name}_cv_auc': score})
        mlflow_tracker.log_metrics({'catboost_ensemble_cv_auc': ensemble_score})
        mlflow_tracker.end_run()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("CATBOOST TRAINING SUMMARY")
    logger.info("="*60)
    
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (model_name, score) in enumerate(sorted_models, 1):
        logger.info(f"{rank}. {model_name:25s}: {score:.6f}")
    
    logger.info(f"\nCatBoost Ensemble: {ensemble_score:.6f}")
    improvement = (ensemble_score - sorted_models[0][1]) * 100
    logger.info(f"Ensemble improvement: {improvement:.3f}%")
    
    return model_scores, ensemble_score


def main():
    """Main training pipeline."""
    logger.info("="*60)
    logger.info("PS5E8 COMPETITION - CATBOOST ADVANCED MODELS")
    logger.info("="*60)
    
    # Load and engineer features
    X_train, y_train, X_test, sample_sub = load_and_engineer_features()
    
    # Train CatBoost models
    model_scores, ensemble_score = train_catboost_models(
        X_train, y_train, X_test, sample_sub
    )
    
    logger.info("\n" + "="*60)
    logger.info("CATBOOST TRAINING COMPLETE!")
    logger.info("="*60)
    logger.info(f"Best submission: {SUBMISSION_DIR}/submission_catboost_ensemble.csv")
    logger.info(f"Best CV Score: {ensemble_score:.6f}")


if __name__ == "__main__":
    main()