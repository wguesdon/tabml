"""
PS5E8 Competition - Advanced LightGBM Models
Train multiple LightGBM configurations with advanced feature engineering
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import tabml
sys.path.append(str(Path(__file__).parent.parent.parent))

from tabml import (
    FeatureEngineer, LightGBMModel, 
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
    
    # Basic feature engineering with FeatureEngineer
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
    
    return X_train_fe, y_train, X_test_fe, sample_sub


def train_lightgbm_models(X_train, y_train, X_test, sample_sub):
    """Train multiple LightGBM models with different configurations."""
    logger.info("\n" + "="*60)
    logger.info("TRAINING LIGHTGBM MODELS")
    logger.info("="*60)
    
    # Initialize OOF manager
    oof_manager = OOFManager(output_dir=str(OOF_DIR))
    
    # Initialize MLflow tracker
    mlflow_tracker = None
    if os.getenv("MLFLOW_TRACKING_URI"):
        mlflow_tracker = MLflowTracker(
            experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "PS5E8-LightGBM"),
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
            tags={'stage': 'lightgbm_training', 'competition': 'PS5E8'}
        )
        mlflow_tracker.start_run(run_name="PS5E8_LightGBM_Advanced")
    
    all_oof_preds = []
    all_test_preds = []
    model_scores = {}
    ensemble = OOFEnsemble(task_type='classification')
    
    # Configuration 1: Standard GBDT
    logger.info("\n--- Training LightGBM Standard ---")
    lgb1_params = {
        'n_estimators': 1000,
        'num_leaves': 31,
        'learning_rate': 0.01,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'lambda_l1': 0.5,
        'lambda_l2': 1.0,
        'min_data_in_leaf': 20,
        'min_gain_to_split': 0.01,
        'max_bin': 255,
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'random_state': RANDOM_SEED,
        'verbosity': -1
    }
    
    lgb1 = LightGBMModel(params=lgb1_params)
    
    oof1 = ensemble.get_oof_predictions(
        models=[lgb1],
        X=X_train,
        y=y_train,
        n_folds=N_FOLDS,
        stratified=True,
        verbose=True
    ).iloc[:, 0]
    
    lgb1.fit(X_train, y_train)
    test1 = lgb1.predict_proba(X_test)[:, 1]
    
    score1 = roc_auc_score(y_train, oof1)
    logger.info(f"LightGBM Standard CV Score: {score1:.6f}")
    
    oof_manager.save_oof(
        predictions=oof1,
        model_name="LightGBM_Standard",
        model_params=lgb1_params,
        cv_score=score1,
        test_predictions=test1,
        experiment_name='PS5E8_lightgbm',
        tags={'competition': 'PS5E8', 'model_type': 'LightGBM', 'version': 'standard'}
    )
    
    all_oof_preds.append(oof1)
    all_test_preds.append(test1)
    model_scores["LightGBM_Standard"] = score1
    
    # Configuration 2: DART (Dropouts meet Multiple Additive Regression Trees)
    logger.info("\n--- Training LightGBM DART ---")
    lgb2_params = {
        'n_estimators': 500,
        'num_leaves': 25,
        'learning_rate': 0.015,
        'boosting_type': 'dart',
        'drop_rate': 0.1,
        'skip_drop': 0.5,
        'max_drop': 50,
        'feature_fraction': 0.6,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'lambda_l1': 0.3,
        'lambda_l2': 0.5,
        'min_data_in_leaf': 30,
        'objective': 'binary',
        'metric': 'auc',
        'random_state': RANDOM_SEED + 42,
        'verbosity': -1
    }
    
    lgb2 = LightGBMModel(params=lgb2_params)
    
    oof2 = ensemble.get_oof_predictions(
        models=[lgb2],
        X=X_train,
        y=y_train,
        n_folds=N_FOLDS,
        stratified=True,
        verbose=True
    ).iloc[:, 0]
    
    lgb2.fit(X_train, y_train)
    test2 = lgb2.predict_proba(X_test)[:, 1]
    
    score2 = roc_auc_score(y_train, oof2)
    logger.info(f"LightGBM DART CV Score: {score2:.6f}")
    
    oof_manager.save_oof(
        predictions=oof2,
        model_name="LightGBM_DART",
        model_params=lgb2_params,
        cv_score=score2,
        test_predictions=test2,
        experiment_name='PS5E8_lightgbm',
        tags={'competition': 'PS5E8', 'model_type': 'LightGBM', 'version': 'dart'}
    )
    
    all_oof_preds.append(oof2)
    all_test_preds.append(test2)
    model_scores["LightGBM_DART"] = score2
    
    # Configuration 3: GOSS (Gradient-based One-Side Sampling)
    logger.info("\n--- Training LightGBM GOSS ---")
    lgb3_params = {
        'n_estimators': 800,
        'num_leaves': 35,
        'learning_rate': 0.012,
        'boosting_type': 'goss',
        'top_rate': 0.2,
        'other_rate': 0.1,
        'feature_fraction': 0.8,
        'lambda_l1': 0.2,
        'lambda_l2': 0.8,
        'min_data_in_leaf': 25,
        'min_gain_to_split': 0.01,
        'objective': 'binary',
        'metric': 'auc',
        'random_state': RANDOM_SEED + 123,
        'verbosity': -1
    }
    
    lgb3 = LightGBMModel(params=lgb3_params)
    
    oof3 = ensemble.get_oof_predictions(
        models=[lgb3],
        X=X_train,
        y=y_train,
        n_folds=N_FOLDS,
        stratified=True,
        verbose=True
    ).iloc[:, 0]
    
    lgb3.fit(X_train, y_train)
    test3 = lgb3.predict_proba(X_test)[:, 1]
    
    score3 = roc_auc_score(y_train, oof3)
    logger.info(f"LightGBM GOSS CV Score: {score3:.6f}")
    
    oof_manager.save_oof(
        predictions=oof3,
        model_name="LightGBM_GOSS",
        model_params=lgb3_params,
        cv_score=score3,
        test_predictions=test3,
        experiment_name='PS5E8_lightgbm',
        tags={'competition': 'PS5E8', 'model_type': 'LightGBM', 'version': 'goss'}
    )
    
    all_oof_preds.append(oof3)
    all_test_preds.append(test3)
    model_scores["LightGBM_GOSS"] = score3
    
    # Create simple ensemble of LightGBM models
    logger.info("\n--- Creating LightGBM Ensemble ---")
    lgb_ensemble = np.mean(all_oof_preds, axis=0)
    lgb_ensemble_test = np.mean(all_test_preds, axis=0)
    ensemble_score = roc_auc_score(y_train, lgb_ensemble)
    logger.info(f"LightGBM Ensemble CV Score: {ensemble_score:.6f}")
    
    # Save ensemble submission
    submission = sample_sub.copy()
    submission['y'] = lgb_ensemble_test
    submission.to_csv(SUBMISSION_DIR / "submission_lightgbm_ensemble.csv", index=False)
    
    # Log to MLflow
    if mlflow_tracker:
        for model_name, score in model_scores.items():
            mlflow_tracker.log_metrics({f'{model_name}_cv_auc': score})
        mlflow_tracker.log_metrics({'lightgbm_ensemble_cv_auc': ensemble_score})
        mlflow_tracker.end_run()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("LIGHTGBM TRAINING SUMMARY")
    logger.info("="*60)
    
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (model_name, score) in enumerate(sorted_models, 1):
        logger.info(f"{rank}. {model_name:25s}: {score:.6f}")
    
    logger.info(f"\nLightGBM Ensemble: {ensemble_score:.6f}")
    improvement = (ensemble_score - sorted_models[0][1]) * 100
    logger.info(f"Ensemble improvement: {improvement:.3f}%")
    
    return model_scores, ensemble_score


def main():
    """Main training pipeline."""
    logger.info("="*60)
    logger.info("PS5E8 COMPETITION - LIGHTGBM ADVANCED MODELS")
    logger.info("="*60)
    
    # Load and engineer features
    X_train, y_train, X_test, sample_sub = load_and_engineer_features()
    
    # Train LightGBM models
    model_scores, ensemble_score = train_lightgbm_models(X_train, y_train, X_test, sample_sub)
    
    logger.info("\n" + "="*60)
    logger.info("LIGHTGBM TRAINING COMPLETE!")
    logger.info("="*60)
    logger.info(f"Best submission: {SUBMISSION_DIR}/submission_lightgbm_ensemble.csv")
    logger.info(f"Best CV Score: {ensemble_score:.6f}")


if __name__ == "__main__":
    main()