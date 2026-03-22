"""
PS5E8 Competition - Advanced XGBoost Models
Train multiple XGBoost configurations with advanced feature engineering
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
    FeatureEngineer, XGBoostModel, 
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

# Feature engineering settings
CREATE_CATEGORICAL_PAIRS = True
TREAT_NUMERICAL_AS_CATEGORICAL = True


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


def train_xgboost_models(X_train, y_train, X_test, sample_sub):
    """Train multiple XGBoost models with different configurations."""
    logger.info("\n" + "="*60)
    logger.info("TRAINING XGBOOST MODELS")
    logger.info("="*60)
    
    # Initialize OOF manager
    oof_manager = OOFManager(output_dir=str(OOF_DIR))
    
    # Initialize MLflow tracker
    mlflow_tracker = None
    if os.getenv("MLFLOW_TRACKING_URI"):
        mlflow_tracker = MLflowTracker(
            experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "PS5E8-XGBoost"),
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
            tags={'stage': 'xgboost_training', 'competition': 'PS5E8'}
        )
        mlflow_tracker.start_run(run_name="PS5E8_XGBoost_Advanced")
    
    all_oof_preds = []
    all_test_preds = []
    model_scores = {}
    
    # Configuration 1: Conservative with heavy regularization
    logger.info("\n--- Training XGBoost Conservative ---")
    xgb1_params = {
        'n_estimators': 800,
        'max_depth': 4,
        'learning_rate': 0.015,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'gamma': 0.1,
        'reg_alpha': 0.5,
        'reg_lambda': 2.0,
        'min_child_weight': 5,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'random_state': RANDOM_SEED,
        'verbosity': 0
    }
    
    xgb1 = XGBoostModel(params=xgb1_params)
    ensemble = OOFEnsemble(task_type='classification')
    
    oof1 = ensemble.get_oof_predictions(
        models=[xgb1],
        X=X_train,
        y=y_train,
        n_folds=N_FOLDS,
        stratified=True,
        verbose=True
    ).iloc[:, 0]
    
    xgb1.fit(X_train, y_train)
    test1 = xgb1.predict_proba(X_test)[:, 1]
    
    score1 = roc_auc_score(y_train, oof1)
    logger.info(f"XGBoost Conservative CV Score: {score1:.6f}")
    
    oof_manager.save_oof(
        predictions=oof1,
        model_name="XGBoost_Conservative",
        model_params=xgb1_params,
        cv_score=score1,
        test_predictions=test1,
        experiment_name='PS5E8_xgboost',
        tags={'competition': 'PS5E8', 'model_type': 'XGBoost', 'version': 'conservative'}
    )
    
    all_oof_preds.append(oof1)
    all_test_preds.append(test1)
    model_scores["XGBoost_Conservative"] = score1
    
    # Configuration 2: Aggressive with less regularization
    logger.info("\n--- Training XGBoost Aggressive ---")
    xgb2_params = {
        'n_estimators': 1200,
        'max_depth': 6,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.01,
        'reg_alpha': 0.1,
        'reg_lambda': 0.5,
        'min_child_weight': 3,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'max_leaves': 31,
        'grow_policy': 'lossguide',
        'random_state': RANDOM_SEED + 42,
        'verbosity': 0
    }
    
    xgb2 = XGBoostModel(params=xgb2_params)
    
    oof2 = ensemble.get_oof_predictions(
        models=[xgb2],
        X=X_train,
        y=y_train,
        n_folds=N_FOLDS,
        stratified=True,
        verbose=True
    ).iloc[:, 0]
    
    xgb2.fit(X_train, y_train)
    test2 = xgb2.predict_proba(X_test)[:, 1]
    
    score2 = roc_auc_score(y_train, oof2)
    logger.info(f"XGBoost Aggressive CV Score: {score2:.6f}")
    
    oof_manager.save_oof(
        predictions=oof2,
        model_name="XGBoost_Aggressive",
        model_params=xgb2_params,
        cv_score=score2,
        test_predictions=test2,
        experiment_name='PS5E8_xgboost',
        tags={'competition': 'PS5E8', 'model_type': 'XGBoost', 'version': 'aggressive'}
    )
    
    all_oof_preds.append(oof2)
    all_test_preds.append(test2)
    model_scores["XGBoost_Aggressive"] = score2
    
    # Configuration 3: Balanced
    logger.info("\n--- Training XGBoost Balanced ---")
    xgb3_params = {
        'n_estimators': 1000,
        'max_depth': 5,
        'learning_rate': 0.012,
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'gamma': 0.05,
        'reg_alpha': 0.3,
        'reg_lambda': 1.0,
        'min_child_weight': 4,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'random_state': RANDOM_SEED + 123,
        'verbosity': 0
    }
    
    xgb3 = XGBoostModel(params=xgb3_params)
    
    oof3 = ensemble.get_oof_predictions(
        models=[xgb3],
        X=X_train,
        y=y_train,
        n_folds=N_FOLDS,
        stratified=True,
        verbose=True
    ).iloc[:, 0]
    
    xgb3.fit(X_train, y_train)
    test3 = xgb3.predict_proba(X_test)[:, 1]
    
    score3 = roc_auc_score(y_train, oof3)
    logger.info(f"XGBoost Balanced CV Score: {score3:.6f}")
    
    oof_manager.save_oof(
        predictions=oof3,
        model_name="XGBoost_Balanced",
        model_params=xgb3_params,
        cv_score=score3,
        test_predictions=test3,
        experiment_name='PS5E8_xgboost',
        tags={'competition': 'PS5E8', 'model_type': 'XGBoost', 'version': 'balanced'}
    )
    
    all_oof_preds.append(oof3)
    all_test_preds.append(test3)
    model_scores["XGBoost_Balanced"] = score3
    
    # Create simple ensemble of XGBoost models
    logger.info("\n--- Creating XGBoost Ensemble ---")
    xgb_ensemble = np.mean(all_oof_preds, axis=0)
    xgb_ensemble_test = np.mean(all_test_preds, axis=0)
    ensemble_score = roc_auc_score(y_train, xgb_ensemble)
    logger.info(f"XGBoost Ensemble CV Score: {ensemble_score:.6f}")
    
    # Save ensemble submission
    submission = sample_sub.copy()
    submission['y'] = xgb_ensemble_test
    submission.to_csv(SUBMISSION_DIR / "submission_xgboost_ensemble.csv", index=False)
    
    # Log to MLflow
    if mlflow_tracker:
        for model_name, score in model_scores.items():
            mlflow_tracker.log_metrics({f'{model_name}_cv_auc': score})
        mlflow_tracker.log_metrics({'xgboost_ensemble_cv_auc': ensemble_score})
        mlflow_tracker.end_run()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("XGBOOST TRAINING SUMMARY")
    logger.info("="*60)
    
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (model_name, score) in enumerate(sorted_models, 1):
        logger.info(f"{rank}. {model_name:25s}: {score:.6f}")
    
    logger.info(f"\nXGBoost Ensemble: {ensemble_score:.6f}")
    improvement = (ensemble_score - sorted_models[0][1]) * 100
    logger.info(f"Ensemble improvement: {improvement:.3f}%")
    
    return model_scores, ensemble_score


def main():
    """Main training pipeline."""
    logger.info("="*60)
    logger.info("PS5E8 COMPETITION - XGBOOST ADVANCED MODELS")
    logger.info("="*60)
    
    # Load and engineer features
    X_train, y_train, X_test, sample_sub = load_and_engineer_features()
    
    # Train XGBoost models
    model_scores, ensemble_score = train_xgboost_models(X_train, y_train, X_test, sample_sub)
    
    logger.info("\n" + "="*60)
    logger.info("XGBOOST TRAINING COMPLETE!")
    logger.info("="*60)
    logger.info(f"Best submission: {SUBMISSION_DIR}/submission_xgboost_ensemble.csv")
    logger.info(f"Best CV Score: {ensemble_score:.6f}")


if __name__ == "__main__":
    main()