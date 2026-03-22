"""
PS5E8 Competition - Baseline Models Training
Train multiple models and save OOF predictions for ensemble
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
from loguru import logger

# Add parent directory to path to import tabml
sys.path.append(str(Path(__file__).parent.parent.parent))

from tabml import (
    DataLoader, FeatureEngineer, AdvancedFeatureEngineer,
    XGBoostModel, LightGBMModel, CatBoostModel, RandomForestModel,
    OOFEnsemble, OOFManager, MLflowTracker
)

# Load environment variables
load_dotenv()

# Setup paths
DATA_DIR = Path("../../data/raw/ PS5E8")
OUTPUT_DIR = Path("output")
OOF_DIR = OUTPUT_DIR / "oof_predictions"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"

# Create directories
OOF_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

# Competition settings
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
N_FOLDS = 5
TARGET_COL = 'y'


def load_competition_data():
    """Load PS5E8 competition data."""
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
    
    # Store IDs for submission
    train_ids = train_df['id']
    test_ids = test_df['id']
    
    return X_train, y_train, X_test, train_ids, test_ids, sample_sub


def engineer_features(X_train, y_train, X_test):
    """Apply feature engineering."""
    logger.info("Engineering features...")
    
    # Basic feature engineering
    # Note: FeatureEngineer auto-detects categorical and numerical columns
    engineer = FeatureEngineer(
        categorical_impute_strategy='constant',
        numeric_impute_strategy='median',
        categorical_encoding='target',  # Target encoding for categorical
        scaling_method='standard',
        max_cardinality=20,  # Limit cardinality for categorical features
        min_frequency=0.01
    )
    
    X_train_fe = engineer.fit_transform(X_train, y_train)
    X_test_fe = engineer.transform(X_test)
    
    # For now, skip polynomial features to avoid complexity
    # Just use the engineered features directly
    X_train_final = X_train_fe
    X_test_final = X_test_fe
    
    # Remove duplicate columns if any
    X_train_final = X_train_final.loc[:, ~X_train_final.columns.duplicated()]
    X_test_final = X_test_final.loc[:, ~X_test_final.columns.duplicated()]
    
    logger.info(f"Final train shape: {X_train_final.shape}")
    logger.info(f"Final test shape: {X_test_final.shape}")
    
    return X_train_final, X_test_final


def train_model_with_tracking(model, model_name, X_train, y_train, X_test, 
                             oof_manager, mlflow_tracker=None):
    """Train a single model with OOF predictions and MLflow tracking."""
    
    logger.info(f"\nTraining {model_name}...")
    
    # Start MLflow run if tracker provided
    if mlflow_tracker:
        mlflow_tracker.start_run(run_name=model_name, nested=True)
        mlflow_tracker.log_params({
            'model_type': model.__class__.__name__,
            'n_folds': N_FOLDS,
            'random_seed': RANDOM_SEED
        })
        
        # Log model parameters
        if hasattr(model, 'params'):
            mlflow_tracker.log_params(model.params)
    
    # Generate OOF predictions
    ensemble = OOFEnsemble(task_type='classification')
    oof_preds = ensemble.get_oof_predictions(
        models=[model],
        X=X_train,
        y=y_train,
        n_folds=N_FOLDS,
        stratified=True,
        verbose=True
    )
    
    # Calculate CV score
    cv_score = roc_auc_score(y_train, oof_preds.iloc[:, 0])
    logger.info(f"{model_name} CV Score: {cv_score:.6f}")
    
    # Log metrics to MLflow
    if mlflow_tracker:
        mlflow_tracker.log_metrics({'cv_auc': cv_score})
    
    # Train on full data for test predictions
    model.fit(X_train, y_train)
    
    # Generate test predictions
    if hasattr(model, 'predict_proba'):
        test_preds = model.predict_proba(X_test)[:, 1]
    else:
        test_preds = model.predict(X_test)
    
    # Save OOF predictions
    oof_manager.save_oof(
        predictions=oof_preds.iloc[:, 0],
        model_name=model_name,
        model_params=model.params if hasattr(model, 'params') else {},
        cv_score=cv_score,
        test_predictions=test_preds,
        experiment_name='PS5E8_baseline',
        tags={'competition': 'PS5E8', 'model_type': model.__class__.__name__}
    )
    
    # Log model to MLflow (skip for now due to compatibility issues)
    if mlflow_tracker:
        # mlflow_tracker.log_model(model, model_name)  # Skip model logging
        mlflow_tracker.end_run()
    
    return oof_preds.iloc[:, 0], test_preds, cv_score


def main():
    """Main training pipeline."""
    
    logger.info("="*60)
    logger.info("PS5E8 COMPETITION - BASELINE MODELS TRAINING")
    logger.info("="*60)
    
    # Initialize MLflow tracker
    mlflow_tracker = None
    if os.getenv("MLFLOW_TRACKING_URI"):
        mlflow_tracker = MLflowTracker(
            experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "PS5E8-Bank-Deposit"),
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
            tags={'stage': 'baseline_training', 'competition': 'PS5E8'}
        )
        mlflow_tracker.start_run(run_name="PS5E8_Baseline_Training")
    
    # Load data
    X_train, y_train, X_test, train_ids, test_ids, sample_sub = load_competition_data()
    
    # Feature engineering
    X_train_fe, X_test_fe = engineer_features(X_train, y_train, X_test)
    
    # Initialize OOF manager
    oof_manager = OOFManager(output_dir=str(OOF_DIR))
    
    # Store all predictions for simple ensemble
    all_oof_preds = []
    all_test_preds = []
    model_scores = {}
    
    # ==========================================
    # Train diverse models
    # ==========================================
    
    # Model 1: XGBoost (conservative)
    xgb1 = XGBoostModel(params={
        'n_estimators': 500,
        'max_depth': 4,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': RANDOM_SEED
    })
    oof1, test1, score1 = train_model_with_tracking(
        xgb1, "XGBoost_Conservative", X_train_fe, y_train, X_test_fe,
        oof_manager, mlflow_tracker
    )
    all_oof_preds.append(oof1)
    all_test_preds.append(test1)
    model_scores["XGBoost_Conservative"] = score1
    
    # Model 2: XGBoost (aggressive)
    xgb2 = XGBoostModel(params={
        'n_estimators': 800,
        'max_depth': 6,
        'learning_rate': 0.02,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'gamma': 0.01,
        'reg_alpha': 0.01,
        'reg_lambda': 0.1,
        'random_state': RANDOM_SEED + 1
    })
    oof2, test2, score2 = train_model_with_tracking(
        xgb2, "XGBoost_Aggressive", X_train_fe, y_train, X_test_fe,
        oof_manager, mlflow_tracker
    )
    all_oof_preds.append(oof2)
    all_test_preds.append(test2)
    model_scores["XGBoost_Aggressive"] = score2
    
    # Model 3: LightGBM
    lgb1 = LightGBMModel(params={
        'n_estimators': 600,
        'num_leaves': 31,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'min_child_samples': 20,
        'random_state': RANDOM_SEED
    })
    oof3, test3, score3 = train_model_with_tracking(
        lgb1, "LightGBM_Standard", X_train_fe, y_train, X_test_fe,
        oof_manager, mlflow_tracker
    )
    all_oof_preds.append(oof3)
    all_test_preds.append(test3)
    model_scores["LightGBM_Standard"] = score3
    
    # Model 4: LightGBM DART
    lgb2 = LightGBMModel(params={
        'n_estimators': 400,
        'num_leaves': 25,
        'learning_rate': 0.015,
        'boosting_type': 'dart',
        'drop_rate': 0.1,
        'skip_drop': 0.5,
        'max_drop': 50,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'random_state': RANDOM_SEED + 2
    })
    oof4, test4, score4 = train_model_with_tracking(
        lgb2, "LightGBM_DART", X_train_fe, y_train, X_test_fe,
        oof_manager, mlflow_tracker
    )
    all_oof_preds.append(oof4)
    all_test_preds.append(test4)
    model_scores["LightGBM_DART"] = score4
    
    # Model 5: CatBoost
    # Prepare data with categorical features for CatBoost
    cat_features = ['job', 'marital', 'education', 'default', 'housing', 
                   'loan', 'contact', 'month', 'poutcome']
    
    # Use original data for CatBoost (it handles categoricals internally)
    cat = CatBoostModel(params={
        'iterations': 500,
        'depth': 6,
        'learning_rate': 0.01,
        'l2_leaf_reg': 3,
        'border_count': 128,
        'random_seed': RANDOM_SEED,  # Use random_seed instead of random_state for CatBoost
        'verbose': False
    })
    
    # For CatBoost, use original categorical features
    oof5, test5, score5 = train_model_with_tracking(
        cat, "CatBoost", X_train, y_train, X_test,
        oof_manager, mlflow_tracker
    )
    all_oof_preds.append(oof5)
    all_test_preds.append(test5)
    model_scores["CatBoost"] = score5
    
    # Model 6: Random Forest
    rf = RandomForestModel(params={
        'n_estimators': 300,
        'max_depth': 12,
        'min_samples_split': 20,
        'min_samples_leaf': 10,
        'max_features': 'sqrt',
        'random_state': RANDOM_SEED
    })
    oof6, test6, score6 = train_model_with_tracking(
        rf, "RandomForest", X_train_fe, y_train, X_test_fe,
        oof_manager, mlflow_tracker
    )
    all_oof_preds.append(oof6)
    all_test_preds.append(test6)
    model_scores["RandomForest"] = score6
    
    # ==========================================
    # Create simple ensemble
    # ==========================================
    
    logger.info("\n" + "="*60)
    logger.info("CREATING SIMPLE ENSEMBLE")
    logger.info("="*60)
    
    # Simple average ensemble
    oof_ensemble = np.mean(all_oof_preds, axis=0)
    test_ensemble = np.mean(all_test_preds, axis=0)
    
    ensemble_score = roc_auc_score(y_train, oof_ensemble)
    logger.info(f"Simple Average Ensemble CV Score: {ensemble_score:.6f}")
    
    # Log ensemble score to MLflow
    if mlflow_tracker:
        mlflow_tracker.log_metrics({'ensemble_cv_auc': ensemble_score})
    
    # ==========================================
    # Save submissions
    # ==========================================
    
    logger.info("\n" + "="*60)
    logger.info("SAVING SUBMISSIONS")
    logger.info("="*60)
    
    # Save individual model submissions
    for i, (model_name, test_pred) in enumerate(zip(model_scores.keys(), all_test_preds)):
        submission = sample_sub.copy()
        submission['y'] = test_pred
        submission.to_csv(SUBMISSION_DIR / f"submission_{model_name.lower()}.csv", index=False)
        logger.info(f"Saved {model_name} submission")
    
    # Save ensemble submission
    submission = sample_sub.copy()
    submission['y'] = test_ensemble
    submission.to_csv(SUBMISSION_DIR / "submission_simple_ensemble.csv", index=False)
    logger.info("Saved simple ensemble submission")
    
    # ==========================================
    # Summary
    # ==========================================
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING SUMMARY")
    logger.info("="*60)
    
    # Sort models by score
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    
    logger.info("\nModel Performance (CV AUC):")
    for rank, (model_name, score) in enumerate(sorted_models, 1):
        logger.info(f"{rank}. {model_name:20s}: {score:.6f}")
    
    logger.info(f"\nSimple Ensemble: {ensemble_score:.6f}")
    improvement = (ensemble_score - sorted_models[0][1]) * 100
    logger.info(f"Ensemble improvement over best single model: {improvement:.3f}%")
    
    # Save summary to file
    summary_df = pd.DataFrame({
        'Model': list(model_scores.keys()) + ['Simple_Ensemble'],
        'CV_AUC': list(model_scores.values()) + [ensemble_score]
    }).sort_values('CV_AUC', ascending=False)
    
    summary_df.to_csv(OUTPUT_DIR / "model_summary.csv", index=False)
    logger.info(f"\nModel summary saved to {OUTPUT_DIR}/model_summary.csv")
    
    # End MLflow run
    if mlflow_tracker:
        mlflow_tracker.end_run()
        logger.info(f"\nMLflow tracking completed. View at {os.getenv('MLFLOW_TRACKING_URI')}")
    
    logger.info("\n" + "="*60)
    logger.info("BASELINE TRAINING COMPLETE!")
    logger.info("="*60)
    logger.info(f"\nNext step: Run 03_ensemble_hill_climb.py to optimize ensemble weights")


if __name__ == "__main__":
    main()