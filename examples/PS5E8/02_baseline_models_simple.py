"""
PS5E8 Competition - Simplified Baseline Models Training
Train multiple models with clear CV metrics display
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
    DataLoader, FeatureEngineer,
    XGBoostModel, LightGBMModel, CatBoostModel, RandomForestModel,
    OOFEnsemble, OOFManager
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
    
    # Check target distribution
    target_dist = train_df[TARGET_COL].value_counts(normalize=True)
    logger.info(f"Target distribution - Class 0: {target_dist[0]:.2%}, Class 1: {target_dist[1]:.2%}")
    
    # Separate features and target
    X_train = train_df.drop([TARGET_COL, 'id'], axis=1)
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop('id', axis=1)
    
    # Store IDs for submission
    test_ids = test_df['id']
    
    return X_train, y_train, X_test, test_ids, sample_sub


def engineer_features(X_train, y_train, X_test):
    """Apply basic feature engineering."""
    logger.info("Engineering features...")
    
    # Basic feature engineering with auto-detection
    engineer = FeatureEngineer(
        categorical_impute_strategy='constant',
        numeric_impute_strategy='median',
        categorical_encoding='target',  # Target encoding for categorical
        scaling_method='standard',
        max_cardinality=20,
        min_frequency=0.01
    )
    
    X_train_fe = engineer.fit_transform(X_train, y_train)
    X_test_fe = engineer.transform(X_test)
    
    logger.info(f"Features after engineering: {X_train_fe.shape[1]}")
    
    return X_train_fe, X_test_fe


def train_single_model(model, model_name, X_train, y_train, X_test, oof_manager):
    """Train a single model and save OOF predictions."""
    
    logger.info(f"\nTraining {model_name}...")
    logger.info("="*50)
    
    # Generate OOF predictions
    ensemble = OOFEnsemble(task_type='classification')
    
    try:
        # Get OOF predictions with fold metrics
        oof_preds = ensemble.get_oof_predictions(
            models=[model],
            X=X_train,
            y=y_train,
            n_folds=N_FOLDS,
            stratified=True,
            verbose=True
        )
        
        # Calculate overall CV score
        cv_score = roc_auc_score(y_train, oof_preds.iloc[:, 0])
        logger.info(f"\n{model_name} Final CV Score: {cv_score:.6f}")
        
        # Train on full data for test predictions
        logger.info(f"Training {model_name} on full dataset...")
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
            tags={'competition': 'PS5E8'}
        )
        
        return oof_preds.iloc[:, 0], test_preds, cv_score
        
    except Exception as e:
        logger.error(f"Error training {model_name}: {str(e)}")
        return None, None, None


def main():
    """Main training pipeline."""
    
    logger.info("="*60)
    logger.info("PS5E8 COMPETITION - SIMPLIFIED BASELINE TRAINING")
    logger.info("="*60)
    
    # Load data
    X_train, y_train, X_test, test_ids, sample_sub = load_competition_data()
    
    # Feature engineering
    X_train_fe, X_test_fe = engineer_features(X_train, y_train, X_test)
    
    # Initialize OOF manager
    oof_manager = OOFManager(output_dir=str(OOF_DIR))
    
    # Store results
    all_oof_preds = []
    all_test_preds = []
    model_scores = {}
    
    # ==========================================
    # Define and train models
    # ==========================================
    
    models_to_train = [
        (XGBoostModel(params={
            'n_estimators': 300,
            'max_depth': 5,
            'learning_rate': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': RANDOM_SEED
        }), "XGBoost_v1"),
        
        (LightGBMModel(params={
            'n_estimators': 300,
            'num_leaves': 31,
            'learning_rate': 0.01,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'random_state': RANDOM_SEED
        }), "LightGBM_v1"),
        
        (CatBoostModel(params={
            'iterations': 300,
            'depth': 6,
            'learning_rate': 0.01,
            'random_seed': RANDOM_SEED,  # CatBoost uses random_seed
            'verbose': False
        }), "CatBoost_v1"),
        
        (RandomForestModel(params={
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'random_state': RANDOM_SEED
        }), "RandomForest_v1"),
    ]
    
    # Train each model
    for model, model_name in models_to_train:
        # Use appropriate data for each model
        if "CatBoost" in model_name:
            # CatBoost works better with original categorical features
            oof, test_pred, score = train_single_model(
                model, model_name, X_train, y_train, X_test, oof_manager
            )
        else:
            # Other models use engineered features
            oof, test_pred, score = train_single_model(
                model, model_name, X_train_fe, y_train, X_test_fe, oof_manager
            )
        
        if oof is not None:
            all_oof_preds.append(oof)
            all_test_preds.append(test_pred)
            model_scores[model_name] = score
    
    # ==========================================
    # Create simple ensemble
    # ==========================================
    
    if len(all_oof_preds) > 0:
        logger.info("\n" + "="*60)
        logger.info("CREATING SIMPLE ENSEMBLE")
        logger.info("="*60)
        
        # Simple average ensemble
        oof_ensemble = np.mean(all_oof_preds, axis=0)
        test_ensemble = np.mean(all_test_preds, axis=0)
        
        ensemble_score = roc_auc_score(y_train, oof_ensemble)
        logger.info(f"Simple Average Ensemble CV Score: {ensemble_score:.6f}")
        
        # Save ensemble submission
        submission = sample_sub.copy()
        submission['y'] = test_ensemble
        submission.to_csv(SUBMISSION_DIR / "submission_simple_ensemble.csv", index=False)
        logger.info(f"Saved ensemble submission")
    
    # ==========================================
    # Final Summary
    # ==========================================
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING SUMMARY")
    logger.info("="*60)
    
    if model_scores:
        # Sort models by score
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        
        logger.info("\nModel Performance (CV ROC-AUC):")
        logger.info("-" * 40)
        for rank, (model_name, score) in enumerate(sorted_models, 1):
            logger.info(f"{rank}. {model_name:20s}: {score:.6f}")
        
        if len(all_oof_preds) > 0:
            logger.info("-" * 40)
            logger.info(f"Simple Ensemble:      {ensemble_score:.6f}")
            
            best_single = sorted_models[0][1]
            improvement = (ensemble_score - best_single) * 100
            logger.info(f"\nEnsemble improvement: {improvement:+.3f}%")
    
    # Save summary
    if model_scores:
        summary_df = pd.DataFrame({
            'Model': list(model_scores.keys()),
            'CV_AUC': list(model_scores.values())
        }).sort_values('CV_AUC', ascending=False)
        
        if len(all_oof_preds) > 0:
            ensemble_row = pd.DataFrame({
                'Model': ['Simple_Ensemble'],
                'CV_AUC': [ensemble_score]
            })
            summary_df = pd.concat([summary_df, ensemble_row], ignore_index=True)
        
        summary_df.to_csv(OUTPUT_DIR / "model_summary.csv", index=False)
        logger.info(f"\nModel summary saved to {OUTPUT_DIR}/model_summary.csv")
    
    logger.info("\n" + "="*60)
    logger.info("BASELINE TRAINING COMPLETE!")
    logger.info("="*60)
    logger.info(f"\nNext step: Run 03_ensemble_hill_climb.py to optimize ensemble weights")


if __name__ == "__main__":
    main()