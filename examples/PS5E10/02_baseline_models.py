"""
PS5E10 Competition - Baseline Models Training
Train multiple regression models with feature engineering and save OOF predictions

Models trained:
- XGBoost (3 variations)
- LightGBM (2 variations)
- CatBoost
- Random Forest
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import tabml
sys.path.append(str(Path(__file__).parent.parent.parent))

from tabml import (
    XGBoostModel, LightGBMModel, CatBoostModel,
    RandomForestModel, OOFEnsemble, OOFManager
)
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from loguru import logger
import gc

# Setup paths
DATA_DIR = Path("../../data/raw/PS5E10")
OUTPUT_DIR = Path("output")
OOF_DIR = OUTPUT_DIR / "oof_predictions"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"
OOF_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

# Competition settings
TARGET_COL = 'accident_risk'
ID_COL = 'id'
RANDOM_SEED = 42
N_FOLDS = 5


def create_features(df, is_train=True):
    """
    Create engineered features for accident risk prediction.

    Args:
        df: Input dataframe
        is_train: Whether this is training data (for target encoding)

    Returns:
        DataFrame with engineered features
    """
    df = df.copy()

    # Categorical features
    categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day']

    # Boolean features (convert to int)
    boolean_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
    for col in boolean_cols:
        df[f'{col}_int'] = df[col].astype(int)

    # 1. Risk interaction features
    df['weather_lighting'] = df['weather'] + '_' + df['lighting']
    df['roadtype_weather'] = df['road_type'] + '_' + df['weather']
    df['time_lighting'] = df['time_of_day'] + '_' + df['lighting']

    # 2. Composite risk scores
    df['visibility_risk'] = ((df['weather'].isin(['foggy', 'rainy'])).astype(int) +
                             (df['lighting'].isin(['dim', 'night'])).astype(int))

    df['geometry_risk'] = df['curvature'] * df['speed_limit']
    df['curvature_squared'] = df['curvature'] ** 2
    df['speed_squared'] = df['speed_limit'] ** 2

    # 3. Historical risk features
    df['has_previous_accidents'] = (df['num_reported_accidents'] > 0).astype(int)
    df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 0.1)

    # 4. Dangerous conditions (boolean combinations)
    df['night_rainy'] = ((df['lighting'] == 'night') &
                         (df['weather'].isin(['rainy', 'foggy']))).astype(int)

    df['high_curve_high_speed'] = ((df['curvature'] > 0.5) &
                                   (df['speed_limit'] > 50)).astype(int)

    df['rural_bad_weather'] = ((df['road_type'] == 'rural') &
                               (df['weather'].isin(['rainy', 'foggy']))).astype(int)

    # 5. Road complexity score
    df['road_complexity'] = (df['num_lanes'] * df['curvature'] *
                            (df['speed_limit'] / 100))

    # 6. Safety features score
    df['safety_score'] = (df['road_signs_present_int'] +
                         df['public_road_int'] -
                         df['visibility_risk'])

    # 7. Temporal features
    df['is_rush_hour'] = ((df['time_of_day'].isin(['morning', 'evening']))).astype(int)
    df['weekend_holiday'] = df['holiday_int']

    # 8. Feature ratios
    df['speed_per_lane'] = df['speed_limit'] / (df['num_lanes'] + 0.1)
    df['accidents_per_curve'] = df['num_reported_accidents'] / (df['curvature'] + 0.01)

    # 9. Polynomial features for key variables
    df['curvature_speed_interaction'] = df['curvature'] * df['speed_limit']
    df['lanes_speed_interaction'] = df['num_lanes'] * df['speed_limit']

    return df


def prepare_data():
    """Load and prepare data with feature engineering."""
    logger.info("="*60)
    logger.info("LOADING AND PREPARING DATA")
    logger.info("="*60)

    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    logger.info(f"Train shape: {train_df.shape}")
    logger.info(f"Test shape: {test_df.shape}")

    # Separate target and IDs
    train_ids = train_df[ID_COL]
    test_ids = test_df[ID_COL]
    y_train = train_df[TARGET_COL]

    # Create features
    logger.info("\nEngineering features...")
    train_features = create_features(train_df.drop([ID_COL, TARGET_COL], axis=1), is_train=True)
    test_features = create_features(test_df.drop([ID_COL], axis=1), is_train=False)

    logger.info(f"Train features shape: {train_features.shape}")
    logger.info(f"Test features shape: {test_features.shape}")

    # Encode categorical features
    categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day',
                       'weather_lighting', 'roadtype_weather', 'time_lighting']

    logger.info("\nEncoding categorical features...")
    for col in categorical_cols:
        if col in train_features.columns:
            le = LabelEncoder()
            train_features[col] = le.fit_transform(train_features[col].astype(str))
            test_features[col] = le.transform(test_features[col].astype(str))

    # Convert boolean columns to int
    boolean_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
    for col in boolean_cols:
        if col in train_features.columns:
            train_features[col] = train_features[col].astype(int)
            test_features[col] = test_features[col].astype(int)

    logger.info(f"\n✓ Data preparation complete")
    logger.info(f"  Features: {train_features.shape[1]}")
    logger.info(f"  Training samples: {len(train_features):,}")
    logger.info(f"  Target mean: {y_train.mean():.4f}")
    logger.info(f"  Target std: {y_train.std():.4f}")

    return train_features, test_features, y_train, train_ids, test_ids


def train_model_with_oof(model, model_name, X_train, y_train, X_test, n_folds=5):
    """
    Train model with K-fold cross-validation and generate OOF predictions.

    Args:
        model: Model instance
        model_name: Name for saving
        X_train: Training features
        y_train: Training target
        X_test: Test features
        n_folds: Number of CV folds

    Returns:
        oof_predictions, test_predictions, cv_scores
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING {model_name}")
    logger.info(f"{'='*60}")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    oof_predictions = np.zeros(len(X_train))
    test_predictions = np.zeros(len(X_test))
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        logger.info(f"\nFold {fold}/{n_folds}")

        # Split data
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]

        # Train model
        model.fit(X_fold_train, y_fold_train, X_fold_val, y_fold_val)

        # Predict on validation fold
        val_preds = model.predict(X_fold_val)
        val_preds = np.clip(val_preds, 0, 1)  # Clip to valid range
        oof_predictions[val_idx] = val_preds

        # Predict on test
        test_preds = model.predict(X_test)
        test_preds = np.clip(test_preds, 0, 1)
        test_predictions += test_preds / n_folds

        # Calculate fold score
        fold_rmse = np.sqrt(mean_squared_error(y_fold_val, val_preds))
        fold_mae = mean_absolute_error(y_fold_val, val_preds)
        fold_r2 = r2_score(y_fold_val, val_preds)

        cv_scores.append(fold_rmse)

        logger.info(f"  RMSE: {fold_rmse:.6f}")
        logger.info(f"  MAE:  {fold_mae:.6f}")
        logger.info(f"  R²:   {fold_r2:.6f}")

        # Clean up
        gc.collect()

    # Overall OOF score
    oof_rmse = np.sqrt(mean_squared_error(y_train, oof_predictions))
    oof_mae = mean_absolute_error(y_train, oof_predictions)
    oof_r2 = r2_score(y_train, oof_predictions)

    logger.info(f"\n{'='*60}")
    logger.info(f"{model_name} - OVERALL OOF SCORES")
    logger.info(f"{'='*60}")
    logger.info(f"RMSE: {oof_rmse:.6f} (±{np.std(cv_scores):.6f})")
    logger.info(f"MAE:  {oof_mae:.6f}")
    logger.info(f"R²:   {oof_r2:.6f}")
    logger.info(f"CV Scores: {cv_scores}")

    return oof_predictions, test_predictions, cv_scores, oof_rmse


def main():
    """Main training pipeline."""
    logger.info("="*60)
    logger.info("PS5E10 - BASELINE MODELS TRAINING")
    logger.info("="*60)

    # Prepare data
    X_train, X_test, y_train, train_ids, test_ids = prepare_data()

    # Initialize OOF Manager
    oof_manager = OOFManager(str(OOF_DIR))

    # Store all test predictions
    all_test_predictions = {}

    # ========================================================================
    # MODEL 1: XGBoost - Configuration 1 (Conservative)
    # ========================================================================
    xgb_params_1 = {
        'n_estimators': 1000,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': RANDOM_SEED
    }

    xgb_model_1 = XGBoostModel(params=xgb_params_1)
    oof_xgb1, test_xgb1, cv_xgb1, score_xgb1 = train_model_with_oof(
        xgb_model_1, "XGBoost_Conservative", X_train, y_train, X_test, N_FOLDS
    )

    # Save OOF
    oof_manager.save_oof(
        predictions=oof_xgb1,
        model_name="xgboost_conservative",
        model_params=xgb_params_1,
        cv_score=score_xgb1,
        cv_scores_per_fold=cv_xgb1,
        test_predictions=test_xgb1
    )
    all_test_predictions['xgboost_conservative'] = test_xgb1

    # ========================================================================
    # MODEL 2: XGBoost - Configuration 2 (Aggressive)
    # ========================================================================
    xgb_params_2 = {
        'n_estimators': 1500,
        'max_depth': 8,
        'learning_rate': 0.03,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 1,
        'gamma': 0.05,
        'reg_alpha': 0.05,
        'reg_lambda': 0.5,
        'random_state': RANDOM_SEED + 1
    }

    xgb_model_2 = XGBoostModel(params=xgb_params_2)
    oof_xgb2, test_xgb2, cv_xgb2, score_xgb2 = train_model_with_oof(
        xgb_model_2, "XGBoost_Aggressive", X_train, y_train, X_test, N_FOLDS
    )

    oof_manager.save_oof(
        predictions=oof_xgb2,
        model_name="xgboost_aggressive",
        model_params=xgb_params_2,
        cv_score=score_xgb2,
        cv_scores_per_fold=cv_xgb2,
        test_predictions=test_xgb2
    )
    all_test_predictions['xgboost_aggressive'] = test_xgb2

    # ========================================================================
    # MODEL 3: LightGBM - Configuration 1
    # ========================================================================
    lgb_params_1 = {
        'n_estimators': 1000,
        'max_depth': 7,
        'learning_rate': 0.05,
        'num_leaves': 63,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': RANDOM_SEED
    }

    lgb_model_1 = LightGBMModel(params=lgb_params_1)
    oof_lgb1, test_lgb1, cv_lgb1, score_lgb1 = train_model_with_oof(
        lgb_model_1, "LightGBM_Standard", X_train, y_train, X_test, N_FOLDS
    )

    oof_manager.save_oof(
        predictions=oof_lgb1,
        model_name="lightgbm_standard",
        model_params=lgb_params_1,
        cv_score=score_lgb1,
        cv_scores_per_fold=cv_lgb1,
        test_predictions=test_lgb1
    )
    all_test_predictions['lightgbm_standard'] = test_lgb1

    # ========================================================================
    # MODEL 4: LightGBM - Configuration 2 (DART)
    # ========================================================================
    lgb_params_2 = {
        'n_estimators': 800,
        'max_depth': 6,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'boosting_type': 'dart',
        'random_state': RANDOM_SEED + 2
    }

    lgb_model_2 = LightGBMModel(params=lgb_params_2)
    oof_lgb2, test_lgb2, cv_lgb2, score_lgb2 = train_model_with_oof(
        lgb_model_2, "LightGBM_DART", X_train, y_train, X_test, N_FOLDS
    )

    oof_manager.save_oof(
        predictions=oof_lgb2,
        model_name="lightgbm_dart",
        model_params=lgb_params_2,
        cv_score=score_lgb2,
        cv_scores_per_fold=cv_lgb2,
        test_predictions=test_lgb2
    )
    all_test_predictions['lightgbm_dart'] = test_lgb2

    # ========================================================================
    # MODEL 5: CatBoost
    # ========================================================================
    cat_params = {
        'iterations': 1000,
        'depth': 6,
        'learning_rate': 0.05,
        'l2_leaf_reg': 3,
        'random_state': RANDOM_SEED,
        'verbose': False
    }

    cat_model = CatBoostModel(params=cat_params)
    oof_cat, test_cat, cv_cat, score_cat = train_model_with_oof(
        cat_model, "CatBoost", X_train, y_train, X_test, N_FOLDS
    )

    oof_manager.save_oof(
        predictions=oof_cat,
        model_name="catboost",
        model_params=cat_params,
        cv_score=score_cat,
        cv_scores_per_fold=cv_cat,
        test_predictions=test_cat
    )
    all_test_predictions['catboost'] = test_cat

    # ========================================================================
    # MODEL 6: Random Forest
    # ========================================================================
    rf_params = {
        'n_estimators': 300,
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_features': 'sqrt',
        'random_state': RANDOM_SEED,
        'n_jobs': -1
    }

    rf_model = RandomForestModel(params=rf_params)
    oof_rf, test_rf, cv_rf, score_rf = train_model_with_oof(
        rf_model, "RandomForest", X_train, y_train, X_test, N_FOLDS
    )

    oof_manager.save_oof(
        predictions=oof_rf,
        model_name="random_forest",
        model_params=rf_params,
        cv_score=score_rf,
        cv_scores_per_fold=cv_rf,
        test_predictions=test_rf
    )
    all_test_predictions['random_forest'] = test_rf

    # ========================================================================
    # SUMMARY
    # ========================================================================
    logger.info("\n" + "="*60)
    logger.info("TRAINING SUMMARY")
    logger.info("="*60)

    summary = oof_manager.list_oofs(sort_by='cv_score', ascending=True)
    logger.info(f"\n{summary}")

    # Simple average ensemble
    logger.info("\n" + "="*60)
    logger.info("SIMPLE AVERAGE ENSEMBLE")
    logger.info("="*60)

    avg_test_preds = np.mean(list(all_test_predictions.values()), axis=0)

    # Create submission
    submission = pd.DataFrame({
        ID_COL: test_ids,
        TARGET_COL: avg_test_preds
    })

    submission_path = SUBMISSION_DIR / f"submission_avg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    submission.to_csv(submission_path, index=False)
    logger.info(f"\n✓ Average ensemble submission saved to: {submission_path}")

    # Also save individual model submissions
    for model_name, preds in all_test_predictions.items():
        individual_submission = pd.DataFrame({
            ID_COL: test_ids,
            TARGET_COL: preds
        })
        individual_path = SUBMISSION_DIR / f"submission_{model_name}.csv"
        individual_submission.to_csv(individual_path, index=False)

    logger.info(f"\n✓ All {len(all_test_predictions)} individual submissions saved")
    logger.info(f"\n{'='*60}")
    logger.info("BASELINE MODELS TRAINING COMPLETE!")
    logger.info(f"{'='*60}")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Review OOF predictions in: {OOF_DIR}")
    logger.info(f"  2. Run ensemble optimization: python 03_ensemble.py")
    logger.info(f"  3. Submit best model to Kaggle")


if __name__ == "__main__":
    main()
