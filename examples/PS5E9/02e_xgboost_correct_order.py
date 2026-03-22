"""
PS5E9 Competition - XGBoost with CORRECT Feature Engineering Order
Fixes the critical issue: Feature engineering must be done on competition data ONLY,
then original data is added AFTER feature engineering (matching the notebook exactly)
Target CV: ~26.46
"""

import os
import sys
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from loguru import logger
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import tabml
sys.path.append(str(Path(__file__).parent.parent.parent))

from tabml import XGBoostModel, OOFManager
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup paths
DATA_DIR = Path("../../data/raw/PS5E9")
OUTPUT_DIR = Path("output")
OOF_DIR = OUTPUT_DIR / "oof_predictions"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"
OOF_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

# Competition settings
TARGET_COL = 'BeatsPerMinute'
ID_COL = 'id'
RANDOM_SEED = 42
N_FOLDS = 5

# MLflow settings
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = "PS5E9"


class ModelPerformanceTracker:
    """Track model performance across experiments in a CSV file."""
    
    def __init__(self, tracker_file="model_performance_tracker.csv"):
        self.tracker_file = Path(tracker_file)
        self.columns = [
            'timestamp', 'script_name', 'dataset_file', 'script_hash', 
            'data_hash', 'model_name', 'n_folds', 'cv_rmse', 'cv_mae', 'cv_r2',
            'public_lb', 'private_lb', 'notes'
        ]
        
        if self.tracker_file.exists():
            self.df = pd.read_csv(self.tracker_file)
            logger.info(f"Loaded existing tracker with {len(self.df)} records")
        else:
            self.df = pd.DataFrame(columns=self.columns)
            logger.info("Created new performance tracker")
    
    def get_file_hash(self, filepath):
        """Calculate MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()[:8]
    
    def add_record(self, script_name, dataset_file, model_name, n_folds, 
                   cv_rmse, cv_mae=None, cv_r2=None, notes=""):
        """Add a new performance record."""
        script_hash = self.get_file_hash(script_name)
        data_hash = self.get_file_hash(dataset_file)
        
        new_record = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'script_name': Path(script_name).name,
            'dataset_file': Path(dataset_file).name,
            'script_hash': script_hash,
            'data_hash': data_hash,
            'model_name': model_name,
            'n_folds': n_folds,
            'cv_rmse': round(cv_rmse, 4),
            'cv_mae': round(cv_mae, 4) if cv_mae else None,
            'cv_r2': round(cv_r2, 4) if cv_r2 else None,
            'public_lb': 'NA',
            'private_lb': 'NA',
            'notes': notes
        }
        
        self.df = pd.concat([self.df, pd.DataFrame([new_record])], ignore_index=True)
        self.df.to_csv(self.tracker_file, index=False)
        logger.info(f"Added performance record: CV RMSE={cv_rmse:.4f}")
        
        return new_record


def load_original_data():
    """Load the original BPM dataset if available."""
    original_path = DATA_DIR / "Train_original.csv"
    
    if original_path.exists():
        logger.info(f"✓ Found original data at {original_path}")
        original_df = pd.read_csv(original_path)
        
        # Select only the needed columns
        cols_needed = ['RhythmScore', 'AudioLoudness', 'VocalContent', 
                      'AcousticQuality', 'InstrumentalScore', 
                      'LivePerformanceLikelihood', 'MoodScore', 
                      'TrackDurationMs', 'Energy', 'BeatsPerMinute']
        
        missing_cols = [col for col in cols_needed if col not in original_df.columns]
        if missing_cols:
            logger.warning(f"Missing columns in original data: {missing_cols}")
            return None
            
        original_df = original_df[cols_needed]
        logger.info(f"Loaded {len(original_df)} samples from original dataset")
        return original_df
    else:
        logger.warning(f"Original dataset not found at {original_path}")
        logger.warning("To achieve best performance, place Train_original.csv in: {DATA_DIR}")
        return None


def create_features_correct_order(train_df, test_df, target_col):
    """
    CRITICAL: This function replicates the EXACT order from the notebook:
    1. Feature engineering on competition data ONLY (train + test)
    2. THEN add original data to the engineered features
    
    This order is crucial for achieving CV ~26.46
    """
    
    # STEP 1: Feature engineering on COMPETITION DATA ONLY
    logger.info("Step 1: Feature engineering on competition data only...")
    
    # Combine competition train and test for feature engineering
    train_features = train_df.drop(columns=[target_col])
    df_combined = pd.concat([train_features, test_df], axis=0, ignore_index=True)
    
    logger.info(f"  Combined competition data shape: {df_combined.shape}")
    
    # Get the base feature columns
    base_cols = ['RhythmScore', 'AudioLoudness', 'VocalContent', 
                 'AcousticQuality', 'InstrumentalScore', 
                 'LivePerformanceLikelihood', 'MoodScore', 
                 'TrackDurationMs', 'Energy']
    
    # Adjust AudioLoudness and apply log1p
    logger.info("  Applying log transformation...")
    min_loudness = df_combined["AudioLoudness"].min()
    df_combined["AudioLoudness"] = df_combined["AudioLoudness"] - min_loudness
    
    # Apply log1p to all base columns
    for col in base_cols:
        df_combined[col] = np.log1p(df_combined[col])
    
    # Create 2-way interactions
    logger.info("  Creating 2-way interactions...")
    for col1, col2 in combinations(base_cols, 2):
        df_combined[f"{col1}_p_{col2}"] = df_combined[col1] + df_combined[col2]
        df_combined[f"{col1}_m_{col2}"] = df_combined[col1] * df_combined[col2]
        df_combined[f"{col1}_d_{col2}"] = df_combined[col1] / (df_combined[col2] + 1e-6)
    
    # Create 3-way interactions
    logger.info("  Creating 3-way interactions...")
    for col1, col2, col3 in combinations(base_cols, 3):
        df_combined[f"{col1}_p_{col2}_p_{col3}"] = df_combined[col1] + df_combined[col2] + df_combined[col3]
        df_combined[f"{col1}_m_{col2}_m_{col3}"] = df_combined[col1] * df_combined[col2] * df_combined[col3]
    
    logger.info(f"  Features after engineering: {df_combined.shape[1]}")
    
    # Split back into train and test
    X_train_comp = df_combined.iloc[:len(train_df)].copy()
    X_test = df_combined.iloc[len(train_df):].copy()
    y_train_comp = train_df[target_col].copy()
    
    # STEP 2: Now add ORIGINAL DATA to the already-engineered features
    logger.info("\nStep 2: Adding original data to engineered features...")
    
    original_df = load_original_data()
    if original_df is not None:
        logger.info("  Processing original data with same transformations...")
        
        # Apply the SAME transformations to original data
        # IMPORTANT: Use the same min_loudness from competition data!
        original_df["AudioLoudness"] = original_df["AudioLoudness"] - min_loudness
        
        # Apply log1p
        for col in base_cols:
            original_df[col] = np.log1p(original_df[col])
        
        # Create same 2-way interactions
        for col1, col2 in combinations(base_cols, 2):
            original_df[f"{col1}_p_{col2}"] = original_df[col1] + original_df[col2]
            original_df[f"{col1}_m_{col2}"] = original_df[col1] * original_df[col2]
            original_df[f"{col1}_d_{col2}"] = original_df[col1] / (original_df[col2] + 1e-6)
        
        # Create same 3-way interactions
        for col1, col2, col3 in combinations(base_cols, 3):
            original_df[f"{col1}_p_{col2}_p_{col3}"] = original_df[col1] + original_df[col2] + original_df[col3]
            original_df[f"{col1}_m_{col2}_m_{col3}"] = original_df[col1] * original_df[col2] * original_df[col3]
        
        # Extract target from original data
        y_train_orig = original_df[target_col].copy()
        X_train_orig = original_df.drop(columns=[target_col])
        
        # Combine competition and original data
        X_train = pd.concat([X_train_comp, X_train_orig], axis=0, ignore_index=True)
        y_train = pd.concat([y_train_comp, y_train_orig], axis=0, ignore_index=True)
        
        logger.info(f"  Combined training data: {len(X_train)} samples")
        logger.info(f"    - Competition: {len(X_train_comp)} samples")
        logger.info(f"    - Original: {len(X_train_orig)} samples")
    else:
        logger.warning("  No original data found - using competition data only")
        X_train = X_train_comp
        y_train = y_train_comp
    
    # Convert to float32 for memory efficiency
    for col in X_train.columns:
        if X_train[col].dtype == 'float64':
            X_train[col] = X_train[col].astype('float32')
            X_test[col] = X_test[col].astype('float32')
    
    logger.info(f"\nFinal shapes: X_train={X_train.shape}, X_test={X_test.shape}")
    
    return X_train, X_test, y_train


def train_xgboost_with_oof(X_train, y_train, X_test, model_params=None):
    """Train XGBoost with out-of-fold predictions."""
    
    # XGBoost parameters matching the notebook with GPU
    if model_params is None:
        model_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 10000,
            'learning_rate': 0.02,
            'max_depth': 8,
            'colsample_bytree': 0.50,
            'colsample_bynode': 0.35,
            'subsample': 0.85,
            'min_child_weight': 1,
            'random_state': RANDOM_SEED,
            'n_jobs': -1,
            'tree_method': 'gpu_hist',
            'predictor': 'gpu_predictor',
            'gpu_id': 0,
            'verbosity': 0
        }
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    oof_predictions = np.zeros(len(X_train))
    test_predictions = np.zeros(len(X_test))
    feature_importance = pd.DataFrame()
    cv_scores = []
    
    logger.info(f"Training XGBoost with {N_FOLDS}-fold CV (GPU)")
    logger.info(f"Samples: {len(X_train)}, Features: {X_train.shape[1]}")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        logger.info(f"\nFold {fold}/{N_FOLDS}")
        
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        model = XGBoostModel(params=model_params)
        model.fit(
            X_fold_train, 
            y_fold_train,
            X_val=X_fold_val,
            y_val=y_fold_val,
            early_stopping_rounds=250
        )
        
        oof_predictions[val_idx] = model.predict(X_fold_val)
        test_predictions += model.predict(X_test) / N_FOLDS
        
        fold_rmse = np.sqrt(mean_squared_error(y_fold_val, oof_predictions[val_idx]))
        cv_scores.append(fold_rmse)
        logger.info(f"Fold {fold} RMSE: {fold_rmse:.4f}")
        
        if hasattr(model.model, 'feature_importances_'):
            fold_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.model.feature_importances_,
                'fold': fold
            })
            feature_importance = pd.concat([feature_importance, fold_importance], ignore_index=True)
    
    cv_rmse = np.sqrt(mean_squared_error(y_train, oof_predictions))
    cv_mae = mean_absolute_error(y_train, oof_predictions)
    cv_r2 = r2_score(y_train, oof_predictions)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Overall CV RMSE: {cv_rmse:.4f}")
    logger.info(f"Overall CV MAE: {cv_mae:.4f}")
    logger.info(f"Overall CV R²: {cv_r2:.4f}")
    logger.info(f"CV mean ± std: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    logger.info(f"{'='*50}")
    
    return oof_predictions, test_predictions, feature_importance, cv_rmse, cv_mae, cv_r2


def main():
    """Main execution function."""
    logger.info("="*60)
    logger.info("PS5E9 - XGBOOST WITH CORRECT FEATURE ORDER")
    logger.info("="*60)
    
    tracker = ModelPerformanceTracker("output/model_performance_tracker.csv")
    
    logger.info(f"\nInitializing MLflow tracking at {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # Load competition data
    logger.info("\nLoading competition data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    submission = pd.read_csv(DATA_DIR / "sample_submission.csv")
    
    # Remove IDs
    train_df = train_df.drop(columns=[ID_COL])
    test_df = test_df.drop(columns=[ID_COL])
    
    logger.info(f"Competition train: {train_df.shape}")
    logger.info(f"Competition test: {test_df.shape}")
    
    # Create features with CORRECT order
    logger.info("\n" + "="*60)
    logger.info("CREATING FEATURES WITH CORRECT ORDER")
    logger.info("="*60)
    
    X_train, X_test, y_train = create_features_correct_order(
        train_df, 
        test_df, 
        TARGET_COL
    )
    
    # Competition samples for OOF scoring
    competition_samples = min(524164, len(X_train))
    
    with mlflow.start_run(run_name="xgboost_correct_order"):
        mlflow.log_param("n_folds", N_FOLDS)
        mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_competition_samples", competition_samples)
        mlflow.log_param("model_type", "XGBoost_Correct_Order")
        mlflow.log_param("gpu_enabled", True)
        
        logger.info("\nTraining XGBoost...")
        oof_preds, test_preds, feature_importance, cv_rmse, cv_mae, cv_r2 = train_xgboost_with_oof(
            X_train, 
            y_train, 
            X_test
        )
        
        mlflow.log_metric("cv_rmse", cv_rmse)
        mlflow.log_metric("cv_mae", cv_mae)
        mlflow.log_metric("cv_r2", cv_r2)
        
        if not feature_importance.empty:
            feature_importance_mean = feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
            logger.info("\nTop 15 Features:")
            for i, (feat, imp) in enumerate(feature_importance_mean.head(15).items(), 1):
                logger.info(f"  {i:2d}. {feat}: {imp:.4f}")
        
        # Save OOF predictions
        logger.info("\nSaving OOF predictions...")
        oof_manager = OOFManager(output_dir=str(OOF_DIR))
        
        oof_df = pd.DataFrame({
            'predictions': oof_preds[:competition_samples],
            'target': y_train[:competition_samples]
        })
        
        oof_manager.save_oof(
            predictions=oof_df,
            test_predictions=test_preds,
            model_name="xgboost_correct_order",
            cv_score=cv_rmse,
            model_params={
                'model_type': 'XGBoost_Correct_Order',
                'n_folds': N_FOLDS,
                'n_features': X_train.shape[1],
                'n_train_samples': len(X_train)
            }
        )
        
        # Create submission
        logger.info("\nCreating submission...")
        submission[TARGET_COL] = test_preds
        submission_file = SUBMISSION_DIR / f"submission_xgboost_correct_cv{cv_rmse:.4f}.csv"
        submission.to_csv(submission_file, index=False)
        logger.info(f"Submission saved to {submission_file}")
        
        mlflow.log_artifact(str(submission_file))
    
    tracker.add_record(
        script_name=__file__,
        dataset_file=DATA_DIR / "train.csv",
        model_name="xgboost_correct_order",
        n_folds=N_FOLDS,
        cv_rmse=cv_rmse,
        cv_mae=cv_mae,
        cv_r2=cv_r2,
        notes="CORRECT feature engineering order (competition first, then original)"
    )
    
    logger.info("\n" + "="*60)
    logger.info("✅ Training Complete!")
    logger.info("="*60)
    logger.info(f"   CV RMSE: {cv_rmse:.4f}")
    logger.info(f"   Target: ~26.46 (notebook benchmark)")
    logger.info(f"   This should match the notebook performance!")


if __name__ == "__main__":
    main()