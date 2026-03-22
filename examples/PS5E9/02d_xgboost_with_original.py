"""
PS5E9 Competition - XGBoost with Original Data
Uses the original BPM dataset (14,633 samples) to improve performance
Target CV: ~26.46 (matching the notebook with original data)
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
    # Path to the original data in PS5E9 folder
    original_path = DATA_DIR / "Train_original.csv"
    
    if original_path.exists():
        logger.info(f"✓ Found original data at {original_path}")
        original_df = pd.read_csv(original_path)
        
        # Select only the needed columns (matching competition data structure)
        cols_needed = ['RhythmScore', 'AudioLoudness', 'VocalContent', 
                      'AcousticQuality', 'InstrumentalScore', 
                      'LivePerformanceLikelihood', 'MoodScore', 
                      'TrackDurationMs', 'Energy', 'BeatsPerMinute']
        
        # Check if columns exist
        missing_cols = [col for col in cols_needed if col not in original_df.columns]
        if missing_cols:
            logger.warning(f"Missing columns in original data: {missing_cols}")
            logger.warning("Proceeding without original data")
            return None
            
        original_df = original_df[cols_needed]
        logger.info(f"Loaded {len(original_df)} samples from original dataset")
        return original_df
    else:
        logger.warning(f"Original dataset not found at {original_path}")
        logger.warning("Proceeding with competition data only")
        logger.warning("To use original data, place Train_original.csv in: {DATA_DIR}")
        return None


def create_features_notebook_style(train_df, test_df, target_col, use_original=True):
    """
    Exactly replicate the notebook's feature engineering approach.
    This matches the winning notebook that achieves CV ~26.46.
    """
    
    # Load and merge original data if requested
    if use_original:
        original_df = load_original_data()
        if original_df is not None:
            logger.info(f"Merging {len(original_df)} original samples with {len(train_df)} competition samples")
            train_df = pd.concat([train_df, original_df], ignore_index=True)
            logger.info(f"Combined training data: {len(train_df)} samples")
    
    # Combine train and test for feature engineering (excluding target)
    train_features = train_df.drop(columns=[target_col])
    df_combined = pd.concat([train_features, test_df], axis=0, ignore_index=True)
    
    # Get the base feature columns (first 9 columns)
    base_cols = ['RhythmScore', 'AudioLoudness', 'VocalContent', 
                 'AcousticQuality', 'InstrumentalScore', 
                 'LivePerformanceLikelihood', 'MoodScore', 
                 'TrackDurationMs', 'Energy']
    
    # Step 1: Adjust AudioLoudness and apply log1p to all features
    logger.info("Applying log transformation to features...")
    df_combined["AudioLoudness"] = df_combined["AudioLoudness"] - df_combined["AudioLoudness"].min()
    
    # Apply log1p to all base columns (this transforms them in place)
    for col in base_cols:
        df_combined[col] = np.log1p(df_combined[col])
    
    # Step 2: Create 2-way interactions (36 combinations × 3 operations = 108 features)
    logger.info("Creating 2-way feature interactions...")
    initial_shape = df_combined.shape[1]
    
    for col1, col2 in combinations(base_cols, 2):
        # Addition
        df_combined[f"{col1}_p_{col2}"] = df_combined[col1] + df_combined[col2]
        # Multiplication  
        df_combined[f"{col1}_m_{col2}"] = df_combined[col1] * df_combined[col2]
        # Division
        df_combined[f"{col1}_d_{col2}"] = df_combined[col1] / (df_combined[col2] + 1e-6)
    
    logger.info(f"After 2-way interactions: {df_combined.shape[1]} features (was {initial_shape})")
    
    # Step 3: Create 3-way interactions (84 combinations × 2 operations = 168 features)
    logger.info("Creating 3-way feature interactions...")
    initial_shape = df_combined.shape[1]
    
    for col1, col2, col3 in combinations(base_cols, 3):
        # Addition
        df_combined[f"{col1}_p_{col2}_p_{col3}"] = df_combined[col1] + df_combined[col2] + df_combined[col3]
        # Multiplication
        df_combined[f"{col1}_m_{col2}_m_{col3}"] = df_combined[col1] * df_combined[col2] * df_combined[col3]
    
    logger.info(f"After 3-way interactions: {df_combined.shape[1]} features (was {initial_shape})")
    logger.info(f"Total features created: {df_combined.shape[1]}")
    
    # Split back into train and test
    X_train = df_combined.iloc[:len(train_df)]
    X_test = df_combined.iloc[len(train_df):]
    y_train = train_df[target_col]
    
    # Reset indices
    X_train.index = range(len(X_train))
    X_test.index = range(len(X_test))
    y_train.index = range(len(y_train))
    
    # Convert to float32 for memory efficiency
    for col in X_train.columns:
        if X_train[col].dtype == 'float64':
            X_train[col] = X_train[col].astype('float32')
            X_test[col] = X_test[col].astype('float32')
    
    return X_train, X_test, y_train


def train_xgboost_with_oof(X_train, y_train, X_test, model_params=None):
    """Train XGBoost with out-of-fold predictions matching notebook parameters."""
    
    # XGBoost parameters with GPU support (matching notebook but with GPU)
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
            'tree_method': 'gpu_hist',  # GPU acceleration
            'predictor': 'gpu_predictor',  # GPU prediction
            'gpu_id': 0,  # Use first GPU (GTX 1650)
            'verbosity': 0
        }
    
    # Initialize KFold
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    # Prepare arrays for OOF predictions
    oof_predictions = np.zeros(len(X_train))
    test_predictions = np.zeros(len(X_test))
    feature_importance = pd.DataFrame()
    
    cv_scores = []
    
    logger.info(f"Training XGBoost with {N_FOLDS}-fold CV (GPU Accelerated)")
    logger.info(f"Training samples: {len(X_train)}, Features: {X_train.shape[1]}")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        logger.info(f"\nFold {fold}/{N_FOLDS}")
        
        # Split data
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        # Train model
        model = XGBoostModel(params=model_params)
        model.fit(
            X_fold_train, 
            y_fold_train,
            X_val=X_fold_val,
            y_val=y_fold_val,
            early_stopping_rounds=250
        )
        
        # Get predictions
        oof_predictions[val_idx] = model.predict(X_fold_val)
        test_predictions += model.predict(X_test) / N_FOLDS
        
        # Calculate fold score
        fold_rmse = np.sqrt(mean_squared_error(y_fold_val, oof_predictions[val_idx]))
        cv_scores.append(fold_rmse)
        logger.info(f"Fold {fold} RMSE: {fold_rmse:.4f}")
        
        # Store feature importance
        if hasattr(model.model, 'feature_importances_'):
            fold_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.model.feature_importances_,
                'fold': fold
            })
            feature_importance = pd.concat([feature_importance, fold_importance], ignore_index=True)
    
    # Calculate overall CV score
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
    logger.info("PS5E9 - XGBOOST WITH ORIGINAL DATA")
    logger.info("="*60)
    
    # Initialize performance tracker
    tracker = ModelPerformanceTracker("output/model_performance_tracker.csv")
    
    # Initialize MLflow
    logger.info(f"\nInitializing MLflow tracking at {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # Load competition data
    logger.info("\nLoading competition data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    submission = pd.read_csv(DATA_DIR / "sample_submission.csv")
    
    # Store IDs and remove from dataframes
    train_ids = train_df[ID_COL].copy()
    test_ids = test_df[ID_COL].copy()
    
    train_df = train_df.drop(columns=[ID_COL])
    test_df = test_df.drop(columns=[ID_COL])
    
    logger.info(f"Competition train shape: {train_df.shape}")
    logger.info(f"Competition test shape: {test_df.shape}")
    
    # Create features using notebook approach WITH ORIGINAL DATA
    logger.info("\n" + "="*60)
    logger.info("Creating features with original data...")
    logger.info("="*60)
    
    X_train, X_test, y_train = create_features_notebook_style(
        train_df, 
        test_df, 
        TARGET_COL,
        use_original=True  # This will load Train_original.csv
    )
    
    logger.info(f"\nFinal shapes:")
    logger.info(f"  X_train: {X_train.shape} (includes original data if found)")
    logger.info(f"  X_test: {X_test.shape}")
    logger.info(f"  y_train: {y_train.shape}")
    
    # Only use competition data for OOF scoring (first 524164 samples)
    competition_samples = min(524164, len(X_train))
    logger.info(f"  Competition samples for OOF: {competition_samples}")
    
    # Start MLflow run
    with mlflow.start_run(run_name="xgboost_with_original"):
        # Log parameters
        mlflow.log_param("n_folds", N_FOLDS)
        mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_competition_samples", competition_samples)
        mlflow.log_param("uses_original_data", len(X_train) > 524164)
        mlflow.log_param("model_type", "XGBoost_With_Original")
        mlflow.log_param("feature_engineering", "notebook_exact_with_original")
        mlflow.log_param("gpu_enabled", True)
        
        # Train model with OOF
        logger.info("\nTraining XGBoost with Original Data...")
        oof_preds, test_preds, feature_importance, cv_rmse, cv_mae, cv_r2 = train_xgboost_with_oof(
            X_train, 
            y_train, 
            X_test
        )
        
        # Log metrics to MLflow
        mlflow.log_metric("cv_rmse", cv_rmse)
        mlflow.log_metric("cv_mae", cv_mae)
        mlflow.log_metric("cv_r2", cv_r2)
        
        # Log top features
        if not feature_importance.empty:
            feature_importance_mean = feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
            
            logger.info("\nTop 15 Most Important Features:")
            for i, (feat, imp) in enumerate(feature_importance_mean.head(15).items(), 1):
                logger.info(f"  {i:2d}. {feat}: {imp:.4f}")
        
        # Save OOF predictions (only for competition data)
        logger.info("\nSaving OOF predictions...")
        oof_manager = OOFManager(output_dir=str(OOF_DIR))
        
        oof_df = pd.DataFrame({
            'predictions': oof_preds[:competition_samples],
            'target': y_train[:competition_samples]
        })
        
        oof_manager.save_oof(
            predictions=oof_df,
            test_predictions=test_preds,
            model_name="xgboost_with_original",
            cv_score=cv_rmse,
            model_params={
                'model_type': 'XGBoost_With_Original',
                'n_folds': N_FOLDS,
                'n_features': X_train.shape[1],
                'n_train_samples': len(X_train),
                'uses_original': len(X_train) > 524164
            }
        )
        
        # Create submission
        logger.info("\nCreating submission file...")
        submission[TARGET_COL] = test_preds
        submission_file = SUBMISSION_DIR / f"submission_xgboost_original_cv{cv_rmse:.4f}.csv"
        submission.to_csv(submission_file, index=False)
        logger.info(f"Submission saved to {submission_file}")
        
        # Log submission as artifact
        mlflow.log_artifact(str(submission_file))
    
    # Track performance in CSV
    tracker.add_record(
        script_name=__file__,
        dataset_file=DATA_DIR / "train.csv",
        model_name="xgboost_with_original",
        n_folds=N_FOLDS,
        cv_rmse=cv_rmse,
        cv_mae=cv_mae,
        cv_r2=cv_r2,
        notes="XGBoost with original BPM data (14,633 samples) + GPU acceleration"
    )
    
    logger.info("\n" + "="*60)
    logger.info("✅ XGBoost with Original Data Training Complete!")
    logger.info("="*60)
    logger.info(f"   CV RMSE: {cv_rmse:.4f}")
    logger.info(f"   Target CV: ~26.46 (notebook with original data)")
    logger.info(f"   Training samples used: {len(X_train)}")
    logger.info(f"   OOF predictions saved to {OOF_DIR}")
    logger.info(f"   Submission saved to {submission_file}")


if __name__ == "__main__":
    main()