"""
PS5E9 Competition - Improved XGBoost Model
Based on insights from top performing notebook (CV: 26.46)
Incorporates log transformations, original data, and extensive feature interactions
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

from tabml import XGBoostModel, OOFManager, MLflowTracker
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
    # Try multiple possible paths for the original data
    possible_paths = [
        Path("../../data/raw/bpm-prediction-challenge/Train.csv"),
        Path("../../data/external/bpm-prediction-challenge/Train.csv"),
        Path("../../data/original/Train.csv"),
    ]
    
    for path in possible_paths:
        if path.exists():
            logger.info(f"Found original data at {path}")
            original_df = pd.read_csv(path)
            # Ensure columns match
            original_df = original_df[['RhythmScore', 'AudioLoudness', 'VocalContent', 
                                      'AcousticQuality', 'InstrumentalScore', 
                                      'LivePerformanceLikelihood', 'MoodScore', 
                                      'TrackDurationMs', 'Energy', 'BeatsPerMinute']]
            original_df['Source'] = 'Original'
            return original_df
    
    logger.warning("Original dataset not found, proceeding with competition data only")
    return None


def create_advanced_features(df, feature_cols):
    """
    Create advanced features based on the winning notebook approach.
    Includes log transformations and multi-way interactions.
    """
    df = df.copy()
    
    # Adjust AudioLoudness to be positive before log transform
    df["AudioLoudness"] = df["AudioLoudness"] - df["AudioLoudness"].min()
    
    # Apply log1p transformation to all numeric features
    for col in feature_cols:
        df[f"{col}_log"] = np.log1p(df[col])
    
    # Get log-transformed columns for interactions
    log_cols = [f"{col}_log" for col in feature_cols]
    
    # Create 2-way interactions (addition, multiplication, division)
    logger.info("Creating 2-way feature interactions...")
    for col1, col2 in list(combinations(log_cols[:9], 2)):  # Use first 9 to limit explosion
        # Addition
        df[f"{col1}_p_{col2}"] = df[col1] + df[col2]
        # Multiplication
        df[f"{col1}_m_{col2}"] = df[col1] * df[col2]
        # Division (with small epsilon to avoid division by zero)
        df[f"{col1}_d_{col2}"] = df[col1] / (df[col2] + 1e-6)
    
    # Create 3-way interactions (addition and multiplication only)
    logger.info("Creating 3-way feature interactions...")
    for col1, col2, col3 in list(combinations(log_cols[:7], 3)):  # Use first 7 to limit size
        # Addition
        df[f"{col1}_p_{col2}_p_{col3}"] = df[col1] + df[col2] + df[col3]
        # Multiplication
        df[f"{col1}_m_{col2}_m_{col3}"] = df[col1] * df[col2] * df[col3]
    
    # Additional domain-specific features
    df['energy_per_second'] = df['Energy_log'] / (df['TrackDurationMs_log'] / 1000 + 1e-6)
    df['rhythm_energy_ratio'] = df['RhythmScore_log'] / (df['Energy_log'] + 1e-6)
    df['vocal_instrumental_balance'] = df['VocalContent_log'] - df['InstrumentalScore_log']
    df['acoustic_electronic'] = df['AcousticQuality_log'] - df['LivePerformanceLikelihood_log']
    df['mood_energy_interaction'] = df['MoodScore_log'] * df['Energy_log']
    
    # Drop the original Source column if it exists
    if 'Source' in df.columns:
        df = df.drop('Source', axis=1)
    
    return df


def train_xgboost_with_oof(X_train, y_train, X_test, model_params=None):
    """Train XGBoost with out-of-fold predictions using improved parameters."""
    
    # Optimized XGBoost parameters based on notebook
    if model_params is None:
        model_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 10000,
            'learning_rate': 0.02,  # Lower learning rate like in notebook
            'max_depth': 8,
            'colsample_bytree': 0.50,
            'colsample_bynode': 0.35,
            'subsample': 0.85,
            'min_child_weight': 1,
            'gamma': 0.01,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': RANDOM_SEED,
            'n_jobs': -1,
            'tree_method': 'hist',  # Faster training
            'enable_categorical': False  # We're using all numeric
        }
    
    # Initialize KFold
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    # Prepare arrays for OOF predictions
    oof_predictions = np.zeros(len(X_train))
    test_predictions = np.zeros(len(X_test))
    feature_importance = pd.DataFrame()
    
    cv_scores = []
    
    logger.info(f"Training XGBoost with {N_FOLDS}-fold CV")
    logger.info(f"Features: {X_train.shape[1]}")
    
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
            early_stopping_rounds=250  # Higher early stopping like notebook
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
    
    logger.info(f"\nOverall CV RMSE: {cv_rmse:.4f}")
    logger.info(f"Overall CV MAE: {cv_mae:.4f}")
    logger.info(f"Overall CV R²: {cv_r2:.4f}")
    logger.info(f"CV scores by fold: {cv_scores}")
    logger.info(f"CV mean ± std: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    return oof_predictions, test_predictions, feature_importance, cv_rmse, cv_mae, cv_r2


def main():
    """Main execution function."""
    logger.info("="*60)
    logger.info("PS5E9 - IMPROVED XGBOOST MODEL")
    logger.info("="*60)
    
    # Initialize performance tracker
    tracker = ModelPerformanceTracker("output/model_performance_tracker.csv")
    
    # Initialize MLflow
    logger.info(f"\nInitializing MLflow tracking at {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # Load competition data
    logger.info("\nLoading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    submission = pd.read_csv(DATA_DIR / "sample_submission.csv")
    
    # Add source column
    train_df['Source'] = 'Competition'
    test_df['Source'] = 'Competition'
    
    # Load and merge original data if available
    original_df = load_original_data()
    if original_df is not None:
        logger.info(f"Merging {len(original_df)} samples from original dataset")
        # Combine with training data
        train_df_combined = pd.concat([
            train_df.drop('id', axis=1),
            original_df
        ], ignore_index=True)
    else:
        train_df_combined = train_df.drop('id', axis=1)
    
    logger.info(f"Combined train shape: {train_df_combined.shape}")
    logger.info(f"Test shape: {test_df.shape}")
    
    # Separate features and target
    test_ids = test_df[ID_COL]
    
    y_train = train_df_combined[TARGET_COL]
    X_train = train_df_combined.drop([TARGET_COL], axis=1)
    X_test = test_df.drop([ID_COL], axis=1)
    
    # Get feature columns (excluding Source)
    feature_cols = [col for col in X_train.columns if col != 'Source']
    
    # Create advanced features
    logger.info("\nCreating advanced features with log transforms and interactions...")
    X_train = create_advanced_features(X_train, feature_cols)
    X_test = create_advanced_features(X_test, feature_cols)
    
    logger.info(f"Features after engineering: {X_train.shape[1]}")
    
    # Reduce memory usage
    logger.info("\nReducing memory usage...")
    for col in X_train.columns:
        if X_train[col].dtype == 'float64':
            X_train[col] = X_train[col].astype('float32')
            X_test[col] = X_test[col].astype('float32')
    
    # Start MLflow run
    with mlflow.start_run(run_name="xgboost_improved"):
        # Log parameters
        mlflow.log_param("n_folds", N_FOLDS)
        mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("model_type", "XGBoost_Improved")
        mlflow.log_param("uses_original_data", original_df is not None)
        mlflow.log_param("feature_engineering", "log_transform_interactions")
        
        # Train model with OOF
        logger.info("\nTraining Improved XGBoost with OOF predictions...")
        oof_preds, test_preds, feature_importance, cv_rmse, cv_mae, cv_r2 = train_xgboost_with_oof(
            X_train, 
            y_train, 
            X_test
        )
        
        # Log metrics to MLflow
        mlflow.log_metric("cv_rmse", cv_rmse)
        mlflow.log_metric("cv_mae", cv_mae)
        mlflow.log_metric("cv_r2", cv_r2)
        
        # Calculate and log feature importance
        if not feature_importance.empty:
            feature_importance_mean = feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
            
            logger.info("\nTop 20 Features:")
            for feat, imp in feature_importance_mean.head(20).items():
                logger.info(f"  {feat}: {imp:.4f}")
        
        # Save OOF predictions using OOFManager
        logger.info("\nSaving OOF predictions...")
        oof_manager = OOFManager(output_dir=str(OOF_DIR))
        
        # For OOF, only use competition data (not original)
        competition_mask = train_df_combined.index < len(train_df)
        oof_df = pd.DataFrame({
            'predictions': oof_preds[competition_mask],
            'target': y_train[competition_mask]
        })
        
        oof_manager.save_oof(
            predictions=oof_df,
            test_predictions=test_preds,
            model_name="xgboost_improved",
            cv_score=cv_rmse,
            model_params={
                'model_type': 'XGBoost_Improved',
                'n_folds': N_FOLDS,
                'n_features': X_train.shape[1],
                'uses_original': original_df is not None
            }
        )
        
        # Create submission
        logger.info("\nCreating submission file...")
        submission[TARGET_COL] = test_preds
        submission_file = SUBMISSION_DIR / f"submission_xgboost_improved_cv{cv_rmse:.4f}.csv"
        submission.to_csv(submission_file, index=False)
        logger.info(f"Submission saved to {submission_file}")
        
        # Log submission as artifact
        mlflow.log_artifact(str(submission_file))
    
    # Track performance in CSV
    tracker.add_record(
        script_name=__file__,
        dataset_file=DATA_DIR / "train.csv",
        model_name="xgboost_improved",
        n_folds=N_FOLDS,
        cv_rmse=cv_rmse,
        cv_mae=cv_mae,
        cv_r2=cv_r2,
        notes="Improved XGBoost with log transforms, interactions, original data"
    )
    
    logger.info("\n✅ Improved XGBoost training complete!")
    logger.info(f"   CV RMSE: {cv_rmse:.4f}")
    logger.info(f"   Expected improvement from baseline: ~0.06 RMSE")
    logger.info(f"   OOF predictions saved to {OOF_DIR}")
    logger.info(f"   Submission saved to {submission_file}")


if __name__ == "__main__":
    main()