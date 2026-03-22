"""
PS5E9 Competition - Baseline XGBoost Model
Train XGBoost model with OOF predictions and MLflow tracking
"""

import os
import sys
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import tabml
sys.path.append(str(Path(__file__).parent.parent.parent))

from tabml import (
    DataLoader, 
    FeatureEngineer, 
    XGBoostModel,
    OOFEnsemble,
    OOFManager,
    MLflowTracker
)
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
        
        # Load existing tracker or create new
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
        return hash_md5.hexdigest()[:8]  # First 8 chars for brevity
    
    def add_record(self, script_name, dataset_file, model_name, n_folds, 
                   cv_rmse, cv_mae=None, cv_r2=None, notes=""):
        """Add a new performance record."""
        # Get hashes
        script_hash = self.get_file_hash(script_name)
        data_hash = self.get_file_hash(dataset_file)
        
        # Create new record
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
            'public_lb': 'NA',  # To be filled manually after submission
            'private_lb': 'NA',  # To be filled at competition end
            'notes': notes
        }
        
        # Append to dataframe
        self.df = pd.concat([self.df, pd.DataFrame([new_record])], ignore_index=True)
        
        # Save to CSV
        self.df.to_csv(self.tracker_file, index=False)
        logger.info(f"Added performance record: CV RMSE={cv_rmse:.4f}")
        
        return new_record
    
    def get_best_models(self, metric='cv_rmse', top_k=5):
        """Get top performing models by metric."""
        return self.df.nsmallest(top_k, metric)[['model_name', 'cv_rmse', 'cv_mae', 'cv_r2', 'public_lb']]


def create_features(df):
    """Create additional features based on domain knowledge."""
    df = df.copy()
    
    # Interaction features
    df['energy_rhythm'] = df['Energy'] * df['RhythmScore']
    df['vocal_acoustic'] = df['VocalContent'] * df['AcousticQuality']
    df['mood_energy'] = df['MoodScore'] * df['Energy']
    
    # Duration-based features
    df['duration_minutes'] = df['TrackDurationMs'] / 60000
    df['energy_per_minute'] = df['Energy'] / (df['duration_minutes'] + 0.001)
    
    # Log transformations for skewed features
    df['log_duration'] = np.log1p(df['TrackDurationMs'])
    df['log_vocal'] = np.log1p(df['VocalContent'])
    df['log_instrumental'] = np.log1p(df['InstrumentalScore'])
    
    # Polynomial features for top predictors
    df['rhythm_squared'] = df['RhythmScore'] ** 2
    df['energy_squared'] = df['Energy'] ** 2
    
    # Ratio features
    df['vocal_instrumental_ratio'] = df['VocalContent'] / (df['InstrumentalScore'] + 0.001)
    df['acoustic_live_ratio'] = df['AcousticQuality'] / (df['LivePerformanceLikelihood'] + 0.001)
    
    return df


def train_xgboost_with_oof(X_train, y_train, X_test, model_params=None):
    """Train XGBoost with out-of-fold predictions."""
    
    # Default XGBoost parameters for regression
    if model_params is None:
        model_params = {
            'n_estimators': 2000,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'min_child_weight': 1,
            'gamma': 0.01,
            'reg_alpha': 0.01,
            'reg_lambda': 0.1,
            'random_state': RANDOM_SEED,
            'n_jobs': -1
        }
    
    # Initialize KFold
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    # Prepare arrays for OOF predictions
    oof_predictions = np.zeros(len(X_train))
    test_predictions = np.zeros(len(X_test))
    feature_importance = pd.DataFrame()
    
    cv_scores = []
    
    logger.info(f"Training XGBoost with {N_FOLDS}-fold CV")
    
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
            early_stopping_rounds=50
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
    logger.info("PS5E9 - BASELINE XGBOOST MODEL")
    logger.info("="*60)
    
    # Initialize performance tracker
    tracker = ModelPerformanceTracker("output/model_performance_tracker.csv")
    
    # Initialize MLflow
    logger.info(f"\nInitializing MLflow tracking at {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # Load data
    logger.info("\nLoading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    submission = pd.read_csv(DATA_DIR / "sample_submission.csv")
    
    logger.info(f"Train shape: {train_df.shape}")
    logger.info(f"Test shape: {test_df.shape}")
    
    # Separate features and target
    train_ids = train_df[ID_COL]
    test_ids = test_df[ID_COL]
    
    X_train = train_df.drop([ID_COL, TARGET_COL], axis=1)
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop([ID_COL], axis=1)
    
    # Create features
    logger.info("\nCreating features...")
    X_train = create_features(X_train)
    X_test = create_features(X_test)
    
    logger.info(f"Features after engineering: {X_train.shape[1]}")
    logger.info(f"Feature names: {list(X_train.columns)}")
    
    # Feature engineering with TabML
    logger.info("\nApplying feature engineering...")
    engineer = FeatureEngineer(
        numeric_impute_strategy='median',
        scaling_method='standard'
    )
    
    X_train_scaled = engineer.fit_transform(X_train, y_train)
    X_test_scaled = engineer.transform(X_test)
    
    # Convert back to DataFrame for easier handling
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Start MLflow run
    with mlflow.start_run(run_name="xgboost_baseline"):
        # Log parameters
        mlflow.log_param("n_folds", N_FOLDS)
        mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_param("n_features", X_train_scaled.shape[1])
        mlflow.log_param("model_type", "XGBoost")
        
        # Train model with OOF
        logger.info("\nTraining XGBoost with OOF predictions...")
        oof_preds, test_preds, feature_importance, cv_rmse, cv_mae, cv_r2 = train_xgboost_with_oof(
            X_train_scaled, 
            y_train, 
            X_test_scaled
        )
        
        # Log metrics to MLflow
        mlflow.log_metric("cv_rmse", cv_rmse)
        mlflow.log_metric("cv_mae", cv_mae)
        mlflow.log_metric("cv_r2", cv_r2)
        
        # Calculate and log feature importance
        if not feature_importance.empty:
            feature_importance_mean = feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
            
            logger.info("\nTop 15 Features:")
            for feat, imp in feature_importance_mean.head(15).items():
                logger.info(f"  {feat}: {imp:.4f}")
                mlflow.log_metric(f"feature_importance_{feat}", imp)
        
        # Save OOF predictions using OOFManager
        logger.info("\nSaving OOF predictions...")
        oof_manager = OOFManager(output_dir=str(OOF_DIR))
        
        oof_df = pd.DataFrame({
            'id': train_ids,
            'predictions': oof_preds,
            'target': y_train
        })
        
        oof_manager.save_oof(
            predictions=oof_df,
            test_predictions=test_preds,
            model_name="xgboost_baseline",
            cv_score=cv_rmse,
            model_params={
                'model_type': 'XGBoost',
                'n_folds': N_FOLDS,
                'features': list(X_train_scaled.columns)
            }
        )
        
        # Create submission
        logger.info("\nCreating submission file...")
        submission[TARGET_COL] = test_preds
        submission_file = SUBMISSION_DIR / f"submission_xgboost_cv{cv_rmse:.4f}.csv"
        submission.to_csv(submission_file, index=False)
        logger.info(f"Submission saved to {submission_file}")
        
        # Log submission as artifact
        mlflow.log_artifact(str(submission_file))
    
    # Track performance in CSV
    tracker.add_record(
        script_name=__file__,
        dataset_file=DATA_DIR / "train.csv",
        model_name="xgboost_baseline",
        n_folds=N_FOLDS,
        cv_rmse=cv_rmse,
        cv_mae=cv_mae,
        cv_r2=cv_r2,
        notes="Baseline XGBoost with feature engineering"
    )
    
    # Show best models so far
    logger.info("\nTop models in tracker:")
    best_models = tracker.get_best_models(top_k=5)
    if not best_models.empty:
        logger.info(f"\n{best_models.to_string()}")
    
    logger.info("\n✅ XGBoost baseline training complete!")
    logger.info(f"   CV RMSE: {cv_rmse:.4f}")
    logger.info(f"   OOF predictions saved to {OOF_DIR}")
    logger.info(f"   Submission saved to {submission_file}")
    logger.info(f"   MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")


if __name__ == "__main__":
    main()