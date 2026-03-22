"""
PS5E9 Competition - XGBoost for Kaggle with T4 GPU
Optimized for Kaggle environment with correct feature engineering order
Target CV: ~26.46
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Install required packages for Kaggle
import subprocess

def install_package(package):
    """Install a package quietly."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

def install_tabml_dependencies():
    """Install all TabML dependencies."""
    dependencies = [
        "numpy",
        "pandas", 
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "catboost",
        "loguru",
        "python-dotenv",
        "matplotlib",
        "seaborn",
        "tqdm"
    ]
    for dep in dependencies:
        try:
            install_package(dep)
        except:
            print(f"Warning: Could not install {dep}")

# Install TabML and dependencies if not available
try:
    from tabml import XGBoostModel, OOFManager
    from tabml import EDAAnalyzer  # Test full import
    print("TabML already installed")
except ImportError as e:
    print(f"TabML not found: {e}")
    print("Installing TabML dependencies first...")
    install_tabml_dependencies()
    
    print("Installing TabML from GitHub...")
    install_package("git+https://github.com/wguesdon/tabml.git")
    
    # Try importing again with better error handling
    try:
        from tabml import XGBoostModel, OOFManager
        print("TabML successfully installed and imported")
    except ImportError as e2:
        print(f"Failed to import TabML after installation: {e2}")
        print("Trying alternative import...")
        # Try importing specific modules
        from tabml.models import XGBoostModel
        from tabml.oof_manager import OOFManager
        print("Successfully imported using direct module imports")

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from loguru import logger

# Kaggle paths
DATA_DIR = Path("/kaggle/input/playground-series-s5e9")
ORIGINAL_DATA_DIR = Path("/kaggle/input/bpm-prediction-challenge")  # If you add as additional dataset
OUTPUT_DIR = Path("/kaggle/working")
OOF_DIR = OUTPUT_DIR / "oof_predictions"
SUBMISSION_DIR = OUTPUT_DIR
OOF_DIR.mkdir(parents=True, exist_ok=True)

# Competition settings
TARGET_COL = 'BeatsPerMinute'
ID_COL = 'id'
RANDOM_SEED = 42
N_FOLDS = 5

# Check GPU availability
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        logger.info(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
except:
    GPU_AVAILABLE = False
    logger.info("No GPU detected, using CPU")


def load_original_data():
    """Load the original BPM dataset if available in Kaggle."""
    # Try different possible locations in Kaggle
    possible_paths = [
        ORIGINAL_DATA_DIR / "Train.csv",
        DATA_DIR / "Train_original.csv",
        Path("/kaggle/working/Train_original.csv"),
    ]
    
    for original_path in possible_paths:
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
    
    logger.warning("Original dataset not found")
    logger.warning("To improve performance, add 'BPM Prediction Challenge' dataset to Kaggle notebook")
    return None


def create_features_correct_order(train_df, test_df, target_col):
    """
    Create features with CORRECT order (matching the winning notebook):
    1. Feature engineering on competition data ONLY
    2. THEN add original data with same transformations
    """
    
    # STEP 1: Feature engineering on COMPETITION DATA ONLY
    logger.info("Step 1: Feature engineering on competition data...")
    
    # Combine competition train and test
    train_features = train_df.drop(columns=[target_col])
    df_combined = pd.concat([train_features, test_df], axis=0, ignore_index=True)
    
    logger.info(f"  Combined shape: {df_combined.shape}")
    
    # Base feature columns
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
    
    logger.info(f"  Total features: {df_combined.shape[1]}")
    
    # Split back
    X_train_comp = df_combined.iloc[:len(train_df)].copy()
    X_test = df_combined.iloc[len(train_df):].copy()
    y_train_comp = train_df[target_col].copy()
    
    # STEP 2: Add original data if available
    logger.info("\nStep 2: Adding original data...")
    
    original_df = load_original_data()
    if original_df is not None:
        logger.info("  Applying same transformations to original data...")
        
        # Apply SAME transformations with SAME min_loudness
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
        
        # Extract target and features
        y_train_orig = original_df[target_col].copy()
        X_train_orig = original_df.drop(columns=[target_col])
        
        # Combine
        X_train = pd.concat([X_train_comp, X_train_orig], axis=0, ignore_index=True)
        y_train = pd.concat([y_train_comp, y_train_orig], axis=0, ignore_index=True)
        
        logger.info(f"  Combined: {len(X_train)} samples ({len(X_train_comp)} competition + {len(X_train_orig)} original)")
    else:
        X_train = X_train_comp
        y_train = y_train_comp
        logger.info("  Using competition data only")
    
    # Convert to float32 for memory efficiency
    for col in X_train.columns:
        if X_train[col].dtype == 'float64':
            X_train[col] = X_train[col].astype('float32')
            X_test[col] = X_test[col].astype('float32')
    
    return X_train, X_test, y_train


def train_xgboost_with_oof(X_train, y_train, X_test):
    """Train XGBoost with OOF predictions, optimized for Kaggle T4 GPU."""
    
    # XGBoost parameters optimized for Kaggle's T4 GPU
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
        'verbosity': 0
    }
    
    # Add GPU parameters if available
    if GPU_AVAILABLE:
        model_params.update({
            'tree_method': 'gpu_hist',
            'predictor': 'gpu_predictor',
            'gpu_id': 0,
        })
        logger.info("Using GPU acceleration (T4)")
    else:
        model_params['tree_method'] = 'hist'
        logger.info("Using CPU")
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    oof_predictions = np.zeros(len(X_train))
    test_predictions = np.zeros(len(X_test))
    feature_importance = pd.DataFrame()
    cv_scores = []
    
    logger.info(f"\nTraining XGBoost with {N_FOLDS}-fold CV")
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
    """Main execution function for Kaggle."""
    logger.info("="*60)
    logger.info("PS5E9 - XGBOOST KAGGLE VERSION")
    logger.info(f"Environment: Kaggle - GPU: {GPU_AVAILABLE}")
    logger.info("="*60)
    
    # Load competition data
    logger.info("\nLoading competition data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    submission = pd.read_csv(DATA_DIR / "sample_submission.csv")
    
    # Store test IDs and remove from dataframes
    test_ids = test_df[ID_COL].copy()
    train_df = train_df.drop(columns=[ID_COL])
    test_df = test_df.drop(columns=[ID_COL])
    
    logger.info(f"Train shape: {train_df.shape}")
    logger.info(f"Test shape: {test_df.shape}")
    
    # Create features with correct order
    logger.info("\n" + "="*60)
    logger.info("FEATURE ENGINEERING")
    logger.info("="*60)
    
    X_train, X_test, y_train = create_features_correct_order(
        train_df, 
        test_df, 
        TARGET_COL
    )
    
    logger.info(f"\nFinal shapes: X_train={X_train.shape}, X_test={X_test.shape}")
    
    # Train model
    logger.info("\n" + "="*60)
    logger.info("MODEL TRAINING")
    logger.info("="*60)
    
    oof_preds, test_preds, feature_importance, cv_rmse, cv_mae, cv_r2 = train_xgboost_with_oof(
        X_train, 
        y_train, 
        X_test
    )
    
    # Show feature importance
    if not feature_importance.empty:
        feature_importance_mean = feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
        logger.info("\nTop 15 Features:")
        for i, (feat, imp) in enumerate(feature_importance_mean.head(15).items(), 1):
            logger.info(f"  {i:2d}. {feat}: {imp:.4f}")
    
    # Save OOF predictions for ensemble (only competition data)
    competition_samples = min(524164, len(X_train))
    logger.info(f"\nSaving OOF predictions for {competition_samples} competition samples...")
    
    oof_manager = OOFManager(output_dir=str(OOF_DIR))
    oof_df = pd.DataFrame({
        'predictions': oof_preds[:competition_samples],
        'target': y_train[:competition_samples]
    })
    
    oof_manager.save_oof(
        predictions=oof_df,
        test_predictions=test_preds,
        model_name="xgboost_kaggle",
        cv_score=cv_rmse,
        model_params={
            'model_type': 'XGBoost_Kaggle',
            'n_folds': N_FOLDS,
            'n_features': X_train.shape[1],
            'gpu_used': GPU_AVAILABLE
        }
    )
    
    # Create submission
    logger.info("\nCreating submission...")
    submission[TARGET_COL] = test_preds
    submission_file = SUBMISSION_DIR / f"submission_xgboost_cv{cv_rmse:.4f}.csv"
    submission.to_csv(submission_file, index=False)
    logger.info(f"Submission saved to {submission_file}")
    
    # Display first few predictions
    logger.info("\nFirst 10 predictions:")
    logger.info(submission.head(10).to_string())
    
    logger.info("\n" + "="*60)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info("="*60)
    logger.info(f"   CV RMSE: {cv_rmse:.4f}")
    logger.info(f"   Target: ~26.46 (with original data)")
    logger.info(f"   Submission: {submission_file}")
    logger.info("\nTo improve score:")
    logger.info("  1. Add 'BPM Prediction Challenge' dataset for original data")
    logger.info("  2. Try LightGBM and CatBoost models")
    logger.info("  3. Create ensemble of models")
    
    return cv_rmse, submission


# For Kaggle notebook execution
if __name__ == "__main__":
    try:
        cv_score, submission_df = main()
        print(f"\n✅ Final CV RMSE: {cv_score:.4f}")
        print(f"Submission shape: {submission_df.shape}")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise