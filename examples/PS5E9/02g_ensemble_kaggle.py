"""
PS5E9 Competition - Multi-Model Ensemble with Hill Climbing and Feature Importance
Trains XGBoost, LightGBM, CatBoost, RandomForest and optimizes ensemble weights
Includes comprehensive feature importance analysis and visualization
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
        "tqdm",
        "optuna"  # For hyperparameter optimization if needed
    ]
    for dep in dependencies:
        try:
            install_package(dep)
        except:
            print(f"Warning: Could not install {dep}")

# Install TabML and dependencies if not available
try:
    from tabml import (
        XGBoostModel, LightGBMModel, CatBoostModel, 
        RandomForestModel, OOFManager
    )
    print("TabML already installed")
except ImportError as e:
    print(f"TabML not found: {e}")
    print("Installing TabML dependencies first...")
    install_tabml_dependencies()
    
    print("Installing TabML from GitHub...")
    install_package("git+https://github.com/wguesdon/tabml.git")
    
    # Try importing again with better error handling
    try:
        from tabml import (
            XGBoostModel, LightGBMModel, CatBoostModel, 
            RandomForestModel, OOFManager
        )
        print("TabML successfully installed and imported")
    except ImportError as e2:
        print(f"Failed to import TabML after installation: {e2}")
        print("Trying alternative import...")
        from tabml.models import (
            XGBoostModel, LightGBMModel, CatBoostModel, 
            RandomForestModel
        )
        from tabml.oof_manager import OOFManager
        print("Successfully imported using direct module imports")

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import minimize
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# Kaggle paths
DATA_DIR = Path("/kaggle/input/playground-series-s5e9")
ORIGINAL_DATA_DIR = Path("/kaggle/input/bpm-prediction-challenge")
OUTPUT_DIR = Path("/kaggle/working")
OOF_DIR = OUTPUT_DIR / "oof_predictions"
SUBMISSION_DIR = OUTPUT_DIR
IMPORTANCE_DIR = OUTPUT_DIR / "feature_importance"
OOF_DIR.mkdir(parents=True, exist_ok=True)
IMPORTANCE_DIR.mkdir(parents=True, exist_ok=True)

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


# ============================================================================
# FEATURE IMPORTANCE FUNCTIONS
# ============================================================================

def get_feature_importance(model, model_type, feature_names):
    """
    Extract feature importance from different model types.
    
    Args:
        model: Trained model instance (TabML wrapper)
        model_type: String indicating model type ('xgboost', 'lightgbm', 'catboost', 'rf')
        feature_names: List of feature names
    
    Returns:
        DataFrame with features and their importance scores
    """
    logger.debug(f"Extracting importance for {model_type}, model type: {type(model.model) if hasattr(model, 'model') else type(model)}")
    if model_type.lower() == 'xgboost':
        # For XGBoost with TabML wrapper
        import xgboost as xgb
        
        # Check if it's a Booster object or has feature_importances_
        if hasattr(model.model, 'feature_importances_'):
            importance = model.model.feature_importances_
        elif isinstance(model.model, xgb.Booster) or hasattr(model.model, 'get_score'):
            # For XGBoost Booster object
            try:
                importance_dict = model.model.get_score(importance_type='gain')
                logger.debug(f"XGBoost importance dict has {len(importance_dict)} entries")
                
                # Convert to array matching feature order
                importance = np.zeros(len(feature_names))
                
                # Check what kind of keys are in the importance dict
                if importance_dict:
                    sample_keys = list(importance_dict.keys())[:3]
                    logger.debug(f"Sample importance keys: {sample_keys}")
                
                # XGBoost can use either actual feature names or internal names (f0, f1, ...)
                matched_count = 0
                for i, feat_name in enumerate(feature_names):
                    # First try the actual feature name
                    if feat_name in importance_dict:
                        importance[i] = importance_dict[feat_name]
                        matched_count += 1
                    else:
                        # Fallback to internal name (f0, f1, ...)
                        internal_name = f'f{i}'
                        if internal_name in importance_dict:
                            importance[i] = importance_dict[internal_name]
                            matched_count += 1
                
                logger.debug(f"Matched {matched_count} features out of {len(feature_names)}")
            except Exception as e:
                # Try without importance_type parameter
                try:
                    importance_dict = model.model.get_score()
                    importance = np.zeros(len(feature_names))
                    for i, feat_name in enumerate(feature_names):
                        if feat_name in importance_dict:
                            importance[i] = importance_dict[feat_name]
                        else:
                            internal_name = f'f{i}'
                            if internal_name in importance_dict:
                                importance[i] = importance_dict[internal_name]
                except:
                    raise ValueError(f"Cannot extract feature importance from XGBoost model: {e}")
        else:
            raise ValueError("Cannot extract feature importance from XGBoost model")
                
    elif model_type.lower() == 'lightgbm':
        # For LightGBM with TabML wrapper
        if hasattr(model.model, 'feature_importances_'):
            importance = model.model.feature_importances_
        elif hasattr(model.model, 'feature_importance'):
            importance = model.model.feature_importance(importance_type='gain')
        else:
            raise ValueError("Cannot extract feature importance from LightGBM model")
            
    elif model_type.lower() == 'catboost':
        # For CatBoost with TabML wrapper
        if hasattr(model.model, 'get_feature_importance'):
            importance = model.model.get_feature_importance()
        elif hasattr(model.model, 'feature_importances_'):
            importance = model.model.feature_importances_
        else:
            raise ValueError("Cannot extract feature importance from CatBoost model")
            
    elif model_type.lower() in ['randomforest', 'rf']:
        # For Random Forest with TabML wrapper
        if hasattr(model.model, 'feature_importances_'):
            importance = model.model.feature_importances_
        else:
            raise ValueError("Cannot extract feature importance from RandomForest model")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Ensure importance is a numpy array
    if not isinstance(importance, np.ndarray):
        importance = np.array(importance)
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    # Log summary
    non_zero_features = len(importance_df[importance_df['importance'] > 0])
    logger.debug(f"Feature importance extraction complete: {non_zero_features} non-zero features")
    if non_zero_features > 0:
        logger.debug(f"Top feature: {importance_df.iloc[0]['feature']} (importance: {importance_df.iloc[0]['importance']:.4f})")
    
    return importance_df


def plot_feature_importance(importance_df, model_name, top_n=20, save_path=None):
    """
    Plot feature importance for a single model.
    
    Args:
        importance_df: DataFrame with features and importance
        model_name: Name of the model for the title
        top_n: Number of top features to display
        save_path: Optional path to save the plot
    """
    # Select top N features
    top_features = importance_df.head(top_n)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create horizontal bar plot
    plt.barh(range(len(top_features)), top_features['importance'].values)
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    
    plt.xlabel('Feature Importance')
    plt.title(f'{model_name} - Top {top_n} Features')
    plt.gca().invert_yaxis()  # Highest importance at the top
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {save_path}")
    
    plt.show()
    plt.close()


def plot_combined_feature_importance(all_importance_dfs, top_n=20, save_path=None):
    """
    Plot combined feature importance from all models.
    
    Args:
        all_importance_dfs: Dictionary of {model_name: importance_df}
        top_n: Number of top features to display
        save_path: Optional path to save the plot
    """
    # Combine all importance DataFrames
    combined_df = None
    
    for model_name, importance_df in all_importance_dfs.items():
        # Normalize importance to 0-1 scale for fair comparison
        importance_df['importance_norm'] = (importance_df['importance'] / 
                                            importance_df['importance'].sum())
        
        # Rename column for this model
        importance_df[f'{model_name}_importance'] = importance_df['importance_norm']
        
        if combined_df is None:
            combined_df = importance_df[['feature', f'{model_name}_importance']]
        else:
            combined_df = pd.merge(
                combined_df, 
                importance_df[['feature', f'{model_name}_importance']], 
                on='feature', 
                how='outer'
            )
    
    # Fill NaN values with 0
    combined_df = combined_df.fillna(0)
    
    # Calculate average importance across all models
    importance_cols = [col for col in combined_df.columns if col.endswith('_importance')]
    combined_df['avg_importance'] = combined_df[importance_cols].mean(axis=1)
    
    # Sort by average importance
    combined_df = combined_df.sort_values('avg_importance', ascending=False)
    
    # Select top N features
    top_features = combined_df.head(top_n)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Average importance
    ax1.barh(range(len(top_features)), top_features['avg_importance'].values)
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features['feature'].values)
    ax1.set_xlabel('Average Normalized Importance')
    ax1.set_title(f'Average Feature Importance - Top {top_n}')
    ax1.invert_yaxis()
    
    # Plot 2: Heatmap of importance by model
    heatmap_data = top_features[importance_cols].T
    heatmap_data.columns = top_features['feature'].values
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax2, cbar_kws={'label': 'Normalized Importance'})
    ax2.set_title('Feature Importance by Model')
    ax2.set_xlabel('Features')
    ax2.set_ylabel('Models')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"Saved combined feature importance plot to {save_path}")
    
    plt.show()
    plt.close()
    
    return combined_df


def display_importance_stats(all_importance_dfs, top_n=10):
    """
    Display statistical summary of feature importance across models.
    
    Args:
        all_importance_dfs: Dictionary of {model_name: importance_df}
        top_n: Number of top features to display
    """
    # Combine all importance DataFrames
    combined_stats = {}
    
    for model_name, importance_df in all_importance_dfs.items():
        # Get top N features for this model
        top_features = importance_df.head(top_n)['feature'].tolist()
        combined_stats[model_name] = top_features
    
    # Find features that appear in top N for all models
    all_top_features = set()
    for features in combined_stats.values():
        all_top_features.update(features)
    
    # Count how many models have each feature in their top N
    feature_counts = {}
    for feature in all_top_features:
        count = sum(1 for features in combined_stats.values() if feature in features)
        feature_counts[feature] = count
    
    # Sort by count
    sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"FEATURE IMPORTANCE CONSENSUS (Top {top_n} features per model)")
    logger.info(f"{'='*60}")
    
    for feature, count in sorted_features[:15]:  # Show top 15 most agreed upon
        consensus_pct = (count / len(all_importance_dfs)) * 100
        logger.info(f"  {feature:40s}: {count}/{len(all_importance_dfs)} models ({consensus_pct:.0f}%)")
    
    # Show top 3 features for each model
    logger.info(f"\n{'='*60}")
    logger.info("TOP 3 FEATURES BY MODEL")
    logger.info(f"{'='*60}")
    
    for model_name, importance_df in all_importance_dfs.items():
        top_3 = importance_df.head(3)
        logger.info(f"\n{model_name}:")
        for idx, row in top_3.iterrows():
            logger.info(f"  {idx+1}. {row['feature']:35s}: {row['importance']:.4f}")


# ============================================================================
# DATA LOADING AND FEATURE ENGINEERING
# ============================================================================

def load_original_data():
    """Load the original BPM dataset if available in Kaggle."""
    possible_paths = [
        ORIGINAL_DATA_DIR / "Train.csv",
        DATA_DIR / "Train_original.csv",
        Path("/kaggle/working/Train_original.csv"),
    ]
    
    for original_path in possible_paths:
        if original_path.exists():
            logger.info(f"✓ Found original data at {original_path}")
            original_df = pd.read_csv(original_path)
            
            cols_needed = ['RhythmScore', 'AudioLoudness', 'VocalContent', 
                          'AcousticQuality', 'InstrumentalScore', 
                          'LivePerformanceLikelihood', 'MoodScore', 
                          'TrackDurationMs', 'Energy', 'BeatsPerMinute']
            
            missing_cols = [col for col in cols_needed if col not in original_df.columns]
            if missing_cols:
                logger.warning(f"Missing columns: {missing_cols}")
                return None
                
            original_df = original_df[cols_needed]
            logger.info(f"Loaded {len(original_df)} samples from original dataset")
            return original_df
    
    logger.warning("Original dataset not found - using competition data only")
    logger.warning("To improve score, add 'BPM Prediction Challenge' dataset")
    return None


def create_features_correct_order(train_df, test_df, target_col):
    """
    Create features with CORRECT order:
    1. Feature engineering on competition data ONLY
    2. Add original data with same transformations
    """
    
    # STEP 1: Feature engineering on COMPETITION DATA ONLY
    logger.info("Step 1: Feature engineering on competition data...")
    
    train_features = train_df.drop(columns=[target_col])
    df_combined = pd.concat([train_features, test_df], axis=0, ignore_index=True)
    
    logger.info(f"  Combined shape: {df_combined.shape}")
    
    base_cols = ['RhythmScore', 'AudioLoudness', 'VocalContent', 
                 'AcousticQuality', 'InstrumentalScore', 
                 'LivePerformanceLikelihood', 'MoodScore', 
                 'TrackDurationMs', 'Energy']
    
    # Adjust AudioLoudness and apply log1p
    logger.info("  Applying log transformation...")
    min_loudness = df_combined["AudioLoudness"].min()
    df_combined["AudioLoudness"] = df_combined["AudioLoudness"] - min_loudness
    
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
        
        y_train_orig = original_df[target_col].copy()
        X_train_orig = original_df.drop(columns=[target_col])
        
        X_train = pd.concat([X_train_comp, X_train_orig], axis=0, ignore_index=True)
        y_train = pd.concat([y_train_comp, y_train_orig], axis=0, ignore_index=True)
        
        logger.info(f"  Combined: {len(X_train)} samples ({len(X_train_comp)} comp + {len(X_train_orig)} orig)")
    else:
        X_train = X_train_comp
        y_train = y_train_comp
        logger.info("  Using competition data only")
    
    # Convert to float32
    for col in X_train.columns:
        if X_train[col].dtype == 'float64':
            X_train[col] = X_train[col].astype('float32')
            X_test[col] = X_test[col].astype('float32')
    
    return X_train, X_test, y_train


# ============================================================================
# TRAINING FUNCTIONS WITH FEATURE IMPORTANCE
# ============================================================================

def train_xgboost(X_train, y_train, X_test, kf):
    """Train XGBoost model with OOF predictions and feature importance."""
    logger.info("\n" + "="*50)
    logger.info("Training XGBoost")
    logger.info("="*50)
    
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
    
    if GPU_AVAILABLE:
        model_params.update({
            'tree_method': 'gpu_hist',
            'predictor': 'gpu_predictor',
            'gpu_id': 0,
        })
    else:
        model_params['tree_method'] = 'hist'
    
    oof_predictions = np.zeros(len(X_train))
    test_predictions = np.zeros(len(X_test))
    feature_importance_list = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        logger.info(f"Fold {fold}/{N_FOLDS}")
        
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
        
        # Get feature importance for this fold
        importance_df = get_feature_importance(model, 'xgboost', X_train.columns.tolist())
        feature_importance_list.append(importance_df['importance'].values)
        
        oof_predictions[val_idx] = model.predict(X_fold_val)
        test_predictions += model.predict(X_test) / N_FOLDS
        
        fold_rmse = np.sqrt(mean_squared_error(y_fold_val, oof_predictions[val_idx]))
        logger.info(f"  Fold {fold} RMSE: {fold_rmse:.4f}")
    
    # Average feature importance across folds
    avg_importance = np.mean(feature_importance_list, axis=0)
    importance_df = pd.DataFrame({
        'feature': X_train.columns.tolist(),
        'importance': avg_importance
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    cv_rmse = np.sqrt(mean_squared_error(y_train, oof_predictions))
    logger.info(f"XGBoost CV RMSE: {cv_rmse:.4f}")
    
    return oof_predictions, test_predictions, cv_rmse, importance_df


def train_lightgbm(X_train, y_train, X_test, kf):
    """Train LightGBM model with OOF predictions and feature importance."""
    logger.info("\n" + "="*50)
    logger.info("Training LightGBM")
    logger.info("="*50)
    
    model_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'n_estimators': 10000,
        'learning_rate': 0.02,
        'num_leaves': 127,
        'max_depth': 8,
        'feature_fraction': 0.50,
        'bagging_fraction': 0.85,
        'bagging_freq': 1,
        'min_child_samples': 20,
        'random_state': RANDOM_SEED,
        'n_jobs': -1,
        'verbosity': -1
    }
    
    if GPU_AVAILABLE:
        model_params['device'] = 'gpu'
    
    oof_predictions = np.zeros(len(X_train))
    test_predictions = np.zeros(len(X_test))
    feature_importance_list = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        logger.info(f"Fold {fold}/{N_FOLDS}")
        
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        model = LightGBMModel(params=model_params)
        model.fit(
            X_fold_train, 
            y_fold_train,
            X_val=X_fold_val,
            y_val=y_fold_val,
            early_stopping_rounds=250
        )
        
        # Get feature importance for this fold
        importance_df = get_feature_importance(model, 'lightgbm', X_train.columns.tolist())
        feature_importance_list.append(importance_df['importance'].values)
        
        oof_predictions[val_idx] = model.predict(X_fold_val)
        test_predictions += model.predict(X_test) / N_FOLDS
        
        fold_rmse = np.sqrt(mean_squared_error(y_fold_val, oof_predictions[val_idx]))
        logger.info(f"  Fold {fold} RMSE: {fold_rmse:.4f}")
    
    # Average feature importance across folds
    avg_importance = np.mean(feature_importance_list, axis=0)
    importance_df = pd.DataFrame({
        'feature': X_train.columns.tolist(),
        'importance': avg_importance
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    cv_rmse = np.sqrt(mean_squared_error(y_train, oof_predictions))
    logger.info(f"LightGBM CV RMSE: {cv_rmse:.4f}")
    
    return oof_predictions, test_predictions, cv_rmse, importance_df


def train_catboost(X_train, y_train, X_test, kf):
    """Train CatBoost model with OOF predictions and feature importance."""
    logger.info("\n" + "="*50)
    logger.info("Training CatBoost")
    logger.info("="*50)
    
    # Base parameters that work for both CPU and GPU
    model_params = {
        'loss_function': 'RMSE',
        'iterations': 10000,
        'learning_rate': 0.02,
        'depth': 8,
        'l2_leaf_reg': 3,
        'bootstrap_type': 'Bernoulli',
        'subsample': 0.85,
        'random_seed': RANDOM_SEED,
        'verbose': False
    }
    
    if GPU_AVAILABLE:
        model_params['task_type'] = 'GPU'
        model_params['devices'] = '0'
    else:
        model_params['rsm'] = 0.5
        model_params['thread_count'] = -1
    
    oof_predictions = np.zeros(len(X_train))
    test_predictions = np.zeros(len(X_test))
    feature_importance_list = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        logger.info(f"Fold {fold}/{N_FOLDS}")
        
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        model = CatBoostModel(params=model_params)
        model.fit(
            X_fold_train, 
            y_fold_train,
            X_val=X_fold_val,
            y_val=y_fold_val,
            early_stopping_rounds=250
        )
        
        # Get feature importance for this fold
        importance_df = get_feature_importance(model, 'catboost', X_train.columns.tolist())
        feature_importance_list.append(importance_df['importance'].values)
        
        oof_predictions[val_idx] = model.predict(X_fold_val)
        test_predictions += model.predict(X_test) / N_FOLDS
        
        fold_rmse = np.sqrt(mean_squared_error(y_fold_val, oof_predictions[val_idx]))
        logger.info(f"  Fold {fold} RMSE: {fold_rmse:.4f}")
    
    # Average feature importance across folds
    avg_importance = np.mean(feature_importance_list, axis=0)
    importance_df = pd.DataFrame({
        'feature': X_train.columns.tolist(),
        'importance': avg_importance
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    cv_rmse = np.sqrt(mean_squared_error(y_train, oof_predictions))
    logger.info(f"CatBoost CV RMSE: {cv_rmse:.4f}")
    
    return oof_predictions, test_predictions, cv_rmse, importance_df


def train_random_forest(X_train, y_train, X_test, kf):
    """Train Random Forest model with OOF predictions and feature importance."""
    logger.info("\n" + "="*50)
    logger.info("Training Random Forest")
    logger.info("="*50)
    
    model_params = {
        'n_estimators': 500,
        'max_depth': 20,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_features': 'sqrt',
        'random_state': RANDOM_SEED,
        'n_jobs': -1
    }
    
    oof_predictions = np.zeros(len(X_train))
    test_predictions = np.zeros(len(X_test))
    feature_importance_list = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        logger.info(f"Fold {fold}/{N_FOLDS}")
        
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        model = RandomForestModel(params=model_params)
        model.fit(X_fold_train, y_fold_train)
        
        # Get feature importance for this fold
        importance_df = get_feature_importance(model, 'randomforest', X_train.columns.tolist())
        feature_importance_list.append(importance_df['importance'].values)
        
        oof_predictions[val_idx] = model.predict(X_fold_val)
        test_predictions += model.predict(X_test) / N_FOLDS
        
        fold_rmse = np.sqrt(mean_squared_error(y_fold_val, oof_predictions[val_idx]))
        logger.info(f"  Fold {fold} RMSE: {fold_rmse:.4f}")
    
    # Average feature importance across folds
    avg_importance = np.mean(feature_importance_list, axis=0)
    importance_df = pd.DataFrame({
        'feature': X_train.columns.tolist(),
        'importance': avg_importance
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    cv_rmse = np.sqrt(mean_squared_error(y_train, oof_predictions))
    logger.info(f"Random Forest CV RMSE: {cv_rmse:.4f}")
    
    return oof_predictions, test_predictions, cv_rmse, importance_df


# ============================================================================
# ENSEMBLE OPTIMIZATION
# ============================================================================

def hill_climbing_ensemble(oof_predictions_dict, y_train, test_predictions_dict, 
                          n_iterations=1000, step_size=0.01):
    """
    Optimize ensemble weights using hill climbing algorithm.
    """
    logger.info("\n" + "="*50)
    logger.info("Hill Climbing Ensemble Optimization")
    logger.info("="*50)
    
    model_names = list(oof_predictions_dict.keys())
    n_models = len(model_names)
    
    # Stack OOF predictions
    oof_stack = np.column_stack([oof_predictions_dict[name] for name in model_names])
    test_stack = np.column_stack([test_predictions_dict[name] for name in model_names])
    
    # Initialize weights uniformly
    best_weights = np.ones(n_models) / n_models
    best_score = np.sqrt(mean_squared_error(y_train, oof_stack @ best_weights))
    
    logger.info(f"Initial uniform weights score: {best_score:.4f}")
    
    # Hill climbing optimization
    for iteration in range(n_iterations):
        improved = False
        
        # Try adjusting each weight
        for i in range(n_models):
            for direction in [-1, 1]:
                # Create new weights
                new_weights = best_weights.copy()
                new_weights[i] += direction * step_size
                
                # Ensure weights are valid (non-negative and sum to 1)
                if new_weights[i] < 0 or new_weights[i] > 1:
                    continue
                
                # Normalize weights to sum to 1
                new_weights = new_weights / new_weights.sum()
                
                # Calculate score
                ensemble_pred = oof_stack @ new_weights
                score = np.sqrt(mean_squared_error(y_train, ensemble_pred))
                
                # Update if better
                if score < best_score:
                    best_score = score
                    best_weights = new_weights
                    improved = True
                    
                    if iteration % 100 == 0:
                        logger.info(f"Iteration {iteration}: New best score {best_score:.4f}")
        
        # Reduce step size if no improvement
        if not improved and iteration % 100 == 0:
            step_size *= 0.9
    
    # Final ensemble predictions
    ensemble_oof = oof_stack @ best_weights
    ensemble_test = test_stack @ best_weights
    
    logger.info("\nOptimized Ensemble Weights:")
    for name, weight in zip(model_names, best_weights):
        logger.info(f"  {name}: {weight:.4f}")
    
    logger.info(f"\nFinal Ensemble CV RMSE: {best_score:.4f}")
    
    return ensemble_oof, ensemble_test, best_weights, best_score


def scipy_optimize_ensemble(oof_predictions_dict, y_train, test_predictions_dict):
    """
    Alternative: Optimize ensemble weights using scipy.optimize.
    """
    logger.info("\n" + "="*50)
    logger.info("Scipy Ensemble Optimization")
    logger.info("="*50)
    
    model_names = list(oof_predictions_dict.keys())
    n_models = len(model_names)
    
    # Stack predictions
    oof_stack = np.column_stack([oof_predictions_dict[name] for name in model_names])
    test_stack = np.column_stack([test_predictions_dict[name] for name in model_names])
    
    # Objective function
    def objective(weights):
        ensemble_pred = oof_stack @ weights
        return np.sqrt(mean_squared_error(y_train, ensemble_pred))
    
    # Constraints: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    
    # Bounds: weights between 0 and 1
    bounds = [(0, 1) for _ in range(n_models)]
    
    # Initial weights
    initial_weights = np.ones(n_models) / n_models
    
    # Optimize
    result = minimize(
        objective, 
        initial_weights, 
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    best_weights = result.x
    best_score = result.fun
    
    # Final predictions
    ensemble_oof = oof_stack @ best_weights
    ensemble_test = test_stack @ best_weights
    
    logger.info("\nOptimized Ensemble Weights (Scipy):")
    for name, weight in zip(model_names, best_weights):
        logger.info(f"  {name}: {weight:.4f}")
    
    logger.info(f"\nFinal Ensemble CV RMSE (Scipy): {best_score:.4f}")
    
    return ensemble_oof, ensemble_test, best_weights, best_score


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function for Kaggle with feature importance analysis."""
    logger.info("="*60)
    logger.info("PS5E9 - MULTI-MODEL ENSEMBLE WITH FEATURE IMPORTANCE")
    logger.info(f"Environment: Kaggle - GPU: {GPU_AVAILABLE}")
    logger.info("="*60)
    
    # Load competition data
    logger.info("\nLoading competition data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    submission = pd.read_csv(DATA_DIR / "sample_submission.csv")
    
    # Remove IDs
    test_ids = test_df[ID_COL].copy()
    train_df = train_df.drop(columns=[ID_COL])
    test_df = test_df.drop(columns=[ID_COL])
    
    logger.info(f"Train shape: {train_df.shape}")
    logger.info(f"Test shape: {test_df.shape}")
    
    # Create features
    logger.info("\n" + "="*60)
    logger.info("FEATURE ENGINEERING")
    logger.info("="*60)
    
    X_train, X_test, y_train = create_features_correct_order(
        train_df, 
        test_df, 
        TARGET_COL
    )
    
    logger.info(f"\nFinal shapes: X_train={X_train.shape}, X_test={X_test.shape}")
    
    # Initialize KFold
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    # Store all predictions and importance
    oof_predictions_dict = {}
    test_predictions_dict = {}
    model_scores = {}
    feature_importance_dict = {}
    
    # Train models
    logger.info("\n" + "="*60)
    logger.info("TRAINING MODELS WITH FEATURE IMPORTANCE")
    logger.info("="*60)
    
    # 1. XGBoost
    oof_xgb, test_xgb, score_xgb, importance_xgb = train_xgboost(X_train, y_train, X_test, kf)
    oof_predictions_dict['XGBoost'] = oof_xgb
    test_predictions_dict['XGBoost'] = test_xgb
    model_scores['XGBoost'] = score_xgb
    feature_importance_dict['XGBoost'] = importance_xgb
    
    # Plot XGBoost importance
    plot_feature_importance(
        importance_xgb, 
        'XGBoost', 
        top_n=20,
        save_path=IMPORTANCE_DIR / 'xgboost_importance.png'
    )
    gc.collect()
    
    # 2. LightGBM
    oof_lgb, test_lgb, score_lgb, importance_lgb = train_lightgbm(X_train, y_train, X_test, kf)
    oof_predictions_dict['LightGBM'] = oof_lgb
    test_predictions_dict['LightGBM'] = test_lgb
    model_scores['LightGBM'] = score_lgb
    feature_importance_dict['LightGBM'] = importance_lgb
    
    # Plot LightGBM importance
    plot_feature_importance(
        importance_lgb, 
        'LightGBM', 
        top_n=20,
        save_path=IMPORTANCE_DIR / 'lightgbm_importance.png'
    )
    gc.collect()
    
    # 3. CatBoost  
    try:
        oof_cat, test_cat, score_cat, importance_cat = train_catboost(X_train, y_train, X_test, kf)
        oof_predictions_dict['CatBoost'] = oof_cat
        test_predictions_dict['CatBoost'] = test_cat
        model_scores['CatBoost'] = score_cat
        feature_importance_dict['CatBoost'] = importance_cat
        
        # Plot CatBoost importance
        plot_feature_importance(
            importance_cat, 
            'CatBoost', 
            top_n=20,
            save_path=IMPORTANCE_DIR / 'catboost_importance.png'
        )
        gc.collect()
    except Exception as e:
        logger.warning(f"CatBoost training failed: {e}")
        logger.warning("Skipping CatBoost and continuing with other models...")
    
    # 4. Random Forest
    oof_rf, test_rf, score_rf, importance_rf = train_random_forest(X_train, y_train, X_test, kf)
    oof_predictions_dict['RandomForest'] = oof_rf
    test_predictions_dict['RandomForest'] = test_rf
    model_scores['RandomForest'] = score_rf
    feature_importance_dict['RandomForest'] = importance_rf
    
    # Plot Random Forest importance
    plot_feature_importance(
        importance_rf, 
        'RandomForest', 
        top_n=20,
        save_path=IMPORTANCE_DIR / 'randomforest_importance.png'
    )
    gc.collect()
    
    # Summary of individual models
    logger.info("\n" + "="*60)
    logger.info("INDIVIDUAL MODEL PERFORMANCE")
    logger.info("="*60)
    for model_name, score in model_scores.items():
        logger.info(f"{model_name:15s}: CV RMSE = {score:.4f}")
    
    # Feature importance analysis
    logger.info("\n" + "="*60)
    logger.info("FEATURE IMPORTANCE ANALYSIS")
    logger.info("="*60)
    
    # Display importance statistics
    display_importance_stats(feature_importance_dict, top_n=10)
    
    # Create combined importance plot
    combined_importance = plot_combined_feature_importance(
        feature_importance_dict,
        top_n=20,
        save_path=IMPORTANCE_DIR / 'combined_importance.png'
    )
    
    # Save importance to CSV
    combined_importance.to_csv(
        IMPORTANCE_DIR / 'feature_importance_all_models.csv',
        index=False
    )
    logger.info(f"Saved feature importance to {IMPORTANCE_DIR / 'feature_importance_all_models.csv'}")
    
    # Save individual model importance to CSV
    for model_name, importance_df in feature_importance_dict.items():
        filename = IMPORTANCE_DIR / f'{model_name.lower()}_importance.csv'
        importance_df.to_csv(filename, index=False)
        logger.info(f"Saved {model_name} importance to {filename}")
    
    # Ensemble optimization
    logger.info("\n" + "="*60)
    logger.info("ENSEMBLE OPTIMIZATION")
    logger.info("="*60)
    
    # Use only competition samples for ensemble optimization
    competition_samples = min(524164, len(X_train))
    y_train_comp = y_train[:competition_samples]
    
    oof_predictions_comp = {
        name: preds[:competition_samples] 
        for name, preds in oof_predictions_dict.items()
    }
    
    # Try both optimization methods
    # 1. Hill Climbing
    ensemble_oof_hc, ensemble_test_hc, weights_hc, score_hc = hill_climbing_ensemble(
        oof_predictions_comp, 
        y_train_comp, 
        test_predictions_dict,
        n_iterations=2000,
        step_size=0.01
    )
    
    # 2. Scipy Optimize
    ensemble_oof_scipy, ensemble_test_scipy, weights_scipy, score_scipy = scipy_optimize_ensemble(
        oof_predictions_comp,
        y_train_comp,
        test_predictions_dict
    )
    
    # Choose best ensemble
    if score_hc < score_scipy:
        logger.info("\n✓ Using Hill Climbing ensemble (better score)")
        ensemble_oof = ensemble_oof_hc
        ensemble_test = ensemble_test_hc
        best_weights = weights_hc
        best_score = score_hc
        method = "hill_climbing"
    else:
        logger.info("\n✓ Using Scipy optimized ensemble (better score)")
        ensemble_oof = ensemble_oof_scipy
        ensemble_test = ensemble_test_scipy
        best_weights = weights_scipy
        best_score = score_scipy
        method = "scipy"
    
    # Save OOF predictions
    logger.info(f"\nSaving OOF predictions...")
    oof_manager = OOFManager(output_dir=str(OOF_DIR))
    
    # Save individual model OOFs
    for model_name, oof_preds in oof_predictions_comp.items():
        oof_df = pd.DataFrame({
            'predictions': oof_preds,
            'target': y_train_comp
        })
        oof_manager.save_oof(
            predictions=oof_df,
            test_predictions=test_predictions_dict[model_name],
            model_name=f"{model_name.lower()}_ensemble",
            cv_score=model_scores[model_name],
            model_params={'model_type': model_name, 'n_folds': N_FOLDS}
        )
    
    # Save ensemble OOF
    ensemble_oof_df = pd.DataFrame({
        'predictions': ensemble_oof,
        'target': y_train_comp
    })
    oof_manager.save_oof(
        predictions=ensemble_oof_df,
        test_predictions=ensemble_test,
        model_name=f"ensemble_{method}",
        cv_score=best_score,
        model_params={
            'model_type': f'Ensemble_{method}',
            'weights': {name: float(w) for name, w in zip(model_scores.keys(), best_weights)},
            'n_models': len(model_scores),
            'optimization_method': method
        }
    )
    
    # Create submission
    logger.info("\nCreating submission...")
    submission[TARGET_COL] = ensemble_test
    submission_file = SUBMISSION_DIR / f"submission_ensemble_cv{best_score:.4f}.csv"
    submission.to_csv(submission_file, index=False)
    logger.info(f"Submission saved to {submission_file}")
    
    # Display results
    logger.info("\nFirst 10 predictions:")
    print(submission.head(10))
    
    # Create a summary visualization of ensemble weights vs importance
    logger.info("\n" + "="*60)
    logger.info("ENSEMBLE WEIGHTS VS FEATURE IMPORTANCE")
    logger.info("="*60)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(model_scores.keys())
    x_pos = np.arange(len(models))
    
    # Plot ensemble weights
    ax.bar(x_pos - 0.2, best_weights, 0.4, label='Ensemble Weights', color='blue', alpha=0.7)
    
    # Plot average feature importance (normalized)
    avg_importances = []
    for model in models:
        if model in feature_importance_dict:
            # Get top 10 features average importance
            top_10_importance = feature_importance_dict[model].head(10)['importance'].mean()
            total_importance = feature_importance_dict[model]['importance'].sum()
            avg_importances.append(top_10_importance / total_importance)
        else:
            avg_importances.append(0)
    
    ax.bar(x_pos + 0.2, avg_importances, 0.4, label='Avg Top-10 Feature Importance (Normalized)', color='green', alpha=0.7)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Weight / Importance')
    ax.set_title('Ensemble Weights vs Feature Importance Contribution')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(IMPORTANCE_DIR / 'weights_vs_importance.png', dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()
    
    logger.info("\n" + "="*60)
    logger.info("✅ ENSEMBLE TRAINING WITH FEATURE IMPORTANCE COMPLETE!")
    logger.info("="*60)
    logger.info(f"   Best Ensemble CV RMSE: {best_score:.4f}")
    logger.info(f"   Optimization Method: {method}")
    logger.info(f"   Target: ~26.38")
    logger.info(f"   Submission: {submission_file}")
    logger.info(f"   Feature Importance: {IMPORTANCE_DIR}")
    
    logger.info("\nFinal Ensemble Weights:")
    for name, weight in zip(model_scores.keys(), best_weights):
        logger.info(f"  {name:15s}: {weight:.4f}")
    
    return best_score, submission


if __name__ == "__main__":
    try:
        cv_score, submission_df = main()
        print(f"\n✅ Final Ensemble CV RMSE: {cv_score:.4f}")
        print(f"Submission shape: {submission_df.shape}")
        print(f"Feature importance analysis saved to: {IMPORTANCE_DIR}")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise