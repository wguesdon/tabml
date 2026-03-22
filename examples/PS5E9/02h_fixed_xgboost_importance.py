"""
PS5E9 Competition - Fixed XGBoost Feature Importance
Fixes the empty feature importance issue for XGBoost models that use xgboost.train()
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

# Import packages
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# Add tabml to path if running locally
if 'kaggle' not in os.getcwd():
    sys.path.insert(0, str(Path('../../').resolve()))
    DATA_DIR = Path("../../data/PS5E9")
    OUTPUT_DIR = Path("./output")
else:
    # Kaggle paths
    DATA_DIR = Path("/kaggle/input/playground-series-s5e9")
    OUTPUT_DIR = Path("/kaggle/working")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMPORTANCE_DIR = OUTPUT_DIR / "feature_importance"
IMPORTANCE_DIR.mkdir(parents=True, exist_ok=True)

# Import TabML
try:
    from tabml import XGBoostModel
    print("TabML imported successfully")
except ImportError:
    print("Installing TabML...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/wguesdon/tabml.git"])
    from tabml import XGBoostModel

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
# FIXED FEATURE IMPORTANCE EXTRACTION
# ============================================================================

def get_xgboost_feature_importance_fixed(model, feature_names, importance_type='gain'):
    """
    Fixed function to extract feature importance from XGBoost model.
    Properly handles both Booster objects and sklearn-like models.
    
    Args:
        model: XGBoostModel instance from TabML or raw XGBoost model
        feature_names: List of feature names
        importance_type: Type of importance ('gain', 'weight', 'cover', 'total_gain', 'total_cover')
    
    Returns:
        DataFrame with features and importance scores
    """
    
    # Get the underlying model
    if hasattr(model, 'model'):
        xgb_model = model.model
        logger.debug(f"Got model from TabML wrapper: {type(xgb_model)}")
    else:
        xgb_model = model
        logger.debug(f"Using model directly: {type(xgb_model)}")
    
    # Initialize importance array
    importance = np.zeros(len(feature_names))
    logger.debug(f"Initialized importance array for {len(feature_names)} features")
    
    # Check if it's a Booster object (from xgboost.train)
    if isinstance(xgb_model, xgb.Booster):
        # Get importance scores from Booster
        try:
            importance_dict = xgb_model.get_score(importance_type=importance_type)
            
            if not importance_dict:
                logger.warning(f"get_score returned empty dict - trying without importance_type")
                importance_dict = xgb_model.get_score()
            
            logger.info(f"Raw importance dict has {len(importance_dict)} entries")
            if importance_dict and len(importance_dict) > 0:
                # Show first few entries for debugging
                sample_keys = list(importance_dict.keys())[:5]
                logger.debug(f"Sample keys from importance dict: {sample_keys}")
            
            # Map feature names - XGBoost can use either internal (f0, f1, ...) or actual names
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
                    
            logger.info(f"Extracted {len(importance_dict)} non-zero importances from Booster, matched {matched_count} features")
            
        except Exception as e:
            logger.error(f"Error extracting importance from Booster: {e}")
            # Try alternative method
            try:
                importance_dict = xgb_model.get_score()
                logger.info(f"Alternative method: got {len(importance_dict)} importances")
                for i, feat_name in enumerate(feature_names):
                    # First try the actual feature name
                    if feat_name in importance_dict:
                        importance[i] = importance_dict[feat_name]
                    else:
                        # Fallback to internal name (f0, f1, ...)
                        internal_name = f'f{i}'
                        if internal_name in importance_dict:
                            importance[i] = importance_dict[internal_name]
            except Exception as e2:
                logger.error(f"Alternative method also failed: {e2}")
                raise ValueError(f"Cannot extract feature importance: {e}")
    
    # Check if it's a sklearn-like model with feature_importances_
    elif hasattr(xgb_model, 'feature_importances_'):
        importance = xgb_model.feature_importances_
        logger.debug("Extracted importance from sklearn-like model")
    
    # Try to get booster from sklearn model
    elif hasattr(xgb_model, 'get_booster'):
        try:
            booster = xgb_model.get_booster()
            importance_dict = booster.get_score(importance_type=importance_type)
            
            for i, feat_name in enumerate(feature_names):
                # First try the actual feature name
                if feat_name in importance_dict:
                    importance[i] = importance_dict[feat_name]
                else:
                    # Fallback to internal name (f0, f1, ...)
                    internal_name = f'f{i}'
                    if internal_name in importance_dict:
                        importance[i] = importance_dict[internal_name]
                    
            logger.debug("Extracted importance from sklearn model's booster")
        except Exception as e:
            raise ValueError(f"Cannot extract feature importance from sklearn model: {e}")
    
    else:
        raise ValueError("Unknown XGBoost model type - cannot extract feature importance")
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    
    # Filter out zero importance features
    importance_df = importance_df[importance_df['importance'] > 0]
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    # Log summary
    logger.info(f"Feature importance extracted: {len(importance_df)} features with non-zero importance")
    if len(importance_df) > 0:
        logger.info(f"Top feature: {importance_df.iloc[0]['feature']} (importance: {importance_df.iloc[0]['importance']:.2f})")
    
    return importance_df


# ============================================================================
# DATA LOADING AND FEATURE ENGINEERING
# ============================================================================

def create_features(train_df, test_df, target_col):
    """
    Create features with proper transformations.
    """
    logger.info("Creating features...")
    
    train_features = train_df.drop(columns=[target_col])
    df_combined = pd.concat([train_features, test_df], axis=0, ignore_index=True)
    
    base_cols = ['RhythmScore', 'AudioLoudness', 'VocalContent', 
                 'AcousticQuality', 'InstrumentalScore', 
                 'LivePerformanceLikelihood', 'MoodScore', 
                 'TrackDurationMs', 'Energy']
    
    # Adjust AudioLoudness to positive values
    min_loudness = df_combined["AudioLoudness"].min()
    df_combined["AudioLoudness"] = df_combined["AudioLoudness"] - min_loudness + 1
    
    # Apply log transformation
    for col in base_cols:
        df_combined[col] = np.log1p(df_combined[col])
    
    # Create 2-way interactions (limited for demo)
    logger.info("Creating interaction features...")
    interaction_count = 0
    for col1, col2 in combinations(base_cols, 2):
        if interaction_count >= 20:  # Limit interactions for faster demo
            break
        df_combined[f"{col1}_x_{col2}"] = df_combined[col1] * df_combined[col2]
        df_combined[f"{col1}_plus_{col2}"] = df_combined[col1] + df_combined[col2]
        interaction_count += 1
    
    logger.info(f"Total features created: {df_combined.shape[1]}")
    
    # Split back
    X_train = df_combined.iloc[:len(train_df)].copy()
    X_test = df_combined.iloc[len(train_df):].copy()
    y_train = train_df[target_col].copy()
    
    # Convert to float32 for memory efficiency
    for col in X_train.columns:
        if X_train[col].dtype == 'float64':
            X_train[col] = X_train[col].astype('float32')
            X_test[col] = X_test[col].astype('float32')
    
    return X_train, X_test, y_train


# ============================================================================
# TRAINING WITH FIXED FEATURE IMPORTANCE
# ============================================================================

def train_xgboost_with_importance(X_train, y_train, X_test, kf):
    """
    Train XGBoost model with proper feature importance extraction.
    """
    logger.info("\n" + "="*50)
    logger.info("Training XGBoost with Fixed Feature Importance")
    logger.info("="*50)
    
    model_params = {
        'objective': 'reg:squarederror',
        'n_estimators': 2000,  # Reduced for demo
        'learning_rate': 0.05,
        'max_depth': 6,
        'colsample_bytree': 0.7,
        'subsample': 0.8,
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
    fold_importances = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        logger.info(f"\nFold {fold}/{N_FOLDS}")
        
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
            early_stopping_rounds=100
        )
        
        # Get feature importance using fixed function
        importance_df = get_xgboost_feature_importance_fixed(
            model, 
            X_train.columns.tolist(),
            importance_type='gain'
        )
        
        # Store importance
        fold_importances.append(importance_df)
        
        # Display top 5 features for this fold
        logger.info(f"Top 5 features for fold {fold}:")
        for idx, row in importance_df.head(5).iterrows():
            logger.info(f"  {row['feature']:30s}: {row['importance']:10.2f}")
        
        # Generate predictions
        oof_predictions[val_idx] = model.predict(X_fold_val)
        test_predictions += model.predict(X_test) / N_FOLDS
        
        # Calculate fold score
        fold_rmse = np.sqrt(mean_squared_error(y_fold_val, oof_predictions[val_idx]))
        logger.info(f"Fold {fold} RMSE: {fold_rmse:.4f}")
    
    # Aggregate feature importance across folds
    all_features = set()
    for imp_df in fold_importances:
        all_features.update(imp_df['feature'].tolist())
    
    avg_importance = {}
    for feature in all_features:
        importances = []
        for imp_df in fold_importances:
            if feature in imp_df['feature'].values:
                imp_value = imp_df[imp_df['feature'] == feature]['importance'].values[0]
                importances.append(imp_value)
            else:
                importances.append(0)
        avg_importance[feature] = np.mean(importances)
    
    # Create final importance DataFrame
    if avg_importance:
        final_importance = pd.DataFrame([
            {'feature': feat, 'importance': imp} 
            for feat, imp in avg_importance.items()
        ])
        # Only sort if DataFrame is not empty
        if not final_importance.empty:
            final_importance = final_importance.sort_values('importance', ascending=False).reset_index(drop=True)
    else:
        # If no importance found, create empty DataFrame with correct columns
        logger.warning("No feature importance found - creating empty DataFrame")
        final_importance = pd.DataFrame(columns=['feature', 'importance'])
    
    cv_rmse = np.sqrt(mean_squared_error(y_train, oof_predictions))
    logger.info(f"\nXGBoost CV RMSE: {cv_rmse:.4f}")
    
    return oof_predictions, test_predictions, cv_rmse, final_importance


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_feature_importance_fixed(importance_df, model_name='XGBoost', top_n=20, save_path=None):
    """
    Create comprehensive feature importance visualizations.
    """
    # Check if DataFrame is empty
    if importance_df.empty:
        logger.warning("Cannot plot - importance DataFrame is empty")
        return None
    
    # Select top N features
    top_features = importance_df.head(top_n)
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Horizontal bar plot
    ax1 = axes[0, 0]
    ax1.barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features['feature'].values, fontsize=8)
    ax1.set_xlabel('Feature Importance (Gain)')
    ax1.set_title(f'{model_name} - Top {top_n} Features')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative importance
    ax2 = axes[0, 1]
    cumulative_importance = np.cumsum(importance_df['importance'].values)
    cumulative_importance = cumulative_importance / cumulative_importance[-1] * 100
    
    ax2.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 'b-', linewidth=2)
    ax2.axhline(y=80, color='r', linestyle='--', label='80% importance')
    ax2.axhline(y=90, color='orange', linestyle='--', label='90% importance')
    
    n_80 = np.argmax(cumulative_importance >= 80) + 1
    n_90 = np.argmax(cumulative_importance >= 90) + 1
    
    ax2.axvline(x=n_80, color='r', linestyle=':', alpha=0.5)
    ax2.axvline(x=n_90, color='orange', linestyle=':', alpha=0.5)
    
    ax2.set_xlabel('Number of Features')
    ax2.set_ylabel('Cumulative Importance (%)')
    ax2.set_title(f'Cumulative Feature Importance\n({n_80} features for 80%, {n_90} features for 90%)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Feature importance distribution
    ax3 = axes[1, 0]
    ax3.hist(importance_df['importance'].values, bins=30, color='green', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Importance Value')
    ax3.set_ylabel('Number of Features')
    ax3.set_title('Distribution of Feature Importance')
    ax3.grid(True, alpha=0.3)
    
    # Add statistics
    mean_imp = importance_df['importance'].mean()
    median_imp = importance_df['importance'].median()
    ax3.axvline(x=mean_imp, color='red', linestyle='--', label=f'Mean: {mean_imp:.2f}')
    ax3.axvline(x=median_imp, color='blue', linestyle='--', label=f'Median: {median_imp:.2f}')
    ax3.legend()
    
    # 4. Top features as percentage
    ax4 = axes[1, 1]
    top_10 = importance_df.head(10)
    total_importance = importance_df['importance'].sum()
    percentages = (top_10['importance'] / total_importance * 100).values
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_10)))
    bars = ax4.bar(range(len(top_10)), percentages, color=colors)
    ax4.set_xticks(range(len(top_10)))
    ax4.set_xticklabels([f"F{i+1}" for i in range(len(top_10))])
    ax4.set_ylabel('% of Total Importance')
    ax4.set_title('Top 10 Features as % of Total Importance')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle(f'{model_name} Feature Importance Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {save_path}")
    
    plt.show()
    plt.close()
    
    return fig


def verify_importance_extraction():
    """
    Verify that feature importance extraction works correctly.
    """
    logger.info("\n" + "="*50)
    logger.info("VERIFYING FEATURE IMPORTANCE EXTRACTION")
    logger.info("="*50)
    
    # Create small test dataset
    X_test = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100),
    })
    y_test = X_test['feature1'] * 2 + X_test['feature2'] * 0.5 + np.random.randn(100) * 0.1
    
    # Train model
    model = XGBoostModel(params={
        'objective': 'reg:squarederror',
        'n_estimators': 100,
        'max_depth': 3,
        'random_state': 42
    })
    model.fit(X_test, y_test)
    
    # Extract importance
    importance_df = get_xgboost_feature_importance_fixed(model, X_test.columns.tolist())
    
    logger.info("\nTest Feature Importance:")
    for idx, row in importance_df.iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    if len(importance_df) > 0:
        logger.info("✓ Feature importance extraction verified successfully!")
    else:
        logger.warning("⚠ No feature importance extracted - check implementation")
    
    return importance_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function with fixed feature importance."""
    logger.info("="*60)
    logger.info("PS5E9 - XGBOOST WITH FIXED FEATURE IMPORTANCE")
    logger.info(f"Environment: {'Kaggle' if 'kaggle' in os.getcwd() else 'Local'}")
    logger.info(f"GPU Available: {GPU_AVAILABLE}")
    logger.info("="*60)
    
    # First verify importance extraction works
    verify_importance_extraction()
    
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
    X_train, X_test, y_train = create_features(train_df, test_df, TARGET_COL)
    logger.info(f"Final shapes: X_train={X_train.shape}, X_test={X_test.shape}")
    
    # Initialize KFold
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    # Train XGBoost with fixed feature importance
    oof_predictions, test_predictions, cv_score, feature_importance = train_xgboost_with_importance(
        X_train, y_train, X_test, kf
    )
    
    # Display top features
    logger.info("\n" + "="*50)
    logger.info("TOP 20 MOST IMPORTANT FEATURES")
    logger.info("="*50)
    
    if not feature_importance.empty:
        for idx, row in feature_importance.head(20).iterrows():
            logger.info(f"{idx+1:2d}. {row['feature']:40s}: {row['importance']:10.2f}")
    else:
        logger.warning("No feature importance found - check model training")
    
    # Create visualizations
    plot_feature_importance_fixed(
        feature_importance,
        model_name='XGBoost (Fixed)',
        top_n=20,
        save_path=IMPORTANCE_DIR / 'xgboost_importance_fixed.png'
    )
    
    # Save feature importance to CSV
    importance_file = IMPORTANCE_DIR / 'xgboost_feature_importance_fixed.csv'
    feature_importance.to_csv(importance_file, index=False)
    logger.info(f"\nFeature importance saved to: {importance_file}")
    
    # Statistics
    logger.info("\n" + "="*50)
    logger.info("FEATURE IMPORTANCE STATISTICS")
    logger.info("="*50)
    
    if not feature_importance.empty and 'importance' in feature_importance.columns:
        logger.info(f"Total features: {len(feature_importance)}")
        logger.info(f"Features with non-zero importance: {len(feature_importance[feature_importance['importance'] > 0])}")
        logger.info(f"Max importance: {feature_importance['importance'].max():.2f}")
        logger.info(f"Mean importance: {feature_importance['importance'].mean():.2f}")
        logger.info(f"Median importance: {feature_importance['importance'].median():.2f}")
        
        # Calculate how many features account for 80% of importance
        if feature_importance['importance'].sum() > 0:
            cumsum = np.cumsum(feature_importance['importance'].values)
            cumsum_pct = cumsum / cumsum[-1] * 100
            n_80 = np.argmax(cumsum_pct >= 80) + 1
            n_90 = np.argmax(cumsum_pct >= 90) + 1
            
            logger.info(f"Features for 80% importance: {n_80}")
            logger.info(f"Features for 90% importance: {n_90}")
    else:
        logger.warning("No feature importance statistics available - DataFrame is empty")
    
    # Create submission
    submission[TARGET_COL] = test_predictions
    submission_file = OUTPUT_DIR / f"submission_xgboost_fixed_cv{cv_score:.4f}.csv"
    submission.to_csv(submission_file, index=False)
    logger.info(f"\nSubmission saved to: {submission_file}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ XGBOOST TRAINING WITH FIXED FEATURE IMPORTANCE COMPLETE!")
    logger.info("="*60)
    logger.info(f"CV RMSE: {cv_score:.4f}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    return cv_score, feature_importance


if __name__ == "__main__":
    try:
        cv_score, importance_df = main()
        print(f"\n✅ Successfully completed!")
        print(f"Final CV RMSE: {cv_score:.4f}")
        print(f"Non-zero importance features: {len(importance_df[importance_df['importance'] > 0])}")
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        raise