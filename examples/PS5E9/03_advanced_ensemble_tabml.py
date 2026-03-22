"""
PS5E9 Competition - Advanced Ensemble with TabML
Incorporates techniques from top public notebooks while using TabML framework:
- Yeo-Johnson target transformation
- Extensive feature engineering (squares, ratios, quantile bins)
- Isotonic calibration
- Two-stage weight optimization
- Adaptive clipping
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from itertools import combinations

warnings.filterwarnings('ignore')

# Add TabML to path if running locally
if 'kaggle' not in os.getcwd():
    sys.path.insert(0, str(Path('../../').resolve()))

# Import TabML components
from tabml import (
    XGBoostModel, LightGBMModel, CatBoostModel,
    RandomForestModel, OOFManager
)

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PowerTransformer
from sklearn.isotonic import IsotonicRegression
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
if 'kaggle' in os.getcwd():
    DATA_DIR = Path("/kaggle/input/playground-series-s5e9")
    ORIGINAL_DATA_DIR = Path("/kaggle/input/bpm-prediction-challenge")
    OUTPUT_DIR = Path("/kaggle/working")
else:
    DATA_DIR = Path("../../data/raw/PS5E9")
    ORIGINAL_DATA_DIR = DATA_DIR  # Adjust as needed
    OUTPUT_DIR = Path("./output")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
CV_SEEDS = [42]  # Can add more seeds for bagging

BASE_NUM_COLS = [
    'RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
    'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
    'TrackDurationMs', 'Energy'
]

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
# UTILITIES
# ============================================================================

def rmse(y_true, y_pred):
    """Calculate RMSE."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


@dataclass
class CVResult:
    """Store cross-validation results."""
    oof: np.ndarray
    test_pred: np.ndarray
    fold_metrics: List[float]
    best_iters: List[int]
    feature_importance: pd.DataFrame = None


# ============================================================================
# ADVANCED FEATURE ENGINEERING
# ============================================================================

def compute_bin_edges(train_col: pd.Series, q: List[float]) -> np.ndarray:
    """Compute quantile bin edges from training data only."""
    qs = np.unique(np.clip(q, 0.0, 1.0))
    edges = np.quantile(train_col.values, qs)
    edges = np.unique(edges)
    
    # Ensure strictly increasing edges
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = edges[i-1] + 1e-9
    
    return np.concatenate(([-np.inf], edges, [np.inf]))


def apply_bins(col: pd.Series, edges: np.ndarray) -> np.ndarray:
    """Apply binning using pre-computed edges."""
    return np.digitize(col.values, edges) - 1


def build_advanced_features(train: pd.DataFrame, test: pd.DataFrame, 
                           base_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Create advanced engineered features using train-only statistics.
    - Squares and pairwise products
    - Safe ratios (both directions)
    - Quantile-based bins (quartile & decile)
    - Log transformations for selected features
    """
    tr = train.copy()
    te = test.copy()
    
    logger.info("Building advanced features...")
    
    # Cast to float32 for memory efficiency
    for c in base_cols:
        tr[c] = tr[c].astype(np.float32)
        te[c] = te[c].astype(np.float32)
    
    # 1. Squares and pairwise products
    logger.info("  Creating polynomial features...")
    for i, c1 in enumerate(base_cols):
        v1_tr = tr[c1]
        v1_te = te[c1]
        
        # Squares
        tr[f"{c1}_sq"] = v1_tr * v1_tr
        te[f"{c1}_sq"] = v1_te * v1_te
        
        # Pairwise products
        for j in range(i+1, len(base_cols)):
            c2 = base_cols[j]
            tr[f"{c1}_x_{c2}"] = v1_tr * tr[c2]
            te[f"{c1}_x_{c2}"] = v1_te * te[c2]
    
    # 2. Safe ratios (both directions)
    logger.info("  Creating ratio features...")
    eps = 1e-6
    for i, c1 in enumerate(base_cols):
        for j, c2 in enumerate(base_cols):
            if i == j:
                continue
            tr[f"{c1}_div_{c2}"] = tr[c1] / (tr[c2].abs() + eps)
            te[f"{c1}_div_{c2}"] = te[c1] / (te[c2].abs() + eps)
    
    # 3. Log transformations for skewed features
    logger.info("  Creating log features...")
    skewed_cols = ['AudioLoudness', 'TrackDurationMs']  # Often skewed
    for c in skewed_cols:
        if c in base_cols:
            # Shift to positive if needed
            min_val = min(tr[c].min(), te[c].min())
            if min_val <= 0:
                shift = abs(min_val) + 1
                tr[f"{c}_log"] = np.log1p(tr[c] + shift)
                te[f"{c}_log"] = np.log1p(te[c] + shift)
            else:
                tr[f"{c}_log"] = np.log1p(tr[c])
                te[f"{c}_log"] = np.log1p(te[c])
    
    # 4. Quantile bins from train only
    logger.info("  Creating quantile bins...")
    quart_q = [0.25, 0.5, 0.75]
    dec_q = [i/10 for i in range(1, 10)]
    
    for c in base_cols:
        # Quartile bins
        edges4 = compute_bin_edges(tr[c], quart_q)
        tr[f"{c}_quartile"] = apply_bins(tr[c], edges4).astype(np.int8)
        te[f"{c}_quartile"] = apply_bins(te[c], edges4).astype(np.int8)
        
        # Decile bins
        edges10 = compute_bin_edges(tr[c], dec_q)
        tr[f"{c}_decile"] = apply_bins(tr[c], edges10).astype(np.int8)
        te[f"{c}_decile"] = apply_bins(te[c], edges10).astype(np.int8)
    
    # 5. Three-way interactions for most important features
    logger.info("  Creating three-way interactions...")
    important_cols = ['Energy', 'RhythmScore', 'MoodScore']  # Domain knowledge
    for combo in combinations(important_cols, 3):
        c1, c2, c3 = combo
        tr[f"{c1}_x_{c2}_x_{c3}"] = tr[c1] * tr[c2] * tr[c3]
        te[f"{c1}_x_{c2}_x_{c3}"] = te[c1] * te[c2] * te[c3]
    
    # Final feature list: all except ID/TARGET
    feat_cols = [c for c in tr.columns if c not in [ID_COL, TARGET_COL]]
    
    logger.info(f"  Total features created: {len(feat_cols)}")
    return tr, te, feat_cols


# ============================================================================
# MODEL TRAINING WITH TABML
# ============================================================================

def train_lgbm_tabml(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame,
                     kf: KFold, use_transform: bool = True) -> CVResult:
    """Train LightGBM using TabML with advanced settings."""
    logger.info("\n" + "="*50)
    logger.info("Training LightGBM with TabML")
    logger.info("="*50)
    
    # Target transformation
    if use_transform:
        pt = PowerTransformer(method="yeo-johnson")
        y_transformed = pt.fit_transform(y.values.reshape(-1, 1)).ravel()
        y_fit = pd.Series(y_transformed, index=y.index)
        
        # Store transform bounds for clipping
        y_min, y_max = y.min(), y.max()
        y_median = y.median()
        transform_min = y_transformed.min()
        transform_max = y_transformed.max()
        
        # Safe inverse transform with NaN handling
        def inv_transform(arr):
            # First clip predictions to reasonable range in transformed space
            arr_clipped = np.clip(arr, transform_min - 2, transform_max + 2)
            # Then inverse transform
            result = pt.inverse_transform(arr_clipped.reshape(-1, 1)).ravel()
            # Final safety check: handle any remaining NaN/inf
            result = np.nan_to_num(result, nan=y_median, posinf=y_max * 1.5, neginf=y_min * 0.5)
            # Clip to reasonable range based on training data
            result = np.clip(result, y_min * 0.8, y_max * 1.2)
            return result
    else:
        y_fit = y.copy()
        inv_transform = lambda arr: arr
    
    model_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'n_estimators': 6000,
        'learning_rate': 0.02,
        'num_leaves': 127,
        'max_depth': -1,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'reg_lambda': 1.0,
        'min_child_samples': 25,
        'random_state': RANDOM_SEED,
        'n_jobs': -1,
        'verbosity': -1
    }
    
    if GPU_AVAILABLE:
        model_params['device'] = 'gpu'
    
    oof_sum = np.zeros(len(X), dtype=float)
    oof_cnt = np.zeros(len(X), dtype=float)
    test_pred = np.zeros(len(X_test), dtype=float)
    fold_metrics = []
    best_iters = []
    feature_importance_list = []
    
    for cv_seed in CV_SEEDS:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=cv_seed)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
            logger.info(f"[LGBM] Seed {cv_seed} | Fold {fold}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_fit.iloc[train_idx], y_fit.iloc[val_idx]
            
            model = LightGBMModel(params=model_params)
            model.fit(
                X_train, y_train,
                X_val=X_val, y_val=y_val,
                early_stopping_rounds=300
            )
            
            # Get predictions
            val_pred = model.predict(X_val)
            oof_sum[val_idx] += inv_transform(val_pred)
            oof_cnt[val_idx] += 1
            
            # Store feature importance
            if hasattr(model.model, 'feature_importances_'):
                importance = model.model.feature_importances_
                feature_importance_list.append(importance)
            
            # Calculate metrics on original scale
            val_pred_orig = inv_transform(val_pred)
            fold_rmse = rmse(y.iloc[val_idx], val_pred_orig)
            fold_metrics.append(fold_rmse)
            logger.info(f"  Fold {fold} RMSE: {fold_rmse:.4f}")
            
            # Test predictions
            test_pred += inv_transform(model.predict(X_test)) / (N_FOLDS * len(CV_SEEDS))
    
    # Average OOF predictions
    oof = oof_sum / np.maximum(oof_cnt, 1)
    
    # Average feature importance
    importance_df = None
    if feature_importance_list:
        avg_importance = np.mean(feature_importance_list, axis=0)
        importance_df = pd.DataFrame({
            'feature': X.columns.tolist(),
            'importance': avg_importance
        }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    logger.info(f"[LGBM] CV RMSE: {rmse(y, oof):.4f}")
    
    return CVResult(oof, test_pred, fold_metrics, best_iters, importance_df)


def train_xgb_tabml(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame,
                    kf: KFold, use_transform: bool = True) -> CVResult:
    """Train XGBoost using TabML with advanced settings."""
    logger.info("\n" + "="*50)
    logger.info("Training XGBoost with TabML")
    logger.info("="*50)
    
    # Target transformation
    if use_transform:
        pt = PowerTransformer(method="yeo-johnson")
        y_transformed = pt.fit_transform(y.values.reshape(-1, 1)).ravel()
        y_fit = pd.Series(y_transformed, index=y.index)
        
        # Store transform bounds for clipping
        y_min, y_max = y.min(), y.max()
        y_median = y.median()
        transform_min = y_transformed.min()
        transform_max = y_transformed.max()
        
        # Safe inverse transform with NaN handling
        def inv_transform(arr):
            # First clip predictions to reasonable range in transformed space
            arr_clipped = np.clip(arr, transform_min - 2, transform_max + 2)
            # Then inverse transform
            result = pt.inverse_transform(arr_clipped.reshape(-1, 1)).ravel()
            # Final safety check: handle any remaining NaN/inf
            result = np.nan_to_num(result, nan=y_median, posinf=y_max * 1.5, neginf=y_min * 0.5)
            # Clip to reasonable range based on training data
            result = np.clip(result, y_min * 0.8, y_max * 1.2)
            return result
    else:
        y_fit = y.copy()
        inv_transform = lambda arr: arr
    
    model_params = {
        'objective': 'reg:squarederror',
        'n_estimators': 7000,
        'learning_rate': 0.02,
        'max_depth': 8,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'reg_lambda': 1.0,
        'reg_alpha': 0.0,
        'min_child_weight': 1.0,
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
    
    oof_sum = np.zeros(len(X), dtype=float)
    oof_cnt = np.zeros(len(X), dtype=float)
    test_pred = np.zeros(len(X_test), dtype=float)
    fold_metrics = []
    best_iters = []
    
    for cv_seed in CV_SEEDS:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=cv_seed)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
            logger.info(f"[XGB] Seed {cv_seed} | Fold {fold}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_fit.iloc[train_idx], y_fit.iloc[val_idx]
            
            model = XGBoostModel(params=model_params)
            model.fit(
                X_train, y_train,
                X_val=X_val, y_val=y_val,
                early_stopping_rounds=300
            )
            
            # Get predictions
            val_pred = model.predict(X_val)
            oof_sum[val_idx] += inv_transform(val_pred)
            oof_cnt[val_idx] += 1
            
            # Calculate metrics on original scale
            val_pred_orig = inv_transform(val_pred)
            fold_rmse = rmse(y.iloc[val_idx], val_pred_orig)
            fold_metrics.append(fold_rmse)
            logger.info(f"  Fold {fold} RMSE: {fold_rmse:.4f}")
            
            # Test predictions
            test_pred += inv_transform(model.predict(X_test)) / (N_FOLDS * len(CV_SEEDS))
    
    # Average OOF predictions
    oof = oof_sum / np.maximum(oof_cnt, 1)
    logger.info(f"[XGB] CV RMSE: {rmse(y, oof):.4f}")
    
    return CVResult(oof, test_pred, fold_metrics, best_iters)


def train_cat_tabml(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame,
                    kf: KFold, use_transform: bool = True) -> CVResult:
    """Train CatBoost using TabML with advanced settings."""
    logger.info("\n" + "="*50)
    logger.info("Training CatBoost with TabML")
    logger.info("="*50)
    
    # Target transformation
    if use_transform:
        pt = PowerTransformer(method="yeo-johnson")
        y_transformed = pt.fit_transform(y.values.reshape(-1, 1)).ravel()
        y_fit = pd.Series(y_transformed, index=y.index)
        
        # Store transform bounds for clipping
        y_min, y_max = y.min(), y.max()
        y_median = y.median()
        transform_min = y_transformed.min()
        transform_max = y_transformed.max()
        
        # Safe inverse transform with NaN handling
        def inv_transform(arr):
            # First clip predictions to reasonable range in transformed space
            arr_clipped = np.clip(arr, transform_min - 2, transform_max + 2)
            # Then inverse transform
            result = pt.inverse_transform(arr_clipped.reshape(-1, 1)).ravel()
            # Final safety check: handle any remaining NaN/inf
            result = np.nan_to_num(result, nan=y_median, posinf=y_max * 1.5, neginf=y_min * 0.5)
            # Clip to reasonable range based on training data
            result = np.clip(result, y_min * 0.8, y_max * 1.2)
            return result
    else:
        y_fit = y.copy()
        inv_transform = lambda arr: arr
    
    model_params = {
        'loss_function': 'RMSE',
        'iterations': 8000,
        'learning_rate': 0.03,
        'depth': 8,
        'l2_leaf_reg': 3.0,
        'subsample': 0.9,
        'rsm': 0.9,
        'random_seed': RANDOM_SEED,
        'verbose': False,
        'allow_const_label': True,
    }
    
    if GPU_AVAILABLE:
        model_params['task_type'] = 'GPU'
        model_params['devices'] = '0'
    else:
        model_params['thread_count'] = -1
    
    oof_sum = np.zeros(len(X), dtype=float)
    oof_cnt = np.zeros(len(X), dtype=float)
    test_pred = np.zeros(len(X_test), dtype=float)
    fold_metrics = []
    best_iters = []
    
    for cv_seed in CV_SEEDS:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=cv_seed)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
            logger.info(f"[CAT] Seed {cv_seed} | Fold {fold}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_fit.iloc[train_idx], y_fit.iloc[val_idx]
            
            model = CatBoostModel(params=model_params)
            model.fit(
                X_train, y_train,
                X_val=X_val, y_val=y_val,
                early_stopping_rounds=300
            )
            
            # Get predictions
            val_pred = model.predict(X_val)
            oof_sum[val_idx] += inv_transform(val_pred)
            oof_cnt[val_idx] += 1
            
            # Calculate metrics on original scale
            val_pred_orig = inv_transform(val_pred)
            fold_rmse = rmse(y.iloc[val_idx], val_pred_orig)
            fold_metrics.append(fold_rmse)
            logger.info(f"  Fold {fold} RMSE: {fold_rmse:.4f}")
            
            # Test predictions
            test_pred += inv_transform(model.predict(X_test)) / (N_FOLDS * len(CV_SEEDS))
    
    # Average OOF predictions
    oof = oof_sum / np.maximum(oof_cnt, 1)
    logger.info(f"[CAT] CV RMSE: {rmse(y, oof):.4f}")
    
    return CVResult(oof, test_pred, fold_metrics, best_iters)


# ============================================================================
# CALIBRATION AND BLENDING
# ============================================================================

def two_stage_weight_search(y_true: np.ndarray, oof_mat: np.ndarray) -> Tuple[float, Tuple[float, ...]]:
    """
    Two-stage weight optimization for ensemble blending.
    Stage 1: Coarse grid search
    Stage 2: Fine grid search around best weights
    """
    n_models = oof_mat.shape[1]
    
    # Stage 1: Coarse grid
    logger.info("Stage 1: Coarse grid search...")
    best = (np.inf, tuple([1.0/n_models] * n_models))
    
    if n_models == 3:
        # For 3 models, search more efficiently
        for a in np.linspace(0, 1, 21):
            for b in np.linspace(0, 1-a, max(2, int((1-a)/0.05)+1)):
                c = 1.0 - a - b
                if c < 0:
                    continue
                w = np.array([a, b, c])
                score = rmse(y_true, (oof_mat * w).sum(1))
                if score < best[0]:
                    best = (score, (a, b, c))
    else:
        # For more models, use random search
        n_trials = 1000
        for _ in range(n_trials):
            w = np.random.dirichlet(np.ones(n_models))
            score = rmse(y_true, (oof_mat * w).sum(1))
            if score < best[0]:
                best = (score, tuple(w))
    
    # Stage 2: Fine grid around best
    logger.info("Stage 2: Fine grid search...")
    best_weights = best[1]
    fine = np.arange(-0.05, 0.0501, 0.005)
    best_fine = best
    
    for deltas in np.random.choice(fine, size=(100, n_models)):
        new_w = np.array(best_weights) + deltas
        if np.any(new_w < 0) or np.any(new_w > 1):
            continue
        new_w = new_w / new_w.sum()  # Normalize
        score = rmse(y_true, (oof_mat * new_w).sum(1))
        if score < best_fine[0]:
            best_fine = (score, tuple(new_w))
    
    return best_fine


def apply_isotonic_calibration(oof_preds: np.ndarray, test_preds: np.ndarray,
                               y_true: np.ndarray, clip_bounds: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Apply isotonic regression calibration."""
    iso_reg = IsotonicRegression(out_of_bounds="clip")
    iso_reg.fit(oof_preds, y_true)
    
    oof_calibrated = np.clip(iso_reg.predict(oof_preds), *clip_bounds)
    test_calibrated = np.clip(iso_reg.predict(test_preds), *clip_bounds)
    
    return oof_calibrated, test_calibrated


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def load_original_data():
    """Load original BPM dataset if available."""
    possible_paths = [
        ORIGINAL_DATA_DIR / "Train.csv",
        DATA_DIR / "Train_original.csv",
        Path("/kaggle/working/Train_original.csv"),
    ]
    
    for original_path in possible_paths:
        if original_path.exists():
            logger.info(f"✓ Found original data at {original_path}")
            original_df = pd.read_csv(original_path)
            
            cols_needed = BASE_NUM_COLS + [TARGET_COL]
            missing_cols = [col for col in cols_needed if col not in original_df.columns]
            
            if missing_cols:
                logger.warning(f"Missing columns: {missing_cols}")
                return None
                
            original_df = original_df[cols_needed]
            logger.info(f"Loaded {len(original_df)} samples from original dataset")
            return original_df
    
    logger.warning("Original dataset not found - using competition data only")
    return None


def main():
    """Main execution pipeline with advanced techniques."""
    logger.info("="*60)
    logger.info("PS5E9 - ADVANCED ENSEMBLE WITH TABML")
    logger.info(f"Environment: {'Kaggle' if 'kaggle' in os.getcwd() else 'Local'}")
    logger.info(f"GPU Available: {GPU_AVAILABLE}")
    logger.info("="*60)
    
    # Load data
    logger.info("\nLoading data...")
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    submission = pd.read_csv(DATA_DIR / "sample_submission.csv")
    
    # Remove IDs for processing
    test_ids = test[ID_COL].copy()
    train = train.drop(columns=[ID_COL])
    test = test.drop(columns=[ID_COL])
    
    logger.info(f"Train shape: {train.shape}")
    logger.info(f"Test shape: {test.shape}")
    
    # Optional: Add original data
    original_df = load_original_data()
    if original_df is not None:
        # Concatenate with competition data
        train = pd.concat([train, original_df.drop(columns=[TARGET_COL])], axis=0, ignore_index=True)
        y_combined = pd.concat([train[TARGET_COL], original_df[TARGET_COL]], axis=0, ignore_index=True)
        train[TARGET_COL] = y_combined
        logger.info(f"Combined train shape: {train.shape}")
    
    # Build advanced features
    train_fe, test_fe, feature_cols = build_advanced_features(train, test, BASE_NUM_COLS)
    
    X = train_fe[feature_cols].copy()
    y = train[TARGET_COL].astype(float).copy()
    X_test = test_fe[feature_cols].copy()
    
    logger.info(f"\nFinal shapes: X={X.shape}, X_test={X_test.shape}")
    
    # Initialize KFold
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    # Train models with TabML
    use_target_transform = True
    
    lgb_cv = train_lgbm_tabml(X, y, X_test, kf, use_target_transform)
    xgb_cv = train_xgb_tabml(X, y, X_test, kf, use_target_transform)
    cat_cv = train_cat_tabml(X, y, X_test, kf, use_target_transform)
    
    # Get predictions
    lgb_oof, lgb_test = lgb_cv.oof, lgb_cv.test_pred
    xgb_oof, xgb_test = xgb_cv.oof, xgb_cv.test_pred
    cat_oof, cat_test = cat_cv.oof, cat_cv.test_pred
    
    logger.info("\n" + "="*60)
    logger.info("MODEL PERFORMANCE SUMMARY")
    logger.info("="*60)
    logger.info(f"LGBM CV RMSE: {rmse(y, lgb_oof):.5f}")
    logger.info(f"XGB  CV RMSE: {rmse(y, xgb_oof):.5f}")
    logger.info(f"CAT  CV RMSE: {rmse(y, cat_oof):.5f}")
    
    # Adaptive clipping bounds from train target
    lo = float(np.quantile(y, 0.005))
    hi = float(np.quantile(y, 0.995))
    clip_lo = max(lo, 40.0)  # Domain knowledge: min BPM
    clip_hi = min(hi, 240.0)  # Domain knowledge: max BPM
    logger.info(f"\nClipping bounds: [{clip_lo:.1f}, {clip_hi:.1f}]")
    
    # Try different calibration modes
    modes = []
    
    # Mode A: No calibration
    A_oof = np.vstack([
        np.clip(lgb_oof, clip_lo, clip_hi),
        np.clip(xgb_oof, clip_lo, clip_hi),
        np.clip(cat_oof, clip_lo, clip_hi)
    ]).T
    A_test = np.vstack([
        np.clip(lgb_test, clip_lo, clip_hi),
        np.clip(xgb_test, clip_lo, clip_hi),
        np.clip(cat_test, clip_lo, clip_hi)
    ]).T
    modes.append(("no_calibration", A_oof, A_test))
    
    # Mode B: Per-model isotonic calibration
    iso_lgb = IsotonicRegression(out_of_bounds="clip").fit(lgb_oof, y)
    iso_xgb = IsotonicRegression(out_of_bounds="clip").fit(xgb_oof, y)
    iso_cat = IsotonicRegression(out_of_bounds="clip").fit(cat_oof, y)
    
    B_oof = np.vstack([
        np.clip(iso_lgb.predict(lgb_oof), clip_lo, clip_hi),
        np.clip(iso_xgb.predict(xgb_oof), clip_lo, clip_hi),
        np.clip(iso_cat.predict(cat_oof), clip_lo, clip_hi)
    ]).T
    B_test = np.vstack([
        np.clip(iso_lgb.predict(lgb_test), clip_lo, clip_hi),
        np.clip(iso_xgb.predict(xgb_test), clip_lo, clip_hi),
        np.clip(iso_cat.predict(cat_test), clip_lo, clip_hi)
    ]).T
    modes.append(("per_model_isotonic", B_oof, B_test))
    
    # Evaluate modes with blending
    logger.info("\n" + "="*60)
    logger.info("ENSEMBLE OPTIMIZATION")
    logger.info("="*60)
    
    best_global = (np.inf, None, None, None)  # (rmse, weights, test_mat, mode)
    
    for mode_name, oof_mat, test_mat in modes:
        logger.info(f"\nEvaluating mode: {mode_name}")
        
        # Find optimal weights
        score, weights = two_stage_weight_search(y.values, oof_mat)
        logger.info(f"  Best blend OOF RMSE: {score:.5f}")
        logger.info(f"  Weights - LGBM: {weights[0]:.3f}, XGB: {weights[1]:.3f}, CAT: {weights[2]:.3f}")
        
        # Apply final isotonic calibration on blended predictions
        oof_blend = (oof_mat * np.array(weights)).sum(1)
        iso_final = IsotonicRegression(out_of_bounds="clip").fit(oof_blend, y.values)
        oof_blend_iso = iso_final.predict(oof_blend)
        score_final = rmse(y.values, oof_blend_iso)
        logger.info(f"  After final isotonic: OOF RMSE: {score_final:.5f}")
        
        if score_final < best_global[0]:
            best_global = (score_final, weights, (test_mat, iso_final), mode_name)
    
    # Apply best configuration
    best_rmse, best_weights, (best_test_mat, best_iso), best_mode = best_global
    
    logger.info("\n" + "="*60)
    logger.info("FINAL RESULTS")
    logger.info("="*60)
    logger.info(f"Best mode: {best_mode}")
    logger.info(f"OOF RMSE: {best_rmse:.5f}")
    logger.info(f"Weights - LGBM: {best_weights[0]:.3f}, XGB: {best_weights[1]:.3f}, CAT: {best_weights[2]:.3f}")
    
    # Generate final predictions
    blended_test = (best_test_mat * np.array(best_weights)).sum(1)
    blended_test = best_iso.predict(blended_test)
    blended_test = np.clip(blended_test, clip_lo, clip_hi)
    
    # Save OOF predictions for stacking
    if OOF_DIR:
        oof_manager = OOFManager(output_dir=str(OOF_DIR))
        
        # Save final ensemble OOF
        oof_blend_final = (best_test_mat * np.array(best_weights)).sum(1)
        oof_df = pd.DataFrame({
            'predictions': best_iso.predict(oof_blend_final[:len(y)]) if len(oof_blend_final) > len(y) else oof_blend_final,
            'target': y
        })
        
        oof_manager.save_oof(
            predictions=oof_df,
            test_predictions=blended_test,
            model_name="advanced_ensemble",
            cv_score=best_rmse,
            model_params={
                'mode': best_mode,
                'weights': {
                    'lgbm': float(best_weights[0]),
                    'xgb': float(best_weights[1]),
                    'cat': float(best_weights[2])
                },
                'use_isotonic': True,
                'use_target_transform': use_target_transform,
                'n_features': len(feature_cols)
            }
        )
    
    # Save submission
    submission[TARGET_COL] = blended_test
    submission_file = SUBMISSION_DIR / f"submission_advanced_cv{best_rmse:.4f}.csv"
    submission.to_csv(submission_file, index=False)
    logger.info(f"\nSubmission saved to: {submission_file}")
    
    # Display feature importance if available
    if lgb_cv.feature_importance is not None:
        logger.info("\n" + "="*60)
        logger.info("TOP 20 FEATURES (LightGBM)")
        logger.info("="*60)
        for idx, row in lgb_cv.feature_importance.head(20).iterrows():
            logger.info(f"{idx+1:2d}. {row['feature']:40s}: {row['importance']:10.2f}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ ADVANCED ENSEMBLE COMPLETE!")
    logger.info("="*60)
    
    return best_rmse, submission


if __name__ == "__main__":
    try:
        cv_score, submission_df = main()
        print(f"\n✅ Final CV RMSE: {cv_score:.5f}")
        print(f"Submission shape: {submission_df.shape}")
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        raise