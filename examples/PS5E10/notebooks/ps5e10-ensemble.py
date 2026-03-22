# %% [markdown]
# # PS5E10 Ensemble - Road Accident Risk Prediction
#
# Multi-model ensemble with GPU acceleration using TabML
# - XGBoost (GPU)
# - LightGBM (GPU)
# - CatBoost (GPU)
# - Random Forest
# - Hill Climbing Optimization

# %%
"""
PS5E10 Competition - Ensemble Modeling (Kaggle Version with GPU)
Train multiple models and optimize ensemble weights using TabML
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

from tabml import (
        XGBoostModel, LightGBMModel, CatBoostModel,
        RandomForestModel, OOFEnsemble, OOFManager
    )
from tabml.features import FeatureEngineer, FeatureSelector
from tabml.advanced_features import AdvancedFeatureEngineer

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
import gc

# Setup paths for Kaggle
DATA_DIR = Path("/kaggle/input/playground-series-s5e10/")
OUTPUT_DIR = Path("/kaggle/working/output")
OOF_DIR = OUTPUT_DIR / "oof_predictions"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"
OOF_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

# Competition settings
TARGET_COL = 'accident_risk'
ID_COL = 'id'
RANDOM_SEED = 42
N_FOLDS = 5

# %% [markdown]
# ## Check GPU Availability

# %%
# Check if GPU is available
import torch

GPU_AVAILABLE = torch.cuda.is_available()
if GPU_AVAILABLE:
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️ No GPU detected, will use CPU")

# %% [markdown]
# ## Load Data

# %%
print("Loading competition data...")
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

print(f"Train: {train_df.shape}, Test: {test_df.shape}")
display(train_df.head())

# Separate IDs and target
train_ids = train_df[ID_COL]
test_ids = test_df[ID_COL]
y_train = train_df[TARGET_COL]

print(f"\nTarget statistics:")
print(f"  Mean: {y_train.mean():.4f}")
print(f"  Std: {y_train.std():.4f}")
print(f"  Range: [{y_train.min():.4f}, {y_train.max():.4f}]")

# %% [markdown]
# ## Feature Engineering with TabML
#
# Using TabML's FeatureEngineer and FeatureSelector for systematic feature creation

# %%
def create_domain_features(df):
    """
    Create domain-specific features for accident risk prediction.
    These are problem-specific features that TabML's generic tools won't create.
    """
    df = df.copy()

    # 1. Risk interaction features (categorical combinations)
    df['weather_lighting'] = df['weather'] + '_' + df['lighting']
    df['roadtype_weather'] = df['road_type'] + '_' + df['weather']
    df['time_lighting'] = df['time_of_day'] + '_' + df['lighting']

    # 2. Composite risk scores
    df['visibility_risk'] = ((df['weather'].isin(['foggy', 'rainy'])).astype(int) +
                             (df['lighting'].isin(['dim', 'night'])).astype(int))

    df['geometry_risk'] = df['curvature'] * df['speed_limit']

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
    df['safety_score'] = (df['road_signs_present'].astype(int) +
                         df['public_road'].astype(int) -
                         df['visibility_risk'])

    # 7. Temporal features
    df['is_rush_hour'] = ((df['time_of_day'].isin(['morning', 'evening']))).astype(int)

    # 8. Feature ratios
    df['speed_per_lane'] = df['speed_limit'] / (df['num_lanes'] + 0.1)
    df['accidents_per_curve'] = df['num_reported_accidents'] / (df['curvature'] + 0.01)

    return df

# Step 1: Create domain-specific features
print("Step 1: Creating domain-specific features...")
X_train_base = train_df.drop([ID_COL, TARGET_COL], axis=1)
X_test_base = test_df.drop([ID_COL], axis=1)

X_train_domain = create_domain_features(X_train_base)
X_test_domain = create_domain_features(X_test_base)

print(f"   After domain features: {X_train_domain.shape}")

# Step 2: Use TabML's FeatureEngineer for systematic feature engineering
print("\nStep 2: Applying TabML FeatureEngineer...")
feature_engineer = FeatureEngineer(
    numeric_impute_strategy='median',
    categorical_impute_strategy='most_frequent',
    scaling_method=None,  # No scaling for tree models (they're scale-invariant)
    categorical_encoding='target',  # Better than label encoding for tree models
    create_interactions=True,  # Automatic numeric interactions
    create_polynomial=True,  # Polynomial features for numeric columns
    max_cardinality=50,
    min_frequency=0.01
)

# Fit on train, transform both
X_train_engineered = feature_engineer.fit_transform(X_train_domain, y_train)
X_test_engineered = feature_engineer.transform(X_test_domain)

print(f"   After TabML engineering: {X_train_engineered.shape}")
print(f"   Features created: {X_train_engineered.shape[1] - X_train_domain.shape[1]}")

# Step 3: Feature selection to reduce noise and improve model performance
print("\nStep 3: Applying feature selection...")
feature_selector = FeatureSelector(
    method='mutual_info',  # Mutual information for regression
    n_features=100,  # Keep top 100 features
    task_type='regression'
)

X_train_selected = feature_selector.fit_transform(X_train_engineered, y_train)
X_test_selected = feature_selector.transform(X_test_engineered)

print(f"   After feature selection: {X_train_selected.shape}")
print(f"   Features reduced from {X_train_engineered.shape[1]} to {X_train_selected.shape[1]}")

# Get selected feature names
selected_features = feature_selector.selected_features_
print(f"\n✅ Feature engineering complete!")
print(f"   Total features: {len(selected_features)}")
print(f"\nTop 10 selected features by importance:")
feature_importance = feature_selector.get_feature_importance()
for i in range(min(10, len(feature_importance))):
    feat = feature_importance.iloc[i]
    print(f"   {i+1:2d}. {feat['feature']:30s} (score: {feat['score']:.4f})")

# X_train_selected and X_test_selected are already DataFrames with proper columns
X_train = X_train_selected.copy()
X_test = X_test_selected.copy()

# %% [markdown]
# ## Model Training Function

# %%
def train_model_with_oof(model, model_name, X_train, y_train, X_test, n_folds=5):
    """
    Train model with K-fold cross-validation and generate OOF predictions.

    Args:
        model: Model instance (with GPU support if available)
        model_name: Name for saving
        X_train: Training features
        y_train: Training target
        X_test: Test features
        n_folds: Number of CV folds

    Returns:
        oof_predictions, test_predictions, cv_scores, oof_rmse
    """
    print(f"\n{'='*60}")
    print(f"TRAINING {model_name}")
    print(f"{'='*60}")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    oof_predictions = np.zeros(len(X_train))
    test_predictions = np.zeros(len(X_test))
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        print(f"\nFold {fold}/{n_folds}")

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

        print(f"  RMSE: {fold_rmse:.6f}")
        print(f"  MAE:  {fold_mae:.6f}")
        print(f"  R²:   {fold_r2:.6f}")

        # Clean up
        gc.collect()

    # Overall OOF score
    oof_rmse = np.sqrt(mean_squared_error(y_train, oof_predictions))
    oof_mae = mean_absolute_error(y_train, oof_predictions)
    oof_r2 = r2_score(y_train, oof_predictions)

    print(f"\n{'='*60}")
    print(f"{model_name} - OVERALL OOF SCORES")
    print(f"{'='*60}")
    print(f"RMSE: {oof_rmse:.6f} (±{np.std(cv_scores):.6f})")
    print(f"MAE:  {oof_mae:.6f}")
    print(f"R²:   {oof_r2:.6f}")
    print(f"CV Scores: {[f'{s:.6f}' for s in cv_scores]}")

    return oof_predictions, test_predictions, cv_scores, oof_rmse

# %% [markdown]
# ## Initialize OOF Manager

# %%
oof_manager = OOFManager(str(OOF_DIR))
all_test_predictions = {}

print("✅ OOF Manager initialized")

# %% [markdown]
# ## Model 1: XGBoost (GPU Accelerated)

# %%
print("\n🚀 Training XGBoost with GPU acceleration...")

xgb_params = {
    'n_estimators': 1000,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': RANDOM_SEED,
    'objective': 'reg:squarederror',  # Explicitly set for regression
    'tree_method': 'gpu_hist' if GPU_AVAILABLE else 'hist',
    'gpu_id': 0 if GPU_AVAILABLE else None
}

# Remove gpu_id if not using GPU
if not GPU_AVAILABLE:
    xgb_params.pop('gpu_id')

xgb_model = XGBoostModel(params=xgb_params)
oof_xgb, test_xgb, cv_xgb, score_xgb = train_model_with_oof(
    xgb_model, "XGBoost_GPU", X_train, y_train, X_test, N_FOLDS
)

# Save OOF
oof_manager.save_oof(
    predictions=oof_xgb,
    model_name="xgboost_gpu",
    model_params=xgb_params,
    cv_score=score_xgb,
    cv_scores_per_fold=cv_xgb,
    test_predictions=test_xgb
)
all_test_predictions['xgboost_gpu'] = test_xgb

# %% [markdown]
# ## Model 2: LightGBM (GPU Accelerated)

# %%
print("\n🚀 Training LightGBM with GPU acceleration...")

lgb_params = {
    'n_estimators': 1000,
    'max_depth': 7,
    'learning_rate': 0.05,
    'num_leaves': 63,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': RANDOM_SEED,
    'objective': 'regression',  # Explicitly set for regression
    'metric': 'rmse',
    'device': 'gpu' if GPU_AVAILABLE else 'cpu',
    'gpu_platform_id': 0 if GPU_AVAILABLE else None,
    'gpu_device_id': 0 if GPU_AVAILABLE else None
}

# Remove GPU params if not available
if not GPU_AVAILABLE:
    lgb_params.pop('gpu_platform_id')
    lgb_params.pop('gpu_device_id')

lgb_model = LightGBMModel(params=lgb_params)
oof_lgb, test_lgb, cv_lgb, score_lgb = train_model_with_oof(
    lgb_model, "LightGBM_GPU", X_train, y_train, X_test, N_FOLDS
)

oof_manager.save_oof(
    predictions=oof_lgb,
    model_name="lightgbm_gpu",
    model_params=lgb_params,
    cv_score=score_lgb,
    cv_scores_per_fold=cv_lgb,
    test_predictions=test_lgb
)
all_test_predictions['lightgbm_gpu'] = test_lgb

# %% [markdown]
# ## Model 3: CatBoost (GPU Accelerated)

# %%
print("\n🚀 Training CatBoost with GPU acceleration...")

cat_params = {
    'iterations': 1000,
    'depth': 6,
    'learning_rate': 0.05,
    'l2_leaf_reg': 3,
    'random_seed': RANDOM_SEED,  # CatBoost uses 'random_seed' not 'random_state'
    'loss_function': 'RMSE',  # Explicitly set for regression
    'task_type': 'GPU' if GPU_AVAILABLE else 'CPU',
    'devices': '0' if GPU_AVAILABLE else None,
    'verbose': False
}

# Remove GPU params if not available
if not GPU_AVAILABLE:
    cat_params.pop('devices')

cat_model = CatBoostModel(params=cat_params)
oof_cat, test_cat, cv_cat, score_cat = train_model_with_oof(
    cat_model, "CatBoost_GPU", X_train, y_train, X_test, N_FOLDS
)

oof_manager.save_oof(
    predictions=oof_cat,
    model_name="catboost_gpu",
    model_params=cat_params,
    cv_score=score_cat,
    cv_scores_per_fold=cv_cat,
    test_predictions=test_cat
)
all_test_predictions['catboost_gpu'] = test_cat

# %% [markdown]
# ## Model 4: Random Forest

# %%
print("\n🚀 Training Random Forest...")

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

# %% [markdown]
# ## Training Summary

# %%
print("\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)

summary = oof_manager.list_oofs(sort_by='cv_score', ascending=True)
display(summary)

# Best model
best_model = summary.iloc[0]
print(f"\n🏆 Best Single Model: {best_model['model_name']}")
print(f"   CV Score: {best_model['cv_score']:.6f}")

# %% [markdown]
# ## Ensemble Optimization
#
# Compare different ensemble methods and optimize weights

# %%
print("\n" + "="*60)
print("ENSEMBLE OPTIMIZATION")
print("="*60)

# Load all OOF predictions
all_oofs = oof_manager.load_all_oofs()
oof_combined = oof_manager.combine_oofs(all_oofs, method='horizontal')

print(f"\nOOF predictions shape: {oof_combined.shape}")
print(f"Models: {list(oof_combined.columns)}")

# %% [markdown]
# ### Method 1: Simple Average

# %%
print("\n" + "="*40)
print("METHOD 1: SIMPLE AVERAGE")
print("="*40)

avg_oof = oof_combined.mean(axis=1)
avg_rmse = np.sqrt(mean_squared_error(y_train, avg_oof))
avg_mae = mean_absolute_error(y_train, avg_oof)
avg_r2 = r2_score(y_train, avg_oof)

print(f"RMSE: {avg_rmse:.6f}")
print(f"MAE:  {avg_mae:.6f}")
print(f"R²:   {avg_r2:.6f}")

# %% [markdown]
# ### Method 2: Hill Climbing Optimization

# %%
print("\n" + "="*40)
print("METHOD 2: HILL CLIMBING OPTIMIZATION")
print("="*40)

ensemble = OOFEnsemble(task_type='regression', metric='rmse', random_state=RANDOM_SEED)

hill_climbing_weights = ensemble.optimize_weights(
    oof_combined,
    y_train,
    method='hill_climbing',
    n_iterations=2000,
    patience=200
)

hill_oof = np.average(oof_combined.values, weights=hill_climbing_weights, axis=1)
hill_rmse = np.sqrt(mean_squared_error(y_train, hill_oof))
hill_mae = mean_absolute_error(y_train, hill_oof)
hill_r2 = r2_score(y_train, hill_oof)

print(f"\nHill Climbing Results:")
print(f"RMSE: {hill_rmse:.6f}")
print(f"MAE:  {hill_mae:.6f}")
print(f"R²:   {hill_r2:.6f}")
print(f"\nOptimized Weights:")
for model_name, weight in zip(oof_combined.columns, hill_climbing_weights):
    print(f"  {model_name}: {weight:.4f}")

# %% [markdown]
# ### Method 3: Stacking with Ridge

# %%
print("\n" + "="*40)
print("METHOD 3: STACKING (Ridge Meta-Learner)")
print("="*40)

meta_model = Ridge(alpha=1.0, random_state=RANDOM_SEED)
ensemble.fit_stacking(oof_combined, y_train, meta_model=meta_model)

stacking_oof = ensemble.meta_model.predict(oof_combined)
stacking_oof = np.clip(stacking_oof, 0, 1)
stacking_rmse = np.sqrt(mean_squared_error(y_train, stacking_oof))
stacking_mae = mean_absolute_error(y_train, stacking_oof)
stacking_r2 = r2_score(y_train, stacking_oof)

print(f"RMSE: {stacking_rmse:.6f}")
print(f"MAE:  {stacking_mae:.6f}")
print(f"R²:   {stacking_r2:.6f}")

# %% [markdown]
# ## Ensemble Comparison

# %%
results = {
    'Simple Average': {'rmse': avg_rmse, 'mae': avg_mae, 'r2': avg_r2},
    'Hill Climbing': {'rmse': hill_rmse, 'mae': hill_mae, 'r2': hill_r2},
    'Stacking': {'rmse': stacking_rmse, 'mae': stacking_mae, 'r2': stacking_r2}
}

results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('rmse')

print("\n" + "="*60)
print("ENSEMBLE METHOD COMPARISON")
print("="*60)
display(results_df)

best_method = results_df.index[0]
print(f"\n🏆 BEST METHOD: {best_method}")
print(f"   RMSE: {results_df.loc[best_method, 'rmse']:.6f}")
print(f"   MAE:  {results_df.loc[best_method, 'mae']:.6f}")
print(f"   R²:   {results_df.loc[best_method, 'r2']:.6f}")

# %% [markdown]
# ## Generate Final Submissions

# %%
print("\n" + "="*60)
print("GENERATING FINAL SUBMISSIONS")
print("="*60)

# Combine test predictions
test_combined = pd.DataFrame(all_test_predictions)

# IMPORTANT: Reorder columns to match OOF training order for stacking
# The meta-model expects features in the same order as during training
test_combined = test_combined[oof_combined.columns]

# Generate submission for each method
submissions = {}

# 1. Simple Average
submissions['avg'] = test_combined.mean(axis=1)

# 2. Hill Climbing
submissions['hill_climbing'] = np.average(test_combined.values, weights=hill_climbing_weights, axis=1)

# 3. Stacking
submissions['stacking'] = ensemble.meta_model.predict(test_combined)

# Clip all predictions to valid range
for key in submissions:
    submissions[key] = np.clip(submissions[key], 0, 1)

# Save all submissions
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

for method, preds in submissions.items():
    submission = pd.DataFrame({
        ID_COL: test_ids,
        TARGET_COL: preds
    })

    submission_path = SUBMISSION_DIR / f"submission_{method}_{timestamp}.csv"
    submission.to_csv(submission_path, index=False)
    print(f"✅ {method:20s} -> {submission_path.name}")

# Save the best submission with a special name
best_method_key = best_method.lower().replace(' ', '_')
best_submission = pd.DataFrame({
    ID_COL: test_ids,
    TARGET_COL: submissions.get(best_method_key, submissions['hill_climbing'])
})

best_path = SUBMISSION_DIR / f"submission_BEST_{timestamp}.csv"
best_submission.to_csv(best_path, index=False)
print(f"\n🏆 BEST submission -> {best_path.name}")

# %% [markdown]
# ## Final Summary

# %%
print("\n" + "="*60)
print("ENSEMBLE MODELING COMPLETE!")
print("="*60)

print(f"\n📊 Summary:")
print(f"  • Models trained: {len(all_test_predictions)}")
print(f"  • Best single model: {best_model['model_name']} ({best_model['cv_score']:.6f} RMSE)")
print(f"  • Best ensemble: {best_method} ({results_df.loc[best_method, 'rmse']:.6f} RMSE)")
print(f"  • Improvement: {(best_model['cv_score'] - results_df.loc[best_method, 'rmse']) / best_model['cv_score'] * 100:.2f}%")

print(f"\n📁 Output files:")
print(f"  • OOF predictions: {OOF_DIR}")
print(f"  • Submissions: {SUBMISSION_DIR}")
print(f"  • Best submission: {best_path.name}")

print(f"\n🚀 GPU Usage: {'✅ Enabled' if GPU_AVAILABLE else '❌ Disabled'}")

print("\n✅ Ready to submit to Kaggle!")
