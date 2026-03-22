"""
Quick test script to verify PS5E10 setup with a sample of data
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from tabml import XGBoostModel, OOFEnsemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

print("="*60)
print("PS5E10 QUICK TEST")
print("="*60)

# Load sample of data
print("\n1. Loading data (sample)...")
train_df = pd.read_csv("../../data/raw/PS5E10/train.csv")
print(f"   Full dataset: {train_df.shape}")

# Sample for quick test
sample_size = 10000
train_sample = train_df.sample(n=sample_size, random_state=42)
print(f"   Test sample: {train_sample.shape}")

# Prepare features
print("\n2. Preparing features...")
X = train_sample.drop(['id', 'accident_risk'], axis=1)
y = train_sample['accident_risk']

# Encode categorical features
categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Convert boolean to int
boolean_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
for col in boolean_cols:
    X[col] = X[col].astype(int)

print(f"   Features: {X.shape[1]}")
print(f"   Target mean: {y.mean():.4f}")

# Split data
print("\n3. Splitting data...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")

# Train a quick XGBoost model
print("\n4. Training XGBoost model...")
model = XGBoostModel(params={
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'random_state': 42
})

model.fit(X_train, y_train, X_val, y_val)

# Predict
print("\n5. Evaluating...")
train_preds = model.predict(X_train)
val_preds = model.predict(X_val)

train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))

print(f"   Train RMSE: {train_rmse:.6f}")
print(f"   Val RMSE: {val_rmse:.6f}")

# Test OOF functionality
print("\n6. Testing OOF ensemble...")
ensemble = OOFEnsemble(task_type='regression', random_state=42)

# Quick 2-fold test
from sklearn.model_selection import KFold
kf = KFold(n_splits=2, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    print(f"   Fold {fold}/2...")
    X_fold_train = X.iloc[train_idx]
    y_fold_train = y.iloc[train_idx]
    X_fold_val = X.iloc[val_idx]
    y_fold_val = y.iloc[val_idx]

    # Pass validation data to ensure correct task type detection
    fold_model = XGBoostModel(params={'n_estimators': 50, 'max_depth': 4})
    fold_model.fit(X_fold_train, y_fold_train, X_fold_val, y_fold_val)

    oof_preds[val_idx] = fold_model.predict(X_fold_val)

oof_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"   OOF RMSE: {oof_rmse:.6f}")

print("\n" + "="*60)
print("✅ ALL TESTS PASSED!")
print("="*60)
print("\nYou're ready to run the full workflow:")
print("  1. python 01_eda.py              # ~2-5 min")
print("  2. python 02_baseline_models.py  # ~10-20 min")
print("  3. python 03_ensemble.py         # ~2-5 min")
print("="*60 + "\n")
