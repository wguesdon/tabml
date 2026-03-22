"""
PS5E10 - LightGBM Regression Detection Test (Synthetic Data)
Test the improved regression detection fix without needing competition data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Import TabML
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tabml import LightGBMModel

print("="*70)
print("LightGBM Regression Detection Test - Synthetic Data")
print("="*70)

# Create synthetic data similar to PS5E10
np.random.seed(42)
n_samples = 10000

print("\n📊 Creating synthetic continuous target data...")

# Simulate accident_risk features
data = {
    'curvature': np.random.uniform(0, 1, n_samples),
    'speed_limit': np.random.choice([30, 35, 40, 50, 60, 70, 80], n_samples),
    'num_lanes': np.random.choice([1, 2, 3, 4], n_samples),
    'num_accidents': np.random.poisson(1, n_samples),
    'feature_1': np.random.randn(n_samples),
    'feature_2': np.random.randn(n_samples),
    'feature_3': np.random.randn(n_samples),
}

df = pd.DataFrame(data)

# Create continuous target in [0, 1] range (like accident_risk)
# This will have float dtype but potentially < 100 unique values due to rounding
df['target'] = (
    0.3 * df['curvature'] +
    0.2 * (df['speed_limit'] / 100) +
    0.1 * (df['num_lanes'] / 4) +
    0.15 * (df['num_accidents'] / 5) +
    0.05 * df['feature_1'] +
    0.1 * df['feature_2'] +
    0.1 * df['feature_3'] +
    np.random.normal(0, 0.05, n_samples)
)

# Clip to [0, 1] and round to simulate limited precision
df['target'] = np.clip(df['target'], 0, 1)
df['target'] = np.round(df['target'], 2)  # Round to 2 decimals

y = df['target']
X = df.drop('target', axis=1)

print(f"\n🎯 Target variable characteristics:")
print(f"   Dtype: {y.dtype}")
print(f"   Unique values: {y.nunique()}")
print(f"   Unique ratio: {y.nunique() / len(y):.4f}")
print(f"   Range: [{y.min():.4f}, {y.max():.4f}]")
print(f"   Mean: {y.mean():.4f}")
print(f"   Std: {y.std():.4f}")

# This is the critical case:
# - Float dtype (should be regression)
# - But potentially < 100 unique values (old heuristic would say classification)
if y.nunique() < 100:
    print(f"\n⚠️  CRITICAL TEST: {y.nunique()} unique values < 100")
    print("   Old heuristic would incorrectly classify as classification!")
    print("   New heuristic should detect as regression (float + unique_ratio > 0.05)")

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n📋 Data split: Train={X_train.shape}, Val={X_val.shape}")

# Test LightGBM
print("\n" + "="*70)
print("TEST 1: LightGBM with automatic task detection")
print("="*70)

lgb_params = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'random_state': 42,
    'verbosity': -1
}

lgb_model = LightGBMModel(params=lgb_params)

print(f"\n🔍 Before fit:")
print(f"   is_classification: {lgb_model.is_classification}")

try:
    print("\n⏳ Training LightGBM...")
    lgb_model.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=10)

    print(f"\n✅ After fit:")
    print(f"   is_classification: {lgb_model.is_classification}")
    print(f"   Model type: {type(lgb_model.model).__name__}")

    # Make predictions
    val_preds = lgb_model.predict(X_val)
    val_preds = np.clip(val_preds, 0, 1)

    # Calculate metrics
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    val_mae = mean_absolute_error(y_val, val_preds)
    val_r2 = r2_score(y_val, val_preds)

    print(f"\n📊 Validation Performance:")
    print(f"   RMSE: {val_rmse:.6f}")
    print(f"   MAE:  {val_mae:.6f}")
    print(f"   R²:   {val_r2:.6f}")

    # Test result
    print("\n" + "="*70)
    if lgb_model.is_classification:
        print("❌ TEST FAILED: Model detected as CLASSIFICATION")
        print("   The fix did not work!")
    else:
        print("✅ TEST PASSED: Model detected as REGRESSION")
        print("   The fix is working correctly!")
    print("="*70)

    # Additional validation
    if isinstance(lgb_model.model, type(None)):
        print("\n❌ ERROR: Model is None")
    else:
        model_name = type(lgb_model.model).__name__
        if 'Regressor' in model_name:
            print(f"\n✅ Correct model instantiated: {model_name}")
        elif 'Classifier' in model_name:
            print(f"\n❌ Wrong model instantiated: {model_name}")
            print("   Expected: LGBMRegressor, Got: LGBMClassifier")

except ValueError as e:
    print(f"\n❌ TEST FAILED with ValueError:")
    print(f"   {str(e)}")
    print("\n   This is the bug we're trying to fix!")
    print("   The model incorrectly tried to use LGBMClassifier on continuous target")

except Exception as e:
    print(f"\n❌ TEST FAILED with unexpected error:")
    print(f"   {type(e).__name__}: {str(e)}")

# Test with different unique value counts
print("\n" + "="*70)
print("TEST 2: Testing edge cases")
print("="*70)

test_cases = [
    ("Float with high unique ratio (should be regression)",
     np.random.uniform(0, 1, 1000), True),
    ("Float with low unique ratio (should be regression)",
     np.round(np.random.uniform(0, 1, 1000), 1), True),
    ("Integer with < 100 unique values (should be classification)",
     np.random.randint(0, 10, 1000), False),
    ("Integer with > 100 unique values (should be regression)",
     np.arange(1000), True),
]

for test_name, test_target, expected_regression in test_cases:
    test_y = pd.Series(test_target)
    test_model = LightGBMModel(params={'n_estimators': 10, 'verbosity': -1})

    print(f"\n{test_name}")
    print(f"  Dtype: {test_y.dtype}, Unique: {test_y.nunique()}, Ratio: {test_y.nunique()/len(test_y):.4f}")

    test_model._determine_task_type(test_y)
    is_regression = not test_model.is_classification

    if is_regression == expected_regression:
        print(f"  ✅ Correct: {'Regression' if is_regression else 'Classification'}")
    else:
        print(f"  ❌ Wrong: Got {'Regression' if is_regression else 'Classification'}, "
              f"Expected {'Regression' if expected_regression else 'Classification'}")

print("\n" + "="*70)
print("All tests complete!")
print("="*70)
