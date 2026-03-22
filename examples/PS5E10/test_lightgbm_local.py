"""
PS5E10 - LightGBM Local Test
Quick test of LightGBM regression detection fix on local machine (CPU)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Import TabML
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tabml import LightGBMModel
from tabml.features import FeatureEngineer, FeatureSelector

print("="*60)
print("PS5E10 LightGBM Local Test")
print("="*60)

# Setup paths
DATA_DIR = Path(__file__).parent / "data" / "raw" / "PS5E10"
TARGET_COL = 'accident_risk'
ID_COL = 'id'
RANDOM_SEED = 42

# Check if data exists
train_file = DATA_DIR / "train.csv"
test_file = DATA_DIR / "test.csv"

if not train_file.exists():
    print(f"\n❌ Data not found at {train_file}")
    print("Please download the data from Kaggle:")
    print("https://www.kaggle.com/competitions/playground-series-s5e10/data")
    sys.exit(1)

# Load data
print("\n📊 Loading data...")
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

print(f"Train: {train_df.shape}, Test: {test_df.shape}")

# Check target dtype and distribution
y_train = train_df[TARGET_COL]
print(f"\n🎯 Target variable analysis:")
print(f"   Dtype: {y_train.dtype}")
print(f"   Unique values: {y_train.nunique()}")
print(f"   Unique ratio: {y_train.nunique() / len(y_train):.4f}")
print(f"   Range: [{y_train.min():.4f}, {y_train.max():.4f}]")
print(f"   Mean: {y_train.mean():.4f}")
print(f"   Std: {y_train.std():.4f}")

# Domain-specific feature engineering
def create_domain_features(df):
    """Create domain-specific features."""
    df = df.copy()

    # Risk interaction features
    df['weather_lighting'] = df['weather'] + '_' + df['lighting']
    df['roadtype_weather'] = df['road_type'] + '_' + df['weather']

    # Composite risk scores
    df['visibility_risk'] = ((df['weather'].isin(['foggy', 'rainy'])).astype(int) +
                             (df['lighting'].isin(['dim', 'night'])).astype(int))
    df['geometry_risk'] = df['curvature'] * df['speed_limit']

    # Historical risk
    df['has_previous_accidents'] = (df['num_reported_accidents'] > 0).astype(int)
    df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 0.1)

    # Boolean combinations
    df['night_rainy'] = ((df['lighting'] == 'night') &
                         (df['weather'].isin(['rainy', 'foggy']))).astype(int)
    df['high_curve_high_speed'] = ((df['curvature'] > 0.5) &
                                   (df['speed_limit'] > 50)).astype(int)

    # Road complexity
    df['road_complexity'] = df['num_lanes'] * df['curvature'] * (df['speed_limit'] / 100)

    # Safety score
    df['safety_score'] = (df['road_signs_present'].astype(int) +
                         df['public_road'].astype(int) -
                         df['visibility_risk'])

    # Feature ratios
    df['speed_per_lane'] = df['speed_limit'] / (df['num_lanes'] + 0.1)
    df['accidents_per_curve'] = df['num_reported_accidents'] / (df['curvature'] + 0.01)

    return df

# Feature engineering
print("\n🔧 Feature Engineering...")
print("Step 1: Domain features")
X_train_base = train_df.drop([ID_COL, TARGET_COL], axis=1)
X_train_domain = create_domain_features(X_train_base)
print(f"   After domain features: {X_train_domain.shape}")

print("Step 2: TabML FeatureEngineer")
feature_engineer = FeatureEngineer(
    numeric_impute_strategy='median',
    categorical_impute_strategy='most_frequent',
    scaling_method='robust',
    categorical_encoding='target',
    create_interactions=True,
    create_polynomial=True,
    max_cardinality=50,
    min_frequency=0.01
)

X_train_engineered = feature_engineer.fit_transform(X_train_domain, y_train)
print(f"   After TabML engineering: {X_train_engineered.shape}")

print("Step 3: Feature selection")
feature_selector = FeatureSelector(
    method='mutual_info',
    n_features=50,  # Use fewer features for faster local testing
    task_type='regression'
)

X_train_selected = feature_selector.fit_transform(X_train_engineered, y_train)
print(f"   After feature selection: {X_train_selected.shape}")

# Split for validation
print("\n📋 Creating train/validation split...")
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_selected, y_train,
    test_size=0.2,
    random_state=RANDOM_SEED
)
print(f"   Train: {X_train_final.shape}, Val: {X_val.shape}")

# Train LightGBM
print("\n🚀 Training LightGBM (CPU)...")
print("This will test the regression detection fix!")

lgb_params = {
    'n_estimators': 500,  # Reduced for faster local testing
    'max_depth': 7,
    'learning_rate': 0.05,
    'num_leaves': 63,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': RANDOM_SEED,
    'objective': 'regression',
    'metric': 'rmse',
    'device': 'cpu',
    'verbosity': -1
}

lgb_model = LightGBMModel(params=lgb_params)

# Check task detection
print(f"\n🔍 Before fit:")
print(f"   is_classification: {lgb_model.is_classification}")

# Fit the model
print("\n⏳ Training...")
lgb_model.fit(X_train_final, y_train_final, X_val, y_val, early_stopping_rounds=50)

print(f"\n✅ After fit:")
print(f"   is_classification: {lgb_model.is_classification}")
print(f"   Model type: {type(lgb_model.model).__name__}")

# Predictions
print("\n📊 Making predictions...")
train_preds = lgb_model.predict(X_train_final)
val_preds = lgb_model.predict(X_val)

# Clip to valid range
train_preds = np.clip(train_preds, 0, 1)
val_preds = np.clip(val_preds, 0, 1)

# Metrics
train_rmse = np.sqrt(mean_squared_error(y_train_final, train_preds))
train_mae = mean_absolute_error(y_train_final, train_preds)
train_r2 = r2_score(y_train_final, train_preds)

val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
val_mae = mean_absolute_error(y_val, val_preds)
val_r2 = r2_score(y_val, val_preds)

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Training Set:")
print(f"  RMSE: {train_rmse:.6f}")
print(f"  MAE:  {train_mae:.6f}")
print(f"  R²:   {train_r2:.6f}")
print(f"\nValidation Set:")
print(f"  RMSE: {val_rmse:.6f}")
print(f"  MAE:  {val_mae:.6f}")
print(f"  R²:   {val_r2:.6f}")

# Feature importance
print("\n🔝 Top 10 Features:")
feature_importance = lgb_model.get_feature_importance()
for i, (feat, score) in enumerate(feature_importance[:10], 1):
    print(f"   {i:2d}. {feat:40s} {score:.4f}")

print("\n" + "="*60)
print("✅ LightGBM Local Test Complete!")
print("="*60)

# Summary
if lgb_model.is_classification:
    print("\n❌ FAILED: Model was classified as classification task!")
    print("The regression detection fix did not work.")
else:
    print("\n✅ SUCCESS: Model correctly detected as regression task!")
    print("The fix is working correctly.")

    if val_rmse < 0.15:
        print("✅ Good performance: RMSE < 0.15")
    elif val_rmse < 0.20:
        print("⚠️  Acceptable performance: RMSE < 0.20")
    else:
        print("❌ Poor performance: RMSE >= 0.20")
        print("Expected RMSE: 0.08-0.12 with full feature engineering")
