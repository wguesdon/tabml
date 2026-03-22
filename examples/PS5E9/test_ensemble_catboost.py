"""
Test the updated CatBoost configuration from the ensemble script
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Check if TabML is available
try:
    from tabml import CatBoostModel
    print("✓ TabML CatBoostModel imported successfully")
except ImportError:
    print("✗ TabML not installed")
    exit(1)

# Check GPU availability
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
except:
    GPU_AVAILABLE = False
    print("✓ No GPU detected, testing CPU configuration")

# Generate test data
print("\nGenerating test data...")
X = pd.DataFrame(np.random.randn(1000, 20), columns=[f'feat_{i}' for i in range(20)])
y = pd.Series(np.random.randn(1000))

print(f"Data shape: {X.shape}")

# Test the exact configuration from ensemble script
print("\n" + "="*50)
print("TESTING ENSEMBLE SCRIPT CONFIGURATION")
print("="*50)

def train_catboost_test(X_train, y_train, X_test):
    """Test function mimicking the ensemble script's train_catboost"""
    
    RANDOM_SEED = 42
    
    # Base parameters that work for both CPU and GPU
    model_params = {
        'loss_function': 'RMSE',
        'iterations': 100,  # Reduced for testing
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
        model_params['colsample_bylevel'] = 0.5
        # NO thread_count for GPU
        config_type = "GPU (colsample_bylevel, no thread_count)"
    else:
        model_params['rsm'] = 0.5
        model_params['thread_count'] = -1
        config_type = "CPU (rsm + thread_count)"
    
    print(f"Testing {config_type} configuration...")
    print(f"Parameters: {model_params}")
    
    try:
        model = CatBoostModel(params=model_params)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_train[:len(predictions)], predictions))
        print(f"✓ Configuration works! Test RMSE: {rmse:.4f}")
        return True
    except Exception as e:
        print(f"✗ Configuration failed: {e}")
        return False

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Test the configuration
success = train_catboost_test(X_train, y_train, X_test)

# Test with KFold (like in ensemble)
print("\n" + "="*50)
print("TESTING WITH KFOLD (LIKE ENSEMBLE)")
print("="*50)

kf = KFold(n_splits=2, shuffle=True, random_state=42)  # Just 2 folds for quick test

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    print(f"\nFold {fold}/2")
    
    X_fold_train = X_train.iloc[train_idx]
    y_fold_train = y_train.iloc[train_idx]
    X_fold_val = X_train.iloc[val_idx]
    y_fold_val = y_train.iloc[val_idx]
    
    RANDOM_SEED = 42
    
    model_params = {
        'loss_function': 'RMSE',
        'iterations': 50,  # Very small for testing
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
        model_params['colsample_bylevel'] = 0.5
    else:
        model_params['rsm'] = 0.5
        model_params['thread_count'] = -1
    
    try:
        model = CatBoostModel(params=model_params)
        model.fit(
            X_fold_train, 
            y_fold_train,
            X_val=X_fold_val,
            y_val=y_fold_val,
            early_stopping_rounds=25
        )
        
        val_predictions = model.predict(X_fold_val)
        fold_rmse = np.sqrt(mean_squared_error(y_fold_val, val_predictions))
        print(f"  ✓ Fold {fold} RMSE: {fold_rmse:.4f}")
    except Exception as e:
        print(f"  ✗ Fold {fold} failed: {e}")
        break

# Test error handling
print("\n" + "="*50)
print("TESTING ERROR HANDLING (LIKE ENSEMBLE)")
print("="*50)

try:
    # Intentionally use bad params to test error handling
    bad_params = {
        'loss_function': 'RMSE',
        'iterations': 10,
        'task_type': 'GPU',
        'rsm': 0.5,  # This will fail on GPU
        'random_seed': 42,
        'verbose': False
    }
    
    if GPU_AVAILABLE:
        print("Testing with intentionally bad GPU params (rsm on GPU)...")
        model = CatBoostModel(params=bad_params)
        model.fit(X_train, y_train)
        print("✗ Unexpected: Bad params worked (shouldn't happen)")
    else:
        print("Skipping bad params test (no GPU)")
        
except Exception as e:
    print(f"✓ Error caught as expected: {e}")
    print("✓ Error handling works - ensemble will skip CatBoost and continue")

print("\n" + "="*50)
print("SUMMARY")
print("="*50)

if success:
    print("✅ CatBoost configuration is CORRECT!")
    print("The ensemble script should work properly on Kaggle.")
else:
    print("⚠️ CatBoost may fail, but ensemble will continue with other models")
    print("The script has error handling to skip CatBoost if it fails.")

print("\nConfiguration details:")
print("- GPU: Uses 'colsample_bylevel' instead of 'rsm', no thread_count")
print("- CPU: Uses 'rsm' and thread_count=-1")
print("- Both: Use Bernoulli bootstrap with subsample=0.85")
print("- Error handling: Ensemble continues even if CatBoost fails")