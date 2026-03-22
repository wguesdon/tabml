"""
Quick test script to verify CatBoost configuration works correctly
Tests both GPU and CPU configurations
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Try importing CatBoost
try:
    from catboost import CatBoostRegressor
    print("✓ CatBoost imported successfully")
except ImportError:
    print("✗ CatBoost not installed")
    exit(1)

# Check GPU availability
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
except:
    GPU_AVAILABLE = False
    print("✓ No GPU detected, will test CPU configuration")

# Generate synthetic data
print("\nGenerating synthetic regression data...")
X, y = make_regression(n_samples=1000, n_features=20, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

# Test configuration 1: CPU with rsm
print("\n" + "="*50)
print("TEST 1: CPU Configuration with rsm")
print("="*50)

cpu_params = {
    'loss_function': 'RMSE',
    'iterations': 100,  # Small number for quick test
    'learning_rate': 0.1,
    'depth': 6,
    'l2_leaf_reg': 3,
    'rsm': 0.5,  # This should work on CPU
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.85,
    'random_seed': 42,
    'verbose': False
}

try:
    model_cpu = CatBoostRegressor(**cpu_params)
    model_cpu.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
    cpu_preds = model_cpu.predict(X_test)
    cpu_rmse = np.sqrt(mean_squared_error(y_test, cpu_preds))
    print(f"✓ CPU config works! Test RMSE: {cpu_rmse:.4f}")
except Exception as e:
    print(f"✗ CPU config failed: {e}")

# Test configuration 2: GPU without rsm (if GPU available)
if GPU_AVAILABLE:
    print("\n" + "="*50)
    print("TEST 2: GPU Configuration without rsm")
    print("="*50)
    
    gpu_params = {
        'loss_function': 'RMSE',
        'iterations': 100,
        'learning_rate': 0.1,
        'depth': 6,
        'l2_leaf_reg': 3,
        'colsample_bylevel': 0.5,  # Use this instead of rsm on GPU
        'bootstrap_type': 'Bernoulli',
        'subsample': 0.85,
        'task_type': 'GPU',
        'devices': '0',
        'random_seed': 42,
        'verbose': False
    }
    
    try:
        model_gpu = CatBoostRegressor(**gpu_params)
        model_gpu.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
        gpu_preds = model_gpu.predict(X_test)
        gpu_rmse = np.sqrt(mean_squared_error(y_test, gpu_preds))
        print(f"✓ GPU config works! Test RMSE: {gpu_rmse:.4f}")
    except Exception as e:
        print(f"✗ GPU config failed: {e}")
    
    # Test configuration 3: GPU with rsm (should fail)
    print("\n" + "="*50)
    print("TEST 3: GPU Configuration with rsm (should fail)")
    print("="*50)
    
    bad_gpu_params = {
        'loss_function': 'RMSE',
        'iterations': 100,
        'learning_rate': 0.1,
        'depth': 6,
        'l2_leaf_reg': 3,
        'rsm': 0.5,  # This should NOT work on GPU
        'bootstrap_type': 'Bernoulli',
        'subsample': 0.85,
        'task_type': 'GPU',
        'devices': '0',
        'random_seed': 42,
        'verbose': False
    }
    
    try:
        model_bad = CatBoostRegressor(**bad_gpu_params)
        model_bad.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
        print(f"✗ Unexpected: GPU with rsm worked (shouldn't happen)")
    except Exception as e:
        print(f"✓ Expected failure: {e}")

# Test the actual configuration from the ensemble script
print("\n" + "="*50)
print("TEST 4: Actual Configuration from Ensemble Script")
print("="*50)

# This mimics what's in the ensemble script
model_params = {
    'loss_function': 'RMSE',
    'iterations': 100,  # Reduced for testing
    'learning_rate': 0.02,
    'depth': 8,
    'l2_leaf_reg': 3,
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.85,
    'random_seed': 42,
    'thread_count': -1,
    'verbose': False
}

if GPU_AVAILABLE:
    model_params['task_type'] = 'GPU'
    model_params['devices'] = '0'
    model_params['colsample_bylevel'] = 0.5
    config_type = "GPU with colsample_bylevel"
else:
    model_params['rsm'] = 0.5
    config_type = "CPU with rsm"

print(f"Testing {config_type} configuration...")

try:
    model_final = CatBoostRegressor(**model_params)
    model_final.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
    final_preds = model_final.predict(X_test)
    final_rmse = np.sqrt(mean_squared_error(y_test, final_preds))
    print(f"✓ Final config works! Test RMSE: {final_rmse:.4f}")
    print(f"✓ Configuration is correct for {config_type}")
except Exception as e:
    print(f"✗ Final config failed: {e}")
    print("This configuration needs to be fixed!")

print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print("The CatBoost configuration has been updated to:")
print("- CPU: Uses 'rsm' parameter for feature sampling")
print("- GPU: Uses 'colsample_bylevel' instead of 'rsm'")
print("- Both: Use Bernoulli bootstrap with subsample=0.85")
print("\nThis ensures compatibility with both CPU and GPU environments.")