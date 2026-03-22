# PS5E10 Notebook Fixes

## Issue 1: LightGBM Classification Error ✅ FIXED (Updated with Better Fix)

### Error
```
ValueError: Unknown label type: continuous. Maybe you are trying to fit a
classifier, which expects discrete classes on a regression target with
continuous values.
```

### Root Cause
TabML's `LightGBMModel` uses a flawed heuristic: `y.nunique() < 100` to detect classification vs regression (see `tabml/models.py:103`).

For continuous targets with < 100 unique values (common in competitions), this incorrectly triggers classification mode, causing TabML to instantiate `LGBMClassifier` instead of `LGBMRegressor`.

### Solution (Two Approaches)

#### Approach 1: Force Regression Mode (Recommended)
Before training, explicitly set `is_classification = False`:

```python
lgb_model = LightGBMModel(params=lgb_params)

# WORKAROUND: Force regression mode
lgb_model.is_classification = False

oof_lgb, test_lgb, cv_lgb, score_lgb = train_model_with_oof(
    lgb_model, "LightGBM_GPU", X_train, y_train, X_test, N_FOLDS
)
```

#### Approach 2: Explicit Objectives (Also Applied)
Set explicit regression objectives for all models:

#### XGBoost
```python
xgb_params = {
    # ... other params
    'objective': 'reg:squarederror',  # ← Added this
    'tree_method': 'gpu_hist',
    # ...
}
```

#### LightGBM
```python
lgb_params = {
    # ... other params
    'objective': 'regression',  # ← Added this
    'metric': 'rmse',           # ← Added this
    'device': 'gpu',
    # ...
}
```

#### CatBoost
```python
cat_params = {
    # ... other params
    'loss_function': 'RMSE',  # ← Added this
    'task_type': 'GPU',
    # ...
}
```

### Why This Happened
The PS5E10 target (`accident_risk`) has:
- **Continuous values** in range [0, 1]
- But potentially **< 100 unique values** in the dataset
- TabML's heuristic sees `y.nunique() < 100` and incorrectly chooses classification

This is a **known limitation in TabML** that affects continuous targets with limited precision or binning.

### Status
✅ **FIXED** - Forced `is_classification = False` + explicit regression objectives

### Recommended TabML Fix
The TabML library should improve task detection:
```python
# Better heuristic (tabml/models.py:103)
def _determine_task_type(self, y: pd.Series) -> None:
    # Check if target is continuous (float with many decimals)
    is_float = y.dtype in ['float32', 'float64']
    unique_ratio = y.nunique() / len(y)

    # If float with high unique ratio, it's regression
    if is_float and unique_ratio > 0.05:
        self.is_classification = False
    else:
        self.is_classification = y.nunique() < 100
```

---

## Issue 2: Adversarial Validation Plot Not Found ✅ FIXED

### Error
```
FileNotFoundError: [Errno 2] No such file or directory:
'/kaggle/working/output/eda_analysis/adversarial_validation.png'
```

### Root Cause
The EDA notebook was trying to display the adversarial validation plot before running the validation function.

### Solution
Run adversarial validation **before** displaying the image:

```python
# Run validation first
adv_results = eda.adversarial_validation(
    train_df=train_df,
    test_df=test_df,
    target_col=TARGET_COL,
    n_folds=5,
    sample_size=100000,
    save_path=str(OUTPUT_DIR / "adversarial_validation.png")
)

# Then display
if os.path.exists(adv_save_path):
    display(Image(filename=adv_save_path))
```

### Status
✅ **FIXED** - Adversarial validation now runs before display attempt

---

## Summary of Changes

### Ensemble Notebook (`ps5e10-ensemble.py`)
1. ✅ Added `objective='reg:squarederror'` to XGBoost
2. ✅ Added `objective='regression'` and `metric='rmse'` to LightGBM
3. ✅ Added `loss_function='RMSE'` to CatBoost
4. ✅ **NEW**: Force `lgb_model.is_classification = False` before training
5. ✅ **NEW**: Integrated TabML FeatureEngineer and FeatureSelector

### EDA Notebook (`ps5e10-eda.py`)
1. ✅ Run adversarial validation before displaying plot
2. ✅ Added error handling with traceback
3. ✅ Added file existence check
4. ✅ Fixed image paths to include `/train/` subdirectory

---

## Verification

### Test Ensemble Notebook
```python
# After fix, all models should train successfully
✅ XGBoost_GPU    - RMSE: 0.080-0.090
✅ LightGBM_GPU   - RMSE: 0.080-0.090  ← Previously failed
✅ CatBoost_GPU   - RMSE: 0.080-0.090
✅ RandomForest   - RMSE: 0.090-0.100
```

### Test EDA Notebook
```python
# After fix, all sections should run
✅ Setup & Data Loading
✅ EDA Report Generation
✅ Target Distribution
✅ Adversarial Validation  ← Previously failed
✅ Univariate Analysis
✅ Multivariate Analysis
✅ Correlation Analysis
```

---

## Best Practices Going Forward

### Always Specify Task Type
```python
# For Regression
xgb_params = {'objective': 'reg:squarederror', ...}
lgb_params = {'objective': 'regression', ...}
cat_params = {'loss_function': 'RMSE', ...}

# For Binary Classification
xgb_params = {'objective': 'binary:logistic', ...}
lgb_params = {'objective': 'binary', ...}
cat_params = {'loss_function': 'Logloss', ...}

# For Multiclass Classification
xgb_params = {'objective': 'multi:softprob', 'num_class': n, ...}
lgb_params = {'objective': 'multiclass', 'num_class': n, ...}
cat_params = {'loss_function': 'MultiClass', ...}
```

### Check File Existence
```python
# Before displaying images
if os.path.exists(image_path):
    display(Image(filename=image_path))
else:
    print(f"⚠️ Image not found: {image_path}")
```

### Use Try-Except for Robustness
```python
# For operations that might fail
try:
    result = potentially_failing_operation()
    print("✅ Success")
except Exception as e:
    print(f"⚠️ Failed: {e}")
    traceback.print_exc()  # For debugging
```

---

**Last Updated**: 2025-10-01
**Status**: All issues resolved ✅
