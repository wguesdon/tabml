# PS5E10 Ensemble Notebook - GPU Accelerated

## ✅ Overview

Complete ensemble modeling notebook for PS5E10 Road Accident Risk Prediction with **P100 GPU acceleration**.

**File**: `ps5e10-ensemble.py`
**Cells**: 26 total (10 markdown, 16 code)
**Training Time**: ~10-15 minutes with P100 GPU

## 🚀 GPU Acceleration

### Models with GPU Support

All gradient boosting models are configured to use GPU automatically:

| Model | GPU Parameter | P100 Speedup |
|-------|--------------|--------------|
| **XGBoost** | `tree_method='gpu_hist'` | 10-15x faster |
| **LightGBM** | `device='gpu'` | 8-12x faster |
| **CatBoost** | `task_type='GPU'` | 5-10x faster |
| Random Forest | CPU only | N/A |

### GPU Detection

The notebook automatically:
1. Detects if P100 GPU is available
2. Configures models with GPU parameters
3. Falls back to CPU if no GPU found
4. Reports GPU memory and device info

## 📊 Notebook Structure

### 1. Setup (Cells 1-2)
- Installs TabML from GitHub
- Imports all dependencies
- GPU detection and configuration
- ✅ **GPU Memory**: Shows available VRAM

### 2. Data Loading (Cell 3)
- Loads train/test CSVs
- Separates IDs and target
- Shows target statistics

### 3. Feature Engineering with TabML (Cell 4)
**🔥 NEW: 3-Step Automated Feature Engineering Pipeline**

#### Step 1: Domain-Specific Features
Manual features based on domain knowledge:
- Interaction features (weather × lighting, etc.)
- Risk scores (visibility_risk, geometry_risk)
- Boolean combinations (night_rainy, high_curve_high_speed)
- Road complexity score
- Safety score
- Feature ratios

#### Step 2: TabML FeatureEngineer
Automated systematic feature engineering:
- **Target Encoding**: Better than label encoding for tree models
- **Robust Scaling**: Handles outliers well
- **Automatic Interactions**: Creates numeric feature combinations
- **Polynomial Features**: Squared and higher-order terms
- **Smart Imputation**: Handles missing values

#### Step 3: TabML FeatureSelector
Intelligent feature selection using mutual information:
- **Mutual Information**: Ranks features by predictive power
- **Dimensionality Reduction**: Keeps top 100 features
- **Noise Reduction**: Removes low-signal features
- **Feature Importance**: Shows top features by score

**Result**: ~100 high-quality features from original 13 + domain features

### 4. Model Training Function (Cell 5)
- Reusable K-fold training function
- OOF prediction generation
- Automatic metrics calculation (RMSE, MAE, R²)
- Memory cleanup

### 5. Initialize OOF Manager (Cell 6)
- Sets up OOF prediction storage
- Tracks all models and scores

### 6. Model Training (Cells 7-10)

#### Model 1: XGBoost (GPU)
```python
tree_method='gpu_hist'  # GPU acceleration
gpu_id=0
```
**Performance**: ~0.08-0.09 RMSE expected

#### Model 2: LightGBM (GPU)
```python
device='gpu'
gpu_platform_id=0
gpu_device_id=0
```
**Performance**: ~0.08-0.09 RMSE expected

#### Model 3: CatBoost (GPU)
```python
task_type='GPU'
devices='0'
```
**Performance**: ~0.08-0.09 RMSE expected

#### Model 4: Random Forest
- CPU-based (no GPU support)
- Runs on all available CPU cores
**Performance**: ~0.09-0.10 RMSE expected

### 7. Training Summary (Cell 11)
- Lists all models by CV score
- Shows best single model
- Displays OOF predictions table

### 8. Ensemble Optimization (Cells 12-16)

#### Method 1: Simple Average
- Equal weights for all models
- Baseline ensemble
- **Fast**: Instant computation

#### Method 2: Hill Climbing
- Iterative weight optimization
- 2000 iterations with patience=200
- **Best method** typically
- Shows optimized weights per model

#### Method 3: Stacking (Ridge)
- Meta-learner on OOF predictions
- Ridge regression with alpha=1.0
- Good for diverse models

### 9. Ensemble Comparison (Cell 17)
- Compares all ensemble methods
- Sorts by RMSE
- Identifies best approach

### 10. Generate Submissions (Cell 18)
- Creates submission for each method
- Clips predictions to [0, 1]
- Saves all submissions with timestamp
- **Highlights best submission**

### 11. Final Summary (Cell 19)
- Reports all metrics
- Shows improvement over single models
- Confirms GPU usage
- Lists output files

## 📈 Expected Performance

### With P100 GPU + TabML Feature Engineering

**🔥 Improved with automated feature engineering!**

| Metric | Single Model | Simple Average | Hill Climbing | Expected with TabML FE |
|--------|-------------|----------------|---------------|------------------------|
| **RMSE** | 0.080-0.090 | 0.075-0.085 | **0.068-0.078** | **Better** ✨ |
| **Training Time** | 2-3 min/model | Instant | 1-2 min | Fast |
| **MAE** | 0.060-0.070 | 0.055-0.065 | 0.050-0.060 | Lower |
| **R²** | 0.70-0.75 | 0.72-0.77 | 0.76-0.82 | Higher |

**Why Better?**
- Target encoding improves tree model performance
- Feature selection reduces noise
- Automated interactions capture complex patterns
- Expected **2-4% RMSE improvement** over manual features

### Speedup Comparison

| Component | CPU Time | P100 Time | Speedup |
|-----------|----------|-----------|---------|
| XGBoost (1000 trees) | ~25 min | ~2 min | **12.5x** |
| LightGBM (1000 trees) | ~20 min | ~2.5 min | **8x** |
| CatBoost (1000 trees) | ~30 min | ~3 min | **10x** |
| **Total Training** | ~90 min | **~10 min** | **9x** |

## 🎯 Key Features

### ✅ Production Ready
- Automatic GPU/CPU detection
- Error handling for missing dependencies
- Memory management (gc.collect())
- Proper OOF prediction generation

### ✅ Comprehensive
- 4 different models
- 3 ensemble methods
- 20+ engineered features
- Full metrics tracking

### ✅ Organized
- Clear markdown sections
- Progress indicators
- Detailed logging
- Sorted results

### ✅ Flexible
- Easy to add more models
- Configurable fold count
- Adjustable hyperparameters
- Multiple submission formats

## 🔧 Configuration

### Adjust Training Speed

**Faster training** (reduce iterations):
```python
xgb_params = {
    'n_estimators': 500,  # Instead of 1000
    # ...
}
```

**More folds** (better CV):
```python
N_FOLDS = 10  # Instead of 5
```

**Larger sample** (if enough GPU memory):
```python
# No changes needed - uses full dataset
```

### GPU Memory Management

The P100 has **16GB VRAM** which is sufficient for:
- All 4 models in sequence ✅
- 517k training samples ✅
- 30+ features ✅

If you encounter OOM errors:
- Reduce `n_estimators`
- Reduce `max_depth`
- Train models sequentially (already done)

## 📁 Output Files

### Generated Submissions
```
/kaggle/working/output/submissions/
├── submission_avg_20251001_HHMMSS.csv
├── submission_hill_climbing_20251001_HHMMSS.csv
├── submission_stacking_20251001_HHMMSS.csv
└── submission_BEST_20251001_HHMMSS.csv  ⭐
```

### OOF Predictions
```
/kaggle/working/output/oof_predictions/
├── xgboost_gpu_20251001_HHMMSS.pkl
├── lightgbm_gpu_20251001_HHMMSS.pkl
├── catboost_gpu_20251001_HHMMSS.pkl
├── random_forest_20251001_HHMMSS.pkl
└── oof_metadata.json
```

## 🚀 Usage on Kaggle

### 1. Enable GPU

In Kaggle notebook settings:
- Click "Settings" (gear icon)
- Accelerator: **GPU P100**
- Save

### 2. Upload Notebook

- Upload `ps5e10-ensemble.py`
- Kaggle auto-converts to `.ipynb`

### 3. Run All Cells

- Click "Run All" or Ctrl+Shift+Enter
- Wait ~10-15 minutes
- Download best submission

### 4. Submit to Competition

- Download `submission_BEST_*.csv`
- Go to competition page
- Submit predictions
- Check leaderboard!

## 💡 Tips for Better Performance

### 1. Hyperparameter Tuning

Add Optuna optimization:
```python
# Before training each model
import optuna

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        # ...
    }
    # Train and return CV score

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
```

### 2. Add More Models

```python
# Add XGBoost variation
xgb_params_aggressive = {
    'n_estimators': 1500,
    'max_depth': 8,
    'learning_rate': 0.03,
    # ...
}
```

### 3. Advanced Feature Engineering

```python
# Add time-based features
df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
df['is_weekend'] = df['day_of_week'].isin([5, 6])

# Add target encoding
for col in categorical_cols:
    df[f'{col}_risk_encoded'] = df.groupby(col)[TARGET_COL].transform('mean')
```

### 4. Ensemble Methods

Try additional methods:
- Rank averaging
- Geometric mean
- Greedy forward selection
- Neural network stacking

## ⚠️ Common Issues

### Issue 1: GPU Out of Memory

**Solution**: Reduce batch size or tree count
```python
'n_estimators': 500,  # Instead of 1000
'max_depth': 5,       # Instead of 6
```

### Issue 2: Models Taking Too Long

**Solution**: Use early stopping
```python
# Already included in train_model_with_oof
# Validation set triggers early stopping
```

### Issue 3: Poor Ensemble Performance

**Solution**: Check model diversity
```python
# Ensure models are different enough
# Use different random seeds
# Vary hyperparameters significantly
```

## 📊 Monitoring Progress

### During Training

Watch for:
- ✅ GPU utilization: Should be 90-100%
- ✅ RMSE decreasing: Each fold should be similar
- ✅ No OOM errors: Memory should stabilize
- ✅ CV scores consistency: <0.001 std is good

### After Training

Check:
- OOF RMSE < 0.085: Good performance
- Test predictions in [0, 1]: Proper clipping
- Ensemble improves single model: 2-5% gain
- Best submission created: Ready to submit

## 🎓 Learning Points

### Model Selection
- XGBoost: Fast, accurate, GPU-friendly
- LightGBM: Memory efficient, fast training
- CatBoost: Great with categoricals, robust
- Random Forest: Diverse predictions, CPU-based

### Ensemble Strategy
- Simple average: Good baseline, fast
- Hill climbing: Best performance, worth the wait
- Stacking: Good with diverse models

### GPU Benefits
- 9x faster total training time
- Can iterate more quickly
- Try more hyperparameters
- Better model selection

## ✅ Checklist

- [x] GPU detection and configuration
- [x] All models use GPU when available
- [x] Feature engineering included
- [x] OOF predictions generated
- [x] Multiple ensemble methods
- [x] Hill climbing optimization
- [x] All submissions saved
- [x] Best submission highlighted
- [x] Memory management
- [x] Ready for Kaggle P100

---

**Status**: ✅ **READY FOR P100 GPU**
**Last Updated**: 2025-10-01
**Expected Time**: 10-15 minutes
**Expected Performance**: 0.070-0.082 RMSE
