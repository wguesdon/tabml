# PS5E8 Competition - Advanced Models Guide

## Overview
This guide explains how to use the advanced TabML scripts to achieve competitive performance (0.977+ AUC) in the PS5E8 competition.

## Scripts

### 1. **04_advanced_models.py** - Advanced Model Training
Trains sophisticated models using advanced feature engineering techniques inspired by high-scoring notebooks.

**Key Features:**
- **Advanced Feature Engineering:**
  - Categorical pair combinations (120+ new features)
  - Numerical features treated as both continuous and categorical
  - Count encoding and target encoding
  - Polynomial and interaction features
  - Log transformations for skewed distributions
  - Original data integration (if available)

- **Multiple XGBoost Configurations:**
  - Conservative model with heavy regularization
  - Aggressive model with deeper trees
  - DART booster for diversity

- **Additional Models:**
  - LightGBM with optimized parameters
  - CatBoost with advanced settings
  - TabNet neural network (if pytorch-tabnet installed)

**Expected Performance:**
- Individual models: 0.970-0.975 AUC
- Simple ensemble: ~0.975-0.976 AUC

### 2. **05_final_ensemble.py** - Final Ensemble Optimization
Combines both baseline and advanced models for optimal performance.

**Ensemble Methods:**
- Equal weights baseline
- CV score weighted
- Top models only
- SciPy optimization
- Optuna optimization (300 trials)
- Hill climbing (10,000 iterations)
- Greedy forward selection
- Stacking with meta-learner
- Meta-ensemble (blend of top 3 methods)

**Expected Performance:**
- Target: 0.977+ AUC
- Best ensemble: ~0.976-0.978 AUC

## Usage Instructions

### Step 1: Prepare Data
```bash
# Ensure you have the PS5E8 data in the correct location
cd examples/PS5E8
mkdir -p ../../data/raw/PS5E8

# Place these files in data/raw/PS5E8/:
# - train.csv
# - test.csv
# - sample_submission.csv
# - original_data.csv (optional but recommended)
```

### Step 2: Install Dependencies
```bash
# Install TabML with all dependencies
pip install -e "../../.[all]"

# For GPU support (optional but recommended):
pip install torch pytorch-tabnet
```

### Step 3: Run Baseline Models (Optional)
If you haven't already trained baseline models:
```bash
python 02_baseline_models.py
```

### Step 4: Train Advanced Models
```bash
python 04_advanced_models.py
```

This will:
- Apply advanced feature engineering
- Train multiple XGBoost, LightGBM, CatBoost, and TabNet models
- Save OOF predictions to `output/oof_predictions_advanced/`
- Generate individual model submissions

### Step 5: Create Final Ensemble
```bash
python 05_final_ensemble.py
```

This will:
- Load all baseline and advanced models
- Try 9 different ensemble methods
- Create visualizations of results
- Save final submissions to `output/submissions_final/`

### Step 6: Submit to Kaggle
The best submission will be saved as:
```
output/submissions_final/submission_BEST.csv
```

## Configuration

### Environment Variables (.env file)
```env
# Random seed for reproducibility
RANDOM_SEED=42

# MLflow tracking (optional)
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=PS5E8-Competition

# Feature engineering flags (in scripts)
USE_ORIGINAL_AS_ROWS=True
USE_ORIGINAL_AS_COLUMNS=True
CREATE_CATEGORICAL_PAIRS=True
TREAT_NUMERICAL_AS_CATEGORICAL=True
```

### Memory Optimization
If you encounter memory issues with large feature sets:

1. Reduce categorical pairs:
   ```python
   # In 04_advanced_models.py, limit pairs
   CREATE_CATEGORICAL_PAIRS = False  # Disable entirely
   # OR
   categorical_cols = categorical_cols[:10]  # Limit to top 10
   ```

2. Use fewer polynomial features:
   ```python
   top_numerical = numerical_cols[:3]  # Reduce from 5 to 3
   ```

3. Reduce model complexity:
   ```python
   # Reduce n_estimators and max_depth
   'n_estimators': 500,  # Instead of 1000+
   'max_depth': 5,       # Instead of 7
   ```

## Performance Tips

### 1. Use Original Data
The original bank marketing dataset significantly improves performance. Place it as `original_data.csv` in the data directory.

### 2. GPU Acceleration
Install CUDA and use GPU-enabled versions:
```bash
# For XGBoost GPU support
pip install xgboost[cuda]

# In model params:
'tree_method': 'gpu_hist',
'predictor': 'gpu_predictor',
```

### 3. Parallel Processing
The OOF generation uses parallel processing by default. Ensure you have sufficient CPU cores.

### 4. Hyperparameter Tuning
For even better results, use Optuna to tune hyperparameters:
```python
from tabml.optimization import OptunaOptimizer

optimizer = OptunaOptimizer(
    model_class=XGBoostModel,
    n_trials=100,
    cv_folds=5
)
best_params = optimizer.optimize(X_train, y_train)
```

## Troubleshooting

### Issue: TabNet fails to install
**Solution:** TabNet requires PyTorch. Install it first:
```bash
pip install torch
pip install pytorch-tabnet
```

### Issue: Out of memory during feature engineering
**Solution:** Reduce feature complexity or use chunking:
```python
# Process in chunks
chunk_size = 10000
for i in range(0, len(X_train), chunk_size):
    X_train_chunk = X_train[i:i+chunk_size]
    # Process chunk
```

### Issue: Models not improving beyond 0.975
**Solution:** Ensure you have:
1. Original data for additional features
2. All categorical pairs enabled
3. Sufficient iterations for optimization (10,000+ for hill climbing)

## Expected Results

| Stage | Expected AUC | Notes |
|-------|-------------|-------|
| Baseline Models | 0.965-0.967 | Standard feature engineering |
| Advanced Models | 0.970-0.975 | With advanced features |
| Final Ensemble | 0.976-0.978 | Combining all models |

## Competition Tips

1. **Feature Engineering is Key:** The advanced feature engineering (especially categorical pairs and original data usage) provides the biggest boost.

2. **Diversity Matters:** Having diverse models (XGBoost, LightGBM, CatBoost, Neural Networks) improves ensemble performance.

3. **Optimization Patience:** Hill climbing with 10,000+ iterations often finds better weights than quick methods.

4. **Validation Strategy:** The 5-fold CV used here is robust. Consider also trying:
   - Repeated stratified K-fold
   - Group K-fold if there are natural groups
   - Time-based splits if temporal

5. **Post-Processing:** Consider:
   - Rank averaging instead of probability averaging
   - Calibration techniques (Platt scaling, isotonic regression)
   - Threshold optimization for specific metrics

## Next Steps

For even better performance (0.978+), consider:

1. **More Advanced Neural Networks:**
   - Custom MLP with embeddings
   - SAINT (Self-Attention and Intersample Attention Transformer)
   - FT-Transformer

2. **Advanced Feature Selection:**
   - Recursive feature elimination
   - Permutation importance
   - SHAP-based selection

3. **Pseudo-Labeling:**
   - Use high-confidence test predictions to augment training data

4. **Advanced Stacking:**
   - Multi-level stacking
   - Cross-validated meta-features

## Support

For issues or questions:
- Check the main TabML documentation
- Review the example notebooks in `example_notebooks/`
- Open an issue on the TabML GitHub repository

Good luck with the competition! 🏆