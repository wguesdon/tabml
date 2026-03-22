# PS5E10 Quick Start Guide

## Overview
This example demonstrates using TabML for the **Road Accident Risk Prediction** competition (Kaggle Playground Series S5E10).

**Competition Details:**
- **Task**: Regression (predict accident_risk [0-1])
- **Metric**: RMSE (Root Mean Squared Error)
- **Dataset**: 517,754 training samples, 172,585 test samples
- **Features**: 13 features (categorical, boolean, numerical)

## Setup (One-Time)

```bash
# Navigate to TabML root
cd ~/Documents/Github/tabml

# Activate your environment (or create a new one)
conda activate tabml  # or conda create -n ps5e10 python=3.10

# Install TabML with all dependencies
pip install -e ".[all]"

# Navigate to PS5E10 example
cd examples/PS5E10

# Verify setup
python 00_verify_setup.py
```

## Workflow

### Step 1: Exploratory Data Analysis (EDA)

```bash
python 01_eda.py
```

**What it does:**
- Loads train/test data
- Analyzes target distribution (accident_risk)
- Examines all 13 features (categorical, boolean, numerical)
- Identifies high-risk factor combinations
- Creates comprehensive visualizations
- Performs adversarial validation (train vs test similarity)
- Generates feature engineering recommendations

**Output:**
- `output/eda_analysis/` - All visualizations and reports
- Console logs with detailed statistics

**Time:** ~2-5 minutes

---

### Step 2: Train Baseline Models

```bash
python 02_baseline_models.py
```

**What it does:**
- Creates 20+ engineered features (interactions, risk scores, etc.)
- Trains 6 models with 5-fold cross-validation:
  1. XGBoost (Conservative)
  2. XGBoost (Aggressive)
  3. LightGBM (Standard)
  4. LightGBM (DART)
  5. CatBoost
  6. Random Forest
- Generates Out-of-Fold (OOF) predictions for each model
- Saves OOF predictions for ensemble optimization
- Creates individual model submissions

**Output:**
- `output/oof_predictions/` - OOF predictions with metadata
- `output/submissions/` - Individual model submissions
- Console logs with CV scores

**Time:** ~10-20 minutes (depending on hardware)

---

### Step 3: Optimize Ensemble

```bash
python 03_ensemble.py
```

**What it does:**
- Loads all saved OOF predictions
- Tests 6 ensemble methods:
  1. Simple Average
  2. CV Score Weighted
  3. **Hill Climbing** (usually best)
  4. Greedy Forward Selection
  5. Stacking (Ridge meta-learner)
  6. Rank Averaging
- Compares all methods and selects the best
- Creates visualizations (weight distributions, comparisons)
- Generates optimized submissions

**Output:**
- `output/submissions/submission_BEST_*.csv` - Best ensemble
- `output/ensemble_analysis/` - Comparison plots and analysis
- All ensemble method submissions

**Time:** ~2-5 minutes

---

## Expected Results

### Individual Model Performance
- **XGBoost**: ~0.08-0.10 RMSE
- **LightGBM**: ~0.08-0.10 RMSE
- **CatBoost**: ~0.08-0.10 RMSE
- **Random Forest**: ~0.09-0.11 RMSE

### Ensemble Performance
- **Simple Average**: ~0.075-0.095 RMSE
- **Optimized Ensemble**: ~0.070-0.090 RMSE
- **Improvement**: 2-5% better than best individual model

## Key Features of This Example

### 1. Comprehensive Feature Engineering
- **Interaction features**: weather × lighting, road_type × weather
- **Risk scores**: visibility_risk, geometry_risk, safety_score
- **Boolean combinations**: night + rainy, high_curve + high_speed
- **Polynomial features**: curvature², speed², interactions
- **Aggregations**: accidents per lane, speed per lane

### 2. Advanced Ensemble Techniques
- **Hill Climbing**: Iterative weight optimization with simulated annealing
- **Greedy Forward**: Selects best model subset
- **Stacking**: Ridge meta-learner on OOF predictions
- **OOF Predictions**: Prevents overfitting in ensemble

### 3. Robust Validation
- **5-Fold Cross-Validation**: Stratified for reliability
- **OOF Predictions**: Used for ensemble optimization
- **Multiple Metrics**: RMSE, MAE, R² tracked

## File Structure

```
PS5E10/
├── 00_verify_setup.py       # Setup verification
├── 01_eda.py                 # Exploratory Data Analysis
├── 02_baseline_models.py    # Train 6 models
├── 03_ensemble.py            # Optimize ensemble
├── README.md                 # Detailed documentation
├── QUICKSTART.md            # This file
├── .env.example             # Configuration template
└── output/
    ├── oof_predictions/     # Saved OOF predictions
    ├── submissions/         # All submissions
    ├── eda_analysis/        # EDA visualizations
    └── ensemble_analysis/   # Ensemble comparisons
```

## Tips for Better Performance

### 1. Feature Engineering
- Focus on **interaction features** (weather × lighting most important)
- Create **domain-specific risk scores**
- Use **target encoding** for high-cardinality categoricals

### 2. Model Tuning
- Run with more folds (`N_FOLDS=10`) for better estimates
- Increase `n_estimators` in tree models
- Try different random seeds for diversity

### 3. Ensemble Optimization
- Hill climbing usually works best
- Use `n_iterations=5000` for better convergence
- Greedy forward selection identifies best model subset

## Troubleshooting

### Memory Issues
```python
# Reduce models or folds in 02_baseline_models.py
N_FOLDS = 3  # Instead of 5
```

### Slow Training
```python
# Reduce iterations in models
'n_estimators': 500  # Instead of 1000
```

### Poor Results
1. Check EDA output for data quality issues
2. Add more interaction features
3. Increase model diversity (different seeds)
4. Use more folds for better OOF predictions

## Kaggle Submission

After running all scripts:

1. Find best submission:
   ```bash
   ls -lh output/submissions/submission_BEST_*.csv
   ```

2. Upload to Kaggle:
   - Go to competition page
   - Click "Submit Predictions"
   - Upload `submission_BEST_*.csv`
   - View leaderboard score

3. Iterate:
   - Analyze leaderboard feedback
   - Adjust feature engineering
   - Re-train models
   - Submit again

## Advanced Usage

### Custom Model Configuration

Edit `02_baseline_models.py` to add your own models:

```python
# Add your custom XGBoost configuration
custom_xgb_params = {
    'n_estimators': 2000,
    'max_depth': 10,
    'learning_rate': 0.01,
    # ... more params
}

xgb_custom = XGBoostModel(params=custom_xgb_params)
oof_custom, test_custom, cv_custom, score_custom = train_model_with_oof(
    xgb_custom, "XGBoost_Custom", X_train, y_train, X_test, N_FOLDS
)
```

### Custom Feature Engineering

Edit `create_features()` function in `02_baseline_models.py`:

```python
def create_features(df, is_train=True):
    df = df.copy()

    # Add your custom features here
    df['my_custom_feature'] = df['curvature'] * df['speed_limit'] ** 2
    df['another_feature'] = (df['weather'] == 'foggy') & (df['lighting'] == 'night')

    # ... rest of the function
    return df
```

### Use MLflow Tracking (Optional)

```bash
# Start MLflow server
mlflow server --host 0.0.0.0 --port 5000

# Create .env file
cp .env.example .env

# Edit .env with your MLflow URI
# Then run your scripts - they'll log to MLflow
```

## Learning Resources

- **TabML Documentation**: `../../README.md`
- **Competition Page**: https://kaggle.com/competitions/playground-series-s5e10
- **Ensemble Methods**: See `tabml/ensemble.py` for implementation details
- **Feature Engineering**: See `02_baseline_models.py` for examples

## Support

If you encounter issues:

1. Run verification: `python 00_verify_setup.py`
2. Check TabML installation: `pip show tabml`
3. Review error logs in console output
4. Check GitHub issues: https://github.com/wguesdon/tabml/issues

---

**Happy Modeling! 🚀**
