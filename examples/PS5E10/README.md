# PS5E10 Competition - Predicting Road Accident Risk

This example demonstrates using TabML for the Kaggle Playground Series Season 5, Episode 10 competition.

## Competition Overview
- **Task**: Regression - Predict the likelihood of accidents on different types of roads (continuous value [0-1])
- **Metric**: Root Mean Squared Error (RMSE)
- **Dataset**: Road accident features with 13 attributes
- **Training samples**: 517,754
- **Test samples**: 172,585

## Setup Instructions

### 1. Create Conda Environment

```bash
# Create a new conda environment for the competition
conda create -n ps5e10 python=3.10 -y

# Activate the environment
conda activate ps5e10

# Install TabML and dependencies
cd ../..  # Navigate to tabml root directory
pip install -e ".[all]"  # Install TabML with all features

# Required dependencies for this example
pip install python-dotenv loguru

# Optional: Additional useful packages
pip install jupyter notebook ipywidgets
```

### 2. Setup MLflow Tracking Server (Optional)

#### Option A: Local MLflow Server (Recommended for this example)

```bash
# Start MLflow server in a separate terminal
tmux
conda activate ps5e10
cd ~/Documents/Github/tabml
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlflow-artifacts
```

#### Option B: Use Existing MLflow Server

If you have MLflow running on your Ubuntu server (as per TabML docs), create `.env` file:

```bash
# Create .env file in PS5E10 directory
cat > .env << EOF
MLFLOW_TRACKING_URI="http://localhost:5000"  # or your server IP
MLFLOW_EXPERIMENT_NAME="PS5E10-Accident-Risk"
RANDOM_SEED=42
EOF
```

### 3. Directory Structure

```
PS5E10/
├── README.md                 # This file
├── .env                      # Environment variables (create this)
├── 01_eda.py                 # Exploratory Data Analysis
├── 02_baseline_models.py    # Train individual models
├── 03_ensemble.py            # Ensemble with hill climbing
├── output/                   # Model outputs and OOF predictions
│   ├── oof_predictions/     # Saved OOF predictions
│   └── submissions/         # Competition submissions
└── notebooks/               # Additional analysis notebooks
```

### Dependencies install for Kaggle

```python
# Install TabML and dependencies
!pip install -q git+https://github.com/wguesdon/tabml.git
!pip install -q xgboost lightgbm catboost
!pip install -q loguru python-dotenv
```

## Workflow

### Step 1: Exploratory Data Analysis
```bash
python 01_eda.py
```
This script will:
- Load and explore the dataset
- Analyze feature distributions and correlations
- Identify patterns in accident risk
- Check for data quality issues
- Visualize relationships between features and target
- Generate adversarial validation to check train/test similarity

### Step 2: Train Baseline Models
```bash
python 02_baseline_models.py
```
This script will:
- Load and preprocess the data
- Train multiple regression models (XGBoost, LightGBM, CatBoost, RandomForest)
- Save OOF predictions for each model
- Track experiments with MLflow (if configured)
- Evaluate models using RMSE metric

### Step 3: Create Ensemble
```bash
python 03_ensemble.py
```
This script will:
- Load all saved OOF predictions
- Compare different ensemble methods
- Use hill climbing to optimize weights for minimum RMSE
- Generate optimized submission

## Data Location

The competition data is located at:
```
../../data/raw/PS5E10/
├── train.csv
├── test.csv
└── sample_submission.csv
```

## Dataset Features

The dataset contains 13 features related to road and environmental conditions:

### Categorical Features:
1. **road_type** - Type of road (urban, rural, highway)
2. **lighting** - Lighting conditions (daylight, dim, night)
3. **weather** - Weather conditions (clear, rainy, foggy)
4. **time_of_day** - Time period (morning, afternoon, evening)

### Boolean Features:
5. **road_signs_present** - Whether road signs are present
6. **public_road** - Whether it's a public road
7. **holiday** - Whether it's a holiday
8. **school_season** - Whether schools are in session

### Numerical Features:
9. **num_lanes** - Number of road lanes (1-4)
10. **curvature** - Road curvature (0-1, where 1 is most curved)
11. **speed_limit** - Posted speed limit
12. **num_reported_accidents** - Number of previously reported accidents

**Target**: `accident_risk` - Continuous variable [0-1] representing accident likelihood

## Key Features for This Competition

### 1. Feature Engineering
   - Interaction features (road_type × weather, lighting × time_of_day)
   - Risk indices (curvature × speed_limit, num_accidents per road_type)
   - Boolean combinations (night + rainy + curved road)
   - Aggregated statistics per categorical groups
   - Polynomial features for numerical variables

### 2. Models
   - XGBoost (multiple configurations)
   - LightGBM (standard and DART)
   - CatBoost (handles categorical features natively)
   - Random Forest
   - Neural Network (TabNet if GPU available)
   - Ridge/Lasso regression for baseline

### 3. Ensemble Techniques
   - Out-of-Fold (OOF) predictions
   - Hill climbing optimization for RMSE
   - Greedy forward selection
   - Stacking with meta-learner
   - Weighted averaging

### 4. MLflow Tracking (Optional)
   - Experiment tracking
   - Model versioning
   - Metric logging (RMSE, MAE, R²)
   - Artifact storage

## Expected Performance

Based on synthetic data regression tasks:
- Single model: 0.08-0.12 RMSE
- Simple ensemble: 0.07-0.10 RMSE
- Optimized ensemble: 0.06-0.09 RMSE

## Tips for Better Performance

### 1. Feature Engineering
   - Create risk score combinations (e.g., bad_conditions = rainy + night + curved)
   - Historical accident rate per road_type/location
   - Speed limit deviations from road type norms
   - Interaction between curvature and speed_limit
   - Time-based patterns (rush hour indicators)

### 2. Model Diversity
   - Use different random seeds
   - Vary tree depths and learning rates
   - Try different feature subsets
   - Include both tree-based and linear models
   - Consider CatBoost's native categorical handling

### 3. Ensemble Optimization
   - Use hill climbing with more iterations for better weights
   - Try greedy forward to identify best model subset
   - Consider stacking with a neural network meta-learner
   - Optimize for RMSE directly

### 4. Domain Knowledge
   - Higher risk typically associated with:
     - Bad weather (rainy, foggy)
     - Night time + dim lighting
     - High curvature + high speed limit
     - More lanes (higher traffic volume)
     - Historical accident patterns
   - Consider clipping extreme predictions to [0, 1]

## Monitoring Progress

Access MLflow UI to track experiments:
```
http://localhost:5000
```

View saved OOF predictions:
```python
from tabml import OOFManager
manager = OOFManager("output/oof_predictions")
summary = manager.list_oofs()
print(summary)
```

## Competition Submission

Final submission will be saved to:
```
output/submissions/submission_final.csv
```

Upload this file to Kaggle to get your leaderboard score.

## Troubleshooting

1. **MLflow Connection Issues**
   - Ensure MLflow server is running
   - Check firewall settings
   - Verify MLFLOW_TRACKING_URI in .env

2. **Memory Issues**
   - Reduce n_estimators in tree models
   - Use smaller n_folds for OOF
   - Process data in chunks

3. **Slow Training**
   - Enable early stopping
   - Use fewer hyperparameter trials
   - Reduce ensemble size

4. **RMSE Not Improving**
   - Check for outliers in predictions
   - Try different feature engineering approaches
   - Ensure proper cross-validation strategy
   - Consider target transformations (logit, etc.)

## Additional Resources

- [Competition Page](https://www.kaggle.com/competitions/playground-series-s5e10)
- [TabML Documentation](../../README.md)
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [Road Safety Analytics Resources](https://www.roadsafety.org/)
