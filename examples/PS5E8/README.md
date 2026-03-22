# PS5E8 Competition - Bank Term Deposit Prediction

This example demonstrates using TabML for the Kaggle Playground Series Season 5, Episode 8 competition.

## Competition Overview
- **Task**: Binary Classification - Predict whether a client will subscribe to a bank term deposit
- **Metric**: ROC AUC
- **Dataset**: Bank marketing data with features like age, job, marital status, education, etc.

## Setup Instructions

### 1. Create Conda Environment

```bash
# Create a new conda environment for the competition
conda create -n ps5e8 python=3.10 -y

# Activate the environment
conda activate ps5e8

# Install TabML and dependencies
cd ../..  # Navigate to tabml root directory
pip install -e ".[all]"  # Install TabML with all features

# Required dependencies for this example
pip install python-dotenv loguru

# Optional: Additional useful packages
pip install jupyter notebook ipywidgets
```

### 2. Setup MLflow Tracking Server

#### Option A: Local MLflow Server (Recommended for this example)

```bash
# Start MLflow server in a separate terminal
tmux
conda activate ps5e8
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
# Create .env file in PS5E8 directory
cat > .env << EOF
MLFLOW_TRACKING_URI="http://localhost:5000"  # or your server IP
MLFLOW_EXPERIMENT_NAME="PS5E8-Bank-Deposit"
RANDOM_SEED=42
EOF
```

### 3. Directory Structure

```
PS5E8/
├── README.md                 # This file
├── .env                      # Environment variables (create this)
├── 01_eda.ipynb             # Exploratory Data Analysis
├── 02_baseline_models.py    # Train individual models
├── 03_ensemble_hill_climb.py # Hill climbing ensemble
├── 04_final_submission.py   # Generate final submission
├── output/                   # Model outputs and OOF predictions
│   ├── oof_predictions/     # Saved OOF predictions
│   └── submissions/         # Competition submissions
└── notebooks/               # Additional analysis notebooks
```

## Workflow

### Step 1: Exploratory Data Analysis
```bash
python 01_eda.py
```

### Step 2: Train Baseline Models
```bash
python 02_baseline_models.py
```
This script will:
- Load and preprocess the data
- Train multiple models (XGBoost, LightGBM, CatBoost, etc.)
- Save OOF predictions for each model
- Track experiments with MLflow

### Step 3: Create Ensemble with Hill Climbing
```bash
python 03_ensemble_hill_climb.py
```
This script will:
- Load all saved OOF predictions
- Compare different ensemble methods
- Use hill climbing to optimize weights
- Generate optimized submission

### Step 4: Generate Final Submission
```bash
python 04_final_submission.py
```

## Data Location

The competition data is located at:
```
../../data/raw/ PS5E8/
├── train.csv
├── test.csv
└── sample_submission.csv
```

## Key Features Used in This Example

1. **Feature Engineering**
   - Categorical encoding for job, marital, education
   - Numerical scaling for age, balance, duration
   - Interaction features
   - Target encoding for high-cardinality features

2. **Models**
   - XGBoost (multiple configurations)
   - LightGBM (standard and DART)
   - CatBoost (with categorical features)
   - Random Forest
   - Neural Network (TabNet if GPU available)

3. **Ensemble Techniques**
   - Out-of-Fold (OOF) predictions
   - Hill climbing optimization
   - Greedy forward selection
   - Stacking with meta-learner

4. **MLflow Tracking**
   - Experiment tracking
   - Model versioning
   - Metric logging
   - Artifact storage

## Expected Performance

Based on similar bank marketing datasets:
- Single model: 0.88-0.91 AUC
- Simple ensemble: 0.91-0.93 AUC
- Optimized ensemble: 0.92-0.94 AUC

## Tips for Better Performance

1. **Feature Engineering**
   - Create ratio features (e.g., balance/age)
   - Aggregate features by job type
   - Time-based features from contact month/day

2. **Model Diversity**
   - Use different random seeds
   - Vary tree depths and learning rates
   - Try different feature subsets

3. **Ensemble Optimization**
   - Use hill climbing with more iterations for better weights
   - Try greedy forward to identify best model subset
   - Consider stacking with a neural network meta-learner

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

## Additional Resources

- [Competition Page](https://www.kaggle.com/competitions/playground-series-s5e8)
- [TabML Documentation](../../README.md)
- [MLflow Documentation](https://mlflow.org/docs/latest/)