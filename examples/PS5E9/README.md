# PS5E9 Competition - Predicting the Beats-per-Minute of Songs

This example demonstrates using TabML for the Kaggle Playground Series Season 5, Episode 9 competition.

## Competition Overview
- **Task**: Regression - Predict the Beats-per-Minute (BPM) of songs
- **Metric**: Root Mean Squared Error (RMSE)
- **Dataset**: Music/audio features with 9 numerical attributes
- **Training samples**: 524,164
- **Test samples**: 174,722

## Setup Instructions

### 1. Create Conda Environment

```bash
# Create a new conda environment for the competition
conda create -n ps5e9 python=3.10 -y

# Activate the environment
conda activate ps5e9

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
conda activate ps5e9
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
# Create .env file in PS5E9 directory
cat > .env << EOF
MLFLOW_TRACKING_URI="http://localhost:5000"  # or your server IP
MLFLOW_EXPERIMENT_NAME="PS5E9-BPM-Prediction"
RANDOM_SEED=42
EOF
```

### 3. Directory Structure

```
PS5E9/
├── README.md                 # This file
├── .env                      # Environment variables (create this)
├── 01_eda.py                 # Exploratory Data Analysis
├── 02_baseline_models.py    # Train individual models
├── 03_ensemble_hill_climb.py # Hill climbing ensemble
├── 04_final_submission.py   # Generate final submission
├── output/                   # Model outputs and OOF predictions
│   ├── oof_predictions/     # Saved OOF predictions
│   └── submissions/         # Competition submissions
└── notebooks/               # Additional analysis notebooks
```

### Dependencies install for Kaggle

```
pip install -q git+https://github.com/wguesdon/tabml.git
pip install -q xgboost lightgbm catboost
pip install -q torch                     
pip install -q optuna scipy                            
pip install -q loguru python-dotenv
```


## Workflow

### Step 1: Exploratory Data Analysis
```bash
python 01_eda.py
```
This script will:
- Load and explore the dataset
- Analyze feature distributions and correlations
- Identify potential feature engineering opportunities
- Check for data quality issues
- Visualize BPM distribution and relationships with features

### Step 2: Train Baseline Models
```bash
python 02_baseline_models.py
```
This script will:
- Load and preprocess the data
- Train multiple regression models (XGBoost, LightGBM, CatBoost, etc.)
- Save OOF predictions for each model
- Track experiments with MLflow
- Evaluate models using RMSE metric

### Step 3: Create Ensemble with Hill Climbing
```bash
python 03_ensemble_hill_climb.py
```
This script will:
- Load all saved OOF predictions
- Compare different ensemble methods
- Use hill climbing to optimize weights for minimum RMSE
- Generate optimized submission

### Step 4: Generate Final Submission
```bash
python 04_final_submission.py
```

## Data Location

The competition data is located at:
```
../../data/raw/PS5E9/
├── train.csv
├── test.csv
└── sample_submission.csv
```

## Dataset Features

The dataset contains 9 numerical features related to audio/music characteristics:

1. **RhythmScore** - Rhythm-related metric
2. **AudioLoudness** - Loudness measure (negative dB scale)
3. **VocalContent** - Vocal content measure
4. **AcousticQuality** - Acoustic quality score
5. **InstrumentalScore** - Instrumental content score
6. **LivePerformanceLikelihood** - Likelihood of live performance
7. **MoodScore** - Mood-related metric
8. **TrackDurationMs** - Track duration in milliseconds
9. **Energy** - Energy level of the track

**Target**: `BeatsPerMinute` - Continuous variable (typical range: 50-200 BPM)

## Key Features Used in This Example

1. **Feature Engineering**
   - Polynomial features and interactions
   - Log transformations for skewed features
   - Ratio features (e.g., Energy/Duration)
   - Statistical aggregations
   - Domain-specific features (tempo groups, genre indicators)

2. **Models**
   - XGBoost (multiple configurations)
   - LightGBM (standard and DART)
   - CatBoost
   - Random Forest
   - Neural Network (TabNet if GPU available)
   - Ridge/Lasso regression for baseline

3. **Ensemble Techniques**
   - Out-of-Fold (OOF) predictions
   - Hill climbing optimization for RMSE
   - Greedy forward selection
   - Stacking with meta-learner
   - Weighted averaging

4. **MLflow Tracking**
   - Experiment tracking
   - Model versioning
   - Metric logging (RMSE, MAE, R²)
   - Artifact storage

## Expected Performance

Based on typical audio/music regression tasks:
- Single model: 8-12 RMSE
- Simple ensemble: 7-10 RMSE
- Optimized ensemble: 6-9 RMSE

## Tips for Better Performance

1. **Feature Engineering**
   - Create tempo-based categorical features (slow/medium/fast)
   - Interaction between Energy and RhythmScore
   - Duration-normalized features
   - Log transform highly skewed features
   - Polynomial features for non-linear relationships

2. **Model Diversity**
   - Use different random seeds
   - Vary tree depths and learning rates
   - Try different feature subsets
   - Include both tree-based and linear models

3. **Ensemble Optimization**
   - Use hill climbing with more iterations for better weights
   - Try greedy forward to identify best model subset
   - Consider stacking with a neural network meta-learner
   - Optimize for RMSE directly

4. **Domain Knowledge**
   - BPM typically ranges from 60-180 for most music
   - Consider clipping extreme predictions
   - Energy and rhythm often correlate with tempo
   - Duration might indicate genre (longer tracks might be different genres)

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
   - Consider target transformations (log, sqrt)

## Additional Resources

- [Competition Page](https://www.kaggle.com/competitions/playground-series-s5e9)
- [TabML Documentation](../../README.md)
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [Music Information Retrieval Resources](https://musicinformationretrieval.com/)