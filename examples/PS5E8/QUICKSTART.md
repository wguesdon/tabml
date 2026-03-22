# PS5E8 Competition - Quick Start Guide

## Fix for MLflow Error

If you encounter the MLflow `artifact_location` error, the code has been updated to handle this. Make sure you have the latest version.

## Quick Setup (if conda environment already exists)

```bash
# Activate environment
conda activate ps5e8

# Install missing dependencies
pip install python-dotenv loguru

# Make sure you're in the PS5E8 directory
cd ~/Documents/Github/tabml/examples/PS5E8
```

## Start MLflow Server (separate terminal)

```bash
mlflow server --host 0.0.0.0 --port 5000
```

## Run the Pipeline

```bash
# 1. Exploratory Data Analysis
python 01_eda.py

# 2. Train baseline models (saves OOF predictions)
python 02_baseline_models.py

# 3. Optimize ensemble with hill climbing
python 03_ensemble_hill_climb.py
```

## Expected Output

After running all scripts, you'll have:
- `output/oof_predictions/` - Saved OOF predictions from each model
- `output/submissions/` - Competition submissions
- `output/plots/` - Visualization plots
- MLflow tracking at http://localhost:5000

## Troubleshooting

### ModuleNotFoundError: No module named 'dotenv'
```bash
pip install python-dotenv
```

### ModuleNotFoundError: No module named 'loguru'
```bash
pip install loguru
```

### MLflow connection error
Make sure MLflow server is running:
```bash
mlflow server --host 0.0.0.0 --port 5000
```

### Memory issues
Reduce model complexity in `02_baseline_models.py`:
- Decrease `n_estimators`
- Reduce `N_FOLDS` from 5 to 3
- Train fewer models

## Competition Submission

Your best submission will be at:
```
output/submissions/submission_best_ensemble.csv
```

Upload this file to Kaggle to get your leaderboard score!