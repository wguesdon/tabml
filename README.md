# TabML

A Python framework for tabular machine learning that provides a unified interface for common ML workflows.

## Overview

TabML handles the standard tabular ML pipeline: data loading, feature engineering, model training, and ensemble creation. It wraps popular libraries (XGBoost, LightGBM, CatBoost) with consistent APIs and includes utilities for out-of-fold predictions and stacking.

## Installation

### Basic Installation
```bash
git clone https://github.com/wguesdon/tabml.git
cd tabml
conda create -n tabml python=3.10 -y
conda activate tabml
pip install -e ".[all]"
```

### Installation with Optional Dependencies
```bash
# Install with specific extras
pip install -e ".[autogluon]"  # AutoGluon support
pip install -e ".[gpu]"  # GPU support for neural networks
pip install -e ".[tracking]"  # MLflow and W&B tracking

# Install with all optional dependencies
pip install -e ".[all]"  # Full installation
```

### Installing AutoGluon
```bash
# Option 1: Install TabML with AutoGluon
pip install -e ".[autogluon]"

# Option 2: Install AutoGluon separately
pip install autogluon

# Option 3: Install AutoGluon with GPU support
pip install autogluon[torch]
```

### Optional Dependencies
- `.[nlp]` - NLTK for text processing
- `.[gpu]` - PyTorch and TabNet support
- `.[tracking]` - Experiment tracking (MLflow, W&B, TensorBoard)
- `.[autogluon]` - AutoGluon for automatic machine learning
- `.[dev]` - Testing and development tools
- `.[all]` - All optional dependencies

### Installing on Kaggle
```python
# Install directly from GitHub in a Kaggle notebook
!pip install git+https://github.com/wguesdon/tabml.git

# With all dependencies
!pip install git+https://github.com/wguesdon/tabml.git#egg=tabml[all]

# With specific extras (e.g., AutoGluon)
!pip install git+https://github.com/wguesdon/tabml.git#egg=tabml[autogluon]
```

**Note:** Kaggle notebooks include most core dependencies (pandas, numpy, scikit-learn, xgboost, lightgbm, catboost) pre-installed, so the basic installation should work without issues.

## Quick Start

```python
from tabml import TabularPipeline

# Run complete pipeline
pipeline = TabularPipeline(data_dir="data")
pipeline.load_data(train_file="train.csv", test_file="test.csv")
submission = pipeline.run_full_pipeline()
```

## Core Components

### Data Loading
```python
from tabml import DataLoader

loader = DataLoader(data_dir="data")
train_df, test_df = loader.load_data(
    train_file="train.csv",
    test_file="test.csv",
    target_column="target"  # Auto-detected if not specified
)
```

### Feature Engineering
```python
from tabml import FeatureEngineer, AdvancedFeatureEngineer

# Basic features
engineer = FeatureEngineer(
    numeric_impute_strategy='median',
    categorical_impute_strategy='constant',
    scaling_method='standard'
)
X_transformed = engineer.fit_transform(X_train, y_train)

# Advanced features (text, dates)
adv_engineer = AdvancedFeatureEngineer()
X_text = adv_engineer.create_text_features(X, text_columns=['description'])
X_date = adv_engineer.create_date_features(X, date_columns=['timestamp'])
```

### Model Training
```python
from tabml import XGBoostModel, LightGBMModel, CatBoostModel, AutoGluonModel

# Individual models
xgb = XGBoostModel(params={'n_estimators': 500, 'max_depth': 6})
xgb.fit(X_train, y_train)
predictions = xgb.predict(X_test)

# AutoGluon for automatic model selection
autogluon = AutoGluonModel(params={'time_limit': 600, 'presets': 'best_quality'})
autogluon.fit(X_train, y_train)
predictions = autogluon.predict(X_test)

# Multiple models
models = [
    XGBoostModel(params={'n_estimators': 500}),
    LightGBMModel(params={'n_estimators': 500}),
    CatBoostModel(params={'iterations': 500}),
    AutoGluonModel(params={'time_limit': 300})
]
```

### Ensemble Methods
```python
from tabml import OOFEnsemble, OOFManager

# Out-of-fold predictions
ensemble = OOFEnsemble(task_type='classification')
oof_preds = ensemble.get_oof_predictions(models, X_train, y_train, n_folds=5)

# Weight optimization methods
weights = ensemble.optimize_weights(oof_preds, y_train, method='hill_climbing', 
                                   n_iterations=2000, patience=200)
# Other methods: 'scipy', 'optuna', 'grid', 'greedy_forward'

# Save and load OOF predictions
manager = OOFManager(output_dir="output/competition")
manager.save_oof(oof_preds, model_name="xgboost_v1", cv_score=0.85, test_predictions=test_preds)
best_oofs = manager.load_all_oofs(top_k=10, min_cv_score=0.85)

# Stacking
ensemble.fit_stacking(oof_preds, y_train)
final_predictions = ensemble.predict_stacking(test_preds)
```

## Available Models

### Tree-based
- `XGBoostModel` - XGBoost gradient boosting
- `LightGBMModel` - LightGBM gradient boosting
- `CatBoostModel` - CatBoost with categorical feature support
- `RandomForestModel` - Random Forest classifier/regressor

### AutoML
- `AutoGluonModel` - Automatic model selection and ensembling (requires autogluon)

### Neural Networks
- `TabNetModel` - Attention-based neural network for tabular data (requires pytorch-tabnet)

### Linear
- `RidgeModel` - L2 regularized regression
- `LinearModel` - Standard linear regression

### Ensembles
- `VotingEnsemble` - Simple voting ensemble
- `OOFEnsemble` - Out-of-fold stacking ensemble
- `AutoEnsemble` - Automatic ensemble selection

## Features

### Feature Engineering
- Missing value imputation (mean, median, mode, constant)
- Encoding (one-hot, label, target encoding)
- Scaling (standard, minmax, robust)
- Text features (TF-IDF, text statistics)
- Date features (year, month, day, cyclical encoding)
- Polynomial and interaction features

### Cross-Validation
- Stratified K-Fold
- Group K-Fold
- Time Series Split
- Repeated K-Fold

### Optimization
- Hyperparameter tuning with Optuna
- Weight optimization for ensembles (scipy, optuna, grid, hill climbing, greedy forward)
- Hill climbing with adaptive learning rate and simulated annealing
- Greedy forward selection for model pruning
- Early stopping callbacks

### Utilities
- Automatic target column detection
- Feature selection (mutual information, tree-based, RFE)
- Data validation and type inference
- Visualization tools for EDA
- OOF predictions manager for saving/loading across sessions
- Competition-ready submission generation

### Experiment Tracking
- MLflow integration for comprehensive tracking
- Weights & Biases (wandb) support
- TensorBoard logging
- Model versioning and registry

## Project Structure

```
tabml/
├── tabml/                      # Main package
│   ├── data.py                 # Data loading
│   ├── features.py             # Feature engineering
│   ├── advanced_features.py    # Text/date features
│   ├── models.py               # Model implementations
│   ├── ensemble.py             # Ensemble methods
│   ├── pipeline.py             # End-to-end pipeline
│   ├── evaluate.py             # Cross-validation
│   └── utils.py                # Helper functions
├── examples/                   # Example scripts
├── tests/                      # Unit tests
└── data/                       # Sample datasets
```

## Experiment Tracking with MLflow

TabML provides comprehensive MLflow integration for experiment tracking, model versioning, and dataset management. Supports both local MLflow servers and DagsHub MLflow.

📚 **[Complete Ubuntu MLflow Server Setup Guide](docs/mlflow_ubuntu_setup.md)** - Detailed instructions for setting up MLflow on Ubuntu with virtual environment.

### Setup MLflow Connection

#### Option A: Local MLflow Server (localhost)
```bash
# Start local MLflow server on your machine
mlflow server --host 0.0.0.0 --port 5000
```

#### Option B: Network MLflow Server (Ubuntu server on local network)
```bash
# On your Ubuntu server:
# 1. Start MLflow server accessible on network
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:///home/username/mlflow/mlflow.db \
    --default-artifact-root /home/username/mlflow/artifacts

# 2. Make it a systemd service for auto-start (optional)
# Create /etc/systemd/system/mlflow.service:
sudo tee /etc/systemd/system/mlflow.service << EOF
[Unit]
Description=MLflow Server
After=network.target

[Service]
Type=simple
User=username
WorkingDirectory=/home/username/mlflow
ExecStart=/usr/local/bin/mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///home/username/mlflow/mlflow.db --default-artifact-root /home/username/mlflow/artifacts
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable mlflow
sudo systemctl start mlflow

# 3. Access from other machines on network:
# Use http://SERVER_IP:5000 or http://SERVER_HOSTNAME:5000
# Example: http://192.168.1.100:5000 or http://ml-server.local:5000
```

#### Option C: DagsHub MLflow (Cloud hosting)
1. Create account at [DagsHub.com](https://dagshub.com)
2. Create a new repository or connect existing GitHub repo
3. Get your token from [Settings → Tokens](https://dagshub.com/user/settings/tokens)
4. Set tracking URI: `https://dagshub.com/USERNAME/REPO_NAME.mlflow`

**Important**: DagsHub requires MLflow 2.x (not 3.x). TabML is configured for compatibility.

### Configure Authentication

**For Local/Network Server**: No authentication needed (unless you've configured it).

```bash
# In .env file:
# For localhost:
MLFLOW_TRACKING_URI="http://localhost:5000"

# For Ubuntu server on your network:
MLFLOW_TRACKING_URI="http://192.168.1.100:5000"  # Use your server's IP
# or
MLFLOW_TRACKING_URI="http://ml-server.local:5000"  # Use hostname if configured
```

**For DagsHub**: Requires authentication:

   **Method A: Using .env file (Recommended)**
   ```bash
   # Copy the example .env file
   cp .env.example .env
   
   # Edit .env with your DagsHub credentials
   MLFLOW_TRACKING_URI="https://dagshub.com/USERNAME/REPO.mlflow"
   MLFLOW_TRACKING_USERNAME="your-dagshub-username"
   MLFLOW_TRACKING_PASSWORD="your-dagshub-token"
   ```
   
   Then load it in your Python code:
   ```python
   from dotenv import load_dotenv
   import os
   
   # Load environment variables from .env file
   load_dotenv()
   
   from tabml import MLflowTracker
   
   # Will automatically use MLFLOW_TRACKING_URI from .env
   tracker = MLflowTracker(
       experiment_name="my-experiment",
       tracking_uri=os.getenv("MLFLOW_TRACKING_URI")
   )
   ```
   
   **Method B: Export to shell environment**
   ```bash
   # For DagsHub
   export MLFLOW_TRACKING_URI="https://dagshub.com/USERNAME/REPO.mlflow"
   export MLFLOW_TRACKING_USERNAME="your-dagshub-username"
   export MLFLOW_TRACKING_PASSWORD="your-dagshub-token"
   
   # For local server
   export MLFLOW_TRACKING_URI="http://localhost:5000"
   ```

### Use MLflow in TabML
```python
from tabml import MLflowTracker, MLflowCallback, MLflowModelRegistry
import os

# Initialize tracker (uses env vars for DagsHub auth)
tracker = MLflowTracker(
    experiment_name="tabml-experiment",
    tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),  # DagsHub or local
    tags={"framework": "tabml", "task": "classification"}
)

# Start run and log dataset
tracker.start_run(run_name="baseline-model")
tracker.log_dataset(X_train, "training_data", version="1.0")
tracker.log_params({"model_type": "xgboost", "n_estimators": 500})

# Train with MLflow callback
from tabml.training import EnhancedTrainer, EarlyStoppingCallback

trainer = EnhancedTrainer(
    callbacks=[
        MLflowCallback(
            experiment_name="tabml-experiment",
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI")
        ),
        EarlyStoppingCallback(patience=10)
    ]
)

# Log model and metrics
tracker.log_model(model, "xgboost_model", registered_model_name="BestModel")
tracker.log_metrics({"accuracy": 0.95, "auc": 0.98})

# Model registry
registry = MLflowModelRegistry(tracking_uri=os.getenv("MLFLOW_TRACKING_URI"))
version = registry.register_model(
    run_id=tracker.current_run.info.run_id,
    model_name="BestModel"
)
registry.transition_model_stage("BestModel", version, "Production")

tracker.end_run()
```

**DagsHub Benefits**:
- Free hosting for public repositories
- Git integration for data versioning
- Collaboration features
- No server maintenance needed

**Note**: For remote MLflow servers, ensure proper authentication is configured. Use environment variables or `.env` files (with python-dotenv) for configuration.

## Examples

See the `examples/` directory for complete workflows:
- `spaceship_titanic_solution.py` - Kaggle competition example
- `oof_ensemble_example.py` - Advanced ensemble techniques
- `tabml_example.py` - Feature tour
- `mlflow_tracking_example.py` - MLflow integration demo

## Command Line Interface

```bash
# Train models
tabml train --data-dir data/titanic

# Run with hyperparameter optimization
tabml train --data-dir data/titanic --optimize

# Exploratory data analysis
tabml eda --data-dir data/titanic --target Survived
```

## Requirements

- Python 3.8+
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- xgboost >= 2.0.0
- lightgbm >= 4.0.0
- catboost >= 1.2
- optuna >= 3.3.0

## License

MIT License - see LICENSE file for details.

## Citation

```bibtex
@software{tabml,
  author = {William Guesdon},
  title = {TabML: Tabular Machine Learning Package},
  year = {2025},
  url = {https://github.com/wguesdon/tabml}
}
```