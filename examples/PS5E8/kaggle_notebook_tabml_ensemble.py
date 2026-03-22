# # TabML: Ensemble Learning for Tabular Data
# 
# This notebook demonstrates how to use TabML for training multiple models and creating optimized ensembles.
# Includes: XGBoost, LightGBM, CatBoost, and custom MLP Neural Network
# 
# **Key Features:**
# - Unified interface for multiple ML algorithms
# - Advanced feature engineering
# - Out-of-fold (OOF) predictions for robust validation
# - Multiple ensemble optimization methods
# - Custom MLP implementation following NN_by_GPT5 approach

# ## 1. Install Dependencies and Setup

#get_ipython().system('pip install -q git+https://github.com/wguesdon/tabml.git')
#get_ipython().system('pip install -q xgboost lightgbm catboost')
#get_ipython().system('pip install -q torch')  # For MLP neural network
#get_ipython().system('pip install -q optuna scipy')  # For ensemble optimization
#get_ipython().system('pip install -q loguru python-dotenv')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Import TabML components
from tabml import (
    FeatureEngineer,
    XGBoostModel,
    LightGBMModel,
    CatBoostModel,
    OOFEnsemble,
    OOFManager
)

# Import PyTorch for MLP
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
    logger.info(f"PyTorch {torch.__version__} available, GPU: {torch.cuda.is_available()}")
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.warning("PyTorch not available, MLP model will be skipped")

# Set style for visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configuration
RANDOM_SEED = 42
N_FOLDS = 5
TARGET_COL = 'y'

logger.info("TabML framework loaded successfully!")


# ## 2. Load Competition Data

# Load competition data
COMPETITION_NAME = "playground-series-s5e8"  # Update this for your competition
DATA_DIR = Path(f"/kaggle/input/{COMPETITION_NAME}")

# If running locally for testing, use local path
if not DATA_DIR.exists():
    DATA_DIR = Path(".")
    logger.info("Using local directory for data")

train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Target column: {TARGET_COL}")

# Display basic info
train_df.info()
print("\nTarget distribution:")
print(train_df[TARGET_COL].value_counts(normalize=True))

# Separate features and target
id_col = 'id' if 'id' in train_df.columns else 'Id'
X_train = train_df.drop([TARGET_COL, id_col], axis=1)
y_train = train_df[TARGET_COL]
X_test = test_df.drop(id_col, axis=1)

print(f"Features shape: {X_train.shape}")
print(f"Number of numeric features: {X_train.select_dtypes(include=['int64', 'float64']).shape[1]}")
print(f"Number of categorical features: {X_train.select_dtypes(include=['object']).shape[1]}")


# ## 3. Feature Engineering

# Apply feature engineering
logger.info("Applying advanced feature engineering...")

engineer = FeatureEngineer(
    categorical_impute_strategy='constant',
    numeric_impute_strategy='median',
    categorical_encoding='target',
    scaling_method='standard',
    create_interactions=True,
    create_polynomial=True,
    max_cardinality=20,
    min_frequency=0.01
)

# Transform features
X_train_fe = engineer.fit_transform(X_train, y_train)
X_test_fe = engineer.transform(X_test)

# Remove duplicate columns
X_train_fe = X_train_fe.loc[:, ~X_train_fe.columns.duplicated()]
X_test_fe = X_test_fe.loc[:, ~X_test_fe.columns.duplicated()]

print(f"Features after engineering: {X_train_fe.shape[1]} (was {X_train.shape[1]})")
print(f"New features created: {X_train_fe.shape[1] - X_train.shape[1]}")

# Also prepare scaled version for neural network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_fe)
X_test_scaled = scaler.transform(X_test_fe)


# ## 4. Custom MLP Neural Network Implementation

def create_mlp_model(input_dim):
    """Create MLP model following NN_by_GPT5 approach."""
    
    class MLP(nn.Module):
        def __init__(self, input_dim):
            super(MLP, self).__init__()
            
            # Architecture: gradually decreasing layer sizes with dropout
            self.layers = nn.Sequential(
                # Input layer
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                # Hidden layer 1
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                # Hidden layer 2
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.2),
                
                # Hidden layer 3
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                
                # Output layer
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            return self.layers(x).squeeze()
    
    return MLP(input_dim)


def train_mlp_with_cv(X_train, y_train, X_test, n_folds=5, verbose=True):
    """Train MLP with cross-validation and return OOF predictions."""
    
    if not PYTORCH_AVAILABLE:
        logger.warning("PyTorch not available, skipping MLP")
        return None, None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training MLP on {device}")
    
    # Prepare for cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train.values if hasattr(y_train, 'values') else y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    
    # Training parameters
    batch_size = 512
    learning_rate = 0.001
    n_epochs = 50
    patience = 10
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        if verbose:
            logger.info(f"Training MLP Fold {fold}/{n_folds}")
        
        # Split data
        X_fold_train = X_train_tensor[train_idx]
        y_fold_train = y_train_tensor[train_idx]
        X_fold_val = X_train_tensor[val_idx]
        y_fold_val = y_train_tensor[val_idx]
        
        # Create data loader
        train_dataset = TensorDataset(X_fold_train, y_fold_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        model = create_mlp_model(X_train.shape[1]).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        best_val_score = 0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(n_epochs):
            # Training phase
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_fold_val.to(device))
                val_preds_np = val_outputs.cpu().numpy()
                val_score = roc_auc_score(y_fold_val.numpy(), val_preds_np)
            
            scheduler.step(-val_score)
            
            # Early stopping
            if val_score > best_val_score:
                best_val_score = val_score
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Generate predictions
        model.eval()
        with torch.no_grad():
            oof_preds[val_idx] = model(X_fold_val.to(device)).cpu().numpy()
            test_preds += model(X_test_tensor.to(device)).cpu().numpy() / n_folds
        
        fold_scores.append(best_val_score)
        if verbose:
            logger.info(f"  Fold {fold} CV Score: {best_val_score:.6f}")
    
    overall_score = roc_auc_score(y_train, oof_preds)
    logger.info(f"MLP Overall CV Score: {overall_score:.6f}")
    
    return oof_preds, test_preds


# ## 5. Initialize Storage and Managers

# Create output directory
OUTPUT_DIR = Path("./tabml_output")
OOF_DIR = OUTPUT_DIR / "oof_predictions"
OOF_DIR.mkdir(parents=True, exist_ok=True)

# Initialize OOF manager
oof_manager = OOFManager(output_dir=str(OOF_DIR))

# Initialize ensemble
ensemble = OOFEnsemble(task_type='classification')

# Storage for predictions
all_oof_predictions = []
all_test_predictions = []
model_scores = {}

# ## 6. Train Models

# ### 6.1 XGBoost

logger.info("Training XGBoost models...")

# XGBoost Conservative
xgb1_params = {
    'n_estimators': 800,
    'max_depth': 4,
    'learning_rate': 0.015,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'gamma': 0.1,
    'reg_alpha': 0.5,
    'reg_lambda': 2.0,
    'min_child_weight': 5,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'random_state': RANDOM_SEED,
    'verbosity': 0
}

xgb1 = XGBoostModel(params=xgb1_params)
oof_xgb1 = ensemble.get_oof_predictions(
    models=[xgb1],
    X=X_train_fe,
    y=y_train,
    n_folds=N_FOLDS,
    stratified=True,
    verbose=True
).iloc[:, 0]

xgb1.fit(X_train_fe, y_train)
test_xgb1 = xgb1.predict_proba(X_test_fe)[:, 1]

score_xgb1 = roc_auc_score(y_train, oof_xgb1)
logger.info(f"XGBoost Conservative CV Score: {score_xgb1:.6f}")

all_oof_predictions.append(oof_xgb1)
all_test_predictions.append(test_xgb1)
model_scores["XGBoost_Conservative"] = score_xgb1

# XGBoost Aggressive
xgb2_params = {
    'n_estimators': 1200,
    'max_depth': 6,
    'learning_rate': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.01,
    'reg_alpha': 0.1,
    'reg_lambda': 0.5,
    'min_child_weight': 3,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'random_state': RANDOM_SEED + 42,
    'verbosity': 0
}

xgb2 = XGBoostModel(params=xgb2_params)
oof_xgb2 = ensemble.get_oof_predictions(
    models=[xgb2],
    X=X_train_fe,
    y=y_train,
    n_folds=N_FOLDS,
    stratified=True,
    verbose=False
).iloc[:, 0]

xgb2.fit(X_train_fe, y_train)
test_xgb2 = xgb2.predict_proba(X_test_fe)[:, 1]

score_xgb2 = roc_auc_score(y_train, oof_xgb2)
logger.info(f"XGBoost Aggressive CV Score: {score_xgb2:.6f}")

all_oof_predictions.append(oof_xgb2)
all_test_predictions.append(test_xgb2)
model_scores["XGBoost_Aggressive"] = score_xgb2

# ### 6.2 LightGBM

logger.info("Training LightGBM...")

lgb_params = {
    'n_estimators': 1000,
    'num_leaves': 31,
    'learning_rate': 0.01,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'lambda_l1': 0.5,
    'lambda_l2': 1.0,
    'min_data_in_leaf': 20,
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'random_state': RANDOM_SEED,
    'verbosity': -1
}

lgb = LightGBMModel(params=lgb_params)
oof_lgb = ensemble.get_oof_predictions(
    models=[lgb],
    X=X_train_fe,
    y=y_train,
    n_folds=N_FOLDS,
    stratified=True,
    verbose=False
).iloc[:, 0]

lgb.fit(X_train_fe, y_train)
test_lgb = lgb.predict_proba(X_test_fe)[:, 1]

score_lgb = roc_auc_score(y_train, oof_lgb)
logger.info(f"LightGBM CV Score: {score_lgb:.6f}")

all_oof_predictions.append(oof_lgb)
all_test_predictions.append(test_lgb)
model_scores["LightGBM"] = score_lgb

# ### 6.3 CatBoost

logger.info("Training CatBoost...")

cat_params = {
    'iterations': 800,
    'depth': 6,
    'learning_rate': 0.015,
    'l2_leaf_reg': 3,
    'border_count': 128,
    'random_seed': RANDOM_SEED,
    'verbose': False
}

cat = CatBoostModel(params=cat_params)
oof_cat = ensemble.get_oof_predictions(
    models=[cat],
    X=X_train_fe,
    y=y_train,
    n_folds=N_FOLDS,
    stratified=True,
    verbose=False
).iloc[:, 0]

cat.fit(X_train_fe, y_train)
test_cat = cat.predict_proba(X_test_fe)[:, 1]

score_cat = roc_auc_score(y_train, oof_cat)
logger.info(f"CatBoost CV Score: {score_cat:.6f}")

all_oof_predictions.append(oof_cat)
all_test_predictions.append(test_cat)
model_scores["CatBoost"] = score_cat

# ### 6.4 MLP Neural Network

logger.info("Training MLP Neural Network...")

# Train MLP using scaled features (important for neural networks)
oof_mlp, test_mlp = train_mlp_with_cv(X_train_scaled, y_train, X_test_scaled, n_folds=N_FOLDS)

if oof_mlp is not None:
    score_mlp = roc_auc_score(y_train, oof_mlp)
    logger.info(f"MLP CV Score: {score_mlp:.6f}")
    
    all_oof_predictions.append(oof_mlp)
    all_test_predictions.append(test_mlp)
    model_scores["MLP_Neural_Network"] = score_mlp
else:
    logger.warning("MLP training skipped")

# ## 7. Model Performance Summary

# Display model scores
model_df = pd.DataFrame(list(model_scores.items()), columns=['Model', 'CV_Score'])
model_df = model_df.sort_values('CV_Score', ascending=False)

print("\n" + "="*50)
print("INDIVIDUAL MODEL PERFORMANCE")
print("="*50)
for idx, row in model_df.iterrows():
    print(f"{row['Model']:25s}: {row['CV_Score']:.6f}")
print("="*50)

# Visualize
plt.figure(figsize=(10, 6))
colors = ['gold' if score >= 0.97 else 'lightgreen' if score >= 0.96 else 'skyblue' 
          for score in model_df['CV_Score']]
bars = plt.bar(model_df['Model'], model_df['CV_Score'], color=colors, alpha=0.8, edgecolor='black')
plt.title('Individual Model Performance', fontsize=14, fontweight='bold')
plt.ylabel('ROC AUC Score', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)

for bar, score in zip(bars, model_df['CV_Score']):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
             f'{score:.4f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

# ## 8. Ensemble Optimization

# Combine OOF predictions
combined_oofs = pd.DataFrame({
    name: pred for name, pred in zip(model_scores.keys(), all_oof_predictions)
})

ensemble_results = {}
ensemble_weights = {}
ensemble_test_preds = {}

# ### 8.1 Simple Average

simple_avg = combined_oofs.mean(axis=1)
simple_avg_test = np.mean(all_test_predictions, axis=0)
simple_score = roc_auc_score(y_train, simple_avg)
ensemble_results['Simple_Average'] = simple_score
ensemble_test_preds['Simple_Average'] = simple_avg_test
print(f"Simple Average Score: {simple_score:.6f}")

# ### 8.2 Weighted by CV Scores

cv_scores = np.array([model_scores[name] for name in combined_oofs.columns])
cv_weights = np.power(cv_scores, 3)
cv_weights = cv_weights / cv_weights.sum()

weighted_avg = np.average(combined_oofs.values, weights=cv_weights, axis=1)
weighted_avg_test = np.average(np.column_stack(all_test_predictions), weights=cv_weights, axis=1)
weighted_score = roc_auc_score(y_train, weighted_avg)
ensemble_results['CV_Weighted'] = weighted_score
ensemble_weights['CV_Weighted'] = cv_weights
ensemble_test_preds['CV_Weighted'] = weighted_avg_test
print(f"CV Weighted Score: {weighted_score:.6f}")

# ### 8.3 Hill Climbing Optimization

logger.info("Optimizing with Hill Climbing...")

hill_weights = ensemble.optimize_weights(
    combined_oofs, y_train,
    method='hill_climbing',
    n_iterations=5000,
    patience=500
)

hill_pred = np.average(combined_oofs.values, weights=hill_weights, axis=1)
hill_test = np.average(np.column_stack(all_test_predictions), weights=hill_weights, axis=1)
hill_score = roc_auc_score(y_train, hill_pred)
ensemble_results['Hill_Climbing'] = hill_score
ensemble_weights['Hill_Climbing'] = hill_weights
ensemble_test_preds['Hill_Climbing'] = hill_test
print(f"Hill Climbing Score: {hill_score:.6f}")

print("\nOptimized weights:")
for name, weight in zip(combined_oofs.columns, hill_weights):
    if weight > 0.01:
        print(f"  {name:25s}: {weight:.3f}")

# ### 8.4 Additional Optimization Methods

# Optuna
logger.info("Optimizing with Optuna...")
optuna_weights = ensemble.optimize_weights(
    combined_oofs, y_train,
    method='optuna',
    n_trials=100
)

optuna_pred = np.average(combined_oofs.values, weights=optuna_weights, axis=1)
optuna_test = np.average(np.column_stack(all_test_predictions), weights=optuna_weights, axis=1)
optuna_score = roc_auc_score(y_train, optuna_pred)
ensemble_results['Optuna'] = optuna_score
ensemble_weights['Optuna'] = optuna_weights
ensemble_test_preds['Optuna'] = optuna_test
print(f"Optuna Score: {optuna_score:.6f}")

# SciPy
logger.info("Optimizing with SciPy...")
scipy_weights = ensemble.optimize_weights(
    combined_oofs, y_train,
    method='scipy'
)

scipy_pred = np.average(combined_oofs.values, weights=scipy_weights, axis=1)
scipy_test = np.average(np.column_stack(all_test_predictions), weights=scipy_weights, axis=1)
scipy_score = roc_auc_score(y_train, scipy_pred)
ensemble_results['SciPy'] = scipy_score
ensemble_weights['SciPy'] = scipy_weights
ensemble_test_preds['SciPy'] = scipy_test
print(f"SciPy Score: {scipy_score:.6f}")

# Stacking with Meta-Learner
logger.info("Training Stacking Ensemble...")
ensemble_stack = OOFEnsemble(task_type='classification')
# Use None to let it use the default meta-model (LogisticRegression for classification)
ensemble_stack.fit_stacking(combined_oofs, y_train, meta_model=None)
stack_pred = ensemble_stack.predict_stacking(combined_oofs)
stack_test = ensemble_stack.predict_stacking(pd.DataFrame(np.column_stack(all_test_predictions), columns=combined_oofs.columns))
stack_score = roc_auc_score(y_train, stack_pred)
ensemble_results['Stacking'] = stack_score
ensemble_test_preds['Stacking'] = stack_test
print(f"Stacking Score: {stack_score:.6f}")

# ## 9. Final Results and Submission

# Display ensemble results
print("\n" + "="*50)
print("ENSEMBLE OPTIMIZATION RESULTS")
print("="*50)

sorted_results = sorted(ensemble_results.items(), key=lambda x: x[1], reverse=True)
for method, score in sorted_results:
    improvement = (score - simple_score) * 100
    print(f"{method:20s}: {score:.6f} ({improvement:+.3f}% vs simple avg)")

# Find best method
best_method = max(ensemble_results, key=ensemble_results.get)
best_score = ensemble_results[best_method]
print(f"\nBest Method: {best_method} with score {best_score:.6f}")


# Visualize ensemble results
plt.figure(figsize=(12, 6))

# Method comparison
plt.subplot(1, 2, 1)
methods = list(ensemble_results.keys())
scores = list(ensemble_results.values())
colors = ['gold' if s == best_score else 'lightgreen' for s in scores]
bars = plt.bar(methods, scores, color=colors, alpha=0.8, edgecolor='black')
plt.title('Ensemble Method Comparison', fontsize=14, fontweight='bold')
plt.ylabel('ROC AUC Score', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)

for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.0001,
             f'{score:.5f}', ha='center', va='bottom', fontsize=9)

# Weight distribution
plt.subplot(1, 2, 2)
if best_method in ensemble_weights:
    weights = ensemble_weights[best_method]
    model_names = combined_oofs.columns
    
    sorted_idx = np.argsort(weights)[::-1]
    sorted_weights = weights[sorted_idx]
    sorted_names = [model_names[i] for i in sorted_idx]
    
    plt.barh(range(len(sorted_weights)), sorted_weights, color='skyblue', alpha=0.8)
    plt.yticks(range(len(sorted_weights)), sorted_names)
    plt.xlabel('Weight', fontsize=12)
    plt.title(f'{best_method} - Weight Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

# ## 10. Generate Submissions

# Create submission with best ensemble
submission = sample_sub.copy()
submission[TARGET_COL] = ensemble_test_preds[best_method]

# Save submission
submission_path = f"submission_tabml_{best_method.lower()}_{best_score:.5f}.csv"
submission.to_csv(submission_path, index=False)
print(f"\nSubmission saved to: {submission_path}")
print(f"Expected LB Score: ~{best_score:.5f}")

# Create additional submissions for top methods
print("\nCreating additional submissions:")
top_methods = sorted(ensemble_results.items(), key=lambda x: x[1], reverse=True)[:3]
for method, score in top_methods:
    submission = sample_sub.copy()
    submission[TARGET_COL] = ensemble_test_preds[method]
    filename = f"submission_{method.lower()}_{score:.5f}.csv"
    submission.to_csv(filename, index=False)
    print(f"  {method}: {filename} (CV: {score:.5f})")

# ## 11. Summary

print("\n" + "="*60)
print("TABML ENSEMBLE COMPLETE!")
print("="*60)
print(f"Best CV Score: {best_score:.6f}")
print(f"Best Method: {best_method}")
print(f"Models Trained: {len(model_scores)}")
print(f"  - XGBoost: 2 configurations")
print(f"  - LightGBM: 1 configuration")
print(f"  - CatBoost: 1 configuration")
if "MLP_Neural_Network" in model_scores:
    print(f"  - MLP Neural Network: 1 configuration")
print(f"Ensemble Methods Tested: {len(ensemble_results)}")
print(f"  - Simple Average")
print(f"  - CV Weighted")  
print(f"  - Hill Climbing")
print(f"  - Optuna Optimization")
print(f"  - SciPy Optimization")
print(f"  - Stacking with Meta-Learner")
print("="*60)

# Model insights
print("\nKey Insights:")
print("1. Feature engineering increased features from", X_train.shape[1], "to", X_train_fe.shape[1])
print("2. Best individual model:", model_df.iloc[0]['Model'], f"({model_df.iloc[0]['CV_Score']:.6f})")
print("3. Ensemble improvement:", f"{(best_score - model_df.iloc[0]['CV_Score'])*100:.3f}%")
if "MLP_Neural_Network" in model_scores:
    print("4. MLP uses same features as XGBoost (NN_by_GPT5 approach)")
print("5. Stacking ensemble uses LogisticRegression meta-learner")
print("\nTabML Repository: https://github.com/wguesdon/tabml")
