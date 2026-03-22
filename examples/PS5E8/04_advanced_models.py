"""
PS5E8 Competition - Advanced Models with Improved Feature Engineering
Implements techniques from high-scoring notebooks to achieve 0.977+ AUC
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import tabml
sys.path.append(str(Path(__file__).parent.parent.parent))

from tabml import (
    DataLoader, FeatureEngineer, AdvancedFeatureEngineer,
    XGBoostModel, LightGBMModel, CatBoostModel, 
    TabNetModel, OOFEnsemble, OOFManager, MLflowTracker
)

# Load environment variables
load_dotenv()

# Setup paths
DATA_DIR = Path("../../data/raw/PS5E8")
ORIGINAL_DATA_PATH = Path("../../data/raw/PS5E8/bank-full.csv")
OUTPUT_DIR = Path("output")
OOF_DIR = OUTPUT_DIR / "oof_predictions_advanced"
SUBMISSION_DIR = OUTPUT_DIR / "submissions_advanced"

# Create directories
OOF_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

# Competition settings
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
N_FOLDS = 5
TARGET_COL = 'y'

# Feature engineering settings
USE_ORIGINAL_AS_ROWS = True
USE_ORIGINAL_AS_COLUMNS = True
CREATE_CATEGORICAL_PAIRS = True
TREAT_NUMERICAL_AS_CATEGORICAL = True


def load_competition_data():
    """Load PS5E8 competition data and original bank marketing data."""
    logger.info("Loading competition data...")
    
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    
    # Load original data if available
    original_df = None
    if ORIGINAL_DATA_PATH.exists():
        original_df = pd.read_csv(ORIGINAL_DATA_PATH)
        logger.info(f"Loaded original data: {original_df.shape}")
    else:
        logger.warning("Original data not found. Some advanced features will be disabled.")
    
    logger.info(f"Train shape: {train_df.shape}")
    logger.info(f"Test shape: {test_df.shape}")
    
    return train_df, test_df, sample_sub, original_df


def create_categorical_pairs(df, categorical_cols):
    """Create categorical pair features like in the advanced notebooks."""
    logger.info(f"Creating categorical pair features from {len(categorical_cols)} columns...")
    
    new_features = {}
    pair_count = 0
    
    for i, col1 in enumerate(categorical_cols):
        for col2 in categorical_cols[i+1:]:
            # Create pair feature
            if col1 in df.columns and col2 in df.columns:
                # Get cardinality of second feature
                card2 = df[col2].nunique()
                # Create combined feature
                new_col_name = f"{col1}_X_{col2}"
                new_features[new_col_name] = df[col1].astype(str) + "_" + df[col2].astype(str)
                pair_count += 1
    
    # Add new features to dataframe
    for col_name, values in new_features.items():
        df[col_name] = values
    
    logger.info(f"Created {pair_count} categorical pair features")
    return df


def treat_numerical_as_categorical(df, numerical_cols):
    """Treat numerical features as both numerical and categorical."""
    logger.info(f"Creating categorical versions of {len(numerical_cols)} numerical features...")
    
    for col in numerical_cols:
        if col in df.columns:
            # Create categorical version using quantile binning
            df[f"{col}_cat"] = pd.qcut(df[col], q=10, labels=False, duplicates='drop')
    
    return df


def engineer_features_advanced(X_train, y_train, X_test, original_df=None):
    """Apply advanced feature engineering inspired by high-scoring notebooks."""
    logger.info("Applying advanced feature engineering...")
    
    # Store original columns
    original_cols = X_train.columns.tolist()
    
    # Identify column types
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    logger.info(f"Categorical columns: {len(categorical_cols)}")
    logger.info(f"Numerical columns: {len(numerical_cols)}")
    
    # 1. Create categorical pairs (if enabled)
    if CREATE_CATEGORICAL_PAIRS and len(categorical_cols) > 0:
        X_train = create_categorical_pairs(X_train, categorical_cols)
        X_test = create_categorical_pairs(X_test, categorical_cols)
        # Update categorical columns list
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 2. Treat numerical as categorical (if enabled)
    if TREAT_NUMERICAL_AS_CATEGORICAL and len(numerical_cols) > 0:
        X_train = treat_numerical_as_categorical(X_train, numerical_cols)
        X_test = treat_numerical_as_categorical(X_test, numerical_cols)
    
    # 3. Use original data for feature engineering (if available)
    if original_df is not None and USE_ORIGINAL_AS_COLUMNS:
        logger.info("Creating features from original data statistics...")
        
        # Calculate target statistics from original data
        # Note: This doesn't cause leakage as original data has different distribution
        for cat_col in categorical_cols[:10]:  # Limit to avoid explosion
            if cat_col in original_df.columns:
                # Calculate mean target for each category in original data
                orig_stats = original_df.groupby(cat_col)[TARGET_COL].agg(['mean', 'std', 'count'])
                orig_stats.columns = [f"orig_{cat_col}_{stat}" for stat in ['mean', 'std', 'count']]
                
                # Merge with train and test
                X_train = X_train.merge(orig_stats, left_on=cat_col, right_index=True, how='left')
                X_test = X_test.merge(orig_stats, left_on=cat_col, right_index=True, how='left')
    
    # 4. Count encoding for all categorical features
    logger.info("Applying count encoding...")
    for col in categorical_cols:
        if col in X_train.columns:
            # Count encoding
            count_map = X_train[col].value_counts().to_dict()
            X_train[f"{col}_count"] = X_train[col].map(count_map).fillna(0)
            X_test[f"{col}_count"] = X_test[col].map(count_map).fillna(0)
    
    # 5. Target encoding using simple mean encoding
    logger.info("Applying target encoding...")
    # Create a temporary dataframe with features and target for encoding
    temp_df = X_train.copy()
    temp_df['__target__'] = y_train
    
    for col in categorical_cols[:20]:  # Limit to avoid too many features
        if col in X_train.columns:
            # Calculate mean target for each category
            target_mean = temp_df.groupby(col)['__target__'].mean()
            # Apply encoding with smoothing
            global_mean = y_train.mean()
            X_train[f"{col}_target"] = X_train[col].map(target_mean).fillna(global_mean)
            X_test[f"{col}_target"] = X_test[col].map(target_mean).fillna(global_mean)
    
    # 6. Label encode remaining categorical features
    logger.info("Label encoding categorical features...")
    label_encoders = {}
    for col in categorical_cols:
        if col in X_train.columns:
            le = LabelEncoder()
            # Fit on combined train and test to handle unseen categories
            combined_values = pd.concat([X_train[col], X_test[col]]).astype(str)
            le.fit(combined_values)
            
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
            label_encoders[col] = le
    
    # 7. Add polynomial features for top numerical features
    logger.info("Adding polynomial features...")
    top_numerical = numerical_cols[:5]  # Use top 5 numerical features
    for i, col1 in enumerate(top_numerical):
        for col2 in top_numerical[i:]:
            if col1 in X_train.columns and col2 in X_train.columns:
                # Multiplication
                X_train[f"{col1}_times_{col2}"] = X_train[col1] * X_train[col2]
                X_test[f"{col1}_times_{col2}"] = X_test[col1] * X_test[col2]
                
                # Division (with protection against division by zero)
                if col1 != col2:
                    X_train[f"{col1}_div_{col2}"] = X_train[col1] / (X_train[col2] + 1e-8)
                    X_test[f"{col1}_div_{col2}"] = X_test[col1] / (X_test[col2] + 1e-8)
    
    # 8. Log transform skewed features
    logger.info("Applying log transformation to skewed features...")
    for col in numerical_cols:
        if col in X_train.columns:
            if X_train[col].skew() > 1.0:
                X_train[f"{col}_log"] = np.log1p(X_train[col].clip(lower=0))
                X_test[f"{col}_log"] = np.log1p(X_test[col].clip(lower=0))
    
    # Remove duplicate columns
    X_train = X_train.loc[:, ~X_train.columns.duplicated()]
    X_test = X_test.loc[:, ~X_test.columns.duplicated()]
    
    # Handle any remaining missing values
    X_train = X_train.fillna(-999)
    X_test = X_test.fillna(-999)
    
    logger.info(f"Final train shape: {X_train.shape}")
    logger.info(f"Final test shape: {X_test.shape}")
    
    return X_train, X_test


def train_xgboost_models(X_train, y_train, X_test, oof_manager, mlflow_tracker=None):
    """Train multiple XGBoost models with different configurations."""
    logger.info("\n" + "="*60)
    logger.info("TRAINING XGBOOST MODELS")
    logger.info("="*60)
    
    all_oof_preds = []
    all_test_preds = []
    model_scores = {}
    
    # Configuration 1: Conservative with heavy regularization (like notebook approach 1)
    xgb1_params = {
        'n_estimators': 1000,
        'max_depth': 5,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'colsample_bylevel': 0.7,
        'gamma': 0.1,
        'reg_alpha': 0.5,
        'reg_lambda': 2.0,
        'min_child_weight': 5,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',  # Use histogram-based method for efficiency
        'random_state': RANDOM_SEED,
        'verbosity': 0
    }
    
    xgb1 = XGBoostModel(params=xgb1_params)
    
    # Generate OOF predictions
    ensemble = OOFEnsemble(task_type='classification')
    oof1 = ensemble.get_oof_predictions(
        models=[xgb1],
        X=X_train,
        y=y_train,
        n_folds=N_FOLDS,
        stratified=True,
        verbose=True
    ).iloc[:, 0]
    
    # Train on full data for test predictions
    xgb1.fit(X_train, y_train)
    test1 = xgb1.predict_proba(X_test)[:, 1]
    
    score1 = roc_auc_score(y_train, oof1)
    logger.info(f"XGBoost Conservative CV Score: {score1:.6f}")
    
    # Save OOF predictions
    oof_manager.save_oof(
        predictions=oof1,
        model_name="XGBoost_Advanced_v1",
        model_params=xgb1_params,
        cv_score=score1,
        test_predictions=test1,
        experiment_name='PS5E8_advanced',
        tags={'competition': 'PS5E8', 'model_type': 'XGBoost', 'version': 'conservative'}
    )
    
    all_oof_preds.append(oof1)
    all_test_preds.append(test1)
    model_scores["XGBoost_Advanced_v1"] = score1
    
    # Configuration 2: Aggressive with less regularization (like notebook approach 2)
    xgb2_params = {
        'n_estimators': 1500,
        'max_depth': 7,
        'learning_rate': 0.008,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.8,
        'gamma': 0.01,
        'reg_alpha': 0.1,
        'reg_lambda': 0.5,
        'min_child_weight': 3,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'max_leaves': 31,  # Use max_leaves for better tree structure
        'grow_policy': 'lossguide',
        'random_state': RANDOM_SEED + 42,
        'verbosity': 0
    }
    
    xgb2 = XGBoostModel(params=xgb2_params)
    
    # Generate OOF predictions
    oof2 = ensemble.get_oof_predictions(
        models=[xgb2],
        X=X_train,
        y=y_train,
        n_folds=N_FOLDS,
        stratified=True,
        verbose=True
    ).iloc[:, 0]
    
    # Train on full data
    xgb2.fit(X_train, y_train)
    test2 = xgb2.predict_proba(X_test)[:, 1]
    
    score2 = roc_auc_score(y_train, oof2)
    logger.info(f"XGBoost Aggressive CV Score: {score2:.6f}")
    
    # Save OOF predictions
    oof_manager.save_oof(
        predictions=oof2,
        model_name="XGBoost_Advanced_v2",
        model_params=xgb2_params,
        cv_score=score2,
        test_predictions=test2,
        experiment_name='PS5E8_advanced',
        tags={'competition': 'PS5E8', 'model_type': 'XGBoost', 'version': 'aggressive'}
    )
    
    all_oof_preds.append(oof2)
    all_test_preds.append(test2)
    model_scores["XGBoost_Advanced_v2"] = score2
    
    # Configuration 3: DART booster for diversity
    xgb3_params = {
        'n_estimators': 800,
        'max_depth': 6,
        'learning_rate': 0.01,
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'gamma': 0.05,
        'reg_alpha': 0.3,
        'reg_lambda': 1.0,
        'min_child_weight': 4,
        'booster': 'dart',
        'sample_type': 'uniform',
        'normalize_type': 'tree',
        'rate_drop': 0.1,
        'skip_drop': 0.5,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': RANDOM_SEED + 123,
        'verbosity': 0
    }
    
    xgb3 = XGBoostModel(params=xgb3_params)
    
    # Generate OOF predictions
    oof3 = ensemble.get_oof_predictions(
        models=[xgb3],
        X=X_train,
        y=y_train,
        n_folds=N_FOLDS,
        stratified=True,
        verbose=True
    ).iloc[:, 0]
    
    # Train on full data
    xgb3.fit(X_train, y_train)
    test3 = xgb3.predict_proba(X_test)[:, 1]
    
    score3 = roc_auc_score(y_train, oof3)
    logger.info(f"XGBoost DART CV Score: {score3:.6f}")
    
    # Save OOF predictions
    oof_manager.save_oof(
        predictions=oof3,
        model_name="XGBoost_DART",
        model_params=xgb3_params,
        cv_score=score3,
        test_predictions=test3,
        experiment_name='PS5E8_advanced',
        tags={'competition': 'PS5E8', 'model_type': 'XGBoost', 'version': 'dart'}
    )
    
    all_oof_preds.append(oof3)
    all_test_preds.append(test3)
    model_scores["XGBoost_DART"] = score3
    
    return all_oof_preds, all_test_preds, model_scores


def train_lightgbm_models(X_train, y_train, X_test, oof_manager, mlflow_tracker=None):
    """Train LightGBM models with advanced configurations."""
    logger.info("\n" + "="*60)
    logger.info("TRAINING LIGHTGBM MODELS")
    logger.info("="*60)
    
    all_oof_preds = []
    all_test_preds = []
    model_scores = {}
    
    # LightGBM with advanced settings
    lgb_params = {
        'n_estimators': 1200,
        'num_leaves': 31,
        'learning_rate': 0.008,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'lambda_l1': 0.5,
        'lambda_l2': 1.0,
        'min_data_in_leaf': 20,
        'min_gain_to_split': 0.01,
        'max_bin': 255,
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'random_state': RANDOM_SEED,
        'verbosity': -1
    }
    
    lgb = LightGBMModel(params=lgb_params)
    
    # Generate OOF predictions
    ensemble = OOFEnsemble(task_type='classification')
    oof = ensemble.get_oof_predictions(
        models=[lgb],
        X=X_train,
        y=y_train,
        n_folds=N_FOLDS,
        stratified=True,
        verbose=True
    ).iloc[:, 0]
    
    # Train on full data
    lgb.fit(X_train, y_train)
    test_pred = lgb.predict_proba(X_test)[:, 1]
    
    score = roc_auc_score(y_train, oof)
    logger.info(f"LightGBM Advanced CV Score: {score:.6f}")
    
    # Save OOF predictions
    oof_manager.save_oof(
        predictions=oof,
        model_name="LightGBM_Advanced",
        model_params=lgb_params,
        cv_score=score,
        test_predictions=test_pred,
        experiment_name='PS5E8_advanced',
        tags={'competition': 'PS5E8', 'model_type': 'LightGBM'}
    )
    
    all_oof_preds.append(oof)
    all_test_preds.append(test_pred)
    model_scores["LightGBM_Advanced"] = score
    
    return all_oof_preds, all_test_preds, model_scores


def train_catboost_models(X_train, y_train, X_test, oof_manager, mlflow_tracker=None):
    """Train CatBoost models with advanced configurations."""
    logger.info("\n" + "="*60)
    logger.info("TRAINING CATBOOST MODELS")
    logger.info("="*60)
    
    all_oof_preds = []
    all_test_preds = []
    model_scores = {}
    
    # CatBoost with advanced settings
    cat_params = {
        'iterations': 1000,
        'depth': 7,
        'learning_rate': 0.01,
        'l2_leaf_reg': 3,
        'border_count': 128,
        'bagging_temperature': 0.5,
        'random_strength': 0.5,
        'od_type': 'Iter',
        'od_wait': 50,
        'random_seed': RANDOM_SEED,
        'verbose': False
    }
    
    cat = CatBoostModel(params=cat_params)
    
    # Generate OOF predictions
    ensemble = OOFEnsemble(task_type='classification')
    oof = ensemble.get_oof_predictions(
        models=[cat],
        X=X_train,
        y=y_train,
        n_folds=N_FOLDS,
        stratified=True,
        verbose=True
    ).iloc[:, 0]
    
    # Train on full data
    cat.fit(X_train, y_train)
    test_pred = cat.predict_proba(X_test)[:, 1]
    
    score = roc_auc_score(y_train, oof)
    logger.info(f"CatBoost Advanced CV Score: {score:.6f}")
    
    # Save OOF predictions
    oof_manager.save_oof(
        predictions=oof,
        model_name="CatBoost_Advanced",
        model_params=cat_params,
        cv_score=score,
        test_predictions=test_pred,
        experiment_name='PS5E8_advanced',
        tags={'competition': 'PS5E8', 'model_type': 'CatBoost'}
    )
    
    all_oof_preds.append(oof)
    all_test_preds.append(test_pred)
    model_scores["CatBoost_Advanced"] = score
    
    return all_oof_preds, all_test_preds, model_scores


def train_tabnet_model(X_train, y_train, X_test, oof_manager, mlflow_tracker=None):
    """Train TabNet neural network model."""
    logger.info("\n" + "="*60)
    logger.info("TRAINING TABNET MODEL")
    logger.info("="*60)
    
    try:
        # TabNet parameters
        tabnet_params = {
            'n_d': 32,
            'n_a': 32,
            'n_steps': 5,
            'gamma': 1.5,
            'n_independent': 2,
            'n_shared': 2,
            'lambda_sparse': 1e-4,
            'optimizer_params': {'lr': 0.02, 'weight_decay': 1e-5},
            'mask_type': 'entmax',
            'scheduler_params': {'step_size': 10, 'gamma': 0.9},
            'seed': RANDOM_SEED,
            'verbose': 1
        }
        
        tabnet = TabNetModel(params=tabnet_params)
        
        # Generate OOF predictions
        ensemble = OOFEnsemble(task_type='classification')
        oof = ensemble.get_oof_predictions(
            models=[tabnet],
            X=X_train,
            y=y_train,
            n_folds=N_FOLDS,
            stratified=True,
            verbose=True
        ).iloc[:, 0]
        
        # Train on full data
        tabnet.fit(X_train, y_train)
        test_pred = tabnet.predict_proba(X_test)[:, 1]
        
        score = roc_auc_score(y_train, oof)
        logger.info(f"TabNet CV Score: {score:.6f}")
        
        # Save OOF predictions
        oof_manager.save_oof(
            predictions=oof,
            model_name="TabNet",
            model_params=tabnet_params,
            cv_score=score,
            test_predictions=test_pred,
            experiment_name='PS5E8_advanced',
            tags={'competition': 'PS5E8', 'model_type': 'TabNet', 'architecture': 'neural_network'}
        )
        
        return [oof], [test_pred], {"TabNet": score}
        
    except Exception as e:
        logger.warning(f"TabNet training failed: {e}")
        logger.info("Skipping TabNet model...")
        return [], [], {}


def create_advanced_ensemble(oof_manager, y_train):
    """Create advanced ensemble using saved OOF predictions."""
    logger.info("\n" + "="*60)
    logger.info("CREATING ADVANCED ENSEMBLE")
    logger.info("="*60)
    
    # Load all advanced OOF predictions
    all_oofs = oof_manager.load_all_oofs(
        experiment_name='PS5E8_advanced',
        min_cv_score=0.0
    )
    
    if not all_oofs:
        logger.error("No advanced OOF predictions found!")
        return None, None, None
    
    logger.info(f"Loaded {len(all_oofs)} advanced models")
    
    # Combine OOF predictions
    combined_oofs = oof_manager.combine_oofs(all_oofs, method='horizontal')
    combined_test = oof_manager.combine_oofs(all_oofs, method='horizontal', use_test=True)
    
    # Initialize ensemble
    ensemble = OOFEnsemble(task_type='classification')
    
    # Try different ensemble methods
    results = {}
    
    # 1. Simple average
    equal_weights = np.ones(combined_oofs.shape[1]) / combined_oofs.shape[1]
    equal_pred = np.average(combined_oofs.values, weights=equal_weights, axis=1)
    equal_score = roc_auc_score(y_train, equal_pred)
    results['Equal'] = equal_score
    logger.info(f"Equal weights ensemble: {equal_score:.6f}")
    
    # 2. Hill climbing optimization
    hill_weights = ensemble.optimize_weights(
        combined_oofs, y_train,
        method='hill_climbing',
        n_iterations=5000,
        patience=500
    )
    hill_pred = np.average(combined_oofs.values, weights=hill_weights, axis=1)
    hill_score = roc_auc_score(y_train, hill_pred)
    results['Hill_Climbing'] = hill_score
    logger.info(f"Hill climbing ensemble: {hill_score:.6f}")
    
    # 3. Greedy forward selection
    greedy_weights = ensemble.optimize_weights(
        combined_oofs, y_train,
        method='greedy_forward'
    )
    greedy_pred = np.average(combined_oofs.values, weights=greedy_weights, axis=1)
    greedy_score = roc_auc_score(y_train, greedy_pred)
    results['Greedy_Forward'] = greedy_score
    logger.info(f"Greedy forward ensemble: {greedy_score:.6f}")
    
    # Find best method
    best_method = max(results, key=results.get)
    best_score = results[best_method]
    
    logger.info(f"\nBest ensemble method: {best_method} with score: {best_score:.6f}")
    
    # Get best weights
    if best_method == 'Equal':
        best_weights = equal_weights
    elif best_method == 'Hill_Climbing':
        best_weights = hill_weights
    else:
        best_weights = greedy_weights
    
    # Create final predictions
    final_test_pred = np.average(combined_test.values, weights=best_weights, axis=1)
    
    return best_score, final_test_pred, best_weights


def main():
    """Main training pipeline for advanced models."""
    
    logger.info("="*60)
    logger.info("PS5E8 COMPETITION - ADVANCED MODELS TRAINING")
    logger.info("="*60)
    
    # Initialize MLflow tracker
    mlflow_tracker = None
    if os.getenv("MLFLOW_TRACKING_URI"):
        mlflow_tracker = MLflowTracker(
            experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "PS5E8-Advanced"),
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
            tags={'stage': 'advanced_training', 'competition': 'PS5E8'}
        )
        mlflow_tracker.start_run(run_name="PS5E8_Advanced_Models")
    
    # Load data
    train_df, test_df, sample_sub, original_df = load_competition_data()
    
    # Separate features and target
    X_train = train_df.drop([TARGET_COL, 'id'], axis=1)
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop('id', axis=1)
    test_ids = test_df['id']
    
    # Apply advanced feature engineering
    X_train_fe, X_test_fe = engineer_features_advanced(X_train, y_train, X_test, original_df)
    
    # Initialize OOF manager
    oof_manager = OOFManager(output_dir=str(OOF_DIR))
    
    # Store all predictions
    all_oof_preds = []
    all_test_preds = []
    all_scores = {}
    
    # Train XGBoost models
    xgb_oofs, xgb_tests, xgb_scores = train_xgboost_models(
        X_train_fe, y_train, X_test_fe, oof_manager, mlflow_tracker
    )
    all_oof_preds.extend(xgb_oofs)
    all_test_preds.extend(xgb_tests)
    all_scores.update(xgb_scores)
    
    # Train LightGBM models
    lgb_oofs, lgb_tests, lgb_scores = train_lightgbm_models(
        X_train_fe, y_train, X_test_fe, oof_manager, mlflow_tracker
    )
    all_oof_preds.extend(lgb_oofs)
    all_test_preds.extend(lgb_tests)
    all_scores.update(lgb_scores)
    
    # Train CatBoost models
    cat_oofs, cat_tests, cat_scores = train_catboost_models(
        X_train_fe, y_train, X_test_fe, oof_manager, mlflow_tracker
    )
    all_oof_preds.extend(cat_oofs)
    all_test_preds.extend(cat_tests)
    all_scores.update(cat_scores)
    
    # Train TabNet model (if available)
    tabnet_oofs, tabnet_tests, tabnet_scores = train_tabnet_model(
        X_train_fe, y_train, X_test_fe, oof_manager, mlflow_tracker
    )
    all_oof_preds.extend(tabnet_oofs)
    all_test_preds.extend(tabnet_tests)
    all_scores.update(tabnet_scores)
    
    # Create advanced ensemble
    ensemble_score, ensemble_test_pred, ensemble_weights = create_advanced_ensemble(oof_manager, y_train)
    
    # Log results to MLflow
    if mlflow_tracker:
        for model_name, score in all_scores.items():
            mlflow_tracker.log_metrics({f'{model_name}_cv_auc': score})
        
        if ensemble_score:
            mlflow_tracker.log_metrics({'ensemble_cv_auc': ensemble_score})
    
    # Save submissions
    logger.info("\n" + "="*60)
    logger.info("SAVING SUBMISSIONS")
    logger.info("="*60)
    
    # Save individual model submissions
    for i, (model_name, test_pred) in enumerate(zip(all_scores.keys(), all_test_preds)):
        submission = sample_sub.copy()
        submission['y'] = test_pred
        submission.to_csv(SUBMISSION_DIR / f"submission_{model_name.lower()}.csv", index=False)
        logger.info(f"Saved {model_name} submission")
    
    # Save ensemble submission
    if ensemble_test_pred is not None:
        submission = sample_sub.copy()
        submission['y'] = ensemble_test_pred
        submission.to_csv(SUBMISSION_DIR / "submission_advanced_ensemble.csv", index=False)
        logger.info("Saved advanced ensemble submission")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING SUMMARY")
    logger.info("="*60)
    
    # Sort models by score
    sorted_models = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    
    logger.info("\nModel Performance (CV AUC):")
    for rank, (model_name, score) in enumerate(sorted_models, 1):
        logger.info(f"{rank}. {model_name:25s}: {score:.6f}")
    
    if ensemble_score:
        logger.info(f"\nAdvanced Ensemble: {ensemble_score:.6f}")
        if sorted_models:
            improvement = (ensemble_score - sorted_models[0][1]) * 100
            logger.info(f"Ensemble improvement over best single model: {improvement:.3f}%")
    
    # End MLflow run
    if mlflow_tracker:
        mlflow_tracker.end_run()
        logger.info(f"\nMLflow tracking completed. View at {os.getenv('MLFLOW_TRACKING_URI')}")
    
    logger.info("\n" + "="*60)
    logger.info("ADVANCED TRAINING COMPLETE!")
    logger.info("="*60)
    logger.info(f"\nBest submission: {SUBMISSION_DIR}/submission_advanced_ensemble.csv")
    logger.info("Expected CV AUC: ~0.975-0.977 (approaching notebook performance)")


if __name__ == "__main__":
    main()