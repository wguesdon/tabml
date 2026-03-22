"""
Out-of-Fold (OOF) Ensemble Example for Kaggle Competitions
===========================================================

This example demonstrates how to use TabML's advanced OOF ensemble capabilities
for creating powerful model ensembles that avoid overfitting.
"""

import pandas as pd
import numpy as np
from tabml import (
    XGBoostModel, LightGBMModel, CatBoostModel, TabNetModel,
    OOFEnsemble, AutoEnsemble
)
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# =============================================================================
# 1. GENERATE SAMPLE DATA
# =============================================================================

def create_sample_data():
    """Create sample data for demonstration."""
    X, y = make_classification(
        n_samples=5000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.7, 0.3],
        random_state=42
    )
    
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(y, name='target')
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# =============================================================================
# 2. BASIC OOF ENSEMBLE
# =============================================================================

def basic_oof_ensemble_example():
    """Basic example of creating OOF predictions and ensemble."""
    print("="*60)
    print("BASIC OOF ENSEMBLE EXAMPLE")
    print("="*60)
    
    # Create sample data
    X_train, X_test, y_train, y_test = create_sample_data()
    
    # Initialize models
    models = [
        XGBoostModel(params={'n_estimators': 100, 'max_depth': 5}),
        LightGBMModel(params={'n_estimators': 100, 'num_leaves': 31}),
        CatBoostModel(params={'iterations': 100, 'depth': 6})
    ]
    
    # Create OOF ensemble
    ensemble = OOFEnsemble(task_type='classification')
    
    # Generate OOF predictions
    print("\n1. Generating OOF predictions...")
    oof_predictions = ensemble.get_oof_predictions(
        models=models,
        X=X_train,
        y=y_train,
        n_folds=5,
        stratified=True,
        verbose=True
    )
    
    print(f"\nOOF predictions shape: {oof_predictions.shape}")
    print(f"Columns: {list(oof_predictions.columns)}")
    
    # Train models on full training data for test predictions
    print("\n2. Training models on full data...")
    for model in models:
        model.fit(X_train, y_train)
    
    # Get test predictions
    test_predictions = ensemble.get_test_predictions(models, X_test)
    
    # Simple averaging
    print("\n3. Simple Average Ensemble:")
    avg_pred = test_predictions.mean(axis=1)
    from sklearn.metrics import roc_auc_score
    simple_score = roc_auc_score(y_test, avg_pred)
    print(f"   Test ROC-AUC: {simple_score:.4f}")
    
    return oof_predictions, test_predictions, models


# =============================================================================
# 3. WEIGHTED ENSEMBLE WITH OPTIMIZATION
# =============================================================================

def weighted_ensemble_example():
    """Example of optimizing ensemble weights."""
    print("\n" + "="*60)
    print("WEIGHTED ENSEMBLE EXAMPLE")
    print("="*60)
    
    X_train, X_test, y_train, y_test = create_sample_data()
    
    # Create models with different strengths
    models = [
        XGBoostModel(params={'n_estimators': 200, 'max_depth': 4}),
        LightGBMModel(params={'n_estimators': 150, 'num_leaves': 25}),
        CatBoostModel(params={'iterations': 100, 'depth': 5}),
        # Add a weaker model to show weight optimization
        XGBoostModel(params={'n_estimators': 50, 'max_depth': 3})
    ]
    
    ensemble = OOFEnsemble(task_type='classification')
    
    # Generate OOF predictions
    print("\n1. Generating OOF predictions...")
    oof_predictions = ensemble.get_oof_predictions(
        models, X_train, y_train, n_folds=5, verbose=False
    )
    
    # Optimize weights using different methods
    print("\n2. Optimizing ensemble weights:")
    
    # Method 1: Scipy optimization
    print("\n   a) Scipy optimization:")
    weights_scipy = ensemble.optimize_weights(
        oof_predictions, y_train, method='scipy'
    )
    
    # Method 2: Optuna optimization
    print("\n   b) Optuna optimization:")
    weights_optuna = ensemble.optimize_weights(
        oof_predictions, y_train, method='optuna', n_trials=100
    )
    
    # Method 3: Grid search
    print("\n   c) Grid search:")
    weights_grid = ensemble.optimize_weights(
        oof_predictions, y_train, method='grid'
    )
    
    # Train models on full data
    for model in models:
        model.fit(X_train, y_train)
    
    # Get test predictions
    test_predictions = ensemble.get_test_predictions(models, X_test)
    
    # Compare results
    from sklearn.metrics import roc_auc_score
    
    print("\n3. Test Results:")
    
    # Equal weights
    equal_pred = test_predictions.mean(axis=1)
    print(f"   Equal weights:  {roc_auc_score(y_test, equal_pred):.4f}")
    
    # Optimized weights
    scipy_pred = np.average(test_predictions.values, weights=weights_scipy, axis=1)
    print(f"   Scipy weights:  {roc_auc_score(y_test, scipy_pred):.4f}")
    
    optuna_pred = np.average(test_predictions.values, weights=weights_optuna, axis=1)
    print(f"   Optuna weights: {roc_auc_score(y_test, optuna_pred):.4f}")
    
    return ensemble


# =============================================================================
# 4. STACKING ENSEMBLE
# =============================================================================

def stacking_ensemble_example():
    """Example of stacking with meta-learner."""
    print("\n" + "="*60)
    print("STACKING ENSEMBLE EXAMPLE")
    print("="*60)
    
    X_train, X_test, y_train, y_test = create_sample_data()
    
    # Create diverse base models
    base_models = [
        XGBoostModel(params={'n_estimators': 100, 'max_depth': 6}),
        LightGBMModel(params={'n_estimators': 100, 'num_leaves': 40}),
        CatBoostModel(params={'iterations': 100, 'depth': 7})
    ]
    
    # Add TabNet if available
    try:
        base_models.append(TabNetModel(params={'n_d': 8, 'n_a': 8}))
        print("\nUsing 4 base models (including TabNet)")
    except:
        print("\nUsing 3 base models (TabNet not available)")
    
    ensemble = OOFEnsemble(task_type='classification')
    
    # Generate OOF predictions
    print("\n1. Generating OOF predictions...")
    oof_predictions = ensemble.get_oof_predictions(
        base_models, X_train, y_train, n_folds=5, verbose=False
    )
    
    # Fit stacking ensemble
    print("\n2. Fitting stacking ensemble...")
    
    # Simple stacking
    ensemble.fit_stacking(oof_predictions, y_train)
    
    # Stacking with original features (select most important)
    important_features = X_train.columns[:5]  # Use top 5 features
    ensemble_with_features = OOFEnsemble(task_type='classification')
    ensemble_with_features.fit_stacking(
        oof_predictions, y_train,
        add_original_features=X_train[important_features]
    )
    
    # Train base models on full data
    print("\n3. Training base models on full data...")
    for model in base_models:
        model.fit(X_train, y_train)
    
    # Get test predictions
    test_predictions = ensemble.get_test_predictions(base_models, X_test)
    
    # Generate stacked predictions
    print("\n4. Generating stacked predictions...")
    
    # Simple stacking
    stacked_pred = ensemble.predict_stacking(test_predictions)
    
    # Stacking with features
    stacked_with_features = ensemble_with_features.predict_stacking(
        test_predictions,
        add_original_features=X_test[important_features]
    )
    
    # Compare results
    from sklearn.metrics import roc_auc_score
    
    print("\n5. Test Results:")
    simple_avg = test_predictions.mean(axis=1)
    print(f"   Simple average:           {roc_auc_score(y_test, simple_avg):.4f}")
    print(f"   Stacking:                 {roc_auc_score(y_test, stacked_pred):.4f}")
    print(f"   Stacking with features:   {roc_auc_score(y_test, stacked_with_features):.4f}")
    
    return ensemble


# =============================================================================
# 5. RANK AVERAGING ENSEMBLE
# =============================================================================

def rank_averaging_example():
    """Example of rank averaging for robust ensemble."""
    print("\n" + "="*60)
    print("RANK AVERAGING ENSEMBLE EXAMPLE")
    print("="*60)
    
    X_train, X_test, y_train, y_test = create_sample_data()
    
    # Create models with different prediction scales
    models = [
        XGBoostModel(params={'n_estimators': 100}),  # Outputs probabilities [0, 1]
        LightGBMModel(params={'n_estimators': 100}),  # Outputs probabilities [0, 1]
        # We'll simulate different scales by transforming predictions
    ]
    
    ensemble = OOFEnsemble(task_type='classification')
    
    # Generate OOF predictions
    oof_predictions = ensemble.get_oof_predictions(
        models, X_train, y_train, n_folds=5, verbose=False
    )
    
    # Train models
    for model in models:
        model.fit(X_train, y_train)
    
    # Get test predictions
    test_predictions = ensemble.get_test_predictions(models, X_test)
    
    # Simulate different scales (as if from different model types)
    test_pred_list = [
        test_predictions.iloc[:, 0].values,  # Original scale [0, 1]
        test_predictions.iloc[:, 1].values * 100,  # Scaled to [0, 100]
        test_predictions.iloc[:, 0].values ** 2,  # Squared (more conservative)
    ]
    
    # Rank averaging
    print("\n1. Applying rank averaging...")
    rank_avg_pred = ensemble.rank_average(test_pred_list)
    
    # Geometric mean (for probabilities)
    print("\n2. Applying geometric mean...")
    # Ensure positive values for geometric mean
    test_pred_list_positive = [np.maximum(p, 1e-15) for p in test_pred_list[:2]]
    geo_mean_pred = ensemble.geometric_mean(test_pred_list_positive)
    
    # Compare results
    from sklearn.metrics import roc_auc_score
    
    print("\n3. Test Results:")
    simple_avg = test_predictions.mean(axis=1)
    print(f"   Simple average:   {roc_auc_score(y_test, simple_avg):.4f}")
    print(f"   Rank average:     {roc_auc_score(y_test, rank_avg_pred):.4f}")
    print(f"   Geometric mean:   {roc_auc_score(y_test, geo_mean_pred):.4f}")
    
    return ensemble


# =============================================================================
# 6. AUTO ENSEMBLE - AUTOMATIC STRATEGY SELECTION
# =============================================================================

def auto_ensemble_example():
    """Example of automatic ensemble strategy selection."""
    print("\n" + "="*60)
    print("AUTO ENSEMBLE EXAMPLE")
    print("="*60)
    
    X_train, X_test, y_train, y_test = create_sample_data()
    
    # Create diverse models
    models = [
        XGBoostModel(params={'n_estimators': 150, 'max_depth': 5}),
        LightGBMModel(params={'n_estimators': 150, 'num_leaves': 35}),
        CatBoostModel(params={'iterations': 150, 'depth': 6})
    ]
    
    # Train models first
    print("\n1. Training base models...")
    for i, model in enumerate(models):
        print(f"   Training model {i+1}/{len(models)}...")
        model.fit(X_train, y_train)
    
    # Create auto ensemble
    print("\n2. Finding best ensemble strategy...")
    auto = AutoEnsemble(task_type='classification')
    
    # Automatically find best strategy
    auto.fit(
        models, X_train, y_train,
        strategies=['weighted', 'stacking', 'rank'],
        cv_folds=5
    )
    
    print(f"\n3. Best strategy selected: {auto.best_strategy}")
    print("\n   Scores for each strategy:")
    for strategy, score in auto.ensemble_scores.items():
        print(f"   - {strategy}: {score:.4f}")
    
    # Generate predictions using best strategy
    print("\n4. Generating predictions with best strategy...")
    final_predictions = auto.predict(models, X_test)
    
    # Evaluate
    from sklearn.metrics import roc_auc_score
    final_score = roc_auc_score(y_test, final_predictions)
    print(f"\n5. Final test score: {final_score:.4f}")
    
    return auto


# =============================================================================
# 7. KAGGLE COMPETITION WORKFLOW
# =============================================================================

def kaggle_competition_workflow():
    """Complete workflow for a Kaggle competition."""
    print("\n" + "="*60)
    print("KAGGLE COMPETITION WORKFLOW WITH OOF ENSEMBLE")
    print("="*60)
    
    # In a real competition, load your data here
    print("\n1. Loading and preparing data...")
    X_train, X_test, y_train, y_test = create_sample_data()
    
    # For demonstration, we'll use y_test as a hold-out validation set
    # In real competition, X_test would be the actual test set without labels
    
    print("\n2. Creating diverse models...")
    models = [
        XGBoostModel(params={
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8
        }),
        LightGBMModel(params={
            'n_estimators': 300,
            'num_leaves': 40,
            'learning_rate': 0.05,
            'feature_fraction': 0.8
        }),
        CatBoostModel(params={
            'iterations': 300,
            'depth': 7,
            'learning_rate': 0.05
        })
    ]
    
    # Try adding TabNet
    try:
        models.append(TabNetModel(params={
            'n_d': 16,
            'n_a': 16,
            'n_steps': 3
        }))
        print(f"   Created {len(models)} diverse models")
    except:
        print(f"   Created {len(models)} models (TabNet unavailable)")
    
    print("\n3. Generating OOF predictions...")
    ensemble = OOFEnsemble(task_type='classification')
    oof_predictions = ensemble.get_oof_predictions(
        models, X_train, y_train,
        n_folds=5,
        stratified=True,
        verbose=False
    )
    
    print("\n4. Finding optimal ensemble weights...")
    weights = ensemble.optimize_weights(
        oof_predictions, y_train,
        method='optuna',
        n_trials=200
    )
    
    print("\n5. Trying stacking ensemble...")
    stacking_ensemble = OOFEnsemble(task_type='classification')
    stacking_ensemble.fit_stacking(oof_predictions, y_train)
    
    print("\n6. Training models on full training data...")
    for i, model in enumerate(models):
        print(f"   Training model {i+1}/{len(models)}...")
        model.fit(X_train, y_train)
    
    print("\n7. Generating test predictions...")
    test_predictions = ensemble.get_test_predictions(models, X_test)
    
    # Create different ensemble predictions
    print("\n8. Creating ensemble predictions:")
    
    # Weighted average
    weighted_pred = np.average(test_predictions.values, weights=weights, axis=1)
    
    # Stacking
    stacked_pred = stacking_ensemble.predict_stacking(test_predictions)
    
    # Rank averaging
    rank_pred = ensemble.rank_average([test_predictions[col].values for col in test_predictions.columns])
    
    # Final blend (average of different strategies)
    final_pred = (weighted_pred + stacked_pred + rank_pred) / 3
    
    # Evaluate (in real competition, you wouldn't have y_test)
    from sklearn.metrics import roc_auc_score
    
    print("\n9. Validation Results:")
    print(f"   Weighted ensemble: {roc_auc_score(y_test, weighted_pred):.4f}")
    print(f"   Stacking ensemble: {roc_auc_score(y_test, stacked_pred):.4f}")
    print(f"   Rank ensemble:     {roc_auc_score(y_test, rank_pred):.4f}")
    print(f"   Final blend:       {roc_auc_score(y_test, final_pred):.4f}")
    
    # Create submission (in real competition)
    print("\n10. Creating submission file...")
    submission = pd.DataFrame({
        'id': range(len(final_pred)),
        'target': final_pred
    })
    
    # Save submission
    submission.to_csv('submission.csv', index=False)
    print("    Submission saved to 'submission.csv'")
    
    return ensemble, final_pred


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║          TabML OOF ENSEMBLE CAPABILITIES DEMO            ║
    ╠══════════════════════════════════════════════════════════╣
    ║  This demo shows how to use TabML's advanced OOF         ║
    ║  ensemble features for Kaggle competitions:              ║
    ║                                                          ║
    ║  • OOF prediction generation                            ║
    ║  • Weight optimization (Scipy, Optuna, Grid)            ║
    ║  • Stacking with meta-learners                          ║
    ║  • Rank averaging for robust ensembles                  ║
    ║  • Geometric mean for probability ensembles             ║
    ║  • Automatic ensemble strategy selection                ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Run examples
    print("\nRunning examples...\n")
    
    # 1. Basic OOF
    oof_preds, test_preds, models = basic_oof_ensemble_example()
    
    # 2. Weighted ensemble
    weighted_ens = weighted_ensemble_example()
    
    # 3. Stacking
    stacking_ens = stacking_ensemble_example()
    
    # 4. Rank averaging
    rank_ens = rank_averaging_example()
    
    # 5. Auto ensemble
    auto_ens = auto_ensemble_example()
    
    # 6. Full Kaggle workflow
    final_ens, predictions = kaggle_competition_workflow()
    
    print("\n" + "="*60)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("""
    TabML now provides comprehensive OOF ensemble capabilities:
    
    ✅ OOF prediction generation with cross-validation
    ✅ Multiple weight optimization methods
    ✅ Stacking with meta-learners
    ✅ Rank averaging for different scales
    ✅ Geometric mean for probabilities
    ✅ Automatic strategy selection
    ✅ Complete Kaggle competition workflow
    
    These features match and exceed what's available in most
    ensemble libraries, giving you powerful tools for winning
    Kaggle competitions!
    """)