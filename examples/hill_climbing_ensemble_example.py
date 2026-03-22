"""Hill Climbing Ensemble Optimization Example for Kaggle Competitions.

This example demonstrates using hill climbing and greedy forward selection
to optimize ensemble weights for Kaggle submissions.
"""

import pandas as pd
import numpy as np
from tabml import (
    DataLoader, FeatureEngineer,
    XGBoostModel, LightGBMModel, CatBoostModel, RandomForestModel,
    OOFEnsemble, OOFManager
)
from sklearn.metrics import roc_auc_score, log_loss
import matplotlib.pyplot as plt
from pathlib import Path


def compare_optimization_methods(oof_predictions, y_true, test_predictions=None):
    """Compare different weight optimization methods."""
    
    ensemble = OOFEnsemble(task_type='classification')
    results = {}
    
    print("\n" + "="*60)
    print("COMPARING WEIGHT OPTIMIZATION METHODS")
    print("="*60)
    
    # Method 1: Equal weights (baseline)
    print("\n1. Equal Weights (Baseline):")
    equal_weights = np.ones(oof_predictions.shape[1]) / oof_predictions.shape[1]
    equal_pred = np.average(oof_predictions.values, weights=equal_weights, axis=1)
    equal_score = roc_auc_score(y_true, equal_pred)
    print(f"   Score: {equal_score:.6f}")
    print(f"   Weights: {equal_weights}")
    results['equal'] = {'score': equal_score, 'weights': equal_weights}
    
    # Method 2: Grid Search
    print("\n2. Grid Search:")
    grid_weights = ensemble.optimize_weights(oof_predictions, y_true, method='grid')
    grid_pred = np.average(oof_predictions.values, weights=grid_weights, axis=1)
    grid_score = roc_auc_score(y_true, grid_pred)
    print(f"   Score: {grid_score:.6f}")
    print(f"   Weights: {grid_weights}")
    results['grid'] = {'score': grid_score, 'weights': grid_weights}
    
    # Method 3: SciPy Optimization
    print("\n3. SciPy Optimization:")
    scipy_weights = ensemble.optimize_weights(oof_predictions, y_true, method='scipy')
    scipy_pred = np.average(oof_predictions.values, weights=scipy_weights, axis=1)
    scipy_score = roc_auc_score(y_true, scipy_pred)
    print(f"   Score: {scipy_score:.6f}")
    print(f"   Weights: {scipy_weights}")
    results['scipy'] = {'score': scipy_score, 'weights': scipy_weights}
    
    # Method 4: Optuna
    print("\n4. Optuna Optimization (100 trials):")
    optuna_weights = ensemble.optimize_weights(oof_predictions, y_true, method='optuna', n_trials=100)
    optuna_pred = np.average(oof_predictions.values, weights=optuna_weights, axis=1)
    optuna_score = roc_auc_score(y_true, optuna_pred)
    print(f"   Score: {optuna_score:.6f}")
    print(f"   Weights: {optuna_weights}")
    results['optuna'] = {'score': optuna_score, 'weights': optuna_weights}
    
    # Method 5: Hill Climbing (NEW)
    print("\n5. Hill Climbing Optimization:")
    hill_weights = ensemble.optimize_weights(
        oof_predictions, y_true, 
        method='hill_climbing',
        n_iterations=2000,
        patience=200
    )
    hill_pred = np.average(oof_predictions.values, weights=hill_weights, axis=1)
    hill_score = roc_auc_score(y_true, hill_pred)
    print(f"   Score: {hill_score:.6f}")
    print(f"   Weights: {hill_weights}")
    results['hill_climbing'] = {'score': hill_score, 'weights': hill_weights}
    
    # Method 6: Greedy Forward Selection (NEW)
    print("\n6. Greedy Forward Selection:")
    greedy_weights = ensemble.optimize_weights(
        oof_predictions, y_true,
        method='greedy_forward'
    )
    greedy_pred = np.average(oof_predictions.values, weights=greedy_weights, axis=1)
    greedy_score = roc_auc_score(y_true, greedy_pred)
    print(f"   Score: {greedy_score:.6f}")
    print(f"   Selected models (non-zero weights): {np.where(greedy_weights > 0)[0]}")
    print(f"   Weights: {greedy_weights[greedy_weights > 0]}")
    results['greedy_forward'] = {'score': greedy_score, 'weights': greedy_weights}
    
    # Find best method
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    best_method = max(results.items(), key=lambda x: x[1]['score'])
    print(f"\nBest Method: {best_method[0].upper()}")
    print(f"Best Score: {best_method[1]['score']:.6f}")
    print(f"Improvement over baseline: {(best_method[1]['score'] - equal_score)*100:.3f}%")
    
    # Generate test predictions if provided
    if test_predictions is not None:
        print("\n" + "="*60)
        print("GENERATING TEST PREDICTIONS")
        print("="*60)
        
        best_weights = best_method[1]['weights']
        final_test_pred = np.average(test_predictions.values, weights=best_weights, axis=1)
        print(f"Using {best_method[0]} weights for final submission")
        
        return final_test_pred, results
    
    return None, results


def visualize_weights(results, model_names):
    """Visualize weights from different optimization methods."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (method_name, result) in enumerate(results.items()):
        ax = axes[idx]
        weights = result['weights']
        score = result['score']
        
        # Create bar plot
        colors = ['green' if w > 0.1 else 'lightgray' for w in weights]
        bars = ax.bar(range(len(weights)), weights, color=colors)
        
        ax.set_title(f'{method_name.upper()}\nScore: {score:.6f}')
        ax.set_xlabel('Model Index')
        ax.set_ylabel('Weight')
        ax.set_xticks(range(len(weights)))
        ax.set_xticklabels([f'M{i}' for i in range(len(weights))], rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Highlight significant weights
        for i, (bar, weight) in enumerate(zip(bars, weights)):
            if weight > 0.1:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{weight:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Weight Distribution Across Different Optimization Methods', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def main():
    """Main example demonstrating hill climbing ensemble optimization."""
    
    # Setup
    output_dir = Path("output/hill_climbing_example")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("HILL CLIMBING ENSEMBLE OPTIMIZATION EXAMPLE")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    loader = DataLoader(data_dir="data")
    train_df, test_df = loader.load_data(
        train_file="train.csv",
        test_file="test.csv"
    )
    
    # Prepare features
    print("Engineering features...")
    engineer = FeatureEngineer()
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df
    
    X_train = engineer.fit_transform(X_train, y_train)
    X_test = engineer.transform(X_test)
    
    # Train diverse models for ensemble
    print("\n" + "="*60)
    print("TRAINING DIVERSE MODELS")
    print("="*60)
    
    models = []
    model_names = []
    
    # Model 1: XGBoost (conservative)
    print("\n1. Training XGBoost (conservative)...")
    xgb1 = XGBoostModel(params={
        'n_estimators': 300,
        'max_depth': 4,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    })
    models.append(xgb1)
    model_names.append("XGB_conservative")
    
    # Model 2: XGBoost (aggressive)
    print("2. Training XGBoost (aggressive)...")
    xgb2 = XGBoostModel(params={
        'n_estimators': 500,
        'max_depth': 8,
        'learning_rate': 0.02,
        'subsample': 0.9,
        'colsample_bytree': 0.9
    })
    models.append(xgb2)
    model_names.append("XGB_aggressive")
    
    # Model 3: LightGBM
    print("3. Training LightGBM...")
    lgb = LightGBMModel(params={
        'n_estimators': 400,
        'num_leaves': 31,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8
    })
    models.append(lgb)
    model_names.append("LightGBM")
    
    # Model 4: CatBoost
    print("4. Training CatBoost...")
    cat = CatBoostModel(params={
        'iterations': 300,
        'depth': 6,
        'learning_rate': 0.01,
        'l2_leaf_reg': 3
    })
    models.append(cat)
    model_names.append("CatBoost")
    
    # Model 5: Random Forest
    print("5. Training Random Forest...")
    rf = RandomForestModel(params={
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_split': 20,
        'min_samples_leaf': 10
    })
    models.append(rf)
    model_names.append("RandomForest")
    
    # Model 6: LightGBM (dart mode)
    print("6. Training LightGBM (DART)...")
    lgb_dart = LightGBMModel(params={
        'n_estimators': 300,
        'num_leaves': 20,
        'learning_rate': 0.015,
        'boosting_type': 'dart',
        'drop_rate': 0.1
    })
    models.append(lgb_dart)
    model_names.append("LGB_DART")
    
    # Generate OOF predictions
    print("\n" + "="*60)
    print("GENERATING OUT-OF-FOLD PREDICTIONS")
    print("="*60)
    
    ensemble = OOFEnsemble(task_type='classification')
    oof_manager = OOFManager(output_dir=str(output_dir))
    
    # Get OOF predictions for all models
    oof_predictions = ensemble.get_oof_predictions(
        models=models,
        X=X_train,
        y=y_train,
        n_folds=5,
        verbose=True
    )
    
    # Also get test predictions
    test_predictions = ensemble.get_test_predictions(models, X_test)
    
    # Save OOF predictions for each model
    for i, model_name in enumerate(model_names):
        cv_score = roc_auc_score(y_train, oof_predictions.iloc[:, i])
        oof_manager.save_oof(
            predictions=oof_predictions.iloc[:, i],
            model_name=model_name,
            cv_score=cv_score,
            test_predictions=test_predictions.iloc[:, i],
            tags={'experiment': 'hill_climbing_demo'}
        )
    
    # Compare optimization methods
    final_test_pred, results = compare_optimization_methods(
        oof_predictions, y_train, test_predictions
    )
    
    # Visualize weights
    print("\n" + "="*60)
    print("VISUALIZING WEIGHT DISTRIBUTIONS")
    print("="*60)
    
    fig = visualize_weights(results, model_names)
    fig.savefig(output_dir / "weight_comparison.png", dpi=100, bbox_inches='tight')
    print(f"Weight visualization saved to {output_dir}/weight_comparison.png")
    
    # Create submissions
    if final_test_pred is not None:
        print("\n" + "="*60)
        print("CREATING SUBMISSION FILES")
        print("="*60)
        
        # Best method submission
        submission = pd.DataFrame({
            'id': test_df.index,
            'target': final_test_pred
        })
        submission.to_csv(output_dir / "submission_best.csv", index=False)
        
        # Also save submissions for each method
        for method_name, result in results.items():
            weights = result['weights']
            test_pred = np.average(test_predictions.values, weights=weights, axis=1)
            
            submission = pd.DataFrame({
                'id': test_df.index,
                'target': test_pred
            })
            submission.to_csv(output_dir / f"submission_{method_name}.csv", index=False)
        
        print(f"All submissions saved to {output_dir}/")
    
    # Performance analysis
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Sort methods by score
    sorted_methods = sorted(results.items(), key=lambda x: x[1]['score'], reverse=True)
    
    print("\nMethods ranked by performance:")
    for rank, (method, result) in enumerate(sorted_methods, 1):
        score = result['score']
        n_models_used = np.sum(result['weights'] > 0.01)  # Models with >1% weight
        print(f"{rank}. {method:20s} - Score: {score:.6f} - Active models: {n_models_used}/{len(models)}")
    
    # Analyze model importance
    print("\n" + "="*60)
    print("MODEL IMPORTANCE ANALYSIS")
    print("="*60)
    
    model_importance = np.zeros(len(models))
    for method, result in results.items():
        # Weight importance by method performance
        method_weight = result['score'] / sum(r['score'] for r in results.values())
        model_importance += result['weights'] * method_weight
    
    # Normalize
    model_importance = model_importance / model_importance.sum()
    
    print("\nOverall Model Importance (weighted by method performance):")
    for model_name, importance in sorted(zip(model_names, model_importance), 
                                         key=lambda x: x[1], reverse=True):
        print(f"  {model_name:20s}: {importance:.3f}")
    
    print("\n" + "="*60)
    print("EXAMPLE COMPLETE!")
    print("="*60)
    print("\nKey Findings:")
    print("1. Hill climbing often finds better local optima than grid search")
    print("2. Greedy forward selection identifies the most complementary models")
    print("3. Not all models contribute equally to the ensemble")
    print("4. The best ensemble method depends on your specific models and data")


if __name__ == "__main__":
    main()