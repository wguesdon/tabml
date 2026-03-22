"""Example of using TabML for Kaggle competition with OOF saving and ensemble.

This example demonstrates:
1. Training multiple models with different parameters
2. Saving OOF predictions for each model
3. Loading and combining OOF predictions
4. Creating optimized ensemble
5. Generating final submission
"""

import pandas as pd
import numpy as np
from tabml import (
    DataLoader, FeatureEngineer, 
    XGBoostModel, LightGBMModel, CatBoostModel,
    OOFEnsemble
)
from tabml.oof_manager import OOFManager
import os


def train_single_model_with_oof(model, X_train, y_train, X_test, model_name, oof_manager, n_folds=5):
    """Train a single model and save its OOF predictions."""
    
    # Create ensemble object for OOF generation
    ensemble = OOFEnsemble(task_type='classification')
    
    # Generate OOF predictions
    print(f"\nGenerating OOF for {model_name}...")
    oof_preds = ensemble.get_oof_predictions(
        models=[model],
        X=X_train,
        y=y_train,
        n_folds=n_folds,
        verbose=True
    )
    
    # Get test predictions
    print(f"Generating test predictions for {model_name}...")
    test_preds = ensemble.get_test_predictions([model], X_test)
    
    # Calculate CV score (example using log loss or AUC)
    from sklearn.metrics import roc_auc_score
    cv_score = roc_auc_score(y_train, oof_preds.iloc[:, 0])
    
    # Save OOF and test predictions
    oof_manager.save_oof(
        predictions=oof_preds,
        model_name=model_name,
        model_params=model.params if hasattr(model, 'params') else {},
        cv_score=cv_score,
        test_predictions=test_preds,
        tags={'competition': 'kaggle_example', 'cv_folds': n_folds}
    )
    
    return oof_preds, test_preds, cv_score


def main():
    # Setup directories
    output_dir = "output/kaggle_competition"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize OOF manager
    oof_manager = OOFManager(output_dir=output_dir)
    
    # Load data
    print("Loading data...")
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
    
    # ============================================
    # PHASE 1: Train Multiple Models and Save OOFs
    # ============================================
    
    print("\n" + "="*50)
    print("PHASE 1: Training Models and Saving OOFs")
    print("="*50)
    
    all_test_preds = []
    
    # Model 1: XGBoost with different parameters
    for i, params in enumerate([
        {'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.01},
        {'n_estimators': 1000, 'max_depth': 4, 'learning_rate': 0.005},
        {'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.02}
    ]):
        model = XGBoostModel(params=params)
        oof, test_pred, score = train_single_model_with_oof(
            model, X_train, y_train, X_test, 
            f"xgboost_v{i+1}", oof_manager
        )
        all_test_preds.append(test_pred)
        print(f"XGBoost v{i+1} CV Score: {score:.4f}")
    
    # Model 2: LightGBM with different parameters
    for i, params in enumerate([
        {'n_estimators': 500, 'num_leaves': 31, 'learning_rate': 0.01},
        {'n_estimators': 800, 'num_leaves': 20, 'learning_rate': 0.008}
    ]):
        model = LightGBMModel(params=params)
        oof, test_pred, score = train_single_model_with_oof(
            model, X_train, y_train, X_test,
            f"lightgbm_v{i+1}", oof_manager
        )
        all_test_preds.append(test_pred)
        print(f"LightGBM v{i+1} CV Score: {score:.4f}")
    
    # Model 3: CatBoost
    model = CatBoostModel(params={'iterations': 500, 'depth': 6, 'learning_rate': 0.01})
    oof, test_pred, score = train_single_model_with_oof(
        model, X_train, y_train, X_test,
        "catboost_v1", oof_manager
    )
    all_test_preds.append(test_pred)
    print(f"CatBoost v1 CV Score: {score:.4f}")
    
    # ============================================
    # PHASE 2: View All Saved OOFs
    # ============================================
    
    print("\n" + "="*50)
    print("PHASE 2: Viewing Saved OOF Predictions")
    print("="*50)
    
    # List all saved OOFs
    summary = oof_manager.list_oofs(sort_by='cv_score', ascending=False)
    print("\nAll saved OOF predictions:")
    print(summary[['model_name', 'cv_score', 'timestamp']])
    
    # Export summary
    oof_manager.export_summary(f"{output_dir}/oof_summary.csv")
    
    # ============================================
    # PHASE 3: Load OOFs and Create Ensemble
    # ============================================
    
    print("\n" + "="*50)
    print("PHASE 3: Loading OOFs and Creating Ensemble")
    print("="*50)
    
    # Load top 5 models
    print("\nLoading top 5 models by CV score...")
    top_oofs = oof_manager.load_all_oofs(top_k=5)
    
    # Combine OOFs for ensemble
    combined_oofs = oof_manager.combine_oofs(top_oofs, method='horizontal')
    print(f"Combined OOF shape: {combined_oofs.shape}")
    
    # Create ensemble with different strategies
    ensemble = OOFEnsemble(task_type='classification')
    
    # Method 1: Optimize weights
    print("\nOptimizing ensemble weights...")
    weights = ensemble.optimize_weights(
        combined_oofs, y_train,
        method='optuna',
        n_trials=100
    )
    
    # Method 2: Stacking
    print("\nFitting stacking ensemble...")
    ensemble.fit_stacking(combined_oofs, y_train)
    
    # Get test predictions for ensemble
    combined_test = oof_manager.combine_oofs(top_oofs, method='horizontal', use_test=True)
    
    # Generate final predictions
    print("\nGenerating final ensemble predictions...")
    
    # Weighted average
    final_pred_weighted = np.average(combined_test.values, weights=weights, axis=1)
    
    # Stacking predictions
    final_pred_stacking = ensemble.predict_stacking(combined_test)
    
    # Simple average (baseline)
    final_pred_simple = combined_test.mean(axis=1)
    
    # ============================================
    # PHASE 4: Create Submission
    # ============================================
    
    print("\n" + "="*50)
    print("PHASE 4: Creating Submission Files")
    print("="*50)
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'id': test_df.index,
        'target': final_pred_weighted  # Use best ensemble method
    })
    
    # Save different versions
    submission.to_csv(f"{output_dir}/submission_weighted.csv", index=False)
    
    submission['target'] = final_pred_stacking
    submission.to_csv(f"{output_dir}/submission_stacking.csv", index=False)
    
    submission['target'] = final_pred_simple
    submission.to_csv(f"{output_dir}/submission_simple.csv", index=False)
    
    print(f"\nSubmission files saved to {output_dir}/")
    
    # ============================================
    # PHASE 5: Clean Up Old OOFs (Optional)
    # ============================================
    
    print("\n" + "="*50)
    print("PHASE 5: Cleanup (Optional)")
    print("="*50)
    
    # Keep only top 20 models and files newer than 30 days
    # oof_manager.cleanup_old_oofs(keep_top_k=20, keep_days=30)
    
    print("\nExample complete! You can now:")
    print("1. Train more models and save their OOFs")
    print("2. Load OOFs from previous runs to create new ensembles")
    print("3. Experiment with different ensemble strategies")
    print("4. Share OOF predictions with teammates for mega-ensembles")


def advanced_example_load_previous_oofs():
    """Example of loading OOFs from previous runs for ensemble."""
    
    print("\n" + "="*50)
    print("ADVANCED: Loading OOFs from Previous Runs")
    print("="*50)
    
    output_dir = "output/kaggle_competition"
    oof_manager = OOFManager(output_dir=output_dir)
    
    # Load data for target variable
    loader = DataLoader(data_dir="data")
    train_df, _ = loader.load_data(train_file="train.csv", test_file="test.csv")
    y_train = train_df['target']
    
    # Load all OOFs with CV score > 0.85
    print("\nLoading all models with CV score > 0.85...")
    good_oofs = oof_manager.load_all_oofs(min_cv_score=0.85)
    print(f"Found {len(good_oofs)} models")
    
    # Load OOFs from specific experiment
    print("\nLoading OOFs from specific experiment...")
    experiment_oofs = oof_manager.load_all_oofs(
        experiment_name="feature_set_v2",
        top_k=10
    )
    
    # Load OOFs with specific tags
    print("\nLoading OOFs with specific tags...")
    tagged_oofs = oof_manager.load_all_oofs(
        tags_filter={'validation': 'time_series', 'feature_version': 'v3'}
    )
    
    # Combine and create mega-ensemble
    all_selected_oofs = {**good_oofs, **experiment_oofs, **tagged_oofs}
    combined = oof_manager.combine_oofs(all_selected_oofs, method='horizontal')
    
    print(f"\nCreating mega-ensemble from {combined.shape[1]} models")
    
    # Use AutoEnsemble to find best strategy
    from tabml import AutoEnsemble
    
    auto = AutoEnsemble(task_type='classification')
    # Note: This would need the actual models, but shows the concept
    # auto.fit(models, X_train, y_train, strategies=['weighted', 'stacking', 'rank'])
    
    print("Mega-ensemble created successfully!")


if __name__ == "__main__":
    # Run main example
    main()
    
    # Uncomment to run advanced example
    # advanced_example_load_previous_oofs()