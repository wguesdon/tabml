"""
PS5E8 Competition - Hill Climbing Ensemble Optimization
Load saved OOF predictions and optimize ensemble weights using hill climbing
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path to import tabml
sys.path.append(str(Path(__file__).parent.parent.parent))

from tabml import OOFEnsemble, OOFManager, MLflowTracker

# Load environment variables
load_dotenv()

# Setup paths
DATA_DIR = Path("../../data/raw/ PS5E8")
OUTPUT_DIR = Path("output")
OOF_DIR = OUTPUT_DIR / "oof_predictions"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Create directories
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Competition settings
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
TARGET_COL = 'y'


def load_train_labels():
    """Load training labels for evaluation."""
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    return train_df[TARGET_COL]


def compare_ensemble_methods(oof_manager, y_true, min_cv_score=0.0, top_k=None):
    """Compare different ensemble optimization methods."""
    
    logger.info("="*60)
    logger.info("COMPARING ENSEMBLE OPTIMIZATION METHODS")
    logger.info("="*60)
    
    # Load all OOF predictions
    logger.info(f"\nLoading OOF predictions (min_cv_score={min_cv_score}, top_k={top_k})...")
    all_oofs = oof_manager.load_all_oofs(
        experiment_name='PS5E8_baseline',
        min_cv_score=min_cv_score,
        top_k=top_k
    )
    
    if not all_oofs:
        logger.error("No OOF predictions found! Run 02_baseline_models.py first.")
        return None, None, None
    
    logger.info(f"Loaded {len(all_oofs)} models")
    
    # Combine OOF predictions
    combined_oofs = oof_manager.combine_oofs(all_oofs, method='horizontal')
    
    # Get model names and scores
    model_info = []
    for filename, data in all_oofs.items():
        model_info.append({
            'name': data['model_name'],
            'cv_score': data.get('cv_score', 0),
            'filename': filename
        })
    
    model_df = pd.DataFrame(model_info).sort_values('cv_score', ascending=False)
    
    logger.info("\nLoaded Models:")
    for idx, row in model_df.iterrows():
        logger.info(f"  {row['name']:20s}: {row['cv_score']:.6f}")
    
    # Initialize ensemble
    ensemble = OOFEnsemble(task_type='classification')
    
    # Store results
    results = {}
    test_predictions = {}
    
    # Get test predictions
    combined_test = oof_manager.combine_oofs(all_oofs, method='horizontal', use_test=True)
    
    # Method 1: Equal weights (baseline)
    logger.info("\n" + "-"*40)
    logger.info("1. EQUAL WEIGHTS (Baseline)")
    equal_weights = np.ones(combined_oofs.shape[1]) / combined_oofs.shape[1]
    equal_pred = np.average(combined_oofs.values, weights=equal_weights, axis=1)
    equal_score = roc_auc_score(y_true, equal_pred)
    logger.info(f"Score: {equal_score:.6f}")
    results['Equal'] = {'score': equal_score, 'weights': equal_weights}
    test_predictions['Equal'] = np.average(combined_test.values, weights=equal_weights, axis=1)
    
    # Method 2: Weighted by CV scores
    logger.info("\n" + "-"*40)
    logger.info("2. WEIGHTED BY CV SCORES")
    cv_scores = np.array([data.get('cv_score', 0.5) for data in all_oofs.values()])
    cv_weights = cv_scores / cv_scores.sum()
    cv_pred = np.average(combined_oofs.values, weights=cv_weights, axis=1)
    cv_score = roc_auc_score(y_true, cv_pred)
    logger.info(f"Score: {cv_score:.6f}")
    results['CV_Weighted'] = {'score': cv_score, 'weights': cv_weights}
    test_predictions['CV_Weighted'] = np.average(combined_test.values, weights=cv_weights, axis=1)
    
    # Method 3: SciPy optimization
    logger.info("\n" + "-"*40)
    logger.info("3. SCIPY OPTIMIZATION")
    scipy_weights = ensemble.optimize_weights(combined_oofs, y_true, method='scipy')
    scipy_pred = np.average(combined_oofs.values, weights=scipy_weights, axis=1)
    scipy_score = roc_auc_score(y_true, scipy_pred)
    logger.info(f"Score: {scipy_score:.6f}")
    results['SciPy'] = {'score': scipy_score, 'weights': scipy_weights}
    test_predictions['SciPy'] = np.average(combined_test.values, weights=scipy_weights, axis=1)
    
    # Method 4: Optuna optimization
    logger.info("\n" + "-"*40)
    logger.info("4. OPTUNA OPTIMIZATION")
    optuna_weights = ensemble.optimize_weights(combined_oofs, y_true, method='optuna', n_trials=200)
    optuna_pred = np.average(combined_oofs.values, weights=optuna_weights, axis=1)
    optuna_score = roc_auc_score(y_true, optuna_pred)
    logger.info(f"Score: {optuna_score:.6f}")
    results['Optuna'] = {'score': optuna_score, 'weights': optuna_weights}
    test_predictions['Optuna'] = np.average(combined_test.values, weights=optuna_weights, axis=1)
    
    # Method 5: Hill Climbing (MAIN METHOD)
    logger.info("\n" + "-"*40)
    logger.info("5. HILL CLIMBING OPTIMIZATION")
    logger.info("Running hill climbing with 3000 iterations...")
    hill_weights = ensemble.optimize_weights(
        combined_oofs, y_true,
        method='hill_climbing',
        n_iterations=3000,
        patience=300
    )
    hill_pred = np.average(combined_oofs.values, weights=hill_weights, axis=1)
    hill_score = roc_auc_score(y_true, hill_pred)
    logger.info(f"Score: {hill_score:.6f}")
    results['Hill_Climbing'] = {'score': hill_score, 'weights': hill_weights}
    test_predictions['Hill_Climbing'] = np.average(combined_test.values, weights=hill_weights, axis=1)
    
    # Method 6: Greedy Forward Selection
    logger.info("\n" + "-"*40)
    logger.info("6. GREEDY FORWARD SELECTION")
    greedy_weights = ensemble.optimize_weights(combined_oofs, y_true, method='greedy_forward')
    greedy_pred = np.average(combined_oofs.values, weights=greedy_weights, axis=1)
    greedy_score = roc_auc_score(y_true, greedy_pred)
    n_selected = np.sum(greedy_weights > 0)
    logger.info(f"Score: {greedy_score:.6f}")
    logger.info(f"Models selected: {n_selected}/{len(greedy_weights)}")
    results['Greedy_Forward'] = {'score': greedy_score, 'weights': greedy_weights}
    test_predictions['Greedy_Forward'] = np.average(combined_test.values, weights=greedy_weights, axis=1)
    
    # Method 7: Stacking
    logger.info("\n" + "-"*40)
    logger.info("7. STACKING WITH META-LEARNER")
    ensemble_stack = OOFEnsemble(task_type='classification')
    ensemble_stack.fit_stacking(combined_oofs, y_true)
    stack_pred = ensemble_stack.predict_stacking(combined_oofs)
    stack_score = roc_auc_score(y_true, stack_pred)
    logger.info(f"Score: {stack_score:.6f}")
    results['Stacking'] = {'score': stack_score, 'weights': None}
    test_predictions['Stacking'] = ensemble_stack.predict_stacking(combined_test)
    
    return results, test_predictions, model_df


def visualize_results(results, model_df, output_dir):
    """Create visualizations of ensemble results."""
    
    logger.info("\n" + "="*60)
    logger.info("CREATING VISUALIZATIONS")
    logger.info("="*60)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Figure 1: Method comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Score comparison
    ax = axes[0, 0]
    methods = list(results.keys())
    scores = [r['score'] for r in results.values()]
    colors = ['red' if m == 'Equal' else 'gold' if m == 'Hill_Climbing' else 'steelblue' for m in methods]
    
    bars = ax.bar(methods, scores, color=colors, alpha=0.7, edgecolor='black')
    ax.set_title('Ensemble Method Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('ROC AUC Score', fontsize=12)
    ax.set_ylim([min(scores) - 0.001, max(scores) + 0.001])
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                f'{score:.6f}', ha='center', va='bottom', fontsize=10)
    
    ax.tick_params(axis='x', rotation=45)
    
    # Plot 2: Weight distribution for Hill Climbing
    ax = axes[0, 1]
    if 'Hill_Climbing' in results and results['Hill_Climbing']['weights'] is not None:
        weights = results['Hill_Climbing']['weights']
        model_names = model_df['name'].values[:len(weights)]
        
        # Sort by weight
        sorted_idx = np.argsort(weights)[::-1]
        sorted_weights = weights[sorted_idx]
        sorted_names = [model_names[i] for i in sorted_idx]
        
        bars = ax.bar(range(len(sorted_weights)), sorted_weights, 
                      color=['green' if w > 0.1 else 'lightgray' for w in sorted_weights])
        ax.set_title('Hill Climbing Weight Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Weight', fontsize=12)
        ax.set_xticks(range(len(sorted_names)))
        ax.set_xticklabels([n[:10] for n in sorted_names], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Highlight top weights
        for i, (bar, weight) in enumerate(zip(bars, sorted_weights)):
            if weight > 0.1:
                ax.text(bar.get_x() + bar.get_width()/2, weight + 0.01,
                       f'{weight:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Score improvement over baseline
    ax = axes[1, 0]
    baseline_score = results['Equal']['score']
    improvements = [(r['score'] - baseline_score) * 100 for r in results.values()]
    
    colors = ['red' if imp <= 0 else 'green' for imp in improvements]
    bars = ax.bar(methods, improvements, color=colors, alpha=0.7, edgecolor='black')
    ax.set_title('Improvement Over Equal Weights Baseline', fontsize=14, fontweight='bold')
    ax.set_ylabel('Improvement (%)', fontsize=12)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        y_pos = height + 0.01 if height > 0 else height - 0.02
        va = 'bottom' if height > 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{imp:.3f}%', ha='center', va=va, fontsize=10)
    
    ax.tick_params(axis='x', rotation=45)
    
    # Plot 4: Model importance (average across methods)
    ax = axes[1, 1]
    n_models = len(model_df)
    model_importance = np.zeros(n_models)
    
    # Calculate average weight across methods (excluding Equal and Stacking)
    weight_methods = ['CV_Weighted', 'SciPy', 'Optuna', 'Hill_Climbing', 'Greedy_Forward']
    for method in weight_methods:
        if method in results and results[method]['weights'] is not None:
            weights = results[method]['weights'][:n_models]
            # Normalize by method performance
            method_score = results[method]['score']
            model_importance += weights * method_score
    
    # Normalize
    if model_importance.sum() > 0:
        model_importance = model_importance / model_importance.sum()
    
    # Sort by importance
    sorted_idx = np.argsort(model_importance)[::-1]
    sorted_importance = model_importance[sorted_idx]
    sorted_names = model_df['name'].values[sorted_idx]
    
    bars = ax.bar(range(len(sorted_importance)), sorted_importance,
                  color='coral', alpha=0.7, edgecolor='black')
    ax.set_title('Overall Model Importance', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Importance Score', fontsize=12)
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels([n[:10] for n in sorted_names], rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('PS5E8 Competition - Ensemble Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / "ensemble_analysis.png"
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    logger.info(f"Saved ensemble analysis plot to {plot_path}")
    plt.close()
    
    # Figure 2: Correlation heatmap of predictions
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create DataFrame with all predictions
    pred_df = pd.DataFrame()
    for method, result in results.items():
        if method != 'Stacking' and result['weights'] is not None:
            continue  # Skip for now
    
    logger.info("Visualizations created successfully")


def save_final_submissions(test_predictions, sample_sub_path, output_dir):
    """Save all ensemble submissions."""
    
    logger.info("\n" + "="*60)
    logger.info("SAVING SUBMISSIONS")
    logger.info("="*60)
    
    sample_sub = pd.read_csv(sample_sub_path)
    
    for method_name, predictions in test_predictions.items():
        submission = sample_sub.copy()
        submission['y'] = predictions
        
        filename = f"submission_ensemble_{method_name.lower()}.csv"
        submission.to_csv(output_dir / filename, index=False)
        logger.info(f"Saved {method_name} ensemble submission")
    
    # Also save the best performing method
    best_method = max(test_predictions.keys(), 
                     key=lambda k: results[k]['score'] if k in results else 0)
    
    submission = sample_sub.copy()
    submission['y'] = test_predictions[best_method]
    submission.to_csv(output_dir / "submission_best_ensemble.csv", index=False)
    logger.info(f"\nBest ensemble ({best_method}) saved as submission_best_ensemble.csv")


def main():
    """Main ensemble optimization pipeline."""
    
    logger.info("="*60)
    logger.info("PS5E8 COMPETITION - HILL CLIMBING ENSEMBLE")
    logger.info("="*60)
    
    # Initialize MLflow tracker
    mlflow_tracker = None
    if os.getenv("MLFLOW_TRACKING_URI"):
        mlflow_tracker = MLflowTracker(
            experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "PS5E8-Bank-Deposit"),
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
            tags={'stage': 'ensemble_optimization', 'competition': 'PS5E8'}
        )
        mlflow_tracker.start_run(run_name="PS5E8_Ensemble_HillClimbing")
    
    # Load training labels
    y_train = load_train_labels()
    
    # Initialize OOF manager
    oof_manager = OOFManager(output_dir=str(OOF_DIR))
    
    # List available OOF predictions
    logger.info("\nAvailable OOF Predictions:")
    oof_summary = oof_manager.list_oofs(sort_by='cv_score', ascending=False)
    
    if oof_summary.empty:
        logger.error("No OOF predictions found! Run 02_baseline_models.py first.")
        return
    
    print(oof_summary[['model_name', 'cv_score']])
    
    # Compare ensemble methods
    global results  # Make accessible for save_final_submissions
    results, test_predictions, model_df = compare_ensemble_methods(
        oof_manager, 
        y_train,
        min_cv_score=0.0,  # Use all models
        top_k=None  # Use all available models
    )
    
    if results is None:
        return
    
    # Create visualizations
    visualize_results(results, model_df, PLOTS_DIR)
    
    # Save submissions
    save_final_submissions(
        test_predictions,
        DATA_DIR / "sample_submission.csv",
        SUBMISSION_DIR
    )
    
    # Log results to MLflow
    if mlflow_tracker:
        for method_name, result in results.items():
            mlflow_tracker.log_metrics({
                f'ensemble_{method_name.lower()}_auc': result['score']
            })
        
        # Log best score
        best_method = max(results.items(), key=lambda x: x[1]['score'])
        mlflow_tracker.log_metrics({
            'best_ensemble_auc': best_method[1]['score'],
            'best_ensemble_method': methods.index(best_method[0])
        })
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("ENSEMBLE OPTIMIZATION SUMMARY")
    logger.info("="*60)
    
    # Sort methods by score
    sorted_methods = sorted(results.items(), key=lambda x: x[1]['score'], reverse=True)
    
    logger.info("\nFinal Rankings:")
    for rank, (method, result) in enumerate(sorted_methods, 1):
        score = result['score']
        improvement = (score - results['Equal']['score']) * 100
        
        if result['weights'] is not None:
            n_active = np.sum(result['weights'] > 0.01)
            logger.info(f"{rank}. {method:15s}: {score:.6f} (+{improvement:.3f}%) - {n_active} active models")
        else:
            logger.info(f"{rank}. {method:15s}: {score:.6f} (+{improvement:.3f}%)")
    
    # Best improvement
    best_method = sorted_methods[0]
    logger.info(f"\nBest Method: {best_method[0]}")
    logger.info(f"Best Score: {best_method[1]['score']:.6f}")
    
    baseline_score = results['Equal']['score']
    best_improvement = (best_method[1]['score'] - baseline_score) * 100
    logger.info(f"Improvement over baseline: {best_improvement:.3f}%")
    
    # Save summary
    summary_data = []
    for method, result in results.items():
        summary_data.append({
            'Method': method,
            'CV_AUC': result['score'],
            'Improvement_%': (result['score'] - baseline_score) * 100
        })
    
    summary_df = pd.DataFrame(summary_data).sort_values('CV_AUC', ascending=False)
    summary_df.to_csv(OUTPUT_DIR / "ensemble_summary.csv", index=False)
    logger.info(f"\nEnsemble summary saved to {OUTPUT_DIR}/ensemble_summary.csv")
    
    # End MLflow run
    if mlflow_tracker:
        mlflow_tracker.end_run()
        logger.info(f"\nMLflow tracking completed. View at {os.getenv('MLFLOW_TRACKING_URI')}")
    
    logger.info("\n" + "="*60)
    logger.info("ENSEMBLE OPTIMIZATION COMPLETE!")
    logger.info("="*60)
    logger.info(f"\nBest submission saved as: submission_best_ensemble.csv")
    logger.info(f"Upload to Kaggle: {SUBMISSION_DIR}/submission_best_ensemble.csv")


if __name__ == "__main__":
    main()