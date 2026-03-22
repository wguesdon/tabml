"""
PS5E8 Competition - Final Ensemble
Combines baseline and advanced models for optimal performance
Target: 0.977+ AUC
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
DATA_DIR = Path("../../data/raw/PS5E8")
OUTPUT_DIR = Path("output")
OOF_DIR_BASELINE = OUTPUT_DIR / "oof_predictions"
OOF_DIR_ADVANCED = OUTPUT_DIR / "oof_predictions_advanced"
SUBMISSION_DIR = OUTPUT_DIR / "submissions_final"
PLOTS_DIR = OUTPUT_DIR / "plots_final"

# Create directories
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Competition settings
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
TARGET_COL = 'y'


def load_train_labels():
    """Load training labels for evaluation."""
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    return train_df[TARGET_COL]


def combine_all_models(y_true):
    """Combine baseline and advanced models for final ensemble."""
    logger.info("="*60)
    logger.info("COMBINING ALL MODELS")
    logger.info("="*60)
    
    # Initialize OOF managers for both directories
    oof_manager_baseline = OOFManager(output_dir=str(OOF_DIR_BASELINE))
    oof_manager_advanced = OOFManager(output_dir=str(OOF_DIR_ADVANCED))
    
    # Load all OOF predictions from both directories
    all_oofs = {}
    
    # Load baseline OOF predictions (if any)
    logger.info("\nLoading baseline models...")
    baseline_oofs = oof_manager_baseline.load_all_oofs(
        experiment_name=None,  # Load all experiments
        min_cv_score=0.0
    )
    
    if baseline_oofs:
        logger.info(f"Found {len(baseline_oofs)} baseline models")
        all_oofs.update(baseline_oofs)
    
    # Load advanced OOF predictions from all experiment types
    logger.info("Loading advanced models...")
    
    # Load all available OOFs from advanced directory
    advanced_oofs = oof_manager_advanced.load_all_oofs(
        experiment_name=None,  # Load all experiments
        min_cv_score=0.0
    )
    
    if advanced_oofs:
        logger.info(f"Found {len(advanced_oofs)} advanced models")
        all_oofs.update(advanced_oofs)
    
    if not all_oofs:
        logger.error("No OOF predictions found! Run training scripts first.")
        return None, None, None, None
    
    logger.info(f"\nTotal models loaded: {len(all_oofs)}")
    
    # Display model scores
    model_info = []
    for filename, data in all_oofs.items():
        model_info.append({
            'name': data['model_name'],
            'cv_score': data.get('cv_score', 0),
            'experiment': data.get('experiment_name', 'unknown')
        })
    
    model_df = pd.DataFrame(model_info).sort_values('cv_score', ascending=False)
    
    logger.info("\nModel Scores:")
    logger.info("-" * 40)
    for idx, row in model_df.iterrows():
        logger.info(f"{row['name']:25s} ({row['experiment']:15s}): {row['cv_score']:.6f}")
    
    # Combine predictions - use the first manager's combine method
    if baseline_oofs:
        combined_oofs = oof_manager_baseline.combine_oofs(all_oofs, method='horizontal')
        combined_test = oof_manager_baseline.combine_oofs(all_oofs, method='horizontal', use_test=True)
    else:
        combined_oofs = oof_manager_advanced.combine_oofs(all_oofs, method='horizontal')
        combined_test = oof_manager_advanced.combine_oofs(all_oofs, method='horizontal', use_test=True)
    
    return combined_oofs, combined_test, all_oofs, model_df


def optimize_final_ensemble(combined_oofs, y_true, model_df):
    """Optimize ensemble weights using multiple strategies."""
    logger.info("\n" + "="*60)
    logger.info("OPTIMIZING FINAL ENSEMBLE")
    logger.info("="*60)
    
    ensemble = OOFEnsemble(task_type='classification')
    results = {}
    weights_dict = {}
    
    # 1. Equal weights baseline
    logger.info("\n1. Equal Weights Baseline...")
    equal_weights = np.ones(combined_oofs.shape[1]) / combined_oofs.shape[1]
    equal_pred = np.average(combined_oofs.values, weights=equal_weights, axis=1)
    equal_score = roc_auc_score(y_true, equal_pred)
    results['Equal'] = equal_score
    weights_dict['Equal'] = equal_weights
    logger.info(f"   Score: {equal_score:.6f}")
    
    # 2. Weighted by CV scores
    logger.info("\n2. CV Score Weighted...")
    cv_scores = model_df['cv_score'].values[:combined_oofs.shape[1]]
    # Apply power transformation to emphasize better models
    cv_weights = np.power(cv_scores, 3)  # Cube the scores for more emphasis
    cv_weights = cv_weights / cv_weights.sum()
    cv_pred = np.average(combined_oofs.values, weights=cv_weights, axis=1)
    cv_score = roc_auc_score(y_true, cv_pred)
    results['CV_Weighted'] = cv_score
    weights_dict['CV_Weighted'] = cv_weights
    logger.info(f"   Score: {cv_score:.6f}")
    
    # 3. Top models only
    logger.info("\n3. Top Models Only...")
    top_k = min(10, combined_oofs.shape[1])
    top_weights = np.zeros(combined_oofs.shape[1])
    top_weights[:top_k] = 1.0 / top_k  # Equal weight among top models
    top_pred = np.average(combined_oofs.values, weights=top_weights, axis=1)
    top_score = roc_auc_score(y_true, top_pred)
    results['Top_Models'] = top_score
    weights_dict['Top_Models'] = top_weights
    logger.info(f"   Score: {top_score:.6f} (using top {top_k} models)")
    
    # 4. SciPy optimization
    logger.info("\n4. SciPy Optimization...")
    scipy_weights = ensemble.optimize_weights(combined_oofs, y_true, method='scipy')
    scipy_pred = np.average(combined_oofs.values, weights=scipy_weights, axis=1)
    scipy_score = roc_auc_score(y_true, scipy_pred)
    results['SciPy'] = scipy_score
    weights_dict['SciPy'] = scipy_weights
    logger.info(f"   Score: {scipy_score:.6f}")
    
    # 5. Optuna optimization
    logger.info("\n5. Optuna Optimization (300 trials)...")
    optuna_weights = ensemble.optimize_weights(
        combined_oofs, y_true, 
        method='optuna', 
        n_trials=300
    )
    optuna_pred = np.average(combined_oofs.values, weights=optuna_weights, axis=1)
    optuna_score = roc_auc_score(y_true, optuna_pred)
    results['Optuna'] = optuna_score
    weights_dict['Optuna'] = optuna_weights
    logger.info(f"   Score: {optuna_score:.6f}")
    
    # 6. Hill Climbing (extensive search)
    logger.info("\n6. Hill Climbing Optimization (10000 iterations)...")
    hill_weights = ensemble.optimize_weights(
        combined_oofs, y_true,
        method='hill_climbing',
        n_iterations=10000,
        patience=1000
    )
    hill_pred = np.average(combined_oofs.values, weights=hill_weights, axis=1)
    hill_score = roc_auc_score(y_true, hill_pred)
    results['Hill_Climbing'] = hill_score
    weights_dict['Hill_Climbing'] = hill_weights
    logger.info(f"   Score: {hill_score:.6f}")
    
    # 7. Greedy Forward Selection
    logger.info("\n7. Greedy Forward Selection...")
    greedy_weights = ensemble.optimize_weights(
        combined_oofs, y_true,
        method='greedy_forward'
    )
    greedy_pred = np.average(combined_oofs.values, weights=greedy_weights, axis=1)
    greedy_score = roc_auc_score(y_true, greedy_pred)
    n_selected = np.sum(greedy_weights > 0)
    results['Greedy_Forward'] = greedy_score
    weights_dict['Greedy_Forward'] = greedy_weights
    logger.info(f"   Score: {greedy_score:.6f} ({n_selected} models selected)")
    
    # 8. Stacking with meta-learner
    logger.info("\n8. Stacking with Meta-Learner...")
    ensemble_stack = OOFEnsemble(task_type='classification')
    # Use None to let it use the default meta-model (LogisticRegression for classification)
    ensemble_stack.fit_stacking(combined_oofs, y_true, meta_model=None)
    stack_pred = ensemble_stack.predict_stacking(combined_oofs)
    stack_score = roc_auc_score(y_true, stack_pred)
    results['Stacking'] = stack_score
    weights_dict['Stacking'] = None  # Stacking doesn't use simple weights
    logger.info(f"   Score: {stack_score:.6f}")
    
    # 9. Ensemble of ensembles (blend top methods)
    logger.info("\n9. Ensemble of Ensembles...")
    # Get predictions from top 3 methods
    top_methods = sorted(results.items(), key=lambda x: x[1], reverse=True)[:3]
    ensemble_preds = []
    for method_name, _ in top_methods:
        if method_name == 'Stacking':
            ensemble_preds.append(stack_pred)
        else:
            weights = weights_dict[method_name]
            pred = np.average(combined_oofs.values, weights=weights, axis=1)
            ensemble_preds.append(pred)
    
    # Average the top ensemble predictions
    meta_ensemble_pred = np.mean(ensemble_preds, axis=0)
    meta_ensemble_score = roc_auc_score(y_true, meta_ensemble_pred)
    results['Meta_Ensemble'] = meta_ensemble_score
    weights_dict['Meta_Ensemble'] = None
    logger.info(f"   Score: {meta_ensemble_score:.6f} (blend of top 3 methods)")
    
    return results, weights_dict, ensemble_stack


def visualize_final_results(results, weights_dict, model_df, output_dir):
    """Create comprehensive visualizations of final ensemble results."""
    logger.info("\n" + "="*60)
    logger.info("CREATING VISUALIZATIONS")
    logger.info("="*60)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Method comparison
    ax1 = plt.subplot(2, 3, 1)
    methods = list(results.keys())
    scores = list(results.values())
    
    # Color code by performance
    colors = []
    for score in scores:
        if score >= 0.977:
            colors.append('gold')
        elif score >= 0.975:
            colors.append('lightgreen')
        elif score >= 0.97:
            colors.append('skyblue')
        else:
            colors.append('lightcoral')
    
    bars = ax1.bar(methods, scores, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_title('Ensemble Method Performance', fontsize=14, fontweight='bold')
    ax1.set_ylabel('ROC AUC Score', fontsize=12)
    ax1.set_ylim([min(scores) - 0.002, max(scores) + 0.002])
    ax1.axhline(y=0.977, color='red', linestyle='--', alpha=0.5, label='Target: 0.977')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add value labels
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                f'{score:.5f}', ha='center', va='bottom', fontsize=9)
    
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 2: Weight distribution for best method
    ax2 = plt.subplot(2, 3, 2)
    best_method = max(results, key=results.get)
    if best_method in weights_dict and weights_dict[best_method] is not None:
        weights = weights_dict[best_method]
        model_names = model_df['name'].values[:len(weights)]
        
        # Sort by weight
        sorted_idx = np.argsort(weights)[::-1][:15]  # Top 15
        sorted_weights = weights[sorted_idx]
        sorted_names = [model_names[i] for i in sorted_idx]
        
        bars = ax2.barh(range(len(sorted_weights)), sorted_weights, 
                       color=['green' if w > 0.05 else 'lightgray' for w in sorted_weights])
        ax2.set_title(f'{best_method} - Top Model Weights', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Weight', fontsize=12)
        ax2.set_yticks(range(len(sorted_names)))
        ax2.set_yticklabels([n[:20] for n in sorted_names], fontsize=9)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, weight) in enumerate(zip(bars, sorted_weights)):
            if weight > 0.01:
                ax2.text(weight + 0.005, bar.get_y() + bar.get_height()/2,
                        f'{weight:.3f}', ha='left', va='center', fontsize=8)
    
    # Plot 3: Model importance across all methods
    ax3 = plt.subplot(2, 3, 3)
    n_models = min(len(model_df), 20)
    model_importance = np.zeros(n_models)
    
    # Calculate importance across methods
    for method, weights in weights_dict.items():
        if weights is not None and method != 'Meta_Ensemble':
            method_score = results[method]
            model_importance[:len(weights)] += weights[:n_models] * method_score
    
    # Normalize
    if model_importance.sum() > 0:
        model_importance = model_importance / model_importance.sum()
    
    # Sort and plot
    sorted_idx = np.argsort(model_importance)[::-1][:10]
    sorted_importance = model_importance[sorted_idx]
    sorted_names = model_df['name'].values[sorted_idx]
    
    bars = ax3.bar(range(len(sorted_importance)), sorted_importance,
                   color='coral', alpha=0.8, edgecolor='black')
    ax3.set_title('Overall Model Importance', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Model', fontsize=12)
    ax3.set_ylabel('Importance Score', fontsize=12)
    ax3.set_xticks(range(len(sorted_names)))
    ax3.set_xticklabels([n[:12] for n in sorted_names], rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Score progression
    ax4 = plt.subplot(2, 3, 4)
    baseline_models = model_df[model_df['experiment'].str.contains('baseline', na=False)]
    advanced_models = model_df[model_df['experiment'].str.contains('advanced', na=False)]
    
    if not baseline_models.empty and not advanced_models.empty:
        categories = ['Baseline\nModels', 'Advanced\nModels', 'Final\nEnsemble']
        avg_scores = [
            baseline_models['cv_score'].mean(),
            advanced_models['cv_score'].mean(),
            max(results.values())
        ]
        
        bars = ax4.bar(categories, avg_scores, 
                      color=['lightblue', 'lightgreen', 'gold'],
                      alpha=0.8, edgecolor='black', width=0.5)
        ax4.set_title('Performance Progression', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Average ROC AUC', fontsize=12)
        ax4.set_ylim([min(avg_scores) - 0.005, max(avg_scores) + 0.005])
        ax4.grid(True, alpha=0.3)
        
        # Add improvement annotations
        for i, (bar, score) in enumerate(zip(bars, avg_scores)):
            ax4.text(bar.get_x() + bar.get_width()/2, score + 0.001,
                    f'{score:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            if i > 0:
                improvement = (score - avg_scores[i-1]) * 100
                ax4.text(bar.get_x() + bar.get_width()/2, score - 0.003,
                        f'+{improvement:.2f}%', ha='center', va='top', fontsize=9, color='green')
    
    # Plot 5: Distribution of model scores
    ax5 = plt.subplot(2, 3, 5)
    all_scores = model_df['cv_score'].values
    
    ax5.hist(all_scores, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax5.axvline(x=np.mean(all_scores), color='red', linestyle='--', label=f'Mean: {np.mean(all_scores):.4f}')
    ax5.axvline(x=max(results.values()), color='gold', linestyle='-', linewidth=2, 
                label=f'Best Ensemble: {max(results.values()):.4f}')
    ax5.set_title('Distribution of Model Scores', fontsize=14, fontweight='bold')
    ax5.set_xlabel('CV AUC Score', fontsize=12)
    ax5.set_ylabel('Count', fontsize=12)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Top models table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('tight')
    ax6.axis('off')
    
    # Create table data
    top_10 = model_df.head(10)
    table_data = []
    for idx, row in top_10.iterrows():
        table_data.append([
            row['name'][:25],
            f"{row['cv_score']:.5f}",
            row['experiment'][:15]
        ])
    
    table = ax6.table(cellText=table_data,
                     colLabels=['Model', 'CV AUC', 'Experiment'],
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.5, 0.25, 0.25])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Color code by score
    for i in range(1, len(table_data) + 1):
        score = float(table_data[i-1][1])
        if score >= 0.975:
            color = 'lightgreen'
        elif score >= 0.97:
            color = 'lightyellow'
        else:
            color = 'white'
        for j in range(3):
            table[(i, j)].set_facecolor(color)
    
    ax6.set_title('Top 10 Individual Models', fontsize=14, fontweight='bold', pad=20)
    
    # Overall title
    plt.suptitle(f'PS5E8 Competition - Final Ensemble Analysis\nBest Score: {max(results.values()):.5f}', 
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / "final_ensemble_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization to {plot_path}")
    plt.close()


def save_final_submissions(results, weights_dict, combined_test, ensemble_stack, sample_sub_path, output_dir):
    """Save all final ensemble submissions."""
    logger.info("\n" + "="*60)
    logger.info("SAVING FINAL SUBMISSIONS")
    logger.info("="*60)
    
    sample_sub = pd.read_csv(sample_sub_path)
    
    # Save predictions for each method
    for method_name, score in results.items():
        if method_name == 'Stacking':
            # Use stacking predictions
            if ensemble_stack is not None:
                predictions = ensemble_stack.predict_stacking(combined_test)
            else:
                continue
        elif method_name == 'Meta_Ensemble':
            # Recreate meta ensemble for test data
            top_methods = sorted(results.items(), key=lambda x: x[1], reverse=True)[:3]
            test_preds = []
            for top_method, _ in top_methods:
                if top_method == 'Stacking' and ensemble_stack is not None:
                    test_preds.append(ensemble_stack.predict_stacking(combined_test))
                elif top_method in weights_dict and weights_dict[top_method] is not None:
                    weights = weights_dict[top_method]
                    pred = np.average(combined_test.values, weights=weights, axis=1)
                    test_preds.append(pred)
            if test_preds:
                predictions = np.mean(test_preds, axis=0)
            else:
                continue
        else:
            # Use weighted average
            if weights_dict[method_name] is not None:
                weights = weights_dict[method_name]
                predictions = np.average(combined_test.values, weights=weights, axis=1)
            else:
                continue
        
        # Save submission
        submission = sample_sub.copy()
        submission['y'] = predictions
        filename = f"submission_final_{method_name.lower()}.csv"
        submission.to_csv(output_dir / filename, index=False)
        logger.info(f"Saved {method_name} submission (CV: {score:.5f})")
    
    # Save best submission
    best_method = max(results, key=results.get)
    best_score = results[best_method]
    
    logger.info(f"\nBest method: {best_method} with CV score: {best_score:.5f}")
    logger.info(f"Best submission saved as: submission_final_{best_method.lower()}.csv")
    
    # Also save as "best" for easy identification
    best_sub_path = output_dir / f"submission_final_{best_method.lower()}.csv"
    if best_sub_path.exists():
        submission = pd.read_csv(best_sub_path)
        submission.to_csv(output_dir / "submission_BEST.csv", index=False)
        logger.info(f"Also saved as: submission_BEST.csv")


def main():
    """Main pipeline for final ensemble."""
    
    logger.info("="*60)
    logger.info("PS5E8 COMPETITION - FINAL ENSEMBLE")
    logger.info("Target: 0.977+ AUC")
    logger.info("="*60)
    
    # Initialize MLflow tracker
    mlflow_tracker = None
    if os.getenv("MLFLOW_TRACKING_URI"):
        mlflow_tracker = MLflowTracker(
            experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "PS5E8-Final"),
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
            tags={'stage': 'final_ensemble', 'competition': 'PS5E8'}
        )
        mlflow_tracker.start_run(run_name="PS5E8_Final_Ensemble")
    
    # Load training labels
    y_train = load_train_labels()
    
    # Combine all models
    combined_oofs, combined_test, all_oofs, model_df = combine_all_models(y_train)
    
    if combined_oofs is None:
        logger.error("Failed to load models. Exiting.")
        return
    
    # Optimize ensemble
    results, weights_dict, ensemble_stack = optimize_final_ensemble(
        combined_oofs, y_train, model_df
    )
    
    # Create visualizations
    visualize_final_results(results, weights_dict, model_df, PLOTS_DIR)
    
    # Save submissions
    save_final_submissions(
        results, weights_dict, combined_test, ensemble_stack,
        DATA_DIR / "sample_submission.csv", SUBMISSION_DIR
    )
    
    # Log to MLflow
    if mlflow_tracker:
        for method_name, score in results.items():
            mlflow_tracker.log_metrics({f'final_{method_name.lower()}_auc': score})
        
        best_score = max(results.values())
        mlflow_tracker.log_metrics({
            'best_final_auc': best_score,
            'target_achieved': best_score >= 0.977
        })
        
        mlflow_tracker.end_run()
        logger.info(f"\nMLflow tracking completed. View at {os.getenv('MLFLOW_TRACKING_URI')}")
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("FINAL SUMMARY")
    logger.info("="*60)
    
    # Sort methods by score
    sorted_methods = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    logger.info("\nFinal Ensemble Rankings:")
    logger.info("-" * 40)
    for rank, (method, score) in enumerate(sorted_methods, 1):
        status = "✓ TARGET MET" if score >= 0.977 else ""
        logger.info(f"{rank:2d}. {method:15s}: {score:.5f} {status}")
    
    # Best result
    best_method, best_score = sorted_methods[0]
    logger.info(f"\nBest Method: {best_method}")
    logger.info(f"Best Score: {best_score:.5f}")
    
    if best_score >= 0.977:
        logger.success(f"🎯 TARGET ACHIEVED! Score: {best_score:.5f} >= 0.977")
    else:
        gap = 0.977 - best_score
        logger.warning(f"Target not met. Gap: {gap:.5f}")
    
    # Improvement summary
    baseline_avg = model_df[model_df['experiment'].str.contains('baseline', na=False)]['cv_score'].mean() if not model_df.empty else 0
    if baseline_avg > 0:
        total_improvement = (best_score - baseline_avg) * 100
        logger.info(f"\nTotal improvement from baseline: {total_improvement:.3f}%")
    
    logger.info("\n" + "="*60)
    logger.info("FINAL ENSEMBLE COMPLETE!")
    logger.info("="*60)
    logger.info(f"\nBest submission: {SUBMISSION_DIR}/submission_BEST.csv")
    logger.info("Ready for Kaggle submission!")


if __name__ == "__main__":
    main()