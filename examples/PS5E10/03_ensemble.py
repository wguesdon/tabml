"""
PS5E10 Competition - Ensemble Optimization
Load OOF predictions and optimize ensemble weights using hill climbing

Methods compared:
- Simple average
- Weighted average (optimized)
- Hill climbing optimization
- Greedy forward selection
- Stacking
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import tabml
sys.path.append(str(Path(__file__).parent.parent.parent))

from tabml import OOFManager, OOFEnsemble
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns

# Setup paths
DATA_DIR = Path("../../data/raw/PS5E10")
OUTPUT_DIR = Path("output")
OOF_DIR = OUTPUT_DIR / "oof_predictions"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"
ENSEMBLE_DIR = OUTPUT_DIR / "ensemble_analysis"
ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)

# Competition settings
TARGET_COL = 'accident_risk'
ID_COL = 'id'
RANDOM_SEED = 42


def plot_ensemble_comparison(results, save_path):
    """Plot comparison of different ensemble methods."""
    methods = list(results.keys())
    scores = [results[m]['score'] for m in methods]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(methods, scores, color='steelblue', alpha=0.7)

    # Color the best bar
    best_idx = np.argmin(scores)
    bars[best_idx].set_color('green')
    bars[best_idx].set_alpha(0.9)

    plt.ylabel('RMSE (lower is better)', fontsize=12)
    plt.title('Ensemble Method Comparison', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (method, score) in enumerate(zip(methods, scores)):
        plt.text(i, score + 0.0001, f'{score:.6f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Ensemble comparison plot saved to: {save_path}")


def plot_weight_distribution(weights, model_names, save_path):
    """Plot distribution of optimized weights."""
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(weights)), weights, color='steelblue', alpha=0.7)

    # Color non-zero weights
    for i, w in enumerate(weights):
        if w > 0.01:
            bars[i].set_color('green')

    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
    plt.ylabel('Weight', fontsize=12)
    plt.title('Optimized Ensemble Weights', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, w in enumerate(weights):
        if w > 0.001:
            plt.text(i, w + 0.01, f'{w:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Weight distribution plot saved to: {save_path}")


def main():
    """Main ensemble optimization pipeline."""
    logger.info("="*60)
    logger.info("PS5E10 - ENSEMBLE OPTIMIZATION")
    logger.info("="*60)

    # Load test data for submission
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    test_ids = test_df[ID_COL]

    # Initialize OOF Manager
    oof_manager = OOFManager(str(OOF_DIR))

    # List all saved OOF predictions
    logger.info("\n" + "="*60)
    logger.info("AVAILABLE OOF PREDICTIONS")
    logger.info("="*60)

    oof_summary = oof_manager.list_oofs(sort_by='cv_score', ascending=True)
    logger.info(f"\n{oof_summary}")

    # Load all OOF predictions
    logger.info("\n" + "="*60)
    logger.info("LOADING OOF PREDICTIONS")
    logger.info("="*60)

    all_oofs = oof_manager.load_all_oofs()

    if not all_oofs:
        logger.error("No OOF predictions found! Please run 02_baseline_models.py first.")
        return

    logger.info(f"✓ Loaded {len(all_oofs)} OOF prediction sets")

    # Extract OOF predictions and test predictions
    oof_predictions_dict = {}
    test_predictions_dict = {}
    model_names = []
    cv_scores = []

    for filename, data in all_oofs.items():
        model_name = data['model_name']
        model_names.append(model_name)
        cv_scores.append(data['cv_score'])

        # Get OOF predictions
        if isinstance(data['predictions'], pd.DataFrame):
            oof_predictions_dict[model_name] = data['predictions'].iloc[:, 0].values
        else:
            oof_predictions_dict[model_name] = data['predictions']

        # Get test predictions
        if data['test_predictions'] is not None:
            if isinstance(data['test_predictions'], pd.DataFrame):
                test_predictions_dict[model_name] = data['test_predictions'].iloc[:, 0].values
            else:
                test_predictions_dict[model_name] = data['test_predictions']

    # Combine OOF predictions into DataFrame
    oof_combined = pd.DataFrame(oof_predictions_dict)
    logger.info(f"\nOOF predictions shape: {oof_combined.shape}")

    # Load training target
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    y_train = train_df[TARGET_COL]

    logger.info(f"Target shape: {y_train.shape}")

    # ========================================================================
    # ENSEMBLE METHOD 1: Simple Average
    # ========================================================================
    logger.info("\n" + "="*60)
    logger.info("METHOD 1: SIMPLE AVERAGE")
    logger.info("="*60)

    avg_oof = oof_combined.mean(axis=1)
    avg_rmse = np.sqrt(mean_squared_error(y_train, avg_oof))
    avg_mae = mean_absolute_error(y_train, avg_oof)
    avg_r2 = r2_score(y_train, avg_oof)

    logger.info(f"RMSE: {avg_rmse:.6f}")
    logger.info(f"MAE:  {avg_mae:.6f}")
    logger.info(f"R²:   {avg_r2:.6f}")

    # ========================================================================
    # ENSEMBLE METHOD 2: Weighted by CV Score
    # ========================================================================
    logger.info("\n" + "="*60)
    logger.info("METHOD 2: WEIGHTED BY CV SCORE")
    logger.info("="*60)

    # Inverse CV score weighting (lower RMSE = higher weight)
    inverse_scores = 1 / np.array(cv_scores)
    cv_weights = inverse_scores / inverse_scores.sum()

    weighted_oof = np.average(oof_combined.values, weights=cv_weights, axis=1)
    weighted_rmse = np.sqrt(mean_squared_error(y_train, weighted_oof))
    weighted_mae = mean_absolute_error(y_train, weighted_oof)
    weighted_r2 = r2_score(y_train, weighted_oof)

    logger.info(f"RMSE: {weighted_rmse:.6f}")
    logger.info(f"MAE:  {weighted_mae:.6f}")
    logger.info(f"R²:   {weighted_r2:.6f}")
    logger.info(f"Weights: {cv_weights}")

    # ========================================================================
    # ENSEMBLE METHOD 3: Hill Climbing Optimization
    # ========================================================================
    logger.info("\n" + "="*60)
    logger.info("METHOD 3: HILL CLIMBING OPTIMIZATION")
    logger.info("="*60)

    ensemble = OOFEnsemble(task_type='regression', metric='rmse', random_state=RANDOM_SEED)

    hill_climbing_weights = ensemble.optimize_weights(
        oof_combined,
        y_train,
        method='hill_climbing',
        n_iterations=2000,
        patience=200
    )

    hill_oof = np.average(oof_combined.values, weights=hill_climbing_weights, axis=1)
    hill_rmse = np.sqrt(mean_squared_error(y_train, hill_oof))
    hill_mae = mean_absolute_error(y_train, hill_oof)
    hill_r2 = r2_score(y_train, hill_oof)

    logger.info(f"\nHill Climbing Results:")
    logger.info(f"RMSE: {hill_rmse:.6f}")
    logger.info(f"MAE:  {hill_mae:.6f}")
    logger.info(f"R²:   {hill_r2:.6f}")

    # ========================================================================
    # ENSEMBLE METHOD 4: Greedy Forward Selection
    # ========================================================================
    logger.info("\n" + "="*60)
    logger.info("METHOD 4: GREEDY FORWARD SELECTION")
    logger.info("="*60)

    greedy_weights = ensemble.optimize_weights(
        oof_combined,
        y_train,
        method='greedy_forward'
    )

    greedy_oof = np.average(oof_combined.values, weights=greedy_weights, axis=1)
    greedy_rmse = np.sqrt(mean_squared_error(y_train, greedy_oof))
    greedy_mae = mean_absolute_error(y_train, greedy_oof)
    greedy_r2 = r2_score(y_train, greedy_oof)

    logger.info(f"\nGreedy Forward Results:")
    logger.info(f"RMSE: {greedy_rmse:.6f}")
    logger.info(f"MAE:  {greedy_mae:.6f}")
    logger.info(f"R²:   {greedy_r2:.6f}")

    # ========================================================================
    # ENSEMBLE METHOD 5: Stacking with Ridge
    # ========================================================================
    logger.info("\n" + "="*60)
    logger.info("METHOD 5: STACKING (Ridge Meta-Learner)")
    logger.info("="*60)

    meta_model = Ridge(alpha=1.0, random_state=RANDOM_SEED)
    ensemble.fit_stacking(oof_combined, y_train, meta_model=meta_model)

    stacking_oof = ensemble.meta_model.predict(oof_combined)
    stacking_oof = np.clip(stacking_oof, 0, 1)
    stacking_rmse = np.sqrt(mean_squared_error(y_train, stacking_oof))
    stacking_mae = mean_absolute_error(y_train, stacking_oof)
    stacking_r2 = r2_score(y_train, stacking_oof)

    logger.info(f"RMSE: {stacking_rmse:.6f}")
    logger.info(f"MAE:  {stacking_mae:.6f}")
    logger.info(f"R²:   {stacking_r2:.6f}")

    # ========================================================================
    # ENSEMBLE METHOD 6: Rank Averaging
    # ========================================================================
    logger.info("\n" + "="*60)
    logger.info("METHOD 6: RANK AVERAGING")
    logger.info("="*60)

    rank_oof = ensemble.rank_average([oof_combined[col].values for col in oof_combined.columns])
    rank_rmse = np.sqrt(mean_squared_error(y_train, rank_oof))
    rank_mae = mean_absolute_error(y_train, rank_oof)
    rank_r2 = r2_score(y_train, rank_oof)

    logger.info(f"RMSE: {rank_rmse:.6f}")
    logger.info(f"MAE:  {rank_mae:.6f}")
    logger.info(f"R²:   {rank_r2:.6f}")

    # ========================================================================
    # COMPARISON AND VISUALIZATION
    # ========================================================================
    logger.info("\n" + "="*60)
    logger.info("ENSEMBLE METHOD COMPARISON")
    logger.info("="*60)

    results = {
        'Simple Average': {'score': avg_rmse, 'mae': avg_mae, 'r2': avg_r2},
        'CV Weighted': {'score': weighted_rmse, 'mae': weighted_mae, 'r2': weighted_r2},
        'Hill Climbing': {'score': hill_rmse, 'mae': hill_mae, 'r2': hill_r2},
        'Greedy Forward': {'score': greedy_rmse, 'mae': greedy_mae, 'r2': greedy_r2},
        'Stacking': {'score': stacking_rmse, 'mae': stacking_mae, 'r2': stacking_r2},
        'Rank Average': {'score': rank_rmse, 'mae': rank_mae, 'r2': rank_r2}
    }

    # Find best method
    best_method = min(results.items(), key=lambda x: x[1]['score'])
    logger.info(f"\n🏆 BEST METHOD: {best_method[0]}")
    logger.info(f"   RMSE: {best_method[1]['score']:.6f}")
    logger.info(f"   MAE:  {best_method[1]['mae']:.6f}")
    logger.info(f"   R²:   {best_method[1]['r2']:.6f}")

    # Create comparison table
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.sort_values('score')
    logger.info(f"\n{comparison_df}")

    # Save comparison
    comparison_df.to_csv(ENSEMBLE_DIR / "ensemble_comparison.csv")

    # Plot comparisons
    plot_ensemble_comparison(results, ENSEMBLE_DIR / "ensemble_comparison.png")

    # Plot best weights (Hill Climbing)
    plot_weight_distribution(
        hill_climbing_weights,
        model_names,
        ENSEMBLE_DIR / "hill_climbing_weights.png"
    )

    # ========================================================================
    # GENERATE FINAL SUBMISSIONS
    # ========================================================================
    logger.info("\n" + "="*60)
    logger.info("GENERATING FINAL SUBMISSIONS")
    logger.info("="*60)

    # Combine test predictions
    test_combined = pd.DataFrame(test_predictions_dict)

    # Generate submission for each method
    submissions = {}

    # 1. Simple Average
    submissions['avg'] = test_combined.mean(axis=1)

    # 2. CV Weighted
    submissions['weighted'] = np.average(test_combined.values, weights=cv_weights, axis=1)

    # 3. Hill Climbing
    submissions['hill_climbing'] = np.average(test_combined.values, weights=hill_climbing_weights, axis=1)

    # 4. Greedy Forward
    submissions['greedy'] = np.average(test_combined.values, weights=greedy_weights, axis=1)

    # 5. Stacking
    submissions['stacking'] = ensemble.meta_model.predict(test_combined)

    # 6. Rank Average
    submissions['rank'] = ensemble.rank_average([test_combined[col].values for col in test_combined.columns])

    # Clip all predictions to valid range
    for key in submissions:
        submissions[key] = np.clip(submissions[key], 0, 1)

    # Save all submissions
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for method, preds in submissions.items():
        submission = pd.DataFrame({
            ID_COL: test_ids,
            TARGET_COL: preds
        })

        submission_path = SUBMISSION_DIR / f"submission_{method}_{timestamp}.csv"
        submission.to_csv(submission_path, index=False)
        logger.info(f"✓ {method:20s} -> {submission_path.name}")

    # Save the best submission with a special name
    best_method_key = best_method[0].lower().replace(' ', '_')
    best_submission = pd.DataFrame({
        ID_COL: test_ids,
        TARGET_COL: submissions.get(best_method_key, submissions['hill_climbing'])
    })

    best_path = SUBMISSION_DIR / f"submission_BEST_{timestamp}.csv"
    best_submission.to_csv(best_path, index=False)
    logger.info(f"\n🏆 BEST submission -> {best_path.name}")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    logger.info("\n" + "="*60)
    logger.info("ENSEMBLE OPTIMIZATION COMPLETE!")
    logger.info("="*60)

    logger.info(f"\n📊 Summary:")
    logger.info(f"  • Models ensembled: {len(model_names)}")
    logger.info(f"  • Methods tested: {len(results)}")
    logger.info(f"  • Best method: {best_method[0]}")
    logger.info(f"  • Best RMSE: {best_method[1]['score']:.6f}")
    logger.info(f"  • Improvement over average: {(avg_rmse - best_method[1]['score']) / avg_rmse * 100:.2f}%")

    logger.info(f"\n📁 Output files:")
    logger.info(f"  • Submissions: {SUBMISSION_DIR}")
    logger.info(f"  • Analysis: {ENSEMBLE_DIR}")
    logger.info(f"  • Best submission: {best_path.name}")

    logger.info(f"\n🚀 Next steps:")
    logger.info(f"  1. Submit {best_path.name} to Kaggle")
    logger.info(f"  2. Review ensemble analysis in {ENSEMBLE_DIR}")
    logger.info(f"  3. Consider additional feature engineering if needed")

    return best_path


if __name__ == "__main__":
    main()
