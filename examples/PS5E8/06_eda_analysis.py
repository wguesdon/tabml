"""
PS5E8 Competition - Exploratory Data Analysis
Comprehensive EDA using TabML's new visualization tools
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import tabml
sys.path.append(str(Path(__file__).parent.parent.parent))

from tabml import EDAAnalyzer

# Setup paths
DATA_DIR = Path("../../data/raw/PS5E8")
OUTPUT_DIR = Path("output/eda_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Competition settings
TARGET_COL = 'y'


def perform_eda():
    """Perform comprehensive EDA on PS5E8 competition data."""
    logger.info("="*60)
    logger.info("PS5E8 COMPETITION - EXPLORATORY DATA ANALYSIS")
    logger.info("="*60)
    
    # Load data
    logger.info("\nLoading competition data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    
    logger.info(f"Train shape: {train_df.shape}")
    logger.info(f"Test shape: {test_df.shape}")
    
    # Drop ID column for analysis
    train_df_analysis = train_df.drop('id', axis=1)
    test_df_analysis = test_df.drop('id', axis=1)
    
    # Initialize EDA Analyzer with Set2 palette for better categorical visibility
    logger.info("\nInitializing EDA Analyzer...")
    eda = EDAAnalyzer(figsize=(14, 8), style='seaborn-v0_8-darkgrid', palette='Set2', n_cols=3)
    
    # 1. Generate complete EDA report for training data
    logger.info("\n" + "="*40)
    logger.info("COMPREHENSIVE EDA REPORT - TRAINING DATA")
    logger.info("="*40)
    
    eda.generate_report(
        train_df_analysis,
        target=TARGET_COL,
        output_dir=str(OUTPUT_DIR / "train"),
        max_categories=20
    )
    
    # 2. Univariate analysis of features
    logger.info("\n" + "="*40)
    logger.info("UNIVARIATE ANALYSIS")
    logger.info("="*40)
    
    # Select subset of interesting features for detailed view
    feature_cols = [col for col in train_df_analysis.columns if col != TARGET_COL][:12]
    
    logger.info(f"\nAnalyzing {len(feature_cols)} features individually...")
    univariate_sample_dir = OUTPUT_DIR / "univariate_sample"
    eda.plot_univariate(
        train_df_analysis[feature_cols],
        max_categories=15,
        save_dir=str(univariate_sample_dir),
        individual_plots=True
    )
    
    # 3. Target distribution analysis
    logger.info("\n" + "="*40)
    logger.info("TARGET DISTRIBUTION ANALYSIS")
    logger.info("="*40)
    
    target_stats = train_df_analysis[TARGET_COL].describe()
    logger.info(f"\nTarget statistics:\n{target_stats}")
    
    # Check if target is binary or continuous
    n_unique = train_df_analysis[TARGET_COL].nunique()
    logger.info(f"Unique target values: {n_unique}")
    
    if n_unique == 2:
        logger.info("Target is binary classification")
        value_counts = train_df_analysis[TARGET_COL].value_counts()
        logger.info(f"Class distribution:\n{value_counts}")
        logger.info(f"Class balance: {value_counts.min() / value_counts.max():.2%}")
    else:
        logger.info("Target appears to be continuous or multi-class")
    
    # 4. Multivariate analysis - Features vs Target
    logger.info("\n" + "="*40)
    logger.info("MULTIVARIATE ANALYSIS - FEATURES VS TARGET")
    logger.info("="*40)
    
    # Select top features for detailed multivariate analysis
    top_features = feature_cols[:9]  # First 9 features for visibility
    
    multivariate_sample_dir = OUTPUT_DIR / "multivariate_sample"
    eda.plot_multivariate(
        train_df_analysis,
        target=TARGET_COL,
        features=top_features,
        max_categories=10,
        save_dir=str(multivariate_sample_dir),
        individual_plots=True
    )
    
    # 5. Correlation analysis
    logger.info("\n" + "="*40)
    logger.info("CORRELATION ANALYSIS")
    logger.info("="*40)
    
    eda.plot_correlation_matrix(
        train_df_analysis,
        method='pearson',
        save_path=str(OUTPUT_DIR / "correlation_matrix.png")
    )
    
    # Also try Spearman correlation for non-linear relationships
    eda.plot_correlation_matrix(
        train_df_analysis,
        method='spearman',
        save_path=str(OUTPUT_DIR / "correlation_spearman.png")
    )
    
    # 6. Missing data patterns
    logger.info("\n" + "="*40)
    logger.info("MISSING DATA ANALYSIS")
    logger.info("="*40)
    
    eda.plot_missing_data(
        train_df_analysis,
        save_path=str(OUTPUT_DIR / "missing_data.png")
    )
    
    # 7. Feature importance hints (correlation with target)
    logger.info("\n" + "="*40)
    logger.info("FEATURE IMPORTANCE HINTS")
    logger.info("="*40)
    
    # Calculate correlations with target (only for numerical columns)
    numerical_cols = train_df_analysis.select_dtypes(include=[np.number]).columns.tolist()
    if len(numerical_cols) > 1 and TARGET_COL in numerical_cols:
        correlations = train_df_analysis[numerical_cols].corr()[TARGET_COL].abs().sort_values(ascending=False)
        
        logger.info("\nTop 10 numerical features by correlation with target:")
        for feat, corr in correlations[1:11].items():  # Skip target itself
            logger.info(f"  {feat}: {corr:.4f}")
    else:
        logger.info("Not enough numerical columns for correlation analysis")
    
    # 8. Adversarial Validation - Check if train and test are from same distribution
    logger.info("\n" + "="*40)
    logger.info("ADVERSARIAL VALIDATION")
    logger.info("="*40)
    
    # Perform adversarial validation
    adv_results = eda.adversarial_validation(
        train_df=train_df,  # Use full DataFrame with ID
        test_df=test_df,
        target_col=TARGET_COL,
        n_folds=5,
        sample_size=50000,  # Sample for speed
        save_path=str(OUTPUT_DIR / "adversarial_validation.png")
    )
    
    logger.info(f"\nAdversarial Validation Summary:")
    logger.info(f"  AUC Score: {adv_results['auc_score']:.4f}")
    logger.info(f"  Interpretation: {adv_results['interpretation']}")
    
    if adv_results['feature_importance'] is not None and not adv_results['feature_importance'].empty:
        logger.info("\nFeatures causing distribution shift:")
        for _, row in adv_results['feature_importance'].head(5).iterrows():
            logger.info(f"  - {row['feature']}: {row['importance']:.4f}")
    
    # 9. Distribution comparison: Train vs Test (Statistical)
    logger.info("\n" + "="*40)
    logger.info("TRAIN VS TEST DISTRIBUTION COMPARISON")
    logger.info("="*40)
    
    # Check if distributions are similar (only for numerical columns)
    common_cols = [col for col in train_df_analysis.columns if col in test_df_analysis.columns]
    # Filter to only numerical columns
    numerical_common_cols = [col for col in common_cols 
                             if train_df_analysis[col].dtype in ['int64', 'float64', 'int32', 'float32']]
    
    logger.info(f"\nComparing {len(numerical_common_cols)} common numerical features...")
    
    distribution_diffs = {}
    for col in numerical_common_cols[:10]:  # Check first 10 numerical features
        try:
            train_mean = train_df_analysis[col].mean()
            test_mean = test_df_analysis[col].mean()
            train_std = train_df_analysis[col].std()
            test_std = test_df_analysis[col].std()
            
            mean_diff = abs(train_mean - test_mean) / (abs(train_mean) + 1e-10)
            std_diff = abs(train_std - test_std) / (abs(train_std) + 1e-10)
            
            distribution_diffs[col] = {
                'mean_diff_%': mean_diff * 100,
                'std_diff_%': std_diff * 100
            }
        except Exception as e:
            logger.debug(f"Could not compare distributions for {col}: {e}")
            continue
    
    # Report significant distribution differences
    logger.info("\nDistribution differences (Train vs Test):")
    for col, diffs in distribution_diffs.items():
        if diffs['mean_diff_%'] > 10 or diffs['std_diff_%'] > 10:
            logger.warning(f"  {col}: Mean diff={diffs['mean_diff_%']:.1f}%, Std diff={diffs['std_diff_%']:.1f}%")
        else:
            logger.info(f"  {col}: Mean diff={diffs['mean_diff_%']:.1f}%, Std diff={diffs['std_diff_%']:.1f}%")
    
    # 10. Feature engineering insights
    logger.info("\n" + "="*40)
    logger.info("FEATURE ENGINEERING INSIGHTS")
    logger.info("="*40)
    
    # Check for potential interaction features
    numerical_feature_cols = train_df_analysis.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET_COL in numerical_feature_cols:
        numerical_feature_cols.remove(TARGET_COL)
    
    if len(numerical_feature_cols) >= 2:
        # Sample interaction analysis
        feat1, feat2 = numerical_feature_cols[0], numerical_feature_cols[1]
        interaction = train_df_analysis[feat1] * train_df_analysis[feat2]
        interaction_corr = interaction.corr(train_df_analysis[TARGET_COL])
        
        # Get individual correlations
        feat1_corr = train_df_analysis[feat1].corr(train_df_analysis[TARGET_COL])
        feat2_corr = train_df_analysis[feat2].corr(train_df_analysis[TARGET_COL])
        
        logger.info(f"\nSample interaction feature: {feat1} * {feat2}")
        logger.info(f"  Correlation with target: {interaction_corr:.4f}")
        logger.info(f"  vs individual correlations: {feat1}={feat1_corr:.4f}, {feat2}={feat2_corr:.4f}")
        
        if abs(interaction_corr) > max(abs(feat1_corr), abs(feat2_corr)):
            logger.info("  ✓ Interaction appears stronger than individual features!")
    
    # 11. Summary and recommendations
    logger.info("\n" + "="*60)
    logger.info("EDA SUMMARY AND RECOMMENDATIONS")
    logger.info("="*60)
    
    logger.info("\n📊 Key Findings:")
    logger.info(f"  • Dataset size: {len(train_df_analysis):,} training samples")
    logger.info(f"  • Feature count: {len(feature_cols)} features")
    logger.info(f"  • Target type: {'Binary' if n_unique == 2 else 'Continuous/Multi-class'}")
    logger.info(f"  • Missing data: {train_df_analysis.isnull().any().sum()} columns with missing values")
    
    logger.info("\n💡 Recommendations:")
    logger.info("  1. Feature Engineering:")
    logger.info("     - Consider polynomial features for top correlated variables")
    logger.info("     - Create interaction features between highly correlated pairs")
    logger.info("     - Apply target encoding for categorical variables")
    
    logger.info("  2. Model Selection:")
    if n_unique == 2:
        logger.info("     - Binary classification: Use XGBoost, LightGBM, CatBoost")
        logger.info("     - Consider ensemble methods for better performance")
        logger.info("     - Use AUC-ROC as primary metric")
    else:
        logger.info("     - Regression/Multi-class: Use gradient boosting methods")
        logger.info("     - Consider neural networks for complex patterns")
    
    logger.info("  3. Data Preprocessing:")
    logger.info("     - Handle missing values based on feature importance")
    logger.info("     - Scale features for neural network models")
    logger.info("     - Consider outlier removal for sensitive models")
    
    logger.info(f"\n✅ All visualizations saved to {OUTPUT_DIR}/")
    logger.info("\nPlot organization:")
    logger.info(f"  - Univariate plots: {OUTPUT_DIR}/train/univariate/")
    logger.info(f"  - Multivariate plots: {OUTPUT_DIR}/train/multivariate/")
    logger.info(f"  - Correlation matrices: {OUTPUT_DIR}/")
    logger.info(f"  - Sample plots: {OUTPUT_DIR}/univariate_sample/ and multivariate_sample/")
    logger.info("\nEDA Complete! Use these insights to guide your modeling approach.")


def main():
    """Main execution function."""
    try:
        perform_eda()
    except Exception as e:
        logger.error(f"Error during EDA: {e}")
        raise


if __name__ == "__main__":
    main()