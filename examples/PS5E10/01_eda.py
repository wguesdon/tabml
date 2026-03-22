"""
PS5E10 Competition - Exploratory Data Analysis
Comprehensive EDA for Road Accident Risk Prediction using TabML's visualization tools
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
DATA_DIR = Path("../../data/raw/PS5E10")
OUTPUT_DIR = Path("output/eda_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Competition settings
TARGET_COL = 'accident_risk'
ID_COL = 'id'


def perform_eda():
    """Perform comprehensive EDA on PS5E10 road accident risk data."""
    logger.info("="*60)
    logger.info("PS5E10 COMPETITION - ROAD ACCIDENT RISK PREDICTION EDA")
    logger.info("="*60)

    # Load data
    logger.info("\nLoading competition data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    logger.info(f"Train shape: {train_df.shape}")
    logger.info(f"Test shape: {test_df.shape}")

    # Drop ID column for analysis
    train_df_analysis = train_df.drop(ID_COL, axis=1)
    test_df_analysis = test_df.drop(ID_COL, axis=1)

    # Initialize EDA Analyzer
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

    # 2. Target (accident_risk) distribution analysis
    logger.info("\n" + "="*40)
    logger.info("TARGET (ACCIDENT RISK) DISTRIBUTION ANALYSIS")
    logger.info("="*40)

    target_stats = train_df_analysis[TARGET_COL].describe()
    logger.info(f"\nAccident Risk statistics:\n{target_stats}")

    # Check for outliers in accident_risk
    Q1 = train_df_analysis[TARGET_COL].quantile(0.25)
    Q3 = train_df_analysis[TARGET_COL].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = train_df_analysis[(train_df_analysis[TARGET_COL] < lower_bound) |
                                 (train_df_analysis[TARGET_COL] > upper_bound)]

    logger.info(f"\nAccident Risk Outliers (using IQR method):")
    logger.info(f"  Lower bound: {lower_bound:.4f}")
    logger.info(f"  Upper bound: {upper_bound:.4f}")
    logger.info(f"  Number of outliers: {len(outliers)} ({len(outliers)/len(train_df_analysis)*100:.2f}%)")

    # Risk ranges (domain knowledge)
    logger.info("\nAccident Risk Distribution by Categories:")
    risk_ranges = {
        'Very Low (0-0.2)': (0, 0.2),
        'Low (0.2-0.4)': (0.2, 0.4),
        'Medium (0.4-0.6)': (0.4, 0.6),
        'High (0.6-0.8)': (0.6, 0.8),
        'Very High (0.8-1.0)': (0.8, 1.0)
    }

    for category, (low, high) in risk_ranges.items():
        count = len(train_df_analysis[(train_df_analysis[TARGET_COL] >= low) &
                                     (train_df_analysis[TARGET_COL] < high)])
        pct = count / len(train_df_analysis) * 100
        logger.info(f"  {category}: {count:,} samples ({pct:.1f}%)")

    # 3. Feature analysis
    logger.info("\n" + "="*40)
    logger.info("FEATURE ANALYSIS")
    logger.info("="*40)

    feature_cols = [col for col in train_df_analysis.columns if col != TARGET_COL]

    # Identify feature types
    categorical_features = ['road_type', 'lighting', 'weather', 'time_of_day']
    boolean_features = ['road_signs_present', 'public_road', 'holiday', 'school_season']
    numerical_features = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']

    logger.info(f"\nFeature categories:")
    logger.info(f"  Categorical: {len(categorical_features)} features - {categorical_features}")
    logger.info(f"  Boolean: {len(boolean_features)} features - {boolean_features}")
    logger.info(f"  Numerical: {len(numerical_features)} features - {numerical_features}")

    # Analyze categorical features
    logger.info("\n" + "-"*40)
    logger.info("CATEGORICAL FEATURES")
    logger.info("-"*40)
    for col in categorical_features:
        logger.info(f"\n{col}:")
        value_counts = train_df_analysis[col].value_counts()
        for val, count in value_counts.items():
            pct = count / len(train_df_analysis) * 100
            avg_risk = train_df_analysis[train_df_analysis[col] == val][TARGET_COL].mean()
            logger.info(f"  {val}: {count:,} ({pct:.1f}%) - Avg Risk: {avg_risk:.4f}")

    # Analyze boolean features
    logger.info("\n" + "-"*40)
    logger.info("BOOLEAN FEATURES")
    logger.info("-"*40)
    for col in boolean_features:
        logger.info(f"\n{col}:")
        true_count = train_df_analysis[col].sum()
        false_count = len(train_df_analysis) - true_count
        true_risk = train_df_analysis[train_df_analysis[col] == True][TARGET_COL].mean()
        false_risk = train_df_analysis[train_df_analysis[col] == False][TARGET_COL].mean()
        logger.info(f"  True: {true_count:,} ({true_count/len(train_df_analysis)*100:.1f}%) - Avg Risk: {true_risk:.4f}")
        logger.info(f"  False: {false_count:,} ({false_count/len(train_df_analysis)*100:.1f}%) - Avg Risk: {false_risk:.4f}")
        logger.info(f"  Risk Difference: {abs(true_risk - false_risk):.4f}")

    # Analyze numerical features
    logger.info("\n" + "-"*40)
    logger.info("NUMERICAL FEATURES")
    logger.info("-"*40)
    for col in numerical_features:
        col_stats = train_df_analysis[col].describe()
        logger.info(f"\n{col}:")
        logger.info(f"  Mean: {col_stats['mean']:.4f}")
        logger.info(f"  Std: {col_stats['std']:.4f}")
        logger.info(f"  Range: [{col_stats['min']:.4f}, {col_stats['max']:.4f}]")
        logger.info(f"  Skewness: {train_df_analysis[col].skew():.4f}")

        # Correlation with target
        corr = train_df_analysis[col].corr(train_df_analysis[TARGET_COL])
        logger.info(f"  Correlation with accident_risk: {corr:.4f}")

    # 4. Univariate analysis of features
    logger.info("\n" + "="*40)
    logger.info("UNIVARIATE ANALYSIS")
    logger.info("="*40)

    logger.info(f"\nCreating distribution plots for all features...")
    univariate_dir = OUTPUT_DIR / "univariate"
    eda.plot_univariate(
        train_df_analysis,
        save_dir=str(univariate_dir),
        individual_plots=True,
        max_categories=10
    )

    # 5. Multivariate analysis - Features vs Accident Risk
    logger.info("\n" + "="*40)
    logger.info("MULTIVARIATE ANALYSIS - FEATURES VS ACCIDENT RISK")
    logger.info("="*40)

    multivariate_dir = OUTPUT_DIR / "multivariate"
    eda.plot_multivariate(
        train_df_analysis,
        target=TARGET_COL,
        features=feature_cols,
        save_dir=str(multivariate_dir),
        individual_plots=True,
        max_categories=10
    )

    # 6. Correlation analysis
    logger.info("\n" + "="*40)
    logger.info("CORRELATION ANALYSIS")
    logger.info("="*40)

    # Create numeric version for correlation (encode booleans and categoricals)
    train_numeric = train_df_analysis.copy()
    for col in categorical_features:
        train_numeric[col] = pd.Categorical(train_numeric[col]).codes
    for col in boolean_features:
        train_numeric[col] = train_numeric[col].astype(int)

    # Pearson correlation
    eda.plot_correlation_matrix(
        train_numeric,
        method='pearson',
        save_path=str(OUTPUT_DIR / "correlation_pearson.png")
    )

    # Spearman correlation for non-linear relationships
    eda.plot_correlation_matrix(
        train_numeric,
        method='spearman',
        save_path=str(OUTPUT_DIR / "correlation_spearman.png")
    )

    # Feature correlations with accident_risk
    correlations = train_numeric.corr()[TARGET_COL].abs().sort_values(ascending=False)

    logger.info("\nFeature correlations with accident_risk (absolute values):")
    for feat, corr in correlations[1:].items():  # Skip target itself
        logger.info(f"  {feat}: {corr:.4f}")

    # 7. Missing data analysis
    logger.info("\n" + "="*40)
    logger.info("MISSING DATA ANALYSIS")
    logger.info("="*40)

    eda.plot_missing_data(
        train_df_analysis,
        save_path=str(OUTPUT_DIR / "missing_data.png")
    )

    missing_counts = train_df_analysis.isnull().sum()
    if missing_counts.sum() > 0:
        logger.info("\nColumns with missing values:")
        for col, count in missing_counts[missing_counts > 0].items():
            pct = count / len(train_df_analysis) * 100
            logger.info(f"  {col}: {count} ({pct:.2f}%)")
    else:
        logger.info("✓ No missing values found in the dataset!")

    # 8. Risk factor combinations
    logger.info("\n" + "="*40)
    logger.info("HIGH-RISK COMBINATIONS ANALYSIS")
    logger.info("="*40)

    # Dangerous conditions
    dangerous_weather = train_df_analysis['weather'].isin(['rainy', 'foggy'])
    poor_visibility = train_df_analysis['lighting'].isin(['dim', 'night'])
    high_curvature = train_df_analysis['curvature'] > train_df_analysis['curvature'].quantile(0.75)
    high_speed = train_df_analysis['speed_limit'] > train_df_analysis['speed_limit'].quantile(0.75)
    previous_accidents = train_df_analysis['num_reported_accidents'] > 0

    # Calculate risk for different combinations
    combinations = {
        'Dangerous weather': dangerous_weather,
        'Poor visibility': poor_visibility,
        'High curvature': high_curvature,
        'High speed limit': high_speed,
        'Previous accidents': previous_accidents,
        'Weather + Visibility': dangerous_weather & poor_visibility,
        'Curvature + Speed': high_curvature & high_speed,
        'Weather + Curvature': dangerous_weather & high_curvature,
        'All danger factors': dangerous_weather & poor_visibility & high_curvature
    }

    logger.info("\nAverage risk by condition combinations:")
    for name, condition in combinations.items():
        count = condition.sum()
        if count > 0:
            avg_risk = train_df_analysis[condition][TARGET_COL].mean()
            baseline_risk = train_df_analysis[TARGET_COL].mean()
            risk_increase = ((avg_risk - baseline_risk) / baseline_risk) * 100
            logger.info(f"  {name}: {avg_risk:.4f} ({count:,} samples, {risk_increase:+.1f}% vs baseline)")

    # 9. Adversarial Validation
    logger.info("\n" + "="*40)
    logger.info("ADVERSARIAL VALIDATION")
    logger.info("="*40)

    adv_results = eda.adversarial_validation(
        train_df=train_df,
        test_df=test_df,
        target_col=TARGET_COL,
        n_folds=5,
        sample_size=100000,
        save_path=str(OUTPUT_DIR / "adversarial_validation.png")
    )

    logger.info(f"\nAdversarial Validation Results:")
    logger.info(f"  AUC Score: {adv_results['auc_score']:.4f}")
    logger.info(f"  Interpretation: {adv_results['interpretation']}")

    if adv_results['feature_importance'] is not None and not adv_results['feature_importance'].empty:
        logger.info("\nFeatures causing distribution shift:")
        for _, row in adv_results['feature_importance'].head(5).iterrows():
            logger.info(f"  - {row['feature']}: {row['importance']:.4f}")

    # 10. Train vs Test distribution comparison
    logger.info("\n" + "="*40)
    logger.info("TRAIN VS TEST DISTRIBUTION COMPARISON")
    logger.info("="*40)

    logger.info(f"\nComparing features between train and test...")

    # For numerical features
    for col in numerical_features:
        train_mean = train_df_analysis[col].mean()
        test_mean = test_df_analysis[col].mean()
        train_std = train_df_analysis[col].std()
        test_std = test_df_analysis[col].std()

        mean_diff = abs(train_mean - test_mean) / (abs(train_mean) + 1e-10) * 100
        std_diff = abs(train_std - test_std) / (abs(train_std) + 1e-10) * 100

        if mean_diff > 5 or std_diff > 5:
            logger.warning(f"  {col}: Mean diff={mean_diff:.1f}%, Std diff={std_diff:.1f}%")
        else:
            logger.info(f"  {col}: Mean diff={mean_diff:.1f}%, Std diff={std_diff:.1f}%")

    # For categorical features
    logger.info("\nCategorical feature distributions:")
    for col in categorical_features:
        train_dist = train_df_analysis[col].value_counts(normalize=True)
        test_dist = test_df_analysis[col].value_counts(normalize=True)

        logger.info(f"\n  {col}:")
        for val in train_dist.index:
            train_pct = train_dist[val] * 100
            test_pct = test_dist.get(val, 0) * 100
            diff = abs(train_pct - test_pct)
            logger.info(f"    {val}: Train={train_pct:.1f}%, Test={test_pct:.1f}%, Diff={diff:.1f}%")

    # 11. Feature engineering recommendations
    logger.info("\n" + "="*40)
    logger.info("FEATURE ENGINEERING RECOMMENDATIONS")
    logger.info("="*40)

    logger.info("\n📊 Based on EDA, consider these feature engineering approaches:")

    logger.info("\n1. Risk factor interactions:")
    logger.info("   - weather × lighting (e.g., rainy + night = high risk)")
    logger.info("   - curvature × speed_limit (dangerous curves at high speeds)")
    logger.info("   - road_type × time_of_day (urban roads during rush hour)")
    logger.info("   - weather × road_type (rural roads in bad weather)")

    logger.info("\n2. Composite risk scores:")
    logger.info("   - visibility_risk = (weather=='foggy' or lighting=='night')")
    logger.info("   - geometry_risk = curvature × speed_limit")
    logger.info("   - historical_risk = num_reported_accidents / road_type_avg")
    logger.info("   - condition_risk = sum of binary dangerous conditions")

    logger.info("\n3. Aggregated statistics:")
    logger.info("   - Average accident_risk by road_type")
    logger.info("   - Average accidents by weather condition")
    logger.info("   - Risk ratios compared to baseline")
    logger.info("   - Target encoding for categorical features")

    logger.info("\n4. Transformations:")
    logger.info("   - Polynomial features for curvature and speed_limit")
    logger.info("   - Binning of continuous features")
    logger.info("   - One-hot encoding for categorical features")
    logger.info("   - Interaction terms between top correlated features")

    # 12. Summary
    logger.info("\n" + "="*60)
    logger.info("EDA SUMMARY")
    logger.info("="*60)

    logger.info("\n📈 Key Findings:")
    logger.info(f"  • Dataset: {len(train_df_analysis):,} training, {len(test_df_analysis):,} test samples")
    logger.info(f"  • Features: {len(feature_cols)} (4 categorical, 4 boolean, 4 numerical)")
    logger.info(f"  • Target: accident_risk (continuous, range: {train_df_analysis[TARGET_COL].min():.4f}-{train_df_analysis[TARGET_COL].max():.4f})")
    logger.info(f"  • Missing data: {'None' if missing_counts.sum() == 0 else f'{missing_counts.sum()} values'}")
    logger.info(f"  • Outliers: {len(outliers)} accident_risk outliers ({len(outliers)/len(train_df_analysis)*100:.2f}%)")

    # Find strongest predictors
    top_predictors = correlations[1:4].index.tolist()
    logger.info(f"  • Top predictors: {', '.join(top_predictors)}")

    baseline_risk = train_df_analysis[TARGET_COL].mean()
    logger.info(f"  • Baseline risk: {baseline_risk:.4f}")

    logger.info("\n🎯 Modeling Strategy:")
    logger.info("  1. Use ensemble of gradient boosting models (XGBoost, LightGBM, CatBoost)")
    logger.info("  2. CatBoost for native categorical feature handling")
    logger.info("  3. Create rich interaction features based on domain knowledge")
    logger.info("  4. Use RMSE as primary metric, monitor MAE and R²")
    logger.info("  5. Implement 5-fold cross-validation with stratification")
    logger.info("  6. Consider target encoding for high-cardinality categoricals")
    logger.info("  7. Clip predictions to [0, 1] range")

    logger.info(f"\n✅ All visualizations saved to {OUTPUT_DIR}/")
    logger.info("\n🚀 EDA Complete! Ready to proceed with modeling.")


def main():
    """Main execution function."""
    try:
        perform_eda()
    except Exception as e:
        logger.error(f"Error during EDA: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
