"""
TabML Enhanced EDA Features - Demonstration Script

This script demonstrates the new automated EDA capabilities added to TabML.
It showcases how the EDA module now provides prescriptive recommendations
rather than just descriptive statistics.

Author: TabML
Date: 2025-09-30
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add TabML to path if needed
if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).parent.parent))

from tabml.eda import EDAAnalyzer


def create_sample_data():
    """Create sample dataset with various data quality issues."""
    np.random.seed(42)
    n = 2000

    # Create base features
    data = {
        # Normal features
        'age': np.random.randint(18, 80, n),
        'income': np.random.lognormal(10, 1, n),  # Skewed
        'experience': np.random.randint(0, 40, n),
        'education_years': np.random.randint(10, 20, n),

        # Features with issues
        'constant_feature': np.ones(n),  # Constant
        'id_column': range(n),  # Row ID
        'mostly_missing': [np.nan] * int(n * 0.97) + list(range(int(n * 0.03))),

        # Categorical
        'department': np.random.choice(['Sales', 'Engineering', 'Marketing', 'HR'], n),
        'city': np.random.choice(['NYC', 'SF', 'LA', 'Boston', 'Seattle'], n),

        # High cardinality
        'user_id': [f'USER_{i:05d}' for i in range(n)],
    }

    df = pd.DataFrame(data)

    # Create target with some features having strong relationships
    df['salary'] = (
        df['age'] * 500 +  # Age effect
        df['experience'] * 1200 +  # Experience effect
        df['education_years'] * 800 +  # Education effect
        np.random.randn(n) * 5000 +  # Random noise
        30000  # Base salary
    )

    # Create a leaky feature (perfect correlation)
    df['leaky_salary_copy'] = df['salary'] * 1.001 + np.random.randn(n) * 10

    # Add some outliers
    outlier_indices = np.random.choice(n, size=20, replace=False)
    df.loc[outlier_indices, 'salary'] *= 3

    return df


def demo_data_quality_detection(eda, df):
    """Demonstrate data quality detection."""
    print("\n" + "="*70)
    print("1. DATA QUALITY DETECTION")
    print("="*70)

    quality = eda.detect_data_quality_issues(
        df,
        variance_threshold=0.01,
        missing_threshold=0.95,
        cardinality_threshold=100
    )

    print(f"\n✓ Constant features found: {len(quality['constant_features'])}")
    for feat in quality['constant_features']:
        print(f"  - {feat['column']} (variance: {feat['variance']:.6f})")

    print(f"\n✓ Duplicate columns found: {len(quality['duplicate_columns'])}")

    print(f"\n✓ High cardinality features: {len(quality['high_cardinality'])}")
    for feat in quality['high_cardinality']:
        print(f"  - {feat['column']} ({feat['unique_count']} unique values)")

    print(f"\n✓ High missing data: {len(quality['high_missing'])}")
    for feat in quality['high_missing']:
        print(f"  - {feat['column']} ({feat['missing_ratio']*100:.1f}% missing)")

    print(f"\n✓ Features with outliers: {len(quality['outlier_features'])}")
    for col, info in list(quality['outlier_features'].items())[:3]:
        print(f"  - {col}: {info['count']} outliers ({info['ratio']*100:.1f}%)")

    print("\n📋 Recommendations:")
    for i, rec in enumerate(quality['recommendations'], 1):
        print(f"  {i}. {rec}")

    return quality


def demo_target_analysis(eda, df):
    """Demonstrate target distribution analysis."""
    print("\n" + "="*70)
    print("2. TARGET DISTRIBUTION ANALYSIS")
    print("="*70)

    target_analysis = eda.analyze_target_distribution(
        df,
        target='salary',
        save_dir='./output/target_analysis'
    )

    print(f"\n✓ Task type: {target_analysis['task_type']}")

    stats = target_analysis['distribution_stats']
    print(f"\n✓ Distribution statistics:")
    print(f"  - Mean: ${stats['mean']:,.2f}")
    print(f"  - Median: ${stats['median']:,.2f}")
    print(f"  - Std: ${stats['std']:,.2f}")
    print(f"  - Skewness: {stats['skewness']:.4f}")
    print(f"  - Kurtosis: {stats['kurtosis']:.4f}")

    print(f"\n✓ Normality tests:")
    for test_name, test_result in target_analysis['normality_tests'].items():
        result = "Normal" if test_result['is_normal'] else "Not normal"
        print(f"  - {test_name}: p={test_result['p_value']:.4f} ({result})")

    print(f"\n✓ Top 3 transformation suggestions:")
    for i, trans in enumerate(target_analysis['suggested_transformations'][:3], 1):
        print(f"  {i}. {trans['name']}: skewness={trans['skewness']:.4f}")

    print("\n📋 Recommendations:")
    for i, rec in enumerate(target_analysis['recommendations'], 1):
        print(f"  {i}. {rec}")

    return target_analysis


def demo_feature_interactions(eda, df):
    """Demonstrate feature interaction detection."""
    print("\n" + "="*70)
    print("3. FEATURE INTERACTION DETECTION")
    print("="*70)

    # Select subset of features for speed
    feature_subset = ['age', 'experience', 'education_years', 'income', 'salary']
    df_subset = df[feature_subset].copy()

    interactions = eda.detect_feature_interactions(
        df_subset,
        target='salary',
        top_k=5,
        method='mutual_info'
    )

    print(f"\n✓ Top ratio candidates:")
    for feat1, feat2, gain in interactions['ratio_candidates'][:3]:
        print(f"  - {feat1} / {feat2} (gain: {gain:.4f})")

    print(f"\n✓ Top product candidates:")
    for feat1, feat2, gain in interactions['product_candidates'][:3]:
        print(f"  - {feat1} * {feat2} (gain: {gain:.4f})")

    print(f"\n✓ Non-linear features:")
    for feat_info in interactions['nonlinear_features'][:3]:
        print(f"  - {feat_info['feature']}: best_transform={feat_info['best_transform']}")

    print("\n📋 Recommendations:")
    for i, rec in enumerate(interactions['recommendations'], 1):
        print(f"  {i}. {rec}")

    return interactions


def demo_leakage_detection(eda, df):
    """Demonstrate leakage detection."""
    print("\n" + "="*70)
    print("4. LEAKAGE DETECTION")
    print("="*70)

    # Create simple test set
    test_df = df.sample(n=100, random_state=42).copy()

    leakage = eda.detect_leakage(
        train_df=df,
        test_df=test_df,
        target='salary',
        threshold=0.99
    )

    print(f"\n✓ Perfect correlations detected: {len(leakage['perfect_correlations'])}")
    for leak in leakage['perfect_correlations']:
        severity = "🚨" if leak['severity'] == 'CRITICAL' else "⚠️"
        print(f"  {severity} {leak['feature']}: {leak['correlation']:.6f} ({leak['severity']})")

    print(f"\n✓ Suspicious features: {len(leakage['suspicious_features'])}")
    for feat in leakage['suspicious_features'][:5]:
        print(f"  - {feat['feature']}: {feat['reason']}")

    print(f"\n✓ Test leakage: {len(leakage['test_leakage'])}")

    print(f"\n✓ Temporal leakage warnings: {len(leakage['temporal_leakage'])}")

    print("\n📋 Recommendations:")
    for i, rec in enumerate(leakage['recommendations'], 1):
        print(f"  {i}. {rec}")

    return leakage


def demo_full_analysis(eda, df):
    """Demonstrate comprehensive EDA analysis."""
    print("\n" + "="*70)
    print("5. COMPREHENSIVE FULL ANALYSIS")
    print("="*70)
    print("\nRunning full_eda_analysis() - this combines all tools...\n")

    # Create simple test set
    test_df = df.sample(n=100, random_state=42).copy()

    results = eda.full_eda_analysis(
        train_df=df,
        test_df=test_df,
        target='salary',
        output_dir='./output/comprehensive_eda',
        include_interactions=False  # Skip for speed in demo
    )

    print("\n" + "="*70)
    print("FULL ANALYSIS COMPLETE")
    print("="*70)

    print(f"\n✓ Total recommendations generated: {len(results['recommendations'])}")

    print("\n📋 Top 10 Consolidated Recommendations:")
    for i, rec in enumerate(results['recommendations'][:10], 1):
        print(f"  {i}. {rec}")

    print("\n📁 Output saved to: ./output/comprehensive_eda/")
    print("  - recommendations.txt (all recommendations)")
    print("  - target_analysis/ (target plots)")
    print("  - standard_eda/ (univariate, multivariate, correlation plots)")
    print("  - adversarial_validation.png")

    return results


def main():
    """Run all demonstrations."""
    print("="*70)
    print("TabML Enhanced EDA Features - Demonstration")
    print("="*70)
    print("\nCreating sample dataset with intentional data quality issues...")

    df = create_sample_data()
    print(f"✓ Dataset created: {df.shape[0]} rows, {df.shape[1]} columns")

    # Initialize EDA analyzer
    eda = EDAAnalyzer(
        figsize=(12, 8),
        style='seaborn',
        palette='viridis'
    )
    print("✓ EDA Analyzer initialized")

    # Run demonstrations
    quality = demo_data_quality_detection(eda, df)
    target_analysis = demo_target_analysis(eda, df)
    interactions = demo_feature_interactions(eda, df)
    leakage = demo_leakage_detection(eda, df)

    # Full analysis (combines everything)
    # results = demo_full_analysis(eda, df)

    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("\n✅ All new EDA features demonstrated successfully!")
    print("\n💡 Key Takeaways:")
    print("  1. Data quality issues are automatically detected")
    print("  2. Target transformations are automatically suggested")
    print("  3. Feature interactions are discovered automatically")
    print("  4. Leakage is caught early before modeling")
    print("  5. All recommendations are actionable")
    print("\n🚀 Use full_eda_analysis() for one-command comprehensive EDA!")


if __name__ == '__main__':
    main()
