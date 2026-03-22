"""
TabML EDA Statistical Plots - Demo Script (ggpubr-style)

Demonstrates the new statistical comparison plotting capabilities with automatic
statistical test annotations, similar to R's ggpubr package.

Features:
- Automatic test selection (t-test, ANOVA, Mann-Whitney, Kruskal-Wallis)
- Statistical annotations on plots
- Pairwise comparison bars with significance stars (*, **, ***)
- Multiple plot types (box, violin, bar, strip)

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
    """Create sample dataset with significant group differences."""
    np.random.seed(42)

    # Create data with intentional group differences
    n_per_group = 100

    data_list = []

    # Group A: Low values
    data_list.append(pd.DataFrame({
        'department': ['Sales'] * n_per_group,
        'salary': np.random.normal(50000, 8000, n_per_group),
        'age': np.random.randint(22, 50, n_per_group),
        'satisfaction': np.random.normal(6.5, 1.5, n_per_group)
    }))

    # Group B: Medium values
    data_list.append(pd.DataFrame({
        'department': ['Engineering'] * n_per_group,
        'salary': np.random.normal(75000, 10000, n_per_group),
        'age': np.random.randint(25, 55, n_per_group),
        'satisfaction': np.random.normal(7.2, 1.3, n_per_group)
    }))

    # Group C: High values
    data_list.append(pd.DataFrame({
        'department': ['Management'] * n_per_group,
        'salary': np.random.normal(95000, 12000, n_per_group),
        'age': np.random.randint(30, 60, n_per_group),
        'satisfaction': np.random.normal(7.8, 1.2, n_per_group)
    }))

    # Group D: Medium-high (to demonstrate pairwise comparisons)
    data_list.append(pd.DataFrame({
        'department': ['Marketing'] * n_per_group,
        'salary': np.random.normal(65000, 9000, n_per_group),
        'age': np.random.randint(24, 52, n_per_group),
        'satisfaction': np.random.normal(6.9, 1.4, n_per_group)
    }))

    df = pd.concat(data_list, ignore_index=True)

    # Add some categorical variables
    df['experience_level'] = pd.cut(df['age'], bins=[0, 30, 45, 100],
                                    labels=['Junior', 'Mid', 'Senior'])

    df['performance'] = pd.cut(df['satisfaction'], bins=[0, 6, 7.5, 10],
                               labels=['Needs Improvement', 'Good', 'Excellent'])

    return df


def demo_basic_comparison(eda, df):
    """Demo 1: Basic statistical comparison with automatic test selection."""
    print("\n" + "="*70)
    print("DEMO 1: Basic Statistical Comparison")
    print("="*70)

    print("\nCreating box plot: Salary by Department")
    print("  - Automatic test selection (ANOVA or Kruskal-Wallis)")
    print("  - Statistical annotation at top of plot")

    results = eda.plot_statistical_comparison(
        df,
        categorical_col='department',
        numerical_col='salary',
        plot_type='box',
        add_stats=True,
        add_pairwise=False,
        save_path='./output/salary_by_dept_basic.png'
    )

    print(f"\n✓ Test performed: {results['test_result']['test']}")
    print(f"✓ P-value: {results['test_result']['p_value']:.6f}")
    print(f"✓ Interpretation: {results['test_result']['interpretation']}")
    print(f"✓ Significance: {results['test_result']['significance']}")

    return results


def demo_pairwise_comparisons(eda, df):
    """Demo 2: Pairwise comparisons with significance bars."""
    print("\n" + "="*70)
    print("DEMO 2: Pairwise Comparisons with Significance Bars")
    print("="*70)

    print("\nCreating box plot: Salary by Department")
    print("  - Overall statistical test")
    print("  - Pairwise comparisons between all groups")
    print("  - Significance bars with stars (*, **, ***)")

    results = eda.plot_statistical_comparison(
        df,
        categorical_col='department',
        numerical_col='salary',
        plot_type='box',
        add_stats=True,
        add_pairwise=True,  # Enable pairwise comparisons
        save_path='./output/salary_by_dept_pairwise.png'
    )

    print(f"\n✓ Overall test: {results['test_result']['interpretation']}")

    if results['pairwise_results']:
        print(f"\n✓ Pairwise comparisons ({len(results['pairwise_results'])} total):")
        for comparison in results['pairwise_results']:
            sig_marker = "✓" if comparison['significant'] else " "
            print(f"  {sig_marker} {comparison['group1']} vs {comparison['group2']}: "
                  f"p={comparison['p_value']:.4f} "
                  f"{'(significant)' if comparison['significant'] else '(not significant)'}")

    return results


def demo_different_plot_types(eda, df):
    """Demo 3: Different plot types with statistics."""
    print("\n" + "="*70)
    print("DEMO 3: Different Plot Types")
    print("="*70)

    plot_types = ['box', 'violin', 'bar', 'strip']

    for plot_type in plot_types:
        print(f"\n✓ Creating {plot_type} plot...")

        results = eda.plot_statistical_comparison(
            df,
            categorical_col='department',
            numerical_col='salary',
            plot_type=plot_type,
            add_stats=True,
            add_pairwise=False,
            save_path=f'./output/salary_by_dept_{plot_type}.png'
        )

        print(f"  Test: {results['test_result']['test']} "
              f"({results['test_result']['interpretation']})")


def demo_two_groups(eda, df):
    """Demo 4: Two-group comparison (t-test or Mann-Whitney)."""
    print("\n" + "="*70)
    print("DEMO 4: Two-Group Comparison (Binary)")
    print("="*70)

    # Create binary variable
    df_binary = df[df['department'].isin(['Sales', 'Management'])].copy()

    print("\nComparing: Sales vs Management")
    print("  - Automatic selection: t-test or Mann-Whitney U")

    results = eda.plot_statistical_comparison(
        df_binary,
        categorical_col='department',
        numerical_col='salary',
        plot_type='violin',
        add_stats=True,
        add_pairwise=False,
        save_path='./output/salary_sales_vs_mgmt.png'
    )

    print(f"\n✓ Test: {results['test_result']['test']}")
    print(f"✓ Result: {results['test_result']['interpretation']}")

    # Show group statistics
    print(f"\n✓ Group Statistics:")
    for group, stats in results['group_stats'].items():
        print(f"  {group}:")
        print(f"    n = {stats['count']}")
        print(f"    mean = ${stats['mean']:,.2f}")
        print(f"    median = ${stats['median']:,.2f}")
        print(f"    std = ${stats['std']:,.2f}")

    return results


def demo_small_groups(eda, df):
    """Demo 5: Satisfaction by experience level (fewer groups)."""
    print("\n" + "="*70)
    print("DEMO 5: Satisfaction by Experience Level")
    print("="*70)

    print("\nComparing: Satisfaction across Junior/Mid/Senior")
    print("  - 3 groups → perfect for pairwise comparisons")

    results = eda.plot_statistical_comparison(
        df,
        categorical_col='experience_level',
        numerical_col='satisfaction',
        plot_type='box',
        add_stats=True,
        add_pairwise=True,
        save_path='./output/satisfaction_by_experience.png'
    )

    print(f"\n✓ Overall: {results['test_result']['interpretation']}")

    if results['pairwise_results']:
        print(f"\n✓ Significant pairwise differences:")
        for comp in results['pairwise_results']:
            if comp['significant']:
                print(f"  - {comp['group1']} vs {comp['group2']}: p={comp['p_value']:.4f}")

    return results


def demo_performance_comparison(eda, df):
    """Demo 6: Age by performance category."""
    print("\n" + "="*70)
    print("DEMO 6: Age by Performance Category")
    print("="*70)

    results = eda.plot_statistical_comparison(
        df,
        categorical_col='performance',
        numerical_col='age',
        plot_type='violin',
        add_stats=True,
        add_pairwise=True,
        save_path='./output/age_by_performance.png'
    )

    print(f"\n✓ Test: {results['test_result']['interpretation']}")

    return results


def main():
    """Run all demonstrations."""
    print("="*70)
    print("TabML EDA Statistical Plots - Demo (ggpubr-style)")
    print("="*70)

    # Create output directory
    Path('./output').mkdir(exist_ok=True)

    # Create sample data
    print("\nCreating sample dataset with 4 departments...")
    df = create_sample_data()
    print(f"✓ Dataset created: {df.shape[0]} rows, {df.shape[1]} columns")

    # Initialize EDA analyzer
    eda = EDAAnalyzer(palette='Set2')
    print("✓ EDA Analyzer initialized")

    # Run demonstrations
    demo_basic_comparison(eda, df)
    demo_pairwise_comparisons(eda, df)
    demo_different_plot_types(eda, df)
    demo_two_groups(eda, df)
    demo_small_groups(eda, df)
    demo_performance_comparison(eda, df)

    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("\n✅ All statistical plots created successfully!")
    print("\n📁 Check ./output/ directory for plots:")
    print("  - salary_by_dept_basic.png (basic with overall test)")
    print("  - salary_by_dept_pairwise.png (with pairwise comparisons)")
    print("  - salary_by_dept_*.png (different plot types)")
    print("  - satisfaction_by_experience.png (3-group comparison)")
    print("  - age_by_performance.png (violin plot)")

    print("\n💡 Key Features:")
    print("  ✓ Automatic statistical test selection")
    print("  ✓ Statistical annotations at top of plots")
    print("  ✓ Pairwise comparison bars with stars")
    print("  ✓ Multiple plot types (box, violin, bar, strip)")
    print("  ✓ Group statistics in returned dictionary")
    print("\n🎯 Similar to R's ggpubr, but in Python!")


if __name__ == '__main__':
    main()
