# TabML EDA Module Enhancements

**Version:** 0.5.0 (Proposed)
**Date:** 2025-09-30
**Status:** Implemented

## Overview

The EDA module has been significantly enhanced with automated analysis capabilities that transform it from **descriptive** to **prescriptive** - not just showing data, but recommending actionable next steps.

## New Features Summary

### 1. Data Quality Detection (`detect_data_quality_issues`)
**Purpose:** Automatically identify common data quality problems.

**Detects:**
- Constant/quasi-constant features (low variance)
- Duplicate columns (high correlation between features)
- High cardinality categorical features (>100 unique values)
- Features with excessive missing data (>95%)
- Outliers (IQR or Z-score methods)

**Returns:** Dictionary with issues and actionable recommendations

**Example:**
```python
eda = EDAAnalyzer()
quality = eda.detect_data_quality_issues(df)
print(quality['recommendations'])
# ['Drop constant features: [feat_1, feat_2]',
#  'Consider target encoding for: [high_card_feature]']
```

### 2. Feature Interaction Detection (`detect_feature_interactions`)
**Purpose:** Discover promising feature combinations for engineering.

**Identifies:**
- Ratio features (A/B) with high predictive power
- Product features (A*B) that capture interactions
- Non-linear transformations (squared, sqrt) that improve prediction
- Compares gain over base features

**Methods:** Mutual information or tree-based importance

**Example:**
```python
interactions = eda.detect_feature_interactions(df, target='price', top_k=10)
print(interactions['ratio_candidates'])
# [('feature_A', 'feature_B', 0.0342), ...]  # (feat1, feat2, gain)

# Automatically generates code suggestions:
# "Create ratio: feature_A / feature_B (gain: 0.0342)"
```

### 3. Target Distribution Analysis (`analyze_target_distribution`)
**Purpose:** Comprehensive target analysis with transformation suggestions.

**For Classification:**
- Class balance detection (severe/moderate/balanced)
- Binary vs multiclass identification
- Evaluation metric recommendations

**For Regression:**
- Distribution statistics (mean, median, std, skewness, kurtosis)
- Normality tests (Shapiro-Wilk, D'Agostino-Pearson)
- Transformation testing (log, sqrt, Box-Cox, Yeo-Johnson)
- Multimodality detection
- Metric recommendations (RMSE vs MAE)

**Generates Diagnostic Plots:**
- Histogram with mean/median
- Box plot
- Q-Q plot
- Best transformation comparison

**Example:**
```python
target_analysis = eda.analyze_target_distribution(df, target='price', save_dir='./plots')
print(target_analysis['suggested_transformations'][0])
# {'name': 'yeo-johnson', 'skewness': 0.05, 'formula': 'yeo-johnson(y)'}

print(target_analysis['recommendations'])
# ['⚠️ Target is highly skewed (2.34). Consider yeo-johnson transformation.',
#  '📊 Suggested metrics: MAE, Huber loss (robust to outliers)']
```

### 4. Leakage Detection (`detect_leakage`)
**Purpose:** Identify potential data leakage patterns early.

**Checks:**
- Perfect/near-perfect correlations with target (>0.99)
- Suspicious features (same cardinality as target, unique per row)
- Test set leakage (values only in test, not in train)
- Temporal leakage (date/time features that may leak future info)

**Example:**
```python
leakage = eda.detect_leakage(train_df, test_df, target='target')

if leakage['perfect_correlations']:
    print("🚨 CRITICAL: Potential leakage detected!")
    for leak in leakage['perfect_correlations']:
        print(f"{leak['feature']}: {leak['correlation']:.4f}")
```

### 5. Full EDA Analysis (`full_eda_analysis`)
**Purpose:** One-command comprehensive analysis orchestrator.

**Runs:**
1. Data quality checks
2. Target distribution analysis
3. Feature interaction detection (optional)
4. Adversarial validation (if test_df provided)
5. Leakage detection
6. Standard EDA visualizations

**Generates:**
- Organized output directory structure
- `recommendations.txt` with consolidated action items
- All diagnostic plots
- Complete results dictionary

**Example:**
```python
results = eda.full_eda_analysis(
    train_df=train,
    test_df=test,
    target='price',
    output_dir='comprehensive_eda',
    include_interactions=True
)

# Access any component
print(results['data_quality'])
print(results['target_analysis'])
print(results['feature_interactions'])
print(results['adversarial_validation'])
print(results['leakage_detection'])

# Get all recommendations
for rec in results['recommendations']:
    print(f"- {rec}")
```

## Updated Existing Features

### `generate_report` (Enhanced)
Now includes:
- Individual plot saving in organized subdirectories
- `/univariate/` - Individual feature distributions
- `/multivariate/` - Feature vs target relationships
- Better handling of large datasets

### `adversarial_validation` (Improved)
- Better ID column detection and removal
- Handles perfect separation gracefully
- More informative warnings for data leakage

## File Structure

Enhanced module now contains **2,191 lines** (was 1,136):

```
tabml/eda.py
├── EDAAnalyzer class
│   ├── __init__              # Configuration (palette, style, figsize)
│   ├── _identify_column_types # Internal: numerical vs categorical
│   ├── _apply_palette        # Internal: color palette management
│   ├── set_palette           # Public: change colors
│   │
│   ├── [Visualization Methods]
│   ├── plot_univariate       # Individual feature distributions
│   ├── plot_multivariate     # Feature vs target relationships
│   ├── plot_correlation_matrix # Heatmap
│   ├── plot_missing_data     # Missing data patterns
│   │
│   ├── [New Analysis Methods] ⭐
│   ├── detect_data_quality_issues      # Quality checks
│   ├── detect_feature_interactions     # Interaction discovery
│   ├── analyze_target_distribution     # Target analysis
│   ├── detect_leakage                  # Leakage detection
│   │
│   ├── [Comprehensive Methods]
│   ├── adversarial_validation # Train/test shift detection (improved)
│   ├── full_eda_analysis      # One-command full analysis ⭐
│   └── generate_report        # Standard EDA report (enhanced)
```

## Google-Style Docstrings

All new methods follow Google-style docstrings for Sphinx compatibility:

```python
def detect_feature_interactions(self,
                                df: pd.DataFrame,
                                target: str,
                                top_k: int = 10) -> Dict[str, Any]:
    """Detect promising feature interactions for engineering.

    Args:
        df: Input DataFrame with features and target.
        target: Name of target column.
        top_k: Number of top interaction pairs to return.

    Returns:
        Dictionary containing:
            - top_interactions: List of top feature pairs
            - ratio_candidates: Features good for ratios
            - recommendations: Action items

    Example:
        >>> eda = EDAAnalyzer()
        >>> interactions = eda.detect_feature_interactions(df, target='price')
        >>> print(interactions['recommendations'])
    """
```

## Integration with Competitions

These features codify techniques used in your PS5E8/PS5E9 competitions:

**PS5E9 BPM Prediction:**
- Manual: Created ratios, squares, quantile bins
- Now: `detect_feature_interactions` suggests them automatically

**Target Transformation:**
- Manual: Applied Yeo-Johnson based on intuition
- Now: `analyze_target_distribution` tests and recommends transformations

**Leakage Checks:**
- Manual: Inspected correlations manually
- Now: `detect_leakage` flags issues automatically

## Usage Patterns

### Quick Start (5 minutes)
```python
from tabml.eda import EDAAnalyzer

eda = EDAAnalyzer()
results = eda.full_eda_analysis(train, target='target', test_df=test)
# Review recommendations.txt and implement suggestions
```

### Competition Workflow
```python
# 1. Initial EDA
eda = EDAAnalyzer(palette='viridis')
results = eda.full_eda_analysis(train, test, target='target', output_dir='01_eda')

# 2. Check for critical issues
if results['leakage_detection']['perfect_correlations']:
    print("⚠️ Fix leakage before modeling!")

# 3. Apply recommendations
target_analysis = results['target_analysis']
if target_analysis['suggested_transformations'][0]['name'] != 'none':
    print(f"Apply: {target_analysis['suggested_transformations'][0]['name']}")

# 4. Feature engineering
interactions = results['feature_interactions']
for feat1, feat2, gain in interactions['ratio_candidates'][:5]:
    train[f'{feat1}_div_{feat2}'] = train[feat1] / train[feat2]

# 5. Validate approach
if results['adversarial_validation']['auc_score'] > 0.7:
    print("⚠️ High train/test shift - use stratified CV")
```

### Production Workflow
```python
# Data quality gate
quality = eda.detect_data_quality_issues(df)
if quality['constant_features']:
    drop_cols = [f['column'] for f in quality['constant_features']]
    df = df.drop(columns=drop_cols)

# Target validation
target_stats = eda.analyze_target_distribution(df, target='target')
if target_stats['class_balance']:
    minority_ratio = min(target_stats['class_balance']['class_ratios'].values())
    if minority_ratio < 0.05:
        print("⚠️ Apply SMOTE or class weights")
```

## Performance Considerations

**Fast Operations (<1s for 100k rows):**
- Data quality detection
- Target distribution analysis
- Leakage detection

**Medium Operations (1-30s):**
- Feature interactions (depends on feature count)
- Adversarial validation

**Slow Operations (>30s):**
- Full EDA with all options
- Feature interactions with >50 features

**Optimization:**
- Use `max_features` parameter to limit interaction search space
- Set `include_interactions=False` for quick analysis
- Sample large datasets for EDA

## Dependencies

**Required (already in TabML):**
- pandas, numpy, scipy, scikit-learn
- matplotlib, seaborn
- loguru

**No new dependencies added!**

## API Stability

**Stable (1.0 Ready):**
- `detect_data_quality_issues`
- `analyze_target_distribution`
- `detect_leakage`
- `full_eda_analysis`

**Experimental:**
- `detect_feature_interactions` (may add more interaction types)

**Unchanged:**
- All existing visualization methods maintain backward compatibility

## Future Enhancements (Not Implemented)

**Phase 2 (Could add later):**
- `statistical_tests_summary()` - Chi-square, ANOVA, Kruskal-Wallis tests
- `recommend_features()` - More specific feature engineering suggestions
- `quick_baseline()` - Train simple models during EDA

**Phase 3:**
- `generate_html_report()` - Interactive Plotly-based reports
- `analyze_temporal_patterns()` - Time series specific EDA
- `kaggle_insights()` - Competition-specific recommendations

## Testing

Create test file `tests/test_eda_enhancements.py`:

```python
def test_data_quality_detection():
    df = pd.DataFrame({
        'constant': np.ones(100),
        'target': np.random.randn(100)
    })
    eda = EDAAnalyzer()
    quality = eda.detect_data_quality_issues(df)
    assert len(quality['constant_features']) == 1

def test_leakage_detection():
    df = pd.DataFrame({
        'feature': np.random.randn(100),
        'leaky': np.random.randn(100),
        'target': np.random.randn(100)
    })
    df['leaky'] = df['target'] * 0.99  # Create leakage

    eda = EDAAnalyzer()
    leakage = eda.detect_leakage(df, None, 'target')
    assert len(leakage['perfect_correlations']) == 1
```

## Documentation

Update these files:
- [x] `tabml/eda.py` - Module and method docstrings (Google style)
- [ ] `README.md` - Add "Enhanced EDA" section
- [ ] `CHANGELOG.md` - Version 0.5.0 entry
- [ ] `docs/eda_guide.md` - Full tutorial (optional)

## Migration Guide

**Existing code works unchanged:**
```python
# This still works exactly as before
eda = EDAAnalyzer()
eda.generate_report(df, target='price', output_dir='eda')
```

**New capabilities (opt-in):**
```python
# Use new features as needed
quality = eda.detect_data_quality_issues(df)
leakage = eda.detect_leakage(train, test, 'target')
```

## Summary

**Lines of Code:** +1,055 (1,136 → 2,191)
**New Public Methods:** 4 major + 1 orchestrator
**Breaking Changes:** None
**New Dependencies:** None
**Test Coverage:** Ready for unit tests
**Documentation:** Google-style docstrings (Sphinx-ready)

**Key Improvement:** TabML EDA now provides **prescriptive recommendations** rather than just descriptive statistics, directly supporting your competition workflow.
