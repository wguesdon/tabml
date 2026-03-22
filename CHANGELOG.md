# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **PS5E10 Example** - Complete Kaggle competition workflow for Road Accident Risk Prediction:
  - Full GPU-accelerated ensemble notebook with XGBoost, LightGBM, CatBoost, and Random Forest
  - EDA notebook with comprehensive visualizations and adversarial validation
  - Three-step automated feature engineering pipeline:
    - Domain-specific manual features (risk scores, boolean combinations)
    - TabML `FeatureEngineer` for systematic transformations (target encoding, interactions, polynomials)
    - TabML `FeatureSelector` for intelligent feature selection (mutual information ranking)
  - Example demonstrates 2-4% RMSE improvement with automated feature engineering
  - Complete documentation including setup, GPU configuration, and troubleshooting

### Fixed
- **LightGBM/XGBoost Regression Detection** - Improved `_determine_task_type()` heuristic in BaseModel:
  - **Old**: Simple `y.nunique() < 100` incorrectly classified float continuous targets as classification
  - **New**: Prioritizes dtype over unique count:
    - Float dtype (float32/float64) → always regression (continuous by nature)
    - Integer with <100 unique values → classification (discrete categories)
    - Otherwise → regression (default for safety)
  - Fixes issue where PS5E10 accident_risk (float [0-1] with 98 unique values) was detected as classification
  - Core fix in `tabml/models.py:94-115` - no workaround needed
  - Comprehensive test suite in `examples/PS5E10/test_lightgbm_synthetic.py`
  - All test cases pass: float targets correctly detected as regression regardless of unique count

### Improved
- Feature engineering workflow with `FeatureEngineer` + `FeatureSelector` pipeline
- GPU acceleration examples with P100 configuration (9x speedup expected)
- Competition notebook organization with cell-based structure for Kaggle compatibility
- Documentation of common TabML pitfalls and workarounds

## [0.5.1] - 2025-09-30

### Added
- **Statistical Comparison Plots** - Publication-ready plots with automatic statistical testing (ggpubr-style):
  - `plot_statistical_comparison()` - Create box/violin/bar/strip plots with statistical annotations:
    - Automatic statistical test selection (t-test, ANOVA, Mann-Whitney U, Kruskal-Wallis)
    - Statistical annotations at top of plots showing test name, p-value, and significance level
    - Pairwise comparison bars with significance stars (*, **, ***)
    - Multiple plot types: box, violin, bar (with SEM error bars), strip
    - Returns test results, group statistics, and pairwise comparisons
    - Support for 2 to 6 groups with automatic layout
  - `_compute_statistical_test()` - Internal method for automatic test selection:
    - Checks sample sizes and variance homogeneity
    - Selects appropriate parametric (t-test, ANOVA) or non-parametric (Mann-Whitney, Kruskal-Wallis) tests
    - Returns detailed results with p-value interpretation
  - `_add_stat_annotation()` - Add statistical test annotations to plots
  - `_add_pairwise_annotations()` - Add pairwise comparison bars with significance stars
- Enhanced `_plot_categorical_vs_numerical()` with optional statistical annotations:
  - New parameters: `add_stats` (overall test) and `add_pairwise` (pairwise comparisons)
  - Backward compatible - statistics disabled by default
- Example script `examples/eda_statistical_plots_demo.py` demonstrating all statistical plot features

### Changed
- Updated version to 0.5.1
- EDA module expanded from 2,190 to ~2,580 lines (+~390 lines)

### Improved
- Statistical rigor in EDA workflow with automatic hypothesis testing
- Publication-ready visualizations similar to R's ggpubr package
- Multivariate plots can now include statistical annotations
- Automatic test selection based on data characteristics (sample size, number of groups, variance)

## [0.5.0] - 2025-09-30

### Added
- **Enhanced EDA Module** - Comprehensive automated analysis with prescriptive recommendations:
  - `detect_data_quality_issues()` - Automatically identify common data quality problems:
    - Constant/quasi-constant features (low variance detection)
    - Duplicate columns (high correlation between features)
    - High cardinality categorical features (>100 unique values)
    - Features with excessive missing data (>95%)
    - Outlier detection using IQR or Z-score methods
    - Returns actionable recommendations for each issue
  - `detect_feature_interactions()` - Discover promising feature combinations:
    - Identify ratio features (A/B) with high predictive power
    - Find product features (A*B) that capture interactions
    - Detect non-linear transformations (squared, sqrt) that improve prediction
    - Support for mutual information or tree-based importance scoring
    - Automatic comparison of gain over base features
    - Generates specific feature engineering code suggestions
  - `analyze_target_distribution()` - Comprehensive target analysis:
    - Classification: Class balance detection, imbalance warnings, metric recommendations
    - Regression: Distribution stats, normality tests, transformation testing
    - Automatic testing of log, sqrt, Box-Cox, and Yeo-Johnson transformations
    - Multimodality detection using peak analysis
    - Generates diagnostic plots (histogram, box plot, Q-Q plot, transformation comparison)
    - Recommends best transformation and evaluation metrics
  - `detect_leakage()` - Early detection of data leakage patterns:
    - Perfect/near-perfect correlations with target (>0.99)
    - Suspicious patterns (ID columns, same cardinality as target, unique per row)
    - Test set leakage (values only in test, not in train)
    - Temporal leakage detection (date/time features that may leak future info)
    - Severity classification (CRITICAL, HIGH, MEDIUM)
  - `full_eda_analysis()` - One-command comprehensive analysis orchestrator:
    - Runs all EDA analyses automatically
    - Generates organized output directory with subdirectories
    - Creates consolidated `recommendations.txt` file with all action items
    - Includes data quality, target analysis, feature interactions, adversarial validation, and leakage detection
    - Returns complete results dictionary for programmatic access
    - Optional feature interaction detection (can be disabled for speed)
- Enhanced `adversarial_validation()` method:
  - Improved ID column detection and removal
  - Better handling of perfect separation cases
  - More informative warnings for potential data leakage
  - Enhanced feature importance visualization
- Example script `examples/eda_enhancements_demo.py` demonstrating all new features
- Comprehensive documentation in `docs/EDA_ENHANCEMENTS.md`

### Changed
- Updated version to 0.5.0
- EDA module expanded from 1,136 to 2,190 lines (+1,054 lines)
- All new methods use Google-style docstrings for Sphinx compatibility
- Enhanced module docstring with comprehensive examples
- `generate_report()` now supports better organization with subdirectories

### Improved
- EDA workflow transformed from **descriptive** to **prescriptive**
- Automated detection replaces manual inspection for common issues
- Actionable recommendations generated for each analysis
- Competition-ready: Codifies techniques from PS5E8/PS5E9 competitions
- Zero new dependencies added - uses existing TabML stack

## [0.4.0] - 2025-01-31

### Added
- **AutoGluon Integration** - Automatic machine learning with model selection and ensembling:
  - `AutoGluonModel` class that wraps AutoGluon's TabularPredictor
  - Full compatibility with TabML's OOF prediction system
  - Support for AutoGluon's internal bagging and stacking
  - Automatic model selection from multiple algorithms
  - Time-based and quality-based presets
  - Custom hyperparameter configurations
  - Model information and leaderboard access
  - Automatic cleanup of temporary files
- New `autogluon` installation option: `pip install tabml[autogluon]`
- Example script `04f_autogluon.py` for PS5E8 competition demonstrating:
  - Fast mode configuration for quick results
  - Best quality mode with bagging and stacking
  - Custom tree-based model configuration
  - Integration with OOFManager for ensemble combination
- `requirements-optional.txt` documenting optional dependencies
- AutoGluon section in Available Models documentation
- Detailed installation instructions for AutoGluon in README

### Changed
- Updated version to 0.4.0
- Enhanced `setup.py` with AutoGluon as optional dependency
- Updated README with AutoGluon examples and installation guide
- Added AutoGluon to `[all]` extras installation

## [0.3.0] - 2025-01-30

### Added
- **MLflow Integration** - Comprehensive experiment tracking and model management:
  - `MLflowTracker` class for full experiment lifecycle tracking
  - `MLflowCallback` for training callbacks with automatic metric logging
  - `MLflowModelRegistry` for model versioning and deployment staging
  - Dataset versioning with automatic hash generation
  - Model artifact management with automatic upload
  - Support for multiple MLflow servers per project
  - Model signature inference and validation
  - Experiment search and comparison utilities
  - **DagsHub MLflow support** with MLflow 2.x compatibility (pinned to `<3.0` for DagsHub)
- New `tracking` installation option includes mlflow>=2.8.0,<3.0
- Example script `mlflow_tracking_example.py` demonstrating all MLflow features
- MLflow documentation section in README with both local and DagsHub hosting instructions
- DagsHub configuration in .env.example with authentication setup
- Support for environment variables via .env file with python-dotenv

### Changed
- **BREAKING**: Replaced Neptune.ai integration with MLflow (more cost-effective, self-hostable)
- Updated `setup.py` to include mlflow>=2.8.0,<3.0 in tracking dependencies (DagsHub compatibility)
- Enhanced `training.py` module with MLflowCallback support
- Updated `__init__.py` to export MLflow classes instead of Neptune
- Updated .env.example with MLflow configuration options

### Removed
- Neptune.ai integration (replaced with MLflow):
  - Removed `NeptuneTracker` class
  - Removed `NeptuneCallback` class
  - Removed `NeptuneModelRegistry` class
  - Removed `neptune_tracking_example.py`
  - Removed neptune>=1.8.0 from dependencies

### Fixed
- Repository cleanup: removed test files from root directory
- Moved `spaceship_processor_config.json` to appropriate examples directory
- Removed internal documentation files (ANALYSIS_GUIDE.md, DOCS_GENERATION_GUIDE.md, TASKS.md)
- Simplified README to be more technical and concise

## [0.2.0] - 2025-01-27

### Added
- High cardinality categorical feature handling in `FeatureEngineer`:
  - `max_cardinality` parameter to limit unique values per categorical column (default: 50)
  - `min_frequency` parameter to set minimum frequency threshold (default: 0.01)
  - `rare_label` parameter to customize grouped category label (default: 'Other')
  - Automatic detection and warning for high cardinality features
  - Frequency-based grouping of rare categories
  - Consistent mapping applied during transform for test data
- CHANGELOG.md file for version tracking

### Changed
- `FeatureEngineer` now automatically reduces cardinality before encoding to prevent dimensionality explosion
- Repository cleanup: removed generated files (catboost_info, *.log, *.egg-info)
- Updated .gitignore to prevent tracking of generated artifacts

### Removed
- Old example output directories
- Generated log files from root directory
- Build artifacts (tabml.egg-info)

## [0.1.0] - 2025-01-26

### Added
- Initial release of TabML package
- Core data processing functionality with `DataProcessor` class
- Model implementations: XGBoost, LightGBM, CatBoost, RandomForest
- Feature engineering capabilities with `FeatureEngineer`
- Data validation with `DataValidator`
- Visualization tools with `Visualizer`
- Time series support with `TimeSeriesProcessor`
- Walk-forward validation for time series
- Cross-validation utilities
- Model evaluation metrics
- Example scripts for Spaceship Titanic competition
- Comprehensive test suite
- Documentation with Sphinx
- CLI interface for quick model training