# PRD: Kaggle Multi-Level Stacking Framework

## Problem Statement

Winning Kaggle solutions for tabular competitions follow a consistent pattern: diverse feature engineering, many independently trained models producing OOF predictions, and multi-level stacking to combine them. This workflow is well understood but tedious to implement from scratch each competition. The existing repo has strong ensemble utilities (`util/smart_ensemble.py`), AWS training infrastructure (`aws_training/src/base_trainer.py`), and per-competition scripts. But each competition still requires re-implementing the same CV loop, OOF prediction management, feature engineering pipeline, and stacking logic.

This package consolidates those patterns into a reusable framework that supports both classification and regression competitions.

## Goals

1. Standardize the multi-level stacking workflow (1st place PS6E3 pattern) as a reusable pipeline.
2. Eliminate boilerplate CV, OOF, and prediction management code from individual model notebooks.
3. Make it easy to add new models, feature sets, and stacking levels without touching infrastructure code.
4. Work locally and on AWS/Colab with minimal changes.
5. Complement existing tools (`util/smart_ensemble.py`, `aws_training/`) rather than replace them.

## Non-Goals

- Auto-ML or automated model selection. The user decides which models to train.
- Kaggle submission management. That stays in `util/smart_ensemble.py`.
- Neural network training loops. Each NN architecture has its own training code. The framework only manages OOF predictions and stacking.
- Automated feature engineering discovery. The user writes feature functions. The framework manages their application across folds.

## Target Users

Internal use. The primary user is someone running Kaggle Playground Series competitions with the existing repo structure.

## Implementation Strategy

Rather than building a new package from scratch, extend `tabml` with a `kaggle` subpackage. This reuses the existing `BaseModel`, `OOFEnsemble`, `OOFManager`, and `FeatureEngineer` as foundations. The new `CVManager` wraps `OOFEnsemble`'s fold logic but adds nested inner CV. The new `StackingLevel` orchestrates multiple `OOFManager` instances across levels.

```
tabml/
├── (existing modules unchanged)
├── kaggle/                        # NEW subpackage
│   ├── __init__.py
│   ├── config.py                  # CompetitionConfig dataclass
│   ├── cv_manager.py              # CVManager with nested OOF support
│   ├── stacking.py                # StackingLevel + MetaLearner
│   └── features/
│       ├── __init__.py
│       ├── snap.py                # SnapTransformer
│       ├── digits.py              # DigitExtractor
│       ├── target_encoding.py     # NestedTargetEncoder (multi-stat, bigrams)
│       ├── categorical.py         # CrossFeatures, FrequencyEncoder
│       └── binning.py             # Multi-scale binning
```

### What existing `tabml` modules provide (no duplication needed)

| Existing module | Reused by |
|---|---|
| `models.BaseModel`, `XGBoostModel`, `LightGBMModel`, `CatBoostModel` | `kaggle.stacking.StackingLevel` trains models via these wrappers |
| `ensemble.OOFEnsemble` | `kaggle.cv_manager.CVManager` wraps its fold logic, adds nested inner CV |
| `oof_manager.OOFManager` | `kaggle.stacking.StackingLevel` uses it for per-level prediction I/O |
| `features.FeatureEngineer` | `kaggle.features` transformers complement it with competition-specific techniques |
| `evaluate.CrossValidator` | `kaggle.cv_manager.CVManager` reuses its metric computation |

## Architecture Overview

```
kaggle_stack/
├── __init__.py
├── config.py              # Competition config dataclass
├── cv.py                  # CV splitter and OOF manager
├── features/
│   ├── __init__.py
│   ├── base.py            # FeatureTransformer base class
│   ├── target_encoding.py # Nested leak-free target encoding
│   ├── numeric.py         # Digit extraction, snap, binning, interactions
│   └── categorical.py     # Cross-features, frequency encoding, count encoding
├── models/
│   ├── __init__.py
│   ├── base.py            # BaseModel wrapper
│   ├── gbdt.py            # XGBoost, LightGBM, CatBoost wrappers
│   └── tabular_nn.py      # RealMLP, TabM, FT-Transformer wrappers
├── stacking/
│   ├── __init__.py
│   ├── level.py           # StackingLevel orchestrator
│   ├── meta.py            # Meta-learner (LogReg, Ridge)
│   └── hill_climbing.py   # Greedy forward model selection
├── experiment.py           # Experiment tracker (results, comparisons)
└── io.py                   # Prediction I/O (npy save/load, naming conventions)
```

## Core Components

### 1. Competition Config

A single config object defines everything shared across all models in a competition.

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class CompetitionConfig:
    name: str                                    # e.g. "playground-series-s6e3"
    task: Literal["classification", "regression"]
    metric: str                                  # e.g. "roc_auc", "rmse"
    higher_is_better: bool
    target_col: str                              # e.g. "Churn"
    id_col: str                                  # e.g. "id"
    n_folds: int = 5
    random_state: int = 42
    data_dir: str = "data/raw"
    predictions_dir: str = "predictions"
    features_dir: str = "data/features"
```

This replaces the scattered constants (`N_FOLDS = 5`, `RANDOM_STATE = 42`, `TARGET_BINS = 10`) that currently appear at the top of every script.

### 2. CV and OOF Manager

The CV manager owns fold indices and OOF prediction arrays. Every model and feature transformer uses the same fold splits.

```python
class CVManager:
    def __init__(self, config: CompetitionConfig, y: np.ndarray):
        """Create fold indices from target array."""

    def get_fold_indices(self, fold: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (train_idx, val_idx) for a given fold."""

    def create_oof_array(self, n_samples: int) -> np.ndarray:
        """Return zeroed array for OOF predictions."""

    def score_oof(self, y_true: np.ndarray, oof_preds: np.ndarray) -> float:
        """Compute competition metric on OOF predictions."""

    def save_predictions(self, name: str, version: int,
                         oof: np.ndarray, test: np.ndarray):
        """Save oof_{name}_v{version}.npy and pred_{name}_v{version}.npy."""

    def load_predictions(self, name: str, version: int
                         ) -> tuple[np.ndarray, np.ndarray]:
        """Load a saved (oof, test) prediction pair."""

    def load_all_predictions(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Load every prediction pair from predictions_dir."""
```

**Why this matters:** The 1st place solution's interface between levels is the `predictions/` folder full of `.npy` files. This component standardizes that interface. Every model saves predictions through the same API. Stacking levels load them through the same API.

**Nested OOF for features:** For leak-free feature computation (target encoding, KNN features), the manager supports a nested inner CV:

```python
class CVManager:
    def nested_transform(self, fold: int, transformer: FeatureTransformer,
                         X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Apply transformer using inner CV within the given outer fold.
        
        Splits the outer-fold training set into inner folds,
        fits the transformer on each inner train split,
        and returns OOF-transformed values for the outer train set
        plus direct-transformed values for the outer val set.
        """
```

This is the "5x5 nested OOF" from the 1st place solution. The outer 5-fold CV controls model training. The inner 5-fold CV within each outer fold controls feature computation. No target leakage.

### 3. Feature Transformers

Feature transformers follow a consistent interface. Each one knows how to fit on training data and transform both training and test data.

```python
class FeatureTransformer(ABC):
    """Base class for feature transformers."""
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "FeatureTransformer":
        """Fit on training data."""

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data, returning new columns only."""

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """Names of output columns."""
```

**Built-in transformers** (based on 1st place solution techniques):

| Transformer | Description | Reference |
|---|---|---|
| `NestedTargetEncoder` | Leak-free TE with configurable stats (mean, std, quantiles). Supports single columns, bigrams, trigrams. | Section 2.3 |
| `SnapTransformer` | Map synthetic floats to nearest original dataset value. Returns snap value and snap diff. | Section 2.1 |
| `DigitExtractor` | Extract decimal digits (d1, d2, frac100, mod10, mod100) from numeric columns. | Section 2.2 |
| `ArithmeticInteractions` | User-defined arithmetic formulas between columns. | Section 2.4 |
| `FrequencyEncoder` | Value frequency as a feature. | Section 2.7 |
| `CategoricalCrossFeatures` | Bigram/trigram concatenation of categorical columns. | Section 2.6 |
| `ServiceAggregator` | Count of "Yes"/"No" across a set of binary columns. | Section 2.8 |
| `BinningTransformer` | Quantile, fixed-width, or log-scale binning. | Section 2.5 |

**Usage pattern in a model notebook:**

```python
from kaggle_stack.features import SnapTransformer, DigitExtractor, NestedTargetEncoder

features = [
    SnapTransformer(original_data=orig_df, columns=["MonthlyCharges", "TotalCharges"]),
    DigitExtractor(columns=["MonthlyCharges", "TotalCharges"]),
    NestedTargetEncoder(columns=cat_cols, stats=["mean", "std"]),
]

# In the CV loop, the CVManager applies these with proper nesting
for fold in range(config.n_folds):
    train_idx, val_idx = cv.get_fold_indices(fold)
    X_train_feat = cv.apply_features(fold, features, X_train, y_train)
    X_val_feat = cv.apply_features_transform(fold, features, X_val)
```

### 4. Model Wrappers

Thin wrappers that normalize the interface across model libraries. Each wrapper handles library-specific fit/predict differences.

```python
class BaseModel(ABC):
    """Wrapper for a single model."""

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray, **kwargs) -> "BaseModel":
        """Fit model with early stopping on validation set."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions (probabilities for classification, values for regression)."""


class XGBoostModel(BaseModel):
    def __init__(self, params: dict, num_boost_round: int = 10000,
                 early_stopping_rounds: int = 100): ...

class LightGBMModel(BaseModel):
    def __init__(self, params: dict, num_boost_round: int = 10000,
                 early_stopping_rounds: int = 50): ...

class CatBoostModel(BaseModel):
    def __init__(self, params: dict, cat_features: list[str] | None = None): ...

class RealMLPModel(BaseModel):
    def __init__(self, n_ens: int = 8, **kwargs): ...
```

**Integration with Optuna** (reuse existing `aws_training/src/base_trainer.py` pattern):

```python
class OptunaTuner:
    def __init__(self, model_cls: type[BaseModel], cv: CVManager,
                 X: pd.DataFrame, y: np.ndarray, n_trials: int = 100):
        """Run Optuna HPO using a single train/val split for speed."""

    def get_search_space(self, trial: optuna.Trial) -> dict:
        """Override to define search space per model type."""

    def run(self) -> dict:
        """Return best params."""
```

### 5. Stacking Levels

This is the core orchestrator. A `StackingLevel` trains multiple models, collects their OOF predictions, and makes them available to the next level.

```python
class StackingLevel:
    def __init__(self, level: int, cv: CVManager, config: CompetitionConfig):
        """Initialize a stacking level.
        
        Args:
            level: 1 for feature extraction, 2 for base models,
                   3 for stacked models, 4 for meta-learner.
        """

    def add_model(self, name: str, version: int, model: BaseModel,
                  features: list[FeatureTransformer] | None = None):
        """Register a model to be trained at this level."""

    def train_model(self, name: str, version: int,
                    X_train: pd.DataFrame, y_train: np.ndarray,
                    X_test: pd.DataFrame,
                    extra_oof_cols: dict[str, np.ndarray] | None = None):
        """Train a single model across all folds.
        
        Applies features with nested CV, trains per fold,
        saves oof and test predictions.
        
        extra_oof_cols: OOF predictions from previous levels
                        to use as additional input features.
        """

    def get_oof_matrix(self) -> tuple[np.ndarray, list[str]]:
        """Return (n_samples, n_models) matrix of all OOF predictions
        from this level, plus model names."""

    def get_test_matrix(self) -> tuple[np.ndarray, list[str]]:
        """Return (n_test, n_models) matrix of all test predictions."""
```

**How the levels connect:**

```python
config = CompetitionConfig(name="ps-s6e3", task="classification", ...)
cv = CVManager(config, y_train)

# Level 1: Feature extraction (KNN, DAE, PCA, TE)
level1 = StackingLevel(level=1, cv=cv, config=config)
level1.train_model("knn_features", 100, knn_model, X_train, y_train, X_test)
level1.train_model("dae_features", 200, dae_model, X_train, y_train, X_test)
l1_oof, l1_names = level1.get_oof_matrix()

# Level 2: Base classifiers (each notebook runs one of these)
level2 = StackingLevel(level=2, cv=cv, config=config)
level2.train_model("xgb", 2100, xgb_model, X_train, y_train, X_test,
                    extra_oof_cols=dict(zip(l1_names, l1_oof.T)))

# Level 3: Stacked models (use L2 OOF as features)
level3 = StackingLevel(level=3, cv=cv, config=config)
l2_oof, l2_names = level2.get_oof_matrix()
level3.train_model("xgb_stk", 8000, xgb_stk_model, X_train, y_train, X_test,
                    extra_oof_cols=dict(zip(l2_names, l2_oof.T)))

# Level 4: Meta-learner
meta = MetaLearner(cv=cv, config=config)
l3_oof, l3_names = level3.get_oof_matrix()
all_oof = np.hstack([l2_oof, l3_oof])
all_test = np.hstack([level2.get_test_matrix()[0], level3.get_test_matrix()[0]])
final_preds = meta.fit_predict(all_oof, y_train, all_test)
```

**Important:** Each `train_model` call is independent. In practice, most Level 2 models run in separate notebooks. The `StackingLevel` loads saved `.npy` files from `predictions/` to build the OOF matrix. Training 850 models does not happen in a single script.

### 6. Meta-Learner

The final combination layer.

```python
class MetaLearner:
    def __init__(self, cv: CVManager, config: CompetitionConfig,
                 method: str = "logistic_regression"):
        """
        Supported methods:
        - logistic_regression: L2-penalized LogReg (classification)
        - ridge: Ridge regression (regression)
        - hill_climbing: Greedy forward selection with weight optimization
        """

    def fit_predict(self, oof_matrix: np.ndarray, y_train: np.ndarray,
                    test_matrix: np.ndarray) -> np.ndarray:
        """Fit on OOF predictions, return final test predictions."""

    def get_model_weights(self) -> dict[str, float]:
        """Return weight assigned to each model (for analysis)."""

    def select_models(self, oof_matrix: np.ndarray, y_train: np.ndarray,
                      model_names: list[str],
                      max_models: int = 150) -> list[str]:
        """Hill climbing: greedily select models that improve ensemble score."""
```

### 7. Experiment Tracker

Lightweight tracking. Not a replacement for MLflow. Just enough to compare models within a competition.

```python
class ExperimentTracker:
    def __init__(self, config: CompetitionConfig):
        """Load or create experiments.json in the competition directory."""

    def log_model(self, name: str, version: int, oof_score: float,
                  fold_scores: list[float], params: dict | None = None,
                  features_used: list[str] | None = None,
                  notes: str | None = None):
        """Record a model's results."""

    def get_leaderboard(self) -> pd.DataFrame:
        """Return all models sorted by OOF score."""

    def compare_models(self, names: list[str]) -> pd.DataFrame:
        """Side-by-side comparison of selected models."""
```

## Workflow: How a Competition Plays Out

### Day 1: Setup

```bash
# Create competition directory
mkdir -p PS6E3/{data/raw,predictions,eda,models/{level2,level3},ensemble}

# Download data
kaggle competitions download -c playground-series-s6e3 -p PS6E3/data/raw
```

```python
# config.py for this competition
config = CompetitionConfig(
    name="playground-series-s6e3",
    task="classification",
    metric="roc_auc",
    higher_is_better=True,
    target_col="Churn",
    id_col="id",
    n_folds=5,
)
```

### Days 1 to 3: EDA and Feature Engineering

Write feature transformer classes. Test them in EDA notebooks. Build the feature store.

### Days 3 to 25: Model Training

Each model gets its own notebook in `models/level2/`. The notebook:

1. Loads data and config
2. Picks a feature set
3. Trains across folds using `CVManager`
4. Saves OOF and test predictions via `cv.save_predictions()`
5. Logs results via `ExperimentTracker`

Run many notebooks. Use LLMs to generate diverse model variants. Each saves its `.npy` files independently.

### Days 20 to 28: Stacking

1. Load all Level 2 `.npy` files
2. Train Level 3 stacked models (raw features + L2 OOF as inputs)
3. Run hill climbing to select best model subset
4. Fit meta-learner on selected models
5. Generate final submission

### Days 28 to 30: Submission Tuning

Use existing `util/smart_ensemble.py` for submission-level blending and Kaggle API interaction.

## Integration with Existing Codebase

| Existing component | Relationship |
|---|---|
| `util/smart_ensemble.py` | Handles submission-level blending and Kaggle API. No overlap. |
| `util/analyze_diversity.py` | Works on submission CSVs. The new package works on OOF `.npy` files. Complementary. |
| `aws_training/src/base_trainer.py` | The new `BaseModel` wrappers are simpler (no Optuna built in). `OptunaTuner` is separate. The AWS trainer can be used as a backend for `BaseModel.fit()` when training on SageMaker. |
| `config/*.json` | The new `CompetitionConfig` replaces per-competition JSON configs for model training. The JSON configs remain for submission automation. |
| Per-competition `src/features.py` | Replaced by `kaggle_stack.features` transformers. Existing domain-specific features become custom `FeatureTransformer` subclasses. |

## Technical Constraints

- Python 3.13+, managed with `uv`.
- No new heavy dependencies beyond what's already in `pyproject.toml`. XGBoost, LightGBM, CatBoost, scikit-learn, pandas, numpy, optuna, torch are all present.
- GPU support is optional. All core logic (CV, OOF, stacking, meta-learner) runs on CPU.
- Must work in Jupyter notebooks (the primary model development environment) and as regular Python scripts.
- Prediction files use `.npy` format with naming convention `{oof|pred}_{name}_v{version}.npy`.

## Milestones

### M1: Core Infrastructure

- `CompetitionConfig` dataclass
- `CVManager` with fold management, OOF arrays, nested transforms, prediction I/O
- `FeatureTransformer` base class
- `BaseModel` wrapper interface
- `ExperimentTracker`

**Validation:** Reproduce the PS6E2 XGBoost baseline using the new framework. Same CV score as the existing `PS6E2/src/train_xgb.py`.

### M2: Feature Transformers

- `NestedTargetEncoder` (single columns, bigrams, trigrams, multi-stat)
- `DigitExtractor`
- `SnapTransformer`
- `ArithmeticInteractions`
- `FrequencyEncoder`
- `CategoricalCrossFeatures`
- `BinningTransformer`

**Validation:** Apply snap features and digit extraction to PS6E3 data. Verify that an XGBoost model trained with these features matches the CV scores reported in the 1st place solution for a comparable single model.

### M3: Model Wrappers and Stacking

- `XGBoostModel`, `LightGBMModel`, `CatBoostModel` wrappers
- `RealMLPModel`, `TabMModel` wrappers
- `StackingLevel` orchestrator
- `MetaLearner` (LogReg, Ridge, hill climbing)
- `OptunaTuner`

**Validation:** Build a 2-level stack (Level 2: 3 GBDTs + Level 4: LogReg) on PS6E3 data. Verify the stacked score exceeds the best single model score.

### M4: Polish and Documentation

- CLI helper for competition scaffolding (`kaggle-stack init ps-s6e4 --task classification`)
- Notebook templates (EDA, model training, stacking)
- Integration tests against PS6E2 and PS6E3

## Open Questions

1. **Should Level 2 models share a single feature set or each define their own?** The 1st place solution uses model-specific feature subsets. The 38th place solution uses a shared feature store with per-model selection. The framework should support both patterns.

2. **How to handle CatBoost's native categorical support?** CatBoost does not need manual target encoding. The model wrapper should accept `cat_features` and skip TE for those columns, while other models in the same stacking level apply TE.

3. **Where does Optuna HPO fit in the level hierarchy?** Currently the AWS pipeline does Optuna on a single split before full CV training. The framework should support the same two-phase pattern without mandating it.

4. **Should the framework manage GPU assignment?** The 1st place solution distributes across 4xA100. For now, GPU selection is the user's responsibility via environment variables (`CUDA_VISIBLE_DEVICES`).

5. **Notebook vs script workflow?** Most Kaggle work happens in notebooks. The framework must be import-friendly in notebooks. No CLI-only features for core functionality.
