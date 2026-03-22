"""
TabML - Tabular Machine Learning Package

A comprehensive package for handling tabular data machine learning tasks.
"""

__version__ = "0.5.2"

# Initialize package and download required data
try:
    from ._init_helpers import _init_status
except ImportError:
    # Silent fail if helper not available
    pass

from .data import DataLoader
from .features import FeatureEngineer, FeatureSelector
from .models import (
    ModelTrainer, XGBoostModel, LightGBMModel, CatBoostModel, 
    RandomForestModel, TabNetModel, VotingEnsemble, RidgeModel, LinearModel
)
from .pipeline import TabularPipeline
from .evaluate import CrossValidator
from .visualize import Visualizer
from .validation import DataValidator
from .timeseries import TimeSeriesDataLoader, TimeSeriesFeatureEngineer
from .preprocessing import DataProcessor
from .advanced_features import AdvancedFeatureEngineer
from .ensemble import OOFEnsemble, AutoEnsemble
from .oof_manager import OOFManager
from .eda import EDAAnalyzer

# Import MLflow tracking if available
try:
    from .mlflow_tracker import MLflowTracker, MLflowModelRegistry
    from .training import MLflowCallback
    MLFLOW_IMPORTS = ["MLflowTracker", "MLflowModelRegistry", "MLflowCallback"]
except ImportError:
    MLFLOW_IMPORTS = []

# Import AutoGluon model if available
try:
    from .autogluon_model import AutoGluonModel
    AUTOGLUON_IMPORTS = ["AutoGluonModel"]
except ImportError:
    AUTOGLUON_IMPORTS = []

__all__ = [
    "DataLoader",
    "FeatureEngineer",
    "FeatureSelector",
    "AdvancedFeatureEngineer",
    "ModelTrainer",
    "XGBoostModel",
    "LightGBMModel", 
    "CatBoostModel",
    "RandomForestModel",
    "TabNetModel",
    "VotingEnsemble",
    "RidgeModel",
    "LinearModel",
    "TabularPipeline",
    "CrossValidator",
    "Visualizer",
    "DataValidator",
    "TimeSeriesDataLoader",
    "TimeSeriesFeatureEngineer",
    "DataProcessor",
    "OOFEnsemble",
    "AutoEnsemble",
    "OOFManager",
    "EDAAnalyzer",
] + MLFLOW_IMPORTS + AUTOGLUON_IMPORTS