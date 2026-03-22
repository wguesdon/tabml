# %% [markdown]
# # PS5E10 EDA - Road Accident Risk Prediction
# 
# Using TabML for comprehensive exploratory data analysis

# %%
"""
PS5E10 Competition - Exploratory Data Analysis (Kaggle Version)
Comprehensive EDA for Road Accident Risk Prediction using TabML
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Install tabml if not already installed (for Kaggle)
import subprocess

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

# Install required packages for Kaggle
try:
    from tabml import EDAAnalyzer
except ImportError:
    print("Installing tabml and dependencies...")
    install_package("git+https://github.com/wguesdon/tabml.git")
    install_package("loguru")
    from tabml import EDAAnalyzer

# Setup paths for Kaggle
DATA_DIR = Path("/kaggle/input/playground-series-s5e10/")
OUTPUT_DIR = Path("/kaggle/working/output/eda_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Competition settings
TARGET_COL = 'accident_risk'
ID_COL = 'id'

print("✅ Setup complete!")

# %% [markdown]
# ## Load Data

# %%
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

print(f"Train: {train_df.shape}, Test: {test_df.shape}")
display(train_df.head())

train_df_analysis = train_df.drop(ID_COL, axis=1)
test_df_analysis = test_df.drop(ID_COL, axis=1)

# %% [markdown]
# ## Generate EDA Report

# %%
eda = EDAAnalyzer(figsize=(14, 8), palette='Set2')

eda.generate_report(
    train_df_analysis,
    target=TARGET_COL,
    output_dir=str(OUTPUT_DIR / "train"),
    max_categories=20
)

print("✅ Report generated!")

# %% [markdown]
# ## Target Distribution

# %%
target_stats = train_df_analysis[TARGET_COL].describe()
print("Accident Risk Statistics:")
display(target_stats)

# Outliers
Q1 = train_df_analysis[TARGET_COL].quantile(0.25)
Q3 = train_df_analysis[TARGET_COL].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = train_df_analysis[(train_df_analysis[TARGET_COL] < lower_bound) |
                             (train_df_analysis[TARGET_COL] > upper_bound)]

print(f"\nOutliers: {len(outliers)} ({len(outliers)/len(train_df_analysis)*100:.2f}%)")

# Risk ranges
print("\nRisk Distribution:")
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
    print(f"  {category}: {count:,} ({pct:.1f}%))")

# %% [markdown]
# ## Adversarial Validation

# %%
# Run adversarial validation to check train/test similarity
from IPython.display import Image
import os

print("Running adversarial validation...")
try:
    adv_save_path = str(OUTPUT_DIR / "adversarial_validation.png")
    print(f"Will save to: {adv_save_path}")

    adv_results = eda.adversarial_validation(
        train_df=train_df,
        test_df=test_df,
        target_col=TARGET_COL,
        n_folds=5,
        sample_size=100000,
        save_path=adv_save_path
    )

    print("\n✅ Adversarial Validation Results:")
    print(f"  AUC Score: {adv_results['auc_score']:.4f}")
    print(f"  Interpretation: {adv_results['interpretation']}")

    if adv_results['feature_importance'] is not None and not adv_results['feature_importance'].empty:
        print("\nTop features causing distribution shift:")
        display(adv_results['feature_importance'].head(10))

    # Display the plot if it exists
    if os.path.exists(adv_save_path):
        print(f"\n✅ Plot saved successfully at: {adv_save_path}")
        display(Image(filename=adv_save_path))
    else:
        print(f"\n⚠️ Plot file not found at: {adv_save_path}")

except Exception as e:
    import traceback
    print(f"\n⚠️ Adversarial validation failed with error:")
    print(f"Error: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    print("\nSkipping adversarial validation...")

# %% [markdown]
# ## Univariate Analysis
#
# Distribution of individual features

# %% [markdown]
# ### Target Variable: Accident Risk

# %%
display(Image(filename='/kaggle/working/output/eda_analysis/train/univariate/accident_risk.png'))

# %% [markdown]
# ### Categorical Features

# %%
display(Image(filename='/kaggle/working/output/eda_analysis/train/univariate/road_type.png'))

# %%
display(Image(filename='/kaggle/working/output/eda_analysis/train/univariate/lighting.png'))

# %%
display(Image(filename='/kaggle/working/output/eda_analysis/train/univariate/weather.png'))

# %%
display(Image(filename='/kaggle/working/output/eda_analysis/train/univariate/time_of_day.png'))

# %% [markdown]
# ### Numerical Features

# %%
display(Image(filename='/kaggle/working/output/eda_analysis/train/univariate/curvature.png'))

# %%
display(Image(filename='/kaggle/working/output/eda_analysis/train/univariate/speed_limit.png'))

# %%
display(Image(filename='/kaggle/working/output/eda_analysis/train/univariate/num_lanes.png'))

# %%
display(Image(filename='/kaggle/working/output/eda_analysis/train/univariate/num_reported_accidents.png'))

# %% [markdown]
# ## Multivariate Analysis
#
# Relationships between features and accident risk

# %% [markdown]
# ### Categorical Features vs Accident Risk

# %%
display(Image(filename='/kaggle/working/output/eda_analysis/train/multivariate/road_type_vs_accident_risk.png'))

# %%
display(Image(filename='/kaggle/working/output/eda_analysis/train/multivariate/lighting_vs_accident_risk.png'))

# %%
display(Image(filename='/kaggle/working/output/eda_analysis/train/multivariate/weather_vs_accident_risk.png'))

# %%
display(Image(filename='/kaggle/working/output/eda_analysis/train/multivariate/time_of_day_vs_accident_risk.png'))

# %% [markdown]
# ### Numerical Features vs Accident Risk

# %%
display(Image(filename='/kaggle/working/output/eda_analysis/train/multivariate/curvature_vs_accident_risk.png'))

# %%
display(Image(filename='/kaggle/working/output/eda_analysis/train/multivariate/speed_limit_vs_accident_risk.png'))

# %%
display(Image(filename='/kaggle/working/output/eda_analysis/train/multivariate/num_lanes_vs_accident_risk.png'))

# %%
display(Image(filename='/kaggle/working/output/eda_analysis/train/multivariate/num_reported_accidents_vs_accident_risk.png'))

# %% [markdown]
# ## Correlation Analysis

# %%
display(Image(filename='/kaggle/working/output/eda_analysis/train/correlation.png'))

# %% [markdown]
# ## EDA Insights
# 
# ### Key Findings:
# - **No missing data**
# - Target mostly in Low-Medium risk (0.2-0.6)
# - **High-risk factors**: Night lighting (+55%), Foggy weather (+24%)
# 
# ### Feature Engineering:
# 1. Interactions: weather × lighting, curvature × speed
# 2. Risk scores: visibility_risk, geometry_risk
# 3. Polynomial features for numerical vars
# 
# ### Strategy:
# - Gradient boosting ensemble (XGB, LGB, Cat)
# - 5-fold CV, RMSE metric
# - Clip predictions to [0, 1]


