"""
PS5E10 - Setup Verification Script
Verify that data and dependencies are properly configured
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

def verify_data():
    """Verify data files exist and are readable."""
    print("="*60)
    print("VERIFYING DATA FILES")
    print("="*60)

    data_dir = Path("../../data/raw/PS5E10")

    required_files = ['train.csv', 'test.csv', 'sample_submission.csv']

    all_good = True
    for file in required_files:
        filepath = data_dir / file
        if filepath.exists():
            import pandas as pd
            df = pd.read_csv(filepath)
            print(f"✓ {file:25s} - {df.shape[0]:,} rows × {df.shape[1]} cols")
        else:
            print(f"✗ {file:25s} - NOT FOUND")
            all_good = False

    return all_good


def verify_tabml():
    """Verify TabML is properly installed."""
    print("\n" + "="*60)
    print("VERIFYING TABML INSTALLATION")
    print("="*60)

    try:
        from tabml import (
            XGBoostModel, LightGBMModel, CatBoostModel,
            RandomForestModel, OOFEnsemble, OOFManager, EDAAnalyzer
        )
        print("✓ TabML core modules")

        import xgboost
        print(f"✓ XGBoost {xgboost.__version__}")

        import lightgbm
        print(f"✓ LightGBM {lightgbm.__version__}")

        import catboost
        print(f"✓ CatBoost {catboost.__version__}")

        import sklearn
        print(f"✓ scikit-learn {sklearn.__version__}")

        import pandas
        print(f"✓ pandas {pandas.__version__}")

        import numpy
        print(f"✓ numpy {numpy.__version__}")

        return True

    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def verify_directories():
    """Create output directories if they don't exist."""
    print("\n" + "="*60)
    print("VERIFYING OUTPUT DIRECTORIES")
    print("="*60)

    directories = [
        Path("output"),
        Path("output/oof_predictions"),
        Path("output/submissions"),
        Path("output/eda_analysis"),
        Path("output/ensemble_analysis")
    ]

    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ {str(dir_path):30s} - ready")

    return True


def show_competition_info():
    """Display competition information."""
    print("\n" + "="*60)
    print("COMPETITION INFORMATION")
    print("="*60)

    import pandas as pd

    train_df = pd.read_csv("../../data/raw/PS5E10/train.csv")

    print(f"\nDataset: Road Accident Risk Prediction")
    print(f"Task: Regression (predict continuous accident_risk [0-1])")
    print(f"Metric: Root Mean Squared Error (RMSE)")
    print(f"\nFeatures:")

    for col in train_df.columns:
        if col not in ['id', 'accident_risk']:
            dtype = train_df[col].dtype
            nunique = train_df[col].nunique()
            print(f"  - {col:25s} ({dtype}, {nunique} unique)")

    print(f"\nTarget: accident_risk")
    print(f"  Range: [{train_df['accident_risk'].min():.4f}, {train_df['accident_risk'].max():.4f}]")
    print(f"  Mean: {train_df['accident_risk'].mean():.4f}")
    print(f"  Std: {train_df['accident_risk'].std():.4f}")


def main():
    """Run all verification checks."""
    print("\n" + "="*70)
    print(" "*15 + "PS5E10 SETUP VERIFICATION")
    print("="*70 + "\n")

    data_ok = verify_data()
    tabml_ok = verify_tabml()
    dirs_ok = verify_directories()

    if data_ok and tabml_ok and dirs_ok:
        show_competition_info()

        print("\n" + "="*60)
        print("✅ ALL CHECKS PASSED!")
        print("="*60)

        print("\nYou're ready to start! Run the following commands:")
        print("\n  1. python 01_eda.py              # Exploratory Data Analysis")
        print("  2. python 02_baseline_models.py  # Train baseline models")
        print("  3. python 03_ensemble.py         # Optimize ensemble")
        print("\n" + "="*60 + "\n")

        return True
    else:
        print("\n" + "="*60)
        print("❌ SETUP INCOMPLETE")
        print("="*60)

        if not data_ok:
            print("\n⚠️  Data files missing. Please check data/raw/PS5E10/")

        if not tabml_ok:
            print("\n⚠️  TabML not installed. Run: pip install -e \".[all]\"")

        print("\n" + "="*60 + "\n")

        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
