"""
PS5E8 Competition - Exploratory Data Analysis
This script demonstrates how to use the tabml library to perform EDA.
Plots are saved to the output/eda_plots directory.
"""

from pathlib import Path
from loguru import logger
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from tabml.data import DataLoader
from tabml.visualize import Visualizer
from tabml.features import FeatureSelector, FeatureEngineer

# Define paths
DATA_DIR = Path("../../data/raw/ PS5E8")
OUTPUT_DIR = Path("output")
PLOTS_DIR = OUTPUT_DIR / "eda_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Monkey-patch matplotlib.pyplot.show to save figures ---
plot_counter = 0
def save_and_close_show(*args, **kwargs):
    """Saves the current figure and closes it, preventing pop-ups."""
    global plot_counter
    plot_counter += 1
    # Try to get the title of the plot for a more descriptive filename
    fig = plt.gcf()
    title = fig._suptitle.get_text() if fig._suptitle else (plt.gca().get_title() or f"plot_{plot_counter}")
    filename = "".join(c for c in title if c.isalnum() or c in (' ', '_')).rstrip()
    filename = filename.replace(' ', '_').lower()
    if not filename:
        filename = f"plot_{plot_counter}"

    fig_path = PLOTS_DIR / f"{filename}.png"
    plt.savefig(fig_path, bbox_inches='tight', dpi=100)
    plt.close(fig)
    logger.info(f"Saved plot: {fig_path}")

# Replace the original show function
plt.show = save_and_close_show
# --- End of monkey-patch ---

def main():
    """Main EDA pipeline using tabml."""
    logger.info("=" * 60)
    logger.info("PS5E8 COMPETITION - EXPLORATORY DATA ANALYSIS (using tabml)")
    logger.info("=" * 60)

    # 1. Load data using DataLoader
    logger.info("Step 1: Loading data...")
    loader = DataLoader(data_dir=DATA_DIR)
    try:
        train_df, test_df = loader.load_data(
            train_file="train.csv", 
            test_file="test.csv",
            target_column="y"
        )
    except FileNotFoundError as e:
        logger.error(f"Data files not found: {e}")
        logger.error(f"Please ensure train.csv and test.csv are in the {DATA_DIR} directory.")
        return

    # 2. Get and display basic data info
    logger.info("\nStep 2: Displaying basic data information...")
    data_info = loader.get_data_info()
    logger.info(f"Train shape: {data_info['train_shape']}")
    logger.info(f"Test shape: {data_info['test_shape']}")
    logger.info(f"Target column: {data_info['target_column']}")
    logger.info(f"Numeric features: {data_info['numeric_features']}")
    logger.info(f"Categorical features: {data_info['categorical_features']}")
    
    # 3. Validate data for potential issues
    logger.info("\nStep 3: Validating data...")
    loader.validate_data()

    # 4. Initialize Visualizer
    viz = Visualizer(figsize=(12, 7), style='whitegrid')

    # 5. Create a comprehensive EDA report
    logger.info("\nStep 4: Generating comprehensive EDA report...")
    # The create_eda_report function will generate and save multiple plots
    viz.create_eda_report(train_df, target_column='y')

    # 6. Showcase individual plotting functions
    logger.info("\nStep 5: Showcasing individual visualization functions...")

    # Plot target distribution
    logger.info("Plotting target distribution...")
    viz.plot_target_distribution(train_df['y'], title="Bank Subscription Target Distribution")

    # Plot missing values
    logger.info("Plotting missing values...")
    viz.plot_missing_values(train_df)

    # Plot correlation matrix
    logger.info("Plotting correlation matrix...")
    viz.plot_correlation_matrix(train_df, method='spearman')

    # Plot feature distributions for top numeric features
    numeric_features = data_info.get('numeric_features', [])
    if 'id' in numeric_features:
        numeric_features.remove('id')
    
    logger.info("Plotting numeric feature distributions...")
    viz.plot_feature_distributions(train_df, features=numeric_features[:6], n_cols=3)

    # Plot feature vs target for top categorical features
    categorical_features = data_info.get('categorical_features', [])
    logger.info("Plotting categorical feature vs target...")
    viz.plot_feature_vs_target(train_df, train_df['y'], features=categorical_features[:6])

    # 7. Showcase FeatureSelector and feature importance plotting
    logger.info("\nStep 6: Running FeatureSelector and plotting importance...")
    
    # Separate features and target
    X = train_df.drop(columns=['y', 'id'])
    y = train_df['y']

    # To use feature selector, we need to handle categorical features first
    # We'll use a simple FeatureEngineer for this
    logger.info("Engineering features for feature selection...")
    feature_engineer = FeatureEngineer(categorical_encoding='label')
    X_encoded = feature_engineer.fit_transform(X)

    # Initialize and run feature selector
    selector = FeatureSelector(method='mutual_info', n_features=15)
    selector.fit(X_encoded, y)
    
    # Get and plot feature importance
    importance_df = selector.get_feature_importance()
    logger.info("Top 10 Features by Mutual Information Score:")
    print(importance_df.head(10))

    # Rename 'score' column to 'importance' for compatibility with plot_feature_importance
    importance_df.rename(columns={'score': 'importance'}, inplace=True)
    
    viz.plot_feature_importance(importance_df, top_n=15, title="Top 15 Features by Mutual Information")

    logger.info("\n" + "=" * 60)
    logger.info("EDA using tabml COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"All plots have been saved to the {PLOTS_DIR} directory.")

if __name__ == "__main__":
    main()