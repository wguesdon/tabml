"""Visualization utilities for exploratory data analysis.

This module provides comprehensive visualization tools for understanding
tabular data, including distributions, correlations, missing values,
and feature-target relationships.

Classes:
    Visualizer: Main visualization class for EDA
    
Example:
    Basic visualization workflow::
    
        from tabml.visualize import Visualizer
        
        # Initialize visualizer
        viz = Visualizer(figsize=(12, 8), style='whitegrid')
        
        # Create comprehensive EDA report
        viz.create_eda_report(df, target_column='price')
        
        # Or use individual plots
        viz.plot_missing_values(df)
        viz.plot_correlation_matrix(df)
        viz.plot_feature_distributions(df, features=['age', 'income', 'score'])
        
        # Feature importance from model
        viz.plot_feature_importance(importance_df, top_n=30)
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class Visualizer:
    """Visualization tools for tabular data analysis.
    
    Provides a comprehensive set of visualization methods for exploring
    tabular datasets, understanding relationships, and presenting results.
    All plots use matplotlib and seaborn for professional visualizations.
    
    Attributes:
        figsize: Default figure size for plots
        style: Seaborn style setting
        
    Example:
        >>> # Initialize with custom settings
        >>> viz = Visualizer(figsize=(14, 8), style='darkgrid')
        >>> 
        >>> # Analyze target distribution
        >>> viz.plot_target_distribution(y_train, title="Training Target Distribution")
        >>> 
        >>> # Check missing values
        >>> viz.plot_missing_values(df)
        >>> 
        >>> # Explore correlations
        >>> viz.plot_correlation_matrix(df, method='spearman')
        >>> 
        >>> # Feature analysis
        >>> viz.plot_feature_distributions(df, features=numeric_columns)
        >>> viz.plot_feature_vs_target(df, target, features=top_features)
        >>> 
        >>> # Model interpretation
        >>> viz.plot_feature_importance(importance_df, top_n=25)
        >>> viz.plot_learning_curves(train_scores, val_scores)
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 6), style: str = 'whitegrid'):
        """Initialize visualizer.
        
        Args:
            figsize: Default figure size as (width, height) tuple.
                Common sizes:
                - (10, 6): Standard single plot
                - (12, 8): Larger detailed plots
                - (16, 10): Presentation size
            style: Seaborn style setting. Options:
                - 'whitegrid': Clean with grid (default)
                - 'darkgrid': Dark background with grid
                - 'white': Minimal white background
                - 'dark': Minimal dark background
                - 'ticks': White with ticks
                
        Example:
            >>> # Standard visualizer
            >>> viz = Visualizer()
            >>> 
            >>> # Large plots for presentations
            >>> viz = Visualizer(figsize=(16, 10), style='white')
        """
        self.figsize = figsize
        sns.set_style(style)
        
    def plot_target_distribution(self, y: pd.Series, title: str = "Target Distribution") -> None:
        """Plot target variable distribution.
        
        Automatically detects whether target is categorical or continuous
        and creates appropriate visualization.
        
        Args:
            y: Target variable as pandas Series
            title: Plot title
            
        Example:
            >>> # Classification target
            >>> viz.plot_target_distribution(y_train, "Class Distribution")
            >>> 
            >>> # Regression target
            >>> viz.plot_target_distribution(prices, "Price Distribution")
            
        Note:
            - Categorical: Bar plot if < 20 unique values
            - Continuous: Histogram with 50 bins otherwise
        """
        plt.figure(figsize=self.figsize)
        
        if y.nunique() < 20:
            # Categorical target
            y.value_counts().plot(kind='bar')
            plt.ylabel('Count')
            plt.xlabel('Target Value')
        else:
            # Continuous target
            plt.hist(y, bins=50, edgecolor='black')
            plt.ylabel('Frequency')
            plt.xlabel('Target Value')
            
        plt.title(title)
        plt.tight_layout()
        plt.show()
        
    def plot_missing_values(self, df: pd.DataFrame, title: str = "Missing Values") -> None:
        """Plot missing values heatmap.
        
        Creates a horizontal bar chart showing percentage of missing values
        for each column, sorted by missingness.
        
        Args:
            df: DataFrame to analyze
            title: Plot title
            
        Example:
            >>> viz.plot_missing_values(df)
            >>> # Shows columns with highest missing percentage first
            
        Note:
            - Only shows columns with missing values
            - Includes percentage labels on bars
            - Figure height scales with number of columns
        """
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        
        if len(missing) == 0:
            logger.info("No missing values found")
            return
            
        plt.figure(figsize=(self.figsize[0], max(6, len(missing) * 0.3)))
        missing_pct = (missing / len(df)) * 100
        
        ax = missing_pct.plot(kind='barh')
        plt.xlabel('Percentage Missing')
        plt.title(title)
        
        # Add percentage labels
        for i, (idx, val) in enumerate(missing_pct.items()):
            ax.text(val + 0.5, i, f'{val:.1f}%', va='center')
            
        plt.tight_layout()
        plt.show()
        
    def plot_correlation_matrix(self, df: pd.DataFrame, 
                               method: str = 'pearson',
                               figsize: Optional[Tuple[int, int]] = None) -> None:
        """Plot correlation matrix heatmap.
        
        Creates a triangular heatmap showing correlations between all
        numeric features in the dataset.
        
        Args:
            df: DataFrame with numeric columns
            method: Correlation method. Options:
                - 'pearson': Linear correlation (default)
                - 'spearman': Rank correlation (robust to outliers)
                - 'kendall': Ordinal association
            figsize: Custom figure size. If None, auto-scales based
                on number of features.
                
        Example:
            >>> # Standard Pearson correlation
            >>> viz.plot_correlation_matrix(df)
            >>> 
            >>> # Spearman for non-linear relationships
            >>> viz.plot_correlation_matrix(df, method='spearman')
            >>> 
            >>> # Large matrix with custom size
            >>> viz.plot_correlation_matrix(df, figsize=(20, 20))
            
        Note:
            - Only shows lower triangle to avoid redundancy
            - Annotations shown only if < 20 features
            - Color scale centered at 0 (white)
            - Red: positive correlation, Blue: negative
        """
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            logger.warning("Not enough numeric columns for correlation matrix")
            return
            
        # Calculate correlation
        corr = numeric_df.corr(method=method)
        
        # Use custom figsize or adjust based on number of features
        if figsize is None:
            size = max(10, numeric_df.shape[1] * 0.5)
            figsize = (size, size)
            
        plt.figure(figsize=figsize)
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr), k=1)
        
        # Plot heatmap
        sns.heatmap(corr, mask=mask, annot=True if corr.shape[0] < 20 else False,
                    fmt='.2f', cmap='coolwarm', center=0,
                    square=True, linewidths=0.5)
        
        plt.title(f'Correlation Matrix ({method})')
        plt.tight_layout()
        plt.show()
        
    def plot_feature_distributions(self, df: pd.DataFrame, 
                                  features: Optional[List[str]] = None,
                                  n_cols: int = 3) -> None:
        """Plot distributions for multiple features.
        
        Creates a grid of subplots showing the distribution of each feature.
        Handles both numeric (histogram) and categorical (bar plot) features.
        
        Args:
            df: DataFrame containing features to plot
            features: List of column names to plot. If None, plots
                first 20 columns.
            n_cols: Number of columns in subplot grid
            
        Example:
            >>> # Plot specific features
            >>> viz.plot_feature_distributions(
            ...     df, 
            ...     features=['age', 'income', 'education', 'city'],
            ...     n_cols=2
            ... )
            >>> 
            >>> # Plot all numeric features
            >>> numeric_cols = df.select_dtypes(include=[np.number]).columns
            >>> viz.plot_feature_distributions(df, features=numeric_cols)
            
        Note:
            - Numeric features: 30-bin histogram
            - Categorical features: Bar plot (top 20 values if many)
            - Auto-adjusts grid layout
        """
        if features is None:
            features = df.columns.tolist()[:20]  # Limit to first 20
            
        n_features = len(features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for i, feature in enumerate(features):
            ax = axes[i]
            
            if df[feature].dtype in ['object', 'category']:
                # Categorical feature
                value_counts = df[feature].value_counts()
                if len(value_counts) > 20:
                    value_counts = value_counts.head(20)
                value_counts.plot(kind='bar', ax=ax)
                ax.set_xlabel('')
            else:
                # Numeric feature
                df[feature].hist(bins=30, ax=ax, edgecolor='black')
                
            ax.set_title(feature)
            ax.tick_params(axis='x', rotation=45)
            
        # Remove empty subplots
        for i in range(n_features, len(axes)):
            fig.delaxes(axes[i])
            
        plt.tight_layout()
        plt.show()
        
    def plot_feature_vs_target(self, df: pd.DataFrame, 
                              target: pd.Series,
                              features: Optional[List[str]] = None,
                              n_features: int = 10) -> None:
        """Plot features vs target relationships.
        
        Visualizes how each feature relates to the target variable.
        Automatically chooses appropriate plot type based on feature
        and target types.
        
        Args:
            df: DataFrame with features (should not include target)
            target: Target variable as Series
            features: List of features to plot. If None, selects
                first n_features numeric columns.
            n_features: Number of features to plot if features=None
            
        Plot types:
            - Numeric feature + numeric target: Scatter plot
            - Numeric feature + categorical target: Box plot by class
            - Categorical feature + numeric target: Box plot by category
            - Categorical feature + categorical target: Stacked bar chart
            
        Example:
            >>> # Analyze top features
            >>> top_features = importance_df.head(10)['feature'].tolist()
            >>> viz.plot_feature_vs_target(
            ...     df[top_features], 
            ...     y_train,
            ...     features=top_features
            ... )
            
        Note:
            Creates individual plots for each feature to allow
            detailed analysis of relationships.
        """
        if features is None:
            # Select top numeric features
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            features = numeric_cols[:n_features]
            
        is_classification = target.nunique() < 20
        
        for feature in features:
            plt.figure(figsize=self.figsize)
            
            if df[feature].dtype in ['object', 'category']:
                # Categorical feature
                if is_classification:
                    # Stacked bar chart
                    pd.crosstab(df[feature], target).plot(kind='bar', stacked=True)
                else:
                    # Box plot
                    df_temp = pd.DataFrame({feature: df[feature], 'target': target})
                    df_temp.boxplot(column='target', by=feature)
                    plt.suptitle('')
            else:
                # Numeric feature
                if is_classification:
                    # Box plot by class
                    df_temp = pd.DataFrame({feature: df[feature], 'target': target})
                    df_temp.boxplot(column=feature, by='target')
                    plt.suptitle('')
                else:
                    # Scatter plot
                    plt.scatter(df[feature], target, alpha=0.5)
                    plt.xlabel(feature)
                    plt.ylabel('Target')
                    
            plt.title(f'{feature} vs Target')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
    def plot_feature_importance(self, importance_df: pd.DataFrame, 
                               top_n: int = 20,
                               title: str = "Feature Importance") -> None:
        """Plot feature importance.
        
        Creates a horizontal bar chart of feature importances,
        typically from a trained model.
        
        Args:
            importance_df: DataFrame with columns:
                - 'feature': Feature names
                - 'importance': Importance scores
                Should be pre-sorted by importance.
            top_n: Number of top features to display
            title: Plot title
            
        Example:
            >>> # From tree-based model
            >>> importance_df = pd.DataFrame({
            ...     'feature': feature_names,
            ...     'importance': model.feature_importances_
            ... }).sort_values('importance', ascending=False)
            >>> 
            >>> viz.plot_feature_importance(importance_df, top_n=30)
            
        Note:
            - Most important features appear at top
            - Figure height scales with number of features
            - Shows exact importance values on bars
        """
        # Get top N features
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(self.figsize[0], max(6, top_n * 0.3)))
        
        # Create horizontal bar plot
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        
        plt.xlabel('Importance')
        plt.title(title)
        plt.gca().invert_yaxis()  # Highest importance at top
        plt.tight_layout()
        plt.show()
        
    def plot_learning_curves(self, train_scores: List[float], 
                           val_scores: List[float],
                           title: str = "Learning Curves") -> None:
        """Plot training and validation learning curves.
        
        Visualizes model performance over training epochs or iterations
        to diagnose overfitting/underfitting.
        
        Args:
            train_scores: List of training scores by epoch
            val_scores: List of validation scores by epoch
            title: Plot title
            
        Example:
            >>> # From model training history
            >>> train_scores = [0.7, 0.8, 0.85, 0.88, 0.9]
            >>> val_scores = [0.68, 0.75, 0.78, 0.79, 0.78]
            >>> viz.plot_learning_curves(train_scores, val_scores)
            
        Interpretation:
            - Converging lines: Good fit
            - Diverging lines: Overfitting
            - Both lines low: Underfitting
            - Gap between lines: Generalization gap
        """
        plt.figure(figsize=self.figsize)
        
        epochs = range(1, len(train_scores) + 1)
        plt.plot(epochs, train_scores, 'b-', label='Training')
        plt.plot(epochs, val_scores, 'r-', label='Validation')
        
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    def create_eda_report(self, df: pd.DataFrame, 
                         target_column: Optional[str] = None,
                         save_path: Optional[str] = None) -> None:
        """Create comprehensive EDA report.
        
        Generates a complete exploratory data analysis including summary
        statistics and multiple visualizations.
        
        Args:
            df: DataFrame to analyze
            target_column: Name of target column (if present in df).
                If provided, includes target analysis and feature-target
                relationships.
            save_path: Path to save plots (not implemented in current version)
            
        Report includes:
            1. Dataset overview (shape, memory usage)
            2. Column type distribution
            3. Missing value analysis
            4. Target distribution (if target_column provided)
            5. Missing values visualization
            6. Correlation matrix
            7. Feature distributions
            8. Feature vs target relationships (if target_column provided)
            
        Example:
            >>> # Full EDA with target
            >>> viz.create_eda_report(train_df, target_column='price')
            >>> 
            >>> # EDA without target (unsupervised)
            >>> viz.create_eda_report(df)
            
        Note:
            - Prints summary statistics to console
            - Displays all plots inline
            - Limits some visualizations to prevent overwhelming output
        """
        logger.info("Creating EDA report...")
        
        # Basic statistics
        print("=== Dataset Overview ===")
        print(f"Shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
        print("\n=== Column Types ===")
        print(df.dtypes.value_counts())
        
        # Missing values
        print("\n=== Missing Values ===")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0].sort_values(ascending=False))
        else:
            print("No missing values")
            
        # Plot visualizations
        if target_column and target_column in df.columns:
            target = df[target_column]
            df_features = df.drop(columns=[target_column])
            
            print(f"\n=== Target Analysis ===")
            print(f"Type: {'Classification' if target.nunique() < 20 else 'Regression'}")
            print(f"Unique values: {target.nunique()}")
            
            self.plot_target_distribution(target)
        else:
            df_features = df
            
        self.plot_missing_values(df)
        self.plot_correlation_matrix(df_features)
        self.plot_feature_distributions(df_features, n_cols=3)
        
        if target_column:
            self.plot_feature_vs_target(df_features, target, n_features=5)
            
        logger.info("EDA report completed")