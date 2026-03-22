"""Exploratory Data Analysis (EDA) tools for TabML.

This module provides comprehensive visualization tools for univariate and multivariate
analysis of tabular data, with automatic plot type selection based on data types.

Classes:
    EDAAnalyzer: Main class for exploratory data analysis and visualization

Example:
    Basic usage::
    
        from tabml.eda import EDAAnalyzer
        
        # Initialize analyzer
        eda = EDAAnalyzer(figsize=(10, 6), style='seaborn')
        
        # Univariate analysis
        eda.plot_univariate(df, max_categories=20)
        
        # Multivariate analysis with target
        eda.plot_multivariate(df, target='price', max_categories=20)
        
        # Generate complete EDA report
        eda.generate_report(df, target='price', output_dir='eda_output')
"""

from typing import Optional, Union, List, Tuple, Dict, Any
import pandas as pd
import numpy as np
import matplotlib
from itertools import combinations
# Only set backend if not already set (allows flexibility)
if matplotlib.get_backend() == 'TkAgg':
    matplotlib.use('Agg')  # Use non-interactive backend to prevent hanging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from loguru import logger
from scipy import stats
from sklearn.preprocessing import LabelEncoder

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
# Suppress the specific warning about converting strings to floats
import matplotlib as mpl
mpl._log.setLevel('ERROR')


class EDAAnalyzer:
    """Exploratory Data Analysis tool for comprehensive data visualization.
    
    Provides automated univariate and multivariate analysis with intelligent
    plot type selection based on variable types and target relationships.
    
    Attributes:
        figsize: Default figure size for plots
        style: Matplotlib/seaborn style
        palette: Color palette for categorical plots
        n_cols: Number of columns for subplot grids
    """
    
    def __init__(self, 
                 figsize: Tuple[int, int] = (12, 8),
                 style: str = 'seaborn-v0_8-darkgrid',
                 palette: Union[str, List[str]] = 'husl',
                 n_cols: int = 3):
        """Initialize the EDA Analyzer.
        
        Args:
            figsize: Default figure size (width, height)
            style: Matplotlib style to use
            palette: Seaborn color palette name or list of colors.
                    Common palettes: 'husl', 'viridis', 'plasma', 'inferno',
                    'magma', 'cividis', 'Set2', 'Set1', 'Dark2', 'Paired',
                    'Accent', 'Pastel1', 'coolwarm', 'RdBu', 'RdYlBu'
            n_cols: Number of columns for subplot grids
        """
        self.figsize = figsize
        self.palette = palette
        self.n_cols = n_cols
        
        # Set style - handle different style names
        try:
            if 'seaborn' in style:
                # Try without version suffix first
                try:
                    plt.style.use('seaborn-darkgrid')
                except:
                    try:
                        plt.style.use('seaborn')
                    except:
                        plt.style.use('default')
            else:
                plt.style.use(style)
        except:
            plt.style.use('default')
            logger.debug(f"Could not set style '{style}', using default")
        
        # Set seaborn defaults
        self._apply_palette(palette)
    
    def _apply_palette(self, palette: Union[str, List[str], None] = None):
        """Apply color palette to plots.
        
        Args:
            palette: Color palette to apply. If None, uses instance default.
        """
        if palette is None:
            palette = self.palette
        
        try:
            if isinstance(palette, str):
                # Try as matplotlib colormap first
                if palette in plt.colormaps():
                    colors = plt.cm.get_cmap(palette).colors if hasattr(plt.cm.get_cmap(palette), 'colors') else None
                    if colors:
                        sns.set_palette(colors)
                    else:
                        # Use as seaborn palette
                        sns.set_palette(palette)
                else:
                    # Try as seaborn palette
                    sns.set_palette(palette)
            elif isinstance(palette, list):
                sns.set_palette(palette)
        except Exception as e:
            logger.debug(f"Could not set palette '{palette}': {e}, using default")
    
    def set_palette(self, palette: Union[str, List[str]]):
        """Change the color palette for subsequent plots.
        
        Args:
            palette: Name of color palette or list of colors.
                    Common options:
                    - Sequential: 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
                    - Categorical: 'Set1', 'Set2', 'Set3', 'Dark2', 'Paired', 'Accent'
                    - Diverging: 'coolwarm', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral'
                    - Perceptual: 'husl', 'hls', 'cubehelix'
        
        Example:
            eda.set_palette('viridis')  # Use viridis colormap
            eda.set_palette(['#1f77b4', '#ff7f0e', '#2ca02c'])  # Custom colors
        """
        self.palette = palette
        self._apply_palette(palette)
        logger.info(f"Palette changed to: {palette}")
        
    def _identify_column_types(self, df: pd.DataFrame, 
                              threshold_unique: int = 20) -> Dict[str, List[str]]:
        """Identify numerical and categorical columns.
        
        Args:
            df: Input DataFrame
            threshold_unique: Threshold for treating numeric as categorical
            
        Returns:
            Dictionary with 'numerical' and 'categorical' column lists
        """
        numerical_cols = []
        categorical_cols = []
        
        for col in df.columns:
            if df[col].dtype in ['object', 'category', 'bool']:
                categorical_cols.append(col)
            elif df[col].nunique() < threshold_unique:
                # Treat low-cardinality numeric as categorical
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
                
        return {
            'numerical': numerical_cols,
            'categorical': categorical_cols
        }
    
    def plot_univariate(self, 
                       df: pd.DataFrame,
                       columns: Optional[List[str]] = None,
                       max_categories: int = 20,
                       save_dir: Optional[str] = None,
                       show: bool = False,
                       max_cols: int = 50,
                       individual_plots: bool = True,
                       palette: Optional[Union[str, List[str]]] = None) -> None:
        """Create univariate plots for all specified columns.
        
        Creates density plots for continuous variables and bar plots for
        categorical variables (showing only top categories if needed).
        
        Args:
            df: Input DataFrame
            columns: Columns to plot (None for all)
            max_categories: Maximum categories to show in bar plots
            save_dir: Directory to save individual plots (if individual_plots=True)
            show: Whether to display the plot interactively (default False)
            max_cols: Maximum number of columns to plot (default 50)
            individual_plots: If True, save each plot separately; if False, create grid
            palette: Optional color palette override for this plot
        """
        # Apply palette if provided
        if palette:
            self._apply_palette(palette)
        
        if columns is None:
            columns = df.columns.tolist()
        
        # Limit number of columns to plot
        if len(columns) > max_cols:
            logger.warning(f"Too many columns ({len(columns)}). Plotting first {max_cols} only.")
            columns = columns[:max_cols]
        
        # Identify column types
        col_types = self._identify_column_types(df)
        
        if len(columns) == 0:
            logger.warning("No columns to plot")
            return
        
        if individual_plots and save_dir:
            # Create individual plots in subdirectory
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            for col in columns:
                try:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    if col in col_types['numerical']:
                        self._plot_univariate_numerical(df[col], ax, col)
                    else:
                        self._plot_univariate_categorical(df[col], ax, col, max_categories)
                    
                    plt.tight_layout()
                    
                    # Save with sanitized filename
                    safe_col_name = col.replace('/', '_').replace('\\', '_').replace(' ', '_')
                    plot_path = save_path / f"{safe_col_name}.png"
                    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    
                except Exception as e:
                    logger.warning(f"Failed to plot column '{col}': {e}")
                    plt.close('all')  # Clean up any open figures
                    continue
            
            logger.info(f"Saved {len(columns)} univariate plots to {save_dir}")
        else:
            # Original grid layout behavior
            n_plots = len(columns)
            n_rows = max(1, (n_plots + self.n_cols - 1) // self.n_cols)
            
            try:
                fig, axes = plt.subplots(n_rows, self.n_cols, 
                                         figsize=(self.figsize[0], self.figsize[1] * n_rows / 3))
                if n_plots == 1:
                    axes = [axes]
                elif n_rows == 1 and self.n_cols == 1:
                    axes = [axes]
                else:
                    axes = axes.flatten()
            except Exception as e:
                logger.error(f"Failed to create subplots: {e}")
                return
            
            # Create plots
            for idx, col in enumerate(columns):
                try:
                    ax = axes[idx] if idx < len(axes) else axes[-1]
                    
                    if col in col_types['numerical']:
                        self._plot_univariate_numerical(df[col], ax, col)
                    else:
                        self._plot_univariate_categorical(df[col], ax, col, max_categories)
                except Exception as e:
                    logger.warning(f"Failed to plot column '{col}': {e}")
                    continue
            
            # Remove empty subplots
            for idx in range(n_plots, len(axes)):
                fig.delaxes(axes[idx])
            
            plt.suptitle('Univariate Analysis', fontsize=16, y=1.02)
            plt.tight_layout()
            
            if save_dir:
                save_path = Path(save_dir)
                save_path.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path / 'univariate_grid.png', dpi=100, bbox_inches='tight')
                logger.info(f"Univariate grid plot saved to {save_path / 'univariate_grid.png'}")
            
            if show:
                plt.show()
            else:
                plt.close(fig)
    
    def _plot_univariate_numerical(self, series: pd.Series, ax: plt.Axes, title: str, 
                                  max_samples: int = 10000) -> None:
        """Plot density curve for numerical variable.
        
        Args:
            series: Data series to plot
            ax: Matplotlib axis
            title: Plot title
            max_samples: Maximum samples for KDE calculation
        """
        # Remove NaN values
        data = series.dropna()
        
        if len(data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return
        
        # Sample if too large for KDE
        if len(data) > max_samples:
            data_sample = data.sample(n=max_samples, random_state=42)
        else:
            data_sample = data
        
        # Create density plot
        try:
            data_sample.plot.kde(ax=ax, color='steelblue', linewidth=2)
        except Exception as e:
            # Fallback to histogram if KDE fails
            logger.debug(f"KDE failed for {title}: {e}, using histogram")
            data.plot.hist(ax=ax, bins=50, alpha=0.7, color='steelblue')
        
        # Add histogram in background
        ax2 = ax.twinx()
        data.plot.hist(ax=ax2, bins=30, alpha=0.3, color='gray', edgecolor='none')
        ax2.set_ylabel('')
        ax2.set_yticks([])
        
        # Add statistics
        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', alpha=0.7, label=f'Median: {median_val:.2f}')
        
        ax.set_title(f'{title}\n(n={len(data):,}, missing={series.isna().sum():,})')
        ax.set_xlabel('')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_univariate_categorical(self, series: pd.Series, ax: plt.Axes, 
                                    title: str, max_categories: int) -> None:
        """Plot bar chart for categorical variable.
        
        Args:
            series: Data series to plot
            ax: Matplotlib axis
            title: Plot title
            max_categories: Maximum number of categories to show
        """
        # Get value counts
        value_counts = series.value_counts()
        
        # Limit to top categories if needed
        if len(value_counts) > max_categories:
            top_counts = value_counts.head(max_categories)
            other_count = value_counts[max_categories:].sum()
            if other_count > 0:
                top_counts['Other'] = other_count
            value_counts = top_counts
        
        # Create bar plot
        bars = ax.bar(range(len(value_counts)), value_counts.values, color='steelblue', alpha=0.8)
        
        # Add value labels on bars
        for bar, val in zip(bars, value_counts.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:,}', ha='center', va='bottom', fontsize=8)
        
        # Set labels
        ax.set_xticks(range(len(value_counts)))
        # Suppress matplotlib warnings about text conversion
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Limit label length and number of labels shown
            labels = [str(x)[:10] for x in value_counts.index]
            if len(labels) > 10:
                # Only show every nth label if too many
                step = len(labels) // 10
                for i in range(len(labels)):
                    if i % step != 0:
                        labels[i] = ''
            ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_title(f'{title}\n(n={series.notna().sum():,}, unique={series.nunique()}, missing={series.isna().sum():,})')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3, axis='y')
    
    def plot_multivariate(self,
                         df: pd.DataFrame,
                         target: str,
                         features: Optional[List[str]] = None,
                         max_categories: int = 20,
                         save_dir: Optional[str] = None,
                         show: bool = False,
                         max_features: int = 50,
                         individual_plots: bool = True,
                         palette: Optional[Union[str, List[str]]] = None) -> None:
        """Create multivariate plots exploring relationships with target.
        
        Plot types are automatically selected based on feature and target types:
        - Continuous target + Continuous feature: Scatter plot with correlation
        - Continuous target + Categorical feature: Box plot
        - Categorical target + Continuous feature: Violin plot or density by class
        - Categorical target + Categorical feature: Stacked bar chart
        
        Args:
            df: Input DataFrame
            target: Target column name
            features: Features to plot (None for all except target)
            max_categories: Maximum categories to show
            save_dir: Directory to save individual plots (if individual_plots=True)
            show: Whether to display the plot interactively (default False)
            max_features: Maximum number of features to plot (default 50)
            individual_plots: If True, save each plot separately; if False, create grid
            palette: Optional color palette override for this plot
        """
        # Apply palette if provided
        if palette:
            self._apply_palette(palette)
        
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame")
        
        if features is None:
            features = [col for col in df.columns if col != target]
        
        # Limit number of features
        if len(features) > max_features:
            logger.warning(f"Too many features ({len(features)}). Plotting first {max_features} only.")
            features = features[:max_features]
        
        # Identify column types
        col_types = self._identify_column_types(df)
        target_is_categorical = target in col_types['categorical']
        
        if individual_plots and save_dir:
            # Create individual plots in subdirectory
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            for feature in features:
                try:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    if feature in col_types['numerical']:
                        if target_is_categorical:
                            self._plot_numerical_vs_categorical(df, feature, target, ax, max_categories)
                        else:
                            self._plot_numerical_vs_numerical(df, feature, target, ax)
                    else:
                        if target_is_categorical:
                            self._plot_categorical_vs_categorical(df, feature, target, ax, max_categories)
                        else:
                            self._plot_categorical_vs_numerical(df, feature, target, ax, max_categories)
                    
                    plt.tight_layout()
                    
                    # Save with sanitized filename
                    safe_feat_name = feature.replace('/', '_').replace('\\', '_').replace(' ', '_')
                    plot_path = save_path / f"{safe_feat_name}_vs_{target}.png"
                    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    
                except Exception as e:
                    logger.warning(f"Failed to plot {feature} vs {target}: {e}")
                    plt.close('all')  # Clean up any open figures
                    continue
            
            logger.info(f"Saved {len(features)} multivariate plots to {save_dir}")
        else:
            # Original grid layout behavior
            n_plots = len(features)
            n_rows = (n_plots + self.n_cols - 1) // self.n_cols
            
            fig, axes = plt.subplots(n_rows, self.n_cols,
                                     figsize=(self.figsize[0], self.figsize[1] * n_rows / 3))
            if n_plots == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            # Create plots
            for idx, feature in enumerate(features):
                ax = axes[idx] if n_plots > 1 else axes[0]
                
                if feature in col_types['numerical']:
                    if target_is_categorical:
                        self._plot_numerical_vs_categorical(df, feature, target, ax, max_categories)
                    else:
                        self._plot_numerical_vs_numerical(df, feature, target, ax)
                else:
                    if target_is_categorical:
                        self._plot_categorical_vs_categorical(df, feature, target, ax, max_categories)
                    else:
                        self._plot_categorical_vs_numerical(df, feature, target, ax, max_categories)
            
            # Remove empty subplots
            for idx in range(n_plots, len(axes)):
                fig.delaxes(axes[idx])
            
            plt.suptitle(f'Multivariate Analysis: Features vs {target}', fontsize=16, y=1.02)
            plt.tight_layout()
            
            if save_dir:
                save_path = Path(save_dir)
                save_path.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path / 'multivariate_grid.png', dpi=100, bbox_inches='tight')
                logger.info(f"Multivariate grid plot saved to {save_path / 'multivariate_grid.png'}")
            
            if show:
                plt.show()
            else:
                plt.close(fig)
    
    def _plot_numerical_vs_numerical(self, df: pd.DataFrame, feature: str, 
                                    target: str, ax: plt.Axes) -> None:
        """Scatter plot for continuous vs continuous variables.
        
        Args:
            df: Input DataFrame
            feature: Feature column name
            target: Target column name
            ax: Matplotlib axis
        """
        # Remove rows with NaN in either column
        data = df[[feature, target]].dropna()
        
        if len(data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{feature} vs {target}')
            return
        
        # Create scatter plot
        ax.scatter(data[feature], data[target], alpha=0.5, s=20)
        
        # Add regression line
        try:
            z = np.polyfit(data[feature], data[target], 1)
            p = np.poly1d(z)
            x_line = np.linspace(data[feature].min(), data[feature].max(), 100)
            ax.plot(x_line, p(x_line), 'r-', alpha=0.8, linewidth=2)
            
            # Calculate correlation
            corr = data[feature].corr(data[target])
            ax.text(0.05, 0.95, f'Corr: {corr:.3f}', transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        except:
            pass
        
        ax.set_xlabel(feature)
        ax.set_ylabel(target)
        ax.set_title(f'{feature} vs {target}')
        ax.grid(True, alpha=0.3)
    
    def _plot_numerical_vs_categorical(self, df: pd.DataFrame, feature: str,
                                      target: str, ax: plt.Axes, max_categories: int) -> None:
        """Violin plot for continuous feature vs categorical target.
        
        Args:
            df: Input DataFrame
            feature: Feature column name
            target: Target column name
            ax: Matplotlib axis
            max_categories: Maximum categories to show
        """
        # Remove rows with NaN
        data = df[[feature, target]].dropna()
        
        if len(data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{feature} by {target}')
            return
        
        # Limit categories if needed
        target_counts = data[target].value_counts()
        if len(target_counts) > max_categories:
            top_categories = target_counts.head(max_categories).index
            data = data[data[target].isin(top_categories)]
        
        # Create violin plot
        try:
            sns.violinplot(data=data, x=target, y=feature, ax=ax, inner='box')
        except:
            # Fallback to box plot if violin fails
            sns.boxplot(data=data, x=target, y=feature, ax=ax)
        
        # Rotate x labels if many categories
        if data[target].nunique() > 5:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                labels = [str(l.get_text())[:10] for l in ax.get_xticklabels()]
                ax.set_xticklabels(labels, rotation=45, ha='right')
        
        ax.set_title(f'{feature} by {target}')
        ax.grid(True, alpha=0.3, axis='y')
    
    def _compute_statistical_test(self,
                                  data: pd.DataFrame,
                                  feature: str,
                                  target: str,
                                  test_type: str = 'auto') -> Dict[str, Any]:
        """Compute appropriate statistical test for categorical vs numerical comparison.

        Args:
            data: DataFrame with feature and target columns.
            feature: Categorical feature column name.
            target: Numerical target column name.
            test_type: Type of test ('auto', 'anova', 'kruskal', 't-test', 'mann-whitney').

        Returns:
            Dictionary with test name, statistic, p-value, and interpretation.
        """
        from scipy.stats import f_oneway, kruskal, ttest_ind, mannwhitneyu, levene

        # Get groups
        groups = data.groupby(feature)[target].apply(list)
        n_groups = len(groups)

        if n_groups < 2:
            return {
                'test': 'none',
                'statistic': None,
                'p_value': None,
                'interpretation': 'Need at least 2 groups'
            }

        # Auto-detect test type
        if test_type == 'auto':
            if n_groups == 2:
                # Check normality assumption (simplified)
                group_sizes = [len(g) for g in groups]
                if all(s >= 30 for s in group_sizes):
                    # Large samples: use t-test
                    test_type = 't-test'
                else:
                    # Small samples: use Mann-Whitney
                    test_type = 'mann-whitney'
            else:
                # Multiple groups: check variance homogeneity
                try:
                    _, p_levene = levene(*groups)
                    if p_levene > 0.05:
                        test_type = 'anova'  # Equal variances
                    else:
                        test_type = 'kruskal'  # Unequal variances
                except:
                    test_type = 'kruskal'  # Default to non-parametric

        # Perform test
        try:
            if test_type == 't-test' and n_groups == 2:
                stat, p_value = ttest_ind(*groups, equal_var=True)
                test_name = "t-test"
            elif test_type == 'mann-whitney' and n_groups == 2:
                stat, p_value = mannwhitneyu(*groups, alternative='two-sided')
                test_name = "Mann-Whitney U"
            elif test_type == 'anova':
                stat, p_value = f_oneway(*groups)
                test_name = "ANOVA"
            elif test_type == 'kruskal':
                stat, p_value = kruskal(*groups)
                test_name = "Kruskal-Wallis"
            else:
                return {
                    'test': 'invalid',
                    'statistic': None,
                    'p_value': None,
                    'interpretation': f'Invalid test type: {test_type}'
                }

            # Interpret result
            if p_value < 0.001:
                interpretation = "*** (p<0.001)"
                significance = "highly significant"
            elif p_value < 0.01:
                interpretation = "** (p<0.01)"
                significance = "very significant"
            elif p_value < 0.05:
                interpretation = "* (p<0.05)"
                significance = "significant"
            else:
                interpretation = "ns (p≥0.05)"
                significance = "not significant"

            return {
                'test': test_name,
                'statistic': float(stat),
                'p_value': float(p_value),
                'interpretation': interpretation,
                'significance': significance
            }

        except Exception as e:
            logger.debug(f"Statistical test failed: {e}")
            return {
                'test': 'failed',
                'statistic': None,
                'p_value': None,
                'interpretation': str(e)
            }

    def _add_stat_annotation(self,
                            ax: plt.Axes,
                            data: pd.DataFrame,
                            feature: str,
                            target: str,
                            test_result: Dict[str, Any],
                            y_offset: float = 0.05) -> None:
        """Add statistical annotation to plot.

        Args:
            ax: Matplotlib axis.
            data: DataFrame with data.
            feature: Feature column name.
            target: Target column name.
            test_result: Result from _compute_statistical_test.
            y_offset: Vertical offset for annotation as fraction of y-range.
        """
        if test_result['p_value'] is None:
            return

        # Get y position for annotation
        y_max = data[target].max()
        y_min = data[target].min()
        y_range = y_max - y_min
        y_pos = y_max + (y_range * y_offset)

        # Create annotation text
        annotation = (f"{test_result['test']}: p={test_result['p_value']:.4f} "
                     f"{test_result['interpretation']}")

        # Add to plot
        ax.text(0.5, 0.98, annotation,
               transform=ax.transAxes,
               ha='center', va='top',
               fontsize=9,
               bbox=dict(boxstyle='round,pad=0.5',
                        facecolor='white' if test_result['p_value'] >= 0.05 else 'yellow',
                        alpha=0.8,
                        edgecolor='gray'))

    def _add_pairwise_annotations(self,
                                  ax: plt.Axes,
                                  data: pd.DataFrame,
                                  feature: str,
                                  target: str,
                                  max_comparisons: int = 10) -> None:
        """Add pairwise comparison annotations (for post-hoc tests).

        Args:
            ax: Matplotlib axis.
            data: DataFrame with data.
            feature: Categorical feature column.
            target: Numerical target column.
            max_comparisons: Maximum number of pairwise comparisons to show.
        """
        from scipy.stats import mannwhitneyu

        categories = sorted(data[feature].unique())
        n_cats = len(categories)

        if n_cats > 6:
            # Too many categories for pairwise comparisons
            return

        # Get all pairs
        pairs = list(combinations(range(n_cats), 2))

        if len(pairs) > max_comparisons:
            # Limit number of comparisons
            return

        # Perform pairwise tests
        y_max = data[target].max()
        y_min = data[target].min()
        y_range = y_max - y_min

        significant_pairs = []
        for i, (idx1, idx2) in enumerate(pairs):
            cat1, cat2 = categories[idx1], categories[idx2]
            group1 = data[data[feature] == cat1][target].values
            group2 = data[data[feature] == cat2][target].values

            try:
                _, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
                if p_val < 0.05:
                    significant_pairs.append((idx1, idx2, p_val))
            except:
                continue

        # Draw significance bars
        for i, (idx1, idx2, p_val) in enumerate(significant_pairs[:5]):  # Limit to 5
            y_level = y_max + (y_range * (0.1 + i * 0.08))

            # Draw bar
            ax.plot([idx1, idx1, idx2, idx2],
                   [y_level - y_range * 0.02, y_level, y_level, y_level - y_range * 0.02],
                   'k-', linewidth=1.5)

            # Add significance stars
            if p_val < 0.001:
                stars = '***'
            elif p_val < 0.01:
                stars = '**'
            else:
                stars = '*'

            ax.text((idx1 + idx2) / 2, y_level + y_range * 0.01, stars,
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    def _plot_categorical_vs_numerical(self, df: pd.DataFrame, feature: str,
                                      target: str, ax: plt.Axes, max_categories: int,
                                      add_stats: bool = True,
                                      add_pairwise: bool = False) -> None:
        """Box plot for categorical feature vs continuous target with statistical annotations.

        Args:
            df: Input DataFrame
            feature: Feature column name
            target: Target column name
            ax: Matplotlib axis
            max_categories: Maximum categories to show
            add_stats: Whether to add overall statistical test annotation
            add_pairwise: Whether to add pairwise comparison annotations
        """
        # Remove rows with NaN
        data = df[[feature, target]].dropna()

        if len(data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{target} by {feature}')
            return

        # Limit categories if needed
        feature_counts = data[feature].value_counts()
        if len(feature_counts) > max_categories:
            top_categories = feature_counts.head(max_categories).index
            data = data[data[feature].isin(top_categories)]

        # Create box plot
        sns.boxplot(data=data, x=feature, y=target, ax=ax)

        # Add mean markers
        means = data.groupby(feature)[target].mean()
        positions = range(len(means))
        ax.scatter(positions, means.values, color='red', s=50, zorder=5, label='Mean')

        # Add statistical test annotation
        if add_stats and data[feature].nunique() >= 2:
            test_result = self._compute_statistical_test(data, feature, target)
            self._add_stat_annotation(ax, data, feature, target, test_result)

        # Add pairwise comparisons if requested
        if add_pairwise and 2 <= data[feature].nunique() <= 6:
            self._add_pairwise_annotations(ax, data, feature, target)

        # Rotate x labels if many categories
        if data[feature].nunique() > 5:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                labels = [str(l.get_text())[:10] for l in ax.get_xticklabels()]
                ax.set_xticklabels(labels, rotation=45, ha='right')

        ax.set_title(f'{target} by {feature}')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_categorical_vs_categorical(self, df: pd.DataFrame, feature: str,
                                        target: str, ax: plt.Axes, max_categories: int) -> None:
        """Stacked bar chart for categorical vs categorical variables.
        
        Args:
            df: Input DataFrame
            feature: Feature column name
            target: Target column name
            ax: Matplotlib axis
            max_categories: Maximum categories to show
        """
        # Remove rows with NaN
        data = df[[feature, target]].dropna()
        
        if len(data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{feature} vs {target}')
            return
        
        # Limit feature categories if needed
        feature_counts = data[feature].value_counts()
        if len(feature_counts) > max_categories:
            top_categories = feature_counts.head(max_categories).index
            data = data[data[feature].isin(top_categories)]
        
        # Create crosstab
        crosstab = pd.crosstab(data[feature], data[target], normalize='index') * 100
        
        # Create stacked bar chart
        crosstab.plot(kind='bar', stacked=True, ax=ax, legend=True)
        
        # Rotate x labels if many categories
        if len(crosstab) > 5:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                labels = [str(l.get_text())[:10] for l in ax.get_xticklabels()]
                ax.set_xticklabels(labels, rotation=45, ha='right')
        
        ax.set_title(f'{feature} vs {target} (% distribution)')
        ax.set_ylabel('Percentage')
        ax.legend(title=target, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    def plot_correlation_matrix(self,
                               df: pd.DataFrame,
                               method: str = 'pearson',
                               save_path: Optional[str] = None,
                               show: bool = False,
                               cmap: Optional[str] = None) -> None:
        """Plot correlation matrix heatmap.
        
        Args:
            df: Input DataFrame (numerical columns only)
            method: Correlation method ('pearson', 'spearman', 'kendall')
            save_path: Path to save the figure
            show: Whether to display the plot interactively (default False)
            cmap: Optional colormap for the heatmap (default 'coolwarm')
        """
        # Select numerical columns
        col_types = self._identify_column_types(df)
        numerical_df = df[col_types['numerical']]
        
        if len(numerical_df.columns) < 2:
            logger.warning("Not enough numerical columns for correlation matrix")
            return
        
        # Calculate correlation matrix
        corr_matrix = numerical_df.corr(method=method)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create heatmap
        colormap = cmap if cmap else 'coolwarm'
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap=colormap,
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   ax=ax)
        
        ax.set_title(f'Correlation Matrix ({method.capitalize()})', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Correlation matrix saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)  # Close figure to free memory
    
    def plot_missing_data(self,
                         df: pd.DataFrame,
                         save_path: Optional[str] = None,
                         show: bool = False) -> None:
        """Plot missing data patterns.
        
        Args:
            df: Input DataFrame
            save_path: Path to save the figure
            show: Whether to display the plot interactively (default False)
        """
        # Calculate missing data
        missing_df = pd.DataFrame({
            'column': df.columns,
            'missing_count': df.isnull().sum(),
            'missing_pct': (df.isnull().sum() / len(df)) * 100
        }).sort_values('missing_pct', ascending=False)
        
        # Filter columns with missing data
        missing_df = missing_df[missing_df['missing_count'] > 0]
        
        if len(missing_df) == 0:
            logger.info("No missing data found")
            return
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0], self.figsize[1]))
        
        # Plot 1: Bar chart of missing percentages
        bars = ax1.barh(range(len(missing_df)), missing_df['missing_pct'].values)
        ax1.set_yticks(range(len(missing_df)))
        ax1.set_yticklabels(missing_df['column'].values)
        ax1.set_xlabel('Missing %')
        ax1.set_title('Missing Data by Column')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Color bars based on severity
        for bar, pct in zip(bars, missing_df['missing_pct'].values):
            if pct > 50:
                bar.set_color('red')
            elif pct > 20:
                bar.set_color('orange')
            else:
                bar.set_color('yellow')
        
        # Plot 2: Missing data pattern matrix
        msno_data = df[missing_df['column'].values].isnull().astype(int)
        ax2.imshow(msno_data.T, cmap='RdYlBu_r', aspect='auto', interpolation='none')
        ax2.set_yticks(range(len(missing_df)))
        ax2.set_yticklabels(missing_df['column'].values)
        ax2.set_xlabel('Sample Index')
        ax2.set_title('Missing Data Pattern (red=missing)')
        
        plt.suptitle('Missing Data Analysis', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Missing data plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)  # Close figure to free memory
    
    def generate_report(self,
                       df: pd.DataFrame,
                       target: Optional[str] = None,
                       output_dir: str = 'eda_output',
                       max_categories: int = 20,
                       palette: Optional[Union[str, List[str]]] = None) -> None:
        """Generate comprehensive EDA report with all visualizations.
        
        Args:
            df: Input DataFrame
            target: Target column name (for multivariate analysis)
            output_dir: Directory to save plots
            max_categories: Maximum categories to show in plots
            palette: Optional color palette override for all plots
        """
        # Apply palette if provided
        if palette:
            self._apply_palette(palette)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating EDA report in {output_dir}")
        logger.info(f"Output directory created: {output_path.absolute()}")
        
        # 1. Data overview
        logger.info("\n=== Data Overview ===")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        col_types = self._identify_column_types(df)
        logger.info(f"Numerical columns: {len(col_types['numerical'])}")
        logger.info(f"Categorical columns: {len(col_types['categorical'])}")
        
        # 2. Missing data analysis
        logger.info("\n=== Missing Data Analysis ===")
        try:
            self.plot_missing_data(df, save_path=str(output_path / 'missing_data.png'))
        except Exception as e:
            logger.error(f"Failed to create missing data plot: {e}")
        
        # 3. Univariate analysis
        logger.info("\n=== Univariate Analysis ===")
        logger.info(f"Creating univariate plots for {len(df.columns)} columns...")
        try:
            univariate_dir = output_path / 'univariate'
            self.plot_univariate(df, max_categories=max_categories,
                               save_dir=str(univariate_dir),
                               individual_plots=True,
                               palette=palette)
        except Exception as e:
            logger.error(f"Failed to create univariate plots: {e}")
        
        # 4. Correlation matrix
        if len(col_types['numerical']) > 1:
            logger.info("\n=== Correlation Analysis ===")
            try:
                self.plot_correlation_matrix(df, save_path=str(output_path / 'correlation.png'))
            except Exception as e:
                logger.error(f"Failed to create correlation matrix: {e}")
        
        # 5. Multivariate analysis with target
        if target and target in df.columns:
            logger.info(f"\n=== Multivariate Analysis (Target: {target}) ===")
            features = [col for col in df.columns if col != target]
            try:
                multivariate_dir = output_path / 'multivariate'
                self.plot_multivariate(df, target, features,  # Will be limited internally
                                     max_categories=max_categories,
                                     save_dir=str(multivariate_dir),
                                     individual_plots=True,
                                     palette=palette)
            except Exception as e:
                logger.error(f"Failed to create multivariate plots: {e}")
        
        # 6. Generate summary statistics
        logger.info("\n=== Summary Statistics ===")
        
        # Numerical statistics
        if col_types['numerical']:
            num_stats = df[col_types['numerical']].describe()
            num_stats.to_csv(output_path / 'numerical_stats.csv')
            logger.info(f"Numerical statistics saved to {output_path / 'numerical_stats.csv'}")
        
        # Categorical statistics
        if col_types['categorical']:
            cat_stats = []
            for col in col_types['categorical']:
                stats = {
                    'column': col,
                    'unique': df[col].nunique(),
                    'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                    'frequency': df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0,
                    'missing': df[col].isna().sum()
                }
                cat_stats.append(stats)
            
            cat_stats_df = pd.DataFrame(cat_stats)
            cat_stats_df.to_csv(output_path / 'categorical_stats.csv', index=False)
            logger.info(f"Categorical statistics saved to {output_path / 'categorical_stats.csv'}")
        
        logger.info(f"\n✅ EDA report completed! All plots saved to {output_dir}/")
    
    def adversarial_validation(self,
                              train_df: pd.DataFrame,
                              test_df: pd.DataFrame,
                              target_col: Optional[str] = None,
                              n_folds: int = 5,
                              sample_size: Optional[int] = None,
                              save_path: Optional[str] = None) -> Dict[str, Any]:
        """Perform adversarial validation to detect distribution shift.
        
        Creates a binary classifier to distinguish between train and test data.
        If the classifier performs well (high AUC), it indicates distribution shift.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            target_col: Name of target column to exclude (if present in train)
            n_folds: Number of CV folds for validation
            sample_size: Sample size if datasets are large (None for full data)
            save_path: Path to save feature importance plot
            
        Returns:
            Dictionary with validation results including:
            - auc_score: ROC-AUC score (>0.5 indicates distribution shift)
            - feature_importance: DataFrame of feature importances
            - interpretation: String interpretation of results
        """
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score
        from sklearn.ensemble import RandomForestClassifier
        import lightgbm as lgb
        
        logger.info("\n=== Adversarial Validation ===")
        logger.info("Checking if train and test come from same distribution...")
        
        # Prepare data
        train_copy = train_df.copy()
        test_copy = test_df.copy()
        
        # Remove ID columns and target
        id_cols = ['id', 'ID', 'Id', 'index', 'Index']
        train_cols_to_drop = []
        test_cols_to_drop = []
        
        # Find columns to drop from train
        for col in train_copy.columns:
            # Drop if it's an ID column or the target
            if col in id_cols or 'id' in col.lower() or col == target_col:
                train_cols_to_drop.append(col)
        
        # Find columns to drop from test (no target in test usually)
        for col in test_copy.columns:
            # Drop if it's an ID column
            if col in id_cols or 'id' in col.lower():
                test_cols_to_drop.append(col)
        
        # Drop columns
        if train_cols_to_drop:
            logger.info(f"Dropping from train: {train_cols_to_drop}")
            train_copy = train_copy.drop(columns=train_cols_to_drop, errors='ignore')
        
        if test_cols_to_drop:
            logger.info(f"Dropping from test: {test_cols_to_drop}")
            test_copy = test_copy.drop(columns=test_cols_to_drop, errors='ignore')
        
        # Double-check target is removed
        if target_col and target_col in train_copy.columns:
            logger.warning(f"Target column '{target_col}' still present, removing...")
            train_copy = train_copy.drop(columns=[target_col])
        
        # Find common columns (excluding ID and target)
        common_cols = list(set(train_copy.columns) & set(test_copy.columns))
        
        if len(common_cols) == 0:
            logger.error("No common columns found between train and test!")
            return {
                'auc_score': 0.5,
                'auc_std': 0,
                'cv_scores': [],
                'feature_importance': None,
                'interpretation': "Error: No common columns",
                'n_train': 0,
                'n_test': 0,
                'n_features': 0
            }
        
        train_copy = train_copy[common_cols]
        test_copy = test_copy[common_cols]
        
        logger.info(f"Using {len(common_cols)} common features for adversarial validation")
        
        # Sample if requested
        if sample_size and len(train_copy) > sample_size:
            train_copy = train_copy.sample(n=sample_size, random_state=42)
        if sample_size and len(test_copy) > sample_size:
            test_copy = test_copy.sample(n=sample_size, random_state=42)
        
        # Create labels: 0 for train, 1 for test
        train_copy['is_test'] = 0
        test_copy['is_test'] = 1
        
        # Combine datasets
        combined_df = pd.concat([train_copy, test_copy], ignore_index=True)
        
        # Separate features and target
        X = combined_df.drop(columns=['is_test'])
        y = combined_df['is_test']
        
        # Handle categorical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            logger.info(f"Encoding {len(categorical_cols)} categorical columns...")
            from sklearn.preprocessing import LabelEncoder
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = X[col].fillna('missing')
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Handle missing values
        X = X.fillna(-999)
        
        # Cross-validation
        cv_scores = []
        feature_importances = []
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Use LightGBM for speed and performance
            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                verbosity=-1,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            val_preds = model.predict_proba(X_val)[:, 1]
            fold_auc = roc_auc_score(y_val, val_preds)
            cv_scores.append(fold_auc)
            
            # Store feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importances.append(model.feature_importances_)
            else:
                logger.warning(f"No feature importances available for fold {fold + 1}")
            
            logger.info(f"  Fold {fold + 1}: AUC = {fold_auc:.4f}")
        
        # Calculate mean AUC
        mean_auc = np.mean(cv_scores)
        std_auc = np.std(cv_scores)
        
        logger.info(f"\nAdversarial Validation Results:")
        logger.info(f"  Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        
        # Feature importance analysis
        feature_importance_df = None
        if feature_importances:
            mean_importance = np.mean(feature_importances, axis=0)
            feature_importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': mean_importance
            }).sort_values('importance', ascending=False)
            
            logger.info("\nTop 10 features that distinguish train from test:")
            for idx, row in feature_importance_df.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Interpretation
        if mean_auc >= 0.99:
            interpretation = "🚨 PERFECT/NEAR-PERFECT separation! Check for data leakage or ID columns."
            logger.error(interpretation)
            logger.error("This usually indicates:")
            logger.error("  1. ID or index columns are still present")
            logger.error("  2. Data leakage (e.g., features computed from test set)")
            logger.error("  3. Temporal features that perfectly separate train/test")
            if feature_importance_df is not None:
                logger.error(f"Check these features: {feature_importance_df.head(3)['feature'].tolist()}")
        elif mean_auc > 0.7:
            interpretation = "⚠️ HIGH distribution shift detected! Train and test sets are significantly different."
            logger.warning(interpretation)
        elif mean_auc > 0.6:
            interpretation = "⚠️ MODERATE distribution shift detected. Some differences between train and test."
            logger.warning(interpretation)
        elif mean_auc > 0.55:
            interpretation = "ℹ️ SMALL distribution shift detected. Minor differences between train and test."
            logger.info(interpretation)
        else:
            interpretation = "✅ NO significant distribution shift detected. Train and test appear similar."
            logger.info(interpretation)
        
        # Plot feature importance if requested
        if save_path and feature_importance_df is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            top_features = feature_importance_df.head(15)
            ax.barh(range(len(top_features)), top_features['importance'].values)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'].values)
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'Adversarial Validation Feature Importance\n(AUC = {mean_auc:.4f})')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Color bars based on importance
            bars = ax.patches
            for bar, importance in zip(bars, top_features['importance'].values):
                if importance > 0.1:
                    bar.set_color('red')
                elif importance > 0.05:
                    bar.set_color('orange')
                else:
                    bar.set_color('green')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Feature importance plot saved to {save_path}")
        
        return {
            'auc_score': mean_auc,
            'auc_std': std_auc,
            'cv_scores': cv_scores,
            'feature_importance': feature_importance_df,
            'interpretation': interpretation,
            'n_train': len(train_copy),
            'n_test': len(test_copy),
            'n_features': len(common_cols)
        }

    def detect_data_quality_issues(self,
                                   df: pd.DataFrame,
                                   variance_threshold: float = 0.01,
                                   missing_threshold: float = 0.95,
                                   cardinality_threshold: int = 100,
                                   duplicate_threshold: float = 0.99,
                                   outlier_method: str = 'iqr') -> Dict[str, Any]:
        """Detect common data quality issues automatically.

        Identifies and reports various data quality problems including constant features,
        duplicates, high cardinality, outliers, and missing data patterns.

        Args:
            df: Input DataFrame to analyze.
            variance_threshold: Minimum variance for numerical features (below = quasi-constant).
            missing_threshold: Maximum missing data ratio (above = problematic).
            cardinality_threshold: Maximum unique values for categorical features.
            duplicate_threshold: Minimum correlation for duplicate columns (above = duplicate).
            outlier_method: Method for outlier detection ('iqr' or 'zscore').

        Returns:
            Dictionary containing:
                - constant_features: List of constant or quasi-constant features
                - duplicate_columns: List of duplicate column pairs
                - high_cardinality: List of high cardinality categorical features
                - high_missing: List of features with excessive missing values
                - outlier_features: Dict of features with outliers and counts
                - recommendations: List of actionable recommendations

        Example:
            >>> eda = EDAAnalyzer()
            >>> issues = eda.detect_data_quality_issues(df)
            >>> print(issues['recommendations'])
            >>> # Drop constant features: ['feature_1', 'feature_2']
        """
        logger.info("\n=== Data Quality Analysis ===")
        issues = {
            'constant_features': [],
            'duplicate_columns': [],
            'high_cardinality': [],
            'high_missing': [],
            'outlier_features': {},
            'recommendations': []
        }

        # 1. Detect constant/quasi-constant features
        col_types = self._identify_column_types(df)
        numerical_cols = col_types['numerical']

        for col in numerical_cols:
            variance = df[col].var()
            if pd.notna(variance) and variance < variance_threshold:
                issues['constant_features'].append({
                    'column': col,
                    'variance': variance,
                    'unique_values': df[col].nunique()
                })

        if issues['constant_features']:
            logger.warning(f"Found {len(issues['constant_features'])} constant/quasi-constant features")
            issues['recommendations'].append(
                f"Drop constant features: {[f['column'] for f in issues['constant_features']]}"
            )

        # 2. Detect duplicate columns
        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols].corr().abs()
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            for col in upper_tri.columns:
                duplicates = upper_tri.index[upper_tri[col] > duplicate_threshold].tolist()
                if duplicates:
                    for dup in duplicates:
                        issues['duplicate_columns'].append({
                            'column1': col,
                            'column2': dup,
                            'correlation': upper_tri.loc[dup, col]
                        })

        if issues['duplicate_columns']:
            logger.warning(f"Found {len(issues['duplicate_columns'])} potential duplicate column pairs")
            issues['recommendations'].append(
                f"Check duplicate columns: {[(d['column1'], d['column2']) for d in issues['duplicate_columns'][:3]]}"
            )

        # 3. Detect high cardinality categorical features
        categorical_cols = col_types['categorical']
        for col in categorical_cols:
            unique_count = df[col].nunique()
            if unique_count > cardinality_threshold:
                issues['high_cardinality'].append({
                    'column': col,
                    'unique_count': unique_count,
                    'total_rows': len(df),
                    'ratio': unique_count / len(df)
                })

        if issues['high_cardinality']:
            logger.warning(f"Found {len(issues['high_cardinality'])} high cardinality categorical features")
            issues['recommendations'].append(
                f"Consider target encoding or grouping for: {[h['column'] for h in issues['high_cardinality']]}"
            )

        # 4. Detect high missing data
        missing_ratios = df.isnull().sum() / len(df)
        high_missing_cols = missing_ratios[missing_ratios > missing_threshold]

        for col in high_missing_cols.index:
            issues['high_missing'].append({
                'column': col,
                'missing_ratio': high_missing_cols[col],
                'missing_count': df[col].isnull().sum()
            })

        if issues['high_missing']:
            logger.warning(f"Found {len(issues['high_missing'])} features with >{missing_threshold*100}% missing")
            issues['recommendations'].append(
                f"Consider dropping: {[h['column'] for h in issues['high_missing']]}"
            )

        # 5. Detect outliers
        for col in numerical_cols:
            data = df[col].dropna()
            if len(data) == 0:
                continue

            if outlier_method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                outliers = ((data < lower_bound) | (data > upper_bound)).sum()
            else:  # zscore
                z_scores = np.abs(stats.zscore(data))
                outliers = (z_scores > 3).sum()

            if outliers > 0:
                outlier_ratio = outliers / len(data)
                if outlier_ratio > 0.01:  # More than 1% outliers
                    issues['outlier_features'][col] = {
                        'count': int(outliers),
                        'ratio': outlier_ratio,
                        'method': outlier_method
                    }

        if issues['outlier_features']:
            logger.info(f"Found {len(issues['outlier_features'])} features with significant outliers (>1%)")
            issues['recommendations'].append(
                f"Review outliers in: {list(issues['outlier_features'].keys())[:5]}"
            )

        # Summary
        total_issues = (len(issues['constant_features']) +
                       len(issues['duplicate_columns']) +
                       len(issues['high_cardinality']) +
                       len(issues['high_missing']))

        if total_issues == 0:
            logger.info("✅ No major data quality issues detected!")
        else:
            logger.warning(f"⚠️ Detected {total_issues} data quality issues")

        return issues

    def detect_feature_interactions(self,
                                    df: pd.DataFrame,
                                    target: str,
                                    top_k: int = 10,
                                    method: str = 'mutual_info',
                                    max_features: int = 50) -> Dict[str, Any]:
        """Detect promising feature interactions for engineering.

        Identifies pairs of features that have high predictive power when combined,
        suggesting candidates for ratio, product, or interaction features.

        Args:
            df: Input DataFrame with features and target.
            target: Name of target column.
            top_k: Number of top interaction pairs to return.
            method: Method for interaction detection ('mutual_info' or 'tree_based').
            max_features: Maximum number of features to analyze (for performance).

        Returns:
            Dictionary containing:
                - top_interactions: List of top feature pairs with scores
                - ratio_candidates: Features good for ratio operations
                - product_candidates: Features good for multiplication
                - nonlinear_features: Features with non-linear relationships to target
                - recommendations: Actionable feature engineering suggestions

        Example:
            >>> eda = EDAAnalyzer()
            >>> interactions = eda.detect_feature_interactions(df, target='price', top_k=5)
            >>> print(interactions['recommendations'])
            >>> # Create ratio: feature_A / feature_B
        """
        from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from itertools import combinations

        logger.info("\n=== Feature Interaction Analysis ===")

        # Identify column types
        col_types = self._identify_column_types(df)
        numerical_cols = [col for col in col_types['numerical'] if col != target]

        # Limit features for performance
        if len(numerical_cols) > max_features:
            logger.warning(f"Too many features ({len(numerical_cols)}). Using top {max_features} by correlation.")
            correlations = df[numerical_cols].corrwith(df[target]).abs()
            numerical_cols = correlations.nlargest(max_features).index.tolist()

        # Determine task type
        target_is_categorical = target in col_types['categorical']

        results = {
            'top_interactions': [],
            'ratio_candidates': [],
            'product_candidates': [],
            'nonlinear_features': [],
            'recommendations': []
        }

        if len(numerical_cols) < 2:
            logger.warning("Need at least 2 numerical features for interaction detection")
            return results

        # Remove rows with missing target
        df_clean = df[numerical_cols + [target]].dropna()

        if len(df_clean) == 0:
            logger.warning("No complete cases for interaction analysis")
            return results

        X = df_clean[numerical_cols]
        y = df_clean[target]

        # 1. Compute base feature importances
        logger.info(f"Analyzing interactions between {len(numerical_cols)} features...")

        if method == 'mutual_info':
            # Mutual information for base features
            mi_func = mutual_info_classif if target_is_categorical else mutual_info_regression
            base_mi = mi_func(X, y, random_state=42)
            base_scores = dict(zip(numerical_cols, base_mi))
        else:  # tree_based
            if target_is_categorical:
                model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)

            model.fit(X, y)
            base_scores = dict(zip(numerical_cols, model.feature_importances_))

        # 2. Test feature interactions (limit to avoid explosion)
        max_pairs = min(1000, len(list(combinations(numerical_cols, 2))))
        feature_pairs = list(combinations(numerical_cols, 2))[:max_pairs]

        interaction_scores = []

        for feat1, feat2 in feature_pairs:
            # Create interaction features
            X_interact = X[[feat1, feat2]].copy()

            # Ratio (handle division by zero)
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = X_interact[feat1] / X_interact[feat2].replace(0, np.nan)
                ratio = ratio.fillna(0)

            # Product
            product = X_interact[feat1] * X_interact[feat2]

            # Test if interactions add value
            X_test = pd.DataFrame({
                'ratio': ratio,
                'product': product
            })

            if method == 'mutual_info':
                mi_scores = mi_func(X_test, y, random_state=42)
                ratio_score = mi_scores[0]
                product_score = mi_scores[1]
            else:
                if target_is_categorical:
                    model_ratio = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42, n_jobs=-1)
                    model_product = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42, n_jobs=-1)
                else:
                    model_ratio = RandomForestRegressor(n_estimators=20, max_depth=3, random_state=42, n_jobs=-1)
                    model_product = RandomForestRegressor(n_estimators=20, max_depth=3, random_state=42, n_jobs=-1)

                model_ratio.fit(X_test[['ratio']], y)
                model_product.fit(X_test[['product']], y)

                ratio_score = model_ratio.score(X_test[['ratio']], y)
                product_score = model_product.score(X_test[['product']], y)

            # Calculate gain over base features
            base_score = max(base_scores.get(feat1, 0), base_scores.get(feat2, 0))

            interaction_scores.append({
                'feature1': feat1,
                'feature2': feat2,
                'ratio_score': ratio_score,
                'product_score': product_score,
                'base_score': base_score,
                'ratio_gain': ratio_score - base_score,
                'product_gain': product_score - base_score
            })

        # 3. Rank interactions
        interaction_scores_sorted = sorted(
            interaction_scores,
            key=lambda x: max(x['ratio_gain'], x['product_gain']),
            reverse=True
        )

        results['top_interactions'] = interaction_scores_sorted[:top_k]

        # 4. Identify best candidates for each operation
        ratio_candidates = sorted(interaction_scores, key=lambda x: x['ratio_gain'], reverse=True)[:top_k]
        product_candidates = sorted(interaction_scores, key=lambda x: x['product_gain'], reverse=True)[:top_k]

        results['ratio_candidates'] = [
            (r['feature1'], r['feature2'], r['ratio_gain'])
            for r in ratio_candidates if r['ratio_gain'] > 0
        ]

        results['product_candidates'] = [
            (p['feature1'], p['feature2'], p['product_gain'])
            for p in product_candidates if p['product_gain'] > 0
        ]

        # 5. Detect non-linear relationships
        for col in numerical_cols:
            # Test polynomial features
            X_poly = pd.DataFrame({
                'linear': X[col],
                'squared': X[col] ** 2,
                'sqrt': np.sqrt(np.abs(X[col]))
            })

            if method == 'mutual_info':
                mi_poly = mi_func(X_poly, y, random_state=42)
                linear_score = mi_poly[0]
                squared_score = mi_poly[1]
                sqrt_score = mi_poly[2]
            else:
                if target_is_categorical:
                    model_poly = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42, n_jobs=-1)
                else:
                    model_poly = RandomForestRegressor(n_estimators=20, max_depth=3, random_state=42, n_jobs=-1)

                model_poly.fit(X_poly, y)
                scores_poly = model_poly.feature_importances_
                linear_score, squared_score, sqrt_score = scores_poly

            # Check if non-linear transforms add value
            if squared_score > linear_score * 1.1 or sqrt_score > linear_score * 1.1:
                results['nonlinear_features'].append({
                    'feature': col,
                    'linear_score': linear_score,
                    'squared_score': squared_score,
                    'sqrt_score': sqrt_score,
                    'best_transform': 'squared' if squared_score > sqrt_score else 'sqrt'
                })

        # 6. Generate recommendations
        if results['ratio_candidates']:
            logger.info(f"Found {len(results['ratio_candidates'])} promising ratio features")
            top_ratio = results['ratio_candidates'][0]
            results['recommendations'].append(
                f"Create ratio: {top_ratio[0]} / {top_ratio[1]} (gain: {top_ratio[2]:.4f})"
            )

        if results['product_candidates']:
            logger.info(f"Found {len(results['product_candidates'])} promising product features")
            top_product = results['product_candidates'][0]
            results['recommendations'].append(
                f"Create product: {top_product[0]} * {top_product[1]} (gain: {top_product[2]:.4f})"
            )

        if results['nonlinear_features']:
            logger.info(f"Found {len(results['nonlinear_features'])} features with non-linear relationships")
            for nl in results['nonlinear_features'][:3]:
                results['recommendations'].append(
                    f"Add {nl['best_transform']} transform: {nl['feature']}^2 or sqrt({nl['feature']})"
                )

        if not results['recommendations']:
            logger.info("✅ No strong feature interactions detected")
            results['recommendations'].append("Feature interactions may not significantly improve model")

        return results

    def analyze_target_distribution(self,
                                    df: pd.DataFrame,
                                    target: str,
                                    save_dir: Optional[str] = None) -> Dict[str, Any]:
        """Comprehensive analysis of target variable distribution.

        Analyzes target distribution, tests for normality, suggests transformations,
        and detects class imbalance or multimodality issues.

        Args:
            df: Input DataFrame containing target variable.
            target: Name of target column to analyze.
            save_dir: Optional directory to save diagnostic plots.

        Returns:
            Dictionary containing:
                - task_type: 'classification' or 'regression'
                - distribution_stats: Mean, median, std, skewness, kurtosis
                - normality_tests: Results of statistical normality tests
                - suggested_transformations: Recommended transformations with scores
                - class_balance: Class distribution (classification only)
                - multimodal: Whether distribution appears multimodal
                - recommendations: Actionable suggestions for modeling

        Example:
            >>> eda = EDAAnalyzer()
            >>> target_analysis = eda.analyze_target_distribution(df, target='price')
            >>> print(target_analysis['suggested_transformations'])
            >>> # ['log', 'yeo-johnson'] - Apply these for better normality
        """
        from scipy.stats import shapiro, normaltest, skew, kurtosis
        from sklearn.preprocessing import PowerTransformer

        logger.info(f"\n=== Target Distribution Analysis: {target} ===")

        results = {
            'task_type': None,
            'distribution_stats': {},
            'normality_tests': {},
            'suggested_transformations': [],
            'class_balance': None,
            'multimodal': False,
            'recommendations': []
        }

        if target not in df.columns:
            logger.error(f"Target column '{target}' not found")
            return results

        target_data = df[target].dropna()

        if len(target_data) == 0:
            logger.error("Target has no non-null values")
            return results

        # Determine task type
        col_types = self._identify_column_types(df)
        is_categorical = target in col_types['categorical']
        results['task_type'] = 'classification' if is_categorical else 'regression'

        logger.info(f"Task type: {results['task_type']}")

        if is_categorical:
            # Classification analysis
            value_counts = target_data.value_counts()
            total = len(target_data)

            results['class_balance'] = {
                'classes': value_counts.to_dict(),
                'class_ratios': (value_counts / total).to_dict(),
                'n_classes': len(value_counts),
                'majority_class': value_counts.index[0],
                'majority_ratio': value_counts.iloc[0] / total
            }

            logger.info(f"Number of classes: {results['class_balance']['n_classes']}")
            logger.info(f"Class distribution:\n{value_counts}")

            # Check for imbalance
            minority_ratio = value_counts.iloc[-1] / total
            if minority_ratio < 0.05:
                results['recommendations'].append(
                    f"⚠️ Severe class imbalance detected (minority: {minority_ratio:.2%}). "
                    "Consider SMOTE, class weights, or stratified sampling."
                )
            elif minority_ratio < 0.2:
                results['recommendations'].append(
                    f"⚠️ Moderate class imbalance (minority: {minority_ratio:.2%}). "
                    "Use stratified CV and consider class weights."
                )
            else:
                results['recommendations'].append(
                    "✅ Classes are reasonably balanced"
                )

            # Binary vs multiclass
            if results['class_balance']['n_classes'] == 2:
                results['recommendations'].append(
                    "Binary classification: Use AUC-ROC, F1-score for evaluation"
                )
            else:
                results['recommendations'].append(
                    f"Multiclass classification ({results['class_balance']['n_classes']} classes): "
                    "Use macro/weighted F1, confusion matrix"
                )

        else:
            # Regression analysis
            target_array = target_data.values

            # Basic statistics
            results['distribution_stats'] = {
                'mean': float(np.mean(target_array)),
                'median': float(np.median(target_array)),
                'std': float(np.std(target_array)),
                'min': float(np.min(target_array)),
                'max': float(np.max(target_array)),
                'skewness': float(skew(target_array)),
                'kurtosis': float(kurtosis(target_array)),
                'cv': float(np.std(target_array) / np.mean(target_array)) if np.mean(target_array) != 0 else 0
            }

            logger.info(f"Mean: {results['distribution_stats']['mean']:.4f}, "
                       f"Median: {results['distribution_stats']['median']:.4f}, "
                       f"Std: {results['distribution_stats']['std']:.4f}")
            logger.info(f"Skewness: {results['distribution_stats']['skewness']:.4f}, "
                       f"Kurtosis: {results['distribution_stats']['kurtosis']:.4f}")

            # Normality tests
            if len(target_array) < 5000:  # Shapiro-Wilk works better for smaller samples
                try:
                    stat, p_value = shapiro(target_array[:5000])  # Limit to 5000 samples
                    results['normality_tests']['shapiro_wilk'] = {
                        'statistic': float(stat),
                        'p_value': float(p_value),
                        'is_normal': p_value > 0.05
                    }
                    logger.info(f"Shapiro-Wilk test: p={p_value:.4f} "
                              f"({'normal' if p_value > 0.05 else 'not normal'})")
                except Exception as e:
                    logger.warning(f"Shapiro-Wilk test failed: {e}")

            # D'Agostino-Pearson test
            try:
                stat, p_value = normaltest(target_array)
                results['normality_tests']['dagostino_pearson'] = {
                    'statistic': float(stat),
                    'p_value': float(p_value),
                    'is_normal': p_value > 0.05
                }
                logger.info(f"D'Agostino-Pearson test: p={p_value:.4f} "
                          f"({'normal' if p_value > 0.05 else 'not normal'})")
            except Exception as e:
                logger.warning(f"D'Agostino-Pearson test failed: {e}")

            # Test transformations
            transformations_to_test = []

            # 1. Log transform (only for positive values)
            if np.all(target_array > 0):
                log_transformed = np.log(target_array)
                log_skew = abs(skew(log_transformed))
                transformations_to_test.append({
                    'name': 'log',
                    'skewness': log_skew,
                    'formula': 'log(y)'
                })

            # 2. Square root (for non-negative)
            if np.all(target_array >= 0):
                sqrt_transformed = np.sqrt(target_array)
                sqrt_skew = abs(skew(sqrt_transformed))
                transformations_to_test.append({
                    'name': 'sqrt',
                    'skewness': sqrt_skew,
                    'formula': 'sqrt(y)'
                })

            # 3. Box-Cox (requires positive values)
            if np.all(target_array > 0):
                try:
                    pt = PowerTransformer(method='box-cox', standardize=False)
                    boxcox_transformed = pt.fit_transform(target_array.reshape(-1, 1)).flatten()
                    boxcox_skew = abs(skew(boxcox_transformed))
                    transformations_to_test.append({
                        'name': 'box-cox',
                        'skewness': boxcox_skew,
                        'formula': 'box-cox(y)'
                    })
                except Exception as e:
                    logger.debug(f"Box-Cox failed: {e}")

            # 4. Yeo-Johnson (works with any values)
            try:
                pt = PowerTransformer(method='yeo-johnson', standardize=False)
                yj_transformed = pt.fit_transform(target_array.reshape(-1, 1)).flatten()
                yj_skew = abs(skew(yj_transformed))
                transformations_to_test.append({
                    'name': 'yeo-johnson',
                    'skewness': yj_skew,
                    'formula': 'yeo-johnson(y)'
                })
            except Exception as e:
                logger.debug(f"Yeo-Johnson failed: {e}")

            # 5. Original (no transform)
            original_skew = abs(results['distribution_stats']['skewness'])
            transformations_to_test.append({
                'name': 'none',
                'skewness': original_skew,
                'formula': 'y'
            })

            # Sort by skewness (lower is better for normality)
            transformations_to_test.sort(key=lambda x: x['skewness'])
            results['suggested_transformations'] = transformations_to_test

            logger.info("\nTransformation suggestions (ordered by improvement):")
            for i, trans in enumerate(transformations_to_test[:3]):
                logger.info(f"  {i+1}. {trans['name']}: skewness={trans['skewness']:.4f}")

            # Generate recommendations
            if original_skew > 1:
                best_transform = transformations_to_test[0]
                if best_transform['name'] != 'none':
                    results['recommendations'].append(
                        f"⚠️ Target is highly skewed ({original_skew:.2f}). "
                        f"Consider {best_transform['name']} transformation."
                    )
                else:
                    results['recommendations'].append(
                        f"⚠️ Target is highly skewed ({original_skew:.2f}). "
                        "Tree-based models may work better than linear models."
                    )
            elif original_skew > 0.5:
                results['recommendations'].append(
                    f"ℹ️ Target is moderately skewed ({original_skew:.2f}). "
                    "May benefit from transformation for linear models."
                )
            else:
                results['recommendations'].append(
                    f"✅ Target distribution is reasonably symmetric (skewness={original_skew:.2f})"
                )

            # Check for multimodality (simple heuristic using KDE)
            try:
                from scipy.signal import find_peaks
                hist, bin_edges = np.histogram(target_array, bins=50)
                peaks, _ = find_peaks(hist, height=len(target_array) * 0.02)
                if len(peaks) > 1:
                    results['multimodal'] = True
                    results['recommendations'].append(
                        f"⚠️ Distribution appears multimodal ({len(peaks)} peaks detected). "
                        "May indicate mixed populations or need for stratification."
                    )
            except Exception as e:
                logger.debug(f"Multimodality check failed: {e}")

            # Metric recommendations
            if original_skew > 1 or results['multimodal']:
                results['recommendations'].append(
                    "📊 Suggested metrics: MAE, Huber loss, or quantile loss (robust to outliers)"
                )
            else:
                results['recommendations'].append(
                    "📊 Suggested metrics: RMSE, R², MAE"
                )

        # Create diagnostic plots if save_dir provided
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

            if is_categorical:
                # Bar plot for classification
                fig, ax = plt.subplots(figsize=(10, 6))
                value_counts.plot(kind='bar', ax=ax, color='steelblue')
                ax.set_title(f'Target Distribution: {target}')
                ax.set_xlabel('Class')
                ax.set_ylabel('Count')
                ax.grid(True, alpha=0.3, axis='y')

                # Add percentage labels
                total = len(target_data)
                for i, v in enumerate(value_counts.values):
                    ax.text(i, v, f'{v}\n({v/total:.1%})', ha='center', va='bottom')

                plt.tight_layout()
                plt.savefig(save_path / f'{target}_distribution.png', dpi=100, bbox_inches='tight')
                plt.close(fig)

            else:
                # Multiple plots for regression
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))

                # Histogram
                axes[0, 0].hist(target_array, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
                axes[0, 0].axvline(results['distribution_stats']['mean'], color='red',
                                  linestyle='--', label='Mean')
                axes[0, 0].axvline(results['distribution_stats']['median'], color='green',
                                  linestyle='--', label='Median')
                axes[0, 0].set_title(f'Distribution: {target}')
                axes[0, 0].set_xlabel(target)
                axes[0, 0].set_ylabel('Frequency')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)

                # Box plot
                axes[0, 1].boxplot(target_array, vert=True)
                axes[0, 1].set_title('Box Plot')
                axes[0, 1].set_ylabel(target)
                axes[0, 1].grid(True, alpha=0.3)

                # Q-Q plot
                stats.probplot(target_array, dist="norm", plot=axes[1, 0])
                axes[1, 0].set_title('Q-Q Plot (Normal Distribution)')
                axes[1, 0].grid(True, alpha=0.3)

                # Transformation comparison (best 4)
                if len(transformations_to_test) > 1:
                    # Show original vs best transformation
                    best_trans = transformations_to_test[0]
                    if best_trans['name'] != 'none':
                        # Apply best transformation
                        if best_trans['name'] == 'log':
                            transformed = np.log(target_array)
                        elif best_trans['name'] == 'sqrt':
                            transformed = np.sqrt(target_array)
                        elif best_trans['name'] == 'box-cox':
                            pt = PowerTransformer(method='box-cox', standardize=False)
                            transformed = pt.fit_transform(target_array.reshape(-1, 1)).flatten()
                        elif best_trans['name'] == 'yeo-johnson':
                            pt = PowerTransformer(method='yeo-johnson', standardize=False)
                            transformed = pt.fit_transform(target_array.reshape(-1, 1)).flatten()

                        axes[1, 1].hist(transformed, bins=50, color='green', alpha=0.7, edgecolor='black')
                        axes[1, 1].set_title(f'Best Transform: {best_trans["name"]} (skew={best_trans["skewness"]:.2f})')
                        axes[1, 1].set_xlabel(f'{best_trans["name"]}({target})')
                        axes[1, 1].set_ylabel('Frequency')
                        axes[1, 1].grid(True, alpha=0.3)
                    else:
                        axes[1, 1].text(0.5, 0.5, 'No transformation needed',
                                       ha='center', va='center', transform=axes[1, 1].transAxes,
                                       fontsize=12)
                        axes[1, 1].set_title('Transformation')

                plt.suptitle(f'Target Analysis: {target}', fontsize=14, y=1.00)
                plt.tight_layout()
                plt.savefig(save_path / f'{target}_analysis.png', dpi=100, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Saved target analysis plots to {save_path}")

        return results

    def detect_leakage(self,
                      train_df: pd.DataFrame,
                      test_df: Optional[pd.DataFrame],
                      target: str,
                      threshold: float = 0.99) -> Dict[str, Any]:
        """Detect potential data leakage in features.

        Identifies features that may cause data leakage, including perfect correlations,
        suspicious patterns, and features that shouldn't be available at prediction time.

        Args:
            train_df: Training DataFrame with target variable.
            test_df: Optional test DataFrame for train/test comparison.
            target: Name of target column.
            threshold: Correlation threshold for flagging potential leakage (default 0.99).

        Returns:
            Dictionary containing:
                - perfect_correlations: Features with correlation > threshold
                - suspicious_features: Features with unusual patterns
                - test_leakage: Features with values only in test set
                - temporal_leakage: Time-based features that may leak
                - recommendations: Actions to investigate or fix leakage

        Example:
            >>> eda = EDAAnalyzer()
            >>> leakage = eda.detect_leakage(train_df, test_df, target='target')
            >>> if leakage['perfect_correlations']:
            >>>     print("⚠️ Potential leakage detected!")
        """
        logger.info("\n=== Leakage Detection ===")

        results = {
            'perfect_correlations': [],
            'suspicious_features': [],
            'test_leakage': [],
            'temporal_leakage': [],
            'recommendations': []
        }

        if target not in train_df.columns:
            logger.error(f"Target column '{target}' not found")
            return results

        # Identify numerical columns
        col_types = self._identify_column_types(train_df)
        numerical_cols = [col for col in col_types['numerical'] if col != target]

        if len(numerical_cols) == 0:
            logger.info("No numerical features to check for leakage")
            return results

        # 1. Check for perfect/near-perfect correlations with target
        logger.info("Checking for perfect correlations with target...")
        for col in numerical_cols:
            try:
                corr = abs(train_df[[col, target]].corr().iloc[0, 1])
                if corr > threshold:
                    results['perfect_correlations'].append({
                        'feature': col,
                        'correlation': float(corr),
                        'severity': 'CRITICAL' if corr > 0.999 else 'HIGH'
                    })
                    logger.warning(f"  🚨 {col}: correlation = {corr:.6f}")
            except:
                continue

        # 2. Check for suspicious value patterns
        logger.info("Checking for suspicious patterns...")
        for col in numerical_cols:
            nunique = train_df[col].nunique()
            nrows = len(train_df)

            # Check if feature has same cardinality as target (potential ID leakage)
            target_nunique = train_df[target].nunique()
            if nunique == target_nunique and nunique > 10:
                results['suspicious_features'].append({
                    'feature': col,
                    'reason': f'Same cardinality as target ({nunique} unique values)',
                    'severity': 'HIGH'
                })

            # Check for features with unique value per row (potential row ID)
            if nunique == nrows:
                results['suspicious_features'].append({
                    'feature': col,
                    'reason': 'Unique value per row (potential ID column)',
                    'severity': 'HIGH'
                })

        # 3. Compare train/test distributions for leakage
        if test_df is not None:
            logger.info("Checking for train/test leakage...")
            common_cols = list(set(numerical_cols) & set(test_df.columns))

            for col in common_cols:
                # Check for values in test but not in train
                train_values = set(train_df[col].dropna().unique())
                test_values = set(test_df[col].dropna().unique())

                test_only = test_values - train_values
                if len(test_only) > 0 and len(test_only) / len(test_values) > 0.1:
                    results['test_leakage'].append({
                        'feature': col,
                        'test_only_values': len(test_only),
                        'test_only_ratio': len(test_only) / len(test_values),
                        'severity': 'MEDIUM'
                    })

        # 4. Check for temporal leakage patterns
        logger.info("Checking for temporal leakage...")
        datetime_cols = train_df.select_dtypes(include=['datetime64']).columns
        potential_temporal_cols = [col for col in train_df.columns
                                   if any(term in col.lower()
                                         for term in ['date', 'time', 'year', 'month', 'day',
                                                     'created', 'updated', 'modified'])]

        if len(datetime_cols) > 0 or len(potential_temporal_cols) > 0:
            temporal_cols = list(set(list(datetime_cols) + potential_temporal_cols))
            for col in temporal_cols:
                results['temporal_leakage'].append({
                    'feature': col,
                    'warning': 'Temporal feature detected - ensure no future information leaks',
                    'severity': 'MEDIUM'
                })

        # Generate recommendations
        if results['perfect_correlations']:
            results['recommendations'].append(
                f"🚨 CRITICAL: {len(results['perfect_correlations'])} feature(s) have near-perfect "
                f"correlation with target. Investigate: {[f['feature'] for f in results['perfect_correlations'][:3]]}"
            )

        if results['suspicious_features']:
            results['recommendations'].append(
                f"⚠️ {len(results['suspicious_features'])} suspicious feature(s) detected. "
                f"Check: {[f['feature'] for f in results['suspicious_features'][:3]]}"
            )

        if results['test_leakage']:
            results['recommendations'].append(
                f"⚠️ {len(results['test_leakage'])} feature(s) have values only in test set. "
                "This may indicate data leakage or distribution shift."
            )

        if results['temporal_leakage']:
            results['recommendations'].append(
                f"ℹ️ {len(results['temporal_leakage'])} temporal feature(s) detected. "
                "Verify no future information is used."
            )

        if not any([results['perfect_correlations'], results['suspicious_features'],
                   results['test_leakage'], results['temporal_leakage']]):
            logger.info("✅ No obvious leakage patterns detected")
            results['recommendations'].append("No obvious data leakage detected")

        return results

    def full_eda_analysis(self,
                         train_df: pd.DataFrame,
                         target: str,
                         test_df: Optional[pd.DataFrame] = None,
                         output_dir: str = 'eda_comprehensive',
                         include_interactions: bool = True,
                         include_baseline: bool = False) -> Dict[str, Any]:
        """Perform comprehensive EDA analysis with all available tools.

        Runs all EDA analyses including quality checks, target analysis, feature interactions,
        adversarial validation, leakage detection, and generates visualizations.

        Args:
            train_df: Training DataFrame.
            target: Target column name.
            test_df: Optional test DataFrame for comparison.
            output_dir: Directory to save all outputs.
            include_interactions: Whether to run feature interaction detection (can be slow).
            include_baseline: Whether to train quick baseline models.

        Returns:
            Dictionary containing all analysis results:
                - data_quality: Data quality issues
                - target_analysis: Target distribution analysis
                - feature_interactions: Feature interaction suggestions (if enabled)
                - adversarial_validation: Train/test similarity (if test_df provided)
                - leakage_detection: Leakage detection results
                - recommendations: Consolidated list of all recommendations

        Example:
            >>> eda = EDAAnalyzer()
            >>> results = eda.full_eda_analysis(
            >>>     train_df=train,
            >>>     test_df=test,
            >>>     target='price',
            >>>     output_dir='./eda'
            >>> )
            >>> print("\\n".join(results['recommendations']))
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 70)
        logger.info("COMPREHENSIVE EDA ANALYSIS")
        logger.info("=" * 70)

        all_results = {
            'data_quality': None,
            'target_analysis': None,
            'feature_interactions': None,
            'adversarial_validation': None,
            'leakage_detection': None,
            'recommendations': []
        }

        # 1. Data quality analysis
        try:
            logger.info("\n[1/6] Running data quality analysis...")
            all_results['data_quality'] = self.detect_data_quality_issues(train_df)
            all_results['recommendations'].extend(all_results['data_quality']['recommendations'])
        except Exception as e:
            logger.error(f"Data quality analysis failed: {e}")

        # 2. Target distribution analysis
        try:
            logger.info("\n[2/6] Analyzing target distribution...")
            all_results['target_analysis'] = self.analyze_target_distribution(
                train_df, target, save_dir=str(output_path / 'target_analysis')
            )
            all_results['recommendations'].extend(all_results['target_analysis']['recommendations'])
        except Exception as e:
            logger.error(f"Target analysis failed: {e}")

        # 3. Feature interactions (optional, can be slow)
        if include_interactions and len(train_df.columns) <= 100:
            try:
                logger.info("\n[3/6] Detecting feature interactions...")
                all_results['feature_interactions'] = self.detect_feature_interactions(
                    train_df, target, top_k=10
                )
                all_results['recommendations'].extend(all_results['feature_interactions']['recommendations'])
            except Exception as e:
                logger.error(f"Feature interaction detection failed: {e}")
        else:
            logger.info("\n[3/6] Skipping feature interactions (disabled or too many features)")

        # 4. Adversarial validation
        if test_df is not None:
            try:
                logger.info("\n[4/6] Running adversarial validation...")
                all_results['adversarial_validation'] = self.adversarial_validation(
                    train_df, test_df, target_col=target,
                    save_path=str(output_path / 'adversarial_validation.png')
                )
                all_results['recommendations'].append(
                    all_results['adversarial_validation']['interpretation']
                )
            except Exception as e:
                logger.error(f"Adversarial validation failed: {e}")
        else:
            logger.info("\n[4/6] Skipping adversarial validation (no test set provided)")

        # 5. Leakage detection
        try:
            logger.info("\n[5/6] Detecting potential data leakage...")
            all_results['leakage_detection'] = self.detect_leakage(
                train_df, test_df, target
            )
            all_results['recommendations'].extend(all_results['leakage_detection']['recommendations'])
        except Exception as e:
            logger.error(f"Leakage detection failed: {e}")

        # 6. Generate standard EDA report
        try:
            logger.info("\n[6/6] Generating standard EDA visualizations...")
            self.generate_report(train_df, target=target, output_dir=str(output_path / 'standard_eda'))
        except Exception as e:
            logger.error(f"Standard EDA report failed: {e}")

        # Save consolidated recommendations
        recommendations_file = output_path / 'recommendations.txt'
        with open(recommendations_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("EDA RECOMMENDATIONS SUMMARY\n")
            f.write("=" * 70 + "\n\n")

            for i, rec in enumerate(all_results['recommendations'], 1):
                f.write(f"{i}. {rec}\n\n")

        logger.info(f"\n{'=' * 70}")
        logger.info(f"✅ Comprehensive EDA completed!")
        logger.info(f"📁 Results saved to: {output_path.absolute()}")
        logger.info(f"📋 Recommendations: {len(all_results['recommendations'])} items")
        logger.info(f"{'=' * 70}\n")

        # Print top recommendations
        logger.info("Top Recommendations:")
        for i, rec in enumerate(all_results['recommendations'][:10], 1):
            logger.info(f"  {i}. {rec}")

        if len(all_results['recommendations']) > 10:
            logger.info(f"  ... and {len(all_results['recommendations']) - 10} more")

        return all_results

    def plot_statistical_comparison(self,
                                    df: pd.DataFrame,
                                    categorical_col: str,
                                    numerical_col: str,
                                    plot_type: str = 'box',
                                    add_stats: bool = True,
                                    add_pairwise: bool = False,
                                    save_path: Optional[str] = None,
                                    show: bool = False) -> Dict[str, Any]:
        """Create statistical comparison plot with annotations (ggpubr-style).

        Creates publication-ready plots comparing a numerical variable across
        categorical groups, with automatic statistical test annotations similar
        to R's ggpubr package.

        Args:
            df: Input DataFrame.
            categorical_col: Name of categorical variable (x-axis).
            numerical_col: Name of numerical variable (y-axis).
            plot_type: Type of plot ('box', 'violin', 'bar', 'strip').
            add_stats: Whether to add overall statistical test annotation.
            add_pairwise: Whether to add pairwise comparison bars with significance stars.
            save_path: Optional path to save the figure.
            show: Whether to display the plot interactively.

        Returns:
            Dictionary containing:
                - test_result: Overall statistical test results
                - pairwise_results: List of significant pairwise comparisons (if add_pairwise=True)
                - group_stats: Summary statistics per group

        Example:
            >>> eda = EDAAnalyzer()
            >>> results = eda.plot_statistical_comparison(
            >>>     df,
            >>>     categorical_col='department',
            >>>     numerical_col='salary',
            >>>     add_stats=True,
            >>>     add_pairwise=True,
            >>>     save_path='salary_by_dept.png'
            >>> )
            >>> print(results['test_result']['interpretation'])
            >>> # "*** (p<0.001)" - Highly significant difference
        """
        from scipy.stats import mannwhitneyu

        logger.info(f"\n=== Statistical Comparison: {numerical_col} by {categorical_col} ===")

        # Remove missing values
        data = df[[categorical_col, numerical_col]].dropna()

        if len(data) == 0:
            logger.error("No data after removing NaN values")
            return {}

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create base plot
        if plot_type == 'box':
            sns.boxplot(data=data, x=categorical_col, y=numerical_col, ax=ax)
        elif plot_type == 'violin':
            sns.violinplot(data=data, x=categorical_col, y=numerical_col, ax=ax)
        elif plot_type == 'bar':
            means = data.groupby(categorical_col)[numerical_col].mean()
            errors = data.groupby(categorical_col)[numerical_col].sem()
            ax.bar(range(len(means)), means.values, yerr=errors.values, alpha=0.7)
            ax.set_xticks(range(len(means)))
            ax.set_xticklabels(means.index)
        elif plot_type == 'strip':
            sns.stripplot(data=data, x=categorical_col, y=numerical_col, ax=ax, alpha=0.5)
            sns.boxplot(data=data, x=categorical_col, y=numerical_col, ax=ax,
                       showcaps=False, boxprops={'facecolor': 'None'},
                       showfliers=False, whiskerprops={'linewidth': 0})

        # Add mean markers for box and violin plots
        if plot_type in ['box', 'violin']:
            means = data.groupby(categorical_col)[numerical_col].mean()
            positions = range(len(means))
            ax.scatter(positions, means.values, color='red', s=100, zorder=10,
                      marker='D', label='Mean', edgecolors='black', linewidths=1)

        # Compute overall statistical test
        test_result = self._compute_statistical_test(data, categorical_col, numerical_col)

        # Add statistical annotation
        if add_stats and test_result['p_value'] is not None:
            self._add_stat_annotation(ax, data, categorical_col, numerical_col, test_result)
            logger.info(f"Overall test: {test_result['test']} p={test_result['p_value']:.4f} "
                       f"{test_result['interpretation']}")

        # Add pairwise comparisons
        pairwise_results = []
        if add_pairwise and 2 <= data[categorical_col].nunique() <= 6:
            self._add_pairwise_annotations(ax, data, categorical_col, numerical_col)

            # Compute all pairwise results for return
            categories = sorted(data[categorical_col].unique())
            for i, (cat1, cat2) in enumerate(combinations(categories, 2)):
                group1 = data[data[categorical_col] == cat1][numerical_col].values
                group2 = data[data[categorical_col] == cat2][numerical_col].values

                try:
                    stat, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
                    pairwise_results.append({
                        'group1': cat1,
                        'group2': cat2,
                        'statistic': float(stat),
                        'p_value': float(p_val),
                        'significant': p_val < 0.05
                    })
                except:
                    continue

        # Compute group statistics
        group_stats = data.groupby(categorical_col)[numerical_col].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).to_dict('index')

        # Format plot
        ax.set_xlabel(categorical_col.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(numerical_col.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{numerical_col.replace("_", " ").title()} by {categorical_col.replace("_", " ").title()}',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        if plot_type in ['box', 'violin']:
            ax.legend(loc='best', fontsize=10)

        # Rotate x labels if many categories
        if data[categorical_col].nunique() > 5:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        # Show if requested
        if show:
            plt.show()
        else:
            plt.close(fig)

        # Return results
        results = {
            'test_result': test_result,
            'group_stats': group_stats,
            'pairwise_results': pairwise_results if add_pairwise else None
        }

        # Print summary
        logger.info(f"\nGroup Statistics:")
        for group, stats in group_stats.items():
            logger.info(f"  {group}: n={stats['count']}, mean={stats['mean']:.2f}, "
                       f"median={stats['median']:.2f}, std={stats['std']:.2f}")

        return results