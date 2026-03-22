"""Command-line interface for tabml.

This module provides a command-line interface for common TabML operations,
making it easy to train models, perform EDA, and validate data without
writing code.

Commands:
    train: Train models and create submission
    eda: Perform exploratory data analysis
    optimize: Optimize hyperparameters
    predict: Generate predictions from saved model
    validate: Check data for issues and leakage
    info: Show available datasets
    
Example:
    Basic CLI usage::
    
        # Train with default settings
        $ tabml train --data-dir data/
        
        # Perform EDA
        $ tabml eda --train-file train.csv --target price
        
        # Optimize hyperparameters
        $ tabml optimize --model xgboost --n-trials 200
        
        # Validate data for leakage
        $ tabml validate --datetime-columns date created_at
"""

import click
from pathlib import Path
from loguru import logger

from .pipeline import TabularPipeline
from .visualize import Visualizer


@click.group()
def cli():
    """TabML - Tabular Machine Learning CLI.
    
    A comprehensive command-line tool for tabular machine learning tasks.
    Use 'tabml COMMAND --help' for more information on each command.
    
    Example:
        $ tabml --help              # Show all commands
        $ tabml train --help        # Show train command options
        $ tabml eda --data-dir ./   # Run EDA on current directory
    """
    pass


@cli.command()
@click.option('--data-dir', '-d', default='data', help='Data directory')
@click.option('--train-file', '-t', default='train.csv', help='Training file name')
@click.option('--test-file', '-s', default='test.csv', help='Test file name')
@click.option('--target', help='Target column name (auto-detected if not specified)')
@click.option('--sample-frac', type=float, help='Sample fraction for quick testing')
@click.option('--optimize', is_flag=True, help='Optimize hyperparameters')
@click.option('--cv-folds', default=5, help='Number of CV folds')
@click.option('--output', '-o', default='submission.csv', help='Output file name')
def train(data_dir, train_file, test_file, target, sample_frac, optimize, cv_folds, output):
    """Train models and create submission.
    
    Runs the complete TabML pipeline including data loading, feature
    engineering, model training, and prediction generation.
    
    Example:
        \b
        # Basic training with auto-detection
        $ tabml train --data-dir ./data
        
        \b
        # Train with specific target and optimization
        $ tabml train --target price --optimize --cv-folds 10
        
        \b  
        # Quick test with 10% of data
        $ tabml train --sample-frac 0.1 --output test_submission.csv
        
    Note:
        - Target column is auto-detected if not specified
        - Uses all available models and selects best
        - Feature engineering and selection are automatic
    """
    logger.info(f"Starting TabML pipeline...")
    
    # Initialize pipeline
    pipeline = TabularPipeline(data_dir=data_dir)
    
    # Load data
    pipeline.load_data(
        train_file=train_file,
        test_file=test_file,
        target_column=target,
        sample_frac=sample_frac
    )
    
    # Run pipeline
    submission = pipeline.run_full_pipeline(
        feature_engineering={
            'numeric_impute': 'median',
            'scaling': 'standard',
            'target_encoding': True
        },
        feature_selection={
            'method': 'mutual_info',
            'n_features': 0.8
        },
        model_training={
            'optimize_hyperparams': optimize,
            'cv_folds': cv_folds
        }
    )
    
    # Save submission
    submission.to_csv(output, index=False)
    logger.info(f"Submission saved to {output}")


@cli.command()
@click.option('--data-dir', '-d', default='data', help='Data directory')
@click.option('--train-file', '-t', default='train.csv', help='Training file name')
@click.option('--target', help='Target column name')
def eda(data_dir, train_file, target):
    """Perform exploratory data analysis.
    
    Generates comprehensive visualizations and statistics to understand
    your dataset including distributions, correlations, and missing values.
    
    Example:
        \b
        # Basic EDA
        $ tabml eda --train-file train.csv
        
        \b
        # EDA with target analysis
        $ tabml eda --target price --data-dir ./kaggle/house-prices
        
    Output includes:
        - Dataset overview and memory usage
        - Missing value analysis
        - Target distribution (if specified)
        - Feature distributions
        - Correlation matrix
        - Feature vs target relationships
    """
    import pandas as pd
    
    # Load data
    train_path = Path(data_dir) / train_file
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
        
    df = pd.read_csv(train_path)
    
    # Create visualizer
    viz = Visualizer()
    
    # Create EDA report
    viz.create_eda_report(df, target_column=target)


@cli.command()
@click.option('--data-dir', '-d', default='data', help='Data directory')
@click.option('--train-file', '-t', default='train.csv', help='Training file name')
@click.option('--test-file', '-s', default='test.csv', help='Test file name')
@click.option('--target', help='Target column name (auto-detected if not specified)')
@click.option('--model', '-m', default='xgboost', type=click.Choice(['xgboost', 'lightgbm', 'catboost', 'rf']), help='Model type')
@click.option('--n-trials', '-n', default=100, help='Number of optimization trials')
@click.option('--cv-folds', default=5, help='Number of CV folds')
@click.option('--output', '-o', default='best_params.json', help='Output file for best parameters')
def optimize(data_dir, train_file, test_file, target, model, n_trials, cv_folds, output):
    """Optimize hyperparameters for a model.
    
    Uses Optuna to find optimal hyperparameters through Bayesian
    optimization with cross-validation.
    
    Example:
        \b
        # Optimize XGBoost with 100 trials
        $ tabml optimize --model xgboost --n-trials 100
        
        \b
        # Extensive optimization for LightGBM
        $ tabml optimize --model lightgbm --n-trials 500 --cv-folds 10
        
        \b
        # Save results to custom file
        $ tabml optimize --model catboost --output catboost_params.json
        
    Output JSON contains:
        - model: Model type
        - best_params: Optimal hyperparameters found
        - best_score: Best cross-validation score
        - n_trials: Number of trials performed
        
    Note:
        Model options: 'xgboost', 'lightgbm', 'catboost', 'rf'
    """
    import json
    from .data import DataLoader
    from .features import FeatureEngineer
    from .models import ModelTrainer
    
    logger.info(f"Starting hyperparameter optimization for {model}")
    
    # Load data
    loader = DataLoader(data_dir=data_dir)
    train_df, test_df = loader.load_data(
        train_file=train_file,
        test_file=test_file,
        target_column=target
    )
    
    # Get train/val split
    X_train, X_val, y_train, y_val = loader.get_train_test_split()
    
    # Basic feature engineering
    engineer = FeatureEngineer()
    X_train = engineer.fit_transform(X_train, y_train)
    X_val = engineer.transform(X_val)
    
    # Initialize trainer
    trainer = ModelTrainer(task_type='auto')
    
    # Optimize hyperparameters
    best_params, best_score = trainer.optimize_hyperparameters(
        model,
        X_train, y_train,
        X_val, y_val,
        n_trials=n_trials,
        cv_folds=cv_folds
    )
    
    # Save best parameters
    with open(output, 'w') as f:
        json.dump({
            'model': model,
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': n_trials
        }, f, indent=2)
    
    logger.info(f"Best parameters saved to {output}")
    logger.info(f"Best score: {best_score:.4f}")


@cli.command()
@click.option('--data-dir', '-d', default='data', help='Data directory')
@click.option('--test-file', '-s', default='test.csv', help='Test file name')
@click.option('--model-path', '-m', required=True, help='Path to saved model')
@click.option('--output', '-o', default='predictions.csv', help='Output file name')
def predict(data_dir, test_file, model_path, output):
    """Generate predictions using a saved model.
    
    Loads a previously trained model and generates predictions on new data.
    Handles preprocessing if saved with the model.
    
    Example:
        \b
        # Basic prediction
        $ tabml predict --model-path models/best_model.pkl
        
        \b
        # Custom test file and output
        $ tabml predict -m trained_model.pkl -s new_test.csv -o final_predictions.csv
        
    Requirements:
        - Model must be saved with joblib
        - Model file should contain 'model' and optionally 'preprocessor'
        - Test data must have same features as training data
    """
    import joblib
    import pandas as pd
    
    logger.info(f"Loading model from {model_path}")
    
    # Load model and preprocessor
    model_data = joblib.load(model_path)
    model = model_data['model']
    preprocessor = model_data.get('preprocessor')
    
    # Load test data
    test_path = Path(data_dir) / test_file
    test_df = pd.read_csv(test_path)
    
    # Apply preprocessing if available
    if preprocessor:
        test_df = preprocessor.transform(test_df)
    
    # Generate predictions
    predictions = model.predict(test_df)
    
    # Save predictions
    output_df = pd.DataFrame({
        'prediction': predictions
    })
    output_df.to_csv(output, index=False)
    
    logger.info(f"Predictions saved to {output}")


@cli.command()
@click.option('--data-dir', '-d', default='data', help='Data directory')
@click.option('--train-file', '-t', default='train.csv', help='Training file name')
@click.option('--test-file', '-s', default='test.csv', help='Test file name')
@click.option('--target', help='Target column name (auto-detected if not specified)')
@click.option('--datetime-columns', '-dt', multiple=True, help='Datetime column names')
def validate(data_dir, train_file, test_file, target, datetime_columns):
    """Validate data for potential issues and leakage.
    
    Performs comprehensive validation to detect data quality issues
    and potential sources of data leakage that could compromise model
    validity.
    
    Example:
        \b
        # Basic validation
        $ tabml validate
        
        \b
        # With temporal validation
        $ tabml validate --datetime-columns date created_at updated_at
        
        \b
        # Full validation with target
        $ tabml validate --target price --datetime-columns listing_date
        
    Checks include:
        - Duplicate columns
        - Constant features
        - Perfect correlations with target
        - Temporal leakage in time series
        - Suspicious feature names
        - ID columns with patterns
        
    Warning:
        Data leakage can lead to overly optimistic performance
        that won't generalize to production!
    """
    from .data import DataLoader
    
    logger.info("Starting data validation...")
    
    # Load data
    loader = DataLoader(data_dir=data_dir)
    loader.load_data(
        train_file=train_file,
        test_file=test_file,
        target_column=target
    )
    
    # Run validation
    datetime_cols = list(datetime_columns) if datetime_columns else None
    validation_results = loader.validate_data(datetime_columns=datetime_cols)
    
    # Summary
    n_warnings = len(validation_results.get('leakage_warnings', []))
    if n_warnings > 0:
        logger.warning(f"Found {n_warnings} potential issues. See report above for details.")
    else:
        logger.info("No data leakage detected!")


@cli.command()
@click.option('--data-dir', '-d', default='data', help='Data directory')
def info(data_dir):
    """Show information about available datasets.
    
    Lists all CSV files in the data directory and subdirectories
    with file sizes.
    
    Example:
        \b
        # Show datasets in default data directory
        $ tabml info
        
        \b
        # Show datasets in custom directory
        $ tabml info --data-dir ./kaggle/competitions
        
    Output format:
        \b
        Datasets in ./data:
        ----------------------------------------
        
        house-prices/
          - train.csv (1.23 MB)
          - test.csv (0.45 MB)
          
        titanic/
          - train.csv (0.08 MB)
          - test.csv (0.03 MB)
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        click.echo(f"Data directory not found: {data_path}")
        return
        
    click.echo(f"\nDatasets in {data_path}:")
    click.echo("-" * 40)
    
    for subdir in data_path.iterdir():
        if subdir.is_dir():
            files = list(subdir.glob("*.csv"))
            if files:
                click.echo(f"\n{subdir.name}/")
                for f in files:
                    size = f.stat().st_size / 1024 / 1024  # MB
                    click.echo(f"  - {f.name} ({size:.2f} MB)")


def main():
    """Main entry point.
    
    Entry point for the TabML command-line interface.
    This function is called when running 'tabml' from the command line.
    """
    cli()


if __name__ == '__main__':
    main()