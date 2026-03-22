"""
Enhanced TabML Usage Example
============================

This example demonstrates how to use the enhanced TabML library that now matches
and exceeds the capabilities of AbdML, including:

1. Advanced models (TabNet, Voting Ensembles, Ridge/Linear)
2. Advanced feature engineering (TF-IDF, date features, text statistics)
3. Custom metrics and weighted metrics
4. Multiple CV strategies
5. GPU support
6. Callbacks and monitoring
7. Optuna hyperparameter optimization
"""

import pandas as pd
import numpy as np
from tabml import (
    DataLoader, 
    FeatureEngineer, 
    AdvancedFeatureEngineer,
    ModelTrainer,
    TabularPipeline,
    XGBoostModel, 
    LightGBMModel, 
    CatBoostModel,
    TabNetModel,
    VotingEnsemble,
    RidgeModel
)
from tabml.training import (
    EnhancedTrainer,
    EarlyStoppingCallback,
    ModelCheckpointCallback,
    ProgressBarCallback
)

# =============================================================================
# 1. BASIC USAGE - Quick Start
# =============================================================================

def basic_example():
    """Basic usage similar to AbdML but with more features."""
    
    # Initialize pipeline
    pipeline = TabularPipeline(data_dir="./data")
    
    # Load data
    pipeline.load_data(
        train_file="train.csv",
        test_file="test.csv",
        target_column="target"  # Auto-detected if not specified
    )
    
    # Run complete pipeline with advanced features
    submission = pipeline.run_full_pipeline(
        feature_engineering={
            'numeric_impute': 'median',
            'scaling': 'robust',
            'categorical_encoding': 'target',  # Target encoding like AbdML
            'create_interactions': True,
            'create_polynomial': True
        },
        feature_selection={
            'method': 'mutual_info',
            'n_features': 0.8  # Keep 80% of features
        },
        model_training={
            'model_types': ['xgboost', 'lightgbm', 'catboost', 'tabnet'],
            'optimize_hyperparams': True,  # Optuna optimization
            'cv_folds': 5
        }
    )
    
    return submission


# =============================================================================
# 2. ADVANCED MODELS - TabNet and Voting Ensembles
# =============================================================================

def advanced_models_example(X_train, y_train, X_val, y_val):
    """Example using TabNet and Voting Ensembles."""
    
    # Initialize trainer with GPU support
    trainer = ModelTrainer(
        task_type='classification',
        metric='roc_auc',
        cv_strategy='stratified',
        n_folds=5,
        gpu=True  # Enable GPU when available
    )
    
    # Train TabNet model (neural network for tabular data)
    tabnet_model = trainer.train_model(
        'tabnet',
        X_train, y_train, X_val, y_val,
        params={
            'n_d': 64,
            'n_a': 64,
            'n_steps': 5,
            'gamma': 1.5,
            'lambda_sparse': 1e-3
        },
        optimize_hyperparams=True,
        n_trials=50
    )
    
    # Train other models
    xgb_model = trainer.train_model('xgboost', X_train, y_train, X_val, y_val)
    lgb_model = trainer.train_model('lightgbm', X_train, y_train, X_val, y_val)
    cat_model = trainer.train_model('catboost', X_train, y_train, X_val, y_val)
    
    # Create voting ensemble
    ensemble = VotingEnsemble(
        models=[xgb_model, lgb_model, cat_model, tabnet_model],
        voting='soft',  # Soft voting for probabilities
        weights=[0.3, 0.3, 0.2, 0.2]  # Custom weights
    )
    ensemble.fit(X_train, y_train, X_val, y_val)
    
    # Make predictions
    predictions = ensemble.predict_proba(X_val)[:, 1]
    
    return ensemble, predictions


# =============================================================================
# 3. ADVANCED FEATURE ENGINEERING
# =============================================================================

def advanced_features_example(df):
    """Example of advanced feature engineering."""
    
    engineer = AdvancedFeatureEngineer()
    
    # 1. Date features with cyclical encoding
    if 'date' in df.columns:
        df = engineer.create_date_features(
            df,
            date_columns=['date'],
            include_cyclical=True,  # Sin/cos transformations
            include_lag=True,       # Lag features
            drop_original=False
        )
    
    # 2. TF-IDF features for text columns
    if 'description' in df.columns:
        df = engineer.create_tfidf_features(
            df,
            text_columns=['description'],
            max_features=500,
            n_components=20,  # SVD for dimensionality reduction
            ngram_range=(1, 2),  # Unigrams and bigrams
            analyzer='word'
        )
    
    # 3. Text statistics features
    if 'review' in df.columns:
        df = engineer.create_text_features(
            df,
            text_columns=['review']
        )
        # Creates features like word_count, punctuation_count, 
        # lexical_diversity, stopword_ratio, etc.
    
    # 4. Interaction features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:5]
    df = engineer.create_interaction_features(
        df,
        numeric_columns=numeric_cols,
        max_interactions=20
    )
    
    # 5. Polynomial features
    df = engineer.create_polynomial_features(
        df,
        numeric_columns=numeric_cols,
        degree=2
    )
    
    # 6. Aggregate features (if you have grouping columns)
    if 'category' in df.columns and 'price' in df.columns:
        df = engineer.create_aggregate_features(
            df,
            group_columns=['category'],
            agg_columns=['price'],
            agg_functions=['mean', 'std', 'min', 'max']
        )
    
    return df


# =============================================================================
# 4. CUSTOM METRICS AND CV STRATEGIES
# =============================================================================

def custom_metrics_example(X_train, y_train):
    """Example with custom metrics and CV strategies."""
    
    # Define custom metric
    def custom_metric(y_true, y_pred):
        """Custom evaluation metric."""
        # Example: Weighted accuracy with more weight on positive class
        from sklearn.metrics import balanced_accuracy_score
        return balanced_accuracy_score(y_true, y_pred.round())
    
    # Trainer with custom metric
    trainer = ModelTrainer(
        task_type='classification',
        metric=custom_metric,  # Custom metric function
        cv_strategy='repeated_stratified',  # Advanced CV
        n_folds=5,
        random_state=42
    )
    
    # Train with different CV strategies
    cv_strategies = [
        'stratified',          # StratifiedKFold
        'kfold',              # Regular KFold
        'repeated',           # RepeatedKFold
        'repeated_stratified', # RepeatedStratifiedKFold
        'timeseries'          # TimeSeriesSplit
    ]
    
    for strategy in cv_strategies:
        trainer.cv_strategy = strategy
        model = trainer.train_model(
            'lightgbm',
            X_train, y_train,
            optimize_hyperparams=True
        )
        print(f"CV Strategy: {strategy}, Score: {trainer.cv_scores.get('lightgbm', 'N/A')}")
    
    return trainer


# =============================================================================
# 5. TRAINING WITH CALLBACKS
# =============================================================================

def callbacks_example(X_train, y_train, X_val, y_val):
    """Example using training callbacks for monitoring."""
    
    # Create trainer with callbacks
    trainer = EnhancedTrainer(
        callbacks=[
            # Early stopping to prevent overfitting
            EarlyStoppingCallback(
                patience=10,
                min_delta=0.0001,
                mode='max',
                restore_best_weights=True
            ),
            
            # Save model checkpoints
            ModelCheckpointCallback(
                filepath='./checkpoints/model_epoch_{epoch}.pkl',
                monitor='val_score',
                save_best_only=True
            ),
            
            # Progress bar for training
            ProgressBarCallback()
        ]
    )
    
    # Train model with callbacks
    model = XGBoostModel(params={'n_estimators': 1000})
    trained_model = trainer.fit(
        model,
        X_train, y_train,
        X_val, y_val,
        epochs=100
    )
    
    # Save training history
    trainer.save_history('./training_history.json')
    
    return trained_model


# =============================================================================
# 6. FULL PIPELINE MATCHING AbdML
# =============================================================================

def full_pipeline_like_abdml():
    """Complete pipeline matching AbdML functionality."""
    
    # Load data
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    target_column = 'target'
    
    # Initialize advanced feature engineer
    adv_engineer = AdvancedFeatureEngineer()
    
    # Add date features if date columns exist
    date_cols = train_data.select_dtypes(include=['datetime64']).columns.tolist()
    if date_cols:
        train_data = adv_engineer.create_date_features(train_data, date_cols)
        test_data = adv_engineer.create_date_features(test_data, date_cols)
    
    # Add TF-IDF features for text columns
    text_cols = train_data.select_dtypes(include=['object']).columns.tolist()
    text_cols = [col for col in text_cols if col != target_column]
    
    if text_cols:
        # Check if columns contain actual text (not just categories)
        for col in text_cols:
            avg_length = train_data[col].astype(str).str.len().mean()
            if avg_length > 20:  # Likely text, not category
                train_data, test_data = adv_engineer.create_multi_column_tfidf(
                    train_data, test_data,
                    text_columns=[col],
                    max_features=100,
                    analyzer='word'
                )
    
    # Split features and target
    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    X_test = test_data
    
    # Feature engineering
    engineer = FeatureEngineer(
        numeric_impute_strategy='median',
        categorical_encoding='target',  # Target encoding like AbdML
        scaling_method='standard',
        create_interactions=True
    )
    
    X_train = engineer.fit_transform(X_train, y_train)
    X_test = engineer.transform(X_test)
    
    # Model training with multiple models
    trainer = ModelTrainer(
        task_type='classification' if y_train.nunique() < 100 else 'regression',
        metric='roc_auc' if y_train.nunique() == 2 else 'rmse',
        cv_strategy='stratified' if y_train.nunique() < 100 else 'kfold',
        n_folds=5,
        gpu=True  # Use GPU if available
    )
    
    # Train multiple models with Optuna optimization
    model_types = ['lgbm', 'xgboost', 'catboost', 'tabnet']
    
    for model_type in model_types:
        print(f"\nTraining {model_type.upper()}...")
        model = trainer.train_model(
            model_type,
            X_train, y_train,
            optimize_hyperparams=True,
            n_trials=100
        )
        print(f"{model_type.upper()} CV Score: {trainer.cv_scores.get(model_type, 'N/A')}")
    
    # Create voting ensemble with all models
    all_models = list(trainer.models.values())
    ensemble = VotingEnsemble(
        models=all_models,
        voting='soft' if trainer.task_type == 'classification' else None,
        weights=None  # Equal weights
    )
    ensemble.fit(X_train, y_train)
    
    # Make predictions
    if trainer.task_type == 'classification':
        predictions = ensemble.predict_proba(X_test)[:, 1]
    else:
        predictions = ensemble.predict(X_test)
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_data.index,
        'prediction': predictions
    })
    
    submission.to_csv('submission.csv', index=False)
    print("\nSubmission saved to submission.csv")
    
    return submission, trainer


# =============================================================================
# 7. COMPARISON WITH AbdML
# =============================================================================

def compare_with_abdml():
    """
    Feature comparison between AbdML and Enhanced TabML:
    
    AbdML Features:
    ✅ LGBM, XGB, CatBoost - Available in TabML
    ✅ TabNet - Now available in TabML
    ✅ Voting Ensemble - Now available in TabML
    ✅ Ridge/Linear Regression - Now available in TabML
    ✅ Optuna optimization - Available in TabML
    ✅ Multiple CV strategies - Available in TabML
    ✅ Custom metrics - Available in TabML
    ✅ TF-IDF features - Available in TabML (AdvancedFeatureEngineer)
    ✅ Date features - Available in TabML (AdvancedFeatureEngineer)
    ✅ Text statistics - Available in TabML (AdvancedFeatureEngineer)
    ✅ Target encoding - Available in TabML
    ✅ Label encoding - Available in TabML
    ✅ One-hot encoding - Available in TabML
    ✅ GPU support - Available in TabML
    
    Additional TabML Features (not in AbdML):
    ✅ Modular architecture - Better code organization
    ✅ Training callbacks - Early stopping, checkpointing, progress bars
    ✅ TensorBoard/W&B integration - Via callbacks
    ✅ More feature engineering options - Polynomial, interactions, aggregates
    ✅ Feature selection methods - Multiple algorithms
    ✅ Data validation - Built-in validation tools
    ✅ Visualization tools - EDA and model interpretation
    ✅ Time series support - Specialized loaders and features
    ✅ Better documentation - Comprehensive docstrings
    ✅ CLI interface - Command-line tools
    """
    print(__doc__)
    
    # Example showing TabML is more capable
    print("\n" + "="*50)
    print("Enhanced TabML Capabilities Demo")
    print("="*50)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'numeric_1': np.random.randn(n_samples),
        'numeric_2': np.random.randn(n_samples),
        'category_1': np.random.choice(['A', 'B', 'C'], n_samples),
        'category_2': np.random.choice(['X', 'Y', 'Z'], n_samples),
        'text_col': ['sample text ' * np.random.randint(1, 5) for _ in range(n_samples)],
        'date_col': pd.date_range('2020-01-01', periods=n_samples, freq='H'),
        'target': np.random.randint(0, 2, n_samples)
    })
    
    # Split data
    train_idx = int(0.8 * n_samples)
    train_data = data[:train_idx].copy()
    test_data = data[train_idx:].drop(columns=['target']).copy()
    
    print("\n1. Advanced Feature Engineering:")
    engineer = AdvancedFeatureEngineer()
    
    # Date features
    train_data = engineer.create_date_features(
        train_data, ['date_col'], include_cyclical=True
    )
    print(f"   - Created {len([c for c in train_data.columns if 'date_col' in c])} date features")
    
    # Text features
    train_data = engineer.create_text_features(train_data, ['text_col'])
    print(f"   - Created {len([c for c in train_data.columns if 'text_col' in c])} text features")
    
    print("\n2. Model Training with GPU Support:")
    X = train_data.drop(columns=['target'])
    y = train_data['target']
    
    trainer = ModelTrainer(
        task_type='classification',
        metric='roc_auc',
        gpu=True
    )
    
    print("   - Training LightGBM with Optuna...")
    model = trainer.train_model(
        'lightgbm', X, y,
        optimize_hyperparams=True,
        n_trials=10  # Quick demo
    )
    
    print("\n3. Advanced Models:")
    print("   - TabNet (Neural Network for Tabular)")
    print("   - Voting Ensemble")
    print("   - Ridge/Linear Models")
    
    print("\n✅ TabML now has all AbdML features and more!")
    print("✅ Better architecture and extensibility")
    print("✅ Production-ready with monitoring and callbacks")


if __name__ == "__main__":
    # Run comparison
    compare_with_abdml()
    
    print("\n" + "="*50)
    print("TabML is now enhanced with all AbdML capabilities!")
    print("Use any of the examples above to get started.")
    print("="*50)