"""
Spaceship Titanic Competition - Complete TabML Solution
========================================================

This script demonstrates a complete end-to-end solution for the Spaceship Titanic
Kaggle competition using TabML's full capabilities including:

1. Automated EDA and visualization
2. Advanced feature engineering
3. Multiple model training with hyperparameter optimization
4. OOF ensemble creation
5. Submission generation

Competition: https://www.kaggle.com/competitions/spaceship-titanic
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# TabML imports
from tabml import (
    DataLoader,
    FeatureEngineer,
    AdvancedFeatureEngineer,
    FeatureSelector,
    ModelTrainer,
    XGBoostModel,
    LightGBMModel,
    CatBoostModel,
    TabNetModel,
    VotingEnsemble,
    OOFEnsemble,
    AutoEnsemble,
    CrossValidator,
    Visualizer,
    DataValidator
)
from tabml.training import (
    EnhancedTrainer,
    EarlyStoppingCallback,
    ModelCheckpointCallback,
    ProgressBarCallback
)

# Standard imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import os

# Set paths - relative to the examples directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # tabml root
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "Spaceship_Titanic")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "spaceship_titanic")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")

# =============================================================================
# 1. DATA LOADING AND INITIAL EXPLORATION
# =============================================================================

def load_and_explore_data():
    """Load data and perform initial exploration."""
    print("="*70)
    print("1. LOADING DATA AND INITIAL EXPLORATION")
    print("="*70)
    
    # Load data
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    submission_df = pd.read_csv(SUBMISSION_PATH)
    
    print(f"\nTrain shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Submission shape: {submission_df.shape}")
    
    # Basic info
    print("\nTrain Data Info:")
    print(train_df.info())
    
    print("\nFirst few rows:")
    print(train_df.head())
    
    print("\nTarget distribution:")
    print(train_df['Transported'].value_counts(normalize=True))
    
    print("\nMissing values in train:")
    missing = train_df.isnull().sum()
    missing_pct = 100 * missing / len(train_df)
    missing_df = pd.DataFrame({'Missing': missing, 'Percentage': missing_pct})
    print(missing_df[missing_df['Missing'] > 0].sort_values('Percentage', ascending=False))
    
    return train_df, test_df, submission_df


# =============================================================================
# 2. AUTOMATED EDA WITH VISUALIZATIONS
# =============================================================================

def automated_eda(train_df, test_df=None):
    """Perform automated exploratory data analysis."""
    print("\n" + "="*70)
    print("2. AUTOMATED EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    # Create visualizer
    viz = Visualizer()
    
    # Set style
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        try:
            plt.style.use('seaborn-darkgrid')
        except:
            plt.style.use('default')
    
    # 1. Target distribution
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    train_df['Transported'].value_counts().plot(kind='bar')
    plt.title('Target Distribution')
    plt.xlabel('Transported')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    train_df['Transported'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Target Distribution (%)')
    plt.ylabel('')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'target_distribution.png'))
    plt.show()
    
    # 2. Numerical features distribution
    numerical_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove target if it exists in numerical columns (it's boolean, so it won't be there)
    if 'Transported' in numerical_cols:
        numerical_cols.remove('Transported')
    
    if len(numerical_cols) > 0:
        n_cols = min(4, len(numerical_cols))
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for idx, col in enumerate(numerical_cols):
            if idx < len(axes):
                train_df[col].hist(bins=30, ax=axes[idx], edgecolor='black')
                axes[idx].set_title(f'Distribution of {col}')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequency')
        
        # Hide extra subplots
        for idx in range(len(numerical_cols), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'numerical_distributions.png'))
        plt.show()
    
    # 3. Categorical features distribution
    categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_cols[:5]:  # Limit to first 5
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        train_df[col].value_counts()[:10].plot(kind='bar')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        # Relationship with target
        if 'Transported' in train_df.columns:
            pd.crosstab(train_df[col], train_df['Transported'], normalize='index').plot(kind='bar', stacked=True)
            plt.title(f'{col} vs Transported')
            plt.xlabel(col)
            plt.ylabel('Proportion')
            plt.legend(title='Transported')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{col}_distribution.png'))
        plt.show()
    
    # 4. Correlation matrix for numerical features
    if len(numerical_cols) > 1:
        plt.figure(figsize=(12, 8))
        correlation_matrix = train_df[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                    fmt='.2f', square=True)
        plt.title('Correlation Matrix - Numerical Features')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_matrix.png'))
        plt.show()
    
    # 5. Missing values heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(train_df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'missing_values_heatmap.png'))
    plt.show()
    
    print(f"\nEDA visualizations saved to {OUTPUT_DIR}")
    
    # Data validation - only if test_df is provided
    validation_results = {}
    if test_df is not None:
        validator = DataValidator()
        validation_results = validator.validate_data(train_df, test_df)
        print("\nData Validation Results:")
        print(f"  - Missing values: {validation_results.get('has_missing', False)}")
        print(f"  - Duplicates: {validation_results.get('has_duplicates', False)}")
        print(f"  - High cardinality: {validation_results.get('high_cardinality_features', [])}")
    
    return validation_results


# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================

def feature_engineering(train_df, test_df):
    """Perform comprehensive feature engineering."""
    print("\n" + "="*70)
    print("3. FEATURE ENGINEERING")
    print("="*70)
    
    # Combine for consistent preprocessing
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    
    # Store IDs and target
    train_ids = train_df['PassengerId']
    test_ids = test_df['PassengerId']
    y_train = train_df['Transported'].astype(int)
    
    # Combine data
    df = pd.concat([train_df, test_df], ignore_index=True)
    
    print(f"\nCombined shape: {df.shape}")
    
    # ==========================================
    # 3.1 Parse existing features
    # ==========================================
    print("\n3.1 Parsing existing features...")
    
    # Parse PassengerId to extract group info
    df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
    df['GroupSize'] = df.groupby('Group')['PassengerId'].transform('count')
    df['IsAlone'] = (df['GroupSize'] == 1).astype(int)
    
    # Parse Cabin to extract deck, number, and side
    df['Deck'] = df['Cabin'].apply(lambda x: x.split('/')[0] if pd.notna(x) else None)
    df['CabinNumber'] = df['Cabin'].apply(lambda x: int(x.split('/')[1]) if pd.notna(x) and len(x.split('/')) > 1 else None)
    df['Side'] = df['Cabin'].apply(lambda x: x.split('/')[2] if pd.notna(x) and len(x.split('/')) > 2 else None)
    
    # Parse Name to extract surname
    df['Surname'] = df['Name'].apply(lambda x: x.split(' ')[0] if pd.notna(x) else None)
    df['FamilySize'] = df.groupby('Surname')['PassengerId'].transform('count')
    
    # ==========================================
    # 3.2 Create new features
    # ==========================================
    print("\n3.2 Creating new features...")
    
    # Total spending across all amenities
    spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpending'] = df[spending_cols].sum(axis=1)
    df['AvgSpending'] = df[spending_cols].mean(axis=1)
    df['MaxSpending'] = df[spending_cols].max(axis=1)
    df['MinSpending'] = df[spending_cols].min(axis=1)
    df['SpendingStd'] = df[spending_cols].std(axis=1)
    
    # Number of amenities used
    df['AmenitiesUsed'] = (df[spending_cols] > 0).sum(axis=1)
    df['NoSpending'] = (df['TotalSpending'] == 0).astype(int)
    
    # Spending patterns
    df['LuxurySpending'] = df['Spa'] + df['VRDeck']
    df['BasicSpending'] = df['RoomService'] + df['FoodCourt']
    df['LuxuryRatio'] = df['LuxurySpending'] / (df['TotalSpending'] + 1)
    
    # Age groups
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                            labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
    df['IsChild'] = (df['Age'] < 18).astype(int)
    df['IsSenior'] = (df['Age'] >= 60).astype(int)
    
    # Cabin features
    df['CabinNumberGroup'] = pd.cut(df['CabinNumber'], bins=10, labels=False)
    
    # Interaction features
    df['CryoVIP'] = ((df['CryoSleep'] == True) & (df['VIP'] == True)).astype(int)
    df['CryoSpending'] = df['CryoSleep'].astype(float) * df['TotalSpending']
    
    # Family/Group features
    df['GroupSpending'] = df.groupby('Group')['TotalSpending'].transform('mean')
    df['FamilySpending'] = df.groupby('Surname')['TotalSpending'].transform('mean')
    
    # ==========================================
    # 3.3 Handle missing values intelligently
    # ==========================================
    print("\n3.3 Handling missing values...")
    
    # For CryoSleep passengers, spending should be 0
    cryo_mask = df['CryoSleep'] == True
    for col in spending_cols:
        df.loc[cryo_mask & df[col].isna(), col] = 0
    
    # Fill other numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numerical_cols:
        if col not in ['is_train', 'Transported']:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Fill categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
    
    # ==========================================
    # 3.4 Encode categorical variables
    # ==========================================
    print("\n3.4 Encoding categorical variables...")
    
    # Label encoding for ordinal features
    label_encoders = {}
    label_encode_cols = ['HomePlanet', 'Destination', 'Deck', 'Side', 'AgeGroup']
    
    for col in label_encode_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_Encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    # One-hot encoding for nominal features (selectively)
    onehot_cols = ['HomePlanet', 'Destination']
    df = pd.get_dummies(df, columns=onehot_cols, prefix=onehot_cols, drop_first=True)
    
    # Boolean to int
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    for col in bool_cols:
        df[col] = df[col].astype(int)
    
    # ==========================================
    # 3.5 Split back to train/test
    # ==========================================
    print("\n3.5 Splitting back to train/test...")
    
    train_df = df[df['is_train'] == 1].drop('is_train', axis=1)
    test_df = df[df['is_train'] == 0].drop('is_train', axis=1)
    
    # Drop columns not useful for modeling
    drop_cols = ['PassengerId', 'Name', 'Cabin', 'Transported', 'Group', 'Surname', 
                 'AgeGroup', 'Deck', 'Side']
    
    for col in drop_cols:
        if col in train_df.columns:
            train_df = train_df.drop(col, axis=1)
        if col in test_df.columns:
            test_df = test_df.drop(col, axis=1)
    
    print(f"\nFinal train shape: {train_df.shape}")
    print(f"Final test shape: {test_df.shape}")
    print(f"Features created: {list(train_df.columns)[:20]}...")
    
    return train_df, test_df, y_train, train_ids, test_ids


# =============================================================================
# 4. ADVANCED FEATURE ENGINEERING WITH TABML
# =============================================================================

def advanced_feature_engineering(train_df, test_df, y_train):
    """Apply TabML's advanced feature engineering."""
    print("\n" + "="*70)
    print("4. ADVANCED FEATURE ENGINEERING WITH TABML")
    print("="*70)
    
    # Initialize advanced feature engineer
    adv_engineer = AdvancedFeatureEngineer()
    
    # 1. Create polynomial features for top numerical columns
    print("\n4.1 Creating polynomial features...")
    numerical_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Select top columns based on variance
    variances = train_df[numerical_cols].var()
    top_cols = variances.nlargest(5).index.tolist()
    
    train_df = adv_engineer.create_polynomial_features(train_df, numeric_columns=top_cols, degree=2)
    test_df = adv_engineer.create_polynomial_features(test_df, numeric_columns=top_cols, degree=2)
    
    # 2. Create interaction features
    print("\n4.2 Creating interaction features...")
    train_df = adv_engineer.create_interaction_features(train_df, numeric_columns=top_cols, max_interactions=20)
    test_df = adv_engineer.create_interaction_features(test_df, numeric_columns=top_cols, max_interactions=20)
    
    # 3. Feature selection
    print("\n4.3 Selecting best features...")
    from tabml import FeatureSelector
    
    selector = FeatureSelector(method='tree_based', n_features=100)
    
    # Fit on train data
    train_df_selected = selector.fit_transform(train_df, y_train)
    test_df_selected = selector.transform(test_df)
    
    print(f"\nSelected {train_df_selected.shape[1]} features from {train_df.shape[1]}")
    
    # Get feature importance
    importance_df = selector.get_feature_importance()
    print("\nTop 10 most important features:")
    print(importance_df.head(10))
    
    return train_df_selected, test_df_selected, selector


# =============================================================================
# 5. MODEL TRAINING WITH HYPERPARAMETER OPTIMIZATION
# =============================================================================

def train_models(X_train, y_train, X_val=None, y_val=None):
    """Train multiple models with hyperparameter optimization."""
    print("\n" + "="*70)
    print("5. MODEL TRAINING WITH HYPERPARAMETER OPTIMIZATION")
    print("="*70)
    
    # Split for validation if not provided
    if X_val is None:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
    
    # Initialize model trainer
    trainer = ModelTrainer(
        task_type='classification',
        metric='roc_auc',
        cv_strategy='stratified',
        n_folds=5,
        gpu=True  # Use GPU if available
    )
    
    # Models to train
    models_config = {
        'lightgbm': {
            'optimize': True,
            'n_trials': 100,
            'params': {
                'n_estimators': 500,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
        },
        'xgboost': {
            'optimize': True,
            'n_trials': 100,
            'params': {
                'n_estimators': 500,
                'learning_rate': 0.05,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
        },
        'catboost': {
            'optimize': True,
            'n_trials': 50,
            'params': {
                'iterations': 500,
                'learning_rate': 0.05,
                'depth': 6,
                'l2_leaf_reg': 3
            }
        }
    }
    
    # Add TabNet if available
    try:
        models_config['tabnet'] = {
            'optimize': False,  # TabNet optimization can be slow
            'params': {
                'n_d': 32,
                'n_a': 32,
                'n_steps': 5,
                'gamma': 1.5,
                'lambda_sparse': 1e-4
            }
        }
        print("\nTabNet available and will be trained")
    except:
        print("\nTabNet not available (install with: pip install pytorch-tabnet)")
    
    # Train models
    trained_models = {}
    
    for model_name, config in models_config.items():
        print(f"\n5.{list(models_config.keys()).index(model_name)+1} Training {model_name.upper()}...")
        
        try:
            model = trainer.train_model(
                model_name,
                X_train, y_train,
                X_val, y_val,
                params=config['params'],
                optimize_hyperparams=config['optimize'],
                n_trials=config.get('n_trials', 50)
            )
            
            trained_models[model_name] = model
            
            # Get CV score
            if model_name in trainer.cv_scores:
                print(f"   CV Score: {trainer.cv_scores[model_name]:.4f}")
            
        except Exception as e:
            print(f"   Error training {model_name}: {str(e)}")
            continue
    
    print(f"\nSuccessfully trained {len(trained_models)} models")
    
    return trained_models, trainer


# =============================================================================
# 6. OOF ENSEMBLE CREATION
# =============================================================================

def create_ensemble(models, X_train, y_train, X_test):
    """Create advanced ensemble using OOF predictions."""
    print("\n" + "="*70)
    print("6. CREATING OOF ENSEMBLE")
    print("="*70)
    
    # Create OOF ensemble
    ensemble = OOFEnsemble(task_type='classification', metric='roc_auc')
    
    # Get model list
    model_list = list(models.values())
    
    print(f"\n6.1 Generating OOF predictions for {len(model_list)} models...")
    oof_predictions = ensemble.get_oof_predictions(
        models=model_list,
        X=X_train,
        y=y_train,
        n_folds=5,
        stratified=True,
        verbose=True
    )
    
    # Try different ensemble strategies
    print("\n6.2 Testing ensemble strategies...")
    
    # Strategy 1: Optimized weights
    print("\n   a) Optimizing weights with Optuna...")
    weights = ensemble.optimize_weights(
        oof_predictions, y_train,
        method='optuna',
        n_trials=200
    )
    
    # Strategy 2: Stacking
    print("\n   b) Fitting stacking ensemble...")
    stacking_ensemble = OOFEnsemble(task_type='classification')
    stacking_ensemble.fit_stacking(oof_predictions, y_train)
    
    # Strategy 3: Auto ensemble
    print("\n   c) Auto ensemble selection...")
    auto_ensemble = AutoEnsemble(task_type='classification')
    auto_ensemble.fit(
        models=model_list,
        X=X_train,
        y=y_train,
        strategies=['weighted', 'stacking', 'rank'],
        cv_folds=5
    )
    
    print(f"\n   Best strategy: {auto_ensemble.best_strategy}")
    print(f"   Best score: {auto_ensemble.ensemble_scores[auto_ensemble.best_strategy]:.4f}")
    
    # Train models on full data for final predictions
    print("\n6.3 Training models on full training data...")
    for name, model in models.items():
        print(f"   Training {name}...")
        model.fit(X_train, y_train)
    
    # Generate test predictions
    print("\n6.4 Generating test predictions...")
    test_predictions = ensemble.get_test_predictions(model_list, X_test)
    
    # Create final predictions using best strategy
    if auto_ensemble.best_strategy == 'weighted':
        final_predictions = np.average(test_predictions.values, weights=weights, axis=1)
    elif auto_ensemble.best_strategy == 'stacking':
        final_predictions = stacking_ensemble.predict_stacking(test_predictions)
    else:  # rank
        final_predictions = ensemble.rank_average([test_predictions[col].values for col in test_predictions.columns])
    
    # Also create a simple average for comparison
    simple_avg = test_predictions.mean(axis=1)
    
    return final_predictions, simple_avg, ensemble, auto_ensemble


# =============================================================================
# 7. GENERATE SUBMISSION
# =============================================================================

def generate_submission(predictions, test_ids, submission_df):
    """Generate submission file for Kaggle."""
    print("\n" + "="*70)
    print("7. GENERATING SUBMISSION")
    print("="*70)
    
    # Convert probabilities to boolean
    predictions_bool = (predictions > 0.5)
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'PassengerId': test_ids,
        'Transported': predictions_bool
    })
    
    # Ensure correct format
    submission['Transported'] = submission['Transported'].map({True: 'True', False: 'False'})
    
    # Save submission
    submission_path = os.path.join(OUTPUT_DIR, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    print(f"\nSubmission saved to '{submission_path}'")
    
    # Display first few predictions
    print("\nFirst 10 predictions:")
    print(submission.head(10))
    
    # Display prediction distribution
    print("\nPrediction distribution:")
    print(submission['Transported'].value_counts(normalize=True))
    
    return submission


# =============================================================================
# 8. MAIN EXECUTION PIPELINE
# =============================================================================

def main():
    """Main execution pipeline."""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║           SPACESHIP TITANIC COMPETITION SOLUTION                 ║
    ║                     Using TabML Framework                        ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  This solution demonstrates TabML's full capabilities:           ║
    ║  • Automated EDA with visualizations                            ║
    ║  • Advanced feature engineering                                 ║
    ║  • Multiple model training with Optuna optimization             ║
    ║  • OOF ensemble creation with multiple strategies               ║
    ║  • Automatic ensemble selection                                 ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # 1. Load data
    train_df, test_df, submission_df = load_and_explore_data()
    
    # 2. Automated EDA (pass test_df for validation)
    validation_results = automated_eda(train_df, test_df)
    
    # 3. Feature engineering
    X_train, X_test, y_train, train_ids, test_ids = feature_engineering(train_df, test_df)
    
    # 4. Advanced feature engineering
    X_train, X_test, selector = advanced_feature_engineering(X_train, X_test, y_train)
    
    # 5. Model training
    models, trainer = train_models(X_train, y_train)
    
    # 6. Create ensemble
    final_predictions, simple_predictions, ensemble, auto_ensemble = create_ensemble(
        models, X_train, y_train, X_test
    )
    
    # 7. Generate submissions
    print("\n" + "="*70)
    print("GENERATING MULTIPLE SUBMISSIONS")
    print("="*70)
    
    # Best ensemble submission
    submission_best = generate_submission(final_predictions, test_ids, submission_df)
    submission_best.to_csv(os.path.join(OUTPUT_DIR, 'submission_best_ensemble.csv'), index=False)
    print(f"\nBest ensemble saved to '{os.path.join(OUTPUT_DIR, 'submission_best_ensemble.csv')}'")
    
    # Simple average submission
    submission_simple = generate_submission(simple_predictions, test_ids, submission_df)
    submission_simple.to_csv(os.path.join(OUTPUT_DIR, 'submission_simple_avg.csv'), index=False)
    print(f"Simple average saved to '{os.path.join(OUTPUT_DIR, 'submission_simple_avg.csv')}'")
    
    print("\n" + "="*70)
    print("SOLUTION COMPLETE!")
    print("="*70)
    print("""
    Results Summary:
    ----------------
    • Models trained: {}
    • Best ensemble strategy: {}
    • Features used: {}
    • Submissions generated: 2
    
    Files created:
    -------------
    • submission_best_ensemble.csv - Best performing ensemble
    • submission_simple_avg.csv - Simple average ensemble
    • Multiple visualization PNG files
    
    Next steps:
    ----------
    1. Review the visualizations to understand the data
    2. Submit 'submission_best_ensemble.csv' to Kaggle
    3. Iterate on feature engineering if needed
    4. Try different model combinations
    
    Good luck with the competition!
    """.format(
        len(models),
        auto_ensemble.best_strategy if auto_ensemble else "N/A",
        X_train.shape[1]
    ))
    
    return models, ensemble, auto_ensemble


# =============================================================================
# EXECUTE
# =============================================================================

if __name__ == "__main__":
    # Run the complete pipeline
    models, ensemble, auto_ensemble = main()
    
    print("\n✅ Pipeline execution completed successfully!")
    print("📊 Check the generated visualizations and submission files")