#!/usr/bin/env python
"""
Spaceship Titanic Competition Example using Real Data

This example demonstrates using TabML with the real Spaceship Titanic dataset.
You need to download the data from Kaggle first:
https://www.kaggle.com/c/spaceship-titanic/data

Place the following files in the data/ directory:
- train.csv
- test.csv

Author: William Guesdon
"""

import os
import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from tabml import (
    DataLoader, DataValidator, DataProcessor, Visualizer,
    FeatureEngineer, FeatureSelector,
    ModelTrainer, CrossValidator,
    XGBoostModel, LightGBMModel, CatBoostModel, RandomForestModel
)

warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "spaceship-titanic"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


class SpaceshipTitanicPipeline:
    """Complete ML pipeline for Spaceship Titanic competition using real data."""
    
    def __init__(self):
        """Initialize the pipeline."""
        self.train_path = DATA_DIR / "train.csv"
        self.test_path = DATA_DIR / "test.csv"
        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.predictions = {}
        self.data_processor = None
        self.feature_engineer = None
        self.feature_selector = None
        
    def check_data_files(self):
        """Check if data files exist."""
        if not self.train_path.exists() or not self.test_path.exists():
            print("\n" + "="*70)
            print("ERROR: Data files not found!")
            print("="*70)
            print("\nPlease download the Spaceship Titanic dataset from Kaggle:")
            print("https://www.kaggle.com/c/spaceship-titanic/data")
            print(f"\nExpected location: {DATA_DIR}/")
            print("- train.csv")
            print("- test.csv")
            print("\nThe data should be in: data/raw/spaceship-titanic/")
            print("\nOption 1: Use the download script")
            print("python examples/download_spaceship_data.py")
            print("\nOption 2: Download manually")
            print("1. Go to https://www.kaggle.com/c/spaceship-titanic/data")
            print("2. Download train.csv and test.csv")
            print("3. Place them in data/raw/spaceship-titanic/")
            print("\nOption 3: Use the synthetic data example")
            print("python examples/spaceship_titanic_complete.py")
            return False
        return True
        
    def load_data(self):
        """Load and validate the data."""
        print("\n" + "="*50)
        print("1. DATA LOADING AND VALIDATION")
        print("="*50)
        
        # Check if files exist
        if not self.check_data_files():
            return None
            
        # Load data
        print(f"\nLoading data from {DATA_DIR}/")
        self.train_df = pd.read_csv(self.train_path)
        self.test_df = pd.read_csv(self.test_path)
        
        # Validate data
        validator = DataValidator()
        validation_report = validator.validate_data(self.train_df, self.test_df)
        
        print(f"\nTraining data shape: {self.train_df.shape}")
        print(f"Test data shape: {self.test_df.shape}")
        print(f"\nData Validation Summary:")
        if 'basic' in validation_report:
            print(f"- Train shape: {validation_report['basic']['train_shape']}")
            print(f"- Test shape: {validation_report['basic']['test_shape']}")
            if validation_report['basic']['missing_in_test']:
                print(f"- Columns in train but not test: {validation_report['basic']['missing_in_test']}")
            if validation_report['basic']['missing_in_train']:
                print(f"- Columns in test but not train: {validation_report['basic']['missing_in_train']}")
        
        # Check for data leakage is handled within validate_data
        if 'leakage_warnings' in validation_report and validation_report['leakage_warnings']:
            print(f"\nWARNING: Potential data leakage detected!")
            for warning in validation_report['leakage_warnings']:
                print(f"  - {warning}")
        
        return self
    
    def perform_eda(self):
        """Perform exploratory data analysis."""
        print("\n" + "="*50)
        print("2. EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        visualizer = Visualizer()
        
        # 1. Target distribution
        if 'Transported' in self.train_df.columns:
            plt.figure(figsize=(8, 6))
            self.train_df['Transported'].value_counts().plot(kind='bar')
            plt.title('Target Distribution')
            plt.xlabel('Transported')
            plt.ylabel('Count')
            plt.xticks([0, 1], ['Not Transported', 'Transported'], rotation=0)
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / 'target_distribution.png')
            plt.close()
            
            print(f"\nTarget distribution:")
            print(self.train_df['Transported'].value_counts(normalize=True))
        
        # 2. Missing values analysis
        missing_df = pd.DataFrame({
            'Count': self.train_df.isnull().sum(),
            'Percentage': (self.train_df.isnull().sum() / len(self.train_df) * 100).round(2)
        }).sort_values('Count', ascending=False)
        missing_df = missing_df[missing_df['Count'] > 0]
        
        if len(missing_df) > 0:
            plt.figure(figsize=(10, 6))
            missing_df['Percentage'].plot(kind='barh')
            plt.title('Missing Values by Feature')
            plt.xlabel('Percentage (%)')
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / 'missing_values.png')
            plt.close()
            
            print(f"\nMissing values:")
            print(missing_df)
        
        # 3. Numeric features distribution
        numeric_cols = self.train_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(
                nrows=(len(numeric_cols) + 3) // 4,
                ncols=4,
                figsize=(16, 4 * ((len(numeric_cols) + 3) // 4))
            )
            axes = axes.flatten()
            
            for idx, col in enumerate(numeric_cols):
                if col != 'Transported':
                    self.train_df[col].hist(bins=30, ax=axes[idx])
                    axes[idx].set_title(f'Distribution of {col}')
                    axes[idx].set_xlabel(col)
                    axes[idx].set_ylabel('Frequency')
            
            # Hide unused subplots
            for idx in range(len(numeric_cols), len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / 'numeric_distributions.png')
            plt.close()
        
        # 4. Correlation analysis
        if len(numeric_cols) > 1:
            plt.figure(figsize=(12, 10))
            corr_matrix = self.train_df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=1, cbar_kws={"shrink": .8})
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / 'correlation_matrix.png')
            plt.close()
        
        print("\nEDA visualizations saved to outputs/")
        return self
    
    def engineer_features(self):
        """Perform feature engineering."""
        print("\n" + "="*50)
        print("3. FEATURE ENGINEERING")
        print("="*50)
        
        # Store passenger IDs for submission
        self.train_passenger_ids = self.train_df['PassengerId'].copy() if 'PassengerId' in self.train_df.columns else None
        self.test_passenger_ids = self.test_df['PassengerId'].copy() if 'PassengerId' in self.test_df.columns else None
        
        # Separate target
        if 'Transported' in self.train_df.columns:
            y_train_full = self.train_df['Transported'].astype(int)
            X_train_full = self.train_df.drop('Transported', axis=1)
        else:
            y_train_full = None
            X_train_full = self.train_df.copy()
        
        X_test_full = self.test_df.copy()
        
        # Custom feature engineering for Spaceship Titanic
        for df in [X_train_full, X_test_full]:
            # Extract features from Cabin
            if 'Cabin' in df.columns:
                df['Deck'] = df['Cabin'].str.extract(r'([A-Z])', expand=False)
                df['CabinNum'] = df['Cabin'].str.extract(r'(\d+)', expand=False).astype('float')
                df['Side'] = df['Cabin'].str.extract(r'([PS])$', expand=False)
            
            # Extract group information from PassengerId
            if 'PassengerId' in df.columns:
                df['Group'] = df['PassengerId'].str.split('_', expand=True)[0]
                
                # Calculate group size
                group_sizes = df.groupby('Group').size()
                df['GroupSize'] = df['Group'].map(group_sizes)
                df['IsAlone'] = (df['GroupSize'] == 1).astype(int)
            
            # Create spending features
            spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
            if all(col in df.columns for col in spending_cols):
                df['TotalSpending'] = df[spending_cols].sum(axis=1)
                df['HasSpending'] = (df['TotalSpending'] > 0).astype(int)
                df['LuxurySpending'] = df[['Spa', 'VRDeck']].sum(axis=1)
                df['BasicSpending'] = df[['RoomService', 'FoodCourt', 'ShoppingMall']].sum(axis=1)
                
                # Spending ratios
                df['LuxuryRatio'] = df['LuxurySpending'] / (df['TotalSpending'] + 1e-6)
                
                # Per-category spending indicators
                for col in spending_cols:
                    df[f'Has_{col}'] = (df[col] > 0).astype(int)
            
            # Age groups
            if 'Age' in df.columns:
                df['AgeGroup'] = pd.cut(df['Age'], 
                                       bins=[0, 12, 18, 25, 35, 50, 65, 100],
                                       labels=['Child', 'Teen', 'YoungAdult', 'Adult', 
                                              'MiddleAge', 'Senior', 'Elder'])
            
            # Interaction features
            if 'CryoSleep' in df.columns and 'VIP' in df.columns:
                # Convert boolean columns to numeric
                df['CryoSleep'] = df['CryoSleep'].astype(float)
                df['VIP'] = df['VIP'].astype(float)
                df['CryoVIP'] = df['CryoSleep'] * df['VIP']
        
        print(f"Original features: {self.train_df.shape[1]}")
        print(f"After feature engineering: {X_train_full.shape[1]}")
        
        # Print new features created
        new_features = set(X_train_full.columns) - set(self.train_df.columns)
        if new_features:
            print(f"\nNew features created ({len(new_features)}):")
            for feat in sorted(new_features):
                print(f"  - {feat}")
        
        # Split data for validation
        if y_train_full is not None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_train_full, y_train_full, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train_full
            )
        else:
            self.X_train = X_train_full
            self.y_train = None
        
        self.X_submission = X_test_full
        
        return self
    
    def preprocess_data(self):
        """Preprocess data using DataProcessor."""
        print("\n" + "="*50)
        print("4. DATA PREPROCESSING")
        print("="*50)
        
        # Configure DataProcessor
        processor_config = {
            'categorical_encoding': {
                'method': 'onehot',
                'handle_unknown': 'ignore',
                'high_cardinality_threshold': 50,
                'high_cardinality_method': 'target'
            },
            'text_processing': {
                'method': 'none'  # No text columns in this dataset
            },
            'scaling': {
                'method': 'standard'
            },
            'imputation': {
                'numeric_strategy': 'median',
                'categorical_strategy': 'most_frequent',
                'add_indicator': True
            },
            'drop_columns': ['PassengerId', 'Cabin', 'Name', 'Group'],
            'datetime_features': False
        }
        
        # Initialize and fit DataProcessor
        self.data_processor = DataProcessor(config=processor_config)
        
        # Fit on training data and transform
        self.X_train_processed = self.data_processor.fit_transform(self.X_train, self.y_train)
        self.X_test_processed = self.data_processor.transform(self.X_test)
        self.X_submission_processed = self.data_processor.transform(self.X_submission)
        
        print(f"\nPreprocessing complete:")
        print(f"  Train shape: {self.X_train.shape} -> {self.X_train_processed.shape}")
        print(f"  Test shape: {self.X_test.shape} -> {self.X_test_processed.shape}")
        print(f"  Submission shape: {self.X_submission.shape} -> {self.X_submission_processed.shape}")
        
        # Feature names
        feature_names = self.data_processor.get_feature_names_out()
        print(f"\nTotal features after preprocessing: {len(feature_names)}")
        
        # Show sample feature names
        print("\nSample processed features:")
        for i, name in enumerate(feature_names[:10]):
            print(f"  {i+1}. {name}")
        
        return self
    
    def train_models(self):
        """Train multiple models."""
        print("\n" + "="*50)
        print("5. MODEL TRAINING")
        print("="*50)
        
        trainer = ModelTrainer(task_type='classification')
        
        # Define models to train
        model_configs = {
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            },
            'lightgbm': {
                'n_estimators': 100,
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5
            },
            'catboost': {
                'iterations': 100,
                'depth': 6,
                'learning_rate': 0.1,
                'verbose': False
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 20,
                'min_samples_leaf': 10
            }
        }
        
        # Train each model
        for model_name, params in model_configs.items():
            print(f"\nTraining {model_name}...")
            
            model = trainer.train_model(
                model_type=model_name,
                X_train=self.X_train_processed,
                y_train=self.y_train,
                X_val=self.X_test_processed,
                y_val=self.y_test,
                params=params
            )
            
            # Make predictions
            y_pred = model.predict(self.X_test_processed)
            y_pred_proba = model.predict_proba(self.X_test_processed)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            auc_score = roc_auc_score(self.y_test, y_pred_proba)
            
            # Store model and results
            self.models[model_name] = {
                'model': model,
                'accuracy': accuracy,
                'auc': auc_score,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  AUC-ROC: {auc_score:.4f}")
        
        # Find best model
        best_model = max(self.models.items(), key=lambda x: x[1]['auc'])
        print(f"\nBest model: {best_model[0]} (AUC: {best_model[1]['auc']:.4f})")
        
        return self
    
    def optimize_models(self):
        """Optimize hyperparameters for best models."""
        print("\n" + "="*50)
        print("6. HYPERPARAMETER OPTIMIZATION")
        print("="*50)
        
        # Select top 2 models for optimization
        sorted_models = sorted(self.models.items(), key=lambda x: x[1]['auc'], reverse=True)[:2]
        
        trainer = ModelTrainer(task_type='classification')
        
        for model_name, model_info in sorted_models:
            print(f"\nOptimizing {model_name}...")
            
            # Train with optimization
            optimized_model = trainer.train_model(
                model_type=model_name,
                X_train=self.X_train_processed,
                y_train=self.y_train,
                X_val=self.X_test_processed,
                y_val=self.y_test,
                optimize_hyperparams=True,
                n_trials=20  # Limited trials for speed
            )
            
            # Evaluate optimized model
            y_pred_opt = optimized_model.predict(self.X_test_processed)
            y_pred_proba_opt = optimized_model.predict_proba(self.X_test_processed)[:, 1]
            
            accuracy_opt = accuracy_score(self.y_test, y_pred_opt)
            auc_opt = roc_auc_score(self.y_test, y_pred_proba_opt)
            
            print(f"  Original AUC: {model_info['auc']:.4f}")
            print(f"  Optimized AUC: {auc_opt:.4f}")
            print(f"  Improvement: {(auc_opt - model_info['auc']):.4f}")
            
            # Update model if improved
            if auc_opt > model_info['auc']:
                self.models[f"{model_name}_optimized"] = {
                    'model': optimized_model,
                    'accuracy': accuracy_opt,
                    'auc': auc_opt,
                    'predictions': y_pred_opt,
                    'probabilities': y_pred_proba_opt
                }
        
        return self
    
    def create_ensemble(self):
        """Create ensemble predictions."""
        print("\n" + "="*50)
        print("7. MODEL ENSEMBLING")
        print("="*50)
        
        # Select models for ensemble (top 3 by AUC)
        sorted_models = sorted(self.models.items(), key=lambda x: x[1]['auc'], reverse=True)[:3]
        
        print(f"\nEnsemble models:")
        ensemble_weights = []
        for model_name, model_info in sorted_models:
            weight = model_info['auc'] / sum(m[1]['auc'] for m in sorted_models)
            ensemble_weights.append(weight)
            print(f"  - {model_name}: weight={weight:.3f}, AUC={model_info['auc']:.4f}")
        
        # Create weighted average ensemble
        ensemble_proba = np.zeros_like(sorted_models[0][1]['probabilities'])
        for (model_name, model_info), weight in zip(sorted_models, ensemble_weights):
            ensemble_proba += weight * model_info['probabilities']
        
        # Convert to predictions
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        
        # Evaluate ensemble
        ensemble_accuracy = accuracy_score(self.y_test, ensemble_pred)
        ensemble_auc = roc_auc_score(self.y_test, ensemble_proba)
        
        print(f"\nEnsemble performance:")
        print(f"  Accuracy: {ensemble_accuracy:.4f}")
        print(f"  AUC-ROC: {ensemble_auc:.4f}")
        
        # Store ensemble
        self.models['ensemble'] = {
            'models': [m[0] for m in sorted_models],
            'weights': ensemble_weights,
            'accuracy': ensemble_accuracy,
            'auc': ensemble_auc,
            'predictions': ensemble_pred,
            'probabilities': ensemble_proba
        }
        
        # Compare with best single model
        best_single = max(self.models.items(), key=lambda x: x[1]['auc'] if x[0] != 'ensemble' else 0)
        print(f"\nBest single model: {best_single[0]} (AUC: {best_single[1]['auc']:.4f})")
        print(f"Ensemble improvement: {(ensemble_auc - best_single[1]['auc']):.4f}")
        
        return self
    
    def cross_validate(self):
        """Perform cross-validation."""
        print("\n" + "="*50)
        print("8. CROSS-VALIDATION")
        print("="*50)
        
        cv = CrossValidator()
        
        # Select best model for CV
        best_model_name = max(self.models.items(), key=lambda x: x[1]['auc'])[0]
        
        # Skip ensemble for CV
        if best_model_name == 'ensemble':
            model_names = [m for m in self.models.keys() if m != 'ensemble']
            best_model_name = max(model_names, key=lambda x: self.models[x]['auc'])
        
        print(f"\nPerforming 5-fold CV with {best_model_name}...")
        
        # Get the model
        best_model = self.models[best_model_name]['model']
        
        # Combine train and test for full CV
        X_full = pd.concat([self.X_train_processed, self.X_test_processed])
        y_full = pd.concat([self.y_train, self.y_test])
        
        # Perform stratified k-fold CV
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
            X_fold_train = X_full.iloc[train_idx]
            y_fold_train = y_full.iloc[train_idx]
            X_fold_val = X_full.iloc[val_idx]
            y_fold_val = y_full.iloc[val_idx]
            
            # Train model
            if best_model_name == 'xgboost' or 'xgboost' in best_model_name:
                fold_model = XGBoostModel()
            elif best_model_name == 'lightgbm' or 'lightgbm' in best_model_name:
                fold_model = LightGBMModel()
            elif best_model_name == 'catboost' or 'catboost' in best_model_name:
                fold_model = CatBoostModel()
            else:
                fold_model = RandomForestModel()
            
            fold_model.fit(X_fold_train, y_fold_train)
            
            # Predict
            y_pred = fold_model.predict_proba(X_fold_val)[:, 1]
            fold_auc = roc_auc_score(y_fold_val, y_pred)
            cv_scores.append(fold_auc)
            
            print(f"  Fold {fold + 1}: AUC = {fold_auc:.4f}")
        
        print(f"\nCV Results:")
        print(f"  Mean AUC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
        print(f"  Min AUC: {np.min(cv_scores):.4f}")
        print(f"  Max AUC: {np.max(cv_scores):.4f}")
        
        return self
    
    def generate_submission(self):
        """Generate competition submission file."""
        print("\n" + "="*50)
        print("9. GENERATING SUBMISSION")
        print("="*50)
        
        # Use ensemble for submission
        if 'ensemble' in self.models:
            print("\nUsing ensemble model for submission...")
            
            # Get ensemble models and weights
            ensemble_models = []
            for model_name in self.models['ensemble']['models']:
                ensemble_models.append(self.models[model_name]['model'])
            
            weights = self.models['ensemble']['weights']
            
            # Make predictions on submission data
            submission_proba = np.zeros(len(self.X_submission_processed))
            
            for model, weight in zip(ensemble_models, weights):
                pred_proba = model.predict_proba(self.X_submission_processed)[:, 1]
                submission_proba += weight * pred_proba
            
            submission_pred = (submission_proba > 0.5).astype(bool)
        else:
            # Use best single model
            best_model_name = max(self.models.items(), key=lambda x: x[1]['auc'])[0]
            print(f"\nUsing {best_model_name} for submission...")
            
            best_model = self.models[best_model_name]['model']
            submission_pred = best_model.predict(self.X_submission_processed)
            submission_pred = submission_pred.astype(bool)
        
        # Create submission DataFrame
        submission = pd.DataFrame({
            'PassengerId': self.test_passenger_ids,
            'Transported': submission_pred
        })
        
        # Save submission
        submission_path = OUTPUT_DIR / 'submission.csv'
        submission.to_csv(submission_path, index=False)
        
        print(f"\nSubmission saved to: {submission_path}")
        print(f"Submission shape: {submission.shape}")
        print(f"\nSubmission preview:")
        print(submission.head(10))
        print(f"\nPrediction distribution:")
        print(submission['Transported'].value_counts(normalize=True))
        
        return self
    
    def save_artifacts(self):
        """Save models and configurations."""
        print("\n" + "="*50)
        print("10. SAVING ARTIFACTS")
        print("="*50)
        
        # Save DataProcessor configuration
        self.data_processor.save_config(str(OUTPUT_DIR / 'data_processor_config.json'))
        print(f"Saved DataProcessor config to: {OUTPUT_DIR / 'data_processor_config.json'}")
        
        # Save feature importance plot for best model
        best_model_name = max(
            [(k, v) for k, v in self.models.items() if k != 'ensemble'], 
            key=lambda x: x[1]['auc']
        )[0]
        
        best_model = self.models[best_model_name]['model']
        if hasattr(best_model, 'feature_importances_'):
            feature_names = self.data_processor.get_feature_names_out()
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Plot top 20 features
            plt.figure(figsize=(10, 8))
            top_features = importance_df.head(20)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importance')
            plt.title(f'Top 20 Feature Importances ({best_model_name})')
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / 'feature_importance_final.png')
            plt.close()
            
            # Save importance CSV
            importance_df.to_csv(OUTPUT_DIR / 'feature_importances.csv', index=False)
            print(f"Saved feature importances to: {OUTPUT_DIR / 'feature_importances.csv'}")
        
        # Save model performance summary
        performance_summary = []
        for model_name, model_info in self.models.items():
            if model_name != 'ensemble':
                performance_summary.append({
                    'model': model_name,
                    'accuracy': model_info['accuracy'],
                    'auc': model_info['auc']
                })
        
        performance_df = pd.DataFrame(performance_summary).sort_values('auc', ascending=False)
        performance_df.to_csv(OUTPUT_DIR / 'model_performance.csv', index=False)
        print(f"Saved model performance to: {OUTPUT_DIR / 'model_performance.csv'}")
        
        print("\nAll artifacts saved successfully!")
        return self


def main():
    """Run the complete Spaceship Titanic pipeline with real data."""
    print("="*50)
    print("SPACESHIP TITANIC COMPLETE PIPELINE - REAL DATA")
    print("="*50)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize pipeline
    pipeline = SpaceshipTitanicPipeline()
    
    # Run complete pipeline
    result = pipeline.load_data()
    
    if result is None:
        print("\nPipeline terminated due to missing data files.")
        return
        
    (pipeline
        .perform_eda()
        .engineer_features()
        .preprocess_data()
        .train_models()
        .optimize_models()
        .create_ensemble()
        .cross_validate()
        .generate_submission()
        .save_artifacts()
    )
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nOutput files saved to: {OUTPUT_DIR}/")
    print("\nFiles generated:")
    for file in sorted(OUTPUT_DIR.glob("*")):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()
