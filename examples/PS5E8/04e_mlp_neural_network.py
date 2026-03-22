"""
PS5E8 Competition - MLP Neural Network (Following NN_by_GPT5 approach)
Train MLP neural network using the SAME feature engineering as XGBoost
Key insight: Neural networks perform best with tree-based model features
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import tabml
sys.path.append(str(Path(__file__).parent.parent.parent))

from tabml import (
    FeatureEngineer, OOFManager, MLflowTracker
)

# Load environment variables
load_dotenv()

# Setup paths
DATA_DIR = Path("../../data/raw/PS5E8")
ORIGINAL_DATA_PATH = Path("../../data/raw/PS5E8/original_data.csv")
OUTPUT_DIR = Path("output")
OOF_DIR = OUTPUT_DIR / "oof_predictions_advanced"
SUBMISSION_DIR = OUTPUT_DIR / "submissions_advanced"

# Create directories
OOF_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

# Competition settings
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
N_FOLDS = 5
TARGET_COL = 'y'


def load_and_engineer_features():
    """Load data and apply THE SAME feature engineering as XGBoost.
    
    This follows the NN_by_GPT5 approach: use XGBoost features for neural networks.
    """
    logger.info("Loading competition data...")
    
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    
    logger.info(f"Train shape: {train_df.shape}")
    logger.info(f"Test shape: {test_df.shape}")
    
    # Separate features and target
    X_train = train_df.drop([TARGET_COL, 'id'], axis=1)
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop('id', axis=1)
    
    # Use EXACTLY the same feature engineering as XGBoost
    logger.info("Applying XGBoost feature engineering for MLP...")
    engineer = FeatureEngineer(
        categorical_impute_strategy='constant',
        numeric_impute_strategy='median',
        categorical_encoding='target',
        scaling_method='standard',
        create_interactions=True,
        create_polynomial=True,
        max_cardinality=20,
        min_frequency=0.01
    )
    
    X_train_fe = engineer.fit_transform(X_train, y_train)
    X_test_fe = engineer.transform(X_test)
    
    # Remove duplicate columns
    X_train_fe = X_train_fe.loc[:, ~X_train_fe.columns.duplicated()]
    X_test_fe = X_test_fe.loc[:, ~X_test_fe.columns.duplicated()]
    
    # Critical for neural networks: normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_fe)
    X_test_scaled = scaler.transform(X_test_fe)
    
    logger.info(f"Final train shape: {X_train_scaled.shape}")
    logger.info(f"Final test shape: {X_test_scaled.shape}")
    
    return X_train_scaled, y_train, X_test_scaled, sample_sub


def create_mlp_model(input_dim):
    """Create MLP model as suggested by GPT5 in the notebook.
    
    Architecture based on NN_by_GPT5 recommendations for tabular data.
    """
    import torch
    import torch.nn as nn
    
    class MLP(nn.Module):
        def __init__(self, input_dim):
            super(MLP, self).__init__()
            
            # Architecture inspired by NN_by_GPT5
            # Gradually decreasing layer sizes with dropout
            self.layers = nn.Sequential(
                # Input layer
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                # Hidden layer 1
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                # Hidden layer 2
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.2),
                
                # Hidden layer 3
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                
                # Output layer
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            return self.layers(x).squeeze()
    
    return MLP(input_dim)


def train_mlp_model(X_train, y_train, X_test, sample_sub):
    """Train MLP model with cross-validation."""
    logger.info("\n" + "="*60)
    logger.info("TRAINING MLP NEURAL NETWORK")
    logger.info("="*60)
    
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
    except ImportError:
        logger.warning("PyTorch is required for MLP. Install with:")
        logger.warning("pip install torch")
        logger.warning("Skipping MLP training...")
        return {}, 0.0
    
    # Initialize OOF manager
    oof_manager = OOFManager(output_dir=str(OOF_DIR))
    
    # Initialize MLflow tracker
    mlflow_tracker = None
    if os.getenv("MLFLOW_TRACKING_URI"):
        mlflow_tracker = MLflowTracker(
            experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "PS5E8-MLP"),
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
            tags={'stage': 'mlp_training', 'competition': 'PS5E8'}
        )
        mlflow_tracker.start_run(run_name="PS5E8_MLP_Neural_Network")
    
    # Prepare for cross-validation
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train.values)
    X_test_tensor = torch.FloatTensor(X_test)
    
    # Training parameters (following NN_by_GPT5 suggestions)
    batch_size = 512
    learning_rate = 0.001
    n_epochs = 50
    patience = 10
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        logger.info(f"\n--- Training Fold {fold}/{N_FOLDS} ---")
        
        # Split data
        X_fold_train = X_train_tensor[train_idx]
        y_fold_train = y_train_tensor[train_idx]
        X_fold_val = X_train_tensor[val_idx]
        y_fold_val = y_train_tensor[val_idx]
        
        # Create data loaders
        train_dataset = TensorDataset(X_fold_train, y_fold_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        model = create_mlp_model(X_train.shape[1]).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        best_val_score = 0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(n_epochs):
            # Training phase
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_fold_val.to(device))
                val_preds_np = val_outputs.cpu().numpy()
                val_score = roc_auc_score(y_fold_val.numpy(), val_preds_np)
            
            scheduler.step(-val_score)  # Minimize negative AUC
            
            # Early stopping
            if val_score > best_val_score:
                best_val_score = val_score
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Val AUC: {val_score:.6f}")
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Generate predictions
        model.eval()
        with torch.no_grad():
            oof_preds[val_idx] = model(X_fold_val.to(device)).cpu().numpy()
            test_preds += model(X_test_tensor.to(device)).cpu().numpy() / N_FOLDS
        
        fold_scores.append(best_val_score)
        logger.info(f"Fold {fold} CV Score: {best_val_score:.6f}")
    
    # Calculate overall score
    cv_score = roc_auc_score(y_train, oof_preds)
    logger.info(f"\nOverall CV Score: {cv_score:.6f}")
    logger.info(f"Mean Fold Score: {np.mean(fold_scores):.6f} ± {np.std(fold_scores):.6f}")
    
    # Save OOF predictions
    oof_manager.save_oof(
        predictions=oof_preds,
        model_name="MLP_Neural_Network",
        model_params={
            'architecture': 'MLP_512_256_128_64',
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'n_epochs': n_epochs,
            'features': 'xgboost_style'
        },
        cv_score=cv_score,
        test_predictions=test_preds,
        experiment_name='PS5E8_mlp',
        tags={'competition': 'PS5E8', 'model_type': 'MLP', 'approach': 'NN_by_GPT5'}
    )
    
    # Save submission
    submission = sample_sub.copy()
    submission['y'] = test_preds
    submission.to_csv(SUBMISSION_DIR / "submission_mlp_neural_network.csv", index=False)
    
    # Log to MLflow
    if mlflow_tracker:
        mlflow_tracker.log_metrics({
            'cv_auc': cv_score,
            'mean_fold_auc': np.mean(fold_scores),
            'std_fold_auc': np.std(fold_scores)
        })
        mlflow_tracker.end_run()
    
    return {'MLP_Neural_Network': cv_score}, cv_score


def main():
    """Main training pipeline."""
    logger.info("="*60)
    logger.info("PS5E8 COMPETITION - MLP NEURAL NETWORK")
    logger.info("Following NN_by_GPT5 approach: XGBoost features + MLP")
    logger.info("="*60)
    
    # Check for PyTorch
    try:
        import torch
        logger.info("PyTorch found. Proceeding with training...")
    except ImportError:
        logger.error("PyTorch is required. Install with:")
        logger.error("pip install torch")
        logger.error("\nFor GPU support:")
        logger.error("pip install torch --index-url https://download.pytorch.org/whl/cu118")
        return
    
    # Load and engineer features (using XGBoost-style features)
    X_train, y_train, X_test, sample_sub = load_and_engineer_features()
    
    # Train MLP model
    model_scores, cv_score = train_mlp_model(X_train, y_train, X_test, sample_sub)
    
    if model_scores:
        logger.info("\n" + "="*60)
        logger.info("MLP TRAINING COMPLETE!")
        logger.info("="*60)
        logger.info(f"Best submission: {SUBMISSION_DIR}/submission_mlp_neural_network.csv")
        logger.info(f"CV Score: {cv_score:.6f}")
        logger.info("\nKey insight applied: Using XGBoost features for neural network")
        logger.info("This follows the NN_by_GPT5 approach for optimal performance")


if __name__ == "__main__":
    main()