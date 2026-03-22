"""
Clean OOF predictions and run ensemble optimization
Removes invalid/empty OOF files and runs hill climbing
"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

def check_and_clean_oofs(oof_dir="output/oof_predictions"):
    """Check OOF files and remove invalid ones."""
    oof_dir = Path(oof_dir)
    
    logger.info("Checking OOF prediction files...")
    
    valid_files = []
    invalid_files = []
    
    for pkl_file in oof_dir.glob("*.pkl"):
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                
            pred = data.get('predictions')
            test_pred = data.get('test_predictions')
            
            # Check if predictions are valid
            is_valid = True
            
            # Check predictions
            if pred is None:
                is_valid = False
            elif hasattr(pred, 'shape'):
                if pred.shape[0] == 0:
                    is_valid = False
                elif hasattr(pred, 'isna') and pred.isna().any().any():
                    is_valid = False
            
            # Check test predictions
            if test_pred is not None and hasattr(test_pred, 'shape'):
                if test_pred.shape[0] == 0:
                    is_valid = False
                    
            if is_valid and pred.shape[0] == 750000:  # Correct size for training data
                valid_files.append(pkl_file.name)
                logger.info(f"  ✓ {pkl_file.name}: Valid (shape: {pred.shape})")
            else:
                invalid_files.append(pkl_file.name)
                logger.warning(f"  ✗ {pkl_file.name}: Invalid or empty")
                
        except Exception as e:
            invalid_files.append(pkl_file.name)
            logger.error(f"  ✗ {pkl_file.name}: Error loading - {e}")
    
    logger.info(f"\nSummary: {len(valid_files)} valid, {len(invalid_files)} invalid files")
    
    # Remove invalid files
    if invalid_files:
        logger.info("\nRemoving invalid files...")
        for filename in invalid_files:
            file_path = oof_dir / filename
            if file_path.exists():
                file_path.unlink()
                logger.info(f"  Removed: {filename}")
    
    return valid_files, invalid_files


def main():
    logger.info("="*60)
    logger.info("CLEANING OOF PREDICTIONS AND RUNNING ENSEMBLE")
    logger.info("="*60)
    
    # Step 1: Clean OOF files
    valid_files, invalid_files = check_and_clean_oofs()
    
    if not valid_files:
        logger.error("No valid OOF files found! Please run 02_baseline_models.py first.")
        return
    
    # Step 2: Check if we have enough diversity
    logger.info("\nChecking model diversity...")
    
    from tabml import OOFManager
    manager = OOFManager(output_dir="output/oof_predictions")
    summary = manager.list_oofs(sort_by='cv_score', ascending=False)
    
    if not summary.empty:
        unique_models = summary['model_name'].str.replace(r'_\d+$', '', regex=True).unique()
        logger.info(f"Found {len(unique_models)} unique model types:")
        for model in unique_models:
            model_scores = summary[summary['model_name'].str.contains(model)]['cv_score'].values
            logger.info(f"  - {model}: {len(model_scores)} versions, scores: {model_scores}")
    
    # Step 3: Run ensemble optimization
    if len(valid_files) >= 2:
        logger.info(f"\nRunning ensemble optimization with {len(valid_files)} models...")
        logger.info("Executing: python 03_ensemble_hill_climb.py")
        
        import subprocess
        import sys
        
        # Use the same Python environment
        result = subprocess.run(
            [sys.executable, "03_ensemble_hill_climb.py"],
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("✓ Ensemble optimization completed successfully!")
        else:
            logger.error("✗ Ensemble optimization failed. Check the output above.")
    else:
        logger.error(f"Not enough valid models for ensemble (found {len(valid_files)}, need at least 2)")
        logger.info("Please run 02_baseline_models.py to train more models.")


if __name__ == "__main__":
    main()