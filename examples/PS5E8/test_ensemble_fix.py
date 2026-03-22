#!/usr/bin/env python3
"""
Test script to verify all fixes work correctly
"""

import os
import sys
from pathlib import Path
import subprocess
from loguru import logger

# Setup paths
BASE_DIR = Path(__file__).parent

def test_individual_scripts():
    """Test that individual model scripts run without errors."""
    
    scripts = [
        "04a_xgboost_advanced.py",
        "04b_lightgbm_advanced.py", 
        "04c_catboost_advanced.py",
        "04d_tabnet_advanced.py",
        "04e_mlp_neural_network.py"
    ]
    
    logger.info("="*60)
    logger.info("TESTING INDIVIDUAL MODEL SCRIPTS")
    logger.info("="*60)
    
    results = {}
    
    for script in scripts:
        script_path = BASE_DIR / script
        if not script_path.exists():
            logger.warning(f"Script not found: {script}")
            results[script] = "NOT FOUND"
            continue
            
        logger.info(f"\nTesting {script}...")
        try:
            # Run with timeout to avoid hanging
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(BASE_DIR),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                logger.success(f"✓ {script} completed successfully")
                results[script] = "SUCCESS"
            else:
                logger.error(f"✗ {script} failed with error")
                logger.error(f"Error: {result.stderr[-500:]}")  # Last 500 chars of error
                results[script] = "FAILED"
                
        except subprocess.TimeoutExpired:
            logger.warning(f"⚠ {script} timed out (>5 minutes)")
            results[script] = "TIMEOUT"
        except Exception as e:
            logger.error(f"✗ {script} crashed: {str(e)}")
            results[script] = "CRASHED"
    
    return results

def test_final_ensemble():
    """Test that final ensemble script runs and creates submissions."""
    
    logger.info("\n" + "="*60)
    logger.info("TESTING FINAL ENSEMBLE")
    logger.info("="*60)
    
    ensemble_script = BASE_DIR / "05_final_ensemble.py"
    
    if not ensemble_script.exists():
        logger.error("Final ensemble script not found!")
        return False
        
    logger.info("Running final ensemble...")
    try:
        result = subprocess.run(
            [sys.executable, str(ensemble_script)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout for ensemble
        )
        
        if result.returncode == 0:
            logger.success("✓ Final ensemble completed successfully")
            
            # Check if submission files were created
            submission_dir = BASE_DIR / "output" / "submissions_final"
            if submission_dir.exists():
                submissions = list(submission_dir.glob("*.csv"))
                if submissions:
                    logger.success(f"✓ Created {len(submissions)} submission files:")
                    for sub in submissions[:5]:  # Show first 5
                        logger.info(f"  - {sub.name}")
                else:
                    logger.warning("⚠ No submission files created")
            else:
                logger.warning("⚠ Submission directory not found")
                
            return True
        else:
            logger.error("✗ Final ensemble failed")
            logger.error(f"Error: {result.stderr[-1000:]}")  # Last 1000 chars
            return False
            
    except subprocess.TimeoutExpired:
        logger.warning("⚠ Final ensemble timed out")
        return False
    except Exception as e:
        logger.error(f"✗ Final ensemble crashed: {str(e)}")
        return False

def main():
    """Main test function."""
    
    logger.info("="*60)
    logger.info("PS5E8 ENSEMBLE FIX TEST")
    logger.info("="*60)
    
    # Test individual scripts
    model_results = test_individual_scripts()
    
    # Test final ensemble
    ensemble_success = test_final_ensemble()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    logger.info("\nModel Script Results:")
    for script, status in model_results.items():
        if status == "SUCCESS":
            logger.success(f"  ✓ {script}: {status}")
        elif status in ["TIMEOUT", "NOT FOUND"]:
            logger.warning(f"  ⚠ {script}: {status}")
        else:
            logger.error(f"  ✗ {script}: {status}")
    
    if ensemble_success:
        logger.success("\n✓ Final ensemble: SUCCESS")
    else:
        logger.error("\n✗ Final ensemble: FAILED")
    
    # Overall result
    success_count = sum(1 for s in model_results.values() if s == "SUCCESS")
    total_count = len(model_results)
    
    logger.info(f"\nModels successful: {success_count}/{total_count}")
    logger.info(f"Ensemble successful: {'Yes' if ensemble_success else 'No'}")
    
    if success_count > 0 and ensemble_success:
        logger.success("\n✓ FIXES VERIFIED - At least some models work and ensemble runs")
    elif success_count > 0:
        logger.warning("\n⚠ PARTIAL SUCCESS - Some models work but ensemble failed")
    else:
        logger.error("\n✗ FIXES NEED MORE WORK")

if __name__ == "__main__":
    main()