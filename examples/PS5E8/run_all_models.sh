#!/bin/bash

# PS5E8 Competition - Run all model training scripts
# This script runs each model type separately for easier debugging

echo "=================================================="
echo "PS5E8 COMPETITION - TRAINING ALL MODELS"
echo "=================================================="

# Activate conda environment if needed
# conda activate ps5e8

# Train XGBoost models
echo ""
echo "1. Training XGBoost models..."
echo "--------------------------------------------------"
python 04a_xgboost_advanced.py

# Train LightGBM models
echo ""
echo "2. Training LightGBM models..."
echo "--------------------------------------------------"
python 04b_lightgbm_advanced.py

# Train CatBoost models
echo ""
echo "3. Training CatBoost models..."
echo "--------------------------------------------------"
python 04c_catboost_advanced.py

# Train TabNet models (optional - requires PyTorch)
echo ""
echo "4. Training TabNet models (attention-based neural network)..."
echo "--------------------------------------------------"
python 04d_tabnet_advanced.py || echo "TabNet training skipped (install pytorch-tabnet if needed)"

# Train MLP Neural Network (following NN_by_GPT5 approach)
echo ""
echo "5. Training MLP Neural Network (XGBoost features + MLP)..."
echo "--------------------------------------------------"
python 04e_mlp_neural_network.py || echo "MLP training skipped (install torch if needed)"

# Create final ensemble
echo ""
echo "6. Creating final ensemble..."
echo "--------------------------------------------------"
python 05_final_ensemble.py

echo ""
echo "=================================================="
echo "ALL MODELS TRAINED SUCCESSFULLY!"
echo "Check output/submissions_advanced/ for submissions"
echo "=================================================="