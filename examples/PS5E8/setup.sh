#!/bin/bash

# PS5E8 Competition Setup Script
echo "======================================"
echo "PS5E8 Competition Setup"
echo "======================================"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

echo "Creating conda environment ps5e8..."
conda create -n ps5e8 python=3.10 -y

echo ""
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ps5e8

echo ""
echo "Installing TabML..."
cd ../..
pip install -e ".[all]"

echo ""
echo "Installing required dependencies..."
pip install python-dotenv loguru

echo ""
echo "Returning to PS5E8 directory..."
cd examples/PS5E8

echo ""
echo "======================================"
echo "Setup complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Activate the environment: conda activate ps5e8"
echo "2. Start MLflow server: mlflow server --host 0.0.0.0 --port 5000"
echo "3. Run the pipeline:"
echo "   python 01_eda.py"
echo "   python 02_baseline_models.py"
echo "   python 03_ensemble_hill_climb.py"