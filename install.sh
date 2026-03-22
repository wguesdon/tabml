#!/bin/bash

# TabML Installation Script
# =========================

echo "╔══════════════════════════════════════════════════════════╗"
echo "║              TabML Installation Script                    ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Function to check if conda is available
check_conda() {
    if command -v conda &> /dev/null; then
        echo "✓ Conda found"
        return 0
    else
        echo "✗ Conda not found"
        return 1
    fi
}

# Function to check if GPU is available
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        echo "✓ GPU detected"
        nvidia-smi --query-gpu=name --format=csv,noheader
        return 0
    else
        echo "✗ No GPU detected"
        return 1
    fi
}

# Main installation
main() {
    echo "1. Checking environment..."
    echo "--------------------------"
    
    # Check for conda
    if check_conda; then
        echo "Using conda environment: $CONDA_DEFAULT_ENV"
    fi
    
    # Check for GPU
    GPU_AVAILABLE=false
    if check_gpu; then
        GPU_AVAILABLE=true
    fi
    
    echo ""
    echo "2. Installing TabML package..."
    echo "------------------------------"
    
    # Install in development mode
    pip install -e .
    
    echo ""
    echo "3. Installing additional dependencies..."
    echo "----------------------------------------"
    
    # Install GPU dependencies if available
    if [ "$GPU_AVAILABLE" = true ]; then
        echo "Installing GPU dependencies..."
        pip install -r requirements-gpu.txt
    else
        echo "Skipping GPU dependencies (no GPU detected)"
    fi
    
    echo ""
    echo "4. Setting up NLTK data..."
    echo "---------------------------"
    
    # Download NLTK data
    python -c "
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
print('✓ NLTK data downloaded')
" 2>/dev/null || echo "✗ NLTK setup skipped"
    
    echo ""
    echo "5. Verifying installation..."
    echo "-----------------------------"
    
    # Test import
    python -c "
import tabml
print(f'✓ TabML version: {tabml.__version__}')

# Check for optional dependencies
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    print('✓ TabNet available')
except:
    print('✗ TabNet not available (install pytorch-tabnet)')

try:
    import nltk
    from nltk.corpus import stopwords
    print('✓ NLTK available')
except:
    print('✗ NLTK not fully configured')

try:
    import torch
    if torch.cuda.is_available():
        print(f'✓ CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        print('✗ CUDA not available')
except:
    print('✗ PyTorch not installed')
" 2>/dev/null
    
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║              Installation Complete!                       ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""
    echo "To test the installation, run:"
    echo "  python spaceship_titanic_solution.py"
    echo ""
    echo "For GPU support with specific CUDA version:"
    echo "  pip install torch --index-url https://download.pytorch.org/whl/cu118"
    echo ""
}

# Run main function
main