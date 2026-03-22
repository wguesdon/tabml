#!/bin/bash

# Build documentation script for TabML

echo "Building TabML documentation..."

# Activate conda environment
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate tabml  # Change 'tabml' to your environment name if different

# Install documentation dependencies if not already installed
echo "Installing documentation dependencies..."
pip install -e ".[docs]" --quiet

# Clean previous builds
echo "Cleaning previous builds..."
cd docs
make clean

# Build HTML documentation
echo "Building HTML documentation..."
make html

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Documentation built successfully!"
    echo "Open docs/build/html/index.html to view the documentation"
    
    # Optionally open in browser (uncomment for your OS)
    # Linux
    # xdg-open build/html/index.html
    
    # macOS
    # open build/html/index.html
    
    # Windows (WSL)
    # explorer.exe build/html/index.html
else
    echo "Documentation build failed. Please check the errors above."
    exit 1
fi