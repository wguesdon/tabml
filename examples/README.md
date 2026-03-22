# TabML Examples

This directory contains example scripts demonstrating various TabML capabilities.

## Available Examples

### 1. `spaceship_titanic_solution.py`
Complete end-to-end solution for the Kaggle Spaceship Titanic competition, demonstrating:
- Automated EDA with visualizations
- Advanced feature engineering
- Multiple model training with hyperparameter optimization
- OOF ensemble creation
- Submission generation

**To run:**
```bash
cd tabml/examples
python spaceship_titanic_solution.py
```

All outputs will be saved to `../output/spaceship_titanic/`

### 2. `oof_ensemble_example.py`
Comprehensive examples of Out-of-Fold ensemble techniques:
- Basic OOF ensemble creation
- Weight optimization methods (Scipy, Optuna, Grid Search)
- Stacking with meta-learners
- Rank averaging for robust ensembles
- Automatic ensemble strategy selection
- Complete Kaggle competition workflow

**To run:**
```bash
cd tabml/examples
python oof_ensemble_example.py
```

### 3. `tabml_example.py`
Demonstrates all TabML features including:
- Comparison with AbdML framework
- Advanced models (TabNet, Voting Ensembles)
- Custom metrics and CV strategies
- Training callbacks
- Complete pipeline matching AbdML functionality

**To run:**
```bash
cd tabml/examples  
python tabml_example.py
```

### 4. `spaceship_titanic_real_data.py`
Original example using real Spaceship Titanic data with basic TabML pipeline.

## Output Directory Structure

All example scripts save their outputs to the `../output/` directory:

```
tabml/
├── examples/          # Example scripts
├── output/           # Generated outputs (gitignored)
│   ├── spaceship_titanic/
│   │   ├── *.png     # Visualization plots
│   │   ├── *.csv     # Submission files
│   │   └── *.log     # Log files
│   └── other_examples/
```

## Requirements

Make sure TabML is installed with all optional dependencies:

```bash
# From the tabml root directory
pip install -e ".[all]"
```

For GPU support (TabNet):
```bash
pip install pytorch-tabnet torch
```

## Notes

- The `output/` directory is gitignored, so your outputs won't be committed
- Data files are expected in `../data/raw/` directory
- All examples use relative paths for portability
- Visualizations are automatically saved to the output directory