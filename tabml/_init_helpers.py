"""Helper functions for package initialization and setup."""

import os
import warnings
from loguru import logger

def setup_nltk_data():
    """Download required NLTK data if not present."""
    try:
        import nltk
        
        # List of required NLTK data
        required_data = ['stopwords', 'punkt', 'averaged_perceptron_tagger']
        
        for data_name in required_data:
            try:
                # Try to find the data
                nltk.data.find(f'corpora/{data_name}' if data_name == 'stopwords' else f'tokenizers/{data_name}')
            except LookupError:
                # Download if not found
                logger.info(f"Downloading NLTK {data_name}...")
                nltk.download(data_name, quiet=True)
                
        return True
    except ImportError:
        return False
    except Exception as e:
        logger.warning(f"Error setting up NLTK data: {e}")
        return False

def check_gpu_availability():
    """Check if GPU is available for computation."""
    gpu_info = {
        'cuda_available': False,
        'device_name': None,
        'device_count': 0
    }
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info['cuda_available'] = True
            gpu_info['device_count'] = torch.cuda.device_count()
            gpu_info['device_name'] = torch.cuda.get_device_name(0)
            logger.info(f"GPU available: {gpu_info['device_name']}")
    except ImportError:
        pass
    
    return gpu_info

def check_optional_dependencies():
    """Check which optional dependencies are available."""
    dependencies = {
        'tabnet': False,
        'nltk': False,
        'torch': False,
        'tensorboard': False,
        'wandb': False
    }
    
    # Check TabNet
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
        dependencies['tabnet'] = True
    except ImportError:
        pass
    
    # Check NLTK
    try:
        import nltk
        dependencies['nltk'] = True
    except ImportError:
        pass
    
    # Check PyTorch
    try:
        import torch
        dependencies['torch'] = True
    except ImportError:
        pass
    
    # Check TensorBoard
    try:
        import tensorboard
        dependencies['tensorboard'] = True
    except ImportError:
        pass
    
    # Check W&B
    try:
        import wandb
        dependencies['wandb'] = True
    except ImportError:
        pass
    
    return dependencies

def initialize_tabml():
    """Initialize TabML with all required setup."""
    # Suppress warnings
    warnings.filterwarnings('ignore')
    
    # Setup NLTK data
    setup_nltk_data()
    
    # Check GPU
    gpu_info = check_gpu_availability()
    
    # Check optional dependencies
    deps = check_optional_dependencies()
    
    # Log initialization status
    logger.info("TabML initialized successfully")
    
    if not deps['tabnet']:
        logger.info("TabNet not available. Install with: pip install pytorch-tabnet")
    
    if not deps['nltk']:
        logger.info("NLTK not available. Install with: pip install nltk")
    
    return {
        'gpu': gpu_info,
        'dependencies': deps
    }

# Auto-initialize when imported
_init_status = initialize_tabml()