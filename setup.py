from setuptools import setup, find_packages
import os
import re

# Read version from __init__.py
init_file = os.path.join(os.path.dirname(__file__), 'tabml', '__init__.py')
with open(init_file, 'r') as f:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if version_match:
        version = version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string.")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tabml",
    version=version,
    author="William Guesdon",
    author_email="wguesdon@gmail.com",
    description="A comprehensive package for tabular machine learning tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/tabml",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
        "catboost>=1.2",
        "optuna>=3.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scipy>=1.10.0",
        "category_encoders>=2.6.0",
        "loguru>=0.7.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "joblib>=1.3.0",
        "plotly>=5.17.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "pre-commit>=3.4.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=2.0.0",
            "sphinx-autodoc-typehints>=2.0.0",
            "sphinx-copybutton>=0.5.0",
            "myst-parser>=2.0.0",
            "nbsphinx>=0.9.0",
        ],
        "nlp": [
            "nltk>=3.8.0",
        ],
        "gpu": [
            "pytorch-tabnet>=4.1.0",
            "torch>=2.0.0",
            "tensorboard>=2.14.0",
        ],
        "tracking": [
            "mlflow>=2.8.0,<3.0",  # DagsHub requires MLflow 2.x
            "wandb>=0.15.0",
            "tensorboard>=2.14.0",
            "python-dotenv>=1.0.0",
        ],
        "autogluon": [
            "autogluon>=1.0.0",
        ],
        "benchmarks": [
            "openml>=0.14.0",
            "pmlb>=1.0.0",
        ],
        "all": [
            "nltk>=3.8.0",
            "pytorch-tabnet>=4.1.0",
            "torch>=2.0.0",
            "tensorboard>=2.14.0",
            "mlflow>=2.8.0,<3.0",  # DagsHub requires MLflow 2.x
            "wandb>=0.15.0",
            "python-dotenv>=1.0.0",
            "autogluon>=1.0.0",
            "openml>=0.14.0",
            "pmlb>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tabml=tabml.cli:main",
            "tabml-benchmark=tabml.benchmarks.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        'tabml': ['data/*/*.csv'],
    },
)
