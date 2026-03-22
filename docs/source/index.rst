.. TabML documentation master file

Welcome to TabML's documentation!
=================================

TabML is a comprehensive Python package for tabular machine learning tasks, providing tools for data processing, feature engineering, model training, and evaluation.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/index
   examples/index
   contributing

Features
--------

* **Data Processing**: Automated data loading, validation, and preprocessing
* **Feature Engineering**: Built-in feature engineering capabilities
* **Model Training**: Support for XGBoost, LightGBM, CatBoost, and scikit-learn models
* **Hyperparameter Optimization**: Integrated Optuna support
* **Time Series**: Specialized time series data handling and validation
* **Visualization**: Comprehensive EDA and results visualization
* **CLI**: Command-line interface for common tasks

Quick Example
-------------

.. code-block:: python

   from tabml.data import DataLoader
   from tabml.pipeline import TabularPipeline
   
   # Load data
   loader = DataLoader()
   train_df = loader.load_data("train.csv")
   
   # Create and run pipeline
   pipeline = TabularPipeline(
       data_loader=loader,
       target_column="target"
   )
   pipeline.run(train_df)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`