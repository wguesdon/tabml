# Google Style Docstring Guide for TabML

This guide shows how to write Google-style docstrings that work well with Sphinx autodoc.

## Module Docstrings

```python
"""Brief module description.

This module provides functionality for [describe what the module does].
It includes classes and functions for [main features].

Example:
    Basic usage of this module::
    
        from tabml.module import MyClass
        obj = MyClass()
        result = obj.process(data)

Note:
    This module requires [any special requirements].
"""
```

## Class Docstrings

```python
class DataProcessor:
    """Handles data preprocessing and transformation.
    
    This class provides methods for cleaning, transforming, and preparing
    tabular data for machine learning models. It handles missing values,
    encoding categorical variables, and feature scaling.
    
    Attributes:
        scaler: The scikit-learn scaler object used for normalization
        encoder: The encoder used for categorical variables
        feature_names: List of feature names after transformation
        is_fitted: Boolean indicating if the processor has been fitted
        
    Example:
        >>> processor = DataProcessor(scaling_method='standard')
        >>> processor.fit(X_train)
        >>> X_transformed = processor.transform(X_test)
    """
```

## Method/Function Docstrings

### Basic Method
```python
def load_data(self, filepath: str, **kwargs) -> pd.DataFrame:
    """Load data from a file.
    
    Args:
        filepath: Path to the data file
        **kwargs: Additional arguments passed to pandas read functions
        
    Returns:
        Loaded data as a pandas DataFrame
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is not supported
    """
```

### Complex Method with Examples
```python
def fit_transform(self, 
                  X: pd.DataFrame, 
                  y: Optional[pd.Series] = None,
                  validation_split: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the processor and transform the data.
    
    This method fits the preprocessing pipeline on the training data and
    returns both transformed training and validation sets.
    
    Args:
        X: Input features as pandas DataFrame
        y: Target variable (optional). If provided, stratified split is used
        validation_split: Fraction of data to use for validation. 
            Must be between 0.0 and 1.0. Defaults to 0.2.
            
    Returns:
        A tuple containing:
            - X_train_transformed: Transformed training features
            - X_val_transformed: Transformed validation features
            
    Raises:
        ValueError: If validation_split is not between 0 and 1
        TypeError: If X is not a pandas DataFrame
        
    Example:
        >>> processor = DataProcessor()
        >>> X_train, X_val = processor.fit_transform(df, y, validation_split=0.3)
        >>> print(f"Training shape: {X_train.shape}")
        >>> print(f"Validation shape: {X_val.shape}")
        
    Note:
        The processor state is saved after fitting, so subsequent calls to
        transform() will use the same preprocessing parameters.
    """
```

### Property Docstrings
```python
@property
def feature_importances_(self) -> pd.DataFrame:
    """Get feature importances as a DataFrame.
    
    Returns:
        DataFrame with columns 'feature' and 'importance', sorted by importance
        
    Raises:
        AttributeError: If model hasn't been fitted yet
    """
```

## Parameter Types

Always specify types clearly:
```python
def process_data(self,
                 data: Union[pd.DataFrame, np.ndarray],
                 columns: List[str],
                 options: Dict[str, Any],
                 threshold: Optional[float] = None) -> pd.DataFrame:
    """Process data with specified options.
    
    Args:
        data: Input data as DataFrame or numpy array
        columns: List of column names to process
        options: Dictionary of processing options where keys are option names
            and values are option values
        threshold: Optional threshold value for filtering. If None, no 
            filtering is applied
    """
```

## Special Sections

### Yields (for generators)
```python
def batch_generator(self, data: pd.DataFrame, batch_size: int = 32):
    """Generate batches of data.
    
    Args:
        data: Input DataFrame
        batch_size: Size of each batch
        
    Yields:
        DataFrame: A batch of data
        
    Example:
        >>> for batch in processor.batch_generator(df, batch_size=64):
        ...     process_batch(batch)
    """
```

### See Also
```python
def train_model(self, X: pd.DataFrame, y: pd.Series) -> 'Model':
    """Train the model on data.
    
    Args:
        X: Training features
        y: Training targets
        
    Returns:
        Trained model instance
        
    See Also:
        predict: For making predictions
        evaluate: For model evaluation
        cross_validate: For cross-validation
    """
```

### Warnings and Important Notes
```python
def risky_operation(self, data: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """Perform a potentially risky operation.
    
    Args:
        data: Input data
        inplace: Whether to modify data in place
        
    Returns:
        Modified DataFrame
        
    Warning:
        This operation modifies the original data if inplace=True.
        Make sure to backup your data first.
        
    Note:
        This method is experimental and may change in future versions.
    """
```

## Best Practices

1. **First line**: Brief one-line summary ending with a period
2. **Blank line**: After the summary if you have more content
3. **Extended description**: More detailed explanation if needed
4. **Args section**: List all parameters with types and descriptions
5. **Returns section**: Describe what is returned
6. **Raises section**: List exceptions that might be raised
7. **Example section**: Show usage examples
8. **Notes/Warnings**: Add when necessary

## Real Example Update

Here's how to update an existing method:

**Before:**
```python
def fit(self, X: pd.DataFrame, y: pd.Series) -> 'Model':
    """Fit the model."""
    # implementation
```

**After:**
```python
def fit(self, X: pd.DataFrame, y: pd.Series) -> 'Model':
    """Fit the model to training data.
    
    Trains the model on the provided features and target values using
    the configured hyperparameters.
    
    Args:
        X: Training features with shape (n_samples, n_features)
        y: Target values with shape (n_samples,)
        
    Returns:
        Self instance for method chaining
        
    Example:
        >>> model = XGBoostModel(params={'max_depth': 5})
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    # implementation
```