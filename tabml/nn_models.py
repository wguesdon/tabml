"""Neural network model wrappers for tabular data.

This module provides unified interfaces for neural network models following
the BaseModel pattern. Includes wrappers for pytabkit models and custom
PyTorch implementations.

Classes:
    RealMLPModel: Wrapper for pytabkit RealMLP_TD.
    TabMModel: Wrapper for pytabkit TabM_D.
    FTTransformerModel: Custom PyTorch FT-Transformer implementation.
    EmbeddingMLPModel: Custom PyTorch MLP with categorical embeddings.
    LogisticRegressionModel: Wrapper for sklearn LogisticRegression.

Example:
    Basic usage::

        from tabml.nn_models import RealMLPModel

        model = RealMLPModel(params={"n_ens": 4, "device": "cpu"})
        model.fit(X_train, y_train, X_val, y_val)
        predictions = model.predict(X_test)
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from .models import BaseModel

try:
    from pytabkit.models.sklearn.sklearn_interfaces import (
        RealMLP_TD_Classifier,
        RealMLP_TD_Regressor,
        TabM_D_Classifier,
        TabM_D_Regressor,
    )

    PYTABKIT_AVAILABLE = True
except ImportError:
    PYTABKIT_AVAILABLE = False
    logger.warning(
        "pytabkit not installed. Install with: uv add 'pytabkit>=1.7.0'"
    )

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed. Install with: uv add 'torch>=2.0.0'")


class RealMLPModel(BaseModel):
    """Wrapper for pytabkit RealMLP_TD.

    Uses RealMLP_TD_Classifier or RealMLP_TD_Regressor depending on the
    detected task type. The pytabkit models handle train/val splitting
    internally.

    Attributes:
        All attributes from BaseModel.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initializes the RealMLPModel.

        Args:
            params: Hyperparameters for the RealMLP_TD model. Defaults
                include n_ens=8 and device="cpu".

        Raises:
            ImportError: If pytabkit is not installed.
        """
        if not PYTABKIT_AVAILABLE:
            raise ImportError(
                "pytabkit is required for RealMLPModel. "
                "Install with: uv add 'pytabkit>=1.7.0'"
            )
        default_params = {
            "n_ens": 8,
            "device": "cpu",
            "verbosity": 0,
        }
        if params:
            default_params.update(params)
        super().__init__("RealMLP", default_params)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs: Any,
    ) -> "RealMLPModel":
        """Fits the RealMLP model.

        Automatically detects classification vs regression and instantiates
        the appropriate pytabkit model.

        Args:
            X: Training features.
            y: Training target.
            X_val: Validation features. pytabkit handles splitting
                internally if not provided.
            y_val: Validation target.
            **kwargs: Additional keyword arguments passed to the underlying
                fit method.

        Returns:
            The fitted model instance for method chaining.
        """
        self._determine_task_type(y)
        self.feature_names = X.columns.tolist()

        if self.is_classification:
            self.model = RealMLP_TD_Classifier(**self.params)
        else:
            self.model = RealMLP_TD_Regressor(**self.params)

        if X_val is not None and y_val is not None:
            self.model.fit(X, y, X_val, y_val)
        else:
            self.model.fit(X, y)

        self.is_fitted = True
        self.feature_importances_ = np.zeros(len(self.feature_names))
        logger.info(
            f"RealMLP fitted ({'classification' if self.is_classification else 'regression'})"
        )
        return self


class TabMModel(BaseModel):
    """Wrapper for pytabkit TabM_D.

    Uses TabM_D_Classifier or TabM_D_Regressor depending on the detected
    task type. The pytabkit models handle train/val splitting internally.

    Attributes:
        All attributes from BaseModel.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initializes the TabMModel.

        Args:
            params: Hyperparameters for the TabM_D model. Defaults include
                arch="tabm-mini-normal" and device="cpu".

        Raises:
            ImportError: If pytabkit is not installed.
        """
        if not PYTABKIT_AVAILABLE:
            raise ImportError(
                "pytabkit is required for TabMModel. "
                "Install with: uv add 'pytabkit>=1.7.0'"
            )
        default_params = {
            "arch": "tabm-mini-normal",
            "device": "cpu",
            "verbosity": 0,
        }
        if params:
            default_params.update(params)
        super().__init__("TabM", default_params)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs: Any,
    ) -> "TabMModel":
        """Fits the TabM model.

        Automatically detects classification vs regression and instantiates
        the appropriate pytabkit model.

        Args:
            X: Training features.
            y: Training target.
            X_val: Validation features. pytabkit handles splitting
                internally if not provided.
            y_val: Validation target.
            **kwargs: Additional keyword arguments passed to the underlying
                fit method.

        Returns:
            The fitted model instance for method chaining.
        """
        self._determine_task_type(y)
        self.feature_names = X.columns.tolist()

        if self.is_classification:
            self.model = TabM_D_Classifier(**self.params)
        else:
            self.model = TabM_D_Regressor(**self.params)

        if X_val is not None and y_val is not None:
            self.model.fit(X, y, X_val, y_val)
        else:
            self.model.fit(X, y)

        self.is_fitted = True
        self.feature_importances_ = np.zeros(len(self.feature_names))
        logger.info(
            f"TabM fitted ({'classification' if self.is_classification else 'regression'})"
        )
        return self


# ---------------------------------------------------------------------------
# PyTorch helper utilities
# ---------------------------------------------------------------------------


def _get_torch_device(device: Optional[str] = None) -> "torch.device":
    """Returns the appropriate torch device.

    Auto-detects CUDA availability when device is None or "auto".

    Args:
        device: Desired device string. Use "auto" or None for automatic
            detection.

    Returns:
        A torch.device instance.
    """
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _prepare_tensors(
    X: pd.DataFrame,
    y: Optional[pd.Series],
    cat_features: Optional[List[str]],
    cat_encoders: Optional[Dict[str, Dict]],
    device: "torch.device",
    fit_encoders: bool = False,
) -> tuple:
    """Converts a DataFrame into numerical and categorical tensors.

    Categorical columns are label-encoded to integer indices. Numerical
    columns are cast to float32.

    Args:
        X: Input features DataFrame.
        y: Target series. None during inference.
        cat_features: Column names that are categorical. If None, all
            columns are treated as numerical.
        cat_encoders: Dictionary mapping column name to a value-to-index
            mapping. Populated in place when fit_encoders is True.
        device: The torch device to place tensors on.
        fit_encoders: If True, build label encoders from the data.

    Returns:
        A tuple of (X_num, X_cat, y_tensor, cat_encoders) where X_num and
        X_cat are torch tensors and y_tensor may be None.
    """
    if cat_features is None:
        cat_features = []
    if cat_encoders is None:
        cat_encoders = {}

    num_features = [c for c in X.columns if c not in cat_features]

    # Numerical tensor
    if num_features:
        X_num = torch.tensor(
            X[num_features].values.astype(np.float32), device=device
        )
    else:
        X_num = torch.zeros(len(X), 0, device=device)

    # Categorical tensor
    if cat_features:
        cat_arrays = []
        for col in cat_features:
            if fit_encoders:
                unique_vals = X[col].unique()
                mapping = {v: i for i, v in enumerate(unique_vals)}
                cat_encoders[col] = mapping
            mapping = cat_encoders[col]
            default_idx = len(mapping)
            encoded = X[col].map(lambda v, m=mapping, d=default_idx: m.get(v, d))
            cat_arrays.append(encoded.values)
        X_cat = torch.tensor(
            np.column_stack(cat_arrays).astype(np.int64), device=device
        )
    else:
        X_cat = torch.zeros(len(X), 0, dtype=torch.long, device=device)

    # Target tensor
    y_tensor = None
    if y is not None:
        y_np = y.values
        if np.issubdtype(y_np.dtype, np.floating):
            y_tensor = torch.tensor(y_np.astype(np.float32), device=device)
        else:
            y_tensor = torch.tensor(y_np.astype(np.int64), device=device)

    return X_num, X_cat, y_tensor, cat_encoders


# ---------------------------------------------------------------------------
# FT-Transformer
# ---------------------------------------------------------------------------


class _FTTransformerNet(nn.Module):
    """PyTorch FT-Transformer network.

    Implements the Feature Tokenizer + Transformer architecture with:
    - Categorical embeddings per feature.
    - Numerical features tokenized via learned linear projections.
    - A prepended [CLS] token for pooling.
    - A standard Transformer encoder.

    Args:
        n_num_features: Number of numerical input features.
        cat_cardinalities: List of cardinalities for each categorical
            feature. Empty list means no categoricals.
        d_model: Transformer hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of Transformer encoder layers.
        n_classes: Number of output classes. Use 1 for regression.
        dropout: Dropout rate used throughout the model.
    """

    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: List[int],
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        n_classes: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Numerical feature tokenizer: one linear projection per feature
        self.num_projections = nn.ModuleList(
            [nn.Linear(1, d_model) for _ in range(n_num_features)]
        )

        # Categorical embeddings
        self.cat_embeddings = nn.ModuleList(
            [
                nn.Embedding(card + 1, d_model)  # +1 for unknown
                for card in cat_cardinalities
            ]
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes),
        )

    def forward(
        self, x_num: "torch.Tensor", x_cat: "torch.Tensor"
    ) -> "torch.Tensor":
        """Forward pass.

        Args:
            x_num: Numerical features tensor of shape (batch, n_num).
            x_cat: Categorical features tensor of shape (batch, n_cat).

        Returns:
            Logits tensor of shape (batch, n_classes).
        """
        tokens = []

        # CLS token expanded to batch size
        batch_size = x_num.size(0) if x_num.size(1) > 0 else x_cat.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens.append(cls)

        # Numerical tokens
        for i, proj in enumerate(self.num_projections):
            token = proj(x_num[:, i : i + 1])  # (batch, d_model)
            tokens.append(token.unsqueeze(1))

        # Categorical tokens
        for i, emb in enumerate(self.cat_embeddings):
            token = emb(x_cat[:, i])  # (batch, d_model)
            tokens.append(token.unsqueeze(1))

        x = torch.cat(tokens, dim=1)  # (batch, seq_len, d_model)
        x = self.transformer(x)

        # Pool from CLS token
        cls_out = x[:, 0, :]
        return self.head(cls_out)


class FTTransformerModel(BaseModel):
    """Custom PyTorch FT-Transformer for tabular data.

    Implements the Feature Tokenizer + Transformer architecture with
    categorical embeddings and numerical feature tokenization.

    Training uses AdamW with cosine annealing learning rate scheduling.

    Attributes:
        All attributes from BaseModel plus training configuration.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initializes the FTTransformerModel.

        Args:
            params: Model and training hyperparameters. Supported keys
                include d_model, n_heads, n_layers, dropout, lr,
                weight_decay, epochs, batch_size, and device.

        Raises:
            ImportError: If PyTorch is not installed.
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for FTTransformerModel. "
                "Install with: uv add 'torch>=2.0.0'"
            )
        default_params = {
            "d_model": 64,
            "n_heads": 4,
            "n_layers": 3,
            "dropout": 0.1,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "epochs": 100,
            "batch_size": 1024,
            "device": "auto",
        }
        if params:
            default_params.update(params)
        super().__init__("FTTransformer", default_params)
        self._cat_encoders: Dict[str, Dict] = {}
        self._cat_features: Optional[List[str]] = None
        self._cat_cardinalities: List[int] = []
        self._n_num_features: int = 0
        self._n_classes: int = 1
        self._device: Optional["torch.device"] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        cat_features: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "FTTransformerModel":
        """Fits the FT-Transformer model.

        Builds the network architecture based on input data, then trains
        using AdamW and cosine annealing LR scheduling.

        Args:
            X: Training features.
            y: Training target.
            X_val: Validation features for monitoring.
            y_val: Validation target for monitoring.
            cat_features: List of column names that are categorical. If
                None, all columns are treated as numerical.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            The fitted model instance for method chaining.
        """
        self._determine_task_type(y)
        self.feature_names = X.columns.tolist()
        self._cat_features = cat_features or []
        self._device = _get_torch_device(self.params.get("device"))

        # Prepare tensors
        X_num, X_cat, y_t, self._cat_encoders = _prepare_tensors(
            X, y, self._cat_features, {}, self._device, fit_encoders=True
        )
        self._n_num_features = X_num.size(1)
        self._cat_cardinalities = [
            len(self._cat_encoders[c]) for c in self._cat_features
        ]

        # Determine output size
        if self.is_classification:
            self._n_classes = int(y.nunique())
        else:
            self._n_classes = 1

        # Build network
        net = _FTTransformerNet(
            n_num_features=self._n_num_features,
            cat_cardinalities=self._cat_cardinalities,
            d_model=self.params["d_model"],
            n_heads=self.params["n_heads"],
            n_layers=self.params["n_layers"],
            n_classes=self._n_classes,
            dropout=self.params["dropout"],
        ).to(self._device)

        self.model = net
        self._train_loop(X_num, X_cat, y_t, X_val, y_val)
        self.is_fitted = True
        self.feature_importances_ = np.zeros(len(self.feature_names))
        logger.info(
            f"FTTransformer fitted ({'classification' if self.is_classification else 'regression'})"
        )
        return self

    def _train_loop(
        self,
        X_num: "torch.Tensor",
        X_cat: "torch.Tensor",
        y_t: "torch.Tensor",
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
    ) -> None:
        """Runs the training loop.

        Args:
            X_num: Numerical features tensor.
            X_cat: Categorical features tensor.
            y_t: Target tensor.
            X_val: Validation features DataFrame.
            y_val: Validation target Series.
        """
        net = self.model
        epochs = self.params["epochs"]
        batch_size = self.params["batch_size"]
        lr = self.params["lr"]
        weight_decay = self.params["weight_decay"]

        optimizer = torch.optim.AdamW(
            net.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )

        if self.is_classification:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

        n_samples = X_num.size(0)
        net.train()

        for epoch in range(epochs):
            perm = torch.randperm(n_samples, device=self._device)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                idx = perm[start : start + batch_size]
                xn_batch = X_num[idx]
                xc_batch = X_cat[idx]
                y_batch = y_t[idx]

                logits = net(xn_batch, xc_batch)

                if self.is_classification:
                    loss = criterion(logits, y_batch)
                else:
                    loss = criterion(logits.squeeze(-1), y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            if (epoch + 1) % 25 == 0:
                avg_loss = epoch_loss / max(n_batches, 1)
                logger.debug(
                    f"FTTransformer epoch {epoch + 1}/{epochs} loss={avg_loss:.4f}"
                )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Makes predictions.

        Args:
            X: Features to predict on.

        Returns:
            An array of predictions. Class labels for classification,
            continuous values for regression.

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        if not self.is_fitted:
            raise ValueError("FTTransformer must be fitted before prediction.")

        logits = self._predict_raw(X)

        if self.is_classification:
            return logits.argmax(dim=1).cpu().numpy()
        return logits.squeeze(-1).cpu().numpy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts class probabilities.

        Args:
            X: Features to predict on.

        Returns:
            An array of shape (n_samples, n_classes) with class
            probabilities.

        Raises:
            ValueError: If the model has not been fitted or if the task
                is not classification.
        """
        if not self.is_classification:
            raise ValueError(
                "predict_proba is only available for classification."
            )
        if not self.is_fitted:
            raise ValueError("FTTransformer must be fitted before prediction.")

        logits = self._predict_raw(X)
        probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    @torch.no_grad()
    def _predict_raw(self, X: pd.DataFrame) -> "torch.Tensor":
        """Runs inference and returns raw logits.

        Args:
            X: Features DataFrame.

        Returns:
            Raw logits tensor.
        """
        self.model.eval()
        X_num, X_cat, _, _ = _prepare_tensors(
            X, None, self._cat_features, self._cat_encoders, self._device
        )
        batch_size = self.params["batch_size"]
        n = X_num.size(0)
        outputs = []
        for start in range(0, n, batch_size):
            xn = X_num[start : start + batch_size]
            xc = X_cat[start : start + batch_size]
            outputs.append(self.model(xn, xc))
        return torch.cat(outputs, dim=0)


# ---------------------------------------------------------------------------
# Embedding MLP
# ---------------------------------------------------------------------------


class _EmbeddingMLPNet(nn.Module):
    """PyTorch MLP with categorical embeddings.

    Architecture: categorical embeddings concatenated with raw numerics,
    followed by MLP layers with BatchNorm, ReLU, and Dropout.

    Args:
        n_num_features: Number of numerical input features.
        cat_cardinalities: List of cardinalities for each categorical
            feature.
        hidden_dims: List of hidden layer dimensions.
        n_classes: Number of output classes. Use 1 for regression.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: List[int],
        hidden_dims: Optional[List[int]] = None,
        n_classes: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        # Categorical embeddings with auto-sized dimensions
        self.cat_embeddings = nn.ModuleList()
        self.cat_embed_dims = []
        for card in cat_cardinalities:
            dim = min(50, (card + 1) // 2)
            dim = max(dim, 2)
            self.cat_embeddings.append(nn.Embedding(card + 1, dim))
            self.cat_embed_dims.append(dim)

        input_dim = n_num_features + sum(self.cat_embed_dims)

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, n_classes))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self, x_num: "torch.Tensor", x_cat: "torch.Tensor"
    ) -> "torch.Tensor":
        """Forward pass.

        Args:
            x_num: Numerical features of shape (batch, n_num).
            x_cat: Categorical features of shape (batch, n_cat).

        Returns:
            Logits of shape (batch, n_classes).
        """
        parts = []

        if x_num.size(1) > 0:
            parts.append(x_num)

        for i, emb in enumerate(self.cat_embeddings):
            parts.append(emb(x_cat[:, i]))

        x = torch.cat(parts, dim=1) if parts else x_num
        return self.mlp(x)


class EmbeddingMLPModel(BaseModel):
    """Custom PyTorch MLP with categorical embeddings for tabular data.

    Categorical features are embedded and concatenated with numerical
    features before passing through a multi-layer perceptron with
    BatchNorm, ReLU, and Dropout.

    Training uses AdamW with cosine annealing and optional label smoothing
    for classification.

    Attributes:
        All attributes from BaseModel plus training configuration.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initializes the EmbeddingMLPModel.

        Args:
            params: Model and training hyperparameters. Supported keys
                include hidden_dims, dropout, lr, weight_decay, epochs,
                batch_size, label_smoothing, and device.

        Raises:
            ImportError: If PyTorch is not installed.
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for EmbeddingMLPModel. "
                "Install with: uv add 'torch>=2.0.0'"
            )
        default_params = {
            "hidden_dims": [512, 256, 128],
            "dropout": 0.3,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "epochs": 100,
            "batch_size": 1024,
            "label_smoothing": 0.02,
            "device": "auto",
        }
        if params:
            default_params.update(params)
        super().__init__("EmbeddingMLP", default_params)
        self._cat_encoders: Dict[str, Dict] = {}
        self._cat_features: Optional[List[str]] = None
        self._cat_cardinalities: List[int] = []
        self._n_num_features: int = 0
        self._n_classes: int = 1
        self._device: Optional["torch.device"] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        cat_features: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "EmbeddingMLPModel":
        """Fits the Embedding MLP model.

        Builds embeddings for categorical features, constructs the MLP,
        and trains using AdamW with cosine annealing.

        Args:
            X: Training features.
            y: Training target.
            X_val: Validation features for monitoring.
            y_val: Validation target for monitoring.
            cat_features: List of column names that are categorical. If
                None, all columns are treated as numerical.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            The fitted model instance for method chaining.
        """
        self._determine_task_type(y)
        self.feature_names = X.columns.tolist()
        self._cat_features = cat_features or []
        self._device = _get_torch_device(self.params.get("device"))

        # Prepare tensors
        X_num, X_cat, y_t, self._cat_encoders = _prepare_tensors(
            X, y, self._cat_features, {}, self._device, fit_encoders=True
        )
        self._n_num_features = X_num.size(1)
        self._cat_cardinalities = [
            len(self._cat_encoders[c]) for c in self._cat_features
        ]

        if self.is_classification:
            self._n_classes = int(y.nunique())
        else:
            self._n_classes = 1

        net = _EmbeddingMLPNet(
            n_num_features=self._n_num_features,
            cat_cardinalities=self._cat_cardinalities,
            hidden_dims=self.params["hidden_dims"],
            n_classes=self._n_classes,
            dropout=self.params["dropout"],
        ).to(self._device)

        self.model = net
        self._train_loop(X_num, X_cat, y_t, X_val, y_val)
        self.is_fitted = True
        self.feature_importances_ = np.zeros(len(self.feature_names))
        logger.info(
            f"EmbeddingMLP fitted ({'classification' if self.is_classification else 'regression'})"
        )
        return self

    def _train_loop(
        self,
        X_num: "torch.Tensor",
        X_cat: "torch.Tensor",
        y_t: "torch.Tensor",
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
    ) -> None:
        """Runs the training loop with optional label smoothing.

        Args:
            X_num: Numerical features tensor.
            X_cat: Categorical features tensor.
            y_t: Target tensor.
            X_val: Validation features DataFrame.
            y_val: Validation target Series.
        """
        net = self.model
        epochs = self.params["epochs"]
        batch_size = self.params["batch_size"]
        lr = self.params["lr"]
        weight_decay = self.params["weight_decay"]

        optimizer = torch.optim.AdamW(
            net.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )

        if self.is_classification:
            smoothing = self.params.get("label_smoothing", 0.02)
            criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
        else:
            criterion = nn.MSELoss()

        n_samples = X_num.size(0)
        net.train()

        for epoch in range(epochs):
            perm = torch.randperm(n_samples, device=self._device)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                idx = perm[start : start + batch_size]
                xn_batch = X_num[idx]
                xc_batch = X_cat[idx]
                y_batch = y_t[idx]

                logits = net(xn_batch, xc_batch)

                if self.is_classification:
                    loss = criterion(logits, y_batch)
                else:
                    loss = criterion(logits.squeeze(-1), y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            if (epoch + 1) % 25 == 0:
                avg_loss = epoch_loss / max(n_batches, 1)
                logger.debug(
                    f"EmbeddingMLP epoch {epoch + 1}/{epochs} loss={avg_loss:.4f}"
                )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Makes predictions.

        Args:
            X: Features to predict on.

        Returns:
            An array of predictions. Class labels for classification,
            continuous values for regression.

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        if not self.is_fitted:
            raise ValueError("EmbeddingMLP must be fitted before prediction.")

        logits = self._predict_raw(X)

        if self.is_classification:
            return logits.argmax(dim=1).cpu().numpy()
        return logits.squeeze(-1).cpu().numpy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts class probabilities.

        Args:
            X: Features to predict on.

        Returns:
            An array of shape (n_samples, n_classes) with class
            probabilities.

        Raises:
            ValueError: If the model has not been fitted or if the task
                is not classification.
        """
        if not self.is_classification:
            raise ValueError(
                "predict_proba is only available for classification."
            )
        if not self.is_fitted:
            raise ValueError("EmbeddingMLP must be fitted before prediction.")

        logits = self._predict_raw(X)
        probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    @torch.no_grad()
    def _predict_raw(self, X: pd.DataFrame) -> "torch.Tensor":
        """Runs inference and returns raw logits.

        Args:
            X: Features DataFrame.

        Returns:
            Raw logits tensor.
        """
        self.model.eval()
        X_num, X_cat, _, _ = _prepare_tensors(
            X, None, self._cat_features, self._cat_encoders, self._device
        )
        batch_size = self.params["batch_size"]
        n = X_num.size(0)
        outputs = []
        for start in range(0, n, batch_size):
            xn = X_num[start : start + batch_size]
            xc = X_cat[start : start + batch_size]
            outputs.append(self.model(xn, xc))
        return torch.cat(outputs, dim=0)


# ---------------------------------------------------------------------------
# Logistic Regression (Level 4 meta-learner)
# ---------------------------------------------------------------------------


class LogisticRegressionModel(BaseModel):
    """Wrapper for sklearn LogisticRegression.

    Intended for use as a Level 4 meta-learner in stacking ensembles.
    Supports only classification tasks.

    Attributes:
        All attributes from BaseModel.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initializes the LogisticRegressionModel.

        Args:
            params: Hyperparameters for sklearn LogisticRegression. Defaults
                include C=1.0, penalty="l2", solver="lbfgs", max_iter=1000.
        """
        default_params = {
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": 42,
            "n_jobs": -1,
        }
        if params:
            default_params.update(params)
        super().__init__("LogisticRegression", default_params)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs: Any,
    ) -> "LogisticRegressionModel":
        """Fits the Logistic Regression model.

        Forces classification mode regardless of the target distribution.

        Args:
            X: Training features.
            y: Training target.
            X_val: Validation features (unused, accepted for API
                compatibility).
            y_val: Validation target (unused, accepted for API
                compatibility).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            The fitted model instance for method chaining.
        """
        from sklearn.linear_model import LogisticRegression

        self.is_classification = True
        self.feature_names = X.columns.tolist()

        self.model = LogisticRegression(**self.params)
        self.model.fit(X, y)

        self.is_fitted = True
        # Use absolute coefficient values as a proxy for feature importance
        coef = self.model.coef_
        if coef.ndim == 2:
            self.feature_importances_ = np.mean(np.abs(coef), axis=0)
        else:
            self.feature_importances_ = np.abs(coef)

        logger.info("LogisticRegression fitted (classification)")
        return self
