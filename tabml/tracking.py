"""Lightweight SQLite-based model tracking for Kaggle competitions.

Tracks model performance across experiments. Each training run records its CV
score, fold scores, parameters, feature set, and optionally the LB score after
submission. This enables comparing models and selecting the best ones for
ensembling.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from loguru import logger

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    version INTEGER,
    model_type TEXT,
    level INTEGER DEFAULT 2,
    cv_score REAL,
    cv_std REAL,
    fold_scores TEXT,
    metric TEXT DEFAULT 'balanced_accuracy',
    params TEXT,
    feature_group TEXT,
    n_features INTEGER,
    oof_path TEXT,
    test_pred_path TEXT,
    notes TEXT,
    lb_score_public REAL,
    lb_score_private REAL,
    training_time_seconds REAL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_models_name ON models (name);
CREATE INDEX IF NOT EXISTS idx_models_cv_score ON models (cv_score);
CREATE INDEX IF NOT EXISTS idx_models_level ON models (level);
"""


class ModelTracker:
    """SQLite-backed tracker for Kaggle model experiments.

    Stores experiment metadata in a local SQLite database so you can
    compare CV scores, LB scores, and OOF prediction diversity across
    many model training runs.

    Attributes:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: str = "experiments.db"):
        """Initialize tracker. Creates SQLite DB and tables if they don't exist.

        Args:
            db_path: File path for the SQLite database. Created automatically
                if it does not exist.
        """
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()
        logger.info("ModelTracker initialized with database at {}", db_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_model(
        self,
        name: str,
        version: int,
        model_type: str,
        cv_score: float,
        fold_scores: list[float],
        metric: str = "balanced_accuracy",
        params: Optional[dict] = None,
        feature_group: Optional[str] = None,
        n_features: Optional[int] = None,
        oof_path: Optional[str] = None,
        test_pred_path: Optional[str] = None,
        notes: Optional[str] = None,
        level: int = 2,
        training_time_seconds: Optional[float] = None,
    ) -> int:
        """Log a model training run.

        Args:
            name: Unique identifier for this run, e.g. ``"xgb_v2100"``.
            version: Numeric version, e.g. ``2100``.
            model_type: Algorithm family such as ``"xgboost"`` or ``"lightgbm"``.
            cv_score: Overall cross-validation metric value.
            fold_scores: Per-fold metric values.
            metric: Name of the evaluation metric.
            params: Hyperparameters dictionary. Stored as JSON.
            feature_group: Label for the feature set used.
            n_features: Number of features in the training data.
            oof_path: Path to the out-of-fold predictions ``.npy`` file.
            test_pred_path: Path to the test predictions ``.npy`` file.
            notes: Free-form text notes about this run.
            level: Stacking level, typically 1 through 4.
            training_time_seconds: Wall-clock training time in seconds.

        Returns:
            The SQLite row ID of the inserted record.

        Raises:
            sqlite3.IntegrityError: If a model with the same *name* already
                exists.
        """
        cv_std = float(np.std(fold_scores)) if fold_scores else None
        now = datetime.now(timezone.utc).isoformat()

        cur = self._conn.execute(
            """
            INSERT INTO models (
                name, version, model_type, level, cv_score, cv_std,
                fold_scores, metric, params, feature_group, n_features,
                oof_path, test_pred_path, notes, training_time_seconds,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                version,
                model_type,
                level,
                cv_score,
                cv_std,
                json.dumps(fold_scores),
                metric,
                json.dumps(params) if params is not None else None,
                feature_group,
                n_features,
                oof_path,
                test_pred_path,
                notes,
                training_time_seconds,
                now,
            ),
        )
        self._conn.commit()
        row_id = cur.lastrowid
        logger.info(
            "Logged model '{}' (cv_score={:.6f}, type={})", name, cv_score, model_type
        )
        return row_id

    def update_lb_score(
        self, name: str, lb_score: float, lb_type: str = "public"
    ) -> None:
        """Update leaderboard score for a model after submission.

        Args:
            name: Model name that was previously logged.
            lb_score: The leaderboard score value.
            lb_type: Either ``"public"`` or ``"private"``.

        Raises:
            ValueError: If *lb_type* is not ``"public"`` or ``"private"``.
            KeyError: If no model with the given *name* exists.
        """
        if lb_type not in ("public", "private"):
            raise ValueError(
                f"lb_type must be 'public' or 'private', got '{lb_type}'"
            )

        column = f"lb_score_{lb_type}"
        cur = self._conn.execute(
            f"UPDATE models SET {column} = ? WHERE name = ?",  # noqa: S608
            (lb_score, name),
        )
        self._conn.commit()

        if cur.rowcount == 0:
            raise KeyError(f"No model found with name '{name}'")

        logger.info(
            "Updated {} LB score for '{}' to {:.6f}", lb_type, name, lb_score
        )

    def get_leaderboard(
        self,
        level: Optional[int] = None,
        model_type: Optional[str] = None,
        top_n: Optional[int] = None,
        sort_by: str = "cv_score",
    ) -> pd.DataFrame:
        """Get models sorted by score with optional filtering.

        Args:
            level: If provided, only return models at this stacking level.
            model_type: If provided, only return models of this type.
            top_n: If provided, limit to the top N results.
            sort_by: Column name to sort by in descending order.

        Returns:
            A DataFrame of model records.
        """
        query = "SELECT * FROM models WHERE 1=1"
        bind: list = []

        if level is not None:
            query += " AND level = ?"
            bind.append(level)
        if model_type is not None:
            query += " AND model_type = ?"
            bind.append(model_type)

        query += f" ORDER BY {sort_by} DESC"

        if top_n is not None:
            query += " LIMIT ?"
            bind.append(top_n)

        rows = self._conn.execute(query, bind).fetchall()
        df = self._rows_to_dataframe(rows)
        return df

    def get_model(self, name: str) -> Optional[dict]:
        """Get full details for a specific model.

        Args:
            name: The unique model name.

        Returns:
            A dictionary of the model record, or ``None`` if not found.
            JSON fields (``fold_scores``, ``params``) are deserialized.
        """
        row = self._conn.execute(
            "SELECT * FROM models WHERE name = ?", (name,)
        ).fetchone()

        if row is None:
            return None

        return self._row_to_dict(row)

    def get_best_models(
        self, n: int = 10, level: Optional[int] = None
    ) -> pd.DataFrame:
        """Get top N models by CV score.

        Args:
            n: Number of models to return.
            level: If provided, filter by stacking level.

        Returns:
            A DataFrame of the top models sorted by ``cv_score`` descending.
        """
        return self.get_leaderboard(level=level, top_n=n, sort_by="cv_score")

    def get_diversity_matrix(
        self, model_names: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """Compute pairwise correlation of OOF predictions.

        Loads the ``.npy`` files referenced by each model's ``oof_path`` and
        returns a correlation matrix. This is useful for selecting diverse
        models for ensembling.

        Args:
            model_names: List of model names to include. If ``None``, uses all
                models that have an ``oof_path`` set.

        Returns:
            A square DataFrame of Pearson correlations indexed by model name.

        Raises:
            FileNotFoundError: If any referenced ``.npy`` file does not exist.
            ValueError: If fewer than two models have OOF predictions.
        """
        if model_names is not None:
            placeholders = ", ".join("?" for _ in model_names)
            rows = self._conn.execute(
                f"SELECT name, oof_path FROM models "  # noqa: S608
                f"WHERE oof_path IS NOT NULL AND name IN ({placeholders})",
                model_names,
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT name, oof_path FROM models WHERE oof_path IS NOT NULL"
            ).fetchall()

        if len(rows) < 2:
            raise ValueError(
                "Need at least 2 models with oof_path to compute diversity matrix. "
                f"Found {len(rows)}."
            )

        preds: dict[str, np.ndarray] = {}
        for row in rows:
            name = row["name"]
            path = Path(row["oof_path"])
            if not path.exists():
                raise FileNotFoundError(
                    f"OOF file for model '{name}' not found: {path}"
                )
            arr = np.load(path)
            if arr.ndim > 1:
                arr = arr.ravel()
            preds[name] = arr

        df = pd.DataFrame(preds)
        corr = df.corr()
        logger.info("Computed diversity matrix for {} models", len(preds))
        return corr

    def compare_cv_lb(self) -> pd.DataFrame:
        """Return models that have both CV and public LB scores.

        Useful for analysing how well local CV tracks the public leaderboard.

        Returns:
            A DataFrame with columns ``name``, ``cv_score``, ``lb_score_public``,
            and ``cv_lb_diff``, sorted by ``cv_score`` descending.
        """
        rows = self._conn.execute(
            "SELECT * FROM models "
            "WHERE cv_score IS NOT NULL AND lb_score_public IS NOT NULL "
            "ORDER BY cv_score DESC"
        ).fetchall()
        df = self._rows_to_dataframe(rows)

        if not df.empty:
            df["cv_lb_diff"] = df["cv_score"] - df["lb_score_public"]

        return df

    def delete_model(self, name: str) -> None:
        """Delete a model entry.

        Args:
            name: The unique model name to delete.

        Raises:
            KeyError: If no model with the given *name* exists.
        """
        cur = self._conn.execute("DELETE FROM models WHERE name = ?", (name,))
        self._conn.commit()

        if cur.rowcount == 0:
            raise KeyError(f"No model found with name '{name}'")

        logger.info("Deleted model '{}'", name)

    def export_csv(self, path: str) -> None:
        """Export the full leaderboard to CSV.

        Args:
            path: Destination file path for the CSV export.
        """
        df = self.get_leaderboard()
        df.to_csv(path, index=False)
        logger.info("Exported {} models to {}", len(df), path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        """Convert a sqlite3.Row to a plain dict, deserializing JSON fields.

        Args:
            row: A single database row.

        Returns:
            Dictionary with deserialized ``fold_scores`` and ``params``.
        """
        d = dict(row)
        if d.get("fold_scores") is not None:
            d["fold_scores"] = json.loads(d["fold_scores"])
        if d.get("params") is not None:
            d["params"] = json.loads(d["params"])
        return d

    def _rows_to_dataframe(self, rows: list[sqlite3.Row]) -> pd.DataFrame:
        """Convert a list of sqlite3.Row objects to a DataFrame.

        JSON columns are left as strings in the DataFrame for display purposes.

        Args:
            rows: List of database rows.

        Returns:
            A pandas DataFrame.
        """
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r) for r in rows])

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
        logger.info("Closed database connection for {}", self.db_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        try:
            self._conn.close()
        except Exception:
            pass
