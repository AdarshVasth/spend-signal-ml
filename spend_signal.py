"""Unsupervised transaction outlier detection for a user's own CSV data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

REQUIRED_COLUMNS = {"timestamp", "amount", "merchant", "category", "channel"}


class InputError(ValueError):
    """Raised when an uploaded transaction table cannot be analyzed safely."""


@dataclass(frozen=True)
class Analysis:
    results: pd.DataFrame
    flagged_count: int
    model: IsolationForest


def validate_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a clean copy of transactions or explain what must be fixed."""
    missing = REQUIRED_COLUMNS - set(frame.columns.str.strip().str.lower())
    if missing:
        raise InputError("Missing required columns: " + ", ".join(sorted(missing)))
    cleaned = frame.copy()
    cleaned.columns = cleaned.columns.str.strip().str.lower()
    if len(cleaned) < 20:
        raise InputError("Upload at least 20 transactions so the model has enough context.")
    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="coerce", utc=True)
    cleaned["amount"] = pd.to_numeric(cleaned["amount"], errors="coerce")
    if cleaned["timestamp"].isna().any():
        raise InputError("Some timestamps could not be read. Use ISO dates, such as 2026-11-01 14:30.")
    if cleaned["amount"].isna().any() or (cleaned["amount"] <= 0).any():
        raise InputError("Amounts must be numeric values greater than zero.")
    for column in ("merchant", "category", "channel"):
        cleaned[column] = cleaned[column].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    return cleaned


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    timestamp = frame["timestamp"]
    numeric = pd.DataFrame({
        "log_amount": np.log1p(frame["amount"].astype(float)),
        "hour": timestamp.dt.hour.astype(float),
        "day_of_week": timestamp.dt.dayofweek.astype(float),
        "weekend": (timestamp.dt.dayofweek >= 5).astype(float),
    }, index=frame.index)
    categories = pd.get_dummies(
        frame[["merchant", "category", "channel"]].astype(str),
        prefix=["merchant", "category", "channel"],
        dtype=float,
    )
    return pd.concat([numeric, categories], axis=1)


def analyze_transactions(frame: pd.DataFrame, contamination: float = 0.05, random_state: int = 42) -> Analysis:
    """Fit an Isolation Forest and return rows sorted from most unusual first.

    ``contamination`` is an expected outlier fraction, not a fraud probability.
    Lower decision scores indicate more unusual observations.
    """
    if not 0.01 <= contamination <= 0.20:
        raise InputError("Contamination must be between 1% and 20%.")
    clean = validate_transactions(frame)
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    features = _features(clean)
    prediction = model.fit_predict(features)
    clean["anomaly_score"] = model.decision_function(features)
    clean["is_unusual"] = prediction == -1
    clean["review_note"] = np.where(
        clean["is_unusual"],
        "Unusual compared with this file; review context manually.",
        "No unusual pattern detected by this model.",
    )
    clean = clean.sort_values("anomaly_score", ascending=True).reset_index(drop=True)
    return Analysis(clean, int((prediction == -1).sum()), model)
