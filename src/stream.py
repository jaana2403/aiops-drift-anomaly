"""Replay a detector over the telemetry stream, window by window."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import LABEL_COLUMN, METRIC_COLUMNS, WINDOW_SIZE, WINDOW_STEP
from src.features import window_bounds


def _majority_label(labels: list[str]) -> str:
    values, counts = np.unique(labels, return_counts=True)
    return str(values[np.argmax(counts)])


def run_stream(
    detector,
    df: pd.DataFrame,
    window: int = WINDOW_SIZE,
    step: int = WINDOW_STEP,
) -> pd.DataFrame:
    """Slide ``detector`` across the stream and collect per-window predictions."""
    x = df[METRIC_COLUMNS].to_numpy()
    labels = df[LABEL_COLUMN].to_numpy() if LABEL_COLUMN in df else None
    rows = []
    for start, end in window_bounds(len(x), window, step):
        result = detector.classify(x[start:end])
        row = {
            "start": start,
            "end": end,
            "pred": result["label"],
            "shift": result["shift"],
            "struct": result["struct"],
            "trend": result["trend"],
            "adapted": result["adapted"],
        }
        if labels is not None:
            row["true"] = _majority_label(list(labels[start:end]))
        rows.append(row)
    return pd.DataFrame(rows)
