"""Fit the DAAD baseline on initial normal traffic and calibrate its thresholds."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from novel.daad import DriftAwareDetector
from src.config import (
    ADAPT_RATE,
    FAULT_LEVEL,
    LABEL_COLUMN,
    METRIC_COLUMNS,
    MODEL_PATH,
    PCA_COMPONENTS,
    PERSISTENCE,
    WINDOW_SIZE,
    ensure_dirs,
)
from src.data_loader import load_telemetry, reference_normal
from src.features import window_bounds


def _normal_windows(df, window=WINDOW_SIZE):
    """Windows from the INITIAL normal regime, matching the baseline fit distribution.

    Only the first contiguous normal block is used: later "normal" segments sit at a
    shifted operating point (post-drift) and would otherwise inflate the thresholds.
    """
    labels = df[LABEL_COLUMN].to_numpy()
    first_shift = np.argmax(labels != "NORMAL")
    x = df[METRIC_COLUMNS].to_numpy()[:first_shift]
    return [x[start:end] for start, end in window_bounds(len(x), window, window)]


def main() -> None:
    ensure_dirs()
    df = load_telemetry()

    ref = reference_normal(df, n=500)
    detector = DriftAwareDetector(
        k=PCA_COMPONENTS,
        fault_level=FAULT_LEVEL,
        persistence=PERSISTENCE,
        adapt_rate=ADAPT_RATE,
    ).fit(ref)

    detector.calibrate(_normal_windows(df))

    joblib.dump(detector, MODEL_PATH)
    print("Fitted DAAD baseline on 500 normal samples.")
    print(f"  shift_threshold  = {detector.shift_threshold:.3f}")
    print(f"  struct_threshold = {detector.struct_threshold:.3f}")
    print(f"Saved detector to {MODEL_PATH}")


if __name__ == "__main__":
    main()
