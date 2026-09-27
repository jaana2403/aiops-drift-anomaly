"""Windowing and structural-descriptor math used by the DAAD algorithm.

The key quantities:
* sliding-window bounds over a stream
* correlation matrix of a window (scale-invariant view of inter-metric structure)
* monotonic-trend score (how steadily a metric ramps in one direction)
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-9


def window_bounds(n: int, window: int, step: int) -> list[tuple[int, int]]:
    """Return (start, end) index pairs for a sliding window over ``n`` samples."""
    if n < window:
        return []
    return [(s, s + window) for s in range(0, n - window + 1, step)]


def correlation_matrix(x: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix, robust to constant (zero-variance) columns."""
    x = np.asarray(x, dtype=float)
    centered = x - x.mean(axis=0, keepdims=True)
    std = centered.std(axis=0, keepdims=True)
    safe = np.where(std < _EPS, 1.0, std)
    z = centered / safe
    corr = (z.T @ z) / len(x)
    # Columns that were constant carry no correlation information.
    dead = (std < _EPS).ravel()
    corr[dead, :] = 0.0
    corr[:, dead] = 0.0
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def monotonic_score(x: np.ndarray) -> float:
    """Strongest sustained directional drift across columns.

    Compares the first third of the window to the last third, in units of the metric's
    in-window standard deviation. A steady ramp scores high even when per-step noise
    hides the sign of individual differences; stationary noise scores near 0.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 6:
        return 0.0
    third = n // 3
    delta = x[-third:].mean(axis=0) - x[:third].mean(axis=0)
    std = x.std(axis=0) + _EPS
    return float(np.max(np.abs(delta / std)))
