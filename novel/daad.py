"""DAAD - Drift-Aware Anomaly Disambiguation.

Novel method. A conventional anomaly detector only answers "is this window
abnormal?". DAAD adds a second stage that answers "WHY?", classifying every flagged
deviation as one of:

    DRIFT      - a legitimate operating-point shift; adapt the baseline, do not alert.
    FAULT      - a genuine anomaly; alert, and never adapt the baseline to it.
    PRECURSOR  - a slow structural degradation trending toward failure; early-warn.

The discriminator is the preservation of inter-metric CORRELATION STRUCTURE:

* Legitimate drift moves the system to a new operating point but keeps the
  relationships between metrics intact (they scale together). => structure preserved.
* A fault violates those relationships (e.g. CPU high while throughput collapses).
  => structure broken.

Baseline adaptation is GATED: the baseline is updated only after drift is confirmed
over several consecutive windows, and never on a fault. This structurally prevents the
"boiling-frog" failure of naive adaptive detectors, which slowly absorb a real fault
into their notion of normal and miss it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import monotonic_score

_EPS = 1e-9


class StructuralBaseline:
    """Reference model of normal operation shared by DAAD and the ablation baselines.

    The baseline captures the *correlation manifold* of normal telemetry: after
    standardizing, the principal subspace spanned by the top-``k`` directions describes
    how the metrics co-vary (driven by shared load). Two departure scores are derived:

    * shift  - total distance of the window mean from the baseline operating point.
    * struct - the OFF-MANIFOLD part of that distance, i.e. the component that the normal
               correlation structure cannot explain.

    Legitimate drift moves the operating point *along* the manifold (metrics scale
    together) -> large shift, near-zero off-manifold residual. A fault or a leak pushes
    the system *off* the manifold (a relationship is violated) -> large residual.
    """

    def __init__(self, k: int = 2):
        self.k = k

    def fit(self, x: np.ndarray) -> "StructuralBaseline":
        x = np.asarray(x, dtype=float)
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0) + _EPS
        z = (x - self.mean_) / self.std_
        _, _, vt = np.linalg.svd(z - z.mean(axis=0), full_matrices=False)
        self.components_ = vt[: self.k]  # (k, d) orthonormal manifold directions
        return self

    def _dz(self, window: np.ndarray) -> np.ndarray:
        return (window.mean(axis=0) - self.mean_) / self.std_

    def shift_score(self, window: np.ndarray) -> float:
        """Total standardized distance of the window mean from the operating point."""
        return float(np.linalg.norm(self._dz(window)))

    def struct_score(self, window: np.ndarray) -> float:
        """Magnitude of the off-manifold (structure-violating) component of the shift."""
        dz = self._dz(window)
        projected = self.components_.T @ (self.components_ @ dz)
        return float(np.linalg.norm(dz - projected))

    def trend_score(self, window: np.ndarray) -> float:
        z = (window - self.mean_) / self.std_
        return monotonic_score(z)

    def adapt(self, window: np.ndarray, rate: float) -> None:
        """EWMA update of the operating point toward a confirmed-drift window.

        Only the operating point (mean/std) moves; the manifold directions are the
        structure and stay fixed, so the detector keeps watching for structural breaks.
        """
        self.mean_ = (1 - rate) * self.mean_ + rate * window.mean(axis=0)
        self.std_ = (1 - rate) * self.std_ + rate * (window.std(axis=0) + _EPS)


class DriftAwareDetector:
    """Two-stage detector: deviation gate + off-manifold disambiguation with gated adaptation."""

    def __init__(
        self,
        k: int = 2,
        struct_threshold: float = 1.0,
        shift_threshold: float = 3.0,
        fault_level: float = 5.0,
        persistence: int = 2,
        adapt_rate: float = 0.3,
    ):
        self.k = k
        self.struct_threshold = struct_threshold
        self.shift_threshold = shift_threshold
        self.fault_level = fault_level
        self.persistence = persistence
        self.adapt_rate = adapt_rate

    def fit(self, x: np.ndarray) -> "DriftAwareDetector":
        self.baseline_ = StructuralBaseline(self.k).fit(x)
        self._drift_run = 0
        return self

    def calibrate(self, windows: list[np.ndarray], q: float = 99.0) -> "DriftAwareDetector":
        """Set the deviation thresholds from the spread of scores on initial-normal windows."""
        shifts = [self.baseline_.shift_score(w) for w in windows]
        structs = [self.baseline_.struct_score(w) for w in windows]
        self.shift_threshold = float(np.percentile(shifts, q)) * 1.15
        self.struct_threshold = float(np.percentile(structs, q)) * 1.5
        return self

    def classify(self, window: np.ndarray) -> dict:
        """Classify a single window and adapt the baseline only on confirmed drift."""
        window = np.asarray(window, dtype=float)
        shift = self.baseline_.shift_score(window)
        struct = self.baseline_.struct_score(window)
        trend = self.baseline_.trend_score(window)

        deviation = shift > self.shift_threshold or struct > self.struct_threshold
        label = "NORMAL"
        adapted = False

        if deviation:
            if struct <= self.struct_threshold:
                # On the manifold but the operating point moved: legitimate drift.
                label = "DRIFT"
                self._drift_run += 1
                if self._drift_run >= self.persistence:
                    self.baseline_.adapt(window, self.adapt_rate)
                    adapted = True
            else:
                # Off the manifold: a genuine anomaly, never adapted to. A small/still-
                # growing excursion is an early-warning precursor; a large one is a fault.
                self._drift_run = 0
                label = "PRECURSOR" if struct <= self.fault_level else "FAULT"
        else:
            self._drift_run = 0

        return {
            "label": label,
            "shift": shift,
            "struct": struct,
            "trend": trend,
            "adapted": adapted,
        }
