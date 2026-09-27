"""Ablation detectors that share DAAD's baseline but drop its disambiguation logic.

* StaticDetector        - flags any deviation as a fault, never adapts. Suffers alert
                          storms whenever the system legitimately drifts.
* NaiveAdaptiveDetector - adapts on every deviation to suppress those storms, and as a
                          result slowly absorbs real faults into "normal" (boiling-frog),
                          missing slow-building failures.

Comparing both against DAAD isolates the value of structural disambiguation + gated
adaptation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from novel.daad import StructuralBaseline


class _ThresholdDetector:
    """Shared deviation gate; subclasses decide labeling and adaptation policy."""

    def __init__(self, k: int = 3, struct_threshold: float = 0.15, shift_threshold: float = 3.0):
        self.k = k
        self.struct_threshold = struct_threshold
        self.shift_threshold = shift_threshold

    def fit(self, x: np.ndarray):
        self.baseline_ = StructuralBaseline(self.k).fit(x)
        return self

    def calibrate(self, windows: list[np.ndarray], q: float = 99.0):
        shifts = [self.baseline_.shift_score(w) for w in windows]
        structs = [self.baseline_.struct_score(w) for w in windows]
        self.shift_threshold = float(np.percentile(shifts, q)) * 1.15
        self.struct_threshold = float(np.percentile(structs, q)) * 1.25
        return self

    def _scores(self, window: np.ndarray):
        shift = self.baseline_.shift_score(window)
        struct = self.baseline_.struct_score(window)
        trend = self.baseline_.trend_score(window)
        deviation = shift > self.shift_threshold or struct > self.struct_threshold
        return shift, struct, trend, deviation


class StaticDetector(_ThresholdDetector):
    """Never adapts: every deviation is reported as a fault."""

    def classify(self, window: np.ndarray) -> dict:
        window = np.asarray(window, dtype=float)
        shift, struct, trend, deviation = self._scores(window)
        return {
            "label": "FAULT" if deviation else "NORMAL",
            "shift": shift,
            "struct": struct,
            "trend": trend,
            "adapted": False,
        }


class NaiveAdaptiveDetector(_ThresholdDetector):
    """Adapts on every deviation regardless of cause (the boiling-frog failure mode)."""

    def __init__(self, *args, adapt_rate: float = 0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.adapt_rate = adapt_rate

    def classify(self, window: np.ndarray) -> dict:
        window = np.asarray(window, dtype=float)
        shift, struct, trend, deviation = self._scores(window)
        label = "FAULT" if deviation else "NORMAL"
        adapted = False
        if deviation:
            self.baseline_.adapt(window, self.adapt_rate)
            adapted = True
        return {
            "label": label,
            "shift": shift,
            "struct": struct,
            "trend": trend,
            "adapted": adapted,
        }
