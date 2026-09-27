"""Head-to-head evaluation: DAAD vs static and naive-adaptive baselines.

Reports the three metrics that motivate the method:
* Drift false-alarm rate  - fraction of legitimate-drift windows wrongly alerted as faults.
* Fault recall            - fraction of true-fault windows detected.
* Precursor lead-time     - how early (in samples) a failure is flagged as a precursor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from novel.daad import DriftAwareDetector
from src.baselines import NaiveAdaptiveDetector, StaticDetector
from src.config import (
    ADAPT_RATE,
    FAULT_LEVEL,
    FIGURES_DIR,
    LABEL_COLUMN,
    PCA_COMPONENTS,
    PERSISTENCE,
    RESULTS_PATH,
    ensure_dirs,
)
from src.data_loader import load_telemetry, reference_normal
from src.stream import run_stream
from src.train import _normal_windows

_REGIME_COLORS = {
    "NORMAL": "#e8f0fe",
    "DRIFT": "#fff3cd",
    "FAULT": "#f8d7da",
    "PRECURSOR": "#ffe5cc",
}


def _build_detectors(df):
    ref = reference_normal(df, n=500)
    windows = _normal_windows(df)
    daad = DriftAwareDetector(
        k=PCA_COMPONENTS,
        fault_level=FAULT_LEVEL,
        persistence=PERSISTENCE,
        adapt_rate=ADAPT_RATE,
    ).fit(ref)
    daad.calibrate(windows)
    static = StaticDetector(k=PCA_COMPONENTS).fit(ref)
    static.calibrate(windows)
    naive = NaiveAdaptiveDetector(k=PCA_COMPONENTS, adapt_rate=ADAPT_RATE).fit(ref)
    naive.calibrate(windows)
    return {"DAAD (novel)": daad, "Static": static, "Naive-adaptive": naive}


def _metrics(res: pd.DataFrame) -> dict:
    is_drift = res["true"] == "DRIFT"
    is_fault = res["true"] == "FAULT"
    alerted = res["pred"] == "FAULT"
    drift_fa = float((alerted & is_drift).sum() / max(is_drift.sum(), 1))
    fault_recall = float((alerted & is_fault).sum() / max(is_fault.sum(), 1))
    total_alerts = int(alerted.sum())

    # Precursor lead-time: samples from the first correct precursor warning to failure onset.
    lead = 0
    prec = res[(res["pred"] == "PRECURSOR") & (res["true"] == "PRECURSOR")]
    faults = res[res["true"] == "FAULT"]
    if not prec.empty and not faults.empty:
        first_prec = prec["end"].iloc[0]
        later = faults[faults["start"] >= first_prec]
        if not later.empty:
            lead = int(later["start"].iloc[0] - first_prec)
    return {
        "drift_false_alarm": drift_fa,
        "fault_recall": fault_recall,
        "total_fault_alerts": total_alerts,
        "precursor_lead_samples": lead,
    }


def _contiguous_segments(labels: np.ndarray):
    segments = []
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[start]:
            segments.append((start, i, labels[start]))
            start = i
    segments.append((start, len(labels), labels[start]))
    return segments


def _plot(df: pd.DataFrame, res: pd.DataFrame) -> Path:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for start, end, label in _contiguous_segments(df[LABEL_COLUMN].to_numpy()):
        for ax in (ax1, ax2):
            ax.axvspan(start, end, color=_REGIME_COLORS.get(label, "#ffffff"), alpha=0.7)

    ax1.plot(res["end"], res["struct"], color="#c0392b", label="structure distance")
    ax1.plot(res["end"], res["shift"] / max(res["shift"].max(), 1e-9), color="#2980b9",
             label="operating-point shift (scaled)")
    ax1.set_ylabel("deviation scores")
    ax1.legend(loc="upper right")
    ax1.set_title("DAAD deviation scores (background = true regime)")

    pred_codes = {"NORMAL": 0, "DRIFT": 1, "PRECURSOR": 2, "FAULT": 3}
    ax2.scatter(res["end"], res["pred"].map(pred_codes), c=res["pred"].map(pred_codes),
                cmap="viridis", s=18)
    ax2.set_yticks(list(pred_codes.values()))
    ax2.set_yticklabels(list(pred_codes.keys()))
    ax2.set_ylabel("DAAD verdict")
    ax2.set_xlabel("sample")
    ax2.set_title("DAAD per-window classification")

    fig.tight_layout()
    path = FIGURES_DIR / "daad_timeline.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> None:
    ensure_dirs()
    df = load_telemetry()
    detectors = _build_detectors(df)

    table = []
    daad_res = None
    for name, det in detectors.items():
        res = run_stream(det, df)
        if name.startswith("DAAD"):
            daad_res = res
        m = _metrics(res)
        table.append((name, m))

    print("\n=== Detector comparison ===")
    header = f"{'Detector':<18}{'DriftFalseAlarm':>17}{'FaultRecall':>13}{'FaultAlerts':>13}{'PrecursorLead':>15}"
    print(header)
    print("-" * len(header))
    for name, m in table:
        print(f"{name:<18}{m['drift_false_alarm']:>17.2f}{m['fault_recall']:>13.2f}"
              f"{m['total_fault_alerts']:>13}{m['precursor_lead_samples']:>15}")

    daad_res.to_csv(RESULTS_PATH, index=False)
    fig_path = _plot(df, daad_res)
    print(f"\nSaved DAAD window results to {RESULTS_PATH}")
    print(f"Saved timeline figure to {fig_path}")


if __name__ == "__main__":
    main()
