"""Exploratory plots: metric timelines and the correlation-structure contrast.

The correlation heatmaps make the core DAAD idea visible: under drift the structure is
preserved (looks like normal), while under a fault it is visibly broken.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (
    FIGURES_DIR,
    LABEL_COLUMN,
    METRIC_COLUMNS,
    METRICS,
    ensure_dirs,
)
from src.data_loader import load_telemetry
from src.features import correlation_matrix

sns.set_theme(style="whitegrid")

_REGIME_COLORS = {
    "NORMAL": "#e8f0fe",
    "DRIFT": "#fff3cd",
    "FAULT": "#f8d7da",
    "PRECURSOR": "#ffe5cc",
}


def _contiguous_segments(labels: np.ndarray):
    segments, start = [], 0
    for i in range(1, len(labels)):
        if labels[i] != labels[start]:
            segments.append((start, i, labels[start]))
            start = i
    segments.append((start, len(labels), labels[start]))
    return segments


def plot_timeline(df) -> Path:
    metrics = ["cpu_util", "mem_util", "req_rate", "latency_ms"]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 8), sharex=True)
    labels = df[LABEL_COLUMN].to_numpy()
    for ax, metric in zip(axes, metrics):
        for start, end, label in _contiguous_segments(labels):
            ax.axvspan(start, end, color=_REGIME_COLORS.get(label, "#fff"), alpha=0.7)
        ax.plot(df["t"], df[metric], color="#2c3e50", linewidth=0.8)
        ax.set_ylabel(METRICS[metric], fontsize=8)
    axes[-1].set_xlabel("sample")
    axes[0].set_title("Telemetry over time (background = true regime)")
    fig.tight_layout()
    path = FIGURES_DIR / "telemetry_timeline.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_structure_contrast(df) -> Path:
    """Correlation structure of normal vs drift vs fault windows, side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, regime in zip(axes, ["NORMAL", "DRIFT", "FAULT"]):
        block = df[df[LABEL_COLUMN] == regime][METRIC_COLUMNS].to_numpy()
        corr = correlation_matrix(block)
        sns.heatmap(corr, ax=ax, cmap="coolwarm", vmin=-1, vmax=1, cbar=False,
                    xticklabels=METRIC_COLUMNS, yticklabels=METRIC_COLUMNS)
        ax.set_title(f"{regime} correlation")
        ax.tick_params(labelsize=6)
    fig.suptitle("Drift preserves structure; a fault breaks it", fontsize=13)
    fig.tight_layout()
    path = FIGURES_DIR / "structure_contrast.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> None:
    ensure_dirs()
    df = load_telemetry()
    print(df[METRIC_COLUMNS].describe().round(2))
    print("\nRegime counts:")
    print(df[LABEL_COLUMN].value_counts())
    p1 = plot_timeline(df)
    p2 = plot_structure_contrast(df)
    print(f"\nSaved {p1}")
    print(f"Saved {p2}")


if __name__ == "__main__":
    main()
