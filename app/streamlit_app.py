"""Interactive replay of DAAD over the telemetry stream.

Run: streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import LABEL_COLUMN, METRIC_COLUMNS
from src.data_loader import load_telemetry, reference_normal
from src.evaluate import _build_detectors
from src.features import correlation_matrix
from src.stream import run_stream

st.set_page_config(page_title="DAAD - Drift-Aware Anomaly Disambiguation", page_icon="📈", layout="wide")

_VERDICT_STYLE = {
    "NORMAL": ("✅ Normal", "normal"),
    "DRIFT": ("🔄 Drift — adapting baseline, no alert", "off"),
    "PRECURSOR": ("⚠️ Precursor — early warning of failure", "inverse"),
    "FAULT": ("🚨 Fault — alert raised", "inverse"),
}


@st.cache_data
def compute():
    df = load_telemetry()
    detector = _build_detectors(df)["DAAD (novel)"]
    base_corr = correlation_matrix(reference_normal(df, 500))
    res = run_stream(detector, df)
    return df, res, base_corr


def main() -> None:
    st.title("DAAD — Drift-Aware Anomaly Disambiguation")
    st.caption(
        "A multivariate telemetry detector that separates legitimate concept drift from "
        "genuine faults using inter-metric correlation structure, and only adapts its "
        "baseline on confirmed drift."
    )

    df, res, base_corr = compute()

    idx = st.slider("Replay position (window)", 0, len(res) - 1, min(20, len(res) - 1))
    row = res.iloc[idx]
    end = int(row["end"])
    window = df[METRIC_COLUMNS].to_numpy()[int(row["start"]):end]

    verdict_text, color = _VERDICT_STYLE[row["pred"]]
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Current verdict")
        st.markdown(f"### {verdict_text}")
        st.metric("Structure distance", f"{row['struct']:.3f}")
        st.metric("Operating-point shift", f"{row['shift']:.2f}")
        st.metric("Monotonic trend", f"{row['trend']:.2f}")
        st.caption(f"True regime at this window: **{row['true']}**")

    with right:
        st.subheader("Correlation structure: baseline vs current window")
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
        for ax, mat, title in zip(axes, [base_corr, correlation_matrix(window)],
                                  ["Baseline", "Current window"]):
            sns.heatmap(mat, ax=ax, cmap="coolwarm", vmin=-1, vmax=1, cbar=False,
                        xticklabels=False, yticklabels=False)
            ax.set_title(title, fontsize=9)
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Telemetry up to now")
    shown = df.iloc[:end]
    fig2, ax = plt.subplots(figsize=(11, 3))
    for metric in ["cpu_util", "mem_util", "req_rate", "latency_ms"]:
        ax.plot(shown["t"], shown[metric], linewidth=0.8, label=metric)
    ax.axvline(end, color="red", linestyle="--", linewidth=1)
    ax.legend(loc="upper left", fontsize=7)
    st.pyplot(fig2)
    plt.close(fig2)

    st.subheader("Alert log")
    alerts = res.iloc[: idx + 1]
    alerts = alerts[alerts["pred"].isin(["FAULT", "PRECURSOR"])]
    if alerts.empty:
        st.info("No alerts raised yet.")
    else:
        st.dataframe(
            alerts[["end", "pred", "true", "struct", "shift"]].rename(
                columns={"end": "sample", "pred": "verdict", "true": "true_regime"}
            ),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
