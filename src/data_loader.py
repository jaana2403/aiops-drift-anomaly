"""Load infrastructure telemetry, or generate a labeled synthetic stream on demand.

The synthetic stream is driven by a single latent "load" signal so all metrics are
correlated under normal operation. Three failure modes are injected with ground-truth
labels so the detector can be evaluated:

* DRIFT      - the operating point shifts (traffic surge) but the inter-metric
               correlation structure is preserved (the whole system scales together).
* FAULT      - the correlation structure BREAKS (e.g. CPU high while throughput drops).
* PRECURSOR  - a slow monotonic ramp on one metric (memory leak) that decouples from
               load and eventually leads to a failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (
    LABEL_COLUMN,
    METRIC_COLUMNS,
    RANDOM_STATE,
    TELEMETRY_PATH,
    ensure_dirs,
)

# Linear map from latent load in [0, 1] to each metric: value = intercept + slope * load.
_METRIC_MODEL = {
    "cpu_util": (20.0, 60.0, 3.0),
    "mem_util": (30.0, 35.0, 2.5),
    "disk_io": (10.0, 90.0, 6.0),
    "net_in": (5.0, 80.0, 5.0),
    "net_out": (5.0, 75.0, 5.0),
    "req_rate": (50.0, 450.0, 20.0),
    "latency_ms": (20.0, 100.0, 6.0),
    "error_rate": (0.5, 2.0, 0.3),
}


def _load_signal(n: int, low: float, high: float, rng: np.random.Generator) -> np.ndarray:
    """Smooth latent load as a bounded random walk within [low, high]."""
    steps = rng.normal(0, 0.03, size=n)
    walk = np.cumsum(steps)
    walk = (walk - walk.min()) / (np.ptp(walk) + 1e-9)
    return low + walk * (high - low)


def _emit(load: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Turn a load signal into correlated metric columns."""
    block = {}
    for name, (intercept, slope, noise) in _METRIC_MODEL.items():
        block[name] = intercept + slope * load + rng.normal(0, noise, size=len(load))
    return block


def _segment(n: int, label: str, load_range: tuple[float, float], rng: np.random.Generator):
    load = _load_signal(n, load_range[0], load_range[1], rng)
    block = _emit(load, rng)
    return block, [label] * n


def generate_synthetic_telemetry(seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Build a multi-regime telemetry stream with ground-truth regime labels."""
    rng = np.random.default_rng(seed)
    blocks: list[dict[str, np.ndarray]] = []
    labels: list[str] = []

    def add(block, seg_labels):
        blocks.append(block)
        labels.extend(seg_labels)

    # 1) Baseline normal operation.
    add(*_segment(600, "NORMAL", (0.30, 0.55), rng))
    # 2) Traffic surge: operating point shifts up, structure preserved -> DRIFT.
    add(*_segment(300, "DRIFT", (0.55, 0.78), rng))
    # 3) Sustained new-normal at high load.
    add(*_segment(400, "NORMAL", (0.66, 0.82), rng))

    # 4) Transient fault: structure BREAKS at a magnitude comparable to the drift above.
    #    Throughput and egress decouple from load (drop) while CPU/disk keep following it.
    fault_block, fault_labels = _segment(150, "FAULT", (0.45, 0.60), rng)
    fault_block["req_rate"] = np.abs(rng.normal(90, 15, size=150))
    fault_block["net_out"] = np.abs(rng.normal(15, 4, size=150))
    fault_block["latency_ms"] += rng.normal(55, 10, size=150)
    fault_block["error_rate"] += rng.normal(6, 1.5, size=150)
    add(fault_block, fault_labels)

    # 5) Recovery to normal.
    add(*_segment(400, "NORMAL", (0.35, 0.60), rng))

    # 6) Memory leak precursor: slow monotonic ramp on mem_util, decoupled from load.
    prec_block, prec_labels = _segment(300, "PRECURSOR", (0.35, 0.60), rng)
    prec_block["mem_util"] = prec_block["mem_util"] + np.linspace(0, 75, 300)
    add(prec_block, prec_labels)

    # 7) Failure at the end of the leak: saturation triggers a hard structural break.
    fail_block, fail_labels = _segment(100, "FAULT", (0.35, 0.60), rng)
    fail_block["mem_util"] += np.linspace(75, 82, 100)
    fail_block["latency_ms"] += rng.normal(120, 15, size=100)
    fail_block["error_rate"] += rng.normal(15, 2, size=100)
    fail_block["req_rate"] = np.abs(rng.normal(50, 10, size=100))
    add(fail_block, fail_labels)

    # 8) Final normal tail.
    add(*_segment(300, "NORMAL", (0.30, 0.55), rng))

    frame = {col: np.concatenate([b[col] for b in blocks]) for col in METRIC_COLUMNS}
    df = pd.DataFrame(frame)
    df = df.clip(lower=0.0)
    df[LABEL_COLUMN] = labels
    df.insert(0, "t", np.arange(len(df)))
    return df


def load_telemetry(force_synthetic: bool = False) -> pd.DataFrame:
    """Load telemetry from disk, generating and caching a synthetic stream if absent."""
    ensure_dirs()
    if TELEMETRY_PATH.exists() and not force_synthetic:
        return pd.read_csv(TELEMETRY_PATH)
    df = generate_synthetic_telemetry()
    df.to_csv(TELEMETRY_PATH, index=False)
    return df


def reference_normal(df: pd.DataFrame, n: int = 500) -> np.ndarray:
    """Return the initial normal window used to fit the detector baseline."""
    normal = df[df[LABEL_COLUMN] == "NORMAL"]
    return normal[METRIC_COLUMNS].to_numpy()[:n]


if __name__ == "__main__":
    data = load_telemetry(force_synthetic=True)
    print(f"Generated {len(data)} samples across {data['label'].nunique()} regimes.")
    print(data["label"].value_counts())
    print(f"Saved to {TELEMETRY_PATH}")
