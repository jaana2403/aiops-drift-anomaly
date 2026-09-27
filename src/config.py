"""Central configuration: paths, telemetry schema, window sizes, and DAAD defaults."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42

# Telemetry schema: metric column -> human-readable label (used in plots and the demo).
METRICS = {
    "cpu_util": "CPU Utilization (%)",
    "mem_util": "Memory Utilization (%)",
    "disk_io": "Disk I/O (MB/s)",
    "net_in": "Network In (MB/s)",
    "net_out": "Network Out (MB/s)",
    "req_rate": "Request Rate (req/s)",
    "latency_ms": "Latency (ms)",
    "error_rate": "Error Rate (%)",
}
METRIC_COLUMNS = list(METRICS.keys())

# Regime labels the detector must distinguish.
LABELS = {"NORMAL": 0, "DRIFT": 1, "FAULT": 2, "PRECURSOR": 3}
LABEL_NAMES = {v: k for k, v in LABELS.items()}
LABEL_COLUMN = "label"

# Windowing over the telemetry stream.
WINDOW_SIZE = 90
WINDOW_STEP = 30
PCA_COMPONENTS = 2

# DAAD defaults (train.py recalibrates thresholds from normal data).
STRUCT_THRESHOLD = 1.0
SHIFT_THRESHOLD = 3.0
FAULT_LEVEL = 5.0
PERSISTENCE = 2
ADAPT_RATE = 0.3
TREND_THRESHOLD = 1.5

MODEL_PATH = MODELS_DIR / "daad.joblib"
TELEMETRY_PATH = RAW_DIR / "telemetry.csv"
RESULTS_PATH = PROCESSED_DIR / "stream_results.csv"


def ensure_dirs() -> None:
    """Create all output directories used across the pipeline."""
    for directory in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
