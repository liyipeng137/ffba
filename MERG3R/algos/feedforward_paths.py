import sys
from pathlib import Path


FEEDFORWARD_ROOT = Path(__file__).resolve().parents[1] / "feedforward"


def ensure_feedforward_on_path():
    if not FEEDFORWARD_ROOT.is_dir():
        raise FileNotFoundError(f"Feedforward model directory not found: {FEEDFORWARD_ROOT}")

    feedforward_root_str = str(FEEDFORWARD_ROOT)
    if feedforward_root_str not in sys.path:
        sys.path.insert(0, feedforward_root_str)
