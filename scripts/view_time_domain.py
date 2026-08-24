"""Open an interactive, zoomable time-domain view of a saved capture.

Run from DigitalOscilloscope_Software:

    python scripts/view_time_domain.py capture.csv

Use the Matplotlib toolbar to zoom, pan, reset the view, or save an image.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from analysis.capture_io import load_capture_csv
from analysis.plot_style import format_thesis_axis


def _time_axis(samples: int, sample_rate_hz: float) -> tuple[np.ndarray, str]:
    """Return a human-friendly time axis and its unit label."""
    seconds = np.arange(samples, dtype=np.float64) / sample_rate_hz
    duration = seconds[-1] if len(seconds) else 0.0
    if duration >= 1.0:
        return seconds, "s"
    if duration >= 1e-3:
        return seconds * 1e3, "ms"
    if duration >= 1e-6:
        return seconds * 1e6, "µs"
    return seconds * 1e9, "ns"


def _signed_codes(raw: np.ndarray, n_bits: int) -> np.ndarray:
    return raw.astype(np.int32) - (1 << (n_bits - 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=pathlib.Path, help="capture CSV to display")
    parser.add_argument(
        "--channels",
        choices=("1", "2", "both"),
        default="both",
        help="channels to display (default: both)",
    )
    parser.add_argument(
        "--raw-codes",
        action="store_true",
        help="plot unsigned ADC codes instead of centred signed codes",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required: pip install matplotlib", file=sys.stderr)
        return 2

    try:
        ch1, ch2, metadata = load_capture_csv(args.capture)
    except (OSError, ValueError) as error:
        print(f"Unable to load capture: {error}", file=sys.stderr)
        return 2

    time, time_unit = _time_axis(len(ch1), metadata.fs_hz)
    if args.raw_codes:
        ch1_plot, ch2_plot = ch1, ch2
        y_label = "Kod ADC bez znaku"
    else:
        ch1_plot = _signed_codes(ch1, metadata.n_bits)
        ch2_plot = _signed_codes(ch2, metadata.n_bits)
        y_label = "Kod ADC ze znakiem"

    fig, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
    if args.channels in ("1", "both"):
        axis.plot(time, ch1_plot, color="tab:orange", linewidth=0.8, label="Kanał 1")
    if args.channels in ("2", "both"):
        axis.plot(time, ch2_plot, color="tab:cyan", linewidth=0.8, label="Kanał 2")

    format_thesis_axis(axis, f"Czas ({time_unit})", y_label)
    fig.canvas.manager.set_window_title(f"Time-domain capture: {args.capture.name}")
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
