"""Plot a thesis-ready comparison of the original and corrected offset paths.

The input is a version-2 capture CSV saved by the oscilloscope application.
CH1 is labelled as the original path and CH2 as the corrected path, matching
the ``offset_path_comparison_100kHz_1to100.csv`` verification capture.

Examples, run from DigitalOscilloscope_Software:

    python scripts/plot_offset_correction.py scripts/results/captures/offset_path_comparison_100kHz_1to100.csv

    python scripts/plot_offset_correction.py scripts/results/captures/offset_path_comparison_100kHz_1to100.csv

Use the Matplotlib toolbar to zoom to the desired transition and save the
cropped figure manually. The vertical axis is the signed ADC amplitude divided
by the ADC full-scale peak-to-peak range. Thus -0.5 and +0.5 are the negative
and positive limits of the nominal ADC input range.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from analysis.capture_io import load_capture_csv
from analysis.plot_style import format_thesis_axis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=pathlib.Path, help="capture CSV to display")
    parser.add_argument(
        "--start-us",
        type=float,
        default=0.0,
        help="start of the displayed interval in microseconds (default: 0)",
    )
    parser.add_argument(
        "--duration-us",
        type=float,
        default=None,
        help="duration of the displayed interval in microseconds (default: full capture)",
    )
    parser.add_argument(
        "--save",
        type=pathlib.Path,
        help="save the complete displayed view to this PNG, PDF, or SVG file",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="do not open the interactive Matplotlib window",
    )
    return parser.parse_args()


def _selected_indices(
    sample_count: int, sample_rate_hz: float, start_us: float, duration_us: float | None
) -> slice:
    if start_us < 0:
        raise ValueError("--start-us must not be negative")
    if duration_us is not None and duration_us <= 0:
        raise ValueError("--duration-us must be positive")

    start = int(np.floor(start_us * 1e-6 * sample_rate_hz))
    stop = sample_count
    if duration_us is not None:
        stop = int(np.ceil((start_us + duration_us) * 1e-6 * sample_rate_hz))
    start = min(start, sample_count)
    stop = min(max(stop, start), sample_count)
    if stop <= start:
        raise ValueError("selected interval contains no samples")
    return slice(start, stop)


def main() -> int:
    args = parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required: pip install matplotlib", file=sys.stderr)
        return 2

    try:
        ch1, ch2, metadata = load_capture_csv(args.capture)
        selected = _selected_indices(len(ch1), metadata.fs_hz, args.start_us, args.duration_us)
    except (OSError, ValueError) as error:
        print(f"Unable to prepare plot: {error}", file=sys.stderr)
        return 2

    full_scale_codes = float(1 << metadata.n_bits)
    midpoint_code = full_scale_codes / 2.0
    time_us = np.arange(selected.start, selected.stop, dtype=np.float64) / metadata.fs_hz * 1e6
    ch1_normalized = (ch1[selected].astype(np.float64) - midpoint_code) / full_scale_codes
    ch2_normalized = (ch2[selected].astype(np.float64) - midpoint_code) / full_scale_codes

    figure, axis = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    axis.plot(time_us, ch1_normalized, color="#C95B2B", linewidth=1.1, label="Kanał 1 - tor pierwotny")
    axis.plot(time_us, ch2_normalized, color="#147D9A", linewidth=1.1, label="Kanał 2 - tor po korekcie")

    format_thesis_axis(
        axis,
        "Czas (µs)",
        r"Amplituda znormalizowana $V_{\mathrm{diff}} / V_{\mathrm{FS,pp}}$",
    )
    axis.axhline(0.0, color="#404040", linewidth=0.7, alpha=0.7)
    figure.canvas.manager.set_window_title("Offset-path correction comparison")

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.save, dpi=300, bbox_inches="tight")
        print(f"Saved: {args.save}")
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
