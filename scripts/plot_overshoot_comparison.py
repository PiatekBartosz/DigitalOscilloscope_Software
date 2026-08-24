"""Plot aligned rising edges from a two-channel capture.

The plot normalizes each channel to its own settled step amplitude. It is
therefore suitable for comparing overshoot even when the two analog paths have
slightly different absolute gains.

Example, run from DigitalOscilloscope_Software:

    python scripts/plot_overshoot_comparison.py \
        scripts/results/captures/debug_overshoot.csv \
        --save ../mastersThesis/masters_thesis/graf/overshoot_comparison_80MSps_100kHz_1to100.png
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
    parser.add_argument("--save", type=pathlib.Path, help="output PNG, PDF, or SVG path")
    parser.add_argument("--no-show", action="store_true", help="do not open the Matplotlib window")
    return parser.parse_args()


def _edge_indices(samples: np.ndarray, rising: bool) -> np.ndarray:
    low = np.percentile(samples, 20)
    high = np.percentile(samples, 80)
    threshold = (low + high) / 2.0
    if rising:
        return np.flatnonzero((samples[:-1] < threshold) & (samples[1:] >= threshold)) + 1
    return np.flatnonzero((samples[:-1] >= threshold) & (samples[1:] < threshold)) + 1


def _representative_edge(samples: np.ndarray, rising: bool, pre: int, post: int) -> int:
    edges = _edge_indices(samples, rising)
    valid = edges[(edges >= pre) & (edges + post < len(samples))]
    if not len(valid):
        raise ValueError("capture does not contain a complete transition")
    return int(valid[len(valid) // 2])


def _normalized_transition(samples: np.ndarray, edge: int, rising: bool, pre: int, post: int) -> np.ndarray:
    before = float(np.mean(samples[edge - pre : edge - 2]))
    after = float(np.mean(samples[edge + 20 : edge + post]))
    low, high = (before, after) if rising else (after, before)
    if high <= low:
        raise ValueError("could not determine the settled step amplitude")
    return (samples[edge - pre : edge + post].astype(np.float64) - low) / (high - low)


def _plot_transition(axis, ch1: np.ndarray, ch2: np.ndarray, sample_rate_hz: float) -> None:
    pre, post = 8, 32
    edge1 = _representative_edge(ch1, True, pre, post)
    edge2 = _representative_edge(ch2, True, pre, post)
    time_ns = np.arange(-pre, post, dtype=np.float64) / sample_rate_hz * 1e9
    axis.plot(time_ns, _normalized_transition(ch1, edge1, True, pre, post), color="#C95B2B", linewidth=1.3,
              label="Kanał 1")
    axis.plot(time_ns, _normalized_transition(ch2, edge2, True, pre, post), color="#147D9A", linewidth=1.3,
              label="Kanał 2")
    format_thesis_axis(axis, "Czas względem zbocza (ns)", "Amplituda znormalizowana do skoku")
    axis.axhline(0.0, color="#404040", linewidth=0.7, alpha=0.7)
    axis.axhline(1.0, color="#404040", linewidth=0.7, alpha=0.7)
    axis.set_ylim(-0.12, 1.32)


def main() -> int:
    args = parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required: pip install matplotlib", file=sys.stderr)
        return 2

    try:
        ch1, ch2, metadata = load_capture_csv(args.capture)
        figure, axis = plt.subplots(figsize=(6.6, 4.8), constrained_layout=True)
        _plot_transition(axis, ch1, ch2, metadata.fs_hz)
    except (OSError, ValueError) as error:
        print(f"Unable to prepare plot: {error}", file=sys.stderr)
        return 2

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.save, dpi=300, bbox_inches="tight")
        print(f"Saved: {args.save}")
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
