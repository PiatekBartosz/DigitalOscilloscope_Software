#!/usr/bin/env python3
"""Plot both-channel spectra for the direct 50 MHz devkit-clock test."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAPTURES_DIR = (
    PROJECT_ROOT / "scripts" / "results" / "captures" / "thesis_measurements"
)
RESULTS_DIR = PROJECT_ROOT / "scripts" / "results" / "implementation"
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.capture_io import load_capture_csv
from analysis.metrics import compute_metrics
from analysis.plot_style import format_thesis_axis

SERIES_ID = "devikit_clk_test_50MHz"
SAMPLE_RATE_HZ = 50_000_000.0
SAMPLE_COUNT = 8192
CAPTURE_COUNT = 10
LEAKAGE_BINS = 3
PEAK_TO_SINE_POWER_DB = 10.0 * math.log10(2.0)


@dataclass(frozen=True)
class ChannelResult:
    channel: int
    color: str
    frequencies_hz: np.ndarray
    spectrum_dbfs: np.ndarray
    fundamental_hz: tuple[float, float]
    snr_db: tuple[float, float]
    sinad_db: tuple[float, float]
    enob: tuple[float, float]


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)), float(np.std(array, ddof=1))


def load_captures() -> list[tuple[np.ndarray, np.ndarray]]:
    paths = sorted(CAPTURES_DIR.glob(f"{SERIES_ID}_[0-9][0-9].csv"))
    if len(paths) != CAPTURE_COUNT:
        raise ValueError(f"expected {CAPTURE_COUNT} captures, found {len(paths)}")

    captures = []
    for path in paths:
        ch1, ch2, metadata = load_capture_csv(path)
        if len(ch1) != SAMPLE_COUNT or len(ch2) != SAMPLE_COUNT:
            raise ValueError(f"{path}: expected {SAMPLE_COUNT} samples per channel")
        if metadata.n_bits != 14:
            raise ValueError(f"{path}: expected 14-bit samples")
        if metadata.fields.get("series_id") != SERIES_ID:
            raise ValueError(f"{path}: unexpected series identifier")
        captures.append((ch1, ch2))
    return captures


def analyse_channel(
    captures: list[tuple[np.ndarray, np.ndarray]], channel_index: int
) -> ChannelResult:
    results = [
        compute_metrics(
            capture[channel_index],
            fs_hz=SAMPLE_RATE_HZ,
            n_bits=14,
            n_harmonics=5,
            window="hann",
            leakage_bins=LEAKAGE_BINS,
        )
        for capture in captures
    ]
    frequency_grids = [result.freqs_hz for result in results]
    if any(
        not np.array_equal(frequency_grids[0], grid) for grid in frequency_grids[1:]
    ):
        raise ValueError(f"channel {channel_index + 1}: inconsistent frequency grids")

    linear_spectra = np.asarray(
        [10.0 ** (result.spectrum_dbfs / 10.0) for result in results]
    )
    averaged_spectrum = (
        10.0 * np.log10(np.mean(linear_spectra, axis=0)) - PEAK_TO_SINE_POWER_DB
    )
    return ChannelResult(
        channel=channel_index + 1,
        color=("#0068B5", "#D1495B")[channel_index],
        frequencies_hz=frequency_grids[0],
        spectrum_dbfs=averaged_spectrum,
        fundamental_hz=mean_std([result.fundamental_freq_hz for result in results]),
        snr_db=mean_std([result.snr_db for result in results]),
        sinad_db=mean_std([result.sinad_db for result in results]),
        enob=mean_std([result.enob for result in results]),
    )


def polish(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def plot_results(results: list[ChannelResult]) -> Path:
    figure, axes = plt.subplots(2, 1, figsize=(9.4, 7.2), sharex=True, sharey=True)
    for axis, result in zip(axes, results):
        axis.plot(
            result.frequencies_hz / 1e6,
            result.spectrum_dbfs,
            color=result.color,
            linewidth=0.72,
        )
        axis.text(
            0.015,
            0.94,
            f"kanał {result.channel}, fs = 50 MHz",
            transform=axis.transAxes,
            va="top",
        )
        metrics = (
            f"f1 = {polish(result.fundamental_hz[0] / 1e6, 3)} MHz\n"
            f"SNR = {polish(result.snr_db[0])} ± {polish(result.snr_db[1])} dB\n"
            f"SINAD = {polish(result.sinad_db[0])} ± {polish(result.sinad_db[1])} dB\n"
            f"ENOB = {polish(result.enob[0])} ± {polish(result.enob[1])} bit"
        )
        axis.text(
            0.985,
            0.94,
            metrics,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9},
        )
        axis.set_xlim(0.0, SAMPLE_RATE_HZ / 2e6)
        axis.set_ylim(-115.0, 5.0)
        format_thesis_axis(axis, "Częstotliwość, MHz", "Poziom, dBFS", legend=False)

    figure.tight_layout()
    output = RESULTS_DIR / "devikit_clk_test_50MHz_ch1_ch2_spectra.png"
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    captures = load_captures()
    results = [analyse_channel(captures, index) for index in range(2)]
    output = plot_results(results)
    print(f"Saved: {output}")
    for result in results:
        print(
            f"channel {result.channel}: fundamental {result.fundamental_hz[0] / 1e6:.6f} MHz, "
            f"SNR {result.snr_db[0]:.3f} +/- {result.snr_db[1]:.3f} dB, "
            f"SINAD {result.sinad_db[0]:.3f} +/- {result.sinad_db[1]:.3f} dB, "
            f"ENOB {result.enob[0]:.3f} +/- {result.enob[1]:.3f} bit"
        )


if __name__ == "__main__":
    main()
