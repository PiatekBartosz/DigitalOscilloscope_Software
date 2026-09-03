#!/usr/bin/env python3
"""Generate spectra accompanying the SNR, SINAD and ENOB studies."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
CAPTURES_DIR = PROJECT_ROOT / "scripts" / "results" / "captures" / "thesis_measurements"
THESIS_GRAPHICS_DIR = REPOSITORY_ROOT / "mastersThesis" / "masters_thesis" / "graf"
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.capture_io import load_capture_csv  # noqa: E402
from analysis.metrics import average_centered_channels, compute_metrics  # noqa: E402
from analysis.plot_style import format_thesis_axis  # noqa: E402


DBFS_CORRECTION_DB = 10.0 * math.log10(2.0)
BIN_WIDTH_HZ = 80_000_000.0 / 8192.0
BIN_TO_HZ_DB = 10.0 * math.log10(BIN_WIDTH_HZ)


def save_dbfs_hz(figure, axes, output: Path, ymin: float, ymax: float) -> None:
    for axis in axes if isinstance(axes, (list, tuple, np.ndarray)) else (axes,):
        for line in axis.lines:
            line.set_ydata(line.get_ydata() - BIN_TO_HZ_DB)
        axis.set_ylabel("Poziom, dBFS/Hz")
        axis.set_ylim(ymin - BIN_TO_HZ_DB, ymax - BIN_TO_HZ_DB)
    figure.savefig(output, dpi=300, bbox_inches="tight")
CHANNELS = ("ch1", "ch2")
CHANNEL_LABELS = {"ch1": "kanał 1", "ch2": "kanał 2"}
GAIN_LEVELS = ((45.0, "45p00"), (47.5, "47p50"), (50.0, "50p00"),
               (52.5, "52p50"), (55.0, "55p00"))
FREQUENCIES = (
    (0.996, "996kHz"), (2.002, "2p002MHz"), (5.000, "5MHz"),
    (10.000, "10MHz"), (14.502, "14p502MHz"), (15.000, "15MHz"),
    (18.496, "18p496MHz"), (20.000, "20MHz"), (30.000, "30MHz"),
    (35.000, "35MHz"),
)


def load_series(pattern: str):
    paths = sorted(CAPTURES_DIR.glob(pattern))
    if len(paths) != 10:
        raise ValueError(f"expected 10 files for {pattern}, found {len(paths)}")
    captures = [load_capture_csv(path) for path in paths]
    for path, (ch1, ch2, meta) in zip(paths, captures):
        if len(ch1) != 8192 or len(ch2) != 8192:
            raise ValueError(f"{path}: unexpected capture depth")
        if not math.isclose(meta.fs_hz, 80_000_000.0):
            raise ValueError(f"{path}: unexpected sample rate")
    return captures


def averaged_spectrum(sample_sets: list[np.ndarray], fs_hz: float = 80_000_000.0):
    results = [
        compute_metrics(samples, fs_hz=fs_hz, n_harmonics=5, window="hann")
        for samples in sample_sets
    ]
    powers = np.asarray([10.0 ** (result.spectrum_dbfs / 10.0) for result in results])
    spectrum = 10.0 * np.log10(np.mean(powers, axis=0)) - DBFS_CORRECTION_DB
    return results[0].freqs_hz, spectrum


def channel_samples(captures, channel: str) -> list[np.ndarray]:
    index = 0 if channel == "ch1" else 1
    return [np.asarray(capture[index], dtype=np.float64) for capture in captures]


def plot_gain_spectra() -> None:
    figure, axes = plt.subplots(2, 1, figsize=(9.6, 7.4), sharex=True, sharey=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(GAIN_LEVELS)))
    for axis, channel in zip(axes, CHANNELS):
        for (gain, token), color in zip(GAIN_LEVELS, colors):
            captures = load_series(
                f"gain_2vpp_1to100_5MHz_g{token}_[0-9][0-9].csv"
            )
            frequencies, spectrum = averaged_spectrum(channel_samples(captures, channel))
            gain_db = 2.0 * (gain - 44.0)
            axis.plot(frequencies / 1e6, spectrum, linewidth=0.72,
                      color=color, label=f"{gain_db:.2f} dB".replace(".", ","))
        axis.text(0.015, 0.92, CHANNEL_LABELS[channel], transform=axis.transAxes)
        axis.set_xlim(0.0, 40.0)
        axis.set_ylim(-105.0, 5.0)
        format_thesis_axis(axis, "Częstotliwość, MHz", "Poziom, dBFS")
    figure.tight_layout()
    output = THESIS_GRAPHICS_DIR / "gain_influence_spectra.png"
    figure.savefig(output, dpi=300, bbox_inches="tight")
    output_hz = THESIS_GRAPHICS_DIR / "gain_influence_spectra_dbfs_hz.png"
    save_dbfs_hz(figure, axes, output_hz, -105.0, 5.0)
    plt.close(figure)
    print(f"Saved: {output}")


def plot_averaging_spectra() -> None:
    captures = load_series("harmonics_2vpp_1to100_profile2Vdiv_g49p66_*.csv")
    series = {
        "kanał 1": channel_samples(captures, "ch1"),
        "kanał 2": channel_samples(captures, "ch2"),
        "średnia kanałów": [
            average_centered_channels(ch1, ch2) for ch1, ch2, _ in captures
        ],
    }
    colors = ("#0068B5", "#D1495B", "#2A9D55")
    figure, axis = plt.subplots(figsize=(9.6, 5.4))
    for (label, samples), color in zip(series.items(), colors):
        frequencies, spectrum = averaged_spectrum(samples)
        axis.plot(frequencies / 1e6, spectrum, linewidth=0.78,
                  color=color, label=label)
    axis.set_xlim(0.0, 40.0)
    axis.set_ylim(-105.0, 5.0)
    format_thesis_axis(axis, "Częstotliwość, MHz", "Poziom, dBFS")
    figure.tight_layout()
    output = THESIS_GRAPHICS_DIR / "channel_averaging_spectra.png"
    figure.savefig(output, dpi=300, bbox_inches="tight")
    output_hz = THESIS_GRAPHICS_DIR / "channel_averaging_spectra_dbfs_hz.png"
    save_dbfs_hz(figure, axis, output_hz, -105.0, 5.0)
    plt.close(figure)
    print(f"Saved: {output}")


def plot_frequency_spectra() -> None:
    figure, axes = plt.subplots(2, 1, figsize=(9.8, 7.6), sharex=True, sharey=True)
    colors = plt.get_cmap("turbo")(np.linspace(0.04, 0.96, len(FREQUENCIES)))
    for axis, channel in zip(axes, CHANNELS):
        for (frequency_mhz, token), color in zip(FREQUENCIES, colors):
            captures = load_series(
                f"freq_2vpp_1to100_{token}_[0-9][0-9].csv"
            )
            frequencies, spectrum = averaged_spectrum(channel_samples(captures, channel))
            axis.plot(frequencies / 1e6, spectrum, linewidth=0.66,
                      color=color, label=f"{frequency_mhz:g} MHz".replace(".", ","))
        axis.text(0.82, 0.92, CHANNEL_LABELS[channel], transform=axis.transAxes)
        axis.set_xlim(0.0, 40.0)
        axis.set_ylim(-110.0, 5.0)
        format_thesis_axis(axis, "Częstotliwość widma, MHz", "Poziom, dBFS")
    figure.tight_layout()
    output = THESIS_GRAPHICS_DIR / "frequency_quality_spectra.png"
    figure.savefig(output, dpi=300, bbox_inches="tight")
    output_hz = THESIS_GRAPHICS_DIR / "frequency_quality_spectra_dbfs_hz.png"
    save_dbfs_hz(figure, axes, output_hz, -110.0, 5.0)
    plt.close(figure)
    print(f"Saved: {output}")


def main() -> None:
    THESIS_GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
    plot_gain_spectra()
    plot_averaging_spectra()
    plot_frequency_spectra()


if __name__ == "__main__":
    main()
