#!/usr/bin/env python3
"""Plot both-channel spectra for the after_clk_output_pll_from_devkit series."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAPTURE_DIR = PROJECT_ROOT / "scripts/results/captures"
CAPTURE_PATTERN = "after_clk_output_pll_from_devkit_*.csv"
OUTPUT_PATH = CAPTURE_DIR / "after_clk_output_pll_from_devkit_spectra.png"
CAPTURE_COUNT = 10
N_HARMONICS = 5
WINDOW_TYPE = "hann"
LEAKAGE_BINS = 3

sys.path.insert(0, str(PROJECT_ROOT))

from analysis.capture_io import CaptureMeta, load_capture_csv  # noqa: E402
from analysis.metrics import SpectralMetrics, compute_metrics  # noqa: E402
from analysis.plot_style import format_thesis_axis  # noqa: E402


def _calculate_channel(
    channel_index: int,
) -> tuple[np.ndarray, np.ndarray, list[SpectralMetrics], CaptureMeta]:
    """Calculate a power-averaged spectrum and per-capture quality metrics."""
    paths = sorted(CAPTURE_DIR.glob(CAPTURE_PATTERN))
    if len(paths) != CAPTURE_COUNT:
        raise ValueError(
            f"expected {CAPTURE_COUNT} captures matching {CAPTURE_PATTERN}, "
            f"found {len(paths)}"
        )

    frequencies: np.ndarray | None = None
    linear_powers: list[np.ndarray] = []
    results: list[SpectralMetrics] = []
    first_metadata = None

    for path in paths:
        ch1, ch2, metadata = load_capture_csv(path)
        samples = ch1 if channel_index == 1 else ch2
        result = compute_metrics(
            samples,
            fs_hz=metadata.fs_hz,
            n_bits=metadata.n_bits,
            n_harmonics=N_HARMONICS,
            window=WINDOW_TYPE,
            leakage_bins=LEAKAGE_BINS,
        )

        # Identyczna korekta poziomu jak w analizie poprawki zegara.
        levels_dbfs = result.spectrum_dbfs - 10.0 * np.log10(2.0)
        if frequencies is None:
            frequencies = result.freqs_hz
            first_metadata = metadata
        elif not np.array_equal(frequencies, result.freqs_hz):
            raise ValueError(f"inconsistent frequency grid in {path}")

        linear_powers.append(10.0 ** (levels_dbfs / 10.0))
        results.append(result)

    assert frequencies is not None and first_metadata is not None
    averaged_dbfs = 10.0 * np.log10(np.mean(linear_powers, axis=0))
    return frequencies, averaged_dbfs, results, first_metadata


def _quality_text(results: list[SpectralMetrics]) -> str:
    snr_db = float(np.mean([result.snr_db for result in results]))
    sinad_db = float(np.mean([result.sinad_db for result in results]))
    enob = float(np.mean([result.enob for result in results]))
    fundamental_mhz = float(
        np.mean([result.fundamental_freq_hz for result in results]) / 1e6
    )
    return (
        f"f₁: {fundamental_mhz:.3f} MHz\n"
        f"SNR: {snr_db:.2f} dB\n"
        f"SINAD: {sinad_db:.2f} dB\n"
        f"ENOB: {enob:.2f} bit"
    )


def plot_both_channels() -> None:
    channel_data = {channel: _calculate_channel(channel) for channel in (1, 2)}
    metadata = channel_data[1][3]
    sample_count = int(metadata.fields.get("capture_depth", 8192))
    resolution_khz = metadata.fs_hz / sample_count / 1e3

    figure, axes = plt.subplots(
        2, 1, figsize=(12.0, 8.0), sharex=True, sharey=True
    )
    colors = {1: "#D1495B", 2: "#0068B5"}

    for axis, channel in zip(axes, (1, 2)):
        frequencies, levels, results, _metadata = channel_data[channel]
        axis.plot(
            frequencies / 1e6,
            levels,
            color=colors[channel],
            linewidth=0.75,
            label=f"kanał {channel}, średnia mocy z {CAPTURE_COUNT} rejestracji",
        )
        format_thesis_axis(axis, "Częstotliwość, MHz", "Poziom, dBFS")
        axis.set_xlim(0.0, metadata.fs_hz / 2e6)
        axis.text(
            0.985,
            0.96,
            _quality_text(results),
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9},
        )

    parameters = (
        f"Seria: after_clk_output_pll_from_devkit, {CAPTURE_COUNT} rejestracji\n"
        "Kanały: 1 i 2, sprzężenie DC, dzielnik 1:100, wzmocnienie 50,00%\n"
        f"FFT: {metadata.fs_hz / 1e6:.0f} MS/s, {sample_count} próbek, "
        f"{resolution_khz:.3f} kHz, okno Hann, 5 harmonicznych, "
        f"szerokość integracji {2 * LEAKAGE_BINS + 1} koszyków\n"
        f"Przetwornik: {metadata.n_bits} bitów, Offset Binary, decymacja "
        f"{metadata.fields.get('decim_factor', '1')}"
    )
    figure.text(0.08, 0.015, parameters, ha="left", va="bottom", fontsize=8.5)
    figure.subplots_adjust(
        left=0.08, right=0.98, bottom=0.16, top=0.98, hspace=0.18
    )
    figure.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(f"Wygenerowano: {OUTPUT_PATH}")
    for channel in (1, 2):
        print(f"Kanał {channel}:\n{_quality_text(channel_data[channel][2])}")


def main() -> None:
    plot_both_channels()


if __name__ == "__main__":
    main()
