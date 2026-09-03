#!/usr/bin/env python3
"""Compare channel-1 spectra used to diagnose acquisition-path noise."""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
CAPTURES_DIR = PROJECT_ROOT / "scripts" / "results" / "captures" / "thesis_measurements"
RESULTS_DIR = PROJECT_ROOT / "scripts" / "results" / "implementation"
THESIS_GRAPHICS_DIR = REPOSITORY_ROOT / "mastersThesis" / "masters_thesis" / "graf"
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.capture_io import load_capture_csv  # noqa: E402
from analysis.metrics import compute_metrics  # noqa: E402
from analysis.plot_style import format_thesis_axis  # noqa: E402

SAMPLE_COUNT = 8192
CAPTURE_COUNT = 10
PEAK_TO_SINE_POWER_DB = 10.0 * math.log10(2.0)


@dataclass(frozen=True)
class Configuration:
    identifier: str
    label: str
    sample_rate_hz: float
    expected_tone_hz: float
    color: str


CONFIGURATIONS = (
    Configuration(
        identifier="harmonics_2vpp_1to1_profile20mVdiv_g49p50",
        label="VGA w torze, zegar 80 MHz",
        sample_rate_hz=80_000_000.0,
        expected_tone_hz=5_000_000.0,
        color="#7F7F7F",
    ),
    Configuration(
        identifier="harmonics_2vpp_1to1_5MHz_osc_fix_ch1_vga_bypass",
        label="VGA zbocznikowany, zegar 80 MHz",
        sample_rate_hz=80_000_000.0,
        expected_tone_hz=5_000_000.0,
        color="#0068B5",
    ),
    Configuration(
        identifier="devikit_clk_test_50MHz",
        label="VGA zbocznikowany, zegar DE0-Nano 50 MHz",
        sample_rate_hz=50_000_000.0,
        expected_tone_hz=3_125_000.0,
        color="#D1495B",
    ),
)


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)), float(np.std(array, ddof=1))


def analyse(configuration: Configuration) -> dict[str, object]:
    paths = sorted(CAPTURES_DIR.glob(f"{configuration.identifier}_[0-9][0-9].csv"))
    if len(paths) != CAPTURE_COUNT:
        raise ValueError(
            f"{configuration.identifier}: expected {CAPTURE_COUNT} captures, "
            f"found {len(paths)}"
        )

    results = []
    levels_dbfs = []
    for path in paths:
        channel_1, _, metadata = load_capture_csv(path)
        if len(channel_1) != SAMPLE_COUNT:
            raise ValueError(f"{path}: expected {SAMPLE_COUNT} channel-1 samples")
        if metadata.n_bits != 14:
            raise ValueError(f"{path}: expected 14-bit samples")
        result = compute_metrics(
            channel_1,
            fs_hz=configuration.sample_rate_hz,
            n_bits=14,
            n_harmonics=5,
            window="hann",
            leakage_bins=3,
        )
        if not math.isclose(result.fundamental_freq_hz, configuration.expected_tone_hz):
            raise ValueError(
                f"{path}: detected tone {result.fundamental_freq_hz:g} Hz, "
                f"expected {configuration.expected_tone_hz:g} Hz"
            )
        results.append(result)
        code_span = float(np.max(channel_1) - np.min(channel_1))
        levels_dbfs.append(20.0 * math.log10(code_span / 16383.0))

    linear_spectra = np.asarray(
        [10.0 ** (result.spectrum_dbfs / 10.0) for result in results]
    )
    averaged_spectrum_dbfs = (
        10.0 * np.log10(np.mean(linear_spectra, axis=0)) - PEAK_TO_SINE_POWER_DB
    )
    normalized_frequency = results[0].freqs_hz / configuration.sample_rate_hz

    return {
        "configuration": configuration,
        "normalized_frequency": normalized_frequency,
        "spectrum_dbfs": averaged_spectrum_dbfs,
        "level_dbfs": mean_std(levels_dbfs),
        "snr_db": mean_std([result.snr_db for result in results]),
        "sinad_db": mean_std([result.sinad_db for result in results]),
        "enob": mean_std([result.enob for result in results]),
    }


def save_summary(rows: list[dict[str, object]]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / "noise_source_diagnostic_metrics.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "configuration",
                "sample_rate_hz",
                "tone_hz",
                "level_dbfs_mean",
                "level_dbfs_std",
                "snr_db_mean",
                "snr_db_std",
                "sinad_db_mean",
                "sinad_db_std",
                "enob_mean",
                "enob_std",
            )
        )
        for row in rows:
            configuration = row["configuration"]
            writer.writerow(
                (
                    configuration.identifier,
                    f"{configuration.sample_rate_hz:.0f}",
                    f"{configuration.expected_tone_hz:.0f}",
                    *(f"{value:.6f}" for key in ("level_dbfs", "snr_db", "sinad_db", "enob") for value in row[key]),
                )
            )
    return output


def save_figure(rows: list[dict[str, object]]) -> Path:
    THESIS_GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9.4, 5.6))
    for row in rows:
        configuration = row["configuration"]
        axis.plot(
            row["normalized_frequency"],
            row["spectrum_dbfs"],
            color=configuration.color,
            linewidth=0.72,
            label=configuration.label,
        )
    axis.set_xlim(0.0, 0.5)
    axis.set_ylim(-130.0, 5.0)
    format_thesis_axis(
        axis,
        "Częstotliwość znormalizowana, f/fs",
        "Poziom, dBFS",
    )
    figure.tight_layout()
    output = THESIS_GRAPHICS_DIR / "noise_source_diagnostic_ch1_spectra.png"
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:
    rows = [analyse(configuration) for configuration in CONFIGURATIONS]
    summary = save_summary(rows)
    figure = save_figure(rows)
    print(f"Saved: {summary}")
    print(f"Saved: {figure}")
    for row in rows:
        configuration = row["configuration"]
        print(
            f"{configuration.label}: "
            f"SNR {row['snr_db'][0]:.3f} +/- {row['snr_db'][1]:.3f} dB, "
            f"SINAD {row['sinad_db'][0]:.3f} +/- {row['sinad_db'][1]:.3f} dB, "
            f"ENOB {row['enob'][0]:.3f} +/- {row['enob'][1]:.3f} bit"
        )


if __name__ == "__main__":
    main()
