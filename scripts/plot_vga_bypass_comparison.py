#!/usr/bin/env python3
"""Compare spectra and dynamic metrics with the channel 1 VGA bypassed."""

from __future__ import annotations

import argparse
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
CAPTURES_DIR = PROJECT_ROOT / "scripts" / "results" / "captures"
RESULTS_DIR = PROJECT_ROOT / "scripts" / "results" / "implementation"
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.capture_io import load_capture_csv
from analysis.metrics import compute_metrics
from analysis.plot_style import format_thesis_axis

DEFAULT_SERIES_ID = "compare_ch1_bypassed_vga_ch2_normal"
EXPECTED_CAPTURES = 10
EXPECTED_SAMPLE_RATE_HZ = 80_000_000.0
EXPECTED_SAMPLE_COUNT = 8192
EXPECTED_TONE_HZ = 5_000_000.0
PEAK_TO_SINE_POWER_DB = 10.0 * math.log10(2.0)
LEAKAGE_BINS = 3


@dataclass(frozen=True)
class ChannelSummary:
    channel: int
    configuration: str
    color: str
    frequencies_hz: np.ndarray
    spectrum_dbfs: np.ndarray
    snr_db: tuple[float, float]
    sinad_db: tuple[float, float]
    enob: tuple[float, float]
    thd_db: tuple[float, float]
    sfdr_db: tuple[float, float]
    noise_floor_dbfs: tuple[float, float]
    minimum_code: int
    maximum_code: int
    noncoherent_captures: int
    overflow_flags: int


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)), float(np.std(array, ddof=1))


def validate_captures(
    paths: list[Path], series_id: str
) -> list[tuple[np.ndarray, np.ndarray, object]]:
    if len(paths) != EXPECTED_CAPTURES:
        raise ValueError(f"expected {EXPECTED_CAPTURES} captures, found {len(paths)}")

    captures = [load_capture_csv(path) for path in paths]
    for path, (ch1, ch2, metadata) in zip(paths, captures):
        if len(ch1) != EXPECTED_SAMPLE_COUNT or len(ch2) != EXPECTED_SAMPLE_COUNT:
            raise ValueError(f"{path}: expected {EXPECTED_SAMPLE_COUNT} samples")
        if not math.isclose(metadata.fs_hz, EXPECTED_SAMPLE_RATE_HZ):
            raise ValueError(f"{path}: unexpected sample rate {metadata.fs_hz}")
        if metadata.n_bits != 14:
            raise ValueError(f"{path}: unexpected ADC resolution {metadata.n_bits}")
        if metadata.fields.get("measurement_type") != "harmonics":
            raise ValueError(f"{path}: not a harmonic measurement")
        if not math.isclose(
            float(metadata.fields["generator_frequency_hz"]), EXPECTED_TONE_HZ
        ):
            raise ValueError(f"{path}: unexpected generator frequency")
        if metadata.fields.get("series_id") != series_id:
            raise ValueError(f"{path}: unexpected series identifier")
    return captures


def sfdr_db(result) -> float:
    spectrum = np.asarray(result.spectrum_dbfs, dtype=np.float64)
    fundamental_index = result.fundamental_bin - 1
    spur_spectrum = spectrum.copy()
    lo = max(fundamental_index - LEAKAGE_BINS, 0)
    hi = min(fundamental_index + LEAKAGE_BINS + 1, len(spur_spectrum))
    spur_spectrum[lo:hi] = -np.inf
    return float(spectrum[fundamental_index] - np.max(spur_spectrum))


def aggregate_channel(
    captures: list[tuple[np.ndarray, np.ndarray, object]], channel_index: int
) -> ChannelSummary:
    configuration = (
        "kanał 1, VGA pominięty" if channel_index == 0 else "kanał 2, VGA aktywny"
    )
    color = "#0068B5" if channel_index == 0 else "#D1495B"
    results = []
    thd_values = []
    sfdr_values = []
    minimum_codes = []
    maximum_codes = []
    overflow_flags = 0

    for ch1, ch2, metadata in captures:
        samples = (ch1, ch2)[channel_index]
        result = compute_metrics(
            samples,
            fs_hz=metadata.fs_hz,
            n_bits=metadata.n_bits,
            n_harmonics=5,
            window="hann",
            leakage_bins=LEAKAGE_BINS,
        )
        if not math.isclose(result.fundamental_freq_hz, EXPECTED_TONE_HZ):
            raise ValueError(
                f"{configuration}: detected tone {result.fundamental_freq_hz} Hz"
            )
        results.append(result)
        harmonic_ratio = sum(
            10.0 ** (level / 10.0) for level in result.harmonic_levels_dbc
        )
        thd_values.append(10.0 * math.log10(harmonic_ratio))
        sfdr_values.append(sfdr_db(result))
        minimum_codes.append(int(np.min(samples)))
        maximum_codes.append(int(np.max(samples)))
        overflow_flags += metadata.fields.get("firmware_overflow") == "1"

    frequency_grids = [result.freqs_hz for result in results]
    if any(
        not np.array_equal(frequency_grids[0], grid) for grid in frequency_grids[1:]
    ):
        raise ValueError(f"{configuration}: inconsistent frequency grids")

    linear_spectra = np.asarray(
        [10.0 ** (result.spectrum_dbfs / 10.0) for result in results]
    )
    averaged_spectrum = (
        10.0 * np.log10(np.mean(linear_spectra, axis=0)) - PEAK_TO_SINE_POWER_DB
    )
    return ChannelSummary(
        channel=channel_index + 1,
        configuration=configuration,
        color=color,
        frequencies_hz=frequency_grids[0],
        spectrum_dbfs=averaged_spectrum,
        snr_db=mean_std([result.snr_db for result in results]),
        sinad_db=mean_std([result.sinad_db for result in results]),
        enob=mean_std([result.enob for result in results]),
        thd_db=mean_std(thd_values),
        sfdr_db=mean_std(sfdr_values),
        noise_floor_dbfs=mean_std(
            [result.noise_floor_dbfs - PEAK_TO_SINE_POWER_DB for result in results]
        ),
        minimum_code=min(minimum_codes),
        maximum_code=max(maximum_codes),
        noncoherent_captures=sum(not result.is_coherent for result in results),
        overflow_flags=overflow_flags,
    )


def plot_spectra(summaries: list[ChannelSummary], output_stem: str) -> Path:
    figure, axis = plt.subplots(figsize=(9.4, 5.2))
    for summary in summaries:
        axis.plot(
            summary.frequencies_hz / 1e6,
            summary.spectrum_dbfs,
            color=summary.color,
            linewidth=0.72,
            label=summary.configuration,
        )

    metrics_text = "\n".join(
        (
            f"{summary.configuration}: "
            f"SNR {summary.snr_db[0]:.2f} dB, "
            f"SINAD {summary.sinad_db[0]:.2f} dB, "
            f"ENOB {summary.enob[0]:.2f} bit"
        ).replace(".", ",")
        for summary in summaries
    )
    axis.text(
        0.015,
        0.04,
        metrics_text,
        transform=axis.transAxes,
        va="bottom",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9},
    )
    axis.set_xlim(0.0, 40.0)
    axis.set_ylim(-110.0, 5.0)
    format_thesis_axis(axis, "Częstotliwość, MHz", "Poziom, dBFS")
    figure.tight_layout()
    output = RESULTS_DIR / f"{output_stem}_spectrum_comparison.png"
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output


def save_summary(summaries: list[ChannelSummary], output_stem: str) -> Path:
    output = RESULTS_DIR / f"{output_stem}_metrics_summary.csv"
    fields = (
        "channel",
        "configuration",
        "capture_count",
        "snr_db_mean",
        "snr_db_std",
        "sinad_db_mean",
        "sinad_db_std",
        "enob_mean",
        "enob_std",
        "thd_db_mean",
        "thd_db_std",
        "sfdr_db_mean",
        "sfdr_db_std",
        "noise_floor_dbfs_per_bin_mean",
        "noise_floor_dbfs_per_bin_std",
        "minimum_code",
        "maximum_code",
        "noncoherent_captures",
        "firmware_overflow_flags",
    )
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "channel": summary.channel,
                    "configuration": summary.configuration,
                    "capture_count": EXPECTED_CAPTURES,
                    "snr_db_mean": f"{summary.snr_db[0]:.6f}",
                    "snr_db_std": f"{summary.snr_db[1]:.6f}",
                    "sinad_db_mean": f"{summary.sinad_db[0]:.6f}",
                    "sinad_db_std": f"{summary.sinad_db[1]:.6f}",
                    "enob_mean": f"{summary.enob[0]:.6f}",
                    "enob_std": f"{summary.enob[1]:.6f}",
                    "thd_db_mean": f"{summary.thd_db[0]:.6f}",
                    "thd_db_std": f"{summary.thd_db[1]:.6f}",
                    "sfdr_db_mean": f"{summary.sfdr_db[0]:.6f}",
                    "sfdr_db_std": f"{summary.sfdr_db[1]:.6f}",
                    "noise_floor_dbfs_per_bin_mean": f"{summary.noise_floor_dbfs[0]:.6f}",
                    "noise_floor_dbfs_per_bin_std": f"{summary.noise_floor_dbfs[1]:.6f}",
                    "minimum_code": summary.minimum_code,
                    "maximum_code": summary.maximum_code,
                    "noncoherent_captures": summary.noncoherent_captures,
                    "firmware_overflow_flags": summary.overflow_flags,
                }
            )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series-id", default=DEFAULT_SERIES_ID)
    parser.add_argument("--output-stem", default="vga_bypass")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(CAPTURES_DIR.glob(f"{args.series_id}_[0-9][0-9].csv"))
    captures = validate_captures(paths, args.series_id)
    summaries = [aggregate_channel(captures, index) for index in range(2)]
    plot_path = plot_spectra(summaries, args.output_stem)
    summary_path = save_summary(summaries, args.output_stem)
    print(f"Saved: {plot_path}")
    print(f"Saved: {summary_path}")
    for summary in summaries:
        print(
            f"{summary.configuration}: SNR {summary.snr_db[0]:.3f} +/- {summary.snr_db[1]:.3f} dB, "
            f"SINAD {summary.sinad_db[0]:.3f} +/- {summary.sinad_db[1]:.3f} dB, "
            f"ENOB {summary.enob[0]:.3f} +/- {summary.enob[1]:.3f} bit, "
            f"THD {summary.thd_db[0]:.3f} +/- {summary.thd_db[1]:.3f} dB, "
            f"SFDR {summary.sfdr_db[0]:.3f} +/- {summary.sfdr_db[1]:.3f} dB"
        )


if __name__ == "__main__":
    main()
