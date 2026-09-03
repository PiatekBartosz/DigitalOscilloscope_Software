#!/usr/bin/env python3
"""Analyse the five-point decimation study and generate summary figures."""

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
CAPTURES_DIR = (
    PROJECT_ROOT
    / "scripts"
    / "results"
    / "captures"
    / "thesis_measurements"
    / "decimation"
)
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "results" / "implementation"
THESIS_GRAPHICS_DIR = REPOSITORY_ROOT / "mastersThesis" / "masters_thesis" / "graf"
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.capture_io import load_capture_csv  # noqa: E402
from analysis.metrics import compute_metrics  # noqa: E402


POINTS = (
    ("5 MHz, D = 1", "decimation_5MHz_D1_*.csv", 5_000_000.0, 1),
    ("1 MHz, D = 5", "decimation_1MHz_D5_*.csv", 1_000_000.0, 5),
    ("500 kHz, D = 10", "decimation_500kHz_D10_*.csv", 500_000.0, 10),
    ("100 kHz, D = 50", "decimation_100kHz_D50_*.csv", 100_000.0, 50),
    ("50 kHz, D = 100", "decimation_50kHz_D100_*.csv", 50_000.0, 100),
)
COLORS = ("#0068B5", "#D1495B")
PEAK_TO_SINE_POWER_DB = 10.0 * math.log10(2.0)


@dataclass(frozen=True)
class ChannelResult:
    spectrum_dbfs: np.ndarray
    snr_mean: float
    snr_std: float
    sinad_mean: float
    sinad_std: float
    enob_mean: float
    enob_std: float
    peak_min: int
    peak_max: int
    noncoherent_captures: int


@dataclass(frozen=True)
class PointResult:
    label: str
    tone_hz: float
    decimation: int
    fs_hz: float
    frequencies_hz: np.ndarray
    channels: tuple[ChannelResult, ChannelResult]
    overflow_flags: int


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)), float(np.std(array, ddof=1))


def analyse() -> list[PointResult]:
    point_results = []
    for label, pattern, tone_hz, decimation in POINTS:
        paths = sorted(CAPTURES_DIR.glob(pattern))
        if len(paths) != 10:
            raise ValueError(f"{label}: expected 10 captures, found {len(paths)}")
        captures = [load_capture_csv(path) for path in paths]
        expected_fs = 80_000_000.0 / decimation
        overflow_flags = 0
        for path, (_, _, metadata) in zip(paths, captures):
            if not math.isclose(metadata.fs_hz, expected_fs):
                raise ValueError(f"{path}: unexpected sample rate {metadata.fs_hz}")
            if int(metadata.fields["decim_factor"]) != decimation:
                raise ValueError(f"{path}: unexpected decimation")
            if not math.isclose(
                float(metadata.fields["generator_frequency_hz"]), tone_hz
            ):
                raise ValueError(f"{path}: unexpected generator frequency")
            overflow_flags += metadata.fields.get("firmware_overflow") == "1"

        channel_results = []
        frequencies = None
        for channel_index in range(2):
            metrics = []
            peak_min = 2**14 - 1
            peak_max = 0
            for ch1, ch2, metadata in captures:
                samples = (ch1, ch2)[channel_index]
                peak_min = min(peak_min, int(np.min(samples)))
                peak_max = max(peak_max, int(np.max(samples)))
                metrics.append(
                    compute_metrics(
                        samples,
                        fs_hz=metadata.fs_hz,
                        n_bits=metadata.n_bits,
                        n_harmonics=5,
                        window="hann",
                        leakage_bins=3,
                    )
                )
            noncoherent_captures = sum(not result.is_coherent for result in metrics)
            spectra_power = np.asarray(
                [10.0 ** (result.spectrum_dbfs / 10.0) for result in metrics]
            )
            spectrum_dbfs = (
                10.0 * np.log10(np.mean(spectra_power, axis=0))
                - PEAK_TO_SINE_POWER_DB
            )
            snr = mean_std([result.snr_db for result in metrics])
            sinad = mean_std([result.sinad_db for result in metrics])
            enob = mean_std([result.enob for result in metrics])
            frequencies = metrics[0].freqs_hz
            channel_results.append(
                ChannelResult(
                    spectrum_dbfs=spectrum_dbfs,
                    snr_mean=snr[0],
                    snr_std=snr[1],
                    sinad_mean=sinad[0],
                    sinad_std=sinad[1],
                    enob_mean=enob[0],
                    enob_std=enob[1],
                    peak_min=peak_min,
                    peak_max=peak_max,
                    noncoherent_captures=noncoherent_captures,
                )
            )
        point_results.append(
            PointResult(
                label=label,
                tone_hz=tone_hz,
                decimation=decimation,
                fs_hz=expected_fs,
                frequencies_hz=frequencies,
                channels=tuple(channel_results),
                overflow_flags=overflow_flags,
            )
        )
    return point_results


def plot_spectra(results: list[PointResult]) -> Path:
    figure, axes = plt.subplots(5, 1, figsize=(10.2, 12.2), constrained_layout=True)
    for axis, point in zip(axes, results):
        for channel_index, channel in enumerate(point.channels):
            axis.plot(
                point.frequencies_hz / 1e6,
                channel.spectrum_dbfs,
                color=COLORS[channel_index],
                linewidth=0.7,
                label=f"kanał {channel_index + 1}",
            )
        axis.set_title(
            f"{point.label}, $F_s$ = {point.fs_hz / 1e6:g} MHz",
            fontsize=10,
        )
        axis.set_xlim(0.0, point.fs_hz / 2e6)
        axis.set_ylim(-120.0, 5.0)
        axis.set_ylabel("Poziom, dBFS")
        axis.grid(True, alpha=0.28)
        axis.legend(loc="lower left", ncol=2, fontsize=8)
    axes[-1].set_xlabel("Częstotliwość, MHz")
    output = THESIS_GRAPHICS_DIR / "decimation_study_spectra.png"
    figure.savefig(output, dpi=220)
    plt.close(figure)
    return output


def plot_metrics(results: list[PointResult]) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 4.2), constrained_layout=True)
    x = np.arange(len(results))
    labels = [f"D={point.decimation}\n{point.tone_hz / 1e3:g} kHz" for point in results]
    metrics = (
        ("snr_mean", "snr_std", "SNR, dB"),
        ("sinad_mean", "sinad_std", "SINAD, dB"),
        ("enob_mean", "enob_std", "ENOB, bit"),
    )
    for axis, (mean_name, std_name, ylabel) in zip(axes, metrics):
        for channel_index in range(2):
            means = [getattr(point.channels[channel_index], mean_name) for point in results]
            stds = [getattr(point.channels[channel_index], std_name) for point in results]
            offset = (-0.08, 0.08)[channel_index]
            axis.errorbar(
                x + offset,
                means,
                yerr=stds,
                marker="o",
                capsize=3,
                color=COLORS[channel_index],
                label=f"kanał {channel_index + 1}",
            )
        axis.set_xticks(x, labels, fontsize=8)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.28)
    axes[0].legend()
    output = THESIS_GRAPHICS_DIR / "decimation_study_quality_metrics.png"
    figure.savefig(output, dpi=220)
    plt.close(figure)
    return output


def write_summary(results: list[PointResult]) -> Path:
    output = OUTPUT_DIR / "decimation_study_quality_metrics.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "point",
                "tone_hz",
                "decimation",
                "effective_fs_hz",
                "channel",
                "snr_db_mean",
                "snr_db_std",
                "sinad_db_mean",
                "sinad_db_std",
                "enob_mean",
                "enob_std",
                "minimum_code",
                "maximum_code",
                "noncoherent_captures",
                "firmware_overflow_flags",
            )
        )
        for point in results:
            for channel_index, channel in enumerate(point.channels, start=1):
                writer.writerow(
                    (
                        point.label,
                        f"{point.tone_hz:.12g}",
                        point.decimation,
                        f"{point.fs_hz:.12g}",
                        channel_index,
                        f"{channel.snr_mean:.6f}",
                        f"{channel.snr_std:.6f}",
                        f"{channel.sinad_mean:.6f}",
                        f"{channel.sinad_std:.6f}",
                        f"{channel.enob_mean:.6f}",
                        f"{channel.enob_std:.6f}",
                        channel.peak_min,
                        channel.peak_max,
                        channel.noncoherent_captures,
                        point.overflow_flags,
                    )
                )
    return output


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    THESIS_GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
    results = analyse()
    spectra_path = plot_spectra(results)
    metrics_path = plot_metrics(results)
    summary_path = write_summary(results)
    print(spectra_path)
    print(metrics_path)
    print(summary_path)
    for point in results:
        values = " | ".join(
            f"CH{index}: SNR={channel.snr_mean:.2f} dB, "
            f"SINAD={channel.sinad_mean:.2f} dB, ENOB={channel.enob_mean:.2f} bit, "
            f"codes={channel.peak_min}..{channel.peak_max}, "
            f"noncoherent={channel.noncoherent_captures}/10"
            for index, channel in enumerate(point.channels, start=1)
        )
        print(f"{point.label}: {values}")


if __name__ == "__main__":
    main()
