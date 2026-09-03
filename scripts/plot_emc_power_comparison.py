#!/usr/bin/env python3
"""Compare captures made with a desktop PC and a battery-powered laptop."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
CAPTURES_DIR = PROJECT_ROOT / "scripts" / "results" / "captures" / "thesis_measurements"
RESULTS_DIR = PROJECT_ROOT / "scripts" / "results" / "completed_studies"
THESIS_GRAPHICS_DIR = REPOSITORY_ROOT / "mastersThesis" / "masters_thesis" / "graf"
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.capture_io import load_capture_csv  # noqa: E402
from analysis.metrics import compute_metrics, compute_noise_metrics  # noqa: E402
from analysis.plot_style import format_thesis_axis  # noqa: E402


SAMPLE_RATE_HZ = 80_000_000.0
SAMPLE_COUNT = 8192
ADC_RANGE_VPP = 2.0
FUNDAMENTAL_HZ = 5_000_000.0
SPECTRUM_DBFS_CORRECTION_DB = 10.0 * math.log10(2.0)
CONDITIONS = {
    "pc_monitors": {
        "label": "PC i monitory",
        "color": "#6A3D9A",
        "linestyle": "-",
    },
    "laptop_battery": {
        "label": "laptop na baterii",
        "color": "#1B9E77",
        "linestyle": "--",
    },
}
CHANNELS = ("ch1", "ch2")
CHANNEL_LABELS = {"ch1": "kanał 1", "ch2": "kanał 2"}


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)), float(np.std(array, ddof=1))


def load_series(condition: str, measurement: str):
    pattern = f"power_{condition}_{measurement}_*.csv"
    paths = sorted(CAPTURES_DIR.glob(pattern))
    if len(paths) != 10:
        raise ValueError(f"expected 10 files for {pattern}, found {len(paths)}")
    captures = [load_capture_csv(path) for path in paths]
    expected_type = "noise" if measurement == "noise" else "harmonics"
    for path, (ch1, ch2, meta) in zip(paths, captures):
        if len(ch1) != SAMPLE_COUNT or len(ch2) != SAMPLE_COUNT:
            raise ValueError(f"{path}: unexpected capture depth")
        if not math.isclose(meta.fs_hz, SAMPLE_RATE_HZ):
            raise ValueError(f"{path}: unexpected sample rate")
        fields = meta.fields
        expected = {
            "decim_factor": "1",
            "trigger_mode": "off",
            "measurement_type": expected_type,
            "sense_ch1_vpp": "2",
            "sense_ch2_vpp": "2",
        }
        if measurement == "noise":
            expected.update(
                {
                    "input_condition": "grounded",
                    "generator_waveform": "off",
                    "attenuation_ch1": "1:1",
                    "attenuation_ch2": "1:1",
                    "gain_pct_ch1": "50.0000",
                    "gain_pct_ch2": "50.0000",
                }
            )
        else:
            expected.update(
                {
                    "input_condition": "generator_splitter",
                    "generator_waveform": "sine",
                    "generator_frequency_hz": "5000000",
                    "generator_amplitude_vpk": "12.5",
                    "attenuation_ch1": "1:100",
                    "attenuation_ch2": "1:100",
                    "gain_pct_ch1": "49.6600",
                    "gain_pct_ch2": "49.6600",
                }
            )
        for key, value in expected.items():
            if fields.get(key) != value:
                raise ValueError(
                    f"{path}: expected {key}={value}, got {fields.get(key)}"
                )
    return captures


def select_channel(capture, channel: str) -> np.ndarray:
    return capture[0] if channel == "ch1" else capture[1]


def aggregate_data() -> dict:
    aggregated: dict = {}
    for condition in CONDITIONS:
        noise_captures = load_series(condition, "noise")
        dynamic_captures = load_series(condition, "p5")
        aggregated[condition] = {"noise": {}, "dynamic": {}}
        for channel in CHANNELS:
            noise_samples = [
                np.asarray(select_channel(capture, channel), dtype=np.float64)
                for capture in noise_captures
            ]
            noise_results = [
                compute_noise_metrics(
                    samples,
                    fs_hz=SAMPLE_RATE_HZ,
                    adc_range_vpp=ADC_RANGE_VPP,
                    window="hann",
                )
                for samples in noise_samples
            ]
            noise_linear_spectra = np.asarray(
                [
                    10.0 ** (result.spectrum_dbfs_per_hz / 10.0)
                    for result in noise_results
                ]
            )
            noise_rms = mean_std(
                [result.noise_rms_codes for result in noise_results]
            )
            representative_index = int(
                np.argmin(
                    np.abs(
                        np.asarray(
                            [result.noise_rms_codes for result in noise_results]
                        )
                        - noise_rms[0]
                    )
                )
            )
            aggregated[condition]["noise"][channel] = {
                "samples": noise_samples,
                "representative": (
                    noise_samples[representative_index]
                    - np.mean(noise_samples[representative_index])
                ),
                "freqs_hz": noise_results[0].freqs_hz,
                "spectrum_dbfs_hz": 10.0
                * np.log10(np.mean(noise_linear_spectra, axis=0)),
                "rms_codes": noise_rms,
                "rms_uv": mean_std(
                    [
                        result.noise_rms_adc_volts * 1e6
                        for result in noise_results
                    ]
                ),
            }

            dynamic_samples = [
                np.asarray(select_channel(capture, channel), dtype=np.float64)
                for capture in dynamic_captures
            ]
            dynamic_results = [
                compute_metrics(
                    samples,
                    fs_hz=SAMPLE_RATE_HZ,
                    n_bits=14,
                    n_harmonics=5,
                    window="hann",
                )
                for samples in dynamic_samples
            ]
            if any(not result.is_coherent for result in dynamic_results):
                raise ValueError(f"non-coherent capture for {condition}, {channel}")
            if any(
                np.any((samples == 0) | (samples == 16383))
                for samples in dynamic_samples
            ):
                raise ValueError(f"clipped capture for {condition}, {channel}")
            dynamic_linear_spectra = np.asarray(
                [
                    10.0 ** (result.spectrum_dbfs / 10.0)
                    for result in dynamic_results
                ]
            )
            aggregated[condition]["dynamic"][channel] = {
                "samples": dynamic_samples,
                "freqs_hz": dynamic_results[0].freqs_hz,
                "spectrum_dbfs": (
                    10.0 * np.log10(np.mean(dynamic_linear_spectra, axis=0))
                    - SPECTRUM_DBFS_CORRECTION_DB
                ),
                "snr_db": mean_std([result.snr_db for result in dynamic_results]),
                "sinad_db": mean_std(
                    [result.sinad_db for result in dynamic_results]
                ),
                "enob": mean_std([result.enob for result in dynamic_results]),
            }
    return aggregated


def phase_aligned_average(sample_sets: list[np.ndarray], reference: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.float64) - np.mean(reference)
    samples_per_period = int(round(SAMPLE_RATE_HZ / FUNDAMENTAL_HZ))
    aligned = []
    for samples in sample_sets:
        centered = np.asarray(samples, dtype=np.float64) - np.mean(samples)
        scores = [
            float(np.dot(np.roll(centered, -shift), reference))
            for shift in range(samples_per_period)
        ]
        aligned.append(np.roll(centered, -int(np.argmax(scores))))
    return np.mean(aligned, axis=0)


def plot_noise_time_and_spectrum(data: dict) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.0))
    visible_samples = 1600
    time_us = np.arange(visible_samples) / SAMPLE_RATE_HZ * 1e6
    for row, channel in enumerate(CHANNELS):
        time_axis, spectrum_axis = axes[row]
        for condition, style in CONDITIONS.items():
            channel_data = data[condition]["noise"][channel]
            time_axis.plot(
                time_us,
                channel_data["representative"][:visible_samples],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=0.65,
                alpha=0.9,
                label=style["label"],
            )
            spectrum_axis.plot(
                channel_data["freqs_hz"] / 1e6,
                channel_data["spectrum_dbfs_hz"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=0.75,
                label=style["label"],
            )
        time_axis.text(
            0.015,
            0.93,
            CHANNEL_LABELS[channel],
            transform=time_axis.transAxes,
            va="top",
        )
        spectrum_axis.text(
            0.015,
            0.93,
            CHANNEL_LABELS[channel],
            transform=spectrum_axis.transAxes,
            va="top",
        )
        format_thesis_axis(time_axis, "Czas, μs", "Kod ADC względem średniej")
        format_thesis_axis(
            spectrum_axis, "Częstotliwość, MHz", "Poziom, dBFS/Hz"
        )
        spectrum_axis.set_xlim(0.0, 40.0)
        spectrum_axis.set_ylim(-135.0, -95.0)
    figure.tight_layout()
    output = THESIS_GRAPHICS_DIR / "emc_noise_time_spectrum_comparison.png"
    figure.savefig(output, dpi=300, bbox_inches="tight")
    bin_width_db = 10.0 * math.log10(SAMPLE_RATE_HZ / SAMPLE_COUNT)
    for axis in axes[:, 1]:
        for line in axis.lines:
            line.set_ydata(np.asarray(line.get_ydata()) + bin_width_db)
        axis.set_ylabel("Poziom, dBFS")
        axis.set_ylim(-135.0 + bin_width_db, -95.0 + bin_width_db)
    figure.savefig(THESIS_GRAPHICS_DIR / "emc_noise_time_spectrum_comparison_dbfs.png",
                   dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {output}")


def plot_dynamic_time_and_spectrum(data: dict) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.0))
    visible_samples = 81
    time_ns = np.arange(visible_samples) / SAMPLE_RATE_HZ * 1e9
    for row, channel in enumerate(CHANNELS):
        time_axis, spectrum_axis = axes[row]
        reference = data["pc_monitors"]["dynamic"][channel]["samples"][0]
        for condition, style in CONDITIONS.items():
            channel_data = data[condition]["dynamic"][channel]
            waveform = phase_aligned_average(channel_data["samples"], reference)
            time_axis.plot(
                time_ns,
                waveform[:visible_samples],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.0,
                label=style["label"],
            )
            spectrum_axis.plot(
                channel_data["freqs_hz"] / 1e6,
                channel_data["spectrum_dbfs"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=0.75,
                label=style["label"],
            )
        time_axis.text(
            0.015,
            0.93,
            CHANNEL_LABELS[channel],
            transform=time_axis.transAxes,
            va="top",
        )
        spectrum_axis.text(
            0.015,
            0.93,
            CHANNEL_LABELS[channel],
            transform=spectrum_axis.transAxes,
            va="top",
        )
        format_thesis_axis(time_axis, "Czas, ns", "Kod ADC względem średniej")
        format_thesis_axis(spectrum_axis, "Częstotliwość, MHz", "Poziom, dBFS")
        spectrum_axis.set_xlim(0.0, 40.0)
        spectrum_axis.set_ylim(-105.0, 5.0)
    figure.tight_layout()
    output = THESIS_GRAPHICS_DIR / "emc_dynamic_time_spectrum_comparison.png"
    figure.savefig(output, dpi=300, bbox_inches="tight")
    for axis in axes[:, 1]:
        for line in axis.lines:
            line.set_ydata(np.asarray(line.get_ydata()) - 10.0 * math.log10(SAMPLE_RATE_HZ / SAMPLE_COUNT))
        axis.set_ylabel("Poziom, dBFS/Hz")
        axis.set_ylim(-105.0 - 10.0 * math.log10(SAMPLE_RATE_HZ / SAMPLE_COUNT),
                      5.0 - 10.0 * math.log10(SAMPLE_RATE_HZ / SAMPLE_COUNT))
    figure.savefig(THESIS_GRAPHICS_DIR / "emc_dynamic_time_spectrum_comparison_dbfs_hz.png",
                   dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {output}")


def plot_metric_comparison(data: dict) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.2, 7.2))
    metric_specs = (
        ("noise", "rms_codes", "Szum skuteczny, kod"),
        ("dynamic", "snr_db", "SNR, dB"),
        ("dynamic", "sinad_db", "SINAD, dB"),
        ("dynamic", "enob", "ENOB, bit"),
    )
    x = np.arange(len(CHANNELS), dtype=np.float64)
    width = 0.34
    for axis, (measurement, metric, y_label) in zip(axes.flat, metric_specs):
        for index, (condition, style) in enumerate(CONDITIONS.items()):
            values = [
                data[condition][measurement][channel][metric][0]
                for channel in CHANNELS
            ]
            errors = [
                data[condition][measurement][channel][metric][1]
                for channel in CHANNELS
            ]
            axis.bar(
                x + (index - 0.5) * width,
                values,
                width,
                yerr=errors,
                capsize=3,
                color=style["color"],
                alpha=0.85,
                label=style["label"],
            )
        axis.set_xticks(x, [CHANNEL_LABELS[channel] for channel in CHANNELS])
        format_thesis_axis(axis, "Kanał", y_label)
        if metric == "enob":
            axis.yaxis.set_major_formatter(
                FuncFormatter(lambda value, _: f"{value:g}".replace(".", ","))
            )
    figure.tight_layout()
    output = THESIS_GRAPHICS_DIR / "emc_metrics_comparison.png"
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {output}")


def save_summary(data: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for condition in CONDITIONS:
        for channel in CHANNELS:
            noise = data[condition]["noise"][channel]
            dynamic = data[condition]["dynamic"][channel]
            rows.append(
                {
                    "condition": condition,
                    "channel": channel,
                    "noise_rms_codes_mean": noise["rms_codes"][0],
                    "noise_rms_codes_std": noise["rms_codes"][1],
                    "noise_rms_uv_mean": noise["rms_uv"][0],
                    "noise_rms_uv_std": noise["rms_uv"][1],
                    "snr_db_mean": dynamic["snr_db"][0],
                    "snr_db_std": dynamic["snr_db"][1],
                    "sinad_db_mean": dynamic["sinad_db"][0],
                    "sinad_db_std": dynamic["sinad_db"][1],
                    "enob_mean": dynamic["enob"][0],
                    "enob_std": dynamic["enob"][1],
                }
            )
    output = RESULTS_DIR / "emc_power_comparison_summary.csv"
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {output}")

    for channel in CHANNELS:
        pc_noise = data["pc_monitors"]["noise"][channel]["rms_codes"][0]
        laptop_noise = data["laptop_battery"]["noise"][channel]["rms_codes"][0]
        pc_dynamic = data["pc_monitors"]["dynamic"][channel]
        laptop_dynamic = data["laptop_battery"]["dynamic"][channel]
        print(
            f"{channel}: noise change {100.0 * (laptop_noise / pc_noise - 1.0):+.2f}%, "
            f"SNR {laptop_dynamic['snr_db'][0] - pc_dynamic['snr_db'][0]:+.3f} dB, "
            f"SINAD {laptop_dynamic['sinad_db'][0] - pc_dynamic['sinad_db'][0]:+.3f} dB, "
            f"ENOB {laptop_dynamic['enob'][0] - pc_dynamic['enob'][0]:+.4f} bit"
        )


def main() -> None:
    THESIS_GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
    data = aggregate_data()
    plot_noise_time_and_spectrum(data)
    plot_dynamic_time_and_spectrum(data)
    plot_metric_comparison(data)
    save_summary(data)


if __name__ == "__main__":
    main()
