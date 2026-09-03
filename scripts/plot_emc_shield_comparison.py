#!/usr/bin/env python3
"""Compare battery-powered measurements before and after shielding channel 2."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_emc_power_comparison as base


CONDITIONS = {
    "laptop_battery": {
        "label": "bez ekranu",
        "color": "#6A3D9A",
        "linestyle": "-",
    },
    "laptop_battery_shield_ch2": {
        "label": "ekran kanału 2",
        "color": "#E67E22",
        "linestyle": "--",
    },
}
OUTPUT_PREFIX = "emc_shield_ch2"


def load_data() -> dict:
    base.CONDITIONS = CONDITIONS
    return base.aggregate_data()


def plot_time_and_spectrum(data: dict, measurement: str) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.0))
    if measurement == "noise":
        visible_samples = 1600
        time = np.arange(visible_samples) / base.SAMPLE_RATE_HZ * 1e6
        x_label = "Czas, μs"
        y_label = "Kod ADC względem średniej"
        spectrum_key = "spectrum_dbfs_hz"
        spectrum_label = "Poziom, dBFS"
        conversion = 10.0 * math.log10(base.SAMPLE_RATE_HZ / base.SAMPLE_COUNT)
        spectrum_ylim = (-135.0 + conversion, -95.0 + conversion)
    else:
        visible_samples = 81
        time = np.arange(visible_samples) / base.SAMPLE_RATE_HZ * 1e9
        x_label = "Czas, ns"
        y_label = "Kod ADC względem średniej"
        spectrum_key = "spectrum_dbfs"
        spectrum_label = "Poziom, dBFS"
        spectrum_ylim = (-105.0, 5.0)

    for row, channel in enumerate(base.CHANNELS):
        time_axis, spectrum_axis = axes[row]
        reference = data["laptop_battery"]["dynamic"][channel]["samples"][0]
        for condition, style in CONDITIONS.items():
            channel_data = data[condition][measurement][channel]
            if measurement == "noise":
                waveform = channel_data["representative"]
            else:
                waveform = base.phase_aligned_average(
                    channel_data["samples"], reference
                )
            time_axis.plot(
                time,
                waveform[:visible_samples],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=0.75 if measurement == "noise" else 1.0,
                label=style["label"],
            )
            spectrum_axis.plot(
                channel_data["freqs_hz"] / 1e6,
                channel_data[spectrum_key],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=0.75,
                label=style["label"],
            )
        for axis in (time_axis, spectrum_axis):
            axis.text(
                0.015,
                0.93,
                base.CHANNEL_LABELS[channel],
                transform=axis.transAxes,
                va="top",
            )
        base.format_thesis_axis(time_axis, x_label, y_label)
        base.format_thesis_axis(
            spectrum_axis, "Częstotliwość, MHz", spectrum_label
        )
        spectrum_axis.set_xlim(0.0, 40.0)
        spectrum_axis.set_ylim(*spectrum_ylim)

        if measurement == "noise":
            for line in spectrum_axis.lines:
                line.set_ydata(np.asarray(line.get_ydata(), dtype=float) + conversion)

    figure.tight_layout()
    output = base.THESIS_GRAPHICS_DIR / f"{OUTPUT_PREFIX}_{measurement}_comparison.png"
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {output}")


def plot_metrics(data: dict) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.2, 7.2))
    specs = (
        ("noise", "rms_codes", "Szum skuteczny, kod"),
        ("dynamic", "snr_db", "SNR, dB"),
        ("dynamic", "sinad_db", "SINAD, dB"),
        ("dynamic", "enob", "ENOB, bit"),
    )
    x = np.arange(len(base.CHANNELS), dtype=float)
    width = 0.34
    for axis, (measurement, metric, label) in zip(axes.flat, specs):
        for index, (condition, style) in enumerate(CONDITIONS.items()):
            values = [
                data[condition][measurement][channel][metric][0]
                for channel in base.CHANNELS
            ]
            errors = [
                data[condition][measurement][channel][metric][1]
                for channel in base.CHANNELS
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
        axis.set_xticks(x, [base.CHANNEL_LABELS[c] for c in base.CHANNELS])
        base.format_thesis_axis(axis, "Kanał", label)
    figure.tight_layout()
    output = base.THESIS_GRAPHICS_DIR / f"{OUTPUT_PREFIX}_metrics_comparison.png"
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {output}")


def save_summary(data: dict) -> None:
    rows = []
    for condition in CONDITIONS:
        for channel in base.CHANNELS:
            noise = data[condition]["noise"][channel]
            dynamic = data[condition]["dynamic"][channel]
            rows.append({
                "condition": condition,
                "channel": channel,
                "noise_rms_codes_mean": noise["rms_codes"][0],
                "noise_rms_codes_std": noise["rms_codes"][1],
                "snr_db_mean": dynamic["snr_db"][0],
                "snr_db_std": dynamic["snr_db"][1],
                "sinad_db_mean": dynamic["sinad_db"][0],
                "sinad_db_std": dynamic["sinad_db"][1],
                "enob_mean": dynamic["enob"][0],
                "enob_std": dynamic["enob"][1],
            })
    output = base.RESULTS_DIR / f"{OUTPUT_PREFIX}_summary.csv"
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {output}")

    for channel in base.CHANNELS:
        before = data["laptop_battery"]
        after = data["laptop_battery_shield_ch2"]
        noise_change = 100.0 * (
            after["noise"][channel]["rms_codes"][0]
            / before["noise"][channel]["rms_codes"][0]
            - 1.0
        )
        print(
            f"{channel}: noise {noise_change:+.2f}%, "
            f"SNR {after['dynamic'][channel]['snr_db'][0] - before['dynamic'][channel]['snr_db'][0]:+.3f} dB, "
            f"SINAD {after['dynamic'][channel]['sinad_db'][0] - before['dynamic'][channel]['sinad_db'][0]:+.3f} dB, "
            f"ENOB {after['dynamic'][channel]['enob'][0] - before['dynamic'][channel]['enob'][0]:+.4f} bit"
        )


def main() -> None:
    data = load_data()
    plot_time_and_spectrum(data, "noise")
    plot_time_and_spectrum(data, "dynamic")
    plot_metrics(data)
    save_summary(data)


if __name__ == "__main__":
    main()
