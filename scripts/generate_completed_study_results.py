#!/usr/bin/env python3
"""Aggregate completed oscilloscope measurements for the thesis."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.capture_io import load_capture_csv
from analysis.metrics import (
    average_centered_channels,
    compute_metrics,
    compute_noise_metrics,
)
from analysis.plot_style import format_thesis_axis


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
CAPTURES_DIR = PROJECT_ROOT / "scripts" / "results" / "captures"
RESULTS_DIR = PROJECT_ROOT / "scripts" / "results" / "completed_studies"
THESIS_GRAPHICS_DIR = REPOSITORY_ROOT / "mastersThesis" / "masters_thesis" / "graf"


@dataclass(frozen=True)
class OperatingPoint:
    key: str
    attenuation: str
    profile: str
    gain_token: str
    gain_pct: float

    @property
    def axis_label(self) -> str:
        return f"{self.key}\n$\\sim${self.attenuation}\n{self.gain_pct:.2f}%"


@dataclass(frozen=True)
class FrequencyPoint:
    stem: str
    input_frequency_hz: float


POINTS = (
    OperatingPoint("P1", "1:1", "20mVdiv", "49p50", 49.50),
    OperatingPoint("P2", "1:1", "50mVdiv", "45p44", 45.44),
    OperatingPoint("P3", "1:100", "0p5Vdiv", "55p50", 55.50),
    OperatingPoint("P4", "1:100", "1Vdiv", "52p53", 52.53),
    OperatingPoint("P5", "1:100", "2Vdiv", "49p66", 49.66),
    OperatingPoint("P6", "1:100", "5Vdiv", "45p00", 45.00),
)

FREQUENCY_POINTS = (
    FrequencyPoint("996kHz", 996_093.75),
    FrequencyPoint("2p002MHz", 2_001_953.125),
    FrequencyPoint("5MHz", 5_000_000.0),
    FrequencyPoint("10MHz", 10_000_000.0),
    FrequencyPoint("14p502MHz", 14_501_953.125),
    FrequencyPoint("15MHz", 15_000_000.0),
    FrequencyPoint("18p496MHz", 18_496_093.75),
    FrequencyPoint("20MHz", 20_000_000.0),
    FrequencyPoint("30MHz", 30_000_000.0),
    FrequencyPoint("35MHz", 35_000_000.0),
)

CHANNELS = ("ch1", "ch2", "average")
CHANNEL_LABELS = {"ch1": "kanał 1", "ch2": "kanał 2", "average": "średnia kanałów"}
COLORS = {"ch1": "#0068B5", "ch2": "#D1495B", "average": "#2A9D55"}
RANGE_STYLES = {1: "--", 2: "-"}


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)), float(np.std(array, ddof=1))


def load_series(pattern: str) -> list[tuple[np.ndarray, np.ndarray, object]]:
    files = sorted(CAPTURES_DIR.glob(pattern))
    if len(files) != 10:
        raise ValueError(f"expected 10 files for {pattern}, found {len(files)}")
    loaded = [load_capture_csv(path) for path in files]
    for ch1, ch2, meta in loaded:
        if len(ch1) != 8192 or len(ch2) != 8192:
            raise ValueError(f"invalid capture depth in {pattern}")
        if not math.isclose(meta.fs_hz, 80_000_000.0):
            raise ValueError(f"invalid sample rate in {pattern}")
    return loaded


def select_samples(ch1: np.ndarray, ch2: np.ndarray, channel: str) -> np.ndarray:
    if channel == "ch1":
        return ch1
    if channel == "ch2":
        return ch2
    return average_centered_channels(ch1, ch2)


def aggregate_noise() -> tuple[list[dict[str, float | int | str]], dict]:
    rows: list[dict[str, float | int | str]] = []
    spectra: dict = {}
    for adc_range in (1, 2):
        captures = load_series(f"noise_{adc_range}vpp_gnd_*.csv")
        spectra[adc_range] = {}
        for channel_index, channel in enumerate(("ch1", "ch2")):
            results = []
            for ch1, ch2, meta in captures:
                samples = (ch1, ch2)[channel_index]
                results.append(
                    compute_noise_metrics(
                        samples,
                        fs_hz=meta.fs_hz,
                        adc_range_vpp=float(adc_range),
                        window="hann",
                    )
                )
            rms_codes = mean_std([result.noise_rms_codes for result in results])
            rms_uv = mean_std(
                [result.noise_rms_adc_volts * 1e6 for result in results]
            )
            density = mean_std(
                [result.noise_density_dbfs_per_hz for result in results]
            )
            floor_bin = mean_std(
                [result.noise_floor_dbfs_per_bin for result in results]
            )
            rows.append(
                {
                    "adc_range_vpp": adc_range,
                    "channel": channel,
                    "noise_rms_codes_mean": rms_codes[0],
                    "noise_rms_codes_std": rms_codes[1],
                    "noise_rms_adc_uv_mean": rms_uv[0],
                    "noise_rms_adc_uv_std": rms_uv[1],
                    "noise_density_dbfs_hz_mean": density[0],
                    "noise_density_dbfs_hz_std": density[1],
                    "noise_floor_dbfs_bin_mean": floor_bin[0],
                    "noise_floor_dbfs_bin_std": floor_bin[1],
                }
            )
            linear_spectra = [
                10.0 ** (result.spectrum_dbfs_per_hz / 10.0) for result in results
            ]
            spectra[adc_range][channel] = {
                "freqs_hz": results[0].freqs_hz,
                "spectrum_dbfs_hz": 10.0
                * np.log10(np.mean(linear_spectra, axis=0)),
            }
    return rows, spectra


def aggregate_dynamic() -> list[dict]:
    rows: list[dict] = []
    for adc_range in (1, 2):
        for point in POINTS:
            pattern = (
                f"harmonics_{adc_range}vpp_{point.attenuation.replace(':', 'to')}_"
                f"profile{point.profile}_g{point.gain_token}_*.csv"
            )
            captures = load_series(pattern)
            for channel in CHANNELS:
                results = []
                peak_to_peak_codes = []
                for ch1, ch2, meta in captures:
                    samples = select_samples(ch1, ch2, channel)
                    results.append(
                        compute_metrics(
                            samples,
                            fs_hz=meta.fs_hz,
                            n_harmonics=5,
                            window="hann",
                        )
                    )
                    peak_to_peak_codes.append(float(np.ptp(samples)))
                if any(not result.is_coherent for result in results):
                    raise ValueError(f"non-coherent capture in {pattern}, {channel}")
                metrics = {
                    "snr_db": mean_std([result.snr_db for result in results]),
                    "sinad_db": mean_std([result.sinad_db for result in results]),
                    "enob": mean_std([result.enob for result in results]),
                    "noise_floor_dbfs": mean_std(
                        [result.noise_floor_dbfs for result in results]
                    ),
                    "peak_to_peak_codes": mean_std(peak_to_peak_codes),
                }
                harmonics = []
                for index in range(4):
                    harmonics.append(
                        mean_std(
                            [result.harmonic_levels_dbc[index] for result in results]
                        )
                    )
                rows.append(
                    {
                        "adc_range_vpp": adc_range,
                        "point": point.key,
                        "attenuation": point.attenuation,
                        "gain_pct": point.gain_pct,
                        "channel": channel,
                        **{
                            f"{name}_{suffix}": value[index]
                            for name, value in metrics.items()
                            for index, suffix in enumerate(("mean", "std"))
                        },
                        **{
                            f"h{order}_dbc_{suffix}": harmonics[order - 2][index]
                            for order in range(2, 6)
                            for index, suffix in enumerate(("mean", "std"))
                        },
                    }
                )
    return rows


def tone_peak_to_peak_codes(
    samples: np.ndarray, fs_hz: float, tone_frequency_hz: float
) -> float:
    samples = np.asarray(samples, dtype=np.float64)
    sample_count = len(samples)
    window = np.hanning(sample_count)
    coherent_gain = float(np.mean(window))
    centered = samples - np.mean(samples)
    spectrum = np.fft.rfft(centered * window)
    amplitude = np.abs(spectrum) / (sample_count * coherent_gain) * 2.0
    amplitude[0] /= 2.0
    if sample_count % 2 == 0:
        amplitude[-1] /= 2.0
    tone_bin = int(round(tone_frequency_hz * sample_count / fs_hz))
    return float(2.0 * amplitude[tone_bin])


def interpolate_bandwidth(rows: list[dict], response_key: str) -> float:
    baseband = [row for row in rows if row["input_frequency_hz"] < 40_000_000.0]
    for lower, upper in zip(baseband, baseband[1:]):
        lower_level = float(lower[response_key])
        upper_level = float(upper[response_key])
        if lower_level > -3.0 and upper_level <= -3.0:
            fraction = (-3.0 - lower_level) / (upper_level - lower_level)
            return float(
                lower["input_frequency_hz"]
                + fraction
                * (upper["input_frequency_hz"] - lower["input_frequency_hz"])
            )
    raise ValueError(f"no -3 dB crossing found for {response_key}")


def aggregate_frequency() -> tuple[list[dict], dict[str, float]]:
    rows: list[dict] = []
    for point in FREQUENCY_POINTS:
        pattern = f"freq_2vpp_1to100_{point.stem}_[0-9][0-9].csv"
        captures = load_series(pattern)
        amplitudes: dict[str, list[float]] = {"ch1": [], "ch2": []}
        for ch1, ch2, meta in captures:
            fields = meta.fields
            if fields.get("measurement_type") != "frequency":
                raise ValueError(f"invalid measurement type in {pattern}")
            if not math.isclose(
                float(fields["generator_frequency_hz"]),
                point.input_frequency_hz,
                abs_tol=0.01,
            ):
                raise ValueError(f"invalid generator frequency in {pattern}")
            if not math.isclose(
                float(fields["generator_amplitude_vpk"]), 2.8, abs_tol=1e-9
            ):
                raise ValueError(f"invalid generator amplitude in {pattern}")
            for channel, samples in (("ch1", ch1), ("ch2", ch2)):
                amplitudes[channel].append(
                    tone_peak_to_peak_codes(
                        samples, meta.fs_hz, point.input_frequency_hz
                    )
                )
        ch1_mean, ch1_std = mean_std(amplitudes["ch1"])
        ch2_mean, ch2_std = mean_std(amplitudes["ch2"])
        rows.append(
            {
                "input_frequency_hz": point.input_frequency_hz,
                "ch1_amplitude_pp_codes_mean": ch1_mean,
                "ch1_amplitude_pp_codes_std": ch1_std,
                "ch2_amplitude_pp_codes_mean": ch2_mean,
                "ch2_amplitude_pp_codes_std": ch2_std,
            }
        )

    reference = rows[0]
    for row in rows:
        for channel in ("ch1", "ch2"):
            mean = float(row[f"{channel}_amplitude_pp_codes_mean"])
            std = float(row[f"{channel}_amplitude_pp_codes_std"])
            reference_mean = float(
                reference[f"{channel}_amplitude_pp_codes_mean"]
            )
            row[f"{channel}_response_db"] = 20.0 * math.log10(
                mean / reference_mean
            )
            row[f"{channel}_response_std_db"] = (
                20.0 / math.log(10.0) * std / mean
            )

    bandwidths = {
        "ch1_hz": interpolate_bandwidth(rows, "ch1_response_db"),
        "ch2_hz": interpolate_bandwidth(rows, "ch2_response_db"),
    }
    return rows, bandwidths


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_noise(spectra: dict) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.0), sharex=True, sharey=True)
    for axis, channel in zip(axes, ("ch1", "ch2")):
        for adc_range, color in ((1, "#D1495B"), (2, "#0068B5")):
            data = spectra[adc_range][channel]
            axis.plot(
                data["freqs_hz"] / 1e6,
                data["spectrum_dbfs_hz"],
                linewidth=0.75,
                color=color,
                label=f"{adc_range} Vpp",
            )
        format_thesis_axis(axis, "Częstotliwość (MHz)", "Poziom (dBFS/Hz)")
        axis.text(0.015, 0.92, CHANNEL_LABELS[channel], transform=axis.transAxes)
        axis.set_xlim(0.0, 40.0)
        axis.set_ylim(-150.0, -75.0)
    fig.tight_layout()
    fig.savefig(THESIS_GRAPHICS_DIR / "noise_adc_range_comparison.png", dpi=220)
    plt.close(fig)


def row_lookup(rows: list[dict], adc_range: int, point: str, channel: str) -> dict:
    return next(
        row
        for row in rows
        if row["adc_range_vpp"] == adc_range
        and row["point"] == point
        and row["channel"] == channel
    )


def plot_harmonics(rows: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), sharex=True, sharey=True)
    x = np.arange(len(POINTS))
    harmonic_colors = ("#0068B5", "#D1495B", "#2A9D55", "#8C5FB3")
    for row_index, channel in enumerate(("ch1", "ch2")):
        for column_index, adc_range in enumerate((1, 2)):
            axis = axes[row_index, column_index]
            for order, color in zip(range(2, 6), harmonic_colors):
                values = [
                    row_lookup(rows, adc_range, point.key, channel)[
                        f"h{order}_dbc_mean"
                    ]
                    for point in POINTS
                ]
                axis.plot(x, values, marker="o", color=color, label=f"H{order}")
            format_thesis_axis(axis, "Punkt pracy", "Poziom (dBc)")
            axis.text(
                0.015,
                0.92,
                f"{CHANNEL_LABELS[channel]}, {adc_range} Vpp",
                transform=axis.transAxes,
            )
            axis.set_xticks(x, [point.axis_label for point in POINTS])
            axis.set_ylim(-70.0, -40.0)
    fig.tight_layout()
    fig.savefig(THESIS_GRAPHICS_DIR / "harmonics_adc_range_comparison.png", dpi=220)
    plt.close(fig)


def plot_dynamic_metrics(rows: list[dict]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 9.0), sharex=True)
    x = np.arange(len(POINTS))
    for axis, (metric, label) in zip(
        axes,
        (("snr_db", "SNR (dB)"), ("sinad_db", "SINAD (dB)"), ("enob", "ENOB (bit)")),
    ):
        for adc_range in (1, 2):
            for channel in ("ch1", "ch2"):
                values = [
                    row_lookup(rows, adc_range, point.key, channel)[f"{metric}_mean"]
                    for point in POINTS
                ]
                axis.plot(
                    x,
                    values,
                    marker="o",
                    color=COLORS[channel],
                    linestyle=RANGE_STYLES[adc_range],
                    label=f"{CHANNEL_LABELS[channel]}, {adc_range} Vpp",
                )
        format_thesis_axis(axis, "Punkt pracy", label)
    axes[-1].set_xticks(x, [point.axis_label for point in POINTS])
    fig.tight_layout()
    fig.savefig(THESIS_GRAPHICS_DIR / "dynamic_metrics_adc_range.png", dpi=220)
    plt.close(fig)


def plot_channel_averaging(rows: list[dict]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 9.0), sharex=True)
    x = np.arange(len(POINTS))
    for axis, (metric, label) in zip(
        axes,
        (("snr_db", "SNR (dB)"), ("sinad_db", "SINAD (dB)"), ("enob", "ENOB (bit)")),
    ):
        for channel in CHANNELS:
            values = [
                row_lookup(rows, 2, point.key, channel)[f"{metric}_mean"]
                for point in POINTS
            ]
            axis.plot(
                x,
                values,
                marker="o",
                color=COLORS[channel],
                label=CHANNEL_LABELS[channel],
            )
        format_thesis_axis(axis, "Punkt pracy", label)
    axes[-1].set_xticks(x, [point.axis_label for point in POINTS])
    fig.tight_layout()
    fig.savefig(THESIS_GRAPHICS_DIR / "channel_averaging_metrics.png", dpi=220)
    plt.close(fig)


def plot_frequency_response(rows: list[dict], bandwidths: dict[str, float]) -> None:
    fig, axis = plt.subplots(figsize=(9.5, 5.8))
    for channel in ("ch1", "ch2"):
        color = COLORS[channel]
        axis.plot(
            [row["input_frequency_hz"] / 1e6 for row in rows],
            [row[f"{channel}_response_db"] for row in rows],
            marker="o",
            color=color,
            linewidth=1.4,
            label=CHANNEL_LABELS[channel],
        )
        axis.plot(
            bandwidths[f"{channel}_hz"] / 1e6,
            -3.0,
            marker="x",
            markersize=8,
            markeredgewidth=1.6,
            color=color,
        )
    axis.axhline(-3.0, color="#555555", linestyle=":", linewidth=1.0)
    axis.set_xlim(0.0, 37.0)
    axis.set_ylim(-32.0, 2.0)
    format_thesis_axis(
        axis,
        "Częstotliwość sygnału wejściowego (MHz)",
        "Względna amplituda (dB)",
    )
    fig.tight_layout()
    fig.savefig(THESIS_GRAPHICS_DIR / "frequency_response.png", dpi=220)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    THESIS_GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
    noise_rows, noise_spectra = aggregate_noise()
    dynamic_rows = aggregate_dynamic()
    frequency_rows, bandwidths = aggregate_frequency()

    save_csv(RESULTS_DIR / "noise_summary.csv", noise_rows)
    save_csv(RESULTS_DIR / "dynamic_summary.csv", dynamic_rows)
    save_csv(RESULTS_DIR / "frequency_response_summary.csv", frequency_rows)
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(
            {
                "operating_points": [asdict(point) for point in POINTS],
                "noise": noise_rows,
                "dynamic": dynamic_rows,
                "frequency_response": frequency_rows,
                "bandwidths": bandwidths,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    plot_noise(noise_spectra)
    plot_harmonics(dynamic_rows)
    plot_dynamic_metrics(dynamic_rows)
    plot_channel_averaging(dynamic_rows)
    plot_frequency_response(frequency_rows, bandwidths)
    print(f"Saved reports to {RESULTS_DIR}")
    print(f"Saved thesis figures to {THESIS_GRAPHICS_DIR}")


if __name__ == "__main__":
    main()
