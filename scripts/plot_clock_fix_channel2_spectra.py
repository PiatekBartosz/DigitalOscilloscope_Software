#!/usr/bin/env python3
"""Compare channel 2 spectra measured before and after the clock fix."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAPTURE_DIR = PROJECT_ROOT / "scripts/results/captures/test_clock_fix"
OUTPUT_PATH = CAPTURE_DIR / "channel_2_spectra_before_after_clock_fix.png"
HARMONICS_OVERLAY_PATH = CAPTURE_DIR / "channel_2_harmonics_overlay.png"
HARMONICS_SUBPLOTS_PATH = CAPTURE_DIR / "channel_2_harmonics_subplots.png"
NOISE_SUBPLOTS_PATH = CAPTURE_DIR / "channel_2_noise_subplots.png"
SINGLE_HARMONICS_PATH = CAPTURE_DIR / "channel_2_harmonics_single_capture.png"
SINGLE_NOISE_PATH = CAPTURE_DIR / "channel_2_noise_single_capture.png"
SINGLE_COMBINED_PATH = CAPTURE_DIR / "channel_2_spectra_single_capture.png"
SAMPLE_RATE_HZ = 80_000_000.0
SAMPLE_COUNT = 8192
CAPTURE_COUNT = 10
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.capture_io import load_capture_csv  # noqa: E402
from analysis.metrics import compute_metrics, compute_noise_metrics  # noqa: E402
from analysis.plot_style import format_thesis_axis  # noqa: E402


CONDITIONS = {
    "przed zmianą": {
        "color": "#D1495B",
        "harmonics": "harmonics_5MHz_before_clock_fix_*.csv",
        "noise": "noise_1_1_gain_50_before_clock_fix_*.csv",
    },
    "po zmianie": {
        "color": "#0068B5",
        "harmonics": "harmonics_5MHz_after_clock_fix_broken_channel_1_*.csv",
        "noise": "../noise_1_1_gain_50_after_clock_fix_broken_channel_1_*.csv",
    },
}


def averaged_spectrum(pattern: str, measurement: str) -> tuple[np.ndarray, np.ndarray]:
    paths = sorted(CAPTURE_DIR.glob(pattern))
    if len(paths) != 10:
        raise ValueError(f"expected 10 captures matching {pattern}, found {len(paths)}")

    frequencies = None
    linear_powers = []
    for path in paths:
        _ch1, ch2, metadata = load_capture_csv(path)
        if measurement == "harmonics":
            result = compute_metrics(
                ch2,
                fs_hz=metadata.fs_hz,
                n_bits=metadata.n_bits,
                n_harmonics=5,
                window="hann",
                leakage_bins=3,
            )
            current_frequencies = result.freqs_hz
            # compute_metrics reports squared peak amplitudes relative to
            # full-scale sine RMS power. Convert a full-scale sine to 0 dBFS.
            levels_db = result.spectrum_dbfs - 10.0 * np.log10(2.0)
        else:
            adc_range_vpp = float(metadata.fields["sense_ch2_vpp"])
            result = compute_noise_metrics(
                ch2,
                fs_hz=metadata.fs_hz,
                adc_range_vpp=adc_range_vpp,
                n_bits=metadata.n_bits,
                window="hann",
            )
            current_frequencies = result.freqs_hz
            bin_width_hz = metadata.fs_hz / len(ch2)
            levels_db = result.spectrum_dbfs_per_hz + 10.0 * np.log10(bin_width_hz)

        if frequencies is None:
            frequencies = current_frequencies
        elif not np.array_equal(frequencies, current_frequencies):
            raise ValueError(f"inconsistent frequency grid in {path}")
        linear_powers.append(10.0 ** (levels_db / 10.0))

    assert frequencies is not None
    averaged_db = 10.0 * np.log10(np.mean(np.asarray(linear_powers), axis=0))
    return frequencies, averaged_db


def single_spectrum(pattern: str, measurement: str) -> tuple[np.ndarray, np.ndarray]:
    paths = sorted(CAPTURE_DIR.glob(pattern))
    if not paths:
        raise ValueError(f"no captures matching {pattern}")
    _ch1, ch2, metadata = load_capture_csv(paths[0])
    if measurement == "harmonics":
        result = compute_metrics(
            ch2, metadata.fs_hz, metadata.n_bits, 5, "hann", leakage_bins=3
        )
        return result.freqs_hz, result.spectrum_dbfs - 10.0 * np.log10(2.0)
    result = compute_noise_metrics(
        ch2,
        metadata.fs_hz,
        float(metadata.fields["sense_ch2_vpp"]),
        metadata.n_bits,
        "hann",
    )
    bin_width_hz = metadata.fs_hz / len(ch2)
    return result.freqs_hz, result.spectrum_dbfs_per_hz + 10.0 * np.log10(bin_width_hz)


def plot_single_capture_versions() -> None:
    data = {
        measurement: {
            condition: single_spectrum(settings[measurement], measurement)
            for condition, settings in CONDITIONS.items()
        }
        for measurement in ("harmonics", "noise")
    }
    specifications = (
        ("harmonics", "Sygnał sinusoidalny 5 MHz", SINGLE_HARMONICS_PATH),
        ("noise", "Wejście zwarte do masy", SINGLE_NOISE_PATH),
    )
    for measurement, title, output in specifications:
        figure, axis = plt.subplots(figsize=(10.0, 5.2))
        for condition, settings in CONDITIONS.items():
            frequencies, levels = data[measurement][condition]
            axis.plot(
                frequencies / 1e6,
                levels,
                color=settings["color"],
                linewidth=0.75,
                label=f"{condition}, rejestracja 01",
            )
        axis.text(0.015, 0.94, title, transform=axis.transAxes, va="top")
        format_thesis_axis(axis, "Częstotliwość, MHz", "Poziom, dBFS")
        axis.set_xlim(0.0, 40.0)
        figure.tight_layout()
        figure.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(10.0, 7.2), sharex=True)
    for axis, (measurement, title, _output) in zip(axes, specifications):
        for condition, settings in CONDITIONS.items():
            frequencies, levels = data[measurement][condition]
            axis.plot(
                frequencies / 1e6,
                levels,
                color=settings["color"],
                linewidth=0.75,
                label=f"{condition}, rejestracja 01",
            )
        axis.text(0.015, 0.94, title, transform=axis.transAxes, va="top")
        format_thesis_axis(axis, "Częstotliwość, MHz", "Poziom, dBFS")
        axis.set_xlim(0.0, 40.0)
    figure.tight_layout()
    figure.savefig(SINGLE_COMBINED_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


def harmonic_quality(pattern: str) -> str:
    results = []
    for path in sorted(CAPTURE_DIR.glob(pattern)):
        _ch1, ch2, metadata = load_capture_csv(path)
        results.append(
            compute_metrics(
                ch2, metadata.fs_hz, metadata.n_bits, 5, "hann", leakage_bins=3
            )
        )
    if len(results) != CAPTURE_COUNT:
        raise ValueError(f"expected {CAPTURE_COUNT} captures matching {pattern}")

    snr = float(np.mean([result.snr_db for result in results]))
    sinad = float(np.mean([result.sinad_db for result in results]))
    enob = float(np.mean([result.enob for result in results]))
    return (
        f"SNR: {snr:.2f} dB\n"
        f"SINAD: {sinad:.2f} dB\n"
        f"ENOB: {enob:.2f} bit"
    )


def plot_harmonic_versions() -> None:
    spectra = {
        condition: averaged_spectrum(settings["harmonics"], "harmonics")
        for condition, settings in CONDITIONS.items()
    }
    quality = {
        condition: harmonic_quality(settings["harmonics"])
        for condition, settings in CONDITIONS.items()
    }
    parameters = (
        "Kanał oscyloskopu: 2\n"
        "Sprzężenie: DC\nDzielnik: 1:100\nWzmocnienie: 49,50%\n"
        "Zakres ADC: 2 Vpp\nGenerator: Hantek HDG6202, kanał 2\n"
        "Sinus: 5 MHz, 12,5 Vpk, offset 0 V\nObciążenie: 50 Ω\n"
        "Połączenie: rozdzielacz\n\nFFT\nPróbkowanie: 80 MS/s\n"
        "Liczba próbek: 8192\nRozdzielczość: 9,766 kHz\nOkno: Hann\n"
        "Uśrednianie mocy: 10 rejestracji\n"
        "Przetwornik: 14 bitów, Offset Binary\nDecymacja: 1\n\n"
        "PRZED ZMIANĄ\n" + quality["przed zmianą"] + "\n\n"
        "PO ZMIANIE\n" + quality["po zmianie"]
    )

    figure, axis = plt.subplots(figsize=(13.2, 5.2))
    for condition, settings in CONDITIONS.items():
        frequencies, levels = spectra[condition]
        axis.plot(
            frequencies / 1e6,
            levels,
            color=settings["color"],
            linewidth=0.75,
            label=condition,
        )
    format_thesis_axis(axis, "Częstotliwość, MHz", "Poziom, dBFS")
    axis.set_xlim(0.0, 40.0)
    axis.text(
        1.015, 0.98, parameters, transform=axis.transAxes, va="top", fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "alpha": 0.9},
    )
    figure.subplots_adjust(left=0.08, right=0.73, bottom=0.13, top=0.97)
    figure.savefig(HARMONICS_OVERLAY_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)

    all_levels = np.concatenate([levels for _frequencies, levels in spectra.values()])
    shared_limits = (float(np.min(all_levels)) - 1.0, float(np.max(all_levels)) + 1.0)
    figure, axes = plt.subplots(2, 1, figsize=(13.2, 7.2), sharex=True, sharey=True)
    for axis, (condition, settings) in zip(axes, CONDITIONS.items()):
        frequencies, levels = spectra[condition]
        axis.plot(
            frequencies / 1e6, levels, color=settings["color"], linewidth=0.75,
            label=condition,
        )
        format_thesis_axis(axis, "Częstotliwość, MHz", "Poziom, dBFS")
        axis.set_xlim(0.0, 40.0)
        axis.set_ylim(*shared_limits)
        axis.text(0.015, 0.93, condition, transform=axis.transAxes, va="top")
        axis.text(
            0.985, 0.73, quality[condition], transform=axis.transAxes,
            va="top", ha="right", fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.86},
        )
    axes[0].text(
        1.015, 0.98, parameters.split("\n\nPRZED ZMIANĄ")[0],
        transform=axes[0].transAxes, va="top", fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "alpha": 0.9},
    )
    figure.subplots_adjust(left=0.08, right=0.73, bottom=0.09, top=0.98, hspace=0.25)
    figure.savefig(HARMONICS_SUBPLOTS_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_noise_subplots() -> None:
    spectra = {
        condition: averaged_spectrum(settings["noise"], "noise")
        for condition, settings in CONDITIONS.items()
    }
    parameters = (
        "Kanał oscyloskopu: 2\nSprzężenie: DC\nDzielnik: 1:1\n"
        "Wzmocnienie: 49,91%\nZakres ADC: 2 Vpp\nWejście: zwarte do masy\n"
        "Generator, kanał 2: wyłączony\n\nFFT\nPróbkowanie: 80 MS/s\n"
        "Liczba próbek: 8192\nRozdzielczość: 9,766 kHz\nOkno: Hann\n"
        "Uśrednianie mocy: 10 rejestracji\n"
        "Przetwornik: 14 bitów, Offset Binary\nDecymacja: 1"
    )
    all_levels = np.concatenate([levels for _frequencies, levels in spectra.values()])
    shared_limits = (float(np.min(all_levels)) - 1.0, float(np.max(all_levels)) + 1.0)
    figure, axes = plt.subplots(2, 1, figsize=(13.2, 7.2), sharex=True, sharey=True)
    for axis, (condition, settings) in zip(axes, CONDITIONS.items()):
        frequencies, levels = spectra[condition]
        axis.plot(
            frequencies / 1e6,
            levels,
            color=settings["color"],
            linewidth=0.75,
            label=condition,
        )
        format_thesis_axis(axis, "Częstotliwość, MHz", "Poziom, dBFS")
        axis.set_xlim(0.0, 40.0)
        axis.set_ylim(*shared_limits)
        axis.text(0.015, 0.93, condition, transform=axis.transAxes, va="top")
    axes[0].text(
        1.015, 0.98, parameters, transform=axes[0].transAxes, va="top", fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "alpha": 0.9},
    )
    figure.subplots_adjust(left=0.08, right=0.73, bottom=0.09, top=0.98, hspace=0.25)
    figure.savefig(NOISE_SUBPLOTS_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)

def main() -> None:
    plot_harmonic_versions()
    plot_noise_subplots()
    plot_single_capture_versions()
    quality = {
        condition: harmonic_quality(settings["harmonics"])
        for condition, settings in CONDITIONS.items()
    }
    figure, axes = plt.subplots(2, 1, figsize=(13.2, 7.2), sharex=True)
    panels = (
        (
            "harmonics",
            "Sygnał sinusoidalny 5 MHz",
            "Poziom, dBFS",
            "Kanał: 2\n"
            "Sprzężenie: DC\n"
            "Dzielnik: 1:100\n"
        "Wzmocnienie: 49,50%\n"
            "Zakres ADC: 2 Vpp\n"
            "Generator: Hantek HDG6202\n"
            "Kanał generatora: 2\n"
            "Przebieg: sinusoidalny\n"
            "Częstotliwość: 5 MHz\n"
            "Amplituda: 12,5 Vpk\n"
            "Offset: 0 V\n"
            "Obciążenie: 50 Ω\n"
            "Połączenie: rozdzielacz",
        ),
        (
            "noise",
            "Wejście zwarte do masy",
            "Poziom, dBFS",
            "Kanał: 2\n"
            "Sprzężenie: DC\n"
            "Dzielnik: 1:1\n"
        "Wzmocnienie: 49,91%\n"
            "Zakres ADC: 2 Vpp\n"
            "Wejście: zwarte do masy\n"
            "Generator, kanał 2: wyłączony\n"
            "Obciążenie generatora: 50 Ω",
        ),
    )

    for axis, (measurement, panel_label, y_label, measurement_parameters) in zip(
        axes, panels
    ):
        for condition, settings in CONDITIONS.items():
            frequencies, levels = averaged_spectrum(
                settings[measurement], measurement
            )
            axis.plot(
                frequencies / 1e6,
                levels,
                color=settings["color"],
                linewidth=0.75,
                label=condition,
            )
        axis.text(0.015, 0.94, panel_label, transform=axis.transAxes, va="top")
        format_thesis_axis(axis, "Częstotliwość, MHz", y_label)
        axis.set_xlim(0.0, 40.0)
        if measurement == "harmonics":
            axis.text(
                0.985,
                0.72,
                "PRZED ZMIANĄ\n" + quality["przed zmianą"] + "\n\n"
                "PO ZMIANIE\n" + quality["po zmianie"],
                transform=axis.transAxes,
                va="top",
                ha="right",
                fontsize=8.0,
                bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.86},
            )
        fft_parameters = (
            f"\n\nFFT\n"
            f"Próbkowanie: {SAMPLE_RATE_HZ / 1e6:.0f} MS/s\n"
            f"Liczba próbek: {SAMPLE_COUNT}\n"
            f"Rozdzielczość: {SAMPLE_RATE_HZ / SAMPLE_COUNT / 1e3:.3f} kHz\n"
            "Okno: Hann\n"
            f"Uśrednianie: moc, {CAPTURE_COUNT} rejestracji\n"
            "Przetwornik: 14 bitów\n"
            "Kod: Offset Binary\n"
            "Decymacja: 1"
        )
        axis.text(
            1.015,
            0.98,
            measurement_parameters + fft_parameters,
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            linespacing=1.2,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "alpha": 0.9},
        )

    figure.subplots_adjust(left=0.08, right=0.73, bottom=0.09, top=0.98, hspace=0.25)
    figure.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
