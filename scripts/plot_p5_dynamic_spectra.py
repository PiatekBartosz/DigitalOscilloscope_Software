#!/usr/bin/env python3
"""Generate thesis figures showing the P5 dynamic-performance spectra."""

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
REPOSITORY_ROOT = PROJECT_ROOT.parent
CAPTURES_DIR = PROJECT_ROOT / "scripts" / "results" / "captures" / "thesis_measurements"
THESIS_GRAPHICS_DIR = REPOSITORY_ROOT / "mastersThesis" / "masters_thesis" / "graf"
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.capture_io import load_capture_csv  # noqa: E402
from analysis.metrics import compute_metrics  # noqa: E402
from analysis.plot_style import format_thesis_axis  # noqa: E402


CAPTURE_PATTERN = "harmonics_2vpp_1to100_profile2Vdiv_g49p66_*.csv"
EXPECTED_SAMPLE_RATE_HZ = 80_000_000.0
EXPECTED_SAMPLE_COUNT = 8192
BIN_TO_HZ_DB = 10.0 * math.log10(EXPECTED_SAMPLE_RATE_HZ / EXPECTED_SAMPLE_COUNT)
EXPECTED_TONE_HZ = 5_000_000.0
LEAKAGE_BINS = 3
CHANNEL_COLORS = ("#0068B5", "#D1495B")
PEAK_TO_SINE_POWER_DB = 10.0 * math.log10(2.0)


@dataclass(frozen=True)
class AveragedSpectrum:
    channel_label: str
    color: str
    frequencies_hz: np.ndarray
    levels_dbfs: np.ndarray
    fundamental_bin: int
    snr_mean_db: float
    snr_std_db: float
    sinad_mean_db: float
    sinad_std_db: float
    enob_mean: float
    enob_std: float
    noise_floor_mean_dbfs: float


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)), float(np.std(array, ddof=1))


def polish_decimal(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def load_p5_captures() -> list[tuple[np.ndarray, np.ndarray, object]]:
    paths = sorted(CAPTURES_DIR.glob(CAPTURE_PATTERN))
    if len(paths) != 10:
        raise ValueError(
            f"expected 10 P5 captures matching {CAPTURE_PATTERN}, found {len(paths)}"
        )

    captures = [load_capture_csv(path) for path in paths]
    for path, (ch1, ch2, meta) in zip(paths, captures):
        if len(ch1) != EXPECTED_SAMPLE_COUNT or len(ch2) != EXPECTED_SAMPLE_COUNT:
            raise ValueError(f"{path}: expected {EXPECTED_SAMPLE_COUNT} samples")
        if not math.isclose(meta.fs_hz, EXPECTED_SAMPLE_RATE_HZ):
            raise ValueError(f"{path}: unexpected sample rate {meta.fs_hz}")
        if meta.n_bits != 14:
            raise ValueError(f"{path}: unexpected ADC resolution {meta.n_bits}")
        fields = meta.fields
        if fields.get("measurement_type") != "harmonics":
            raise ValueError(f"{path}: not a harmonic measurement")
        if not math.isclose(
            float(fields["generator_frequency_hz"]), EXPECTED_TONE_HZ
        ):
            raise ValueError(f"{path}: unexpected generator frequency")
        for channel in (1, 2):
            if fields.get(f"attenuation_ch{channel}") != "1:100":
                raise ValueError(f"{path}: unexpected channel {channel} attenuation")
            if not math.isclose(
                float(fields[f"gain_pct_ch{channel}"]), 49.66, abs_tol=0.01
            ):
                raise ValueError(f"{path}: unexpected channel {channel} gain")
            if not math.isclose(float(fields[f"sense_ch{channel}_vpp"]), 2.0):
                raise ValueError(f"{path}: unexpected channel {channel} ADC range")
    return captures


def average_spectra(
    captures: list[tuple[np.ndarray, np.ndarray, object]],
) -> list[AveragedSpectrum]:
    averaged: list[AveragedSpectrum] = []
    for channel_index, (channel_label, color) in enumerate(
        zip(("kanał 1", "kanał 2"), CHANNEL_COLORS)
    ):
        results = []
        for ch1, ch2, meta in captures:
            samples = (ch1, ch2)[channel_index]
            result = compute_metrics(
                samples,
                fs_hz=meta.fs_hz,
                n_bits=meta.n_bits,
                n_harmonics=5,
                window="hann",
                leakage_bins=LEAKAGE_BINS,
            )
            if not result.is_coherent:
                raise ValueError(f"non-coherent P5 capture in {channel_label}")
            results.append(result)

        frequency_grids = [result.freqs_hz for result in results]
        if any(
            not np.array_equal(frequency_grids[0], grid)
            for grid in frequency_grids[1:]
        ):
            raise ValueError(f"inconsistent frequency grids in {channel_label}")
        fundamental_bins = {result.fundamental_bin for result in results}
        if len(fundamental_bins) != 1:
            raise ValueError(f"inconsistent fundamental bins in {channel_label}")

        linear_power = np.asarray(
            [10.0 ** (result.spectrum_dbfs / 10.0) for result in results]
        )
        # compute_metrics stores squared peak amplitudes. Subtracting 3.01 dB
        # expresses each sinusoidal component relative to the RMS power of a
        # full-scale sine, for which the displayed level must be 0 dBFS.
        mean_levels_dbfs = (
            10.0 * np.log10(np.mean(linear_power, axis=0)) - PEAK_TO_SINE_POWER_DB
        )
        snr_mean, snr_std = mean_std([result.snr_db for result in results])
        sinad_mean, sinad_std = mean_std([result.sinad_db for result in results])
        enob_mean, enob_std = mean_std([result.enob for result in results])
        averaged.append(
            AveragedSpectrum(
                channel_label=channel_label,
                color=color,
                frequencies_hz=frequency_grids[0],
                levels_dbfs=mean_levels_dbfs,
                fundamental_bin=fundamental_bins.pop(),
                snr_mean_db=snr_mean,
                snr_std_db=snr_std,
                sinad_mean_db=sinad_mean,
                sinad_std_db=sinad_std,
                enob_mean=enob_mean,
                enob_std=enob_std,
                noise_floor_mean_dbfs=float(
                    np.mean([result.noise_floor_dbfs for result in results])
                )
                - PEAK_TO_SINE_POWER_DB,
            )
        )
    return averaged


def level_at_bin(spectrum: AveragedSpectrum, bin_index: int) -> float:
    return float(spectrum.levels_dbfs[bin_index - 1])


def plot_full_spectrum(spectra: list[AveragedSpectrum]) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(9.4, 7.2), sharex=True, sharey=True)
    for axis, spectrum in zip(axes, spectra):
        frequencies_mhz = spectrum.frequencies_hz / 1e6
        axis.plot(
            frequencies_mhz,
            spectrum.levels_dbfs,
            color=spectrum.color,
            linewidth=0.72,
            label="uśrednione widmo mocy",
        )

        fundamental_frequency_mhz = (
            spectrum.fundamental_bin * EXPECTED_SAMPLE_RATE_HZ
            / EXPECTED_SAMPLE_COUNT
            / 1e6
        )
        axis.scatter(
            fundamental_frequency_mhz,
            level_at_bin(spectrum, spectrum.fundamental_bin),
            color="#C62828",
            marker="o",
            s=30,
            zorder=4,
            label="ton podstawowy",
        )
        for order in range(2, 9):
            harmonic_bin = spectrum.fundamental_bin * order
            included_as_distortion = order <= 5
            axis.scatter(
                harmonic_bin * EXPECTED_SAMPLE_RATE_HZ / EXPECTED_SAMPLE_COUNT / 1e6,
                level_at_bin(spectrum, harmonic_bin),
                color="#7B4FA3" if included_as_distortion else "#E07A1F",
                marker="D" if included_as_distortion else "^",
                s=23,
                zorder=4,
                label=(
                    "H2-H5, zniekształcenia"
                    if order == 2
                    else "H6-H8, w puli szumu"
                    if order == 6
                    else None
                ),
            )
            axis.annotate(
                f"H{order}",
                (
                    harmonic_bin
                    * EXPECTED_SAMPLE_RATE_HZ
                    / EXPECTED_SAMPLE_COUNT
                    / 1e6,
                    level_at_bin(spectrum, harmonic_bin),
                ),
                xytext=(-3 if order == 8 else 0, 7),
                textcoords="offset points",
                ha="right" if order == 8 else "center",
                fontsize=8,
            )
        axis.axhline(
            spectrum.noise_floor_mean_dbfs,
            color="#555555",
            linestyle=":",
            linewidth=1.0,
            label="średni poziom szumu na prążek",
        )
        axis.text(
            0.015,
            0.94,
            spectrum.channel_label,
            transform=axis.transAxes,
            va="top",
        )
        axis.text(
            0.985,
            0.94,
            (
                f"SNR = {polish_decimal(spectrum.snr_mean_db, 2)} ± "
                f"{polish_decimal(spectrum.snr_std_db, 2)} dB\n"
                f"SINAD = {polish_decimal(spectrum.sinad_mean_db, 2)} ± "
                f"{polish_decimal(spectrum.sinad_std_db, 2)} dB\n"
                f"ENOB = {polish_decimal(spectrum.enob_mean, 2)} ± "
                f"{polish_decimal(spectrum.enob_std, 2)} bit"
            ),
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9},
        )
        axis.set_xlim(0.0, 40.2)
        axis.set_ylim(-115.0, 5.0)
        format_thesis_axis(axis, "Częstotliwość, MHz", "Poziom, dBFS")

    figure.tight_layout()
    output = THESIS_GRAPHICS_DIR / "dynamic_spectrum_p5_2vpp.png"
    figure.savefig(output, dpi=300, bbox_inches="tight")
    for axis in axes:
        for line in axis.lines:
            line.set_ydata(np.asarray(line.get_ydata(), dtype=float) - BIN_TO_HZ_DB)
        axis.set_ylabel("Poziom, dBFS/Hz")
        axis.set_ylim(-115.0 - BIN_TO_HZ_DB, 5.0 - BIN_TO_HZ_DB)
    figure.savefig(THESIS_GRAPHICS_DIR / "dynamic_spectrum_p5_2vpp_dbfs_hz.png",
                   dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {output}")


def plot_fundamental_zoom(spectra: list[AveragedSpectrum]) -> None:
    # The thesis uses channel 1 as the representative example of the
    # classification procedure described in section 7.1.1.
    figure, axis = plt.subplots(figsize=(9.4, 3.8))
    bin_width_hz = EXPECTED_SAMPLE_RATE_HZ / EXPECTED_SAMPLE_COUNT
    spectrum = spectra[0]
    offsets_khz = (spectrum.frequencies_hz - EXPECTED_TONE_HZ) / 1e3
    visible = np.abs(offsets_khz) <= 120.0
    axis.plot(
        offsets_khz[visible],
        spectrum.levels_dbfs[visible],
        color=spectrum.color,
        linewidth=0.9,
        marker="o",
        markersize=2.8,
        label="uśrednione widmo mocy",
    )
    peak = spectrum.fundamental_bin - 1
    left = peak - 1
    while left > 0 and spectrum.levels_dbfs[left] > spectrum.levels_dbfs[left - 1]:
        left -= 1
    right = peak + 1
    while right < len(spectrum.levels_dbfs) - 1 and spectrum.levels_dbfs[right] > spectrum.levels_dbfs[right + 1]:
        right += 1
    axis.axvspan(
        (spectrum.frequencies_hz[left] - EXPECTED_TONE_HZ) / 1e3,
        (spectrum.frequencies_hz[right] - EXPECTED_TONE_HZ) / 1e3,
        color="#F4A261",
        alpha=0.24,
        label=f"prążki mocy sygnału ({right - left + 1})",
    )
    axis.axvline(0.0, color="#C62828", linestyle="--", linewidth=0.9)
    axis.axhline(
        spectrum.noise_floor_mean_dbfs,
        color="#555555",
        linestyle=":",
        linewidth=1.0,
        label="średni poziom szumu na prążek",
    )
    axis.text(
        0.015,
        0.93,
        spectrum.channel_label,
        transform=axis.transAxes,
        va="top",
    )
    axis.text(
        0.985,
        0.93,
        f"Δf = {polish_decimal(bin_width_hz / 1e3, 6)} kHz",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
    )
    axis.set_xlim(-120.0, 120.0)
    axis.set_ylim(-115.0, 5.0)
    format_thesis_axis(
        axis,
        "Odsunięcie od częstotliwości 5 MHz, kHz",
        "Poziom, dBFS",
    )

    figure.tight_layout()
    output = THESIS_GRAPHICS_DIR / "dynamic_spectrum_p5_fundamental_zoom.png"
    figure.savefig(output, dpi=300, bbox_inches="tight")
    for line in axis.lines:
        line.set_ydata(np.asarray(line.get_ydata(), dtype=float) - BIN_TO_HZ_DB)
    axis.set_ylabel("Poziom, dBFS/Hz")
    axis.set_ylim(-115.0 - BIN_TO_HZ_DB, 5.0 - BIN_TO_HZ_DB)
    figure.savefig(THESIS_GRAPHICS_DIR / "dynamic_spectrum_p5_fundamental_zoom_dbfs_hz.png",
                   dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {output}")


def main() -> None:
    THESIS_GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
    captures = load_p5_captures()
    spectra = average_spectra(captures)
    plot_full_spectrum(spectra)
    plot_fundamental_zoom(spectra)


if __name__ == "__main__":
    main()
