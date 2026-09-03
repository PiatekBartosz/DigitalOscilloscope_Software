#!/usr/bin/env python3
"""Compare the bypassed-VGA result with prior channel 1 SINAD extremes."""

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
CAPTURES_ROOT = PROJECT_ROOT / "scripts" / "results" / "captures"
THESIS_CAPTURES = CAPTURES_ROOT / "thesis_measurements"
RESULTS_DIR = PROJECT_ROOT / "scripts" / "results" / "implementation"
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.capture_io import load_capture_csv
from analysis.metrics import compute_metrics
from analysis.plot_style import format_thesis_axis

SAMPLE_RATE_HZ = 80_000_000.0
SAMPLE_COUNT = 8192
TONE_HZ = 5_000_000.0
CAPTURE_COUNT = 10
LEAKAGE_BINS = 3
PEAK_TO_SINE_POWER_DB = 10.0 * math.log10(2.0)
BYPASS_SERIES = "vga_bypass"
PRIOR_SERIES = (
    ("P1", "harmonics_2vpp_1to1_profile20mVdiv_g49p50"),
    ("P2", "harmonics_2vpp_1to1_profile50mVdiv_g45p44"),
    ("P3", "harmonics_2vpp_1to100_profile0p5Vdiv_g55p50"),
    ("P4", "harmonics_2vpp_1to100_profile1Vdiv_g52p53"),
    ("P5", "harmonics_2vpp_1to100_profile2Vdiv_g49p66"),
    ("P6", "harmonics_2vpp_1to100_profile5Vdiv_g45p00"),
)


@dataclass(frozen=True)
class SeriesResult:
    role: str
    point: str
    series_id: str
    color: str
    attenuation: str
    gain_pct: float
    generator_amplitude_vpk: float
    adc_range_vpp: float
    frequencies_hz: np.ndarray
    spectrum_dbfs: np.ndarray
    snr_db: tuple[float, float]
    sinad_db: tuple[float, float]
    enob: tuple[float, float]
    thd_db: tuple[float, float]
    sfdr_db: tuple[float, float]
    noise_floor_dbfs: tuple[float, float]
    harmonics_dbc: tuple[tuple[float, float], ...]
    peak_to_peak_codes: tuple[float, float]
    minimum_code: int
    maximum_code: int
    noncoherent_captures: int


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)), float(np.std(array, ddof=1))


def load_series(series_id: str, directory: Path) -> list[tuple[np.ndarray, object]]:
    paths = sorted(directory.glob(f"{series_id}_[0-9][0-9].csv"))
    if len(paths) != CAPTURE_COUNT:
        raise ValueError(
            f"{series_id}: expected {CAPTURE_COUNT} files, found {len(paths)}"
        )

    captures = []
    for path in paths:
        ch1, _, metadata = load_capture_csv(path)
        if len(ch1) != SAMPLE_COUNT:
            raise ValueError(f"{path}: unexpected sample count")
        if not math.isclose(metadata.fs_hz, SAMPLE_RATE_HZ):
            raise ValueError(f"{path}: unexpected sample rate")
        if metadata.fields.get("series_id") != series_id:
            raise ValueError(f"{path}: unexpected series identifier")
        if not math.isclose(float(metadata.fields["sense_ch1_vpp"]), 2.0):
            raise ValueError(f"{path}: expected the 2 Vpp ADC range")
        if not math.isclose(float(metadata.fields["generator_frequency_hz"]), TONE_HZ):
            raise ValueError(f"{path}: unexpected generator frequency")
        captures.append((ch1, metadata))
    return captures


def calculate_sfdr(result) -> float:
    spectrum = np.asarray(result.spectrum_dbfs, dtype=np.float64)
    fundamental_index = result.fundamental_bin - 1
    spurs = spectrum.copy()
    lo = max(fundamental_index - LEAKAGE_BINS, 0)
    hi = min(fundamental_index + LEAKAGE_BINS + 1, len(spurs))
    spurs[lo:hi] = -np.inf
    return float(spectrum[fundamental_index] - np.max(spurs))


def analyse_series(
    role: str, point: str, series_id: str, directory: Path, color: str
) -> SeriesResult:
    captures = load_series(series_id, directory)
    results = []
    thd_values = []
    sfdr_values = []
    peak_to_peak_codes = []
    minimum_codes = []
    maximum_codes = []
    for samples, metadata in captures:
        result = compute_metrics(
            samples,
            fs_hz=metadata.fs_hz,
            n_bits=metadata.n_bits,
            n_harmonics=5,
            window="hann",
            leakage_bins=LEAKAGE_BINS,
        )
        results.append(result)
        harmonic_ratio = sum(
            10.0 ** (level / 10.0) for level in result.harmonic_levels_dbc
        )
        thd_values.append(10.0 * math.log10(harmonic_ratio))
        sfdr_values.append(calculate_sfdr(result))
        peak_to_peak_codes.append(float(np.ptp(samples)))
        minimum_codes.append(int(np.min(samples)))
        maximum_codes.append(int(np.max(samples)))

    frequency_grids = [result.freqs_hz for result in results]
    if any(
        not np.array_equal(frequency_grids[0], grid) for grid in frequency_grids[1:]
    ):
        raise ValueError(f"{series_id}: inconsistent frequency grids")
    linear_spectra = np.asarray(
        [10.0 ** (result.spectrum_dbfs / 10.0) for result in results]
    )
    metadata = captures[0][1].fields
    return SeriesResult(
        role=role,
        point=point,
        series_id=series_id,
        color=color,
        attenuation=metadata["attenuation_ch1"],
        gain_pct=float(metadata["gain_pct_ch1"]),
        generator_amplitude_vpk=float(metadata["generator_amplitude_vpk"]),
        adc_range_vpp=float(metadata["sense_ch1_vpp"]),
        frequencies_hz=frequency_grids[0],
        spectrum_dbfs=(
            10.0 * np.log10(np.mean(linear_spectra, axis=0)) - PEAK_TO_SINE_POWER_DB
        ),
        snr_db=mean_std([result.snr_db for result in results]),
        sinad_db=mean_std([result.sinad_db for result in results]),
        enob=mean_std([result.enob for result in results]),
        thd_db=mean_std(thd_values),
        sfdr_db=mean_std(sfdr_values),
        noise_floor_dbfs=mean_std(
            [result.noise_floor_dbfs - PEAK_TO_SINE_POWER_DB for result in results]
        ),
        harmonics_dbc=tuple(
            mean_std([result.harmonic_levels_dbc[index] for result in results])
            for index in range(4)
        ),
        peak_to_peak_codes=mean_std(peak_to_peak_codes),
        minimum_code=min(minimum_codes),
        maximum_code=max(maximum_codes),
        noncoherent_captures=sum(not result.is_coherent for result in results),
    )


def select_prior_extremes() -> tuple[SeriesResult, SeriesResult]:
    candidates = [
        analyse_series("kandydat", point, series_id, THESIS_CAPTURES, "#777777")
        for point, series_id in PRIOR_SERIES
    ]
    best = max(candidates, key=lambda result: result.sinad_db[0])
    worst = min(candidates, key=lambda result: result.sinad_db[0])
    return (
        SeriesResult(
            **{**best.__dict__, "role": "najlepszy wcześniejszy", "color": "#2A9D55"}
        ),
        SeriesResult(
            **{**worst.__dict__, "role": "najgorszy wcześniejszy", "color": "#D1495B"}
        ),
    )


def polish(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def plot_results(results: list[SeriesResult]) -> Path:
    figure, axes = plt.subplots(3, 1, figsize=(9.4, 10.4), sharex=True, sharey=True)
    subplot_labels = ("a", "b", "c")
    for axis, label, result in zip(axes, subplot_labels, results):
        axis.plot(
            result.frequencies_hz / 1e6,
            result.spectrum_dbfs,
            color=result.color,
            linewidth=0.72,
        )
        configuration = (
            f"({label}) {result.role}, {result.point}\n"
            f"tłumienie {result.attenuation}, "
            f"G_VGA = {polish(2.0 * (result.gain_pct - 44.0))} dB, "
            f"Ugen = {polish(result.generator_amplitude_vpk)} Vpk"
        )
        axis.text(0.015, 0.94, configuration, transform=axis.transAxes, va="top")
        metrics = (
            f"SNR = {polish(result.snr_db[0])} ± {polish(result.snr_db[1])} dB\n"
            f"SINAD = {polish(result.sinad_db[0])} ± {polish(result.sinad_db[1])} dB\n"
            f"ENOB = {polish(result.enob[0])} ± {polish(result.enob[1])} bit"
        )
        axis.text(
            0.985,
            0.94,
            metrics,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8.2,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9},
        )
        axis.set_xlim(0.0, 40.0)
        axis.set_ylim(-115.0, 5.0)
        format_thesis_axis(axis, "Częstotliwość, MHz", "Poziom, dBFS", legend=False)

    figure.tight_layout()
    output = RESULTS_DIR / "vga_bypass_vs_ch1_extremes_spectra.png"
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output


def save_summary(results: list[SeriesResult]) -> Path:
    output = RESULTS_DIR / "vga_bypass_vs_ch1_extremes_metrics.csv"
    fieldnames = [
        "role",
        "point",
        "series_id",
        "attenuation",
        "gain_pct",
        "generator_amplitude_vpk",
        "adc_range_vpp",
        "capture_count",
        "snr_db_mean",
        "snr_db_std",
        "sinad_db_mean",
        "sinad_db_std",
        "enob_mean",
        "enob_std",
        "thd_h2_h5_db_mean",
        "thd_h2_h5_db_std",
        "sfdr_db_mean",
        "sfdr_db_std",
        "noise_floor_dbfs_per_bin_mean",
        "noise_floor_dbfs_per_bin_std",
        "h2_dbc_mean",
        "h2_dbc_std",
        "h3_dbc_mean",
        "h3_dbc_std",
        "h4_dbc_mean",
        "h4_dbc_std",
        "h5_dbc_mean",
        "h5_dbc_std",
        "peak_to_peak_codes_mean",
        "peak_to_peak_codes_std",
        "minimum_code",
        "maximum_code",
        "noncoherent_captures",
    ]
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {
                "role": result.role,
                "point": result.point,
                "series_id": result.series_id,
                "attenuation": result.attenuation,
                "gain_pct": f"{result.gain_pct:.4f}",
                "generator_amplitude_vpk": f"{result.generator_amplitude_vpk:.6f}",
                "adc_range_vpp": f"{result.adc_range_vpp:.1f}",
                "capture_count": CAPTURE_COUNT,
                "snr_db_mean": f"{result.snr_db[0]:.6f}",
                "snr_db_std": f"{result.snr_db[1]:.6f}",
                "sinad_db_mean": f"{result.sinad_db[0]:.6f}",
                "sinad_db_std": f"{result.sinad_db[1]:.6f}",
                "enob_mean": f"{result.enob[0]:.6f}",
                "enob_std": f"{result.enob[1]:.6f}",
                "thd_h2_h5_db_mean": f"{result.thd_db[0]:.6f}",
                "thd_h2_h5_db_std": f"{result.thd_db[1]:.6f}",
                "sfdr_db_mean": f"{result.sfdr_db[0]:.6f}",
                "sfdr_db_std": f"{result.sfdr_db[1]:.6f}",
                "noise_floor_dbfs_per_bin_mean": f"{result.noise_floor_dbfs[0]:.6f}",
                "noise_floor_dbfs_per_bin_std": f"{result.noise_floor_dbfs[1]:.6f}",
                "peak_to_peak_codes_mean": f"{result.peak_to_peak_codes[0]:.6f}",
                "peak_to_peak_codes_std": f"{result.peak_to_peak_codes[1]:.6f}",
                "minimum_code": result.minimum_code,
                "maximum_code": result.maximum_code,
                "noncoherent_captures": result.noncoherent_captures,
            }
            for order, harmonic in enumerate(result.harmonics_dbc, start=2):
                row[f"h{order}_dbc_mean"] = f"{harmonic[0]:.6f}"
                row[f"h{order}_dbc_std"] = f"{harmonic[1]:.6f}"
            writer.writerow(row)
    return output


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    bypass = analyse_series(
        "VGA pominięty",
        "obejście",
        BYPASS_SERIES,
        CAPTURES_ROOT,
        "#0068B5",
    )
    best, worst = select_prior_extremes()
    results = [bypass, best, worst]
    plot_path = plot_results(results)
    summary_path = save_summary(results)
    print(f"Saved: {plot_path}")
    print(f"Saved: {summary_path}")
    for result in results:
        print(
            f"{result.role} ({result.point}): SNR {result.snr_db[0]:.3f} dB, "
            f"SINAD {result.sinad_db[0]:.3f} dB, ENOB {result.enob[0]:.3f} bit, "
            f"THD {result.thd_db[0]:.3f} dB, SFDR {result.sfdr_db[0]:.3f} dB"
        )


if __name__ == "__main__":
    main()
