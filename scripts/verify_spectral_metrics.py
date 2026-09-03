# ruff: noqa: I001
"""Verify SNR, SINAD and ENOB calculations with a synthetic capture.

The generated waveform contains a coherent fundamental tone, its second
harmonic and white noise of known RMS value. The script asserts that the
spectral analysis recovers the expected metrics and saves a verification plot.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from analysis.metrics import SpectralMetrics, compute_metrics


SAMPLE_COUNT = 8192
SAMPLE_RATE_HZ = 80_000_000.0
FUNDAMENTAL_BIN = 73
FUNDAMENTAL_AMPLITUDE = 4000.0
HARMONIC_AMPLITUDE = 400.0
NOISE_RMS = 20.0
TOLERANCE_DB = 0.25


def expected_metrics() -> tuple[float, float, float]:
    """Return analytical values for the generated coherent waveform."""
    signal_power = FUNDAMENTAL_AMPLITUDE**2
    noise_power = 2.0 * NOISE_RMS**2
    harmonic_power = HARMONIC_AMPLITUDE**2
    snr_db = 10.0 * np.log10(signal_power / noise_power)
    sinad_db = 10.0 * np.log10(signal_power / (noise_power + harmonic_power))
    return snr_db, sinad_db, (sinad_db - 1.76) / 6.02


def synthetic_capture(bin_index: float = FUNDAMENTAL_BIN) -> np.ndarray:
    """Generate a deterministic capture with a coherent or offset tone."""
    indices = np.arange(SAMPLE_COUNT)
    phase = 2.0 * np.pi * bin_index * indices / SAMPLE_COUNT
    rng = np.random.default_rng(20260816)
    noise = rng.normal(size=SAMPLE_COUNT)
    noise *= NOISE_RMS / np.sqrt(np.mean(noise**2))
    return (
        FUNDAMENTAL_AMPLITUDE * np.sin(phase)
        + HARMONIC_AMPLITUDE * np.sin(2.0 * phase)
        + noise
    )


def verify() -> tuple[np.ndarray, SpectralMetrics, SpectralMetrics, tuple[float, float, float]]:
    samples = synthetic_capture()
    result = compute_metrics(
        samples,
        fs_hz=SAMPLE_RATE_HZ,
        n_harmonics=5,
        window="hann",
        leakage_bins=3,
    )
    expected = expected_metrics()
    measured = (result.snr_db, result.sinad_db, result.enob)
    names = ("SNR", "SINAD", "ENOB")
    tolerances = (TOLERANCE_DB, TOLERANCE_DB, TOLERANCE_DB / 6.02)
    for name, actual, target, tolerance in zip(names, measured, expected, tolerances):
        if abs(actual - target) > tolerance:
            raise AssertionError(
                f"{name}: expected {target:.3f}, got {actual:.3f}, "
                f"tolerance {tolerance:.3f}"
            )
    offset_result = compute_metrics(
        synthetic_capture(FUNDAMENTAL_BIN + 0.5),
        fs_hz=SAMPLE_RATE_HZ,
        n_harmonics=5,
        window="hann",
        leakage_bins=3,
    )
    return samples, result, offset_result, expected


def save_plot(
    output: pathlib.Path,
    samples: np.ndarray,
    result: SpectralMetrics,
    offset_result: SpectralMetrics,
    expected: tuple[float, float, float],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, (time_axis, coherent_axis, offset_axis) = plt.subplots(3, 1, figsize=(9, 9))

    shown_samples = 5 * SAMPLE_COUNT // FUNDAMENTAL_BIN
    time_axis.plot(
        np.arange(shown_samples) / SAMPLE_RATE_HZ * 1e6, samples[:shown_samples]
    )
    time_axis.set_xlabel("Czas, µs")
    time_axis.set_ylabel("Kod ADC")
    time_axis.set_title("Sygnał syntetyczny")
    time_axis.grid()

    coherent_axis.plot(result.freqs_hz / 1e6, result.spectrum_dbfs)
    coherent_axis.set_xlim(
        result.freqs_hz[0] / 1e6, 3 * result.fundamental_freq_hz / 1e6
    )
    coherent_axis.set_ylim(-110, 0)
    coherent_axis.set_xlabel("Częstotliwość, MHz")
    coherent_axis.set_ylabel("Amplituda, dBFS")
    coherent_axis.set_title(f"Trafienie w bin: ton przy binie {FUNDAMENTAL_BIN}")
    coherent_axis.grid()

    offset_axis.plot(offset_result.freqs_hz / 1e6, offset_result.spectrum_dbfs)
    offset_axis.set_xlim(offset_result.freqs_hz[0] / 1e6, 3 * result.fundamental_freq_hz / 1e6)
    offset_axis.set_ylim(-110, 0)
    offset_axis.set_xlabel("Częstotliwość, MHz")
    offset_axis.set_ylabel("Amplituda, dBFS")
    offset_axis.set_title(f"Nietrafienie w bin: ton przesunięty do {FUNDAMENTAL_BIN + 0.5:.1f} binu")
    offset_axis.grid()

    figure.tight_layout()
    figure.savefig(output, dpi=150)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent
        / "results"
        / "implementation"
        / "metrics_verification.png",
        help="path of the generated verification plot",
    )
    args = parser.parse_args()

    samples, result, offset_result, expected = verify()
    save_plot(args.output, samples, result, offset_result, expected)
    print(f"SNR: expected {expected[0]:.2f} dB, measured {result.snr_db:.2f} dB")
    print(f"SINAD: expected {expected[1]:.2f} dB, measured {result.sinad_db:.2f} dB")
    print(f"ENOB: expected {expected[2]:.2f}, measured {result.enob:.2f}")
    print(
        "Off-bin: expected bin {:.1f}, detected bin {}, SNR {:.2f} dB, "
        "SINAD {:.2f} dB, ENOB {:.2f}".format(
            FUNDAMENTAL_BIN + 0.5,
            offset_result.fundamental_bin,
            offset_result.snr_db,
            offset_result.sinad_db,
            offset_result.enob,
        )
    )
    print(f"Saved verification plot -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
