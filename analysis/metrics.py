from __future__ import annotations

import dataclasses

import numpy as np

_WINDOWS = {
    "hann": np.hanning,
    "hamming": np.hamming,
    "blackman": np.blackman,
    "rect": lambda n: np.ones(n),
}


@dataclasses.dataclass
class SpectralMetrics:
    snr_db: float
    thd_db: float
    thd_percent: float
    sinad_db: float
    enob: float
    sfdr_db: float
    noise_floor_dbfs: float
    fundamental_freq_hz: float
    fundamental_bin: int
    n_harmonics_used: int
    is_coherent: bool
    coherence_margin_db: float
    freqs_hz: np.ndarray
    spectrum_dbfs: np.ndarray


def compute_metrics(
    samples: np.ndarray,
    fs_hz: float,
    n_bits: int = 14,
    n_harmonics: int = 5,
    window: str = "hann",
    leakage_bins: int = 3,
) -> SpectralMetrics:
    samples = np.asarray(samples, dtype=np.float64)
    n = len(samples)
    if n < 8:
        raise ValueError(f"need at least 8 samples, got {n}")

    win_fn = _WINDOWS.get(window)
    if win_fn is None:
        raise ValueError(f"unknown window {window!r}; choose one of {list(_WINDOWS)}")
    win = win_fn(n)
    coherent_gain = np.mean(win)

    centered = samples - np.mean(samples)
    windowed = centered * win


    rect_power = np.abs(np.fft.rfft(centered)) ** 2
    rect_peak_bin = int(np.argmax(rect_power[1:])) + 1
    neighbor_power = 0.0
    for nb in (rect_peak_bin - 1, rect_peak_bin + 1):
        if 0 <= nb < len(rect_power):
            neighbor_power += rect_power[nb]
    if neighbor_power <= np.finfo(float).tiny:
        coherence_margin_db = float("inf")
    else:
        coherence_margin_db = 10.0 * np.log10(rect_power[rect_peak_bin] / neighbor_power)
    is_coherent = bool(coherence_margin_db > 40.0)

    spectrum = np.fft.rfft(windowed)
    amp = np.abs(spectrum) / (n * coherent_gain) * 2.0
    amp[0] /= 2.0
    if n % 2 == 0:
        amp[-1] /= 2.0

    power = amp**2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)

    fundamental_bin = int(np.argmax(power[1:])) + 1
    fundamental_freq = float(freqs[fundamental_bin])

    max_nonoverlap_width = max((fundamental_bin - 1) // 2, 0)
    effective_leakage_bins = min(leakage_bins, max_nonoverlap_width)

    signal_mask = np.zeros(len(power), dtype=bool)
    lo = max(fundamental_bin - effective_leakage_bins, 1)
    hi = min(fundamental_bin + effective_leakage_bins + 1, len(power))
    signal_mask[lo:hi] = True
    signal_power = float(np.sum(power[signal_mask]))

    harmonic_mask = np.zeros(len(power), dtype=bool)
    n_harmonics_used = 0
    for k in range(2, n_harmonics + 1):
        h_bin = fundamental_bin * k
        if h_bin >= len(power):
            break
        lo = max(h_bin - effective_leakage_bins, 1)
        hi = min(h_bin + effective_leakage_bins + 1, len(power))
        harmonic_mask[lo:hi] = True
        n_harmonics_used += 1

    harmonic_mask &= ~signal_mask
    harmonic_power_total = float(np.sum(power[harmonic_mask]))

    noise_mask = ~(signal_mask | harmonic_mask)
    noise_mask[0] = False
    noise_power = float(np.sum(power[noise_mask]))
    noise_power = max(noise_power, np.finfo(float).tiny)

    noise_bin_count = max(int(np.sum(noise_mask)), 1)

    snr_db = 10.0 * np.log10(signal_power / noise_power)
    thd_db = 10.0 * np.log10(max(harmonic_power_total, np.finfo(float).tiny) / signal_power)
    thd_percent = 100.0 * np.sqrt(harmonic_power_total / signal_power)
    sinad_db = 10.0 * np.log10(signal_power / (noise_power + harmonic_power_total))
    enob = (sinad_db - 1.76) / 6.02

    remaining_power = power.copy()
    remaining_power[~noise_mask] = 0.0
    spur_bin = int(np.argmax(remaining_power))
    spur_power = float(remaining_power[spur_bin])
    if spur_power > 0:
        sfdr_db = 10.0 * np.log10(signal_power / spur_power)
    else:
        sfdr_db = float("inf")

    full_scale_amplitude = 2.0 ** (n_bits - 1)
    full_scale_power = full_scale_amplitude**2 / 2.0
    noise_floor_dbfs = 10.0 * np.log10(
        (noise_power / noise_bin_count) / full_scale_power
    )

    with np.errstate(divide="ignore"):
        spectrum_dbfs = 10.0 * np.log10(
            np.maximum(power, np.finfo(float).tiny) / full_scale_power
        )

    return SpectralMetrics(
        snr_db=snr_db,
        thd_db=thd_db,
        thd_percent=thd_percent,
        sinad_db=sinad_db,
        enob=enob,
        sfdr_db=sfdr_db,
        noise_floor_dbfs=noise_floor_dbfs,
        fundamental_freq_hz=fundamental_freq,
        fundamental_bin=fundamental_bin,
        n_harmonics_used=n_harmonics_used,
        is_coherent=is_coherent,
        coherence_margin_db=float(coherence_margin_db),
        freqs_hz=freqs,
        spectrum_dbfs=spectrum_dbfs,
    )


def suggest_coherent_frequency(desired_hz: float, fs_hz: float, n: int) -> tuple[float, int]:
    k = max(1, round(desired_hz * n / fs_hz))
    return k * fs_hz / n, k


def format_report(metrics: SpectralMetrics, channel_label: str = "CH1") -> str:
    """Human-readable report matching typical thesis-table formatting."""
    lines = [
        f"--- {channel_label} dynamic performance ---",
        f"Fundamental        : {metrics.fundamental_freq_hz:,.2f} Hz (bin {metrics.fundamental_bin})",
        f"SNR                 : {metrics.snr_db:.2f} dB",
        f"THD                 : {metrics.thd_db:.2f} dB  ({metrics.thd_percent:.4f} %)",
        f"SINAD               : {metrics.sinad_db:.2f} dB",
        f"ENOB                : {metrics.enob:.2f} bits",
        f"SFDR                : {metrics.sfdr_db:.2f} dB",
        f"Noise floor         : {metrics.noise_floor_dbfs:.2f} dBFS/bin",
        f"Harmonics used      : {metrics.n_harmonics_used}",
        f"Coherent sampling   : {'yes' if metrics.is_coherent else 'NO'} "
        f"(peak/neighbor-bin margin {metrics.coherence_margin_db:.1f} dB)",
    ]
    if not metrics.is_coherent:
        lines.append(
            "WARNING: capture does not look coherent"
        )
    return "\n".join(lines)
