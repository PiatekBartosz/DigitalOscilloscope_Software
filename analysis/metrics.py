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
    sinad_db: float
    enob: float
    noise_floor_dbfs: float
    fundamental_freq_hz: float
    fundamental_bin: int
    n_harmonics_used: int
    harmonic_levels_dbc: tuple[float, ...]
    is_coherent: bool
    coherence_margin_db: float
    freqs_hz: np.ndarray
    spectrum_dbfs: np.ndarray


@dataclasses.dataclass
class NoiseMetrics:
    dc_code: float
    noise_rms_codes: float
    noise_rms_adc_volts: float
    noise_floor_dbfs_per_bin: float
    noise_density_dbfs_per_hz: float
    equivalent_noise_bandwidth_hz: float
    freqs_hz: np.ndarray
    spectrum_dbfs_per_hz: np.ndarray


def average_centered_channels(ch1: np.ndarray, ch2: np.ndarray) -> np.ndarray:
    """Average simultaneous channels after removing their individual means."""
    ch1 = np.asarray(ch1, dtype=np.float64)
    ch2 = np.asarray(ch2, dtype=np.float64)
    if ch1.shape != ch2.shape or ch1.ndim != 1 or len(ch1) == 0:
        raise ValueError("channels must be equal, non-empty one-dimensional arrays")
    return ((ch1 - np.mean(ch1)) + (ch2 - np.mean(ch2))) / 2.0


def compute_noise_metrics(
    samples: np.ndarray,
    fs_hz: float,
    adc_range_vpp: float,
    n_bits: int = 14,
    window: str = "hann",
) -> NoiseMetrics:
    """Measure zero-input noise without treating a random bin as a signal tone."""
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 1 or len(samples) < 8:
        raise ValueError(f"need at least 8 one-dimensional samples, got {samples.shape}")
    if fs_hz <= 0.0:
        raise ValueError(f"fs_hz must be positive, got {fs_hz}")
    if adc_range_vpp <= 0.0:
        raise ValueError(f"adc_range_vpp must be positive, got {adc_range_vpp}")
    if n_bits <= 0:
        raise ValueError(f"n_bits must be positive, got {n_bits}")

    win_fn = _WINDOWS.get(window)
    if win_fn is None:
        raise ValueError(f"unknown window {window!r}; choose one of {list(_WINDOWS)}")

    dc_code = float(np.mean(samples))
    centered = samples - dc_code
    noise_rms_codes = float(np.sqrt(np.mean(centered**2)))
    noise_rms_adc_volts = noise_rms_codes * float(adc_range_vpp) / (2.0**n_bits)

    win = win_fn(len(centered))
    window_power = float(np.sum(win**2))
    window_sum = float(np.sum(win))
    if window_power <= 0.0 or window_sum == 0.0:
        raise ValueError(f"window {window!r} has invalid normalization")

    spectrum = np.fft.rfft(centered * win)
    power_density = np.abs(spectrum) ** 2 / (fs_hz * window_power)
    if len(centered) % 2 == 0:
        power_density[1:-1] *= 2.0
    else:
        power_density[1:] *= 2.0

    all_freqs = np.fft.rfftfreq(len(centered), d=1.0 / fs_hz)
    bin_width_hz = fs_hz / len(centered)
    equivalent_noise_bandwidth_hz = fs_hz * window_power / (window_sum**2)

    # Bin zero is the DC component and is deliberately excluded from every
    # reported noise quantity and from the spectrum returned for plotting.
    power_density = power_density[1:]
    freqs = all_freqs[1:]
    mean_noise_density = float(np.mean(power_density))
    mean_noise_density = max(mean_noise_density, np.finfo(float).tiny)

    full_scale_amplitude = 2.0 ** (n_bits - 1)
    full_scale_power = full_scale_amplitude**2 / 2.0
    noise_density_dbfs_per_hz = 10.0 * np.log10(
        mean_noise_density / full_scale_power
    )
    noise_floor_dbfs_per_bin = 10.0 * np.log10(
        mean_noise_density * bin_width_hz / full_scale_power
    )
    with np.errstate(divide="ignore"):
        spectrum_dbfs_per_hz = 10.0 * np.log10(
            np.maximum(power_density, np.finfo(float).tiny) / full_scale_power
        )

    return NoiseMetrics(
        dc_code=dc_code,
        noise_rms_codes=noise_rms_codes,
        noise_rms_adc_volts=noise_rms_adc_volts,
        noise_floor_dbfs_per_bin=float(noise_floor_dbfs_per_bin),
        noise_density_dbfs_per_hz=float(noise_density_dbfs_per_hz),
        equivalent_noise_bandwidth_hz=float(equivalent_noise_bandwidth_hz),
        freqs_hz=freqs,
        spectrum_dbfs_per_hz=spectrum_dbfs_per_hz,
    )


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
        coherence_margin_db = 10.0 * np.log10(
            rect_power[rect_peak_bin] / neighbor_power
        )
    is_coherent = bool(coherence_margin_db > 40.0)

    spectrum = np.fft.rfft(windowed)
    # A one-sided amplitude spectrum needs a factor of two because the
    # negative-frequency half is omitted. Dividing by coherent_gain applies
    # the second factor of about two for a Hann window (coherent gain ~= 0.5).
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
    harmonic_powers: list[float] = []
    n_harmonics_used = 0
    for k in range(2, n_harmonics + 1):
        h_bin = fundamental_bin * k
        if h_bin >= len(power):
            break
        lo = max(h_bin - effective_leakage_bins, 1)
        hi = min(h_bin + effective_leakage_bins + 1, len(power))
        current_harmonic = np.zeros(len(power), dtype=bool)
        current_harmonic[lo:hi] = True
        current_harmonic &= ~signal_mask
        harmonic_mask |= current_harmonic
        harmonic_powers.append(float(np.sum(power[current_harmonic])))
        n_harmonics_used += 1

    harmonic_power_total = float(np.sum(power[harmonic_mask]))
    harmonic_levels_dbc = tuple(
        float(
            10.0
            * np.log10(
                max(harmonic_power, np.finfo(float).tiny) / signal_power
            )
        )
        for harmonic_power in harmonic_powers
    )

    noise_mask = ~(signal_mask | harmonic_mask)
    noise_mask[0] = False
    noise_power = float(np.sum(power[noise_mask]))
    noise_power = max(noise_power, np.finfo(float).tiny)

    noise_bin_count = max(int(np.sum(noise_mask)), 1)

    snr_db = 10.0 * np.log10(signal_power / noise_power)
    sinad_db = 10.0 * np.log10(signal_power / (noise_power + harmonic_power_total))
    enob = (sinad_db - 1.76) / 6.02

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
        sinad_db=sinad_db,
        enob=enob,
        noise_floor_dbfs=noise_floor_dbfs,
        fundamental_freq_hz=fundamental_freq,
        fundamental_bin=fundamental_bin,
        n_harmonics_used=n_harmonics_used,
        harmonic_levels_dbc=harmonic_levels_dbc,
        is_coherent=is_coherent,
        coherence_margin_db=float(coherence_margin_db),
        freqs_hz=freqs[1:],
        spectrum_dbfs=spectrum_dbfs[1:],
    )


def suggest_coherent_frequency(
    desired_hz: float, fs_hz: float, n: int
) -> tuple[float, int]:
    k = max(1, round(desired_hz * n / fs_hz))
    return k * fs_hz / n, k


def format_report(metrics: SpectralMetrics, channel_label: str = "CH1") -> str:
    """Human-readable report matching typical thesis-table formatting."""
    lines = [
        f"--- {channel_label} dynamic performance ---",
        f"Fundamental        : {metrics.fundamental_freq_hz:,.2f} Hz (bin {metrics.fundamental_bin})",
        f"SNR                 : {metrics.snr_db:.2f} dB",
        f"SINAD               : {metrics.sinad_db:.2f} dB",
        f"ENOB                : {metrics.enob:.2f} bits",
        f"Noise floor         : {metrics.noise_floor_dbfs:.2f} dBFS/bin",
        *(
            f"H{order}                  : {level:.2f} dBc"
            for order, level in enumerate(metrics.harmonic_levels_dbc, start=2)
        ),
        f"Harmonics used      : {metrics.n_harmonics_used}",
        f"Coherent sampling   : {'yes' if metrics.is_coherent else 'NO'} "
        f"(peak/neighbor-bin margin {metrics.coherence_margin_db:.1f} dB)",
    ]
    if not metrics.is_coherent:
        lines.append("WARNING: capture does not look coherent")
    return "\n".join(lines)


def format_noise_report(
    metrics: NoiseMetrics, channel_label: str, adc_range_vpp: float
) -> str:
    return "\n".join(
        [
            f"--- {channel_label} zero-input noise ---",
            f"ADC range           : {adc_range_vpp:g} Vpp differential",
            f"DC code              : {metrics.dc_code:.3f}",
            f"Noise RMS            : {metrics.noise_rms_codes:.4f} codes",
            f"ADC-input noise RMS  : {metrics.noise_rms_adc_volts * 1e6:.4f} uV",
            f"Noise floor          : {metrics.noise_floor_dbfs_per_bin:.2f} dBFS/bin",
            f"Noise density        : {metrics.noise_density_dbfs_per_hz:.2f} dBFS/Hz",
            f"Hann ENBW            : {metrics.equivalent_noise_bandwidth_hz:.3f} Hz",
            "DC bin              : excluded",
        ]
    )
