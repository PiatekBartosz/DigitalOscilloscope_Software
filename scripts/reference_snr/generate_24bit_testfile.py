# Reference implementation from Hans Rosenberg, “Are you measuring the right noise?”
# https://www.youtube.com/watch?v=1jTqRlIfdKY
# This file is intentionally unchanged from the supplied reference except for this provenance notice.
import numpy as np
import math
from scipy.io import wavfile

# Signal and noise parameters +/-8.38M
# 24 bit range
Fs = 48000
F = 4000
As = 7.8e6
time = 60
An = 46800

# Create timesignal
N = Fs * time
t = np.arange(N) / Fs
signal = np.sin(2 * math.pi * F * t) * As
# Create noise signal
noise = np.random.uniform(low=-An, high=An, size=N)
# 24-bit RPDF dithering
dith1 = np.random.uniform(-0.5, 0.5, N)
# Add signal, noise and dithering and scale
signaloriginal = signal + noise + dith1

# Quantize to 24-bit range, stored in int32
signal = signaloriginal.copy() * 256
signal = np.rint(signal).astype(np.int32)
stereo = np.stack((signal, signal), axis=-1)
filename = (
    "WAVtestsignals/signal"
    + str(round(As / 1e6, 2))
    + "M_noise"
    + str(round(An))
    + "int32_24bit.wav"
)
wavfile.write(filename, Fs, stereo)

# Quantize to 24-bit range, stored as float32
INT24_MAX = (2**23) - 1
x_float = (signaloriginal.copy() / INT24_MAX).astype(np.float32)  # normalize to [-1,1]
stereo_f = np.column_stack([x_float, x_float])
filename = (
    "WAVtestsignals/signal"
    + str(round(As / 1e6, 2))
    + "M_noise"
    + str(round(An))
    + "float32_24bit.wav"
)
wavfile.write(filename, Fs, stereo_f)
