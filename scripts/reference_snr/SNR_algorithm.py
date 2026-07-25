# Reference implementation from Hans Rosenberg, “Are you measuring the right noise?”
# https://www.youtube.com/watch?v=1jTqRlIfdKY
# This file is intentionally unchanged from the supplied reference except for this provenance notice.
import pandas as pd
import numpy as np
import os
import math

basedir = os.path.dirname(os.path.abspath(__file__))
from scipy.signal import windows
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def normal_plot(
    data,
    filename=None,
    xmin="a",
    xmax="a",
    ymin="a",
    ymax="a",
    ytick="a",
    xtick="a",
    ytitle="",
    xtitle="",
    titletext="",
    xlin="lin",
    ylin="lin",
    color1="yellow",
    linewidth=4,
    ticksize=40,
    titlesize=48,
    titlecolor="yellow",
    plottype="line",
):
    fig, ax = plt.subplots(figsize=(12, 10))
    bgcol = "black"
    fig.patch.set_facecolor(bgcol)
    ax.set_facecolor(bgcol)
    gridcolor = "white"
    tickcolor = "white"
    axis_line_color = "white"
    label_color = "yellow"
    traces = [
        {"col": "y1", "color": color1},
        {"col": "y2", "color": "magenta"},
        {"col": "y3", "color": "cyan"},
        {"col": "y4", "color": "orangered"},
    ]
    for trace in traces:
        if trace["col"] in data.columns:
            ax.plot(
                data["x"], data[trace["col"]], color=trace["color"], linewidth=linewidth
            )
    if xlin == "lin":
        ax.set_xscale("linear")
        if xtick != "a":
            ax.xaxis.set_major_locator(ticker.MultipleLocator(xtick))
        if xmin != "a":
            ax.set_xlim(xmin, xmax)
    elif xlin == "log":
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))
        ax.xaxis.set_minor_locator(
            ticker.LogLocator(base=10, subs=np.arange(1.0, 10.0), numticks=100)
        )
        if xmin != "a":
            ax.set_xlim(xmin, xmax)
    if ylin == "lin":
        ax.set_yscale("linear")
        if ytick != "a":
            ax.yaxis.set_major_locator(ticker.MultipleLocator(ytick))
        if ymin != "a":
            ax.set_ylim(ymin, ymax)
    elif ylin == "log":
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))
        if ymin != "a":
            ax.set_ylim(ymin, ymax)
    ax.grid(True, color=gridcolor, linestyle="-", linewidth=2, alpha=1.0)
    if xlin == "log" or ylin == "log":
        ax.grid(
            True,
            which="minor",
            color=gridcolor,
            linestyle="-",
            linewidth=0.5,
            alpha=0.5,
        )
    for spine in ax.spines.values():
        spine.set_edgecolor(axis_line_color)
        spine.set_linewidth(2)
    # font_axis = {'family': 'Arial Black', 'size': ticksize, 'color': label_color}
    # font_title = {'family': 'Arial Black', 'size': titlesize, 'color': titlecolor}
    font_axis = {"size": ticksize, "color": label_color}
    font_title = {"size": titlesize, "color": titlecolor}
    ax.set_xlabel(xtitle, **font_axis)
    ax.set_ylabel(ytitle, **font_axis)
    ax.set_title(titletext, **font_title, pad=20)
    ax.tick_params(
        axis="both",
        which="major",
        colors=tickcolor,
        labelsize=ticksize,
        width=2,
        length=10,
        pad=10,
    )
    ax.tick_params(axis="both", which="minor", colors=tickcolor, width=1, length=5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname("DejaVu Sans")
    plt.tight_layout()
    plt.show()


def compute_psd_fft(signal, fs):
    # Calculates a PSD spectrum in V^2/Hz based on a time-signal
    # Lets get some main parameters first
    N = len(signal)
    binsize = fs / N  # in Hz
    # We first create a window function
    window = windows.hann(N, sym=False)
    # Next, we apply that window function to the signal
    signal = signal * window
    # We also need the amplitude correction factor for the window function. As you've
    # seen in the video: A window reduces the amplitude of the signal. This needs to
    # be corrected.
    wincorr = np.sum(window**2) / N
    # Calculate FFT
    signal = np.asarray(signal)
    spectrum = np.fft.rfft(signal)  # Do the FFT
    f = np.fft.rfftfreq(N, 1 / fs)  # Generate frequency axis
    # Calculate normalized PSD spectrum. A number of things go on here
    # Divide by N to normalize the spectrum. Otherwise the 'height' of signals would
    # keep growing with the amount of samples in the timesignal.
    # Divide by Fs to normalize to V^2/Hz (in combination with dividing by N)
    # Take the ABS value: An FFT produces complex numbers, we just want the magnitude
    # Square the spectrum: We want V^2
    # Apply the window correction we calculated earlier.
    spectrum = (np.abs(spectrum) ** 2) / (fs * N * wincorr)
    # Just keep the positive frequencies (An FFT runs from -Fs/2 to Fs/2)
    if N % 2 == 0:
        spectrum[1:-1] = spectrum[1:-1] * 2
    else:
        spectrum[1:] = spectrum[1:] * 2
    return f, spectrum, binsize


def remove_peak(f, spectrum, fftbinsize):
    # Function: Cutout main tone and replace it with noise to create a noise-spectrum
    # find the highest peak in the spectrum
    tone_idx = spectrum.argmax()
    ftone = f[tone_idx]
    # Make a copy of the spectrum and make a dataframe findfloor, we'll use this to find the noise-floor
    # above the tone
    findfloor = pd.DataFrame()
    findfloor["f"] = f.copy()
    findfloor["P"] = spectrum.copy()
    # Set the size of the 'binwindow' The number of bins we are going to average with each step
    binwindow = 30  # Too small and you don't have any averaging anymore so the first rising sample triggers a stop way too early...
    searchrange = 600  # How many bins do we 'walk away' from the carrier to find a flat noise floor
    # So first, we make a list of mean values of binwindows as they 'walk away' from the center of the tone
    win_mean = []
    # for safety, it may be wise to set idx_150, so you're sure you're going to catch the flat spot
    for windowcenter in np.arange(tone_idx, tone_idx + searchrange, binwindow / 2):
        # Find edges of current binwindow
        winmin = round(windowcenter - binwindow / 2)
        winmax = round(windowcenter + binwindow / 2)
        # Prevent window going outside of the spectrum which causes a crash
        if winmax > len(findfloor):
            winmax = len(findfloor)
        # get window average and add it to the list of mean window values
        temp = np.mean(findfloor[winmin:winmax]["P"])
        win_mean = win_mean + [temp]
    # Now calculate relative difference with previous sample win_mean[i]/win_mean[i-1]
    win_mean = np.array(win_mean)
    rel_change = win_mean[1:] / win_mean[:-1]
    # Remove first result, it can be larger than one due to rounding of bins. This can also be
    # solved by making sure binwindow only has even numbers I think, but I'm solving it this way
    rel_change = rel_change[1:]
    # Visualize Relative value binwindows compared to previous binwindow
    plotdata = pd.DataFrame()
    plotdata["x"] = np.arange(len(rel_change)) * binwindow / 2
    plotdata["y1"] = rel_change.copy()
    normal_plot(
        data=plotdata,
        titletext="Relative value current binwindow compared to previous binwindow",
        xtitle="Binwindow distance to carrier",
        ytitle="Ratio current / previous binwindow",
        filename=basedir + "/plot1.html",
        xlin="lin",
        ylin="log",
        xmin=-searchrange,
        xmax=searchrange,
        # ymin=10e-9, ymax=40,
        linewidth=4,
    )
    # Now find the offset frequency (with respect to the carrier) where rel-change is biger than 1
    # delta_f = next(i for i, v in enumerate(rel_change) if v > 1)
    delta_f_bins = [i for i, v in enumerate(rel_change) if v >= 1]
    # Pick first or second time rel_change > 1. second time gives more margin
    delta_f_bins = delta_f_bins[0]
    delta_f_bins = round(delta_f_bins * binwindow / 2)
    # Now we need to replace the tone with representative noise. We take noise from the top of the spectrum. This is better
    # because for very low frequencies, you can encounter dc at the lower side of the tone. This happens with 100Hz tones
    # and only 100ms of data for instance
    # select spectrum just above the area we want to cutout
    noisewindow = 100  # Sets the amount of bins to average to get the 'fill in' value for the noise-floor
    replacementnoise = np.mean(
        spectrum[tone_idx + delta_f_bins : tone_idx + delta_f_bins + noisewindow]
    )
    # copy spectrum to noise_spectrum and replace carrier by replacementnoise
    noise_spectrum = spectrum.copy()
    idxmin = tone_idx - delta_f_bins
    idxmax = tone_idx + delta_f_bins
    if idxmin < 0:
        idxmin = 0
    if idxmax > len(noise_spectrum):
        idxmax = len(noise_spectrum)
    noise_spectrum[idxmin:idxmax] = replacementnoise
    # Visualize the spectrum for debugging
    plotdata = pd.DataFrame()
    plotdata["x"] = f
    plotdata["y1"] = spectrum
    plotdata["y2"] = noise_spectrum
    normal_plot(
        data=plotdata,
        titletext="Spectrum and constructed noise-spectrum without carrier",
        xtitle="Frequency (Hz)",
        ytitle="Level (V²/Hz)",
        filename=basedir + "/plot2.html",
        xlin="lin",
        ylin="log",
        xmin=ftone - delta_f_bins * fftbinsize * 10,
        xmax=ftone + delta_f_bins * fftbinsize * 10,
        linewidth=4,
    )
    return ftone, noise_spectrum


def remove_spurious(f, noise_spectrum):
    # Remove spurious tones using the Median Absolute Deviation (MAD) approach.
    # This method is very effective at detecting outliers because it measures
    # how far each spectral bin deviates from the median level (which represents
    # the noise floor), rather than from the mean, which can be strongly skewed
    # by narrowband tones or other strong signals.
    ns_db = 20 * np.log10(noise_spectrum.copy())
    median = np.median(ns_db)
    mad = np.median(np.abs(ns_db - median))
    sigma = 1.4826 * mad
    upper = median + 3 * sigma
    ns_db = np.minimum(ns_db, upper)
    ns = 10 ** (ns_db / 20)
    # Store the result
    noise_spectrum_nospurs = ns.copy()
    # Visualize MAD method
    plotdata = pd.DataFrame()
    plotdata["x"] = f
    plotdata["y1"] = 20 * np.log10(noise_spectrum.copy())
    plotdata["y2"] = ns_db
    plotdata["y3"] = [upper] * len(f)
    plotdata["y4"] = [median] * len(f)
    normal_plot(
        data=plotdata,
        titletext="Spur removal",
        xtitle="Frequency (Hz)",
        ytitle="Level (dB)",
        filename=basedir + "/plot3.html",
        xlin="log",
        ylin="lin",
        linewidth=4,
    )
    # Return spurious free noise spectrum
    return noise_spectrum_nospurs


############################################################################
#### INPUT PARAMETERS ######################################################
############################################################################
##### SNR measurement parameters
snrfmin = 100  # Start bandwidth for SNR measurement
snrfmax = 10100  # Stop bandwidth for SNR measurement
Fs = 48000  # Samplerate of the signal
##### Select if you want to load a real measurement file (captured time-signal from QA403) or
##### generate an ideal timesignal
ideal_timesignal = (
    0  # 1 creates an ideal timesignal. 0 loads an actual measured timesignal
)
filename = "QA403timesignals/120dB_64k.csv"
# filename         = "QA403timesignals/110dB_64k.csv"
# filename         = "QA403timesignals/105dB_64k.csv"
# filename         = "QA403timesignals/100dB_64k.csv"
# filename         = "QA403timesignals/90dB_64k.csv"
# filename         = "QA403timesignals/70dB_64k.csv"
# filename         = "QA403timesignals/50dB_64k.csv"
# filename         = "QA403timesignals/30dB_4k.csv"
# filename         = "QA403timesignals/30dB_8k.csv"
# filename         = "QA403timesignals/30dB_16k.csv"
# filename         = "QA403timesignals/30dB_32k.csv"
# filename         = "QA403timesignals/30dB_64k.csv"
# filename         = "QA403timesignals/30dB_128k.csv"
# filename         = "QA403timesignals/30dB_256k.csv"
# filename         = "QA403timesignals/30dB_512k.csv"
# filename         = "QA403timesignals/30dB_1024k.csv"
##### SIGNAL PARAMETERS WHEN IDEAL SIGNAL IS SELECTED
F = 2000.05  # Frequency of sinewave
signal_peak = 7.8e6  # peak voltage of sinewave
time = 1  # Length of testsignal in seconds, longer time = better accuracy
noise_peak = 46800  # peak voltage of the noise
############################################################################
#### END INPUT PARAMETERS ##################################################
############################################################################


############################################################################
#### LOAD A QA403 TIME-SIGNAL OR CREATE AN IDEAL TIME-SIGNAL ###############
############################################################################
if ideal_timesignal == 0:
    ts = pd.read_csv(filename, sep=r",", comment="#", header=None, engine="python")
    # Assign column names based on the header description
    ts.columns = ["time", "output_left", "output_right", "left", "right"]
    # Keep only what you want
    ts = ts[["time", "left", "right"]]
    Fs = 48000
    N = len(ts)
    time = np.arange(N)
    time = time / Fs
    ts["time"] = time
else:
    N = int(Fs * time)
    t = np.arange(0, N)
    t = t * 1 / Fs
    signal = np.sin(2 * math.pi * F * t)
    signal = signal * signal_peak
    # Now create the noise
    noise = np.random.uniform(low=-noise_peak, high=noise_peak, size=N)
    # combine them and convert to 32-bit
    signal = signal + noise
    ts = pd.DataFrame()
    ts["time"] = t
    ts["left"] = signal / 10e3  # divide signal so the level fits nicely in the plots

############################################################################
#### CALCULATE PSD SPECTRUM ################################################
############################################################################
[f, spectrum, fftbinsize] = compute_psd_fft(ts["left"], Fs)
# CUT BAND OF INTEREST
mask = (f >= snrfmin) & (f <= snrfmax)
f = f[mask]
spectrum = spectrum[mask]

############################################################################
#### REMOVE HIGHEST PEAK AND REPLACE WITH NOISEFLOOR #######################
############################################################################
[ftone, noise_spectrum] = remove_peak(f, spectrum, fftbinsize)

############################################################################
#### REMOVE SPURIOUS TONES AND REPLACE BY AVERAGE NOISE FLOOR ##############
############################################################################
noise_spectrum = remove_spurious(f, noise_spectrum)

############################################################################
#### GET RMS VALUE OF MAIN TONE (SHOULD BE THE ONLY LARGE TONE) ############
############################################################################
# Ad 5 bins left and right of the highest peak in the spectrum.
# Multiply by fft binsize because we're converting to RMS
k0 = np.argmax(spectrum)
left = max(0, k0 - 5)
right = min(len(spectrum), k0 + 5)
signal_rms = np.sqrt(np.sum(spectrum[left:right]) * fftbinsize)

############################################################################
#### GET RMS VALUE OF THE NOISE-FLOOR ######################################
############################################################################
# Sum all the bins together, but you must correct for binsize since you're calculating to voltages now.
Pnoise = np.sum(noise_spectrum.copy()) * fftbinsize
noise_rms = np.sqrt(Pnoise)  # Then simply take the sqrt to get the voltage

############################################################################
#### CALCULATE / REPORT RESULTS TO CONSOLE #################################
############################################################################
if ideal_timesignal == 0:
    # QA403 captured time-signal was processed
    print("The signal level in Vrms is: " + str(round(signal_rms, 5)) + " V")
    print("The noise level in Vrms is:  " + str(round(1000 * noise_rms, 6)) + " mV")
    print(
        "The SNR is:                  "
        + str(round(20 * np.log10(signal_rms / noise_rms), 6))
        + " dB"
    )
    print(
        "The average noise is:        "
        + str(round(1e6 * noise_rms / np.sqrt(snrfmax - snrfmin), 4))
        + " uV/rtHz"
    )
else:
    # Ideal time-signal was processed
    # Calculate ideal SNR value
    As_rms = signal_peak / np.sqrt(2)
    An_rms = noise_peak / np.sqrt(3)
    An_v_rthz = An_rms / np.sqrt(Fs / 2)
    An_rms_in_bw = An_v_rthz * np.sqrt(snrfmax - snrfmin)
    ideal_snr = 20 * np.log10(As_rms / An_rms_in_bw)
    print("Ideal SNR from input params: " + str(round(ideal_snr, 2)) + " dB")
    print(
        "The CalculatedSNR is:        "
        + str(round(20 * np.log10(signal_rms / noise_rms), 2))
        + " dB"
    )
    print(
        "The SNR error is:            "
        + str(round(20 * np.log10(signal_rms / noise_rms) - ideal_snr, 3))
        + " dB"
    )

############################################################################
#### VISUALIZE RESULTING SPECTRUM WITH NOISE REMOVAL #######################
############################################################################
plotdata = pd.DataFrame()
plotdata["x"] = f
plotdata["y1"] = spectrum.copy()
plotdata["y2"] = noise_spectrum.copy()
normal_plot(
    data=plotdata,
    titletext="Full Spectrum",
    xtitle="Frequency (Hz)",
    ytitle="Level (V²/Hz)",
    filename=basedir + "/plot4.html",
    xlin="log",
    ylin="log",
)
