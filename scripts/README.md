# Analysis scripts

Run both SNR tools together, with one interactive setup, from `DigitalOscilloscope_Software`:

```bash
python scripts/run_snr_comparison.py
```

## Verification of spectral metrics

Generate a deterministic synthetic signal, verify SNR, SINAD and ENOB against
their analytical values, and save the plot:

```bash
python scripts/verify_spectral_metrics.py
```

The launcher starts the native implementation and QA403-reference adapter concurrently, then saves one PNG from each tool. The individual tools can still be run interactively as well.

Generated files are organized by implementation:

- `results/implementation/` contains plots, reports, and live captures from `snr_analysis.py`.
- `results/reference/` contains spectrum plots from `run_reference_snr.py`.
- `results/captures/` retains raw CSV captures. The captures used for the thesis
  are grouped in `results/captures/thesis_measurements/`.
- `results/comparison/` contains the saved side-by-side SNR summaries.
- `results/legacy/` contains debug and check captures retained for archival use.

The unmodified third-party reference material used by the adapter is in `reference_snr/`.

## Measurement-plan analysis

Analyze a driven sine wave from one channel or the sample-wise average of both
channels:

```bash
python scripts/snr_analysis.py --from-csv capture.csv --channel 1 --window hann
python scripts/snr_analysis.py --from-csv capture.csv --channel average --window hann
```

The average-channel mode subtracts the mean of each channel before averaging
corresponding samples. The reported spectrum omits the 0 Hz bin. For the Hann
window, the one-sided tone amplitude includes both the factor of two for the
omitted negative-frequency spectrum and the coherent-gain correction of about
two.

For a grounded-input capture, use the dedicated noise mode. It does not search
for a fundamental tone:

```bash
python scripts/snr_analysis.py --from-csv grounded.csv --channel 1 \
  --noise-only --adc-range-vpp 2 --window hann \
  --save-report grounded_ch1.json --save-plot grounded_ch1.png
```

`--adc-range-vpp` may be omitted when the CSV metadata contains the matching
physical ADC range. Noise reports include RMS codes, ADC-input RMS voltage,
dBFS/bin, dBFS/Hz, and the Hann equivalent noise bandwidth. Use raw captures
for all measurement-plan results.
