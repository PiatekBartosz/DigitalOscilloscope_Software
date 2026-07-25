# Analysis scripts

Run both SNR tools together, with one interactive setup, from `DigitalOscilloscope_Software`:

```bash
python scripts/run_snr_comparison.py
```

The launcher starts the native implementation and QA403-reference adapter concurrently, then saves one PNG from each tool. The individual tools can still be run interactively as well.

Generated files are organized by implementation:

- `results/implementation/` contains plots, reports, and live captures from `snr_analysis.py`.
- `results/reference/` contains spectrum plots from `run_reference_snr.py`.
- `results/captures/` retains the raw CSV used by each comparison run.
- `results/comparison/` contains the saved side-by-side SNR summaries.

The unmodified third-party reference material used by the adapter is in `reference_snr/`.
