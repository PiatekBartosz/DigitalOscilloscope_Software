# Analysis scripts

Run both SNR tools together, with one interactive setup, from `DigitalOscilloscope_Software`:

```bash
python scripts/run_snr_comparison.py
```

The launcher starts the native implementation and QA403-reference adapter concurrently, then saves one PNG from each tool. The individual tools can still be run interactively as well.

## Interactive time-domain viewer

Open a saved CSV capture in a zoomable time-domain plot:

```bash
python scripts/view_time_domain.py capture.csv
```

Use the Matplotlib toolbar to zoom, pan, reset the view, or save an image.
The default plot uses centred signed ADC codes; add `--raw-codes` to see the
stored unsigned codes, or use `--channels 1` / `--channels 2` for one channel.

Generated files are organized by implementation:

- `results/implementation/` contains plots, reports, and live captures from `snr_analysis.py`.
- `results/reference/` contains spectrum plots from `run_reference_snr.py`.
- `results/captures/` retains the raw CSV used by each comparison run.
- `results/comparison/` contains the saved side-by-side SNR summaries.

The unmodified third-party reference material used by the adapter is in `reference_snr/`.
