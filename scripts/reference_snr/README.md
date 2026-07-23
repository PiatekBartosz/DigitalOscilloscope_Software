# SNR reference scripts

`SNR_algorithm.py` and `generate_24bit_testfile.py` are copied reference material from Hans Rosenberg's video, [Are you measuring the right noise?](https://www.youtube.com/watch?v=1jTqRlIfdKY). They are retained for comparison and are not to be edited beyond their provenance headers.

Run the adapter interactively from the `DigitalOscilloscope_Software` directory:

```bash
python scripts/run_reference_snr.py
```

The adapter converts the capture to the five-column QA403 format in a temporary directory, configures only that temporary copy with the capture's sample rate and selected band, and then runs it. It writes the spectrum plot to `scripts/results/reference/` by default. Use `--keep-workdir` to inspect those generated comparison inputs. The reference dependencies are isolated in `requirements.txt` so they are not required by the GUI.

Command-line options remain available for repeatable runs, for example:

```bash
python scripts/run_reference_snr.py scripts/results/captures/correct_frame.csv --channel 1 --band-min 100 --band-max 10100
```
