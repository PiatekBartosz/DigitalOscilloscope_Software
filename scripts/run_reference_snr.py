#!/usr/bin/env python3
"""Run the bundled Hans Rosenberg SNR reference against an oscilloscope CSV."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


SOFTWARE_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = SCRIPTS_DIR / "reference_snr"
REFERENCE_SCRIPT = REFERENCE_DIR / "SNR_algorithm.py"
REFERENCE_RESULTS_DIR = SCRIPTS_DIR / "results" / "reference"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a DigitalOscilloscope capture CSV to QA403's five-column "
            "format and run the bundled SNR reference implementation."
        )
    )
    parser.add_argument(
        "capture_csv", type=Path, nargs="?", help="capture CSV saved by the GUI"
    )
    parser.add_argument(
        "--channel", choices=("1", "2"), default="1", help="channel to analyse"
    )
    parser.add_argument("--band-min", type=float, default=100.0, metavar="HZ")
    parser.add_argument("--band-max", type=float, default=10100.0, metavar="HZ")
    parser.add_argument(
        "--save-plot",
        type=Path,
        metavar="PATH",
        help="save the reference full-spectrum plot as a PNG",
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="keep the converted QA403 CSV and patched temporary script for inspection",
    )
    return parser.parse_args()


def _prompt(prompt: str, default: str) -> str:
    answer = input(f"{prompt} [{default}]: ").strip()
    return answer or default


def capture_sample_rate_and_depth(path: Path) -> tuple[float, int]:
    """Read the metadata needed to choose a usable reference-analysis band."""
    metadata: dict[str, str] = {}
    with path.open() as capture:
        for line in capture:
            if not line.startswith("#"):
                break
            key, separator, value = line[1:].strip().partition("=")
            if separator:
                metadata[key] = value
    return float(metadata["fs_hz"]), int(metadata.get("capture_depth", "8192"))


def configure_interactively(args: argparse.Namespace) -> None:
    """Ask for all analysis settings when the script is launched without arguments."""
    captures = sorted(
        (SCRIPTS_DIR / "results" / "captures").glob("*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    default_capture = str(captures[0]) if captures else "capture.csv"
    capture_text = _prompt("Capture CSV", default_capture)
    args.capture_csv = Path(capture_text)
    fs_hz, capture_depth = capture_sample_rate_and_depth(args.capture_csv)
    args.channel = _prompt("Channel", args.channel)
    if args.channel not in {"1", "2"}:
        raise ValueError("channel must be 1 or 2")
    args.band_min = float(_prompt("Band minimum (Hz)", f"{args.band_min:g}"))
    max_bin_hz = fs_hz / 2 - fs_hz / capture_depth
    args.band_max = float(_prompt("Band maximum (Hz)", f"{max_bin_hz:g}"))
    default_plot = (
        REFERENCE_RESULTS_DIR / f"{args.capture_csv.stem}_ch{args.channel}.png"
    )
    args.save_plot = Path(_prompt("Output plot", str(default_plot)))


def require_reference_dependencies() -> bool:
    missing = [
        package
        for package in ("pandas", "scipy", "matplotlib")
        if importlib.util.find_spec(package) is None
    ]
    if not missing:
        return True
    print(
        "Reference dependencies are missing: "
        + ", ".join(missing)
        + ". Install scripts/reference_snr/requirements.txt.",
        file=sys.stderr,
    )
    return False


def write_qa403_csv(
    path: Path, ch1, ch2, fs_hz: float, n_bits: int, channel: int
) -> None:
    selected, other = (ch1, ch2) if channel == 1 else (ch2, ch1)
    midpoint = 1 << (n_bits - 1)
    with path.open("w", newline="") as output:
        writer = csv.writer(output)
        for index, (value, other_value) in enumerate(zip(selected, other)):
            writer.writerow(
                (
                    f"{index / fs_hz:.12g}",
                    "0",
                    "0",
                    int(value) - midpoint,
                    int(other_value) - midpoint,
                )
            )


def configure_temporary_reference(
    source: str, fs_hz: float, band_min: float, band_max: float
) -> str:
    replacements = {
        "snrfmin": repr(band_min),
        "snrfmax": repr(band_max),
        "Fs": repr(fs_hz),
        "ideal_timesignal": "0",
        "filename": '"QA403timesignals/adapter_input.csv"',
    }
    for name, value in replacements.items():
        source, substitutions = re.subn(
            # These settings are module-level assignments.  Limiting the match
            # to column zero prevents replacing similarly named function
            # arguments (notably ``filename`` in plotting calls).
            rf"^{name}[ \t]*=.*$",
            f"{name} = {value}",
            source,
            flags=re.MULTILINE,
        )
        if not substitutions:
            raise RuntimeError(f"could not configure reference setting {name!r}")
    source, substitutions = re.subn(
        r"^    Fs\s*=\s*48000\s*$",
        "    # Keep the sample rate configured above for this capture.",
        source,
        flags=re.MULTILINE,
    )
    if substitutions != 1:
        raise RuntimeError(
            "could not disable the reference script's fixed 48 kHz sample rate"
        )
    source = source.replace("import numpy as np", "import os\nimport numpy as np", 1)
    source = source.replace(
        "    plt.show()",
        '    plot_path = os.environ.get("DSO_REFERENCE_PLOT_PATH")\n'
        "    if plot_path:\n"
        "        plt.savefig(plot_path, dpi=150)\n"
        "        plt.close(fig)\n"
        "    else:\n"
        "        plt.show()",
        1,
    )
    return source


def run() -> int:
    args = parse_args()
    if args.capture_csv is None:
        try:
            configure_interactively(args)
        except (EOFError, ValueError) as error:
            print(f"Interactive configuration failed: {error}", file=sys.stderr)
            return 2
    if not args.capture_csv.is_file():
        print(f"Capture CSV not found: {args.capture_csv}", file=sys.stderr)
        return 2
    if args.band_min < 0 or args.band_max <= args.band_min:
        print(
            "--band-max must be greater than --band-min, and --band-min cannot be negative.",
            file=sys.stderr,
        )
        return 2
    if not require_reference_dependencies():
        return 2

    sys.path.insert(0, str(SOFTWARE_DIR))
    from analysis.capture_io import load_capture_csv

    ch1, ch2, metadata = load_capture_csv(args.capture_csv)
    if args.band_max >= metadata.fs_hz / 2:
        print(
            "The selected SNR band must be below the capture Nyquist frequency.",
            file=sys.stderr,
        )
        return 2
    fft_bin_hz = metadata.fs_hz / len(ch1)
    band_bin_count = (
        math.floor(args.band_max / fft_bin_hz)
        - math.ceil(args.band_min / fft_bin_hz)
        + 1
    )
    if band_bin_count < 16:
        print(
            f"The selected SNR band contains only {band_bin_count} FFT bins "
            f"({fft_bin_hz:g} Hz/bin); choose a wider band.",
            file=sys.stderr,
        )
        return 2

    workdir = Path(tempfile.mkdtemp(prefix="dso_reference_snr_"))
    try:
        qa_dir = workdir / "QA403timesignals"
        qa_dir.mkdir()
        write_qa403_csv(
            qa_dir / "adapter_input.csv",
            ch1,
            ch2,
            metadata.fs_hz,
            metadata.n_bits,
            int(args.channel),
        )
        temporary_script = workdir / REFERENCE_SCRIPT.name
        temporary_script.write_text(
            configure_temporary_reference(
                REFERENCE_SCRIPT.read_text(),
                metadata.fs_hz,
                args.band_min,
                args.band_max,
            )
        )
        print(
            f"Reference input: {len(ch1)} samples, {metadata.fs_hz:g} Hz, "
            f"CH{args.channel}, {args.band_min:g}–{args.band_max:g} Hz."
        )
        print(
            "The reference reports RMS in centred ADC-code units; its SNR result is scale-independent."
        )
        environment = os.environ | {"MPLBACKEND": "Agg"}
        if args.save_plot:
            plot_path = args.save_plot.resolve()
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            environment["DSO_REFERENCE_PLOT_PATH"] = str(plot_path)
        result = subprocess.run(
            [sys.executable, temporary_script.name], cwd=workdir, env=environment
        )
        if result.returncode == 0 and args.save_plot:
            print(f"Saved reference plot -> {args.save_plot}")
        return result.returncode
    finally:
        if args.keep_workdir:
            print(f"Temporary reference workspace retained at: {workdir}")
        else:
            shutil.rmtree(workdir)


if __name__ == "__main__":
    raise SystemExit(run())
