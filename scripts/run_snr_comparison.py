#!/usr/bin/env python3
"""Interactively run the native and reference SNR analyses on one capture."""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
import sys

from run_reference_snr import capture_sample_rate_and_depth


SCRIPTS_DIR = Path(__file__).resolve().parent
IMPLEMENTATION_SCRIPT = SCRIPTS_DIR / "snr_analysis.py"
REFERENCE_SCRIPT = SCRIPTS_DIR / "run_reference_snr.py"
RESULTS_DIR = SCRIPTS_DIR / "results"


def prompt(label: str, default: str) -> str:
    answer = input(f"{label} [{default}]: ").strip()
    return answer or default


def default_capture() -> str:
    captures = sorted(
        (RESULTS_DIR / "captures").glob("*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return str(captures[0]) if captures else "capture.csv"


def main() -> int:
    print("SNR comparison: native implementation and QA403 reference")
    try:
        capture_csv = Path(prompt("Capture CSV", default_capture()))
        if not capture_csv.is_file():
            raise ValueError(f"capture CSV not found: {capture_csv}")
        channel = prompt("Channel", "1")
        if channel not in {"1", "2"}:
            raise ValueError("channel must be 1 or 2")
        fs_hz, capture_depth = capture_sample_rate_and_depth(capture_csv)
        max_bin_hz = fs_hz / 2 - fs_hz / capture_depth
        band_min = float(prompt("Reference band minimum (Hz)", "100"))
        band_max = float(prompt("Reference band maximum (Hz)", f"{max_bin_hz:g}"))
        if band_min < 0 or band_max <= band_min:
            raise ValueError(
                "band maximum must be greater than band minimum, and minimum cannot be negative"
            )
    except (EOFError, ValueError) as error:
        print(f"Interactive configuration failed: {error}", file=sys.stderr)
        return 2

    implementation_dir = RESULTS_DIR / "implementation"
    reference_dir = RESULTS_DIR / "reference"
    captures_dir = RESULTS_DIR / "captures"
    comparison_dir = RESULTS_DIR / "comparison"
    implementation_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)
    captures_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    archived_capture = captures_dir / capture_csv.name
    if capture_csv.resolve() != archived_capture.resolve():
        shutil.copy2(capture_csv, archived_capture)
    stem = f"{capture_csv.stem}_ch{channel}"
    implementation_plot = implementation_dir / f"{stem}.png"
    reference_plot = reference_dir / f"{stem}.png"

    commands = (
        (
            "Native implementation",
            [
                sys.executable,
                str(IMPLEMENTATION_SCRIPT),
                "--from-csv",
                str(capture_csv),
                "--channel",
                channel,
                "--save-plot",
                str(implementation_plot),
            ],
        ),
        (
            "QA403 reference",
            [
                sys.executable,
                str(REFERENCE_SCRIPT),
                str(capture_csv),
                "--channel",
                channel,
                "--band-min",
                str(band_min),
                "--band-max",
                str(band_max),
                "--save-plot",
                str(reference_plot),
            ],
        ),
    )

    print("Starting both analyses…", flush=True)
    running = [
        (
            name,
            subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            ),
        )
        for name, command in commands
    ]
    failed = False
    outputs: dict[str, str] = {}
    for name, process in running:
        output, _ = process.communicate()
        outputs[name] = output
        print(f"\n--- {name} output ---")
        print(output, end="" if output.endswith("\n") else "\n")
        return_code = process.returncode
        if return_code:
            print(f"{name} failed with exit status {return_code}.", file=sys.stderr)
            failed = True

    if failed:
        return 1
    native_match = re.search(
        r"^SNR\s*:\s*([-+\d.]+)\s+dB", outputs["Native implementation"], re.MULTILINE
    )
    reference_match = re.search(
        r"^The SNR is:\s*([-+\d.]+)\s+dB", outputs["QA403 reference"], re.MULTILINE
    )
    if not native_match or not reference_match:
        print(
            "Could not extract both SNR values from the analysis output.",
            file=sys.stderr,
        )
        return 1
    native_snr = float(native_match.group(1))
    reference_snr = float(reference_match.group(1))
    difference = reference_snr - native_snr
    summary = (
        "SNR comparison\n"
        f"  Native implementation: {native_snr:.6f} dB\n"
        f"  QA403 reference:      {reference_snr:.6f} dB\n"
        f"  Reference − native:   {difference:+.6f} dB\n"
    )
    summary_path = comparison_dir / f"{stem}.txt"
    summary_path.write_text(summary)
    print(f"\n{summary}", end="")
    print(f"Comparison log: {summary_path}")
    print(f"Raw capture:    {archived_capture}")
    print(f"Native plot:    {implementation_plot}")
    print(f"Reference plot: {reference_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
