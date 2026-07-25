"""Versioned CSV interchange format for oscilloscope captures.

Every capture contains the original 14-bit ADC codes and the effective sample
rate needed for spectral analysis.  Desktop captures may additionally contain
the display-voltage conversion used for the waveform view.  Metadata is kept
as ``# key=value`` lines so the file remains easy to inspect in Octave, Excel,
or a text editor.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime
import pathlib
from typing import Mapping

import numpy as np


@dataclasses.dataclass
class CaptureMeta:
    fs_hz: float
    n_bits: int = 14
    timestamp: str = ""
    fields: dict[str, str] = dataclasses.field(default_factory=dict)


def save_capture_csv(
    path: str | pathlib.Path,
    ch1: np.ndarray,
    ch2: np.ndarray,
    fs_hz: float,
    n_bits: int = 14,
    metadata: Mapping[str, object] | None = None,
    ch1_volts: np.ndarray | None = None,
    ch2_volts: np.ndarray | None = None,
) -> None:
    """Save a two-channel raw-code capture and optional voltage conversion."""
    ch1 = np.asarray(ch1, dtype=np.uint16)
    ch2 = np.asarray(ch2, dtype=np.uint16)
    if len(ch1) != len(ch2) or len(ch1) == 0:
        raise ValueError("capture must contain equal, non-empty channel arrays")
    if fs_hz <= 0:
        raise ValueError(f"fs_hz must be positive, got {fs_hz}")

    have_volts = ch1_volts is not None or ch2_volts is not None
    if have_volts:
        if ch1_volts is None or ch2_volts is None:
            raise ValueError("both ch1_volts and ch2_volts are required together")
        ch1_volts = np.asarray(ch1_volts, dtype=np.float64)
        ch2_volts = np.asarray(ch2_volts, dtype=np.float64)
        if len(ch1_volts) != len(ch1) or len(ch2_volts) != len(ch2):
            raise ValueError("voltage and raw-code arrays must have equal lengths")

    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        f.write(f"# DigitalOscilloscope capture\n")
        f.write("# format_version=2\n")
        f.write(f"# fs_hz={fs_hz!r}\n")
        f.write(f"# n_bits={n_bits}\n")
        f.write(f"# timestamp={timestamp}\n")
        for key, value in (metadata or {}).items():
            if not key or "=" in key or "\n" in str(value):
                raise ValueError(f"invalid capture metadata entry: {key!r}")
            f.write(f"# {key}={value}\n")
        writer = csv.writer(f)
        header = ["index", "ch1_raw", "ch2_raw"]
        if have_volts:
            header.extend(["ch1_volts", "ch2_volts"])
        writer.writerow(header)
        for i, (a, b) in enumerate(zip(ch1.tolist(), ch2.tolist())):
            row = [i, a, b]
            if have_volts:
                row.extend([f"{ch1_volts[i]:.9g}", f"{ch2_volts[i]:.9g}"])
            writer.writerow(row)


def load_capture_csv(
    path: str | pathlib.Path,
) -> tuple[np.ndarray, np.ndarray, CaptureMeta]:
    """Load a capture saved by save_capture_csv(). Returns (ch1, ch2, meta)."""
    path = pathlib.Path(path)
    meta_kv: dict[str, str] = {}
    data_lines: list[str] = []

    with open(path, "r", newline="") as f:
        for line in f:
            if line.startswith("#"):
                if "=" in line:
                    key, _, value = line[1:].strip().partition("=")
                    meta_kv[key.strip()] = value.strip()
            else:
                data_lines.append(line)

    if "fs_hz" not in meta_kv:
        raise ValueError(
            f"{path}: missing '# fs_hz=' metadata line — not a valid capture file"
        )

    reader = csv.DictReader(data_lines)
    ch1_list: list[int] = []
    ch2_list: list[int] = []
    for row in reader:
        ch1_key = "ch1_raw" if "ch1_raw" in row else "ch1"
        ch2_key = "ch2_raw" if "ch2_raw" in row else "ch2"
        ch1_list.append(int(row[ch1_key], 0))
        ch2_list.append(int(row[ch2_key], 0))

    meta = CaptureMeta(
        fs_hz=float(meta_kv["fs_hz"]),
        n_bits=int(meta_kv.get("n_bits", 14)),
        timestamp=meta_kv.get("timestamp", ""),
        fields=meta_kv,
    )
    return (
        np.array(ch1_list, dtype=np.uint16),
        np.array(ch2_list, dtype=np.uint16),
        meta,
    )
