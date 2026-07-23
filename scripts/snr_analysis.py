from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from analysis.capture_io import load_capture_csv, save_capture_csv
from analysis.metrics import compute_metrics, format_report, suggest_coherent_frequency
from core.command_client import CommandClient


SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
IMPLEMENTATION_RESULTS_DIR = SCRIPTS_DIR / "results" / "implementation"


class DeviceError(RuntimeError):
    pass


def _prompt(prompt: str, default: str) -> str:
    answer = input(f"{prompt} [{default}]: ").strip()
    return answer or default


def configure_interactively(args: argparse.Namespace) -> None:
    """Configure a local-capture or live-device analysis without CLI options."""
    source = _prompt("Analysis source (capture/live)", "capture").lower()
    if source not in {"capture", "live"}:
        raise ValueError("analysis source must be 'capture' or 'live'")

    args.channel = int(_prompt("Channel", str(args.channel)))
    if args.channel not in (1, 2):
        raise ValueError("channel must be 1 or 2")
    args.n_harmonics = int(_prompt("Number of harmonics", str(args.n_harmonics)))
    args.window = _prompt("FFT window (hann/hamming/blackman/rect)", args.window)
    if args.window not in {"hann", "hamming", "blackman", "rect"}:
        raise ValueError("unsupported FFT window")

    if source == "capture":
        args.from_csv = pathlib.Path(_prompt("Capture CSV", "capture.csv"))
        base_name = f"{args.from_csv.stem}_ch{args.channel}"
    else:
        args.host = _prompt("Oscilloscope IP address", "192.168.1.1")
        args.fs_hz = float(_prompt("Sample rate (Hz)", "40000000"))
        base_name = f"live_ch{args.channel}"
        args.save_csv = IMPLEMENTATION_RESULTS_DIR / f"{base_name}.csv"

    args.save_report = IMPLEMENTATION_RESULTS_DIR / f"{base_name}.json"
    args.save_plot = IMPLEMENTATION_RESULTS_DIR / f"{base_name}.png"


async def _wait_for_reply(text_lines: list[str], timeout: float) -> str:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while not text_lines:
        if loop.time() > deadline:
            raise TimeoutError("timed out waiting for a reply from the device")
        await asyncio.sleep(0.05)
    line = text_lines.pop(0)
    if line.startswith("ERR"):
        raise DeviceError(f"device rejected command: {line}")
    return line


async def _live_acquire(
    host: str,
    port: int,
    decim: int | None,
    sample_size: int | None,
    timeout: float,
) -> tuple[np.ndarray, np.ndarray]:
    frame: dict = {}
    frame_event = asyncio.Event()
    text_lines: list[str] = []

    def on_frame(seq, ch1, ch2):
        frame["ch1"] = ch1
        frame["ch2"] = ch2
        frame_event.set()

    client = CommandClient(host, port, frame_cb=on_frame, text_cb=text_lines.append)
    await client.connect()
    try:
        if decim is not None:
            await client.send_command(f"afe decim {decim}")
            await _wait_for_reply(text_lines, timeout)
        if sample_size is not None:
            await client.send_command(f"afe sample_size {sample_size}")
            await _wait_for_reply(text_lines, timeout)

        await client.send_command("acquire")
        await asyncio.wait_for(frame_event.wait(), timeout=timeout)
    finally:
        await client.disconnect()

    return frame["ch1"], frame["ch2"]


def _save_plot(path: pathlib.Path, samples: np.ndarray, fs_hz: float, result, channel_label: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping --save-plot "
              "(pip install matplotlib)", file=sys.stderr)
        return

    fig, (ax_time, ax_spec) = plt.subplots(2, 1, figsize=(9, 6))

    t = np.arange(len(samples)) / fs_hz
    ax_time.plot(t * 1e3, samples, linewidth=0.8)
    ax_time.set_xlabel("Time (ms)")
    ax_time.set_ylabel(f"{channel_label} code")
    ax_time.set_title("Captured waveform")

    plot_floor_dbfs = -180.0
    spectrum_plot_dbfs = np.maximum(result.spectrum_dbfs, plot_floor_dbfs)
    ax_spec.plot(result.freqs_hz / 1e3, spectrum_plot_dbfs, linewidth=0.8)
    ax_spec.axvline(result.fundamental_freq_hz / 1e3, color="r", linestyle="--",
                     linewidth=0.8, label="fundamental")
    display_max_hz = min(result.freqs_hz[-1], max(10.0 * result.fundamental_freq_hz, 1.0))
    ax_spec.set_xlim(0.0, display_max_hz / 1e3)
    peak_dbfs = float(np.max(spectrum_plot_dbfs))
    ax_spec.set_ylim(plot_floor_dbfs, max(0.0, peak_dbfs + 6.0))
    ax_spec.set_xlabel("Frequency (kHz)")
    ax_spec.set_ylabel("dBFS")
    ax_spec.set_title(f"Spectrum — SNR={result.snr_db:.1f} dB  "
                       f"SINAD={result.sinad_db:.1f} dB  ENOB={result.enob:.2f} bits")
    ax_spec.legend()

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"Saved plot -> {path}")


def _save_report(path: pathlib.Path, result, channel_label: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        payload = {
            "channel": channel_label,
            "snr_db": result.snr_db,
            "thd_db": result.thd_db,
            "thd_percent": result.thd_percent,
            "sinad_db": result.sinad_db,
            "enob": result.enob,
            "sfdr_db": result.sfdr_db,
            "noise_floor_dbfs": result.noise_floor_dbfs,
            "fundamental_freq_hz": result.fundamental_freq_hz,
            "n_harmonics_used": result.n_harmonics_used,
        }
        path.write_text(json.dumps(payload, indent=2))
    else:
        path.write_text(format_report(result, channel_label) + "\n")
    print(f"Saved report -> {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--host", help="oscilloscope IP address (live acquisition)")
    src.add_argument("--from-csv", type=pathlib.Path,
                      help="analyze a previously saved capture instead of a live device")

    parser.add_argument("--suggest-freq", type=float, metavar="HZ",
                         help="print the nearest coherent-sampling frequency to HZ for the given "
                              "--fs-hz/--decim and --sample-size, then exit (no acquisition). Set "
                              "your signal generator to the printed frequency for accurate SNR.")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--channel", type=int, choices=(1, 2), default=1,
                         help="which channel to analyze (default: 1)")
    parser.add_argument("--fs-hz", type=float,
                         help="effective sample rate in Hz (required for --host unless --decim "
                              "is given; ignored for --from-csv, whose file already records it)")
    parser.add_argument("--decim", type=int,
                         help="decimation factor to configure on the device before capturing "
                              "(fs_hz = 80 MHz / decim if --fs-hz is not also given)")
    parser.add_argument("--sample-size", type=int,
                         help="capture depth to configure on the device before capturing "
                              "(power of two, 1..8192)")
    parser.add_argument("--n-bits", type=int, default=14, help="ADC resolution (default: 14)")
    parser.add_argument("--n-harmonics", type=int, default=5,
                         help="harmonics 2..N included in THD/SINAD (default: 5)")
    parser.add_argument("--window", default="hann", choices=("hann", "hamming", "blackman", "rect"),
                         help="FFT window (default: hann; use 'rect' only for coherent sampling)")
    parser.add_argument("--timeout", type=float, default=10.0,
                         help="live acquisition timeout in seconds (default: 10)")
    parser.add_argument("--save-csv", type=pathlib.Path,
                         help="save the raw captured waveform to this CSV path")
    parser.add_argument("--save-report", type=pathlib.Path,
                         help="save the computed metrics to this path (.json for JSON, else text)")
    parser.add_argument("--save-plot", type=pathlib.Path,
                         help="save a waveform+spectrum PNG plot (requires matplotlib)")

    args = parser.parse_args()
    if len(sys.argv) == 1:
        try:
            configure_interactively(args)
        except (EOFError, ValueError) as error:
            print(f"Interactive configuration failed: {error}", file=sys.stderr)
            return 2

    if args.suggest_freq is not None:
        if args.fs_hz is None and args.decim is None:
            parser.error("--suggest-freq requires --fs-hz or --decim")
        fs_hz = args.fs_hz if args.fs_hz is not None else 80_000_000.0 / args.decim
        n = args.sample_size or 8192
        coherent_hz, k = suggest_coherent_frequency(args.suggest_freq, fs_hz, n)
        print(f"Nearest coherent frequency to {args.suggest_freq:,.1f} Hz "
              f"(fs={fs_hz:,.1f} Hz, N={n}): {coherent_hz:,.3f} Hz ({k} cycles/record)")
        return 0

    if not args.host and not args.from_csv:
        parser.error("one of --host, --from-csv, or --suggest-freq is required")

    if args.from_csv:
        ch1, ch2, meta = load_capture_csv(args.from_csv)
        fs_hz = meta.fs_hz
        n_bits = meta.n_bits
        print(f"Loaded {len(ch1)} samples from {args.from_csv} "
              f"(fs_hz={fs_hz:,.1f}, n_bits={n_bits}, captured {meta.timestamp})")
    else:
        if args.fs_hz is None and args.decim is None:
            parser.error("--host requires --fs-hz or --decim so the frequency axis is known")
        fs_hz = args.fs_hz if args.fs_hz is not None else 80_000_000.0 / args.decim
        n_bits = args.n_bits

        try:
            ch1, ch2 = asyncio.run(_live_acquire(
                args.host, args.port, args.decim, args.sample_size, args.timeout))
        except (TimeoutError, DeviceError, OSError) as e:
            print(f"Acquisition failed: {e}", file=sys.stderr)
            return 1

        print(f"Captured {len(ch1)} samples from {args.host}:{args.port} (fs_hz={fs_hz:,.1f})")

        if args.save_csv:
            save_capture_csv(args.save_csv, ch1, ch2, fs_hz, n_bits)
            print(f"Saved capture -> {args.save_csv}")

    samples = ch1 if args.channel == 1 else ch2
    channel_label = f"CH{args.channel}"

    result = compute_metrics(
        samples, fs_hz, n_bits=n_bits, n_harmonics=args.n_harmonics, window=args.window)

    print()
    print(format_report(result, channel_label))

    if args.save_report:
        _save_report(args.save_report, result, channel_label)
    if args.save_plot:
        _save_plot(args.save_plot, samples, fs_hz, result, channel_label)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
