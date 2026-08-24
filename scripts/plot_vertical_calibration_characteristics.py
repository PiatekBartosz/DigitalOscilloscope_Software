"""Generate calibration-transfer characteristics for the thesis.

Run from ``DigitalOscilloscope_Software``:

    .venv/bin/python scripts/plot_vertical_calibration_characteristics.py --no-show

The script reads the saved calibration coefficients from
``calibration_profiles.json``. The cursor-derived peak-to-peak code values and
the corresponding generator settings are kept below as measurement metadata.
They are used to mark the actual calibration interval on each characteristic.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from analysis.plot_style import format_thesis_axis
from utils.calibration import profile_key


ADC_COUNTS = 1 << 14
REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
CALIBRATION_PATH = REPOSITORY_ROOT / "DigitalOscilloscope_Software" / "calibration_profiles.json"
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "mastersThesis"
    / "masters_thesis"
    / "graf"
    / "vertical_calibration_characteristics.png"
)

# Values measured with application cursors. ``code_pp`` and ``input_vpp`` are
# peak-to-peak values. ``zero_code`` is expressed relative to ADC midscale.
MEASUREMENTS = {
    "ch1_1to1": {
        0.0: {"zero_code": 5.2180175781, "code_pp": 13186.6, "input_vpp": 0.460},
        25.0: {"zero_code": 57.691699219, "code_pp": 11505.2, "input_vpp": 0.200},
        50.0: {"zero_code": 61.767822266, "code_pp": 8933.8, "input_vpp": 0.080},
    },
    "ch2_1to1": {
        0.0: {"zero_code": 0.7665283203, "code_pp": 13186.6, "input_vpp": 0.460},
        25.0: {"zero_code": -5.308496094, "code_pp": 11505.2, "input_vpp": 0.200},
        50.0: {"zero_code": 23.652832031, "code_pp": 8933.8, "input_vpp": 0.080},
    },
    "ch1_1to100": {
        0.0: {"zero_code": 67.823095703, "code_pp": 5895.8, "input_vpp": 20.0},
        50.0: {"zero_code": 128.071435547, "code_pp": 7007.6, "input_vpp": 6.0},
        100.0: {"zero_code": -36.781152344, "code_pp": 6401.2, "input_vpp": 1.4},
    },
    "ch2_1to100": {
        0.0: {"zero_code": -3.304003906, "code_pp": 5895.8, "input_vpp": 20.0},
        50.0: {"zero_code": 35.953369141, "code_pp": 7007.6, "input_vpp": 6.0},
        100.0: {"zero_code": -38.640625, "code_pp": 6401.2, "input_vpp": 1.4},
    },
}

COLORS = {0.0: "#C95B2B", 25.0: "#147D9A", 50.0: "#4C8B3B", 100.0: "#6F4C9B"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def load_profiles() -> dict:
    with CALIBRATION_PATH.open(encoding="utf-8") as source:
        document = json.load(source)
    return document["profiles"]


def plot_attenuation(axis, profiles: dict, attenuation: str) -> None:
    signed_codes = np.linspace(-ADC_COUNTS / 2, ADC_COUNTS / 2 - 1, 800)
    gains = (0.0, 25.0, 50.0) if attenuation == "1:1" else (0.0, 50.0, 100.0)

    for channel, linestyle in ((1, "-"), (2, "--")):
        key = profile_key(channel, attenuation)
        points = {point["gain_pct"]: point for point in profiles[key]["gain_points"]}
        for gain in gains:
            point = points[gain]
            slope = point["volts_per_code"]
            offset = point["offset_v"]
            voltage = slope * signed_codes + offset
            axis.plot(
                signed_codes,
                voltage,
                color=COLORS[gain],
                linestyle=linestyle,
                linewidth=1.35,
                label=f"Gain {gain:g}% - kanał {channel}",
            )

            measured = MEASUREMENTS[key][gain]
            endpoint_codes = measured["zero_code"] + np.array(
                [-measured["code_pp"] / 2, measured["code_pp"] / 2]
            )
            endpoint_voltages = np.array(
                [-measured["input_vpp"] / 2, measured["input_vpp"] / 2]
            )
            axis.plot(
                endpoint_codes,
                endpoint_voltages,
                linestyle="none",
                marker="o" if channel == 1 else "s",
                markersize=3.8,
                color=COLORS[gain],
            )

    axis.axhline(0.0, color="#404040", linewidth=0.7, alpha=0.7)
    axis.axvline(0.0, color="#404040", linewidth=0.7, alpha=0.7)
    format_thesis_axis(axis, "Kod próbki po przekształceniu do reprezentacji ze znakiem", "Napięcie wejściowe (V)")
    axis.set_title(f"Tłumienie wejściowe {attenuation}", loc="left", fontweight="bold")


def main() -> int:
    args = parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required to generate the figure", file=sys.stderr)
        return 2

    profiles = load_profiles()
    figure, axes = plt.subplots(2, 1, figsize=(11.5, 10.0), constrained_layout=True)
    plot_attenuation(axes[0], profiles, "1:1")
    plot_attenuation(axes[1], profiles, "1:100")
    axes[0].legend(ncol=2, fontsize=8.5)
    axes[1].legend(ncol=2, fontsize=8.5)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Saved: {args.output}")
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
