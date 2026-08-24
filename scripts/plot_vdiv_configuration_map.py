"""Generate the V/div-to-AFE configuration map used by normal mode.

Run from ``DigitalOscilloscope_Software``:

    .venv/bin/python scripts/plot_vdiv_configuration_map.py --no-show
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from analysis.plot_style import format_thesis_axis
from utils.calibration import configuration_for_volts_per_div, profile_key
from utils.vertical_scales import requested_volts_per_div_values


ADC_COUNTS = 1 << 14
DISPLAY_VERTICAL_DIVISIONS = 8
REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
CALIBRATION_PATH = REPOSITORY_ROOT / "DigitalOscilloscope_Software" / "calibration_profiles.json"
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT / "mastersThesis" / "masters_thesis" / "graf" / "vdiv_configuration_map.png"
)
CHANNEL_STYLE = {1: ("#C95B2B", "-"), 2: ("#147D9A", "--")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def vdiv_limits(profile: dict) -> tuple[float, float]:
    values = [
        abs(point["volts_per_code"]) * ADC_COUNTS / DISPLAY_VERTICAL_DIVISIONS
        for point in profile["gain_points"]
    ]
    return min(values), max(values)


def load_profiles() -> dict:
    with CALIBRATION_PATH.open(encoding="utf-8") as source:
        return json.load(source)["profiles"]


def plot_attenuation(gain_axis, offset_axis, profiles: dict, attenuation: str) -> None:
    for channel in (1, 2):
        profile = profiles[profile_key(channel, attenuation)]
        minimum, maximum = vdiv_limits(profile)
        requested_vdiv = np.geomspace(minimum, maximum, 300)
        configuration = [
            configuration_for_volts_per_div(
                profile,
                float(value),
                adc_counts=ADC_COUNTS,
                divisions=DISPLAY_VERTICAL_DIVISIONS,
            )
            for value in requested_vdiv
        ]
        gain = np.array([item[0] for item in configuration])
        afe_offset = np.array([item[1] for item in configuration])
        color, linestyle = CHANNEL_STYLE[channel]
        label = f"kanał {channel}"
        gain_axis.plot(requested_vdiv, gain, color=color, linestyle=linestyle, label=label)
        offset_axis.plot(requested_vdiv, afe_offset, color=color, linestyle=linestyle, label=label)

        calibration_points = sorted(profile["gain_points"], key=lambda point: point["gain_pct"])
        calibration_vdiv = [
            abs(point["volts_per_code"]) * ADC_COUNTS / DISPLAY_VERTICAL_DIVISIONS
            for point in calibration_points
        ]
        gain_axis.plot(
            calibration_vdiv,
            [point["gain_pct"] for point in calibration_points],
            linestyle="none",
            marker="o",
            markersize=5.5,
            color=color,
        )
        offset_axis.plot(
            calibration_vdiv,
            [point["afe_offset_pct"] for point in calibration_points],
            linestyle="none",
            marker="o",
            markersize=5.5,
            color=color,
        )

        selected_vdiv = [
            value
            for value in requested_volts_per_div_values(attenuation)
            if minimum <= value <= maximum
        ]
        selected_configuration = [
            configuration_for_volts_per_div(
                profile,
                value,
                adc_counts=ADC_COUNTS,
                divisions=DISPLAY_VERTICAL_DIVISIONS,
            )
            for value in selected_vdiv
        ]
        gain_axis.plot(
            selected_vdiv,
            [item[0] for item in selected_configuration],
            linestyle="none",
            marker="x",
            markersize=6,
            markeredgewidth=1.5,
            color=color,
        )
        offset_axis.plot(
            selected_vdiv,
            [item[1] for item in selected_configuration],
            linestyle="none",
            marker="x",
            markersize=6,
            markeredgewidth=1.5,
            color=color,
        )

    for axis in (gain_axis, offset_axis):
        axis.set_xscale("log")
    format_thesis_axis(gain_axis, "Żądana wartość V/div (V/div)", "Gain (%)")
    format_thesis_axis(offset_axis, "Żądana wartość V/div (V/div)", "Offset AFE (%)")


def main() -> int:
    args = parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required to generate the figure", file=sys.stderr)
        return 2

    profiles = load_profiles()
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.4), constrained_layout=True)
    plot_attenuation(axes[0, 0], axes[1, 0], profiles, "1:1")
    plot_attenuation(axes[0, 1], axes[1, 1], profiles, "1:100")
    axes[0, 0].set_title("Tłumienie wejściowe 1:1", loc="left", fontweight="bold")
    axes[0, 1].set_title("Tłumienie wejściowe 1:100", loc="left", fontweight="bold")
    for axis in axes[0]:
        axis.plot([], [], color="#404040", linestyle="none", marker="o", label="punkt kalibracyjny")
        axis.plot([], [], color="#404040", linestyle="none", marker="x", label="nastawa V/div")
        axis.legend(loc="best", fontsize=8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Saved: {args.output}")
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
