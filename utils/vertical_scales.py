"""Requested vertical scales used by the normal oscilloscope mode.

The list is a user-interface requirement, independent of a particular
calibration file.  Calibration only decides whether the analogue front end can
realize a requested scale and, if so, which settings it needs.
"""

from __future__ import annotations


# The calibrated ranges currently support these conventional 1-2-5 scales.
# Add a value here before taking the calibration measurements needed to support
# it.  Values are input voltage per vertical display division, in volts.
VDIV_SCALES_BY_ATTENUATION: dict[str, tuple[float, ...]] = {
    "1:1": (0.020, 0.050),
    "1:100": (0.5, 1.0, 2.0, 5.0),
}


def requested_volts_per_div_values(attenuation: str) -> tuple[float, ...]:
    """Return the fixed V/div scales requested for one input attenuation."""
    try:
        return VDIV_SCALES_BY_ATTENUATION[attenuation]
    except KeyError as exc:
        raise ValueError(f"unsupported attenuation {attenuation}") from exc
