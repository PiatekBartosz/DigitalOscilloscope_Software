"""Discrete vertical-calibration profiles for the oscilloscope UI.

Each profile maps a requested vertical sensitivity directly to the measured VGA
setting and voltage-per-code coefficient. The AFE offset is deliberately not
part of calibration: it remains a live percentage control used to position a
waveform on the screen.
"""

from __future__ import annotations

import copy
import json
import math
import os
from itertools import pairwise
from pathlib import Path
from typing import Any

CALIBRATION_PATH = Path(__file__).resolve().parents[1] / "calibration_profiles.json"
ADC_FORMATS = ("Offset Binary", "2's Complement")
_PATHS = tuple(
    (channel, attenuation, sense_vpp)
    for channel in (1, 2)
    for attenuation in ("1:1", "1:100")
    for sense_vpp in (1.0, 2.0)
)
_SCALES = {
    ("1:1", 1.0): (0.02, 0.05),
    ("1:100", 1.0): (0.5, 1.0, 2.0, 5.0),
    ("1:1", 2.0): (0.02, 0.05),
    ("1:100", 2.0): (1.0, 2.0, 5.0),
}


def profile_key(channel: int, attenuation: str, sense_vpp: float = 1.0) -> str:
    if channel not in (1, 2) or (attenuation, float(sense_vpp)) not in _SCALES:
        raise ValueError("invalid channel, attenuation, or SENSE range")
    return f"ch{channel}_{attenuation.replace(':', 'to')}_{int(sense_vpp)}vpp"


def default_document() -> dict[str, Any]:
    profiles: dict[str, dict[str, Any]] = {}
    for channel, attenuation, sense_vpp in _PATHS:
        profiles[profile_key(channel, attenuation, sense_vpp)] = {
            "channel": channel,
            "attenuation": attenuation,
            "sense_vpp": sense_vpp,
            "adc_format": "Offset Binary",
            "points": [
                {"volts_per_div": value, "gain_pct": None, "volts_per_code": None}
                for value in _SCALES[(attenuation, sense_vpp)]
            ],
        }
    return {"schema_version": 3, "profiles": profiles}


def _number(value: object, *, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool):
        raise TypeError("boolean is not a calibration number")
    return float(value)


def validate_document(document: object) -> dict[str, Any]:
    """Validate and normalize discrete sensitivity calibration profiles."""
    if not isinstance(document, dict) or document.get("schema_version") != 3:
        raise ValueError("unsupported calibration document")
    raw_profiles = document.get("profiles")
    if not isinstance(raw_profiles, dict):
        raise TypeError("profiles must be an object")

    normalized = default_document()
    for channel, attenuation, expected_sense_vpp in _PATHS:
        key = profile_key(channel, attenuation, expected_sense_vpp)
        raw = raw_profiles.get(key)
        if not isinstance(raw, dict):
            raise TypeError(f"missing profile {key}")
        if raw.get("channel") != channel or raw.get("attenuation") != attenuation:
            raise ValueError(f"profile {key} does not match its physical path")
        sense_vpp = _number(raw.get("sense_vpp"))
        if sense_vpp != expected_sense_vpp:
            raise ValueError(f"profile {key} does not match its SENSE range")
        adc_format = raw.get("adc_format")
        if adc_format not in ADC_FORMATS:
            raise ValueError(f"profile {key} has invalid ADC format")
        points = raw.get("points")
        expected_scales = set(_SCALES[(attenuation, expected_sense_vpp)])
        if not isinstance(points, list) or len(points) != len(expected_scales):
            raise ValueError(f"profile {key} has an invalid number of points")

        normalized_points = []
        for point in points:
            if not isinstance(point, dict):
                raise TypeError(f"profile {key} contains an invalid point")
            volts_per_div = _number(point.get("volts_per_div"))
            gain_pct = _number(point.get("gain_pct"), allow_none=True)
            volts_per_code = _number(point.get("volts_per_code"), allow_none=True)
            if volts_per_div not in expected_scales:
                raise ValueError(f"profile {key} has an unsupported V/div value")
            if (gain_pct is None) != (volts_per_code is None):
                raise ValueError(f"profile {key} must store gain and V/code together")
            if gain_pct is not None and not 0.0 <= gain_pct <= 100.0:
                raise ValueError(f"profile {key} has gain outside 0-100 %")
            if volts_per_code == 0.0:
                raise ValueError(f"profile {key} has a zero V/code coefficient")
            normalized_points.append(
                {
                    "volts_per_div": volts_per_div,
                    "gain_pct": gain_pct,
                    "volts_per_code": volts_per_code,
                }
            )
        if {point["volts_per_div"] for point in normalized_points} != expected_scales:
            raise ValueError(f"profile {key} must contain every requested V/div value")
        normalized_points.sort(key=lambda point: point["volts_per_div"])
        normalized["profiles"][key] = {
            "channel": channel,
            "attenuation": attenuation,
            "sense_vpp": sense_vpp,
            "adc_format": adc_format,
            "points": normalized_points,
        }
    return normalized


def load_calibration(path: Path = CALIBRATION_PATH) -> dict[str, Any]:
    try:
        return validate_document(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return default_document()


def save_calibration(document: object, path: Path = CALIBRATION_PATH) -> None:
    normalized = validate_document(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(normalized, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_path, path)


def configuration_for_volts_per_div(
    profile: dict[str, Any], volts_per_div: float, **_ignored: object
) -> tuple[float, float] | None:
    """Return the measured ``(gain_pct, volts_per_code)`` for one sensitivity."""
    for point in profile["points"]:
        if math.isclose(point["volts_per_div"], float(volts_per_div), rel_tol=0.0, abs_tol=1e-15):
            if point["gain_pct"] is None:
                return None
            return float(point["gain_pct"]), float(point["volts_per_code"])
    return None


def interpolate_profile(profile: dict[str, Any], gain_pct: float) -> float | None:
    """Estimate V/code for Advanced-mode calibrated display at a VGA setting."""
    points = [point for point in profile["points"] if point["gain_pct"] is not None]
    if not points:
        return None
    points.sort(key=lambda point: float(point["gain_pct"]))
    gain_pct = float(gain_pct)
    if gain_pct <= points[0]["gain_pct"]:
        return float(points[0]["volts_per_code"])
    if gain_pct >= points[-1]["gain_pct"]:
        return float(points[-1]["volts_per_code"])
    for lower, upper in pairwise(points):
        if lower["gain_pct"] <= gain_pct <= upper["gain_pct"]:
            fraction = (gain_pct - lower["gain_pct"]) / (upper["gain_pct"] - lower["gain_pct"])
            lower_slope = float(lower["volts_per_code"])
            upper_slope = float(upper["volts_per_code"])
            return math.copysign(
                math.exp(math.log(abs(lower_slope)) + fraction * (math.log(abs(upper_slope)) - math.log(abs(lower_slope)))),
                lower_slope,
            )
    raise AssertionError("validated points must bracket the VGA setting")


def clone_document(document: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(document)
