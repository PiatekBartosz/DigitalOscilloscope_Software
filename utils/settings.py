"""Small, local persistence for user-selected oscilloscope settings."""

import json
from pathlib import Path


_SETTINGS_PATH = Path(__file__).resolve().parents[1] / "settings.json"
_DEFAULTS = {
    "adc_range_vpp": {"ch1": 2, "ch2": 2},
    "ui_mode": "normal",
    "normal_volts_per_div": {"ch1": 5.0, "ch2": 5.0},
}


def load_settings() -> dict:
    """Return valid settings, falling back to safe defaults on any error."""
    settings = {
        "adc_range_vpp": dict(_DEFAULTS["adc_range_vpp"]),
        "ui_mode": _DEFAULTS["ui_mode"],
        "normal_volts_per_div": dict(_DEFAULTS["normal_volts_per_div"]),
    }
    try:
        loaded = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
        ranges = loaded.get("adc_range_vpp", {})
        for channel in ("ch1", "ch2"):
            if ranges.get(channel) in (1, 2):
                settings["adc_range_vpp"][channel] = ranges[channel]
        if loaded.get("ui_mode") in ("normal", "advanced"):
            settings["ui_mode"] = loaded["ui_mode"]
        normal_scales = loaded.get("normal_volts_per_div", {})
        for channel in ("ch1", "ch2"):
            value = normal_scales.get(channel)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
                settings["normal_volts_per_div"][channel] = float(value)
    except (OSError, ValueError, TypeError):
        pass
    return settings


def save_settings(settings: dict) -> None:
    """Persist settings atomically enough for a desktop configuration file."""
    try:
        _SETTINGS_PATH.write_text(
            json.dumps(settings, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except OSError:
        pass
