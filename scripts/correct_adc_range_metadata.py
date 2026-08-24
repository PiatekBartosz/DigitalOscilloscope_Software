#!/usr/bin/env python3
"""Correct swapped 1 Vpp/2 Vpp labels in the accepted measurement series.

The correction changes file names and metadata only.  Original files are
copied to a backup directory before any source file is renamed.  Raw sample
rows are preserved byte-for-byte apart from the unchanged CSV header line.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAPTURES_DIR = PROJECT_ROOT / "scripts" / "results" / "captures"
BACKUP_DIR = CAPTURES_DIR / "old" / "adc_range_label_backup_20260823"


@dataclass(frozen=True)
class SeriesCorrection:
    source_prefix: str
    target_prefix: str
    recorded_vpp: int
    physical_vpp: int


CORRECTIONS = (
    SeriesCorrection("noise_2vpp_gnd", "noise_1vpp_gnd", 2, 1),
    SeriesCorrection("noise_1vpp_gnd", "noise_2vpp_gnd", 1, 2),
    SeriesCorrection(
        "harmonics_2vpp_1to1_20mVdiv",
        "harmonics_1vpp_1to1_profile50mVdiv_g45p44",
        2,
        1,
    ),
    SeriesCorrection(
        "harmonics_2vpp_1to1_20mVdiv_g49p50",
        "harmonics_1vpp_1to1_profile20mVdiv_g49p50",
        2,
        1,
    ),
    SeriesCorrection(
        "harmonics_2vpp_1to100_0p5Vdiv_g55p50",
        "harmonics_1vpp_1to100_profile0p5Vdiv_g55p50",
        2,
        1,
    ),
    SeriesCorrection(
        "harmonics_2vpp_1to100_1Vdiv_g52p53",
        "harmonics_1vpp_1to100_profile1Vdiv_g52p53",
        2,
        1,
    ),
    SeriesCorrection(
        "harmonics_2vpp_1to100_2Vdiv_g49p66",
        "harmonics_1vpp_1to100_profile2Vdiv_g49p66",
        2,
        1,
    ),
    SeriesCorrection(
        "harmonics_2vpp_1to100_5Vdiv_g45p00",
        "harmonics_1vpp_1to100_profile5Vdiv_g45p00",
        2,
        1,
    ),
    SeriesCorrection(
        "harmonics_1vpp_1to1_profile20mVdiv_g49p50",
        "harmonics_2vpp_1to1_profile20mVdiv_g49p50",
        1,
        2,
    ),
)


def _series_files(prefix: str) -> list[Path]:
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d{{2}})\.csv$")
    return sorted(
        path
        for path in CAPTURES_DIR.iterdir()
        if path.is_file() and pattern.fullmatch(path.name)
    )


def _replace_metadata(
    text: str,
    correction: SeriesCorrection,
    target_name: str,
) -> str:
    required = {
        "capture_name": None,
        "firmware_afe_ch1_range_vpp": str(correction.recorded_vpp),
        "firmware_afe_ch2_range_vpp": str(correction.recorded_vpp),
        "sense_ch1_vpp": str(correction.recorded_vpp),
        "sense_ch2_vpp": str(correction.recorded_vpp),
        "series_id": correction.source_prefix,
    }
    found: dict[str, str] = {}
    lines = text.splitlines(keepends=True)
    for line in lines:
        if not line.startswith("# ") or "=" not in line:
            continue
        key, value = line[2:].rstrip("\r\n").split("=", 1)
        if key in required:
            found[key] = value

    missing = sorted(set(required).difference(found))
    if missing:
        raise ValueError(f"missing metadata fields: {', '.join(missing)}")
    for key, expected in required.items():
        if expected is not None and found[key] != expected:
            raise ValueError(
                f"unexpected {key}: expected {expected!r}, found {found[key]!r}"
            )

    target_series = correction.target_prefix
    replacement_values = {
        "capture_name": target_name,
        "firmware_afe_ch1_range_vpp": str(correction.physical_vpp),
        "firmware_afe_ch2_range_vpp": str(correction.physical_vpp),
        "sense_ch1_vpp": str(correction.physical_vpp),
        "sense_ch2_vpp": str(correction.physical_vpp),
        "series_id": target_series,
    }
    corrected: list[str] = []
    inserted_audit = False
    has_voltage_columns = any(
        line.startswith("index,") and "ch1_volts" in line for line in lines
    )
    for line in lines:
        if line.startswith("# ") and "=" in line:
            key = line[2:].split("=", 1)[0]
            if key in replacement_values:
                newline = "\r\n" if line.endswith("\r\n") else "\n"
                line = f"# {key}={replacement_values[key]}{newline}"
        corrected.append(line)
        if line.startswith("# format_version="):
            newline = "\r\n" if line.endswith("\r\n") else "\n"
            voltage_status = (
                "not_validated_after_adc_range_correction"
                if has_voltage_columns
                else "not_present"
            )
            corrected.extend(
                [
                    f"# metadata_correction=adc_range_labels_swapped_after_measurement{newline}",
                    f"# original_recorded_adc_range_vpp_ch1={correction.recorded_vpp}{newline}",
                    f"# original_recorded_adc_range_vpp_ch2={correction.recorded_vpp}{newline}",
                    f"# physical_adc_range_vpp_ch1={correction.physical_vpp}{newline}",
                    f"# physical_adc_range_vpp_ch2={correction.physical_vpp}{newline}",
                    f"# voltage_columns_status={voltage_status}{newline}",
                ]
            )
            inserted_audit = True
    if not inserted_audit:
        raise ValueError("format_version metadata field not found")
    return "".join(corrected)


def main() -> int:
    if BACKUP_DIR.exists():
        raise FileExistsError(f"backup directory already exists: {BACKUP_DIR}")

    planned: list[tuple[Path, Path, str]] = []
    source_paths: set[Path] = set()
    target_paths: set[Path] = set()
    for correction in CORRECTIONS:
        files = _series_files(correction.source_prefix)
        if len(files) != 10:
            raise ValueError(
                f"{correction.source_prefix}: expected 10 files, found {len(files)}"
            )
        for source in files:
            repetition = source.stem.rsplit("_", 1)[1]
            target = CAPTURES_DIR / f"{correction.target_prefix}_{repetition}.csv"
            corrected = _replace_metadata(
                source.read_text(encoding="utf-8"), correction, target.name
            )
            planned.append((source, target, corrected))
            source_paths.add(source)
            if target in target_paths:
                raise ValueError(f"duplicate target path: {target}")
            target_paths.add(target)

    conflicts = sorted(path for path in target_paths if path.exists() and path not in source_paths)
    if conflicts:
        raise FileExistsError(f"target path already exists: {conflicts[0]}")

    BACKUP_DIR.mkdir(parents=True)
    manifest_lines = []
    for source, _, _ in planned:
        backup = BACKUP_DIR / source.name
        shutil.copy2(source, backup)
        digest = hashlib.sha256(backup.read_bytes()).hexdigest()
        manifest_lines.append(f"{digest}  {source.name}\n")
    (BACKUP_DIR / "SHA256SUMS").write_text(
        "".join(sorted(manifest_lines)), encoding="utf-8"
    )

    staged: list[tuple[Path, Path, str]] = []
    for source, target, corrected in planned:
        temporary = source.with_name(f".{source.name}.adc_range_fix_tmp")
        if temporary.exists():
            raise FileExistsError(f"temporary path already exists: {temporary}")
        source.rename(temporary)
        staged.append((temporary, target, corrected))

    for temporary, target, corrected in staged:
        temporary.write_text(corrected, encoding="utf-8")
        temporary.rename(target)

    print(f"Corrected {len(planned)} files")
    print(f"Original files backed up in {BACKUP_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
