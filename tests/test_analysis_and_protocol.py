"""Regression tests for the measurement CSV contract and TCP frame parser."""

import os
import pathlib
import queue
import sys
import tempfile
import unittest
from itertools import pairwise
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.capture_io import load_capture_csv, save_capture_csv
from analysis.metrics import compute_metrics
from core.command_client import CommandClient
from scripts.verify_spectral_metrics import verify
from utils.calibration import (
    configuration_for_volts_per_div,
    default_document,
    interpolate_profile,
    load_calibration,
    profile_key,
    save_calibration,
    validate_document,
)

# UI layout tests run without a desktop session in CI.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtWidgets import QApplication

from ui.oscilloscope import (
    ADC_COUNTS,
    DISPLAY_VERTICAL_DIVISIONS,
    FFT_Y_DIVISION_DB,
    FFT_Y_MAX_DBFS,
    FFT_Y_MIN_DBFS,
    MAX_DISPLAY_POINTS,
    Oscilloscope,
)
from utils.vertical_scales import requested_volts_per_div_values


class CaptureIoTests(unittest.TestCase):
    def test_fft_display_reduction_preserves_local_extrema(self):
        levels = np.zeros(4096)
        levels[512] = 10.0
        levels[513] = -10.0

        indices = Oscilloscope._display_indices(levels)

        self.assertLessEqual(len(indices), MAX_DISPLAY_POINTS)
        self.assertTrue(np.all(np.diff(indices) > 0))
        self.assertIn(512, indices)
        self.assertIn(513, indices)

    def test_synthetic_capture_recovers_known_snr_sinad_and_enob(self):
        _, result, expected = verify()

        self.assertAlmostEqual(result.snr_db, expected[0], delta=0.25)
        self.assertAlmostEqual(result.sinad_db, expected[1], delta=0.25)
        self.assertAlmostEqual(result.enob, expected[2], delta=0.25 / 6.02)

    def test_round_trip_preserves_raw_data_and_metadata(self):
        ch1 = np.array([0, 1, 8192, 16383], dtype=np.uint16)
        ch2 = np.array([16383, 8192, 1, 0], dtype=np.uint16)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "capture.csv"
            save_capture_csv(
                path,
                ch1,
                ch2,
                fs_hz=8_000_000.0,
                metadata={"decim_factor": 10, "trigger_mode": "normal"},
                ch1_volts=np.array([-5.0, -4.99, 0.0, 4.999]),
                ch2_volts=np.array([4.999, 0.0, -4.99, -5.0]),
            )
            got1, got2, meta = load_capture_csv(path)

        np.testing.assert_array_equal(got1, ch1)
        np.testing.assert_array_equal(got2, ch2)
        self.assertEqual(meta.fs_hz, 8_000_000.0)
        self.assertEqual(meta.fields["decim_factor"], "10")
        self.assertEqual(meta.fields["format_version"], "2")

    def test_coherent_sine_has_plausible_dynamic_metrics(self):
        n = 8192
        cycles = 73
        sample_index = np.arange(n)
        samples = np.round(8192 + 7000 * np.sin(2 * np.pi * cycles * sample_index / n))
        result = compute_metrics(
            samples, fs_hz=80_000_000.0, window="rect", leakage_bins=0
        )

        self.assertTrue(result.is_coherent)
        self.assertEqual(result.fundamental_bin, cycles)
        self.assertGreater(result.snr_db, 60.0)
        self.assertGreater(result.enob, 9.0)


class CalibrationProfileTests(unittest.TestCase):
    def test_requested_vdiv_values_are_defined_before_calibration(self):
        self.assertEqual(
            requested_volts_per_div_values("1:1"), (0.02, 0.05)
        )
        self.assertEqual(
            requested_volts_per_div_values("1:100"), (0.5, 1.0, 2.0, 5.0)
        )

    def test_vdiv_configuration_returns_the_measured_discrete_point(self):
        document = default_document()
        profile = document["profiles"][profile_key(1, "1:1", 1.0)]
        profile["points"][0].update(
            {"gain_pct": 49.5, "volts_per_code": 1.0040444607857001e-05}
        )
        profile = validate_document(document)["profiles"]["ch1_1to1_1vpp"]

        gain, slope = configuration_for_volts_per_div(profile, 0.020)
        self.assertAlmostEqual(gain, 49.5)
        self.assertAlmostEqual(slope, 1.0040444607857001e-05)

    def test_default_document_contains_the_six_requested_sensitivities(self):
        document = validate_document(default_document())

        self.assertEqual(
            set(document["profiles"]),
            {
                "ch1_1to1_1vpp",
                "ch1_1to1_2vpp",
                "ch1_1to100_1vpp",
                "ch1_1to100_2vpp",
                "ch2_1to1_1vpp",
                "ch2_1to1_2vpp",
                "ch2_1to100_1vpp",
                "ch2_1to100_2vpp",
            },
        )
        self.assertEqual(
            [
                point["volts_per_div"]
                for point in document["profiles"]["ch1_1to1_1vpp"]["points"]
            ],
            [0.02, 0.05],
        )
        self.assertEqual(
            [
                point["volts_per_div"]
                for point in document["profiles"]["ch1_1to100_1vpp"]["points"]
            ],
            [0.5, 1.0, 2.0, 5.0],
        )
        self.assertEqual(
            [
                point["volts_per_div"]
                for point in document["profiles"]["ch1_1to100_2vpp"]["points"]
            ],
            [1.0, 2.0, 5.0],
        )

    def test_saved_calibration_round_trip_and_interpolation(self):
        document = default_document()
        profile = document["profiles"][profile_key(1, "1:1", 1.0)]
        profile["points"][0].update(
            {"gain_pct": 40.0, "volts_per_code": 1e-5}
        )
        profile["points"][1].update(
            {"gain_pct": 60.0, "volts_per_code": 4e-5}
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "calibration.json"
            save_calibration(document, path)
            loaded = load_calibration(path)

        slope = interpolate_profile(loaded["profiles"]["ch1_1to1_1vpp"], 50.0)
        self.assertAlmostEqual(slope, 2e-5)
        self.assertEqual(
            loaded["profiles"]["ch1_1to1_1vpp"]["points"][0]["gain_pct"], 40.0
        )

    def test_calibration_rejects_partial_point(self):
        document = default_document()
        document["profiles"]["ch1_1to1_1vpp"]["points"][0]["volts_per_code"] = 1e-3
        with self.assertRaises(ValueError):
            validate_document(document)

    def test_low_bin_tone_does_not_overlap_harmonic_clusters(self):
        """Default leakage width must remain finite for a bin-4 tone."""
        n = 8192
        samples = np.round(8192 + 2500 * np.sin(2 * np.pi * 4 * np.arange(n) / n))
        result = compute_metrics(samples, fs_hz=40_000_000.0, window="hann")

        self.assertTrue(np.isfinite(result.snr_db))
        self.assertTrue(np.isfinite(result.noise_floor_dbfs))
        self.assertGreater(result.snr_db, 60.0)


class ProtocolParserTests(unittest.TestCase):
    def test_parser_handles_text_and_fragmented_binary_frame(self):
        frames = []
        replies = []
        client = CommandClient(
            "unused",
            0,
            frame_cb=lambda *args: frames.append(args),
            text_cb=replies.append,
        )
        payload = bytes(
            [
                0xAD,
                0xC1,
                0,
                0,
                0,
                7,
                0,
                2,
                0x00,
                0x01,
                0x3F,
                0xFF,
                0x20,
                0x00,
                0x00,
                0x02,
            ]
        )

        remaining = client._parse_frames(bytearray(b"OK\n" + payload[:11]))
        self.assertEqual(replies, ["OK"])
        self.assertEqual(frames, [])
        remaining.extend(payload[11:])
        remaining = client._parse_frames(remaining)

        self.assertEqual(remaining, bytearray())
        self.assertEqual(len(frames), 1)
        sequence, ch1, ch2 = frames[0]
        self.assertEqual(sequence, 7)
        np.testing.assert_array_equal(ch1, np.array([1, 8192], dtype=np.uint16))
        np.testing.assert_array_equal(ch2, np.array([16383, 2], dtype=np.uint16))


class OscilloscopeLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_sidebar_controls_fit_without_horizontal_clipping(self):
        scope = Oscilloscope(None, queue.Queue())
        scope.show()
        self.app.processEvents()
        try:
            sidebar_width = scope._sidebar_scroll.viewport().width()
            content_width = scope._sidebar_scroll.widget().sizeHint().width()
            self.assertLessEqual(content_width, sidebar_width)
        finally:
            scope.close()
            scope.deleteLater()

    def test_small_timebases_reduce_capture_depth(self):
        expected_depths = {
            200e-9: 256,
            500e-9: 512,
            1e-6: 1024,
            2e-6: 2048,
            5e-6: 4096,
            10e-6: 8192,
        }
        for target, expected_depth in expected_depths.items():
            self.assertEqual(
                Oscilloscope._largest_compatible_capture_size(target), expected_depth
            )

    def test_default_timebase_matches_default_capture_configuration(self):
        scope = Oscilloscope(None, queue.Queue())
        try:
            self.assertEqual(scope.capture_size, 8192)
            self.assertEqual(scope.decimation, 1)
            self.assertEqual(scope._timebase_target, 10e-6)
            self.assertAlmostEqual(scope._time_per_div(scope.decimation), 10.24e-6)
        finally:
            scope.close()
            scope.deleteLater()

    def test_decimation_study_preset_applies_frequency_and_decimation(self):
        scope = Oscilloscope(None, queue.Queue())
        try:
            study_index = scope._series_study_combo.findData("decimation")
            scope._series_study_combo.setCurrentIndex(study_index)
            point_index = next(
                index
                for index in range(scope._series_decimation_point_combo.count())
                if scope._series_decimation_point_combo.itemData(index)["decimation"] == 50
            )
            scope._series_decimation_point_combo.setCurrentIndex(point_index)

            self.assertEqual(scope._decimation_spinbox.value(), 50)
            self.assertEqual(scope.decimation, 50)
            self.assertEqual(scope._series_frequency.value(), 100_000.0)
            self.assertEqual(scope._series_amplitude.value(), 0.13)
            self.assertFalse(scope._series_generator_frequency_confirmed.isChecked())
            self.assertIn("16 samples/period", scope._series_decimation_instruction.text())

            error = scope._series_setup_error()
            self.assertIn("Confirm that the physical SENSE", error)
            scope._series_sense_confirmed.setChecked(True)
            error = scope._series_setup_error()
            self.assertIn("Confirm that the generator frequency", error)
            scope._series_generator_frequency_confirmed.setChecked(True)
            self.assertIsNone(scope._series_setup_error())
        finally:
            scope.close()
            scope.deleteLater()

    def test_normal_and_advanced_modes_expose_the_expected_controls(self):
        scope = Oscilloscope(None, queue.Queue())
        try:
            scope._on_connected()
            scope._set_ui_mode("normal", persist=False)
            self.assertFalse(scope._normal_controls_group.isHidden())
            self.assertTrue(scope._acquisition_group.isHidden())
            self.assertFalse(scope._calibration_action.isEnabled())

            scope._set_ui_mode("advanced", persist=False)
            self.assertTrue(scope._normal_controls_group.isHidden())
            self.assertFalse(scope._acquisition_group.isHidden())
            self.assertTrue(scope._calibration_action.isEnabled())
            self.assertTrue(scope._decimation_spinbox.isEnabled())
            self.assertTrue(scope._sample_size_combo.isEnabled())
            self.assertTrue(scope._timebase_combo.isEnabled())
            self.assertIs(scope._timebase_combo.parent(), scope._acquisition_group)
            self.assertIs(scope._trigger_source_combo.parent(), scope._trigger_group)
            self.assertIs(scope._run_btn.parent(), scope._acquisition_group)
            self.assertEqual(scope._normal_trigger_value.value(), 50.0)
        finally:
            scope.close()
            scope.deleteLater()

    def test_calibrated_input_voltage_uses_saved_volts_per_code(self):
        scope = Oscilloscope(None, queue.Queue())
        try:
            scope._set_ui_mode("normal", persist=False)
            volts_per_code = 2.0 * DISPLAY_VERTICAL_DIVISIONS / ADC_COUNTS
            document = default_document()
            for profile in document["profiles"].values():
                for point in profile["points"]:
                    point.update(
                        {"gain_pct": 50.0, "volts_per_code": volts_per_code}
                    )
            scope._calibration = validate_document(document)
            scope._settings["adc_range_vpp"] = {"ch1": 1, "ch2": 1}
            scope._refresh_normal_scale_controls()
            volts = scope._input_volts(
                np.array([8192, 8193], dtype=np.uint16), 1
            )
            np.testing.assert_allclose(volts, np.array([0.0, volts_per_code]))
            self.assertTrue(scope._normal_ch1_scale_combo.isEnabled())
        finally:
            scope.close()
            scope.deleteLater()

    def test_normal_mode_offers_range_specific_calibrated_scales(self):
        scope = Oscilloscope(None, queue.Queue())
        try:
            document = default_document()
            for profile in document["profiles"].values():
                for point in profile["points"]:
                    point.update({"gain_pct": 50.0, "volts_per_code": 1e-5})
            scope._calibration = validate_document(document)

            scope._settings["adc_range_vpp"] = {"ch1": 2, "ch2": 2}
            scope._refresh_normal_scale_controls()
            values_2vpp = {
                data["volts_per_div"]
                for index in range(scope._normal_ch1_scale_combo.count())
                if isinstance(
                    (data := scope._normal_ch1_scale_combo.itemData(index)), dict
                )
            }
            self.assertEqual(values_2vpp, {0.02, 0.05, 1.0, 2.0, 5.0})

            scope._settings["adc_range_vpp"] = {"ch1": 1, "ch2": 1}
            scope._refresh_normal_scale_controls()
            values_1vpp = {
                data["volts_per_div"]
                for index in range(scope._normal_ch1_scale_combo.count())
                if isinstance(
                    (data := scope._normal_ch1_scale_combo.itemData(index)), dict
                )
            }
            self.assertEqual(values_1vpp, {0.02, 0.05, 0.5, 1.0, 2.0, 5.0})
        finally:
            scope.close()
            scope.deleteLater()

    def test_saved_raw_capture_records_vdiv_and_channel_configuration(self):
        scope = Oscilloscope(None, queue.Queue())
        try:
            scope._set_ui_mode("advanced", persist=False)
            scope._advanced_calibrated_display.setChecked(False)
            scope._normal_volts_per_div = {"ch1": 0.02, "ch2": 0.05}
            scope._ch1_raw = np.array([8191, 8192], dtype=np.uint16)
            scope._ch2_raw = np.array([8192, 8193], dtype=np.uint16)
            scope._have_frame = True

            with tempfile.TemporaryDirectory() as tmpdir:
                path = pathlib.Path(tmpdir) / "capture.csv"
                with patch(
                    "ui.oscilloscope.QFileDialog.getSaveFileName",
                    return_value=(str(path), "CSV file (*.csv)"),
                ):
                    scope._save_frame()
                _, _, metadata = load_capture_csv(path)

            self.assertEqual(metadata.fields["voltage_display_mode"], "raw_adc_codes")
            self.assertEqual(metadata.fields["normal_requested_volts_per_div_ch1"], "0.02")
            self.assertEqual(metadata.fields["normal_requested_volts_per_div_ch2"], "0.05")
            self.assertEqual(metadata.fields["displayed_volts_per_div_ch1"], "raw_adc_codes")
            self.assertEqual(metadata.fields["attenuation_ch1"], "1:1")
            self.assertIn("gain_pct_ch1", metadata.fields)
            self.assertIn("afe_offset_pct_ch2", metadata.fields)
            self.assertIn("actual_seconds_per_div", metadata.fields)
        finally:
            scope.close()
            scope.deleteLater()

    def test_saved_capture_uses_applied_status_and_records_traceability(self):
        scope = Oscilloscope(None, queue.Queue())
        try:
            scope._ch1_raw = np.array([100, 200], dtype=np.uint16)
            scope._ch2_raw = np.array([300, 400], dtype=np.uint16)
            scope._have_frame = True
            scope._frame_sequence = 42
            scope.decimation = 5
            scope._last_status_timestamp = "2026-08-23T12:00:00+00:00"
            scope._afe_state.update(
                {
                    "build": "0x12",
                    "version": "0x08",
                    "depth": "2",
                    "pretrigger": "0",
                    "decim": "2",
                    "overflow": "1",
                }
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                path = pathlib.Path(tmpdir) / "capture.csv"
                scope._write_current_frame(path, {"measurement_type": "noise"})
                _, _, metadata = load_capture_csv(path)

            self.assertEqual(metadata.fs_hz, 40_000_000.0)
            self.assertEqual(metadata.fields["decim_factor"], "2")
            self.assertEqual(metadata.fields["requested_decim_factor"], "5")
            self.assertEqual(metadata.fields["firmware_overflow"], "1")
            self.assertEqual(metadata.fields["fpga_build"], "0x12")
            self.assertEqual(metadata.fields["fpga_version"], "0x08")
            self.assertEqual(metadata.fields["frame_sequence"], "42")
            self.assertEqual(metadata.fields["measurement_type"], "noise")
        finally:
            scope.close()
            scope.deleteLater()

    def test_series_saves_distinct_numbered_frames_with_repetition_metadata(self):
        scope = Oscilloscope(None, queue.Queue())
        try:
            scope._afe_state.update(
                {
                    "depth": "2",
                    "pretrigger": "0",
                    "decim": "1",
                    "overflow": "1",
                    "build": "0x12",
                    "version": "0x08",
                }
            )
            scope._batch_active = True
            scope._batch_state = "waiting_frame_status"
            scope._batch_total = 2
            scope._batch_saved = 0
            scope._batch_metadata = {"measurement_type": "noise"}

            with tempfile.TemporaryDirectory() as tmpdir:
                base_path = pathlib.Path(tmpdir) / "noise_2vpp.csv"
                scope._batch_base_path = base_path
                with patch("ui.oscilloscope.QTimer.singleShot"):
                    scope._frame_sequence = 100
                    scope._ch1_raw = np.array([100, 101], dtype=np.uint16)
                    scope._ch2_raw = np.array([200, 201], dtype=np.uint16)
                    scope._complete_current_series_frame()

                    scope._batch_state = "waiting_frame_status"
                    scope._frame_sequence = 101
                    scope._ch1_raw = np.array([110, 111], dtype=np.uint16)
                    scope._ch2_raw = np.array([210, 211], dtype=np.uint16)
                    scope._complete_current_series_frame()

                first_path = pathlib.Path(tmpdir) / "noise_2vpp_01.csv"
                second_path = pathlib.Path(tmpdir) / "noise_2vpp_02.csv"
                first_ch1, _, first_meta = load_capture_csv(first_path)
                second_ch1, _, second_meta = load_capture_csv(second_path)

            np.testing.assert_array_equal(first_ch1, np.array([100, 101]))
            np.testing.assert_array_equal(second_ch1, np.array([110, 111]))
            self.assertEqual(first_meta.fields["repetition_index"], "1")
            self.assertEqual(second_meta.fields["repetition_index"], "2")
            self.assertFalse(scope._batch_active)
        finally:
            scope.close()
            scope.deleteLater()

    def test_fft_view_has_fixed_10_dbfs_vertical_divisions(self):
        scope = Oscilloscope(None, queue.Queue())
        try:
            scope._configure_fft_axes()
            y_min, y_max = scope.plotWidget.getViewBox().viewRange()[1]
            self.assertEqual(y_min, FFT_Y_MIN_DBFS)
            self.assertEqual(y_max, FFT_Y_MAX_DBFS)
            tick_values = [
                value
                for value, _label in scope.plotWidget.getAxis("left")._tickLevels[0]
            ]
            self.assertEqual(tick_values[0], FFT_Y_MIN_DBFS)
            self.assertEqual(tick_values[-1], 0)
            self.assertTrue(
                all(
                    next_value - value == FFT_Y_DIVISION_DB
                    for value, next_value in pairwise(tick_values)
                )
            )
        finally:
            scope.close()
            scope.deleteLater()

    def test_time_view_restores_vertical_divisions_after_fft(self):
        scope = Oscilloscope(None, queue.Queue())
        try:
            scope._calibration = validate_document(default_document())
            scope._normal_volts_per_div = {"ch1": None, "ch2": None}
            scope._configure_fft_axes()
            scope._update_sample_axis()
            tick_values = [
                value
                for value, _label in scope.plotWidget.getAxis("left")._tickLevels[0]
            ]
            expected_step = ADC_COUNTS / DISPLAY_VERTICAL_DIVISIONS
            self.assertEqual(
                tick_values,
                [
                    -ADC_COUNTS / 2 + expected_step * index
                    for index in range(DISPLAY_VERTICAL_DIVISIONS + 1)
                ],
            )
        finally:
            scope.close()
            scope.deleteLater()

    def test_normal_vdiv_uses_standard_calibrated_scale_and_restores_afe_state(self):
        scope = Oscilloscope(None, queue.Queue())
        try:
            scope._set_ui_mode("normal", persist=False)
            document = default_document()
            for channel in (1, 2):
                profile = document["profiles"][profile_key(channel, "1:1", 1.0)]
                profile["points"][0].update(
                    {"gain_pct": 49.5, "volts_per_code": 1.0040444607857001e-05}
                )
                profile["points"][1].update(
                    {"gain_pct": 45.44, "volts_per_code": 2.4411651763673265e-05}
                )
            scope._calibration = validate_document(document)
            scope._normal_volts_per_div = {"ch1": None, "ch2": None}
            scope._settings["adc_range_vpp"] = {"ch1": 1, "ch2": 1}
            scope._refresh_normal_scale_controls()

            values = {
                data["volts_per_div"]
                for index in range(scope._normal_ch1_scale_combo.count())
                if isinstance(
                    (data := scope._normal_ch1_scale_combo.itemData(index)), dict
                )
            }
            self.assertEqual(values, {0.02, 0.05})

            index = next(
                index
                for index in range(scope._normal_ch1_scale_combo.count())
                if isinstance(scope._normal_ch1_scale_combo.itemData(index), dict)
                and scope._normal_ch1_scale_combo.itemData(index)["volts_per_div"] == 0.02
            )
            scope._normal_ch1_scale_combo.setCurrentIndex(index)
            self.assertAlmostEqual(scope._gain_value.value(), 49.5, places=2)
            self.assertEqual(scope._offset_value.value(), 50.0)
            scope._normal_volts_per_div = {"ch1": 0.02, "ch2": 0.05}
            scope.ch2_enabled = True
            scope._update_sample_axis()
            self.assertTrue(scope.plotWidget.getAxis("right").isVisible())
            left_ticks = scope.plotWidget.getAxis("left")._tickLevels[0]
            right_ticks = scope.plotWidget.getAxis("right")._tickLevels[0]
            self.assertIn((1, "20 mV"), left_ticks)
            self.assertIn((1, "50 mV"), right_ticks)
        finally:
            scope.close()
            scope.deleteLater()


if __name__ == "__main__":
    unittest.main()
