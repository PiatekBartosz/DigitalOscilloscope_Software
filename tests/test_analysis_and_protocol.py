"""Regression tests for the measurement CSV contract and TCP frame parser."""

import os
import pathlib
import queue
import sys
import tempfile
import unittest

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.capture_io import load_capture_csv, save_capture_csv
from analysis.metrics import compute_metrics
from core.command_client import CommandClient

# UI layout tests run without a desktop session in CI.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtWidgets import QApplication
from ui.oscilloscope import Oscilloscope


class CaptureIoTests(unittest.TestCase):
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



if __name__ == "__main__":
    unittest.main()
