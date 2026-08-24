"""Tests for the measurement-plan noise and dual-channel analysis paths."""

import pathlib
import sys
import unittest

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.metrics import (
    average_centered_channels,
    compute_metrics,
    compute_noise_metrics,
)
from scripts.snr_analysis import resolve_adc_range_vpp, select_channel_samples


class MeasurementAnalysisTests(unittest.TestCase):
    def test_hann_spectrum_applies_one_sided_and_coherent_gain_corrections(self):
        sample_count = 8192
        sample_rate_hz = 80_000_000.0
        fundamental_bin = 512
        amplitude = 4000.0
        indices = np.arange(sample_count)
        samples = 8192.0 + amplitude * np.sin(
            2.0 * np.pi * fundamental_bin * indices / sample_count
        )

        result = compute_metrics(samples, sample_rate_hz, window="hann")
        fundamental_hz = fundamental_bin * sample_rate_hz / sample_count
        plotted_index = int(np.argmin(np.abs(result.freqs_hz - fundamental_hz)))
        full_scale_power = (2.0**13) ** 2 / 2.0
        expected_dbfs = 10.0 * np.log10(amplitude**2 / full_scale_power)

        self.assertAlmostEqual(
            result.spectrum_dbfs[plotted_index], expected_dbfs, delta=0.02
        )
        self.assertGreater(result.freqs_hz[0], 0.0)

    def test_harmonic_levels_are_reported_in_dbc(self):
        sample_count = 8192
        indices = np.arange(sample_count)
        phase = 2.0 * np.pi * 256 * indices / sample_count
        samples = 8192.0 + 4000.0 * np.sin(phase) + 400.0 * np.sin(2.0 * phase)

        result = compute_metrics(
            samples,
            fs_hz=80_000_000.0,
            window="rect",
            leakage_bins=0,
        )

        self.assertAlmostEqual(result.harmonic_levels_dbc[0], -20.0, delta=0.02)

    def test_noise_metrics_exclude_dc_and_convert_codes_to_adc_input_volts(self):
        samples = 8192.0 + np.tile(np.array([-2.0, 2.0]), 4096)
        result = compute_noise_metrics(
            samples,
            fs_hz=80_000_000.0,
            adc_range_vpp=2.0,
            window="hann",
        )

        self.assertAlmostEqual(result.dc_code, 8192.0)
        self.assertAlmostEqual(result.noise_rms_codes, 2.0)
        self.assertAlmostEqual(result.noise_rms_adc_volts, 2.0 / 8192.0)
        self.assertGreater(result.freqs_hz[0], 0.0)
        self.assertTrue(np.all(np.isfinite(result.spectrum_dbfs_per_hz)))

    def test_average_channel_removes_individual_dc_offsets(self):
        ch1 = np.array([101.0, 99.0, 101.0, 99.0])
        ch2 = np.array([201.0, 201.0, 199.0, 199.0])

        averaged = average_centered_channels(ch1, ch2)
        selected, label = select_channel_samples(ch1, ch2, "average")

        np.testing.assert_allclose(averaged, np.array([1.0, 0.0, 0.0, -1.0]))
        np.testing.assert_allclose(selected, averaged)
        self.assertEqual(label, "(CH1+CH2)/2")
        self.assertAlmostEqual(np.sqrt(np.mean(averaged**2)), 1.0 / np.sqrt(2.0))

    def test_adc_range_is_read_from_metadata_and_must_match_for_average(self):
        fields = {
            "firmware_afe_ch1_range_vpp": "2",
            "firmware_afe_ch2_range_vpp": "2",
        }
        self.assertEqual(resolve_adc_range_vpp(None, fields, "average"), 2.0)

        fields["firmware_afe_ch2_range_vpp"] = "1"
        with self.assertRaisesRegex(ValueError, "different recorded ADC ranges"):
            resolve_adc_range_vpp(None, fields, "average")


if __name__ == "__main__":
    unittest.main()
